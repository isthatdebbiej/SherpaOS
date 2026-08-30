"""Runs one MuJoCo G1 episode end to end: scenario setup, the PD
stepping controller, a control-rate guard callback, telemetry capture,
and evaluator-only ground-truth capture, until a fall or `max_steps`.

Control-rate design (read this before changing `dt`/`max_steps`):
  - The physics substeps at the vendored G1 XML's native timestep
    (`model.opt.timestep`, 0.002s / 500Hz) -- this is not overridden here,
    since every gain/amplitude/threshold in this sim lane was tuned
    against that exact timestep.
  - `dt` is the *control* period: how often the guard callback is
    consulted and a `RobotTelemetry` sample is captured. `run_episode`
    computes `substeps = round(dt / model.opt.timestep)` and holds the
    guard's `(speed_scale, hold)` decision constant (zero-order hold)
    across those substeps -- a standard two-rate real-robot layering
    (fast low-level joint control, slower high-level supervisory
    decisions). The default `dt=0.02` is a 50Hz control loop.
  - `max_steps` counts *control* steps, not physics steps, so
    `EpisodeResult.steps_survived` and `len(EpisodeResult.telemetry)`
    are directly comparable across calls with different `dt`. Default
    `max_steps=500` at the default `dt` is a 10s episode.
  - `sherpaos.sim.controller.PDStepController.step` is still called
    every *physics* substep (not just every control step), because its
    gait-phase accumulator must advance at the physics timestep to stay
    smooth; only the guard's `(speed_scale, hold)` inputs to it are held
    fixed for `substeps` physics steps at a time.

Fall detection: `pelvis z < FALL_PELVIS_Z_M` OR `tilt-from-vertical >
FALL_TILT_DEG`. Both were picked with wide empirical margin (see the
sim-lane report): every stable hold/nominal episode observed stayed at
pelvis z in [0.79, 0.792] and tilt under ~5 degrees even under a
sub-threshold disturbance, while every actual fall observed collapsed to
pelvis z in [0.04, 0.09] and tilt past ~88 degrees within a couple of
physics steps of each other -- i.e. there is no borderline zone between
these two constants and what was ever actually produced by this
controller, so a small nudge in either constant is not fall-detector-
sensitive.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from sherpaos.contracts import RobotTelemetry
from sherpaos.evaluation.ground_truth import ScenarioGroundTruth, classify_unsafe
from sherpaos.sim import disturbances
from sherpaos.sim.battery import enrich_battery_telemetry
from sherpaos.sim.controller import STAND_KEYFRAME_NAME, PDStepController
from sherpaos.sim.mujoco_source import MuJoCoTelemetrySource
from sherpaos.sim.scenario import Scenario

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
G1_SCENE_XML_PATH = _REPO_ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"

DEFAULT_CONTROL_DT_S = 0.02  # 50Hz guard-decision / telemetry-sampling rate
DEFAULT_MAX_CONTROL_STEPS = 500  # 500 * 0.02s = 10s nominal episode

# See module docstring for the empirical margin behind these two.
FALL_PELVIS_Z_M = 0.55
FALL_TILT_DEG = 35.0

LEFT_FOOT_BODY_NAME = "left_ankle_roll_link"
RIGHT_FOOT_BODY_NAME = "right_ankle_roll_link"
FLOOR_GEOM_NAME = "floor"

# guard_fn: given the telemetry history so far (oldest first, current
# episode only), return (speed_scale in [0, 1], hold). See
# sherpaos/sim/controller.py for what these mean physically.
GuardFn = Callable[[list[RobotTelemetry]], tuple[float, bool]]


def _default_guard_fn(_history: list[RobotTelemetry]) -> tuple[float, bool]:
    """No-op guard: always PASS at full speed. This is the implicit
    "controller-only" baseline when `run_episode` is called without a
    `guard_fn` -- i.e. no SherpaOS supervision at all.
    """
    return 1.0, False


@dataclass(slots=True, frozen=True)
class EpisodeResult:
    """Everything one `run_episode` call produced.

    `telemetry[i]` and `ground_truth[i]` describe the same control step
    `i` (same length, index-aligned) -- but `telemetry` is exactly what
    an estimator/policy would see, and `ground_truth` is evaluator-only;
    see `sherpaos.evaluation.ground_truth`'s module docstring for why
    these must never be merged into one object.
    """

    telemetry: list[RobotTelemetry]
    ground_truth: list[ScenarioGroundTruth]
    fell: bool
    steps_survived: int
    scenario: Scenario
    seed: int


def run_episode(
    scenario: Scenario,
    seed: int,
    guard_fn: GuardFn | None = None,
    max_steps: int = DEFAULT_MAX_CONTROL_STEPS,
    dt: float = DEFAULT_CONTROL_DT_S,
    model_path: str | Path | None = None,
    live_viewer: bool = False,
) -> EpisodeResult:
    """Run one G1 episode under `scenario` and return the full trace.

    `seed` seeds this run's sensor-noise RNG (independent of whatever
    seed `scenario` itself was constructed with, e.g. via
    `sherpaos.sim.scenario.mixed_traction_disturbance_scenario`), so the
    same `(scenario, seed)` pair reproduces byte-identical telemetry.
    """
    guard = guard_fn if guard_fn is not None else _default_guard_fn
    xml_path = str(model_path) if model_path is not None else str(G1_SCENE_XML_PATH)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, STAND_KEYFRAME_NAME)
    if key_id < 0:
        raise ValueError(f"model has no '{STAND_KEYFRAME_NAME}' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    disturbances.apply_scenario_to_model(model, scenario)
    mujoco.mj_forward(model, data)

    viewer = mujoco.viewer.launch_passive(model, data) if live_viewer else None

    controller = PDStepController()
    controller.reset(model)
    telemetry_source = MuJoCoTelemetrySource()
    noise_rng = np.random.default_rng(seed)

    physics_dt = float(model.opt.timestep)
    substeps = max(1, round(dt / physics_dt))

    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_FOOT_BODY_NAME)
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_FOOT_BODY_NAME)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    left_foot_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == left_foot_id}
    right_foot_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == right_foot_id}

    telemetry_history: list[RobotTelemetry] = []
    ground_truth_history: list[ScenarioGroundTruth] = []

    fell = False
    steps_survived = 0
    physics_step_idx = 0

    for control_step in range(max_steps):
        speed_scale, hold = guard(telemetry_history)
        speed_scale = float(np.clip(speed_scale, 0.0, 1.0))

        disturbance_active_this_period = False
        for _ in range(substeps):
            physics_step_started_at = time.perf_counter()
            controller.step(model, data, data.time, speed_scale, hold)
            if disturbances.disturbance_active_at_step(scenario, physics_step_idx):
                disturbance_active_this_period = True
            disturbances.apply_disturbance_wrench(model, data, scenario, physics_step_idx)
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
                remaining_time = physics_dt - (time.perf_counter() - physics_step_started_at)
                if remaining_time > 0.0:
                    time.sleep(remaining_time)
            physics_step_idx += 1

        gait_mode = "hold" if (hold or speed_scale <= 0.0) else "stepping"
        sample = telemetry_source.sample(
            model,
            data,
            sequence=control_step,
            monotonic_time=data.time,
            speed_scale=speed_scale,
            gait_mode=gait_mode,
        )
        sample = enrich_battery_telemetry(
            sample,
            scenario,
            elapsed_seconds=data.time,
            speed_scale=speed_scale,
        )
        if scenario.sensor_noise_std > 0.0:
            sample = disturbances.inject_sensor_noise(sample, scenario.sensor_noise_std, noise_rng)
        telemetry_history.append(sample)

        tilt_deg = _tilt_from_vertical_deg(data.qpos[3:7])
        slip_mps = max(
            _planted_foot_slip_mps(model, data, left_foot_id, left_foot_geoms, floor_id),
            _planted_foot_slip_mps(model, data, right_foot_id, right_foot_geoms, floor_id),
        )
        unsafe = classify_unsafe(
            tilt_from_vertical_deg=tilt_deg,
            actuator_health=scenario.actuator_health,
            planted_foot_slip_mps=slip_mps,
        )
        ground_truth_history.append(
            ScenarioGroundTruth(
                true_friction=scenario.friction,
                true_slope_deg=scenario.slope_deg,
                disturbance_active=disturbance_active_this_period,
                actuator_health=scenario.actuator_health,
                tilt_from_vertical_deg=tilt_deg,
                planted_foot_slip_mps=slip_mps,
                true_unsafe=unsafe,
            )
        )

        steps_survived = control_step + 1

        pelvis_z = float(data.qpos[2])
        if pelvis_z < FALL_PELVIS_Z_M or tilt_deg > FALL_TILT_DEG:
            fell = True
            break

    if viewer is not None:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
        viewer.close()

    return EpisodeResult(
        telemetry=telemetry_history,
        ground_truth=ground_truth_history,
        fell=fell,
        steps_survived=steps_survived,
        scenario=scenario,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Ground-truth helpers (MuJoCo-state readers; kept private to this module
# since sherpaos.evaluation.ground_truth itself must not import mujoco).
# ---------------------------------------------------------------------------


def _tilt_from_vertical_deg(quat_wxyz: np.ndarray) -> float:
    """Angle (degrees) between the pelvis's local +Z axis (rotated by
    `quat_wxyz`) and world +Z -- a single scalar capturing combined
    roll+pitch lean, robust to the gimbal-lock edge cases separate
    roll/pitch Euler angles can hit near vertical.
    """
    w, x, y, z = (float(c) for c in quat_wxyz)
    up_x = 2.0 * (x * z + w * y)
    up_y = 2.0 * (y * z - w * x)
    up_z = 1.0 - 2.0 * (x * x + y * y)
    norm = float(np.sqrt(up_x * up_x + up_y * up_y + up_z * up_z)) + 1e-12
    cos_tilt = float(np.clip(up_z / norm, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_tilt)))


def _planted_foot_slip_mps(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_body_id: int,
    foot_geom_ids: set[int],
    floor_geom_id: int,
) -> float:
    """Horizontal speed (m/s) of `foot_body_id`, but only if MuJoCo
    currently reports an active contact between one of `foot_geom_ids`
    and `floor_geom_id` (i.e. the foot is actually planted this step).
    A lifted/swinging foot moving isn't "slipping", so this returns 0.0
    when there is no such contact.
    """
    planted = False
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if (g1 in foot_geom_ids and g2 == floor_geom_id) or (
            g2 in foot_geom_ids and g1 == floor_geom_id
        ):
            planted = True
            break
    if not planted:
        return 0.0

    vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, foot_body_id, vel, 0)
    linear_xy = vel[3:5]
    return float(np.linalg.norm(linear_xy))
