"""Controller-only v26 rollouts on the full Menagerie G1 Himalayan scene."""

from __future__ import annotations

import tempfile
from collections import deque
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from sherpaos.evaluation.ground_truth import ScenarioGroundTruth, classify_unsafe
from sherpaos.sim import disturbances
from sherpaos.sim.battery import enrich_battery_telemetry
from sherpaos.sim.himalaya_scene import scene_xml, terrain_slope_for_geom
from sherpaos.sim.mujoco_source import MuJoCoTelemetrySource
from sherpaos.sim.runner import FALL_PELVIS_Z_M, FALL_TILT_DEG, EpisodeResult
from sherpaos.sim.scenario import Scenario
from sherpaos.sim.v26_playground import (
    DEFAULT_POSE,
    JOINTS,
    KD,
    KP,
    V26ObservationHistory,
    action_target,
    projected_gravity,
)
from sherpaos.sim.weather import aerodynamic_force_n, wind_speed_at_step

TERRAIN_GEOMS = (
    "spawn_apron",
    "terrain_segment_0",
    "terrain_segment_1",
    "terrain_segment_2",
    "terrain_segment_3",
    "crust_a",
    "crust_b",
    "rock_step",
)
LEFT_FOOT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_BODY = "right_ankle_roll_link"


def run_v26_episode(
    scenario: Scenario,
    seed: int,
    *,
    policy_path: Path,
    g1_dir: Path,
    max_steps: int = 500,
    command: tuple[float, float, float] = (0.4, 0.0, 0.0),
    terrain_zone: int = 0,
    wind_target_mps: float = 0.0,
) -> EpisodeResult:
    """Run frozen v26 with no SherpaOS intervention and return aligned traces."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for v26 generation") from exc
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=g1_dir, delete=False) as handle:
        handle.write(scene_xml(terrain_zone))
        scene_path = Path(handle.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    finally:
        scene_path.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key)
    qadr = np.array([_qpos_address(model, name) for name in JOINTS])
    aids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINTS]
    )
    data.qpos[qadr] = DEFAULT_POSE
    target_health = float(np.clip(scenario.actuator_health, 0.05, 1.0))
    for index, aid in enumerate(aids):
        model.actuator_gainprm[aid, 0] = KP[index]
        model.actuator_biasprm[aid, 1] = -KP[index]
        model.actuator_biasprm[aid, 2] = -KD[index]
    terrain_ids = {_geom_id(model, name) for name in TERRAIN_GEOMS}
    terrain_ids.discard(-1)
    if scenario.friction < 0.99:
        for geom_id in terrain_ids:
            model.geom_friction[geom_id, 0] = scenario.friction
    mujoco.mj_forward(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    feet = (_body_geoms(model, LEFT_FOOT_BODY), _body_geoms(model, RIGHT_FOOT_BODY))
    if scenario.friction < 0.99:
        for foot_geoms in feet:
            for geom_id in foot_geoms:
                model.geom_friction[geom_id, 0] = scenario.friction
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape[-1] != 240 or session.get_outputs()[0].shape[-1] != 12:
        raise ValueError("unexpected v26 ONNX input/output contract")
    history, previous_target = V26ObservationHistory(), None
    source, telemetry, truth = MuJoCoTelemetrySource(), [], []
    physics_step = 0
    slip_history: deque[float] = deque(maxlen=5)
    fell = False
    noise_rng = np.random.default_rng(seed)
    for step in range(max_steps):
        health = 1.0
        if target_health < 1.0 and step >= 200:
            alpha = min(1.0, (step - 200) / 50.0)
            health = 1.0 + alpha * (target_health - 1.0)
        for index, aid in enumerate(aids):
            model.actuator_gainprm[aid, 0] = KP[index] * health
            model.actuator_biasprm[aid, 1] = -KP[index] * health
            model.actuator_biasprm[aid, 2] = -KD[index] * health
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis, velocity, 1)
        policy_observation = history.build(
            data.qpos[qadr],
            data.qpos[3:7],
            velocity[:3],
            np.array([*command, 0.0], dtype=np.float32),
        )
        action = session.run(None, {session.get_inputs()[0].name: policy_observation})[0][0]
        history.record_action(action)
        target = action_target(action)
        if step < 25:
            alpha = (step + 1) / 25
            target = (1 - alpha) * data.qpos[qadr] + alpha * target
        if previous_target is not None:
            target = np.clip(target, previous_target - 0.2, previous_target + 0.2)
        previous_target = target.copy()
        data.ctrl[aids] = target
        wind_speed_mps = wind_speed_at_step(step, wind_target_mps)
        wind_force_n = aerodynamic_force_n(wind_speed_mps)
        disturbance_active = wind_force_n > 0.0
        for _ in range(10):
            if disturbances.disturbance_active_at_step(scenario, physics_step):
                disturbance_active = True
            disturbances.apply_disturbance_wrench(model, data, scenario, physics_step)
            data.xfrc_applied[pelvis, 1] += wind_force_n
            mujoco.mj_step(model, data)
            physics_step += 1
        sample = source.sample(
            model,
            data,
            sequence=step,
            monotonic_time=float(data.time),
            speed_scale=1.0,
            gait_mode="walking",
        )
        sample = replace(sample, commanded_velocity=np.asarray(command, dtype=float))
        sample = enrich_battery_telemetry(
            sample, scenario, elapsed_seconds=float(data.time), speed_scale=1.0
        )
        if scenario.sensor_noise_std:
            sample = disturbances.inject_sensor_noise(sample, scenario.sensor_noise_std, noise_rng)
        telemetry.append(sample)
        gravity = projected_gravity(data.qpos[3:7])
        tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))))
        contact_geoms = _contacting_terrain_geoms(data, feet, terrain_ids)
        instantaneous_slip = max(_foot_slip(model, data, foot, terrain_ids) for foot in feet)
        slip_history.append(instantaneous_slip)
        true_friction = min(
            (float(model.geom_friction[geom_id, 0]) for geom_id in contact_geoms),
            default=scenario.friction,
        )
        true_slope = max(
            (
                terrain_slope_for_geom(
                    terrain_zone,
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                )
                for geom_id in contact_geoms
            ),
            default=0.0,
        )
        slip = float(np.median(slip_history))
        truth.append(
            ScenarioGroundTruth(
                true_friction=true_friction,
                true_slope_deg=true_slope,
                disturbance_active=disturbance_active,
                actuator_health=health,
                tilt_from_vertical_deg=tilt,
                planted_foot_slip_mps=slip,
                true_unsafe=classify_unsafe(
                    tilt_from_vertical_deg=tilt,
                    actuator_health=health,
                    planted_foot_slip_mps=slip,
                    tilt_unsafe_deg=25.0,
                    actuator_health_unsafe=0.90,
                    foot_slip_unsafe_mps=0.8,
                ),
            )
        )
        if float(data.qpos[2]) < FALL_PELVIS_Z_M or tilt > FALL_TILT_DEG:
            fell = True
            break
    return EpisodeResult(telemetry, truth, fell, len(telemetry), scenario, seed)


def _qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        raise ValueError(f"missing G1 joint {name}")
    return int(model.jnt_qposadr[joint])


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _body_geoms(model: mujoco.MjModel, name: str) -> set[int]:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return {index for index in range(model.ngeom) if model.geom_bodyid[index] == body}


def _contacting_terrain_geoms(
    data: mujoco.MjData, feet: tuple[set[int], set[int]], terrain: set[int]
) -> set[int]:
    """Return authored terrain geoms physically touching either foot."""
    foot_geoms = feet[0] | feet[1]
    contacts: set[int] = set()
    for index in range(data.ncon):
        pair = {int(data.contact[index].geom1), int(data.contact[index].geom2)}
        if pair & foot_geoms:
            contacts.update(pair & terrain)
    return contacts


def _foot_slip(
    model: mujoco.MjModel, data: mujoco.MjData, foot_geoms: set[int], terrain: set[int]
) -> float:
    contacting = any(
        ({int(data.contact[index].geom1), int(data.contact[index].geom2)} & foot_geoms)
        and ({int(data.contact[index].geom1), int(data.contact[index].geom2)} & terrain)
        for index in range(data.ncon)
    )
    if not contacting:
        return 0.0
    body = int(model.geom_bodyid[next(iter(foot_geoms))])
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
    return float(np.linalg.norm(velocity[3:5]))
