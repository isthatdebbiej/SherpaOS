"""Per-episode and per-step `Scenario` effects applied to a live MuJoCo
model/data pair, plus telemetry-level noise injection and adversarial
telemetry-corruption helpers other lanes' tests can use.

Two effect classes, applied differently by `sherpaos/sim/runner.py`:
  - **Static, per-episode** (`apply_scenario_to_model`): floor/foot
    traction, ground slope (via gravity tilt), actuator authority. Call
    once at episode setup, on a freshly loaded `mujoco.MjModel` (this
    module mutates `model` in place and does not track/undo previous
    calls, so reusing one `MjModel` object across scenarios without
    reloading it would accumulate scaling errors -- `runner.py` avoids
    this by loading a fresh model per episode).
  - **Per-step** (`apply_disturbance_wrench`): the external pelvis
    wrench during the scenario's disturbance window. Call every physics
    step; it always writes `data.xfrc_applied` (zeroing it outside the
    window) so a wrench never lingers past its configured duration.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from sherpaos.contracts import RobotTelemetry
from sherpaos.sim.scenario import Scenario

FLOOR_GEOM_NAME = "floor"
PELVIS_BODY_NAME = "pelvis"

__all__ = [
    "apply_scenario_to_model",
    "apply_disturbance_wrench",
    "disturbance_active_at_step",
    "inject_sensor_noise",
    "make_stale_telemetry",
    "make_nan_telemetry",
    "make_invalid_telemetry",
    "make_out_of_order_telemetry",
]


def apply_scenario_to_model(model: mujoco.MjModel, scenario: Scenario) -> None:
    """Mutate `model` in place for the static (per-episode) parts of
    `scenario`: floor/foot traction, ground slope, actuator authority.
    """
    _set_friction(model, scenario.friction)
    _set_slope(model, scenario.slope_deg)
    _set_actuator_health(model, scenario.actuator_health)


def _set_friction(model: mujoco.MjModel, friction: float) -> None:
    """Set sliding friction on the floor geom AND every geom with
    `geom_priority == 1` (the G1's foot-sole contact points).

    Why both: MuJoCo resolves a contact's friction from whichever
    participating geom has the *higher* `geom_priority` (not an
    average/max of both) when priorities differ. The vendored G1 XML
    gives each foot's 4 corner contact spheres `priority=1` (friction
    0.6, fixed) while the floor geom is `priority=0` -- so setting only
    the floor geom's friction has **no effect at all** on foot/ground
    contact behavior. This was verified empirically while tuning this
    module: episodes were bit-for-bit identical from `friction=1.0` down
    to `friction=0.03` until the foot-corner geoms' friction was also
    changed. Only the sliding-friction component (`friction[0]`) is
    touched; torsional/rolling friction stay at their authored defaults.
    """
    friction = float(friction)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    if floor_id < 0:
        raise ValueError(f"model has no geom named '{FLOOR_GEOM_NAME}'")
    model.geom_friction[floor_id, 0] = friction
    for gid in range(model.ngeom):
        if model.geom_priority[gid] == 1:
            model.geom_friction[gid, 0] = friction


def _set_slope(model: mujoco.MjModel, slope_deg: float) -> None:
    """Approximate a ground slope by tilting the gravity vector rather
    than remodeling the floor geometry: for a flat floor, walking on a
    `slope_deg` incline under vertical gravity is kinematically and
    dynamically equivalent (in the robot's own frame) to walking on a
    flat floor with gravity tilted `slope_deg` off vertical. This is a
    single vector write with no geometry/contact-normal remodeling.
    Rotation is about the world Y axis (a pitch-only incline).
    """
    theta = np.radians(float(slope_deg))
    g = 9.81
    model.opt.gravity[:] = [g * np.sin(theta), 0.0, -g * np.cos(theta)]


def _set_actuator_health(model: mujoco.MjModel, actuator_health: float) -> None:
    """Scale actuator authority by `actuator_health` (1.0 = nominal).

    The G1's actuators are MuJoCo `<position>` actuators:
    `gainprm[:, 0]` is kp, `biasprm[:, 1]` is -kp, `biasprm[:, 2]` is -kv
    (joint-specific built-in damping). Scaling all three by the same
    factor keeps the actuator's natural frequency/damping ratio
    unchanged while proportionally reducing how hard it can push --
    degrading torque authority, not literally breaking/faulting the
    actuator, matching this field's "degraded gain scaling" framing.
    """
    health = float(np.clip(actuator_health, 0.0, 1.0))
    model.actuator_gainprm[:, 0] *= health
    model.actuator_biasprm[:, 1] *= health
    model.actuator_biasprm[:, 2] *= health


def apply_disturbance_wrench(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario: Scenario,
    physics_step_idx: int,
) -> None:
    """Apply (or clear) the scenario's external wrench on the pelvis for
    this physics step. Call every physics step, before `mujoco.mj_step`.
    """
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY_NAME)
    data.xfrc_applied[pelvis_id, :] = 0.0

    if not disturbance_active_at_step(scenario, physics_step_idx):
        return
    direction = scenario.disturbance_direction
    if direction is None:
        return
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return
    unit_direction = direction / norm
    data.xfrc_applied[pelvis_id, 0:3] = unit_direction * scenario.disturbance_force_n


def disturbance_active_at_step(scenario: Scenario, physics_step_idx: int) -> bool:
    """Whether `scenario`'s disturbance window covers `physics_step_idx`
    (a physics-step index, at the model's native timestep).
    """
    if scenario.disturbance_force_n <= 0.0:
        return False
    if scenario.disturbance_start_step is None or scenario.disturbance_duration_steps is None:
        return False
    start = scenario.disturbance_start_step
    end = start + scenario.disturbance_duration_steps
    return start <= physics_step_idx < end


def inject_sensor_noise(
    telemetry: RobotTelemetry, std: float, rng: np.random.Generator
) -> RobotTelemetry:
    """Return a copy of `telemetry` with i.i.d. Gaussian noise (std=`std`)
    added to the channels a real noisy sensor would actually corrupt:
    angular velocity, linear acceleration, and joint velocity. Joint
    *position* (encoders, typically much higher precision) and every
    other field is left untouched. `std <= 0` returns `telemetry`
    unchanged (no-op, not even a copy).
    """
    if std <= 0.0:
        return telemetry
    return dataclasses.replace(
        telemetry,
        base_angular_velocity=telemetry.base_angular_velocity
        + rng.normal(0.0, std, size=telemetry.base_angular_velocity.shape),
        base_linear_acceleration=telemetry.base_linear_acceleration
        + rng.normal(0.0, std, size=telemetry.base_linear_acceleration.shape),
        joint_velocity=telemetry.joint_velocity
        + rng.normal(0.0, std, size=telemetry.joint_velocity.shape),
    )


# ---------------------------------------------------------------------------
# Adversarial telemetry corruption helpers, for other lanes' robustness
# tests (e.g. AGENTS.md safety constraint #3: stale/malformed/missing/NaN/
# out-of-order telemetry must fail conservatively). Each takes a valid
# `RobotTelemetry` and returns a corrupted copy; `RobotTelemetry` is a
# frozen dataclass so none of these mutate the input.
# ---------------------------------------------------------------------------


def make_stale_telemetry(telemetry: RobotTelemetry, age_seconds: float) -> RobotTelemetry:
    """Return a copy whose `monotonic_time` is `age_seconds` further in
    the past, so `result.age_seconds(telemetry.monotonic_time) >= age_seconds`.
    """
    return dataclasses.replace(telemetry, monotonic_time=telemetry.monotonic_time - age_seconds)


def make_nan_telemetry(
    telemetry: RobotTelemetry, fields: tuple[str, ...] = ("joint_position",)
) -> RobotTelemetry:
    """Return a copy with NaN written into the given numpy-array fields
    (any field that is `None` on the input, e.g. an already-absent
    `joint_effort`, is silently skipped).
    """
    updates: dict[str, np.ndarray] = {}
    for name in fields:
        value = getattr(telemetry, name)
        if value is None:
            continue
        corrupted = np.array(value, dtype=float, copy=True)
        corrupted[:] = np.nan
        updates[name] = corrupted
    return dataclasses.replace(telemetry, **updates)


def make_invalid_telemetry(telemetry: RobotTelemetry) -> RobotTelemetry:
    """Return a copy with `valid=False` (simulates an adapter marking a
    sample untrustworthy, e.g. a dropped/garbled read).
    """
    return dataclasses.replace(telemetry, valid=False)


def make_out_of_order_telemetry(
    telemetry: RobotTelemetry, sequence_delta: int = -5
) -> RobotTelemetry:
    """Return a copy with `sequence` shifted by `sequence_delta` (default:
    -5, i.e. backwards), simulating a reordered/duplicated sample arriving
    after a logically later one.
    """
    return dataclasses.replace(telemetry, sequence=telemetry.sequence + sequence_delta)
