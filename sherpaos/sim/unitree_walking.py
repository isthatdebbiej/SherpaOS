"""Standalone rollout for Unitree's pinned 12-DOF G1 walking policy.

This module is observational only. It drives the upstream policy and publishes
contract telemetry, but does not yet pass a SherpaOS guard decision back into
the policy because zero velocity commands still produce marching in place.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from sherpaos.contracts import RobotTelemetry
from sherpaos.sim.g1_sensors import build_sensorized_scene
from sherpaos.sim.gestures import GestureCue, gesture_at

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "unitree" / "g1_walking.yaml"

TelemetryObserver = Callable[[RobotTelemetry], None]
FrameObserver = Callable[[mujoco.MjModel, mujoco.MjData, int], None]

# Optional arm articulation. The pinned policy only drives the 12 leg joints;
# when enabled we swap in the 23-DOF description (same legs, plus waist/arms)
# and hold the extra joints with a small fixed PD toward a battery-dependent
# pose -- neutral (matches the frozen 12-DOF arm mesh) when charged, lowered
# when low on charge, both as an energy-saving posture and a visual signal.
# Order: waist_yaw, then left/right shoulder_pitch/roll/yaw, elbow, wrist_roll.
_ARM_NEUTRAL_ANGLES = np.zeros(11, dtype=np.float32)
_ARM_LOWERED_ANGLES = np.array(
    [0.0, 1.55, 0.35, 0.0, 0.55, 0.0, 1.55, -0.35, 0.0, 0.55, 0.0],
    dtype=np.float32,
)
_ARM_KP = 45.0
_ARM_KD = 2.0
_ARM_LOWER_BATTERY_HIGH = 0.35
_ARM_LOWER_BATTERY_LOW = 0.15


def _arm_target_angles(battery_fraction: float) -> np.ndarray:
    span = _ARM_LOWER_BATTERY_HIGH - _ARM_LOWER_BATTERY_LOW
    weight = (_ARM_LOWER_BATTERY_HIGH - battery_fraction) / span
    weight = float(np.clip(weight, 0.0, 1.0))
    return _ARM_NEUTRAL_ANGLES + weight * (_ARM_LOWERED_ANGLES - _ARM_NEUTRAL_ANGLES)


@dataclass(frozen=True, slots=True)
class SimulatedWalkingAuxiliary:
    """Simulator-only display data; never part of RobotTelemetry or policy input."""

    actual_speed_mps: float
    contacts: dict[str, bool]
    forces_n: dict[str, float]
    battery_fraction: float
    battery_voltage_v: float
    battery_current_a: float
    battery_temperature_c: float
    electrical_power_w: float


AuxiliaryObserver = Callable[[SimulatedWalkingAuxiliary], None]


class _SimulatedBattery:
    """Small deterministic pack model for walking-demo display telemetry."""

    def __init__(self, initial_fraction: float, temperature_c: float):
        self._fraction = float(np.clip(initial_fraction, 0.0, 1.0))
        self._temperature_c = temperature_c
        self._energy_used_wh = 0.0
        self._previous_time: float | None = None

    def update(self, time_s: float, mechanical_power_w: float) -> tuple[float, float, float, float]:
        electrical_power_w = mechanical_power_w / 0.30 + 100.0
        dt = 0.0 if self._previous_time is None else max(0.0, time_s - self._previous_time)
        self._previous_time = time_s
        self._energy_used_wh += electrical_power_w * dt / 3600.0
        self._fraction = max(0.0, self._fraction - electrical_power_w * dt / (500.0 * 3600.0))
        open_circuit_voltage = 38.0 + 16.0 * self._fraction
        current_a = electrical_power_w / max(open_circuit_voltage, 1.0)
        voltage_v = max(0.0, open_circuit_voltage - 0.05 * current_a)
        return self._fraction, voltage_v, current_a, electrical_power_w


@dataclass(frozen=True, slots=True)
class UnitreeWalkingResult:
    control_steps: int
    elapsed_seconds: float
    planar_distance_m: float
    fell: bool
    uphill_slope_deg: float


def run_unitree_walking_episode(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    max_steps: int | None = None,
    live_viewer: bool = False,
    telemetry_observer: TelemetryObserver | None = None,
    frame_observer: FrameObserver | None = None,
    frame_stride_controls: int = 2,
    simulated_auxiliary_observer: AuxiliaryObserver | None = None,
    simulated_auxiliary_temperature_c: float = 20.0,
    simulated_battery_initial_fraction: float = 1.0,
    terrain_slope_deg: float = 0.0,
    scenic_environment: bool = False,
    arms_enabled: bool = False,
    gesture_schedule: tuple[GestureCue, ...] | None = None,
) -> UnitreeWalkingResult:
    """Run the pinned Unitree policy and publish one sample per control tick."""
    config = _load_config(config_path)
    source_xml = _resolve_path(config_path, config["source_xml"])
    policy_path = _resolve_path(config_path, config["policy_path"])
    expected_hash = str(config["policy_sha256"])
    if _sha256(policy_path) != expected_hash:
        raise ValueError("Unitree policy hash does not match configs/unitree/g1_walking.yaml")

    arms_enabled = arms_enabled or gesture_schedule is not None
    scene = _build_scene(source_xml, scenic_environment, arms_enabled)
    model, data = scene.model, scene.data
    model.opt.timestep = float(config["simulation_dt_s"])
    _set_uphill_terrain(model, data, terrain_slope_deg)
    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()

    actions = np.zeros(int(config["num_actions"]), dtype=np.float32)
    default_angles = np.asarray(config["default_angles"], dtype=np.float32)
    target_dof_position = default_angles.copy()
    command = np.asarray(config["commanded_velocity_mps"], dtype=np.float32)
    kps = np.asarray(config["kps"], dtype=np.float32)
    kds = np.asarray(config["kds"], dtype=np.float32)
    control_decimation = int(config["control_decimation"])
    control_steps = int(config["default_max_control_steps"] if max_steps is None else max_steps)
    if control_steps < 1:
        raise ValueError("max_steps must be positive")
    if frame_stride_controls < 1:
        raise ValueError("frame_stride_controls must be positive")

    start_xy = data.qpos[:2].copy()
    foot_geom_ids = _foot_geom_ids(model)
    battery = _SimulatedBattery(
        simulated_battery_initial_fraction, simulated_auxiliary_temperature_c
    )
    target_arm_position = _ARM_NEUTRAL_ANGLES.copy()
    viewer = mujoco.viewer.launch_passive(model, data) if live_viewer else None
    try:
        for physics_step in range(control_steps * control_decimation):
            wall_started_at = time.perf_counter()
            torque = (target_dof_position - data.qpos[7:19]) * kps - data.qvel[6:18] * kds
            data.ctrl[:12] = torque
            if arms_enabled:
                arm_torque = (
                    (target_arm_position - data.qpos[19:30]) * _ARM_KP
                    - data.qvel[18:29] * _ARM_KD
                )
                data.ctrl[12:23] = arm_torque
            mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
                remaining_time = model.opt.timestep - (time.perf_counter() - wall_started_at)
                if remaining_time > 0.0:
                    time.sleep(remaining_time)

            if (physics_step + 1) % control_decimation:
                continue

            control_step = physics_step // control_decimation
            observation = _policy_observation(
                data=data,
                action=actions,
                command=command,
                default_angles=default_angles,
                time_s=float(data.time),
                config=config,
            )
            with torch.inference_mode():
                actions = policy(torch.from_numpy(observation).unsqueeze(0)).numpy().squeeze()
            target_dof_position = actions * float(config["action_scale"]) + default_angles

            sample = scene.suite.low_state(
                sequence=control_step,
                monotonic_time=float(data.time),
                commanded_velocity=command,
                gait_mode="walking",
            )
            if simulated_auxiliary_observer is not None or arms_enabled:
                auxiliary = None
                try:
                    auxiliary = _simulated_auxiliary(
                        model,
                        data,
                        foot_geom_ids,
                        battery,
                        simulated_auxiliary_temperature_c,
                    )
                except Exception:
                    pass
                if auxiliary is not None and simulated_auxiliary_observer is not None:
                    try:
                        simulated_auxiliary_observer(auxiliary)
                    except Exception:
                        pass
                if auxiliary is not None and arms_enabled and gesture_schedule is None:
                    target_arm_position = _arm_target_angles(auxiliary.battery_fraction)
            if gesture_schedule is not None:
                target_arm_position, _name, _label, _level = gesture_at(
                    float(data.time), gesture_schedule
                )
            if telemetry_observer is not None:
                try:
                    telemetry_observer(sample)
                except Exception:
                    pass
            if frame_observer is not None and control_step % frame_stride_controls == 0:
                try:
                    frame_observer(model, data, control_step)
                except Exception:
                    pass

        xy_delta = data.qpos[:2] - start_xy
        tilt_deg = _tilt_deg(data.qpos[3:7])
        return UnitreeWalkingResult(
            control_steps=control_steps,
            elapsed_seconds=float(data.time),
            planar_distance_m=float(np.linalg.norm(xy_delta)),
            fell=bool(data.qpos[2] < 0.3 or tilt_deg > 50.0),
            uphill_slope_deg=float(terrain_slope_deg),
        )
    finally:
        if viewer is not None:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
            viewer.close()


def _load_config(config_path: Path) -> dict:
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("walking config must be a mapping")
    return config


def _resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    resolved = path if path.is_absolute() else _REPO_ROOT / path
    if not resolved.is_file():
        raise FileNotFoundError(f"walking asset is unavailable: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_scene(source_xml: Path, scenic_environment: bool, arms_enabled: bool = False):
    """Build the walking scene, optionally swapping in the 23-DOF (arms) description.

    ``g1_23dof.xml`` bundles its own floor/skybox/lighting (unlike ``g1_12dof.xml``,
    which relies on ``scene.xml`` for those), so the arm-enabled path reads it
    directly instead of routing through ``scene.xml`` -- combining both would
    declare the "groundplane" texture/material twice and fail to compile.
    """
    if not scenic_environment and not arms_enabled:
        return build_sensorized_scene(source_xml)
    base_path = source_xml.parent / "g1_23dof.xml" if arms_enabled else source_xml
    scene = base_path.read_text(encoding="utf-8")
    if scenic_environment:
        scene = scene.replace(
            '<texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" '
            'width="512" height="3072"/>',
            '<texture type="skybox" builtin="gradient" rgb1="0.55 0.72 0.88" '
            'rgb2="0.04 0.10 0.19" width="512" height="3072"/>',
        ).replace(
            'rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"',
            'rgb1="0.84 0.90 0.95" rgb2="0.60 0.72 0.82"',
        )
        himalaya_lights = (
            '<light directional="true" pos="-4 -5 9" dir="0.35 0.25 -1" diffuse="1 0.94 0.84"/>\n'
            '    <light directional="true" pos="4 3 6" dir="-0.5 -0.25 -1" '
            'diffuse="0.32 0.48 0.72"/>'
        )
        if arms_enabled:
            scene = scene.replace(
                '<light pos="1 0 3.5" dir="0 0 -1" directional="true"/>',
                f'{himalaya_lights}\n    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>',
            )
        else:
            scene = scene.replace("  <worldbody>", f"  <worldbody>\n    {himalaya_lights}")
    with NamedTemporaryFile(
        mode="w", suffix=".xml", prefix=".sherpa_himalaya_", dir=base_path.parent, delete=False
    ) as temporary:
        temporary.write(scene)
        temporary_path = Path(temporary.name)
    try:
        return build_sensorized_scene(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _policy_observation(
    *,
    data: mujoco.MjData,
    action: np.ndarray,
    command: np.ndarray,
    default_angles: np.ndarray,
    time_s: float,
    config: dict,
) -> np.ndarray:
    qj = (data.qpos[7:19] - default_angles) * float(config["dof_pos_scale"])
    dqj = data.qvel[6:18] * float(config["dof_vel_scale"])
    quaternion = data.qpos[3:7]
    gravity_orientation = np.array(
        [
            2.0 * (-quaternion[3] * quaternion[1] + quaternion[0] * quaternion[2]),
            -2.0 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1]),
            1.0 - 2.0 * (quaternion[0] ** 2 + quaternion[3] ** 2),
        ],
        dtype=np.float32,
    )
    phase = (time_s % float(config["gait_period_s"])) / float(config["gait_period_s"])
    return np.concatenate(
        (
            data.qvel[3:6] * float(config["ang_vel_scale"]),
            gravity_orientation,
            command * np.asarray(config["cmd_scale"], dtype=np.float32),
            qj,
            dqj,
            action,
            np.array([math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)]),
        ),
        dtype=np.float32,
    )


def _tilt_deg(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    up_z = 1.0 - 2.0 * (x * x + y * y)
    norm = math.sqrt((2.0 * (x * z + w * y)) ** 2 + (2.0 * (y * z - w * x)) ** 2 + up_z**2)
    return math.degrees(math.acos(float(np.clip(up_z / max(norm, 1.0e-12), -1.0, 1.0))))


def _set_uphill_terrain(model: mujoco.MjModel, data: mujoco.MjData, slope_deg: float) -> None:
    """Rotate the MuJoCo floor so positive world-X walking climbs the requested grade."""
    slope_deg = float(slope_deg)
    if not np.isfinite(slope_deg) or not 0.0 <= slope_deg <= 20.0:
        raise ValueError("terrain_slope_deg must be finite and between 0 and 20 degrees")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("walking model has no 'floor' geom")
    half_angle = -math.radians(slope_deg) / 2.0
    model.geom_quat[floor_id] = [math.cos(half_angle), 0.0, math.sin(half_angle), 0.0]
    mujoco.mj_forward(model, data)


def _foot_geom_ids(model: mujoco.MjModel) -> dict[str, set[int]]:
    foot_body_names = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}
    foot_geoms: dict[str, set[int]] = {}
    for side, body_name in foot_body_names.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"walking model has no '{body_name}' body")
        foot_geoms[side] = {
            geom_id for geom_id in range(model.ngeom) if model.geom_bodyid[geom_id] == body_id
        }
    return foot_geoms


def _simulated_auxiliary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_geom_ids: dict[str, set[int]],
    battery: _SimulatedBattery,
    temperature_c: float,
) -> SimulatedWalkingAuxiliary:
    forces_n = {"left": 0.0, "right": 0.0}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, contact_index, force)
        for side, geom_ids in foot_geom_ids.items():
            if contact.geom1 in geom_ids or contact.geom2 in geom_ids:
                forces_n[side] += abs(float(force[0]))
    mechanical_power_w = float(np.abs(data.ctrl * data.qvel[6:]).sum())
    fraction, voltage_v, current_a, electrical_power_w = battery.update(
        float(data.time), mechanical_power_w
    )
    return SimulatedWalkingAuxiliary(
        actual_speed_mps=float(np.linalg.norm(data.qvel[:2])),
        contacts={side: force > 1.0 for side, force in forces_n.items()},
        forces_n=forces_n,
        battery_fraction=fraction,
        battery_voltage_v=voltage_v,
        battery_current_a=current_a,
        battery_temperature_c=temperature_c,
        electrical_power_w=electrical_power_w,
    )