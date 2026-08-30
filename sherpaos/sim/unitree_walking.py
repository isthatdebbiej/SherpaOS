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

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from sherpaos.contracts import RobotTelemetry
from sherpaos.sim.g1_sensors import build_sensorized_scene

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "unitree" / "g1_walking.yaml"

TelemetryObserver = Callable[[RobotTelemetry], None]


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
    simulated_auxiliary_observer: AuxiliaryObserver | None = None,
    simulated_auxiliary_temperature_c: float = 20.0,
    simulated_battery_initial_fraction: float = 1.0,
    terrain_slope_deg: float = 0.0,
) -> UnitreeWalkingResult:
    """Run the pinned Unitree policy and publish one sample per control tick."""
    config = _load_config(config_path)
    source_xml = _resolve_path(config_path, config["source_xml"])
    policy_path = _resolve_path(config_path, config["policy_path"])
    expected_hash = str(config["policy_sha256"])
    if _sha256(policy_path) != expected_hash:
        raise ValueError("Unitree policy hash does not match configs/unitree/g1_walking.yaml")

    scene = build_sensorized_scene(source_xml)
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

    start_xy = data.qpos[:2].copy()
    foot_geom_ids = _foot_geom_ids(model)
    battery = _SimulatedBattery(
        simulated_battery_initial_fraction, simulated_auxiliary_temperature_c
    )
    viewer = mujoco.viewer.launch_passive(model, data) if live_viewer else None
    try:
        for physics_step in range(control_steps * control_decimation):
            wall_started_at = time.perf_counter()
            torque = (target_dof_position - data.qpos[7:]) * kps - data.qvel[6:] * kds
            data.ctrl[:] = torque
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
            if simulated_auxiliary_observer is not None:
                try:
                    simulated_auxiliary_observer(
                        _simulated_auxiliary(
                            model,
                            data,
                            foot_geom_ids,
                            battery,
                            simulated_auxiliary_temperature_c,
                        )
                    )
                except Exception:
                    pass
            if telemetry_observer is not None:
                try:
                    telemetry_observer(sample)
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


def _policy_observation(
    *,
    data: mujoco.MjData,
    action: np.ndarray,
    command: np.ndarray,
    default_angles: np.ndarray,
    time_s: float,
    config: dict,
) -> np.ndarray:
    qj = (data.qpos[7:] - default_angles) * float(config["dof_pos_scale"])
    dqj = data.qvel[6:] * float(config["dof_vel_scale"])
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