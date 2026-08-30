"""Onboard-observable MuJoCo sensor suite for supported G1 MJCF variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import mujoco
import numpy as np

from sherpaos.contracts import RobotTelemetry, TelemetrySource

_PREFIX = "sherpa_"
_PELVIS_BODY = "pelvis"


@dataclass(slots=True)
class G1SensorSuite:
    """Read only encoder, actuator, and pelvis-IMU observations from a model."""

    model: mujoco.MjModel
    data: mujoco.MjData
    joint_count: int

    def low_state(
        self,
        *,
        sequence: int,
        monotonic_time: float,
        commanded_velocity: np.ndarray | None = None,
        gait_mode: str | None = None,
        valid: bool = True,
    ) -> RobotTelemetry:
        """Return one adapter-agnostic, onboard-observable telemetry sample."""
        joint_position = np.array(
            [
                self._read(f"{_PREFIX}joint_position_{index}")[0]
                for index in range(self.joint_count)
            ],
            dtype=float,
        )
        joint_velocity = np.array(
            [
                self._read(f"{_PREFIX}joint_velocity_{index}")[0]
                for index in range(self.joint_count)
            ],
            dtype=float,
        )
        joint_effort = np.array(
            [self._read(f"{_PREFIX}joint_effort_{index}")[0] for index in range(self.joint_count)],
            dtype=float,
        )
        return RobotTelemetry(
            monotonic_time=float(monotonic_time),
            source_time=float(monotonic_time),
            sequence=int(sequence),
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            joint_effort=joint_effort,
            base_orientation=self._read(f"{_PREFIX}pelvis_orientation"),
            base_angular_velocity=self._read(f"{_PREFIX}pelvis_angular_velocity"),
            base_linear_acceleration=self._read(f"{_PREFIX}pelvis_linear_acceleration"),
            commanded_velocity=(
                None
                if commanded_velocity is None
                else np.asarray(commanded_velocity, dtype=float).copy()
            ),
            gait_mode=gait_mode,
            source=TelemetrySource.SIM,
            valid=valid,
            field_provenance={
                "joint_position": "mujoco_sensor:jointpos",
                "joint_velocity": "mujoco_sensor:jointvel",
                "joint_effort": "mujoco_sensor:actuatorfrc",
                "base_orientation": "mujoco_sensor:framequat",
                "base_angular_velocity": "mujoco_sensor:frameangvel",
                "base_linear_acceleration": "mujoco_sensor:framelinacc",
            },
        )

    def _read(self, name: str) -> np.ndarray:
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise ValueError(f"sensorized model has no sensor named '{name}'")
        address = int(self.model.sensor_adr[sensor_id])
        dimension = int(self.model.sensor_dim[sensor_id])
        return np.array(self.data.sensordata[address : address + dimension], dtype=float, copy=True)


@dataclass(slots=True)
class SensorizedScene:
    """Compiled model/data pair and its bound observable sensor suite."""

    model: mujoco.MjModel
    data: mujoco.MjData
    suite: G1SensorSuite


def build_sensorized_scene(src_xml: str | Path) -> SensorizedScene:
    """Compile ``src_xml`` with explicit observable G1 sensors attached.

    The source asset is never modified. A short-lived wrapper preserves the
    original relative includes while supplying encoder, actuator-force, and
    pelvis frame sensors for both Unitree's 12-DOF and Menagerie's 29-DOF G1.
    """
    source = Path(src_xml).resolve()
    base_model = mujoco.MjModel.from_xml_path(str(source))
    actuator_joint_names = _actuator_joint_names(base_model)
    if mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_BODY, _PELVIS_BODY) < 0:
        raise ValueError(f"model has no '{_PELVIS_BODY}' body for IMU sensors")

    sensor_lines = [
        "<framequat name=\"sherpa_pelvis_orientation\" objtype=\"body\" objname=\"pelvis\"/>",
        (
            "<frameangvel name=\"sherpa_pelvis_angular_velocity\" "
            "objtype=\"body\" objname=\"pelvis\"/>"
        ),
        (
            "<framelinacc name=\"sherpa_pelvis_linear_acceleration\" "
            "objtype=\"body\" objname=\"pelvis\"/>"
        ),
        (
            "<frameangacc name=\"sherpa_pelvis_angular_acceleration\" "
            "objtype=\"body\" objname=\"pelvis\"/>"
        ),
    ]
    if base_model.nsensor == 0:
        sensor_lines.append('<clock name="sherpa_control_clock"/>')
    for index, (actuator_name, joint_name) in enumerate(actuator_joint_names):
        sensor_lines.extend(
            (
                f'<jointpos name="{_PREFIX}joint_position_{index}" joint="{joint_name}"/>',
                f'<jointvel name="{_PREFIX}joint_velocity_{index}" joint="{joint_name}"/>',
                f'<actuatorfrc name="{_PREFIX}joint_effort_{index}" actuator="{actuator_name}"/>',
            )
        )

    wrapper = "\n".join(
        (
            '<mujoco model="sherpa_sensorized_g1">',
            f'  <include file="{source.as_posix()}"/>',
            "  <sensor>",
            *[f"    {line}" for line in sensor_lines],
            "  </sensor>",
            "</mujoco>",
        )
    )
    with NamedTemporaryFile(
        mode="w", suffix=".xml", prefix=".sherpa_sensorized_", dir=source.parent, delete=False
    ) as temporary:
        temporary.write(wrapper)
        temporary_path = Path(temporary.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(temporary_path))
    finally:
        temporary_path.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    suite = G1SensorSuite(model=model, data=data, joint_count=len(actuator_joint_names))
    return SensorizedScene(model=model, data=data, suite=suite)


def _actuator_joint_names(model: mujoco.MjModel) -> list[tuple[str, str]]:
    names: list[tuple[str, str]] = []
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if actuator_name is None or joint_name is None:
            raise ValueError(f"actuator {actuator_id} lacks a named joint transmission")
        names.append((actuator_name, joint_name))
    return names