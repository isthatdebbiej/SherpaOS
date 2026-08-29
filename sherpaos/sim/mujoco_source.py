"""The sole seam that reads live MuJoCo `model`/`data` state to build a
`sherpaos.contracts.RobotTelemetry` sample.

Per `docs/CONTRACTS.md`'s adapter boundary rule, everything downstream of
telemetry production must be adapter-agnostic; per this task's scope this
module is the *only* place in `sherpaos/sim` allowed to read `mujoco.MjData`
for the purpose of building a `RobotTelemetry`. `sherpaos/sim/runner.py`
owns the physics loop and passes this module already-known scalars
(sequence number, wall/sim time, the controller's current `speed_scale`
and gait mode); it never reaches back into `RobotTelemetry` construction
itself, and this module never reads `sherpaos.sim.scenario` or
`sherpaos.evaluation.ground_truth` (no simulator ground truth is available
here by construction -- see the module-level field-mapping notes below).

Field mapping decisions (documented once, here, since they're easy to get
subtly wrong):
  - `joint_position`/`joint_velocity`: `qpos[7:]`/`qvel[6:]`, the 29
    actuated hinge joints, skipping the floating base's 7 qpos / 6 qvel
    slots (`floating_base_joint` is a freejoint: 3 pos + 4 quat in qpos,
    3 linvel + 3 angvel in qvel). Joint order matches actuator order,
    which matches joint declaration order in the vendored G1 MJCF.
  - `joint_effort`: `data.actuator_force` (the scalar force/torque each
    of the 29 actuators actually output this step), **not**
    `data.qfrc_actuator`. `qfrc_actuator` is expressed in generalized
    (qvel-indexed) coordinates and would need remapping through the
    joint Jacobian to line up 1:1 with the 29 actuators; `actuator_force`
    is already actuator-indexed (shape `(29,)`, matching `joint_position`)
    so no remapping is needed.
  - `base_orientation`: `qpos[3:7]`, MuJoCo's native wxyz quaternion --
    matches the contract's documented wxyz convention directly.
  - `base_angular_velocity`/`base_linear_acceleration`: the **pelvis**
    IMU sensors (`imu-pelvis-angular-velocity` /
    `imu-pelvis-linear-acceleration`), not `qvel[3:6]`/a finite difference.
    A real G1 only exposes IMU readings, never privileged base twist, so
    reading the sensor here (rather than the more "convenient" qvel
    slice) keeps the sim adapter honest about what a live-G1 adapter
    could actually provide. The pelvis sensor (rather than torso) is used
    because `base_orientation` above is the *pelvis's* freejoint pose --
    keeping all three `base_*` fields anchored to the same physical body.
  - `commanded_velocity`: repurposed to carry the controller's current
    `speed_scale` as a 1-element array, e.g. `np.array([0.7])`. This
    posture/stepping task has no translational body-frame velocity
    command (no locomotion), so there is nothing else meaningful to put
    here; a 1-element array (rather than overloading it with 3 zeros) is
    the least-surprising way to expose the actuation-channel signal the
    guard controls without adding a non-contract field. Downstream
    consumers should treat `commanded_velocity[0]` as `speed_scale`.
  - `gait_mode`: `"stepping"` or `"hold"`, passed in by the caller (which
    already knows the guard's `hold` decision -- see `runner.py`).
  - `battery_fraction`/`battery_voltage`: always `None`. There is no
    battery model in this pass; fabricating a value would violate the
    "never invent data" spirit of `RobotTelemetry.valid`/
    `field_provenance`.
"""

from __future__ import annotations

import mujoco
import numpy as np

from sherpaos.contracts import RobotTelemetry, TelemetrySource

N_ACTUATED_JOINTS = 29
FREE_JOINT_QPOS_DOFS = 7  # 3 position + 4 quaternion
FREE_JOINT_QVEL_DOFS = 6  # 3 linear velocity + 3 angular velocity

DEFAULT_GYRO_SENSOR = "imu-pelvis-angular-velocity"
DEFAULT_ACCEL_SENSOR = "imu-pelvis-linear-acceleration"


class MuJoCoTelemetrySource:
    """Builds one `RobotTelemetry` sample per call from live sim state."""

    def __init__(
        self,
        gyro_sensor_name: str = DEFAULT_GYRO_SENSOR,
        accel_sensor_name: str = DEFAULT_ACCEL_SENSOR,
    ) -> None:
        self._gyro_sensor_name = gyro_sensor_name
        self._accel_sensor_name = accel_sensor_name

    def sample(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        sequence: int,
        monotonic_time: float,
        source_time: float | None = None,
        speed_scale: float = 1.0,
        gait_mode: str = "stepping",
        valid: bool = True,
    ) -> RobotTelemetry:
        """Read `model`/`data` and return one `RobotTelemetry` sample.

        `monotonic_time` and `source_time` are supplied by the caller
        (typically `data.time`, the elapsed simulated seconds) rather
        than read from `data` here, so this method's signature matches
        what a dump-replay or live-G1 adapter would also need to supply
        without depending on MuJoCo-specific state.
        """
        joint_position = np.array(
            data.qpos[FREE_JOINT_QPOS_DOFS : FREE_JOINT_QPOS_DOFS + N_ACTUATED_JOINTS],
            dtype=float,
            copy=True,
        )
        joint_velocity = np.array(
            data.qvel[FREE_JOINT_QVEL_DOFS : FREE_JOINT_QVEL_DOFS + N_ACTUATED_JOINTS],
            dtype=float,
            copy=True,
        )
        joint_effort = np.array(data.actuator_force, dtype=float, copy=True)

        base_orientation = np.array(data.qpos[3:7], dtype=float, copy=True)

        base_angular_velocity = _read_sensor(model, data, self._gyro_sensor_name)
        base_linear_acceleration = _read_sensor(model, data, self._accel_sensor_name)

        return RobotTelemetry(
            monotonic_time=monotonic_time,
            source_time=source_time if source_time is not None else monotonic_time,
            sequence=sequence,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            joint_effort=joint_effort,
            base_orientation=base_orientation,
            base_angular_velocity=base_angular_velocity,
            base_linear_acceleration=base_linear_acceleration,
            commanded_velocity=np.array([float(speed_scale)], dtype=float),
            gait_mode=gait_mode,
            battery_fraction=None,
            battery_voltage=None,
            source=TelemetrySource.SIM,
            valid=valid,
            field_provenance={},
        )


def _read_sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise ValueError(f"model has no sensor named '{name}'")
    adr = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])
    return np.array(data.sensordata[adr : adr + dim], dtype=float, copy=True)
