from __future__ import annotations

import numpy as np
import pytest

from sherpaos.adapters.playground import (
    PlaygroundObservationLayout,
    PlaygroundTelemetryAdapter,
)
from sherpaos.contracts import TelemetrySource


def _mapping_layout() -> PlaygroundObservationLayout:
    return PlaygroundObservationLayout(
        joint_position="joint_pos",
        joint_velocity="joint_vel",
        joint_effort="motor_effort",
        base_orientation="imu_quaternion",
        base_angular_velocity="imu_gyro",
        base_linear_acceleration="imu_accel",
        commanded_velocity="command",
    )


def _mapping_observation() -> dict[str, object]:
    return {
        "joint_pos": np.arange(29, dtype=float),
        "joint_vel": np.arange(29, dtype=float) * 0.1,
        "motor_effort": np.arange(29, dtype=float) * 0.01,
        "imu_quaternion": [1.0, 0.0, 0.0, 0.0],
        "imu_gyro": [0.1, 0.2, 0.3],
        "imu_accel": [0.0, 0.0, 9.81],
        "command": [0.4, 0.0, 0.1],
        "true_friction": 0.05,
        "injected_fault": "left_knee",
    }


def test_maps_observable_mapping_fields_and_ignores_ground_truth():
    adapter = PlaygroundTelemetryAdapter(_mapping_layout())
    telemetry = adapter.sample(
        _mapping_observation(),
        sequence=7,
        monotonic_time=12.5,
        source_time=12.4,
        gait_mode="walk",
    )

    assert telemetry.valid is True
    assert telemetry.source is TelemetrySource.SIM
    assert telemetry.sequence == 7
    assert telemetry.source_time == pytest.approx(12.4)
    assert telemetry.joint_position.shape == (29,)
    assert telemetry.commanded_velocity is not None
    np.testing.assert_allclose(telemetry.commanded_velocity, [0.4, 0.0, 0.1])
    assert all("friction" not in value for value in telemetry.field_provenance.values())
    assert all("fault" not in value for value in telemetry.field_provenance.values())


def test_flat_observation_requires_explicit_slices():
    layout = PlaygroundObservationLayout(
        joint_position=slice(0, 2),
        joint_velocity=slice(2, 4),
        joint_effort=slice(4, 6),
        base_orientation=slice(6, 10),
        base_angular_velocity=slice(10, 13),
        base_linear_acceleration=slice(13, 16),
        commanded_velocity=slice(16, 19),
    )
    observation = np.arange(19, dtype=float)
    adapter = PlaygroundTelemetryAdapter(layout, joint_count=2)

    telemetry = adapter.sample(observation, sequence=1, monotonic_time=2.0)

    assert telemetry.valid is True
    np.testing.assert_allclose(telemetry.joint_position, [0.0, 1.0])
    np.testing.assert_allclose(telemetry.base_angular_velocity, [10.0, 11.0, 12.0])
    np.testing.assert_allclose(telemetry.commanded_velocity, [16.0, 17.0, 18.0])


def test_missing_required_field_returns_invalid_sample_without_throwing():
    observation = _mapping_observation()
    del observation["imu_gyro"]
    adapter = PlaygroundTelemetryAdapter(_mapping_layout())

    telemetry = adapter.sample(observation, sequence=2, monotonic_time=3.0)

    assert telemetry.valid is False
    assert np.all(np.isnan(telemetry.base_angular_velocity))
    assert telemetry.field_provenance["base_angular_velocity"] == "missing_or_invalid"


def test_nonfinite_field_returns_invalid_sample_and_copies_input():
    observation = _mapping_observation()
    joint_pos = np.arange(29, dtype=float)
    observation["joint_pos"] = joint_pos
    adapter = PlaygroundTelemetryAdapter(_mapping_layout())

    telemetry = adapter.sample(observation, sequence=3, monotonic_time=4.0)
    joint_pos[0] = 999.0
    assert telemetry.joint_position[0] == 0.0

    observation["imu_accel"] = [0.0, np.nan, 9.81]
    invalid = adapter.sample(observation, sequence=4, monotonic_time=5.0)
    assert invalid.valid is False
    assert np.all(np.isnan(invalid.base_linear_acceleration))


def test_optional_fields_may_be_unavailable_without_invalidating_sample():
    layout = PlaygroundObservationLayout(
        joint_position="joint_pos",
        joint_velocity="joint_vel",
        base_orientation="imu_quaternion",
        base_angular_velocity="imu_gyro",
        base_linear_acceleration="imu_accel",
    )
    telemetry = PlaygroundTelemetryAdapter(layout).sample(
        _mapping_observation(), sequence=5, monotonic_time=6.0
    )

    assert telemetry.valid is True
    assert telemetry.joint_effort is None
    assert telemetry.commanded_velocity is None
    assert telemetry.field_provenance["joint_effort"] == "unavailable"


@pytest.mark.parametrize(
    "selector",
    ["true_friction", "contact_force", "injected_fault", "fall_state", "privileged_qpos"],
)
def test_layout_rejects_simulator_truth_selectors(selector: str):
    layout = PlaygroundObservationLayout(
        joint_position=selector,
        joint_velocity="joint_vel",
        base_orientation="imu_quaternion",
        base_angular_velocity="imu_gyro",
        base_linear_acceleration="imu_accel",
    )

    with pytest.raises(ValueError, match="simulator-only truth"):
        PlaygroundTelemetryAdapter(layout)
