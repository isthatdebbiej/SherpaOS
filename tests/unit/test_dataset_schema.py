from __future__ import annotations

import numpy as np

from sherpaos.contracts import RobotTelemetry
from sherpaos.datasets.schema import OBSERVATION_WIDTH, telemetry_vector


def test_one_element_simulator_command_is_finite_and_padded() -> None:
    sample = RobotTelemetry(
        monotonic_time=0.0,
        source_time=0.0,
        sequence=0,
        joint_position=np.zeros(29),
        joint_velocity=np.zeros(29),
        joint_effort=np.zeros(29),
        base_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        base_angular_velocity=np.zeros(3),
        base_linear_acceleration=np.array([0.0, 0.0, 9.81]),
        commanded_velocity=np.array([0.7]),
    )
    vector = telemetry_vector(sample)
    assert vector.shape == (OBSERVATION_WIDTH,)
    assert np.all(np.isfinite(vector))
    np.testing.assert_allclose(vector[98:101], [0.7, 0.0, 0.0])
