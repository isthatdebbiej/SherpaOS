from __future__ import annotations

import numpy as np
import pytest

from sherpaos.sim.unitree_walking import DEFAULT_CONFIG_PATH, run_unitree_walking_episode


@pytest.mark.integration
def test_unitree_walking_publishes_sensorized_telemetry():
    if not DEFAULT_CONFIG_PATH.exists():
        pytest.skip("Unitree walking configuration is unavailable")

    samples = []
    auxiliary_samples = []
    result = run_unitree_walking_episode(
        config_path=DEFAULT_CONFIG_PATH,
        max_steps=3,
        telemetry_observer=samples.append,
        simulated_auxiliary_observer=auxiliary_samples.append,
        terrain_slope_deg=4.17,
    )

    assert result.control_steps == 3
    assert result.elapsed_seconds == pytest.approx(0.06)
    assert result.fell is False
    assert result.uphill_slope_deg == pytest.approx(4.17)
    assert len(samples) == 3
    assert len(auxiliary_samples) == 3
    assert all(sample.gait_mode == "walking" for sample in samples)
    assert all(sample.joint_position.shape == (12,) for sample in samples)
    assert all(np.all(np.isfinite(sample.base_orientation)) for sample in samples)
    assert all(auxiliary.actual_speed_mps >= 0.0 for auxiliary in auxiliary_samples)
    assert all(set(auxiliary.contacts) == {"left", "right"} for auxiliary in auxiliary_samples)
    assert all(auxiliary.battery_fraction > 0.0 for auxiliary in auxiliary_samples)