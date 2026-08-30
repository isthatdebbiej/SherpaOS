from __future__ import annotations

import numpy as np

from sherpaos.sim.weather import HAZARD_ONSET_STEP, wind_speed_at_step


def test_extreme_wind_has_stable_baseline_and_smooth_storm_onset() -> None:
    values = np.asarray([wind_speed_at_step(step, 55.6) for step in range(500)])
    assert np.max(values[: HAZARD_ONSET_STEP + 1]) < 10.0
    assert values[-1] > 50.0
    assert np.max(np.abs(np.diff(values))) < 0.25


def test_ordinary_wind_is_continuous_at_hazard_onset() -> None:
    values = np.asarray([wind_speed_at_step(step, 20.0) for step in range(500)])
    assert values[0] > 0.0
    assert values[-1] > values[0]
    assert np.max(np.abs(np.diff(values))) < 0.1
