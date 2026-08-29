"""Build future-horizon targets without exposing evaluator truth to observations."""

from __future__ import annotations

import numpy as np

from sherpaos.contracts import RobotTelemetry
from sherpaos.datasets.schema import OBSERVATION_WIDTH, WindowedRiskDataset, telemetry_vector
from sherpaos.evaluation.ground_truth import (
    DEFAULT_ACTUATOR_HEALTH_UNSAFE,
    DEFAULT_FOOT_SLIP_UNSAFE_MPS,
    DEFAULT_TILT_UNSAFE_DEG,
    ScenarioGroundTruth,
)


def build_episode_windows(
    telemetry: list[RobotTelemetry],
    truth: list[ScenarioGroundTruth],
    *,
    episode_id: str,
    scenario_group: str = "unspecified",
    window_steps: int,
    horizon_steps: int,
    stride: int = 1,
    fell: bool = False,
) -> WindowedRiskDataset:
    if len(telemetry) != len(truth):
        raise ValueError("telemetry and privileged truth must be index-aligned")
    if window_steps < 2 or horizon_steps < 1 or stride < 1:
        raise ValueError("invalid window, horizon, or stride")

    encoded = np.asarray([telemetry_vector(sample) for sample in telemetry], dtype=np.float32)
    windows: list[np.ndarray] = []
    mobility: list[float] = []
    dynamics: list[float] = []
    falls: list[float] = []
    ids: list[str] = []
    groups: list[str] = []
    last_end = len(telemetry) - horizon_steps
    for end in range(window_steps, last_end + 1, stride):
        future = truth[end : end + horizon_steps]
        windows.append(encoded[end - window_steps : end])
        mobility.append(
            float(
                any(
                    (
                        item.true_friction <= 0.15
                        or item.planted_foot_slip_mps >= DEFAULT_FOOT_SLIP_UNSAFE_MPS
                    )
                    for item in future
                )
            )
        )
        dynamics.append(
            float(
                any(
                    item.tilt_from_vertical_deg >= DEFAULT_TILT_UNSAFE_DEG
                    or item.actuator_health <= DEFAULT_ACTUATOR_HEALTH_UNSAFE
                    for item in future
                )
            )
        )
        falls.append(float(fell and end + horizon_steps >= len(telemetry)))
        ids.append(episode_id)
        groups.append(scenario_group)

    shape = (0, window_steps, OBSERVATION_WIDTH)
    return WindowedRiskDataset(
        observations=np.stack(windows) if windows else np.empty(shape, dtype=np.float32),
        mobility_targets=np.asarray(mobility, dtype=np.float32),
        dynamics_targets=np.asarray(dynamics, dtype=np.float32),
        fall_targets=np.asarray(falls, dtype=np.float32),
        episode_ids=np.asarray(ids, dtype=str),
        scenario_groups=np.asarray(groups, dtype=str),
    )
