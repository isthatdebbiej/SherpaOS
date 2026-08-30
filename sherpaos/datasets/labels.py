"""Build future-horizon targets without exposing evaluator truth to observations."""

from __future__ import annotations

import numpy as np

from sherpaos.contracts import RobotTelemetry
from sherpaos.datasets.schema import OBSERVATION_WIDTH, WindowedRiskDataset, telemetry_vector
from sherpaos.evaluation.ground_truth import ScenarioGroundTruth

# Early-warning threshold: sustained 0.6 m/s planted-foot slip is already
# operationally hazardous even when the controller subsequently recovers.
MOBILITY_SLIP_UNSAFE_MPS = 0.6
DYNAMICS_TILT_UNSAFE_DEG = 25.0
DYNAMICS_ACTUATOR_HEALTH_UNSAFE = 0.90


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
    mobility_failure: bool = False,
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
    last_end = len(telemetry) if fell else len(telemetry) - horizon_steps
    for end in range(window_steps, last_end + 1, stride):
        future = truth[end : end + horizon_steps]
        fall_in_horizon = fell and end + horizon_steps >= len(telemetry)
        windows.append(encoded[end - window_steps : end])
        mobility.append(
            float(
                (mobility_failure and fall_in_horizon)
                or any(item.planted_foot_slip_mps >= MOBILITY_SLIP_UNSAFE_MPS for item in future)
            )
        )
        dynamics.append(
            float(
                fall_in_horizon
                or any(
                    item.tilt_from_vertical_deg >= DYNAMICS_TILT_UNSAFE_DEG
                    or item.actuator_health <= DYNAMICS_ACTUATOR_HEALTH_UNSAFE
                    for item in future
                )
            )
        )
        falls.append(float(fall_in_horizon))
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
