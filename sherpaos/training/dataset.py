"""Build leakage-resistant temporal risk datasets from simulator episodes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sherpaos.contracts import RobotTelemetry
from sherpaos.evaluation.ground_truth import (
    DEFAULT_ACTUATOR_HEALTH_UNSAFE,
    DEFAULT_FOOT_SLIP_UNSAFE_MPS,
    DEFAULT_TILT_UNSAFE_DEG,
    ScenarioGroundTruth,
)

JOINT_COUNT = 29
OBSERVATION_WIDTH = 103


@dataclass(frozen=True, slots=True)
class WindowedRiskDataset:
    """Observation windows and aligned targets held in separate arrays."""

    observations: np.ndarray  # [window, time, feature]
    mobility_targets: np.ndarray  # [window]
    dynamics_targets: np.ndarray  # [window]
    episode_ids: np.ndarray  # [window], never used as a model feature


def telemetry_vector(sample: RobotTelemetry) -> np.ndarray:
    """Encode only onboard-observable fields; battery/geography are excluded."""
    effort = _fixed(sample.joint_effort, JOINT_COUNT) if sample.joint_effort is not None else None
    command = (
        _fixed(sample.commanded_velocity, 3)
        if sample.commanded_velocity is not None
        else None
    )
    vector = np.concatenate(
        [
            _fixed(sample.joint_position, JOINT_COUNT),
            _fixed(sample.joint_velocity, JOINT_COUNT),
            np.zeros(JOINT_COUNT) if effort is None else effort,
            np.array([effort is not None], dtype=float),
            _fixed(sample.base_orientation, 4),
            _fixed(sample.base_angular_velocity, 3),
            _fixed(sample.base_linear_acceleration, 3),
            np.zeros(3) if command is None else command,
            np.array([command is not None, sample.valid], dtype=float),
        ]
    )
    if vector.shape != (OBSERVATION_WIDTH,):
        raise AssertionError(f"observation schema produced {vector.shape}")
    return vector


def build_episode_windows(
    telemetry: list[RobotTelemetry],
    truth: list[ScenarioGroundTruth],
    *,
    episode_id: str,
    window_steps: int,
    horizon_steps: int,
    stride: int = 1,
) -> WindowedRiskDataset:
    """Create causal windows with future-horizon evaluator-only targets."""
    if len(telemetry) != len(truth):
        raise ValueError("telemetry and privileged truth must be index-aligned")
    if window_steps < 2 or horizon_steps < 1 or stride < 1:
        raise ValueError("invalid window, horizon, or stride")

    encoded = np.asarray([telemetry_vector(sample) for sample in telemetry], dtype=np.float32)
    windows: list[np.ndarray] = []
    mobility: list[float] = []
    dynamics: list[float] = []
    ids: list[str] = []
    last_end = len(telemetry) - horizon_steps
    for end in range(window_steps, last_end + 1, stride):
        future = truth[end : end + horizon_steps]
        windows.append(encoded[end - window_steps : end])
        mobility.append(
            float(
                any(item.planted_foot_slip_mps >= DEFAULT_FOOT_SLIP_UNSAFE_MPS for item in future)
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
        ids.append(episode_id)

    shape = (0, window_steps, OBSERVATION_WIDTH)
    observation_array = np.stack(windows) if windows else np.empty(shape, dtype=np.float32)
    return WindowedRiskDataset(
        observations=observation_array,
        mobility_targets=np.asarray(mobility, dtype=np.float32),
        dynamics_targets=np.asarray(dynamics, dtype=np.float32),
        episode_ids=np.asarray(ids, dtype=str),
    )


def split_for_group(group_id: str, *, split_seed: int = 20260829) -> str:
    """Assign an entire scenario group deterministically; never split frames."""
    digest = hashlib.sha256(f"{split_seed}:{group_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def write_dataset_pair(
    observation_path: Path,
    label_path: Path,
    dataset: WindowedRiskDataset,
) -> None:
    """Persist observations and privileged-derived targets in different files."""
    observation_path = Path(observation_path)
    label_path = Path(label_path)
    if observation_path.resolve() == label_path.resolve():
        raise ValueError("observations and labels must use separate files")
    observation_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        observation_path,
        observations=dataset.observations,
        episode_ids=dataset.episode_ids,
    )
    np.savez_compressed(
        label_path,
        mobility_targets=dataset.mobility_targets,
        dynamics_targets=dataset.dynamics_targets,
        episode_ids=dataset.episode_ids,
    )


def _fixed(value: object, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size or not np.all(np.isfinite(array)):
        return np.full(size, np.nan, dtype=float)
    return array
