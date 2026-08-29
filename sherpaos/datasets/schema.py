"""Frozen observation schema for the risk-model dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sherpaos.contracts import RobotTelemetry

JOINT_COUNT = 29
OBSERVATION_WIDTH = 103
PRIVILEGED_FIELD_NAMES = frozenset(
    {
        "true_friction",
        "true_slope",
        "true_slope_deg",
        "slip_truth",
        "planted_foot_slip_mps",
        "actuator_health",
        "disturbance_identity",
        "disturbance_active",
        "fall_truth",
        "fell",
        "true_unsafe",
    }
)


@dataclass(frozen=True, slots=True)
class WindowedRiskDataset:
    observations: np.ndarray
    mobility_targets: np.ndarray
    dynamics_targets: np.ndarray
    fall_targets: np.ndarray
    episode_ids: np.ndarray
    scenario_groups: np.ndarray


def telemetry_vector(sample: RobotTelemetry) -> np.ndarray:
    """Encode onboard-observable motion fields only."""
    effort = _fixed(sample.joint_effort, JOINT_COUNT) if sample.joint_effort is not None else None
    command = _command_vector(sample.commanded_velocity)
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


def _fixed(value: object, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size or not np.all(np.isfinite(array)):
        return np.full(size, np.nan, dtype=float)
    return array


def _command_vector(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)) or array.size not in {1, 3}:
        return np.full(3, np.nan, dtype=float)
    if array.size == 1:
        return np.array([array[0], 0.0, 0.0], dtype=float)
    return array
