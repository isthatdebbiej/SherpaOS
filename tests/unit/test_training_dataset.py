from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sherpaos.contracts import RobotTelemetry
from sherpaos.evaluation.ground_truth import ScenarioGroundTruth
from sherpaos.training.dataset import (
    OBSERVATION_WIDTH,
    build_episode_windows,
    split_for_group,
    telemetry_vector,
    write_dataset_pair,
)


def _sample(sequence: int) -> RobotTelemetry:
    return RobotTelemetry(
        monotonic_time=sequence * 0.02,
        source_time=sequence * 0.02,
        sequence=sequence,
        joint_position=np.zeros(29),
        joint_velocity=np.zeros(29),
        joint_effort=np.zeros(29),
        base_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        base_angular_velocity=np.zeros(3),
        base_linear_acceleration=np.array([0.0, 0.0, 9.81]),
        commanded_velocity=np.array([0.2, 0.0, 0.0]),
    )


def _truth(index: int) -> ScenarioGroundTruth:
    return ScenarioGroundTruth(
        true_friction=0.1,
        true_slope_deg=0.0,
        disturbance_active=False,
        actuator_health=1.0,
        tilt_from_vertical_deg=0.0,
        planted_foot_slip_mps=0.2 if index == 5 else 0.0,
        true_unsafe=index == 5,
    )


def test_vector_excludes_battery_and_has_frozen_width() -> None:
    sample = _sample(0)
    vector = telemetry_vector(sample)
    assert vector.shape == (OBSERVATION_WIDTH,)
    assert sample.battery_fraction is None


def test_windows_use_future_truth_only_as_separate_targets() -> None:
    telemetry = [_sample(i) for i in range(8)]
    truth = [_truth(i) for i in range(8)]
    dataset = build_episode_windows(
        telemetry, truth, episode_id="episode-a", window_steps=3, horizon_steps=2
    )
    assert dataset.observations.shape[1:] == (3, OBSERVATION_WIDTH)
    assert dataset.mobility_targets.max() == 1.0
    assert dataset.dynamics_targets.max() == 0.0
    assert set(dataset.episode_ids) == {"episode-a"}


def test_group_split_is_stable_and_dataset_files_must_be_separate(tmp_path: Path) -> None:
    assert split_for_group("rough-family") == split_for_group("rough-family")
    empty = build_episode_windows([], [], episode_id="empty", window_steps=3, horizon_steps=1)
    with pytest.raises(ValueError, match="separate files"):
        write_dataset_pair(tmp_path / "same.npz", tmp_path / "same.npz", empty)
    write_dataset_pair(tmp_path / "observations.npz", tmp_path / "labels.npz", empty)
    assert set(np.load(tmp_path / "observations.npz").files) == {
        "observations",
        "episode_ids",
    }
    assert "mobility_targets" not in np.load(tmp_path / "observations.npz").files
