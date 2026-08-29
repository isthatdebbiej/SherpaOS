"""Compatibility facade for dataset code moved to :mod:`sherpaos.datasets`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sherpaos.datasets.labels import build_episode_windows
from sherpaos.datasets.schema import (
    JOINT_COUNT,
    OBSERVATION_WIDTH,
    WindowedRiskDataset,
    telemetry_vector,
)
from sherpaos.datasets.split import split_for_group


def write_dataset_pair(
    observation_path: Path, label_path: Path, dataset: WindowedRiskDataset
) -> None:
    """Retain the original small two-file writer for existing callers."""
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


__all__ = [
    "JOINT_COUNT",
    "OBSERVATION_WIDTH",
    "WindowedRiskDataset",
    "build_episode_windows",
    "split_for_group",
    "telemetry_vector",
    "write_dataset_pair",
]
