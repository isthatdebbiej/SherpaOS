from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from sherpaos.contracts import RobotTelemetry
from sherpaos.datasets.generate import generate_dataset
from sherpaos.datasets.manifest import read_json, write_checksums
from sherpaos.datasets.split import build_split_manifest
from sherpaos.datasets.validate import DatasetValidationError, validate_dataset
from sherpaos.evaluation.ground_truth import ScenarioGroundTruth


@dataclass
class _Result:
    telemetry: list[RobotTelemetry]
    ground_truth: list[ScenarioGroundTruth]
    fell: bool = False
    steps_survived: int = 500


def _fake_episode(scenario, seed, guard_fn, max_steps):
    assert guard_fn is None
    assert max_steps == 500
    telemetry = []
    truth = []
    for index in range(500):
        telemetry.append(
            RobotTelemetry(
                monotonic_time=index * 0.02,
                source_time=index * 0.02,
                sequence=index,
                joint_position=np.full(29, seed % 7, dtype=float),
                joint_velocity=np.zeros(29),
                joint_effort=np.zeros(29),
                base_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                base_angular_velocity=np.zeros(3),
                base_linear_acceleration=np.array([0.0, 0.0, 9.81]),
                commanded_velocity=np.array([0.2, 0.0, 0.0]),
            )
        )
        truth.append(
            ScenarioGroundTruth(
                true_friction=scenario.friction,
                true_slope_deg=scenario.slope_deg,
                disturbance_active=False,
                actuator_health=scenario.actuator_health,
                tilt_from_vertical_deg=0.0,
                planted_foot_slip_mps=0.2 if scenario.friction < 0.2 else 0.0,
                true_unsafe=scenario.friction < 0.2 or scenario.actuator_health < 0.3,
            )
        )
    return _Result(telemetry, truth)


@pytest.fixture
def matrix_path() -> Path:
    return Path("configs/scenario_matrix.yaml")


def test_deterministic_generation_and_observation_label_separation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matrix_path: Path
) -> None:
    monkeypatch.setattr("sherpaos.datasets.generate.run_episode", _fake_episode)
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(matrix_path, 2, first)
    generate_dataset(matrix_path, 2, second)
    with (
        np.load(first / "observations/shard-000.npz", allow_pickle=False) as first_obs,
        np.load(second / "observations/shard-000.npz", allow_pickle=False) as second_obs,
    ):
        assert np.array_equal(first_obs["observations"], second_obs["observations"])
        assert set(first_obs.files) == {"observations", "episode_ids", "scenario_groups"}
        assert "true_friction" not in first_obs.files
        assert "mobility_targets" not in first_obs.files
    with np.load(first / "labels/shard-000.npz", allow_pickle=False) as labels:
        assert "mobility_targets" in labels.files
        assert set(labels["completed_episode_ids"]) == {"episode-000", "episode-050"}
    assert validate_dataset(first)["status"] == "GREEN"


def test_leakage_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matrix_path: Path
) -> None:
    monkeypatch.setattr("sherpaos.datasets.generate.run_episode", _fake_episode)
    output = tmp_path / "leak"
    generate_dataset(matrix_path, 2, output)
    path = output / "observations/shard-000.npz"
    with np.load(path, allow_pickle=False) as artifact:
        values = {key: artifact[key] for key in artifact.files}
    np.savez_compressed(path, **values, true_friction=np.zeros(len(values["episode_ids"])))
    write_checksums(output)
    with pytest.raises(DatasetValidationError, match="privileged field"):
        validate_dataset(output)


def test_group_split_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matrix_path: Path
) -> None:
    monkeypatch.setattr("sherpaos.datasets.generate.run_episode", _fake_episode)
    output = tmp_path / "split"
    generate_dataset(matrix_path, 2, output)
    manifest = build_split_manifest(output, Path("configs/splits.yaml"))
    for episode, split in manifest["episodes"].items():
        category = "nominal" if episode == "episode-000" else "mobility"
        assert split == manifest["groups"][f"{category}-group-00"]


def test_resumability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matrix_path: Path) -> None:
    calls = 0

    def counting_episode(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _fake_episode(*args, **kwargs)

    monkeypatch.setattr("sherpaos.datasets.generate.run_episode", counting_episode)
    output = tmp_path / "resume"
    generate_dataset(matrix_path, 2, output)
    assert calls == 2
    first_manifest = read_json(output / "scenario_manifest.json")
    generate_dataset(matrix_path, 2, output)
    assert calls == 2
    resumed_manifest = read_json(output / "scenario_manifest.json")
    assert resumed_manifest["completed"] == first_manifest["completed"]
    assert all(
        "terrain_zone" in row and "max_slope_deg" in row for row in resumed_manifest["completed"]
    )


def test_checksum_corruption_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matrix_path: Path
) -> None:
    monkeypatch.setattr("sherpaos.datasets.generate.run_episode", _fake_episode)
    output = tmp_path / "corrupt"
    generate_dataset(matrix_path, 2, output)
    path = output / "observations/shard-000.npz"
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(DatasetValidationError, match="checksum mismatch"):
        validate_dataset(output)


def test_validation_rejects_episode_without_pre_failure_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matrix_path: Path
) -> None:
    monkeypatch.setattr("sherpaos.datasets.generate.run_episode", _fake_episode)
    output = tmp_path / "short"
    generate_dataset(matrix_path, 20, output)
    manifest_path = output / "scenario_manifest.json"
    import json

    manifest = json.loads(manifest_path.read_text())
    manifest["completed"][0]["windows"] = 0
    manifest_path.write_text(json.dumps(manifest))
    write_checksums(output)
    with pytest.raises(DatasetValidationError, match="fewer than 10 usable"):
        validate_dataset(output)
