"""Fail-closed validation for immutable SherpaOS dataset artifacts."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from sherpaos.datasets.manifest import (
    read_checksums,
    read_json,
    sha256_file,
    write_checksums,
    write_json,
)
from sherpaos.datasets.schema import OBSERVATION_WIDTH, PRIVILEGED_FIELD_NAMES

REQUIRED_TOP_LEVEL = {
    "DATASET_CARD.md",
    "source_manifest.json",
    "scenario_manifest.json",
    "split_manifest.json",
    "quality_report.json",
    "checksums.sha256",
}


class DatasetValidationError(ValueError):
    """The dataset cannot be admitted to a release."""


def validate_dataset(dataset: Path) -> dict[str, Any]:
    dataset = Path(dataset)
    errors: list[str] = []
    for name in sorted(REQUIRED_TOP_LEVEL):
        if not (dataset / name).is_file():
            errors.append(f"missing required artifact: {name}")
    if errors:
        raise DatasetValidationError("; ".join(errors))

    _verify_checksums(dataset, errors)
    scenario = read_json(dataset / "scenario_manifest.json")
    split = read_json(dataset / "split_manifest.json")
    requested = int(scenario.get("episodes_requested", -1))
    contract_mode = bool(scenario.get("contract_mode", False))
    qualification_mode = bool(scenario.get("qualification_mode", False))
    expected = 2 if contract_mode else (20 if qualification_mode else 200)
    completed = scenario.get("completed", [])
    completed_ids = [row.get("episode_id") for row in completed]
    if requested != expected or len(completed_ids) != expected:
        errors.append(f"expected exactly {expected} completed episodes, got {len(completed_ids)}")
    if len(set(completed_ids)) != len(completed_ids):
        errors.append("duplicate episode IDs")
    if not contract_mode:
        settings = scenario.get("settings", {})
        stress_slope_min = float(settings.get("stress_slope_min_deg", 16.0))
        terrain_max_slope = float(settings.get("terrain_max_slope_deg", 30.0))
        short = [row.get("episode_id") for row in completed if int(row.get("windows", 0)) < 10]
        if short:
            errors.append(
                "episodes with fewer than 10 usable pre-failure windows: "
                + ", ".join(map(str, short))
            )
        nominal = [row for row in completed if row.get("category") == "nominal"]
        nominal_fall_rate = (
            sum(bool(row.get("fell")) for row in nominal) / len(nominal) if nominal else 1.0
        )
        if nominal_fall_rate > 0.20:
            errors.append(f"nominal fall rate {nominal_fall_rate:.6f} above 0.20")
        missed_steep = [
            row.get("episode_id")
            for row in completed
            if row.get("terrain_zone") == 4
            and float(row.get("max_slope_deg", 0.0)) < stress_slope_min - 0.5
        ]
        if missed_steep:
            errors.append(
                "stress terrain scenarios never contacted a "
                f">={stress_slope_min:g} degree segment: "
                + ", ".join(str(value) for value in missed_steep)
            )
        excessive_slope = [
            row.get("episode_id")
            for row in completed
            if float(row.get("max_slope_deg", 0.0)) > terrain_max_slope + 0.5
        ]
        if excessive_slope:
            errors.append(
                f"episodes exceeded {terrain_max_slope:g} degree terrain cap: "
                + ", ".join(str(value) for value in excessive_slope)
            )

    shard_size = int(scenario.get("settings", {}).get("shard_episodes", 10))
    expected_shards = math.ceil(expected / shard_size)
    observation_paths = sorted((dataset / "observations").glob("shard-*.npz"))
    label_paths = sorted((dataset / "labels").glob("shard-*.npz"))
    context_paths = sorted((dataset / "context").glob("shard-*.npz"))
    expected_names = [f"shard-{index:03d}.npz" for index in range(expected_shards)]
    if [path.name for path in observation_paths] != expected_names:
        errors.append("missing or unexpected observation shards")
    if [path.name for path in label_paths] != expected_names:
        errors.append("missing or unexpected label shards")
    if [path.name for path in context_paths] != expected_names:
        errors.append("missing or unexpected context shards")

    mobility_parts: list[np.ndarray] = []
    dynamics_parts: list[np.ndarray] = []
    warning_episodes: set[str] = set()
    risk_episodes: set[str] = set()
    observation_digests: dict[str, str] = {}
    for name in expected_names:
        observation_path = dataset / "observations" / name
        label_path = dataset / "labels" / name
        context_path = dataset / "context" / name
        if not observation_path.is_file() or not label_path.is_file() or not context_path.is_file():
            continue
        try:
            with (
                np.load(observation_path, allow_pickle=False) as observations,
                np.load(label_path, allow_pickle=False) as labels,
            ):
                _validate_pair(observations, labels, errors, name)
                with np.load(context_path, allow_pickle=False) as context:
                    if len(context["episode_ids"]) != int(np.sum(labels["completed_steps"])):
                        errors.append(f"context/telemetry length mismatch in {name}")
                mobility_parts.append(np.asarray(labels["mobility_targets"]))
                dynamics_parts.append(np.asarray(labels["dynamics_targets"]))
                episode_ids = np.asarray(labels["episode_ids"])
                observation_episode_ids = np.asarray(observations["episode_ids"])
                for episode_id in np.unique(observation_episode_ids):
                    episode_windows = observations["observations"][
                        observation_episode_ids == episode_id
                    ]
                    digest = hashlib.sha256(episode_windows.tobytes()).hexdigest()
                    previous = observation_digests.get(digest)
                    if previous is not None:
                        errors.append(
                            f"duplicate observation trajectory: {previous} and {episode_id}"
                        )
                    observation_digests[digest] = str(episode_id)
                risk = np.maximum(labels["mobility_targets"], labels["dynamics_targets"])
                for episode_id in np.unique(episode_ids):
                    values = risk[episode_ids == episode_id]
                    if np.any(values > 0.5):
                        risk_episodes.add(str(episode_id))
                        first = int(np.argmax(values > 0.5))
                        if first > 0 and np.any(values[:first] <= 0.5):
                            warning_episodes.add(str(episode_id))
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"missing or corrupt shard {name}: {exc}")

    _validate_splits(scenario, split, errors)
    mobility_rate = _positive_rate(mobility_parts)
    dynamics_rate = _positive_rate(dynamics_parts)
    if not contract_mode:
        if not 0.05 <= mobility_rate <= 0.80:
            errors.append(f"mobility positive rate {mobility_rate:.6f} outside [0.05, 0.80]")
        if not 0.05 <= dynamics_rate <= 0.80:
            errors.append(f"dynamics positive rate {dynamics_rate:.6f} outside [0.05, 0.80]")
        warning_coverage = len(warning_episodes) / len(risk_episodes) if risk_episodes else 0.0
        if warning_coverage < 0.90:
            errors.append(f"pre-failure warning coverage {warning_coverage:.6f} below 0.90")
    else:
        warning_coverage = 0.0

    report = {
        "status": "RED" if errors else "GREEN",
        "episodes_completed": len(completed_ids),
        "expected_episodes": expected,
        "feature_width": OBSERVATION_WIDTH,
        "mobility_positive_rate": mobility_rate,
        "dynamics_positive_rate": dynamics_rate,
        "pre_failure_warning_coverage": warning_coverage,
        "errors": errors,
    }
    if errors:
        raise DatasetValidationError("; ".join(errors))
    write_json(dataset / "quality_report.json", report)
    write_checksums(dataset)
    return report


def _verify_checksums(dataset: Path, errors: list[str]) -> None:
    try:
        expected = read_checksums(dataset)
    except (OSError, ValueError) as exc:
        errors.append(f"invalid checksum manifest: {exc}")
        return
    actual_files = {
        path.relative_to(dataset).as_posix()
        for path in dataset.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    if actual_files != set(expected):
        errors.append("checksum manifest file set mismatch")
    for relative, digest in expected.items():
        path = dataset / relative
        if not path.is_file() or sha256_file(path) != digest:
            errors.append(f"checksum mismatch: {relative}")


def _validate_pair(observations: Any, labels: Any, errors: list[str], name: str) -> None:
    lowered_keys = {key.lower() for key in observations.files}
    leaks = {
        key
        for key in lowered_keys
        if any(privileged in key for privileged in PRIVILEGED_FIELD_NAMES)
    }
    if leaks:
        errors.append(f"privileged field names in observation artifact {name}: {sorted(leaks)}")
    required_observation = {"observations", "episode_ids", "scenario_groups"}
    required_labels = {
        "mobility_targets",
        "dynamics_targets",
        "fall_targets",
        "episode_ids",
        "scenario_groups",
        "completed_episode_ids",
        "completed_steps",
        "episode_fell",
        "episode_window_counts",
    }
    if not required_observation.issubset(observations.files):
        errors.append(f"missing observation arrays in {name}")
        return
    if not required_labels.issubset(labels.files):
        errors.append(f"missing label arrays in {name}")
        return
    inputs = observations["observations"]
    if inputs.ndim != 3 or inputs.shape[2] != OBSERVATION_WIDTH:
        errors.append(f"unexpected feature width in {name}: {inputs.shape}")
    if not np.all(np.isfinite(inputs)):
        errors.append(f"NaN or infinity in model inputs: {name}")
    window_length = len(inputs)
    aligned = [
        observations["episode_ids"],
        observations["scenario_groups"],
        labels["mobility_targets"],
        labels["dynamics_targets"],
        labels["fall_targets"],
        labels["episode_ids"],
        labels["scenario_groups"],
    ]
    if any(len(array) != window_length for array in aligned):
        errors.append(f"observation/label length mismatch in {name}")
    elif not np.array_equal(observations["episode_ids"], labels["episode_ids"]):
        errors.append(f"observation/label episode alignment mismatch in {name}")


def _validate_splits(scenario: dict[str, Any], split: dict[str, Any], errors: list[str]) -> None:
    memberships = split.get("groups", {})
    episode_memberships = split.get("episodes", {})
    split_groups = {
        name: {group for group, assigned in memberships.items() if assigned == name}
        for name in ("train", "validation", "test")
    }
    if (
        split_groups["train"] & split_groups["validation"]
        or split_groups["train"] & split_groups["test"]
        or split_groups["validation"] & split_groups["test"]
    ):
        errors.append("overlap between train/validation/test scenario groups")
    seen_episodes: set[str] = set()
    for spec in scenario.get("specs", []):
        episode = spec["episode_id"]
        group = spec["scenario_group"]
        if episode in seen_episodes:
            errors.append("overlap between train/validation/test episodes")
        seen_episodes.add(episode)
        if episode_memberships.get(episode) != memberships.get(group):
            errors.append(f"episode {episode} split differs from scenario group {group}")


def _positive_rate(parts: list[np.ndarray]) -> float:
    if not parts:
        return 0.0
    values = np.concatenate(parts)
    return float(np.mean(values)) if len(values) else 0.0
