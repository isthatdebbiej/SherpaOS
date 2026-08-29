"""Deterministic, resumable controller-only MuJoCo episode generation."""

from __future__ import annotations

import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sherpaos.datasets.labels import build_episode_windows
from sherpaos.datasets.manifest import git_sha, read_json, write_checksums, write_json
from sherpaos.datasets.schema import OBSERVATION_WIDTH, WindowedRiskDataset
from sherpaos.datasets.split import build_split_manifest
from sherpaos.sim.runner import run_episode
from sherpaos.sim.scenario import Scenario, nominal_scenario

CATEGORIES = ("nominal", "mobility", "dynamics", "combined")
LABEL_RULE_VERSION = "risk-horizon-v1"


def load_matrix(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "dataset" not in value or "categories" not in value:
        raise ValueError("scenario matrix must contain dataset and categories mappings")
    return value


def episode_specs(matrix: dict[str, Any], episodes: int) -> list[dict[str, Any]]:
    if episodes not in {2, 200}:
        raise ValueError("episodes must be exactly 200 (or 2 for the local contract run)")
    configured = int(matrix["dataset"]["episodes"])
    if configured != 200:
        raise ValueError("production scenario matrix must specify exactly 200 episodes")
    seed_base = int(matrix["dataset"]["seed_base"])
    full: list[dict[str, Any]] = []
    for category in CATEGORIES:
        count = int(matrix["categories"][category]["episodes"])
        if count != 50:
            raise ValueError(f"category {category} must specify exactly 50 episodes")
        for category_index in range(count):
            absolute_index = len(full)
            full.append(
                {
                    "episode_id": f"episode-{absolute_index:03d}",
                    "category": category,
                    "category_index": category_index,
                    "scenario_group": f"{category}-group-{category_index // 10:02d}",
                    "seed": seed_base + absolute_index,
                }
            )
    if episodes == 2:
        return [full[0], full[50]]
    return full


def scenario_for(category: str, seed: int) -> Scenario:
    if category == "nominal":
        return nominal_scenario(seed)
    rng = np.random.default_rng(seed)
    friction = 1.0
    actuator_health = 1.0
    force = 0.0
    direction = None
    start = None
    duration = None
    if category in {"mobility", "combined"}:
        friction = float(rng.uniform(0.04, 0.12))
    if category in {"dynamics", "combined"}:
        actuator_health = 0.30
        force = float(rng.uniform(45.0, 70.0))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        start = int(rng.integers(1800, 2600))
        duration = int(rng.integers(100, 180))
    return Scenario(
        friction=friction,
        slope_deg=0.0,
        disturbance_force_n=force,
        disturbance_direction=direction,
        disturbance_start_step=start,
        disturbance_duration_steps=duration,
        actuator_health=actuator_health,
        sensor_noise_std=float(rng.uniform(0.0, 0.01)),
        seed=seed,
    )


def generate_dataset(matrix_path: Path, episodes: int, output: Path) -> dict[str, Any]:
    matrix_path = Path(matrix_path)
    output = Path(output)
    matrix = load_matrix(matrix_path)
    settings = matrix["dataset"]
    specs = episode_specs(matrix, episodes)
    shard_size = int(settings["shard_episodes"])
    if shard_size != 10:
        raise ValueError("shard_episodes must be 10")
    if int(settings["control_steps"]) != 500 or int(settings["control_hz"]) != 50:
        raise ValueError("generation requires 500 control steps at 50 Hz")

    output.mkdir(parents=True, exist_ok=True)
    (output / "observations").mkdir(exist_ok=True)
    (output / "labels").mkdir(exist_ok=True)
    existing_manifest = output / "scenario_manifest.json"
    if existing_manifest.exists():
        previous = read_json(existing_manifest)
        if previous.get("episodes_requested") != episodes or previous.get("specs") != specs:
            raise ValueError("existing dataset manifest does not match this deterministic run")

    window_steps = int(settings["window_seconds"] * settings["control_hz"])
    horizon_steps = int(settings["prediction_horizon_seconds"] * settings["control_hz"])
    stride = int(settings["stride_samples"])
    completed: list[dict[str, Any]] = []
    resumed_shards = 0
    for shard_index in range(math.ceil(len(specs) / shard_size)):
        shard_specs = specs[shard_index * shard_size : (shard_index + 1) * shard_size]
        observation_path = output / "observations" / f"shard-{shard_index:03d}.npz"
        label_path = output / "labels" / f"shard-{shard_index:03d}.npz"
        if _valid_existing_shard(observation_path, label_path, shard_specs):
            resumed_shards += 1
            completed.extend(_episode_rows_from_label(label_path, shard_specs))
            continue

        datasets: list[WindowedRiskDataset] = []
        shard_rows: list[dict[str, Any]] = []
        for spec in shard_specs:
            scenario = scenario_for(spec["category"], spec["seed"])
            result = run_episode(scenario, seed=spec["seed"], guard_fn=None, max_steps=500)
            dataset = build_episode_windows(
                result.telemetry,
                result.ground_truth,
                episode_id=spec["episode_id"],
                scenario_group=spec["scenario_group"],
                window_steps=window_steps,
                horizon_steps=horizon_steps,
                stride=stride,
                fell=result.fell,
            )
            datasets.append(dataset)
            shard_rows.append(
                {
                    **spec,
                    "steps_completed": result.steps_survived,
                    "fell": result.fell,
                    "windows": int(dataset.observations.shape[0]),
                    "scenario": _jsonable_scenario(scenario),
                }
            )
        _write_shard(observation_path, label_path, datasets, shard_rows)
        completed.extend(shard_rows)

    source_manifest = {
        "version": 1,
        "code_sha": git_sha(),
        "matrix": str(matrix_path.as_posix()),
        "generator": "sherpaos.datasets.generate",
        "controller_mode": "controller-only",
        "sherpaos_intervention": False,
    }
    scenario_manifest = {
        "version": 1,
        "dataset_id": output.name,
        "episodes_requested": episodes,
        "production_episode_target": 200,
        "contract_mode": episodes == 2,
        "label_rule_version": LABEL_RULE_VERSION,
        "settings": settings,
        "specs": specs,
        "completed": completed,
    }
    write_json(output / "source_manifest.json", source_manifest)
    write_json(output / "scenario_manifest.json", scenario_manifest)
    build_split_manifest(output, Path("configs/splits.yaml"))
    quality = {
        "status": "UNVALIDATED",
        "episodes_completed": len(completed),
        "resumed_shards": resumed_shards,
        "feature_width": OBSERVATION_WIDTH,
    }
    write_json(output / "quality_report.json", quality)
    (output / "DATASET_CARD.md").write_text(_dataset_card(scenario_manifest), encoding="utf-8")
    write_checksums(output)
    return quality


def _valid_existing_shard(obs_path: Path, label_path: Path, specs: list[dict[str, Any]]) -> bool:
    if not obs_path.is_file() or not label_path.is_file():
        return False
    try:
        with (
            np.load(obs_path, allow_pickle=False) as obs,
            np.load(label_path, allow_pickle=False) as labels,
        ):
            expected = {item["episode_id"] for item in specs}
            return (
                obs["observations"].ndim == 3
                and obs["observations"].shape[2] == OBSERVATION_WIDTH
                and np.all(np.isfinite(obs["observations"]))
                and set(labels["completed_episode_ids"].tolist()) == expected
                and len(obs["episode_ids"]) == len(labels["episode_ids"])
            )
    except (OSError, ValueError, KeyError):
        return False


def _episode_rows_from_label(path: Path, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as labels:
        steps = labels["completed_steps"].tolist()
        falls = labels["episode_fell"].tolist()
        windows = labels["episode_window_counts"].tolist()
    return [
        {**spec, "steps_completed": int(step), "fell": bool(fell), "windows": int(count)}
        for spec, step, fell, count in zip(specs, steps, falls, windows, strict=True)
    ]


def _write_shard(
    observation_path: Path,
    label_path: Path,
    datasets: list[WindowedRiskDataset],
    rows: list[dict[str, Any]],
) -> None:
    def combine(field: str) -> np.ndarray:
        arrays = [getattr(dataset, field) for dataset in datasets]
        return np.concatenate(arrays) if arrays else np.asarray([])

    observation_tmp = observation_path.with_suffix(".tmp.npz")
    label_tmp = label_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        observation_tmp,
        observations=combine("observations"),
        episode_ids=combine("episode_ids"),
        scenario_groups=combine("scenario_groups"),
    )
    np.savez_compressed(
        label_tmp,
        mobility_targets=combine("mobility_targets"),
        dynamics_targets=combine("dynamics_targets"),
        fall_targets=combine("fall_targets"),
        episode_ids=combine("episode_ids"),
        scenario_groups=combine("scenario_groups"),
        completed_episode_ids=np.asarray([row["episode_id"] for row in rows]),
        completed_steps=np.asarray([row["steps_completed"] for row in rows], dtype=np.int32),
        episode_fell=np.asarray([row["fell"] for row in rows], dtype=np.bool_),
        episode_window_counts=np.asarray([row["windows"] for row in rows], dtype=np.int32),
    )
    os.replace(observation_tmp, observation_path)
    os.replace(label_tmp, label_path)


def _jsonable_scenario(scenario: Scenario) -> dict[str, Any]:
    value = asdict(scenario)
    if isinstance(value["disturbance_direction"], np.ndarray):
        value["disturbance_direction"] = value["disturbance_direction"].tolist()
    return value


def _dataset_card(manifest: dict[str, Any]) -> str:
    return (
        f"# SherpaOS dataset {manifest['dataset_id']}\n\n"
        "Controller-only MuJoCo simulation dataset with "
        f"{manifest['episodes_requested']} episodes.\n\n"
        "Observations contain only RobotTelemetry-compatible motion fields. Privileged simulator "
        "truth is stored separately under `labels/` and is evaluator-only.\n"
    )
