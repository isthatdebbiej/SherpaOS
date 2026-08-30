"""Deterministic, resumable controller-only MuJoCo episode generation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sherpaos.datasets.context import EpisodeContext, build_episode_context
from sherpaos.datasets.labels import build_episode_windows
from sherpaos.datasets.manifest import (
    git_sha,
    read_json,
    sha256_file,
    write_checksums,
    write_json,
)
from sherpaos.datasets.schema import OBSERVATION_WIDTH, WindowedRiskDataset
from sherpaos.datasets.split import build_split_manifest
from sherpaos.geography.terrain import load_route
from sherpaos.sim.runner import run_episode
from sherpaos.sim.scenario import Scenario, nominal_scenario

CATEGORIES = ("nominal", "mobility", "dynamics", "combined")
PROVENANCE_FILES = (
    "configs/scenario_matrix.yaml",
    "configs/data_sources.yaml",
    "configs/splits.yaml",
    "sherpaos/datasets/generate.py",
    "sherpaos/datasets/labels.py",
    "sherpaos/datasets/context.py",
    "sherpaos/datasets/schema.py",
    "sherpaos/datasets/validate.py",
    "sherpaos/sim/himalaya_scene.py",
    "sherpaos/sim/v26_runner.py",
    "sherpaos/sim/weather.py",
)
LABEL_RULE_VERSION = "risk-horizon-v4-causal-fall"
SHARD_SCHEMA_VERSION = 3


def load_matrix(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "dataset" not in value or "categories" not in value:
        raise ValueError("scenario matrix must contain dataset and categories mappings")
    return value


def episode_specs(matrix: dict[str, Any], episodes: int) -> list[dict[str, Any]]:
    if episodes not in {2, 20, 200}:
        raise ValueError("episodes must be 200, 20 for cloud qualification, or 2 locally")
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
    if episodes == 20:
        return [item for start in (0, 50, 100, 150) for item in full[start : start + 5]]
    return full


def scenario_for(category: str, seed: int) -> Scenario:
    if category == "nominal":
        return replace(nominal_scenario(seed), friction=0.90)
    rng = np.random.default_rng(seed)
    variant = seed % 5
    friction = (
        (0.50, 0.36, 0.30, 0.58, 0.40)[variant]
        if category == "mobility"
        else (0.42, 0.36, 0.32, 0.38, 0.65)[variant]
        if category == "combined"
        else 0.65
    )
    actuator_health = 1.0
    force = 0.0
    direction = None
    start = None
    duration = None
    if category in {"dynamics", "combined"}:
        health_ladder = (0.98, 0.95, 0.90, 0.85, 0.98)
        force_ladder = (0.0, 10.0, 20.0, 0.0, 0.0)
        actuator_health = health_ladder[variant]
        force = force_ladder[variant]
        if force:
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            start = int(rng.integers(2100, 2800))
            duration = int(rng.integers(75, 150))
    if category == "combined" and variant == 4:
        friction = 0.65
        actuator_health = 0.98
        force = 0.0
        direction = None
        start = None
        duration = None
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
        battery_initial_fraction=0.18 if category == "combined" and variant == 4 else 0.8,
        battery_temperature_c=-25.0 if category == "combined" and variant == 3 else 15.0,
    )


def _command_for(category_index: int) -> tuple[float, float, float]:
    """Matched command families shared by every category."""
    base = (
        (0.25, 0.0, 0.0),
        (0.40, 0.0, 0.0),
        (0.30, 0.0, 0.10),
        (0.30, 0.0, 0.0),
        (0.35, 0.0, -0.05),
    )[category_index % 5]
    group = category_index // 5
    speed_delta = (group - 4.5) * 0.004
    yaw_delta = ((group % 3) - 1) * 0.003
    return (base[0] + speed_delta, base[1], base[2] + yaw_delta)


def _terrain_zone_for(category: str, category_index: int) -> int:
    """Select safe baselines plus isolated stress terrain without spawn overlap."""
    if category in {"nominal", "dynamics"}:
        return 0
    if category == "mobility":
        return (1, 2, 3, 4, 2)[category_index % 5]
    if category == "combined" and category_index % 5 == 4:
        return 0
    return category_index % 5


def _route_fraction_for(category_index: int) -> float:
    return (0.0, 0.25, 0.50, 0.75, 0.90)[category_index % 5]


def _wind_for(category: str, category_index: int) -> float:
    """Deterministic Himalayan wind targets, including repeated 200 km/h stress."""
    targets = {
        "nominal": (5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 6.0, 7.0),
        "mobility": (5.0, 8.0, 10.0, 8.0, 15.0, 10.0, 12.0, 15.0, 8.0, 12.0),
        "dynamics": (8.0, 10.0, 12.0, 15.0, 25.0, 12.0, 15.0, 18.0, 22.0, 15.0),
        "combined": (10.0, 15.0, 25.0, 35.0, 55.6, 15.0, 20.0, 30.0, 45.0, 55.6),
    }
    target = targets[category][category_index % 10]
    if target >= 50.0:
        return target
    group_offset = (-0.4, -0.2, 0.0, 0.2, 0.4)[category_index // 10]
    return target + group_offset


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
    (output / "context").mkdir(exist_ok=True)
    route = load_route()
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
        context_path = output / "context" / f"shard-{shard_index:03d}.npz"
        if _valid_existing_shard(observation_path, label_path, context_path, shard_specs):
            resumed_shards += 1
            completed.extend(_episode_rows_from_label(label_path, shard_specs))
            continue

        datasets: list[WindowedRiskDataset] = []
        contexts: list[EpisodeContext] = []
        truths: list[list[Any]] = []
        shard_rows: list[dict[str, Any]] = []
        for spec in shard_specs:
            scenario = scenario_for(spec["category"], spec["seed"])
            command = _command_for(spec["category_index"])
            policy_path = os.environ.get("SHERPA_V26_POLICY")
            if policy_path:
                from sherpaos.sim.v26_runner import run_v26_episode

                result = run_v26_episode(
                    scenario,
                    spec["seed"],
                    policy_path=Path(policy_path),
                    g1_dir=Path(os.environ["SHERPA_G1_DIR"]),
                    max_steps=500,
                    command=command,
                    terrain_zone=_terrain_zone_for(spec["category"], spec["category_index"]),
                    wind_target_mps=_wind_for(spec["category"], spec["category_index"]),
                )
            else:
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
                mobility_failure=result.fell and scenario.friction < 0.6,
            )
            datasets.append(dataset)
            contexts.append(
                build_episode_context(
                    result.telemetry,
                    route,
                    route_fraction=_route_fraction_for(spec["category_index"]),
                    wind_mps=_wind_for(spec["category"], spec["category_index"]),
                )
            )
            truths.append(result.ground_truth)
            shard_rows.append(
                {
                    **spec,
                    "steps_completed": result.steps_survived,
                    "fell": result.fell,
                    "windows": int(dataset.observations.shape[0]),
                    "max_slip_mps": max(item.planted_foot_slip_mps for item in result.ground_truth),
                    "max_tilt_deg": max(
                        item.tilt_from_vertical_deg for item in result.ground_truth
                    ),
                    "max_slope_deg": max(item.true_slope_deg for item in result.ground_truth),
                    "unsafe_steps": sum(item.true_unsafe for item in result.ground_truth),
                    "terrain_zone": _terrain_zone_for(spec["category"], spec["category_index"]),
                    "wind_target_mps": _wind_for(spec["category"], spec["category_index"]),
                    "command": list(command),
                    "scenario": _jsonable_scenario(scenario),
                }
            )
        _write_shard(
            observation_path, label_path, context_path, datasets, contexts, truths, shard_rows
        )
        completed.extend(shard_rows)

    policy_path = os.environ.get("SHERPA_V26_POLICY")
    source_manifest = {
        "version": 2,
        "code_sha": git_sha(),
        "source_file_sha256": {
            relative: sha256_file(Path(relative)) for relative in PROVENANCE_FILES
        },
        "locomotion_policy_sha256": (sha256_file(Path(policy_path)) if policy_path else None),
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
        "qualification_mode": episodes == 20,
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


def _valid_existing_shard(
    obs_path: Path, label_path: Path, context_path: Path, specs: list[dict[str, Any]]
) -> bool:
    if not obs_path.is_file() or not label_path.is_file() or not context_path.is_file():
        return False
    try:
        with (
            np.load(obs_path, allow_pickle=False) as obs,
            np.load(label_path, allow_pickle=False) as labels,
            np.load(context_path, allow_pickle=False) as context,
        ):
            expected = {item["episode_id"] for item in specs}
            return (
                obs["observations"].ndim == 3
                and obs["observations"].shape[2] == OBSERVATION_WIDTH
                and np.all(np.isfinite(obs["observations"]))
                and set(labels["completed_episode_ids"].tolist()) == expected
                and int(labels["shard_schema_version"]) == SHARD_SCHEMA_VERSION
                and len(obs["episode_ids"]) == len(labels["episode_ids"])
                and len(context["episode_ids"]) == int(np.sum(labels["completed_steps"]))
            )
    except (OSError, ValueError, KeyError):
        return False


def _episode_rows_from_label(path: Path, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as labels:
        steps = labels["completed_steps"].tolist()
        falls = labels["episode_fell"].tolist()
        windows = labels["episode_window_counts"].tolist()
        max_slip = labels["episode_max_slip_mps"].tolist()
        max_tilt = labels["episode_max_tilt_deg"].tolist()
        max_slope = labels["episode_max_slope_deg"].tolist()
        unsafe_steps = labels["episode_unsafe_steps"].tolist()
        terrain_zones = labels["episode_terrain_zone"].tolist()
        winds = labels["episode_wind_target_mps"].tolist()
        commands = labels["episode_command"].tolist()
        scenarios = [json.loads(value) for value in labels["episode_scenario_json"].tolist()]
    return [
        {
            **spec,
            "steps_completed": int(step),
            "fell": bool(fell),
            "windows": int(count),
            "max_slip_mps": float(slip),
            "max_tilt_deg": float(tilt),
            "max_slope_deg": float(slope),
            "unsafe_steps": int(unsafe),
            "terrain_zone": int(zone),
            "wind_target_mps": float(wind),
            "command": [float(value) for value in command],
            "scenario": scenario,
        }
        for (
            spec,
            step,
            fell,
            count,
            slip,
            tilt,
            slope,
            unsafe,
            zone,
            wind,
            command,
            scenario,
        ) in zip(
            specs,
            steps,
            falls,
            windows,
            max_slip,
            max_tilt,
            max_slope,
            unsafe_steps,
            terrain_zones,
            winds,
            commands,
            scenarios,
            strict=True,
        )
    ]


def _write_shard(
    observation_path: Path,
    label_path: Path,
    context_path: Path,
    datasets: list[WindowedRiskDataset],
    contexts: list[EpisodeContext],
    truths: list[list[Any]],
    rows: list[dict[str, Any]],
) -> None:
    def combine(field: str) -> np.ndarray:
        arrays = [getattr(dataset, field) for dataset in datasets]
        return np.concatenate(arrays) if arrays else np.asarray([])

    observation_tmp = observation_path.with_suffix(".tmp.npz")
    label_tmp = label_path.with_suffix(".tmp.npz")
    context_tmp = context_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        observation_tmp,
        observations=combine("observations"),
        episode_ids=combine("episode_ids"),
        scenario_groups=combine("scenario_groups"),
    )
    np.savez_compressed(
        label_tmp,
        shard_schema_version=np.asarray(SHARD_SCHEMA_VERSION, dtype=np.int32),
        mobility_targets=combine("mobility_targets"),
        dynamics_targets=combine("dynamics_targets"),
        fall_targets=combine("fall_targets"),
        episode_ids=combine("episode_ids"),
        scenario_groups=combine("scenario_groups"),
        completed_episode_ids=np.asarray([row["episode_id"] for row in rows]),
        completed_steps=np.asarray([row["steps_completed"] for row in rows], dtype=np.int32),
        episode_fell=np.asarray([row["fell"] for row in rows], dtype=np.bool_),
        episode_window_counts=np.asarray([row["windows"] for row in rows], dtype=np.int32),
        episode_max_slip_mps=np.asarray([row["max_slip_mps"] for row in rows], dtype=np.float32),
        episode_max_tilt_deg=np.asarray([row["max_tilt_deg"] for row in rows], dtype=np.float32),
        episode_max_slope_deg=np.asarray([row["max_slope_deg"] for row in rows], dtype=np.float32),
        episode_unsafe_steps=np.asarray([row["unsafe_steps"] for row in rows], dtype=np.int32),
        episode_terrain_zone=np.asarray([row["terrain_zone"] for row in rows], dtype=np.int8),
        episode_wind_target_mps=np.asarray(
            [row["wind_target_mps"] for row in rows], dtype=np.float64
        ),
        episode_command=np.asarray([row["command"] for row in rows], dtype=np.float64),
        episode_scenario_json=np.asarray(
            [json.dumps(row["scenario"], sort_keys=True) for row in rows], dtype=str
        ),
        truth_episode_ids=np.concatenate(
            [
                np.repeat(row["episode_id"], len(trace))
                for row, trace in zip(rows, truths, strict=True)
            ]
        ),
        truth_control_steps=np.concatenate(
            [np.arange(len(trace), dtype=np.int32) for trace in truths]
        ),
        true_friction=np.concatenate(
            [
                np.asarray([item.true_friction for item in trace], dtype=np.float32)
                for trace in truths
            ]
        ),
        true_slope_deg=np.concatenate(
            [
                np.asarray([item.true_slope_deg for item in trace], dtype=np.float32)
                for trace in truths
            ]
        ),
        persistent_slip_mps=np.concatenate(
            [
                np.asarray([item.planted_foot_slip_mps for item in trace], dtype=np.float32)
                for trace in truths
            ]
        ),
        actuator_health=np.concatenate(
            [
                np.asarray([item.actuator_health for item in trace], dtype=np.float32)
                for trace in truths
            ]
        ),
        disturbance_active=np.concatenate(
            [
                np.asarray([item.disturbance_active for item in trace], dtype=np.bool_)
                for trace in truths
            ]
        ),
        tilt_deg=np.concatenate(
            [
                np.asarray([item.tilt_from_vertical_deg for item in trace], dtype=np.float32)
                for trace in truths
            ]
        ),
        unsafe_truth=np.concatenate(
            [np.asarray([item.true_unsafe for item in trace], dtype=np.bool_) for trace in truths]
        ),
    )
    context_arrays = {
        key: np.concatenate([item.arrays[key] for item in contexts]) for key in contexts[0].arrays
    }
    context_arrays["episode_ids"] = np.concatenate(
        [
            np.repeat(row["episode_id"], len(item.arrays["control_step"]))
            for row, item in zip(rows, contexts, strict=True)
        ]
    )
    np.savez_compressed(context_tmp, **context_arrays)
    os.replace(observation_tmp, observation_path)
    os.replace(label_tmp, label_path)
    os.replace(context_tmp, context_path)


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
