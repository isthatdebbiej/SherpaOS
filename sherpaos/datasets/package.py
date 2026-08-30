"""Build a checksum-verifiable multi-cohort package for Hugging Face."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from sherpaos.datasets.manifest import read_checksums, read_json, sha256_file, write_json


def package_huggingface_collection(cohorts: list[tuple[str, Path]], output: Path) -> dict[str, Any]:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"package output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cohort_root = output / "cohorts"
    cohort_root.mkdir()
    indexes: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    cohort_records: list[dict[str, Any]] = []
    global_ids: set[str] = set()
    global_groups: dict[str, str] = {}

    for cohort_id, source in cohorts:
        source = Path(source)
        _verify_source(source)
        scenario = read_json(source / "scenario_manifest.json")
        split = read_json(source / "split_manifest.json")
        quality = read_json(source / "quality_report.json")
        destination = cohort_root / cohort_id
        shutil.copytree(source, destination)
        split_counts = Counter(split["episodes"].values())
        cohort_records.append(
            {
                "cohort_id": cohort_id,
                "dataset_id": scenario["dataset_id"],
                "episodes": len(scenario["specs"]),
                "splits": dict(sorted(split_counts.items())),
                "quality": quality,
                "source_manifest": read_json(source / "source_manifest.json"),
            }
        )
        shard_size = int(scenario["settings"]["shard_episodes"])
        for index, spec in enumerate(scenario["specs"]):
            local_episode_id = spec["episode_id"]
            role = split["episodes"][local_episode_id]
            global_episode_id = f"{cohort_id}/{local_episode_id}"
            global_group = f"{cohort_id}/{spec['scenario_group']}"
            if global_episode_id in global_ids:
                raise ValueError(f"duplicate global episode ID: {global_episode_id}")
            global_ids.add(global_episode_id)
            previous_role = global_groups.setdefault(global_group, role)
            if previous_role != role:
                raise ValueError(f"scenario group crosses splits: {global_group}")
            shard = f"shard-{index // shard_size:03d}.npz"
            indexes[role].append(
                {
                    "global_episode_id": global_episode_id,
                    "global_scenario_group": global_group,
                    "cohort_id": cohort_id,
                    "episode_id": local_episode_id,
                    "category": spec["category"],
                    "seed": spec["seed"],
                    "observations": f"cohorts/{cohort_id}/observations/{shard}",
                    "labels": f"cohorts/{cohort_id}/labels/{shard}",
                    "context": f"cohorts/{cohort_id}/context/{shard}",
                }
            )

    for role, rows in indexes.items():
        path = output / f"{role}_index.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    counts = {role: len(rows) for role, rows in indexes.items()}
    manifest = {
        "version": 1,
        "format": "sherpaos-huggingface-npz-v1",
        "feature_width": 103,
        "window_steps": 100,
        "prediction_horizon_steps": 50,
        "stride_samples": 10,
        "total_episodes": sum(counts.values()),
        "splits": counts,
        "cohorts": cohort_records,
        "privileged_labels_separate": True,
        "context_excluded_from_motion_model": True,
    }
    write_json(output / "collection_manifest.json", manifest)
    (output / "README.md").write_text(_dataset_card(manifest), encoding="utf-8")
    checksum_lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            relative = path.relative_to(output).as_posix()
            checksum_lines.append(f"{sha256_file(path)}  {relative}\n")
    (output / "checksums.sha256").write_text("".join(checksum_lines), encoding="utf-8")
    return manifest


def _verify_source(dataset: Path) -> None:
    quality = read_json(dataset / "quality_report.json")
    if quality.get("status") != "GREEN":
        raise ValueError(f"source dataset is not GREEN: {dataset}")
    expected = read_checksums(dataset)
    for relative, digest in expected.items():
        path = dataset / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"source checksum mismatch: {dataset}/{relative}")


def _dataset_card(manifest: dict[str, Any]) -> str:
    splits = manifest["splits"]
    return f"""---
pretty_name: SherpaOS Himalayan Risk 400
license: apache-2.0
task_categories:
  - time-series-forecasting
---

# SherpaOS Himalayan Risk 400

Controller-only MuJoCo G1 risk-supervision data from two immutable cohorts.
The collection contains {manifest["total_episodes"]} episodes: {splits["train"]} train,
{splits["validation"]} validation, and {splits["test"]} test.

Model inputs are 100x103 motion windows. Mobility, dynamics, fall, friction, slope,
slip, actuator, and unsafe truth remain in separate label artifacts. Battery,
telemetry-health, geographic context, current/forecast wind, and deterministic
GO/CAUTION/NO_GO labels remain in context artifacts and are excluded from the
learned motion model.

Use the split JSONL indexes as authoritative membership. Prefixing local IDs with
the cohort ID prevents collisions between the two source datasets.
"""
