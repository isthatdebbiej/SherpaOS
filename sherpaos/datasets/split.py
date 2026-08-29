"""Immutable scenario-group train/validation/test assignment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from sherpaos.datasets.manifest import read_json, write_json


def split_for_group(group_id: str, *, split_seed: int = 20260829) -> str:
    digest = hashlib.sha256(f"{split_seed}:{group_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def build_split_manifest(dataset: Path, config: Path) -> dict[str, Any]:
    dataset = Path(dataset)
    config = Path(config)
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenario = read_json(dataset / "scenario_manifest.json")
    seed = int(settings["seed"])
    group_membership: dict[str, str] = {}
    episode_membership: dict[str, str] = {}
    for spec in scenario["specs"]:
        group = spec["scenario_group"]
        split = group_membership.setdefault(group, split_for_group(group, split_seed=seed))
        episode_membership[spec["episode_id"]] = split
    result = {
        "version": 1,
        "seed": seed,
        "group_key": "scenario_group",
        "groups": group_membership,
        "episodes": episode_membership,
    }
    write_json(dataset / "split_manifest.json", result)
    return result
