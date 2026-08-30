"""Create balanced group-isolated indexes without reading NPZ payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from sherpaos.datasets.manifest import sha256_file

ROLES = ("train", "validation", "test")


def assign_group_roles(rows, seed):
    strata = defaultdict(set)
    for row in rows:
        strata[(row["cohort_id"], row["category"])].add(row["global_scenario_group"])
    assignments = {}
    for stratum, groups in sorted(strata.items()):
        if len(groups) != 5:
            raise ValueError(f"{stratum} must contain five scenario groups")
        ordered = sorted(
            groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest()
        )
        for role, selected in (
            ("train", ordered[:3]),
            ("validation", ordered[3:4]),
            ("test", ordered[4:]),
        ):
            assignments.update({group: role for group in selected})
    return assignments


def reindex(source, output, seed=20260830):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        if any(output.iterdir()):
            raise ValueError(f"output must be empty: {output}")
        output.rmdir()
    shutil.copytree(source, output, copy_function=os.link)
    rows = []
    for role in ROLES:
        rows += [
            json.loads(line) for line in (source / f"{role}_index.jsonl").read_text().splitlines()
        ]
    if len({row["global_episode_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate global episode IDs")
    assignments = assign_group_roles(rows, seed)
    indexes = {role: [] for role in ROLES}
    for row in rows:
        indexes[assignments[row["global_scenario_group"]]].append(row)
    for role in ROLES:
        indexes[role].sort(key=lambda row: row["global_episode_id"])
        path = output / f"{role}_index.jsonl"
        path.unlink()
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in indexes[role]))
    counts = {role: len(indexes[role]) for role in ROLES}
    if counts != {"train": 240, "validation": 80, "test": 80}:
        raise ValueError(f"unexpected counts: {counts}")
    path = output / "collection_manifest.json"
    manifest = json.loads(path.read_text())
    path.unlink()
    manifest["version"] = 2
    manifest["splits"] = counts
    manifest["split_strategy"] = {
        "name": "cohort_category_stratified_scenario_group",
        "seed": seed,
        "groups_per_cohort_category": {"train": 3, "validation": 1, "test": 1},
        "test_payload_opened": False,
    }
    for cohort in manifest["cohorts"]:
        cohort["splits"] = {"train": 120, "validation": 40, "test": 40}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    readme = output / "README.md"
    text = readme.read_text()
    readme.unlink()
    readme.write_text(
        text.replace(
            "240 train,\n20 validation, and 140 test", "240 train,\n80 validation, and 80 test"
        )
        + "\nV2 balances every cohort/scenario-family stratum by immutable scenario group.\n"
    )
    checksums = output / "checksums.sha256"
    checksums.unlink()
    checksums.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in sorted(output.rglob("*"))
            if path.is_file()
        )
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    print(json.dumps(reindex(args.source, args.output, args.seed)["splits"], sort_keys=True))


if __name__ == "__main__":
    main()
