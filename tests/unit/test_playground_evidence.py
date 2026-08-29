from __future__ import annotations

import json
from pathlib import Path

import pytest

from sherpaos.evidence.manifest import sha256_of_file
from sherpaos.evidence.playground import (
    build_playground_rollout_manifest,
    write_playground_rollout_manifest,
)


def test_rollout_records_required_provenance_without_policy_claim(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.npz"
    trajectory.write_bytes(b"rollout")

    manifest = build_playground_rollout_manifest(
        seed=17,
        environment_name="G1JoystickFlatTerrain",
        environment_config={"episode_length": 1000, "command": "walk"},
        artifact_paths=[trajectory],
    )

    assert manifest.seed == 17
    assert manifest.environment_name == "G1JoystickFlatTerrain"
    assert len(manifest.environment_config_hash) == 64
    assert manifest.platform["python"]
    assert "devices" in manifest.gpu
    assert {"jax_version", "jaxlib_version", "backend", "devices"} <= manifest.jax.keys()
    assert manifest.policy == {
        "learned": False,
        "path": None,
        "sha256": None,
        "size_bytes": None,
        "provenance": None,
        "license": None,
    }
    assert manifest.artifacts[0]["sha256"] == sha256_of_file(trajectory)


def test_learned_policy_claim_requires_present_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a policy artifact"):
        build_playground_rollout_manifest(
            seed=1,
            environment_name="env",
            environment_config={},
            learned_policy=True,
        )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        build_playground_rollout_manifest(
            seed=1,
            environment_name="env",
            environment_config={},
            policy_path=tmp_path / "missing.ckpt",
            learned_policy=True,
        )


def test_policy_hash_and_artifact_manifest_are_written(tmp_path: Path) -> None:
    policy = tmp_path / "policy.ckpt"
    policy.write_bytes(b"weights-v1")
    video = tmp_path / "rollout.mp4"
    video.write_bytes(b"video")

    manifest = build_playground_rollout_manifest(
        seed=3,
        environment_name="G1RoughTerrain",
        environment_config={"friction_range": [0.1, 1.0]},
        artifact_paths=[video],
        policy_path=policy,
        learned_policy=True,
        policy_provenance="https://example.invalid/policy-card",
        policy_license="Apache-2.0",
        command="walk-forward",
        code_repository=Path.cwd(),
    )
    output = tmp_path / "manifest.json"
    write_playground_rollout_manifest(output, manifest)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["policy"]["learned"] is True
    assert payload["policy"]["provenance"] == "https://example.invalid/policy-card"
    assert payload["policy"]["license"] == "Apache-2.0"
    assert payload["command"] == "walk-forward"
    assert payload["code"]["commit_sha"]
    assert payload["policy"]["sha256"] == sha256_of_file(policy)
    assert {Path(item["path"]).name for item in payload["artifacts"]} == {
        "policy.ckpt",
        "rollout.mp4",
    }
    assert all(len(item["sha256"]) == 64 for item in payload["artifacts"])


def test_config_hash_is_independent_of_mapping_order() -> None:
    first = build_playground_rollout_manifest(
        seed=0,
        environment_name="env",
        environment_config={"a": 1, "b": 2},
    )
    second = build_playground_rollout_manifest(
        seed=0,
        environment_name="env",
        environment_config={"b": 2, "a": 1},
    )
    assert first.environment_config_hash == second.environment_config_hash
