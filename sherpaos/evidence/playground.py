"""Evidence metadata for MuJoCo Playground rollout artifacts.

This module deliberately records what ran, without inferring that a policy was
learned.  A rollout may use a hand-written, bundled, or absent policy; the
``learned`` flag can only be true when a real policy artifact is present.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sherpaos.evidence.manifest import sha256_of_file, sha256_of_text


@dataclass(frozen=True)
class PlaygroundRolloutManifest:
    """Auditable metadata attached to one Playground rollout."""

    schema_version: str
    created_at: str
    seed: int
    environment_name: str
    environment_config_hash: str
    command: str
    code: dict[str, Any]
    playground: dict[str, Any]
    platform: dict[str, Any]
    gpu: dict[str, Any]
    jax: dict[str, Any]
    policy: dict[str, Any]
    artifacts: tuple[dict[str, Any], ...]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _platform_metadata() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def _git_metadata(repository: Path | None) -> dict[str, Any]:
    if repository is None:
        return {"commit_sha": None, "dirty_worktree": None}
    root = Path(repository).resolve()

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-c", f"safe.directory={root.as_posix()}", *args],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit_sha": commit,
        "dirty_worktree": None if status is None else bool(status),
    }


def _gpu_metadata() -> dict[str, Any]:
    """Return best-effort GPU evidence without making GPU availability a requirement."""
    metadata: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": [],
        "probe": "nvidia-smi",
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        metadata["probe_status"] = "unavailable"
        return metadata

    if result.returncode != 0:
        metadata["probe_status"] = "failed"
        return metadata

    metadata["probe_status"] = "ok"
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            name, uuid, driver, memory_mib = fields
            metadata["devices"].append(
                {
                    "name": name,
                    "uuid": uuid,
                    "driver_version": driver,
                    "memory_total_mib": memory_mib,
                }
            )
    return metadata


def _jax_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "jax_version": _package_version("jax"),
        "jaxlib_version": _package_version("jaxlib"),
        "backend": None,
        "devices": [],
    }
    try:
        import jax  # type: ignore[import-not-found]

        metadata["backend"] = jax.default_backend()
        metadata["devices"] = [str(device) for device in jax.devices()]
    except (ImportError, RuntimeError):
        # CPU-only test and evidence hosts may intentionally have no working JAX.
        pass
    return metadata


def _artifact_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"evidence artifact does not exist: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_of_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def build_playground_rollout_manifest(
    *,
    seed: int,
    environment_name: str,
    environment_config: Mapping[str, Any],
    artifact_paths: Iterable[Path] = (),
    policy_path: Path | None = None,
    learned_policy: bool = False,
    policy_provenance: str | None = None,
    policy_license: str | None = None,
    command: str = "",
    code_repository: Path | None = None,
    playground_repository: Path | None = None,
) -> PlaygroundRolloutManifest:
    """Capture reproducible rollout provenance.

    ``learned_policy=True`` is an evidence claim, so it is rejected unless a
    readable policy file is supplied. Merely naming a controller or running a
    rollout never upgrades it to a learned-policy claim.
    """
    if learned_policy and policy_path is None:
        raise ValueError("learned_policy=True requires a policy artifact")

    policy: dict[str, Any] = {
        "learned": False,
        "path": None,
        "sha256": None,
        "size_bytes": None,
        "provenance": policy_provenance,
        "license": policy_license,
    }
    artifact_records = [_artifact_record(path) for path in artifact_paths]
    if policy_path is not None:
        policy_record = _artifact_record(policy_path)
        policy = {
            "learned": learned_policy,
            "path": policy_record["path"],
            "sha256": policy_record["sha256"],
            "size_bytes": policy_record["size_bytes"],
            "provenance": policy_provenance,
            "license": policy_license,
        }
        if all(record["path"] != policy_record["path"] for record in artifact_records):
            artifact_records.append(policy_record)

    config_json = json.dumps(environment_config, sort_keys=True, separators=(",", ":"))
    return PlaygroundRolloutManifest(
        schema_version="sherpaos.playground-rollout/v1",
        created_at=datetime.now(UTC).isoformat(),
        seed=int(seed),
        environment_name=environment_name,
        environment_config_hash=sha256_of_text(config_json),
        command=command,
        code=_git_metadata(code_repository),
        playground=_git_metadata(playground_repository),
        platform=_platform_metadata(),
        gpu=_gpu_metadata(),
        jax=_jax_metadata(),
        policy=policy,
        artifacts=tuple(sorted(artifact_records, key=lambda record: record["path"])),
    )


def write_playground_rollout_manifest(
    path: Path, manifest: PlaygroundRolloutManifest
) -> None:
    """Write rollout metadata as deterministic, human-readable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dataclasses.asdict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
