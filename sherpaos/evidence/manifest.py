"""Reproducibility manifest: build/write/read `RunManifest` records.

A `RunManifest` pins down everything needed to reconstruct a run later:
the code (commit + dependency lock), the configuration and controller/model
identities that were active, the machine it ran on, and checksums of any
artifacts worth tying to the run. See `docs/CONTRACTS.md` for intent and
`sherpaos.contracts.RunManifest` for the frozen schema.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from sherpaos.contracts import RunManifest

# sherpaos/sherpaos/evidence/manifest.py -> parents[0]=evidence, [1]=sherpaos
# (inner package), [2]=repo root (contains pyproject.toml, uv.lock, .git).
_REPO_ROOT = Path(__file__).resolve().parents[2]

_UNKNOWN_COMMIT_SENTINEL = "unknown"
_MISSING_LOCK_SENTINEL = "unknown-missing-uv-lock"

_CHUNK_SIZE = 1 << 16  # 64 KiB


def sha256_of_file(path: Path) -> str:
    """Return the hex sha256 digest of a file's contents.

    Reusable by both this module (manifest artifact checksums, dependency
    lock hash) and `evidence/bundle.py` (per-file bundle checksums) so
    there is exactly one place that decides how a file becomes a digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_text(text: str) -> str:
    """Return the hex sha256 digest of a UTF-8 encoded string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit_sha(repo_root: Path) -> str:
    """Best-effort `git rev-parse HEAD`; never raises.

    Falls back to a clear sentinel (rather than crashing) if git is
    missing, the directory isn't a repo yet, or there are no commits.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return _UNKNOWN_COMMIT_SENTINEL
    if result.returncode != 0:
        return _UNKNOWN_COMMIT_SENTINEL
    sha = result.stdout.strip()
    return sha if sha else _UNKNOWN_COMMIT_SENTINEL


def build_run_manifest(
    run_id: str,
    seed: int,
    scenario_name: str,
    config: dict,
    controller_id: str,
    model_id: str | None = None,
    extra_artifact_paths: list[Path] = (),
) -> RunManifest:
    """Construct a `RunManifest` describing the current run/environment.

    `controller_id`/`model_id` are caller-supplied identity strings (e.g.
    a version tag or hash the caller already computed) -- this function
    just hashes whatever identity it is given, it does not hash actual
    controller/model code itself.
    """
    config_hash = sha256_of_text(json.dumps(config, sort_keys=True))
    controller_hash = sha256_of_text(controller_id)
    model_hash = sha256_of_text(model_id) if model_id is not None else None

    lock_path = _REPO_ROOT / "uv.lock"
    dependency_lock_hash = (
        sha256_of_file(lock_path) if lock_path.exists() else _MISSING_LOCK_SENTINEL
    )

    artifact_checksums = {
        str(path): sha256_of_file(path) for path in extra_artifact_paths if Path(path).exists()
    }

    return RunManifest(
        run_id=run_id,
        commit_sha=_git_commit_sha(_REPO_ROOT),
        config_hash=config_hash,
        controller_hash=controller_hash,
        model_hash=model_hash,
        data_hash=None,
        dependency_lock_hash=dependency_lock_hash,
        container_identity=None,
        seed=seed,
        scenario_name=scenario_name,
        runtime_identity=f"{platform.system()}-{platform.release()}-py{platform.python_version()}",
        hardware_identity=f"{platform.machine()}-{platform.processor()}",
        artifact_checksums=artifact_checksums,
        created_at=datetime.now(UTC).isoformat(),
    )


def write_manifest(path: Path, manifest: RunManifest) -> None:
    """Write a `RunManifest` as pretty-printed, deterministically ordered JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(manifest), indent=2, sort_keys=True))


def read_manifest(path: Path) -> RunManifest:
    """Read back a `RunManifest` written by `write_manifest` (lossless round-trip).

    `RunManifest`'s fields are all JSON-native (str/int/None/dict[str, str]),
    so a plain `json.loads` + keyword-construction round-trips exactly --
    no numpy/enum conversion is needed here (unlike the telemetry/decision/
    receipt records handled in `evidence/bundle.py`).
    """
    data = json.loads(Path(path).read_text())
    return RunManifest(**data)
