"""On-disk evidence bundle format.

An evidence bundle is a directory containing:

- `telemetry.jsonl`  -- one `RobotTelemetry` per line
- `decisions.jsonl`  -- one `GuardDecision` per line
- `receipts.jsonl`   -- one `ActuationReceipt` per line
- `manifest.json`    -- a `RunManifest` (only written if one was supplied)
- `checksums.json`   -- sha256 of every file above, for later tamper-evident
                         verification via `verify_bundle`

JSONL (one JSON object per line) is used instead of a single JSON array so a
partially-written or truncated file still yields whatever complete records
were flushed, and so incident bundles can be appended-to/streamed in the
future without rewriting the whole file. `sherpaos.recorder.incident` writes
per-incident bundles here; a full-run evidence bundle can use the exact same
writer.

`RobotTelemetry`/`GuardDecision`/`ActuationReceipt` contain numpy arrays and
str-backed enums that plain `json` cannot serialize on its own. `_jsonable`
below is the single shared conversion used for all three record types (and
is also how a manifest's `artifact_checksums` etc. would be handled, though
`RunManifest` itself has no numpy/enum fields and round-trips via plain
`dataclasses.asdict`/`json` in `evidence/manifest.py`). Reconstructing a
record from JSON *does* need one function per dataclass, since JSON itself
doesn't carry enough type information to know that e.g. `"action": "PASS"`
should become `GuardAction.PASS` rather than stay a plain string.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from sherpaos.contracts import (
    ActuationReceipt,
    GuardAction,
    GuardDecision,
    ReasonCode,
    RobotTelemetry,
    RunManifest,
    TelemetrySource,
)
from sherpaos.evidence.manifest import read_manifest, sha256_of_file, write_manifest

TELEMETRY_FILENAME = "telemetry.jsonl"
DECISIONS_FILENAME = "decisions.jsonl"
RECEIPTS_FILENAME = "receipts.jsonl"
MANIFEST_FILENAME = "manifest.json"
CHECKSUMS_FILENAME = "checksums.json"


def _jsonable(value):
    """Recursively convert numpy arrays into plain lists.

    str-backed enums (`GuardAction`, `ReasonCode`, `TelemetrySource`) are
    already genuine `str` instances (they subclass `str`), so `json.dumps`
    serializes them directly to their value (e.g. `"PASS"`) with no help
    needed here -- only `np.ndarray` leaves require conversion.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _encode(obj) -> dict:
    """Turn a frozen dataclass (telemetry/decision/receipt) into a JSON-safe dict."""
    return _jsonable(dataclasses.asdict(obj))


def telemetry_to_dict(telemetry: RobotTelemetry) -> dict:
    return _encode(telemetry)


def telemetry_from_dict(data: dict) -> RobotTelemetry:
    def _array(key: str) -> np.ndarray | None:
        value = data.get(key)
        return np.asarray(value, dtype=float) if value is not None else None

    return RobotTelemetry(
        monotonic_time=data["monotonic_time"],
        source_time=data["source_time"],
        sequence=data["sequence"],
        joint_position=np.asarray(data["joint_position"], dtype=float),
        joint_velocity=np.asarray(data["joint_velocity"], dtype=float),
        joint_effort=_array("joint_effort"),
        base_orientation=np.asarray(data["base_orientation"], dtype=float),
        base_angular_velocity=np.asarray(data["base_angular_velocity"], dtype=float),
        base_linear_acceleration=np.asarray(data["base_linear_acceleration"], dtype=float),
        commanded_velocity=_array("commanded_velocity"),
        gait_mode=data.get("gait_mode"),
        battery_fraction=data.get("battery_fraction"),
        battery_voltage=data.get("battery_voltage"),
        source=TelemetrySource(data.get("source", TelemetrySource.SIM.value)),
        valid=data.get("valid", True),
        field_provenance=dict(data.get("field_provenance") or {}),
    )


def decision_to_dict(decision: GuardDecision) -> dict:
    return _encode(decision)


def decision_from_dict(data: dict) -> GuardDecision:
    return GuardDecision(
        decision_id=data["decision_id"],
        action=GuardAction(data["action"]),
        score=data["score"],
        confidence=data["confidence"],
        reason_codes=tuple(ReasonCode(code) for code in data.get("reason_codes", ())),
        input_age_seconds=data["input_age_seconds"],
        requested_speed_limit=data.get("requested_speed_limit"),
        timestamp=data["timestamp"],
        rules_version=data["rules_version"],
        model_version=data.get("model_version"),
    )


def receipt_to_dict(receipt: ActuationReceipt) -> dict:
    return _encode(receipt)


def receipt_from_dict(data: dict) -> ActuationReceipt:
    return ActuationReceipt(
        decision_id=data["decision_id"],
        requested_action=GuardAction(data["requested_action"]),
        applied_action=GuardAction(data["applied_action"]),
        accepted=data["accepted"],
        rejection_reason=data.get("rejection_reason"),
        adapter_timestamp=data["adapter_timestamp"],
        acknowledgement_source=data["acknowledgement_source"],
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_evidence_bundle(
    dir_path: Path,
    telemetry: list[RobotTelemetry],
    decisions: list[GuardDecision],
    receipts: list[ActuationReceipt],
    manifest: RunManifest | None = None,
) -> Path:
    """Write a full evidence bundle to `dir_path` and return that path.

    Overwrites files in place if `dir_path` already exists (used by the
    incident recorder, which always writes a fresh, complete bundle per
    incident rather than appending).
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    _write_jsonl(dir_path / TELEMETRY_FILENAME, [telemetry_to_dict(t) for t in telemetry])
    _write_jsonl(dir_path / DECISIONS_FILENAME, [decision_to_dict(d) for d in decisions])
    _write_jsonl(dir_path / RECEIPTS_FILENAME, [receipt_to_dict(r) for r in receipts])

    written_files = [TELEMETRY_FILENAME, DECISIONS_FILENAME, RECEIPTS_FILENAME]

    if manifest is not None:
        write_manifest(dir_path / MANIFEST_FILENAME, manifest)
        written_files.append(MANIFEST_FILENAME)

    checksums = {name: sha256_of_file(dir_path / name) for name in written_files}
    (dir_path / CHECKSUMS_FILENAME).write_text(json.dumps(checksums, indent=2, sort_keys=True))

    return dir_path


def read_evidence_bundle(
    dir_path: Path,
) -> tuple[list[RobotTelemetry], list[GuardDecision], list[ActuationReceipt], RunManifest | None]:
    """Read back everything `write_evidence_bundle` wrote (does not verify checksums)."""
    dir_path = Path(dir_path)
    telemetry = [telemetry_from_dict(row) for row in _read_jsonl(dir_path / TELEMETRY_FILENAME)]
    decisions = [decision_from_dict(row) for row in _read_jsonl(dir_path / DECISIONS_FILENAME)]
    receipts = [receipt_from_dict(row) for row in _read_jsonl(dir_path / RECEIPTS_FILENAME)]
    manifest_path = dir_path / MANIFEST_FILENAME
    manifest = read_manifest(manifest_path) if manifest_path.exists() else None
    return telemetry, decisions, receipts, manifest


def verify_bundle(dir_path: Path) -> bool:
    """Recompute checksums for every file `checksums.json` lists and compare.

    Returns `False` (never raises) on a missing `checksums.json`, a missing
    listed file, a corrupted/truncated file, or a malformed checksums file --
    this is meant to be a safe, boolean-returning tamper check.
    """
    dir_path = Path(dir_path)
    checksums_path = dir_path / CHECKSUMS_FILENAME
    if not checksums_path.exists():
        return False
    try:
        checksums = json.loads(checksums_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(checksums, dict):
        return False

    for filename, expected_hash in checksums.items():
        file_path = dir_path / filename
        if not file_path.exists():
            return False
        try:
            actual_hash = sha256_of_file(file_path)
        except OSError:
            return False
        if actual_hash != expected_hash:
            return False
    return True
