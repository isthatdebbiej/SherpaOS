"""Unit tests for sherpaos.evidence.manifest and sherpaos.evidence.bundle.

Synthetic RobotTelemetry/GuardDecision/ActuationReceipt instances are built
locally so this test does not depend on any other lane's code.
"""

from __future__ import annotations

import dataclasses
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
from sherpaos.evidence.bundle import verify_bundle, write_evidence_bundle
from sherpaos.evidence.manifest import (
    build_run_manifest,
    read_manifest,
    sha256_of_file,
    write_manifest,
)


def _telemetry(seq: int = 0) -> RobotTelemetry:
    return RobotTelemetry(
        monotonic_time=float(seq),
        source_time=float(seq),
        sequence=seq,
        joint_position=np.arange(6, dtype=float),
        joint_velocity=np.zeros(6),
        joint_effort=None,
        base_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        base_angular_velocity=np.zeros(3),
        base_linear_acceleration=np.array([0.0, 0.0, -9.81]),
        source=TelemetrySource.SIM,
        valid=True,
    )


def _decision(seq: int = 0) -> GuardDecision:
    return GuardDecision(
        decision_id=f"dec-{seq}",
        action=GuardAction.LIMIT_SPEED,
        score=0.7,
        confidence=0.9,
        reason_codes=(ReasonCode.SLIP_RISK_HIGH,),
        input_age_seconds=0.02,
        requested_speed_limit=0.5,
        timestamp=float(seq),
        rules_version="v-test",
    )


def _receipt(decision: GuardDecision) -> ActuationReceipt:
    return ActuationReceipt(
        decision_id=decision.decision_id,
        requested_action=decision.action,
        applied_action=decision.action,
        accepted=True,
        rejection_reason=None,
        adapter_timestamp=decision.timestamp,
        acknowledgement_source="test-adapter",
    )


# --- build_run_manifest -----------------------------------------------------


def test_build_run_manifest_field_types_match_contract() -> None:
    manifest = build_run_manifest(
        run_id="run-123",
        seed=42,
        scenario_name="icy_ramp",
        config={"b": 2, "a": 1},
        controller_id="pd-controller-v0",
        model_id=None,
    )

    assert isinstance(manifest, RunManifest)
    assert manifest.run_id == "run-123"
    assert isinstance(manifest.commit_sha, str) and manifest.commit_sha
    assert isinstance(manifest.config_hash, str) and len(manifest.config_hash) == 64
    assert isinstance(manifest.controller_hash, str) and len(manifest.controller_hash) == 64
    assert manifest.model_hash is None
    assert manifest.data_hash is None
    assert isinstance(manifest.dependency_lock_hash, str) and manifest.dependency_lock_hash
    assert manifest.container_identity is None
    assert manifest.seed == 42
    assert manifest.scenario_name == "icy_ramp"
    assert isinstance(manifest.runtime_identity, str) and manifest.runtime_identity
    assert isinstance(manifest.hardware_identity, str) and manifest.hardware_identity
    assert isinstance(manifest.artifact_checksums, dict) and manifest.artifact_checksums == {}
    assert isinstance(manifest.created_at, str) and manifest.created_at


def test_build_run_manifest_config_hash_independent_of_key_order() -> None:
    manifest1 = build_run_manifest(
        run_id="run-a",
        seed=1,
        scenario_name="s",
        config={"a": 1, "b": 2},
        controller_id="controller-x",
    )
    manifest2 = build_run_manifest(
        run_id="run-b",
        seed=1,
        scenario_name="s",
        config={"b": 2, "a": 1},
        controller_id="controller-x",
    )
    assert manifest1.config_hash == manifest2.config_hash
    # same controller identity string -> same controller hash too
    assert manifest1.controller_hash == manifest2.controller_hash


def test_build_run_manifest_with_model_id_and_artifact_checksums(tmp_path: Path) -> None:
    artifact = tmp_path / "plot.png"
    artifact.write_bytes(b"fake-png-bytes")

    manifest = build_run_manifest(
        run_id="run-125",
        seed=7,
        scenario_name="wet_tile",
        config={"x": 1},
        controller_id="pd-controller-v0",
        model_id="risk-model-v1",
        extra_artifact_paths=[artifact],
    )

    assert manifest.model_hash is not None and len(manifest.model_hash) == 64
    assert manifest.artifact_checksums == {str(artifact): sha256_of_file(artifact)}


def test_build_run_manifest_skips_missing_extra_artifacts(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.bin"
    manifest = build_run_manifest(
        run_id="run-126",
        seed=0,
        scenario_name="s",
        config={},
        controller_id="c",
        extra_artifact_paths=[missing],
    )
    assert manifest.artifact_checksums == {}


# --- write_manifest / read_manifest -----------------------------------------


def test_write_read_manifest_roundtrips_losslessly(tmp_path: Path) -> None:
    manifest = build_run_manifest(
        run_id="run-127",
        seed=1,
        scenario_name="baseline",
        config={"k": "v"},
        controller_id="controller-x",
        model_id="model-y",
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)

    assert path.exists()
    round_tripped = read_manifest(path)
    assert round_tripped == manifest
    assert dataclasses.asdict(round_tripped) == dataclasses.asdict(manifest)


# --- evidence bundle: write / verify / corruption ---------------------------


def test_evidence_bundle_round_trips_and_verifies(tmp_path: Path) -> None:
    telemetry = [_telemetry(i) for i in range(3)]
    decisions = [_decision(i) for i in range(3)]
    receipts = [_receipt(d) for d in decisions]
    manifest = build_run_manifest(
        run_id="run-bundle", seed=0, scenario_name="s", config={}, controller_id="c"
    )

    bundle_dir = tmp_path / "incident-xyz"
    write_evidence_bundle(bundle_dir, telemetry, decisions, receipts, manifest=manifest)

    expected_files = (
        "telemetry.jsonl",
        "decisions.jsonl",
        "receipts.jsonl",
        "manifest.json",
        "checksums.json",
    )
    for filename in expected_files:
        assert (bundle_dir / filename).exists()

    assert verify_bundle(bundle_dir) is True


def test_verify_bundle_detects_corrupted_byte(tmp_path: Path) -> None:
    telemetry = [_telemetry(0)]
    decisions = [_decision(0)]
    receipts = [_receipt(decisions[0])]
    bundle_dir = tmp_path / "incident-corrupt"
    write_evidence_bundle(bundle_dir, telemetry, decisions, receipts)
    assert verify_bundle(bundle_dir) is True

    telem_path = bundle_dir / "telemetry.jsonl"
    data = bytearray(telem_path.read_bytes())
    data[0] = (data[0] + 1) % 256
    telem_path.write_bytes(bytes(data))

    assert verify_bundle(bundle_dir) is False


def test_verify_bundle_detects_truncated_file(tmp_path: Path) -> None:
    telemetry = [_telemetry(i) for i in range(2)]
    decisions = [_decision(i) for i in range(2)]
    receipts = [_receipt(d) for d in decisions]
    bundle_dir = tmp_path / "incident-truncated"
    write_evidence_bundle(bundle_dir, telemetry, decisions, receipts)
    assert verify_bundle(bundle_dir) is True

    decisions_path = bundle_dir / "decisions.jsonl"
    truncated = decisions_path.read_bytes()[:5]
    decisions_path.write_bytes(truncated)

    assert verify_bundle(bundle_dir) is False


def test_verify_bundle_missing_directory_returns_false(tmp_path: Path) -> None:
    assert verify_bundle(tmp_path / "does-not-exist") is False


def test_verify_bundle_missing_listed_file_returns_false(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "incident-missing-file"
    write_evidence_bundle(bundle_dir, [_telemetry(0)], [_decision(0)], [])
    (bundle_dir / "telemetry.jsonl").unlink()

    assert verify_bundle(bundle_dir) is False
