"""Unit tests for sherpaos.recorder.incident.IncidentRecorder.

Synthetic RobotTelemetry/GuardDecision/ActuationReceipt instances are built
locally so this test does not depend on any other lane's code.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np

from sherpaos.contracts import (
    ActuationReceipt,
    GuardAction,
    GuardDecision,
    ReasonCode,
    RobotTelemetry,
    TelemetrySource,
)
from sherpaos.evidence.bundle import verify_bundle
from sherpaos.recorder.incident import IncidentRecorder


def make_telemetry(seq: int, t: float = 0.0) -> RobotTelemetry:
    return RobotTelemetry(
        monotonic_time=t,
        source_time=t,
        sequence=seq,
        joint_position=np.zeros(12),
        joint_velocity=np.zeros(12),
        joint_effort=None,
        base_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        base_angular_velocity=np.zeros(3),
        base_linear_acceleration=np.array([0.0, 0.0, -9.81]),
        source=TelemetrySource.SIM,
        valid=True,
    )


def make_decision(seq: int, action: GuardAction, t: float = 0.0) -> GuardDecision:
    return GuardDecision(
        decision_id=f"dec-{seq}-{uuid.uuid4().hex[:8]}",
        action=action,
        score=0.1 if action == GuardAction.PASS else 0.8,
        confidence=0.9,
        reason_codes=(ReasonCode.NOMINAL,)
        if action == GuardAction.PASS
        else (ReasonCode.SLIP_RISK_HIGH,),
        input_age_seconds=0.01,
        requested_speed_limit=None if action == GuardAction.PASS else 0.3,
        timestamp=t,
        rules_version="v-test",
    )


def make_receipt(decision: GuardDecision, t: float = 0.0) -> ActuationReceipt:
    return ActuationReceipt(
        decision_id=decision.decision_id,
        requested_action=decision.action,
        applied_action=decision.action,
        accepted=True,
        rejection_reason=None,
        adapter_timestamp=t,
        acknowledgement_source="test-adapter",
    )


def test_pass_only_stream_produces_no_incidents(tmp_path: Path) -> None:
    recorder = IncidentRecorder(output_dir=tmp_path, pre_event_window=5, post_event_window=2)
    for i in range(20):
        recorder.observe(make_telemetry(i, float(i)), make_decision(i, GuardAction.PASS, float(i)))

    assert recorder.finalized_bundle_paths == []
    assert list(tmp_path.iterdir()) == []


def test_pass_limit_speed_pass_produces_exactly_one_incident(tmp_path: Path) -> None:
    recorder = IncidentRecorder(output_dir=tmp_path, pre_event_window=3, post_event_window=1)

    seq = 0
    for _ in range(4):
        recorder.observe(
            make_telemetry(seq, float(seq)), make_decision(seq, GuardAction.PASS, float(seq))
        )
        seq += 1

    recorder.observe(
        make_telemetry(seq, float(seq)), make_decision(seq, GuardAction.LIMIT_SPEED, float(seq))
    )
    seq += 1

    recorder.observe(
        make_telemetry(seq, float(seq)), make_decision(seq, GuardAction.PASS, float(seq))
    )
    seq += 1

    assert len(recorder.finalized_bundle_paths) == 1
    bundle_dir = recorder.finalized_bundle_paths[0]
    assert bundle_dir.exists()
    assert verify_bundle(bundle_dir) is True

    telemetry_lines = (bundle_dir / "telemetry.jsonl").read_text().strip().splitlines()
    decisions_lines = (bundle_dir / "decisions.jsonl").read_text().strip().splitlines()
    assert len(telemetry_lines) == len(decisions_lines) >= 3

    actions = [json.loads(line)["action"] for line in decisions_lines]
    assert GuardAction.LIMIT_SPEED.value in actions
    assert GuardAction.PASS.value in actions  # both pre- and post-event context present


def test_receipt_is_recorded_when_supplied(tmp_path: Path) -> None:
    recorder = IncidentRecorder(output_dir=tmp_path, pre_event_window=2, post_event_window=1)

    recorder.observe(make_telemetry(0, 0.0), make_decision(0, GuardAction.PASS, 0.0))
    decision = make_decision(1, GuardAction.REQUEST_HOLD, 1.0)
    receipt = make_receipt(decision, 1.0)
    recorder.observe(make_telemetry(1, 1.0), decision, receipt)
    recorder.observe(make_telemetry(2, 2.0), make_decision(2, GuardAction.PASS, 2.0))

    assert len(recorder.finalized_bundle_paths) == 1
    bundle_dir = recorder.finalized_bundle_paths[0]
    receipts_lines = (bundle_dir / "receipts.jsonl").read_text().strip().splitlines()
    assert len(receipts_lines) == 1
    assert json.loads(receipts_lines[0])["decision_id"] == decision.decision_id


def test_overlapping_incidents_each_finalize_independently(tmp_path: Path) -> None:
    recorder = IncidentRecorder(output_dir=tmp_path, pre_event_window=2, post_event_window=2)
    seq = 0

    def step(action: GuardAction) -> None:
        nonlocal seq
        recorder.observe(make_telemetry(seq, float(seq)), make_decision(seq, action, float(seq)))
        seq += 1

    step(GuardAction.PASS)
    step(GuardAction.LIMIT_SPEED)  # trigger #1
    step(GuardAction.REQUEST_HOLD)  # escalation -> trigger #2, while #1 still open
    step(GuardAction.PASS)
    step(GuardAction.PASS)

    assert len(recorder.finalized_bundle_paths) == 2
    # distinct incident directories, both individually valid
    assert len(set(recorder.finalized_bundle_paths)) == 2
    for bundle_dir in recorder.finalized_bundle_paths:
        assert verify_bundle(bundle_dir) is True


def test_flush_active_incidents_forces_finalization_at_stream_end(tmp_path: Path) -> None:
    recorder = IncidentRecorder(output_dir=tmp_path, pre_event_window=3, post_event_window=10)
    recorder.observe(make_telemetry(0, 0.0), make_decision(0, GuardAction.PASS, 0.0))
    recorder.observe(make_telemetry(1, 1.0), make_decision(1, GuardAction.REQUEST_HOLD, 1.0))

    # post_event_window (10) hasn't elapsed yet -> nothing finalized on its own
    assert recorder.finalized_bundle_paths == []

    flushed = recorder.flush_active_incidents()
    assert len(flushed) == 1
    assert verify_bundle(flushed[0]) is True
    assert recorder.finalized_bundle_paths == flushed
    # calling it again with nothing left active is a no-op, not an error
    assert recorder.flush_active_incidents() == []
