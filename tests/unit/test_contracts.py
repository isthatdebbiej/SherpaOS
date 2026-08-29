"""Structural/schema tests for sherpaos.contracts.

These tests pin down the frozen contract described in docs/CONTRACTS.md:
dataclasses are immutable, the action/source enums have exactly the
membership the rest of the system depends on, `age_seconds` never goes
negative, and no field name on RobotTelemetry hints at simulator ground
truth.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sherpaos.contracts import (
    ActuationReceipt,
    GuardAction,
    GuardDecision,
    ReasonCode,
    RobotTelemetry,
    RunManifest,
    TelemetrySource,
)

# ---------------------------------------------------------------------------
# Frozen-ness
# ---------------------------------------------------------------------------


def test_robot_telemetry_is_frozen(make_telemetry):
    telemetry = make_telemetry()
    with pytest.raises(dataclasses.FrozenInstanceError):
        telemetry.sequence = 99  # type: ignore[misc]


def test_guard_decision_is_frozen():
    decision = GuardDecision(
        decision_id="d-1",
        action=GuardAction.PASS,
        score=0.0,
        confidence=1.0,
        reason_codes=(ReasonCode.NOMINAL,),
        input_age_seconds=0.0,
        requested_speed_limit=None,
        timestamp=0.0,
        rules_version="v0",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.action = GuardAction.REQUEST_HOLD  # type: ignore[misc]


def test_actuation_receipt_is_frozen():
    receipt = ActuationReceipt(
        decision_id="d-1",
        requested_action=GuardAction.PASS,
        applied_action=GuardAction.PASS,
        accepted=True,
        rejection_reason=None,
        adapter_timestamp=0.0,
        acknowledgement_source="sim",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.accepted = False  # type: ignore[misc]


def test_run_manifest_is_frozen():
    manifest = RunManifest(
        run_id="r-1",
        commit_sha="deadbeef",
        config_hash="cfg",
        controller_hash="ctrl",
        model_hash=None,
        data_hash=None,
        dependency_lock_hash="lock",
        container_identity=None,
        seed=0,
        scenario_name="nominal",
        runtime_identity="local",
        hardware_identity="cpu",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.seed = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum membership
# ---------------------------------------------------------------------------


def test_guard_action_has_exactly_three_members():
    assert {member.value for member in GuardAction} == {
        "PASS",
        "LIMIT_SPEED",
        "REQUEST_HOLD",
    }
    assert len(GuardAction) == 3


def test_telemetry_source_has_exactly_three_members():
    assert {member.value for member in TelemetrySource} == {"sim", "dump", "g1_live"}
    assert len(TelemetrySource) == 3


# ---------------------------------------------------------------------------
# age_seconds
# ---------------------------------------------------------------------------


def test_age_seconds_never_negative_on_clock_rewind(make_telemetry):
    telemetry = make_telemetry(monotonic_time=100.0)
    # `now` is before the sample's own monotonic_time (clock weirdness).
    assert telemetry.age_seconds(50.0) == 0.0


def test_age_seconds_positive_difference(make_telemetry):
    telemetry = make_telemetry(monotonic_time=100.0)
    assert telemetry.age_seconds(103.5) == pytest.approx(3.5)


def test_age_seconds_zero_when_equal(make_telemetry):
    telemetry = make_telemetry(monotonic_time=100.0)
    assert telemetry.age_seconds(100.0) == 0.0


# ---------------------------------------------------------------------------
# No simulator ground truth hiding in the schema
# ---------------------------------------------------------------------------

_FORBIDDEN_NAME_FRAGMENTS = ("friction", "true_", "ground_truth", "injected_fault", "true_fall")


def test_robot_telemetry_has_no_ground_truth_fields():
    field_names = [f.name for f in dataclasses.fields(RobotTelemetry)]
    for name in field_names:
        lowered = name.lower()
        for fragment in _FORBIDDEN_NAME_FRAGMENTS:
            assert fragment not in lowered, (
                f"RobotTelemetry field {name!r} looks like simulator ground truth "
                f"(matched fragment {fragment!r}); see docs/CONTRACTS.md leakage rule."
            )


def test_make_telemetry_fixture_produces_expected_shapes(make_telemetry):
    telemetry = make_telemetry()
    assert telemetry.joint_position.shape == (29,)
    assert telemetry.joint_velocity.shape == (29,)
    assert telemetry.joint_effort is not None
    assert telemetry.joint_effort.shape == (29,)
    assert telemetry.base_orientation.shape == (4,)
    assert np.isclose(np.linalg.norm(telemetry.base_orientation), 1.0)
    assert telemetry.source is TelemetrySource.SIM
    assert telemetry.valid is True


def test_make_telemetry_fixture_accepts_overrides(make_telemetry):
    stale = make_telemetry(monotonic_time=-1000.0)
    assert stale.monotonic_time == -1000.0

    corrupt = make_telemetry(joint_position=np.full(29, np.nan))
    assert np.all(np.isnan(corrupt.joint_position))
