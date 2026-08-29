from __future__ import annotations

import numpy as np

from sherpaos.contracts import (
    GuardAction,
    GuardName,
    GuardReport,
    MissionContext,
    ReasonCode,
    RobotTelemetry,
    TelemetrySource,
)
from sherpaos.evidence.bundle import decision_from_dict, decision_to_dict
from sherpaos.policy.guards import FiveGuardSupervisor, fuse_guard_reports


def report(
    guard: GuardName, score: float, action: GuardAction, confidence: float = 1.0
) -> GuardReport:
    return GuardReport(
        guard=guard,
        score=score,
        confidence=confidence,
        reason_codes=(ReasonCode.NOMINAL,),
        recommended_action=action,
    )


def telemetry(now: float = 10.0) -> RobotTelemetry:
    return RobotTelemetry(
        monotonic_time=now,
        source_time=now,
        sequence=1,
        joint_position=np.zeros(29),
        joint_velocity=np.zeros(29),
        joint_effort=np.zeros(29),
        base_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        base_angular_velocity=np.zeros(3),
        base_linear_acceleration=np.array([0.0, 0.0, 9.81]),
        commanded_velocity=np.zeros(3),
        battery_fraction=0.8,
        battery_voltage=50.0,
        battery_current_a=2.0,
        battery_temperature_c=20.0,
        source=TelemetrySource.SIM,
        valid=True,
    )


def context(now: float = 10.0) -> MissionContext:
    return MissionContext(
        latitude=27.7,
        longitude=86.7,
        elevation_m=3000.0,
        slope_deg=2.0,
        route_segment="test",
        distance_to_safe_waypoint_m=1000.0,
        exposure_class="LOW",
        terrain_source="test",
        terrain_version="1",
        coordinate_reference_system="EPSG:4326",
        lookup_timestamp=now,
        valid=True,
        resolution_m=30.0,
        provenance="unit-test",
    )


def test_hold_report_cannot_be_diluted_by_nominal_guards():
    reports = (
        report(GuardName.MOBILITY, 0.1, GuardAction.PASS),
        report(GuardName.DYNAMICS, 0.2, GuardAction.PASS),
        report(GuardName.BATTERY, 0.2, GuardAction.REQUEST_HOLD),
        report(GuardName.GEOGRAPHIC, 0.0, GuardAction.PASS),
    )
    fused = fuse_guard_reports(reports, age_seconds=0.01)
    assert fused.score >= 0.7


def test_lowest_guard_confidence_is_preserved():
    reports = (
        report(GuardName.MOBILITY, 0.0, GuardAction.PASS, confidence=0.9),
        report(GuardName.TELEMETRY_HEALTH, 0.0, GuardAction.PASS, confidence=0.2),
    )
    assert fuse_guard_reports(reports, age_seconds=0.0).confidence == 0.2


def test_supervisor_emits_all_five_named_reports():
    decision = FiveGuardSupervisor().decide(telemetry(), context(), now=10.0)
    assert {item.guard for item in decision.guard_reports} == set(GuardName)
    assert decision.action in GuardAction


def test_missing_geography_never_silently_passes():
    supervisor = FiveGuardSupervisor()
    decision = supervisor.decide(telemetry(), None, now=10.0)
    geographic = next(
        item for item in decision.guard_reports if item.guard == GuardName.GEOGRAPHIC
    )
    assert geographic.recommended_action == GuardAction.LIMIT_SPEED
    assert ReasonCode.GEOGRAPHIC_CONTEXT_UNAVAILABLE in geographic.reason_codes


def test_decision_evidence_round_trip_preserves_guard_reports():
    decision = FiveGuardSupervisor().decide(telemetry(), context(), now=10.0)
    restored = decision_from_dict(decision_to_dict(decision))
    assert restored == decision
