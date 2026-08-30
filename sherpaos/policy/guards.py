"""Five independent guard reports and conservative fusion.

The motion-related guards share one bounded feature window so they evaluate
the same telemetry sample without duplicating history. Battery and geography
retain their dedicated implementations. Fusion is deliberately max-based:
one severe guard cannot be diluted by four nominal guards.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from sherpaos.battery.guard import BatteryMarginGuard
from sherpaos.contracts import (
    GuardAction,
    GuardDecision,
    GuardName,
    GuardReport,
    MissionContext,
    ReasonCode,
    RobotTelemetry,
)
from sherpaos.estimator.features import Features, FeatureWindow
from sherpaos.estimator.risk import (
    ASYMMETRY_THRESHOLD,
    BODY_ANOMALY_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    ORIENT_ELEVATED_THRESHOLD,
    SLIP_ELEVATED_THRESHOLD,
    SLIP_HIGH_THRESHOLD,
    RiskEstimate,
    asymmetry_component,
    body_component,
    data_quality_gate,
    orientation_component,
    slip_component,
)
from sherpaos.geography.guard import GeographicRiskGuard
from sherpaos.policy.state_machine import (
    SCORE_LIMIT_SPEED_THRESHOLD,
    SCORE_REQUEST_HOLD_THRESHOLD,
    PolicyStateMachine,
)

_ACTION_RANK = {
    GuardAction.PASS: 0,
    GuardAction.LIMIT_SPEED: 1,
    GuardAction.REQUEST_HOLD: 2,
}


def _clip_score(value: float) -> float:
    if not math.isfinite(value):
        return 1.0
    return min(1.0, max(0.0, value))


def _clip_confidence(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _action_for(score: float, confidence: float) -> GuardAction:
    if score >= SCORE_REQUEST_HOLD_THRESHOLD:
        return GuardAction.REQUEST_HOLD
    if score >= SCORE_LIMIT_SPEED_THRESHOLD or confidence < LOW_CONFIDENCE_THRESHOLD:
        return GuardAction.LIMIT_SPEED
    return GuardAction.PASS


def _report(
    guard: GuardName,
    score: float,
    confidence: float,
    reasons: Iterable[ReasonCode],
    provenance: dict[str, str],
    *,
    action: GuardAction | None = None,
) -> GuardReport:
    bounded_score = _clip_score(score)
    bounded_confidence = _clip_confidence(confidence)
    reason_tuple = tuple(dict.fromkeys(reasons)) or (ReasonCode.NOMINAL,)
    return GuardReport(
        guard=guard,
        score=bounded_score,
        confidence=bounded_confidence,
        reason_codes=reason_tuple,
        recommended_action=action or _action_for(bounded_score, bounded_confidence),
        provenance=provenance,
    )


class MotionGuardSuite:
    """Produce mobility, dynamics, and telemetry-health reports per sample."""

    def __init__(self, window: FeatureWindow | None = None) -> None:
        self._window = window if window is not None else FeatureWindow()

    def observe(self, telemetry: RobotTelemetry, now: float) -> tuple[GuardReport, ...]:
        try:
            self._window.push(telemetry)
            features = self._window.compute_features(now)
            return self._reports(features, telemetry)
        except Exception:
            failed = _report(
                GuardName.TELEMETRY_HEALTH,
                1.0,
                0.0,
                (ReasonCode.NAN_OR_INVALID,),
                {"source": "unknown", "error": "motion guard suite failed"},
                action=GuardAction.REQUEST_HOLD,
            )
            unknown_mobility = _report(
                GuardName.MOBILITY,
                0.0,
                0.0,
                (ReasonCode.LOW_CONFIDENCE,),
                {"source": "unknown"},
                action=GuardAction.LIMIT_SPEED,
            )
            unknown_dynamics = _report(
                GuardName.DYNAMICS,
                0.0,
                0.0,
                (ReasonCode.LOW_CONFIDENCE,),
                {"source": "unknown"},
                action=GuardAction.LIMIT_SPEED,
            )
            return unknown_mobility, unknown_dynamics, failed

    def _reports(self, features: Features, telemetry: RobotTelemetry) -> tuple[GuardReport, ...]:
        quality_score, quality_confidence, quality_reasons = data_quality_gate(features)
        source = telemetry.source.value
        shared_provenance = {
            "source": source,
            "sample_count": str(features.sample_count),
            "input_age_seconds": f"{features.age_seconds:.6f}",
            "imu_roll_deg": f"{math.degrees(features.roll):.2f}",
            "imu_pitch_deg": f"{math.degrees(features.pitch):.2f}",
            "imu_angular_rate_rad_s": f"{features.angular_velocity_magnitude:.3f}",
            "joint_velocity_residual": f"{features.joint_velocity_residual:.3f}",
            "left_right_asymmetry": f"{features.asymmetry_score:.3f}",
        }

        mobility_score = slip_component(features)
        mobility_reasons: list[ReasonCode] = []
        if mobility_score > SLIP_HIGH_THRESHOLD:
            mobility_reasons.append(ReasonCode.SLIP_RISK_HIGH)
        elif mobility_score > SLIP_ELEVATED_THRESHOLD:
            mobility_reasons.append(ReasonCode.SLIP_RISK_ELEVATED)
        mobility = _report(
            GuardName.MOBILITY,
            mobility_score,
            quality_confidence,
            mobility_reasons,
            {**shared_provenance, "slip_proxy": f"{mobility_score:.6f}"},
        )

        orientation = orientation_component(features)
        body = body_component(features)
        asymmetry = asymmetry_component(features)
        dynamics_score = max(orientation, body, asymmetry)
        dynamics_reasons: list[ReasonCode] = []
        if orientation > ORIENT_ELEVATED_THRESHOLD:
            dynamics_reasons.append(ReasonCode.ORIENTATION_INSTABILITY)
        if body > BODY_ANOMALY_THRESHOLD:
            dynamics_reasons.append(ReasonCode.BODY_ANOMALY)
        if asymmetry > ASYMMETRY_THRESHOLD:
            dynamics_reasons.append(ReasonCode.ASYMMETRY_DETECTED)
        dynamics = _report(
            GuardName.DYNAMICS,
            dynamics_score,
            quality_confidence,
            dynamics_reasons,
            {
                **shared_provenance,
                "orientation_component": f"{orientation:.6f}",
                "body_component": f"{body:.6f}",
                "asymmetry_component": f"{asymmetry:.6f}",
            },
        )

        telemetry_action = _action_for(quality_score, quality_confidence)
        if features.missing_optional_fields and telemetry_action == GuardAction.PASS:
            telemetry_action = GuardAction.LIMIT_SPEED
        telemetry_health = _report(
            GuardName.TELEMETRY_HEALTH,
            quality_score,
            quality_confidence,
            quality_reasons,
            {
                **shared_provenance,
                "missing_fields": ",".join(features.missing_optional_fields),
            },
            action=telemetry_action,
        )
        return mobility, dynamics, telemetry_health


def fuse_guard_reports(reports: Iterable[GuardReport], *, age_seconds: float) -> RiskEstimate:
    """Fuse reports without allowing a severe report to be averaged away."""
    items = tuple(reports)
    if not items:
        return RiskEstimate(
            score=1.0,
            confidence=0.0,
            reason_codes=(ReasonCode.NAN_OR_INVALID,),
            age_seconds=max(0.0, age_seconds),
        )

    worst_report = max(items, key=lambda report: _ACTION_RANK[report.recommended_action])
    worst_action = worst_report.recommended_action
    score = max(_clip_score(report.score) for report in items)
    if worst_action == GuardAction.REQUEST_HOLD:
        score = max(score, SCORE_REQUEST_HOLD_THRESHOLD)
    elif worst_action == GuardAction.LIMIT_SPEED:
        score = max(score, SCORE_LIMIT_SPEED_THRESHOLD)

    confidence = min(_clip_confidence(report.confidence) for report in items)
    reasons = tuple(
        dict.fromkeys(reason for report in items for reason in report.reason_codes)
    ) or (ReasonCode.NOMINAL,)
    safe_age = age_seconds if math.isfinite(age_seconds) and age_seconds >= 0.0 else 1.0e9
    return RiskEstimate(
        score=_clip_score(score),
        confidence=confidence,
        reason_codes=reasons,
        age_seconds=safe_age,
    )


class FiveGuardSupervisor:
    """Stateful end-to-end five-guard runtime for one continuous session."""

    def __init__(self) -> None:
        self.motion = MotionGuardSuite()
        self.battery = BatteryMarginGuard()
        self.geography = GeographicRiskGuard()
        self.policy = PolicyStateMachine()

    def decide(
        self,
        telemetry: RobotTelemetry,
        mission_context: MissionContext | None,
        now: float,
    ) -> GuardDecision:
        motion_reports = self.motion.observe(telemetry, now)
        battery_report = self.battery.observe(telemetry, now)
        geographic_report = self.geography.evaluate(mission_context, now)
        reports = (*motion_reports, battery_report, geographic_report)
        try:
            age_seconds = float(now) - float(telemetry.monotonic_time)
        except Exception:
            age_seconds = 1.0e9
        risk = fuse_guard_reports(reports, age_seconds=age_seconds)
        return self.policy.decide(risk, now, guard_reports=reports)
