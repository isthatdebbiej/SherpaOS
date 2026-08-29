"""Unit tests for sherpaos.policy.state_machine.

`RiskEstimate`s are constructed directly here (not via the estimator) so the
hysteresis/threshold logic in `PolicyStateMachine` can be exercised in
isolation, deterministically. No import of `sherpaos.sim` or `mujoco`.
"""

from __future__ import annotations

import re

import pytest

from sherpaos.contracts import GuardAction, GuardDecision, ReasonCode
from sherpaos.estimator.risk import RiskEstimate
from sherpaos.policy.state_machine import (
    BAD_DATA_HOLD_STREAK,
    CONSECUTIVE_TO_ESCALATE,
    CONSECUTIVE_TO_RECOVER,
    COOLDOWN_SECONDS,
    RULES_VERSION,
    PolicyStateMachine,
)

_HEX32 = re.compile(r"^[0-9a-f]{32}$")


def nominal(age: float = 0.01) -> RiskEstimate:
    return RiskEstimate(
        score=0.05, confidence=0.95, reason_codes=(ReasonCode.NOMINAL,), age_seconds=age
    )


def elevated(score: float = 0.5, confidence: float = 0.8, age: float = 0.01) -> RiskEstimate:
    return RiskEstimate(
        score=score,
        confidence=confidence,
        reason_codes=(ReasonCode.ORIENTATION_INSTABILITY,),
        age_seconds=age,
    )


def severe(score: float = 0.9, confidence: float = 0.9, age: float = 0.01) -> RiskEstimate:
    return RiskEstimate(
        score=score,
        confidence=confidence,
        reason_codes=(ReasonCode.SLIP_RISK_HIGH,),
        age_seconds=age,
    )


def stale(score: float = 0.5, confidence: float = 0.25, age: float = 5.0) -> RiskEstimate:
    return RiskEstimate(
        score=score,
        confidence=confidence,
        reason_codes=(ReasonCode.STALE_TELEMETRY,),
        age_seconds=age,
    )


def missing_field(score: float = 0.25, confidence: float = 0.65, age: float = 0.01) -> RiskEstimate:
    return RiskEstimate(
        score=score,
        confidence=confidence,
        reason_codes=(ReasonCode.MISSING_FIELD,),
        age_seconds=age,
    )


# ---------------------------------------------------------------------------
# Basic decision hygiene
# ---------------------------------------------------------------------------


def test_every_decision_has_nonempty_reason_codes_and_valid_decision_id():
    machine = PolicyStateMachine()
    seen_ids = set()
    inputs = [nominal(), elevated(), severe(), stale(), missing_field(), nominal()]
    for risk in inputs:
        decision: GuardDecision = machine.decide(risk, now=0.0)
        assert len(decision.reason_codes) >= 1
        assert _HEX32.match(decision.decision_id), decision.decision_id
        assert decision.decision_id not in seen_ids
        seen_ids.add(decision.decision_id)
        assert decision.rules_version == RULES_VERSION


def test_score_and_confidence_clipped_even_if_upstream_is_out_of_range():
    machine = PolicyStateMachine()
    bad = RiskEstimate(
        score=1.7, confidence=-0.3, reason_codes=(ReasonCode.NOMINAL,), age_seconds=0.0
    )
    decision = machine.decide(bad, now=0.0)
    assert 0.0 <= decision.score <= 1.0
    assert 0.0 <= decision.confidence <= 1.0


# ---------------------------------------------------------------------------
# Fail-conservative on untrustworthy telemetry
# ---------------------------------------------------------------------------


def test_stale_telemetry_never_yields_pass_even_on_first_sample():
    machine = PolicyStateMachine()  # fresh instance, starts at PASS
    decision = machine.decide(stale(), now=0.0)
    assert decision.action != GuardAction.PASS
    assert decision.action == GuardAction.LIMIT_SPEED
    assert ReasonCode.STALE_TELEMETRY in decision.reason_codes


def test_nan_or_invalid_never_yields_pass():
    machine = PolicyStateMachine()
    risk = RiskEstimate(
        score=0.55, confidence=0.15, reason_codes=(ReasonCode.NAN_OR_INVALID,), age_seconds=0.0
    )
    decision = machine.decide(risk, now=0.0)
    assert decision.action != GuardAction.PASS


def test_persistent_bad_data_escalates_limit_speed_to_request_hold():
    machine = PolicyStateMachine()
    last_action = None
    for i in range(BAD_DATA_HOLD_STREAK + 2):
        decision = machine.decide(stale(), now=i * 0.02)
        last_action = decision.action
        assert last_action != GuardAction.PASS
    assert last_action == GuardAction.REQUEST_HOLD


def test_missing_field_caps_at_limit_speed_and_never_escalates_to_hold_alone():
    # A permanently-missing optional sensor should hold at LIMIT_SPEED
    # forever, not ratchet to REQUEST_HOLD purely from persistence -- see
    # the _NEVER_PASS_REASONS / _HOLD_ESCALATION_REASONS split in
    # policy/state_machine.py.
    machine = PolicyStateMachine()
    last_action = None
    for i in range(BAD_DATA_HOLD_STREAK + 20):
        decision = machine.decide(missing_field(), now=i * 0.02)
        last_action = decision.action
        assert last_action != GuardAction.PASS
    assert last_action == GuardAction.LIMIT_SPEED


def test_decide_never_raises_on_pathological_risk_estimate():
    # A RiskEstimate with NaN score/confidence and no reason codes is
    # something RiskEstimator itself should never produce (it always fills
    # in NAN_OR_INVALID/NOMINAL/etc), but policy must still degrade
    # gracefully -- never raise -- if it ever sees one. Since there's no
    # bad-data reason code here, the normal escalation debounce still
    # applies (that's the point of hysteresis: a lone severe-but-unflagged
    # reading doesn't itself bypass debounce) -- what must hold is that
    # decide() doesn't crash and every output field stays well-formed.
    machine = PolicyStateMachine()
    weird = RiskEstimate(
        score=float("nan"), confidence=float("nan"), reason_codes=(), age_seconds=float("nan")
    )
    decision = machine.decide(weird, now=0.0)  # must not raise
    assert decision.action in (GuardAction.PASS, GuardAction.LIMIT_SPEED, GuardAction.REQUEST_HOLD)
    assert 0.0 <= decision.score <= 1.0
    assert 0.0 <= decision.confidence <= 1.0
    assert len(decision.reason_codes) >= 1
    assert decision.requested_speed_limit is None or 0.0 <= decision.requested_speed_limit <= 1.0

    # But it must not get *stuck* undetected: three such readings in a row
    # (score clips to 1.0 = max risk each time) do escalate past PASS via
    # the normal hysteresis path, exactly like any other sustained severe
    # reading would.
    machine.decide(weird, now=0.02)
    final = machine.decide(weird, now=0.04)
    assert final.action == GuardAction.REQUEST_HOLD


# ---------------------------------------------------------------------------
# Hysteresis: no flapping on a single noisy/borderline sample
# ---------------------------------------------------------------------------


def test_single_borderline_sample_sandwiched_in_nominal_does_not_flip():
    machine = PolicyStateMachine()
    t = 0.0
    for _ in range(5):
        decision = machine.decide(nominal(), now=t)
        assert decision.action == GuardAction.PASS
        t += 0.02

    # one borderline/noisy elevated reading, not a data-quality problem
    decision = machine.decide(elevated(score=0.5), now=t)
    assert decision.action == GuardAction.PASS
    assert ReasonCode.HYSTERESIS_HOLD in decision.reason_codes
    t += 0.02

    for _ in range(5):
        decision = machine.decide(nominal(), now=t)
        assert decision.action == GuardAction.PASS
        t += 0.02


def test_sustained_bad_run_escalates_then_sustained_good_run_recovers():
    machine = PolicyStateMachine()
    t = 0.0

    for _ in range(3):
        decision = machine.decide(nominal(), now=t)
        assert decision.action == GuardAction.PASS
        t += 0.02

    # sustained elevated run -> escalates to LIMIT_SPEED after
    # CONSECUTIVE_TO_ESCALATE consecutive readings, not before.
    last_action = None
    for i in range(CONSECUTIVE_TO_ESCALATE):
        decision = machine.decide(elevated(score=0.5), now=t)
        last_action = decision.action
        t += 0.02
        if i < CONSECUTIVE_TO_ESCALATE - 1:
            assert last_action == GuardAction.PASS
    assert last_action == GuardAction.LIMIT_SPEED

    # jump well past the cooldown window before offering recovery samples
    t += COOLDOWN_SECONDS + 0.5

    last_action = None
    for i in range(CONSECUTIVE_TO_RECOVER):
        decision = machine.decide(nominal(), now=t)
        last_action = decision.action
        t += 0.02
        if i < CONSECUTIVE_TO_RECOVER - 1:
            assert last_action == GuardAction.LIMIT_SPEED
    assert last_action == GuardAction.PASS
    assert ReasonCode.RECOVERY_CONFIRMED in decision.reason_codes


def test_recovery_blocked_during_cooldown_even_with_enough_consecutive_good_reads():
    machine = PolicyStateMachine()
    t = 0.0
    for _i in range(CONSECUTIVE_TO_ESCALATE):
        machine.decide(elevated(score=0.5), now=t)
        t += 0.02
    assert machine._current_action == GuardAction.LIMIT_SPEED  # sanity on test setup

    # Immediately (well within cooldown) offer more than enough nominal
    # reads to satisfy CONSECUTIVE_TO_RECOVER -- cooldown must still block.
    last_action = None
    for _ in range(CONSECUTIVE_TO_RECOVER + 3):
        decision = machine.decide(nominal(), now=t)
        last_action = decision.action
        t += 0.001  # far less than COOLDOWN_SECONDS in total
    assert last_action == GuardAction.LIMIT_SPEED


# ---------------------------------------------------------------------------
# requested_speed_limit semantics
# ---------------------------------------------------------------------------


def test_requested_speed_limit_semantics_per_action():
    machine = PolicyStateMachine()
    pass_decision = machine.decide(nominal(), now=0.0)
    assert pass_decision.action == GuardAction.PASS
    assert pass_decision.requested_speed_limit is None

    t = 0.02
    for _ in range(CONSECUTIVE_TO_ESCALATE):
        limit_decision = machine.decide(elevated(score=0.5), now=t)
        t += 0.02
    assert limit_decision.action == GuardAction.LIMIT_SPEED
    assert limit_decision.requested_speed_limit is not None
    assert 0.0 < limit_decision.requested_speed_limit <= 1.0

    # The bad-data streak forces the *raw* action to REQUEST_HOLD once it
    # reaches BAD_DATA_HOLD_STREAK, but that raw action still has to clear
    # the normal CONSECUTIVE_TO_ESCALATE confirmation window before it
    # becomes the reported action -- see state_machine.py's docstring on
    # BAD_DATA_HOLD_STREAK.
    for _ in range(BAD_DATA_HOLD_STREAK + CONSECUTIVE_TO_ESCALATE - 1):
        hold_decision = machine.decide(stale(), now=t)
        t += 0.02
    assert hold_decision.action == GuardAction.REQUEST_HOLD
    assert hold_decision.requested_speed_limit == 0.0


def test_higher_score_requests_lower_speed_limit_fraction():
    machine = PolicyStateMachine()
    t = 0.0
    for _ in range(CONSECUTIVE_TO_ESCALATE):
        low_risk_decision = machine.decide(elevated(score=0.4), now=t)
        t += 0.02
    assert low_risk_decision.action == GuardAction.LIMIT_SPEED

    # push a worse-but-still-LIMIT_SPEED-band score
    high_risk_decision = machine.decide(elevated(score=0.65), now=t)
    assert high_risk_decision.action == GuardAction.LIMIT_SPEED
    assert high_risk_decision.requested_speed_limit < low_risk_decision.requested_speed_limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
