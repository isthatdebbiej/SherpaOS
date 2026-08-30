"""Unit tests for sherpaos.emotion (mapping.py + gestures.py).

Constructs GuardDecision directly (no filesystem/mujoco I/O) so the
classification logic is exercised deterministically and in isolation. No
import of sherpaos.sim/mujoco -- this package is presentation-only and its
tests reflect that boundary.
"""

from __future__ import annotations

import math

from sherpaos.contracts import GuardAction, ReasonCode
from sherpaos.emotion.gestures import (
    MIN_GESTURE_INTENSITY,
    VERIFIED_STABLE_SKILLS,
    gesture_for_emotion,
    should_gesture,
)
from sherpaos.emotion.mapping import EmotionLabel, EmotionState, classify_emotion

NOW = 1_000.0


def make_decision(**overrides: object):
    from sherpaos.contracts import GuardDecision

    defaults: dict[str, object] = {
        "decision_id": "test-decision",
        "action": GuardAction.PASS,
        "score": 0.1,
        "confidence": 0.9,
        "reason_codes": (ReasonCode.NOMINAL,),
        "input_age_seconds": 0.01,
        "requested_speed_limit": None,
        "timestamp": NOW,
        "rules_version": "policy-v1.0.0",
        "guard_reports": (),
    }
    defaults.update(overrides)
    return GuardDecision(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# classify_emotion: action -> label


def test_request_hold_maps_to_fear():
    decision = make_decision(action=GuardAction.REQUEST_HOLD, score=0.85, confidence=0.9)
    emotion = classify_emotion(decision)
    assert emotion.label == EmotionLabel.FEAR
    assert 0.0 <= emotion.intensity <= 1.0


def test_limit_speed_maps_to_worry():
    decision = make_decision(action=GuardAction.LIMIT_SPEED, score=0.5, confidence=0.9)
    emotion = classify_emotion(decision)
    assert emotion.label == EmotionLabel.WORRY
    assert 0.0 <= emotion.intensity <= 1.0


def test_pass_low_confidence_maps_to_uncertain_not_calm():
    decision = make_decision(action=GuardAction.PASS, score=0.05, confidence=0.1)
    emotion = classify_emotion(decision)
    assert emotion.label == EmotionLabel.UNCERTAIN


def test_pass_high_confidence_maps_to_calm():
    decision = make_decision(action=GuardAction.PASS, score=0.05, confidence=0.95)
    emotion = classify_emotion(decision)
    assert emotion.label == EmotionLabel.CALM


def test_pass_with_milestone_maps_to_joy():
    decision = make_decision(action=GuardAction.PASS, score=0.05, confidence=0.95)
    emotion = classify_emotion(decision, milestone_reached=True)
    assert emotion.label == EmotionLabel.JOY


def test_pass_after_recent_non_pass_maps_to_relief_before_joy():
    decision = make_decision(action=GuardAction.PASS, score=0.05, confidence=0.95)
    emotion = classify_emotion(
        decision,
        milestone_reached=True,
        recent_actions=(GuardAction.LIMIT_SPEED,),
    )
    # RELIEF takes precedence over JOY when both conditions hold, since
    # "just recovered from a guarded state" is more specific.
    assert emotion.label == EmotionLabel.RELIEF


def test_pass_after_recent_pass_does_not_trigger_relief():
    decision = make_decision(action=GuardAction.PASS, score=0.05, confidence=0.95)
    emotion = classify_emotion(decision, recent_actions=(GuardAction.PASS, GuardAction.PASS))
    assert emotion.label == EmotionLabel.CALM


# ---------------------------------------------------------------------
# Bounds and monotonicity.


def test_intensity_always_bounded_and_finite():
    for action in (GuardAction.PASS, GuardAction.LIMIT_SPEED, GuardAction.REQUEST_HOLD):
        for score in (0.0, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0):
            for confidence in (0.0, 0.3, 0.5, 0.9, 1.0):
                decision = make_decision(action=action, score=score, confidence=confidence)
                emotion = classify_emotion(decision)
                assert 0.0 <= emotion.intensity <= 1.0
                assert math.isfinite(emotion.intensity)


def test_higher_score_within_request_hold_band_is_more_intense():
    calm_fear = classify_emotion(make_decision(action=GuardAction.REQUEST_HOLD, score=0.72))
    intense_fear = classify_emotion(make_decision(action=GuardAction.REQUEST_HOLD, score=1.0))
    assert intense_fear.intensity > calm_fear.intensity


def test_non_finite_score_is_handled_conservatively():
    decision = make_decision(action=GuardAction.PASS, score=float("nan"), confidence=0.9)
    emotion = classify_emotion(decision)
    # _clip01 maps non-finite score to 1.0 -> action stays PASS but the
    # "calm" score component (1 - score) collapses to 0, not a crash.
    assert 0.0 <= emotion.intensity <= 1.0
    assert math.isfinite(emotion.intensity)


def test_classify_emotion_never_raises_on_malformed_decision():
    class BadDecision:
        decision_id = "bad"
        timestamp = NOW
        action = "not-a-guard-action"
        score = 0.5
        confidence = 0.5
        reason_codes = ()
        rules_version = "x"

    emotion = classify_emotion(BadDecision())  # type: ignore[arg-type]
    assert emotion.label == EmotionLabel.UNCERTAIN
    assert emotion.intensity == 1.0
    assert ReasonCode.NAN_OR_INVALID in emotion.reason_codes


# ---------------------------------------------------------------------
# reason_codes / provenance passthrough.


def test_reason_codes_pass_through_from_decision():
    decision = make_decision(
        action=GuardAction.LIMIT_SPEED,
        reason_codes=(ReasonCode.SLIP_RISK_ELEVATED, ReasonCode.NOMINAL),
    )
    emotion = classify_emotion(decision)
    assert emotion.reason_codes == (ReasonCode.SLIP_RISK_ELEVATED, ReasonCode.NOMINAL)


def test_provenance_records_source_action_and_rules_version():
    decision = make_decision(action=GuardAction.PASS, confidence=0.95)
    emotion = classify_emotion(decision)
    assert emotion.provenance["source_action"] == "PASS"
    assert emotion.provenance["rules_version"] == "policy-v1.0.0"


# ---------------------------------------------------------------------
# gestures.py


def _emotion(label: str, intensity: float) -> EmotionState:
    return EmotionState(
        label=label,
        intensity=intensity,
        reason_codes=(ReasonCode.NOMINAL,),
        decision_id="test",
        timestamp=NOW,
    )


def test_joy_above_threshold_proposes_dance():
    emotion = _emotion(EmotionLabel.JOY, 0.9)
    assert gesture_for_emotion(emotion) == "dance"


def test_relief_above_threshold_proposes_kick():
    emotion = _emotion(EmotionLabel.RELIEF, 0.9)
    assert gesture_for_emotion(emotion) == "kick"


def test_joy_below_min_intensity_proposes_nothing():
    emotion = _emotion(EmotionLabel.JOY, MIN_GESTURE_INTENSITY - 0.01)
    assert gesture_for_emotion(emotion) is None


def test_worry_fear_calm_uncertain_never_propose_a_gesture():
    for label in (EmotionLabel.WORRY, EmotionLabel.FEAR, EmotionLabel.CALM, EmotionLabel.UNCERTAIN):
        emotion = _emotion(label, 1.0)
        assert gesture_for_emotion(emotion) is None


def test_gesture_for_emotion_only_ever_proposes_verified_stable_skills():
    for label in (EmotionLabel.JOY, EmotionLabel.RELIEF):
        emotion = _emotion(label, 1.0)
        skill = gesture_for_emotion(emotion)
        assert skill is None or skill in VERIFIED_STABLE_SKILLS


def test_should_gesture_refuses_when_mobility_not_ok():
    emotion = _emotion(EmotionLabel.JOY, 0.9)
    assert should_gesture(emotion, mobility_ok=False) is None
    assert should_gesture(emotion, mobility_ok=True) == "dance"


def test_gesture_for_emotion_never_raises_on_unknown_label():
    emotion = _emotion("SOME_FUTURE_LABEL", 1.0)
    assert gesture_for_emotion(emotion) is None
