"""Deterministic policy state machine: RiskEstimate -> GuardDecision.

`PolicyStateMachine` is stateful (one instance per continuous control-loop
session) so it can apply hysteresis across successive `decide()` calls: it
must not flap between actions on single-sample noise (AGENTS.md: "never
flap on noise"). Escalation (getting more cautious) requires fewer
consecutive confirmations than recovery (getting less cautious) requires,
on purpose -- a false-safe costs a moment of unneeded caution, a false-clear
could remove a needed safeguard. Imports only from `sherpaos.contracts` and
`sherpaos.estimator.risk` (for the `RiskEstimate` type) -- see the leakage
rule in docs/CONTRACTS.md.
"""

from __future__ import annotations

import math
import uuid

from sherpaos.contracts import GuardAction, GuardDecision, GuardReport, ReasonCode
from sherpaos.estimator.risk import RiskEstimate

RULES_VERSION = "policy-v1.0.0"

_ACTION_RANK: dict[GuardAction, int] = {
    GuardAction.PASS: 0,
    GuardAction.LIMIT_SPEED: 1,
    GuardAction.REQUEST_HOLD: 2,
}

# --------------------------------------------------------------------------
# Raw (pre-hysteresis) action thresholds on (score, confidence).
#
# Simple fixed thresholds per docs/idea.txt section 24 ("if simple
# threshold wins, use the threshold"). Not fit to any dataset -- flagged in
# the handoff report as most likely to need retuning once evaluation data
# exists.
SCORE_LIMIT_SPEED_THRESHOLD = 0.35
SCORE_REQUEST_HOLD_THRESHOLD = 0.7
CONFIDENCE_LOW_THRESHOLD = 0.4  # below this we don't trust the estimate enough to PASS

# --------------------------------------------------------------------------
# Hysteresis tuning. Numbers assume the same ~50-200 Hz loop rate assumed in
# estimator/features.py's FeatureWindow docstring.
# ~15-60ms of sustained above-threshold signal before we act on it
CONSECUTIVE_TO_ESCALATE = 3
# ~30-120ms of sustained below-threshold signal -- stricter than escalate, on purpose
CONSECUTIVE_TO_RECOVER = 6
# minimum dwell time in an escalated action before recovery is even considered
COOLDOWN_SECONDS = 1.0

# Reason codes whose presence means "this sample's data cannot be trusted".
# `_NEVER_PASS_REASONS` is enforced as a hard, un-debounced floor: none of
# these may ever resolve to PASS, regardless of hysteresis state (that
# invariant is more important than avoiding a one-tick blip in the reported
# action). `_HOLD_ESCALATION_REASONS` is the narrower subset that also
# ratchets LIMIT_SPEED -> REQUEST_HOLD if it persists across
# `BAD_DATA_HOLD_STREAK` consecutive samples.
#
# MISSING_FIELD is deliberately in the first set but not the second: a G1
# unit that simply has no effort sensor wired up produces MISSING_FIELD on
# *every* sample forever, and that should cap it at LIMIT_SPEED, not
# eventually ratchet to REQUEST_HOLD purely from persistence. STALE /
# NAN_OR_INVALID / OUT_OF_ORDER describe the *current* sample being
# corrupt/late/reordered -- if that keeps happening, escalating all the way
# to REQUEST_HOLD is the appropriate conservative response.
_NEVER_PASS_REASONS = frozenset(
    {
        ReasonCode.STALE_TELEMETRY,
        ReasonCode.NAN_OR_INVALID,
        ReasonCode.OUT_OF_ORDER,
        ReasonCode.MISSING_FIELD,
    }
)
_HOLD_ESCALATION_REASONS = frozenset(
    {
        ReasonCode.STALE_TELEMETRY,
        ReasonCode.NAN_OR_INVALID,
        ReasonCode.OUT_OF_ORDER,
    }
)
BAD_DATA_HOLD_STREAK = 5
# Note: once the streak reaches this many, the *raw* action is forced to
# REQUEST_HOLD, but that raw value still has to clear the normal
# CONSECUTIVE_TO_ESCALATE confirmation window (like any other escalation)
# before it becomes the reported action -- so persisting bad data actually
# reaches REQUEST_HOLD after BAD_DATA_HOLD_STREAK + CONSECUTIVE_TO_ESCALATE
# - 1 samples, not exactly BAD_DATA_HOLD_STREAK.

# requested_speed_limit convention (documented per task spec: pick one and
# document it):
#   PASS          -> None  (no restriction is being requested/applicable)
#   LIMIT_SPEED   -> a fraction in [LIMIT_SPEED_FRACTION_FLOOR, LIMIT_SPEED_FRACTION_CEILING]
#   REQUEST_HOLD  -> 0.0   (an explicit, actionable "go to zero commanded speed",
#                           distinct from PASS's "not applicable" None)
LIMIT_SPEED_FRACTION_CEILING = 0.8
LIMIT_SPEED_FRACTION_FLOOR = 0.2


def _clip_score(x: float) -> float:
    """Fail-conservative clip for a risk score: non-finite -> 1.0 (max
    risk), never silently 0.0."""
    if not math.isfinite(x):
        return 1.0
    return min(1.0, max(0.0, x))


def _clip_confidence(x: float) -> float:
    """Fail-conservative clip for confidence: non-finite -> 0.0 (min
    trust), never silently 1.0 -- the opposite direction from
    `_clip_score`, since here "conservative" means "don't trust it"."""
    if not math.isfinite(x):
        return 0.0
    return min(1.0, max(0.0, x))


class PolicyStateMachine:
    """Stateful mapping from successive `RiskEstimate`s to `GuardDecision`s.

    Hysteresis state (current action, pending-change streak, bad-data
    streak, last-change time) lives on the instance and persists across
    calls to `decide()` -- construct one per continuous session, not one
    per sample.
    """

    def __init__(self) -> None:
        self._current_action: GuardAction = GuardAction.PASS
        self._pending_action: GuardAction | None = None
        self._pending_count: int = 0
        self._bad_data_streak: int = 0
        self._last_change_time: float | None = None

    def decide(
        self,
        risk: RiskEstimate,
        now: float,
        guard_reports: tuple[GuardReport, ...] = (),
    ) -> GuardDecision:
        try:
            action, extra_reasons = self._decide_action(risk, now)
        except Exception:
            # A bug in the policy layer must never crash the control loop,
            # and must never silently keep PASS -- fail all the way to
            # REQUEST_HOLD (AGENTS.md safety constraint 3).
            action = GuardAction.REQUEST_HOLD
            extra_reasons = (ReasonCode.NAN_OR_INVALID,)
            self._current_action = action
            self._last_change_time = now
            self._pending_action = None
            self._pending_count = 0

        reason_codes = tuple(dict.fromkeys((*risk.reason_codes, *extra_reasons)))

        if action == GuardAction.REQUEST_HOLD:
            requested_speed_limit: float | None = 0.0
        elif action == GuardAction.LIMIT_SPEED:
            requested_speed_limit = self._speed_limit_fraction(risk.score)
        else:
            requested_speed_limit = None

        try:
            input_age_seconds = max(0.0, float(risk.age_seconds))
            if not math.isfinite(input_age_seconds):
                input_age_seconds = 1.0e9
        except Exception:
            input_age_seconds = 1.0e9

        return GuardDecision(
            decision_id=uuid.uuid4().hex,
            action=action,
            score=_clip_score(risk.score),
            confidence=_clip_confidence(risk.confidence),
            reason_codes=reason_codes,
            input_age_seconds=input_age_seconds,
            requested_speed_limit=requested_speed_limit,
            timestamp=now,
            rules_version=RULES_VERSION,
            guard_reports=guard_reports,
        )

    def _decide_action(
        self, risk: RiskEstimate, now: float
    ) -> tuple[GuardAction, tuple[ReasonCode, ...]]:
        score = _clip_score(risk.score)
        confidence = _clip_confidence(risk.confidence)
        reasons = set(risk.reason_codes)

        raw = self._raw_action(score, confidence)

        if reasons & _HOLD_ESCALATION_REASONS:
            self._bad_data_streak += 1
        else:
            self._bad_data_streak = 0
        if self._bad_data_streak >= BAD_DATA_HOLD_STREAK:
            raw = GuardAction.REQUEST_HOLD

        new_action, extra = self._apply_hysteresis(raw, now)

        # Hard safety floor, applied *after* hysteresis so it cannot be
        # debounced away: untrustworthy input may never resolve to PASS.
        below_limit_speed = _ACTION_RANK[new_action] < _ACTION_RANK[GuardAction.LIMIT_SPEED]
        if reasons & _NEVER_PASS_REASONS and below_limit_speed:
            new_action = GuardAction.LIMIT_SPEED
            if new_action != self._current_action:
                self._current_action = new_action
                self._pending_action = None
                self._pending_count = 0
                self._last_change_time = now

        return new_action, extra

    def _raw_action(self, score: float, confidence: float) -> GuardAction:
        if score >= SCORE_REQUEST_HOLD_THRESHOLD:
            return GuardAction.REQUEST_HOLD
        if score >= SCORE_LIMIT_SPEED_THRESHOLD or confidence < CONFIDENCE_LOW_THRESHOLD:
            return GuardAction.LIMIT_SPEED
        return GuardAction.PASS

    def _apply_hysteresis(
        self, raw: GuardAction, now: float
    ) -> tuple[GuardAction, tuple[ReasonCode, ...]]:
        current = self._current_action
        if raw == current:
            self._pending_action = None
            self._pending_count = 0
            return current, ()

        wants_escalate = _ACTION_RANK[raw] > _ACTION_RANK[current]

        if wants_escalate:
            if self._pending_action == raw:
                self._pending_count += 1
            else:
                self._pending_action = raw
                self._pending_count = 1

            if self._pending_count >= CONSECUTIVE_TO_ESCALATE:
                self._current_action = raw
                self._pending_action = None
                self._pending_count = 0
                self._last_change_time = now
                return raw, ()
            return current, (ReasonCode.HYSTERESIS_HOLD,)

        # raw wants to de-escalate / recover.
        in_cooldown = (
            self._last_change_time is not None
            and (now - self._last_change_time) < COOLDOWN_SECONDS
        )
        if in_cooldown:
            self._pending_action = None
            self._pending_count = 0
            return current, (ReasonCode.HYSTERESIS_HOLD,)

        if self._pending_action == raw:
            self._pending_count += 1
        else:
            self._pending_action = raw
            self._pending_count = 1

        if self._pending_count >= CONSECUTIVE_TO_RECOVER:
            self._current_action = raw
            self._pending_action = None
            self._pending_count = 0
            self._last_change_time = now
            return raw, (ReasonCode.RECOVERY_CONFIRMED,)

        return current, (ReasonCode.HYSTERESIS_HOLD,)

    def _speed_limit_fraction(self, score: float) -> float:
        span = SCORE_REQUEST_HOLD_THRESHOLD - SCORE_LIMIT_SPEED_THRESHOLD
        if span <= 0:
            return LIMIT_SPEED_FRACTION_FLOOR
        t = (_clip_score(score) - SCORE_LIMIT_SPEED_THRESHOLD) / span
        t = min(1.0, max(0.0, t))
        fraction = LIMIT_SPEED_FRACTION_CEILING - t * (
            LIMIT_SPEED_FRACTION_CEILING - LIMIT_SPEED_FRACTION_FLOOR
        )
        return float(min(LIMIT_SPEED_FRACTION_CEILING, max(LIMIT_SPEED_FRACTION_FLOOR, fraction)))
