"""Deterministic GuardDecision -> EmotionState mapping.

`classify_emotion` is a pure function of its inputs (no hidden state, no
randomness, no learned model, per AGENTS.md rule 6 "deterministic rules
preferred over learned models"). It reads only fields already present on
`sherpaos.contracts.GuardDecision` plus small, explicitly-passed journey
context (milestone reached, mission progress fraction) -- it never queries
simulator ground truth, the internet, or an LLM. This keeps it usable
identically in `sherpa demo --offline`, a replayed dump, and (eventually) a
live G1, matching the adapter-boundary rule in docs/CONTRACTS.md.

Design (fixed thresholds, mirroring policy/state_machine.py's own style):

  - `GuardAction.REQUEST_HOLD`            -> FEAR      (score-scaled intensity)
  - `GuardAction.LIMIT_SPEED`             -> WORRY     (score-scaled intensity)
  - `GuardAction.PASS` + low confidence   -> UNCERTAIN (never a confident PASS
                                             is read as calm -- mirrors the
                                             guard layer's own "low confidence
                                             is not nominal" rule)
  - `GuardAction.PASS` + `milestone_reached=True` -> JOY (a named waypoint was
                                             just reached safely)
  - `GuardAction.PASS`, otherwise         -> CALM

  A short recent history of `GuardDecision.action` (`recent_actions`) lets
  the classifier additionally emit RELIEF exactly once when the action
  *recovers* from LIMIT_SPEED/REQUEST_HOLD back to PASS -- an explicit edge
  condition on the hysteresis-confirmed action sequence in
  `sherpaos.policy.state_machine`, not a new competing state machine: this
  module has no memory of its own, the caller supplies the short window.

`intensity` is always in [0, 1] and is a monotonic function of `score`
within each action band, so a WORRY at score=0.36 renders visibly calmer
than a WORRY at score=0.69, and a FEAR at score=1.0 (the fail-conservative
ceiling `RiskEstimate`/`GuardDecision` clip to on invalid input) is at
`intensity=1.0` -- both fully bounded and comparable to how
`policy/state_machine.py`'s own `_speed_limit_fraction` scales severity to
a graded response instead of a single binary "unsafe" flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sherpaos.contracts import GuardAction, GuardDecision, ReasonCode

# --------------------------------------------------------------------------
# Thresholds. Deliberately reuse the policy layer's own action bands rather
# than inventing new score cutoffs -- an emotion should never disagree with
# what the safety policy just decided about the same sample.
CONFIDENCE_LOW_THRESHOLD = 0.4  # same value as policy/state_machine.py

# Score spans used only to scale *intensity within* an already-decided
# action band -- these do not gate which EmotionLabel is chosen, only how
# strongly it is expressed.
_REQUEST_HOLD_FLOOR = 0.7  # SCORE_REQUEST_HOLD_THRESHOLD in state_machine.py
_LIMIT_SPEED_FLOOR = 0.35  # SCORE_LIMIT_SPEED_THRESHOLD in state_machine.py


class EmotionLabel:
    """String constants (not StrEnum) so this module has zero import-time
    coupling to sherpaos.contracts beyond what it already needs -- kept as
    plain strings because the demo UI / evidence bundle only ever needs to
    render or log the value, never branch dispatch on it outside this
    package. See gestures.py for the (separate, deliberately small) mapping
    from these labels to an actual FSM skill trigger."""

    CALM = "CALM"
    JOY = "JOY"
    RELIEF = "RELIEF"
    WORRY = "WORRY"
    FEAR = "FEAR"
    UNCERTAIN = "UNCERTAIN"


_ALL_LABELS = frozenset(
    {
        EmotionLabel.CALM,
        EmotionLabel.JOY,
        EmotionLabel.RELIEF,
        EmotionLabel.WORRY,
        EmotionLabel.FEAR,
        EmotionLabel.UNCERTAIN,
    }
)


@dataclass(slots=True, frozen=True)
class EmotionState:
    """Bounded output of `classify_emotion`.

    Deliberately mirrors the shape of `sherpaos.contracts.GuardReport` /
    `RiskEstimate` (score-like `intensity`, `reason_codes` passthrough,
    `provenance`) so it composes the same way in logs/evidence bundles
    instead of inventing a new shape for one presentation-only lane.
    """

    label: str  # one of EmotionLabel's constants
    intensity: float  # 0..1
    reason_codes: tuple[ReasonCode, ...]
    decision_id: str
    timestamp: float
    provenance: dict[str, str] = field(default_factory=dict)


def _clip01(x: float) -> float:
    if not math.isfinite(x):
        return 1.0
    return min(1.0, max(0.0, x))


def _scale_within_band(score: float, floor: float, ceiling: float = 1.0) -> float:
    span = ceiling - floor
    if span <= 0:
        return 1.0
    return _clip01((score - floor) / span)


def classify_emotion(
    decision: GuardDecision,
    *,
    milestone_reached: bool = False,
    recent_actions: tuple[GuardAction, ...] = (),
) -> EmotionState:
    """Map one `GuardDecision` to one `EmotionState`.

    `recent_actions` should be the last few *reported* (post-hysteresis)
    actions ending just before `decision.action`, oldest first -- e.g. from
    a small deque the caller maintains alongside the `FiveGuardSupervisor`
    it is already driving. Passing `()` is always safe (RELIEF then simply
    never fires; every other label is unaffected).
    """
    try:
        score = _clip01(decision.score)
        confidence = _clip01(decision.confidence)
        action = decision.action

        if action == GuardAction.REQUEST_HOLD:
            label = EmotionLabel.FEAR
            intensity = _scale_within_band(score, _REQUEST_HOLD_FLOOR)
        elif action == GuardAction.LIMIT_SPEED:
            label = EmotionLabel.WORRY
            intensity = _scale_within_band(score, _LIMIT_SPEED_FLOOR, _REQUEST_HOLD_FLOOR)
        elif confidence < CONFIDENCE_LOW_THRESHOLD:
            label = EmotionLabel.UNCERTAIN
            intensity = _clip01(1.0 - confidence)
        elif recent_actions and recent_actions[-1] != GuardAction.PASS:
            label = EmotionLabel.RELIEF
            intensity = _clip01(1.0 - score)
        elif milestone_reached:
            label = EmotionLabel.JOY
            intensity = _clip01(1.0 - score)
        else:
            label = EmotionLabel.CALM
            intensity = _clip01(1.0 - score)

        provenance = {
            "rules_version": decision.rules_version,
            "source_action": decision.action.value,
            "milestone_reached": str(milestone_reached),
        }
        return EmotionState(
            label=label,
            intensity=intensity,
            reason_codes=decision.reason_codes,
            decision_id=decision.decision_id,
            timestamp=decision.timestamp,
            provenance=provenance,
        )
    except Exception:
        # A bug in this presentation-only module must never crash the
        # caller's loop and must never fabricate a calm reading -- fail to
        # the most attention-worthy label instead (mirrors AGENTS.md rule 3
        # applied to a non-safety lane: fail loud, not silently pleasant).
        return EmotionState(
            label=EmotionLabel.UNCERTAIN,
            intensity=1.0,
            reason_codes=(ReasonCode.NAN_OR_INVALID,),
            decision_id=getattr(decision, "decision_id", "unknown"),
            timestamp=getattr(decision, "timestamp", 0.0),
            provenance={"error": "emotion classification failed"},
        )
