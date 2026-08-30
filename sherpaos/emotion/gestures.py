"""EmotionLabel -> physical G1 gesture, gated to only the skills this repo
has actually verified stable (see docs/G1_DANCE_DEMO.md).

Deliberately narrow: this module is the *only* place emotion output is
allowed to turn into an actuation request, and it only ever proposes one of
the three skills `docs/G1_DANCE_DEMO.md` records as repeatedly ending
standing (`dance`, `kungfu`, `kick` -- NOT `beyondmimic`, which fell
repeatedly in testing, and NOT any KungFu2-style skill with no reachable
FSM trigger). A proposal from here still has to pass the caller's own
mobility gate (see `should_gesture`) before anything runs -- this module
never calls into `scripts/run_g1_dance.py` itself, it only returns a
skill name string for the caller to hand to it.

This is intentionally NOT wired into `sherpaos.policy` or the runtime
control loop: it is a downstream *consumer* of an already-finalized
`GuardDecision`/`EmotionState`, exactly like the Field Journal in `web/`
consumes finalized incident evidence. Nothing in `sherpaos.policy` imports
this module.
"""

from __future__ import annotations

from sherpaos.emotion.mapping import EmotionLabel, EmotionState

# The exact set verified in docs/G1_DANCE_DEMO.md. Keep this list in sync
# with that document -- if a skill's verified status there changes, update
# it here in the same commit.
VERIFIED_STABLE_SKILLS = frozenset({"dance", "kungfu", "kick"})

# One skill per emotion that is allowed to ever trigger a gesture. CALM and
# UNCERTAIN deliberately map to None -- "keep walking normally" and "we are
# not sure what is happening" are exactly the two states where triggering
# an 18s scripted skill (which owns the FSM for its whole duration, see
# scripts/run_g1_dance.py's docstring) would be the wrong call.
_EMOTION_TO_SKILL: dict[str, str | None] = {
    EmotionLabel.JOY: "dance",
    EmotionLabel.RELIEF: "kick",  # short (3.6s), low-commitment celebratory beat
    EmotionLabel.WORRY: None,
    EmotionLabel.FEAR: None,
    EmotionLabel.CALM: None,
    EmotionLabel.UNCERTAIN: None,
}

# Minimum intensity before a gesture is even proposed -- a JOY of intensity
# 0.05 (i.e. score close to the LIMIT_SPEED boundary) is not worth
# interrupting locomotion for.
MIN_GESTURE_INTENSITY = 0.3


def gesture_for_emotion(emotion: EmotionState) -> str | None:
    """Return a verified-stable skill name, or None if no gesture should
    run for this `EmotionState`. Never raises -- an unrecognized label
    (e.g. a future EmotionLabel this module has not been updated for)
    degrades to None, the same conservative default as WORRY/FEAR/CALM."""
    skill = _EMOTION_TO_SKILL.get(emotion.label)
    if skill is None:
        return None
    if skill not in VERIFIED_STABLE_SKILLS:
        # Defensive: a typo/renamed skill above must never silently
        # propose an unverified one.
        return None
    if emotion.intensity < MIN_GESTURE_INTENSITY:
        return None
    return skill


def should_gesture(emotion: EmotionState, *, mobility_ok: bool) -> str | None:
    """Same as `gesture_for_emotion`, plus the hard external gate: the
    caller must tell us the mobility guard is currently PASS-ing
    (`mobility_ok`) before a gesture is proposed at all. A gesture is a
    frivolous, non-essential action -- if the caller cannot vouch that it
    is currently safe to run one, refuse regardless of how strong the
    emotion is.
    """
    if not mobility_ok:
        return None
    return gesture_for_emotion(emotion)
