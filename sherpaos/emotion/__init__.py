"""Emotion-expression lane: SherpaOS guard output -> a robot "mood".

This package is presentation-only, same boundary as `web/`'s Field Journal
(see AGENTS.md / docs/DECISIONS.md "LLM strictly post-mission, no actuation
role"): it *reads* an already-computed `GuardDecision` (and, for gesture
triggering, an already-verified-stable dance skill name) and never feeds
anything back into `sherpaos.policy`, `sherpaos.estimator`, or the
locomotion controller. Deleting this package changes nothing about mission
safety.

- `mapping.py` -- deterministic, fixed-threshold rules mapping a
  `GuardDecision` (+ small amount of journey context) to an `EmotionState`.
  No learned model, per AGENTS.md rule 6.
- `gestures.py` -- the (small, deliberately conservative) subset of
  emotions allowed to trigger a physical gesture on the G1, and the hard
  safety gate around it.
"""
