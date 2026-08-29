"""Deterministic risk scoring over `Features`.

`RiskEstimator` wraps a `FeatureWindow`, ingests one `RobotTelemetry` sample
at a time, and produces a bounded `RiskEstimate`. Per AGENTS.md rule 6 and
docs/idea.txt section 24 ("if simple threshold wins, use the threshold, do
not force ML"), this is a weighted-sum-and-clip scorer over the features
computed in `estimator/features.py` -- no learned model, no ML, no network
calls. Imports only from `sherpaos.contracts`, `sherpaos.estimator.features`,
stdlib, and numpy (via features.py) -- see the leakage rule in
docs/CONTRACTS.md.

Design: data-quality problems (NaN, staleness, reordering, producer-flagged
invalid, missing risk-relevant fields) set a *floor* under `score` and a
*ceiling* over `confidence` -- they can only push the estimate toward "more
dangerous, less trustworthy", never cancel out a behavioral risk signal.
Behavioral risk (orientation instability, body-residual anomaly, asymmetry,
slip proxy) is combined as a separate weighted sum. The final score is the
max of the data-quality floor and the behavioral weighted sum, so a single
NaN sample can't be "diluted" by an otherwise-calm window.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from sherpaos.contracts import ReasonCode, RobotTelemetry
from sherpaos.estimator.features import Features, FeatureWindow

# --------------------------------------------------------------------------
# Data-quality gating.
#
# Floors/ceilings below are chosen so a *single* occurrence lands solidly in
# the policy layer's LIMIT_SPEED band (see policy/state_machine.py:
# SCORE_LIMIT_SPEED_THRESHOLD=0.35, SCORE_REQUEST_HOLD_THRESHOLD=0.7) rather
# than jumping straight to REQUEST_HOLD -- persistence-based escalation to
# REQUEST_HOLD is the policy layer's job (its bad-data streak counter), not
# this module's. The exception is `missing_optional_fields`, which is
# deliberately mild here (see below) and enforced as a hard action floor at
# the policy layer instead of via score.
NAN_SCORE_FLOOR = 0.55
NAN_CONFIDENCE_CEILING = 0.15
INVALID_SCORE_FLOOR = 0.5
INVALID_CONFIDENCE_CEILING = 0.2
STALE_SCORE_FLOOR = 0.5
STALE_CONFIDENCE_CEILING = 0.25
OUT_OF_ORDER_SCORE_FLOOR = 0.45
OUT_OF_ORDER_CONFIDENCE_CEILING = 0.4

# A permanently-missing optional sensor (e.g. a G1 unit with no effort
# sensing) is not "corrupt data" the way NaN/stale/reordered samples are --
# it is a persistent, known reduction in what we can assess. We reflect that
# as a small score nudge + confidence cap here (not enough to alone cross
# SCORE_LIMIT_SPEED_THRESHOLD), while the policy layer separately guarantees
# "never PASS while a risk-relevant field is missing" as a hard action floor
# -- see policy/state_machine.py's _NEVER_PASS_REASONS. Splitting it this
# way means the *severity* stays visible in `score` without this module
# unilaterally deciding the resulting action.
MISSING_FIELD_SCORE_FLOOR = 0.25
MISSING_FIELD_CONFIDENCE_CEILING = 0.7

# Below this many samples in the window, short-horizon variability features
# (std-devs, residuals) are based on too little history to fully trust, so
# confidence is capped on a ramp up to this many samples. 10 samples is
# ~0.1s at 100Hz / 0.2s at 50Hz -- a brief but bounded warm-up.
MIN_SAMPLES_FOR_FULL_CONFIDENCE = 10
_WARM_UP_CONFIDENCE_FLOOR = 0.5

# --------------------------------------------------------------------------
# Behavioral risk component scaling.
#
# Converts each raw feature magnitude into a 0..1 sub-score before summing.
# Deliberately simple fixed thresholds (docs/idea.txt section 24: "if simple
# threshold wins, use the threshold") -- not fit to any dataset, since no
# real/sim G1 telemetry exists yet for this hackathon pass. Flagged in the
# handoff report as the values most likely to need retuning once evaluation
# data exists.
ORIENT_ANGLE_SCALE = 0.6  # rad (~34 deg) of |roll|+|pitch| treated as "fully unsafe" tilt
ANGULAR_VELOCITY_MAG_SCALE = 3.0  # rad/s
ORIENT_STD_SCALE = 0.2  # rad
ANGULAR_VELOCITY_STD_SCALE = 1.5  # rad/s

JOINT_VELOCITY_RESIDUAL_SCALE = 4.0  # rad/s RMS
JOINT_EFFORT_RESIDUAL_SCALE = 8.0  # N*m RMS -- placeholder, real G1 effort ranges TBD

# asymmetry_score (already a 0..1 relative ratio) considered "fully anomalous" from ~0.6 up
ASYMMETRY_SCALE = 0.6

WEIGHT_ORIENTATION = 0.30
WEIGHT_BODY = 0.25
WEIGHT_ASYMMETRY = 0.20
WEIGHT_SLIP = 0.25

ORIENT_ELEVATED_THRESHOLD = 0.5
BODY_ANOMALY_THRESHOLD = 0.5
ASYMMETRY_THRESHOLD = 0.5
SLIP_ELEVATED_THRESHOLD = 0.35
SLIP_HIGH_THRESHOLD = 0.65

LOW_CONFIDENCE_THRESHOLD = 0.5


def _clip_score(x: float) -> float:
    """Fail-conservative clip for score: a non-finite value becomes 1.0
    (maximal risk), never silently 0.0."""
    if not math.isfinite(x):
        return 1.0
    return min(1.0, max(0.0, x))


def _clip_confidence(x: float) -> float:
    """Fail-conservative clip for confidence: a non-finite value becomes
    0.0 (minimal trust), never silently 1.0."""
    if not math.isfinite(x):
        return 0.0
    return min(1.0, max(0.0, x))


@dataclass(slots=True, frozen=True)
class RiskEstimate:
    """Bounded output of `RiskEstimator.update()`."""

    score: float  # 0..1
    confidence: float  # 0..1
    reason_codes: tuple[ReasonCode, ...]  # always non-empty
    age_seconds: float  # age of the input sample this estimate is based on


def _fallback_estimate(age_seconds: float) -> RiskEstimate:
    """Used only if scoring itself raises unexpectedly -- see
    `RiskEstimator.update`'s try/except. Maximally conservative by
    construction: score=1.0, confidence=0.0."""
    return RiskEstimate(
        score=1.0,
        confidence=0.0,
        reason_codes=(ReasonCode.NAN_OR_INVALID,),
        age_seconds=age_seconds,
    )


class RiskEstimator:
    """Stateful: wraps one `FeatureWindow` and scores each new sample."""

    def __init__(self, window: FeatureWindow | None = None) -> None:
        self._window = window if window is not None else FeatureWindow()

    def update(self, telemetry: RobotTelemetry, now_monotonic: float | None = None) -> RiskEstimate:
        # `now_monotonic` is documented and always passed explicitly by the
        # real control loop and by this lane's own tests, so estimates stay
        # reproducible given a fixed clock trace. It defaults to
        # `time.monotonic()` only so a caller that genuinely has no
        # simulated/replayed clock of its own can still call `update(telemetry)`
        # -- the same clock domain `RobotTelemetry.monotonic_time` is drawn
        # from -- without that being a required ceremony.
        if now_monotonic is None:
            now_monotonic = time.monotonic()

        # Top-level safety net: a bug in feature/scoring math must never
        # crash the control loop (AGENTS.md safety constraint 3). If
        # anything above raises, fail all the way conservative instead.
        try:
            age_seconds = max(0.0, float(now_monotonic) - float(telemetry.monotonic_time))
        except Exception:
            age_seconds = 0.0
        try:
            self._window.push(telemetry)
            features = self._window.compute_features(now_monotonic)
            return _score(features)
        except Exception:
            return _fallback_estimate(age_seconds)


def data_quality_gate(f: Features) -> tuple[float, float, list[ReasonCode]]:
    """The "can only push toward worse" data-quality block, factored out so
    `sherpaos.policy.guards.telemetry_health_report` can build the
    telemetry-health `GuardReport` from the exact same logic `_score` uses,
    instead of duplicating it. Returns (score_floor, confidence, reasons) --
    `confidence` here has already had every applicable ceiling applied
    (including the warm-up ramp), so callers should treat it as final for
    the "how much can we trust *this sample*" question, not just fold their
    own confidence on top of a partial value.
    """
    reasons: list[ReasonCode] = []
    score_floor = 0.0
    confidence = 1.0

    if f.has_nan:
        score_floor = max(score_floor, NAN_SCORE_FLOOR)
        confidence = min(confidence, NAN_CONFIDENCE_CEILING)
        reasons.append(ReasonCode.NAN_OR_INVALID)
    if f.producer_invalid:
        score_floor = max(score_floor, INVALID_SCORE_FLOOR)
        confidence = min(confidence, INVALID_CONFIDENCE_CEILING)
        if ReasonCode.NAN_OR_INVALID not in reasons:
            reasons.append(ReasonCode.NAN_OR_INVALID)
    if f.is_stale:
        score_floor = max(score_floor, STALE_SCORE_FLOOR)
        confidence = min(confidence, STALE_CONFIDENCE_CEILING)
        reasons.append(ReasonCode.STALE_TELEMETRY)
    if f.is_out_of_order:
        score_floor = max(score_floor, OUT_OF_ORDER_SCORE_FLOOR)
        confidence = min(confidence, OUT_OF_ORDER_CONFIDENCE_CEILING)
        reasons.append(ReasonCode.OUT_OF_ORDER)
    if f.missing_optional_fields:
        score_floor = max(score_floor, MISSING_FIELD_SCORE_FLOOR)
        confidence = min(confidence, MISSING_FIELD_CONFIDENCE_CEILING)
        reasons.append(ReasonCode.MISSING_FIELD)

    if f.sample_count < MIN_SAMPLES_FOR_FULL_CONFIDENCE:
        warm_up = _WARM_UP_CONFIDENCE_FLOOR + (1.0 - _WARM_UP_CONFIDENCE_FLOOR) * (
            f.sample_count / MIN_SAMPLES_FOR_FULL_CONFIDENCE
        )
        confidence = min(confidence, warm_up)

    return score_floor, confidence, reasons


def orientation_component(f: Features) -> float:
    """See `sherpaos.policy.guards.dynamics_report` -- shared with `_score`."""
    return _clip_score(
        (abs(f.roll) + abs(f.pitch)) / ORIENT_ANGLE_SCALE * 0.5
        + f.angular_velocity_magnitude / ANGULAR_VELOCITY_MAG_SCALE * 0.25
        + (f.roll_std + f.pitch_std) / ORIENT_STD_SCALE * 0.15
        + f.angular_velocity_std / ANGULAR_VELOCITY_STD_SCALE * 0.10
    )


def body_component(f: Features) -> float:
    """See `sherpaos.policy.guards.dynamics_report` -- shared with `_score`."""
    return _clip_score(
        f.joint_velocity_residual / JOINT_VELOCITY_RESIDUAL_SCALE * 0.7
        + (
            (f.joint_effort_residual / JOINT_EFFORT_RESIDUAL_SCALE * 0.3)
            if f.joint_effort_residual is not None
            else 0.0
        )
    )


def asymmetry_component(f: Features) -> float:
    """See `sherpaos.policy.guards.dynamics_report` -- shared with `_score`."""
    return _clip_score(f.asymmetry_score / ASYMMETRY_SCALE)


def slip_component(f: Features) -> float:
    """See `sherpaos.policy.guards.mobility_report` -- shared with `_score`."""
    return _clip_score(f.slip_proxy_score)


def _score(f: Features) -> RiskEstimate:
    score_floor, confidence, reasons = data_quality_gate(f)
    reasons = list(reasons)

    orientation_component_val = orientation_component(f)
    body_component_val = body_component(f)
    asymmetry_component_val = asymmetry_component(f)
    slip_component_val = slip_component(f)

    weighted = (
        WEIGHT_ORIENTATION * orientation_component_val
        + WEIGHT_BODY * body_component_val
        + WEIGHT_ASYMMETRY * asymmetry_component_val
        + WEIGHT_SLIP * slip_component_val
    )

    score = _clip_score(max(score_floor, weighted))

    if orientation_component_val > ORIENT_ELEVATED_THRESHOLD:
        reasons.append(ReasonCode.ORIENTATION_INSTABILITY)
    if body_component_val > BODY_ANOMALY_THRESHOLD:
        reasons.append(ReasonCode.BODY_ANOMALY)
    if asymmetry_component_val > ASYMMETRY_THRESHOLD:
        reasons.append(ReasonCode.ASYMMETRY_DETECTED)
    if slip_component_val > SLIP_HIGH_THRESHOLD:
        reasons.append(ReasonCode.SLIP_RISK_HIGH)
    elif slip_component_val > SLIP_ELEVATED_THRESHOLD:
        reasons.append(ReasonCode.SLIP_RISK_ELEVATED)

    confidence = _clip_confidence(confidence)
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        reasons.append(ReasonCode.LOW_CONFIDENCE)

    if not reasons:
        reasons.append(ReasonCode.NOMINAL)

    # De-dup while preserving first-seen order (a code could in principle be
    # appended twice above as features evolve; keep this robust to that).
    reasons_tuple = tuple(dict.fromkeys(reasons))

    return RiskEstimate(
        score=score,
        confidence=confidence,
        reason_codes=reasons_tuple,
        age_seconds=f.age_seconds,
    )
