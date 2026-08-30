"""Battery-margin guard: `BatteryMarginGuard`, one of SherpaOS's five
independent guard families (`docs/CONTRACTS.md` "Guard families" item 4).

Scope boundary: this module is *analysis only*. It consumes
`sherpaos.contracts.RobotTelemetry.battery_*` fields -- state of charge,
voltage, current, temperature -- and never generates or simulates a battery
trace itself (that is `sherpaos/sim/`'s job, a different lane). It imports
nothing but `sherpaos.contracts` and stdlib, so it works identically against
simulated telemetry, a replayed dump, or (eventually) a live G1 -- see the
adapter-boundary rule in `docs/CONTRACTS.md`.

Per docs/plan.md's labeling requirement, every `GuardReport` this guard
emits carries the input samples' `TelemetrySource` in `provenance["source"]`
so a report can be traced back to "based on simulated battery data" and
never quietly presented as a calibrated real-G1 predictor.

Design mirrors `sherpaos/estimator/risk.py`'s conventions for consistency
across guards: fixed, documented thresholds (no learned model, per AGENTS.md
rule 6); fail-conservative on missing/invalid input (AGENTS.md rule 3, never
crash, never silently PASS); a top-level try/except in `observe()` so a bug
in this module's own math degrades to a maximally conservative report
instead of taking down the control loop.

Modeling choices (all self-defined for this hackathon pass -- no real G1
battery-chemistry data exists to calibrate against; see the anchor-point
constants below and the final handoff notes for what is most likely to need
retuning once real/simulated battery traces exist):

  - "Margin" is a *leading* indicator, not merely today's state of charge:
    this guard also estimates a discharge rate from the rolling window and
    projects a remaining-operating-time, so a battery declining fast can
    score worse than one at the same charge declining slowly.
  - Voltage sag is only flagged as a load-related signature
    (`ReasonCode.BATTERY_VOLTAGE_SAG`) when voltage is below what a
    self-defined expected-voltage-vs-SoC reference curve predicts *and*
    reported current indicates a high discharge load at the same time --
    voltage alone, or current alone, is not enough (see docstring on
    `_sag_component`).
  - Cold temperature reduces the *effective* margin (an "effective SoC"
    lower than the raw reported fraction), so a cold, mid-charge battery
    compounds with an already-marginal SoC rather than being scored as an
    independent, additive concern -- see `_temperature_retention` and the
    `effective_fraction` computation in `BatteryMarginGuard._score`.

`battery_current_a` sign convention (this module's own; RobotTelemetry
itself does not fix one -- see contracts.py): positive = discharging,
negative or zero = charging/idle. Only positive current above
`HIGH_DISCHARGE_CURRENT_A` counts as "high load" for the sag check.
"""

from __future__ import annotations

import math
from collections import deque

from sherpaos.contracts import (
    GuardAction,
    GuardName,
    GuardReport,
    ReasonCode,
    RobotTelemetry,
)

# --------------------------------------------------------------------------
# Window sizing.
#
# Unlike joint kinematics (assumed 50-200 Hz elsewhere in this repo, see
# estimator/features.py), battery state changes on the order of minutes, not
# milliseconds, and this contract does not guarantee battery fields update
# at the same cadence as the rest of a RobotTelemetry sample (a real gauge
# might report once a second while joints stream at 100 Hz, or a replay
# might carry battery fields at whatever rate the dump captured them).
# Rather than assume a rate, the discharge-rate estimate below is computed
# from the *elapsed monotonic time* spanned by the window, not the sample
# count -- so this module works the same whether "240 samples" means 2.4s
# (100 Hz full-rate feed) or several minutes (a slow battery-only feed).
# 240 is chosen generously (~2x the mobility estimator's 100-sample default)
# so a slow-cadence feed still keeps several minutes of history; the memory
# footprint of 240 small telemetry records is trivial either way.
DEFAULT_WINDOW_SIZE = 240

# --------------------------------------------------------------------------
# Missing-data handling.
#
# `battery_fraction is None` (or non-finite) on the current sample means
# this guard has nothing to assess. Per AGENTS.md rule 3, that must not
# resolve to a confident PASS. MISSING_DATA_SCORE is chosen to land
# solidly in the LIMIT_SPEED band (see SCORE_LIMIT_SPEED_THRESHOLD /
# SCORE_REQUEST_HOLD_THRESHOLD below) without ever reaching REQUEST_HOLD on
# its own -- escalating a single guard's *absence of data* all the way to a
# hold is the fusion layer's job (weighing this against the other four
# guards), not this one's.
MISSING_DATA_SCORE = 0.40
MISSING_DATA_CONFIDENCE = 0.15

# --------------------------------------------------------------------------
# State-of-charge margin bands.
#
# Piecewise-linear anchors (fraction -> score contribution), escalating as
# charge drops. Smooth/continuous on purpose (vs. hard cliffs) so noise near
# a boundary doesn't flap the score -- flapping the reported *action* is
# handled by hysteresis at the policy layer, but there's no reason to make
# this guard's own score jumpy either. Anchors are round, documented
# placeholders (25%/20%/10% bands are common rules of thumb for legged
# robots' usable margin, not measured from any real G1 pack) -- flagged as
# likely to need retuning once real/simulated battery traces exist.
SOC_SCORE_ANCHORS: list[tuple[float, float]] = [
    (0.00, 1.00),
    (0.10, 0.80),
    (0.20, 0.50),
    (0.35, 0.20),
    (1.00, 0.00),
]
# Fraction at/below which low SoC alone is worth calling out with its own
# reason code (roughly the "low" band above).
SOC_LOW_REASON_THRESHOLD = 0.20

# --------------------------------------------------------------------------
# Expected voltage-vs-SoC reference curve.
#
# We have no ground-truth open-circuit-voltage curve for the G1's actual
# pack, so this is a self-defined, documented, monotonic reference curve
# shaped like a typical multi-cell Li-ion pack: a steep initial drop off
# full charge, a broad flat plateau through the middle of the discharge
# curve, and a steep drop-off again near empty. Anchors assume a nominal
# ~48V pack (13S Li-ion: ~4.2V/cell full, ~2.6V/cell cutoff) purely as a
# plausible round-number stand-in -- not a claim about the real G1 pack.
EXPECTED_VOLTAGE_ANCHORS: list[tuple[float, float]] = [
    (0.00, 33.8),
    (0.05, 38.0),
    (0.10, 41.0),
    (0.20, 44.0),
    (0.50, 48.0),
    (0.90, 52.0),
    (1.00, 54.6),
]
# Deficit (expected - observed) below which we don't consider it worth
# flagging at all -- ordinary reporting noise/quantization.
VOLTAGE_SAG_MIN_DEFICIT_V = 1.5
# Deficit at/above which the sag component saturates to 1.0 ("fully
# sagging"). Both values are round placeholders against the curve above,
# not derived from real pack internal-resistance data.
VOLTAGE_SAG_FULL_DEFICIT_V = 4.0
# Discharge current above which we call the load "high" for the purposes of
# the sag check. No real G1 current-draw spec is available for this pass;
# chosen as a round multiple of a plausible nominal walking current on a
# ~48V pack. Flagged as a placeholder most likely to need retuning.
HIGH_DISCHARGE_CURRENT_A = 15.0
# Sag observed without a confirmed high-load current is ambiguous (could be
# a resting/low-load reading that's just naturally a bit low) -- it still
# nudges the score, but at reduced weight, and does not by itself raise
# `ReasonCode.BATTERY_VOLTAGE_SAG` (that code is reserved for the
# load-confirmed signature described in the module docstring).
UNCONFIRMED_SAG_WEIGHT = 0.3

# --------------------------------------------------------------------------
# Discharge rate / estimated remaining operating time.
#
# Require at least this much *real elapsed time* spanned by same-window
# samples before trusting a slope estimate -- below this, a slope is
# dominated by SoC reporting resolution/quantization rather than real
# discharge behavior. Deliberately low so a modest-cadence synthetic/replay
# sequence can exercise this logic within a small window; flagged as likely
# to need retuning once a real gauge's update rate/resolution is known.
MIN_WINDOW_SPAN_SECONDS_FOR_RATE = 5.0
# A magnitude below this (fraction/second) is treated as "not clearly
# discharging" (flat or noisy-flat), not as an extremely slow discharge.
DISCHARGE_RATE_NOISE_EPS_PER_SEC = 1.0e-6

# Piecewise-linear anchors (estimated remaining minutes -> score
# contribution). This is the "leading indicator" half of the guard: a
# battery at a comfortable SoC but draining fast enough to project under
# ~15 minutes remaining should already read as elevated risk, not wait
# until SoC itself crosses a low-charge band. Round placeholders, not
# calibrated to any real mission-length data.
REMAINING_TIME_SCORE_ANCHORS_MIN: list[tuple[float, float]] = [
    (0.0, 1.00),
    (5.0, 0.85),
    (15.0, 0.45),
    (30.0, 0.15),
    (60.0, 0.00),
]
REMAINING_TIME_LOW_REASON_THRESHOLD_MIN = 15.0

# --------------------------------------------------------------------------
# Cold-temperature derating.
#
# Below COLD_DERATE_START_C, usable capacity is treated as linearly reduced
# from 100% down to RETENTION_AT_SEVERE_COLD at COLD_DERATE_SEVERE_C and
# below (flat beyond that). Real Li-ion packs do lose a substantial chunk
# of usable capacity in deep cold (commonly cited as roughly a third to a
# half around -20C, worse with load), so 50% retention at -20C is a
# plausible round placeholder, not a measured curve for the G1's pack.
COLD_DERATE_START_C = 5.0
COLD_DERATE_SEVERE_C = -20.0
RETENTION_AT_SEVERE_COLD = 0.5
COLD_RETENTION_ANCHORS: list[tuple[float, float]] = [
    (COLD_DERATE_SEVERE_C, RETENTION_AT_SEVERE_COLD),
    (COLD_DERATE_START_C, 1.00),
]

# --------------------------------------------------------------------------
# Action thresholds. Mirrors policy/state_machine.py's
# SCORE_LIMIT_SPEED_THRESHOLD / SCORE_REQUEST_HOLD_THRESHOLD bands so a
# `score` means roughly the same thing across guards even before fusion --
# not imported from there (that module is a different lane and this guard
# must stay self-contained), just intentionally aligned.
SCORE_LIMIT_SPEED_THRESHOLD = 0.35
SCORE_REQUEST_HOLD_THRESHOLD = 0.7


def _clip01(x: float) -> float:
    """Clip to [0, 1]; a non-finite input fails conservative (-> 1.0)."""
    if not math.isfinite(x):
        return 1.0
    return min(1.0, max(0.0, x))


def _clip_confidence(x: float) -> float:
    """Clip to [0, 1]; a non-finite input fails conservative (-> 0.0, i.e.
    minimal trust, never silently maximal)."""
    if not math.isfinite(x):
        return 0.0
    return min(1.0, max(0.0, x))


def _is_finite_number(x: object) -> bool:
    try:
        return math.isfinite(float(x))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _piecewise_linear(x: float, anchors: list[tuple[float, float]]) -> float:
    """Interpolate `x` against ascending-`x` `(x, y)` anchors. Flat
    extrapolation beyond either end. Callers are responsible for passing a
    finite `x` -- every call site below only reaches this after an
    explicit finite-value check, so this stays a simple total function."""
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:], strict=False):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return anchors[-1][1]  # unreachable given the bounds checks above


def _temperature_retention(temperature_c: float) -> float:
    """Fraction of nominal usable capacity retained at this temperature.
    1.0 above COLD_DERATE_START_C, ramping down to RETENTION_AT_SEVERE_COLD
    at/below COLD_DERATE_SEVERE_C."""
    return _piecewise_linear(temperature_c, COLD_RETENTION_ANCHORS)


def _discharge_rate_per_second(points: list[tuple[float, float]]) -> float | None:
    """Least-squares slope of fraction vs. time over `points`
    (monotonic_time, battery_fraction), returned as a *positive* rate
    (fraction lost per second) when the fit shows a clear decline.

    Returns None (never inf/nan, never a fabricated number) when there
    isn't a reliable declining trend to report: too few points, a
    degenerate (zero-variance) time axis, or a slope that is flat/rising
    within noise. Callers must treat None as "cannot estimate", not as
    "rate is zero".
    """
    n = len(points)
    if n < 2:
        return None
    times = [t for t, _ in points]
    fracs = [f for _, f in points]
    mean_t = sum(times) / n
    mean_f = sum(fracs) / n
    var_t = sum((t - mean_t) ** 2 for t in times)
    if var_t < 1.0e-9:
        return None
    cov_tf = sum((t - mean_t) * (f - mean_f) for t, f in zip(times, fracs, strict=True))
    slope = cov_tf / var_t  # d(fraction)/d(time); negative while discharging
    if not math.isfinite(slope):
        return None
    rate = -slope
    if rate <= DISCHARGE_RATE_NOISE_EPS_PER_SEC:
        return None
    return rate


def _sag_component(
    fraction: float,
    voltage: float | None,
    current_a: float | None,
) -> tuple[float, float, bool]:
    """Returns (sag_component, expected_voltage, is_load_confirmed_sag).

    See the module docstring: sag is only reported as the "load" signature
    (`ReasonCode.BATTERY_VOLTAGE_SAG`) when a meaningful voltage deficit vs.
    the expected-voltage-vs-SoC curve coincides with a high discharge
    current. A deficit without a confirmed high load still nudges the
    score (a resting battery that's simply weak is still worth knowing
    about) but at reduced weight, since we can't rule out "just a low-load
    reading, not sag".
    """
    expected_voltage = _piecewise_linear(fraction, EXPECTED_VOLTAGE_ANCHORS)
    if voltage is None or not math.isfinite(voltage):
        return 0.0, expected_voltage, False

    deficit = expected_voltage - voltage
    if deficit < VOLTAGE_SAG_MIN_DEFICIT_V:
        return 0.0, expected_voltage, False

    span = VOLTAGE_SAG_FULL_DEFICIT_V - VOLTAGE_SAG_MIN_DEFICIT_V
    ratio = _clip01((deficit - VOLTAGE_SAG_MIN_DEFICIT_V) / span) if span > 0 else 1.0

    high_load = (
        current_a is not None and math.isfinite(current_a) and current_a >= HIGH_DISCHARGE_CURRENT_A
    )
    if high_load:
        return ratio, expected_voltage, True
    return ratio * UNCONFIRMED_SAG_WEIGHT, expected_voltage, False


def _fmt(x: float | None, digits: int = 3) -> str:
    return "unavailable" if x is None else f"{x:.{digits}f}"


def _unavailable_report(
    telemetry: RobotTelemetry,
    window_len: int,
    all_missing: bool,
) -> GuardReport:
    reasons: list[ReasonCode] = [ReasonCode.BATTERY_DATA_UNAVAILABLE]
    confidence = MISSING_DATA_CONFIDENCE
    if all_missing and window_len > 1:
        # Persistently missing (not just this one sample) -- even less to
        # go on than a single dropped reading.
        confidence = min(confidence, MISSING_DATA_CONFIDENCE * 0.5)
    confidence = _clip_confidence(confidence)
    score = _clip01(MISSING_DATA_SCORE)
    # Never PASS on missing data; never force HOLD from absence alone
    # (escalating a single guard's silence all the way is the fusion layer's job).
    action = GuardAction.LIMIT_SPEED
    return GuardReport(
        guard=GuardName.BATTERY,
        score=score,
        confidence=confidence,
        reason_codes=tuple(reasons),
        recommended_action=action,
        provenance={
            "source": str(telemetry.source.value),
            "window_sample_count": str(window_len),
            "battery_fraction": "unavailable",
            "discharge_rate_per_min": "unavailable",
            "estimated_remaining_min": "unavailable",
            "expected_voltage_at_soc": "unavailable",
        },
    )


def _fallback_report() -> GuardReport:
    """Used only if scoring itself raises unexpectedly (see `observe`'s
    try/except) -- maximally conservative by construction, mirroring
    estimator/risk.py's `_fallback_estimate`."""
    return GuardReport(
        guard=GuardName.BATTERY,
        score=1.0,
        confidence=0.0,
        reason_codes=(ReasonCode.NAN_OR_INVALID,),
        recommended_action=GuardAction.REQUEST_HOLD,
        provenance={"source": "unknown", "error": "battery_guard_internal_failure"},
    )


class BatteryMarginGuard:
    """Stateful battery-margin guard wrapping a bounded rolling window of
    recent `RobotTelemetry` samples (see `DEFAULT_WINDOW_SIZE` for sizing
    rationale). Not thread-safe; intended for one control-loop thread
    calling `observe()` once per new sample, same pattern as
    `estimator.risk.RiskEstimator`.
    """

    def __init__(self, maxlen: int = DEFAULT_WINDOW_SIZE) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self._window: deque[RobotTelemetry] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._window)

    def observe(self, telemetry: RobotTelemetry, now_monotonic: float) -> GuardReport:
        # `now_monotonic` is accepted for signature parity with the other
        # guards/estimators (and for a caller's own logging) but is
        # deliberately not used to score staleness here: detecting stale/
        # frozen/out-of-order telemetry is TELEMETRY_HEALTH's job per
        # docs/CONTRACTS.md's guard-family split, not this guard's. This
        # guard's own "freshness" concept is about the *window* -- whether
        # it has enough same-window history to trust a discharge-rate
        # estimate (see MIN_WINDOW_SPAN_SECONDS_FOR_RATE) -- not wall-clock
        # age of the latest sample.
        del now_monotonic
        try:
            self._window.append(telemetry)
            return self._score(telemetry)
        except Exception:
            # AGENTS.md safety constraint 3: never crash, never silently
            # pass -- an unexpected bug in this module's own math must fail
            # all the way conservative instead of propagating.
            return _fallback_report()

    def _score(self, telemetry: RobotTelemetry) -> GuardReport:
        window = list(self._window)
        fraction = telemetry.battery_fraction

        if fraction is None or not _is_finite_number(fraction):
            valid_count = sum(
                1
                for t in window
                if t.battery_fraction is not None and _is_finite_number(t.battery_fraction)
            )
            return _unavailable_report(telemetry, len(window), all_missing=(valid_count == 0))

        fraction = _clip01(float(fraction))

        voltage = telemetry.battery_voltage
        voltage = float(voltage) if voltage is not None and _is_finite_number(voltage) else None
        current_a = telemetry.battery_current_a
        current_a = (
            float(current_a) if current_a is not None and _is_finite_number(current_a) else None
        )
        temperature_c = telemetry.battery_temperature_c
        temperature_c = (
            float(temperature_c)
            if temperature_c is not None and _is_finite_number(temperature_c)
            else None
        )

        reasons: list[ReasonCode] = []
        confidence = 1.0

        # --- cold derating -> effective SoC margin (compounds with low SoC) ---
        if temperature_c is not None:
            retention = _temperature_retention(temperature_c)
            if temperature_c < COLD_DERATE_START_C:
                reasons.append(ReasonCode.BATTERY_COLD_DERATED)
        else:
            retention = 1.0
            confidence = min(confidence, 0.9)  # unknown thermal state -> slightly less trust

        effective_fraction = _clip01(fraction * retention)
        soc_component = _clip01(_piecewise_linear(effective_fraction, SOC_SCORE_ANCHORS))
        if effective_fraction <= SOC_LOW_REASON_THRESHOLD:
            reasons.append(ReasonCode.BATTERY_MARGIN_LOW)

        # --- discharge rate / estimated remaining operating time ---
        points = [
            (t.monotonic_time, float(t.battery_fraction))
            for t in window
            if t.battery_fraction is not None
            and _is_finite_number(t.battery_fraction)
            and _is_finite_number(t.monotonic_time)
        ]
        rate: float | None = None
        if len(points) >= 2:
            span = max(t for t, _ in points) - min(t for t, _ in points)
            if span >= MIN_WINDOW_SPAN_SECONDS_FOR_RATE:
                rate = _discharge_rate_per_second(points)
        if rate is None:
            confidence = min(confidence, 0.7)

        remaining_seconds: float | None = None
        if rate is not None and rate > DISCHARGE_RATE_NOISE_EPS_PER_SEC:
            candidate = fraction / rate
            if math.isfinite(candidate) and candidate >= 0.0:
                remaining_seconds = candidate
        remaining_minutes = remaining_seconds / 60.0 if remaining_seconds is not None else None

        if remaining_minutes is not None:
            remaining_time_component = _clip01(
                _piecewise_linear(remaining_minutes, REMAINING_TIME_SCORE_ANCHORS_MIN)
            )
            if remaining_minutes <= REMAINING_TIME_LOW_REASON_THRESHOLD_MIN:
                reasons.append(ReasonCode.BATTERY_MARGIN_LOW)
        else:
            remaining_time_component = 0.0

        # --- voltage sag under load ---
        sag_component, expected_voltage, load_confirmed_sag = _sag_component(
            fraction, voltage, current_a
        )
        if load_confirmed_sag:
            reasons.append(ReasonCode.BATTERY_VOLTAGE_SAG)
        if voltage is None:
            confidence = min(confidence, 0.85)
        if current_a is None:
            confidence = min(confidence, 0.9)

        # --- combine: worst single signal drives the score, not an average
        # of three separate concerns (same philosophy as estimator/risk.py:
        # a single severe signal must not be diluted by two calm ones). ---
        score = _clip01(max(soc_component, remaining_time_component, sag_component))

        # --- window warm-up: not enough same-window SoC history yet ---
        if len(points) < 2:
            confidence = min(confidence, 0.5)

        confidence = _clip_confidence(confidence)

        if not reasons:
            reasons.append(ReasonCode.NOMINAL)
        reasons_tuple = tuple(dict.fromkeys(reasons))  # de-dup, preserve first-seen order

        if score >= SCORE_REQUEST_HOLD_THRESHOLD:
            action = GuardAction.REQUEST_HOLD
        elif score >= SCORE_LIMIT_SPEED_THRESHOLD:
            action = GuardAction.LIMIT_SPEED
        else:
            action = GuardAction.PASS

        provenance = {
            "source": str(telemetry.source.value),
            "battery_fraction": _fmt(fraction),
            "battery_voltage_v": _fmt(voltage, digits=2),
            "battery_current_a": _fmt(current_a, digits=2),
            "battery_temperature_c": _fmt(temperature_c, digits=1),
            "effective_fraction": _fmt(effective_fraction),
            "temperature_retention_factor": _fmt(retention, digits=3),
            "discharge_rate_per_min": _fmt(rate * 60.0 if rate is not None else None, digits=4),
            "estimated_remaining_min": _fmt(remaining_minutes, digits=1),
            "expected_voltage_at_soc": _fmt(expected_voltage, digits=2),
            "window_sample_count": str(len(window)),
            "window_soc_sample_count": str(len(points)),
        }

        return GuardReport(
            guard=GuardName.BATTERY,
            score=score,
            confidence=confidence,
            reason_codes=reasons_tuple,
            recommended_action=action,
            provenance=provenance,
        )
