"""Unit tests for sherpaos.battery.guard (BatteryMarginGuard).

Everything here constructs `RobotTelemetry` directly -- no import of
`sherpaos.sim`, `sherpaos.estimator`, `sherpaos.policy`, or mujoco. The
battery guard must generalize to any telemetry stream (sim, dump, or
eventually a live G1), not just a canned fixture from another lane.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sherpaos.battery.guard import (
    COLD_DERATE_SEVERE_C,
    EXPECTED_VOLTAGE_ANCHORS,
    SCORE_LIMIT_SPEED_THRESHOLD,
    SCORE_REQUEST_HOLD_THRESHOLD,
    SOC_LOW_REASON_THRESHOLD,
    BatteryMarginGuard,
    _piecewise_linear,
)
from sherpaos.contracts import GuardAction, GuardName, ReasonCode, RobotTelemetry, TelemetrySource

N_JOINTS = 29


def make_telemetry(
    *,
    monotonic_time: float = 0.0,
    sequence: int = 0,
    battery_fraction: float | None = 0.8,
    battery_voltage: float | None = 48.0,
    battery_current_a: float | None = 3.0,
    battery_temperature_c: float | None = 20.0,
    source: TelemetrySource = TelemetrySource.SIM,
    valid: bool = True,
) -> RobotTelemetry:
    """Build a synthetic, self-contained RobotTelemetry sample for tests.
    Non-battery fields are filled with harmless, well-formed defaults --
    this guard never reads them, but RobotTelemetry requires them."""
    return RobotTelemetry(
        monotonic_time=monotonic_time,
        source_time=monotonic_time,
        sequence=sequence,
        joint_position=np.zeros(N_JOINTS),
        joint_velocity=np.zeros(N_JOINTS),
        joint_effort=np.zeros(N_JOINTS),
        base_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        base_angular_velocity=np.zeros(3),
        base_linear_acceleration=np.array([0.0, 0.0, 9.81]),
        commanded_velocity=np.zeros(3),
        gait_mode="walk",
        battery_fraction=battery_fraction,
        battery_voltage=battery_voltage,
        battery_current_a=battery_current_a,
        battery_temperature_c=battery_temperature_c,
        source=source,
        valid=valid,
    )


def expected_voltage(fraction: float) -> float:
    """Same self-defined expected-voltage-vs-SoC curve the guard uses,
    reused here so tests can construct "no sag" or "sag" samples relative
    to it without hardcoding numbers that would silently drift if the
    curve's anchors are retuned."""
    return _piecewise_linear(fraction, EXPECTED_VOLTAGE_ANCHORS)


def feed_sequence(
    guard: BatteryMarginGuard,
    *,
    n: int,
    dt: float,
    start_fraction: float,
    end_fraction: float,
    voltage_offset: float = -0.5,
    current_a: float = 3.0,
    temperature_c: float = 20.0,
):
    """Feed `guard` a linearly-interpolated fraction sequence (start ->
    end over `n` samples spaced `dt` seconds apart), with voltage held a
    fixed, safe offset below the expected-voltage-vs-SoC curve (so no
    sag is triggered unless the caller wants it) and constant
    current/temperature. Returns the last GuardReport."""
    report = None
    for i in range(n):
        t = i * dt
        frac = start_fraction + (end_fraction - start_fraction) * (i / max(1, n - 1))
        volt = expected_voltage(frac) + voltage_offset
        telemetry = make_telemetry(
            monotonic_time=t,
            sequence=i,
            battery_fraction=frac,
            battery_voltage=volt,
            battery_current_a=current_a,
            battery_temperature_c=temperature_c,
        )
        report = guard.observe(telemetry, now_monotonic=t)
    assert report is not None
    return report


def assert_bounded(report):
    assert 0.0 <= report.score <= 1.0
    assert 0.0 <= report.confidence <= 1.0
    assert math.isfinite(report.score)
    assert math.isfinite(report.confidence)
    assert report.guard == GuardName.BATTERY
    assert len(report.reason_codes) > 0


# --------------------------------------------------------------------------


def test_nominal_slow_decline_room_temperature_stays_pass():
    guard = BatteryMarginGuard()
    report = feed_sequence(
        guard,
        n=30,
        dt=2.0,
        start_fraction=0.90,
        end_fraction=0.885,  # slow decline over ~58s -> projected remaining >> 60 min
    )
    assert_bounded(report)
    assert report.recommended_action == GuardAction.PASS
    assert report.confidence > 0.5
    assert report.reason_codes == (ReasonCode.NOMINAL,)
    assert report.provenance["source"] == TelemetrySource.SIM.value


def test_missing_battery_fraction_reports_unavailable_and_caps_action():
    guard = BatteryMarginGuard()
    telemetry = make_telemetry(monotonic_time=0.0, sequence=0, battery_fraction=None)
    report = guard.observe(telemetry, now_monotonic=0.0)

    assert_bounded(report)
    assert ReasonCode.BATTERY_DATA_UNAVAILABLE in report.reason_codes
    assert report.confidence < 0.3
    # No more permissive than LIMIT_SPEED: never PASS on missing data, and a
    # single guard's absence of data must not alone force REQUEST_HOLD.
    assert report.recommended_action == GuardAction.LIMIT_SPEED


def test_missing_battery_fraction_persistently_across_window_stays_capped():
    guard = BatteryMarginGuard()
    report = None
    for i in range(10):
        telemetry = make_telemetry(monotonic_time=i * 1.0, sequence=i, battery_fraction=None)
        report = guard.observe(telemetry, now_monotonic=i * 1.0)
    assert report is not None
    assert_bounded(report)
    assert ReasonCode.BATTERY_DATA_UNAVAILABLE in report.reason_codes
    assert report.recommended_action == GuardAction.LIMIT_SPEED
    assert report.recommended_action != GuardAction.REQUEST_HOLD


def test_fast_discharge_is_a_leading_indicator_beyond_soc_alone():
    """Same final SoC (0.55, still a 'moderate' absolute charge level) for
    both sequences; only the decline rate differs. The fast-declining
    sequence must score strictly higher than the slow one, and must show
    the margin reason code even though absolute SoC alone would not yet
    warrant it."""
    same_final_fraction = 0.55
    assert same_final_fraction > SOC_LOW_REASON_THRESHOLD  # isolate the rate signal, not raw SoC

    fast_guard = BatteryMarginGuard()
    fast_report = feed_sequence(
        fast_guard,
        n=20,
        dt=2.0,
        start_fraction=0.70,
        end_fraction=same_final_fraction,  # drops 0.15 over 38s -> a few minutes remaining
    )

    slow_guard = BatteryMarginGuard()
    slow_report = feed_sequence(
        slow_guard,
        n=20,
        dt=2.0,
        start_fraction=0.552,
        end_fraction=same_final_fraction,  # drops 0.002 over 38s -> hours remaining
    )

    assert_bounded(fast_report)
    assert_bounded(slow_report)

    assert fast_report.score > slow_report.score
    assert ReasonCode.BATTERY_MARGIN_LOW in fast_report.reason_codes
    assert fast_report.provenance["estimated_remaining_min"] != "unavailable"
    assert float(fast_report.provenance["estimated_remaining_min"]) < 15.0

    # The slow control should read as comfortably nominal.
    assert slow_report.recommended_action == GuardAction.PASS
    # The fast case must be flagged as worse than PASS.
    assert fast_report.recommended_action != GuardAction.PASS


def test_voltage_sag_under_high_load_is_flagged():
    guard = BatteryMarginGuard()
    fraction = 0.60
    sagged_voltage = expected_voltage(fraction) - 3.0  # deficit well past the min-deficit threshold
    report = None
    for i in range(10):
        telemetry = make_telemetry(
            monotonic_time=i * 2.0,
            sequence=i,
            battery_fraction=fraction,  # flat SoC: isolate sag from the rate signal
            battery_voltage=sagged_voltage,
            battery_current_a=20.0,  # high discharge load
            battery_temperature_c=20.0,
        )
        report = guard.observe(telemetry, now_monotonic=i * 2.0)
    assert report is not None
    assert_bounded(report)
    assert ReasonCode.BATTERY_VOLTAGE_SAG in report.reason_codes
    assert report.recommended_action != GuardAction.PASS


def test_voltage_deficit_without_high_load_does_not_claim_sag():
    """Same voltage deficit as the sag test, but low current (no confirmed
    load) -- must not raise the load-confirmed BATTERY_VOLTAGE_SAG code."""
    guard = BatteryMarginGuard()
    fraction = 0.60
    sagged_voltage = expected_voltage(fraction) - 3.0
    report = None
    for i in range(10):
        telemetry = make_telemetry(
            monotonic_time=i * 2.0,
            sequence=i,
            battery_fraction=fraction,
            battery_voltage=sagged_voltage,
            battery_current_a=1.0,  # low/idle current -- no confirmed load
            battery_temperature_c=20.0,
        )
        report = guard.observe(telemetry, now_monotonic=i * 2.0)
    assert report is not None
    assert_bounded(report)
    assert ReasonCode.BATTERY_VOLTAGE_SAG not in report.reason_codes


def test_cold_temperature_derates_moderate_soc_vs_warm_control():
    fraction = 0.50  # moderate SoC that would look fine on its own

    cold_guard = BatteryMarginGuard()
    cold_report = None
    warm_guard = BatteryMarginGuard()
    warm_report = None
    for i in range(10):
        t = i * 2.0
        cold_report = cold_guard.observe(
            make_telemetry(
                monotonic_time=t,
                sequence=i,
                battery_fraction=fraction,
                battery_voltage=expected_voltage(fraction) - 0.5,
                battery_current_a=3.0,
                battery_temperature_c=COLD_DERATE_SEVERE_C,  # well below freezing
            ),
            now_monotonic=t,
        )
        warm_report = warm_guard.observe(
            make_telemetry(
                monotonic_time=t,
                sequence=i,
                battery_fraction=fraction,
                battery_voltage=expected_voltage(fraction) - 0.5,
                battery_current_a=3.0,
                battery_temperature_c=20.0,
            ),
            now_monotonic=t,
        )

    assert cold_report is not None
    assert warm_report is not None
    assert_bounded(cold_report)
    assert_bounded(warm_report)

    assert ReasonCode.BATTERY_COLD_DERATED in cold_report.reason_codes
    assert ReasonCode.BATTERY_COLD_DERATED not in warm_report.reason_codes

    # Same raw SoC, only temperature differs -- cold must score meaningfully worse.
    assert cold_report.score > warm_report.score + 0.15
    assert warm_report.recommended_action == GuardAction.PASS
    assert cold_report.recommended_action != GuardAction.PASS


def test_flat_or_rising_soc_does_not_fabricate_remaining_time():
    """A data glitch (or a charging event) where SoC is flat/rising must
    not crash and must not report an infinite or negative remaining-time
    estimate -- it should fall back to 'cannot estimate'."""
    guard = BatteryMarginGuard()
    report = None
    for i in range(10):
        t = i * 2.0
        frac = 0.50 + 0.005 * i  # slowly rising ("charging"/glitch)
        telemetry = make_telemetry(
            monotonic_time=t,
            sequence=i,
            battery_fraction=frac,
            battery_voltage=expected_voltage(frac) - 0.5,
            battery_current_a=2.0,
            battery_temperature_c=20.0,
        )
        report = guard.observe(telemetry, now_monotonic=t)
    assert report is not None
    assert_bounded(report)
    assert report.provenance["estimated_remaining_min"] == "unavailable"
    assert report.provenance["discharge_rate_per_min"] == "unavailable"


def test_perfectly_flat_soc_two_samples_does_not_crash_or_divide_by_zero():
    guard = BatteryMarginGuard()
    for i in range(2):
        t = i * 2.0
        telemetry = make_telemetry(
            monotonic_time=t,
            sequence=i,
            battery_fraction=0.50,
            battery_voltage=expected_voltage(0.50) - 0.5,
            battery_current_a=2.0,
            battery_temperature_c=20.0,
        )
        report = guard.observe(telemetry, now_monotonic=t)
    assert_bounded(report)
    assert report.provenance["estimated_remaining_min"] == "unavailable"


@pytest.mark.parametrize(
    "battery_fraction,battery_voltage,battery_current_a,battery_temperature_c",
    [
        (0.02, 20.0, 25.0, -25.0),  # severe: near-empty, sagging, freezing
        (1.0, 54.6, 0.0, 20.0),  # full charge, no load, warm
        (float("nan"), 48.0, 3.0, 20.0),  # malformed SoC reading
    ],
)
def test_score_and_confidence_always_bounded(
    battery_fraction, battery_voltage, battery_current_a, battery_temperature_c
):
    guard = BatteryMarginGuard()
    telemetry = make_telemetry(
        monotonic_time=0.0,
        sequence=0,
        battery_fraction=battery_fraction,
        battery_voltage=battery_voltage,
        battery_current_a=battery_current_a,
        battery_temperature_c=battery_temperature_c,
    )
    report = guard.observe(telemetry, now_monotonic=0.0)
    assert_bounded(report)


def test_severe_low_soc_short_remaining_time_can_reach_request_hold():
    guard = BatteryMarginGuard()
    report = feed_sequence(
        guard,
        n=15,
        dt=2.0,
        start_fraction=0.12,
        end_fraction=0.05,  # fast drop deep into the critical band
    )
    assert_bounded(report)
    assert report.score >= SCORE_REQUEST_HOLD_THRESHOLD
    assert report.recommended_action == GuardAction.REQUEST_HOLD
    assert ReasonCode.BATTERY_MARGIN_LOW in report.reason_codes


def test_recommended_action_thresholds_are_internally_consistent():
    assert 0.0 < SCORE_LIMIT_SPEED_THRESHOLD < SCORE_REQUEST_HOLD_THRESHOLD < 1.0
