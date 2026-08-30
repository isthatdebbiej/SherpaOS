"""Unit tests for sherpaos.estimator (features.py + risk.py).

Everything here constructs `RobotTelemetry` directly (no import of
`sherpaos.sim` or `mujoco`) -- estimator/policy code must generalize to any
telemetry stream, not just a golden canned fixture.
"""

from __future__ import annotations

import math

import numpy as np

from sherpaos.contracts import ReasonCode, RobotTelemetry, TelemetrySource
from sherpaos.estimator.features import (
    DEFAULT_STALE_THRESHOLD_SECONDS,
    FeatureWindow,
)
from sherpaos.estimator.risk import RiskEstimate, RiskEstimator

N_JOINTS = 29


def make_telemetry(
    *,
    monotonic_time: float = 0.0,
    source_time: float = 0.0,
    sequence: int = 0,
    joint_position: np.ndarray | None = None,
    joint_velocity: np.ndarray | None = None,
    joint_effort: np.ndarray | None = None,
    base_orientation: np.ndarray | None = None,
    base_angular_velocity: np.ndarray | None = None,
    base_linear_acceleration: np.ndarray | None = None,
    commanded_velocity: np.ndarray | None = None,
    gait_mode: str | None = None,
    battery_fraction: float | None = None,
    battery_voltage: float | None = None,
    source: TelemetrySource = TelemetrySource.SIM,
    valid: bool = True,
) -> RobotTelemetry:
    """Build a synthetic, self-contained RobotTelemetry sample for tests."""
    return RobotTelemetry(
        monotonic_time=monotonic_time,
        source_time=source_time,
        sequence=sequence,
        joint_position=(
            np.zeros(N_JOINTS)
            if joint_position is None
            else np.asarray(joint_position, dtype=float)
        ),
        joint_velocity=(
            np.zeros(N_JOINTS)
            if joint_velocity is None
            else np.asarray(joint_velocity, dtype=float)
        ),
        joint_effort=None if joint_effort is None else np.asarray(joint_effort, dtype=float),
        base_orientation=(
            np.array([1.0, 0.0, 0.0, 0.0])
            if base_orientation is None
            else np.asarray(base_orientation, dtype=float)
        ),
        base_angular_velocity=(
            np.zeros(3)
            if base_angular_velocity is None
            else np.asarray(base_angular_velocity, dtype=float)
        ),
        base_linear_acceleration=(
            np.array([0.0, 0.0, 9.81])
            if base_linear_acceleration is None
            else np.asarray(base_linear_acceleration, dtype=float)
        ),
        commanded_velocity=(
            None if commanded_velocity is None else np.asarray(commanded_velocity, dtype=float)
        ),
        gait_mode=gait_mode,
        battery_fraction=battery_fraction,
        battery_voltage=battery_voltage,
        source=source,
        valid=valid,
    )


def _pitch_quat(angle_rad: float) -> np.ndarray:
    """Pure-pitch (rotation about body Y) unit quaternion, wxyz."""
    return np.array([math.cos(angle_rad / 2.0), 0.0, math.sin(angle_rad / 2.0), 0.0])


def _idle_sequence(n: int, dt: float = 0.01):
    """n perfectly nominal, evenly spaced, in-order, *complete* telemetry
    samples (joint_effort/commanded_velocity populated, so nothing trips
    MISSING_FIELD -- this is meant to exercise the true "nothing wrong"
    steady state, not the narrower "one optional field absent" state)."""
    return [
        make_telemetry(
            monotonic_time=i * dt,
            source_time=i * dt,
            sequence=i,
            joint_effort=np.zeros(N_JOINTS),
            commanded_velocity=np.zeros(3),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# FeatureWindow behavior
# ---------------------------------------------------------------------------


def test_feature_window_respects_maxlen():
    window = FeatureWindow(maxlen=5)
    for t in _idle_sequence(20):
        window.push(t)
    assert len(window) == 5


def test_feature_window_out_of_order_detected_and_not_sticky():
    window = FeatureWindow()
    window.push(make_telemetry(monotonic_time=1.0, sequence=5))
    f = window.compute_features(now_monotonic=1.0)
    assert f.is_out_of_order is False

    # sequence goes backwards -> flagged
    window.push(make_telemetry(monotonic_time=1.01, sequence=2))
    f = window.compute_features(now_monotonic=1.01)
    assert f.is_out_of_order is True

    # next sample resumes proper ordering vs the last *pushed* sample -> not sticky
    window.push(make_telemetry(monotonic_time=1.02, sequence=6))
    f = window.compute_features(now_monotonic=1.02)
    assert f.is_out_of_order is False


def test_feature_window_staleness_threshold():
    window = FeatureWindow()
    window.push(make_telemetry(monotonic_time=0.0, sequence=0))

    fresh = window.compute_features(now_monotonic=DEFAULT_STALE_THRESHOLD_SECONDS / 2)
    assert fresh.is_stale is False

    stale = window.compute_features(now_monotonic=DEFAULT_STALE_THRESHOLD_SECONDS * 4)
    assert stale.is_stale is True


def test_feature_window_detects_future_dated_sample():
    window = FeatureWindow()
    window.push(make_telemetry(monotonic_time=1.0, sequence=1))

    features = window.compute_features(now_monotonic=0.0)

    assert features.is_future_dated


def test_feature_window_missing_optional_fields_scoped_narrowly():
    window = FeatureWindow()
    # joint_effort + commanded_velocity missing -> reported (risk-relevant)
    window.push(make_telemetry(joint_effort=None, commanded_velocity=None))
    f = window.compute_features(now_monotonic=0.0)
    assert set(f.missing_optional_fields) == {"joint_effort", "commanded_velocity"}

    # battery-only gap is out of scope for mobility-risk features
    window2 = FeatureWindow()
    window2.push(
        make_telemetry(
            joint_effort=np.zeros(N_JOINTS),
            commanded_velocity=np.zeros(3),
            battery_fraction=None,
            battery_voltage=None,
        )
    )
    f2 = window2.compute_features(now_monotonic=0.0)
    assert f2.missing_optional_fields == ()


def test_feature_window_has_nan_flags_latest_sample():
    window = FeatureWindow()
    bad_velocity = np.zeros(N_JOINTS)
    bad_velocity[3] = float("nan")
    window.push(make_telemetry(joint_velocity=bad_velocity))
    f = window.compute_features(now_monotonic=0.0)
    assert f.has_nan is True


def test_feature_window_asymmetry_detects_left_right_imbalance():
    window = FeatureWindow()
    vel = np.zeros(N_JOINTS)
    vel[0:6] = 2.0  # left leg moving fast, right leg (6:12) stationary
    window.push(make_telemetry(joint_velocity=vel))
    f = window.compute_features(now_monotonic=0.0)
    assert f.leg_asymmetry > 0.9  # near-total imbalance -> ratio near 1.0

    window2 = FeatureWindow()
    vel2 = np.zeros(N_JOINTS)
    vel2[0:6] = 1.0
    vel2[6:12] = 1.0  # symmetric
    window2.push(make_telemetry(joint_velocity=vel2))
    f2 = window2.compute_features(now_monotonic=0.0)
    assert f2.leg_asymmetry == 0.0


# ---------------------------------------------------------------------------
# RiskEstimator: bounds + adversarial robustness
# ---------------------------------------------------------------------------


def test_score_and_confidence_always_bounded_nan_array():
    est = RiskEstimator()
    bad_vel = np.zeros(N_JOINTS)
    bad_vel[0] = float("nan")
    t = make_telemetry(joint_velocity=bad_vel)
    result = est.update(t, now_monotonic=0.0)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert ReasonCode.NAN_OR_INVALID in result.reason_codes
    assert len(result.reason_codes) > 0


def test_score_and_confidence_always_bounded_inf_value():
    est = RiskEstimator()
    bad_accel = np.array([float("inf"), 0.0, 9.81])
    t = make_telemetry(base_linear_acceleration=bad_accel)
    result = est.update(t, now_monotonic=0.0)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_score_and_confidence_always_bounded_extreme_value():
    est = RiskEstimator()
    huge_vel = np.full(N_JOINTS, 1.0e6)
    t = make_telemetry(joint_velocity=huge_vel, joint_effort=np.full(N_JOINTS, 1.0e6))
    result = est.update(t, now_monotonic=0.0)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.reason_codes) > 0


def test_score_and_confidence_always_bounded_all_zeros():
    est = RiskEstimator()
    t = RobotTelemetry(
        monotonic_time=0.0,
        source_time=0.0,
        sequence=0,
        joint_position=np.zeros(N_JOINTS),
        joint_velocity=np.zeros(N_JOINTS),
        joint_effort=None,
        base_orientation=np.zeros(4),  # degenerate (non-unit) quaternion
        base_angular_velocity=np.zeros(3),
        base_linear_acceleration=np.zeros(3),
    )
    result = est.update(t, now_monotonic=0.0)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.reason_codes) > 0


def test_score_bounded_across_many_random_adversarial_samples():
    rng = np.random.default_rng(1234)
    est = RiskEstimator()
    for i in range(200):
        scale = rng.choice([1.0, 1e3, 1e6])
        vel = rng.normal(scale=scale, size=N_JOINTS)
        if i % 17 == 0:
            vel[rng.integers(0, N_JOINTS)] = float("nan")
        quat = rng.normal(size=4) * scale
        t = make_telemetry(
            monotonic_time=i * 0.01,
            sequence=i,
            joint_velocity=vel,
            base_orientation=quat,
        )
        result = est.update(t, now_monotonic=i * 0.01)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.reason_codes) > 0


def test_nominal_idle_stream_settles_to_nominal_low_risk():
    est = RiskEstimator()
    result = None
    for t in _idle_sequence(30, dt=0.01):
        result = est.update(t, now_monotonic=t.monotonic_time)
    assert result is not None
    assert result.reason_codes == (ReasonCode.NOMINAL,)
    assert result.score < 0.15
    assert result.confidence > 0.85


def test_stale_sample_forces_high_score_low_confidence_and_reason_code():
    est = RiskEstimator()
    t = make_telemetry(monotonic_time=0.0, sequence=0)
    # "now" is far past the sample's timestamp -> stale
    result = est.update(t, now_monotonic=10.0)
    assert result.score > 0.4
    assert result.confidence < 0.5
    assert ReasonCode.STALE_TELEMETRY in result.reason_codes


def test_producer_invalid_flag_forces_conservative_estimate():
    est = RiskEstimator()
    t = make_telemetry(valid=False)
    result = est.update(t, now_monotonic=0.0)
    assert result.score > 0.4
    assert result.confidence < 0.3
    assert ReasonCode.NAN_OR_INVALID in result.reason_codes


def test_out_of_order_sample_flagged():
    est = RiskEstimator()
    est.update(make_telemetry(monotonic_time=1.0, sequence=10), now_monotonic=1.0)
    result = est.update(make_telemetry(monotonic_time=1.01, sequence=3), now_monotonic=1.01)
    assert ReasonCode.OUT_OF_ORDER in result.reason_codes
    assert result.score > 0.3


# ---------------------------------------------------------------------------
# Worsening-trend test
# ---------------------------------------------------------------------------


def test_worsening_orientation_and_residuals_trend_score_upward():
    est = RiskEstimator()
    scores: list[float] = []
    n = 60
    for i in range(n):
        angle = i * 0.01  # tilt grows from 0 to ~0.59 rad (~34 deg)
        quat = _pitch_quat(angle)
        vel = np.zeros(N_JOINTS)
        vel[0:6] = 0.05 * i  # growing, asymmetric (left-leg-only) velocity drift
        t = make_telemetry(
            monotonic_time=i * 0.01,
            source_time=i * 0.01,
            sequence=i,
            base_orientation=quat,
            joint_velocity=vel,
            joint_effort=np.zeros(N_JOINTS),
            commanded_velocity=np.zeros(3),
        )
        result = est.update(t, now_monotonic=i * 0.01)
        scores.append(result.score)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    quarter = n // 4
    early_avg = sum(scores[:quarter]) / quarter
    late_avg = sum(scores[-quarter:]) / quarter
    assert late_avg > early_avg + 0.2, (early_avg, late_avg)
    assert scores[-1] > scores[0]


# ---------------------------------------------------------------------------
# RiskEstimate sanity
# ---------------------------------------------------------------------------


def test_risk_estimate_reason_codes_never_empty():
    est = RiskEstimator()
    for t in _idle_sequence(5):
        result: RiskEstimate = est.update(t, now_monotonic=t.monotonic_time)
        assert len(result.reason_codes) >= 1
        assert all(isinstance(r, ReasonCode) for r in result.reason_codes)
