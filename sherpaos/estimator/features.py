"""Rolling-window feature extraction over RobotTelemetry.

This module owns exactly one job: turn a bounded history of onboard-observable
`RobotTelemetry` samples into a `Features` snapshot that `estimator/risk.py` can
score. It imports nothing but `sherpaos.contracts`, stdlib, and numpy -- see
`docs/CONTRACTS.md`'s leakage rule. It must never raise on malformed input;
every helper here degrades to a conservative default instead of crashing, per
AGENTS.md safety constraint 3.

Feature groups (mirroring docs/idea.txt section 6, collapsed into one module
per docs/DECISIONS.md's "single estimator" decision):
  - validity/freshness: is this sample fresh, complete, well-ordered, finite?
  - orientation instability ("is the body doing something unexpected?")
  - body residual / anomaly proxy ("BodySense"-style, section 6.2)
  - left/right asymmetry
  - traction/slip proxy ("IceSense"-style, section 6.1 -- see docstring on
    `_slip_proxy` for why this is an approximation, not literal foot-slip
    detection: this contract has no foot-contact/foot-velocity fields.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from sherpaos.contracts import RobotTelemetry

# --------------------------------------------------------------------------
# Window sizing.
#
# G1's control loop is assumed to run somewhere in the 50-200 Hz range (not
# specified further anywhere in docs/, so we design for the whole range
# rather than pick a single number). DEFAULT_WINDOW_SIZE=100 samples gives:
#   - 2.0s of trailing history at 50 Hz (slow end)
#   - 0.5s of trailing history at 200 Hz (fast end)
# That's enough history to compute meaningful short-horizon variability
# (orientation std-dev, velocity residual) without dragging in multi-second
# lag, and 100 samples of ~29-element float64 arrays is a trivial memory
# footprint (well under 1 MB) for a deque.
DEFAULT_WINDOW_SIZE = 100

# A sample older than this (age_seconds vs "now") is considered stale.
# At the slow end of the assumed loop rate (50 Hz => 20ms nominal period),
# 250ms is 12+ missed/delayed frames -- comfortably past ordinary scheduling
# jitter (a frame or two) while still catching a genuinely stalled feed
# quickly (well under a second).
DEFAULT_STALE_THRESHOLD_SECONDS = 0.25

# Sentinel age used only when features are computed on a window that has
# never had a sample pushed into it. This should be unreachable via
# RiskEstimator (which always pushes before computing), but we keep it a
# large *finite* value -- never inf/nan -- so anything downstream that does
# arithmetic on it (e.g. GuardDecision.input_age_seconds) stays well-behaved.
_NO_DATA_AGE_SECONDS = 1.0e9

N_JOINTS = 29

# Joint index groups, per the G1 actuator order documented in contracts.py /
# the task brief: left leg 0-5 pairs with right leg 6-11 in the same
# within-leg order (hip_pitch<->hip_pitch, ..., ankle_roll<->ankle_roll);
# left arm 15-21 pairs with right arm 22-28 the same way.
LEFT_LEG_SLICE = slice(0, 6)
RIGHT_LEG_SLICE = slice(6, 12)
BOTH_LEGS_SLICE = slice(0, 12)
LEFT_ARM_SLICE = slice(15, 22)
RIGHT_ARM_SLICE = slice(22, 29)

# --------------------------------------------------------------------------
# Traction/slip proxy tuning constants.
#
# G1 has no foot-contact or foot-velocity sensing in this contract (see
# docs/CONTRACTS.md -- deliberately excluded). We cannot detect literal foot
# slip. Instead we combine two indirect, onboard-observable signals that
# correlate with loss-of-traction events in legged locomotion:
#   (1) leg joints moving substantially while the robot was commanded to
#       stay (near-)stationary -- consistent with a foot sliding/skating
#       instead of holding a planted stance;
#   (2) short-horizon orientation instability co-occurring with (1) --
#       consistent with the body reacting to an unexpected/unstable
#       contact, rather than a smooth intended motion;
#   (3) (if effort telemetry exists) an effort residual spike while leg
#       speed stays low -- consistent with a foot "catching" against
#       resistance without producing the expected motion.
# This is a coarse heuristic, not a physical slip detector: e.g. a fast,
# jerky *intentional* turn could also trip it. Read the output as "traction
# / body-response anomaly signature" per docs/idea.txt 6.1's own framing
# ("detect traction-loss signatures", not "detect ice type") -- never claim
# more than that.
LOW_COMMAND_SPEED_MPS = 0.05
# rad/s treated as "fully suspicious" leg motion while uncommanded
UNCOMMANDED_LEG_SPEED_SCALE = 2.0
ORIENTATION_STD_SCALE = 0.15  # rad (~8.6 deg) std treated as "fully unstable"
ANGULAR_VELOCITY_STD_SCALE = 1.0  # rad/s std treated as "fully unstable"
# N*m RMS residual treated as "fully spiking" -- placeholder, retune with real G1 effort ranges
EFFORT_RESIDUAL_SCALE = 5.0
_LOW_LEG_SPEED_FOR_EFFORT_CHECK = UNCOMMANDED_LEG_SPEED_SCALE * 0.25


@dataclass(slots=True, frozen=True)
class Features:
    """One feature snapshot derived from the current window contents."""

    # --- validity / freshness ---
    age_seconds: float
    is_stale: bool
    has_nan: bool
    is_out_of_order: bool
    missing_optional_fields: tuple[str, ...]
    producer_invalid: bool
    sample_count: int

    # --- orientation instability ---
    roll: float
    pitch: float
    angular_velocity_magnitude: float
    roll_std: float
    pitch_std: float
    angular_velocity_std: float

    # --- body residual / anomaly proxy ---
    joint_velocity_residual: float
    joint_effort_residual: float | None

    # --- left/right asymmetry ---
    leg_asymmetry: float
    arm_asymmetry: float
    asymmetry_score: float

    # --- traction/slip proxy ---
    slip_proxy_score: float


def _clip01(x: float) -> float:
    """Clip to [0, 1]; a non-finite input fails conservative (-> 1.0, i.e.
    treated as maximal signal rather than silently dropped to 0)."""
    if not math.isfinite(x):
        return 1.0
    return min(1.0, max(0.0, x))


def _safe_norm(vec: np.ndarray | None) -> float:
    try:
        if vec is None:
            return 0.0
        arr = np.asarray(vec, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return 0.0
        return float(np.linalg.norm(arr))
    except Exception:
        return 0.0


def _quat_to_roll_pitch(quat: np.ndarray | None) -> tuple[float, float]:
    """wxyz quaternion -> (roll, pitch) in radians. Defensive: any malformed
    input (wrong shape, non-finite, near-zero norm) yields (0.0, 0.0) rather
    than raising -- callers combine this with the `has_nan` flag elsewhere so
    a genuinely bad quaternion still shows up as untrustworthy input."""
    try:
        arr = np.asarray(quat, dtype=float)
        if arr.shape != (4,) or not np.all(np.isfinite(arr)):
            return 0.0, 0.0
        w, x, y, z = arr
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm < 1e-8:
            return 0.0, 0.0
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = math.asin(sinp)
        if not (math.isfinite(roll) and math.isfinite(pitch)):
            return 0.0, 0.0
        return roll, pitch
    except Exception:
        return 0.0, 0.0


def _relative_asymmetry(vec: np.ndarray | None, left: slice, right: slice) -> float:
    """|left_norm - right_norm| / (left_norm + right_norm), bounded [0, 1]
    (equals 0 when magnitudes match, approaches 1 the more one side
    dominates). Returns 0.0 (not anomalous) on any malformed input -- this
    is a summary comparison, not itself a data-quality signal."""
    try:
        arr = np.asarray(vec, dtype=float)
        if arr.shape[0] < right.stop:
            return 0.0
        left_vals, right_vals = arr[left], arr[right]
        if not (np.all(np.isfinite(left_vals)) and np.all(np.isfinite(right_vals))):
            return 0.0
        left_mag = float(np.linalg.norm(left_vals))
        right_mag = float(np.linalg.norm(right_vals))
        denom = left_mag + right_mag
        if denom < 1e-6:
            return 0.0
        return _clip01(abs(left_mag - right_mag) / denom)
    except Exception:
        return 0.0


def _joint_velocity_residual(buf: list[RobotTelemetry]) -> float:
    """RMS, across joints, of (latest velocity - mean of prior window
    velocities): a simple "expected vs observed" residual. Skips any sample
    whose velocity array is malformed rather than letting it poison the
    mean. Returns 0.0 if there isn't enough clean history to compare
    against (not "anomalous" -- "unknown")."""
    try:
        vels = []
        for t in buf:
            arr = np.asarray(t.joint_velocity, dtype=float)
            if arr.shape == (N_JOINTS,) and np.all(np.isfinite(arr)):
                vels.append(arr)
        if len(vels) < 2:
            return 0.0
        current = vels[-1]
        expected = np.mean(np.stack(vels[:-1]), axis=0)
        residual = current - expected
        val = float(np.sqrt(np.mean(np.square(residual))))
        return val if math.isfinite(val) else 0.0
    except Exception:
        return 0.0


def _joint_effort_residual(buf: list[RobotTelemetry]) -> float | None:
    """Same idea as `_joint_velocity_residual` for effort, when available.
    Returns None (not 0.0) whenever a residual can't be computed -- callers
    must treat None as "no signal", never as "residual is zero"."""
    try:
        latest = buf[-1]
        if latest.joint_effort is None:
            return None
        current = np.asarray(latest.joint_effort, dtype=float)
        if current.shape != (N_JOINTS,) or not np.all(np.isfinite(current)):
            return None
        history = []
        for t in buf[:-1]:
            if t.joint_effort is None:
                continue
            arr = np.asarray(t.joint_effort, dtype=float)
            if arr.shape == (N_JOINTS,) and np.all(np.isfinite(arr)):
                history.append(arr)
        if not history:
            return None
        expected = np.mean(np.stack(history), axis=0)
        residual = current - expected
        val = float(np.sqrt(np.mean(np.square(residual))))
        return val if math.isfinite(val) else None
    except Exception:
        return None


def _sample_has_nan(t: RobotTelemetry) -> bool:
    """True if any numeric field present on this sample contains NaN/Inf,
    or is malformed enough that we can't tell (fails conservative -> True)."""
    arrays = [
        t.joint_position,
        t.joint_velocity,
        t.base_orientation,
        t.base_angular_velocity,
        t.base_linear_acceleration,
    ]
    if t.joint_effort is not None:
        arrays.append(t.joint_effort)
    if t.commanded_velocity is not None:
        arrays.append(t.commanded_velocity)
    for a in arrays:
        try:
            if a is None:
                return True
            if not np.all(np.isfinite(np.asarray(a, dtype=float))):
                return True
        except Exception:
            return True

    for s in (t.monotonic_time, t.source_time, t.battery_fraction, t.battery_voltage):
        if s is None:
            continue
        try:
            if not math.isfinite(float(s)):
                return True
        except Exception:
            return True
    return False


def _missing_optional_fields(t: RobotTelemetry) -> tuple[str, ...]:
    """Optional fields whose absence measurably degrades *this module's*
    signal quality. Deliberately scoped narrowly: `joint_effort` feeds the
    body-anomaly/slip proxies and `commanded_velocity` feeds the slip
    proxy's "moving while told to hold" check. `battery_fraction` /
    `battery_voltage` / `gait_mode` are real optional fields on
    RobotTelemetry but have no bearing on mobility-risk features computed
    here, so their absence is not reported as a mobility-relevant gap."""
    missing = []
    if t.joint_effort is None:
        missing.append("joint_effort")
    if t.commanded_velocity is None:
        missing.append("commanded_velocity")
    return tuple(missing)


def _slip_proxy(
    latest: RobotTelemetry,
    roll_std: float,
    pitch_std: float,
    angular_velocity_std: float,
    joint_effort_residual: float | None,
) -> float:
    try:
        leg_vel = np.asarray(latest.joint_velocity, dtype=float)[BOTH_LEGS_SLICE]
        leg_speed = _safe_norm(leg_vel) if np.all(np.isfinite(leg_vel)) else 0.0
    except Exception:
        leg_speed = 0.0

    commanded_speed: float | None = None
    if latest.commanded_velocity is not None:
        # `_safe_norm` returns 0.0 both for "genuinely zero" and "invalid".
        # That ambiguity is harmless here: an invalid commanded_velocity
        # means the sample already has `has_nan=True`, which already forces
        # a conservative score in risk.py independent of this proxy.
        commanded_speed = _safe_norm(latest.commanded_velocity)

    if commanded_speed is not None and commanded_speed < LOW_COMMAND_SPEED_MPS:
        uncommanded_motion = _clip01(leg_speed / UNCOMMANDED_LEG_SPEED_SCALE)
    else:
        uncommanded_motion = 0.0

    instability = _clip01(
        0.5 * ((roll_std + pitch_std) / ORIENTATION_STD_SCALE)
        + 0.5 * (angular_velocity_std / ANGULAR_VELOCITY_STD_SCALE)
    )

    if joint_effort_residual is not None and leg_speed < _LOW_LEG_SPEED_FOR_EFFORT_CHECK:
        effort_spike = _clip01(joint_effort_residual / EFFORT_RESIDUAL_SCALE)
    else:
        effort_spike = 0.0

    return _clip01(0.45 * uncommanded_motion + 0.35 * instability + 0.20 * effort_spike)


def _empty_features() -> Features:
    """Conservative placeholder for the (practically unreachable) case
    where features are requested before any sample has been pushed."""
    return Features(
        age_seconds=_NO_DATA_AGE_SECONDS,
        is_stale=True,
        has_nan=True,
        is_out_of_order=False,
        missing_optional_fields=(),
        producer_invalid=True,
        sample_count=0,
        roll=0.0,
        pitch=0.0,
        angular_velocity_magnitude=0.0,
        roll_std=0.0,
        pitch_std=0.0,
        angular_velocity_std=0.0,
        joint_velocity_residual=0.0,
        joint_effort_residual=None,
        leg_asymmetry=0.0,
        arm_asymmetry=0.0,
        asymmetry_score=0.0,
        slip_proxy_score=0.0,
    )


class FeatureWindow:
    """Bounded rolling buffer of RobotTelemetry + feature computation.

    Not thread-safe; intended for one control-loop thread calling
    `push()` then `compute_features()` once per new telemetry sample, which
    is exactly how `estimator.risk.RiskEstimator` drives it.
    """

    def __init__(
        self,
        maxlen: int = DEFAULT_WINDOW_SIZE,
        stale_threshold_seconds: float = DEFAULT_STALE_THRESHOLD_SECONDS,
    ) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self._buf: deque[RobotTelemetry] = deque(maxlen=maxlen)
        self.stale_threshold_seconds = stale_threshold_seconds
        self._prev_sequence: int | None = None
        self._prev_monotonic_time: float | None = None
        self._last_is_out_of_order: bool = False

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, sample: RobotTelemetry) -> None:
        """Append a sample and update ordering state. Comparison is against
        the immediately preceding *pushed* sample (whether or not that one
        was itself flagged out-of-order), so a single reordering blip is
        caught once rather than permanently poisoning future comparisons."""
        is_out_of_order = False
        try:
            if self._prev_sequence is not None and sample.sequence <= self._prev_sequence:
                is_out_of_order = True
            if (
                self._prev_monotonic_time is not None
                and sample.monotonic_time < self._prev_monotonic_time
            ):
                is_out_of_order = True
        except Exception:
            is_out_of_order = True
        self._last_is_out_of_order = is_out_of_order

        try:
            self._prev_sequence = sample.sequence
            self._prev_monotonic_time = sample.monotonic_time
        except Exception:
            pass

        self._buf.append(sample)

    def compute_features(self, now_monotonic: float) -> Features:
        buf = list(self._buf)
        if not buf:
            return _empty_features()

        latest = buf[-1]

        try:
            age_seconds = float(latest.age_seconds(now_monotonic))
            if not math.isfinite(age_seconds):
                age_seconds = _NO_DATA_AGE_SECONDS
        except Exception:
            age_seconds = _NO_DATA_AGE_SECONDS

        has_nan = _sample_has_nan(latest)
        missing_optional = _missing_optional_fields(latest)
        try:
            producer_invalid = not bool(latest.valid)
        except Exception:
            producer_invalid = True
        is_stale = age_seconds > self.stale_threshold_seconds
        is_out_of_order = self._last_is_out_of_order

        roll, pitch = _quat_to_roll_pitch(latest.base_orientation)
        angular_velocity_magnitude = _safe_norm(latest.base_angular_velocity)

        rolls: list[float] = []
        pitches: list[float] = []
        ang_mags: list[float] = []
        for t in buf:
            try:
                quat = np.asarray(t.base_orientation, dtype=float)
                if quat.shape != (4,) or not np.all(np.isfinite(quat)):
                    continue
                ang = np.asarray(t.base_angular_velocity, dtype=float)
                if not np.all(np.isfinite(ang)):
                    continue
                r, p = _quat_to_roll_pitch(quat)
                rolls.append(r)
                pitches.append(p)
                ang_mags.append(_safe_norm(ang))
            except Exception:
                continue
        roll_std = float(np.std(rolls)) if len(rolls) >= 2 else 0.0
        pitch_std = float(np.std(pitches)) if len(pitches) >= 2 else 0.0
        angular_velocity_std = float(np.std(ang_mags)) if len(ang_mags) >= 2 else 0.0
        if not math.isfinite(roll_std):
            roll_std = 0.0
        if not math.isfinite(pitch_std):
            pitch_std = 0.0
        if not math.isfinite(angular_velocity_std):
            angular_velocity_std = 0.0

        joint_velocity_residual = _joint_velocity_residual(buf)
        joint_effort_residual = _joint_effort_residual(buf)

        leg_asym_vel = _relative_asymmetry(latest.joint_velocity, LEFT_LEG_SLICE, RIGHT_LEG_SLICE)
        arm_asym_vel = _relative_asymmetry(latest.joint_velocity, LEFT_ARM_SLICE, RIGHT_ARM_SLICE)
        if latest.joint_effort is not None:
            leg_asym_effort = _relative_asymmetry(
                latest.joint_effort, LEFT_LEG_SLICE, RIGHT_LEG_SLICE
            )
            arm_asym_effort = _relative_asymmetry(
                latest.joint_effort, LEFT_ARM_SLICE, RIGHT_ARM_SLICE
            )
            leg_asymmetry = 0.5 * (leg_asym_vel + leg_asym_effort)
            arm_asymmetry = 0.5 * (arm_asym_vel + arm_asym_effort)
        else:
            leg_asymmetry = leg_asym_vel
            arm_asymmetry = arm_asym_vel
        # Legs weighted far higher than arms: leg asymmetry is directly
        # mobility/fall-relevant, arm asymmetry much less so.
        asymmetry_score = _clip01(0.75 * leg_asymmetry + 0.25 * arm_asymmetry)

        slip_proxy_score = _slip_proxy(
            latest, roll_std, pitch_std, angular_velocity_std, joint_effort_residual
        )

        return Features(
            age_seconds=age_seconds,
            is_stale=is_stale,
            has_nan=has_nan,
            is_out_of_order=is_out_of_order,
            missing_optional_fields=missing_optional,
            producer_invalid=producer_invalid,
            sample_count=len(buf),
            roll=roll,
            pitch=pitch,
            angular_velocity_magnitude=angular_velocity_magnitude,
            roll_std=roll_std,
            pitch_std=pitch_std,
            angular_velocity_std=angular_velocity_std,
            joint_velocity_residual=joint_velocity_residual,
            joint_effort_residual=joint_effort_residual,
            leg_asymmetry=_clip01(leg_asymmetry),
            arm_asymmetry=_clip01(arm_asymmetry),
            asymmetry_score=asymmetry_score,
            slip_proxy_score=slip_proxy_score,
        )
