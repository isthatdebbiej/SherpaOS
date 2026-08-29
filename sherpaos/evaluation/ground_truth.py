"""Evaluator-only simulator ground truth.

Per `docs/CONTRACTS.md`'s Leakage rule and `AGENTS.md` safety constraint
#1: `ScenarioGroundTruth` must never be passed into, imported by, or
reconstructed inside `sherpaos/estimator` or `sherpaos/policy`. It lives
in `sherpaos/evaluation` specifically so it stays on the far side of that
boundary from `sherpaos.contracts.RobotTelemetry`. `tests/unit/
test_leakage.py` statically enforces this for the estimator/policy lanes;
nothing in this module (or `sherpaos/sim`) is exempt from keeping these
two families of objects apart.

`sherpaos/sim/runner.py` builds one `ScenarioGroundTruth` per control step
by reading live MuJoCo `model`/`data` state (true friction/slope come
straight from the driving `Scenario`; `true_unsafe` is derived via
`classify_unsafe` below from *physical* quantities runner.py computes each
step -- tilt-from-vertical, a MuJoCo-contact-confirmed planted-foot slip
speed). This module itself never touches `mujoco.MjModel`/`MjData` -- it
only deals in plain floats/bools, which keeps `classify_unsafe` trivially
unit-testable in isolation and keeps this module import-light for whoever
builds the paired evaluator/baselines in `sherpaos/evaluation` on top of
`sherpaos/sim/runner.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Thresholds below were chosen empirically against the exact G1 "stand +
# weight-shift" controller in sherpaos/sim/controller.py (see the sim-lane
# report for the full sweep data):
#   - Stable nominal/undisturbed episodes stay under ~2-5 deg of tilt
#     from vertical even under a sub-threshold disturbance; a fall (see
#     sherpaos/sim/runner.py's FALL_TILT_DEG) blows straight past ~90 deg
#     once it starts. 20 deg sits well above ordinary sway noise and well
#     below a committed fall, i.e. it is an early-warning band, not a
#     fall detector -- deliberately lower than runner.py's hard-fall
#     tilt threshold so "unsafe" fires *before* "fell" does.
#   - actuator_health below ~0.2-0.3 destabilizes the plain stand pose
#     outright for this controller (insufficient torque authority against
#     gravity); above ~0.4 it is stable indefinitely.
#   - A foot MuJoCo-contact-confirmed as planted (see runner.py) sliding
#     faster than a brisk walking pace is a slip, not tracking noise --
#     a stationary planted foot under this controller reads ~0 m/s.
DEFAULT_TILT_UNSAFE_DEG = 20.0
DEFAULT_ACTUATOR_HEALTH_UNSAFE = 0.3
DEFAULT_FOOT_SLIP_UNSAFE_MPS = 0.15


@dataclass(slots=True, frozen=True)
class ScenarioGroundTruth:
    """Evaluator-only ground truth for one sim step.

    `disturbance_active` mirrors
    `sherpaos.sim.disturbances.disturbance_active_at_step` (true if the
    scenario's external wrench was active during any physics substep of
    this control step). `actuator_health` here is a passthrough of the
    driving `Scenario.actuator_health` (constant for the whole episode
    given the current scenario model, but included per-step so a future
    time-varying degradation model doesn't require a shape change).
    """

    true_friction: float
    true_slope_deg: float
    disturbance_active: bool
    actuator_health: float
    tilt_from_vertical_deg: float
    planted_foot_slip_mps: float
    true_unsafe: bool


def classify_unsafe(
    *,
    tilt_from_vertical_deg: float,
    actuator_health: float,
    planted_foot_slip_mps: float,
    tilt_unsafe_deg: float = DEFAULT_TILT_UNSAFE_DEG,
    actuator_health_unsafe: float = DEFAULT_ACTUATOR_HEALTH_UNSAFE,
    foot_slip_unsafe_mps: float = DEFAULT_FOOT_SLIP_UNSAFE_MPS,
) -> bool:
    """Threshold rule for "unsafe continuation" at one sim step.

    Unsafe if ANY of:
      - body tilt from vertical exceeds `tilt_unsafe_deg` (losing
        balance, ahead of an actual fall),
      - `actuator_health` has degraded below `actuator_health_unsafe`
        (authority loss severe enough that this controller's stand pose
        is no longer statically stable), or
      - a foot confirmed (by ground-truth MuJoCo contact) planted on the
        ground is sliding faster than `foot_slip_unsafe_mps`
        (traction/slip loss).

    This is what the evaluator (built later on top of
    `sherpaos/sim/runner.py`) uses to score "unsafe continuation" against
    the guard's actual decisions -- it is intentionally a simple,
    auditable threshold rule (per AGENTS.md's "prefer deterministic rules"
    principle), not a learned classifier.
    """
    return (
        tilt_from_vertical_deg > tilt_unsafe_deg
        or actuator_health < actuator_health_unsafe
        or planted_foot_slip_mps > foot_slip_unsafe_mps
    )
