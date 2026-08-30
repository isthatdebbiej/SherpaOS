"""Deterministic Himalayan wind physics shared by rollouts and replays."""

from __future__ import annotations

import math

HIMALAYAN_AIR_DENSITY_KG_M3 = 0.70
G1_DRAG_COEFFICIENT = 1.10
G1_FRONTAL_AREA_M2 = 0.60
EXTREME_STORM_MPS = 50.0
EPISODE_STEPS = 500
HAZARD_ONSET_STEP = 125


def wind_speed_at_step(control_step: int, target_mps: float) -> float:
    """Return a temporally smooth wind trace for one independent episode.

    The rollout begins with a stable but non-zero Himalayan baseline. After
    2.5 seconds the wind evolves continuously toward the target, giving the
    risk model a causal pre-hazard observation period without an unphysical
    one-step jump. Extreme targets therefore represent a storm front, not an
    instantaneous change from city wind to 200 km/h.
    """
    target = max(0.0, float(target_mps))
    if target == 0.0:
        return 0.0
    initial = min(0.70 * target, 8.0)
    phase = min(
        1.0,
        max(0.0, (control_step - HAZARD_ONSET_STEP) / (EPISODE_STEPS - HAZARD_ONSET_STEP - 1)),
    )
    smooth = phase * phase * (3.0 - 2.0 * phase)
    trend = initial + smooth * (target - initial)
    gust = 0.03 * target * math.sin(control_step * 0.010)
    return max(0.0, trend + gust)


def aerodynamic_force_n(wind_speed_mps: float) -> float:
    """Lateral drag force from F=0.5*rho*Cd*A*v^2."""
    speed = max(0.0, float(wind_speed_mps))
    return 0.5 * HIMALAYAN_AIR_DENSITY_KG_M3 * G1_DRAG_COEFFICIENT * G1_FRONTAL_AREA_M2 * speed**2
