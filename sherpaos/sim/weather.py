"""Deterministic Himalayan wind physics shared by rollouts and replays."""

from __future__ import annotations

import math

HIMALAYAN_AIR_DENSITY_KG_M3 = 0.70
G1_DRAG_COEFFICIENT = 1.10
G1_FRONTAL_AREA_M2 = 0.60
EXTREME_STORM_MPS = 50.0
EPISODE_STEPS = 500


def wind_speed_at_step(control_step: int, target_mps: float) -> float:
    """Return a temporally smooth wind trace for one independent episode.

    Ordinary regimes begin at 70% of their target and evolve with a
    smoothstep trend plus a low-frequency, bounded gust. Extreme regimes
    begin already inside a sustained storm; they never fake a 200 km/h
    onset inside a ten-second rollout.
    """
    target = max(0.0, float(target_mps))
    if target == 0.0:
        return 0.0
    if target >= EXTREME_STORM_MPS:
        return target
    phase = min(1.0, max(0.0, control_step / (EPISODE_STEPS - 1)))
    smooth = phase * phase * (3.0 - 2.0 * phase)
    initial = 0.70 * target
    trend = initial + smooth * (target - initial)
    gust = 0.03 * target * math.sin(control_step * 0.010)
    return max(0.0, trend + gust)


def aerodynamic_force_n(wind_speed_mps: float) -> float:
    """Lateral drag force from F=0.5*rho*Cd*A*v^2."""
    speed = max(0.0, float(wind_speed_mps))
    return 0.5 * HIMALAYAN_AIR_DENSITY_KG_M3 * G1_DRAG_COEFFICIENT * G1_FRONTAL_AREA_M2 * speed**2
