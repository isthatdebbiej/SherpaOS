"""Geographic-risk guard: `MissionContext` -> `GuardReport`.

`GeographicRiskGuard.evaluate()` is a pure function of `(MissionContext,
now)` -- it imports only `sherpaos.contracts`, stdlib `math`, and (for the
end-to-end convenience wrapper at the bottom of this module) its sibling
`sherpaos.geography.terrain`. It does not import `sherpaos.sim`,
`sherpaos.estimator`, or `mujoco`: this guard has no business knowing about
robot joints, gait, or simulator internals -- only geography.

Threshold rationale mirrors `sherpaos/estimator/risk.py` and
`sherpaos/policy/state_machine.py`'s convention: simple, explicitly
documented, fixed thresholds (docs/idea.txt section 24: "if simple
threshold wins, use the threshold"), not fit to any dataset -- flagged as
most likely to need retuning once real mountaineering/exposure data
exists.
"""

from __future__ import annotations

import math
from pathlib import Path

from sherpaos.contracts import GuardAction, GuardName, GuardReport, MissionContext, ReasonCode
from sherpaos.geography import terrain

# ---------------------------------------------------------------------
# Context availability.
#
# A context with valid=False (or missing a required field despite claiming
# valid=True) means the guard has *no* terrain awareness for the current
# position. In mountain terrain that is itself a degraded-margin situation
# (unknown slope, unknown exposure, unknown distance to safety) -- not a
# "nothing to worry about" situation. So we report a *moderate*, non-zero
# score and a *low but not zero* confidence, and lean the recommended
# action toward LIMIT_SPEED rather than PASS. See docs/CONTRACTS.md's
# MissionContext section: "must lower the geographic guard's confidence
# accordingly" (not "must report PASS").
UNAVAILABLE_SCORE = 0.45
UNAVAILABLE_CONFIDENCE = 0.15

# Fields that must be populated for a `valid=True` context to actually be
# usable. `slope_deg` is deliberately *excluded* from this list: the route
# artifact's first waypoint on any route legitimately has no "previous
# segment" (see configs/terrain/ebc_route.json's "Lukla" entry, whose
# `slope_deg_from_prev` is `null`), so a real, valid, in-range position can
# have `slope_deg=None` without the context being "unavailable" -- see
# terrain.py's `build_mission_context` docstring. `None` in any field below
# on an otherwise `valid=True` context indicates a producer bug, not a
# legitimate edge case, so it is treated the same as `valid=False`.
_REQUIRED_FIELDS = (
    "latitude",
    "longitude",
    "elevation_m",
    "route_segment",
    "distance_to_safe_waypoint_m",
)

# ---------------------------------------------------------------------
# Staleness.
#
# How old can `lookup_timestamp` be (relative to `now`) before we stop
# trusting that the resolved position still reflects where the robot
# actually is right now. 5 minutes is generous for a walking-pace robot on
# a mountain trail (this route's shortest inter-waypoint leg is ~4km) but
# tight enough that a lookup from an earlier leg of the route can't be
# silently reused for a materially different position.
MAX_AGE_SECONDS = 300.0
# A stale context can't be reported as confidently as a fresh one: floor
# the score (can't read as "confidently safe") and cap confidence (can't
# be trusted as much as a fresh reading), rather than reusing the
# fresh-context score/confidence unchanged.
STALE_SCORE_FLOOR = 0.3
STALE_CONFIDENCE_CEILING = 0.35  # below CONFIDENCE_LOW_THRESHOLD -> forces >= LIMIT_SPEED

# ---------------------------------------------------------------------
# Nominal scoring: slope + exposure + distance-to-safe-waypoint.

# |slope_deg| past this is "steep" for a loaded biped on a mountain trail
# (a sustained >20 degree grade is beyond typical maintained-trail
# switchback grades). Risk contribution saturates (reaches 1.0) at
# SLOPE_FULL_RISK_DEG. Past EXTREME_SLOPE_THRESHOLD_DEG the guard forces
# REQUEST_HOLD outright, regardless of the weighted score.
STEEP_SLOPE_THRESHOLD_DEG = 20.0
SLOPE_FULL_RISK_DEG = 30.0
EXTREME_SLOPE_THRESHOLD_DEG = 30.0

# Exposure classification now comes straight from MissionContext.exposure_class
# (populated by sherpaos.geography.terrain from the artifact's own precomputed
# per-waypoint value, or the nearer waypoint's value when interpolating -- see
# terrain.py's build_mission_context docstring). This used to be recomputed
# here from elevation_m/slope_deg via a duplicated copy of
# scripts/prepare_terrain_artifact.py's heuristic; MissionContext gained a
# dedicated exposure_class field (docs/DECISIONS.md, five-guard architecture
# update) so that duplication was removed -- an unusable/missing
# exposure_class is treated as "LOW" (i.e. contributes no exposure risk)
# only as an explicit, documented fallback for producers that legitimately
# have no classification (e.g. live-G1 telemetry with no terrain lookup at
# all), never silently for this artifact's own waypoints, which always
# populate it.
_EXPOSURE_SCORE = {"LOW": 0.0, "MODERATE": 0.35, "HIGH": 0.7, "SEVERE": 1.0}
_HIGH_EXPOSURE_CLASSES = frozenset({"HIGH", "SEVERE"})

# Distance to the nearest safe waypoint: this artifact's Everest Base Camp
# waypoint is ~28km from the nearest safe waypoint (Namche Bazaar), so a
# 30km scale keeps EBC itself near-but-not-at full risk contribution while
# still discriminating shorter distances earlier on the route. 10km is
# where we start calling it out as a named reason.
FAR_FROM_SAFE_WAYPOINT_THRESHOLD_M = 10_000.0
DISTANCE_FULL_RISK_M = 30_000.0

WEIGHT_SLOPE = 0.35
WEIGHT_EXPOSURE = 0.40
WEIGHT_DISTANCE = 0.25

# Optional environmental context. These are transparent operational
# thresholds, not learned weather predictions. Missing values do not invent
# risk; non-finite values fail through the guard's unavailable fallback.
HIGH_WIND_THRESHOLD_MPS = 15.0
EXTREME_WIND_THRESHOLD_MPS = 25.0
WIND_FULL_RISK_MPS = 30.0
EXTREME_COLD_THRESHOLD_C = -20.0
CRITICAL_COLD_THRESHOLD_C = -30.0
COLD_FULL_RISK_C = -35.0

# Base confidence for a fresh, valid, fully-populated context. Not 1.0:
# configs/terrain/PROVENANCE.md documents this artifact as Wikipedia-
# sourced (not survey-grade) with great-circle, not trail, distances, so
# even a "healthy" lookup carries some irreducible uncertainty.
FRESH_CONFIDENCE = 0.9

# Action thresholds on the combined score.
SCORE_LIMIT_SPEED_THRESHOLD = 0.3
SCORE_REQUEST_HOLD_THRESHOLD = 0.65
CONFIDENCE_LOW_THRESHOLD = 0.4  # confidence below this -> never report PASS

# "Severe combo" override: SEVERE exposure *and* meaningfully far from a
# safe waypoint forces REQUEST_HOLD even if the weighted score lands just
# under SCORE_REQUEST_HOLD_THRESHOLD -- standing at a severe-exposure point
# a long way from safety is exactly the situation this guard exists to
# catch, so it should not depend on landing on the right side of a linear
# blend.
_SEVERE_COMBO_DISTANCE_M = FAR_FROM_SAFE_WAYPOINT_THRESHOLD_M


def _clip01(x: float) -> float:
    """Clip to [0, 1]; a non-finite value becomes 1.0 (maximal risk),
    never silently 0.0 -- mirrors estimator/risk.py's `_clip_score`."""
    if not math.isfinite(x):
        return 1.0
    return min(1.0, max(0.0, x))


def _clip_confidence(x: float) -> float:
    """Clip to [0, 1]; a non-finite value becomes 0.0 (minimal trust),
    never silently 1.0 -- mirrors estimator/risk.py's `_clip_confidence`."""
    if not math.isfinite(x):
        return 0.0
    return min(1.0, max(0.0, x))


def _is_usable(mission_context: MissionContext | None) -> bool:
    if mission_context is None or not mission_context.valid:
        return False
    return all(getattr(mission_context, f) is not None for f in _REQUIRED_FIELDS)


def _unavailable_report(mission_context: MissionContext | None) -> GuardReport:
    provenance: dict[str, str] = {}
    if mission_context is not None:
        provenance["terrain_source"] = mission_context.terrain_source
        provenance["terrain_version"] = mission_context.terrain_version
        provenance["mission_context_provenance"] = mission_context.provenance
    return GuardReport(
        guard=GuardName.GEOGRAPHIC,
        score=UNAVAILABLE_SCORE,
        confidence=UNAVAILABLE_CONFIDENCE,
        reason_codes=(ReasonCode.GEOGRAPHIC_CONTEXT_UNAVAILABLE,),
        recommended_action=GuardAction.LIMIT_SPEED,
        provenance=provenance,
    )


def _action_for(score: float, confidence: float) -> GuardAction:
    if score >= SCORE_REQUEST_HOLD_THRESHOLD:
        return GuardAction.REQUEST_HOLD
    if score >= SCORE_LIMIT_SPEED_THRESHOLD or confidence < CONFIDENCE_LOW_THRESHOLD:
        return GuardAction.LIMIT_SPEED
    return GuardAction.PASS


class GeographicRiskGuard:
    """Stateless: `.evaluate()` is a pure function of `(MissionContext, now)`."""

    def evaluate(self, mission_context: MissionContext | None, now: float) -> GuardReport:
        try:
            return self._evaluate(mission_context, now)
        except Exception:
            # Top-level safety net, mirroring estimator/risk.py and
            # policy/state_machine.py (AGENTS.md constraint 3): a bug here
            # must never crash the runtime, and must never silently read
            # as safe.
            return GuardReport(
                guard=GuardName.GEOGRAPHIC,
                score=UNAVAILABLE_SCORE,
                confidence=0.0,
                reason_codes=(ReasonCode.GEOGRAPHIC_CONTEXT_UNAVAILABLE,),
                recommended_action=GuardAction.LIMIT_SPEED,
                provenance={"error": "geographic guard raised unexpectedly"},
            )

    def _evaluate(self, mission_context: MissionContext | None, now: float) -> GuardReport:
        if not _is_usable(mission_context):
            return _unavailable_report(mission_context)

        mc = mission_context
        assert mc is not None  # _is_usable already ruled out None

        reasons: list[ReasonCode] = []
        confidence = FRESH_CONFIDENCE

        age_seconds = float(now) - float(mc.lookup_timestamp)
        is_stale = (not math.isfinite(age_seconds)) or age_seconds > MAX_AGE_SECONDS

        slope_deg = mc.slope_deg
        slope_component = (
            _clip01(abs(slope_deg) / SLOPE_FULL_RISK_DEG) if slope_deg is not None else 0.0
        )
        if slope_deg is not None and abs(slope_deg) > STEEP_SLOPE_THRESHOLD_DEG:
            reasons.append(ReasonCode.GEOGRAPHIC_STEEP_SLOPE)

        exposure_class = mc.exposure_class or "LOW"
        exposure_component = _EXPOSURE_SCORE.get(exposure_class, 0.0)
        if exposure_class in _HIGH_EXPOSURE_CLASSES:
            reasons.append(ReasonCode.GEOGRAPHIC_HIGH_EXPOSURE)

        distance_m = mc.distance_to_safe_waypoint_m
        assert distance_m is not None  # _is_usable already ruled out None
        distance_component = _clip01(distance_m / DISTANCE_FULL_RISK_M)
        if distance_m > FAR_FROM_SAFE_WAYPOINT_THRESHOLD_M:
            reasons.append(ReasonCode.GEOGRAPHIC_FAR_FROM_SAFE_WAYPOINT)

        score = (
            WEIGHT_SLOPE * slope_component
            + WEIGHT_EXPOSURE * exposure_component
            + WEIGHT_DISTANCE * distance_component
        )

        wind_mps = mc.wind_mps
        temperature_c = mc.temperature_c
        wind_component = 0.0
        cold_component = 0.0
        if wind_mps is not None:
            wind = float(wind_mps)
            if not math.isfinite(wind) or wind < 0.0:
                return _unavailable_report(mc)
            wind_component = _clip01(wind / WIND_FULL_RISK_MPS)
            if wind >= HIGH_WIND_THRESHOLD_MPS:
                reasons.append(ReasonCode.ENVIRONMENT_HIGH_WIND)
        if temperature_c is not None:
            temperature = float(temperature_c)
            if not math.isfinite(temperature):
                return _unavailable_report(mc)
            if temperature <= EXTREME_COLD_THRESHOLD_C:
                reasons.append(ReasonCode.ENVIRONMENT_EXTREME_COLD)
                cold_component = _clip01(
                    (EXTREME_COLD_THRESHOLD_C - temperature)
                    / (EXTREME_COLD_THRESHOLD_C - COLD_FULL_RISK_C)
                )

        # Environmental hazards cannot be diluted by a calm route score.
        environment_component = max(wind_component, cold_component)
        score = max(score, environment_component)

        if is_stale:
            reasons.append(ReasonCode.GEOGRAPHIC_CONTEXT_STALE)
            score = max(score, STALE_SCORE_FLOOR)
            confidence = min(confidence, STALE_CONFIDENCE_CEILING)

        score = _clip01(score)
        confidence = _clip_confidence(confidence)

        action = _action_for(score, confidence)

        severe_combo = exposure_class == "SEVERE" and distance_m >= _SEVERE_COMBO_DISTANCE_M
        extreme_slope = slope_deg is not None and abs(slope_deg) >= EXTREME_SLOPE_THRESHOLD_DEG
        extreme_wind = wind_mps is not None and float(wind_mps) >= EXTREME_WIND_THRESHOLD_MPS
        critical_cold = (
            temperature_c is not None and float(temperature_c) <= CRITICAL_COLD_THRESHOLD_C
        )
        if severe_combo or extreme_slope or extreme_wind or critical_cold:
            action = GuardAction.REQUEST_HOLD

        if not reasons:
            reasons.append(ReasonCode.NOMINAL)

        provenance = {
            "terrain_source": mc.terrain_source,
            "terrain_version": mc.terrain_version,
            "route_segment": mc.route_segment or "",
            "exposure_class": exposure_class,
            "age_seconds": f"{age_seconds:.1f}",
            "wind_mps": "unavailable" if wind_mps is None else f"{float(wind_mps):.2f}",
            "temperature_c": (
                "unavailable" if temperature_c is None else f"{float(temperature_c):.2f}"
            ),
        }

        return GuardReport(
            guard=GuardName.GEOGRAPHIC,
            score=score,
            confidence=confidence,
            reason_codes=tuple(dict.fromkeys(reasons)),
            recommended_action=action,
            provenance=provenance,
        )


def evaluate_route_position(
    guard: GeographicRiskGuard,
    now: float,
    *,
    route_path: Path | None = None,
    waypoint_name: str | None = None,
    waypoint_index: int | None = None,
    cumulative_distance_m: float | None = None,
) -> GuardReport:
    """End-to-end convenience helper: load the route artifact, resolve a
    position, and evaluate it.

    This is the one place in this package that touches the filesystem for
    the runtime path (as opposed to `terrain.load_route`, which is called
    from here and from tests). It catches `terrain.TerrainLoadError` so a
    missing/corrupt artifact degrades to a low-confidence
    `GEOGRAPHIC_CONTEXT_UNAVAILABLE` report instead of propagating an
    exception -- per the task's requirement that a load failure become a
    `GuardReport`, not a crash. `GeographicRiskGuard.evaluate()` itself
    stays a pure function of `MissionContext` throughout.
    """
    try:
        route = terrain.load_route(route_path or terrain.DEFAULT_ROUTE_PATH)
        mission_context = terrain.build_mission_context(
            route,
            now,
            waypoint_name=waypoint_name,
            waypoint_index=waypoint_index,
            cumulative_distance_m=cumulative_distance_m,
        )
    except terrain.TerrainLoadError:
        mission_context = terrain.unavailable_mission_context(now)
    return guard.evaluate(mission_context, now)
