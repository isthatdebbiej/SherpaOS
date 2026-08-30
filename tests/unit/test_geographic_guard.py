"""Unit tests for sherpaos.geography (terrain.py + guard.py).

Most cases construct `MissionContext` directly (no filesystem I/O) so the
guard's scoring/threshold logic can be exercised in isolation and
deterministically. A smaller set of tests goes through `terrain.py`'s real
loader against the actual `configs/terrain/ebc_route.json` artifact to
prove the loader and the guard integrate correctly end to end. No import of
`sherpaos.sim`, `sherpaos.estimator`, `sherpaos.policy`, or `mujoco`.
"""

from __future__ import annotations

import math

import pytest

from sherpaos.contracts import GuardAction, GuardName, MissionContext, ReasonCode
from sherpaos.geography import terrain
from sherpaos.geography.guard import (
    CRITICAL_COLD_THRESHOLD_C,
    EXTREME_SLOPE_THRESHOLD_DEG,
    EXTREME_WIND_THRESHOLD_MPS,
    FAR_FROM_SAFE_WAYPOINT_THRESHOLD_M,
    MAX_AGE_SECONDS,
    STEEP_SLOPE_THRESHOLD_DEG,
    GeographicRiskGuard,
    evaluate_route_position,
)

NOW = 1_000.0


def make_context(**overrides: object) -> MissionContext:
    """Build a valid, self-consistent synthetic MissionContext for tests."""
    defaults: dict[str, object] = {
        "latitude": 27.68889,
        "longitude": 86.73056,
        "elevation_m": 2800.0,
        "slope_deg": 1.0,
        "route_segment": "seg_00_test",
        "distance_to_safe_waypoint_m": 50.0,
        "exposure_class": "LOW",
        "terrain_source": "test-route",
        "terrain_version": "test-1",
        "coordinate_reference_system": "WGS84 (EPSG:4326)",
        "lookup_timestamp": NOW,
        "valid": True,
        "resolution_m": None,
        "provenance": "synthetic test fixture",
        "wind_mps": None,
        "temperature_c": None,
    }
    defaults.update(overrides)
    return MissionContext(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Nominal case (direct MissionContext construction).


def test_nominal_low_elevation_low_slope_near_safe_waypoint_passes():
    guard = GeographicRiskGuard()
    ctx = make_context()  # low elevation, gentle slope, 50m from a safe waypoint

    report = guard.evaluate(ctx, now=NOW)

    assert report.guard == GuardName.GEOGRAPHIC
    assert report.recommended_action == GuardAction.PASS
    assert report.confidence > 0.5  # non-trivial confidence
    assert ReasonCode.NOMINAL in report.reason_codes
    # no elevated geographic reason codes
    elevated = {
        ReasonCode.GEOGRAPHIC_STEEP_SLOPE,
        ReasonCode.GEOGRAPHIC_HIGH_EXPOSURE,
        ReasonCode.GEOGRAPHIC_FAR_FROM_SAFE_WAYPOINT,
        ReasonCode.GEOGRAPHIC_CONTEXT_UNAVAILABLE,
        ReasonCode.GEOGRAPHIC_CONTEXT_STALE,
    }
    assert not (elevated & set(report.reason_codes))
    assert 0.0 <= report.score <= 1.0
    assert 0.0 <= report.confidence <= 1.0


# ---------------------------------------------------------------------
# Integration: real loader against the actual artifact.


def test_real_route_everest_base_camp_scores_meaningfully_higher_than_lukla():
    route = terrain.load_route()  # default path: configs/terrain/ebc_route.json
    guard = GeographicRiskGuard()

    lukla_ctx = terrain.build_mission_context(route, now=NOW, waypoint_name="Lukla")
    ebc_ctx = terrain.build_mission_context(route, now=NOW, waypoint_name="Everest Base Camp")

    assert lukla_ctx.valid is True
    assert ebc_ctx.valid is True

    lukla_report = guard.evaluate(lukla_ctx, now=NOW)
    ebc_report = guard.evaluate(ebc_ctx, now=NOW)

    # Lukla is a safe waypoint at low elevation/gentle grade -> nominal.
    assert lukla_report.recommended_action == GuardAction.PASS
    assert ReasonCode.NOMINAL in lukla_report.reason_codes

    # Everest Base Camp is marked exposure_class "SEVERE" in the artifact
    # and is the farthest point from any safe waypoint on the route.
    assert ebc_report.score > lukla_report.score + 0.3  # meaningfully higher
    assert ReasonCode.GEOGRAPHIC_HIGH_EXPOSURE in ebc_report.reason_codes
    assert ReasonCode.GEOGRAPHIC_FAR_FROM_SAFE_WAYPOINT in ebc_report.reason_codes
    assert ebc_report.recommended_action == GuardAction.REQUEST_HOLD

    # provenance names the route/version and the resolved waypoint/segment.
    assert ebc_report.provenance["terrain_source"] == route.terrain_source
    assert ebc_report.provenance["terrain_version"] == route.terrain_version
    assert ebc_report.provenance["route_segment"] == "seg_07_everest_base_camp"

    for report in (lukla_report, ebc_report):
        assert 0.0 <= report.score <= 1.0
        assert 0.0 <= report.confidence <= 1.0


def test_real_route_waypoint_index_and_name_agree():
    route = terrain.load_route()
    by_name = terrain.build_mission_context(route, now=NOW, waypoint_name="Namche Bazaar")
    by_index = terrain.build_mission_context(route, now=NOW, waypoint_index=2)
    assert by_name == by_index


def test_real_route_interpolated_position_between_waypoints():
    route = terrain.load_route()
    lukla = route.by_name("Lukla")
    phakding = route.by_name("Phakding")
    midpoint = (lukla.cumulative_distance_m + phakding.cumulative_distance_m) / 2

    ctx = terrain.build_mission_context(route, now=NOW, cumulative_distance_m=midpoint)

    assert ctx.valid is True
    assert ctx.elevation_m is not None
    # Elevation should sit strictly between the two bracketing waypoints.
    lo, hi = sorted((lukla.elevation_m, phakding.elevation_m))
    assert lo < ctx.elevation_m < hi
    # Slope is taken from the segment being traversed (Phakding's segment).
    assert ctx.slope_deg == phakding.slope_deg_from_prev
    # distance_to_safe_waypoint_m is taken from the nearer named waypoint,
    # not interpolated -- both endpoints here are safe waypoints (0.0).
    assert ctx.distance_to_safe_waypoint_m == 0.0


def test_real_route_out_of_range_position_is_invalid():
    route = terrain.load_route()
    beyond_end = route.max_cumulative_distance_m + 5_000.0

    ctx = terrain.build_mission_context(route, now=NOW, cumulative_distance_m=beyond_end)

    assert ctx.valid is False
    assert ctx.latitude is None
    assert ctx.elevation_m is None
    assert ctx.lookup_timestamp == NOW  # still set, per spec

    guard = GeographicRiskGuard()
    report = guard.evaluate(ctx, now=NOW)
    assert ReasonCode.GEOGRAPHIC_CONTEXT_UNAVAILABLE in report.reason_codes
    assert report.recommended_action in (GuardAction.LIMIT_SPEED, GuardAction.REQUEST_HOLD)


def test_evaluate_route_position_end_to_end_on_real_artifact():
    guard = GeographicRiskGuard()
    report = evaluate_route_position(guard, now=NOW, waypoint_name="Lukla")
    assert report.recommended_action == GuardAction.PASS


def test_evaluate_route_position_missing_file_degrades_gracefully(tmp_path):
    guard = GeographicRiskGuard()
    bad_path = tmp_path / "does_not_exist.json"

    report = evaluate_route_position(guard, now=NOW, route_path=bad_path)

    assert report.recommended_action != GuardAction.PASS
    assert ReasonCode.GEOGRAPHIC_CONTEXT_UNAVAILABLE in report.reason_codes
    assert report.confidence < 0.5


# ---------------------------------------------------------------------
# valid=False / unavailable context.


@pytest.mark.parametrize(
    "ctx",
    [
        make_context(valid=False, latitude=None, longitude=None, elevation_m=None,
                     slope_deg=None, route_segment=None, distance_to_safe_waypoint_m=None),
        None,
        make_context(route_segment=None),  # valid=True but a required field is None
    ],
)
def test_unavailable_context_is_never_a_confident_pass(ctx):
    guard = GeographicRiskGuard()
    report = guard.evaluate(ctx, now=NOW)

    assert report.recommended_action != GuardAction.PASS
    assert ReasonCode.GEOGRAPHIC_CONTEXT_UNAVAILABLE in report.reason_codes
    assert report.confidence < 0.3  # low confidence
    assert 0.0 <= report.score <= 1.0
    assert 0.0 <= report.confidence <= 1.0


# ---------------------------------------------------------------------
# Staleness.


def test_stale_context_flags_and_reduces_confidence_vs_fresh():
    guard = GeographicRiskGuard()
    fresh_ctx = make_context(lookup_timestamp=NOW)
    stale_ctx = make_context(lookup_timestamp=NOW - MAX_AGE_SECONDS - 1_000.0)

    fresh_report = guard.evaluate(fresh_ctx, now=NOW)
    stale_report = guard.evaluate(stale_ctx, now=NOW)

    assert ReasonCode.GEOGRAPHIC_CONTEXT_STALE not in fresh_report.reason_codes
    assert ReasonCode.GEOGRAPHIC_CONTEXT_STALE in stale_report.reason_codes
    assert stale_report.confidence < fresh_report.confidence
    assert stale_report.score >= fresh_report.score  # staleness only pushes risk up
    assert 0.0 <= stale_report.score <= 1.0
    assert 0.0 <= stale_report.confidence <= 1.0


def test_context_exactly_at_max_age_is_not_stale():
    guard = GeographicRiskGuard()
    ctx = make_context(lookup_timestamp=NOW - MAX_AGE_SECONDS)
    report = guard.evaluate(ctx, now=NOW)
    assert ReasonCode.GEOGRAPHIC_CONTEXT_STALE not in report.reason_codes


# ---------------------------------------------------------------------
# Individual reason-code triggers and score bounds.


def test_steep_slope_triggers_reason_code_and_raises_score():
    guard = GeographicRiskGuard()
    calm = make_context(slope_deg=2.0)
    steep = make_context(slope_deg=STEEP_SLOPE_THRESHOLD_DEG + 5.0)

    calm_report = guard.evaluate(calm, now=NOW)
    steep_report = guard.evaluate(steep, now=NOW)

    assert ReasonCode.GEOGRAPHIC_STEEP_SLOPE not in calm_report.reason_codes
    assert ReasonCode.GEOGRAPHIC_STEEP_SLOPE in steep_report.reason_codes
    assert steep_report.score > calm_report.score


def test_extreme_slope_forces_request_hold():
    guard = GeographicRiskGuard()
    ctx = make_context(
        slope_deg=EXTREME_SLOPE_THRESHOLD_DEG,
        elevation_m=2800.0,
        distance_to_safe_waypoint_m=10.0,
    )
    report = guard.evaluate(ctx, now=NOW)
    assert report.recommended_action == GuardAction.REQUEST_HOLD


def test_high_exposure_class_triggers_reason_code():
    guard = GeographicRiskGuard()
    # exposure_class comes from MissionContext directly (populated by
    # terrain.py from the artifact), not recomputed from elevation/slope by
    # the guard -- so a HIGH/SEVERE classification must be set explicitly
    # here to exercise this path.
    low = make_context(exposure_class="LOW")
    severe = make_context(
        exposure_class="SEVERE", elevation_m=5400.0, slope_deg=0.0, distance_to_safe_waypoint_m=0.0
    )
    low_report = guard.evaluate(low, now=NOW)
    severe_report = guard.evaluate(severe, now=NOW)
    assert ReasonCode.GEOGRAPHIC_HIGH_EXPOSURE not in low_report.reason_codes
    assert ReasonCode.GEOGRAPHIC_HIGH_EXPOSURE in severe_report.reason_codes
    assert severe_report.score > low_report.score


def test_far_from_safe_waypoint_triggers_reason_code():
    guard = GeographicRiskGuard()
    near = make_context(distance_to_safe_waypoint_m=100.0)
    far = make_context(distance_to_safe_waypoint_m=FAR_FROM_SAFE_WAYPOINT_THRESHOLD_M + 1_000.0)

    near_report = guard.evaluate(near, now=NOW)
    far_report = guard.evaluate(far, now=NOW)

    assert ReasonCode.GEOGRAPHIC_FAR_FROM_SAFE_WAYPOINT not in near_report.reason_codes
    assert ReasonCode.GEOGRAPHIC_FAR_FROM_SAFE_WAYPOINT in far_report.reason_codes
    assert far_report.score > near_report.score


def test_high_wind_is_environmental_risk_and_extreme_wind_holds():
    guard = GeographicRiskGuard()
    calm = guard.evaluate(make_context(wind_mps=4.0), now=NOW)
    extreme = guard.evaluate(
        make_context(wind_mps=EXTREME_WIND_THRESHOLD_MPS), now=NOW
    )

    assert ReasonCode.ENVIRONMENT_HIGH_WIND not in calm.reason_codes
    assert ReasonCode.ENVIRONMENT_HIGH_WIND in extreme.reason_codes
    assert extreme.score > calm.score
    assert extreme.recommended_action == GuardAction.REQUEST_HOLD
    assert extreme.provenance["wind_mps"] == f"{EXTREME_WIND_THRESHOLD_MPS:.2f}"


def test_critical_cold_is_environmental_risk_and_holds():
    report = GeographicRiskGuard().evaluate(
        make_context(temperature_c=CRITICAL_COLD_THRESHOLD_C), now=NOW
    )

    assert ReasonCode.ENVIRONMENT_EXTREME_COLD in report.reason_codes
    assert report.recommended_action == GuardAction.REQUEST_HOLD


@pytest.mark.parametrize("bad_wind", [-1.0, math.nan, math.inf])
def test_malformed_wind_fails_to_unavailable_context(bad_wind):
    report = GeographicRiskGuard().evaluate(make_context(wind_mps=bad_wind), now=NOW)
    assert ReasonCode.GEOGRAPHIC_CONTEXT_UNAVAILABLE in report.reason_codes
    assert report.recommended_action != GuardAction.PASS


@pytest.mark.parametrize(
    "ctx_kwargs",
    [
        {},  # nominal
        {"elevation_m": 5364.0, "slope_deg": 2.7, "distance_to_safe_waypoint_m": 27_994.5},
        {"slope_deg": 45.0},
        {"distance_to_safe_waypoint_m": 50_000.0},
        {"lookup_timestamp": NOW - MAX_AGE_SECONDS - 5_000.0},
        {"valid": False, "latitude": None, "longitude": None, "elevation_m": None,
         "slope_deg": None, "route_segment": None, "distance_to_safe_waypoint_m": None},
    ],
)
def test_score_and_confidence_always_bounded(ctx_kwargs):
    guard = GeographicRiskGuard()
    ctx = make_context(**ctx_kwargs)
    report = guard.evaluate(ctx, now=NOW)
    assert 0.0 <= report.score <= 1.0
    assert 0.0 <= report.confidence <= 1.0
    assert math.isfinite(report.score)
    assert math.isfinite(report.confidence)


# ---------------------------------------------------------------------
# terrain.load_route error handling.


def test_load_route_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(terrain.TerrainLoadError):
        terrain.load_route(missing)


def test_load_route_corrupt_json_raises(tmp_path):
    bad = tmp_path / "corrupt.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(terrain.TerrainLoadError):
        terrain.load_route(bad)


def test_load_route_missing_required_field_raises(tmp_path):
    bad = tmp_path / "incomplete.json"
    bad.write_text('{"terrain_source": "x"}', encoding="utf-8")
    with pytest.raises(terrain.TerrainLoadError):
        terrain.load_route(bad)


def test_load_route_does_not_fabricate_data_on_failure(tmp_path):
    """A load failure must raise, not return some default/empty-but-valid
    RouteData that could be mistaken for real terrain data."""
    missing = tmp_path / "nope.json"
    try:
        terrain.load_route(missing)
        pytest.fail("expected TerrainLoadError")
    except terrain.TerrainLoadError:
        pass
