"""Offline loader + MissionContext builder for the geographic-risk guard's
route artifact (`configs/terrain/ebc_route.json`).

**Module split (documented explicitly so it stays unambiguous):** this
module is responsible for producing a *snapshot* `MissionContext` -- it
decides `valid=False` only for reasons intrinsic to resolving a position
against the route artifact itself (the requested position falls outside the
route's cumulative-distance range). It does **not** decide staleness:
`lookup_timestamp` is always set to the caller-supplied `now` (never
`time.time()`, to keep this deterministic/testable), and it is
`sherpaos.geography.guard.GeographicRiskGuard` that later compares
`lookup_timestamp` against its own "now" at evaluation time and decides
whether the snapshot is too old to act on. This keeps "can we resolve a
position" (here) cleanly separate from "is this resolution still fresh
enough to trust" (guard.py).

`load_route()` raises `TerrainLoadError` on a missing/corrupt artifact
rather than fabricating data -- it is the caller's job (see
`sherpaos.geography.guard.evaluate_route_position`) to turn that failure
into a low-confidence `GuardReport` instead of crashing the runtime
(AGENTS.md safety constraint 3).

The runtime must never query the internet for terrain data -- this module
only ever reads the already-committed local JSON file (see
`configs/terrain/PROVENANCE.md`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from sherpaos.contracts import MissionContext

# sherpaos/sherpaos/geography/terrain.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ROUTE_PATH = REPO_ROOT / "configs" / "terrain" / "ebc_route.json"


class TerrainLoadError(Exception):
    """Raised when the route artifact is missing, unreadable, or malformed.

    Never caught silently here -- callers must decide how to degrade.
    """


@dataclass(slots=True, frozen=True)
class Waypoint:
    """One named waypoint parsed from the route artifact."""

    name: str
    latitude: float
    longitude: float
    elevation_m: float
    is_safe_waypoint: bool
    route_segment: str
    cumulative_distance_m: float
    distance_from_prev_m: float
    slope_deg_from_prev: float | None
    distance_to_safe_waypoint_m: float
    exposure_class: str  # the Waypoint's own field (from the artifact), not MissionContext's


@dataclass(slots=True, frozen=True)
class RouteData:
    """In-memory parse of `configs/terrain/ebc_route.json`."""

    terrain_source: str
    terrain_version: str
    coordinate_reference_system: str
    resolution_m: float | None
    provenance: str
    safe_waypoint_names: tuple[str, ...]
    waypoints: tuple[Waypoint, ...]

    def by_name(self, name: str) -> Waypoint:
        for wp in self.waypoints:
            if wp.name == name:
                return wp
        raise KeyError(f"no waypoint named {name!r} in route {self.terrain_source!r}")

    @property
    def min_cumulative_distance_m(self) -> float:
        return self.waypoints[0].cumulative_distance_m

    @property
    def max_cumulative_distance_m(self) -> float:
        return self.waypoints[-1].cumulative_distance_m


_REQUIRED_TOP_LEVEL_FIELDS = (
    "terrain_source",
    "terrain_version",
    "coordinate_reference_system",
    "resolution_m",
    "provenance",
    "safe_waypoint_names",
    "waypoints",
)
_REQUIRED_WAYPOINT_FIELDS = (
    "name",
    "latitude",
    "longitude",
    "elevation_m",
    "is_safe_waypoint",
    "route_segment",
    "cumulative_distance_m",
    "distance_from_prev_m",
    "slope_deg_from_prev",
    "distance_to_safe_waypoint_m",
    "exposure_class",
)


def load_route(path: Path = DEFAULT_ROUTE_PATH) -> RouteData:
    """Parse the route artifact into a `RouteData`.

    Raises `TerrainLoadError` -- never returns fabricated/partial data --
    if the file is missing, unreadable, not valid JSON, missing an
    expected field, or internally inconsistent (waypoints not sorted by
    `cumulative_distance_m`, which the interpolation below assumes).
    """
    try:
        raw_text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise TerrainLoadError(f"could not read terrain artifact at {path}: {exc}") from exc

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise TerrainLoadError(f"terrain artifact at {path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise TerrainLoadError(
            f"terrain artifact at {path} must be a JSON object, got {type(raw).__name__}"
        )

    for key in _REQUIRED_TOP_LEVEL_FIELDS:
        if key not in raw:
            raise TerrainLoadError(f"terrain artifact at {path} is missing required field {key!r}")

    raw_waypoints = raw["waypoints"]
    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        raise TerrainLoadError(f"terrain artifact at {path} has no waypoints")

    waypoints: list[Waypoint] = []
    for i, wp in enumerate(raw_waypoints):
        if not isinstance(wp, dict):
            raise TerrainLoadError(f"waypoint {i} in {path} is not a JSON object")
        for key in _REQUIRED_WAYPOINT_FIELDS:
            if key not in wp:
                raise TerrainLoadError(
                    f"waypoint {i} in {path} is missing required field {key!r}"
                )
        try:
            waypoints.append(
                Waypoint(
                    name=str(wp["name"]),
                    latitude=float(wp["latitude"]),
                    longitude=float(wp["longitude"]),
                    elevation_m=float(wp["elevation_m"]),
                    is_safe_waypoint=bool(wp["is_safe_waypoint"]),
                    route_segment=str(wp["route_segment"]),
                    cumulative_distance_m=float(wp["cumulative_distance_m"]),
                    distance_from_prev_m=float(wp["distance_from_prev_m"]),
                    slope_deg_from_prev=(
                        None
                        if wp["slope_deg_from_prev"] is None
                        else float(wp["slope_deg_from_prev"])
                    ),
                    distance_to_safe_waypoint_m=float(wp["distance_to_safe_waypoint_m"]),
                    exposure_class=str(wp["exposure_class"]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise TerrainLoadError(f"waypoint {i} in {path} has a malformed field: {exc}") from exc

    distances = [wp.cumulative_distance_m for wp in waypoints]
    if distances != sorted(distances):
        raise TerrainLoadError(
            f"waypoints in {path} are not sorted ascending by cumulative_distance_m"
        )

    try:
        resolution_m = None if raw["resolution_m"] is None else float(raw["resolution_m"])
    except (TypeError, ValueError) as exc:
        raise TerrainLoadError(
            f"terrain artifact at {path} has a malformed resolution_m: {exc}"
        ) from exc

    try:
        safe_waypoint_names = tuple(str(n) for n in raw["safe_waypoint_names"])
    except TypeError as exc:
        raise TerrainLoadError(
            f"terrain artifact at {path} has a malformed safe_waypoint_names: {exc}"
        ) from exc

    return RouteData(
        terrain_source=str(raw["terrain_source"]),
        terrain_version=str(raw["terrain_version"]),
        coordinate_reference_system=str(raw["coordinate_reference_system"]),
        resolution_m=resolution_m,
        provenance=str(raw["provenance"]),
        safe_waypoint_names=safe_waypoint_names,
        waypoints=tuple(waypoints),
    )


def unavailable_mission_context(
    now: float,
    *,
    terrain_source: str = "unknown",
    terrain_version: str = "unknown",
    coordinate_reference_system: str = "unknown",
    resolution_m: float | None = None,
    provenance: str = "terrain context unavailable",
) -> MissionContext:
    """Build an explicitly-invalid `MissionContext` snapshot.

    Used both for a position outside the route's range (below) and by
    `sherpaos.geography.guard.evaluate_route_position` when `load_route`
    itself raises `TerrainLoadError` -- in both cases we represent "no
    usable terrain context" explicitly (`valid=False`, position/derived
    fields `None`) rather than inventing a plausible-looking value.
    """
    return MissionContext(
        latitude=None,
        longitude=None,
        elevation_m=None,
        slope_deg=None,
        route_segment=None,
        distance_to_safe_waypoint_m=None,
        exposure_class=None,
        terrain_source=terrain_source,
        terrain_version=terrain_version,
        coordinate_reference_system=coordinate_reference_system,
        lookup_timestamp=now,
        valid=False,
        resolution_m=resolution_m,
        provenance=provenance,
    )


def _context_from_waypoint(
    route: RouteData,
    wp: Waypoint,
    now: float,
    wind_mps: float | None,
    temperature_c: float | None,
) -> MissionContext:
    return MissionContext(
        latitude=wp.latitude,
        longitude=wp.longitude,
        elevation_m=wp.elevation_m,
        slope_deg=wp.slope_deg_from_prev,
        route_segment=wp.route_segment,
        distance_to_safe_waypoint_m=wp.distance_to_safe_waypoint_m,
        exposure_class=wp.exposure_class,
        terrain_source=route.terrain_source,
        terrain_version=route.terrain_version,
        coordinate_reference_system=route.coordinate_reference_system,
        lookup_timestamp=now,
        valid=True,
        resolution_m=route.resolution_m,
        provenance=f"{route.provenance} [waypoint {wp.name!r}]",
        wind_mps=wind_mps,
        temperature_c=temperature_c,
    )


def _bracketing_waypoints(route: RouteData, x: float) -> tuple[Waypoint, Waypoint]:
    wps = route.waypoints
    for i in range(len(wps) - 1):
        if wps[i].cumulative_distance_m <= x <= wps[i + 1].cumulative_distance_m:
            return wps[i], wps[i + 1]
    # Only reachable for a single-waypoint route; caller already validated
    # x is within [min, max].
    return wps[-1], wps[-1]


def build_mission_context(
    route: RouteData,
    now: float,
    *,
    waypoint_name: str | None = None,
    waypoint_index: int | None = None,
    cumulative_distance_m: float | None = None,
    wind_mps: float | None = None,
    temperature_c: float | None = None,
) -> MissionContext:
    """Build a `MissionContext` snapshot for a position along `route`.

    Exactly one of `waypoint_name`, `waypoint_index`, `cumulative_distance_m`
    must be given to specify the position:

    - `waypoint_name` / `waypoint_index` resolve to that waypoint exactly
      (no interpolation) -- `valid=True` always, since it is a named point
      already in the artifact.
    - `cumulative_distance_m` linearly interpolates `elevation_m` (and
      `latitude`/`longitude`) between the two bracketing waypoints by
      distance along the route. `slope_deg` is **not** interpolated
      between two different segments' slopes -- it is taken from the
      *current segment* (the outgoing segment from the lower bracketing
      waypoint, i.e. the upper waypoint's `slope_deg_from_prev`), because
      slope is only meaningfully defined per-segment in this artifact (see
      `configs/terrain/PROVENANCE.md`): a segment's slope is a single
      number describing that whole stretch of trail, so it applies
      unchanged to every point within it, whereas "interpolating" between
      two adjacent segments' slopes would blend two physically distinct
      pieces of trail into a number that describes neither. Categorical /
      route-derived fields that have no natural interpolation
      (`route_segment`, `distance_to_safe_waypoint_m`, `exposure_class`) are
      taken from whichever bracketing waypoint is nearer by distance, not
      interpolated, per the task's guidance to prefer "nearest waypoint"
      over inventing an interpolated categorical value.

    `lookup_timestamp` is always set to the caller-supplied `now` (never
    wall-clock time) so context construction stays deterministic/testable.
    A `cumulative_distance_m` outside the route's
    `[min_cumulative_distance_m, max_cumulative_distance_m]` range returns
    an explicitly invalid (`valid=False`) context (via
    `unavailable_mission_context`-equivalent fields) rather than
    extrapolating terrain past the edge of the artifact.
    """
    specified = [v is not None for v in (waypoint_name, waypoint_index, cumulative_distance_m)]
    if sum(specified) != 1:
        raise ValueError(
            "build_mission_context requires exactly one of waypoint_name, "
            "waypoint_index, cumulative_distance_m"
        )

    if waypoint_name is not None:
        wp = route.by_name(waypoint_name)
        return _context_from_waypoint(route, wp, now, wind_mps, temperature_c)

    if waypoint_index is not None:
        try:
            wp = route.waypoints[waypoint_index]
        except IndexError as exc:
            raise IndexError(
                f"waypoint_index {waypoint_index} out of range for "
                f"{len(route.waypoints)} waypoints"
            ) from exc
        return _context_from_waypoint(route, wp, now, wind_mps, temperature_c)

    x = float(cumulative_distance_m)  # type: ignore[arg-type]
    if (
        not math.isfinite(x)
        or x < route.min_cumulative_distance_m
        or x > route.max_cumulative_distance_m
    ):
        return MissionContext(
            latitude=None,
            longitude=None,
            elevation_m=None,
            slope_deg=None,
            route_segment=None,
            distance_to_safe_waypoint_m=None,
            exposure_class=None,
            terrain_source=route.terrain_source,
            terrain_version=route.terrain_version,
            coordinate_reference_system=route.coordinate_reference_system,
            lookup_timestamp=now,
            valid=False,
            resolution_m=route.resolution_m,
            provenance=(
                f"{route.provenance} [position {x!r}m is outside route range "
                f"[{route.min_cumulative_distance_m:.1f}, "
                f"{route.max_cumulative_distance_m:.1f}]m]"
            ),
        )

    lower, upper = _bracketing_waypoints(route, x)
    if lower is upper:
        return _context_from_waypoint(route, lower, now, wind_mps, temperature_c)

    span = upper.cumulative_distance_m - lower.cumulative_distance_m
    t = 0.0 if span <= 0 else (x - lower.cumulative_distance_m) / span
    t = min(1.0, max(0.0, t))

    latitude = lower.latitude + t * (upper.latitude - lower.latitude)
    longitude = lower.longitude + t * (upper.longitude - lower.longitude)
    elevation_m = lower.elevation_m + t * (upper.elevation_m - lower.elevation_m)
    slope_deg = upper.slope_deg_from_prev

    nearer = (
        lower
        if (x - lower.cumulative_distance_m) <= (upper.cumulative_distance_m - x)
        else upper
    )

    return MissionContext(
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation_m,
        slope_deg=slope_deg,
        route_segment=nearer.route_segment,
        distance_to_safe_waypoint_m=nearer.distance_to_safe_waypoint_m,
        exposure_class=nearer.exposure_class,
        terrain_source=route.terrain_source,
        terrain_version=route.terrain_version,
        coordinate_reference_system=route.coordinate_reference_system,
        lookup_timestamp=now,
        valid=True,
        resolution_m=route.resolution_m,
        provenance=f"{route.provenance} [interpolated between {lower.name!r} and {upper.name!r}]",
        wind_mps=wind_mps,
        temperature_c=temperature_c,
    )
