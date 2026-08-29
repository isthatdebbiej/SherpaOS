"""One-time offline prep: build configs/terrain/ebc_route.json.

Not part of the runtime — this script is run once (with network access) to
produce a small, pinned, offline route artifact. The runtime (geographic-risk
guard / MissionContext loader) only ever reads the generated JSON file and
must not make network calls. Re-run this script only if the route/provenance
needs to change; commit the resulting JSON.

Source of the waypoint coordinates and published elevations: English
Wikipedia articles for each named location on the standard Everest Base Camp
(EBC) trek, Khumbu region, Nepal. See configs/terrain/PROVENANCE.md for the
exact per-waypoint source URL and the fetch date. This is a small, hand-pinned
waypoint/route artifact (not a gridded DEM raster) sized for a bounded demo
route, consistent with docs/plan.md's "small open Himalayan DEM/terrain
artifact" requirement without pulling in a heavyweight GIS toolchain.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "configs" / "terrain" / "ebc_route.json"

# name, latitude (deg), longitude (deg), elevation (m), is_safe_waypoint
# Coordinates/elevations as published on English Wikipedia (see PROVENANCE.md).
WAYPOINTS = [
    ("Lukla", 27.68889, 86.73056, 2860.0, True),
    ("Phakding", 27.75000, 86.71700, 2610.0, True),
    ("Namche Bazaar", 27.81700, 86.71700, 3440.0, True),
    ("Tengboche", 27.83611, 86.76389, 3860.0, False),
    ("Dingboche", 27.88300, 86.81700, 4410.0, False),
    ("Lobuche", 27.94806, 86.81028, 4940.0, False),
    ("Gorak Shep", 27.98056, 86.82861, 5164.0, False),
    ("Everest Base Camp", 28.00722, 86.85944, 5364.0, False),
]

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def exposure_class(elevation_m: float, slope_deg: float | None) -> str:
    """Bounded heuristic proxy, not a validated avalanche/exposure model.

    Combines altitude (thinner air, colder, less rescue margin) and local
    trail slope into a coarse four-level class. Documented here rather than
    left implicit so it can be challenged/tuned without spelunking code.
    """
    slope = slope_deg or 0.0
    if elevation_m >= 5300 or slope > 20:
        return "SEVERE"
    if elevation_m >= 4500 and slope > 5:
        return "HIGH"
    if elevation_m >= 3500 or slope > 10:
        return "MODERATE"
    return "LOW"


def build() -> dict:
    segments = []
    cumulative_m = 0.0
    for i, (name, lat, lon, elev, is_safe) in enumerate(WAYPOINTS):
        if i == 0:
            slope_deg_from_prev = None
            dist_from_prev_m = 0.0
        else:
            _, plat, plon, pelev, _ = WAYPOINTS[i - 1]
            dist_from_prev_m = haversine_m(plat, plon, lat, lon)
            rise = elev - pelev
            slope_deg_from_prev = math.degrees(math.atan2(rise, max(dist_from_prev_m, 1e-6)))
        cumulative_m += dist_from_prev_m
        segments.append(
            {
                "name": name,
                "latitude": lat,
                "longitude": lon,
                "elevation_m": elev,
                "is_safe_waypoint": is_safe,
                "route_segment": f"seg_{i:02d}_{name.lower().replace(' ', '_')}",
                "cumulative_distance_m": round(cumulative_m, 1),
                "distance_from_prev_m": round(dist_from_prev_m, 1),
                "slope_deg_from_prev": None
                if slope_deg_from_prev is None
                else round(slope_deg_from_prev, 2),
            }
        )

    # distance to the nearest safe waypoint at or before this point on the route
    last_safe_cumulative = None
    for seg in segments:
        if seg["is_safe_waypoint"]:
            last_safe_cumulative = seg["cumulative_distance_m"]
        seg["distance_to_safe_waypoint_m"] = (
            0.0
            if last_safe_cumulative is None
            else round(seg["cumulative_distance_m"] - last_safe_cumulative, 1)
        )
        seg["exposure_class"] = exposure_class(seg["elevation_m"], seg["slope_deg_from_prev"])

    return {
        "terrain_source": "Everest Base Camp (EBC) trek waypoints, Khumbu region, Nepal",
        "terrain_version": "2026-08-29",
        "coordinate_reference_system": "WGS84 (EPSG:4326)",
        "resolution_m": None,
        "resolution_note": (
            "Waypoint/route artifact (named-location samples along one trekking "
            "route), not a gridded DEM raster -- there is no single grid "
            "resolution. Slope is computed between consecutive named waypoints, "
            "not from a continuous elevation surface."
        ),
        "provenance": (
            "Coordinates and published elevations from English Wikipedia "
            "articles for each named waypoint, fetched 2026-08-29. See "
            "configs/terrain/PROVENANCE.md for per-waypoint source URLs. "
            "Distances are great-circle (haversine) between named waypoints, "
            "not actual trail distance (the real trail is longer due to "
            "switchbacks) -- treat distance_to_safe_waypoint_m as a lower "
            "bound / relative-risk signal, not a calibrated hiking distance. "
            "Exposure classification is a bounded heuristic proxy defined in "
            "scripts/prepare_terrain_artifact.py, not a validated avalanche/"
            "exposure model."
        ),
        "safe_waypoint_names": [w[0] for w in WAYPOINTS if w[4]],
        "waypoints": segments,
    }


if __name__ == "__main__":
    artifact = build()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({len(artifact['waypoints'])} waypoints)")
