"""Synchronized deterministic-guard context for dataset episodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sherpaos.battery.guard import BatteryMarginGuard
from sherpaos.contracts import GuardReport, RobotTelemetry
from sherpaos.estimator.features import FeatureWindow
from sherpaos.geography.guard import GeographicRiskGuard
from sherpaos.geography.terrain import RouteData, build_mission_context
from sherpaos.policy.guards import MotionGuardSuite
from sherpaos.sim.weather import wind_speed_at_step


@dataclass(frozen=True, slots=True)
class EpisodeContext:
    """Per-control-step inputs and reports excluded from learned observations."""

    arrays: dict[str, np.ndarray]


def build_episode_context(
    telemetry: list[RobotTelemetry],
    route: RouteData,
    *,
    route_fraction: float = 0.0,
    wind_mps: float = 5.0,
) -> EpisodeContext:
    """Replay guards 3–5 without allowing their decisions to alter the rollout."""
    motion = MotionGuardSuite(FeatureWindow())
    battery = BatteryMarginGuard()
    geography = GeographicRiskGuard()
    rows: dict[str, list[object]] = {
        "control_step": [],
        "timestamp_s": [],
        "sequence": [],
        "telemetry_valid": [],
        "battery_fraction": [],
        "battery_voltage_v": [],
        "battery_current_a": [],
        "battery_temperature_c": [],
        "latitude": [],
        "longitude": [],
        "elevation_m": [],
        "terrain_slope_deg": [],
        "distance_to_safety_m": [],
        "localization_valid": [],
        "wind_mps": [],
        "ambient_temperature_c": [],
        "telemetry_score": [],
        "telemetry_confidence": [],
        "telemetry_action": [],
        "battery_score": [],
        "battery_confidence": [],
        "battery_action": [],
        "geographic_score": [],
        "geographic_confidence": [],
        "geographic_action": [],
    }
    count = max(1, len(telemetry) - 1)
    route_span = route.max_cumulative_distance_m - route.min_cumulative_distance_m
    route_start = route.min_cumulative_distance_m + np.clip(route_fraction, 0.0, 1.0) * route_span
    route_progress = min(500.0, route.max_cumulative_distance_m - route_start)
    for step, sample in enumerate(telemetry):
        now = float(sample.monotonic_time)
        current_wind_mps = wind_speed_at_step(step, wind_mps)
        distance = route_start + step / count * route_progress
        context = build_mission_context(
            route,
            now,
            cumulative_distance_m=distance,
            wind_mps=current_wind_mps,
            temperature_c=sample.battery_temperature_c,
        )
        reports = motion.observe(sample, now)
        telemetry_report = reports[2]
        battery_report = battery.observe(sample, now)
        geographic_report = geography.evaluate(context, now)
        values = {
            "control_step": step,
            "timestamp_s": now,
            "sequence": sample.sequence,
            "telemetry_valid": sample.valid,
            "battery_fraction": sample.battery_fraction,
            "battery_voltage_v": sample.battery_voltage,
            "battery_current_a": sample.battery_current_a,
            "battery_temperature_c": sample.battery_temperature_c,
            "latitude": context.latitude,
            "longitude": context.longitude,
            "elevation_m": context.elevation_m,
            "terrain_slope_deg": context.slope_deg,
            "distance_to_safety_m": context.distance_to_safe_waypoint_m,
            "localization_valid": context.valid,
            "wind_mps": context.wind_mps,
            "ambient_temperature_c": context.temperature_c,
        }
        _add_report(values, "telemetry", telemetry_report)
        _add_report(values, "battery", battery_report)
        _add_report(values, "geographic", geographic_report)
        for key, value in values.items():
            rows[key].append(np.nan if value is None else value)
    arrays = {
        key: np.asarray(values, dtype=str if key.endswith("_action") else None)
        for key, values in rows.items()
    }
    return EpisodeContext(arrays)


def _add_report(values: dict[str, object], prefix: str, report: GuardReport) -> None:
    values[f"{prefix}_score"] = report.score
    values[f"{prefix}_confidence"] = report.confidence
    values[f"{prefix}_action"] = report.recommended_action.value
