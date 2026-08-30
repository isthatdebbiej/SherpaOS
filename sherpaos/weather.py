"""Best-effort external weather lookup for display-only telemetry context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass(frozen=True, slots=True)
class WeatherSnapshot:
    available: bool
    source: str
    observed_at: str | None = None
    fetched_at_utc: str | None = None
    temperature_c: float | None = None
    apparent_temperature_c: float | None = None
    relative_humidity_pct: float | None = None
    wind_speed_kmh: float | None = None
    model_elevation_m: float | None = None
    error: str | None = None


def fetch_open_meteo_current_weather(
    latitude: float,
    longitude: float,
    *,
    timeout_s: float = 3.0,
) -> WeatherSnapshot:
    """Fetch one current-weather snapshot; never raise on provider failures."""
    query = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m"
            ),
            "timezone": "auto",
        }
    )
    try:
        url = f"https://api.open-meteo.com/v1/forecast?{query}"
        with urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        current = payload["current"]
        return WeatherSnapshot(
            available=True,
            source="Open-Meteo",
            observed_at=str(current["time"]),
            fetched_at_utc=datetime.now(UTC).isoformat(),
            temperature_c=float(current["temperature_2m"]),
            apparent_temperature_c=float(current["apparent_temperature"]),
            relative_humidity_pct=float(current["relative_humidity_2m"]),
            wind_speed_kmh=float(current["wind_speed_10m"]),
            model_elevation_m=float(payload["elevation"]),
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return WeatherSnapshot(available=False, source="Open-Meteo", error=str(exc))