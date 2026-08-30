from __future__ import annotations

from sherpaos.weather import fetch_open_meteo_current_weather


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.body


def test_open_meteo_weather_extracts_current_conditions(monkeypatch):
    payload = b"""{
        "elevation": 5315,
        "current": {
            "time": "2026-08-29T12:00",
            "temperature_2m": 2.0,
            "apparent_temperature": 1.0,
            "relative_humidity_2m": 93,
            "wind_speed_10m": 3.1
        }
    }"""
    monkeypatch.setattr("sherpaos.weather.urlopen", lambda *_args, **_kwargs: _Response(payload))

    weather = fetch_open_meteo_current_weather(28.00722, 86.85944)

    assert weather.available is True
    assert weather.temperature_c == 2.0
    assert weather.apparent_temperature_c == 1.0
    assert weather.wind_speed_kmh == 3.1


def test_open_meteo_weather_failure_is_explicit(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr("sherpaos.weather.urlopen", unavailable)

    weather = fetch_open_meteo_current_weather(28.00722, 86.85944)

    assert weather.available is False
    assert weather.source == "Open-Meteo"
    assert weather.error == "network unavailable"
