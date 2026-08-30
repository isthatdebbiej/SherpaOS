"""Observer-only video recorder for the Unitree walking demonstration.

The HUD tells one story: published telemetry (battery/range, weather,
location) feeds a continuously re-evaluated go/no-go check. Everything shown
is read from the same aggregate snapshot the live feed publishes; the verdict
is an advisory display, not the guard loop.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_OK = (96, 222, 140, 255)
_CAUTION = (255, 200, 90, 255)
_STOP = (255, 105, 97, 255)
_TEXT = (235, 247, 255, 255)
_DIM = (170, 205, 228, 255)
_PANEL = (8, 18, 28, 208)

_LEVEL_OK = 0
_LEVEL_CAUTION = 1
_LEVEL_STOP = 2
_LEVEL_COLOR = {_LEVEL_OK: _OK, _LEVEL_CAUTION: _CAUTION, _LEVEL_STOP: _STOP}
_RESERVE_FRACTION = 0.25


class WalkingVideoRecorder:
    """Encode a heading-tracking walk recording with a telemetry-decision HUD."""

    def __init__(self, output: Path, *, width: int = 1280, height: int = 720, fps: int = 25):
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self._renderer: mujoco.Renderer | None = None
        self._writer: Any | None = None
        self._azimuth: float | None = None
        self._fonts: dict[str, ImageFont.ImageFont] = {}

    def start(self, model: mujoco.MjModel) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self._fonts = _load_fonts()
        self._writer = imageio_ffmpeg.write_frames(
            str(self.output),
            (self.width, self.height),
            fps=self.fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            quality=8,
        )
        self._writer.send(None)

    def capture(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        snapshot: Mapping[str, Any],
        gesture: tuple[str, int] | None = None,
    ) -> None:
        if self._renderer is None or self._writer is None:
            self.start(model)
        self._renderer.update_scene(data, camera=self._camera(data))
        frame = self._renderer.render()
        self._writer.send(self._hud(frame, snapshot, gesture).tobytes())

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> WalkingVideoRecorder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- camera -----------------------------------------------------------

    def _camera(self, data: mujoco.MjData) -> mujoco.MjvCamera:
        """Front three-quarter view that follows the robot's actual heading.

        A fixed azimuth makes any yaw drift read as the robot walking
        sideways across the frame; anchoring the azimuth to the pelvis yaw
        keeps the robot facing a consistent screen direction.
        """
        w, x, y, z = (float(v) for v in data.qpos[3:7])
        yaw_deg = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        target = yaw_deg + 150.0
        if self._azimuth is None:
            self._azimuth = target
        else:
            delta = (target - self._azimuth + 180.0) % 360.0 - 180.0
            self._azimuth += 0.04 * delta
        heading = math.radians(yaw_deg)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 3.6
        camera.azimuth = self._azimuth
        camera.elevation = -15.0
        camera.lookat[:] = [
            float(data.qpos[0]) + 0.30 * math.cos(heading),
            float(data.qpos[1]) + 0.30 * math.sin(heading),
            float(data.qpos[2]),
        ]
        return camera

    # -- HUD ----------------------------------------------------------------

    def _hud(
        self,
        frame: np.ndarray,
        snapshot: Mapping[str, Any],
        gesture: tuple[str, int] | None,
    ) -> np.ndarray:
        image = Image.fromarray(frame, mode="RGB")
        draw = ImageDraw.Draw(image, "RGBA")

        battery = _mapping(snapshot.get("battery"))
        gauge = _mapping(battery.get("gauge"))
        range_model = _mapping(battery.get("range_model"))
        environment = _mapping(snapshot.get("environment"))
        weather = _mapping(environment.get("weather"))
        himalaya = _mapping(environment.get("himalaya"))
        terrain = _mapping(environment.get("terrain_simulation"))
        locomotion = _mapping(_mapping(snapshot.get("decision_context")).get("locomotion"))
        orientation = _mapping(snapshot.get("orientation"))

        self._battery_panel(draw, 20, 20, gauge, range_model)
        self._weather_panel(draw, 20, 196, weather, range_model)
        self._location_panel(draw, 20, 352, himalaya, terrain)
        checks, verdict = _evaluate(range_model, weather, himalaya)
        self._decision_panel(draw, 856, 20, checks, verdict)
        if gesture is not None:
            self._signal_panel(draw, 410, 20, gesture)
        self._footer(draw, snapshot, locomotion, orientation)
        return np.asarray(image)

    def _battery_panel(self, draw, x, y, gauge, range_model) -> None:
        self._panel(draw, x, y, 372, 160, "BATTERY & RANGE")
        fraction = _number(gauge.get("fraction"))
        bar_w = 200
        draw.rectangle((x + 16, y + 40, x + 16 + bar_w, y + 56), outline=_DIM, width=1)
        fill = _OK if fraction > 0.4 else (_CAUTION if fraction > 0.15 else _STOP)
        draw.rectangle((x + 18, y + 42, x + 18 + int((bar_w - 4) * fraction), y + 54), fill=fill)
        self._text(draw, x + 226, y + 41, f"{100.0 * fraction:.1f}%", "value")
        cold = _number(range_model.get("cold_capacity_fraction"))
        usable = _number(range_model.get("usable_capacity_wh"))
        range_m = _number(range_model.get("estimated_range_remaining_m"))
        endurance_s = _number(range_model.get("estimated_endurance_s"))
        self._text(draw, x + 16, y + 66, f"USABLE (COLD-DERATED)  {usable:.0f} Wh"
                   f"  ({100.0 * cold:.0f}% OF PACK)", "small", _DIM)
        self._text(draw, x + 16, y + 88, f"RANGE  {range_m / 1000.0:.2f} km", "value")
        self._text(draw, x + 200, y + 88, f"ENDURANCE  {endurance_s / 60.0:.0f} min", "value")
        self._text(draw, x + 16, y + 116, "SOURCE: SIMULATED PACK + MODELLED RANGE",
                   "small", _DIM)
        self._text(draw, x + 16, y + 134, "RANGE MODEL: JOINT WORK + IDLE LOAD", "small", _DIM)

    def _weather_panel(self, draw, x, y, weather, range_model) -> None:
        self._panel(draw, x, y, 372, 140, "WEATHER  (LIVE, DISPLAY-ONLY)")
        if not bool(weather.get("available")):
            self._text(draw, x + 16, y + 44, "NO DATA — TREATED AS CAUTION", "value", _CAUTION)
            return
        temp = _number(weather.get("temperature_c"))
        wind = _number(weather.get("wind_speed_kmh"))
        self._text(draw, x + 16, y + 40,
                   f"{temp:+.1f} C  (FEELS {_number(weather.get('apparent_temperature_c')):+.1f})",
                   "value")
        self._text(draw, x + 16, y + 66, f"WIND {wind:.0f} km/h   "
                   f"HUMIDITY {_number(weather.get('relative_humidity_pct')):.0f}%", "value")
        cold = _number(range_model.get("cold_capacity_fraction"))
        self._text(draw, x + 16, y + 94,
                   f"-> COLD DERATES PACK TO {100.0 * cold:.0f}%", "small", _CAUTION)
        self._text(draw, x + 16, y + 116, f"SOURCE: {weather.get('source', '?')}", "small", _DIM)

    def _location_panel(self, draw, x, y, himalaya, terrain) -> None:
        self._panel(draw, x, y, 372, 172, "LOCATION  (OFFLINE EBC ROUTE)")
        if not bool(himalaya.get("available")):
            self._text(draw, x + 16, y + 44, "NO ROUTE CONTEXT", "value", _CAUTION)
            return
        elevation_m = _number(himalaya.get("elevation_m"))
        self._text(draw, x + 16, y + 40,
                   f"{himalaya.get('waypoint', '?')}   {elevation_m:.0f} m", "value")
        self._text(draw, x + 16, y + 66,
                   f"UPHILL {_number(terrain.get('uphill_slope_deg')):.1f} deg", "value")
        exposure = str(himalaya.get("exposure_class") or "?")
        color = {"LOW": _OK, "MODERATE": _CAUTION}.get(exposure, _STOP)
        self._text(draw, x + 16, y + 92, f"EXPOSURE {exposure}", "value", color)
        distance = _number(himalaya.get("distance_to_safe_waypoint_m"))
        self._text(draw, x + 16, y + 118,
                   f"SAFE WAYPOINT  {distance / 1000.0:.1f} km AWAY", "value")
        self._text(draw, x + 16, y + 146, f"SEGMENT: {himalaya.get('route_segment', '?')}",
                   "small", _DIM)

    def _decision_panel(self, draw, x, y, checks, verdict) -> None:
        self._panel(draw, x, y, 404, 268, "GO / NO-GO CHECK  (EVERY TICK)")
        row_y = y + 42
        for label, level, detail in checks:
            color = _LEVEL_COLOR[level]
            draw.ellipse((x + 16, row_y + 3, x + 30, row_y + 17), fill=color)
            self._text(draw, x + 40, row_y, label, "value")
            self._text(draw, x + 40, row_y + 22, detail, "small", _DIM)
            row_y += 48
        level, title, reason = verdict
        color = _LEVEL_COLOR[level]
        draw.rectangle((x + 12, row_y + 4, x + 392, row_y + 46), fill=color)
        self._text(draw, x + 24, row_y + 12, title, "title", (10, 16, 24, 255))
        self._text(draw, x + 12, row_y + 52, reason, "small", _DIM)

    def _signal_panel(self, draw, x, y, gesture: tuple[str, int]) -> None:
        label, level = gesture
        self._panel(draw, x, y, 430, 90, "ROBOT SIGNAL  (ARM GESTURE)")
        color = _LEVEL_COLOR[level]
        draw.rectangle((x + 12, y + 40, x + 418, y + 78), fill=color)
        self._text(draw, x + 24, y + 50, label, "value", (10, 16, 24, 255))

    def _footer(self, draw, snapshot, locomotion, orientation) -> None:
        top = self.height - 46
        draw.rectangle((20, top, self.width - 20, self.height - 12), fill=_PANEL)
        actual = _number(locomotion.get("actual_speed_m_s"))
        commanded = _number(locomotion.get("commanded_speed_m_s"))
        self._text(draw, 36, top + 10,
                   f"T {_number(snapshot.get('t')):5.1f} s    "
                   f"SPEED {actual:.2f} / {commanded:.2f} m/s    "
                   f"TILT {_number(orientation.get('tilt_deg')):.1f} deg", "value")
        self._text(draw, self.width - 560, top + 12,
                   "SHERPAOS | ADVISORY DISPLAY FROM PUBLISHED TELEMETRY | MUJOCO SIM",
                   "small", _DIM)

    def _panel(self, draw, x, y, w, h, title: str) -> None:
        draw.rectangle((x, y, x + w, y + h), fill=_PANEL)
        self._text(draw, x + 16, y + 12, title, "title")

    def _text(self, draw, x, y, message: str, font_key: str, color=_TEXT) -> None:
        draw.text((x, y), message, font=self._fonts.get(font_key), fill=color)


def _evaluate(
    range_model: Mapping[str, Any],
    weather: Mapping[str, Any],
    himalaya: Mapping[str, Any],
) -> tuple[list[tuple[str, int, str]], tuple[int, str, str]]:
    """Grade the three telemetry families and fuse them conservatively.

    Missing data is graded CAUTION, never OK. The fused verdict is the worst
    individual grade, mirroring the guard fusion's no-averaging rule.
    """
    checks: list[tuple[str, int, str]] = []

    range_m = _number(range_model.get("estimated_range_remaining_m"))
    distance_m = _number(himalaya.get("distance_to_safe_waypoint_m"))
    reason = ""
    if not bool(range_model.get("available")) or not bool(himalaya.get("available")):
        energy = (_LEVEL_CAUTION, "RANGE OR ROUTE DATA MISSING")
    elif distance_m <= 0.0:
        energy = (_LEVEL_OK, "ALREADY AT A SAFE WAYPOINT")
    else:
        needed_m = distance_m / (1.0 - _RESERVE_FRACTION)
        if range_m >= needed_m:
            energy = (
                _LEVEL_OK,
                f"RANGE {range_m / 1000.0:.1f} km >= NEED {needed_m / 1000.0:.1f} km",
            )
        else:
            energy = (
                _LEVEL_STOP,
                f"RANGE {range_m / 1000.0:.1f} km < NEED {needed_m / 1000.0:.1f} km (25% RSV)",
            )
            reason = "CANNOT REACH SAFE WAYPOINT ON REMAINING CHARGE"
    checks.append(("ENERGY REACH", energy[0], energy[1]))

    if not bool(weather.get("available")):
        wx = (_LEVEL_CAUTION, "NO WEATHER DATA")
    else:
        temp = _number(weather.get("temperature_c"))
        wind = _number(weather.get("wind_speed_kmh"))
        if temp <= -15.0 or wind >= 60.0:
            wx = (_LEVEL_STOP, f"{temp:+.0f} C / {wind:.0f} km/h — SEVERE")
            reason = reason or "SEVERE WEATHER"
        elif temp <= -5.0 or wind >= 35.0:
            wx = (_LEVEL_CAUTION, f"{temp:+.0f} C / {wind:.0f} km/h — COLD/WINDY")
        else:
            wx = (_LEVEL_OK, f"{temp:+.0f} C / {wind:.0f} km/h WIND")
    checks.append(("WEATHER", wx[0], wx[1]))

    exposure = str(himalaya.get("exposure_class") or "")
    if not bool(himalaya.get("available")) or not exposure:
        terrain_check = (_LEVEL_CAUTION, "NO EXPOSURE DATA")
    elif exposure == "LOW":
        terrain_check = (_LEVEL_OK, "LOW EXPOSURE SEGMENT")
    elif exposure == "MODERATE":
        terrain_check = (_LEVEL_CAUTION, "MODERATE EXPOSURE SEGMENT")
    else:
        terrain_check = (_LEVEL_STOP, f"{exposure} EXPOSURE SEGMENT")
        reason = reason or f"{exposure} TERRAIN EXPOSURE"
    checks.append(("TERRAIN EXPOSURE", terrain_check[0], terrain_check[1]))

    worst = max(level for _, level, _ in checks)
    if worst == _LEVEL_STOP:
        verdict = (worst, "TURN BACK / HOLD", reason or "A CHECK FAILED")
    elif worst == _LEVEL_CAUTION:
        verdict = (worst, "PROCEED — LIMIT SPEED", "CAUTION ON AT LEAST ONE CHECK")
    else:
        verdict = (worst, "PROCEED", "ALL CHECKS GREEN")
    return checks, verdict


def _load_fonts() -> dict[str, ImageFont.ImageFont]:
    try:
        return {
            "title": ImageFont.truetype("arialbd.ttf", 17),
            "value": ImageFont.truetype("arial.ttf", 16),
            "small": ImageFont.truetype("arial.ttf", 13),
        }
    except OSError:
        default = ImageFont.load_default()
        return {"title": default, "value": default, "small": default}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0
