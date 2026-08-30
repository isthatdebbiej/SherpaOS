"""Telemetry feed -- makes G1 sensor data easy for other people to consume.

Scope: this module PROVIDES NUMBERS. It makes no decisions. No GO/TURN BACK,
no thresholds, no policy. Whoever owns the decision layer reads from here.

Three ways to consume it, pick whichever suits you:

  1. In-process        feed.snapshot()        -> dict
  2. A file on disk    feed.write("t.json")   -> any language, no coupling
  3. HTTP              feed.serve(port=8088) -> GET /telemetry

Plus one specifically for LLM prompts:

     feed.as_llm_text()  -> ~15 readable lines with units

Why that last one exists: raw telemetry is 50 samples/second across 29 joints.
Putting that in a prompt is thousands of useless tokens. as_llm_text()
aggregates a time window into something a model can actually reason about.

    feed = TelemetryFeed(window_s=1.0)
    ...each control tick:
        feed.push(telemetry, command_speed_mps=0.5)
    ...whenever the consumer wants it:
        feed.write("telemetry.json")
"""

import json
import math
import os
import tempfile
import threading
from collections import deque

import numpy as np

from derived_state import DerivedState
from sherpaos.contracts import MissionContext, RobotTelemetry
from sherpaos.weather import WeatherSnapshot

SCHEMA_VERSION = "1.0"


def _f(x):
    """JSON-safe float (NaN and numpy types break json.dump)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(v) or np.isinf(v)) else round(v, 4)


def _rpy_from_quaternion(quaternion_wxyz):
    """Convert a finite wxyz quaternion to roll, pitch, and yaw in radians."""
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        return np.full(3, np.nan, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-8:
        return np.full(3, np.nan, dtype=float)
    w, x, y, z = quaternion / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=float)


def _himalaya_snapshot(context: MissionContext | None, location_name: str | None) -> dict:
    """Serialize fixed, offline route context without inventing terrain data."""
    if context is None:
        return {"available": False, "waypoint": location_name}
    return {
        "available": context.valid,
        "waypoint": location_name,
        "latitude_deg": _f(context.latitude),
        "longitude_deg": _f(context.longitude),
        "elevation_m": _f(context.elevation_m),
        "slope_deg": _f(context.slope_deg),
        "distance_to_safe_waypoint_m": _f(context.distance_to_safe_waypoint_m),
        "exposure_class": context.exposure_class,
        "route_segment": context.route_segment,
        "terrain_source": context.terrain_source,
        "terrain_version": context.terrain_version,
        "provenance": context.provenance,
    }


def _weather_snapshot(weather: WeatherSnapshot | None) -> dict:
    """Serialize external weather separately from safety-critical telemetry."""
    if weather is None:
        return {"available": False, "source": None}
    return {
        "available": weather.available,
        "source": weather.source,
        "observed_at": weather.observed_at,
        "fetched_at_utc": weather.fetched_at_utc,
        "temperature_c": _f(weather.temperature_c),
        "apparent_temperature_c": _f(weather.apparent_temperature_c),
        "relative_humidity_pct": _f(weather.relative_humidity_pct),
        "wind_speed_kmh": _f(weather.wind_speed_kmh),
        "model_elevation_m": _f(weather.model_elevation_m),
        "error": weather.error,
    }


def _battery_snapshot(record: dict) -> dict:
    """Keep raw gauge data distinct from the display-only range model."""
    estimate = record["battery_estimate"]
    gauge = {
        "source": record["battery_gauge_source"],
        "fraction": _f(record["battery_fraction"]),
        "voltage_v": _f(record["battery_voltage"]),
        "current_a": _f(record["battery_current_a"]),
        "temperature_c": _f(record["battery_temperature_c"]),
    }
    if estimate is None:
        return {
            "gauge": gauge,
            "range_model": {"available": False, "source": None},
        }
    return {
        "gauge": gauge,
        "range_model": {
            "available": True,
            "source": "modelled:joint-work-plus-idle-load",
            "speed_basis": "commanded_velocity",
            "speed_assumption_m_s": _f(record["range_speed_mps"]),
            "initial_charge_fraction": _f(estimate["initial_battery_fraction"]),
            "state_of_charge_fraction": _f(estimate["soc"]),
            "nominal_capacity_wh": _f(estimate["nominal_capacity_wh"]),
            "cold_capacity_fraction": _f(estimate["cold_capacity_frac"]),
            "usable_capacity_wh": _f(estimate["usable_capacity_wh"]),
            "remaining_energy_wh": _f(estimate["remaining_energy_wh"]),
            "energy_used_wh": _f(estimate["energy_wh_used"]),
            "estimated_electrical_w": _f(estimate["power_w"]),
            "estimated_electrical_mean_w": _f(estimate["mean_power_w"]),
            "estimated_range_remaining_m": _f(estimate["range_remaining_m"]),
            "estimated_endurance_s": _f(estimate["endurance_s"]),
        },
    }


def _decision_context(record: dict, battery: dict, tilt_deg: float, tilt_max_deg: float) -> dict:
    """Expose interpretable evidence and gaps without making a safety decision."""
    has_contacts = record["lc"] is not None and record["rc"] is not None
    has_gauge = any(
        value is not None for name, value in battery["gauge"].items() if name != "source"
    )
    has_electrical_power = record["power_w"] is not None
    has_range_model = battery["range_model"]["available"]
    data_gaps = []
    if record["actual_speed_mps"] is None:
        data_gaps.append("actual_base_speed_m_s")
    if not has_contacts:
        data_gaps.append("foot_contact_and_load")
    if not has_gauge:
        data_gaps.append("raw_battery_gauge")
    if not has_electrical_power:
        data_gaps.append("validated_electrical_power")
    if not has_range_model:
        data_gaps.append("battery_range_estimate")
    simulated_inputs = [
        field
        for field, source in (
            ("actual_base_speed_m_s", record["actual_speed_source"]),
            ("foot_contact_and_load", record["contacts_source"]),
            ("raw_battery_gauge", record["battery_gauge_source"]),
            ("validated_electrical_power", record["electrical_power_source"]),
        )
        if source is not None and source.startswith("simulated:")
    ]
    return {
        "telemetry_valid": record["telemetry_valid"],
        "stability": {
            "current_tilt_deg": _f(tilt_deg),
            "peak_tilt_window_deg": _f(tilt_max_deg),
        },
        "locomotion": {
            "commanded_speed_m_s": _f(record["cmd_vel"]),
            "actual_speed_m_s": _f(record["actual_speed_mps"]),
            "speed_tracking_ratio": _f(
                record["actual_speed_mps"] / record["cmd_vel"]
                if record["actual_speed_mps"] is not None and record["cmd_vel"] not in (None, 0.0)
                else None
            ),
            "gait_evidence_available": has_contacts,
        },
        "energy": {
            "range_model_available": has_range_model,
            "raw_gauge_available": has_gauge,
            "validated_electrical_power_available": has_electrical_power,
        },
        "data_gaps": data_gaps,
        "simulation_only_fields": simulated_inputs,
    }


class TelemetryFeed:
    """Rolling window of G1 telemetry, exposed in several shapes."""

    def __init__(self, window_s=1.0, rate_hz=50.0, range_estimator: DerivedState | None = None):
        self.window_s = window_s
        self.n = max(1, int(window_s * rate_hz))
        self._lock = threading.Lock()
        self._buf = deque(maxlen=self.n)
        self._latest = None
        self._energy_wh = 0.0
        self._distance_m = 0.0
        self._t_prev = None
        self._strikes = deque(maxlen=16)
        self._prev_lc = False
        self._range_estimator = range_estimator

    # ------------------------------------------------------------------
    def push(
        self,
        telemetry: RobotTelemetry,
        *,
        command_speed_mps=None,
        contacts=None,
        forces=None,
        electrical_power_w=None,
        ambient_c=None,
        mission_context: MissionContext | None = None,
        location_name: str | None = None,
        weather: WeatherSnapshot | None = None,
        range_speed_mps: float | None = None,
        actual_speed_mps: float | None = None,
        actual_speed_source: str | None = None,
        contacts_source: str | None = None,
        battery_gauge: dict[str, float] | None = None,
        battery_gauge_source: str | None = None,
        electrical_power_source: str | None = None,
        simulated_terrain_slope_deg: float | None = None,
    ):
        """Call once per control tick.

        ``telemetry`` must be a :class:`RobotTelemetry` sample. Optional foot
        contacts/loads and actual speed are intentionally not inferred from
        simulator state: omitted values are published as ``null``.
        """
        if not isinstance(telemetry, RobotTelemetry):
            raise TypeError("telemetry must be a RobotTelemetry sample")
        tau = (
            np.full(telemetry.joint_velocity.shape, np.nan, dtype=float)
            if telemetry.joint_effort is None
            else np.asarray(telemetry.joint_effort, dtype=float)
        )
        dq = np.asarray(telemetry.joint_velocity, dtype=float)
        mech_w = float(np.abs(tau * dq).sum()) if np.all(np.isfinite(tau)) else None

        if electrical_power_w is None:
            voltage = telemetry.battery_voltage
            current = telemetry.battery_current_a
            if voltage is not None and current is not None:
                electrical_power_w = float(voltage) * float(current)

        lc = bool(contacts["left"]) if contacts else None
        rc = bool(contacts["right"]) if contacts else None
        rpy = _rpy_from_quaternion(telemetry.base_orientation)
        battery_estimate = None
        if (
            self._range_estimator is not None
            and range_speed_mps is not None
            and ambient_c is not None
            and np.all(np.isfinite(tau))
            and np.all(np.isfinite(dq))
            and np.all(np.isfinite(rpy))
        ):
            battery_estimate = self._range_estimator.update(
                float(telemetry.monotonic_time),
                tau,
                dq,
                rpy,
                (0.0, 0.0),
                (False, False),
                float(command_speed_mps or 0.0),
                float(range_speed_mps),
                ambient_c=float(ambient_c),
            )

        rec = {
            "t": float(telemetry.monotonic_time),
            "quat": np.asarray(telemetry.base_orientation, dtype=float),
            "rpy": rpy,
            "gyro": np.asarray(telemetry.base_angular_velocity, dtype=float),
            "acc": np.asarray(telemetry.base_linear_acceleration, dtype=float),
            "q": np.asarray(telemetry.joint_position, dtype=float),
            "dq": dq,
            "tau": tau,
            "names": [f"joint_{index}" for index in range(dq.size)],
            "mech_w": mech_w,
            "power_w": electrical_power_w,
            "electrical_power_source": electrical_power_source,
            "cmd_vel": command_speed_mps,
            "lc": lc, "rc": rc,
            "fl": (forces or {}).get("left"),
            "fr": (forces or {}).get("right"),
            "contacts_source": contacts_source,
            "actual_speed_mps": actual_speed_mps,
            "actual_speed_source": actual_speed_source,
            "ambient_c": ambient_c,
            "mission_context": mission_context,
            "location_name": location_name,
            "weather": weather,
            "range_speed_mps": range_speed_mps,
            "battery_estimate": battery_estimate,
            "battery_fraction": (battery_gauge or {}).get(
                "fraction", telemetry.battery_fraction
            ),
            "battery_voltage": (battery_gauge or {}).get("voltage_v", telemetry.battery_voltage),
            "battery_current_a": (battery_gauge or {}).get(
                "current_a", telemetry.battery_current_a
            ),
            "battery_temperature_c": (battery_gauge or {}).get(
                "temperature_c", telemetry.battery_temperature_c
            ),
            "battery_gauge_source": battery_gauge_source,
            "simulated_terrain_slope_deg": simulated_terrain_slope_deg,
            "telemetry_valid": telemetry.valid,
        }
        with self._lock:
            dt = 0.0 if self._t_prev is None else max(0.0, rec["t"] - self._t_prev)
            self._t_prev = rec["t"]
            consumed_power = electrical_power_w if electrical_power_w is not None else mech_w
            if consumed_power is not None and math.isfinite(consumed_power):
                self._energy_wh += consumed_power * dt / 3600.0
            if lc and not self._prev_lc:
                self._strikes.append(rec["t"])
            self._prev_lc = bool(lc)
            self._buf.append(rec)
            self._latest = rec

    # ------------------------------------------------------------------
    def snapshot(self):
        """Aggregated window as a plain dict. Full fidelity, JSON-safe."""
        with self._lock:
            buf = list(self._buf)
            last = self._latest
        if not buf:
            return {"schema": SCHEMA_VERSION, "ready": False}

        rpy = np.array([b["rpy"] for b in buf])
        gyro = np.array([b["gyro"] for b in buf])
        acc = np.array([b["acc"] for b in buf])
        tau = np.array([b["tau"] for b in buf])
        mech = np.array([np.nan if b["mech_w"] is None else b["mech_w"] for b in buf])
        pw = [b["power_w"] for b in buf if b["power_w"] is not None]
        tilt = np.degrees(np.hypot(rpy[:, 0], rpy[:, 1]))

        lc = [b["lc"] for b in buf if b["lc"] is not None]
        rc = [b["rc"] for b in buf if b["rc"] is not None]
        gait_hz = None
        if len(self._strikes) >= 3:
            iv = np.diff(np.asarray(self._strikes))
            if iv.mean() > 0:
                gait_hz = float(1.0 / iv.mean())

        names = last["names"]
        has_effort = bool(np.any(np.isfinite(tau)))
        peak_i = int(np.nanargmax(np.abs(tau).max(axis=0))) if has_effort else None
        battery = _battery_snapshot(last)
        decision_context = _decision_context(last, battery, tilt[-1], tilt.max())

        return {
            "schema": SCHEMA_VERSION,
            "ready": True,
            "t": _f(last["t"]),
            "window_s": self.window_s,
            "samples": len(buf),

            "orientation": {
                "roll_deg": _f(np.degrees(rpy[-1, 0])),
                "pitch_deg": _f(np.degrees(rpy[-1, 1])),
                "yaw_deg": _f(np.degrees(rpy[-1, 2])),
                "tilt_deg": _f(tilt[-1]),
                "tilt_max_deg": _f(tilt.max()),
                "quaternion_wxyz": [_f(v) for v in last["quat"]],
            },
            "motion": {
                "gyro_rms_rad_s": _f(np.sqrt((gyro ** 2).sum(axis=1)).mean()),
                "accel_rms_m_s2": _f(np.sqrt((acc ** 2).sum(axis=1)).mean()),
                "accel_z_var": _f(acc[:, 2].var()),
                "cmd_vel_m_s": _f(last["cmd_vel"]),
            },
            "gait": {
                "cadence_hz": _f(gait_hz),
                "left_contact": last["lc"],
                "right_contact": last["rc"],
                "single_support_frac": _f(
                    np.mean([a != b for a, b in zip(lc, rc, strict=True)])
                )
                if lc and rc else None,
                "foot_load_left_n": _f(last["fl"]),
                "foot_load_right_n": _f(last["fr"]),
                "source": last["contacts_source"],
            },
            "power": {
                "mechanical_w": _f(mech[-1]),
                "mechanical_mean_w": _f(np.nanmean(mech)) if has_effort else None,
                "electrical_w": _f(pw[-1]) if pw else None,
                "electrical_mean_w": _f(np.mean(pw)) if pw else None,
                "electrical_source": last["electrical_power_source"],
                "energy_wh_cumulative": _f(self._energy_wh),
            },
            "battery": battery,
            "decision_context": decision_context,
            "joints": {
                "count": len(names),
                "peak_torque_nm": _f(np.nanmax(np.abs(tau))) if has_effort else None,
                "peak_torque_joint": names[peak_i] if peak_i is not None else None,
                "mean_abs_torque_nm": _f(np.nanmean(np.abs(tau))) if has_effort else None,
            },
            "environment": {
                "ambient_c": _f(last["ambient_c"]),
                "terrain_simulation": {
                    "uphill_slope_deg": _f(last["simulated_terrain_slope_deg"]),
                    "source": (
                        "simulated:mujoco_rotated_floor"
                        if last["simulated_terrain_slope_deg"] is not None
                        else None
                    ),
                },
                "himalaya": _himalaya_snapshot(
                    last["mission_context"], last["location_name"]
                ),
                "weather": _weather_snapshot(last["weather"]),
            },
        }

    # ------------------------------------------------------------------
    def as_json(self, indent=None):
        return json.dumps(self.snapshot(), indent=indent)

    def write(self, path, indent=1):
        """Atomic write, so a reader never sees a half-written file."""
        d = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.snapshot(), f, indent=indent)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ------------------------------------------------------------------
    def as_llm_text(self):
        """Compact, unit-labelled block suitable for dropping into a prompt.

        Roughly 150 tokens instead of the ~5000 a raw window would cost.
        """
        s = self.snapshot()
        if not s.get("ready"):
            return "G1 TELEMETRY: no samples yet."
        o, m, g, p = s["orientation"], s["motion"], s["gait"], s["power"]
        context = s["decision_context"]
        L = []
        L.append(f"G1 TELEMETRY  t={s['t']}s  window={s['window_s']}s "
                 f"({s['samples']} samples)")
        cv = m["cmd_vel_m_s"]
        L.append(f"commanded speed : {cv} m/s" if cv is not None
                 else "commanded speed : unknown")
        L.append(f"posture         : tilt {o['tilt_deg']} deg "
                 f"(roll {o['roll_deg']}, pitch {o['pitch_deg']}, "
                 f"yaw {o['yaw_deg']}); peak tilt this window "
                 f"{o['tilt_max_deg']} deg")
        L.append(f"imu             : accel {m['accel_rms_m_s2']} m/s2 rms, "
                 f"gyro {m['gyro_rms_rad_s']} rad/s rms, "
                 f"vertical accel variance {m['accel_z_var']}")
        if g["cadence_hz"] is not None:
            L.append(f"gait            : cadence {g['cadence_hz']} Hz, "
                     f"single-support {g['single_support_frac']}")
        foot = []
        if g["left_contact"] is not None:
            foot.append("L " + ("down" if g["left_contact"] else "swing"))
            foot.append("R " + ("down" if g["right_contact"] else "swing"))
        if g["foot_load_left_n"] is not None:
            foot.append(f"load {g['foot_load_left_n']}/{g['foot_load_right_n']} N")
        if foot:
            L.append("feet            : " + ", ".join(foot))
        pw = p["electrical_mean_w"]
        if pw is not None:
            L.append(f"power           : {pw} W electrical (mean over window), "
                     f"{p['energy_wh_cumulative']} Wh used so far")
        else:
            L.append(f"power           : {p['mechanical_mean_w']} W mechanical "
                     f"at the joints (NOT electrical draw), "
                     f"{p['energy_wh_cumulative']} Wh used so far")
        battery_model = s["battery"]["range_model"]
        if battery_model["available"]:
            L.append(
                f"battery model   : {battery_model['state_of_charge_fraction'] * 100:.1f}% "
                f"estimated charge, {battery_model['estimated_range_remaining_m']} m range, "
                f"{battery_model['estimated_electrical_mean_w']} W estimated electrical"
            )
        L.append(
            "decision context: stability tilt "
            f"{context['stability']['current_tilt_deg']} deg now / "
            f"{context['stability']['peak_tilt_window_deg']} deg peak; "
            "gait evidence "
            f"{'available' if context['locomotion']['gait_evidence_available'] else 'unavailable'}"
        )
        if context["data_gaps"]:
            L.append("data gaps       : " + ", ".join(context["data_gaps"]))
        if context["simulation_only_fields"]:
            L.append("simulated only  : " + ", ".join(context["simulation_only_fields"]))
        amb = s["environment"]["ambient_c"]
        if amb is not None:
            L.append(f"environment     : ambient {amb} C")
        terrain = s["environment"]["terrain_simulation"]
        if terrain["uphill_slope_deg"] is not None:
            L.append(
                f"terrain          : uphill {terrain['uphill_slope_deg']} deg "
                "(MuJoCo simulated)"
            )
        weather = s["environment"]["weather"]
        if weather["available"]:
            L.append(
                f"weather          : {weather['temperature_c']} C, feels like "
                f"{weather['apparent_temperature_c']} C, wind {weather['wind_speed_kmh']} km/h "
                f"(Open-Meteo at {weather['observed_at']})"
            )
        himalaya = s["environment"]["himalaya"]
        if himalaya["available"]:
            L.append(
                f"himalaya        : {himalaya['waypoint']} at "
                f"{himalaya['elevation_m']} m, slope {himalaya['slope_deg']} deg, "
                f"exposure {himalaya['exposure_class']}"
            )
        return "\n".join(L)

    # ------------------------------------------------------------------
    def serve(self, port=8088, host="127.0.0.1"):
        """Background HTTP server. GET /telemetry (JSON), /llm (text)."""
        from http.server import BaseHTTPRequestHandler, HTTPServer
        feed = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/llm":
                    body = feed.as_llm_text().encode()
                    ctype = "text/plain; charset=utf-8"
                elif self.path == "/telemetry":
                    body = feed.as_json(indent=1).encode()
                    ctype = "application/json"
                else:
                    self.send_error(404, "not found")
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = HTTPServer((host, port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"[telemetry] http://{host}:{port}/telemetry  and  /llm")
        return srv
