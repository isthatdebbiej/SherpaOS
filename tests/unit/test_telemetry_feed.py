from __future__ import annotations

import json
from urllib.request import urlopen

import numpy as np

from derived_state import DerivedState
from sherpaos.contracts import RobotTelemetry
from sherpaos.geography.terrain import build_mission_context, load_route
from sherpaos.weather import WeatherSnapshot
from telemetry_feed import TelemetryFeed


def _telemetry(*, sequence: int = 0, effort: bool = True) -> RobotTelemetry:
    return RobotTelemetry(
        monotonic_time=sequence * 0.02,
        source_time=sequence * 0.02,
        sequence=sequence,
        joint_position=np.zeros(12),
        joint_velocity=np.full(12, 0.1),
        joint_effort=np.full(12, 2.0) if effort else None,
        base_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        base_angular_velocity=np.array([0.1, 0.2, 0.3]),
        base_linear_acceleration=np.array([0.0, 0.0, 9.81]),
    )


def test_feed_aggregates_contract_telemetry_and_preserves_unavailable_fields():
    feed = TelemetryFeed()
    himalaya = build_mission_context(load_route(), now=0.0, waypoint_name="Lobuche")
    feed.push(
        _telemetry(),
        command_speed_mps=0.5,
        ambient_c=20.0,
        mission_context=himalaya,
        location_name="Lobuche",
        weather=WeatherSnapshot(
            available=True,
            source="Open-Meteo",
            observed_at="2026-08-29T12:00",
            temperature_c=2.0,
            apparent_temperature_c=1.0,
            relative_humidity_pct=93.0,
            wind_speed_kmh=3.1,
            model_elevation_m=5315.0,
        ),
        simulated_terrain_slope_deg=4.17,
    )
    snapshot = feed.snapshot()

    assert snapshot["ready"] is True
    assert snapshot["joints"]["count"] == 12
    assert snapshot["motion"]["cmd_vel_m_s"] == 0.5
    assert snapshot["environment"]["ambient_c"] == 20.0
    assert snapshot["environment"]["himalaya"]["available"] is True
    assert snapshot["environment"]["himalaya"]["waypoint"] == "Lobuche"
    assert snapshot["environment"]["himalaya"]["elevation_m"] == 4940.0
    assert snapshot["environment"]["weather"]["temperature_c"] == 2.0
    assert snapshot["environment"]["weather"]["wind_speed_kmh"] == 3.1
    assert snapshot["environment"]["terrain_simulation"]["uphill_slope_deg"] == 4.17
    assert snapshot["decision_context"]["locomotion"]["actual_speed_m_s"] is None
    assert "foot_contact_and_load" in snapshot["decision_context"]["data_gaps"]
    assert snapshot["gait"]["left_contact"] is None
    assert snapshot["gait"]["foot_load_left_n"] is None
    assert "NOT electrical draw" in feed.as_llm_text()


def test_feed_tolerates_unavailable_optional_effort_telemetry():
    feed = TelemetryFeed()
    feed.push(_telemetry(effort=False))
    snapshot = feed.snapshot()

    assert snapshot["power"]["mechanical_w"] is None
    assert snapshot["joints"]["peak_torque_nm"] is None


def test_feed_publishes_explicitly_modelled_battery_and_range():
    feed = TelemetryFeed(
        range_estimator=DerivedState(pack_wh=100.0, initial_battery_fraction=0.5)
    )
    simulated_auxiliary = {
        "actual_speed_mps": 0.45,
        "actual_speed_source": "simulated:mujoco_base_velocity",
        "contacts": {"left": True, "right": False},
        "contacts_source": "simulated:mujoco_contact_force",
        "forces": {"left": 200.0, "right": 0.0},
        "electrical_power_w": 200.0,
        "electrical_power_source": "simulated:walking_pack_model",
        "battery_gauge": {
            "fraction": 0.5,
            "voltage_v": 46.0,
            "current_a": 4.3,
            "temperature_c": 0.0,
        },
        "battery_gauge_source": "simulated:walking_pack_model",
    }
    feed.push(
        _telemetry(sequence=0),
        command_speed_mps=0.5,
        ambient_c=0.0,
        range_speed_mps=0.5,
        **simulated_auxiliary,
    )
    feed.push(
        _telemetry(sequence=50),
        command_speed_mps=0.5,
        ambient_c=0.0,
        range_speed_mps=0.5,
        **simulated_auxiliary,
    )

    snapshot = feed.snapshot()
    model = snapshot["battery"]["range_model"]
    context = snapshot["decision_context"]

    assert model["available"] is True
    assert model["source"] == "modelled:joint-work-plus-idle-load"
    assert model["initial_charge_fraction"] == 0.5
    assert model["state_of_charge_fraction"] < 0.5
    assert model["estimated_range_remaining_m"] > 0.0
    assert snapshot["battery"]["gauge"]["source"] == "simulated:walking_pack_model"
    assert context["locomotion"]["actual_speed_m_s"] == 0.45
    assert context["locomotion"]["speed_tracking_ratio"] == 0.9
    assert context["data_gaps"] == []
    assert "raw_battery_gauge" in context["simulation_only_fields"]
    assert "battery model" in feed.as_llm_text()
    assert "simulated only" in feed.as_llm_text()
    assert "data gaps" not in feed.as_llm_text()


def test_feed_writes_atomically_and_serves_json_and_llm(tmp_path):
    feed = TelemetryFeed()
    feed.push(_telemetry(), command_speed_mps=0.5)
    output = tmp_path / "telemetry.json"
    feed.write(output)
    assert json.loads(output.read_text())["ready"] is True

    server = feed.serve(port=0)
    port = server.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/telemetry") as response:
            assert json.loads(response.read())["ready"] is True
        with urlopen(f"http://127.0.0.1:{port}/llm") as response:
            assert "G1 TELEMETRY" in response.read().decode()
    finally:
        server.shutdown()
        server.server_close()