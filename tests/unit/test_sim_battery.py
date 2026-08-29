from __future__ import annotations

import numpy as np

from sherpaos.contracts import RobotTelemetry, TelemetrySource
from sherpaos.sim.battery import enrich_battery_telemetry
from sherpaos.sim.scenario import nominal_scenario


def raw_telemetry() -> RobotTelemetry:
    return RobotTelemetry(
        monotonic_time=10.0,
        source_time=10.0,
        sequence=1,
        joint_position=np.zeros(29),
        joint_velocity=np.zeros(29),
        joint_effort=np.zeros(29),
        base_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        base_angular_velocity=np.zeros(3),
        base_linear_acceleration=np.array([0.0, 0.0, 9.81]),
        source=TelemetrySource.SIM,
    )


def test_synthetic_battery_is_deterministic_and_explicitly_provenanced():
    scenario = nominal_scenario(seed=1)
    first = enrich_battery_telemetry(
        raw_telemetry(), scenario, elapsed_seconds=10.0, speed_scale=1.0
    )
    second = enrich_battery_telemetry(
        raw_telemetry(), scenario, elapsed_seconds=10.0, speed_scale=1.0
    )
    assert first.battery_fraction == second.battery_fraction
    assert first.battery_voltage == second.battery_voltage
    assert first.battery_current_a == second.battery_current_a
    assert first.field_provenance["battery_fraction"].startswith("simulated:")


def test_motion_load_increases_current_and_voltage_sag():
    scenario = nominal_scenario(seed=1)
    idle = enrich_battery_telemetry(
        raw_telemetry(), scenario, elapsed_seconds=1.0, speed_scale=0.0
    )
    moving = enrich_battery_telemetry(
        raw_telemetry(), scenario, elapsed_seconds=1.0, speed_scale=1.0
    )
    assert moving.battery_current_a > idle.battery_current_a
    assert moving.battery_voltage < idle.battery_voltage
