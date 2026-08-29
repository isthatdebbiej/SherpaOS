"""Deterministic synthetic battery telemetry for MuJoCo episodes.

This is an explicitly simulated signal source, not a calibrated G1 pack
model. It exists to exercise the battery guard with reproducible cold,
load, discharge, and voltage-sag scenarios while preserving provenance.
"""

from __future__ import annotations

import dataclasses

from sherpaos.contracts import RobotTelemetry
from sherpaos.sim.scenario import Scenario


def enrich_battery_telemetry(
    telemetry: RobotTelemetry,
    scenario: Scenario,
    *,
    elapsed_seconds: float,
    speed_scale: float,
) -> RobotTelemetry:
    """Return telemetry carrying deterministic synthetic battery fields."""
    elapsed = max(0.0, float(elapsed_seconds))
    speed = min(1.0, max(0.0, float(speed_scale)))
    fraction = max(
        0.0,
        min(1.0, scenario.battery_initial_fraction - scenario.battery_discharge_per_s * elapsed),
    )
    current_a = scenario.battery_idle_current_a + scenario.battery_motion_current_a * speed
    # Transparent linear reference used only by the simulator. The guard
    # owns its independent expected-voltage curve.
    open_circuit_voltage = 38.0 + 16.0 * fraction
    voltage = max(
        0.0,
        open_circuit_voltage - current_a * scenario.battery_internal_resistance_ohm,
    )
    provenance = dict(telemetry.field_provenance)
    provenance.update(
        {
            "battery_fraction": "simulated:linear_discharge",
            "battery_voltage": "simulated:open_circuit_minus_ir_sag",
            "battery_current_a": "simulated:idle_plus_motion_load",
            "battery_temperature_c": "simulated:scenario_ambient",
        }
    )
    return dataclasses.replace(
        telemetry,
        battery_fraction=fraction,
        battery_voltage=voltage,
        battery_current_a=current_a,
        battery_temperature_c=scenario.battery_temperature_c,
        field_provenance=provenance,
    )
