"""Shared pytest fixtures for the SherpaOS test suite.

These fixtures build synthetic-but-valid `sherpaos.contracts` objects so
every lane's tests (estimator, policy, recorder, evidence, ...) can
construct realistic telemetry without duplicating boilerplate. Nothing
here may import from `sherpaos.sim` or `sherpaos.evaluation` — fixtures
must stay on the same side of the leakage boundary as the code they
help test.
"""

from __future__ import annotations

import numpy as np
import pytest

from sherpaos.contracts import RobotTelemetry, TelemetrySource

# The real Unitree G1 has 29 actuated joints.
G1_NUM_JOINTS = 29


def _make_telemetry(**overrides: object) -> RobotTelemetry:
    """Build a valid, self-consistent synthetic RobotTelemetry.

    Every field has a sensible default; pass keyword overrides to build
    edge cases, e.g. `make_telemetry(monotonic_time=-1000.0)` for a
    stale sample or `make_telemetry(joint_position=np.full(29, np.nan))`
    for a corrupt one.
    """
    defaults: dict[str, object] = {
        "monotonic_time": 100.0,
        "source_time": 100.0,
        "sequence": 0,
        "joint_position": np.zeros(G1_NUM_JOINTS, dtype=float),
        "joint_velocity": np.zeros(G1_NUM_JOINTS, dtype=float),
        "joint_effort": np.zeros(G1_NUM_JOINTS, dtype=float),
        # Identity quaternion, wxyz order.
        "base_orientation": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        "base_angular_velocity": np.zeros(3, dtype=float),
        "base_linear_acceleration": np.array([0.0, 0.0, 9.81], dtype=float),
        "commanded_velocity": np.zeros(3, dtype=float),
        "gait_mode": "walk",
        "battery_fraction": 1.0,
        "battery_voltage": 48.0,
        "source": TelemetrySource.SIM,
        "valid": True,
        "field_provenance": {},
    }
    defaults.update(overrides)
    return RobotTelemetry(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def make_telemetry():
    """Factory fixture: call `make_telemetry(**overrides)` to build a RobotTelemetry."""
    return _make_telemetry


@pytest.fixture
def seeded_rng() -> np.random.Generator:
    """Deterministic RNG for reproducible randomized test data."""
    return np.random.default_rng(0)
