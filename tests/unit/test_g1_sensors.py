from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from sherpaos.sim.g1_sensors import build_sensorized_scene


@pytest.mark.parametrize(
    ("xml_path", "joint_count", "sensor_count"),
    (
        ("third_party/mujoco_menagerie/unitree_g1/scene.xml", 29, 95),
        ("third_party/unitree_rl_gym/resources/robots/g1_description/scene.xml", 12, 41),
    ),
)
def test_sensorized_scene_exposes_observable_low_state(
    xml_path: str, joint_count: int, sensor_count: int
):
    source = Path(xml_path)
    if not source.exists():
        pytest.skip(f"local third-party asset unavailable: {source}")

    scene = build_sensorized_scene(source)
    mujoco.mj_forward(scene.model, scene.data)
    telemetry = scene.suite.low_state(
        sequence=3,
        monotonic_time=1.25,
        commanded_velocity=np.array([0.5, 0.0, 0.0]),
        gait_mode="walking",
    )

    assert scene.model.nsensor == sensor_count
    assert telemetry.valid is True
    assert telemetry.joint_position.shape == (joint_count,)
    assert telemetry.joint_velocity.shape == (joint_count,)
    assert telemetry.joint_effort is not None
    assert telemetry.joint_effort.shape == (joint_count,)
    assert telemetry.base_orientation.shape == (4,)
    assert telemetry.base_angular_velocity.shape == (3,)
    assert telemetry.base_linear_acceleration.shape == (3,)
    assert np.all(np.isfinite(telemetry.base_orientation))
