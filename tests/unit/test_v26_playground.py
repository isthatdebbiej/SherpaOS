import numpy as np

from scripts.render_v26_himalaya import _condition_overlay, _risk_state
from sherpaos.sim.himalaya_scene import (
    TERRAIN_ZONE_LENGTHS_M,
    TERRAIN_ZONE_NAMES,
    TERRAIN_ZONE_PROFILES,
    scene_xml,
    terrain_heightmap,
    terrain_slope_deg_at,
    terrain_slope_for_geom,
    write_terrain_png,
)
from sherpaos.sim.v26_playground import (
    DEFAULT_POSE,
    V26ObservationHistory,
    action_target,
    projected_gravity,
    robot_visibility,
)


def test_v26_observation_contract_and_reset_history() -> None:
    history = V26ObservationHistory()
    observation = history.build(
        DEFAULT_POSE, np.array([1, 0, 0, 0]), np.zeros(3), np.array([0.4, 0, 0, 0])
    )
    assert observation.shape == (1, 240)
    np.testing.assert_allclose(observation[0, :48], observation[0, -48:])


def test_action_target_scales_and_clips() -> None:
    np.testing.assert_allclose(action_target(np.zeros(12)), DEFAULT_POSE)
    assert np.all(np.isfinite(action_target(np.full(12, 100))))


def test_visibility_measurement() -> None:
    segmentation = np.full((20, 30, 2), -1, dtype=np.int32)
    segmentation[4:15, 8:20, 0] = 7
    assert robot_visibility(segmentation, np.array([7])) == (132, 11)


def test_video_condition_overlay_and_forecast_no_go() -> None:
    state = _risk_state(
        forecast_wind_mps=55.6,
        current_wind_mps=8.0,
        slope_deg=4.0,
        slip_mps=0.1,
        tilt_deg=5.0,
    )
    assert state == "NO-GO"
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rendered = _condition_overlay(frame, ["Wind now: 8.0 m/s"], state)
    assert rendered.shape == frame.shape
    assert np.any(rendered != frame)


def test_scene_uses_connected_visible_collision_segments() -> None:
    scenes = [scene_xml(index) for index in range(len(TERRAIN_ZONE_NAMES))]
    assert all('<include file="g1.xml"/>' in xml for xml in scenes)
    assert all('name="spawn_apron" type="box"' in xml for xml in scenes)
    assert all(
        all(f'name="terrain_segment_{index}" type="box"' in xml for index in range(4))
        for xml in scenes
    )
    assert all("ridge_" not in xml for xml in scenes)
    np.testing.assert_allclose(projected_gravity(np.array([1, 0, 0, 0])), [0, 0, -1])
    assert TERRAIN_ZONE_PROFILES[0] == (2.0, 4.0, 5.0, 4.0)
    assert TERRAIN_ZONE_PROFILES[4] == (10.0, 16.0, 22.0, 30.0)
    assert TERRAIN_ZONE_LENGTHS_M[4] == (0.65, 0.90, 1.30, 2.65)
    assert all(abs(sum(lengths) - 4.5) < 1e-9 for lengths in TERRAIN_ZONE_LENGTHS_M[:4])
    assert abs(sum(TERRAIN_ZONE_LENGTHS_M[4]) - 5.5) < 1e-9
    assert terrain_slope_for_geom(4, "terrain_segment_2") == 22.0
    assert terrain_slope_for_geom(4, "spawn_apron") == 0.0


def test_himalayan_heightfields_are_deterministic_distinct_and_spawn_flat(tmp_path) -> None:
    maps = [terrain_heightmap(index) for index in range(len(TERRAIN_ZONE_NAMES))]
    assert all(value.shape == (129, 129) for value in maps)
    assert all(np.isfinite(value).all() and 0 <= value.min() <= value.max() <= 1 for value in maps)
    assert all(np.ptp(value[:, :70]) == 0 for value in maps)
    assert len({value.tobytes() for value in maps}) == len(maps)
    np.testing.assert_array_equal(maps[4], terrain_heightmap(4))
    assert terrain_slope_deg_at(0, 0.0, 0.0) == 0.0
    assert 3.0 <= terrain_slope_deg_at(0, 5.0, 0.0) <= 5.0
    assert 27.0 <= terrain_slope_deg_at(4, 5.0, 0.0) <= 31.0
    output = tmp_path / "terrain.png"
    write_terrain_png(output, 4)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
