import numpy as np

from sherpaos.sim.himalaya_scene import scene_xml
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


def test_scene_has_full_g1_and_no_occluding_mountains() -> None:
    xml = scene_xml()
    assert '<include file="g1.xml"/>' in xml
    assert 'name="steep_boundary"' in xml and 'name="cross_slope"' in xml
    assert "mountain" not in xml.lower()
    np.testing.assert_allclose(projected_gravity(np.array([1, 0, 0, 0])), [0, 0, -1])
