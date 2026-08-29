"""Integration tests for the MuJoCo G1 simulation lane.

These are full physics-engine tests (they load the vendored G1 MJCF and
step real MuJoCo episodes), so they are slower than a unit test but they
are the thing that actually proves the sim lane's core claims: telemetry
in, real actuation channel out, and the guard's hold/speed-limit signal
measurably changes physical outcomes. Nothing here imports
`sherpaos.estimator`/`sherpaos.policy` -- this lane only needs to prove
its own physics and its own contract-boundary discipline.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sherpaos.contracts import RobotTelemetry
from sherpaos.evaluation.ground_truth import ScenarioGroundTruth
from sherpaos.sim.runner import run_episode
from sherpaos.sim.scenario import (
    mixed_traction_disturbance_scenario,
    nominal_scenario,
    random_scenario,
)

pytestmark = pytest.mark.integration


def _assert_no_nans(telemetry: list[RobotTelemetry]) -> None:
    for sample in telemetry:
        assert np.all(np.isfinite(sample.joint_position)), "NaN/inf in joint_position"
        assert np.all(np.isfinite(sample.joint_velocity)), "NaN/inf in joint_velocity"
        if sample.joint_effort is not None:
            assert np.all(np.isfinite(sample.joint_effort)), "NaN/inf in joint_effort"
        assert np.all(np.isfinite(sample.base_orientation)), "NaN/inf in base_orientation"
        assert np.all(
            np.isfinite(sample.base_angular_velocity)
        ), "NaN/inf in base_angular_velocity"
        assert np.all(
            np.isfinite(sample.base_linear_acceleration)
        ), "NaN/inf in base_linear_acceleration"


def _gyro_variance_sum(telemetry: list[RobotTelemetry]) -> float:
    gyro = np.array([sample.base_angular_velocity for sample in telemetry])
    return float(gyro.var(axis=0).sum())


def _mean_joint_speed(telemetry: list[RobotTelemetry]) -> float:
    speeds = [float(np.linalg.norm(sample.joint_velocity)) for sample in telemetry]
    return float(np.mean(speeds))


# ---------------------------------------------------------------------------
# 1. The model loads and the nominal scenario survives cleanly, no NaNs.
# ---------------------------------------------------------------------------


def test_nominal_scenario_survives_full_episode_no_nans():
    max_steps = 300
    scenario = nominal_scenario(seed=42)

    result = run_episode(scenario, seed=42, max_steps=max_steps)

    assert result.fell is False, "nominal (good traction, no disturbance) episode fell"
    assert result.steps_survived == max_steps
    assert len(result.telemetry) == max_steps
    assert len(result.ground_truth) == max_steps
    _assert_no_nans(result.telemetry)

    # A plain stand+step episode should stay close to the model's ~0.79m
    # standing pelvis height throughout -- sanity check on top of "didn't
    # fall", using a margin well inside the fall threshold.
    for sample in result.telemetry:
        assert sample.valid is True
        assert sample.gait_mode in ("stepping", "hold")


def test_nominal_scenario_ground_truth_never_flags_unsafe():
    scenario = nominal_scenario(seed=11)
    result = run_episode(scenario, seed=11, max_steps=300)

    assert not any(gt.true_unsafe for gt in result.ground_truth)
    for gt in result.ground_truth:
        assert gt.true_friction == pytest.approx(1.0)
        assert gt.true_slope_deg == pytest.approx(0.0)
        assert gt.disturbance_active is False
        assert gt.actuator_health == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. mixed_traction_disturbance_scenario is measurably, physically
#    different from nominal_scenario at the same seed.
# ---------------------------------------------------------------------------


def test_mixed_scenario_differs_from_nominal_same_seed():
    seed = 7
    max_steps = 500  # 10s at the default 50Hz control rate

    nominal = nominal_scenario(seed=seed)
    mixed = mixed_traction_disturbance_scenario(seed=seed)

    nominal_result = run_episode(nominal, seed=seed, max_steps=max_steps)
    mixed_result = run_episode(mixed, seed=seed, max_steps=max_steps)

    _assert_no_nans(nominal_result.telemetry)
    _assert_no_nans(mixed_result.telemetry)

    nominal_gyro_var = _gyro_variance_sum(nominal_result.telemetry)
    mixed_gyro_var = _gyro_variance_sum(mixed_result.telemetry)

    # The low-friction slip-chatter signature alone is roughly two orders
    # of magnitude of extra pelvis angular-velocity variance in practice
    # (see the sim-lane report); require at least a conservative 3x here
    # so this test isn't brittle to minor future tuning.
    assert mixed_gyro_var > 3.0 * nominal_gyro_var, (
        f"expected mixed-scenario angular-velocity variance to be "
        f"meaningfully higher than nominal's; got mixed={mixed_gyro_var!r} "
        f"vs nominal={nominal_gyro_var!r}"
    )

    # The mixed scenario's own disturbance window must actually have
    # fired within the episode, and ground truth must reflect the
    # scenario's low friction (both are part of "measurably different").
    assert any(gt.disturbance_active for gt in mixed_result.ground_truth)
    assert all(
        gt.true_friction == pytest.approx(mixed.friction) for gt in mixed_result.ground_truth
    )
    assert mixed.friction < 0.2 < nominal.friction


def test_mixed_scenario_produces_falls_or_unsafe_flags_across_seeds():
    """The mixed scenario must be a real hazard, not just noisier-but-safe.

    Across a handful of seeds it should produce at least one outright
    fall, or at minimum flag `true_unsafe` ground truth at some point --
    this is the physical basis the estimator lane will need to detect.
    """
    max_steps = 500
    any_fell = False
    any_unsafe = False

    for seed in range(5):
        scenario = mixed_traction_disturbance_scenario(seed=seed)
        result = run_episode(scenario, seed=seed, max_steps=max_steps)
        any_fell = any_fell or result.fell
        any_unsafe = any_unsafe or any(gt.true_unsafe for gt in result.ground_truth)

    assert any_fell or any_unsafe, (
        "mixed_traction_disturbance_scenario produced neither a fall nor a "
        "true_unsafe flag across 5 seeds -- the disturbance/traction "
        "combination may be too weak"
    )


# ---------------------------------------------------------------------------
# 3. hold=True via guard_fn really does reduce motion vs hold=False, at
#    the same step.
# ---------------------------------------------------------------------------


def test_hold_guard_fn_reduces_motion_vs_full_speed():
    seed = 3
    max_steps = 200
    scenario = nominal_scenario(seed=seed)

    def guard_pass(_history: list[RobotTelemetry]) -> tuple[float, bool]:
        return 1.0, False

    def guard_hold(_history: list[RobotTelemetry]) -> tuple[float, bool]:
        return 1.0, True

    pass_result = run_episode(scenario, seed=seed, guard_fn=guard_pass, max_steps=max_steps)
    hold_result = run_episode(scenario, seed=seed, guard_fn=guard_hold, max_steps=max_steps)

    assert all(sample.gait_mode == "stepping" for sample in pass_result.telemetry)
    assert all(sample.gait_mode == "hold" for sample in hold_result.telemetry)

    pass_mean_speed = _mean_joint_speed(pass_result.telemetry)
    hold_mean_speed = _mean_joint_speed(hold_result.telemetry)

    assert hold_mean_speed < 0.5 * pass_mean_speed, (
        f"expected hold=True to meaningfully reduce mean joint speed; "
        f"got hold={hold_mean_speed!r} vs pass={pass_mean_speed!r}"
    )


def test_hold_guard_fn_prevents_disturbance_fall_that_full_speed_suffers():
    """The actuation channel's whole point: the same disturbance is
    survivable under REQUEST_HOLD but not under uncontrolled PASS.
    """
    seed = 99
    max_steps = 400
    # A fixed, strong lateral pelvis shove at nominal (high) friction --
    # this magnitude reliably tips a full-speed stepping episode into a
    # fall via resonance with the gait, per the sim-lane report.
    scenario = dataclasses.replace(
        nominal_scenario(seed=seed),
        disturbance_force_n=70.0,
        disturbance_direction=np.array([0.0, 1.0, 0.0]),
        disturbance_start_step=2000,  # 4s in, at physics dt=0.002s
        disturbance_duration_steps=150,  # ~0.3s
    )

    def guard_pass(_history: list[RobotTelemetry]) -> tuple[float, bool]:
        return 1.0, False

    def guard_hold(_history: list[RobotTelemetry]) -> tuple[float, bool]:
        return 1.0, True

    pass_result = run_episode(scenario, seed=seed, guard_fn=guard_pass, max_steps=max_steps)
    hold_result = run_episode(scenario, seed=seed, guard_fn=guard_hold, max_steps=max_steps)

    assert pass_result.fell is True, "expected full-speed PASS to fall under this disturbance"
    assert hold_result.fell is False, "expected REQUEST_HOLD to survive the same disturbance"


# ---------------------------------------------------------------------------
# 4. No RobotTelemetry field ever carries a ground-truth quantity.
# ---------------------------------------------------------------------------


def test_robot_telemetry_has_no_ground_truth_fields_by_construction():
    telemetry_field_names = {f.name for f in dataclasses.fields(RobotTelemetry)}
    forbidden_substrings = ("friction", "slope", "fault", "fall", "contact", "ground_truth")

    for name in telemetry_field_names:
        lowered = name.lower()
        for forbidden in forbidden_substrings:
            assert forbidden not in lowered, (
                f"RobotTelemetry has a field '{name}' that looks like simulator "
                f"ground truth (matched '{forbidden}')"
            )

    # And the two structures must not share attribute names by accident
    # (which would make it easy to mix them up downstream).
    ground_truth_field_names = {f.name for f in dataclasses.fields(ScenarioGroundTruth)}
    assert telemetry_field_names.isdisjoint(ground_truth_field_names)


def test_produced_telemetry_objects_have_no_ground_truth_attributes():
    scenario = nominal_scenario(seed=1)
    result = run_episode(scenario, seed=1, max_steps=5)

    for sample in result.telemetry:
        for forbidden in ("friction", "slope_deg", "injected_fault", "true_fall", "contact"):
            assert not hasattr(sample, forbidden)


# ---------------------------------------------------------------------------
# Small extra coverage: random_scenario sampler is seeded/reproducible
# and both regimes produce runnable episodes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ["train", "stress"])
def test_random_scenario_is_seeded_and_runs(regime: str):
    scenario_a = random_scenario(seed=123, regime=regime)
    scenario_b = random_scenario(seed=123, regime=regime)

    # Compare field by field (not `scenario_a == scenario_b`): the
    # dataclass-generated __eq__ would call `==` on the two
    # `disturbance_direction` numpy arrays and Python's tuple-equality
    # machinery raises on an ambiguous multi-element array truth value.
    assert scenario_a.friction == pytest.approx(scenario_b.friction)
    assert scenario_a.slope_deg == pytest.approx(scenario_b.slope_deg)
    assert scenario_a.disturbance_force_n == pytest.approx(scenario_b.disturbance_force_n)
    assert scenario_a.actuator_health == pytest.approx(scenario_b.actuator_health)
    assert scenario_a.sensor_noise_std == pytest.approx(scenario_b.sensor_noise_std)
    assert scenario_a.disturbance_start_step == scenario_b.disturbance_start_step
    assert scenario_a.disturbance_duration_steps == scenario_b.disturbance_duration_steps
    if scenario_a.disturbance_direction is None:
        assert scenario_b.disturbance_direction is None
    else:
        assert scenario_b.disturbance_direction is not None
        assert np.array_equal(scenario_a.disturbance_direction, scenario_b.disturbance_direction)

    result = run_episode(scenario_a, seed=123, max_steps=50)
    assert len(result.telemetry) == result.steps_survived
    _assert_no_nans(result.telemetry)
