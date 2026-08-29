"""Scenario definitions and seeded samplers for the MuJoCo G1 sim lane.

A `Scenario` is a plain, evaluator-*and*-runner-visible bundle of episode
parameters -- it is not itself simulator ground truth (it never reaches
`RobotTelemetry`), it is the *recipe* `sherpaos/sim/disturbances.py` and
`sherpaos/sim/runner.py` use to configure one MuJoCo episode. The
per-step ground truth derived *from* running a scenario (true friction at
this instant, whether a fall/unsafe condition occurred, etc.) is a
separate, evaluator-only structure: `sherpaos.evaluation.ground_truth.
ScenarioGroundTruth`.

All scenario fields are physically grounded against the exact
posture/stepping controller in `sherpaos/sim/controller.py`, empirically
tuned while building this module (see the sim-lane report for the full
sweep data):
  - `friction` below ~0.15 (with `slope_deg=0`, `disturbance_force_n=0`)
    produces a persistent slip-chatter signature -- roughly two orders of
    magnitude higher pelvis angular-velocity variance than
    `friction=1.0` -- even with *zero* disturbance, because MuJoCo
    resolves foot/ground friction from the G1's foot-corner contact
    geoms (see `disturbances.py`'s `_set_friction` docstring for why
    just touching the floor geom's friction alone has *no* effect).
  - `slope_deg` above ~3-4 degrees reliably destabilizes the plain
    (undisturbed) `stand` pose within ~10s, because this controller has
    no active balance/ZMP feedback -- it is a fixed-target PD tracker,
    so *any* uncompensated slope eventually topples it. Below ~3 degrees
    it is stable indefinitely.
  - `actuator_health` below ~0.2-0.3 destabilizes the plain `stand` pose
    outright (insufficient torque authority to hold the pose against
    gravity); above ~0.4 it is stable indefinitely.
  - `disturbance_force_n` in roughly 45-90N, sustained ~0.2-0.4s
    (100-200 physics steps at the model's 2ms timestep) and applied
    laterally to the pelvis, reliably tips a *full-speed*
    (`speed_scale=1`, `hold=False`) stepping episode into a fall at
    nominal friction, purely via resonance with the stepping gait's own
    ~0.5Hz sway -- while the identical push is comfortably survived by
    the same episode with `hold=True` (or even `speed_scale` merely
    reduced, e.g. to 0.9). This is the physical basis of the guard's
    `REQUEST_HOLD`/`LIMIT_SPEED` intervention actually mattering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Reference episode length (in physics steps, at the vendored G1 model's
# native 2ms timestep) used only to scale where a *sampled* disturbance
# window falls within "a typical episode". This is deliberately decoupled
# from whatever `max_steps`/`dt` a given `sherpaos.sim.runner.run_episode`
# call actually uses -- if a caller runs a much shorter episode, a sampled
# disturbance simply may not trigger before the episode ends, which is
# acceptable stochastic behavior for a Monte Carlo sampler.
REFERENCE_MAX_PHYSICS_STEPS = 5000  # 10s at dt=0.002s


@dataclass(slots=True, frozen=True)
class Scenario:
    """One episode's MuJoCo environment recipe.

    `disturbance_direction` is a 3-vector (only the horizontal x/y
    components are physically meaningful for the lateral pelvis shove
    `disturbances.py` applies; z is accepted but expected to be 0) and is
    `None` exactly when there is no disturbance in this scenario.
    `disturbance_start_step`/`disturbance_duration_steps` are in physics
    steps (the model's native 2ms timestep), matching how
    `sherpaos/sim/runner.py` and `sherpaos/sim/disturbances.py` count
    physics steps internally.
    """

    friction: float
    slope_deg: float
    disturbance_force_n: float
    disturbance_direction: np.ndarray | None
    disturbance_start_step: int | None
    disturbance_duration_steps: int | None
    actuator_health: float
    sensor_noise_std: float
    seed: int
    # Synthetic battery inputs. These are scenario parameters, never
    # evaluator labels, and are emitted with explicit simulated provenance.
    battery_initial_fraction: float = 0.8
    battery_discharge_per_s: float = 0.00005
    battery_idle_current_a: float = 2.0
    battery_motion_current_a: float = 8.0
    battery_internal_resistance_ohm: float = 0.03
    battery_temperature_c: float = 15.0


def nominal_scenario(seed: int) -> Scenario:
    """Good traction, flat ground, no disturbance, healthy actuators,
    clean sensors. Mandatory deliverable: "Normal case."
    """
    return Scenario(
        friction=1.0,
        slope_deg=0.0,
        disturbance_force_n=0.0,
        disturbance_direction=None,
        disturbance_start_step=None,
        disturbance_duration_steps=None,
        actuator_health=1.0,
        sensor_noise_std=0.0,
        seed=seed,
        battery_initial_fraction=0.85,
        battery_temperature_c=15.0,
    )


def mixed_traction_disturbance_scenario(seed: int) -> Scenario:
    """Meaningfully low floor traction for the whole run, plus a
    mid-episode lateral pelvis shove. Mandatory deliverable:
    "mixed-traction/disturbance case."

    Reproducible per `seed`. See the module docstring for the physical
    grounding of these ranges: low friction alone already produces a
    robust, ~100x pelvis-angular-velocity-variance difference from
    `nominal_scenario`; the added shove exercises the guard's
    hold/speed-limit intervention pathway on top of that.
    """
    rng = np.random.default_rng(seed)

    friction = float(rng.uniform(0.03, 0.15))
    force_n = float(rng.uniform(50.0, 90.0))
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float)
    start_step = int(
        rng.integers(
            int(0.3 * REFERENCE_MAX_PHYSICS_STEPS), int(0.6 * REFERENCE_MAX_PHYSICS_STEPS)
        )
    )
    duration_steps = int(rng.integers(100, 200))  # 0.2-0.4s at dt=0.002s
    sensor_noise_std = float(rng.uniform(0.005, 0.02))

    return Scenario(
        friction=friction,
        slope_deg=0.0,
        disturbance_force_n=force_n,
        disturbance_direction=direction,
        disturbance_start_step=start_step,
        disturbance_duration_steps=duration_steps,
        actuator_health=1.0,
        sensor_noise_std=sensor_noise_std,
        seed=seed,
        battery_initial_fraction=0.55,
        battery_discharge_per_s=0.0002,
        battery_motion_current_a=18.0,
        battery_internal_resistance_ohm=0.12,
        battery_temperature_c=-20.0,
    )


def random_scenario(seed: int, regime: str = "train") -> Scenario:
    """Seeded Monte Carlo sampler over the full `Scenario` space, for
    later evaluation-lane use (`sherpaos/evaluation`, built on top of
    this module and `sherpaos/sim/runner.py`).

    `regime="train"` samples a moderate range that this controller
    mostly survives; `regime="stress"` samples a harsher/OOD range where
    falls become common -- in the spirit of `docs/idea.txt` section 17's
    train-vs-stress/OOD partition (not reproduced verbatim; kept simple
    per this task's scope).
    """
    if regime not in ("train", "stress"):
        raise ValueError(f"unknown regime {regime!r}, expected 'train' or 'stress'")
    rng = np.random.default_rng(seed)

    if regime == "train":
        friction = float(rng.uniform(0.5, 1.0))
        slope_deg = float(rng.uniform(0.0, 3.0))
        actuator_health = float(rng.uniform(0.7, 1.0))
        max_force_n = 40.0
        sensor_noise_std = float(rng.uniform(0.0, 0.01))
        battery_initial_fraction = float(rng.uniform(0.45, 1.0))
        battery_temperature_c = float(rng.uniform(-5.0, 25.0))
        battery_internal_resistance_ohm = float(rng.uniform(0.02, 0.08))
    else:
        friction = float(rng.uniform(0.03, 1.0))
        slope_deg = float(rng.uniform(0.0, 8.0))
        actuator_health = float(rng.uniform(0.15, 1.0))
        max_force_n = 150.0
        sensor_noise_std = float(rng.uniform(0.0, 0.05))
        battery_initial_fraction = float(rng.uniform(0.08, 1.0))
        battery_temperature_c = float(rng.uniform(-25.0, 25.0))
        battery_internal_resistance_ohm = float(rng.uniform(0.02, 0.18))

    if rng.random() < 0.5:
        force_n = 0.0
        direction = None
        start_step = None
        duration_steps = None
    else:
        force_n = float(rng.uniform(0.2 * max_force_n, max_force_n))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float)
        start_step = int(
            rng.integers(
                int(0.2 * REFERENCE_MAX_PHYSICS_STEPS), int(0.7 * REFERENCE_MAX_PHYSICS_STEPS)
            )
        )
        duration_steps = int(rng.integers(75, 250))

    return Scenario(
        friction=friction,
        slope_deg=slope_deg,
        disturbance_force_n=force_n,
        disturbance_direction=direction,
        disturbance_start_step=start_step,
        disturbance_duration_steps=duration_steps,
        actuator_health=actuator_health,
        sensor_noise_std=sensor_noise_std,
        seed=seed,
        battery_initial_fraction=battery_initial_fraction,
        battery_discharge_per_s=float(rng.uniform(0.00002, 0.0005)),
        battery_motion_current_a=float(rng.uniform(6.0, 22.0)),
        battery_internal_resistance_ohm=battery_internal_resistance_ohm,
        battery_temperature_c=battery_temperature_c,
    )
