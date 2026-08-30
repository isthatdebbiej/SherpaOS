"""Continuously publish paced MuJoCo decisions for the Vultr live demo."""

from __future__ import annotations

import itertools
import os
import time

from sherpaos.geography.terrain import build_mission_context, load_route
from sherpaos.sim.runner import run_episode
from sherpaos.sim.scenario import mixed_traction_disturbance_scenario, nominal_scenario
from sherpaos.sim.supervisor import SimulationSupervisorAdapter


def main() -> None:
    delay = float(os.environ.get("SHERPA_LIVE_CONTROL_DELAY_SECONDS", "0.02"))
    steps = int(os.environ.get("SHERPA_LIVE_EPISODE_STEPS", "500"))
    scenarios = itertools.cycle((("nominal", "Lukla"), ("mixed", "Lobuche")))
    for seed, (scenario_name, waypoint) in enumerate(scenarios, start=9000):
        scenario = (
            nominal_scenario(seed)
            if scenario_name == "nominal"
            else mixed_traction_disturbance_scenario(seed)
        )
        context = build_mission_context(load_route(), now=0.0, waypoint_name=waypoint)
        adapter = SimulationSupervisorAdapter(context)
        run_episode(
            scenario,
            seed=seed,
            guard_fn=adapter,
            max_steps=steps,
            telemetry_observer=lambda _sample: time.sleep(delay),
        )


if __name__ == "__main__":
    main()