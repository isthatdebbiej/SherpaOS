"""Continuously publish paced MuJoCo decisions for the Vultr live demo."""

from __future__ import annotations

import itertools
import os
import time

from sherpaos.cli.main import _run_supervised


def main() -> None:
    delay = float(os.environ.get("SHERPA_LIVE_CONTROL_DELAY_SECONDS", "0.02"))
    steps = int(os.environ.get("SHERPA_LIVE_EPISODE_STEPS", "500"))
    scenarios = itertools.cycle((("nominal", "Lukla"), ("mixed_traction_disturbance", "Lobuche")))
    for seed, (scenario, waypoint) in enumerate(scenarios, start=9000):
        _run_supervised(
            scenario,
            seed=seed,
            max_steps=steps,
            waypoint=waypoint,
            telemetry_observer=lambda _sample: time.sleep(delay),
        )


if __name__ == "__main__":
    main()
