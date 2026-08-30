"""Simulate the full Everest Base Camp journey, waypoint by waypoint, and
turn each leg's SherpaOS guard decision into an EmotionState + (optionally)
a recommended G1 gesture.

This script is a *consumer* of the existing offline pipeline -- it reuses
`sherpaos.sim.runner.run_episode`, `sherpaos.sim.supervisor.
SimulationSupervisorAdapter`, and `sherpaos.geography.terrain` exactly as
`sherpa demo` does (see sherpaos/cli/main.py's `_run_supervised`), then
feeds each leg's final `GuardDecision` through `sherpaos.emotion.mapping.
classify_emotion` and `sherpaos.emotion.gestures.should_gesture`. It does
not modify the guard/policy/estimator runtime and it does not run the
robot's actuators -- it only prints/writes what emotion+gesture *would*
be expressed, matching the "presentation only" boundary documented in
sherpaos/emotion/__init__.py.

Each of the 8 waypoints in configs/terrain/ebc_route.json gets one short
episode (`--steps-per-leg`, default 300 == same length as `sherpa demo`'s
scenarios) whose terrain-driven difficulty (friction/slope) scales with
that waypoint's own `exposure_class` (LOW/MODERATE/SEVERE) and
`slope_deg_from_prev` -- steeper/more exposed terrain is *harder*, not
just relabeled, so the guard's REQUEST_HOLD/LIMIT_SPEED/PASS actions (and
therefore the emotions derived from them) actually track real simulated
difficulty along the route rather than a scripted mood arc.

Usage:
    uv run python scripts/himalaya_emotion_journey.py
    uv run python scripts/himalaya_emotion_journey.py --steps-per-leg 500 \
        --output artifacts/emotion_journey/run.json
    uv run python scripts/himalaya_emotion_journey.py --run-gestures
        # after printing the journal, actually triggers each leg's
        # recommended gesture via scripts/run_g1_dance.py (requires the
        # vendored third_party/FSMDeployG1 fetched via
        # scripts/fetch_fsm_dance_repo.py -- see docs/G1_DANCE_DEMO.md)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sherpaos.contracts import GuardAction  # noqa: E402
from sherpaos.emotion.gestures import should_gesture  # noqa: E402
from sherpaos.emotion.mapping import classify_emotion  # noqa: E402
from sherpaos.geography.terrain import (  # noqa: E402
    RouteData,
    Waypoint,
    build_mission_context,
    load_route,
)
from sherpaos.sim.runner import run_episode  # noqa: E402
from sherpaos.sim.scenario import Scenario  # noqa: E402
from sherpaos.sim.supervisor import SimulationSupervisorAdapter  # noqa: E402

# Recent-action window handed to classify_emotion's `recent_actions`, so a
# leg immediately after a LIMIT_SPEED/REQUEST_HOLD leg can register RELIEF
# instead of flatly CALM. Small on purpose -- this is presentation only,
# not a second hysteresis layer competing with policy/state_machine.py's.
_RECENT_ACTIONS_WINDOW = 3

_EXPOSURE_FRICTION = {
    "LOW": 1.0,
    "MODERATE": 0.4,
    "SEVERE": 0.12,
}
_EXPOSURE_ACTUATOR_HEALTH = {
    "LOW": 1.0,
    "MODERATE": 0.9,
    "SEVERE": 0.7,
}


def _scenario_for_waypoint(wp: Waypoint, seed: int) -> Scenario:
    """Derive a MuJoCo episode recipe from this waypoint's own terrain
    fields, so difficulty follows the real route instead of a fixed
    per-leg script. See sherpaos/sim/scenario.py's module docstring for
    the empirically-grounded friction/slope/actuator_health ranges this
    draws from."""
    friction = _EXPOSURE_FRICTION.get(wp.exposure_class, 0.6)
    actuator_health = _EXPOSURE_ACTUATOR_HEALTH.get(wp.exposure_class, 0.9)
    # Cap: sherpaos/sim/scenario.py documents plain `stand` as reliably
    # destabilized above ~3-4 degrees with no active balance feedback --
    # capping here keeps every leg a survivable (if sometimes guarded)
    # episode rather than a guaranteed fall on the steepest real segments.
    slope_deg = min(3.0, abs(wp.slope_deg_from_prev or 0.0))
    return Scenario(
        friction=friction,
        slope_deg=slope_deg,
        disturbance_force_n=0.0,
        disturbance_direction=None,
        disturbance_start_step=None,
        disturbance_duration_steps=None,
        actuator_health=actuator_health,
        sensor_noise_std=0.01,
        seed=seed,
        battery_initial_fraction=max(0.15, 0.9 - 0.08 * (wp.cumulative_distance_m / 10_000.0)),
        battery_temperature_c=max(-15.0, 15.0 - wp.elevation_m / 250.0),
    )


def _leg_summary(
    route: RouteData,
    wp: Waypoint,
    seed: int,
    steps: int,
    recent_actions: deque[GuardAction],
) -> dict:
    mission_context = build_mission_context(route, now=float(seed), waypoint_name=wp.name)
    adapter = SimulationSupervisorAdapter(mission_context)
    result = run_episode(
        _scenario_for_waypoint(wp, seed),
        seed=seed,
        guard_fn=adapter,
        max_steps=steps,
    )

    if not adapter.decisions:
        raise RuntimeError(f"leg for waypoint {wp.name!r} produced no guard decisions")
    final_decision = adapter.decisions[-1]
    final_reports = final_decision.guard_reports
    mobility_report = next((r for r in final_reports if r.guard.value == "mobility"), None)
    mobility_ok = mobility_report is None or mobility_report.recommended_action == GuardAction.PASS

    emotion = classify_emotion(
        final_decision,
        milestone_reached=not result.fell,
        recent_actions=tuple(recent_actions),
    )
    gesture = should_gesture(emotion, mobility_ok=mobility_ok)

    recent_actions.append(final_decision.action)
    while len(recent_actions) > _RECENT_ACTIONS_WINDOW:
        recent_actions.popleft()

    action_counts: dict[str, int] = {}
    for decision in adapter.decisions:
        action_counts[decision.action.value] = action_counts.get(decision.action.value, 0) + 1

    return {
        "waypoint": wp.name,
        "elevation_m": wp.elevation_m,
        "exposure_class": wp.exposure_class,
        "fell": result.fell,
        "steps_survived": result.steps_survived,
        "action_counts": action_counts,
        "final_action": final_decision.action.value,
        "final_score": final_decision.score,
        "final_confidence": final_decision.confidence,
        "emotion": emotion.label,
        "emotion_intensity": emotion.intensity,
        "emotion_reason_codes": [rc.value for rc in emotion.reason_codes],
        "recommended_gesture": gesture,
        "mobility_ok": mobility_ok,
    }


def _run_gesture(skill: str) -> None:
    runner = REPO_ROOT / "scripts" / "run_g1_dance.py"
    fsm_repo = REPO_ROOT / "third_party" / "FSMDeployG1"
    if not fsm_repo.is_dir():
        print(
            "  [skip] third_party/FSMDeployG1 not fetched yet -- run "
            "scripts/fetch_fsm_dance_repo.py first",
            file=sys.stderr,
        )
        return
    venv_python = fsm_repo / ".venv" / "bin" / "python"
    python_bin = str(venv_python) if venv_python.exists() else sys.executable
    print(f"  [gesture] running {skill!r} via run_g1_dance.py ...")
    subprocess.run(  # noqa: S603
        [python_bin, str(runner), skill],
        cwd=str(fsm_repo),
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-per-leg", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "emotion_journey" / "himalaya_journey.json",
    )
    parser.add_argument(
        "--run-gestures",
        action="store_true",
        help="Actually trigger each leg's recommended gesture on the MuJoCo G1 "
        "(requires third_party/FSMDeployG1, see docs/G1_DANCE_DEMO.md).",
    )
    args = parser.parse_args()

    route = load_route()
    recent_actions: deque[GuardAction] = deque(maxlen=_RECENT_ACTIONS_WINDOW)
    legs: list[dict] = []

    print(f"Simulating {len(route.waypoints)}-waypoint Everest Base Camp journey...\n")
    for i, wp in enumerate(route.waypoints):
        leg_seed = args.seed + i
        summary = _leg_summary(route, wp, leg_seed, args.steps_per_leg, recent_actions)
        legs.append(summary)
        gesture_note = f" -> gesture: {summary['recommended_gesture']}" if summary[
            "recommended_gesture"
        ] else ""
        print(
            f"[{i + 1}/{len(route.waypoints)}] {wp.name:<20} "
            f"({wp.exposure_class:<8}) action={summary['final_action']:<12} "
            f"emotion={summary['emotion']:<10} "
            f"intensity={summary['emotion_intensity']:.2f}{gesture_note}"
        )
        if args.run_gestures and summary["recommended_gesture"]:
            _run_gesture(summary["recommended_gesture"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "route": route.terrain_source,
        "route_version": route.terrain_version,
        "steps_per_leg": args.steps_per_leg,
        "seed": args.seed,
        "legs": legs,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote journey report to {args.output}")

    fell_legs = [leg["waypoint"] for leg in legs if leg["fell"]]
    if fell_legs:
        print(f"Note: robot fell on legs: {', '.join(fell_legs)}", file=sys.stderr)


if __name__ == "__main__":
    main()
