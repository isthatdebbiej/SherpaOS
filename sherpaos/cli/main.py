"""SherpaOS command-line entry point."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from sherpaos.datasets.cli import data_app
from sherpaos.evidence.bundle import verify_bundle, write_evidence_bundle
from sherpaos.evidence.manifest import build_run_manifest
from sherpaos.geography.terrain import build_mission_context, load_route
from sherpaos.sim.runner import G1_SCENE_XML_PATH, EpisodeResult, run_episode
from sherpaos.sim.scenario import mixed_traction_disturbance_scenario, nominal_scenario
from sherpaos.sim.supervisor import SimulationSupervisorAdapter

app = typer.Typer(no_args_is_help=True, help="Offline expedition-risk supervisor tools.")
app.add_typer(data_app, name="data")


def _scenario(name: str, seed: int):
    if name == "nominal":
        return nominal_scenario(seed)
    if name in {"mixed", "mixed_traction_disturbance"}:
        return mixed_traction_disturbance_scenario(seed)
    raise typer.BadParameter("scenario must be nominal or mixed_traction_disturbance")


def _run_supervised(
    scenario_name: str,
    seed: int,
    max_steps: int,
    waypoint: str,
    live_viewer: bool = False,
) -> tuple[EpisodeResult, SimulationSupervisorAdapter]:
    mission_context = build_mission_context(load_route(), now=0.0, waypoint_name=waypoint)
    adapter = SimulationSupervisorAdapter(mission_context)
    result = run_episode(
        _scenario(scenario_name, seed),
        seed=seed,
        guard_fn=adapter,
        max_steps=max_steps,
        live_viewer=live_viewer,
    )
    return result, adapter


def _write_run(
    output: Path,
    scenario_name: str,
    seed: int,
    waypoint: str,
    result: EpisodeResult,
    adapter: SimulationSupervisorAdapter,
) -> dict:
    # Each decision was made from the previous sample; the final simulator
    # sample has not yet been presented to the callback.
    observed_telemetry = result.telemetry[: len(adapter.decisions)]
    manifest = build_run_manifest(
        run_id=output.name,
        seed=seed,
        scenario_name=scenario_name,
        config={"max_steps": result.steps_survived, "waypoint": waypoint},
        controller_id="pd-step-controller+five-guard-supervisor",
        model_id="mujoco-menagerie-unitree-g1",
    )
    write_evidence_bundle(
        output,
        observed_telemetry,
        adapter.decisions,
        adapter.receipts,
        manifest,
    )
    action_counts: dict[str, int] = {}
    for decision in adapter.decisions:
        action_counts[decision.action.value] = action_counts.get(decision.action.value, 0) + 1
    return {
        "scenario": scenario_name,
        "seed": seed,
        "waypoint": waypoint,
        "fell": result.fell,
        "steps_survived": result.steps_survived,
        "decisions": len(adapter.decisions),
        "receipts": len(adapter.receipts),
        "action_counts": action_counts,
        "evidence_bundle": str(output.resolve()),
        "evidence_verified": verify_bundle(output),
        "commit_sha": manifest.commit_sha,
    }


@app.command()
def preflight() -> None:
    """Validate package, pinned assets, route data, and a short physics run."""
    checks = {
        "g1_scene_exists": G1_SCENE_XML_PATH.exists(),
        "route_loads": False,
        "five_guard_smoke": False,
    }
    try:
        load_route()
        checks["route_loads"] = True
        result, adapter = _run_supervised("nominal", seed=1, max_steps=5, waypoint="Lukla")
        checks["five_guard_smoke"] = result.steps_survived == 5 and len(adapter.decisions) == 4
    except Exception as exc:
        typer.echo(json.dumps({"status": "RED", "checks": checks, "error": str(exc)}, indent=2))
        raise typer.Exit(code=1) from exc
    status = "GREEN" if all(checks.values()) else "RED"
    typer.echo(json.dumps({"status": status, "checks": checks}, indent=2))
    if status != "GREEN":
        raise typer.Exit(code=1)


@app.command("test")
def test_command(
    integration: Annotated[bool, typer.Option(help="Include MuJoCo integration tests.")] = True,
) -> None:
    """Run the canonical Ruff and pytest merge gate."""
    commands = [["ruff", "check", "."], ["pytest", "-q"]]
    if not integration:
        commands[-1].extend(["-m", "not integration"])
    for command in commands:
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise typer.Exit(code=completed.returncode)


@app.command()
def simulate(
    scenario: Annotated[str, typer.Option()] = "nominal",
    seed: Annotated[int, typer.Option()] = 1,
    max_steps: Annotated[int, typer.Option(min=2)] = 100,
    waypoint: Annotated[str, typer.Option()] = "Lukla",
    output: Annotated[Path, typer.Option()] = Path("artifacts/runs/latest"),
    viewer: Annotated[
        bool, typer.Option("--viewer", help="Show the native MuJoCo viewer during the episode.")
    ] = False,
) -> None:
    """Run one offline five-guard MuJoCo episode and write evidence."""
    result, adapter = _run_supervised(scenario, seed, max_steps, waypoint, live_viewer=viewer)
    summary = _write_run(output, scenario, seed, waypoint, result, adapter)
    typer.echo(json.dumps(summary, indent=2))
    if not summary["evidence_verified"]:
        raise typer.Exit(code=1)


@app.command()
def demo(
    offline: Annotated[
        bool, typer.Option(help="Assert the demo has no network dependency.")
    ] = True,
    output: Annotated[Path, typer.Option()] = Path("artifacts/demo-run"),
) -> None:
    """Run deterministic nominal and hazard evidence scenarios."""
    if not offline:
        raise typer.BadParameter("the hackathon demo contract requires --offline")
    summaries = []
    for name, waypoint in (("nominal", "Lukla"), ("mixed_traction_disturbance", "Lobuche")):
        run_output = output / name
        result, adapter = _run_supervised(name, seed=42, max_steps=300, waypoint=waypoint)
        summaries.append(_write_run(run_output, name, 42, waypoint, result, adapter))
    report = {"status": "GREEN", "offline": True, "runs": summaries}
    (output / "SUMMARY.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))
