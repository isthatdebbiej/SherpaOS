"""SherpaOS command-line entry point."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from derived_state import DerivedState
from sherpaos.datasets.cli import data_app
from sherpaos.evidence.bundle import verify_bundle, write_evidence_bundle
from sherpaos.evidence.manifest import build_run_manifest
from sherpaos.geography.terrain import (
    TerrainLoadError,
    build_mission_context,
    load_route,
    unavailable_mission_context,
)
from sherpaos.sim.runner import G1_SCENE_XML_PATH, EpisodeResult, run_episode
from sherpaos.sim.scenario import mixed_traction_disturbance_scenario, nominal_scenario
from sherpaos.sim.supervisor import SimulationSupervisorAdapter
from sherpaos.sim.unitree_walking import (
    DEFAULT_CONFIG_PATH,
    SimulatedWalkingAuxiliary,
    run_unitree_walking_episode,
)
from sherpaos.weather import WeatherSnapshot, fetch_open_meteo_current_weather
from telemetry_feed import TelemetryFeed

app = typer.Typer(no_args_is_help=True, help="Offline expedition-risk supervisor tools.")
app.add_typer(data_app, name="data")

DEFAULT_SIMULATE_MAX_CONTROL_STEPS = 1_500


def _scenario(name: str, seed: int):
    if name == "nominal":
        return nominal_scenario(seed)
    if name in {"mixed", "mixed_traction_disturbance"}:
        return mixed_traction_disturbance_scenario(seed)
    raise typer.BadParameter("scenario must be nominal or mixed_traction_disturbance")


def _walking_mission_context(waypoint: str, ambient_c: float | None):
    try:
        return build_mission_context(
            load_route(), now=0.0, waypoint_name=waypoint, temperature_c=ambient_c
        )
    except (KeyError, TerrainLoadError) as exc:
        return unavailable_mission_context(
            now=0.0,
            provenance=f"Himalaya route context unavailable: {exc}",
        )


def _walking_weather(mission_context, enabled: bool) -> WeatherSnapshot | None:
    if not enabled or not mission_context.valid:
        return None
    if mission_context.latitude is None or mission_context.longitude is None:
        return None
    return fetch_open_meteo_current_weather(mission_context.latitude, mission_context.longitude)


def _walking_uphill_slope(mission_context, enabled: bool) -> float:
    if not enabled or mission_context.slope_deg is None:
        return 0.0
    return max(0.0, float(mission_context.slope_deg))


def _run_supervised(
    scenario_name: str,
    seed: int,
    max_steps: int,
    waypoint: str,
    live_viewer: bool = False,
    telemetry_observer: Callable | None = None,
) -> tuple[EpisodeResult, SimulationSupervisorAdapter]:
    mission_context = build_mission_context(load_route(), now=0.0, waypoint_name=waypoint)
    adapter = SimulationSupervisorAdapter(mission_context)
    result = run_episode(
        _scenario(scenario_name, seed),
        seed=seed,
        guard_fn=adapter,
        max_steps=max_steps,
        live_viewer=live_viewer,
        telemetry_observer=telemetry_observer,
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
    max_steps: Annotated[
        int, typer.Option(min=2, help="50 Hz control ticks; defaults to a 30-second episode.")
    ] = DEFAULT_SIMULATE_MAX_CONTROL_STEPS,
    waypoint: Annotated[str, typer.Option()] = "Lukla",
    output: Annotated[Path, typer.Option()] = Path("artifacts/runs/latest"),
    viewer: Annotated[
        bool, typer.Option("--viewer", help="Show the native MuJoCo viewer during the episode.")
    ] = False,
    telemetry_output: Annotated[
        Path | None, typer.Option(help="Write the final aggregated telemetry snapshot as JSON.")
    ] = None,
    telemetry_port: Annotated[
        int | None, typer.Option(min=1, max=65535, help="Serve live telemetry on localhost.")
    ] = None,
) -> None:
    """Run one offline five-guard MuJoCo episode and write evidence."""
    feed = TelemetryFeed()
    server = feed.serve(port=telemetry_port) if telemetry_port is not None else None
    try:
        result, adapter = _run_supervised(
            scenario,
            seed,
            max_steps,
            waypoint,
            live_viewer=viewer,
            telemetry_observer=feed.push,
        )
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
    if telemetry_output is not None:
        feed.write(telemetry_output)
    summary = _write_run(output, scenario, seed, waypoint, result, adapter)
    if telemetry_output is not None:
        summary["telemetry_snapshot"] = str(telemetry_output.resolve())
    typer.echo(json.dumps(summary, indent=2))
    if not summary["evidence_verified"]:
        raise typer.Exit(code=1)


@app.command()
def walk(
    max_steps: Annotated[int | None, typer.Option(min=1, help="50 Hz control ticks.")] = None,
    viewer: Annotated[
        bool, typer.Option("--viewer/--headless", help="Show the native MuJoCo viewer.")
    ] = True,
    telemetry_output: Annotated[
        Path, typer.Option(help="Continuously refreshed aggregate telemetry JSON.")
    ] = Path("artifacts/walk/telemetry.json"),
    telemetry_port: Annotated[
        int | None, typer.Option(min=1, max=65535, help="Serve live telemetry on localhost.")
    ] = 8088,
    ambient_c: Annotated[
        float | None,
        typer.Option(help="Fallback ambient temperature in degrees C; omit when unavailable."),
    ] = None,
    waypoint: Annotated[
        str,
        typer.Option(help="Named EBC route waypoint published in the live environment context."),
    ] = "Lobuche",
    weather: Annotated[
        bool,
        typer.Option(
            "--weather/--no-weather", help="Fetch display-only current weather on activation."
        ),
    ] = True,
    initial_battery_fraction: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Display-model initial battery charge fraction."),
    ] = 1.0,
    simulate_auxiliary: Annotated[
        bool,
        typer.Option(
            "--simulate-auxiliary/--no-simulate-auxiliary",
            help="Publish simulator-only battery, speed, and foot telemetry for this walking demo.",
        ),
    ] = True,
    uphill: Annotated[
        bool,
        typer.Option(
            "--uphill/--level",
            help="Tilt the MuJoCo floor to the selected route's uphill grade.",
        ),
    ] = True,
) -> None:
    """Run the pinned Unitree policy and publish its sensorized telemetry."""
    feed = TelemetryFeed(
        range_estimator=DerivedState(initial_battery_fraction=initial_battery_fraction)
    )
    mission_context = _walking_mission_context(waypoint, ambient_c)
    weather_snapshot = _walking_weather(mission_context, weather)
    observed_ambient_c = (
        weather_snapshot.temperature_c
        if weather_snapshot is not None and weather_snapshot.available
        else ambient_c
    )
    uphill_slope_deg = _walking_uphill_slope(mission_context, uphill)
    latest_auxiliary: SimulatedWalkingAuxiliary | None = None

    def observe_auxiliary(auxiliary: SimulatedWalkingAuxiliary) -> None:
        nonlocal latest_auxiliary
        latest_auxiliary = auxiliary

    def observe(sample) -> None:
        auxiliary = latest_auxiliary
        feed.push(
            sample,
            command_speed_mps=0.5,
            contacts=None if auxiliary is None else auxiliary.contacts,
            forces=None if auxiliary is None else auxiliary.forces_n,
            electrical_power_w=None if auxiliary is None else auxiliary.electrical_power_w,
            ambient_c=observed_ambient_c,
            mission_context=mission_context,
            location_name=waypoint,
            weather=weather_snapshot,
            range_speed_mps=0.5,
            actual_speed_mps=None if auxiliary is None else auxiliary.actual_speed_mps,
            actual_speed_source=None if auxiliary is None else "simulated:mujoco_base_velocity",
            contacts_source=None if auxiliary is None else "simulated:mujoco_contact_force",
            battery_gauge=(
                None
                if auxiliary is None
                else {
                    "fraction": auxiliary.battery_fraction,
                    "voltage_v": auxiliary.battery_voltage_v,
                    "current_a": auxiliary.battery_current_a,
                    "temperature_c": auxiliary.battery_temperature_c,
                }
            ),
            battery_gauge_source=None if auxiliary is None else "simulated:walking_pack_model",
            electrical_power_source=None if auxiliary is None else "simulated:walking_pack_model",
            simulated_terrain_slope_deg=uphill_slope_deg,
        )
        feed.write(telemetry_output)

    server = feed.serve(port=telemetry_port) if telemetry_port is not None else None
    try:
        result = run_unitree_walking_episode(
            config_path=DEFAULT_CONFIG_PATH,
            max_steps=max_steps,
            live_viewer=viewer,
            telemetry_observer=observe,
            simulated_auxiliary_observer=observe_auxiliary if simulate_auxiliary else None,
            simulated_auxiliary_temperature_c=float(observed_ambient_c or 20.0),
            simulated_battery_initial_fraction=initial_battery_fraction,
            terrain_slope_deg=uphill_slope_deg,
        )
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
    typer.echo(
        json.dumps(
            {
                "controller": "unitree-g1-motion.pt",
                "control_steps": result.control_steps,
                "elapsed_seconds": result.elapsed_seconds,
                "planar_distance_m": result.planar_distance_m,
                "fell": result.fell,
                "uphill_slope_deg": result.uphill_slope_deg,
                "telemetry_snapshot": str(telemetry_output.resolve()),
            },
            indent=2,
        )
    )


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
