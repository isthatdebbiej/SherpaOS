# Telemetry Feed API

`TelemetryFeed` publishes a one-second, unit-labelled summary of the same
`RobotTelemetry` samples presented to SherpaOS guards. It is an observability
interface only: it does not call an LLM and cannot affect actuation decisions.

## Consumers

- In process: `feed.snapshot()` returns a JSON-safe dictionary.
- File: `feed.write("telemetry.json")` atomically replaces a JSON snapshot.
- HTTP: `feed.serve(port=8088)` provides `GET /telemetry` (JSON) and `GET /llm`
  (compact plain text). The server binds to `127.0.0.1` by default.

Run a live feed during simulation:

```powershell
uv run --no-sync sherpa walk
```

This starts the pinned Unitree walking policy in MuJoCo, continuously replaces
`artifacts/walk/telemetry.json`, and serves the same data at
`http://127.0.0.1:8088/telemetry` and `http://127.0.0.1:8088/llm`. During the
rollout, open a second terminal to inspect the file or query either endpoint.
Every activation resolves the selected EBC waypoint from the bundled offline
route artifact and publishes it under `environment.himalaya`.

`sherpa walk` fetches a current Open-Meteo weather snapshot once when the feed
is activated and publishes it under `environment.weather`. It uses the reported
temperature as `environment.ambient_c`. `--no-weather` disables the lookup;
`--ambient-c <degrees-C>` provides an explicit fallback when live weather is
unavailable. Weather is display-only and never reaches the guard or policy.

`sherpa walk` enables `--simulate-auxiliary` by default. In the simulator it
publishes base speed, foot contacts/load, electrical draw, and a battery gauge
under their normal fields so the complete telemetry shape can be exercised.
Every generated field is marked `simulated:*` in its source field and in
`decision_context.simulation_only_fields`. Use `--no-simulate-auxiliary` to
show the same gaps expected from the current real-sensor adapter.
`sherpa simulate --telemetry-output <path> --telemetry-port <port>` exposes the
same interface for the existing 29-DOF stepping simulation.

`sherpa walk --waypoint Lobuche` selects the fixed Himalayan location for this
activation; `Lukla`, `Phakding`, `Namche Bazaar`, `Tengboche`, `Dingboche`,
`Lobuche`, `Gorak Shep`, and `Everest Base Camp` are available. The route data
is local and auditable, not a live online weather or mapping service.

By default, `sherpa walk --uphill` rotates the MuJoCo floor so walking along
positive world X climbs the selected waypoint's positive route grade. The
physical simulation grade is published in `environment.terrain_simulation`.
`--level` runs the same location context on a flat floor for comparison. This
is a uniform incline, not a reconstructed Himalayan terrain mesh.

## Field Notes

All JSON numbers include units in their field names. `null` means unavailable;
it never means zero. The ambient value can be supplied as fixed environment
metadata through `sherpa walk --ambient-c`; foot contact, foot load, and actual
base speed remain `null` unless supplied by an onboard-equivalent estimator.

`power.mechanical_w` is absolute joint mechanical power, not battery draw. Use
`power.electrical_w` only when calibrated electrical power is supplied or when
the telemetry includes voltage and current. Mechanical power alone excludes
drivetrain losses, compute, sensors, and thermal loads.

`battery.gauge` contains raw battery fields when an onboard gauge supplies
them. For `sherpa walk`, `battery.range_model` is a display-only estimate built
from observed joint work, a 500 Wh nominal-pack assumption, an always-on load,
configured initial charge (`--initial-battery-fraction`, default `1.0`), and
the current ambient temperature. It labels its source and speed basis, and
publishes estimated electrical draw, cold-derated capacity, charge, endurance,
and remaining range. It is not a measured G1 battery value and never reaches a
guard or the walking policy.

`GET /llm` is a compact summary intended for an external consumer. Treat it as
observability context, not a safety decision source; SherpaOS remains offline
and all guard decisions use the typed runtime contract directly.

`decision_context` makes the summary usable for reasoning without converting
the feed into a decision-maker. It reports stability evidence, commanded-speed
context, range-model and gauge availability, and a `data_gaps` list. Consumers
must treat listed gaps as unknown, rather than infer speed tracking, gait
quality, battery state, or electrical draw from unavailable measurements.

The current `sherpa walk` command publishes Unitree-policy telemetry but does
not feed SherpaOS actions back into that policy. Unitree's policy marches in
place when commanded to zero, so mapping `REQUEST_HOLD` to zero velocity would
not satisfy the supervisor's hold semantics. Treat it as a walking telemetry
source until a dedicated 12-DOF actuation adapter is validated.