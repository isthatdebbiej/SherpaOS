# DECISIONS.md — timestamped decisions and rejected alternatives

## 2026-08-30 — Environmental context remains deterministic

Wind and ambient temperature already existed as optional `MissionContext` fields but
were not scored. The geographic guard now evaluates high wind and extreme cold with
explicit fixed thresholds and conservative severe-condition hold overrides. New reason
codes were added to the shared contract so evidence can distinguish terrain from weather.
Depth remains outside `RobotTelemetry`: raw camera/depth frames require their own timed,
calibrated adapter contract before they can safely influence runtime decisions.

## 2026-08-29 11:24 PT — Repo location and layout

Repo root created at `sherpaos/` (containing package `sherpaos/sherpaos/`), per explicit
user instruction, nested under the workspace `SherpaOS/` that holds `docs/plan.md` and
`docs/idea.txt`. Standard src-less Python layout: repo root name == importable package
name.

## 2026-08-29 11:24 PT — plan.md is authoritative over idea.txt

`docs/plan.md` (workspace-level) is the "operational source of truth" per its own header
and is dated the same day as kickoff. `docs/idea.txt` is the earlier, more detailed
spec plan.md was distilled from (IceSense/BodySense/EnergyTerrain -> collapsed into a
single deterministic risk estimator + state machine; six-state policy
CONTINUE/CAUTION/SLOW/HOLD/RETREAT/STOP -> collapsed into
PASS/LIMIT_SPEED/REQUEST_HOLD). Where they conflict, plan.md's architecture wins;
idea.txt is used only for implementation detail/rationale (e.g. residual-based anomaly
detection, energy-margin math, evaluation suite structure) that plan.md references but
doesn't spell out.

**Rejected alternative**: implementing idea.txt's five-module architecture
(IceSense/BodySense/EnergyTerrain/AnchorSense/SherpaPolicy) verbatim. Rejected because
plan.md explicitly simplifies to one estimator + one state machine for the hackathon
timeline and because idea.txt itself says "two implementation decisions I'd lock now:
submit under Track 3... treat IceSense + BodySense + EnergyTerrain + policy + evaluation
as the non-negotiable core" — i.e. the modules can be internal feature groups inside a
single estimator rather than separate top-level products. The estimator module keeps
internal feature groups named after these concepts (traction/slip features, body-residual
features) so the distinction is preserved without three separate top-level packages.

## 2026-08-29 11:24 PT — Simulation controller fallback

No pre-trained G1 walking policy is vendored in this pass (network/weight-size risk,
offline-constraint risk, integration risk within the session budget). First
implementation uses a constrained MuJoCo G1 posture/stepping task under a simple
built-in PD controller. This is explicitly sanctioned by plan.md Lane B's own fallback
clause ("If stable walking is still unavailable at the two-hour cutoff, use a
constrained G1 posture/stepping task... label it honestly") and idea.txt's Plan B/C.
Upgrading to a real trained locomotion controller (LeRobot G1) is a stretch item, not a
blocker for the intervention-proof gate.

**Rejected alternative**: integrating LeRobot's G1 sim/locomotion path directly.
Rejected for this pass due to added dependency weight and setup risk; revisit once the
core estimator/policy/evaluation loop is proven end-to-end.

## 2026-08-29 12:05 PT — Five-guard architecture supersedes the collapsed single estimator

`docs/plan.md` was revised mid-build to specify five independent guard families
(mobility, dynamics/body, telemetry-health, battery-margin, geographic-risk), each
emitting its own `GuardReport` (score/confidence/reasons/recommended action/provenance),
fused conservatively (a high-severity guard must not be diluted by averaging with calm
guards) into the final `GuardDecision`. This **revises** the 11:24 AM entry above, which
collapsed IceSense/BodySense/EnergyTerrain into one estimator — that collapse was correct
for the plan version active at the time, but the plan has since been updated to require
five separately-reported guards plus two new ones (battery, geographic) that didn't exist
in `idea.txt` at all. `sherpaos/contracts.py` was extended additively (new `GuardReport`,
`GuardName`, `MissionContext` dataclasses; new `battery_current_a`/
`battery_temperature_c` fields on `RobotTelemetry`; new `guard_reports` field on
`GuardDecision` defaulting to `()`) — nothing existing was renamed or removed, so code
already written against the old shape keeps working; it just doesn't yet populate the new
fields until the guard-split/battery/geographic work lands.

Also fixed while touching this file: `TelemetrySource`/`GuardAction`/`ReasonCode` were
`(str, enum.Enum)` (ruff UP042); switched to `enum.StrEnum` (Python 3.11+, matches
`requires-python`) for the existing enums and the two new ones.

## 2026-08-29 12:05 PT — Geographic terrain artifact: waypoint route, not a DEM raster

Built `configs/terrain/ebc_route.json` (via `scripts/prepare_terrain_artifact.py`): 8
named Everest Base Camp trek waypoints (Lukla through Everest Base Camp) with
Wikipedia-sourced coordinates/elevations (see `configs/terrain/PROVENANCE.md` for
per-waypoint URLs, fetched 2026-08-29), plus derived slope/distance/exposure fields.
Chose a small hand-pinned waypoint/route artifact over a gridded DEM raster (e.g.
SRTM/Copernicus clip via rasterio/GDAL) to avoid a heavy geospatial dependency chain on
Windows within the session budget, while still being real, sourced, and provenance-
documented — consistent with `docs/idea.txt`'s "physics matter more than snow textures"
philosophy applied to terrain data. Distances are great-circle (haversine) between
waypoints, not actual trail distance — documented as a lower bound, not a calibrated
hiking distance. `exposure_class` is an explicit bounded heuristic (altitude + local
slope), not a validated avalanche/exposure model.

**Rejected alternative**: querying a live elevation API (Open-Elevation/OpenTopoData) at
runtime. Rejected because `docs/plan.md` requires the geographic guard to work with the
internet absent — the artifact is fetched/prepared once (this step, with network access)
and committed; runtime only ever reads the local JSON file.

## 2026-08-29 11:40 PT — Menagerie vendored via sparse+partial clone, not full clone

`mujoco_menagerie` is only needed locally for the G1 model files (`unitree_g1/`) so the
local/offline demo has an asset to load — this is unrelated to Vultr/HF cloud usage,
which is for parallel Monte Carlo/overnight validation at scale, not for hosting this
asset. A first attempt did a full `git clone` (1.6GB, every robot in the menagerie);
corrected to `git clone --filter=blob:none --sparse --depth 1` +
`git sparse-checkout set unitree_g1`, which fetches blobs on demand and lands at ~56MB.
Pinned commit: `da76818e269b82289eba39808e2fb91d679d6994`. The directory is gitignored
in this repo (vendored, not committed) — see `third_party/ATTRIBUTIONS.md` for the
license/commit record.

## 2026-08-29 11:24 PT — Cloud (HF Jobs / Vultr) not exercised in this pass

No credentials are configured in this environment. `sherpa overnight launch/status/fetch`
are implemented as CLI surface + local-only stand-ins (they validate preflight
conditions and can run a local shard) rather than issuing real HF/Vultr API calls. Wiring
real cloud calls is left for whoever has the account/credentials, per plan.md section 6's
preflight-refusal contract, which this local stand-in already enforces.
## 2026-08-29 — Learned scope frozen to the temporal risk supervisor

The only model trained during the hackathon is a compact temporal SherpaOS risk model.
It consumes 1–3 seconds of onboard-observable telemetry and emits mobility risk,
dynamics/body risk, and confidence. It never controls joints. Telemetry health, battery,
geography, policy hysteresis, and hard action floors remain deterministic. Locomotion
policy training/fine-tuning, Transformers, VLAs, and end-to-end control are out of scope.
Simulator observations and privileged labels are stored separately; evaluator truth now
retains tilt and planted-foot slip magnitude so targets describe physical outcomes rather
than merely copying scenario parameters.

## 2026-08-29 — Native MuJoCo viewer is an optional local visualization path

`sherpa simulate --viewer` launches MuJoCo's native viewer and synchronizes it to the
physics timestep, then holds the final frame until the user closes the viewer window. It
is opt-in so batch simulation, evidence generation, and all runtime decision behavior
remain headless and deterministic by default. The viewer reads the same model/data
objects already driven by the existing controller and does not create a new telemetry,
policy, or ground-truth data path.

## 2026-08-29 — Unitree pretrained walking policy is a separately pinned simulator path

The local `third_party/unitree_rl_gym/` checkout at
`276801e46c5d433564f24658bac64f254b7d2d4b` supplies the BSD-3-Clause licensed,
TorchScript `deploy/pre_train/g1/motion.pt` policy and its matching 12-actuator G1
MuJoCo model. The upstream `deploy_mujoco.py g1.yaml` rollout has been run locally with
the config's 0.5 m/s forward command. It is not connected to SherpaOS's existing
29-actuator telemetry/controller path: a new, verified 12-DOF telemetry and actuation
adapter is required before claims about five-guard supervision of the walking policy.

## 2026-08-29 — Sensorized scenes emit only adapter-visible telemetry

`sherpaos.sim.g1_sensors.build_sensorized_scene(src_xml)` compiles a temporary MJCF
wrapper with joint position, joint velocity, actuator force, pelvis orientation, angular
velocity, and linear acceleration sensors. It produces 95 sensor elements for the
29-DOF Menagerie model and 41 for the 12-DOF Unitree model, without modifying either
source asset. `G1SensorSuite.low_state()` produces `RobotTelemetry` at each 50 Hz
control tick in the existing runner, so the estimator/policy still receive only
contract-defined onboard-observable values. The separately pinned Unitree walking-policy
config is in `configs/unitree/g1_walking.yaml`; the Playground config remains unset
because that policy is not Playground-compatible.

## 2026-08-29 — Telemetry delivery is observational and contract-bound

`TelemetryFeed` consumes finalized `RobotTelemetry` samples after they have been
produced for the guard loop. It can publish an aggregated snapshot in process, by atomic
JSON file, or through a localhost-only HTTP server with a compact `/llm` text form. The
feed is an optional observer: it has no return path into the controller or guard policy,
and feed failures cannot interrupt the runtime decision loop. Foot contacts, loads, and
actual base speed are emitted only when an onboard-equivalent estimator provides them;
the feed does not inspect simulator contacts or other evaluator-only state.

## 2026-08-29 — Unitree walking telemetry is published before supervised actuation

`sherpa walk` runs the pinned 12-DOF Unitree TorchScript policy, samples the
sensorized model at its native 50 Hz policy rate, and publishes snapshots through
`TelemetryFeed`. It remains an observational integration: the command does not
route a Sherpa guard action back to the locomotion policy, because the policy's
zero velocity command marches in place rather than honoring `REQUEST_HOLD`.
This prevents an unsupported safety claim while still exposing live walking data
to external consumers.

## 2026-08-29 — Himalayan context is resolved offline on walking-feed activation

`sherpa walk --waypoint <name>` resolves a named EBC waypoint from the bundled,
versioned route artifact once when the feed is activated. Its location, elevation,
slope, route segment, nearest-safe-waypoint distance, and exposure classification
are published under `environment.himalaya` with every current snapshot. This is
static route context, not a claim of live GPS, weather, or online terrain lookup;
the runtime remains network-free and simulator truth remains excluded.

## 2026-08-29 — Walking battery range is a display-only model, not a battery gauge

The Unitree walking model does not expose an onboard battery gauge. `sherpa
walk` therefore publishes `battery.range_model` only when explicitly enabled
by its local `DerivedState` model. The model uses observed joint work, a fixed
500 Wh nominal pack, fixed idle load, configured initial charge, activated
ambient temperature, and commanded velocity as a speed assumption. It is
separately labelled from `battery.gauge`, never changes `RobotTelemetry`, and
has no return path into guard or locomotion decisions. This supplies usable
display context without claiming a measured battery state.

## 2026-08-29 — LLM telemetry reports evidence and data gaps, not safety actions

`TelemetryFeed` publishes `decision_context` and prioritizes it in `/llm`.
It summarizes stability and available energy evidence alongside an explicit
list of absent decision inputs. This supports external explanation and human
review while preventing an LLM from mistaking missing actual speed, contact,
raw battery-gauge, or calibrated electrical-power data for nominal evidence.
The deterministic SherpaOS guard/policy path remains the only action source.

## 2026-08-29 — Walking auxiliary sensors are simulated display inputs

The Unitree walking rollout may publish simulated base speed, foot contact/load,
electrical power, and battery gauge values through `--simulate-auxiliary`.
Those values are passed only to `TelemetryFeed`, with `simulated:*` provenance;
they are not inserted into frozen `RobotTelemetry` or presented to a guard or
locomotion policy. This permits a complete telemetry demonstration while
preserving the real-versus-simulated evidence boundary.

## 2026-08-29 — Walking terrain uses a physical MuJoCo incline from route grade

`sherpa walk --uphill` rotates the MuJoCo floor geometry about world Y so the
Unitree policy walks uphill along positive world X. The positive slope from the
selected EBC route waypoint is used once at activation and emitted as
`environment.terrain_simulation.uphill_slope_deg` with simulated provenance.
This is a controlled uniform-grade terrain test, not a claim that the local
waypoint artifact reconstructs the Himalayan surface.

## 2026-08-29 — External weather is a bounded, display-only activation lookup

`sherpa walk` fetches a single Open-Meteo snapshot for the selected route
waypoint before the policy loop begins. The API marks the data with its external
source and timestamps under `environment.weather`; failure is represented
explicitly and does not block walking. No weather response is converted into
`RobotTelemetry`, sent to a guard, or used by policy control. `--no-weather`
preserves a fully offline walking activation.

## 2026-08-29 - Frozen Zealot v26 policy is the Himalayan visualization lane

The playground wraps g1_v26_iter42290.onnx as a frozen controller using its exact 5x48 observation and 12-action contract. SherpaOS neither trains locomotion nor intervenes. Weights stay outside Git and are accepted only at the pinned SHA-256. The full MuJoCo Menagerie G1 is used for rendering.

Foreground mountain silhouettes were rejected because they occluded the robot. Terrain severity is collision geometry, and a segmentation-based minimum visibility gate is mandatory.

## 2026-08-29 — All deployable services run on Vultr

The user explicitly rejected Vercel for this product. The existing Vultr host is the
single deployment target for the Python expedition API, local ROS-bag storage and
processing, OpenAI-backed reflection/voice worker, Next.js Field Journal, and generated
diary/audio artifacts. Hugging Face is also removed from the current deployment path.

This is a deployment simplification, not a change to the runtime safety architecture.
The LLM remains strictly post-mission and cannot participate in guard fusion or
actuation. A general Vultr account API key is not required when services read the local
instance filesystem. `OPENAI_API_KEY` remains server-only.

**Rejected alternative:** Vercel frontend plus a remote Vultr worker. Rejected because
the team wants one host, one artifact boundary, and no cross-provider operational
dependency during the demo.

## 2026-08-29 — Field Journal is the approved presentation UI

The repository now contains `web/`, a Next.js Field Journal with an original animated
grayscale Everest relief, central ascent route, clickable five-day camps, written past
entries, current-day speech, locked future days, Pemba reactions, snow, parallax,
collectible memories, and reduced-motion support. The UX uses mock day fixtures until
the API connection is completed.

The design follows the spatial grammar of Robot Everest (front-facing relief, luminous
route, altitude/camp progression) without copying its proprietary assets or source.
Unity was rejected in favor of web-native SVG/CSS/Motion because the experience needs
fast browser startup, responsive text, and straightforward deployment from Vultr.
## 2026-08-30 — Live demo renders runtime evidence, never video labels

The Vultr supervisor publishes each GuardDecision together with its matching
ActuationReceipt and requested/applied velocities. The browser consumes this
separate WebSocket stream beside the raw robot stream. It shows NO LIVE EVIDENCE
when the runtime is absent and has no code path that fabricates GO/CAUTION/NO-GO.
Navigation order is Live Demo, Radio, Journal.
