# DECISIONS.md — timestamped decisions and rejected alternatives

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

## 2026-08-29 - Frozen Zealot v26 policy is the Himalayan visualization lane

The playground wraps g1_v26_iter42290.onnx as a frozen controller using its exact 5x48 observation and 12-action contract. SherpaOS neither trains locomotion nor intervenes. Weights stay outside Git and are accepted only at the pinned SHA-256. The full MuJoCo Menagerie G1 is used for rendering.

Foreground mountain silhouettes were rejected because they occluded the robot. Terrain severity is collision geometry, and a segmentation-based minimum visibility gate is mandatory.
