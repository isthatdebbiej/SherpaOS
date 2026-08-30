# SherpaOS risk data generation plan

**Status:** execution plan  
**Time budget:** 6–9 hours for generation/training, then 2 hours for inference/demo  
**Primary controller:** frozen Zealot G1 v26 `iter42290`  
**Learned outputs:** mobility risk and dynamics/body risk only

## Objective

Generate a physically meaningful, leakage-safe dataset in which the full Unitree G1 is
commanded through icy and inclined terrain. Train a compact temporal model to predict
mobility and dynamics risk during the following second from the preceding two seconds of
onboard-observable telemetry.

SherpaOS does not control joints or intervene during training-data generation. The
frozen locomotion policy produces joint targets; MuJoCo supplies dynamics, contacts,
friction, terrain, and disturbances.

Guards 3–5 remain deterministic:

1. learned mobility risk;
2. learned dynamics/body risk;
3. deterministic telemetry health;
4. deterministic battery margin;
5. deterministic geographic/environmental risk.

Inputs and reports for guards 3–5 are synchronized and preserved for full-supervisor
evaluation, but they are not inputs or training targets for the temporal model.

## Non-negotiable terrain behavior

The robot must walk onto and across the hazards. It must not path-plan around them.
Terrain collision geometry spans the traversal corridor.

The qualification course is:

```text
spawn
  -> packed snow
  -> low-friction ice
  -> 10–15 degree ascent
  -> 8–15 degree cross-slope
  -> uneven crust and rock step
  -> 18–25 degree recovery/failure boundary
```

A vertical wall is not a valid walking-policy test. Slopes above the policy's useful
range are failure-boundary experiments and must be labeled as such.

Every rendered qualification episode must prove:

- the G1 reaches the first ice segment;
- the feet contact the intended hazard geometry;
- the G1 climbs onto the inclined geometry;
- the video contains approach, contact, and response;
- the G1 remains visible with at least 5,000 segmented pixels and 120 px bounding-box
  height at every sampled visibility check;
- commanded and achieved forward progress are recorded;
- termination is explicitly `completed`, `physical_fall`, or `qualification_failure`.

## Frozen temporal contract

- Control and telemetry rate: 50 Hz.
- Maximum episode length: 500 control steps (10 seconds).
- Physical falls may terminate an episode early.
- Observation window: 100 samples (2 seconds).
- Prediction horizon: 50 samples (1 second).
- Window stride: 10 samples (0.2 seconds).
- Observation shape: `[window, 103]`.
- Model outputs: `mobility_risk`, `dynamics_risk`.

A complete episode produces approximately 36 overlapping windows. Two hundred complete
episodes produce about 7,200 windows; 1,000 produce about 36,000. These windows are
correlated and never count as independent episodes.

## Artifact separation

Each dataset is immutable after validation:

```text
artifacts/datasets/<dataset-id>/
  observations/
    shard-000.npz
  labels/
    shard-000.npz
  context/
    shard-000.npz
  videos/
    selected qualification videos only
  DATASET_CARD.md
  source_manifest.json
  scenario_manifest.json
  split_manifest.json
  quality_report.json
  checksums.sha256
```

### Observations

Only the frozen 103-field onboard schema:

- 29 joint positions;
- 29 joint velocities;
- 29 joint efforts plus presence flag;
- IMU orientation, angular velocity, and linear acceleration;
- commanded velocity plus presence flag;
- telemetry validity.

Observations never contain true friction, true slope, slip truth, actuator health,
disturbance identity/timing, fall truth, battery, GPS, terrain lookup, weather, or guard
outputs.

### Privileged labels

Store index-aligned evaluator truth separately:

- true friction and slope;
- planted-foot slip;
- actuator health;
- disturbance vector, onset, and duration;
- tilt and fall outcome;
- contacted terrain segment;
- command-tracking error and forward progress;
- hazard phase and termination reason.

The preferred targets are physical outcomes:

```text
mobility positive =
  unsafe planted-foot slip
  OR material traction-related command-tracking loss
  OR traction-related fall

dynamics positive =
  unsafe tilt/body response
  OR material actuator-response residual
  OR failed recovery
  OR dynamics-related fall
```

Scenario parameters explain an outcome but do not alone prove one. If the existing
parameter-based labels are retained for comparison, preserve them as versioned `v1`
labels and add outcome targets without silently changing semantics.

### Synchronized context for guards 3–5

Use the control-step timeline and preserve:

- episode ID, control step, sequence, and decision timestamp;
- telemetry validity and telemetry-health report;
- battery fraction, voltage, current, pack temperature, value timestamp, and age;
- simulated or GNSS position, validity, covariance, value timestamp, and age;
- route progress, elevation, DEM/route slope, resolution and provenance;
- exposure, distance to safety, wind, temperature, value timestamp, and age;
- score, confidence, reasons, and recommended action from guards 3–5.

Slower sources use last-known-valid joins with recorded timestamps and age. Missing
values are never interpolated and represented as if observed.

Freshness targets:

- joints/IMU/command: 250 ms;
- battery: 2 seconds;
- localization: 5 seconds;
- weather: 5 minutes;
- terrain lookup: update after material accepted position change.

## Scenario design

The first 200 episodes remain balanced:

| Category | Episodes |
|---|---:|
| Nominal and hard-negative | 50 |
| Mobility/traction | 50 |
| Dynamics/body | 50 |
| Combined | 50 |

Use five scenario groups per category and ten episodes per group. Groups describe
meaningful scenario families, not merely consecutive seeds.

Nominal groups cover forward walking, turning, lateral commands, stop/start, and
aggressive-but-safe traversal. Mobility groups cover flat ice, icy ascent/descent,
cross-slope ice, alternating patches, and rough low-traction crust. Dynamics groups
cover directional pushes, actuator degradation, payload/asymmetry, uneven steps, and
aggressive safe recovery. Combined groups compose these hazards with randomized order.

Disturbance onset must be reachable inside 500 steps. Use approximately steps 200–350
with duration 15–60 steps, leaving adequate pre-hazard history and recovery time.

The deterministic split is category-stratified by scenario group:

```text
per category: 3 train groups, 1 validation group, 1 test group
total: 120 train, 40 validation, 40 test episodes
```

No seed, command trace, terrain family, episode, or near-identical environment may cross
splits. Windows inherit their episode's immutable membership.

## Staged generation

### Gate A: 20-episode v26 qualification

Generate five episodes per category. Capture four representative videos: nominal,
mobility, dynamics, and combined. Do not proceed unless:

1. nominal episodes walk meaningfully without persistent hazards;
2. ice produces measurable slip or tracking-response changes;
3. inclines produce measurable posture/load changes;
4. disturbances occur during the episode and change body dynamics;
5. actuator degradation changes measured joint response;
6. label onset is aligned with the future prediction horizon;
7. observations contain no privileged truth;
8. the visual terrain-contact and robot-visibility gates pass.

### Gate B: immutable 200-episode core

Generate headlessly in resumable ten-episode shards. Render only selected audit
episodes. Validate integrity, separation, balance, physical response, group isolation,
and checksums before training.

### Gate C: optional expansion to 1,000

Begin only after the 200-episode dataset and first model are healthy. Add 800 episodes
that expand scenario families rather than merely adding seeds, for a final balance of
250 episodes per category. Stop expansion if it threatens held-out evaluation or
inference integration.

Benchmark ten headless episodes first and calculate expected wall time. Run independent
shards across CPU workers while preserving deterministic seeds. Never render all 1,000
episodes.

## Geographic, GPS, and depth scope

The existing route artifact may drive deterministic simulated route progress today. A
small offline Copernicus GLO-30 or NASADEM clip is an optional strategic context layer:
crop once, reproject locally, derive elevation/slope, and freeze its provenance and
checksum. A 30 m DEM is not foot-scale collision geometry and cannot prove ice,
crevasses, rocks, or traversability.

Simulation may emit `NavSatFix`-compatible position at 1–5 Hz with timestamp, WGS84
position, status, covariance, frame, and explicit simulated provenance. Inject dropout,
staleness, high covariance, frozen fixes, and jumps for geographic-guard evaluation.
Live ROS 2 integration is not required for model training.

Depth is auxiliary evidence initially. Record it at 5–10 Hz only for selected episodes,
with timestamp, frame, intrinsics, range, and validity. Do not add depth to the 103-field
model or safety decision until a calibrated adapter and traversability contract exist.

## Quality report required before training

Report per category, group, and split:

- episode count, completion count, falls, and termination reasons;
- mobility/dynamics-positive episodes and windows;
- nominal false-hazard duration;
- slip distribution by friction, slope, and terrain contact;
- tilt/body response by disturbance and actuator condition;
- commanded versus achieved velocity/progress;
- recovery rate and early termination rate;
- label onset relative to hazard and fall;
- invalid, missing, duplicate, stale, and non-finite samples;
- controller, simulator, policy, asset, config, and code hashes.

Do not train if pilot gates 1–4 fail. Do not expand if the 200-episode model shows no
held-out improvement from added scenario diversity.

## Training plan

Train in increasing complexity:

1. existing deterministic feature baseline;
2. separate logistic-regression baselines over window statistics;
3. compact TCN with input `[batch, 100, 103]`, channels `[32, 32, 16]`, kernel 5,
   dropout 0.10, and two logits.

Use batch size 128, learning rate 0.001, at most 25 epochs, patience 4, gradient clipping
at 1.0, and output-specific class weighting computed from training groups only. Run one
seed first; run seeds 2 and 3 only after the first beats baseline without severe
overfitting.

Normalization statistics and class weights come from train groups only. Select and
calibrate on validation groups only. Open test results once for the shipping decision.

Report average precision, AUROC, recall at the selected operating point, false alarms
per nominal minute, early-warning lead time, Brier score/calibration error, and metrics
by held-out scenario family. Accuracy alone is not a shipping metric.

## Compute and deployment environments

The required execution path is:

```text
Vultr CPU
  MuJoCo v26 episode generation and validation
          ->
Hugging Face dataset repository
  immutable validated shards and manifests
          ->
Hugging Face Jobs
  baseline/TCN training, calibration, held-out evaluation, and batch inference
          ->
Hugging Face model repository
  model, normalization, schema, thresholds, reports, and checksums
```

Hugging Face batch inference is a required acceptance environment. Run it as a finite Job
against frozen test/replay artifacts, not as an always-on public endpoint. The Job must
pull immutable dataset/model revisions, verify checksums, run CPU and/or GPU inference,
and upload a signed-off inference report containing predictions, metrics, latency,
hardware identity, code SHA, model revision, and artifact checksums.

A managed Hugging Face Inference Endpoint or Space is optional for a live cloud demo and
must never be the only way SherpaOS can infer. The exported ONNX model remains runnable
offline. Jetson AGX Thor is an optional later deployment benchmark; lack of Thor access
does not block the Hugging Face training and inference acceptance path.

Required Hugging Face model bundle:

```text
model.onnx
model.pt
normalization.json
schema.json
calibration.json
thresholds.json
training_manifest.json
evaluation_report.json
inference_report.json
checksums.sha256
MODEL_CARD.md
```
## Runtime inference

Maintain a 100x103 circular buffer, infer every ten valid samples, apply the frozen
training normalization, calibrate both probabilities, and fuse learned guards 1/2 with
deterministic guards 3/4/5. Before warm-up or on schema mismatch, NaN, staleness, model
failure, or unexpected output, inference must not emit a confident pass.

The browser demo may send bounded `[vx, vy, yaw]` commands over a WebSocket to the frozen
v26 policy and stream video/telemetry/risk back through an SSH-tunneled service. It may
not expose direct joint control. Interactive sessions are demo/evaluation evidence, not
training episodes.

## Execution schedule

| Time | Work and exit condition |
|---|---|
| 0:00–1:00 | Fix v26 generation, reachable hazards, outcome labels, and contact gates; pass 20-episode qualification. |
| 1:00–3:00 | Generate and validate immutable 200-episode core; begin baseline and first TCN. |
| 3:00–5:00 | Audit results, add validated scenario diversity, optionally generate toward 1,000. |
| 5:00–7:00 | Train up to three seeds in Hugging Face Jobs, calibrate, and evaluate held-out groups. |
| 7:00–9:00 | Freeze the Hub model bundle; run Hugging Face batch inference, robustness tests, and optional geographic upgrade. |
| Inference 0:00–1:00 | Rolling buffer, runtime adapter, parity, latency, and fail-conservative behavior. |
| Inference 1:00–2:00 | Five-guard fusion, browser command demo, tunneled viewing, and recorded fallback. |

## Priority and stop rules

Preserve this order when time is constrained:

1. physical traversal and measurable terrain response;
2. correct future labels and leakage prevention;
3. independent groups and valid 200-episode core;
4. trustworthy baseline/TCN evaluation;
5. fail-conservative runtime inference;
6. expansion toward 1,000;
7. simulated GPS and offline DEM;
8. depth evidence;
9. browser polish.

Do not trade evaluation time for episode count. Do not train guards 3–5 merely because
their synchronized inputs exist. Do not call visual snow an ice experiment without
low-friction contact and a measurable physical response.
