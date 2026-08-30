# SherpaOS

SherpaOS is an offline, auditable mobility-risk supervisor for a Unitree G1. It runs above an existing locomotion controller, evaluates onboard-observable telemetry and local route context, and produces one of three bounded decisions:

- `PASS`: continue with the requested motion;
- `LIMIT_SPEED`: continue at a reduced velocity;
- `REQUEST_HOLD`: stop and hold position.

SherpaOS does not train or replace the locomotion controller. Its role is to detect deteriorating mobility, unstable body dynamics, unhealthy telemetry, insufficient battery margin, and geographic or environmental risk. Every applied decision is paired with an actuation receipt and stored evidence so the result can be reconstructed later.

## Why this problem matters

A legged robot operating far from immediate recovery may encounter changing traction, body disturbances, actuator degradation, sensor faults, low battery margin, steep terrain, wind, and cold. These conditions are only partially observable from the robot. A safety system must therefore:

- operate from measurements available onboard rather than privileged simulator state;
- react locally without requiring a network or language model;
- treat missing, stale, malformed, or contradictory data conservatively;
- prevent one severe risk signal from being averaged away by nominal signals;
- record the requested response and the response actually applied;
- expose enough evidence for later diagnosis and operator review.

SherpaOS implements and evaluates this supervisory boundary in MuJoCo. Physical G1 validation remains future work.

## System overview

```text
Frozen locomotion policy
        |
        v
MuJoCo Menagerie G1 + terrain + disturbances
        |
        +---- privileged simulator truth ----> labels and evaluation only
        |
        v
Onboard-observable telemetry at 50 Hz
        |
        +---- temporal mobility/dynamics model
        |
        +---- deterministic guard reports
                1. mobility
                2. dynamics/body
                3. telemetry health
                4. battery margin
                5. geographic/environment
        |
        v
Conservative fusion + hysteresis
        |
        v
PASS / LIMIT_SPEED / REQUEST_HOLD
        |
        v
Applied velocity or hold + actuation receipt + evidence log
        |
        +---- Live Demo
        +---- Journal
        +---- grounded Radio Q&A
```

The fast safety path is local and independent of the web application, voice services, and network connectivity. Journal and Radio consume recorded evidence after or between decisions; they cannot authorize motion or modify a safety decision.

## Runtime inputs

The learned motion-risk model receives a two-second window containing 100 samples at 50 Hz. Each sample contains 103 onboard-observable values:

| Feature family | Values |
|---|---:|
| Joint positions | 29 |
| Joint velocities | 29 |
| Joint efforts | 29 |
| Effort-present flag | 1 |
| Base orientation quaternion | 4 |
| Base angular velocity | 3 |
| Base linear acceleration | 3 |
| Commanded velocity | 3 |
| Command-present and telemetry-valid flags | 2 |
| **Total** | **103** |

The deterministic supervisor also consumes context that is intentionally kept outside the motion model:

- battery state of charge, voltage sag, current, pack temperature, and time margin when available;
- offline route context, slope, exposure, distance to safety, and localization freshness;
- current and forecast wind and ambient temperature;
- message timestamps, sequence information, completeness, and validity.

Joint effort is zero-filled only when unavailable, and an explicit flag distinguishes unavailable effort from a measured zero value.

### Privileged data boundary

Simulator-only values are physically separated from observation artifacts. True friction, contacted slope, slip truth, actuator-health truth, disturbance identity, fall truth, and scenario labels may be used to construct labels or evaluate outcomes, but they cannot enter runtime features. Dataset validation rejects privileged fields, non-finite observations, incorrect feature widths, corrupt checksums, duplicate episode IDs, and split overlap.

## Decisions and evidence

Five guard families produce independent reports with risk scores, confidence, and reason codes. Fusion selects the most conservative applicable action floor; a severe guard cannot be cancelled by lower scores from other guards. Hysteresis prevents rapid state changes around a threshold.

The actuation adapter applies the requested velocity limit or hold and returns an `ActuationReceipt` tied to the originating decision ID. Evidence records include:

- the decision and contributing guard reports;
- telemetry freshness and validity;
- requested and applied velocity;
- whether the action was accepted;
- the acknowledgement source;
- timestamps, model version, and reason codes.

REST and WebSocket endpoints expose the latest supervisor evidence and recent event history to the Live Demo. The visual robot feed is a selected simulation replay; the live decision stream is produced by a separate continuously running MuJoCo supervisor and is not frame-synchronized with that replay.

## Simulation and dataset generation

The primary simulation lane uses the frozen Zealot G1 v26 `iter42290` ONNX policy with its exact 240-input, 12-action contract at 50 Hz. SherpaOS did not train or fine-tune this locomotion policy. A pinned Unitree `unitree_rl_gym` checkpoint is retained as a fallback and reference path.

Scenarios vary:

- connected collision terrain and physically contacted slope;
- friction and mixed-traction surfaces;
- command trajectories;
- external pushes and body disturbances;
- actuator degradation;
- sensor noise and data-quality faults;
- battery state and environmental context;
- persistent wind and weather envelopes.

Rendered snow and particles are visual only. Traction comes from collision geometry and friction. Videos are used for qualitative inspection of robot visibility, physical contact, disturbance timing, and label plausibility; video pixels are not model inputs.

The packaged dataset contains two independently generated 200-episode cohorts. Each cohort contains 50 nominal, 50 mobility/traction, 50 dynamics/body, and 50 combined episodes. Episodes are generated deterministically in resumable shards of ten and run for at most 500 control steps unless a physical fall ends the rollout.

The combined 400-episode package is split before windows are loaded:

- 240 training episodes;
- 80 validation episodes;
- 80 locked test episodes.

Membership is assigned by scenario group and episode. Overlapping windows from one episode cannot cross splits. Intervention is disabled during training-data generation so unsafe trajectories are not censored by the supervisor that will later learn to detect them.

## Risk-model training

The trained risk estimator is a compact residual temporal convolutional network with a shared encoder and two output heads:

- mobility/traction risk;
- dynamics/body risk.

It predicts risk over the next one-second horizon from the preceding two-second observation window. The network uses three causal-style one-dimensional convolution stages with dilations 1, 2, and 4, residual connections, batch normalization, GELU activations, and dropout.

Training uses weighted binary cross entropy, AdamW, deterministic seed `20260830`, gradient clipping, early stopping, and normalization statistics computed from the training split only. Decision thresholds are selected on validation data to minimize false positives while retaining at least 90% recall.

Current validation results:

| Head | Average precision | AUROC | Brier | Threshold | Recall | False-positive rate | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mobility | 0.559 | 0.904 | 0.108 | 0.270 | 0.903 | 0.264 | 0.307 |
| Dynamics | 0.911 | 0.977 | 0.056 | 0.705 | 0.901 | 0.048 | 0.792 |

The learned model is not the sole authority for actuation. Deterministic guards and action floors remain active, particularly because the mobility head's validation false-positive rate is still high.

- Dataset: `isthatdebbiej/sherpaos-himalayan-risk-400-balanced-v2`
- Model: `iteratehack/sherpaos-risk-tcn-balanced-v2`

## Operator interfaces

The Next.js application contains three views:

- **Live Demo** displays the G1 simulation replay, current five-guard reports, requested and applied velocity, decision IDs, receipts, and recent events.
- **Journal** presents mission memories derived from immutable simulation evidence and separates factual records from expressive reflection.
- **Radio** accepts operator speech, retrieves selected Journal evidence and the latest supervisor evidence, and returns a grounded spoken response.

The deployed Radio pipeline uses `openai/whisper-large-v3` for transcription through Hugging Face Inference Providers, `meta-llama/Llama-3.1-8B-Instruct` for grounded answers, and `facebook/mms-tts-eng` for local speech synthesis. Radio has no connection to the locomotion or safety-control path.

## Repository layout

```text
configs/                    scenario, terrain, split, and training contracts
sherpaos/contracts.py       runtime telemetry and decision types
sherpaos/sim/               G1 controller, sensors, terrain, weather, and supervisor
sherpaos/estimator/         deterministic and learned risk features
sherpaos/policy/            five-guard fusion and hysteresis
sherpaos/recorder/          incident recording and store-and-forward queue
sherpaos/evidence/          evidence bundles and manifests
sherpaos/datasets/          generation, labels, splitting, packaging, validation
sherpaos/runtime/           live evidence bus, REST/WebSocket, grounded voice API
sherpaos/expedition/        immutable MCAP/rosbag2 ingestion and inspection
scripts/                    training, rendering, packaging, and Vultr workflows
web/                        Next.js operator interface
tests/                      unit, property, integration, and leakage tests
```

## Installation

Python 3.12 and `uv` are required for the core project. Node.js and npm are required for the web application.

```bash
uv sync --extra dev
uv run sherpa preflight
```

The frozen locomotion checkpoints and terrain artifacts are verified against pinned hashes during their respective setup workflows. See `docs/V26_HIMALAYA_PLAYGROUND.md` for the v26 controller setup.

## Running locally

Run the offline end-to-end demonstration:

```bash
uv run sherpa demo --offline
```

Run a scenario:

```bash
uv run sherpa simulate --scenario nominal --seed 42
```

Run the test suite and static checks:

```bash
uv run ruff check .
uv run pytest
```

Generate and validate a small dataset-contract sample:

```bash
uv run sherpa data generate \
  --matrix configs/scenario_matrix.yaml \
  --episodes 2 \
  --output artifacts/datasets/pilot-contract

uv run sherpa data validate \
  --dataset artifacts/datasets/pilot-contract
```

Run the web application:

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000`.

Operational commands and deployment steps are documented in `docs/RUNBOOK.md`. Data contracts and acceptance gates are documented in `docs/RISK_DATA_REQUIREMENTS.md` and `docs/CONTRACTS.md`.

## Deployment

The current deployment target is a Vultr host running:

- the continuously running MuJoCo supervisor;
- the live evidence REST/WebSocket API;
- the Next.js application;
- Journal artifacts and voice services;
- immutable uploaded bag storage and inspection.

The local safety loop must continue operating if the network, web application, or voice service is unavailable. In a future field deployment, compact decisions and receipts should be transmitted before larger telemetry chunks, while full recording continues locally during communication loss.

## Current status

Implemented and verified:

- frozen G1 locomotion integration and sensorized MuJoCo runtime;
- five deterministic guard families and conservative fusion;
- applied speed-limit/hold behavior with matching receipts;
- leakage-resistant dataset generation, validation, and packaging;
- 400-episode balanced training package and trained two-head TCN;
- live evidence REST/WebSocket API;
- immutable `.mcap` and rosbag2 `.db3` upload and manifest inspection;
- Live Demo, Journal, and Radio web views;
- offline CLI demonstration and automated test coverage.

## Unfinished work

- Run and publish the locked 80-episode test evaluation. It has not been used for model selection or threshold tuning.
- Reduce mobility-head false positives before considering learned mobility risk as a primary authority.
- Collect and ingest physical G1 ROS 2 bags; current mission memories and model results are simulation-derived.
- Validate sensor calibration, inference latency, hardware timing, actuator acknowledgement, and fail-operational behavior on a physical G1.
- Evaluate sim-to-real transfer and recalibrate thresholds using controlled hardware tests.
- Replace rigid, non-deformable snow with validated terrain/contact models where required.
- Add foot-scale local terrain perception for ice, holes, and crevasses; the current DEM and route context are coarse.
- Synchronize the displayed robot stream with the live supervisor decision stream if frame-level visual correspondence is required.
- Complete decoded common-clock rows, gap/rate/overlap audits, and evidence-linked event intervals for long-duration ROS bag reflection.
- Connect all Journal views to persisted backend artifacts rather than frontend fixtures.
- Complete production service management and reverse-proxy configuration for the Vultr deployment.

This repository is a simulation-validated safety-supervision system and evidence pipeline. It is not a field-qualified safety controller or a certification artifact.
