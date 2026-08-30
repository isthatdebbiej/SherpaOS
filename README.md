# SherpaOS

**An offline, auditable risk supervisor for a Unitree G1 operating beyond easy rescue.**

SherpaOS does not teach a robot to walk. It wraps a frozen open-source locomotion policy, watches onboard-observable telemetry, estimates whether traction or body stability is deteriorating, and combines that evidence with deterministic telemetry, battery, and geographic/environment guards. The result is an inspectable `PASS`, `LIMIT_SPEED`, or `REQUEST_HOLD` decision with reason codes and an actuation receipt.

> **Fast thinking keeps the robot safe in the moment. Slow thinking helps the team learn from the journey.**

Live judge demo: <https://96.30.206.111.nip.io>

## What judges can see

The web application has three operator-facing views:

1. **Live Demo** — a Himalayan G1 visual feed beside real, continuously produced MuJoCo supervisor decisions, five guard reports, requested versus applied velocity, decision IDs, and actuation receipts.
2. **Radio** — a voice-only HTTPS demonstration. Tap the microphone, speak, and hear Pemba answer from verified journal memory plus the latest supervisor evidence.
3. **Journal** — four mission-day narratives derived from immutable simulation evidence: nominal, mobility/traction, dynamics/body, and combined stress.

The current Live Demo is deliberately described as a hybrid: the robot image is a selected simulation replay, while the decision stream is produced live by a separate continuously running deterministic MuJoCo supervisor. They are not frame-synchronized, and the trained Hugging Face TCN is not yet connected to the displayed episode. Text burned into a video is never accepted as proof that the supervisor acted. A decision is credible only when the API publishes its decision ID, guard evidence, requested action, and matching adapter receipt.

## System architecture

```text
Frozen OSS locomotion policy
(v26 iter42290 ONNX; 12 actions at 50 Hz)
               |
               v
MuJoCo Menagerie G1 + physical terrain + disturbances
               |
               +---- evaluator-only truth --------------------+
               |   friction, contacted slope, slip, fall,     |
               |   disturbance identity (labels only)          |
               |                                                v
               v                                         offline training
103 onboard-observable motion features                two-head residual TCN
               |                                                |
               +---------------- fast safety path <-------------+
               |
        five independent guards
        1. mobility risk
        2. dynamics/body risk
        3. telemetry health
        4. battery margin
        5. geographic/environment risk
               |
       conservative max/action-floor fusion
               |
      PASS / LIMIT_SPEED / REQUEST_HOLD
               |
        actuation receipt + evidence log
               |
               +---------------- slow learning path -----------+
                    immutable run history, Journal, Radio,
                    operator review, later ROS-bag ingestion
```

The LLM and voice system have no path to the joint controller. They explain recorded evidence; they do not create safety truth or authorize motion.

## Fast thinking: the safety loop

The 50 Hz runtime is intentionally small and inspectable:

- **Mobility** looks for loss of traction and slip-like motion.
- **Dynamics/body** looks for orientation instability, anomalous body motion, and asymmetry.
- **Telemetry health** rejects malformed, stale, future-dated, duplicated, or incomplete sensor streams.
- **Battery margin** evaluates charge, cold derating, voltage sag, current, pack temperature, and time margin when those measurements are available.
- **Geographic/environment** evaluates offline route context, slope, exposure, distance to safety, localization freshness, wind, and temperature.

Fusion is conservative: one severe guard cannot be averaged away by four nominal guards. Hysteresis prevents rapid GO/NO-GO oscillation. Every applied action receives an `ActuationReceipt` tied to the originating `decision_id`.

The learned TCN is not yet the sole authority for actuation. Its validation is promising, but mobility false positives remain too high for an unsupported deployment claim. Deterministic guards and action floors remain the operational safety boundary.

## Slow thinking: mission memory

The slow path exists for review and learning after or between traversals:

- preserve checksummed observations, labels, context, decisions, and receipts;
- summarize why risk rose, when speed was limited, and what physical outcome followed;
- ground operator Q&A in stored evidence and the latest live decision;
- compare cohorts and identify missing coverage before generating more data;
- eventually ingest immutable G1 ROS 2 bags instead of simulation-only mission memories.

Today, the four Journal days are derived from simulation train/validation evidence. They are not represented as real Himalayan deployments. The repository already accepts immutable `.mcap` and rosbag2 `.db3` uploads, computes SHA-256, inspects topic/message/time coverage without mutating the bag, and rejects privileged-topic leakage. Full physical G1 ingestion and field validation remain future work.

## Models and services

| Role | Model or implementation | What we did | Safety authority |
|---|---|---|---|
| Locomotion, primary simulation lane | Zealot G1 v26 `iter42290`, ONNX | Frozen checkpoint, exact 240-input/12-action contract, 50 Hz, pinned SHA-256 | Produces joint targets only |
| Locomotion fallback/reference | Unitree `unitree_rl_gym` G1 `motion.pt` | Pinned TorchScript policy and matching 12-DOF MuJoCo configuration | Not trained by SherpaOS |
| Learned risk | SherpaOS two-head residual TCN | Trained on Hugging Face Jobs from 400 simulated episodes | Mobility/dynamics evidence; not sole actuation authority |
| Speech recognition | `openai/whisper-large-v3` | Called through Hugging Face Inference Providers | None |
| Grounded Radio answer | `meta-llama/Llama-3.1-8B-Instruct` | Receives only selected journal evidence and latest live decision | None |
| Speech synthesis | `facebook/mms-tts-eng` | Runs locally on Vultr CPU and returns WAV audio | None |

SherpaOS never trained or fine-tuned locomotion. The only model trained by this project is the compact temporal risk model.

## What the risk model sees

Each training example is a two-second window: **100 control samples at 50 Hz**, each containing **103 onboard-observable values**:

| Feature family | Width |
|---|---:|
| 29 joint positions | 29 |
| 29 joint velocities | 29 |
| 29 joint efforts (zero-filled only when unavailable) | 29 |
| effort-present flag | 1 |
| base orientation quaternion | 4 |
| base angular velocity | 3 |
| base linear acceleration | 3 |
| commanded velocity | 3 |
| command-present and telemetry-valid flags | 2 |
| **Total** | **103** |

The model predicts two probabilities over the following one-second horizon (50 control samples):

- mobility/traction risk;
- dynamics/body risk.

Windows advance by 10 samples. Because adjacent windows overlap, splitting individual windows would leak nearly identical motion across train and validation. Membership is therefore immutable and assigned by scenario group/episode before windows are loaded.

### What the model never sees

Observation artifacts are physically separated from privileged labels. Validation fails if an observation contains true friction, true/contacted slope, slip truth, actuator health, disturbance identity, fall truth, NaN/Inf, the wrong feature width, duplicated episode IDs, corrupt checksums, missing shards, or split overlap.

Battery, GPS/geographic context, forecast wind, and deterministic guard outputs are stored in separate context artifacts. They are useful to the supervisor and for full-system evaluation, but they do not leak into the 103-feature motion TCN.

## Dataset and simulation method

The production collection contains two independently generated 200-episode cohorts: a higher-slope Himalayan stress cohort and a later 10–15 degree cohort. Each 200-episode cohort contains:

- 50 nominal episodes;
- 50 mobility/traction episodes;
- 50 dynamics/body episodes;
- 50 combined episodes.

Generation is deterministic, controller-only, resumable in shards of 10, and capped at 500 control steps unless a physical fall ends the episode. SherpaOS intervention is disabled during training-data generation so the supervisor cannot censor the unsafe trajectories it must learn to anticipate.

The balanced Hugging Face package contains **400 episodes** with immutable group-level membership:

- 240 train;
- 80 validation;
- 80 locked final test.

The final 80-episode test split was not opened during model selection or threshold tuning.

### Physics and environment

The frozen v26 controller drives the full MuJoCo Menagerie G1. Scenarios vary connected collision terrain, contacted slope, friction, command trajectory, pushes/body disturbances, actuator degradation, sensor noise, battery state, route context, and persistent weather envelopes.

Snow rendering is qualitative; traction comes from collision geometry and friction, not pixels. Wind is a time-continuous physical force with a slowly varying envelope rather than implausible frame-to-frame jumps. Current and forecast wind are stored separately. Extreme scenarios include winds near 200 km/h, but only the explicitly configured extreme family uses that envelope.

The v6q terrain qualification uses connected multi-grade surfaces. One qualifying mobility episode physically contacted 10- and 16-degree ice segments before falling. Authored 22- and 30-degree geometry was not contacted and is therefore not claimed as training exposure. The low-slope cohort is capped and qualified separately at 15 degrees.

Video does not train the TCN. It is synchronized qualitative QA used to check robot visibility, terrain contact, disturbance timing, weather rendering, recovery/fall behavior, and label plausibility. Machine-readable telemetry and manifests remain the source of truth.

## The v1–v6 journey

I started this project without a robotics background. The difficult part was not making one dramatic video; it was learning to reject simulations and datasets that looked persuasive but could not support an honest safety claim.

### v1 — a pipeline, not yet a trustworthy experiment

The first milestone established the 200-episode contract, separate observation/label shards, deterministic seeds, checksums, resumability, and group splits. It proved data engineering, but the simple controller and flat-looking scenes did not demonstrate competent locomotion or Himalayan terrain.

**Lesson:** reproducibility is necessary, but reproducibly generating weak data is still weak data.

### v2 — frozen locomotion instead of training the wrong model

We stopped trying to make SherpaOS responsible for locomotion and wrapped a pretrained G1 policy. We evaluated Unitree’s official `motion.pt` path and chose the downloadable v26 ONNX controller for the main terrain lane because its contract was small, stable, and reproducible.

**Lesson:** keep locomotion frozen and spend the project budget on risk supervision.

### v3 — a visible robot is an evidence requirement

Early camera angles cropped arms or hid the robot behind foreground mountains. We moved to the full Menagerie G1 renderer and added segmentation-based gates for minimum robot pixels, body height, and border margin. A render now fails rather than silently publishing an occluded robot.

**Lesson:** a beautiful landscape is useless evidence if the robot’s body response cannot be inspected.

### v4 — visual roughness was not physical terrain

Early snow/ice looked Himalayan but remained too horizontal. We replaced decorative relief with connected collision segments, multi-grade ice/snow/rock profiles, exact contacted-geometry slope truth, low-friction surfaces, and qualification checks requiring the policy to physically reach the intended grade.

**Lesson:** visual slope, route-context slope, and contacted physical slope are three different quantities and must never be conflated.

### v5 — storms had to obey time and physics

Initial extreme-weather experiments risked unrealistic wind changes. We introduced persistent wind envelopes, bounded gust evolution, aerodynamic force, separately recorded current/forecast wind, blowing-snow rendering, and labeled HUD videos. Nominal weather stays nominal; extreme weather is isolated to explicit stress scenarios.

**Lesson:** a “200 km/h” label is not evidence unless the force, telemetry, timing, and visuals agree.

### v6 — causal warning data and honest qualification

We diversified command traces and hazard onset, labeled future risk rather than copying scenario parameters, introduced recovery and failure boundaries, required physical terrain contact, generated a second lower-slope cohort, and audited positive rates. We also found that our first packaged validation split was mostly nominal: its mobility AUROC looked acceptable while mobility false-positive behavior was poor. We did not hide that result. We reindexed only by scenario group—without changing payloads or opening test data—and retrained on balanced hazard coverage.

**Lesson:** high aggregate metrics can be gamed by a bad split; scenario-group isolation and per-cohort metrics matter more than a flattering headline.

## Risk-model training

Training ran as a reproducible Hugging Face Job, not in a Space. The job downloaded the immutable private dataset package, trained, evaluated validation only, and uploaded the model artifacts.

Dataset: `isthatdebbiej/sherpaos-himalayan-risk-400-balanced-v2`
Model: `iteratehack/sherpaos-risk-tcn-balanced-v2`

The model is a residual temporal convolutional network:

- input shape `100 x 103`;
- causal-style 1D convolutions with kernel size 5 and dilations 1, 2, and 4;
- channels 32, 32, and 16;
- batch normalization, GELU, residual/skip connections, and 0.1 dropout;
- shared temporal encoder with a two-logit mobility/dynamics head;
- weighted binary cross entropy from natural training prevalence;
- AdamW, learning rate 0.001, batch size 128;
- deterministic seed `20260830`;
- at most 25 epochs, gradient clipping at 1.0, early stopping patience 4;
- normalization computed from train only;
- thresholds selected on validation by minimum false-positive rate while retaining at least 90% recall.

### Validation result

| Head | Average precision | AUROC | Brier | Threshold | Recall | False-positive rate | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mobility | 0.559 | 0.904 | 0.108 | 0.270 | 0.903 | 0.264 | 0.307 |
| Dynamics | 0.911 | 0.977 | 0.056 | 0.705 | 0.901 | 0.048 | 0.792 |

These are validation metrics, not field performance. Dynamics is strong for this simulated distribution. Mobility recall is useful, but a 26.4% false-positive rate is too high to make the learned mobility head the sole GO/NO-GO authority. The locked test remains unopened, and sim-to-real transfer is unproven.

## Radio today and in the field

The UI keeps the human metaphor **Radio**, but today it is voice over HTTPS:

```text
operator browser microphone
        -> HTTPS audio upload to Vultr
        -> Whisper speech-to-text
        -> evidence retrieval (Journal + latest decision/receipt)
        -> Llama grounded answer
        -> local MMS text-to-speech
        -> HTTPS WAV response to the operator
```

There is no claim of RF communication in the current demo.

A field deployment can preserve the same evidence contract while changing the transport:

```text
G1 onboard ROS 2
  joint_states / IMU / commands / battery / localization / diagnostics
        -> local fast supervisor (must remain safe if disconnected)
        -> rosbag2/MCAP recorder + signed/checksummed event log
        -> radio modem, private LTE/5G, Wi-Fi mesh, or satellite IP link
        -> operator backend on Vultr or an expedition base-station computer
        -> slow memory/indexing + grounded voice Q&A
```

The G1 should make immediate safety decisions locally; a network round trip must never be required to stop. When connectivity exists, it uploads compact live decisions and receipts first, then telemetry chunks or complete rosbag files. The backend verifies the bag hash, topic allowlist, timestamps, gaps, and provenance before Pemba can answer questions such as “Why did you stop?” If the link fails, local recording continues and sync resumes later.

A realistic ROS 2 source contract would include joint state, IMU, command, battery state, diagnostic validity/sequence, `sensor_msgs/NavSatFix` or fused localization with covariance, and calibrated local perception. DEM data supports strategic route risk; it cannot identify foot-scale ice or crevasses. Depth/local perception and dynamics remain necessary for nearby traversability.

## Reproduce locally

```bash
uv sync --extra dev
uv run sherpa preflight
uv run sherpa demo --offline
uv run pytest -q
```

Dataset contract run (the only small local generation expected):

```bash
uv run sherpa data generate \
  --matrix configs/scenario_matrix.yaml \
  --episodes 2 \
  --output artifacts/datasets/pilot-contract
uv run sherpa data validate --dataset artifacts/datasets/pilot-contract
```

Web application:

```bash
cd web
npm install
npm run dev
```

Frozen v26 Himalayan renderer and teammate setup are documented in [docs/V26_HIMALAYA_PLAYGROUND.md](docs/V26_HIMALAYA_PLAYGROUND.md). Operational commands are in [docs/RUNBOOK.md](docs/RUNBOOK.md), data requirements in [docs/RISK_DATA_REQUIREMENTS.md](docs/RISK_DATA_REQUIREMENTS.md), and current evidence/limitations in [docs/STATUS.md](docs/STATUS.md).

## Repository map

```text
configs/                    dataset, scenario, split, terrain, training contracts
sherpaos/datasets/          schema, generation, labels, context, split, validation
sherpaos/sim/               G1 sensors, v26 controller, terrain, weather, supervisor
sherpaos/policy/            five-guard fusion and hysteresis
sherpaos/runtime/           live evidence bus, REST/WebSocket, grounded voice API
sherpaos/expedition/        immutable MCAP/rosbag2 upload and inspection
scripts/train_risk_model.py reproducible two-head TCN trainer
scripts/render_v26_himalaya.py labeled, visibility-gated video renderer
scripts/hf_balanced_job/    Hugging Face Job entrypoint
web/                        Live Demo, Radio, and four-day operator Journal
```

## Honest limitations

- This is simulation evidence, not Himalayan ground truth or a safety certification.
- No physical G1 ROS bag has yet been ingested into the four displayed mission memories.
- Snow is rigid/non-deformable; visual particles do not create traction physics.
- The DEM/route artifact is coarse strategic context, not foot-scale terrain perception.
- The visual feed is a selected, labeled simulation recording; live decisions are generated separately by a continuously running MuJoCo supervisor.
- The learned risk model has not passed locked-test, hardware, sim-to-real, calibration-drift, latency, or fail-operational qualification.
- Mobility validation false positives remain too high for sole-authority deployment.
- Voice and LLM outputs are explanatory only and must never authorize actuation.

SherpaOS is best understood as a rigorous first safety experiment: frozen competent locomotion, leakage-resistant risk data, a conservative real-time supervisor, immutable evidence, and a human interface that can explain what the robot observed and why it slowed or stopped.