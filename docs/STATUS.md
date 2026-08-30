# STATUS.md — current state

**Last updated:** 2026-08-29 PT
**Base SHA:** `f8562169` (dataset-pipeline implementation base)
**Working tree:** dataset pipeline implementation ready for commit
**Test state:** Ruff green; full pytest suite green (150 passed); 2-episode dataset contract GREEN

## Implemented

- MuJoCo Menagerie G1 posture/stepping simulator with nominal, mixed-traction,
  disturbance, actuator-health, slope, sensor-noise, and synthetic battery scenarios.
- Five independent guard reports: mobility, dynamics, telemetry health, battery margin,
  and offline geographic risk.
- Conservative max/action-floor fusion: one severe guard cannot be averaged away.
- Policy hysteresis with `PASS`, `LIMIT_SPEED`, and `REQUEST_HOLD`.
- Simulation actuation adapter with one `ActuationReceipt` per decision.
- Battery simulation with explicit simulated provenance for state of charge, current,
  voltage sag, and temperature.
- Offline Everest Base Camp route artifact and geographic guard.
- Incident/evidence bundle serialization, including battery fields and all guard reports.
- Executable CLI: `sherpa preflight`, `sherpa test`, `sherpa simulate`, and
  `sherpa demo --offline`.
- Optional native live visualization via `sherpa simulate --viewer`.
- Local Unitree pretrained G1 walking-policy source and checkpoint, pinned with
  attribution; standalone upstream MuJoCo rollout verified.
- Checksum-verified nominal and hazard demo smoke run.
- Isolated, pinned MuJoCo Playground v0.2.0 bootstrap with CUDA JAX GPU gate.
- Flat- and rough-terrain G1 reset/step smoke checks with non-finite rejection.
- Explicit Playground observation-to-telemetry adapter that rejects privileged truth.
- Rollout evidence metadata with code/Playground SHAs, GPU/JAX identity, policy hash,
  provenance, license, command, seed, and artifact checksums.
- Vultr clean-SHA validation/evidence packaging and optional rscope viewer launcher.

## Verified

- `ruff check .`: green.
- Full pytest suite: 127 passed.
- `sherpa preflight`: GREEN (G1 asset, terrain artifact, five-guard smoke).
- `sherpa demo --offline`: GREEN; 300/300 steps survived in nominal and hazard runs;
  299 decisions and 299 receipts per run; evidence checksums verified.
- `sherpa simulate --viewer --max-steps 2`: GREEN under a mocked native viewer; the
  CLI forwards the option, synchronizes physics frames, waits for the window to close,
  and then closes the viewer handle.
- Unitree `deploy/deploy_mujoco/deploy_mujoco.py g1.yaml`: GREEN after adding the
  repository root to `PYTHONPATH`; loaded the 12-DOF TorchScript walking policy in the
  native viewer using its configured 0.5 m/s forward command.
- Nominal run produced no `REQUEST_HOLD`; it did spend 112/299 decisions in
  `LIMIT_SPEED` after 17 transient dynamics elevations plus policy cooldown. Do not tune
  this away without paired evaluation of nominal-progress impact.

## Environment note

Claude's `.venv` is owned by a different Windows SID from the Codex runner. Codex used
an isolated `.venv-codex` for verification and did not replace Claude's environment.
Git commands in evidence generation now pass the repository as an explicit safe directory.

## Next three tasks

1. Review and commit this integration checkpoint; Vultr validation intentionally refuses
   dirty-tree evidence.
2. Run `scripts/vultr_playground_smoke.sh` on the provisioned GPU and retrieve its logs.
3. Wire the pinned Unitree 12-DOF policy's observation and action contracts to a
  dedicated adapter, then generate a five-guard supervised MP4 rollout. The verified
  standalone viewer run is not yet supervision evidence.

## Dataset pipeline checkpoint (2026-08-29 PT)

- Added frozen 200-episode scenario/config contracts: 50 nominal, 50 mobility,
  50 dynamics, and 50 combined controller-only episodes.
- Added deterministic, resumable 10-episode NPZ shards with 500 control steps,
  100-sample windows, 50-sample prediction horizon, and stride 10.
- Observation arrays are fixed at 103 onboard-observable features; privileged
  friction/slope/slip/actuator/disturbance/fall truth remains under `labels/`.
- Added `sherpa data generate`, `sherpa data validate`, and `sherpa data split` only.
- Validation rejects count, integrity, leakage, alignment, finite-value, width,
  duplicate-ID, group-overlap, missing-shard, and positive-rate failures.
- Acceptance commands completed GREEN for the allowed two-episode local contract;
  mobility positive rate was 0.50 and feature width was 103.
- Full suite: 150 passed. Ruff: green.

## Next three tasks (dataset pipeline)

1. Push the committed dataset-pipeline SHA and clone that exact SHA on Vultr.
2. Run exactly 200 episodes on Vultr, then validate and freeze checksums.
3. Download and checksum-verify the immutable dataset before any later training task.
