# STATUS.md — current state

**Last updated:** 2026-08-29 12:37 PT  
**Base SHA:** `2382a8c` (`checkpoint-1-smoke`)  
**Working tree:** uncommitted integration checkpoint; human must review/commit  
**Test state:** 127 tests passed under Codex's isolated Python 3.13 environment; Ruff green

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
- Checksum-verified nominal and hazard demo smoke run.

## Verified

- `ruff check .`: green.
- Full pytest suite: 127 passed.
- `sherpa preflight`: GREEN (G1 asset, terrain artifact, five-guard smoke).
- `sherpa demo --offline`: GREEN; 300/300 steps survived in nominal and hazard runs;
  299 decisions and 299 receipts per run; evidence checksums verified.
- Nominal run produced no `REQUEST_HOLD`; it did spend 112/299 decisions in
  `LIMIT_SPEED` after 17 transient dynamics elevations plus policy cooldown. Do not tune
  this away without paired evaluation of nominal-progress impact.

## Environment note

Claude's `.venv` is owned by a different Windows SID from the Codex runner. Codex used
an isolated `.venv-codex` for verification and did not replace Claude's environment.
Git commands in evidence generation now pass the repository as an explicit safe directory.

## Next three tasks

1. Human reviews and commits this integration checkpoint, then reruns official evidence
   from the clean SHA so manifests do not represent a dirty tree.
2. Implement paired evaluator/baselines and quantify nominal progress, unsafe
   continuation, false HOLD, intervention timing, and guard ablations over fixed seeds.
3. Add deterministic programmatic MP4 capture and the two-minute comparison view.
