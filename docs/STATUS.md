# STATUS.md — current state

**Last updated:** 2026-08-29 11:24 PT
**Green SHA:** none yet (scaffold not committed)
**Test state:** no tests written yet

## What's done

- Repo scaffold: directory tree, `pyproject.toml`, `.gitignore`.
- Frozen contracts (`sherpaos/contracts.py`): `RobotTelemetry`, `GuardDecision`,
  `ActuationReceipt`, `RunManifest`, `ReasonCode`, `GuardAction`, `TelemetrySource`.
- Shared-context docs: `AGENTS.md`, `CLAUDE.md`, `docs/BUILD_SPEC.md`,
  `docs/CONTRACTS.md`, `docs/DECISIONS.md`.

## In progress

Parallel implementation launched across four module groups (see `docs/HANDOFF.md` once
each lands):

1. Simulation — MuJoCo G1 adapter, scenario generator, disturbance injection, PD
   posture/stepping controller (`sherpaos/sim/`).
2. Estimator + policy — rolling-window features, deterministic risk score, state
   machine with hysteresis (`sherpaos/estimator/`, `sherpaos/policy/`).
3. Recorder + evidence — incident buffer, store-and-forward queue, manifest writer
   (`sherpaos/recorder/`, `sherpaos/evidence/`).
4. Test harness — pytest/ruff config, contract/property tests, leakage sentinel
   (`tests/`).

## Blockers

None yet.

## Next three tasks

1. Integrate: baselines (controller-only, IMU-threshold, always-slow) + paired
   evaluator (`sherpaos/evaluation/`), wired against the sim adapter and policy.
2. CLI (`sherpaos/cli/main.py`): `preflight`, `test`, `simulate`, `evaluate`, `demo
   --offline`.
3. Run `sherpa demo --offline` end-to-end and confirm the 3:30 PM gate: intervention
   visibly changes one simulated outcome.
