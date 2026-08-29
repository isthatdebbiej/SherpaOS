# CONTRACTS.md — schemas and adapter behavior

Authoritative implementation: `sherpaos/contracts.py`. This document explains intent;
the code is the schema.

## RobotTelemetry

One timestamped, onboard-observable sample. Fields: monotonic + source timestamps,
sequence number, joint position/velocity/effort, base orientation/angular
velocity/linear acceleration, optional commanded velocity + gait mode, optional battery
fields, `source` (`sim`/`dump`/`g1_live`), `valid` flag, and `field_provenance` map.

**Never add**: true friction, simulator contact ground truth, terrain class, injected-fault
label, or true fall state. Those are evaluator-only (`sherpaos/evaluation/`).

## GuardDecision

`action` in `{PASS, LIMIT_SPEED, REQUEST_HOLD}`, plus `score`, `confidence`,
`reason_codes` (from `sherpaos.contracts.ReasonCode`), `input_age_seconds`,
`requested_speed_limit`, `timestamp`, `rules_version`, optional `model_version`.

## ActuationReceipt

Links a `decision_id` to what was actually applied: `requested_action`,
`applied_action`, `accepted`, optional `rejection_reason`, `adapter_timestamp`,
`acknowledgement_source`. Every acted-upon `GuardDecision` must produce exactly one
receipt.

## RunManifest

Reproducibility record: commit/config/controller/model/data hashes, dependency lock
hash, container identity, seed, scenario name, runtime/hardware identity, artifact
checksums.

## Adapter boundary rule

The simulator adapter, dump-replay adapter, and live-G1 adapter differ **only** at the
point that produces `RobotTelemetry`. Everything downstream (estimator, policy,
recorder, CLI) must be adapter-agnostic. If a downstream module imports anything from
`sherpaos.sim` or a specific adapter module directly, that is a contract violation.

## Leakage rule

Estimator/policy code may only read fields present on `RobotTelemetry`. Simulator
ground truth used for scoring (`sherpaos/evaluation/ground_truth.py`, once written)
must never be passed into estimator/policy call paths. This is enforced by a dedicated
`leakage`-marked test in `tests/unit/`.
