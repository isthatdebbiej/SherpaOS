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

## GuardReport (per-guard, pre-fusion)

SherpaOS runs five independent guards (see "Guard families" below). Each one emits its
own `GuardReport`: `guard` (`GuardName`), `score` (0..1), `confidence` (0..1),
`reason_codes`, `recommended_action`, `provenance` (free-form string map — e.g. which
telemetry fields or artifact version it used). Nothing in one guard's `GuardReport` may
depend on another guard's internals.

## GuardDecision (fused)

`action` in `{PASS, LIMIT_SPEED, REQUEST_HOLD}`, plus `score`, `confidence`,
`reason_codes` (from `sherpaos.contracts.ReasonCode`), `input_age_seconds`,
`requested_speed_limit`, `timestamp`, `rules_version`, optional `model_version`, and
`guard_reports` (the full tuple of per-guard `GuardReport`s that produced this decision —
this is what lets the incident timeline and dashboard show "mobility, dynamics,
telemetry, battery, and geographic reasons" separately, per `docs/plan.md` section 7).

**Fusion rule**: conservative, not averaged. A single high-severity guard must be able to
drive the fused action (e.g. `REQUEST_HOLD`) even if every other guard reports `PASS` —
never let four calm guards dilute one alarmed guard's signal into a mid-range average.

## Guard families

1. **Mobility/traction guard** (`GuardName.MOBILITY`) — command-response residuals, IMU
   instability, joint effort/velocity residuals, asymmetry. Detects reduced operating
   margin; does not classify literal ice.
2. **Dynamics/body guard** (`GuardName.DYNAMICS`) — actuator under-response/saturation,
   vibration, payload change, abnormal effort, external impulse, orientation instability.
3. **Telemetry-health guard** (`GuardName.TELEMETRY_HEALTH`) — stale, missing, frozen,
   NaN, malformed, future-dated, or out-of-order telemetry, and inference failure.
4. **Battery-margin guard** (`GuardName.BATTERY`) — state of charge, voltage sag under
   load, discharge rate, estimated remaining operating time, temperature when available.
   Reads `RobotTelemetry.battery_fraction/battery_voltage/battery_current_a/
   battery_temperature_c`. Simulated input is valid hackathon evidence but must be
   labeled `sim` (via `RobotTelemetry.source`) and varied across held-out battery
   scenarios — never presented as a calibrated real-G1 predictor without real telemetry.
5. **Geographic-risk guard** (`GuardName.GEOGRAPHIC`) — slope, elevation, route exposure,
   distance to safe waypoint, terrain-data quality, optional high-wind/extreme-cold
   context, read from `MissionContext` (below).

## ActuationReceipt

Links a `decision_id` to what was actually applied: `requested_action`,
`applied_action`, `accepted`, optional `rejection_reason`, `adapter_timestamp`,
`acknowledgement_source`. Every acted-upon `GuardDecision` must produce exactly one
receipt.

## RunManifest

Reproducibility record: commit/config/controller/model/data hashes, dependency lock
hash, container identity, seed, scenario name, runtime/hardware identity, artifact
checksums.

## MissionContext

Offline-preprocessed geographic/route context feeding the geographic-risk guard:
latitude/longitude, elevation, slope, route segment, distance to safe waypoint, terrain
source/version/CRS, lookup timestamp, validity, resolution, provenance, optional
wind/temperature context. Loaded once from `configs/terrain/ebc_route.json` (see
`configs/terrain/PROVENANCE.md`) — **the runtime must never query the internet for
this**. Missing, out-of-bounds, low-resolution, or stale context must set `valid=False`
(or an old `lookup_timestamp`) rather than inventing terrain, and must lower the
geographic guard's confidence accordingly.

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

## Expedition memory and reflection boundary

Authoritative backend types live in `sherpaos/expedition/models.py`; the current visual
fixture type lives in `web/src/types/expedition.ts` until API integration is complete.
Raw `.mcap`/rosbag2 `.db3` files are immutable per-day inputs. Inspection records their
hash, size, time range, topic coverage, and allowlist decision before any voice or
reflection call is allowed.

The reflection layer may receive the approved day plan, verified onboard-topic summary,
derived event report, bag hash, and prior diary entries. It must not receive evaluator
ground truth, hidden friction/contact labels, or actuation authority. Its output is a
presentation artifact (`diary.json` plus optional narration), never a `GuardDecision`.
