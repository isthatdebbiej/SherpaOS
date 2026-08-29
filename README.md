# SherpaOS

Offline, auditable mobility-risk supervisor for Unitree G1. It consumes only
onboard-observable telemetry, detects deteriorating mobility/dynamics, and requests one
of `PASS`, `LIMIT_SPEED`, `REQUEST_HOLD` from an underlying locomotion controller.

Full plan and background: `../docs/plan.md`, `../docs/idea.txt`.
Day-to-day contract: `AGENTS.md` / `CLAUDE.md`.
Schemas: `docs/CONTRACTS.md`. Current state: `docs/STATUS.md`.

## Quickstart

```bash
uv sync --extra dev
uv run sherpa preflight
uv run sherpa demo --offline
```

See `docs/RUNBOOK.md` for the full command set.

## Status

Under active hackathon-pace development. See `docs/STATUS.md` for what's implemented.

## Limitations

See `docs/BUILD_SPEC.md` and the limitations section that will be added to this README
before submission (simulation is not Himalayan ground truth; thresholds are
experimental, not safety-certified; sim-to-real transfer is unproven).
