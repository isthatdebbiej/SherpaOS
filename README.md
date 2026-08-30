# SherpaOS

Offline, auditable mobility-risk supervisor for Unitree G1. It consumes only
onboard-observable telemetry, detects deteriorating mobility/dynamics, and requests one
of `PASS`, `LIMIT_SPEED`, `REQUEST_HOLD` from an underlying locomotion controller.

The repository also contains **Pemba's Field Journal** in `web/`: an interactive
five-day Everest trail and post-mission diary. Verified ROS recordings are processed
after each day; an LLM may write and narrate a reflection, but it is never part of the
runtime safety or actuation loop. The application and processing services are intended
to run entirely on the existing Vultr host.

Full plan and background: `../docs/plan.md`, `../docs/idea.txt`.
Day-to-day contract: `AGENTS.md` / `CLAUDE.md`.
Schemas: `docs/CONTRACTS.md`. Current state: `docs/STATUS.md`.

## Quickstart

```bash
uv sync --extra dev
uv run sherpa preflight
uv run sherpa demo --offline
```

Field Journal development:

```bash
cd web
npm install
npm run dev
```

See `docs/RUNBOOK.md` for the full command set.

## Status

The safety runtime, expedition-memory backend, and mock-data Field Journal UX are under
active hackathon-pace development. See `docs/STATUS.md` for the verified checkpoint and
remaining integration work.

## Limitations

Simulation is not Himalayan ground truth; thresholds are experimental and not
safety-certified; sim-to-real transfer is unproven. Diary prose is an evidence-grounded
interpretation of a processed report, not a source of safety truth. See
`docs/BUILD_SPEC.md` for the complete boundary.
