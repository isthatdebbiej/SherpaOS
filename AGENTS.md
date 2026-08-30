# AGENTS.md — SherpaOS

Operational contract for any coding agent (Claude, Codex, or human) working in this repo.
Full plan: `../docs/plan.md` (hackathon schedule) and `../docs/idea.txt` (original spec/background).
This file is the day-to-day contract; `docs/plan.md` is the operational source of truth for scope and gates.

## What this is

An offline, auditable mobility-risk supervisor for Unitree G1. It consumes only
onboard-observable telemetry, estimates deteriorating mobility/dynamics, and requests
one of `PASS`, `LIMIT_SPEED`, `REQUEST_HOLD`. It sits above an existing/assumed
locomotion controller — it does not replace it.

## Commands

```text
sherpa preflight                          # env/deps/contract sanity check
sherpa test                               # ruff + pytest (unit/property/contract/integration)
sherpa simulate --scenario <name> --seed <n>
sherpa evaluate --matrix configs/eval.yaml
sherpa demo --offline                     # one-command end-to-end demo, no network
sherpa reproduce <run-id>
sherpa overnight launch|status|fetch      # cloud validation pipeline (stubbed until credentials exist)
```

Or via uv: `uv run sherpa ...`, `uv run pytest`.

Field Journal:

```text
cd web
npm ci
npm run build
npm run dev
```

## Ownership boundaries (module lanes)

| Area | Path | Notes |
|---|---|---|
| Contracts | `sherpaos/contracts.py` | Frozen. Changes require a docs/DECISIONS.md entry. |
| Simulation | `sherpaos/sim/`, `sherpaos/adapters/mujoco_*` | MuJoCo G1 adapter, scenario generator, disturbances |
| Estimator | `sherpaos/estimator/` | Rolling-window features, deterministic risk score |
| Policy | `sherpaos/policy/` | State machine, hysteresis, reason codes -> GuardDecision |
| Recorder/Evidence | `sherpaos/recorder/`, `sherpaos/evidence/` | Incident buffer, store-and-forward queue, manifest |
| Evaluation | `sherpaos/evaluation/` | Baselines, paired evaluator, metrics |
| CLI | `sherpaos/cli/` | Wires everything together; touches all modules by import only |
| Tests | `tests/` | unit, property (Hypothesis), integration, regression |
| Expedition memory | `sherpaos/expedition/`, `sherpaos/voice/` | Post-day immutable bag inspection/reflection boundary; never actuation |
| Field Journal | `web/` | Next.js presentation UI and mock/API day state |

Do not edit another lane's files without reviewing `docs/DECISIONS.md` first and recording
why. Shared types live only in `sherpaos/contracts.py`.

## Safety constraints (non-negotiable)

1. Simulator ground truth (true friction, true contact class, injected-fault label, true
   fall state) must never reach estimator/policy features. It is evaluator-only —
   see `sherpaos/evaluation/`. There must be a test (`leakage` marker) enforcing this.
2. No LLM/network call inside the runtime decision loop (telemetry -> features -> score ->
   policy -> decision). The core must run fully offline.
3. Stale, malformed, missing, NaN, and out-of-order telemetry must fail conservatively
   (toward `REQUEST_HOLD`/rejecting the sample), never crash and never silently pass.
4. Every `GuardDecision` that results in an action must produce an `ActuationReceipt`.
   Every intervention must be reconstructable from an incident evidence bundle.
5. Every new estimator/policy behavior needs a test. Every randomized-failure seed found
   during evaluation becomes a regression test (`tests/regression/`).
6. Prefer deterministic rules before any learned model. A learned model ships only if it
   beats the deterministic rules on held-out scenario groups and passes ONNX parity.
7. Reflection and voice are post-mission presentation features. They must never enter the
   telemetry-to-guard-to-actuation loop or receive privileged simulator truth.
8. Raw bags, `OPENAI_API_KEY`, ingest secrets, and SSH credentials must never enter the
   browser bundle or Git. Deployment is Vultr-only unless the user explicitly revises it.

## Definition of done (per change)

- `uv run ruff check .` clean.
- `uv run pytest` green (unit + property + contract + integration smoke).
- If touching telemetry/features: leakage test still passes.
- If touching policy: at least one scenario shows the decision changing the simulated
  outcome (the "3:30 PM gate" in `docs/plan.md`).
- `docs/STATUS.md` updated with current state, blockers, next steps.
- If touching `web/`, `npm run build` is green and current/past/locked day behavior is
  checked at desktop and mobile widths.
