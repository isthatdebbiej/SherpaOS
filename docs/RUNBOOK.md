# RUNBOOK.md — local, HF, Vultr, Jetson, recording, recovery commands

## Setup

```bash
uv sync --extra dev --extra expedition
```

## Expedition memory and radio

The authoritative recordings are written beneath the gitignored `var/expeditions/`
directory. Set `SHERPA_EXPEDITION_STORE` to use another local disk.

```bash
uv run uvicorn sherpaos.expedition.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd web
npm run dev
```

Open the Field Journal, choose a day in **Call Pemba**, and upload a real `.mcap` or
rosbag2 `.db3`. Voice turns additionally require `OPENAI_API_KEY`. The frontend uses
`http://127.0.0.1:8000` by default; override it with `NEXT_PUBLIC_SHERPA_API`.

## Local

```bash
uv run sherpa preflight
uv run sherpa test
uv run sherpa simulate --scenario nominal --seed 1                 # 30 s (1,500 ticks at 50 Hz)
uv run sherpa simulate --scenario mixed_traction_disturbance --seed 1
uv run sherpa evaluate --matrix configs/eval.yaml
uv run sherpa demo --offline
```

`simulate --max-steps` counts 50 Hz control ticks, not 500 Hz physics steps:
`--max-steps 300` is 6 seconds and the default `1,500` is 30 seconds.

```bash
uv run ruff check .
uv run pytest -m "not integration"
uv run pytest -m integration
uv run pytest -n auto            # parallel via pytest-xdist
```

## Hugging Face / Vultr (cloud)

Not wired to real accounts in this environment (no credentials configured — see
`docs/DECISIONS.md`). `sherpa overnight launch/status/fetch` currently validate the
preflight contract (clean tree, tagged commit, lock resolves, tests green, cost
ceiling/timeout/destination present) and run a local shard as a stand-in. To wire real
jobs: set `HF_TOKEN` / Vultr API key as environment variables (never commit them), then
implement the actual submission calls in `sherpaos/cli/overnight.py`.

## Jetson

No Jetson available in this environment. `sherpa` produces an ONNX export step once a
model exists (`sherpaos/evaluation/` model path); benchmarking on a real Orin/Thor is a
manual step to run on-device once hardware access exists — see `docs/BUILD_SPEC.md`
mandatory deliverable 9.

## Recording / demo

`uv run sherpa demo --offline` runs the deterministic end-to-end scenario used for the
live demo. It writes incident evidence to `artifacts/incidents/` and any plots to
`artifacts/plots/`. There is no camera/video capture step in this repo — that is Lane D
(human-owned) work per `docs/plan.md`.

## Recovery

If a run leaves the tree dirty or a lock stale: `git status`, then either commit or
`git stash -u` — do not discard without checking. Re-run `uv sync --extra dev` if the
environment looks broken. If a regression seed is found, save it under
`tests/regression/` per the failure-driven-development workflow in `docs/idea.txt`
section 26.
