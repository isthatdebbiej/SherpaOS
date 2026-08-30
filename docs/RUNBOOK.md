# RUNBOOK.md — local, Vultr, recording, Field Journal, and recovery commands

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

The redesigned trail currently reads `web/src/data/days.ts` so UX review is independent
of cloud and model credentials. Connecting that page to the existing API is unfinished.

## Local

```bash
uv run sherpa preflight
uv run sherpa test
uv run sherpa simulate --scenario nominal --seed 1
uv run sherpa simulate --scenario mixed_traction_disturbance --seed 1
uv run sherpa evaluate --matrix configs/eval.yaml
uv run sherpa demo --offline
```

```bash
uv run ruff check .
uv run pytest -m "not integration"
uv run pytest -m integration
uv run pytest -n auto            # parallel via pytest-xdist
```

## Vultr application deployment

Vultr is the only deployment target. Run the Python API/ROS-processing service and the
Next.js application on the same host. Do not copy a general Vultr account API key into
the application; it is not needed when the code reads local files on the instance.

Recommended host layout:

```text
/opt/sherpa/app/                         # clean repository checkout
/opt/sherpa/expeditions/day-01/raw/      # immutable incoming bags
/opt/sherpa/expeditions/day-01/derived/  # deterministic report/events/trajectory
/opt/sherpa/expeditions/day-01/diary/    # diary JSON and narration
```

Server-only environment:

```bash
export SHERPA_EXPEDITION_STORE=/opt/sherpa/expeditions
export OPENAI_API_KEY=...                 # required only for reflection/voice
export SHERPA_INGEST_SECRET=...           # authenticate upload/completion requests
export NEXT_PUBLIC_SHERPA_API=/api         # same-origin reverse-proxied API
```

Start the services for development:

```bash
uv run uvicorn sherpaos.expedition.api:app --host 127.0.0.1 --port 8000
cd web
npm ci
npm run build
npm run start -- --hostname 127.0.0.1 --port 3000
```

Use nginx or Caddy to terminate TLS and route `/api/` to port 8000 and everything else
to port 3000. Use systemd (or another supervised service manager) in production. The
exact service/proxy files are not yet committed.

`sherpa overnight launch/status/fetch` remains a validation stub and is not the Field
Journal deployment mechanism.

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

## Secret handling

- Never commit `.env`, API keys, SSH keys, raw bags, or generated narration.
- Never expose `OPENAI_API_KEY` through a `NEXT_PUBLIC_*` name.
- Keep the API bound to loopback behind the reverse proxy unless direct access is
  explicitly secured.
- Rotate any credential that appears in logs or screenshots.
