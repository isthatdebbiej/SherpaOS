#!/usr/bin/env bash
set -Eeuo pipefail

# Viewer only: it neither creates a policy nor generates a rollout.
PLAYGROUND_DIR="${1:-$PWD/.cloud-work/mujoco_playground}"

fail() { printf '[playground-viewer] ERROR: %s\n' "$*" >&2; exit 1; }
[[ -x "$PLAYGROUND_DIR/.venv/bin/python" ]] || \
  fail "Playground environment missing; run vultr_playground_smoke.sh first"

cd "$PLAYGROUND_DIR"
uv pip install rscope
printf '[playground-viewer] Starting rscope; open its URL via VDI or an SSH tunnel.\n'
exec uv --no-config run python -m rscope
