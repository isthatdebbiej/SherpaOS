#!/usr/bin/env bash
set -Eeuo pipefail

# Isolated MuJoCo Playground G1/GPU smoke test. It does not alter SherpaOS's uv environment.
# Usage: ./scripts/vultr_playground_smoke.sh [work_root] [artifact_root]

WORK_ROOT="${1:-$PWD/.cloud-work}"
ARTIFACT_ROOT="${2:-$PWD/artifacts/playground}"
PLAYGROUND_REF="${PLAYGROUND_REF:-v0.2.0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-5400}"
REPO_URL="https://github.com/google-deepmind/mujoco_playground.git"

log() { printf '[playground-smoke] %s\n' "$*"; }
fail() { printf '[playground-smoke] ERROR: %s\n' "$*" >&2; exit 1; }

command -v uv >/dev/null || fail "uv is missing; run vultr_bootstrap.sh first"
command -v git >/dev/null || fail "git is required"
command -v nvidia-smi >/dev/null || fail "NVIDIA driver/GPU not detected"
[[ "$WORK_ROOT" != "/" && "$WORK_ROOT" != "$HOME" ]] || fail "unsafe work_root"

mkdir -p "$WORK_ROOT" "$ARTIFACT_ROOT"
PLAYGROUND_DIR="$WORK_ROOT/mujoco_playground"
LOG_FILE="$ARTIFACT_ROOT/playground-smoke.log"

if [[ ! -d "$PLAYGROUND_DIR/.git" ]]; then
  git clone --filter=blob:none --branch "$PLAYGROUND_REF" "$REPO_URL" "$PLAYGROUND_DIR"
fi
cd "$PLAYGROUND_DIR"
git fetch --tags origin
git checkout --detach "$PLAYGROUND_REF"

{
  printf 'timestamp_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'playground_ref=%s\n' "$PLAYGROUND_REF"
  printf 'playground_sha=%s\n' "$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} | tee "$ARTIFACT_ROOT/playground-provenance.txt"

log "Creating isolated Playground environment"
uv venv --python 3.12
export VIRTUAL_ENV="$PLAYGROUND_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export JAX_DEFAULT_MATMUL_PRECISION="highest"

timeout "$TIMEOUT_SECONDS" uv pip install -U "jax[cuda12]" \
  --index-url https://pypi.org/simple 2>&1 | tee "$LOG_FILE"
timeout "$TIMEOUT_SECONDS" uv --no-config sync --all-extras 2>&1 | tee -a "$LOG_FILE"

log "Checking JAX GPU backend and resetting/stepping both official G1 environments"
timeout "$TIMEOUT_SECONDS" uv --no-config run python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json
import jax
import jax.numpy as jnp
from mujoco_playground import locomotion

backend = jax.default_backend()
devices = [str(device) for device in jax.devices()]
if backend != "gpu":
    raise SystemExit(f"Expected JAX GPU backend, got {backend}: {devices}")
results = []
for index, environment_name in enumerate(
    ("G1JoystickFlatTerrain", "G1JoystickRoughTerrain")
):
    env = locomotion.load(environment_name)
    state = env.reset(jax.random.PRNGKey(42 + index))
    reset_finite = all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(state.obs)
    )
    if not reset_finite:
        raise SystemExit(f"Non-finite reset observation: {environment_name}")
    state = env.step(state, jnp.zeros(env.action_size))
    step_finite = all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(state.obs)
    )
    if not step_finite:
        raise SystemExit(f"Non-finite step observation: {environment_name}")
    results.append({
        "environment": environment_name,
        "action_size": int(env.action_size),
        "reset_finite": reset_finite,
        "step_finite": step_finite,
    })
print(json.dumps({
    "status": "GREEN",
    "jax_backend": backend,
    "jax_devices": devices,
    "environments": results,
    "claim": "infrastructure-smoke-only-no-locomotion-policy",
}, indent=2))
PY

sha256sum "$LOG_FILE" "$ARTIFACT_ROOT/playground-provenance.txt" \
  > "$ARTIFACT_ROOT/playground-files.sha256"
log "Playground G1 smoke test is GREEN"
