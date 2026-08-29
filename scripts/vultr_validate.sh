#!/usr/bin/env bash
set -Eeuo pipefail

# Run the reproducible SherpaOS cloud validation gate and package its evidence.
# Usage: ./scripts/vultr_validate.sh [artifact_root]

ARTIFACT_ROOT="${1:-artifacts/cloud}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$ARTIFACT_ROOT/$RUN_ID"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-7200}"

log() { printf '[vultr-validate] %s\n' "$*"; }
fail() { printf '[vultr-validate] ERROR: %s\n' "$*" >&2; exit 1; }

command -v uv >/dev/null || fail "uv is missing; run vultr_bootstrap.sh first"
command -v git >/dev/null || fail "git is required"
[[ -f pyproject.toml && -f uv.lock && -d .git ]] || fail "run from the SherpaOS repo root"

mkdir -p "$RUN_DIR"
STATUS="RED"
on_exit() {
  rc=$?
  printf '{"status":"%s","exit_code":%d,"run_id":"%s","git_sha":"%s"}\n' \
    "$STATUS" "$rc" "$RUN_ID" "$(git rev-parse HEAD)" > "$RUN_DIR/cloud-status.json"
  tar -C "$ARTIFACT_ROOT" -czf "$ARTIFACT_ROOT/$RUN_ID.tar.gz" "$RUN_ID"
  sha256sum "$ARTIFACT_ROOT/$RUN_ID.tar.gz" > "$ARTIFACT_ROOT/$RUN_ID.tar.gz.sha256"
  log "Evidence: $ARTIFACT_ROOT/$RUN_ID.tar.gz"
  exit "$rc"
}
trap on_exit EXIT

if [[ -n "$(git status --porcelain)" ]]; then
  fail "working tree is dirty; official evidence requires a clean SHA"
fi

{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'timestamp_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'git_sha=%s\n' "$(git rev-parse HEAD)"
  printf 'git_ref=%s\n' "$(git describe --always --dirty)"
  printf 'uv_lock_sha256=%s\n' "$(sha256sum uv.lock | cut -d ' ' -f 1)"
  printf 'host=%s\n' "$(hostname)"
  uname -a
  if command -v nvidia-smi >/dev/null; then nvidia-smi || true; fi
} > "$RUN_DIR/provenance.txt"

log "Verifying locked dependencies"
timeout "$TIMEOUT_SECONDS" uv sync --frozen --extra dev 2>&1 | tee "$RUN_DIR/uv-sync.log"

log "Running preflight, merge gate, and deterministic offline demo"
timeout "$TIMEOUT_SECONDS" uv run sherpa preflight 2>&1 | tee "$RUN_DIR/preflight.log"
timeout "$TIMEOUT_SECONDS" uv run sherpa test 2>&1 | tee "$RUN_DIR/test.log"
timeout "$TIMEOUT_SECONDS" uv run sherpa demo --offline \
  --output "$RUN_DIR/demo" 2>&1 | tee "$RUN_DIR/demo.log"

find "$RUN_DIR" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_DIR/files.sha256"
STATUS="GREEN"
log "Validation is GREEN"
