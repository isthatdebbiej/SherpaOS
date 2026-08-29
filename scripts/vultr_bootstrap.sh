#!/usr/bin/env bash
set -Eeuo pipefail

# Bootstrap a disposable Ubuntu 22.04/24.04 Vultr instance for SherpaOS.
# Usage: ./scripts/vultr_bootstrap.sh [repo_url] [git_ref] [install_root]
# If repo_url is omitted, run this script from an existing SherpaOS checkout.

REPO_URL="${1:-}"
GIT_REF="${2:-main}"
INSTALL_ROOT="${3:-$PWD}"
UV_VERSION="${UV_VERSION:-0.8.13}"

log() { printf '[vultr-bootstrap] %s\n' "$*"; }
fail() { printf '[vultr-bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || fail "Ubuntu Linux is required"
command -v sudo >/dev/null || fail "sudo is required"

log "Installing OS packages"
sudo DEBIAN_FRONTEND=noninteractive apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates curl ffmpeg git jq tmux build-essential python3-dev

if ! command -v uv >/dev/null; then
  log "Installing uv ${UV_VERSION} for the current user"
  curl --proto '=https' --tlsv1.2 -LsSf \
    "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

if [[ -n "$REPO_URL" ]]; then
  [[ "$INSTALL_ROOT" != "/" && "$INSTALL_ROOT" != "$HOME" ]] || \
    fail "install_root must be a dedicated directory"
  mkdir -p "$INSTALL_ROOT"
  if [[ ! -d "$INSTALL_ROOT/.git" ]]; then
    [[ -z "$(find "$INSTALL_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
      fail "install_root exists and is not empty: $INSTALL_ROOT"
    log "Cloning SherpaOS"
    git clone --filter=blob:none "$REPO_URL" "$INSTALL_ROOT"
  fi
fi

cd "$INSTALL_ROOT"
[[ -f pyproject.toml && -f uv.lock ]] || fail "SherpaOS checkout not found at $INSTALL_ROOT"
[[ -d .git ]] || fail "A git checkout is required for evidence provenance"

if [[ -n "$REPO_URL" ]]; then
  git fetch --tags --prune origin
  git checkout --detach "$GIT_REF"
fi

log "Synchronizing the locked SherpaOS environment"
uv sync --frozen --extra dev

mkdir -p artifacts/cloud
{
  printf 'timestamp_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'host=%s\n' "$(hostname)"
  printf 'git_sha=%s\n' "$(git rev-parse HEAD)"
  printf 'uv=%s\n' "$(uv --version)"
  printf 'python=%s\n' "$(uv run python --version 2>&1)"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total \
      --format=csv,noheader || true
  else
    printf 'gpu=none-detected\n'
  fi
} | tee artifacts/cloud/vultr-host.txt

log "Bootstrap complete. Next: ./scripts/vultr_validate.sh"
