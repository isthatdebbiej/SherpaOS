# Vultr and MuJoCo Playground runbook

This runbook provides two independent cloud checks:

1. SherpaOS validation: the exact project lock, tests, offline demo, and evidence bundle.
2. Playground capability: NVIDIA/JAX works and the official
   `G1JoystickFlatTerrain` environment can load.

The Playground check is an infrastructure proof, not evidence that SherpaOS controls a
real G1 or closes the sim-to-real gap. SherpaOS currently uses its own deterministic
MuJoCo adapter; do not delay the demo to integrate Playground unless the smoke test is
green and the integration has a strict timebox.

## Vultr instance

This runbook describes a disposable validation/Playground host. The Field Journal now
runs on a persistent Vultr application host; never destroy or clean that host by
following this validation runbook. If both workloads share one instance, use isolated
directories/services and skip the destructive cleanup guidance.

Use Ubuntu 22.04 or 24.04. For the SherpaOS validation, a CPU instance is sufficient.
For Playground, choose an NVIDIA GPU instance with a working vendor driver image and
enough disk for CUDA/JAX packages and Menagerie assets. Start with at least 40 GB free.

Do not put API keys in commands, cloud-init, the repository, or captured logs. Restrict
SSH ingress and use an SSH key. Destroy the instance after artifacts are retrieved only
when it was explicitly provisioned as a disposable validation host.

Before paying for GPU time:

```bash
nvidia-smi
df -h
```

If `nvidia-smi` fails, replace/fix the instance rather than installing random CUDA
toolkits. JAX wheels include their CUDA user-space dependencies; the host still needs a
compatible NVIDIA driver.

## Bootstrap SherpaOS

SSH to the instance, clone the repository, and run from its root:

```bash
chmod +x scripts/vultr_*.sh
./scripts/vultr_bootstrap.sh
```

Alternatively, bootstrap and clone into a dedicated empty directory:

```bash
./vultr_bootstrap.sh <git-url> <commit-or-tag> /opt/sherpaos-run
cd /opt/sherpaos-run
```

Use an immutable commit SHA or checkpoint tag, not a moving branch, for official
evidence. The bootstrap uses `uv sync --frozen --extra dev`; lock drift therefore fails
instead of silently changing dependencies.

## Run SherpaOS validation

The validation refuses a dirty tree. It runs preflight, Ruff/pytest, and the deterministic
offline demo, then packages logs and evidence with SHA-256 hashes.

```bash
./scripts/vultr_validate.sh
```

Optional controls:

```bash
RUN_ID=checkpoint-3 TIMEOUT_SECONDS=7200 ./scripts/vultr_validate.sh /mnt/evidence
```

Success means `cloud-status.json` contains `GREEN`. Retrieve both the archive and its
checksum before terminating the instance:

```bash
scp user@host:/path/to/SherpaOS/artifacts/cloud/<run-id>.tar.gz .
scp user@host:/path/to/SherpaOS/artifacts/cloud/<run-id>.tar.gz.sha256 .
sha256sum -c <run-id>.tar.gz.sha256
```

Inspect the archive rather than trusting the final terminal line. Preserve the commit
SHA, lock hash, raw logs, demo summary, and evidence bundle for the submission.

## Run the isolated Playground smoke test

This clones the official repository at `v0.2.0`, creates a separate virtual environment,
installs the CUDA 12 JAX wheel, syncs Playground extras, verifies that JAX selected the
GPU backend, and loads the official G1 task. The first environment load downloads
MuJoCo Menagerie assets and therefore needs network access.

```bash
./scripts/vultr_playground_smoke.sh
```

To test another reviewed release or commit:

```bash
PLAYGROUND_REF=<commit-or-tag> ./scripts/vultr_playground_smoke.sh
```

Do not use an unrecorded `main` checkout for evidence. Keep
`artifacts/playground/playground-provenance.txt`, the smoke log, and hashes. The
smoke gate resets and steps both `G1JoystickFlatTerrain` and
`G1JoystickRoughTerrain` and rejects non-finite observations. Its single zero-action
step is infrastructure validation, not locomotion or policy evidence. The
authoritative install flow is the official
[MuJoCo Playground repository](https://github.com/google-deepmind/mujoco_playground).

## Interactive viewer

After the smoke test, start rscope in a second `tmux` pane:

```bash
./scripts/vultr_rscope.sh
```

Open the URL it prints through the Vultr graphical desktop or an SSH tunnel. A viewer
with no trajectory producer will be empty. Policy training/playback must explicitly
enable rscope output; starting the viewer alone is not visual-rollout evidence.

## Failure triage

- **JAX reports CPU:** confirm `nvidia-smi`, inspect driver compatibility, and ensure the
  CUDA JAX wheel was installed inside Playground's `.venv`. Do not count CPU fallback as
  a GPU result.
- **Out of disk:** delete the disposable `.cloud-work` directory or provision a larger
  volume. Never delete the repository or evidence directory indiscriminately.
- **Menagerie download fails:** retry only the download/network stage. Record that the
  test is infrastructure-blocked; the SherpaOS offline demo remains the primary path.
- **Playground dependency conflict:** keep the failure log and the pinned ref. Do not
  modify SherpaOS's `uv.lock` to accommodate Playground.
- **SherpaOS deterministic test fails:** do not automatically rerun until green. Retrieve
  the failing archive, reproduce against its SHA locally, fix on a branch, and rerun with
  a new run ID.
- **SSH disconnects:** run validation inside `tmux`; the process and logs remain on the
  instance. Do not run any real-robot command unattended.

## Timebox and stop conditions

- Allocate 20 minutes to provisioning/bootstrap and 30 minutes to the Playground smoke.
- If GPU setup is not green after 45 minutes, stop and use the existing MuJoCo demo.
- Do not start locomotion training merely because the environment loads. Training needs
  an explicit hypothesis, budget, checkpoint plan, and evaluation against the frozen
  deterministic baseline.
- Cloud success never substitutes for an on-device Orin/Thor inference benchmark.

## Cleanup

After checksum verification on the local machine, remove or destroy only an explicitly
disposable validation instance through the provider console. Never destroy the
persistent Field Journal/application host. Confirm billing has stopped for disposable
resources. Retain the evidence archive and exact repository SHA needed to reproduce it.
