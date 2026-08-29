# HANDOFF.md — exact commands, artifacts, unfinished work

Chat is not memory. This file is updated at every checkpoint with: commit SHA, owned
files touched, command run, result, artifact path, remaining risk, next action.

## Checkpoint 0 — scaffold

- **SHA:** (pending first commit)
- **Files:** repo tree, `pyproject.toml`, `.gitignore`, `sherpaos/contracts.py`,
  `sherpaos/__init__.py`, module `__init__.py` stubs, `AGENTS.md`, `CLAUDE.md`,
  `docs/BUILD_SPEC.md`, `docs/CONTRACTS.md`, `docs/DECISIONS.md`, `docs/STATUS.md`,
  `docs/RUNBOOK.md`, this file.
- **Command:** none run yet (`uv sync` not yet executed).
- **Result:** contracts frozen; nothing importable/runnable yet beyond `contracts.py`.
- **Artifact path:** n/a.
- **Remaining risk:** MuJoCo G1 model (mujoco_menagerie) not yet vendored/pinned;
  `mujoco` package not yet installed in this environment.
- **Next action:** parallel implementation of sim/estimator-policy/recorder-evidence/
  test-harness, then integration (baselines, evaluator, CLI).
