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

## Checkpoint 2 — five-guard vertical slice (working tree; pending human commit)

- **Base SHA:** `2382a8c` (`checkpoint-1-smoke`).
- **Files:** estimator risk component refactor; five-guard fusion; policy report
  attachment; synthetic battery simulation; supervisor/receipt adapter; evidence
  round-trip updates; CLI; tests; status docs.
- **Commands:** isolated `.venv-codex`; `ruff check .`; full `pytest`; `sherpa preflight`;
  `sherpa demo --offline`.
- **Result:** Ruff green; 127 tests passed; preflight GREEN; offline nominal/hazard demo
  GREEN with checksum-verified evidence and one receipt per decision.
- **Known limitation:** no paired evaluator or video renderer yet. Nominal has transient
  dynamics limits that must be measured against progress before threshold changes.
- **Artifacts:** smoke evidence was generated from a dirty working tree and is not
  acceptable as final judging evidence; regenerate after the human commits/tags.
- **Next action:** human review/commit, then build paired baselines/evaluator and video.
