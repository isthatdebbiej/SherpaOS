# G1 dance/skill demo (FSMDeployG1 lane)

This is a separate, standalone demo lane from `docs/V26_HIMALAYA_PLAYGROUND.md`.
It drives the pinned **FSMDeployG1** fork (RoboMimic Deploy multi-policy FSM)
end-to-end in MuJoCo to perform pretrained dance/skill routines on the
Unitree G1. SherpaOS does not intervene in this lane and it is not part of
the runtime safety/actuation path — it is an offline visualization/demo
harness only.

## What it is

FSMDeployG1 ships several pretrained ONNX policies behind a finite-state
machine: stand up (`FixedPose`), walk (`LocoMode`), and skill routines
(`Dance`, `KungFu`, `Kick`, `KungFu2`, `BeyondMimic`, plus mimic policies
that need motion data this repo does not vendor).

`scripts/run_g1_dance.py` scripts the full sequence automatically —
PassiveMode settle → stand up → LocoMode → trigger skill → hold → cooldown
→ back to LocoMode — using deterministic sim-time waits instead of manual
keypress timing.

## Setup

```bash
uv run python scripts/fetch_fsm_dance_repo.py
cd third_party/FSMDeployG1
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install numpy pyyaml onnx onnxruntime mujoco torch
```

`fetch_fsm_dance_repo.py` clones the fork at a pinned commit
(`18f517b48c3eb7acce1f4c45bbb5db3900b5c2f1`) into gitignored
`third_party/FSMDeployG1/` (same convention as `third_party/mujoco_menagerie/`)
and patches `FSM/FSM.py` to skip mimic policies whose motion `.npz` assets
aren't vendored here (`GAE_Mimic`, `SONIC_ROBOT_Mimic`, `SONIC_HUMAN_Mimic`),
so the FSM initializes without those files present.

**License note:** FSMDeployG1 has no explicit license (`license: null` via
the GitHub API as of the pinned commit) — see `docs/DECISIONS.md`. It is
used here strictly as a local, offline demo harness; nothing from it is
redistributed in this repository or used in SherpaOS's guard/policy logic.

## Run

From `third_party/FSMDeployG1` with its venv active:

```bash
python ../../scripts/run_g1_dance.py --list
python ../../scripts/run_g1_dance.py dance
python ../../scripts/run_g1_dance.py kungfu
python ../../scripts/run_g1_dance.py kick
python ../../scripts/run_g1_dance.py beyondmimic
```

## Verified results

Each skill run below is the full automated sequence: settle → stand →
LocoMode → trigger skill → hold through `motion_length` + margin → cooldown
→ back to LocoMode. "Final pelvis height" is read at exit; > 0.5 m means
still standing.

| Skill | Length | Runs (vendored `third_party/FSMDeployG1` path) | Verdict |
|---|---|---|---|
| `dance` (Charleston) | 18.0s | 2/3 clean (0.770 m, 0.770 m final); 1/3 fell during `skill_cooldown` (~0.08 m) | Mostly stable — re-run and watch the live window if used for the actual demo; one anomalous run seen |
| `kungfu` | 17.4s | 1/1 clean, ended standing at 0.770 m | Stable |
| `kick` | 3.6s | 1/1 clean, ended standing at 0.770 m | Stable |
| `beyondmimic` | 140s | interactive smoke test only — fell repeatedly (~0.06 m) partway through, never recovered | Not demo-ready |
| `kungfu2` | 18.4s | not reachable — no `FSMCommand` wired to it in this fork's `LocoMode.checkChange()`; excluded from `SKILLS` | Excluded |

**Recommended for the hackathon demo:** `kungfu` and `kick` are the most
consistently reproducible so far (single clean run each, no observed
failures). `dance` is usable but showed one intermittent fall out of three
runs in this environment — the cause (display/EGL timing vs. a genuine
`skill_cooldown` transition issue) is not yet root-caused; watch the live
MuJoCo window rather than trusting it fully unattended. Do not use
`beyondmimic` for a live demo without further stabilization work.
