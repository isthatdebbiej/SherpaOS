# STATUS.md — current state

**Last updated:** 2026-08-29 PT
**Base SHA:** `f8562169` (dataset-pipeline implementation base)
**Working tree:** uncommitted dataset, expedition-memory, Field Journal, and documentation work
**Latest Python verification on record:** Ruff green; full pytest suite green (150 passed); 2-episode dataset contract GREEN
**Latest web verification:** Next.js 16.3.3 production build GREEN after the animated trail redesign

## Implemented

- Real expedition-memory vertical slice: immutable per-day `.mcap`/rosbag2 `.db3`
  upload, streaming SHA-256, read-only bag inspection, topic/message/time coverage,
  allowlisted onboard-topic policy, and explicit privileged-topic rejection.
- Local FastAPI boundary for day manifests, uploads, read-only voice tools, and
  push-to-talk audio turns. Voice turns are locked until a verified bag exists and
  require `OPENAI_API_KEY`; the model receives the verified bag hash and approved
  topic coverage, never simulator truth or actuation authority.
- Expedition-memory backend routes support real day selection, upload status, manifest
  checksum/message/topic data, and grounded read-only voice turns. The redesigned
  frontend has not yet reconnected these routes.
- Redesigned `web/` presentation shell with original Everest relief/topographic styling,
  animated central ascent, clickable five-day navigation, written past-day entries,
  current-day spoken-thought fixture, locked future days, Pemba reactions, snow/parallax,
  memory charms, lantern mode, responsive layout, and reduced-motion support.

- MuJoCo Menagerie G1 posture/stepping simulator with nominal, mixed-traction,
  disturbance, actuator-health, slope, sensor-noise, and synthetic battery scenarios.
- Five independent guard reports: mobility, dynamics, telemetry health, battery margin,
  and offline geographic risk.
- Conservative max/action-floor fusion: one severe guard cannot be averaged away.
- Policy hysteresis with `PASS`, `LIMIT_SPEED`, and `REQUEST_HOLD`.
- Simulation actuation adapter with one `ActuationReceipt` per decision.
- Battery simulation with explicit simulated provenance for state of charge, current,
  voltage sag, and temperature.
- Offline Everest Base Camp route artifact and geographic guard.
- Incident/evidence bundle serialization, including battery fields and all guard reports.
- Executable CLI: `sherpa preflight`, `sherpa test`, `sherpa simulate`, and
  `sherpa demo --offline`.
- Checksum-verified nominal and hazard demo smoke run.
- Isolated, pinned MuJoCo Playground v0.2.0 bootstrap with CUDA JAX GPU gate.
- Flat- and rough-terrain G1 reset/step smoke checks with non-finite rejection.
- Explicit Playground observation-to-telemetry adapter that rejects privileged truth.
- Rollout evidence metadata with code/Playground SHAs, GPU/JAX identity, policy hash,
  provenance, license, command, seed, and artifact checksums.
- Vultr clean-SHA validation/evidence packaging and optional rscope viewer launcher.

## Verified

- `ruff check .`: green.
- Full pytest suite: 127 passed.
- `sherpa preflight`: GREEN (G1 asset, terrain artifact, five-guard smoke).
- `sherpa demo --offline`: GREEN; 300/300 steps survived in nominal and hazard runs;
  299 decisions and 299 receipts per run; evidence checksums verified.
- Nominal run produced no `REQUEST_HOLD`; it did spend 112/299 decisions in
  `LIMIT_SPEED` after 17 transient dynamics elevations plus policy cooldown. Do not tune
  this away without paired evaluation of nominal-progress impact.

## Environment note

Claude's `.venv` is owned by a different Windows SID from the Codex runner. Codex used
an isolated `.venv-codex` for verification and did not replace Claude's environment.
Git commands in evidence generation now pass the repository as an explicit safe directory.

## Expedition memory checkpoint (2026-08-29 PT)

- Raw bags live under gitignored `var/expeditions/<expedition>/day-XX/raw/`; promotion is
  atomic and an existing day cannot be replaced implicitly.
- rosbag2 SQLite inspection works without a ROS runtime. MCAP inspection uses the locked
  `mcap` expedition extra. Full message decoding/time-series Parquet derivation remains
  the next backend slice.
- Full pytest suite and Ruff are green; the Next.js production build is green.
- Live voice was not called during verification because no API credential was supplied.

## Field Journal / Vultr checkpoint (2026-08-29 PT)

- Deployment is now Vultr-only. Vercel and Hugging Face are not application dependencies.
- The intended Vultr layout keeps raw day bags, derived reports, diary JSON, narration,
  Python API, and Next.js app on the same host.
- No general Vultr account API key is required for the app to read local instance files.
  Real reflection/voice requires a server-only `OPENAI_API_KEY`; uploads/completion
  notifications require a separate application secret.
- The new visual experience is currently fixture-driven and is not yet connected to the
  existing expedition API. Browser `speechSynthesis` is a temporary interaction fixture,
  not the final narration path.
- `npm run build` passes. Browser QA confirmed clickable past diary entries, current-day
  thought controls, locked future days, and the Everest-style vertical route composition.
- Documentation was reconciled across the workspace plan, README, scope, contracts,
  decisions, runbook, status, handoff, attribution, and web README.

## Next three tasks

1. Freeze a derived `DayReport`/`DiaryEntry` API contract and connect the redesigned
   Next.js page to `sherpaos.expedition.api` without sending raw bags to the browser.
2. Add the post-day reflection and narration worker on Vultr; persist evidence references,
   diary JSON, and audio beside each immutable day.
3. Add Vultr systemd plus nginx/Caddy deployment files, run clean-host verification, and
   commit/tag the integration checkpoint before generating judging evidence.

## Dataset pipeline checkpoint (2026-08-29 PT)

- Added frozen 200-episode scenario/config contracts: 50 nominal, 50 mobility,
  50 dynamics, and 50 combined controller-only episodes.
- Added deterministic, resumable 10-episode NPZ shards with 500 control steps,
  100-sample windows, 50-sample prediction horizon, and stride 10.
- Observation arrays are fixed at 103 onboard-observable features; privileged
  friction/slope/slip/actuator/disturbance/fall truth remains under `labels/`.
- Added `sherpa data generate`, `sherpa data validate`, and `sherpa data split` only.
- Validation rejects count, integrity, leakage, alignment, finite-value, width,
  duplicate-ID, group-overlap, missing-shard, and positive-rate failures.
- Acceptance commands completed GREEN for the allowed two-episode local contract;
  mobility positive rate was 0.50 and feature width was 103.
- Full suite: 150 passed. Ruff: green.

## Next three tasks (dataset pipeline)

1. Push the committed dataset-pipeline SHA and clone that exact SHA on Vultr.
2. Run exactly 200 episodes on Vultr, then validate and freeze checksums.
3. Download and checksum-verify the immutable dataset before any later training task.

## Frozen v26 Himalayan playground checkpoint (2026-08-29 PT)

- Added reproducible v26 iter42290 ONNX download with pinned SHA-256 verification.
- Added its exact 240-input/12-action controller and full Menagerie G1 renderer.
- Added a segmentation visibility gate so an occluded G1 cannot pass.
- See docs/V26_HIMALAYA_PLAYGROUND.md for teammate commands.

## FSMDeployG1 dance/skill demo lane (2026-08-30 PT)

- Added `scripts/fetch_fsm_dance_repo.py` (pins and vendors the
  Renforce-Dynamics/FSMDeployG1 fork into gitignored `third_party/FSMDeployG1/`,
  patches its FSM to skip unavailable mimic policies) and `scripts/run_g1_dance.py`
  (fully automated stand -> walk -> trigger skill -> cooldown sequence, no manual
  keypress timing).
- Verified per-skill: `kungfu` and `kick` completed cleanly and ended standing in
  every run tested; `dance` completed cleanly in 2/3 runs and fell once during
  cooldown (cause not yet root-caused); `beyondmimic` repeatedly falls and is not
  demo-ready; `kungfu2` has no reachable FSM trigger in this fork and was excluded.
- This is a separate, non-safety demo lane — SherpaOS does not intervene in it. See
  docs/G1_DANCE_DEMO.md for full results and docs/DECISIONS.md for the license note
  (FSMDeployG1 has no declared upstream license; nothing from it is redistributed).
- Next: re-run `dance` several more times to root-cause the intermittent cooldown
  fall before relying on it for a live demo.
