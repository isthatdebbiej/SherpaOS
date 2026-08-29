# DECISIONS.md — timestamped decisions and rejected alternatives

## 2026-08-29 11:24 PT — Repo location and layout

Repo root created at `sherpaos/` (containing package `sherpaos/sherpaos/`), per explicit
user instruction, nested under the workspace `SherpaOS/` that holds `docs/plan.md` and
`docs/idea.txt`. Standard src-less Python layout: repo root name == importable package
name.

## 2026-08-29 11:24 PT — plan.md is authoritative over idea.txt

`docs/plan.md` (workspace-level) is the "operational source of truth" per its own header
and is dated the same day as kickoff. `docs/idea.txt` is the earlier, more detailed
spec plan.md was distilled from (IceSense/BodySense/EnergyTerrain -> collapsed into a
single deterministic risk estimator + state machine; six-state policy
CONTINUE/CAUTION/SLOW/HOLD/RETREAT/STOP -> collapsed into
PASS/LIMIT_SPEED/REQUEST_HOLD). Where they conflict, plan.md's architecture wins;
idea.txt is used only for implementation detail/rationale (e.g. residual-based anomaly
detection, energy-margin math, evaluation suite structure) that plan.md references but
doesn't spell out.

**Rejected alternative**: implementing idea.txt's five-module architecture
(IceSense/BodySense/EnergyTerrain/AnchorSense/SherpaPolicy) verbatim. Rejected because
plan.md explicitly simplifies to one estimator + one state machine for the hackathon
timeline and because idea.txt itself says "two implementation decisions I'd lock now:
submit under Track 3... treat IceSense + BodySense + EnergyTerrain + policy + evaluation
as the non-negotiable core" — i.e. the modules can be internal feature groups inside a
single estimator rather than separate top-level products. The estimator module keeps
internal feature groups named after these concepts (traction/slip features, body-residual
features) so the distinction is preserved without three separate top-level packages.

## 2026-08-29 11:24 PT — Simulation controller fallback

No pre-trained G1 walking policy is vendored in this pass (network/weight-size risk,
offline-constraint risk, integration risk within the session budget). First
implementation uses a constrained MuJoCo G1 posture/stepping task under a simple
built-in PD controller. This is explicitly sanctioned by plan.md Lane B's own fallback
clause ("If stable walking is still unavailable at the two-hour cutoff, use a
constrained G1 posture/stepping task... label it honestly") and idea.txt's Plan B/C.
Upgrading to a real trained locomotion controller (LeRobot G1) is a stretch item, not a
blocker for the intervention-proof gate.

**Rejected alternative**: integrating LeRobot's G1 sim/locomotion path directly.
Rejected for this pass due to added dependency weight and setup risk; revisit once the
core estimator/policy/evaluation loop is proven end-to-end.

## 2026-08-29 11:40 PT — Menagerie vendored via sparse+partial clone, not full clone

`mujoco_menagerie` is only needed locally for the G1 model files (`unitree_g1/`) so the
local/offline demo has an asset to load — this is unrelated to Vultr/HF cloud usage,
which is for parallel Monte Carlo/overnight validation at scale, not for hosting this
asset. A first attempt did a full `git clone` (1.6GB, every robot in the menagerie);
corrected to `git clone --filter=blob:none --sparse --depth 1` +
`git sparse-checkout set unitree_g1`, which fetches blobs on demand and lands at ~56MB.
Pinned commit: `da76818e269b82289eba39808e2fb91d679d6994`. The directory is gitignored
in this repo (vendored, not committed) — see `third_party/ATTRIBUTIONS.md` for the
license/commit record.

## 2026-08-29 11:24 PT — Cloud (HF Jobs / Vultr) not exercised in this pass

No credentials are configured in this environment. `sherpa overnight launch/status/fetch`
are implemented as CLI surface + local-only stand-ins (they validate preflight
conditions and can run a local shard) rather than issuing real HF/Vultr API calls. Wiring
real cloud calls is left for whoever has the account/credentials, per plan.md section 6's
preflight-refusal contract, which this local stand-in already enforces.
