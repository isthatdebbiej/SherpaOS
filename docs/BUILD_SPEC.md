# BUILD_SPEC.md — frozen product/demo scope

Source of truth: `../../docs/plan.md` (workspace-level). This file freezes the scope that
code in this repo implements; if it drifts from `plan.md`, `plan.md` wins and this file
should be updated.

## Product

SherpaOS is an offline, auditable expedition-risk supervisor for Unitree G1. It combines
onboard telemetry with locally cached terrain context, evaluates five independent guard
families (mobility, dynamics/body, telemetry-health, battery-margin, geographic-risk),
and requests one of `PASS`, `LIMIT_SPEED`, `REQUEST_HOLD` via conservative fusion of the
guards. It sits above an existing locomotion controller.

The core proof: the same hidden traction change or disturbance produces a safer outcome
with SherpaOS than with (1) controller only, (2) a naive IMU threshold, (3) an
always-slow policy.

Every intervention has a reason, timestamp, input-freshness result, requested action,
applied/rejected receipt, and incident evidence bundle.

## Must work with all of these absent

physical robot, host telemetry dump, internet, Hugging Face, Vultr, Jetson Thor, remote
G1 connection. Those resources improve evidence; none may be a stage dependency.

## Mandatory deliverables (see plan.md section 2 for full list)

1. Reproducible G1 MuJoCo scenario with a controller.
2. Normal case + mixed-traction/disturbance case.
3. Runtime telemetry contract with a leakage test.
4. Deterministic risk estimator + safety state machine.
5. Learned temporal model only if it beats the deterministic rules (stretch).
6. Real simulated speed-limit/hold intervention.
7. Paired baseline evaluation, >=100 episodes (target 500).
8. Incident recorder + store-and-forward queue.
9. ONNX export/parity test + latency benchmark (stretch, needs Jetson access).
10. One-command local demo, one-command evaluation.
11. Video/repo/description/track submission (human-owned, out of this repo's scope).
12. Simulated battery guard: state of charge, voltage sag, discharge rate, thermal
    stress; real fields replace simulated fields through the same contract when
    available.
13. Offline geographic-risk guard using a pinned open terrain artifact
    (`configs/terrain/ebc_route.json`) packaged on-device — see
    `configs/terrain/PROVENANCE.md`.

## Explicitly out of scope until every mandatory gate is green

photorealistic or full-Himalaya 3D rendering, calibrated energy-to-return prediction
beyond the bounded simulated battery-margin guard, LiveKit voice, phone teleop, ROS 2
bridge, Isaac Sim/Lab, TensorRT, second UI, new locomotion policy. (The small offline
geographic-risk artifact/guard itself is in scope — only fancy 3D map rendering is cut.)

## Simulation fallback (recorded decision)

Full dynamic G1 walking requires an external trained locomotion policy (LeRobot/Unitree
SDK weights) which is heavy, network-dependent to fetch, and risks violating the offline
constraint if not vendored carefully. Per `plan.md` Lane B's own fallback clause and
`idea.txt` Plan B/C, the first implementation uses a constrained G1 posture/stepping/
weight-shift task under a simple built-in PD controller, honestly labeled as such (not a
trained walking policy). This preserves the same supervisory product proof: telemetry in,
risk out, intervention changes outcome. Upgrading to a real trained controller is a
stretch goal, not a blocker. See `DECISIONS.md`.

## Acceptance gates (see plan.md section 9)

- Intervention changes a simulated outcome.
- Simulator secrets cannot reach runtime features (leakage test).
- Nominal progress >= 95% of controller-only.
- False `REQUEST_HOLD` <= 10%, target < 5%.
- Unsafe continuation improves over controller-only and naive IMU threshold.
- Safety/progress tradeoff improves over always-slow.
- Stale/malformed/missing/NaN/out-of-order telemetry fails conservatively.
- Every intervention has an actuation receipt and incident artifact.
- >=100 paired episodes complete.
- Offline demo and recorded replay both work.
