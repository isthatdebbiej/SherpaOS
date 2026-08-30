# Final demo artifacts

Rendered, observer-only MP4s for the SherpaOS walking demonstration. Everything
here is display-only: the HUD reads the same aggregate telemetry snapshot the
live feed publishes, and no video/gesture path ever feeds a decision back into
the locomotion controller.

## What we added

- **`telemetry-decision-{lobuche,namche}.mp4`** — the core story: battery/range,
  live weather, and offline route/location telemetry are continuously fused into
  a per-tick go/no-go verdict (`PROCEED` / `LIMIT SPEED` / `TURN BACK`). Lobuche
  reads STOP (cold-derated range can't reach the safe waypoint); Namche reads
  PROCEED. Camera tracks the pelvis yaw so heading drift no longer looks like the
  robot walking sideways.

- **`gesture-signals-namche.mp4`** — the arm-articulated robot playing a scripted
  sequence of mountaineering hand signals (NOMINAL → HALT → PATH CLEAR → ENERGY
  LOW → SOS), labelled live in a "ROBOT SIGNAL" HUD panel. Uses the 23-DOF G1
  description (same 12 leg joints as the pinned policy, plus waist + arms); the
  legs are still driven by the untouched pinned policy, and the arms are held by a
  separate fixed PD toward the scheduled pose.

## Pose tuning = the robot's "emotions"

`_pose_tuning/` holds the still-frame renders used to design each arm pose — this
is the robot's body language, i.e. how it *reacts* and signals its internal state
to a human companion on the mountain:

- **`energy_depletion*.png`** — arms droop / hang low when the battery is low, both
  to conserve power and to *show* it is tired and running out of charge.
- **`halt_x.png`** — arms thrown up into a blocking "X": danger ahead, stop.
- **`sos_wave.png`** — both arms raised overhead: emergency / system failure, a
  distress call when something has gone wrong.
- **`nominal.png`** — relaxed neutral posture: all systems fine, confident.

`tune_poses.py` is the throwaway script that renders these static previews so the
joint-angle signs could be eyeballed before wiring them into the live schedule.

> Known limitation: the largest overhead swings (HALT, SOS) currently inject
> enough reaction torque to destabilize the walking policy mid-stride — the robot
> can topple during those gestures. The poses read clearly as still frames; making
> them stable while walking needs gentler/rate-limited arm motion.
