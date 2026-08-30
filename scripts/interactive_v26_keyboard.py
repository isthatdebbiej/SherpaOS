"""Interactive keyboard control of the Unitree G1 using the frozen,
checksum-verified v26 locomotion policy (haixuantao/zealot-g1-locomotion),
on a flat MuJoCo Menagerie G1 scene.

This reuses SherpaOS's exact v26 observation/action pipeline
(sherpaos.sim.v26_playground) - the same pinned policy verified in
docs/V26_HIMALAYA_PLAYGROUND.md to survive 300+ steps without falling -
but swaps the offscreen ffmpeg renderer for mujoco.viewer.launch_passive
with a live key_callback, so you can drive it with the keyboard in
real time instead of watching a pre-rendered MP4.

Run from the SherpaOS repo root with its venv active:
    python scripts/interactive_v26_keyboard.py

Controls (click the MuJoCo window first so it has keyboard focus):
  UP / W      - increase forward velocity command
  DOWN / S    - decrease forward velocity command (reverse)
  LEFT        - increase leftward (+y) velocity command
  RIGHT       - increase rightward (-y) velocity command
  A / D       - increase / decrease yaw rate command (turn left/right)
  SPACE       - zero all velocity commands (stand in place)
  ESC         - quit
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sherpaos.sim.v26_playground import (  # noqa: E402
    DEFAULT_POSE,
    JOINTS,
    KD,
    KP,
    POLICY_SHA256,
    V26ObservationHistory,
    action_target,
    projected_gravity,
)

POLICY_PATH = REPO_ROOT / "var/policies/v26/g1_v26_iter42290.onnx"
G1_DIR = REPO_ROOT / "third_party/mujoco_menagerie/unitree_g1"

VEL_STEP = 0.15
YAW_STEP = 0.3
VEL_MAX = 0.8
YAW_MAX = 1.0


class Commander:
    """Holds the live [vx, vy, yaw] command, updated from keypresses."""

    def __init__(self) -> None:
        self.cmd = np.zeros(3, dtype=np.float32)  # vx, vy, yaw_rate

    def key_callback(self, key: int) -> None:
        if key in (glfw.KEY_UP, glfw.KEY_W):
            self.cmd[0] = float(np.clip(self.cmd[0] + VEL_STEP, -VEL_MAX, VEL_MAX))
        elif key in (glfw.KEY_DOWN, glfw.KEY_S):
            self.cmd[0] = float(np.clip(self.cmd[0] - VEL_STEP, -VEL_MAX, VEL_MAX))
        elif key == glfw.KEY_LEFT:
            self.cmd[1] = float(np.clip(self.cmd[1] + VEL_STEP, -VEL_MAX, VEL_MAX))
        elif key == glfw.KEY_RIGHT:
            self.cmd[1] = float(np.clip(self.cmd[1] - VEL_STEP, -VEL_MAX, VEL_MAX))
        elif key == glfw.KEY_A:
            self.cmd[2] = float(np.clip(self.cmd[2] + YAW_STEP, -YAW_MAX, YAW_MAX))
        elif key == glfw.KEY_D:
            self.cmd[2] = float(np.clip(self.cmd[2] - YAW_STEP, -YAW_MAX, YAW_MAX))
        elif key == glfw.KEY_SPACE:
            self.cmd[:] = 0.0
        else:
            return
        print(f"[cmd] vx={self.cmd[0]:+.2f} vy={self.cmd[1]:+.2f} yaw={self.cmd[2]:+.2f}")


def main() -> None:
    if not G1_DIR.joinpath("scene.xml").is_file():
        raise SystemExit(f"missing MuJoCo Menagerie G1 at {G1_DIR}")
    if not POLICY_PATH.is_file() or (
        hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() != POLICY_SHA256
    ):
        raise SystemExit("missing or unverified v26 policy; run scripts/fetch_v26_policy.py")

    import onnxruntime as ort

    model = mujoco.MjModel.from_xml_path(str(G1_DIR / "scene.xml"))
    data = mujoco.MjData(model)

    # Use the Menagerie G1's built-in "stand"/"home" keyframe if present
    # (bent-knee, balanced pose) rather than qpos=0 (straight legs, easily
    # toppled) - same idea as render_v26_himalaya.py's mj_resetDataKeyframe.
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print(f"[init] Reset to keyframe id {key_id} (bent-knee stand pose)")
    else:
        print("[init] WARNING: no 'stand'/'home' keyframe found; using default qpos")

    qadr = np.array(
        [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in JOINTS
        ]
    )
    aids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINTS]
    )
    data.qpos[qadr] = DEFAULT_POSE
    for index, aid in enumerate(aids):
        model.actuator_gainprm[aid, 0] = KP[index]
        model.actuator_biasprm[aid, 1] = -KP[index]
        model.actuator_biasprm[aid, 2] = -KD[index]
    mujoco.mj_forward(model, data)

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    session = ort.InferenceSession(str(POLICY_PATH), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape[-1] != 240 or session.get_outputs()[0].shape[-1] != 12:
        raise SystemExit("unexpected v26 ONNX input/output contract")

    history = V26ObservationHistory()
    previous_target = None
    commander = Commander()
    step = 0

    print("\n=== v26 Interactive Keyboard Control ===")
    print(__doc__)
    print("==========================================\n")

    with mujoco.viewer.launch_passive(
        model, data, show_right_ui=False,
        key_callback=lambda key: commander.key_callback(key),
    ) as viewer:
        while viewer.is_running():
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis, velocity, 1)
            observation = history.build(
                data.qpos[qadr],
                data.qpos[3:7],
                velocity[:3],
                np.array([commander.cmd[0], commander.cmd[1], commander.cmd[2], 0], np.float32),
            )
            action = session.run(None, {session.get_inputs()[0].name: observation})[0][0]
            history.record_action(action)
            target = action_target(action)
            if step < 25:
                alpha = (step + 1) / 25
                target = (1 - alpha) * data.qpos[qadr] + alpha * target
            if previous_target is not None:
                target = np.clip(target, previous_target - 0.2, previous_target + 0.2)
            previous_target = target.copy()
            data.ctrl[aids] = target
            for _ in range(10):
                mujoco.mj_step(model, data)
            step += 1

            if step % 250 == 0:
                gravity = projected_gravity(data.qpos[3:7])
                tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))))
                print(f"[diag] step={step} height={data.qpos[2]:.3f}m tilt={tilt:.1f}deg")

            viewer.sync()

    print("Program exited")


if __name__ == "__main__":
    main()
