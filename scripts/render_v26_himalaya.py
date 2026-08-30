"""Render the full-body G1 with the frozen v26 policy on icy Himalayan terrain."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from sherpaos.sim.himalaya_scene import scene_xml
from sherpaos.sim.v26_playground import (
    DEFAULT_POSE,
    JOINTS,
    KD,
    KP,
    POLICY_SHA256,
    V26ObservationHistory,
    action_target,
    projected_gravity,
    robot_visibility,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", type=Path, default=Path("var/policies/v26/g1_v26_iter42290.onnx")
    )
    parser.add_argument(
        "--g1-dir", type=Path, default=Path("third_party/mujoco_menagerie/unitree_g1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/playground/v26-himalaya-visible.mp4")
    )
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--command-vx", type=float, default=0.4)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")
    if not args.g1_dir.joinpath("g1.xml").is_file():
        raise SystemExit(f"missing MuJoCo Menagerie G1 at {args.g1_dir}")
    if (
        not args.policy.is_file()
        or hashlib.sha256(args.policy.read_bytes()).hexdigest() != POLICY_SHA256
    ):
        raise SystemExit("missing or unverified v26 policy; run scripts/fetch_v26_policy.py")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("install the playground extra: uv sync --extra playground") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=args.g1_dir, delete=False) as handle:
        handle.write(scene_xml())
        scene_path = Path(handle.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    finally:
        scene_path.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key)
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
    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape[-1] != 240 or session.get_outputs()[0].shape[-1] != 12:
        raise SystemExit("unexpected v26 ONNX input/output contract")
    history, previous_target = V26ObservationHistory(), None
    robot_geom_ids = np.flatnonzero(model.geom_bodyid > 0)
    visibility: list[tuple[int, int]] = []
    start_x, min_height, max_tilt, fell, completed = float(data.qpos[0]), 10.0, 0.0, False, 0
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1280x720",
            "-r",
            "25",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    renderer = mujoco.Renderer(model, height=720, width=1280)
    try:
        for step in range(args.steps):
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis, velocity, 1)
            observation = history.build(
                data.qpos[qadr],
                data.qpos[3:7],
                velocity[:3],
                np.array([args.command_vx, args.command_vy, args.command_yaw, 0], np.float32),
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
            gravity = projected_gravity(data.qpos[3:7])
            tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))))
            min_height, max_tilt, completed = (
                min(min_height, float(data.qpos[2])),
                max(max_tilt, tilt),
                step + 1,
            )
            if step % 2 == 0:
                camera = mujoco.MjvCamera()
                camera.type, camera.distance, camera.azimuth, camera.elevation = (
                    mujoco.mjtCamera.mjCAMERA_FREE,
                    4.0,
                    90,
                    -20,
                )
                camera.lookat[:] = [float(data.qpos[0]) + 0.1, 0, 0.62]
                renderer.update_scene(data, camera=camera)
                assert encoder.stdin is not None
                encoder.stdin.write(renderer.render().tobytes())
                if step % 50 == 0:
                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(data, camera=camera)
                    visibility.append(robot_visibility(renderer.render(), robot_geom_ids))
                    renderer.disable_segmentation_rendering()
            if data.qpos[2] < 0.5 or tilt > 50:
                fell = True
                break
    finally:
        renderer.close()
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
    if return_code:
        raise SystemExit("ffmpeg failed")
    min_pixels = min((value[0] for value in visibility), default=0)
    min_height_px = min((value[1] for value in visibility), default=0)
    if min_pixels < 5000 or min_height_px < 120:
        args.output.unlink(missing_ok=True)
        raise SystemExit(
            f"robot visibility gate failed: {min_pixels} pixels, {min_height_px}px tall"
        )
    report = {
        "policy": "v26 iter42290 frozen",
        "controller_only": True,
        "sherpaos_intervention": False,
        "steps": completed,
        "fell": fell,
        "forward_m": float(data.qpos[0] - start_x),
        "min_height_m": min_height,
        "max_tilt_deg": max_tilt,
        "minimum_robot_pixels": min_pixels,
        "minimum_robot_height_px": min_height_px,
        "visibility_gate": "PASS",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
