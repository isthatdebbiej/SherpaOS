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
from PIL import Image, ImageDraw, ImageFont

from sherpaos.datasets.generate import (
    _command_for,
    _terrain_zone_for,
    _wind_for,
    scenario_for,
)
from sherpaos.sim import disturbances
from sherpaos.sim.himalaya_scene import scene_xml, terrain_slope_for_geom
from sherpaos.sim.runner import FALL_PELVIS_Z_M, FALL_TILT_DEG
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
from sherpaos.sim.v26_runner import (
    BASELINE_FRICTION,
    HAZARD_ONSET_STEP,
    HAZARD_RAMP_STEPS,
    LEFT_FOOT_BODY,
    RIGHT_FOOT_BODY,
    TERRAIN_GEOMS,
    _body_geoms,
    _contacting_terrain_geoms,
    _foot_slip,
    _geom_id,
)
from sherpaos.sim.weather import aerodynamic_force_n, wind_speed_at_step


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
    parser.add_argument("--category", choices=("nominal", "mobility", "dynamics", "combined"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--storm", action="store_true")
    return parser.parse_args()


def _blowing_snow(frame: np.ndarray, rng: np.random.Generator, wind_mps: float) -> np.ndarray:
    """Add deterministic camera-space snow streaks; wind physics is applied separately."""
    result = frame.copy()
    height, width, _ = result.shape
    count = int(180 + 14 * wind_mps)
    xs = rng.integers(0, width, count)
    ys = rng.integers(0, height, count)
    length = np.maximum(2, (wind_mps / 5.0 + rng.random(count) * 5).astype(int))
    for x, y, streak in zip(xs, ys, length, strict=True):
        end = min(width, x + int(streak))
        result[max(0, y - 1) : min(height, y + 1), x:end] = (235, 245, 255)
    return result


def _robot_frame_margin(segmentation: np.ndarray, robot_geom_ids: np.ndarray) -> int:
    """Minimum distance from any robot pixel to a frame edge; zero means cropped."""
    mask = np.isin(segmentation[:, :, 0], robot_geom_ids)
    rows, columns = np.nonzero(mask)
    if not rows.size:
        return 0
    height, width = mask.shape
    return int(min(rows.min(), height - 1 - rows.max(), columns.min(), width - 1 - columns.max()))


def _risk_state(
    *,
    forecast_wind_mps: float,
    current_wind_mps: float,
    slope_deg: float,
    slip_mps: float,
    tilt_deg: float,
) -> str:
    if forecast_wind_mps >= 50.0 or slip_mps >= 0.6 or tilt_deg >= 25.0:
        return "NO-GO"
    if current_wind_mps >= 15.0 or slope_deg >= 16.0 or slip_mps >= 0.4 or tilt_deg >= 15.0:
        return "CAUTION"
    return "GO"


def _condition_overlay(frame: np.ndarray, lines: list[str], risk_state: str) -> np.ndarray:
    """Draw readable, deterministic condition evidence without hiding the robot."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 21)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 25)
    except OSError:
        font = title_font = ImageFont.load_default()
    color = {"GO": (40, 190, 90), "CAUTION": (240, 175, 35), "NO-GO": (225, 65, 65)}[risk_state]
    draw.rounded_rectangle(
        (24, 22, 470, 224), radius=14, fill=(5, 12, 24, 205), outline=color + (255,), width=3
    )
    draw.text((44, 34), f"SHERPA RISK: {risk_state}", font=title_font, fill=color + (255,))
    for index, line in enumerate(lines):
        draw.text((44, 75 + 28 * index), line, font=font, fill=(245, 248, 255, 255))
    return np.asarray(image)


def main() -> None:
    args = arguments()
    scenario = scenario_for(args.category, args.seed) if args.category and args.seed else None
    category_index = args.seed % 50 if scenario is not None else 0
    terrain_zone = _terrain_zone_for(args.category, category_index) if scenario is not None else 0
    wind_target_mps = (
        55.6
        if args.storm
        else _wind_for(args.category, category_index)
        if scenario is not None
        else 0.0
    )
    if scenario is not None:
        args.command_vx, args.command_vy, args.command_yaw = _command_for(category_index)
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
        handle.write(scene_xml(terrain_zone))
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
    terrain_ids = {_geom_id(model, name) for name in TERRAIN_GEOMS}
    terrain_ids.discard(-1)
    feet = tuple(_body_geoms(model, name) for name in (LEFT_FOOT_BODY, RIGHT_FOOT_BODY))
    if scenario is not None:
        initial_friction = max(scenario.friction, BASELINE_FRICTION)
        for geom_id in terrain_ids:
            model.geom_friction[geom_id, 0] = initial_friction
        for foot_geoms in feet:
            for geom_id in foot_geoms:
                model.geom_friction[geom_id, 0] = initial_friction
    mujoco.mj_forward(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape[-1] != 240 or session.get_outputs()[0].shape[-1] != 12:
        raise SystemExit("unexpected v26 ONNX input/output contract")
    history, previous_target = V26ObservationHistory(), None
    robot_geom_ids = np.flatnonzero(model.geom_bodyid > 0)
    visibility: list[tuple[int, int, int]] = []
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
    physics_step = 0
    storm_rng = np.random.default_rng((args.seed or 0) + 991)
    wind_speed_mps = 0.0
    try:
        for step in range(args.steps):
            effective_friction = 1.0
            if scenario is not None:
                friction_phase = np.clip((step - HAZARD_ONSET_STEP) / HAZARD_RAMP_STEPS, 0.0, 1.0)
                friction_phase = friction_phase * friction_phase * (3.0 - 2.0 * friction_phase)
                effective_friction = initial_friction + friction_phase * (
                    scenario.friction - initial_friction
                )
                for geom_id in terrain_ids:
                    model.geom_friction[geom_id, 0] = effective_friction
                for foot_geoms in feet:
                    for geom_id in foot_geoms:
                        model.geom_friction[geom_id, 0] = effective_friction
            if scenario is not None and scenario.actuator_health < 1.0 and step >= 200:
                alpha = min(1.0, (step - 200) / 50.0)
                health = 1.0 + alpha * (scenario.actuator_health - 1.0)
                for index, aid in enumerate(aids):
                    model.actuator_gainprm[aid, 0] = KP[index] * health
                    model.actuator_biasprm[aid, 1] = -KP[index] * health
                    model.actuator_biasprm[aid, 2] = -KD[index] * health
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
            wind_speed_mps = wind_speed_at_step(step, wind_target_mps)
            drag_force = aerodynamic_force_n(wind_speed_mps)
            for _ in range(10):
                if scenario is not None:
                    disturbances.apply_disturbance_wrench(model, data, scenario, physics_step)
                data.xfrc_applied[pelvis, 1] += drag_force
                mujoco.mj_step(model, data)
                physics_step += 1
            gravity = projected_gravity(data.qpos[3:7])
            tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))))
            contact_geoms = _contacting_terrain_geoms(data, feet, terrain_ids)
            slope_deg = max(
                (
                    terrain_slope_for_geom(
                        terrain_zone,
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                    )
                    for geom_id in contact_geoms
                ),
                default=0.0,
            )
            slip_mps = max(_foot_slip(model, data, foot, terrain_ids) for foot in feet)
            risk_state = _risk_state(
                forecast_wind_mps=wind_target_mps,
                current_wind_mps=wind_speed_mps,
                slope_deg=slope_deg,
                slip_mps=slip_mps,
                tilt_deg=tilt,
            )
            min_height, max_tilt, completed = (
                min(min_height, float(data.qpos[2])),
                max(max_tilt, tilt),
                step + 1,
            )
            if step % 2 == 0:
                camera = mujoco.MjvCamera()
                camera.type, camera.distance, camera.azimuth, camera.elevation = (
                    mujoco.mjtCamera.mjCAMERA_FREE,
                    3.3,
                    90,
                    -20,
                )
                camera.lookat[:] = [float(data.qpos[0]) + 0.1, 0, 0.62]
                renderer.update_scene(data, camera=camera)
                assert encoder.stdin is not None
                frame = renderer.render()
                frame = _blowing_snow(frame, storm_rng, wind_speed_mps)
                frame = _condition_overlay(
                    frame,
                    [
                        f"Wind now: {wind_speed_mps:4.1f} m/s ({wind_speed_mps * 3.6:5.1f} km/h)",
                        (
                            f"Wind forecast: {wind_target_mps:4.1f} m/s "
                            f"({wind_target_mps * 3.6:5.1f} km/h)"
                        ),
                        f"Slope/contact: {slope_deg:4.1f} deg   friction: {effective_friction:.2f}",
                        f"Foot slip: {slip_mps:.2f} m/s   body tilt: {tilt:.1f} deg",
                    ],
                    risk_state,
                )
                encoder.stdin.write(frame.tobytes())
                if step % 50 == 0:
                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(data, camera=camera)
                    segmentation = renderer.render()
                    pixels, height_px = robot_visibility(segmentation, robot_geom_ids)
                    visibility.append(
                        (pixels, height_px, _robot_frame_margin(segmentation, robot_geom_ids))
                    )
                    renderer.disable_segmentation_rendering()
            if data.qpos[2] < FALL_PELVIS_Z_M or tilt > FALL_TILT_DEG:
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
    min_frame_margin_px = min((value[2] for value in visibility), default=0)
    if min_pixels < 3500 or min_height_px < 150 or min_frame_margin_px < 30:
        args.output.unlink(missing_ok=True)
        raise SystemExit(
            f"robot visibility gate failed: {min_pixels} pixels, {min_height_px}px tall, "
            f"{min_frame_margin_px}px border margin"
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
        "minimum_robot_frame_margin_px": min_frame_margin_px,
        "visibility_gate": "PASS",
        "category": args.category,
        "seed": args.seed,
        "storm": wind_target_mps >= 25.0,
        "terrain_zone": terrain_zone,
        "wind_target_mps": wind_target_mps,
        "condition_overlay": True,
        "overlay_fields": [
            "risk_state",
            "current_wind_mps",
            "forecast_wind_mps",
            "contact_slope_deg",
            "friction",
            "foot_slip_mps",
            "body_tilt_deg",
        ],
        "wind_physics_applied": True,
        "wind_speed_at_end_mps": wind_speed_mps,
        "wind_speed_at_end_kmh": wind_speed_mps * 3.6,
        "snow_rendering": "deterministic visual overlay; physics from aerodynamic force",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
