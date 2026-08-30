"""Scratch script: render single frames for candidate gesture arm poses."""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sherpaos.sim.unitree_walking import DEFAULT_CONFIG_PATH, _build_scene, _resolve_path  # noqa: E402

config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
source_xml = _resolve_path(DEFAULT_CONFIG_PATH, config["source_xml"])
default_angles = np.asarray(config["default_angles"], dtype=np.float32)

scene = _build_scene(source_xml, scenic_environment=True, arms_enabled=True)
model, data = scene.model, scene.data
model.vis.global_.offwidth = 1280
model.vis.global_.offheight = 720
renderer = mujoco.Renderer(model, height=720, width=1280)

# order: waist_yaw, L_sp, L_sr, L_sy, L_elbow, L_wr, R_sp, R_sr, R_sy, R_elbow, R_wr
POSES = {
    "energy_depletion_v2": np.array(
        [0.0, 0.0, 0.6, 0.0, 1.2, 0.0, 0.0, -0.6, 0.0, 1.2, 0.0], dtype=np.float32
    ),
    "energy_depletion_v3": np.array(
        [0.0, 0.0, 1.1, 0.0, 1.6, 0.0, 0.0, -1.1, 0.0, 1.6, 0.0], dtype=np.float32
    ),
}

camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.distance = 2.6
camera.azimuth = 160.0
camera.elevation = -8.0

out_dir = Path(__file__).parent
for name, pose in POSES.items():
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.793
    data.qpos[3] = 1.0
    data.qpos[7:19] = default_angles
    data.qpos[19:30] = pose
    mujoco.mj_forward(model, data)
    camera.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), 0.95]
    renderer.update_scene(data, camera=camera)
    frame = renderer.render()
    import imageio_ffmpeg

    from PIL import Image

    Image.fromarray(frame).save(out_dir / f"{name}.png")
    print("wrote", name)

renderer.close()
