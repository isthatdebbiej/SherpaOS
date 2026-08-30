# G1 v26 Himalayan playground

This reproduces the full-body G1 video using the frozen Zealot v26 `iter42290` policy.
It does not train locomotion and SherpaOS does not intervene.

## Setup

Install `ffmpeg`, then run:

```bash
uv sync --extra playground --extra dev
git clone --filter=blob:none --sparse --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie sparse-checkout set unitree_g1
git -C third_party/mujoco_menagerie checkout da76818e269b82289eba39808e2fb91d679d6994
uv run python scripts/fetch_v26_policy.py
```

The policy download is rejected unless its SHA-256 is
`1e21412a09f3af7fa2dbdec58de4d4600e2679862a1b24c502c0a02916bd440f`.

## Render

On a headless Linux/Vultr host:

```bash
MUJOCO_GL=egl uv run python scripts/render_v26_himalaya.py --output artifacts/playground/v26-himalaya-visible.mp4
```

The default produces 15 seconds at 50 Hz control and 25 FPS video. The scene contains
a 10-degree icy ascent, 11-degree cross-slope, uneven crusted snow, a rock step, and a
20-degree low-friction boundary. Snow is visual and rigid, not deformable.

The command deletes the MP4 and fails if sampled frames show fewer than 5,000 robot
pixels or a robot bounding-box height below 120 pixels. A JSON report beside the MP4
records survival, displacement, tilt, and visibility evidence.

Policy source: <https://huggingface.co/haixuantao/zealot-g1-locomotion>. Weights are
downloaded into gitignored `var/` and are not redistributed by SherpaOS.
