"""Frozen Zealot G1 v26 policy contract used by the Himalayan playground."""

from __future__ import annotations

from collections import deque

import numpy as np

POLICY_REPOSITORY = "haixuantao/zealot-g1-locomotion"
POLICY_FILENAME = "g1_v26_iter42290.onnx"
POLICY_SHA256 = "1e21412a09f3af7fa2dbdec58de4d4600e2679862a1b24c502c0a02916bd440f"
POLICY_URL = f"https://huggingface.co/{POLICY_REPOSITORY}/resolve/main/{POLICY_FILENAME}"
JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
DEFAULT_POSE = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2, dtype=np.float32)
JOINT_LIMITS = np.array(
    [
        [-2.5307, 2.8798],
        [-0.5236, 2.9671],
        [-2.7576, 2.7576],
        [-0.087267, 2.8798],
        [-0.87267, 0.5236],
        [-0.2618, 0.2618],
        [-2.5307, 2.8798],
        [-2.9671, 0.5236],
        [-2.7576, 2.7576],
        [-0.087267, 2.8798],
        [-0.87267, 0.5236],
        [-0.2618, 0.2618],
    ],
    dtype=np.float32,
)
KP = np.array([100, 100, 100, 200, 40, 40] * 2, dtype=np.float64)
KD = np.array([2.5, 2.5, 2.5, 5, 2, 2] * 2, dtype=np.float64)


def projected_gravity(quaternion: np.ndarray) -> np.ndarray:
    """Return gravity in the base frame for a MuJoCo wxyz quaternion."""
    w, x, y, z = (float(value) for value in quaternion)
    return np.array(
        [
            -2 * (x * z - w * y),
            -2 * (y * z + w * x),
            -(1 - 2 * (x * x + y * y)),
        ],
        dtype=np.float32,
    )


class V26ObservationHistory:
    """Build the policy's five-frame, 240-value observation history."""

    def __init__(self) -> None:
        self.frames: deque[np.ndarray] = deque(maxlen=5)
        self.actions: deque[np.ndarray] = deque(
            [np.zeros(12, np.float32), np.zeros(12, np.float32)], maxlen=2
        )
        self.previous_q: np.ndarray | None = None
        self.phase = 0.0
        self.steps = 0

    def build(
        self,
        q: np.ndarray,
        quaternion: np.ndarray,
        body_gyro: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32)
        qdot = np.zeros(12, np.float32)
        if self.previous_q is not None:
            qdot = (q - self.previous_q) / 0.02
        if self.steps < 2:
            qdot.fill(0)
        self.previous_q = q.copy()
        self.phase = (self.phase + 0.02 / 0.7) % 1.0
        frame = np.zeros(48, np.float32)
        frame[:12] = self.actions[0] if self.steps >= 2 else 0
        frame[12:16] = command
        frame[16:28] = q - DEFAULT_POSE
        frame[28:40] = qdot
        frame[40:43] = projected_gravity(quaternion)
        frame[43:45] = [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)]
        frame[45:48] = body_gyro
        if not self.frames:
            self.frames.extend(frame.copy() for _ in range(5))
        else:
            self.frames.append(frame)
        self.steps += 1
        return np.concatenate(self.frames)[None].astype(np.float32)

    def record_action(self, action: np.ndarray) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32).copy())


def action_target(action: np.ndarray) -> np.ndarray:
    """Apply the published v26 action scale and G1 joint limits."""
    action = np.clip(np.asarray(action, dtype=np.float32), -10, 10)
    return np.clip(DEFAULT_POSE + 0.5 * action, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def robot_visibility(segmentation: np.ndarray, robot_geom_ids: np.ndarray) -> tuple[int, int]:
    """Return robot pixel count and bounding-box height from MuJoCo segmentation."""
    mask = np.isin(segmentation[:, :, 0], robot_geom_ids)
    rows, _columns = np.nonzero(mask)
    height = int(rows.max() - rows.min() + 1) if rows.size else 0
    return int(mask.sum()), height
