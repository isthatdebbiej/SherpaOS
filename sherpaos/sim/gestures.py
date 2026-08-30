"""Scripted mountaineering hand-signal gestures for the arm-articulated G1 demo.

Pure, offline, display-only: pose vectors and the time schedule live here so
both the physics loop (``unitree_walking.py``) and the video HUD
(``walk_video.py``) can independently derive "which gesture is active right
now" from ``data.time``, without either module owning the other's state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Order: waist_yaw, L_shoulder_pitch/roll/yaw, L_elbow, L_wrist_roll,
#        R_shoulder_pitch/roll/yaw, R_elbow, R_wrist_roll (11 joints, g1_23dof.xml).
_NOMINAL = np.zeros(11, dtype=np.float32)
_HALT = np.array(
    [0.0, -2.3, 1.9, 0.0, 1.9, 0.0, -2.3, -1.9, 0.0, 1.9, 0.0], dtype=np.float32
)
_PATH_CLEAR = np.array(
    [0.0, -1.5, 0.0, 0.0, 0.2, 0.0, 1.55, -0.35, 0.0, 0.55, 0.0], dtype=np.float32
)
_ENERGY_DEPLETION = np.array(
    [0.0, 0.0, 0.6, 0.0, 1.2, 0.0, 0.0, -0.6, 0.0, 1.2, 0.0], dtype=np.float32
)
_SOS_WAVE = np.array(
    [0.0, -2.5, 1.6, 0.0, 0.3, 0.0, -2.5, -1.6, 0.0, 0.3, 0.0], dtype=np.float32
)

GESTURE_POSES: dict[str, np.ndarray] = {
    "nominal": _NOMINAL,
    "halt": _HALT,
    "path_clear": _PATH_CLEAR,
    "energy_depletion": _ENERGY_DEPLETION,
    "sos_wave": _SOS_WAVE,
}

GESTURE_LABELS: dict[str, str] = {
    "nominal": "NOMINAL",
    "halt": "HALT — path obstructed",
    "path_clear": "PATH CLEAR — proceed",
    "energy_depletion": "ENERGY LOW — conserving power",
    "sos_wave": "SOS — system failure",
}

# 0 = OK, 1 = CAUTION, 2 = STOP -- matches walk_video.py's traffic-light levels.
GESTURE_LEVELS: dict[str, int] = {
    "nominal": 0,
    "path_clear": 0,
    "energy_depletion": 1,
    "halt": 2,
    "sos_wave": 2,
}


@dataclass(frozen=True, slots=True)
class GestureCue:
    start_s: float
    end_s: float
    name: str


DEFAULT_GESTURE_SCHEDULE: tuple[GestureCue, ...] = (
    GestureCue(0.0, 4.0, "nominal"),
    GestureCue(4.0, 8.0, "halt"),
    GestureCue(8.0, 9.0, "nominal"),
    GestureCue(9.0, 13.0, "path_clear"),
    GestureCue(13.0, 14.0, "nominal"),
    GestureCue(14.0, 18.0, "energy_depletion"),
    GestureCue(18.0, 19.0, "nominal"),
    GestureCue(19.0, 23.0, "sos_wave"),
    GestureCue(23.0, 27.0, "nominal"),
)

_BLEND_S = 0.6


def gesture_at(
    time_s: float, schedule: tuple[GestureCue, ...] = DEFAULT_GESTURE_SCHEDULE
) -> tuple[np.ndarray, str, str, int]:
    """Return (target arm pose, gesture name, label, level) for an episode time.

    Poses blend linearly for ``_BLEND_S`` seconds after each cue starts so the
    arms ease into a gesture instead of snapping.
    """
    index = len(schedule) - 1
    for candidate_index, cue in enumerate(schedule):
        if time_s < cue.end_s:
            index = candidate_index
            break
    active = schedule[index]
    pose = GESTURE_POSES[active.name].copy()

    if index > 0:
        elapsed = time_s - active.start_s
        if elapsed < _BLEND_S:
            previous_pose = GESTURE_POSES[schedule[index - 1].name]
            weight = float(np.clip(elapsed / _BLEND_S, 0.0, 1.0))
            pose = previous_pose + weight * (pose - previous_pose)

    return (
        pose.astype(np.float32),
        active.name,
        GESTURE_LABELS[active.name],
        GESTURE_LEVELS[active.name],
    )
