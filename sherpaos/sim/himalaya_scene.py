"""Collision-backed icy Himalayan terrain for the full MuJoCo Menagerie G1."""

# ruff: noqa: E501 -- keep XML elements searchable for terrain review.

from __future__ import annotations

import struct
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

TERRAIN_ZONE_NAMES = ("snow_ramp", "cross_slope", "snow_crust", "rocky_snow", "steep_ice")
TERRAIN_ZONE_PROFILES = (
    (2.0, 4.0, 5.0, 4.0),
    (5.0, 8.0, 12.0, 9.0),
    (4.0, 7.0, 10.0, 6.0),
    (5.0, 9.0, 13.0, 8.0),
    (8.0, 10.0, 12.0, 15.0),
)
TERRAIN_ZONE_SLOPE_DEG = tuple(max(profile) for profile in TERRAIN_ZONE_PROFILES)
TERRAIN_ZONE_LENGTHS_M = (
    (1.125, 1.125, 1.125, 1.125),
    (1.125, 1.125, 1.125, 1.125),
    (1.125, 1.125, 1.125, 1.125),
    (1.125, 1.125, 1.125, 1.125),
    (0.65, 0.90, 1.30, 2.65),
)
TERRAIN_GRID_SIZE = 129
TERRAIN_HALF_X_M = 6.0
TERRAIN_HALF_Y_M = 2.5
TERRAIN_MAX_HEIGHT_M = 2.0


def terrain_heightmap(terrain_zone: int, size: int = TERRAIN_GRID_SIZE) -> np.ndarray:
    """Generate one deterministic snow/ice surface with a flat spawn apron."""
    if terrain_zone not in range(len(TERRAIN_ZONE_NAMES)):
        raise ValueError(f"unknown terrain zone {terrain_zone}")
    x = np.linspace(-TERRAIN_HALF_X_M, TERRAIN_HALF_X_M, size)
    y = np.linspace(-TERRAIN_HALF_Y_M, TERRAIN_HALF_Y_M, size)
    xx, yy = np.meshgrid(x, y)
    transition_width = 4.0
    progress = np.clip((xx - 1.50) / transition_width, 0.0, 1.0)
    blend = progress * progress * (3.0 - 2.0 * progress)
    forward = np.maximum(0.0, xx - 1.50)
    smooth_rise = transition_width * (progress**3 - 0.5 * progress**4)
    smooth_rise += np.maximum(0.0, forward - transition_width)
    if terrain_zone == 0:
        height = np.tan(np.deg2rad(4.0)) * smooth_rise
        height += 0.018 * blend * (np.sin(2.1 * xx) + 0.5 * np.sin(3.7 * yy))
    elif terrain_zone == 1:
        height = blend * (0.10 * forward + np.tan(np.deg2rad(8.0)) * (yy + 2.5))
    elif terrain_zone == 2:
        height = np.tan(np.deg2rad(6.0)) * smooth_rise
        height += blend * (
            0.16 * np.exp(-((xx - 1.65) ** 2 / 0.10 + (yy + 0.20) ** 2 / 0.30))
            + 0.22 * np.exp(-((xx - 2.35) ** 2 / 0.14 + (yy - 0.18) ** 2 / 0.25))
        )
    elif terrain_zone == 3:
        height = np.tan(np.deg2rad(7.0)) * smooth_rise
        rough = 0.055 * (np.sin(3.2 * xx + 0.8) * np.cos(2.7 * yy) + 1.0)
        rocks = 0.14 * np.maximum(0.0, np.sin(4.5 * xx) * np.cos(3.4 * yy)) ** 3
        height += blend * (rough + rocks)
    else:
        height = np.tan(np.deg2rad(15.0)) * smooth_rise
        height += 0.025 * blend * np.sin(2.4 * yy)
    height -= float(height.min())
    return np.clip(height / TERRAIN_MAX_HEIGHT_M, 0.0, 1.0)


@lru_cache(maxsize=len(TERRAIN_ZONE_NAMES))
def _terrain_gradient(terrain_zone: int) -> tuple[np.ndarray, np.ndarray]:
    height_m = terrain_heightmap(terrain_zone) * TERRAIN_MAX_HEIGHT_M
    dx = 2.0 * TERRAIN_HALF_X_M / (height_m.shape[1] - 1)
    dy = 2.0 * TERRAIN_HALF_Y_M / (height_m.shape[0] - 1)
    gradient_y, gradient_x = np.gradient(height_m, dy, dx)
    return gradient_x, gradient_y


def terrain_slope_deg_at(terrain_zone: int, x_m: float, y_m: float) -> float:
    """Return local surface grade from the same sampled heightfield MuJoCo loads."""
    gradient_x, gradient_y = _terrain_gradient(terrain_zone)
    column = int(
        round((x_m + TERRAIN_HALF_X_M) / (2 * TERRAIN_HALF_X_M) * (gradient_x.shape[1] - 1))
    )
    row = int(round((y_m + TERRAIN_HALF_Y_M) / (2 * TERRAIN_HALF_Y_M) * (gradient_x.shape[0] - 1)))
    row = int(np.clip(row, 0, gradient_x.shape[0] - 1))
    column = int(np.clip(column, 0, gradient_x.shape[1] - 1))
    grade = np.hypot(gradient_x[row, column], gradient_y[row, column])
    return float(np.degrees(np.arctan(grade)))


def write_terrain_png(path: Path, terrain_zone: int) -> None:
    """Write an 8-bit grayscale PNG without optional image dependencies."""
    pixels = np.rint(terrain_heightmap(terrain_zone) * 255.0).astype(np.uint8)
    height, width = pixels.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in pixels)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw, level=9))
    payload += chunk(b"IEND", b"")
    Path(path).write_bytes(payload)


def terrain_slope_for_geom(terrain_zone: int, geom_name: str | None) -> float:
    """Return the authored grade of the exact collision geom under a foot."""
    if terrain_zone not in range(len(TERRAIN_ZONE_NAMES)):
        raise ValueError(f"unknown terrain zone {terrain_zone}")
    if geom_name and geom_name.startswith("terrain_segment_"):
        index = int(geom_name.rsplit("_", 1)[1])
        return TERRAIN_ZONE_PROFILES[terrain_zone][index]
    return 0.0


def _profile_start_x(terrain_zone: int) -> float:
    return 0.5 if terrain_zone == 4 else 1.5


def _profile_height_m(terrain_zone: int, x_m: float) -> float:
    start_x = _profile_start_x(terrain_zone)
    height = 0.0
    for slope_deg, projected_length in zip(
        TERRAIN_ZONE_PROFILES[terrain_zone], TERRAIN_ZONE_LENGTHS_M[terrain_zone], strict=True
    ):
        covered = float(np.clip(x_m - start_x, 0.0, projected_length))
        height += covered * np.tan(np.deg2rad(slope_deg))
        start_x += projected_length
    return float(height)


def _terrain_segment_xml(terrain_zone: int, material: str) -> str:
    start_x = _profile_start_x(terrain_zone)
    thickness, height = 0.03, 0.0
    geoms: list[str] = []
    for index, (slope_deg, projected_length) in enumerate(
        zip(
            TERRAIN_ZONE_PROFILES[terrain_zone],
            TERRAIN_ZONE_LENGTHS_M[terrain_zone],
            strict=True,
        )
    ):
        angle = float(np.deg2rad(slope_deg))
        half_length = projected_length / (2.0 * np.cos(angle))
        center_x = start_x + projected_length / 2.0
        center_z = height + half_length * np.sin(angle) - thickness * np.cos(angle)
        geoms.append(
            f'<geom name="terrain_segment_{index}" type="box" '
            f'pos="{center_x:.6f} 0 {center_z:.6f}" '
            f'size="{half_length:.6f} 2.5 {thickness}" '
            f'euler="0 {-angle:.8f} 0" material="{material}" '
            'friction="0.65 0.01 0.001"/>'
        )
        height += projected_length * np.tan(angle)
        start_x += projected_length
    return "\n    ".join(geoms)


def scene_xml(terrain_zone: int = 0, heightmap_file: str = "terrain.png") -> str:
    """Build connected, visible, multi-grade collision terrain v26 can traverse."""
    del heightmap_file
    if terrain_zone not in range(len(TERRAIN_ZONE_NAMES)):
        raise ValueError(f"unknown terrain zone {terrain_zone}")
    material = "ice" if terrain_zone in {1, 4} else "snow"
    segments = _terrain_segment_xml(terrain_zone, material)
    apron_end = _profile_start_x(terrain_zone)
    apron_half_length = (apron_end + 6.0) / 2.0
    apron_center_x = (-6.0 + apron_end) / 2.0
    obstacles = ""
    if terrain_zone == 2:
        obstacles = f"""
    <geom name="crust_a" type="box" pos="3.05 -0.22 {_profile_height_m(terrain_zone, 3.05) + 0.055:.6f}" size="0.20 0.38 0.055" material="snow" friction="0.48 0.01 0.001"/>
    <geom name="crust_b" type="box" pos="4.05 0.20 {_profile_height_m(terrain_zone, 4.05) + 0.075:.6f}" size="0.22 0.34 0.075" material="snow" friction="0.42 0.01 0.001"/>"""
    elif terrain_zone == 3:
        obstacles = f"""
    <geom name="rock_step" type="box" pos="3.10 -0.24 {_profile_height_m(terrain_zone, 3.10) + 0.09:.6f}" size="0.18 0.30 0.09" material="rock" friction="0.72 0.02 0.002"/>
    <geom name="crust_a" type="box" pos="4.15 0.22 {_profile_height_m(terrain_zone, 4.15) + 0.07:.6f}" size="0.22 0.34 0.07" material="snow" friction="0.48 0.01 0.001"/>"""
    return f"""<mujoco model="connected himalayan snow grades">
  <include file="g1.xml"/>
  <option timestep="0.002" integrator="implicitfast" iterations="12"/>
  <visual><headlight diffuse="0.78 0.82 0.9" ambient="0.25 0.28 0.34" specular="1 1 1"/>
    <rgba haze="0.74 0.82 0.90 1"/><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.78 0.86 0.96" rgb2="0.10 0.16 0.25" width="512" height="3072"/>
    <texture type="2d" name="snowtex" builtin="checker" rgb1="0.95 0.98 1" rgb2="0.73 0.82 0.9" width="512" height="512"/>
    <material name="snow" texture="snowtex" texuniform="true" texrepeat="10 4" specular="0.32" shininess="0.22" reflectance="0.10"/>
    <material name="ice" rgba="0.40 0.68 0.84 1" specular="0.98" shininess="0.98" reflectance="0.42"/>
    <material name="rock" rgba="0.18 0.20 0.23 1" specular="0.12"/>
  </asset>
  <worldbody>
    <light directional="true" pos="-3 -4 8" dir="0.4 0.3 -1" diffuse="1 0.98 0.92" castshadow="true"/>
    <geom name="spawn_apron" type="box" pos="{apron_center_x:.6f} 0 -0.03" size="{apron_half_length:.6f} 2.5 0.03" material="snow" friction="0.65 0.01 0.001"/>
    {segments}{obstacles}
  </worldbody>
</mujoco>\n"""
