"""Icy Himalayan qualification scene for the full MuJoCo Menagerie G1."""

from __future__ import annotations


def scene_xml() -> str:
    return """<mujoco model="himalaya icy snow qualification">
  <include file="g1.xml"/>
  <option timestep="0.002" integrator="implicitfast" iterations="8"/>
  <visual><headlight diffuse="0.75 0.8 0.9" ambient="0.22 0.25 0.3" specular="1 1 1"/>
    <rgba haze="0.72 0.82 0.92 1"/><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.72 0.84 0.96" rgb2="0.1 0.18 0.28" width="512" height="3072"/>
    <texture type="2d" name="snowtex" builtin="checker" rgb1="0.92 0.96 1" rgb2="0.72 0.82 0.9" width="512" height="512"/>
    <material name="snow" texture="snowtex" texuniform="true" texrepeat="6 3" specular="0.25" shininess="0.18" reflectance="0.08"/>
    <material name="ice" rgba="0.42 0.68 0.82 1" specular="0.95" shininess="0.95" reflectance="0.35"/>
    <material name="rock" rgba="0.16 0.18 0.21 1" specular="0.15" shininess="0.1"/>
  </asset>
  <worldbody>
    <light directional="true" pos="-3 -4 8" dir="0.4 0.3 -1" diffuse="1 0.98 0.92" castshadow="true"/>
    <geom name="snow_floor" type="plane" size="20 10 0.05" material="snow" friction="0.65 0.01 0.001"/>
    <geom name="ice_ramp" type="box" pos="1.85 0 0.08" size="0.85 0.75 0.06" euler="0 -0.174533 0" material="ice" friction="0.16 0.005 0.0005"/>
    <geom name="landing" type="box" pos="3.15 0 0.24" size="0.48 0.75 0.06" material="snow" friction="0.5 0.01 0.001"/>
    <geom name="cross_slope" type="box" pos="4.05 0 0.24" size="0.48 0.75 0.06" euler="0.191986 0 0" material="ice" friction="0.12 0.004 0.0005"/>
    <geom name="crust1" type="box" pos="4.72 -0.18 0.31" size="0.24 0.48 0.10" euler="0 -0.087266 0.04" material="snow" friction="0.42 0.01 0.001"/>
    <geom name="crust2" type="box" pos="5.18 0.15 0.38" size="0.22 0.50 0.14" euler="-0.08 0.10 -0.04" material="snow" friction="0.38 0.01 0.001"/>
    <geom name="rock_step" type="box" pos="5.58 -0.05 0.43" size="0.18 0.55 0.18" material="rock" friction="0.75 0.02 0.002"/>
    <geom name="steep_boundary" type="box" pos="6.20 0 0.52" size="0.55 0.65 0.07" euler="0 -0.349066 0" material="ice" friction="0.08 0.003 0.0003"/>
  </worldbody>
</mujoco>\n"""
