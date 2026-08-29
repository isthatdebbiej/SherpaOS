"""PD posture/stepping controller for the vendored Unitree G1 MJCF model.

Per `docs/DECISIONS.md`'s "Simulation controller fallback" entry, this is a
constrained posture/weight-shift task under a simple built-in PD controller,
**not** a trained locomotion policy. It drives the robot to the model's
`stand` keyframe pose and layers a small, continuous, low-amplitude
sinusoidal weight-shift/stepping motion on top of the hip/ankle targets
(out of phase left/right) so there is real stance-foot dynamics to observe
-- the robot sways/steps in place rather than standing like a statue.

Actuator model note: the G1 XML's 29 actuators are all MuJoCo `<position>`
actuators -- `actuator_gaintype == mjGAIN_FIXED` with `gainprm[0] == 500`
(kp) and `actuator_biastype == mjBIAS_AFFINE` with `biasprm == [0, -kp,
-kv]` (a joint-specific built-in damping `kv`). That is exactly MuJoCo's
built-in position-servo model, so this controller does **not** hand-roll
PD torque math itself -- it just writes joint-angle targets to `data.ctrl`
each physics step and lets MuJoCo's actuator model apply the PD torque
internally (see `sherpaos/sim/disturbances.py` for how `actuator_health`
scales this gain/damping to model degraded authority).

`speed_scale`/`hold` (both threaded through `step()`) are literally the
actuation channel `sherpaos/sim/runner.py`'s guard callback controls:
  - `hold=True` freezes the target at the plain `stand` pose (zero
    stepping motion) -- the physical meaning of `REQUEST_HOLD`.
  - `hold=False` scales the stepping amplitude *and* frequency by
    `speed_scale` in `[0, 1]` -- `speed_scale < 1` is `LIMIT_SPEED`,
    `speed_scale == 1` is `PASS`.
The gait phase is an accumulator advanced by `2*pi*freq*dt` every call
(not `phase = freq * t`), and it does not advance at all while frozen, so
toggling hold/speed_scale never introduces a phase discontinuity/jerk --
stepping resumes smoothly from wherever it left off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

STAND_KEYFRAME_NAME = "stand"

# Actuator indices (0-based), matching the actuator/joint declaration order
# in third_party/mujoco_menagerie/unitree_g1/g1.xml. Only the leg joints
# used by the weight-shift/stepping motion are named here.
LEFT_HIP_PITCH, RIGHT_HIP_PITCH = 0, 6
LEFT_HIP_ROLL, RIGHT_HIP_ROLL = 1, 7
LEFT_KNEE, RIGHT_KNEE = 3, 9
LEFT_ANKLE_PITCH, RIGHT_ANKLE_PITCH = 4, 10


@dataclass
class PDStepController:
    """Drives the G1 to a standing pose plus an optional weight-shift/
    stepping sinusoid layered onto the hip/ankle targets.

    Gains below were tuned empirically against this exact model (see the
    sim-lane report): at `base_freq_hz=0.5` with these amplitudes the
    stand pose is stable indefinitely at nominal friction with *zero*
    disturbance (max pelvis tilt ~1.2-1.7 deg over a 15-20s hold), while
    still being large enough that a 50-90N/~0.3s lateral pelvis shove
    reliably tips a *full-speed* (`speed_scale=1`, `hold=False`) episode
    into a fall that the same shove does not cause when `hold=True` or
    `speed_scale` is reduced even slightly (e.g. 0.9) -- i.e. there is a
    real, sizeable, demonstrable margin for the runtime guard to buy back
    by intervening. Roughly 30% more amplitude (e.g. hip_pitch 0.065 rad)
    is enough to make the *nominal, undisturbed* gait itself resonate and
    fall on its own within ~10s, so do not casually scale these up.
    """

    base_freq_hz: float = 0.5
    hip_pitch_amp_rad: float = 0.05
    hip_roll_amp_rad: float = 0.02
    ankle_pitch_amp_rad: float = 0.02
    knee_amp_rad: float = 0.02

    _stand_ctrl: np.ndarray | None = field(default=None, repr=False, compare=False)
    _phase: float = field(default=0.0, repr=False, compare=False)

    def reset(self, model: mujoco.MjModel) -> None:
        """(Re)load the `stand` keyframe's ctrl target and zero the gait
        phase. Call this once at the start of each episode.
        """
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, STAND_KEYFRAME_NAME)
        if key_id < 0:
            raise ValueError(f"model has no '{STAND_KEYFRAME_NAME}' keyframe")
        self._stand_ctrl = model.key_ctrl[key_id].copy()
        self._phase = 0.0

    def step(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
        speed_scale: float = 1.0,
        hold: bool = False,
    ) -> None:
        """Write this physics step's actuator targets to `data.ctrl`.

        Intended to be called once per physics step (not once per
        control/guard-decision step) so the internal phase accumulator
        advances at the model's true timestep -- see `sherpaos/sim/runner.py`
        for how the control-rate guard decision is zero-order-held across
        several calls to this method. `t` (currently `data.time`) is
        accepted for interface clarity / future use but is not read here;
        the phase accumulator is the source of truth so it survives
        `speed_scale` changing between calls without discontinuity.
        """
        del t  # unused: phase is accumulator-driven, see class docstring.
        if self._stand_ctrl is None:
            self.reset(model)
        assert self._stand_ctrl is not None

        target = self._stand_ctrl.copy()
        speed_scale = float(np.clip(speed_scale, 0.0, 1.0))

        if not hold and speed_scale > 0.0:
            dt = model.opt.timestep
            freq_hz = self.base_freq_hz * speed_scale
            self._phase += 2.0 * np.pi * freq_hz * dt
            s = float(np.sin(self._phase))

            hip_pitch = self.hip_pitch_amp_rad * speed_scale * s
            hip_roll = self.hip_roll_amp_rad * speed_scale * s
            ankle_pitch = self.ankle_pitch_amp_rad * speed_scale * s
            knee_lift = self.knee_amp_rad * speed_scale

            target[LEFT_HIP_PITCH] += hip_pitch
            target[RIGHT_HIP_PITCH] -= hip_pitch
            target[LEFT_HIP_ROLL] += hip_roll
            target[RIGHT_HIP_ROLL] -= hip_roll
            target[LEFT_ANKLE_PITCH] -= ankle_pitch
            target[RIGHT_ANKLE_PITCH] += ankle_pitch
            target[LEFT_KNEE] += knee_lift * max(0.0, s)
            target[RIGHT_KNEE] += knee_lift * max(0.0, -s)
        # else: hold (or speed_scale == 0) -- freeze at the plain stand
        # pose. The phase accumulator is deliberately *not* advanced here,
        # so stepping resumes in-phase (no jerk) once hold is lifted.

        data.ctrl[:] = target
