"""Derived state: turn raw G1 telemetry into decisions an expedition can act on.

Sensors give you numbers. Guards need judgements. This is the layer between.

Deliberately dependency-free and type-agnostic -- every input is a plain float
or numpy array, so it slots in behind whatever `RobotTelemetry` shape you
settled on without touching frozen contracts.

    ds = DerivedState(mass_kg=32.1)
    out = ds.update(t, tau=..., dq=..., rpy=..., foot_forces=(l, r),
                    foot_contacts=(l, r), cmd_vel=0.5, speed=0.47,
                    ambient_c=-15.0)
    out["range_remaining_m"], out["traction_ratio"], out["gait_regularity"]

MODELLING ASSUMPTIONS (state these on your slide, do not hide them):
  * Pack energy defaults to 9 Ah x 15 cells x 3.7 V nominal ~= 500 Wh.
  * Drivetrain efficiency and idle draw are CALIBRATED so that level walking at
    0.5 m/s reproduces Unitree's quoted ~2 h runtime. Joint mechanical power
    alone underestimates draw by ~3x -- it ignores gearbox and motor losses,
    the onboard computer, LiDAR and thermal management. Do not remove the
    calibration and quote raw mechanical watts; it will read as a 6-hour
    battery, which is wrong by a factor of three.
  * The cold-capacity curve is a documented Li-ion approximation, not a
    measurement of this pack. It is a model, and you should say so.
"""

from collections import deque

import numpy as np

# Li-ion usable capacity vs temperature. Approximate, widely-cited shape:
# full capacity around 25 C, roughly half by -20 C.
_COLD_TEMP_C = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 25.0, 40.0])
_COLD_FRAC = np.array([0.40, 0.55, 0.72, 0.85, 0.94, 1.00, 0.98])


def cold_capacity_fraction(ambient_c):
    """Fraction of rated capacity actually usable at this temperature."""
    return float(np.interp(ambient_c, _COLD_TEMP_C, _COLD_FRAC))


class DerivedState:
    def __init__(self, mass_kg=32.1, pack_wh=500.0, idle_w=100.0,
                 drivetrain_eff=0.30, stride_window=12, slip_threshold=0.6,
                 initial_battery_fraction=1.0):
        self.mass = mass_kg
        self.pack_wh = pack_wh
        self.idle_w = idle_w              # compute + LiDAR + thermal, always on
        self.drivetrain_eff = drivetrain_eff
        self.slip_threshold = slip_threshold
        self.initial_battery_fraction = float(np.clip(initial_battery_fraction, 0.0, 1.0))

        self.energy_wh = 0.0          # cumulative consumption
        self.distance_m = 0.0
        self._t_prev = None
        self._strikes = deque(maxlen=stride_window)   # footstrike timestamps
        self._prev_contact = (False, False)
        self._power_hist = deque(maxlen=100)
        self._speed_hist = deque(maxlen=100)

    # ------------------------------------------------------------------
    def update(self, t, tau, dq, rpy, foot_forces, foot_contacts,
               cmd_vel, speed, ambient_c=25.0):
        tau = np.asarray(tau, dtype=float)
        dq = np.asarray(dq, dtype=float)
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        self._t_prev = t

        # -- power and energy ------------------------------------------
        # Mechanical power at the joints plus a fixed compute/idle draw.
        mech_w = float(np.abs(tau * dq).sum())
        # Electrical draw = mechanical work / efficiency, plus always-on load.
        power_w = mech_w / self.drivetrain_eff + self.idle_w
        self.energy_wh += power_w * dt / 3600.0
        self.distance_m += abs(speed) * dt
        self._power_hist.append(power_w)
        self._speed_hist.append(abs(speed))

        # -- cold-derated remaining range ------------------------------
        cold_frac = cold_capacity_fraction(ambient_c)
        usable_wh = self.pack_wh * cold_frac
        remaining_wh = max(0.0, usable_wh * self.initial_battery_fraction - self.energy_wh)
        soc = remaining_wh / usable_wh if usable_wh > 0 else 0.0

        mean_p = float(np.mean(self._power_hist)) if self._power_hist else power_w
        mean_v = float(np.mean(self._speed_hist)) if self._speed_hist else 0.0
        # Wh per metre at the terrain difficulty we are actually seeing.
        if mean_v > 0.05:
            wh_per_m = (mean_p / 3600.0) / mean_v
            range_m = remaining_wh / wh_per_m if wh_per_m > 0 else 0.0
        else:
            wh_per_m = float("nan")
            range_m = 0.0
        endurance_s = remaining_wh / mean_p * 3600.0 if mean_p > 0 else 0.0

        # -- traction --------------------------------------------------
        # The robot commands a velocity; the ground decides what it gets.
        # Sustained shortfall IS loss of traction -- no extra sensor needed.
        traction = abs(speed) / abs(cmd_vel) if abs(cmd_vel) > 0.05 else 1.0
        traction = float(np.clip(traction, 0.0, 1.5))
        slipping = bool(abs(cmd_vel) > 0.05 and traction < self.slip_threshold)

        # -- gait timing -----------------------------------------------
        lc, rc = bool(foot_contacts[0]), bool(foot_contacts[1])
        if lc and not self._prev_contact[0]:
            self._strikes.append(t)
        self._prev_contact = (lc, rc)

        gait_hz = gait_cv = float("nan")
        if len(self._strikes) >= 3:
            iv = np.diff(np.asarray(self._strikes))
            if iv.mean() > 0:
                gait_hz = float(1.0 / iv.mean())
                # Coefficient of variation: low = metronomic, high = stumbling.
                gait_cv = float(iv.std() / iv.mean())

        both_down = lc and rc
        airborne = not lc and not rc

        # -- posture / stability ---------------------------------------
        roll, pitch = float(rpy[0]), float(rpy[1])
        tilt_deg = float(np.degrees(np.hypot(roll, pitch)))
        fallen = tilt_deg > 50.0

        # -- load symmetry (limping / one-sided load) ------------------
        fl, fr = float(foot_forces[0]), float(foot_forces[1])
        tot = fl + fr
        asymmetry = abs(fl - fr) / tot if tot > 1.0 else 0.0

        return {
            # energy
            "power_w": power_w,
            "mean_power_w": mean_p,
            "energy_wh_used": self.energy_wh,
            "nominal_capacity_wh": self.pack_wh,
            "usable_capacity_wh": usable_wh,
            "remaining_energy_wh": remaining_wh,
            "initial_battery_fraction": self.initial_battery_fraction,
            "soc": soc,
            "cold_capacity_frac": cold_frac,
            "wh_per_m": wh_per_m,
            "range_remaining_m": range_m,
            "endurance_s": endurance_s,
            "distance_m": self.distance_m,
            # terrain
            "traction_ratio": traction,
            "slipping": slipping,
            # gait
            "gait_hz": gait_hz,
            "gait_regularity_cv": gait_cv,
            "double_support": both_down,
            "airborne": airborne,
            "load_asymmetry": asymmetry,
            # posture
            "tilt_deg": tilt_deg,
            "fallen": fallen,
        }

    # ------------------------------------------------------------------
    def can_reach(self, distance_m, ambient_c=25.0, round_trip=True,
                  reserve=0.25):
        """THE turn-back question: can we still get there and back?

        reserve=0.25 keeps a quarter of usable range in hand, the sort of
        margin a real party insists on. Returns a dict, or None if we have
        not moved enough yet to estimate consumption.
        """
        mean_p = float(np.mean(self._power_hist)) if self._power_hist else 0.0
        mean_v = float(np.mean(self._speed_hist)) if self._speed_hist else 0.0
        if mean_p <= 0.0 or mean_v <= 0.05:
            return None
        usable_wh = self.pack_wh * cold_capacity_fraction(ambient_c)
        remaining_wh = max(0.0, usable_wh - self.energy_wh)
        wh_per_m = (mean_p / 3600.0) / mean_v
        range_m = remaining_wh / wh_per_m
        required_m = distance_m * (2.0 if round_trip else 1.0) / (1.0 - reserve)
        return {
            "range_m": range_m,
            "required_m": required_m,
            "margin_m": range_m - required_m,
            "decision": "GO" if range_m > required_m else "TURN BACK",
        }
