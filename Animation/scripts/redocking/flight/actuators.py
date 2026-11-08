# -*- coding: utf-8 -*-
"""
actuators.py - flight actuator suite: commanded [dv, tau] -> REALIZED [dv, tau].

Uses the REAL nominal_v2 hardware when reachable (flight/hardware_bridge):
  * TRANSLATION : Aerojet MPS-130 (AF-M315E) via thruster.apply_burn - real Isp,
    minimum-impulse-bit quantization, thrust-vector misalignment, finite-burn loss,
    rocket-equation propellant.
  * ATTITUDE    : XACT-50-class reaction-wheel assembly via ReactionWheelParams.
    realized_torque - back-EMF ceiling, viscous+Coulomb friction, torque saturation,
    with per-axis wheel-momentum accumulation and saturation.
Plus command LATENCY and stuck-thruster FAULT injection.  Falls back to a
self-contained analytic model if the nominal_v2 hardware is not reachable.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from flight import config_flight as F
from flight import hardware_bridge as HWB


def _rot_small(angles):
    ax, ay, az = angles
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class RCS:
    """Deputy actuator suite (name kept for the flight loop): MPS-130 RCS for Δv +
    XACT-50 reaction wheels for torque."""
    def __init__(self, rng, fault_axis=None, dt=None):
        a = F.ACTUATORS; self.rng = rng
        self.dt = dt if dt is not None else F.GNC["dt"]
        self.fault_axis = fault_axis
        self.hw = HWB.AVAILABLE
        self.mass = HWB.deputy_mass_kg()
        # analytic fallback params
        self.dv_q = a["dv_min_impulse_mps"]; self.dv_max = a["dv_max_mps"]
        self.tau_q = a["tau_min_impulse_Nm"]; self.tau_max = a["tau_max_Nm"]
        self.mis = _rot_small(np.radians(a["misalign_sigma_deg"]) * rng.normal(size=3))
        self.scale = 1.0 + a["scale_sigma_frac"] * rng.normal()
        self.isp = a["isp_s"]; self.g0 = 9.80665; self.prop0 = a["prop_mass_kg"]
        # command latency buffer (dv, tau)
        self.lat = deque(maxlen=a["latency_steps"] + 1); self.lat_n = a["latency_steps"]
        # reaction-wheel state (per-axis stored momentum) and propellant
        self.h_wheel = np.zeros(3); self.prop_used = 0.0
        if self.hw:
            self.RWA = HWB.RWA; self.max_h = self.RWA.max_momentum_Nms

    @staticmethod
    def _quantize(x, q):
        return np.round(np.asarray(x, float) / q) * q

    def apply(self, dv_cmd, tau_cmd):
        dv = np.clip(np.asarray(dv_cmd, float), -self.dv_max, self.dv_max)
        tau = np.clip(np.asarray(tau_cmd, float), -self.tau_max, self.tau_max)

        # ---- translation: MPS-130 (real) or analytic ----
        if self.hw:
            dv_r, m_prop, _ = HWB.thruster.apply_burn(dv, self.mass, self.rng, HWB.MPS130)
            self.prop_used += m_prop
        else:
            dvq = self._quantize(dv, self.dv_q)
            dv_r = self.mis @ dvq * self.scale
            self.prop_used += self.mass * (1 - np.exp(-np.linalg.norm(dv_r) / (self.isp * self.g0)))
        if self.fault_axis is not None:
            dv_r = np.asarray(dv_r, float).copy(); dv_r[self.fault_axis] = 0.0   # stuck thruster

        # ---- attitude: XACT-50 reaction wheels (real) or analytic ----
        if self.hw:
            w_wheel = self.RWA.speed_from_momentum(self.h_wheel)          # flywheel speed
            tau_r = self.RWA.realized_torque(tau, w_wheel)               # back-EMF + friction + sat
            self.h_wheel = np.clip(self.h_wheel - tau_r * self.dt,       # wheels absorb reaction
                                   -self.max_h, self.max_h)
            # at momentum saturation the wheel cannot torque further (would need RCS dumping)
            sat = np.abs(self.h_wheel) >= self.max_h - 1e-9
            tau_r = np.where(sat & (np.sign(tau_r) == -np.sign(self.h_wheel)), 0.0, tau_r)
        else:
            tau_r = self._quantize(tau, self.tau_q) * self.scale

        # ---- command latency ----
        self.lat.append((np.asarray(dv_r, float), np.asarray(tau_r, float)))
        if len(self.lat) > self.lat_n:
            return self.lat[0]
        return np.zeros(3), np.zeros(3)

    @property
    def prop_frac(self):
        return self.prop_used / (self.prop0 + 1e-12)

    def depleted(self):
        return self.prop_used >= self.prop0

    @property
    def wheel_frac(self):
        return float(np.max(np.abs(self.h_wheel)) / self.max_h) if self.hw else 0.0


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    print(HWB.summary())
    rcs = RCS(rng)
    for _ in range(3):
        dv, tau = rcs.apply([0.03, -0.01, 0.0], [1e-3, 0, 0])
    print("realized dv:", np.round(dv, 4), "| tau:", np.round(tau, 6),
          "| prop %:", round(100 * rcs.prop_frac, 2), "| wheel %:", round(100 * rcs.wheel_frac, 1))
    rcs2 = RCS(rng, fault_axis=0)
    for _ in range(2):
        dvf, _ = rcs2.apply([0.03, 0.03, 0.03], [0, 0, 0])
    print("stuck-axis-0 dv:", np.round(dvf, 4)); assert abs(dvf[0]) < 1e-6
    print("actuators OK")
