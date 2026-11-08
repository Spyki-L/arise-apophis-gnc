# -*- coding: utf-8 -*-
"""
fdir.py - Fault Detection, Isolation and Recovery for the docking GNC.

Monitors the closed loop each step and raises isolated faults with a recommended
recovery action for the executive:
  THRUSTER_STUCK    commanded burn not realised (cmd/realized mismatch persists)
  NAV_DIVERGED      EKF position covariance above the flight limit
  CORRIDOR_BREACH   persistent approach-cone violation near the port
  CONTACT_OVERSPEED closing speed above the abort factor near the port
  PROP_LOW          propellant budget nearly exhausted
Recommended actions: NOMINAL | SAFE_HOLD | ABORT.  (The executive owns the response;
FDIR only detects + recommends - clean separation of concerns.)
"""
from __future__ import annotations
import numpy as np
from flight import config_flight as F


class FDIR:
    def __init__(self):
        d = F.FDIR
        self.thr_gate = d["thruster_stuck_steps"]; self.cor_gate = d["corridor_breach_steps"]
        self.over = d["overspeed_factor"]; self.prop_abort = d["dv_budget_frac_abort"]
        self._thr = 0; self._cor = 0; self._comm = 0; self.active = set()
        self.comm_gate = d.get("comm_loss_steps", 20)

    def monitor(self, dv_cmd, dv_realized, nav_sigma, dist, in_cone, v_axial, prop_frac,
                comm_ok=True):
        faults = []
        # thruster: nonzero command not realised
        cmag = np.linalg.norm(dv_cmd)
        if cmag > 2 * F.ACTUATORS["dv_min_impulse_mps"]:
            self._thr = self._thr + 1 if np.linalg.norm(dv_realized) < 0.3 * cmag else 0
        else:
            self._thr = 0
        if self._thr >= self.thr_gate:
            faults.append("THRUSTER_STUCK")
        # nav divergence
        if nav_sigma > F.FDIR["nav_cov_diverge_m"]:
            faults.append("NAV_DIVERGED")
        # corridor breach near the port
        if dist < 5.0:
            self._cor = self._cor + 1 if not in_cone else 0
            if self._cor >= self.cor_gate:
                faults.append("CORRIDOR_BREACH")
        else:
            self._cor = 0
        # contact overspeed near the port
        if dist < 1.0 and abs(v_axial) > self.over * F.SAFETY["v_contact_max_mps"]:
            faults.append("CONTACT_OVERSPEED")
        # propellant
        if prop_frac > self.prop_abort:
            faults.append("PROP_LOW")
        # ISL comm health: transient outages are normal; a persistent loss of the
        # fleet link degrades collision-avoidance coordination
        self._comm = self._comm + 1 if not comm_ok else 0
        if self._comm >= self.comm_gate:
            faults.append("COMM_DEGRADED")

        self.active = set(faults)
        if {"CORRIDOR_BREACH", "CONTACT_OVERSPEED"} & self.active:
            action = "ABORT"
        elif {"THRUSTER_STUCK", "NAV_DIVERGED", "PROP_LOW", "COMM_DEGRADED"} & self.active:
            action = "SAFE_HOLD"
        else:
            action = "NOMINAL"
        return faults, action


if __name__ == "__main__":
    fd = FDIR()
    # simulate a stuck thruster: command a burn, realise ~0, repeatedly
    for _ in range(6):
        f, a = fd.monitor(np.array([0.03, 0, 0]), np.zeros(3), 1.0, 50.0, True, 0.0, 0.1)
    print("stuck thruster -> faults:", f, "action:", a); assert "THRUSTER_STUCK" in f and a == "SAFE_HOLD"
    fd2 = FDIR()
    f2, a2 = fd2.monitor(np.zeros(3), np.zeros(3), 1.0, 0.5, True, 0.20, 0.1)  # overspeed at contact
    print("contact overspeed -> faults:", f2, "action:", a2); assert a2 == "ABORT"
    print("fdir OK")
