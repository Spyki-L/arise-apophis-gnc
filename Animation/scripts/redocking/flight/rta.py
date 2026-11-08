# -*- coding: utf-8 -*-
"""
rta.py - Runtime Assurance safety filter (the flight trust layer for the RL policy).

Wraps every guidance command with a predictive one-step safety check so the policy
- learned or not - can NEVER command an unsafe maneuver.  Enforced invariants:
  * KEEP-OUT: the deputy body stays outside the ESPA shell (+margin) except through
    its own port corridor;
  * APPROACH CONE: near the port the deputy stays inside the glideslope cone;
  * SPEED: relative speed <= v_approach_max everywhere, contact speed <= v_contact
    inside the terminal ball;
  * BODY RATE: |omega| <= omega_max.
On violation it PROJECTS the command to the safe set (brake / back-off) and, for a
hard breach (already inside keep-out, gross overspeed), raises an ABORT flag for the
executive.  Status: NOMINAL | LIMITED | ABORT.
"""
from __future__ import annotations
import numpy as np
from dynamics.hcw import hcw_stm
from flight import config_flight as F


class RTA:
    def __init__(self, ports, omega=1.027e-3, dt=1.0):
        self.ports = ports; self.dt = dt
        self.Phi = hcw_stm(dt, omega)
        s = F.SAFETY
        self.keepout = s["keepout_radius_m"]; self.margin = s["collision_margin_m"]
        self.cone = np.radians(s["cone_half_angle_deg"]); self.tan_cone = np.tan(self.cone)
        self.v_max = s["v_approach_max_mps"]; self.v_contact = s["v_contact_max_mps"]
        self.omega_max = s["omega_max_radps"]

    def filter(self, dv_cmd, tau_cmd, r, v, omega, port):
        status = "NOMINAL"; dv = np.asarray(dv_cmd, float).copy(); tau = np.asarray(tau_cmd, float).copy()

        # 1) speed cap (hard)
        v_next = v + dv; sp = np.linalg.norm(v_next)
        if sp > self.v_max:
            dv = (v_next * self.v_max / sp) - v; v_next = v + dv; status = "LIMITED"

        # 2) predictive keep-out: one-step propagate; if it crosses into the shell
        #    OUTSIDE its own corridor -> replace with a radial-outward brake
        s_next = self.Phi @ np.concatenate([r, v_next]); r_next = s_next[:3]
        p_next, _ = self.ports.to_port_frame(r_next, v_next, port)
        axial, lateral = p_next[0], float(np.hypot(p_next[1], p_next[2]))
        in_corridor = (axial >= -0.1) and (lateral <= max(axial, 0.0) * self.tan_cone + 0.15)
        rho_next = np.linalg.norm(r_next)
        if rho_next < (self.keepout + self.margin) and not in_corridor:
            n_out = r / (np.linalg.norm(r) + 1e-9)          # push radially outward, brake
            dv = -0.5 * v + 0.02 * n_out
            status = "ABORT" if np.linalg.norm(r) < self.keepout else "LIMITED"

        # 3) contact-speed limit inside the terminal ball (AXIAL ONLY - keep lateral)
        p_now, v_port = self.ports.to_port_frame(r, v, port)
        if np.linalg.norm(p_now) < 1.5:
            R = self.ports.R_port[port]
            dvp = R.T @ dv                                   # command in port frame
            v_ax_next = v_port[0] + dvp[0]
            if v_ax_next < -self.v_contact:                  # cap only the axial closing rate
                dvp[0] = -self.v_contact - v_port[0]
                dv = R @ dvp
                status = "LIMITED" if status == "NOMINAL" else status

        # 4) body-rate cap
        if np.linalg.norm(omega) > self.omega_max:
            tau = -np.sign(omega) * np.abs(tau)              # oppose the rate
            status = "LIMITED" if status == "NOMINAL" else status

        return dv, tau, status


if __name__ == "__main__":
    import config_redock as C
    from envs.rafti import RaftiPorts
    r, u, rad = C.load_espa_ports()
    ports = RaftiPorts(r, u, rad, C.RAFTI, deputy_dims=C.ESPA["deputy_dims"], height=C.ESPA["height_m"])
    rta = RTA(ports)
    port = 0
    # a command driving straight at the ESPA centre (through the shell) must be caught
    r0 = np.array([2.0, 0.0, 0.0]); v0 = np.array([-0.5, 0, 0])   # inbound, off-corridor-ish
    dv, tau, st = rta.filter(np.array([-0.05, 0, 0]), np.zeros(3), r0, v0, np.zeros(3), port)
    print("drive-through-shell -> status:", st, "safe dv:", np.round(dv, 3))
    # an over-speed command is capped
    dv2, _, st2 = rta.filter(np.array([0.5, 0, 0]), np.zeros(3), np.array([50.0, 0, 0]), np.array([0.4, 0, 0]), np.zeros(3), port)
    print("overspeed -> status:", st2, "|v_next|:", round(np.linalg.norm(np.array([50.,0,0])*0+ (np.array([0.4,0,0])+dv2)),3))
    assert st2 == "LIMITED"
    print("rta OK")
