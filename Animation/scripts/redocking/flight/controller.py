# -*- coding: utf-8 -*-
"""
controller.py - flight guidance/control for terminal docking.

FlightController runs EITHER a trained RL policy (fixed weights, port-frame
canonicalised observation) OR a verified model-based fallback (glideslope PD),
with automatic fallback if the RL policy is missing/faulted.  It conditions the
command: clips to RCS authority, applies a deadband, and limits the commanded
closing speed (far vs terminal).  Consumes the ESTIMATED state from navigation.
"""
from __future__ import annotations
import numpy as np
from dynamics.attitude import mrp_error_to_point, mrp_to_dcm, dock_attitude_error
from eval.rollout import PortFramePD
from flight import config_flight as F

POS_S, VEL_S, RATE_S = 100.0, 0.10, 0.10


def load_flight_agent(ckpt_path):
    """Load a trained terminal-docking policy for the flight loop.

    Infers the network dims from the checkpoint so single (12->6) and HER (15->6)
    policies both load.  Rejects a swarm policy (96->48): the flight loop is
    per-deputy.  Returns (agent, in_dim, n_act) or raises with a clear message.
    """
    import torch
    from agents.ddpg import DDPGAgent
    ck = torch.load(ckpt_path, map_location="cpu")
    aw = ck["actor"]
    in_dim = int(aw["fc1.weight"].shape[1]); n_act = int(aw["mu.weight"].shape[0])
    if n_act != 6 or in_dim not in (12, 15):
        raise ValueError(f"{ckpt_path}: policy is {in_dim}->{n_act}; the flight loop needs a "
                         f"single-dock 6-DOF policy (12->6 or HER 15->6). Train with --env single|her.")
    agent = DDPGAgent(in_dim, n_act)
    agent.load_models(ckpt_path)
    return agent, in_dim, n_act


class FlightController:
    def __init__(self, ports, agent=None, prefer_rl=True):
        self.ports = ports
        self.agent = agent
        self.fallback = PortFramePD(k_lat=0.20, k_ax=0.05, kd=4.0)  # flight-tuned: null lateral first
        self.prefer_rl = prefer_rl and (agent is not None)
        self.max_dv = F.ACTUATORS["dv_max_mps"]; self.max_tau = F.ACTUATORS["tau_max_Nm"]
        self.mode = "RL" if self.prefer_rl else "FALLBACK"
        gi = getattr(F, 'INTEGRAL', {})
        self._int_on = bool(gi.get('enabled', True))
        self.ki = float(gi.get('ki', 3e-3)); self.i_clip = float(gi.get('clip_mps', 0.03))
        self.dt = F.GNC['dt']; self.I = np.zeros(3)

    def _obs(self, r, v, sigma, port):
        p, vp = self.ports.to_port_frame(r, v, port)
        err_b = mrp_to_dcm(sigma) @ mrp_error_to_point(sigma, -self.ports.dir_B[port])
        # omega is appended by caller-provided estimate; return 9 dims here + omega outside
        return np.concatenate([p / POS_S, vp / VEL_S, err_b]), (p, vp)

    def command(self, r_est, v_est, sigma_est, omega_est, port, force_fallback=False):
        o9, (p, vp) = self._obs(r_est, v_est, sigma_est, port)
        obs = np.concatenate([o9, np.asarray(omega_est, float) / RATE_S])   # 12-D
        use_rl = self.prefer_rl and not force_fallback
        if use_rl:
            try:
                a = self.agent.choose_action(self._match_dim(obs), add_noise=False)
                self.mode = "RL"
            except Exception:
                a = self.fallback.choose_action(obs); self.mode = "FALLBACK(err)"
        else:
            a = self.fallback.choose_action(obs); self.mode = "FALLBACK"
        a = np.clip(np.asarray(a, float).reshape(6), -1, 1)
        dv_port = a[:3] * self.max_dv; tau = a[3:] * self.max_tau
        # guidance integral action: null the steady-state port-frame offset a constant
        # perturbation induces under proportional control (impulsive FF cannot - see synthetic).
        if self._int_on:
            lim = self.i_clip / max(self.ki, 1e-12)
            self.I = np.clip(self.I + p * self.dt, -lim, lim)     # anti-windup on the contribution
            dv_port = dv_port - self.ki * self.I
        # --- speed limiting IN THE PORT FRAME: cap AXIAL closing, preserve LATERAL ---
        dist = float(np.linalg.norm(p))
        vmax = F.GNC["approach_speed_terminal"] if dist < 3.0 else F.GNC["approach_speed_max"]
        v_next = vp + dv_port                       # resulting port-frame velocity
        if v_next[0] < -vmax:                       # +x is outboard, closing is negative
            v_next[0] = -vmax
        lat_sp = np.linalg.norm(v_next[1:])
        if lat_sp > vmax:                            # cap lateral speed too (but don't zero it)
            v_next[1:] *= vmax / lat_sp
        dv_port = v_next - vp
        dv_port = np.where(np.abs(dv_port) < 0.2 * F.ACTUATORS["dv_min_impulse_mps"], 0.0, dv_port)
        # --- attitude: VERIFIED full-3-DOF PD (reaction wheels), overrides policy tau ---
        serr = dock_attitude_error(sigma_est, self.ports.dir_B[port])
        g = F.ATT_CTRL
        tau = np.clip(-g["Kp"] * serr - g["Kd"] * np.asarray(omega_est, float),
                      -self.max_tau, self.max_tau)
        return self.ports.action_to_lvlh(dv_port, port), tau

    def _match_dim(self, obs12):
        """Pad the 12-D obs to the agent's input dim (HER policies expect +3 goal=0)."""
        n = None
        try:
            n = self.agent.actor.fc1.in_features
        except Exception:
            n = 12
        if n == 12:
            return obs12
        if n == 15:
            return np.concatenate([obs12, np.zeros(3)])
        return obs12


if __name__ == "__main__":
    import config_redock as C
    from envs.rafti import RaftiPorts
    r, u, rad = C.load_espa_ports()
    ports = RaftiPorts(r, u, rad, C.RAFTI, deputy_dims=C.ESPA["deputy_dims"], height=C.ESPA["height_m"])
    fc = FlightController(ports, agent=None)                 # fallback only
    port = 0
    r0 = ports.capture_pt[port] + 60 * ports.dir_B[port]     # 60 m out on the corridor
    dv, tau = fc.command(r0, np.zeros(3), np.zeros(3), np.zeros(3), port)
    print("mode:", fc.mode, "| dv:", np.round(dv, 4), "| tau:", np.round(tau, 6))
    assert np.linalg.norm(dv) > 0 and np.linalg.norm(dv) <= F.ACTUATORS["dv_max_mps"] * np.sqrt(3) + 1e-9
    print("controller OK")
