# -*- coding: utf-8 -*-
"""
multi_chaser_env.py - centralised 8-deputy 6-DOF RAFTI re-docking.

Full 6-DOF: each deputy has state [r, v, sigma, omega] (LVLH position/velocity +
MRP attitude body<-LVLH + body rate) and is commanded [dv, tau] (impulsive LVLH
Delta-v in the port frame + body-frame control torque).  The deputy must both fly
its cuboid onto the RAFTI port (radial glideslope, soft contact) AND slew its
docking FACE (+x_body) onto the port normal at a low body rate - a real docking.

  * dynamics: dynamics.get_sixdof  (HCW translation + rigid-body attitude | Basilisk)
  * capture : rafti.capture_status with attitude (face alignment + rate)
  * obs per deputy (12): port-frame [pos(3), vel(3)] + [MRP pointing-error(3), omega(3)]
  * action per deputy (6): [dv(3) port-frame, tau(3) body]
Centralised policy: 96 -> 48.  A captured deputy is frozen (docked).
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_redock as C
from dynamics import get_sixdof
from dynamics.attitude import mrp_error_to_point, pointing_error_deg, mrp_to_dcm
from envs.rafti import RaftiPorts

OBS_POS_SCALE = 100.0
OBS_VEL_SCALE = 0.10
OBS_RATE_SCALE = 0.10


class MultiChaserRedock:
    def __init__(self, n=None, dynamics=None, seed=None, ports=None, env_cfg=None,
                 handoff=None, rafti=None, chief_state_fn=None, t0_mission=0.0):
        self.N = int(n or C.N_DEPUTIES)
        e = dict(C.ENV); e.update(env_cfg or {})
        self.cfg = e
        self.dt = e["dt"]; self.max_steps = e["max_steps"]
        self.max_dv = e["max_dv_mps"]; self.max_tau = e["max_torque_Nm"]
        self.handoff = dict(C.HANDOFF); self.handoff.update(handoff or {})
        rcfg = dict(C.RAFTI); rcfg.update(rafti or {})
        r, u, rad = C.load_espa_ports()
        self.ports = ports or RaftiPorts(r, u, rad, rcfg, deputy_dims=C.ESPA["deputy_dims"],
                                         height=C.ESPA["height_m"])
        self.J = np.asarray(C.ESPA["deputy_inertia"], float)
        self.prop = get_sixdof(dynamics or e["dynamics"], e["omega"], self.dt, inertia=self.J,
                               chief_state_fn=chief_state_fn, t0=t0_mission)
        self.rng = np.random.default_rng(seed)
        self.STATE_SIZE = 12 * self.N; self.ACTION_SIZE = 6 * self.N
        self.init_cone_deg = self.handoff.get("init_cone_deg", 50.0)
        self.init_att_deg = self.handoff.get("init_att_err_deg", 90.0)
        self._tan_cone = np.tan(np.radians(rcfg["cone_half_angle_deg"]))
        self._reset_internal()

    def _reset_internal(self):
        self.state = np.zeros((self.N, 12)); self.port_of = np.arange(self.N)
        self.captured = np.zeros(self.N, bool); self.prev_dist = np.zeros(self.N)
        self.t = 0.0; self.steps = 0

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_internal()
        h = self.handoff
        self.port_of = self.rng.permutation(self.ports.P)[:self.N]
        # attitude dispersal magnitude: MRP radius for up to init_att_deg off-point
        smax = np.tan(np.radians(self.init_att_deg) / 4.0)
        for k in range(self.N):
            i = self.port_of[k]
            rr = h["envelope_min_m"] + (h["envelope_max_m"] - h["envelope_min_m"]) \
                * self.rng.random() ** (1.0 / 3.0)
            d = self._cone_dir(self.ports.dir_B[i], self.init_cone_deg)
            pos = self.ports.capture_pt[i] + rr * d
            vel = -h["init_speed_max_mps"] * self.rng.random() * self.ports.dir_B[i] \
                + 0.01 * self.rng.normal(size=3)
            sig = self.rng.normal(size=3); sig = smax * self.rng.random() * sig / (np.linalg.norm(sig) + 1e-15)
            om = 0.01 * self.rng.normal(size=3)
            self.state[k] = np.concatenate([pos, vel, sig, om])
            self.prev_dist[k] = self.ports.capture_status(pos, vel, i)["dist"]
        return self._obs(), self._info()

    def _cone_dir(self, axis, half_deg):
        ax = axis / (np.linalg.norm(axis) + 1e-15)
        ref = np.array([0, 0, 1.0]) if abs(ax @ np.array([0, 0, 1.0])) < 0.95 else np.array([0, 1.0, 0])
        e1 = np.cross(ref, ax); e1 /= np.linalg.norm(e1) + 1e-15; e2 = np.cross(ax, e1)
        ang = np.radians(half_deg) * np.sqrt(self.rng.random()); az = 2 * np.pi * self.rng.random()
        return np.cos(ang) * ax + np.sin(ang) * (np.cos(az) * e1 + np.sin(az) * e2)

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = np.clip(np.asarray(action, float).reshape(self.N, 6), -1.0, 1.0)
        act = np.zeros((self.N, 6))                       # [dv_lvlh(3), tau_body(3)]
        for k in range(self.N):
            if not self.captured[k]:
                act[k, :3] = self.ports.action_to_lvlh(a[k, :3] * self.max_dv, self.port_of[k])
                act[k, 3:] = a[k, 3:] * self.max_tau
        prev = self.state.copy()
        self.state = self.prop(self.state, act, self.dt)
        for k in range(self.N):
            if self.captured[k]:
                self.state[k] = prev[k]
        self.t += self.dt; self.steps += 1

        e = self.cfg; reward = 0.0; newly = 0; collided = False
        dists = np.zeros(self.N)
        for k in range(self.N):
            i = self.port_of[k]
            pos, vel, sig, om = self.state[k, :3], self.state[k, 3:6], self.state[k, 6:9], self.state[k, 9:12]
            st = self.ports.capture_status(pos, vel, i, sigma=sig, omega=om); dists[k] = st["dist"]
            if self.captured[k]:
                continue
            reward += 1.0 * (self.prev_dist[k] - st["dist"])                  # translational progress
            reward -= e["dist_penalty_scale"] * 0.01 * st["dist"]
            cone_excess = max(0.0, st["lateral"] - max(st["axial"], 0.0) * self._tan_cone)
            reward -= e["cone_funnel_scale"] * cone_excess                    # radial funnel
            reward -= e["point_funnel_scale"] * st["att_err"]                # attitude funnel (deg)
            if st["dist"] < 3.0:
                reward -= e["brake_scale"] * float(np.linalg.norm(vel))       # brake
                reward -= e["rate_penalty_scale"] * st["rate"]               # de-spin near port
            reward -= e["dv_penalty_scale"] * np.linalg.norm(a[k, :3] * self.max_dv)
            reward -= 20.0 * np.linalg.norm(a[k, 3:] * self.max_tau)         # torque fuel
            if self.ports.body_collision(pos, i):
                reward -= e["collision_penalty"]; collided = True
            if np.linalg.norm(pos) > e.get("bounds_radius_m", 200.0):
                reward -= e["collision_penalty"]; collided = True
            if st["captured"]:
                self.captured[k] = True; reward += e["success_bonus"]; newly += 1
            self.prev_dist[k] = st["dist"]

        hit, _ = self.ports.swarm_collision(self.state[:, :3])
        if hit:
            reward -= e["collision_penalty"]; collided = True

        done = False
        if self.captured.all():
            reward += e["success_bonus"]; done = True
        elif self.steps >= self.max_steps or collided:
            done = True
        info = self._info(); info.update(distances=dists, n_captured=int(self.captured.sum()),
                                         newly_captured=newly, collided=collided,
                                         dv_used=float(np.linalg.norm(a[:, :3] * self.max_dv)))
        return self._obs(), float(reward), done, False, info

    # ------------------------------------------------------------------ obs
    def _obs(self):
        o = np.zeros((self.N, 12))
        for k in range(self.N):
            i = self.port_of[k]
            p, v = self.ports.to_port_frame(self.state[k, :3], self.state[k, 3:6], i)
            err_l = mrp_error_to_point(self.state[k, 6:9], -self.ports.dir_B[i])  # slew-to-port (LVLH)
            err_b = mrp_to_dcm(self.state[k, 6:9]) @ err_l                        # -> body frame
            o[k, :3] = p / OBS_POS_SCALE; o[k, 3:6] = v / OBS_VEL_SCALE
            o[k, 6:9] = err_b; o[k, 9:12] = self.state[k, 9:12] / OBS_RATE_SCALE
        return o.reshape(-1)

    def _info(self):
        return dict(t=self.t, steps=self.steps, port_of=self.port_of.copy(),
                    captured=self.captured.copy())

    def positions_lvlh(self):
        return self.state[:, :3].copy()

    def attitudes(self):
        return self.state[:, 6:9].copy()


def MultiChaserRedockGym(**kw):
    import gymnasium as gym
    from gymnasium import spaces

    class _Gym(gym.Env):
        metadata = {"render_modes": []}
        def __init__(self, **kwargs):
            super().__init__(); self.core = MultiChaserRedock(**kwargs)
            self.action_space = spaces.Box(-1.0, 1.0, (self.core.ACTION_SIZE,), np.float32)
            high = np.full(self.core.STATE_SIZE, np.inf, np.float32)
            self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        def reset(self, *, seed=None, options=None):
            o, i = self.core.reset(seed=seed); return o.astype(np.float32), i
        def step(self, action):
            o, r, term, trunc, i = self.core.step(action); return o.astype(np.float32), r, term, trunc, i
    return _Gym(**kw)


if __name__ == "__main__":
    env = MultiChaserRedock(seed=0, dynamics="hcw")
    obs, info = env.reset()
    print(f"6-DOF: STATE={env.STATE_SIZE} ACTION={env.ACTION_SIZE} (12/deputy state, 6/deputy action)")
    print("ports:", info["port_of"], "unique:", len(set(info["port_of"])) == env.N)
    print("obs dim:", obs.shape, "| sample att-err MRP norms:",
          np.round(np.linalg.norm(obs.reshape(env.N, 12)[:, 6:9], axis=1), 2))
    print("multi_chaser_env (6-DOF) self-test OK")
