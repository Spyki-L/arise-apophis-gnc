# -*- coding: utf-8 -*-
"""rollout.py - policy roll-out helpers + the model-based 6-DOF guidance baseline.
Port/extension of rollout_swarm.m."""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GreedyPolicy:
    """Deterministic wrapper (no exploration noise) around a DDPGAgent."""
    def __init__(self, agent, goal_based=False):
        self.agent = agent; self.goal_based = goal_based
    def choose_action(self, obs, add_noise=False):
        if self.goal_based:
            obs = np.concatenate([obs["observation"], obs["desired_goal"]])
        return self.agent.choose_action(obs, add_noise=False)


def rollout(env, policy, max_steps=None, record=True):
    o = env.reset(); states, done, k = [], False, 0
    max_steps = max_steps or env.max_steps
    while not done and k < max_steps:
        k += 1
        if record and hasattr(env, "positions_lvlh"):
            states.append(env.positions_lvlh())
        a = policy.choose_action(o, add_noise=False)
        o, r, done, trunc, info = env.step(a)
    return dict(states=np.array(states) if states else None, info=info, steps=k)


class PortFramePD:
    """6-DOF model-based guidance baseline (port frame).  Translation: glideslope
    (null lateral, run in, damp velocity).  Attitude: a rate-command law that
    slews the dock face onto the port and de-spins.  Consumes the 12-D per-deputy
    obs [pos(3), vel(3), att_err_body(3), omega(3)] -> action [dv(3), tau(3)]."""
    def __init__(self, k_lat=0.06, k_ax=0.05, kd=3.0, k_slew=0.5, kp_rate=25.0, w_max=0.015):
        self.k_lat = k_lat; self.k_ax = k_ax; self.kd = kd
        self.k_slew = k_slew; self.kp_rate = kp_rate; self.w_max = w_max
    def choose_action(self, obs, add_noise=False):
        o = np.asarray(obs).reshape(12)
        p = o[:3] * 100.0; v = o[3:6] * 0.1; err_b = o[6:9]; w = o[9:12] * 0.1
        lat = np.array([0.0, p[1], p[2]]); ax = np.array([p[0], 0.0, 0.0])
        dv = np.clip(-self.k_lat * lat - self.k_ax * ax - self.kd * v, -1, 1)
        w_des = np.clip(self.k_slew * err_b, -self.w_max, self.w_max)
        tau = np.clip(self.kp_rate * (w_des - w), -1, 1)
        return np.concatenate([dv, tau])


class SwarmFromSingle:
    """Apply ONE single-dock policy to all N deputies (canonicalisation makes each
    an identical port-frame problem).  Multi-env 96-D obs (8x12) -> 48-D action.
    goal_dim>0 appends a zero desired-goal per deputy (HER-trained 15->6 policies)."""
    def __init__(self, agent, n=8, goal_dim=0):
        self.agent = agent; self.n = n; self.goal_dim = goal_dim
    def choose_action(self, obs, add_noise=False):
        o = np.asarray(obs).reshape(self.n, 12); acts = []
        for i in range(self.n):
            v = o[i] if not self.goal_dim else np.concatenate([o[i], np.zeros(self.goal_dim)])
            acts.append(self.agent.choose_action(v, add_noise=False))
        return np.concatenate(acts)
