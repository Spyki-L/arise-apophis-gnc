# -*- coding: utf-8 -*-
"""
replay_buffer.py - experience replay (port of ReplayBuffer.m) + a Hindsight
Experience Replay buffer for the goal-based single-dock task.

The MATLAB ReplayBuffer stored (s,a,r,s',1-done); this reproduces it with a fast
pre-allocated ring buffer, and adds HERReplayBuffer which relabels the desired
goal with a FUTURE achieved goal (Andrychowicz et al., 2017) - the source RESULTS
are labelled "DDPG + HER" though the MATLAB buffer had none.
"""
from __future__ import annotations
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_dims, n_actions):
        self.mem_size = int(max_size); self.mem_cntr = 0
        self.state = np.zeros((self.mem_size, input_dims), np.float32)
        self.new_state = np.zeros((self.mem_size, input_dims), np.float32)
        self.action = np.zeros((self.mem_size, n_actions), np.float32)
        self.reward = np.zeros(self.mem_size, np.float32)
        self.terminal = np.zeros(self.mem_size, np.float32)   # stores (1 - done)

    def store_transition(self, state, action, reward, state_, done):
        i = self.mem_cntr % self.mem_size
        self.state[i] = state; self.new_state[i] = state_
        self.action[i] = action; self.reward[i] = reward
        self.terminal[i] = 1.0 - float(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        maxm = min(self.mem_cntr, self.mem_size)
        idx = np.random.randint(0, maxm, batch_size)
        return (self.state[idx], self.action[idx], self.reward[idx],
                self.new_state[idx], self.terminal[idx])

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)


class HERReplayBuffer:
    """Episode-based buffer with future-strategy hindsight relabelling.

    Stores whole episodes of dicts with keys: obs, action, reward, next_obs,
    achieved, next_achieved, desired, done.  On sample, with prob p_her a
    transition's goal is replaced by a future achieved goal from the same episode
    and its reward recomputed by env.compute_reward.
    """
    def __init__(self, max_episodes, compute_reward, obs_dim, goal_dim, n_actions,
                 k_future=4):
        self.max_ep = int(max_episodes)
        self.compute_reward = compute_reward
        self.k = k_future
        self.episodes = []

    def store_episode(self, transitions):
        self.episodes.append(transitions)
        if len(self.episodes) > self.max_ep:
            self.episodes.pop(0)

    def sample(self, batch_size):
        obs, act, rew, nobs, term = [], [], [], [], []
        for _ in range(batch_size):
            ep = self.episodes[np.random.randint(len(self.episodes))]
            t = np.random.randint(len(ep))
            tr = ep[t]
            o = np.concatenate([tr["obs"], tr["desired"]])
            no = np.concatenate([tr["next_obs"], tr["desired"]])
            r, d = tr["reward"], tr["done"]
            if np.random.random() < self.k / (self.k + 1):     # HER relabel
                fut = np.random.randint(t, len(ep))
                g = ep[fut]["next_achieved"]
                o = np.concatenate([tr["obs"], g])
                no = np.concatenate([tr["next_obs"], g])
                r = float(self.compute_reward(tr["next_achieved"], g, {}))
                d = 1.0 if r >= 0.0 else 0.0
            obs.append(o); act.append(tr["action"]); rew.append(r)
            nobs.append(no); term.append(1.0 - d)
        return (np.array(obs, np.float32), np.array(act, np.float32),
                np.array(rew, np.float32), np.array(nobs, np.float32),
                np.array(term, np.float32))

    def __len__(self):
        return sum(len(e) for e in self.episodes)
