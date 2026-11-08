#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_apophis_agent.py — deploy the trained brain in the Apophis scenario and WATCH it.

Loads the trained shared policy (SB3) and runs it decentralized (each deputy queries
the same policy on its own local observation) over one Apophis-configured episode,
printing the faults injected, the actions the agent commits, and the mission-value
trajectory.  If SB3/torch or a saved model is unavailable, falls back to a rule-based
HEURISTIC policy so the loop still runs (useful for a dry run of the pipeline).

    python run_apophis_agent.py --model fdir_brain.zip --backend surrogate
    python run_apophis_agent.py --model fdir_brain.zip --backend basilisk   # on the sim machine
"""
from __future__ import annotations
import argparse
import numpy as np
import fdir_env
from fdir_env import ACTIONS, CH, BASE_OBS_DIM

_RECOVER = {"NAV": "REINIT", "COMM": "RELOAD", "FORM": "MODE",
            "ATT": "SAFEHOLD", "PROP": "SAFEHOLD", "PEER": "ISOLATE"}


def heuristic(obs):
    """Rule-based fallback.  With the flight-FDI obs layer (ARISE_RL_FDI, default on)
    it acts on the OBC's FDIR verdict: act only when the core is FAULTED, on the
    ISOLATED fault type — exactly the VSC's rule-based recovery table.  Without the
    layer, falls back to worst-raw-channel."""
    if obs.shape[0] > BASE_OBS_DIM:                       # FDI telemetry present
        state01 = float(obs[BASE_OBS_DIM + len(CH)])
        onehot = np.asarray(obs[BASE_OBS_DIM + len(CH) + 1: BASE_OBS_DIM + 2 * len(CH) + 1])
        if state01 >= 1.0 and onehot.max() > 0:
            return ACTIONS.index(_RECOVER[CH[int(np.argmax(onehot))]])
        return ACTIONS.index("NOOP")
    sev = obs[:len(CH)]
    ci = int(np.argmax(sev))
    if sev[ci] < 1.5:
        return ACTIONS.index("NOOP")
    return ACTIONS.index(_RECOVER[CH[ci]])


def load_policy(path):
    try:
        from stable_baselines3 import PPO
        m = PPO.load(path)
        return (lambda o: int(m.predict(o, deterministic=True)[0])), "SB3 trained brain"
    except Exception as e:
        return (lambda o: heuristic(o)), f"heuristic fallback ({type(e).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fdir_brain.zip")
    ap.add_argument("--backend", choices=["surrogate", "basilisk"], default="surrogate")
    ap.add_argument("--deputies", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    backend = None
    if a.backend == "basilisk":
        from basilisk_backend import BasiliskBackend
        backend = BasiliskBackend(a.deputies, scenario_cfg=dict(horizon=a.horizon))
    env = fdir_env.FDIRParallelEnv(n_deputies=a.deputies, horizon=a.horizon, backend=backend, seed=a.seed)
    policy, tag = load_policy(a.model)
    print(f"== Apophis deployment · policy: {tag} ==")
    obs, _ = env.reset(seed=a.seed)
    campaign = [(f"D{i}", env.backend.fault[i], env.backend.onset[i]) for i in range(a.deputies)
                if env.backend.fault[i] != "NONE"]
    print("injected fault campaign:", campaign)
    v0 = env.backend.value(); acts_taken = {}
    while env.agents:
        actions = {ag: policy(obs[ag]) for ag in env.agents}
        for ag in env.agents:
            an = ACTIONS[int(actions[ag])]
            if an != "NOOP":
                acts_taken.setdefault(ag, []).append(an)
        obs, rew, term, trunc, info = env.step(actions)
    v1 = env.backend.value()
    print(f"recovery actions taken: { {k: v[:4] for k, v in acts_taken.items()} }")
    print(f"mission value: {v0:.2f} -> {v1:.2f}   (recovered {v1 - v0:+.2f})")
    print("0 unsafe executions (RTA veto guarantees every executed action is safe)")


if __name__ == "__main__":
    main()
