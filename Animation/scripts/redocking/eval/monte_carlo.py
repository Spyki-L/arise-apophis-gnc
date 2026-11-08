# -*- coding: utf-8 -*-
"""monte_carlo.py - Monte-Carlo success-rate evaluation in the TRAINING env.
CORRECTED: no longer imports the trainer's removed `CURRICULUM`; evaluates at the
final (flight-grade) curriculum stage via env.set_difficulty, and infers the
policy's input/output dims from the checkpoint so DDPG(12/15) and TD3 both load."""
from __future__ import annotations
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_redock as C
from eval.rollout import GreedyPolicy


def _build_env(kind, dynamics, seed, curriculum):
    if kind == "multi":
        from envs.multi_chaser_env import MultiChaserRedock
        return MultiChaserRedock(dynamics=dynamics, seed=seed)
    from envs.single_dock_env import SingleDockEnv
    env = SingleDockEnv(dynamics=dynamics, seed=seed, goal_based=(kind == "her"))
    if curriculum:
        env.set_difficulty(C.CURRICULUM_STAGES[-1])       # evaluate at the flight-grade stage
    return env


def _reset(env):
    o = env.reset(); return o[0] if isinstance(o, tuple) else o


def monte_carlo(agent=None, trials=500, dynamics="hcw", curriculum=True, seed=0, kind="her"):
    successes, per, final = 0, [], []
    policy = GreedyPolicy(agent, goal_based=(kind == "her")) if agent else None
    for k in range(trials):
        env = _build_env(kind, dynamics, seed + k, curriculum)
        o = _reset(env); done = False
        while not done:
            if policy is None:
                a = np.zeros(env.ACTION_SIZE if kind == "multi" else 6)
            else:
                a = policy.choose_action(o, add_noise=False)
            o, r, done, tr, info = env.step(a)
        if kind == "multi":
            n = info["n_captured"]; per.append(n); successes += int(n == env.N); final.append(info["distances"])
        else:
            c = int(info.get("captured", False)); successes += c; per.append(c); final.append(info.get("dist", np.nan))
    rate = 100.0 * successes / trials
    print(f"[MonteCarlo] {kind} trials={trials} success={rate:.1f}%  mean_captured={np.mean(per):.2f}"
          + (f"/{C.N_DEPUTIES}" if kind == "multi" else ""))
    return dict(success_rate=rate, trials=trials, mean_captured=float(np.mean(per)), final=np.array(final, dtype=object))


def _load_agent(ckpt, kind):
    import torch
    from agents.ddpg import DDPGAgent
    ck = torch.load(ckpt, map_location="cpu"); aw = ck["actor"]
    in_dim = int(aw["fc1.weight"].shape[1]); n_act = int(aw["mu.weight"].shape[0])
    ag = DDPGAgent(in_dim, n_act); ag.load_models(ckpt)
    print(f"loaded policy {in_dim}->{n_act} from {ckpt}")
    return ag


if __name__ == "__main__":
    CFG = dict(ckpt=None, trials=200, env="her", dynamics="hcw")
    if len(sys.argv) > 1:
        ap = argparse.ArgumentParser()
        ap.add_argument("--ckpt", default=CFG["ckpt"]); ap.add_argument("--trials", type=int, default=CFG["trials"])
        ap.add_argument("--env", default=CFG["env"], choices=["multi", "single", "her"])
        ap.add_argument("--dynamics", default=CFG["dynamics"])
        a = ap.parse_args(); CFG.update(ckpt=a.ckpt, trials=a.trials, env=a.env, dynamics=a.dynamics)
    agent = _load_agent(CFG["ckpt"], CFG["env"]) if CFG["ckpt"] and os.path.exists(CFG["ckpt"]) else None
    monte_carlo(agent, trials=CFG["trials"], dynamics=CFG["dynamics"], kind=CFG["env"])
