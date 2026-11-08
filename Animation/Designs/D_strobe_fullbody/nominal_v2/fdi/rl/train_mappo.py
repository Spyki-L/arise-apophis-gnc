#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mappo.py — train the multi-agent FDIR brain (SB3 + PettingZoo + SuperSuit).

Parameter-sharing PPO over the PettingZoo ParallelEnv: SuperSuit flattens the
multi-agent env into a vectorized single-policy env where ONE shared policy controls
all deputies (decentralized execution, shared weights).  This is the standard SB3
multi-agent recipe and is effectively IPPO-with-parameter-sharing.

CTDE / true MAPPO note: SB3's PPO uses a DECENTRALIZED critic (the value net sees local
obs).  For a true centralized critic (MAPPO) feed env.state() to the value network:
either subclass ActorCriticPolicy so the critic input is the global state (hook
`CentralizedCriticPolicy` sketched at the bottom), or swap in a dedicated MAPPO
implementation.  Parameter-sharing PPO below is the working baseline and already gives
coordinated recovery via the shared policy + team reward.

Run (after `pip install "stable-baselines3[extra]" torch supersuit pettingzoo gymnasium`):
    python train_mappo.py --backend surrogate --timesteps 2000000
    python train_mappo.py --backend basilisk  --timesteps 200000   # hi-fi fine-tune
"""
from __future__ import annotations
import argparse
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

import fdir_env


def build_backend(kind, n, horizon):
    if kind == "basilisk":
        from basilisk_backend import BasiliskBackend
        return BasiliskBackend(n, scenario_cfg=dict(horizon=horizon))   # episode window = horizon
    return fdir_env.SurrogateBackend(n)


def make_vec(kind, n, horizon, num_vec, num_cpus):
    env = fdir_env.FDIRParallelEnv(n_deputies=n, horizon=horizon, backend=build_backend(kind, n, horizon))
    env = ss.pettingzoo_env_to_vec_env_v1(env)                       # parallel -> vector (param sharing)
    env = ss.concat_vec_envs_v1(env, num_vec, num_cpus=num_cpus, base_class="stable_baselines3")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["surrogate", "basilisk"], default="surrogate")
    ap.add_argument("--deputies", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--num-vec", type=int, default=8)     # keep SMALL (1-2) for basilisk
    ap.add_argument("--num-cpus", type=int, default=4)
    ap.add_argument("--out", default="fdir_brain")
    ap.add_argument("--init", default=None,
                    help="warm-start from a saved brain (.zip) — the surrogate->hi-fi fine-tune path")
    a = ap.parse_args()

    if a.backend == "basilisk":                            # hi-fi: few envs, IN-PROCESS.
        # num_cpus=0 is deliberate: supersuit's process-vec (spawn) can deadlock on
        # Windows, and the replay backend steps fast enough in-process anyway.
        a.num_vec = min(a.num_vec, 2); a.num_cpus = 0
    print(f"[train] building {a.num_vec}x vec env ({a.backend}, num_cpus={a.num_cpus}) ...", flush=True)
    env = make_vec(a.backend, a.deputies, a.horizon, a.num_vec, a.num_cpus)
    print(f"[train] env ready ({env.num_envs} agent-slots). NOTE: with the basilisk/replay "
          f"backend the FIRST PPO iteration = n_steps x slots timesteps of hi-fi replay "
          f"(~1-3 min SILENT before the first log block). Not hung — watch tensorboard.", flush=True)

    if a.init:
        model = PPO.load(a.init, env=env, tensorboard_log="./tb_fdir")
        print(f"[train] warm-started from {a.init} (fine-tune, continues timestep count)", flush=True)
    else:
        model = PPO(MlpPolicy, env, verbose=1, n_steps=1024, batch_size=256, gae_lambda=0.95,
                    gamma=0.995, ent_coef=0.01, learning_rate=3e-4,
                    policy_kwargs=dict(net_arch=[128, 128]), tensorboard_log="./tb_fdir")
    model.learn(total_timesteps=a.timesteps, reset_num_timesteps=not a.init,
                tb_log_name=f"ppo_{a.backend}")
    model.save(a.out)
    print(f"saved trained FDIR brain -> {a.out}.zip")


# ---- CTDE upgrade hook (true MAPPO): centralized critic over env.state() -------
# from stable_baselines3.common.policies import ActorCriticPolicy
# class CentralizedCriticPolicy(ActorCriticPolicy):
#     """Feed the global state() to the value network while the actor keeps local obs.
#     Requires storing state() in the rollout buffer and overriding forward()/
#     predict_values(); see CORTEX_RL_TRAINING_FRAMEWORK.md, section 'CTDE / MAPPO'."""
#     ...

if __name__ == "__main__":
    main()
