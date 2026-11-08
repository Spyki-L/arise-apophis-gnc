# -*- coding: utf-8 -*-
"""
bc_pretrain.py - Behavior-clone the VERIFIED docking PD into the flight ActorNetwork.

Why: from-scratch sparse RL on the 6-DOF soft-capture keeps stalling (reward-hacking
in --env her; approach-only local optimum in --env single).  But the model-based PD
(eval.rollout.PortFramePD) docks 12/12 at EVERY curriculum stage, and the 400/300 actor
CAN represent it (BC probe: action MSE 1e-4).  So we train the actor by supervised
imitation of the PD over dispersed states across all 5 stages (DAgger-style: execute a
noised action for state coverage, but LABEL each visited state with the clean PD action).

Output: artifacts/<out>/ckpt_best.pt in the DDPGAgent checkpoint format that
flight/controller.load_flight_agent and eval/monte_carlo.py load unchanged (12->6).
This alone should dock ~like the PD; optionally RL-fine-tune it afterwards.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_redock as C
from eval.rollout import PortFramePD


def collect(n_per_stage=200, noise=0.15, seed=0, dynamics="hcw"):
    """Roll out the docking PD across all curriculum stages; return (obs, clean_PD_action).
    dynamics="basilisk" clones the PD on the hi-fi backend so the actor is trained on the
    SAME dynamics it is evaluated on (closes the sim2sim gap behind the Basilisk 0%).
    SPICE-safe: reuse ONE env across all episodes (like the trainer). Creating a fresh
    Basilisk env per episode furnishes SPICE kernels that are never released and overflows
    SPICE's 5300-file KEEPER pool (SPICE(NOMOREROOM)) once n_per_stage is scaled up."""
    from envs.single_dock_env import SingleDockEnv
    pd = PortFramePD(k_lat=0.3504, k_ax=0.0932, kd=9.923, k_slew=0.3193, kp_rate=28.12, w_max=0.02247)      # the PD that docks 12/12 every stage
    rng = np.random.default_rng(seed)
    env = SingleDockEnv(seed=int(rng.integers(1 << 30)), dynamics=dynamics)   # SPICE-safe: one reused env
    X, Y = [], []
    for stage in C.CURRICULUM_STAGES:
        env.set_difficulty(stage)
        for _ in range(n_per_stage):
            o = env.reset(); done = False
            while not done:
                a_clean = np.asarray(pd.choose_action(o), np.float32)      # supervised LABEL
                X.append(np.asarray(o, np.float32)); Y.append(a_clean)
                a_exec = np.clip(a_clean + noise * rng.standard_normal(6), -1, 1)  # coverage
                o, r, done, t, info = env.step(a_exec)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def _dock_rate(actor, torch, stage, trials, seed0=10_000, dynamics="hcw"):
    from envs.single_dock_env import SingleDockEnv
    @torch.no_grad()
    def act(o):
        return actor(torch.as_tensor(np.asarray(o, np.float32)).unsqueeze(0)).numpy()[0]
    env = SingleDockEnv(seed=seed0, dynamics=dynamics); env.set_difficulty(stage)   # SPICE-safe: one reused env
    caps = 0
    for s in range(trials):
        o = env.reset(); done = False
        while not done:
            o, r, done, t, info = env.step(act(o))
        caps += int(info["captured"])
    return caps


def train(out="artifacts/bc", n_per_stage=200, epochs=30, batch=256, lr=1e-3, seed=0, eval_trials=20, dynamics="hcw"):
    os.makedirs(out, exist_ok=True)
    import torch
    try:
        _n = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or len(os.sched_getaffinity(0))
        torch.set_num_threads(max(1, _n))
    except Exception:
        pass
    from agents.networks import ActorNetwork, CriticNetwork
    np.random.seed(seed); torch.manual_seed(seed)

    print(f"[bc] collecting PD demos on {dynamics}: {n_per_stage} eps x {len(C.CURRICULUM_STAGES)} stages ...", flush=True)
    X, Y = collect(n_per_stage=n_per_stage, seed=seed, dynamics=dynamics)
    print(f"[bc] dataset = {X.shape[0]} (obs,action) pairs", flush=True)
    Xt = torch.as_tensor(X); Yt = torch.as_tensor(Y)

    actor = ActorNetwork(12, 400, 300, 6)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    n = X.shape[0]
    for ep in range(1, epochs + 1):
        idx = np.random.permutation(n); tot = 0.0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            loss = torch.nn.functional.mse_loss(actor(Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.item()) * len(b)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[bc] epoch {ep:3d}/{epochs}  mse={tot / n:.6f}", flush=True)

    critic = CriticNetwork(12, 400, 300, 6)      # flight never uses the critic; fill fresh
    ck = dict(actor=actor.state_dict(), critic=critic.state_dict(),
              target_actor=actor.state_dict(), target_critic=critic.state_dict())
    torch.save(ck, os.path.join(out, "ckpt_best.pt"))
    torch.save(ck, os.path.join(out, "ckpt_last.pt"))
    print("[bc] saved", os.path.join(out, "ckpt_best.pt"), "(12->6, flight-loadable)", flush=True)

    print("[bc] strict 6-DOF dock rate of the BC actor (greedy):", flush=True)
    total = 0
    for stage in C.CURRICULUM_STAGES:
        caps = _dock_rate(actor, torch, stage, eval_trials, dynamics=dynamics)
        total += caps
        print(f"[bc]   stage {stage['name']:9s}  {caps}/{eval_trials}", flush=True)
    print(f"[bc] OVERALL {total}/{eval_trials * len(C.CURRICULUM_STAGES)}", flush=True)
    return os.path.join(out, "ckpt_best.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/bc")
    ap.add_argument("--n-per-stage", type=int, default=200, dest="n_per_stage")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-trials", type=int, default=20, dest="eval_trials")
    ap.add_argument("--dynamics", default="hcw", choices=["hcw", "basilisk"])
    a = ap.parse_args()
    train(out=a.out, n_per_stage=a.n_per_stage, epochs=a.epochs, batch=a.batch,
          lr=a.lr, seed=a.seed, eval_trials=a.eval_trials, dynamics=a.dynamics)
