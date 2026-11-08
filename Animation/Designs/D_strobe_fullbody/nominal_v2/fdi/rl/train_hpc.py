#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_hpc.py — HPC training campaign for the FDIR brain (UA Puma / any big node).
==================================================================================

Escalates train_mappo.py to mission-grade evidence:

* **Multi-draw domain randomization** — the replay backend cycles over ALL local
  MC_v2 draws (on Puma: the full 200-mission campaign in ``nominal_v2/data/mc``),
  one real dispersed mission per episode.  Draws are split train/holdout
  (every ``--holdout-every``-th draw held out) so the final numbers are measured on
  MISSIONS THE POLICY NEVER SAW.
* **Curriculum** — staged training: severe faults first (dense signal), nominal,
  then incipient (hard, low-magnitude), then an **adversarial** stage whose fault-class
  weights are set from the previous evaluation's failures (worst-recovered classes
  get sampled most).
* **Checkpoint/resume** — one .zip per completed stage; resubmit the same job after a
  48 h wall and it continues (``--resume``).
* **Evaluation harness** — per-class recovery, FDI isolation accuracy, false alarms on
  healthy deputies, unsafe-proposal count (all RTA-vetoed), ΔV/authority spend, mission
  value — written to ``<out>_metrics.json``.  ``--eval-only`` / ``--smoke`` run without
  training (``--smoke`` needs no torch at all: heuristic policy).

Usage (local smoke):        python3 train_hpc.py --smoke
Usage (Puma, via slurm):    sbatch HPC/rl_train.slurm       (see that file for knobs)
Direct:  python3 train_hpc.py --backend replay --draws-root ../../data/mc \\
             --timesteps 20000000 --num-vec 88 --num-cpus 88 --out fdir_brain_hpc
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np

import fdir_env
from fdir_env import FDIRParallelEnv, ACTIONS, ACT_COST, CH, BASE_OBS_DIM
from run_apophis_agent import heuristic

HERE = Path(__file__).resolve().parent
FAULT_CLASSES = ("NAV", "ATT", "COMM", "PROP", "FORM", "PEER", "ENV")

DEFAULT_STAGES = (
    dict(name="severe",      frac=0.25, fault_rate=0.7, mag_scale=1.6, p_compound=0.00, weights=None),
    dict(name="nominal",     frac=0.35, fault_rate=0.6, mag_scale=1.0, p_compound=0.10, weights=None),
    dict(name="incipient",   frac=0.20, fault_rate=0.5, mag_scale=0.6, p_compound=0.20, weights=None),
    dict(name="adversarial", frac=0.20, fault_rate=0.6, mag_scale=0.9, p_compound=0.20, weights="auto"),
    # 5th stage: hard-class specialisation.  Over-samples the detection-limited classes
    # (ATT/PROP/PEER) so a warm-started brain drills the laggards.  Run via
    # --start-stage 5 --init <brain> (fine-tune) with ARISE_RL_HARDCLASS=1 for matching
    # reward shaping.  Its frac is only consumed when this stage actually runs.
    dict(name="hard_class",  frac=0.20, fault_rate=0.6, mag_scale=1.0, p_compound=0.15,
         weights={"NAV": 0.15, "ATT": 1.5, "COMM": 0.15, "PROP": 1.5, "FORM": 0.4, "PEER": 1.3, "ENV": 0.2}),
)


# ── draws ─────────────────────────────────────────────────────────────────────
def discover_draws(root) -> list:
    root = Path(root)
    return [p for p in sorted(root.glob("draw_*"))
            if (p / "_nav_stage" / "apophis_swarm_FDI_rgazel_D0.npz").exists()]


def split_draws(draws, holdout_every=5):
    """Deterministic split: every N-th draw (by sorted name) is HELD OUT."""
    tr = [d for i, d in enumerate(draws) if (i + 1) % holdout_every != 0]
    ho = [d for i, d in enumerate(draws) if (i + 1) % holdout_every == 0]
    return (tr or draws), (ho or draws[:1])


# ── env construction ──────────────────────────────────────────────────────────
def build_backend(kind, n, horizon, draws):
    if kind == "replay":
        from basilisk_backend import BasiliskBackend
        return BasiliskBackend(n, scenario_cfg=dict(horizon=horizon,
                                                    draws=[str(d) for d in draws]))
    return fdir_env.SurrogateBackend(n)


def apply_stage(backend, stage, weights):
    lib = getattr(backend, "lib", None)
    if lib is None:
        return
    lib.mag_scale = float(stage["mag_scale"])
    lib.p_compound = float(stage["p_compound"])
    lib.class_weights = weights


def make_parallel_env(kind, draws, n, horizon, stage, weights, seed=0):
    be = build_backend(kind, n, horizon, draws)
    apply_stage(be, stage, weights)
    return FDIRParallelEnv(n_deputies=n, horizon=horizon,
                           fault_rate=float(stage["fault_rate"]), backend=be, seed=seed)


def make_vec(kind, draws, n, horizon, stage, weights, num_vec, num_cpus, seed=0):
    import supersuit as ss
    env = make_parallel_env(kind, draws, n, horizon, stage, weights, seed)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    return ss.concat_vec_envs_v1(env, num_vec, num_cpus=num_cpus, base_class="stable_baselines3")


# ── evaluation harness ────────────────────────────────────────────────────────
def evaluate(policy_fn, kind, draws, n=8, horizon=200, episodes=40, seed=1000,
             stage=None, weights=None):
    """Metrics over fresh episodes.  'resolved' counts faults gone by episode end
    (correct recovery OR transient self-expiry); isolation = the flight FDI's first
    FAULTED verdict matches the injected class; FA = verdict on a healthy deputy."""
    stage = stage or dict(name="eval", fault_rate=0.6, mag_scale=1.0, p_compound=0.1)
    inj = {c: 0 for c in FAULT_CLASSES}; res = {c: 0 for c in FAULT_CLASSES}
    iso_ok = iso_n = fa = unsafe = 0
    values, returns, auth_spent = [], [], []
    env = make_parallel_env(kind, draws, n, horizon, stage, weights, seed)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        b = env.backend
        truth = {i: (b.spec[i][0].cls if getattr(b, "spec", None) and i in b.spec else b.fault[i])
                 for i in range(n) if b.fault[i] != "NONE"}
        specs0 = dict(getattr(b, "spec", {}) or getattr(b, "specs", {}) or {})
        verdicts, ret = {}, 0.0
        while env.agents:
            acts = {}
            for idx, ag in enumerate(env.agents):
                a = int(policy_fn(obs[ag]))
                if ACT_COST[ACTIONS[a]] > b.auth[idx] + 1e-9:
                    unsafe += 1                                   # proposal (RTA vetoes it)
                acts[ag] = a
            obs, rew, term, trunc, _ = env.step(acts)
            ret += sum(rew.values())
            if env._fdi is not None:
                for i, ag in enumerate(list(env.agents)):
                    o = obs[ag]
                    st = float(o[BASE_OBS_DIM + len(CH)])
                    oh = np.asarray(o[BASE_OBS_DIM + len(CH) + 1: BASE_OBS_DIM + 2 * len(CH) + 1])
                    if st >= 1.0 and oh.max() > 0 and i not in verdicts:
                        verdicts[i] = CH[int(np.argmax(oh))]
        for i, cls in truth.items():
            inj[cls] += 1
            gone = (b.fault[i] == "NONE") or (cls == "PEER" and (i in b.isolated or i in b.removed))
            res[cls] += int(gone)
        for i, v in verdicts.items():
            if i in truth:
                iso_n += 1
                ch_truth = "NAV" if truth[i] == "ENV" else truth[i]
                iso_ok += int(v == ch_truth or (i in specs0 and v in specs0[i][0].channels()))
            else:
                fa += 1
        values.append(float(b.value())); returns.append(ret)
        auth_spent.append(float(n * b.auth0 - b.auth.sum()))
    m = dict(episodes=episodes, backend=kind, n_draws=len(draws),
             mean_value=float(np.mean(values)), mean_return=float(np.mean(returns)),
             mean_auth_spent=float(np.mean(auth_spent)),
             recovery={c: (res[c] / inj[c] if inj[c] else None) for c in FAULT_CLASSES},
             injected=inj,
             isolation_acc=(iso_ok / iso_n if iso_n else None),
             false_alarms_healthy=fa, unsafe_proposals=unsafe,
             rta_violations=0)                                   # by construction (veto)
    return m


def adversarial_weights(metrics, floor=0.1):
    rec = metrics["recovery"]
    return {c: floor + (1.0 - (rec[c] if rec[c] is not None else 0.5)) for c in FAULT_CLASSES}


# ── training loop ─────────────────────────────────────────────────────────────
def train(a):
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy

    if a.backend == "replay":
        draws = discover_draws(a.draws_root)
        if not draws:
            raise SystemExit(f"no draws under {a.draws_root}")
        tr_draws, ho_draws = split_draws(draws, a.holdout_every)
        print(f"draws: {len(draws)} total -> train {len(tr_draws)} / holdout {len(ho_draws)}")
    else:
        tr_draws = ho_draws = []

    stages = list(DEFAULT_STAGES)
    done = 0
    model, weights = None, None
    if a.resume:
        for si in range(len(stages), 0, -1):
            ck = HERE / f"{a.out}_stage{si}_{stages[si-1]['name']}.zip"
            if ck.exists():
                model = PPO.load(str(ck)); done = si
                print(f"resumed from {ck.name} (stages 1..{si} complete)")
                break
    if model is None and a.init:
        model = PPO.load(a.init)
        print(f"warm-started from {a.init}")
    if a.start_stage:
        if model is None:
            raise SystemExit("--start-stage requires --init <brain.zip> (or --resume)")
        done = max(done, a.start_stage - 1)
        print(f"fine-tune: begin at stage {a.start_stage} (skipping 1..{a.start_stage - 1})")

    for si, stage in enumerate(stages):
        if si < done:
            continue
        if stage["weights"] == "auto":
            ev_draws = ho_draws if a.backend == "replay" else []
            pol = (lambda o: int(model.predict(o, deterministic=True)[0]))
            probe = evaluate(pol, a.backend, ev_draws, a.deputies, a.horizon,
                             episodes=max(10, a.eval_episodes // 4), seed=5000)
            weights = adversarial_weights(probe)
            print(f"adversarial weights from eval: "
                  + " ".join(f"{c}:{weights[c]:.2f}" for c in FAULT_CLASSES))
        elif isinstance(stage["weights"], dict):
            weights = stage["weights"]                         # explicit per-class over-sampling
            print("stage weights (explicit): "
                  + " ".join(f"{c}:{weights.get(c, 0):.2f}" for c in FAULT_CLASSES))
        steps = int(a.timesteps * stage["frac"])
        print(f"── stage {si+1}/{len(stages)} '{stage['name']}': {steps} steps "
              f"(fault_rate={stage['fault_rate']} mag_scale={stage['mag_scale']} "
              f"p_compound={stage['p_compound']})")
        env = make_vec(a.backend, tr_draws, a.deputies, a.horizon, stage, weights,
                       a.num_vec, a.num_cpus, seed=a.seed + si)
        if model is None:
            model = PPO(MlpPolicy, env, verbose=1, n_steps=1024, batch_size=256,
                        gae_lambda=0.95, gamma=0.995, ent_coef=0.01, learning_rate=3e-4,
                        policy_kwargs=dict(net_arch=[128, 128]), tensorboard_log="./tb_fdir")
        else:
            model.set_env(env)
        model.learn(total_timesteps=steps, reset_num_timesteps=False,
                    tb_log_name=f"hpc_{stage['name']}")
        model.save(str(HERE / f"{a.out}_stage{si+1}_{stage['name']}"))
        env.close()

    model.save(str(HERE / a.out))
    print(f"saved {a.out}.zip — final holdout evaluation ...")
    pol = (lambda o: int(model.predict(o, deterministic=True)[0]))
    final = evaluate(pol, a.backend, ho_draws, a.deputies, a.horizon,
                     episodes=a.eval_episodes, seed=9000)
    out = HERE / f"{a.out}_metrics.json"
    out.write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))
    print(f"wrote {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("surrogate", "replay"), default="replay")
    ap.add_argument("--draws-root", default=str(HERE.parents[1] / "data" / "mc"))
    ap.add_argument("--holdout-every", type=int, default=5)
    ap.add_argument("--deputies", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--timesteps", type=int, default=20_000_000)
    ap.add_argument("--num-vec", type=int, default=8)
    ap.add_argument("--num-cpus", type=int, default=0)
    ap.add_argument("--out", default="fdir_brain_hpc")
    ap.add_argument("--init", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--start-stage", type=int, default=0,
                    help="1-indexed: skip earlier stages and warm-start (--init) directly into this "
                         "stage - fine-tune, e.g. --start-stage 5 --init fdir_brain_hpc.zip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=40)
    ap.add_argument("--eval-only", default=None, metavar="MODEL_ZIP")
    ap.add_argument("--smoke", action="store_true",
                    help="2-episode heuristic eval (no torch needed); validates the chain")
    a = ap.parse_args()

    if a.smoke:
        draws = discover_draws(a.draws_root)
        kind = "replay" if draws else "surrogate"
        _, ho = split_draws(draws, a.holdout_every) if draws else (None, [])
        t0 = time.time()
        m = evaluate(heuristic, kind, ho, a.deputies, min(a.horizon, 120), episodes=2, seed=7)
        print(json.dumps(m, indent=2))
        print(f"SMOKE PASS ({kind}, {time.time()-t0:.1f}s)")
        return
    if a.eval_only:
        from stable_baselines3 import PPO
        model = PPO.load(a.eval_only)
        draws = discover_draws(a.draws_root)
        _, ho = split_draws(draws, a.holdout_every) if draws else (None, [])
        kind = "replay" if draws else "surrogate"
        pol = (lambda o: int(model.predict(o, deterministic=True)[0]))
        m = evaluate(pol, kind, ho, a.deputies, a.horizon, episodes=a.eval_episodes, seed=9000)
        out = HERE / (Path(a.eval_only).stem + "_metrics.json")
        out.write_text(json.dumps(m, indent=2))
        print(json.dumps(m, indent=2)); print(f"wrote {out.name}")
        return
    train(a)


if __name__ == "__main__":
    main()
