# -*- coding: utf-8 -*-
"""
train_ddpg_torch.py - CORRECTED faithful trainer.  Now TD3 (not vanilla DDPG,
which diverged: best avg reward -197 -> -730) with a REAL annealed curriculum and
a CONSISTENT HER buffer.

Modes
-----
  --env single : 12->6 dense-reward TD3 (one policy, all 8 deputies).  Flight-native.
  --env her    : 15->6 TD3 + HER (goal = port-frame capture position).  The env
                 reward and the HER-relabelled reward go through the SAME
                 SingleDockEnv._reward_core, so real and virtual transitions are
                 consistent (the old code mixed dense ~-800/ep and sparse 0/-1).
  --env multi  : 96->48 swarm TD3 (Phase-A fleet guidance; NOT the terminal dock).

Key fixes vs the 0%-dock run:
  * curriculum ANNEALS stage-by-stage (env.set_difficulty) advancing on rolling
    success - and sizes the episode horizon so capture is actually reachable;
  * TD3 twin critics/target smoothing/delayed actor kill the DDPG divergence;
  * the Stage-3 crash is fixed: MultiChaserRedock.reset() returns (obs, info); we
    now unpack it (the old loop shoved the tuple into the replay buffer).
"""
from __future__ import annotations
import os, sys, time, argparse, json
from collections import deque
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_redock as C
from agents.td3 import TD3Agent


def _log(fp, msg):
    print(msg, flush=True); fp.write(msg + "\n"); fp.flush()


def build_env(kind, dynamics, seed, curriculum, stage_idx=0):
    if kind == "multi":
        from envs.multi_chaser_env import MultiChaserRedock
        env = MultiChaserRedock(dynamics=dynamics, seed=seed)
    else:
        from envs.single_dock_env import SingleDockEnv
        env = SingleDockEnv(dynamics=dynamics, seed=seed, goal_based=(kind == "her"))
    if curriculum and kind != "multi":
        env.set_difficulty(C.CURRICULUM_STAGES[stage_idx])
    return env


def _reset(env):
    o = env.reset()
    return o[0] if isinstance(o, tuple) else o          # multi env returns (obs, info)


def _add_her_episode(agent, env, ep, k_future, bonus):
    """Add real + future-relabelled transitions to ONE buffer with a CONSISTENT
    reward (env._reward_core).  ep: list of dicts (obs,next_obs,action,achieved,
    next_achieved,desired,her,done)."""
    n = len(ep)
    for t in range(n):
        tr = ep[t]
        # real transition (desired goal = 0)
        r_real = env._reward_core(tr["next_achieved"], tr["desired"], tr["her"])
        agent.remember(np.concatenate([tr["obs"], tr["desired"]]), tr["action"], r_real,
                       np.concatenate([tr["next_obs"], tr["desired"]]), tr["done"])
        # k hindsight goals from the future of THIS episode
        for _ in range(k_future):
            f = np.random.randint(t, n)
            g = ep[f]["next_achieved"]
            r = env._reward_core(tr["next_achieved"], g, tr["her"])
            done_r = bool(r >= 0.5 * bonus)         # relabelled goal effectively reached
            agent.remember(np.concatenate([tr["obs"], g]), tr["action"], r,
                           np.concatenate([tr["next_obs"], g]), done_r)


def pd_warmstart(agent, env, kind, n_episodes, bonus, her_k):
    """Seed the replay buffer with verified-PD demonstrations (which DOCK) so TD3
    has success transitions from step 1.  Off-policy RL bootstraps far faster from
    an expert prior - and the terminal soft-dock is precision-sensitive (a behaviour
    -cloning probe of the PD docks even with tiny action error), so a good prior
    matters.  Uses eval.rollout.PortFramePD on the CURRENT curriculum stage."""
    from eval.rollout import PortFramePD
    pd = PortFramePD(k_lat=0.3504, k_ax=0.0932, kd=9.923, k_slew=0.3193, kp_rate=28.12, w_max=0.02247)
    docked = 0
    for _ in range(n_episodes):
        o = _reset(env); done = False; ep_tr = []
        while not done:
            obs_vec = o["observation"] if kind == "her" else o
            a = pd.choose_action(obs_vec)
            o2, r, done, tr, info = env.step(a)
            if kind == "her":
                ep_tr.append(dict(obs=o["observation"], next_obs=o2["observation"], action=a,
                                  achieved=o["achieved_goal"], next_achieved=o2["achieved_goal"],
                                  desired=o["desired_goal"], her=info["her"], done=bool(done)))
            else:
                agent.remember(o, a, r, o2, done)
            o = o2
        if kind == "her":
            _add_her_episode(agent, env, ep_tr, her_k, bonus)
        docked += int(info.get("captured", False))
    return docked


def train(kind="her", episodes=2000, dynamics="hcw", curriculum=True,
          out="artifacts/run", seed=0, resume=False, warmstart_pd=0):
    os.makedirs(out, exist_ok=True)
    logfp = open(os.path.join(out, "train.log"), "a" if resume else "w")
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)
    try:
        # use the ALLOCATED cpus (SLURM), not all node cores - os.cpu_count() sees every core on
        # an HPC node (e.g. 94), so torch spawns that many threads and glibc per-thread malloc
        # arenas blow RSS past the job's --mem -> OOM kill.  (Pair with MALLOC_ARENA_MAX in slurm.)
        _ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or len(os.sched_getaffinity(0))
        torch.set_num_threads(max(1, _ncpu))
    except Exception:
        try: torch.set_num_threads(max(1, os.cpu_count() or 1))
        except Exception: pass

    stage_idx = 0
    env = build_env(kind, dynamics, seed, curriculum, stage_idx)
    if kind == "multi":
        in_dim, act_dim = env.STATE_SIZE, env.ACTION_SIZE
    elif kind == "single":
        in_dim, act_dim = env.obs_dim, env.act_dim
    else:  # her
        in_dim, act_dim = env.obs_dim + 3, env.act_dim

    d = C.TD3
    agent = TD3Agent(in_dim, act_dim, alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"],
                     tau=d["tau"], layer1_size=d["layer1_size"], layer2_size=d["layer2_size"],
                     batch_size=d["batch_size"], buffer_size=d["buffer_size"],
                     policy_delay=d["policy_delay"], target_noise=d["target_noise"],
                     noise_clip=d["noise_clip"], expl_noise=d["expl_noise"])
    cur = C.CURRICULUM
    bonus = C.REWARD["success_bonus"]

    hist, succ_hist, best = [], [], -1e9
    best_key = (-1, -1.0, -1e9)   # (stage_idx, rolling success, avg reward): checkpoint the best-DOCKING snapshot, not the highest-reward one
    if resume and os.path.exists(os.path.join(out, 'ckpt_last.pt')):
        agent.load_models(os.path.join(out, 'ckpt_last.pt'))
        rp = os.path.join(out, 'reward_history.npz')
        if os.path.exists(rp):
            d0 = np.load(rp); hist = list(d0['reward']); succ_hist = list(d0['success'])
            stage_idx = int(d0['stage_idx']) if 'stage_idx' in d0.files else 0
            if curriculum and kind != "multi":
                env.set_difficulty(C.CURRICULUM_STAGES[stage_idx])

    _log(logfp, f"[start] TD3 kind={kind} in={in_dim} act={act_dim} dyn={dynamics} "
                f"curriculum={curriculum} stage={C.CURRICULUM_STAGES[stage_idx]['name'] if kind!='multi' else 'n/a'} "
                f"episodes={episodes} device={agent.device}")

    import signal
    stop = {"flag": False}
    def _on_signal(signum, frame):
        stop["flag"] = True; _log(logfp, f"[signal {signum}] will checkpoint at end of episode")
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try: signal.signal(_sig, _on_signal)
        except Exception: pass

    def _save():
        agent.save_models(os.path.join(out, "ckpt_last.pt"))
        np.savez(os.path.join(out, "reward_history.npz"),
                 reward=np.array(hist), success=np.array(succ_hist), stage_idx=stage_idx)

    if warmstart_pd and kind != "multi" and not (resume and len(hist)):
        nd = pd_warmstart(agent, env, kind, warmstart_pd, bonus, d["her_k"])
        _log(logfp, f"[warmstart] seeded {len(agent.memory)} PD transitions from {warmstart_pd} eps ({nd} docked)")
    warmup = d["warmup_steps"]; total_steps = 0
    roll = deque(maxlen=cur["window"]); eps_on_stage = 0
    t0 = time.time()

    for ep in range(1, episodes + 1):
        # anneal exploration noise from expl_noise -> expl_noise_final
        frac = min(1.0, ep / max(1, episodes))
        agent.set_expl_noise(d["expl_noise"] + frac * (d["expl_noise_final"] - d["expl_noise"]))
        o = _reset(env); done = False; ep_r = 0.0; ep_tr = []
        while not done:
            if kind == "her":
                obs_vec = np.concatenate([o["observation"], o["desired_goal"]])
            else:
                obs_vec = o
            if total_steps < warmup:
                a = np.random.uniform(-1, 1, act_dim)
            else:
                a = agent.choose_action(obs_vec, add_noise=True)
            o2, r, done, trunc, info = env.step(a)
            total_steps += 1
            if kind == "her":
                ep_tr.append(dict(obs=o["observation"], next_obs=o2["observation"], action=a,
                                  achieved=o["achieved_goal"], next_achieved=o2["achieved_goal"],
                                  desired=o["desired_goal"], her=info["her"], done=bool(done)))
            else:
                agent.remember(obs_vec, a, r, o2, done)
                if total_steps >= warmup:
                    for _ in range(d["updates_per_step"]):
                        agent.learn()
            o = o2; ep_r += r
        # end of episode
        if kind == "her":
            _add_her_episode(agent, env, ep_tr, d["her_k"], bonus)
            n_upd = min(len(ep_tr) * d["updates_per_step"], 400)
            if len(agent.memory) > agent.batch_size:
                for _ in range(n_upd):
                    agent.learn()

        if kind == "multi":
            captured = info["n_captured"]; succ_hist.append(captured / max(1, env.N))
        else:
            captured = int(info.get("captured", False)); succ_hist.append(captured)
        hist.append(ep_r); roll.append(0 if kind == "multi" else captured); eps_on_stage += 1

        # ---- curriculum advancement on rolling success ----
        if curriculum and kind != "multi" and stage_idx < len(C.CURRICULUM_STAGES) - 1:
            if eps_on_stage >= cur["min_eps"] and len(roll) == roll.maxlen and np.mean(roll) >= cur["advance_at"]:
                stage_idx += 1; env.set_difficulty(C.CURRICULUM_STAGES[stage_idx])
                _log(logfp, f"[curriculum] -> stage {stage_idx} '{C.CURRICULUM_STAGES[stage_idx]['name']}' "
                            f"at ep {ep} (rolling succ {np.mean(roll)*100:.0f}%)")
                roll.clear(); eps_on_stage = 0

        if ep % 10 == 0:
            avg = np.mean(hist[-50:]); sr = np.mean(succ_hist[-50:]) * (100.0 if kind != "multi" else 8.0)
            rate = ep / (time.time() - t0 + 1e-9)
            stg = C.CURRICULUM_STAGES[stage_idx]['name'] if kind != 'multi' else '-'
            _log(logfp, f"ep {ep:6d}/{episodes} stg={stg:9s} R={ep_r:8.1f} avg50={avg:8.1f} "
                        f"succ50={sr:5.1f}{'%' if kind!='multi' else '/8'} steps={total_steps} {rate:4.1f}ep/s")
        if ep > 50:
            sr50 = float(np.mean(succ_hist[-50:])); avg50 = float(np.mean(hist[-50:]))
            key = (stage_idx, sr50, avg50)           # prefer later curriculum stage, then dock rate, then reward
            if key > best_key:                       # dock-rate checkpointing (was: highest avg reward -> kept a non-docking policy)
                best_key = key; best = avg50
                agent.save_models(os.path.join(out, "ckpt_best.pt"))
        if ep % 50 == 0:
            _save()
        if stop["flag"]:
            _save(); _log(logfp, f"[stopped] ep={ep}: checkpointed. Resubmit with --resume."); break

    agent.save_models(os.path.join(out, "ckpt_last.pt"))
    if not os.path.exists(os.path.join(out, "ckpt_best.pt")):
        agent.save_models(os.path.join(out, "ckpt_best.pt"))
    _save()
    json.dump(dict(kind=kind, episodes=episodes, dynamics=dynamics, curriculum=curriculum,
                   final_stage=C.CURRICULUM_STAGES[stage_idx]['name'] if kind != 'multi' else 'n/a',
                   best_avg=float(best), wall_s=time.time() - t0),
              open(os.path.join(out, "summary.json"), "w"), indent=2)
    _log(logfp, f"[done] best_avg50={best:.1f} final_stage="
                f"{C.CURRICULUM_STAGES[stage_idx]['name'] if kind!='multi' else 'n/a'} wall={time.time()-t0:.0f}s")
    logfp.close(); return agent


if __name__ == "__main__":
    CFG = dict(env="her", episodes=400, dynamics="hcw", curriculum=True,
               out="artifacts/local_train", seed=0, resume=False, warmstart_pd=0)
    if len(sys.argv) > 1:
        ap = argparse.ArgumentParser()
        ap.add_argument("--env", default=CFG["env"], choices=["multi", "single", "her"])
        ap.add_argument("--episodes", type=int, default=CFG["episodes"])
        ap.add_argument("--dynamics", default=CFG["dynamics"], choices=["hcw", "basilisk"])
        ap.add_argument("--curriculum", action="store_true", default=CFG["curriculum"])
        ap.add_argument("--out", default=CFG["out"]); ap.add_argument("--seed", type=int, default=CFG["seed"])
        ap.add_argument("--resume", action="store_true", default=CFG["resume"])
        ap.add_argument("--warmstart-pd", type=int, default=0, dest="warmstart_pd")
        a = ap.parse_args()
        CFG.update(env=a.env, episodes=a.episodes, dynamics=a.dynamics, curriculum=a.curriculum,
                   out=a.out, seed=a.seed, resume=a.resume, warmstart_pd=a.warmstart_pd)
    train(kind=CFG["env"], episodes=CFG["episodes"], dynamics=CFG["dynamics"],
          curriculum=CFG["curriculum"], out=CFG["out"], seed=CFG["seed"], resume=CFG["resume"],
          warmstart_pd=CFG.get("warmstart_pd", 0))
