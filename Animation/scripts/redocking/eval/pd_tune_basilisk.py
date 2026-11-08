# -*- coding: utf-8 -*-
"""pd_tune_basilisk.py - Retune the PortFramePD gains for BASILISK dynamics.

The PortFramePD gains were tuned for the HCW surrogate: on Basilisk they dock ~100%
at easy stages but only ~20% at s4_flight, which CAPS the behavior-cloned actor (the
clone cannot beat its teacher). This does a random search over all SIX gains
(k_lat,k_ax,kd,k_slew,kp_rate,w_max), scores each candidate by the strict 6-DOF dock
rate on Basilisk at the hard stage(s), and prints the best set to paste back into
bc_pretrain.PortFramePD (then re-run bc_bsk.slurm to re-clone on the better teacher).

SPICE-safe: one env per (candidate, stage), created with the SAME seed, so every
candidate faces the identical initial-condition set (fair comparison) and total SPICE
kernel loads stay tiny (= candidates*stages, not candidates*stages*trials).
"""
from __future__ import annotations
import os, sys, argparse, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_redock as C
from eval.rollout import PortFramePD

# the currently-deployed teacher (what bc_pretrain instantiates today)
BASE = dict(k_lat=0.20, k_ax=0.15, kd=5.0, k_slew=0.5, kp_rate=25.0, w_max=0.015)
# multiplicative (log-uniform) search range around BASE, per gain
RANGE = dict(k_lat=(0.4, 2.6), k_ax=(0.4, 2.6), kd=(0.4, 2.6),
             k_slew=(0.5, 2.2), kp_rate=(0.5, 2.0), w_max=(0.6, 1.8))


def dock_counts(gains, stages, trials, dynamics, seed):
    """Strict dock counts of PortFramePD(**gains) over `trials` eps at each stage.
    Fresh env seeded identically per call -> all candidates see the same scenarios."""
    from envs.single_dock_env import SingleDockEnv
    pd = PortFramePD(**gains)
    out = {}
    for st in stages:
        env = SingleDockEnv(seed=seed, dynamics=dynamics); env.set_difficulty(st)  # SPICE-safe: reused across trials
        caps = 0
        for _ in range(trials):
            o = env.reset(); done = False
            while not done:
                o, r, done, t, info = env.step(np.asarray(pd.choose_action(o), np.float32))
            caps += int(info.get("captured", False))
        out[st["name"]] = caps
    return out


def sample(rng):
    return {k: float(BASE[k] * np.exp(rng.uniform(np.log(lo), np.log(hi))))
            for k, (lo, hi) in RANGE.items()}


def fmt(g):
    return ("k_lat=%.4f k_ax=%.4f kd=%.3f k_slew=%.4f kp_rate=%.2f w_max=%.5f"
            % (g["k_lat"], g["k_ax"], g["kd"], g["k_slew"], g["kp_rate"], g["w_max"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics", default="basilisk", choices=["hcw", "basilisk"])
    ap.add_argument("--n-candidates", type=int, default=60, dest="ncand")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--stages", default="s3,s4_flight", help="comma list of stage names to score")
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--out", default="artifacts/pd_tune")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    by_name = {s["name"]: s for s in C.CURRICULUM_STAGES}
    stages = [by_name[n] for n in a.stages.split(",")]
    w = {s["name"]: (i + 1) for i, s in enumerate(stages)}          # weight harder (later) stages more

    def score(res):
        num = sum(w[n] * res[n] for n in res); den = sum(w[n] * a.trials for n in res)
        return num / den

    rng = np.random.default_rng(a.seed)
    print(f"[pd-tune] {a.dynamics}: {a.ncand} candidates x {a.trials} trials on {list(w)} (weights {w})", flush=True)
    t0 = time.time(); results = []
    for i in range(a.ncand + 1):
        g = dict(BASE) if i == 0 else sample(rng)
        res = dock_counts(g, stages, a.trials, a.dynamics, a.seed)
        sc = score(res)
        results.append((sc, g, res))
        tag = "BASELINE" if i == 0 else f"cand{i:3d}"
        rates = "  ".join(f"{n} {res[n]:2d}/{a.trials}" for n in res)
        print(f"[pd-tune] {tag}  score={sc:.3f}  {rates}   [{fmt(g)}]", flush=True)
    base_sc = results[0][0]
    results.sort(key=lambda x: -x[0])
    print("\n[pd-tune] ===== TOP 5 (baseline score = %.3f) =====" % base_sc, flush=True)
    for sc, g, res in results[:5]:
        rates = "  ".join(f"{n} {res[n]}/{a.trials}" for n in res)
        print(f"  score={sc:.3f}  {rates}   [{fmt(g)}]", flush=True)

    best_sc, best_g, _ = results[0]
    # full 5-stage profile for BASELINE vs BEST so a hard-stage win doesn't secretly wreck easy stages
    allst = C.CURRICULUM_STAGES
    print("\n[pd-tune] full 5-stage profile (trials=%d):" % a.trials, flush=True)
    prof_base = dock_counts(BASE, allst, a.trials, a.dynamics, a.seed)
    prof_best = dock_counts(best_g, allst, a.trials, a.dynamics, a.seed)
    print("  stage        BASELINE   BEST", flush=True)
    for st in allst:
        n = st["name"]
        print(f"  {n:11s}  {prof_base[n]:2d}/{a.trials}     {prof_best[n]:2d}/{a.trials}", flush=True)

    improved = best_sc > base_sc + 1e-9
    print(f"\n[pd-tune] {'IMPROVED' if improved else 'NO IMPROVEMENT'}: best score {best_sc:.3f} vs baseline {base_sc:.3f}", flush=True)
    if improved:
        print("[pd-tune] paste into bc_pretrain.collect (and the bc_bsk sanity / pd_warmstart) then re-run bc_bsk.slurm:", flush=True)
        print("    PortFramePD(k_lat=%.4f, k_ax=%.4f, kd=%.3f, k_slew=%.4f, kp_rate=%.2f, w_max=%.5f)"
              % (best_g["k_lat"], best_g["k_ax"], best_g["kd"], best_g["k_slew"], best_g["kp_rate"], best_g["w_max"]), flush=True)
    else:
        print("[pd-tune] the current gains are already ~best in this range; widen RANGE or try RL fine-tune (ft_bsk2).", flush=True)
    json.dump(dict(base=BASE, base_score=base_sc, best=best_g, best_score=best_sc,
                   profile_base=prof_base, profile_best=prof_best),
              open(os.path.join(a.out, "best_gains.json"), "w"), indent=2)
    print(f"[pd-tune] wrote {os.path.join(a.out,'best_gains.json')}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
