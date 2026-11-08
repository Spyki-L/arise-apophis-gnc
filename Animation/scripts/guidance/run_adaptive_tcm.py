#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_adaptive_tcm.py
===================

Runnable driver for adaptive_tcm.py.  Loads the fleet trajectories, builds the
relative-motion STM per deputy, and reports — per deputy and fleet-wide:

  * terminal-miss leverage L(t) at the current fixed-fraction TCM epochs;
  * the targeting miss the *recorded* TCMs removed and what they cost;
  * deterministic re-timed Δv (fixed-fraction vs leverage-optimized nodes);
  * realistic closed-loop terminal dispersion under nav + execution error.

Runs WITHOUT Basilisk: it uses the built-in linear terminal model, so the
numbers establish structure/bounds.  For flight-grade values, pass a
Basilisk-backed ``terminal_miss_fn`` to AdaptiveTCMScheduler / simulate_policy
(propagate the estimate to t_f with your force model; the hook is already there).

    python Animation/scripts/guidance/run_adaptive_tcm.py
"""
from __future__ import annotations
import os, sys, glob, traceback

print("=" * 60, flush=True)
print("ARISE adaptive-TCM driver starting...", flush=True)
print("python:", sys.version.split()[0], "| cwd:", os.getcwd(), flush=True)
print("=" * 60, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))   # .../Animation/scripts  (config.py)
sys.path.insert(0, HERE)                        # this folder (adaptive_tcm.py)

try:
    import numpy as np
    import numpy.linalg as la
    import config                               # noqa: E402
    import adaptive_tcm as atcm                  # noqa: E402
except Exception:
    print("\n*** IMPORT FAILED ***", flush=True)
    traceback.print_exc()
    print("\nCheck: this Python has numpy/scipy, and config.py + adaptive_tcm.py "
          "are in scripts/ and scripts/guidance/.", flush=True)
    raise

# ---- knobs ---------------------------------------------------------------
TOL_M          = 0.05      # close-approach design tolerance [m]
K_SIGMA        = 3.0
MAX_NODES      = 4
EARLIEST_FRAC  = 0.08      # nav gate: no TCM before this fraction of TOF
NAV_POS_SIGMA  = 0.5       # [m]    convergent EKF position knowledge (per axis)
NAV_VEL_SIGMA  = 0.003     # [m/s]  convergent EKF velocity knowledge (per axis)
N_MC           = 400
ACCURACY_CAP_M = 12.0      # penalize re-timed schedules whose p95 miss exceeds this
FIXED_FRACS    = (0.25, 0.55, 0.80, 0.93)   # current guidance schedule


def load_bundle(path):
    """Load a deputy bundle; return t, rC_N, vC_N, impulses (in-arc TCMs)."""
    z = np.load(path, allow_pickle=True)
    t  = np.asarray(z["t"], float).reshape(-1)
    rC = np.asarray(z["rC_N"], float).reshape(-1, 3)
    vC = np.asarray(z["vC_N"], float).reshape(-1, 3)
    if "impulses" in z.files:
        imp = np.asarray(z["impulses"], float).reshape(-1, 4)
    elif "dv_times" in z.files and "dv_list" in z.files:
        ts = np.asarray(z["dv_times"], float).reshape(-1, 1)
        dv = np.asarray(z["dv_list"], float).reshape(-1, 3)
        imp = np.hstack([ts, dv])
    else:
        imp = np.zeros((0, 4))
    return t, rC, vC, imp


def analyse_deputy(tag, path):
    t, rC, vC, imp = load_bundle(path)
    stm = atcm.RelativeSTM(t, rC, vC, n_grid=6000)
    t0, tf = stm.t0, stm.t_f
    TOF = tf - t0

    # in-arc TCMs (exclude the pre-window deploy burn)
    crop0 = t0 + 0.5 * 3600.0
    tcms = [r for r in imp if r[0] > crop0]
    net = np.zeros(3); recorded = 0.0
    for r in tcms:
        net += stm.Phi_rv(r[0]) @ r[1:4]
        recorded += la.norm(r[1:4])

    # equivalent initial offset that reproduces the corrected terminal miss
    y0 = la.solve(stm.Phi(t0), np.concatenate([-net, np.zeros(3)]))
    miss0 = la.norm((stm.Phi(t0) @ y0)[0:3])

    cfg = atcm.AdaptiveTCMConfig(tol_m=TOL_M, k_sigma=K_SIGMA,
                                 max_nodes=MAX_NODES, earliest_frac=EARLIEST_FRAC)
    sched = atcm.AdaptiveTCMScheduler(stm, cfg, target_rel_pos_N=np.zeros(3))
    fixed = np.array([t0 + f * TOF for f in FIXED_FRACS])

    def mc(times, pos, vel, nmc=N_MC, seed=7):
        return atcm.simulate_policy(stm, cfg, times, y0, np.zeros(3),
                                    nav_pos_sigma=pos, nav_vel_sigma=vel,
                                    n_mc=nmc, seed=seed)

    # deterministic targeting Δv (perfect nav; execution error still on)
    det_fixed = mc(fixed, 0.0, 0.0)["mean_total_dv"]

    # robust node-timing objective: realistic-MC Δv + a penalty when the
    # terminal dispersion exceeds the accuracy cap.  This stops the optimizer
    # from clustering nodes early and abandoning the late clean-up burn (which
    # is cheap deterministically but blows the dispersion up under nav noise).
    def score(ts):
        r = mc(ts, NAV_POS_SIGMA, NAV_VEL_SIGMA, nmc=120, seed=11)
        pen = 0.05 * max(0.0, r["p95_terminal_miss"] - ACCURACY_CAP_M)
        return r["mean_total_dv"] + pen

    opt = sched.optimize_node_times(score)
    det_opt = mc(opt, 0.0, 0.0)["mean_total_dv"]

    # realistic closed-loop dispersion
    real = mc(opt, NAV_POS_SIGMA, NAV_VEL_SIGMA)

    print(f"\n===== {tag}  (TOF {TOF/3600:.2f} h) =====")
    print("  leverage L(t) at fixed-fraction nodes [m per m/s]:")
    print("    " + "  ".join(f"{int(f*100)}%:{stm.leverage(t0+f*TOF):.0f}" for f in FIXED_FRACS))
    print(f"  recorded TCMs: {len(tcms)}  cost {recorded*1000:6.1f} mm/s   (removed ~{miss0:.0f} m terminal miss)")
    print(f"  deterministic re-timing:  fixed {det_fixed*1000:6.1f}  ->  optimized {det_opt*1000:6.1f} mm/s"
          f"   (save {100*(1-det_opt/det_fixed):4.0f}% vs fixed; {100*(1-det_opt/recorded):4.0f}% vs recorded)")
    print(f"  optimized nodes [%TOF]: {[f'{(x-t0)/TOF*100:.0f}' for x in opt]}")
    print(f"  realistic dispersion (nav {NAV_POS_SIGMA} m / {NAV_VEL_SIGMA*1000:.0f} mm/s):"
          f"  median miss {np.median(real['terminal_miss']):.2f} m,  p95 {real['p95_terminal_miss']:.2f} m,"
          f"  Δv {real['mean_total_dv']*1000:.1f} mm/s")
    return dict(tag=tag, recorded=recorded, det_fixed=det_fixed, det_opt=det_opt,
                real_dv=real["mean_total_dv"], real_p95=real["p95_terminal_miss"])


def main():
    files = sorted(glob.glob(os.path.join(config.TRAJ_DIR,
                                           "apophis_rendezvous_full_D*.npz")))
    if not files:
        raise FileNotFoundError(f"No deputy bundles in {config.TRAJ_DIR}")
    rows = []
    for f in files:
        tag = os.path.basename(f).replace("apophis_rendezvous_full_", "").replace(".npz", "")
        try:
            rows.append(analyse_deputy(tag, f))
        except Exception as e:
            print(f"\n[{tag}] FAILED: {e}")

    if rows:
        rec = sum(r["recorded"] for r in rows) * 1000
        opt = sum(r["det_opt"] for r in rows) * 1000
        print("\n" + "=" * 60)
        print("FLEET ROLL-UP (TCM portion only; deploy burns excluded)")
        print(f"  recorded TCM Δv (sum)      : {rec:7.1f} mm/s")
        print(f"  re-timed/deadband Δv (sum) : {opt:7.1f} mm/s   ->  {100*(1-opt/rec):.0f}% reduction")
        print("  NOTE: deploy burns (~60-65% of each deputy's total) are set by the")
        print("        transfer geometry and are NOT TCM-reducible; fleet-level deploy/")
        print("        station reallocation (global_dv_opt.py) is the lever for those.")
        print("  NOTE: numbers use the linear terminal model; swap in the Basilisk")
        print("        propagator via terminal_miss_fn for flight-grade values.")
        print("=" * 60)


if __name__ == "__main__":
    main()
