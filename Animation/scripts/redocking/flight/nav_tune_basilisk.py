# -*- coding: utf-8 -*-
"""nav_tune_basilisk.py - tune the RelativeNavEKF process noise for Basilisk closed-loop.

The accel-bias-augmented EKF docks on Basilisk truth but under-tuned (~12%). Sweep the
process-noise knobs (q_accel_bias, accel_bias0_sigma; q_vel optional) against the
closed-loop Basilisk dock rate on the DEPLOYED policy and print the best F.NAV to paste
into config_flight.NAV.  Basilisk is slow, so keep n modest; total sims = configs*n stays
well under SPICE's 5300-kernel cap.
"""
from __future__ import annotations
import os, sys, argparse, itertools, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flight import config_flight as F
from flight import montecarlo as MC


def evaluate(nav_cfg, agent, n, seed, fault_frac=0.15):
    F.NAV = dict(estimate_accel_bias=True, q_pos=1e-4, **nav_cfg)
    st = MC.run_campaign(n=n, fault_frac=fault_frac, dynamics="basilisk", seed0=seed,
                         agent=agent, verbose=False)
    navs = [c["nav_err"] for c in st["cases"] if np.isfinite(c["nav_err"])]
    return dict(dock=st["dock_rate"], safe=st["safe_rate"], fail=st["failed"],
                nav_med=(float(np.median(navs)) if navs else float("nan")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/bc_bsk/ckpt_best.pt")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    agent = None
    try:
        from flight.controller import load_flight_agent
        agent, ind, na = load_flight_agent(a.ckpt); print(f"policy {ind}->{na} ({a.ckpt})", flush=True)
    except Exception as e:
        print(f"FALLBACK PD (could not load {a.ckpt}: {e})", flush=True)

    qb_list = [3e-5, 3e-4, 3e-3]            # accel-bias random-walk (adaptation speed)
    b0_list = [3e-2, 1e-1, 3e-1]            # initial accel-bias 1-sigma (lock-on speed)
    grid = [dict(q_accel_bias=qb, accel_bias0_sigma=b0, q_vel=1e-6)
            for qb, b0 in itertools.product(qb_list, b0_list)]
    base = dict(q_accel_bias=3e-6, accel_bias0_sigma=3e-2, q_vel=1e-6)   # current deployed
    print(f"[nav-tune] Basilisk closed-loop, n={a.n}/config, {len(grid)+1} configs", flush=True)
    results = []; t0 = time.time()
    for i, cfg in enumerate([base] + grid):
        r = evaluate(cfg, agent, a.n, a.seed)
        tag = "BASE" if i == 0 else f"c{i:02d}"
        results.append((r["dock"], r["fail"], cfg, r))
        print(f"[nav-tune] {tag} dock={r['dock']:5.1f}% safe={r['safe']:5.1f}% fail={r['fail']} "
              f"nav_med={r['nav_med']:6.2f}m  qb={cfg['q_accel_bias']:.0e} b0={cfg['accel_bias0_sigma']:.0e}",
              flush=True)
    results.sort(key=lambda x: (x[1] > 0, -x[0]))       # 0-fail first, then dock rate
    print("\n[nav-tune] TOP 5:", flush=True)
    for dock, fail, cfg, r in results[:5]:
        print(f"  dock={dock:5.1f}% fail={fail} nav_med={r['nav_med']:.2f}m  {cfg}", flush=True)
    bd, bf, bc, br = results[0]
    print(f"\n[nav-tune] BEST dock={bd:.1f}% fail={bf} -> paste into config_flight.NAV:", flush=True)
    print(f"    estimate_accel_bias=True, q_pos=1.0e-4, q_vel={bc['q_vel']:.1e}, "
          f"q_accel_bias={bc['q_accel_bias']:.1e}, accel_bias0_sigma={bc['accel_bias0_sigma']:.1e}", flush=True)
    print(f"[nav-tune] ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
