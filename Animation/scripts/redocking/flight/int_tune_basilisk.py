# -*- coding: utf-8 -*-
"""int_tune_basilisk.py - tune the guidance INTEGRAL (ki, clip_mps) for Basilisk closed-loop.

Integral action nulls the perturbation-induced steady-state offset (dock 7% -> ~48% at
ki=3e-3); sweep ki x clip against the closed-loop Basilisk dock rate on the deployed policy
and print the best F.INTEGRAL to paste into config_flight.
"""
from __future__ import annotations
import os, sys, argparse, itertools, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flight import config_flight as F
from flight import montecarlo as MC


def evaluate(cfg, agent, n, seed):
    F.INTEGRAL = dict(enabled=True, **cfg)
    st = MC.run_campaign(n=n, fault_frac=0.15, dynamics="basilisk", seed0=seed, agent=agent, verbose=False)
    return dict(dock=st["dock_rate"], safe=st["safe_rate"], fail=st["failed"],
                prop=(100 * float(np.mean(st["props"])) if st["props"] else float("nan")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/bc_bsk/ckpt_best.pt")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    agent = None
    try:
        from flight.controller import load_flight_agent
        agent, ind, na = load_flight_agent(a.ckpt); print(f"policy {ind}->{na} ({a.ckpt})", flush=True)
    except Exception as e:
        print(f"FALLBACK PD ({e})", flush=True)
    ki_list = [3e-3, 6e-3, 1.2e-2, 2.4e-2]
    clip_list = [0.03, 0.05]
    grid = [dict(ki=ki, clip_mps=cl) for ki, cl in itertools.product(ki_list, clip_list)]
    base = dict(ki=3e-3, clip_mps=0.03)
    print(f"[int-tune] Basilisk closed-loop, n={a.n}/config, {len(grid)+1} configs", flush=True)
    results = []; t0 = time.time()
    for i, cfg in enumerate([base] + grid):
        r = evaluate(cfg, agent, a.n, a.seed)
        tag = "BASE" if i == 0 else f"c{i:02d}"
        results.append((r["dock"], r["fail"], cfg, r))
        print(f"[int-tune] {tag} dock={r['dock']:5.1f}% safe={r['safe']:5.1f}% fail={r['fail']} "
              f"prop={r['prop']:4.1f}%  ki={cfg['ki']:.1e} clip={cfg['clip_mps']:.2f}", flush=True)
    results.sort(key=lambda x: (x[1] > 0, -x[0]))
    print("\n[int-tune] TOP 5:", flush=True)
    for dock, fail, cfg, r in results[:5]:
        print(f"  dock={dock:5.1f}% fail={fail} prop={r['prop']:.1f}%  {cfg}", flush=True)
    bd, bf, bc, br = results[0]
    print(f"\n[int-tune] BEST dock={bd:.1f}% fail={bf} -> paste into config_flight.INTEGRAL:", flush=True)
    print(f"    enabled=True, ki={bc['ki']:.2e}, clip_mps={bc['clip_mps']:.2f}", flush=True)
    print(f"[int-tune] ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
