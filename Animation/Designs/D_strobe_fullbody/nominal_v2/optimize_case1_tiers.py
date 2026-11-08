#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_case1_tiers.py - ML-optimize the 8 deputy stations for ALL THREE coverage tiers.

Objective (robust over resampled spin poses, maximise):
    J = 1.0*cov@0.15m  +  0.5*cov@<=1m  +  0.25*shape-closed   - separation_penalty
subject to body clearance (rho in [700,850] m by construction) and inter-deputy sep >= 150 m.

Optimizer: CMA-ES (cma) in a tanh-bounded 24-D space - the right tool for a 24-dimensional,
non-differentiable, NOISY black box.  (A deep-learning surrogate was considered; it would need
thousands of evaluations just to train a network that is *less* accurate than evaluating the true
coverage directly, so CMA-ES is both faster and more accurate here.)

  python optimize_case1_tiers.py            # evolve (checkpointed; re-run to continue)
  python optimize_case1_tiers.py --final     # validate best on full mesh, write targets npz
"""
import os, sys, math, time, pickle, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import optimize_refly_stations as OPT
import optimize_refly_v2 as V2          # genes_from_x (tanh-bounded mapping)
import simulate_case1_nac as SIM

KPOSE = 6; TBUDGET = 35.0; NAC_KM = 3.0; ESH = 50
PKL = os.path.join(HERE, "case1_tiers_cma.pkl")
GEOM = SIM.load_geometry()
# subsampled mesh for fast fitness (occ off -> only centroids/normals/areas used)
_m = np.load(os.path.join(HERE, "data", "radar_mesh.npz"))
_idx = np.arange(0, len(_m["centroids"]), 4)
MESH_SUB = (_m["centroids"][_idx], _m["normals"][_idx], _m["areas"][_idx], _m["vertices"], _m["faces"])
ATOT = MESH_SUB[2].sum()

def tiers(rs, R):
    cat, lit, _ = SIM.score_pass(rs, R, MESH_SUB, GEOM, False, ESH, NAC_KM)
    a = MESH_SUB[2]
    return (100*a[cat == 0].sum()/ATOT, 100*a[cat <= 1].sum()/ATOT, 100*a[cat <= 2].sum()/ATOT)

def fitness(x, rng):
    g = V2.genes_from_x(x); rs = OPT.stations(g); sp = OPT.sep_min(rs); pen = max(0., 150.-sp)*0.3
    t0 = t1 = t2 = 0.0
    for _ in range(KPOSE):
        p = OPT.TUM.sample_tumbler(rng); _, dc = OPT.TUM.integrate_window(p, np.array([0., 1.])); R = dc[0]
        a, b, c = tiers(rs, R); t0 += a; t1 += b; t2 += c
    t0 /= KPOSE; t1 /= KPOSE; t2 /= KPOSE
    return (2.0*t0 + 0.6*t1 + 0.2*t2 - pen), (t0, t1, t2, sp)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--final", action="store_true"); a = ap.parse_args()
    import cma
    if os.path.exists(PKL):
        es, best = pickle.load(open(PKL, "rb"))
    else:
        # seed from the already-CMA-optimized stations (so we start at ~74% and improve)
        tg = np.load(os.path.join(HERE, "refly_targets_cma.npz"))["DEPUTY_TARGETS_M"]; rs = (OPT.C_A.T @ tg.T).T
        rs = rs * (800.0 / np.median([np.linalg.norm(r) for r in rs]))
        x0 = np.zeros((8, 3))
        for i, r in enumerate(rs):
            tau = float(r @ OPT.vhat); bp = r - tau*OPT.vhat; rho = np.linalg.norm(bp)
            x0[i, 0] = math.atan2(bp @ OPT.e2, bp @ OPT.e1)
            x0[i, 1] = np.arctanh(np.clip((rho-775.)/75., -0.99, 0.99))
            x0[i, 2] = np.arctanh(np.clip(tau/1500., -0.99, 0.99))
        es = cma.CMAEvolutionStrategy(x0.reshape(24), 0.35, {"popsize": 14, "seed": 1, "verbose": -9}); best = (-1e9, None, None)
    rng = np.random.default_rng()
    if a.final:
        g = V2.genes_from_x(best[1]); rs = OPT.stations(g)
        full = (_m["centroids"], _m["normals"], _m["areas"], _m["vertices"], _m["faces"]); At = _m["areas"].sum()
        rngv = np.random.default_rng(7); T0 = []; T1 = []; T2 = []
        for _ in range(24):
            p = OPT.TUM.sample_tumbler(rngv); _, dc = OPT.TUM.integrate_window(p, np.array([0., 1.])); R = dc[0]
            cat, lit, _e = SIM.score_pass(rs, R, full, GEOM, False, ESH, NAC_KM)
            T0.append(100*_m["areas"][cat == 0].sum()/At); T1.append(100*_m["areas"][cat <= 1].sum()/At); T2.append(100*_m["areas"][cat <= 2].sum()/At)
        tgt = (OPT.C_A @ rs.T).T; np.savez(os.path.join(HERE, "case1_targets_tiers_opt.npz"), DEPUTY_TARGETS_M=tgt)
        print("OPTIMIZED (full mesh, 24 poses): 0.15m=%.1f%%  <=1m=%.1f%%  +shape=%.1f%%" % (np.mean(T0), np.mean(T1), np.mean(T2)))
        print("min-range %d m (clearance %d m), sep %.0f m" % (round(np.clip(g[:,1],*OPT.RHO).min()), round(np.clip(g[:,1],*OPT.RHO).min()-205), OPT.sep_min(rs)))
        print("wrote case1_targets_tiers_opt.npz"); return
    t0 = time.time(); ng = 0
    while time.time()-t0 < TBUDGET and not es.stop():
        X = es.ask(); vals = [fitness(x, rng) for x in X]; es.tell(X, [-v[0] for v in vals])
        gi = int(np.argmax([v[0] for v in vals]))
        if vals[gi][0] > best[0]: best = (vals[gi][0], X[gi].copy(), vals[gi][1])
        ng += 1
    pickle.dump((es, best), open(PKL, "wb"))
    t0_, t1_, t2_, sp = best[2]
    print("CMA-ES +%d gens (iter %d): best J=%.1f  ->  0.15m=%.1f  <=1m=%.1f  shape=%.1f  sep=%.0fm (sigma %.3f)"
          % (ng, es.countiter, best[0], t0_, t1_, t2_, sp, es.sigma))

if __name__ == "__main__":
    main()
