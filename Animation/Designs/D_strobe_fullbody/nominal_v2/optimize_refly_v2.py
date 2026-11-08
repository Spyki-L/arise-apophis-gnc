#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_refly_v2.py - robust CMA-ES + NSGA-II re-optimizer for the tighter-standoff fleet.

Reuses the kinematic coverage machinery in optimize_refly_stations.py (CASE-1 fidelity).
Improvements over the plain GA:
  * ROBUST objective: coverage is averaged/percentiled over POSES *resampled every eval*
    (the CA spin phase is unknown) -> no overfit to a fixed set of poses.
  * CMA-ES (well-conditioned continuous search) in a tanh-bounded normalized space.
  * NSGA-II: Pareto front of [cov@0.15m, cov@0.5m, clearance-margin] under sep>=150 m.

Modes:
  python optimize_refly_v2.py --cma           # evolve CMA-ES (pickled resume)
  python optimize_refly_v2.py --cma --final    # validate CMA best, write targets npz
  python optimize_refly_v2.py --nsga          # evolve NSGA-II (checkpointed)
  python optimize_refly_v2.py --nsga --final   # dump Pareto front (npz) + summary
"""
import os, sys, time, math, pickle, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import optimize_refly_stations as OPT   # kinematics, coverage(), stations(), sep_min(), C0/N0/areas, TUM, BRDF

RHO = OPT.RHO; SEP_MIN = OPT.SEP_MIN
KPOSE = 20; TBUDGET = 35.0
CMA_PKL = os.path.join(HERE, "refly_cma.pkl")
NSGA_NPZ = os.path.join(HERE, "refly_nsga_state.npz")

def genes_from_x(x):
    g = np.asarray(x, float).reshape(8, 3).copy()
    g[:, 1] = 775.0 + 75.0*np.tanh(g[:, 1])      # rho -> [700,850]
    g[:, 2] = 1500.0*np.tanh(g[:, 2])            # tau -> [-1500,1500]
    return g

def cov_samples(genes, gsd, kpose, rng):
    rs = OPT.stations(genes); out = []
    for _ in range(kpose):
        p = OPT.TUM.sample_tumbler(rng); _, dc = OPT.TUM.integrate_window(p, np.array([0., 1.])); R = dc[0]; h = OPT.BRDF.sample_hapke(rng)
        out.append(OPT.coverage(rs, OPT.C0, OPT.N0, OPT.areas, R, h, gsd))
    return np.nan_to_num(np.array(out), nan=0.0, posinf=0.0, neginf=0.0)

def robust_obj(x, rng):
    g = genes_from_x(x); rs = OPT.stations(g); sp = OPT.sep_min(rs)
    pen = max(0.0, SEP_MIN - sp)*0.3
    c = cov_samples(g, OPT.GSD_OBJ, KPOSE, rng)
    return float(np.percentile(c, 5)) - pen, float(c.mean()), sp

# ───────────────────────── CMA-ES ─────────────────────────
def run_cma(final):
    import cma
    if os.path.exists(CMA_PKL):
        with open(CMA_PKL, "rb") as f:
            es, best = pickle.load(f)
    else:
        x0 = np.zeros(24); x0[0:8] = np.linspace(0, 2*np.pi, 8, endpoint=False)  # seed b-plane ring
        es = cma.CMAEvolutionStrategy(x0, 0.6, {"popsize": 14, "seed": 1, "verbose": -9})
        best = (-1e9, None)
    rng = np.random.default_rng()
    if final:
        x = best[1]; g = genes_from_x(x); rs = OPT.stations(g)
        C0f, N0f, areasf = OPT.load_mesh(1); rngv = np.random.default_rng(7)
        gsds = [0.15, 0.2, 0.3, 0.5, 1.0]; acc = np.zeros((40, len(gsds)))
        for i in range(40):
            p = OPT.TUM.sample_tumbler(rngv); _, dc = OPT.TUM.integrate_window(p, np.array([0., 1.])); R = dc[0]; h = OPT.BRDF.sample_hapke(rngv)
            acc[i] = [OPT.coverage(rs, C0f, N0f, areasf, R, h, gg) for gg in gsds]
        rho = g[:, 1]; tgt = (OPT.C_A @ rs.T).T
        np.savez(os.path.join(HERE, "refly_targets_cma.npz"), DEPUTY_TARGETS_M=tgt)
        print("CMA-ES best: P5=%.1f  fit-mean=%.1f" % (best[0], best[0]))
        print("min-range/deputy [m]:", np.round(rho).astype(int).tolist(), " clearance %d m" % int(rho.min()-205))
        print("inter-deputy sep: %.0f m (limit %d)" % (OPT.sep_min(rs), SEP_MIN))
        print("coverage@0.15m: mean %.1f  P5 %.1f  P95 %.1f  (40 fresh poses)" % (acc[:,0].mean(), np.percentile(acc[:,0],5), np.percentile(acc[:,0],95)))
        print("coverage vs GSD (mean): ", {g_: round(acc[:,k].mean(),1) for k,g_ in enumerate(gsds)})
        print("DEPUTY_TARGETS_KM = np.array(["); [print("    [%6.3f, %6.3f, %6.3f]," % tuple(v/1e3)) for v in tgt]; print("])")
        print("wrote refly_targets_cma.npz"); return
    t0 = time.time(); ng = 0
    while time.time()-t0 < TBUDGET and not es.stop():
        X = es.ask(); vals = [robust_obj(x, rng) for x in X]
        es.tell(X, [-v[0] for v in vals])
        gi = int(np.argmax([v[0] for v in vals]))
        if vals[gi][0] > best[0]: best = (vals[gi][0], X[gi].copy())
        ng += 1
    with open(CMA_PKL, "wb") as f:
        pickle.dump((es, best), f)
    print("CMA-ES +%d gens (iter=%d), best robust P5@0.15m = %.1f%%  (sigma=%.3f)" % (ng, es.countiter, best[0], es.sigma))

# ───────────────────────── NSGA-II ─────────────────────────
NPOP = 28
def objectives(x, rng):
    g = genes_from_x(x); rs = OPT.stations(g); sp = OPT.sep_min(rs)
    c15 = cov_samples(g, 0.15, 5, rng).mean(); c50 = cov_samples(g, 0.5, 5, rng).mean()
    clr = float(g[:, 1].min() - 205.0)               # body clearance margin [m]
    con = max(0.0, SEP_MIN - sp)                      # 0 = feasible
    return np.array([c15, c50, clr]), con

def dominates(o1, c1, o2, c2):
    if c1 < c2: return True
    if c1 > c2: return False
    return np.all(o1 >= o2) and np.any(o1 > o2)

def nondom_sort(O, C):
    N = len(O); S = [[] for _ in range(N)]; n = np.zeros(N, int); rank = np.zeros(N, int); fronts = [[]]
    for p in range(N):
        for qd in range(N):
            if p == qd: continue
            if dominates(O[p], C[p], O[qd], C[qd]): S[p].append(qd)
            elif dominates(O[qd], C[qd], O[p], C[p]): n[p] += 1
        if n[p] == 0: rank[p] = 0; fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for qd in S[p]:
                n[qd] -= 1
                if n[qd] == 0: rank[qd] = i+1; nxt.append(qd)
        i += 1; fronts.append(nxt)
    return fronts[:-1], rank

def crowding(front, O):
    l = len(front); d = np.zeros(l)
    if l <= 2: return {front[k]: 1e9 for k in range(l)}
    O = np.array([O[i] for i in front])
    for m in range(O.shape[1]):
        idx = np.argsort(O[:, m]); d[idx[0]] = d[idx[-1]] = 1e9; rng = O[idx[-1], m]-O[idx[0], m] + 1e-9
        for k in range(1, l-1): d[idx[k]] += (O[idx[k+1], m]-O[idx[k-1], m])/rng
    return {front[k]: d[k] for k in range(l)}

def run_nsga(final):
    rng = np.random.default_rng()
    if os.path.exists(NSGA_NPZ):
        z = np.load(NSGA_NPZ); pop = z["pop"]; gen = int(z["gen"])
    else:
        pop = rng.normal(0, 0.8, (NPOP, 24)); pop[:, 0:8] = rng.uniform(0, 2*np.pi, (NPOP, 8)); gen = 0
    if final:
        O = []; C = []
        for x in pop:
            o, c = objectives(x, np.random.default_rng(7)); O.append(o); C.append(c)
        O = np.array(O); C = np.array(C); fronts, rank = nondom_sort(O, C)
        pf = [i for i in fronts[0] if C[i] == 0]
        np.savez(os.path.join(HERE, "refly_pareto.npz"), objs=O[pf], pop=pop[pf])
        print("Pareto front (%d feasible non-dominated):" % len(pf))
        for i in sorted(pf, key=lambda i:-O[i][0])[:12]:
            print("  cov@0.15=%.1f  cov@0.5=%.1f  clearance=%.0f m" % (O[i][0], O[i][1], O[i][2]))
        print("wrote refly_pareto.npz"); return
    t0 = time.time(); ng = 0
    while time.time()-t0 < TBUDGET:
        O = []; C = []
        for x in pop:
            o, c = objectives(x, rng); O.append(o); C.append(c)
        O = np.array(O); C = np.array(C); fronts, rank = nondom_sort(O, C)
        cd = {}
        for fr in fronts: cd.update(crowding(fr, O))
        def better(i, j): return (rank[i] < rank[j]) or (rank[i] == rank[j] and cd.get(i,0) > cd.get(j,0))
        children = []
        while len(children) < NPOP:
            a, b = rng.integers(NPOP), rng.integers(NPOP); p1 = a if better(a, b) else b
            a, b = rng.integers(NPOP), rng.integers(NPOP); p2 = a if better(a, b) else b
            mask = rng.random(24) < 0.5; ch = np.where(mask, pop[p1], pop[p2]).copy()
            ch += rng.normal(0, 0.25, 24)*(rng.random(24) < 0.3)
            children.append(ch)
        comb = np.vstack([pop, np.array(children)]); Oc = []; Cc = []
        for x in comb:
            o, c = objectives(x, rng); Oc.append(o); Cc.append(c)
        Oc = np.array(Oc); Cc = np.array(Cc); fr, rk = nondom_sort(Oc, Cc)
        newidx = []
        for f in fr:
            if len(newidx)+len(f) <= NPOP: newidx += f
            else:
                cdf = crowding(f, Oc); newidx += sorted(f, key=lambda i:-cdf[i])[:NPOP-len(newidx)]; break
        pop = comb[newidx]; gen += 1; ng += 1
    np.savez(NSGA_NPZ, pop=pop, gen=gen)
    print("NSGA-II +%d gens (total %d)" % (ng, gen))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cma", action="store_true"); ap.add_argument("--nsga", action="store_true"); ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.cma: run_cma(a.final)
    elif a.nsga: run_nsga(a.final)
