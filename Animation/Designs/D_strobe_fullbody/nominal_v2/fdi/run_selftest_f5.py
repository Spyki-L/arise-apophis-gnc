#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_selftest_f5.py — F5 proofs: two-tier onboard stereo-coverage re-optimizer.

  T1 surrogate: a ground-optimized 8-deputy config gives high coverage
  T2 losing a deputy drops coverage (a real gap to recover)
  T3 Tier-2 onboard re-optimization recovers most of the gap
  T4 the reachable (F3 tube) constraint is respected (a degraded deputy stays in its box)
  T5 Tier-1 contingency lookup beats doing nothing
  T6 bounded iterations -> deterministic WCET
  T7 RTA-gated reconfigure returns >= Tier-1 with a valid controller source
"""
import numpy as np
from reoptimizer import (fibonacci_sphere, coverage_surrogate, optimize_stations,
                         build_contingency_table, ReOptimizer)

Rt = 200.; pts, norms = fibonacci_sphere(200, Rt)
M = 8; ang = np.linspace(0, 2*np.pi, M, endpoint=False); shell = 1400.
geom = np.array([[shell*np.cos(a), shell*np.sin(a), 500*np.sin(2*a)] for a in ang])
reach = np.full(M, 700.); kw = dict(R_max=4000., th_min=8., th_max=45.)

st0, _, _ = optimize_stations(geom, list(range(M)), pts, norms, reach, n_iter=400, seed=3, **kw)
nominal = geom.copy()
for d in range(M): nominal[d] = st0[d]
cov_nom = coverage_surrogate(nominal, pts, norms, **kw)

ok = 0; tot = 0
def check(name, cond, extra=""):
    global ok, tot; tot += 1
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else "")); ok += int(bool(cond))

check("T1 ground-optimized 8-deputy coverage is high", cov_nom > 0.8, f"cov_nom={cov_nom:.3f}")

active = [d for d in range(M) if d != 2]
cov_lost = coverage_surrogate(nominal[active], pts, norms, **kw)
check("T2 deputy loss drops coverage (real gap)", cov_lost < cov_nom - 0.02, f"lost={cov_lost:.3f} < nom={cov_nom:.3f}")

st, cov_opt, ev = optimize_stations(nominal, active, pts, norms, reach, n_iter=300, seed=5, **kw)
rec = (cov_opt - cov_lost) / max(cov_nom - cov_lost, 1e-6)
check("T3 Tier-2 re-opt recovers most of the gap", rec > 0.7, f"recovered {100*rec:.0f}% ({cov_lost:.3f} -> {cov_opt:.3f})")

reach2 = reach.copy(); reach2[5] = 50.
st2, _, _ = optimize_stations(nominal, active, pts, norms, reach2, n_iter=200, seed=9, **kw)
moved = np.linalg.norm(st2[active.index(5)] - nominal[5])
check("T4 reachable (F3 tube) constraint respected", moved <= 50.0 + 1e-6, f"degraded D5 moved {moved:.1f} m (<=50)")

check("T6 bounded iterations -> deterministic WCET", ev == 301, f"evals={ev} == n_iter+1")

table = build_contingency_table(nominal, pts, norms, reach, n_iter=180, **kw)
ro = ReOptimizer(nominal, pts, norms, table, **kw)
t1 = ro.tier1({2})
check("T5 Tier-1 contingency lookup beats do-nothing", t1["coverage"] > cov_lost,
      f"Tier1={t1['coverage']:.3f} > lost={cov_lost:.3f}")

rc = ro.reconfigure({2}, reach, n_iter=120)
check("T7 RTA reconfigure: >= Tier-1 with a valid source",
      rc["coverage"] >= t1["coverage"] - 1e-4 and rc["source"] in ("AC(Tier2)", "VSC(Tier1)"),
      f"source={rc['source']} cov={rc['coverage']:.3f}")

print(f"\n{ok}/{tot} passed")
