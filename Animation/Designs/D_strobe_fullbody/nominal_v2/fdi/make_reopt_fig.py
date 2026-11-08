#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_reopt_fig.py — F5 figure: coverage recovery after deputy loss (single + all-losses)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reoptimizer import fibonacci_sphere, coverage_surrogate, optimize_stations, build_contingency_table, ReOptimizer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs"); os.makedirs(OUT, exist_ok=True)
Rt = 200.; pts, norms = fibonacci_sphere(180, Rt)
M = 8; ang = np.linspace(0, 2*np.pi, M, endpoint=False); shell = 1400.
geom = np.array([[shell*np.cos(a), shell*np.sin(a), 500*np.sin(2*a)] for a in ang])
reach = np.full(M, 700.); kw = dict(R_max=4000., th_min=8., th_max=45.)
st0, _, _ = optimize_stations(geom, list(range(M)), pts, norms, reach, n_iter=350, seed=3, **kw)
nominal = geom.copy()
for d in range(M): nominal[d] = st0[d]
cov_nom = coverage_surrogate(nominal, pts, norms, **kw)
table = build_contingency_table(nominal, pts, norms, reach, n_iter=160, **kw)
ro = ReOptimizer(nominal, pts, norms, table, **kw)

lost_fixed, t1c, t2c = [], [], []
for L in range(M):
    act = [d for d in range(M) if d != L]
    lost_fixed.append(coverage_surrogate(nominal[act], pts, norms, **kw))
    t1c.append(ro.tier1({L})["coverage"])
    t2c.append(ro.reconfigure({L}, reach, n_iter=120)["coverage"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
i = 2
cats = ["nominal\n(8 dep)", "lose D2\n(unhandled)", "Tier-1\ncontingency", "Tier-2\nonboard"]
vals = [cov_nom, lost_fixed[i], t1c[i], t2c[i]]; cols = ["#2E7D32", "#C62828", "#F9A825", "#1565C0"]
b = ax1.bar(cats, vals, color=cols, edgecolor="0.3"); ax1.set_ylim(0, 1.05); ax1.set_ylabel("stereo coverage (surrogate)")
for r, v in zip(b, vals): ax1.text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=9)
ax1.set_title("Onboard recovery after losing one deputy (D2)", fontsize=11); ax1.grid(axis="y", alpha=.3)
x = np.arange(M); w = 0.4
ax2.bar(x-w/2, lost_fixed, w, label="unhandled (stations fixed)", color="#C62828")
ax2.bar(x+w/2, t2c, w, label="Tier-2 re-optimized", color="#1565C0")
ax2.axhline(cov_nom, ls="--", color="#2E7D32", label=f"nominal 8-dep = {cov_nom:.2f}")
ax2.set_xticks(x); ax2.set_xticklabels([f"D{d}" for d in range(M)]); ax2.set_ylim(0, 1.05)
ax2.set_xlabel("lost deputy"); ax2.set_ylabel("stereo coverage"); ax2.grid(axis="y", alpha=.3)
ax2.set_title("Robust to ANY single-deputy loss", fontsize=11); ax2.legend(fontsize=8, loc="lower right")
fig.suptitle("F5 — two-tier onboard re-optimization recovers stereo coverage after deputy loss", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "reopt_coverage_recovery.png"), dpi=160)
print(f"mean across 8 losses: unhandled={np.mean(lost_fixed):.3f}  Tier2={np.mean(t2c):.3f}  nominal={cov_nom:.3f}")
print("wrote figs/reopt_coverage_recovery.png")
