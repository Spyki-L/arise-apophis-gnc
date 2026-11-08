#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_mission_fig.py — F8 capstone figure: whole-mission autonomy coverage timeline."""
import os
import numpy as np, numpy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mission_profile import load_profile
from autonomy_executive import AutonomyExecutive, Phase, Event
from reoptimizer import fibonacci_sphere
from spin_estimator import make_roi_facets, rot_about_axis

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs"); os.makedirs(OUT, exist_ok=True)
profile = load_profile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles", "apophis.json"))
wb_pts, wb_norms = fibonacci_sphere(120, 200.); roi_cb, roi_nb = make_roi_facets(200., [1, 0, 0], 22., n=120)
M = 8; ang = np.linspace(0, 2*np.pi, M, endpoint=False); shell = 1300.
geom = np.array([[shell*np.cos(a), shell*np.sin(a), 350*np.sin(2*a)] for a in ang]); reach = np.full(M, 1500.)
kw_wb = dict(R_max=4000., th_min=8., th_max=45.); kw_roi = dict(R_nac=3000., incidence_max_deg=25.)
exe = AutonomyExecutive(profile, wb_pts, wb_norms, roi_cb, roi_nb, geom, reach,
                        d_safe=50., w_wb=0.6, w_roi=0.4, dv_axis=1.0, kw_wb=kw_wb, kw_roi=kw_roi)
nom, _ = exe._optimize(np.eye(3), n_iter=350, seed=3); exe.nominal = nom.copy(); exe.stations = nom.copy()
cov0 = exe.mission_coverage()
d6 = nom[6]; others = [nom[j] for j in range(M) if j != 6]; nearest = min(others, key=lambda p: la.norm(p - d6))
v6 = 2.0*(nearest - d6)/la.norm(nearest - d6); fleet_vel = [np.zeros(3)]*M
tl = [(Phase.COAST, 3000., Event.DEPUTY_LOSS, dict(lost=2, n_iter=120)),
      (Phase.COAST, 2500., Event.ACTUATION_DEGRADE, dict(dep=5, new_reach=60., n_iter=120)),
      (Phase.TERMINAL, 1500., Event.SPIN_CHANGE, dict(body_R=rot_about_axis([0, 0, 1], np.radians(60.)), n_iter=120)),
      (Phase.TERMINAL, 1200., Event.MISBEHAVING, dict(suspect=6, fleet_vel=fleet_vel, suspect_state=(d6, v6), n_iter=100)),
      (Phase.TERMINAL, 12., Event.SPIN_CHANGE, dict(body_R=rot_about_axis([0, 0, 1], np.radians(95.)), n_iter=120))]
res = exe.run_mission(tl)
labels = ["nominal", "D2 lost", "D5 actuator\ndegraded", "spin change\n(reachable)", "D6 misbehaving\n(collision)", "spin change\n(too late)"]
acts = ["—", "reconfigure", "re-task", "re-point", "dispose+reopt", "HOLD (deadline)"]
pre = [cov0] + [r.get("dropped", r.get("before")) for r in res]
post = [cov0] + [r["after"] for r in res]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(x, post, '-o', color="#1565C0", lw=2, ms=7, label="after autonomous action", zorder=3)
ax.plot(x[1:], pre[1:], 'v', color="#C62828", ms=9, label="event impact (pre-action)", zorder=3)
for i in range(1, len(x)):
    ax.annotate("", xy=(x[i], post[i]), xytext=(x[i], pre[i]), arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.5))
    ax.text(x[i], max(post[i], pre[i]) + 0.03, acts[i], ha="center", fontsize=8, color="#2E7D32")
ax.axhline(cov0, ls="--", color="0.6", lw=1, label="nominal coverage")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylim(0.45, 1.25)
ax.set_ylabel("mission coverage (0.6 whole-body + 0.4 NAC-ROI)")
ax.set_title("F8 — closed-loop Autonomy Executive: whole mission, no ground intervention\n"
             "each fault is diagnosed, valued by recoverable coverage, and acted on under run-time assurance", fontsize=11)
ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=.3)
ax.text(0.99, 0.03, f"RTA: AC accepts={exe.rta.n_ac}  VSC fallbacks={exe.rta.n_vsc}  "
        f"unsafe executed={exe.executed_unsafe}  ground calls={exe.ground_calls}",
        transform=ax.transAxes, ha="right", fontsize=8, style="italic", color="#444")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "mission_autonomy_timeline.png"), dpi=160)
print(f"wrote figs/mission_autonomy_timeline.png  final={exe.mission_coverage():.3f} nominal={cov0:.3f}")
