#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_form_figs.py — F3 figures: recoverability frontier + 4D tube cross-sections."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from reachability import FormTube, recoverability_frontier, propagate_pos

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

n = 1e-4; t_go = 3600.; rho = np.array([1000., 2000., -300.]); rhod = np.zeros(3)
P0 = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6]); dv = np.array([0.1, 0.1, 0.1])
c = propagate_pos(rho, rhod, n, t_go); slot = c + np.array([200., 0., 0.])
base = FormTube(n=n, rho=rho, rhod=rhod, P0=P0, dv_axis=dv, t_go=t_go, slot=slot)

# Fig 1 — recoverability frontier
gg = np.linspace(0.05, 1.0, 40); tg = np.linspace(300., 12000., 60)
fr = recoverability_frontier(base, gg, tg)
g_ok = [g for g in gg if fr[g] is not None]; y = [fr[g]/60. for g in g_ok]
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.plot([g*100 for g in g_ok], y, '-o', ms=3, color="#0072B2", lw=1.6)
ax.fill_between([g*100 for g in g_ok], y, 0, color="#0072B2", alpha=0.10)
ax.set_xlabel("Surviving control authority  [% of nominal]")
ax.set_ylabel("Latest moment still recoverable\n(time-to-CA, min)")
ax.set_title("FORM 4D-tube recoverability frontier\n(slot still reachable above the curve; lost below)", fontsize=11)
ax.grid(alpha=.3)
if any(fr[g] is None for g in gg):
    ax.axvspan(5, max(g for g in gg if fr[g] is None)*100, color="#D55E00", alpha=.08)
    ax.text(7, ax.get_ylim()[1]*0.85, "too degraded\nto recover", color="#D55E00", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "recoverability_frontier.png"), dpi=160)

# Fig 2 — tube cross-sections
def ellipse(ax, c2, S2, color, label=None):
    w, V = np.linalg.eigh(S2); ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
    ax.add_patch(Ellipse(c2, 2*np.sqrt(max(w[0], 1e-9)), 2*np.sqrt(max(w[1], 1e-9)), angle=ang,
                         fill=False, edgecolor=color, lw=1.6, label=label))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (tube, title) in zip(axes, [(base, "Nominal authority — slot INSIDE tube"),
                                     (base.with_authority(dv*0.18), "Thruster-degraded — slot OUTSIDE tube")]):
    for dt, col in [(t_go, "#0072B2"), (0.66*t_go, "#56B4E9"), (0.33*t_go, "#9ad0ec")]:
        cc, S = tube.cross_section(dt); ellipse(ax, cc[:2], S[:2, :2], col, label=f"reach @ {dt/60:.0f} min")
    ax.plot(slot[0], slot[1], '*', ms=16, color="#D55E00", label="required slot s*")
    ax.plot(c[0], c[1], 'k.', ms=6, label="uncontrolled terminal")
    d = tube.r_form()
    ax.set_title(f"{title}\nslot Mahalanobis d={d:.2f}  ({'reachable' if d <= 1 else 'FORM FAULT'})", fontsize=10)
    ax.set_xlabel("Hill x [m]"); ax.set_ylabel("Hill y [m]"); ax.grid(alpha=.3); ax.set_aspect('equal', 'datalim')
axes[0].legend(fontsize=7, loc="best")
fig.suptitle("FORM channel as a 4D reachable tube — cross-sections propagate to the CA slot", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "form_tube_xsections.png"), dpi=160)
print("wrote figs/recoverability_frontier.png and figs/form_tube_xsections.png")
