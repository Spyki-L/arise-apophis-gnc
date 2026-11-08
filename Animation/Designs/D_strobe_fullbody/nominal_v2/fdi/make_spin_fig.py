#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_spin_fig.py — F6 figure: onboard spin estimation + spin-aware NAC re-point recovery."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spin_estimator import SpinEstimator, rot_about_axis, make_roi_facets, nac_roi_coverage, optimize_for_spin

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0); axis = np.array([0, 0, 1.]); t0 = 0.; t_ca = 600.; w_true = np.radians(2.0)
est = SpinEstimator(axis=axis, omega_prior=np.radians(1.0), t0=t0)
t_obs = np.linspace(0, 120, 40); phi = w_true*(t_obs-t0) + rng.normal(0, np.radians(0.3), t_obs.size)
est.update(t_obs, phi)
Rt = 200.; cb, nb = make_roi_facets(Rt, [1, 0, 0], 22., n=300)
M = 8; ang = np.linspace(0, 2*np.pi, M, endpoint=False); shell = 1200.
geom = np.array([[shell*np.cos(a), shell*np.sin(a), 300*np.sin(2*a)] for a in ang]); reach = np.full(M, 1500.)
kw = dict(R_nac=3000., incidence_max_deg=25.)
Rp = rot_about_axis(axis, est.omega_prior*(t_ca-t0)); Ra = est.predict_R(t_ca)
st0, c0 = optimize_for_spin(geom, list(range(M)), cb, nb, Rp, reach, n_iter=300, seed=2, **kw)
nominal = geom.copy()
for d in range(M): nominal[d] = st0[d]
cdrop = nac_roi_coverage(nominal, cb, nb, Ra, **kw)
st1, crec = optimize_for_spin(nominal, list(range(M)), cb, nb, Ra, reach, n_iter=300, seed=4, **kw)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
tt = np.linspace(0, t_ca, 100)
ax1.plot(t_obs, np.degrees(phi), 'o', ms=4, color="#888", label="imagery obs (noisy)")
ax1.plot(tt, np.degrees(est.omega_prior*tt), '--', color="#C62828", label="a-priori (1.0 deg/s)")
ax1.plot(tt, np.degrees(est.predict_phase(tt)), '-', color="#1565C0", lw=1.8,
         label="onboard estimate (%.2f deg/s)" % np.degrees(est.omega))
ax1.axvline(120, ls=':', color="0.5"); ax1.text(125, 50, "obs arc ends", fontsize=8, color="0.4")
ax1.set_xlabel("time [s]"); ax1.set_ylabel("body spin phase [deg]")
ax1.set_title("Onboard spin estimation from imagery\n(detects the off-nominal rate; AM-consistent)", fontsize=10)
ax1.legend(fontsize=8, loc="upper left"); ax1.grid(alpha=.3)

cats = ["a-priori-opt\n(planned spin)", "actual spin\n(fixed stations)", "spin-aware\nre-point", "too late\n(hold)"]
vals = [c0, cdrop, crec, cdrop]; cols = ["#2E7D32", "#C62828", "#1565C0", "#F9A825"]
b = ax2.bar(cats, vals, color=cols, edgecolor="0.3"); ax2.set_ylim(0, 1.08); ax2.set_ylabel("NAC 0.15 m ROI coverage")
for r, v in zip(b, vals): ax2.text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=9)
ax2.set_title("Re-point recovers high-res coverage after a spin change\n(deadline-gated: hold if unreachable in time)", fontsize=10)
ax2.grid(axis="y", alpha=.3)
fig.suptitle("F6 — Apophis spin-change reconfiguration (estimate -> re-point -> recover 0.15 m NAC)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "spin_repoint_recovery.png"), dpi=160)
print(f"wrote figs/spin_repoint_recovery.png  (c0={c0:.2f} drop={cdrop:.2f} rec={crec:.2f})")
