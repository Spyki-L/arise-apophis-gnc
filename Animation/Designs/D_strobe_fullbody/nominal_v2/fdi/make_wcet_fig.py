#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_wcet_fig.py — F9 figure: onboard WCET vs cadence budget per layer."""
import os
import numpy as np, numpy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import flight_readiness as fr
from fdi_core import FDICore, FDIConfig
from reachability import FormTube
from peer_cbf import PeerCBF
from isl_scheduler import schedule_isl
from reoptimizer import fibonacci_sphere, optimize_stations

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs"); os.makedirs(OUT, exist_ok=True)
I3 = np.eye(3); Z3 = np.zeros(3)
pts, norms = fibonacci_sphere(120, 200.)
nom = np.array([[1300*np.cos(a), 1300*np.sin(a), 0] for a in np.linspace(0, 2*np.pi, 8, endpoint=False)])
c = FDICore(FDIConfig(use_form_channel=True, use_peer_channel=True)); c.reset(0.)
tube = FormTube(n=1e-4, rho=np.array([1e3, 2e3, -3e2]), rhod=Z3, P0=np.diag([1, 1, 1, 1e-6, 1e-6, 1e-6]),
                dv_axis=np.full(3, 0.1), t_go=3600., slot=np.array([1.2e3, 2e3, -2e2]))
peer = PeerCBF(); vel = np.zeros((8, 3)); rep = nom.copy(); mr = np.full((8, 8), np.nan)
for i in range(8):
    for j in range(8):
        if i != j: mr[i, j] = la.norm(nom[i]-nom[j])
L = ["FDI step\n(10 Hz)", "FORM tube\n(1 Hz)", "PEER-CBF\n(10 Hz)", "ISL sched\n(1 Hz)", "Re-optimizer\n(event)"]
meas = {L[0]: fr.time_call(lambda: c.step(t=1., innov=Z3, S_innov=I3, dv_est_H=None, P_vv=None, burn_flag=False,
                                          sigma_BR=Z3, omega_BR_B=Z3, hb_age=60., hb_nominal=60.)),
        L[1]: fr.time_call(tube.r_form),
        L[2]: fr.time_call(lambda: peer.evaluate(nom, vel, rep, mr)),
        L[3]: fr.time_call(lambda: schedule_isl(nom, B=4, k_diag=3)),
        L[4]: fr.time_call(lambda: optimize_stations(nom, list(range(8)), pts, norms, np.full(8, 600.), n_iter=120, seed=1), n=5)}
slow = {L[0]: 30, L[1]: 30, L[2]: 30, L[3]: 30, L[4]: 3}
cad = {L[0]: 0.1, L[1]: 1.0, L[2]: 0.1, L[3]: 1.0, L[4]: 60.0}
rows = fr.compute_budget(meas, cad, slow)
wcet = [r["onboard_ms"] for r in rows]; budg = [r["budget_ms"] for r in rows]
proc = ["VSC (rad-hard OBC)", "VSC", "VSC", "VSC", "AC (Zynq SoC)"]
x = np.arange(len(L))
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x, wcet, 0.55, color=["#1565C0"]*4 + ["#6A1B9A"], edgecolor="0.3", label="onboard WCET estimate")
for i in range(len(x)):
    ax.plot([x[i]-0.32, x[i]+0.32], [budg[i], budg[i]], color="#C62828", lw=2.2)
    ax.text(x[i], wcet[i]*1.35, f"{wcet[i]:.2f} ms\n/{budg[i]:.0f} ms", ha="center", fontsize=8)
ax.plot([], [], color="#C62828", lw=2.2, label="cadence deadline")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(L, fontsize=8.5)
ax.set_ylabel("time per call [ms, log]"); ax.set_ylim(0.2, 2e5)
ax.set_title("F9 — onboard WCET vs cadence budget (every layer fits, with margin)\n"
             "diagnosis/safety on a rad-hard VSC OBC; the heavy re-optimizer on the AC co-processor under RTA", fontsize=11)
ax.legend(loc="upper left", fontsize=9); ax.grid(axis="y", alpha=.3, which="both")
for i, p in enumerate(proc):
    ax.text(x[i], 0.27, p, ha="center", fontsize=7, color="#444")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "wcet_compute_budget.png"), dpi=160)
print("wrote figs/wcet_compute_budget.png")
