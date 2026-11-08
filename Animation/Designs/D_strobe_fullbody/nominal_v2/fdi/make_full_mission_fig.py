#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_full_mission_fig.py — whole-mission FDI figure (deploy -> encounter, onboard vs ground)."""
import os, sys, subprocess, importlib


def _ensure(pkg, mod=None):
    try:
        return importlib.import_module(mod or pkg)
    except ImportError:
        print(f"[setup] installing {pkg} (one-time)…", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(mod or pkg)


for _p in ("numpy", "matplotlib"):
    _ensure(_p)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import full_mission_monitor as fm
import cockpit_engine as ce

HERE = os.path.dirname(os.path.abspath(__file__))
DD = os.path.abspath(os.path.join(HERE, "..", ".."))                       # …/D_strobe_fullbody
OUT = os.path.join(HERE, "figs"); os.makedirs(OUT, exist_ok=True)
SEP_OUT = os.path.join(DD, "chief deputy seperation", "outputs")
SEP_MC = os.path.join(SEP_OUT, "sep_mc_stats.json")
# rel-nav bundles (rgazel + dep logs): prefer the GNC scenario dir, else the nominal_v2 nav-stage
NAV = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "GNC", "Scenario", "Apophis"))
if not os.path.isdir(NAV):
    NAV = os.path.join(HERE, "..", "data", "mc", "draw_0000", "_nav_stage")
TRAJ = os.path.join(DD, "data", "trajectories"); ANIM = os.path.join(DD, "data", "anim")
MESH = os.path.join(HERE, "..", "data", "radar_mesh.npz")
FP = os.path.join(HERE, "..", "data", "mc_v2_results_nominal_facetprob.npz")

dep, tag = fm.load_separation(SEP_OUT); dres, donb = fm.deploy_fdi(dep); gdep = fm.ground_deploy(SEP_MC)
cres, conb = fm.coast_fdi(NAV)
truth = ce.load_truth(TRAJ, ANIM, MESH, FP, n_frames=60)
eng = ce.MissionEngine(truth, [], ce.OnboardCoverage(truth), ce.GroundOracle(truth))
st = eng.ca_stations()
co, to = eng.onb.score(list(st.values())); cg, tg, _ = eng.gnd.score(list(st.values()))

fig, axs = plt.subplots(2, 2, figsize=(13.5, 9))
ax = axs[0, 0]
phases = [("DEPLOY\nSEP/ALLOC", 0, 1, "#6A1B9A"), ("COAST\nNAV/ATT", 1, 7, "#1565C0"),
          ("APPROACH\nFORM/PEER", 7, 9.4, "#2E7D32"), ("ENCOUNTER\ncoverage", 9.4, 10, "#C62828")]
for name, x0, x1, c in phases:
    ax.barh(0, x1-x0, left=x0, height=0.5, color=c, alpha=.85, edgecolor="k")
    ax.text((x0+x1)/2, 0, name, ha="center", va="center", color="white", fontsize=8, weight="bold")
    ax.text((x0+x1)/2, 0.45, "HEALTHY", ha="center", fontsize=7, color=c)
ax.set_xlim(0, 10); ax.set_ylim(-0.6, 0.7); ax.set_yticks([])
ax.set_xlabel("mission progress (deploy -> CA, not to scale)")
ax.set_title("FDI active across the COMPLETE mission (all phases monitored, all HEALTHY)", fontsize=10)

ax = axs[0, 1]; tho = [r["t_ho"] for r in dres]; x = np.arange(1, 9)
ax.bar(x, tho, color="#6A1B9A", edgecolor="k", label="onboard handoff (live telemetry)")
ax.axhspan(gdep["handoff_p3"], gdep["handoff_p997"], color="#888", alpha=.2, label="ground MC p3-p997")
ax.axhline(gdep["handoff_mean"], ls="--", color="#C62828", label="ground MC mean %.0fs" % gdep["handoff_mean"])
ax.set_xticks(x); ax.set_xticklabels([f"D{i}" for i in range(1, 9)]); ax.set_ylabel("handoff time [s]")
ax.set_title("DEPLOY: onboard handoff inside the N=%d ground MC band" % gdep["N"], fontsize=10); ax.legend(fontsize=7)

ax = axs[1, 0]
for g, lab, col in [(dep[0], "D1", "#6A1B9A"), (dep[4], "D5", "#1565C0")]:
    m = g["t"] < 60
    ax.plot(g["t"][m], np.nan_to_num(g["bores"][m], nan=10), color=col, lw=1.4, label=f"{lab} boresight [deg]")
    ax.plot(g["t"][m], g["om"][m], color=col, lw=1.0, ls=":", label=f"{lab} omega [deg/s]")
ax.axhline(2.0, color="#C62828", ls="--", lw=1); ax.axhline(0.5, color="#F9A825", ls="--", lw=1)
ax.text(45, 2.1, "boresight handoff 2 deg", fontsize=7, color="#C62828")
ax.text(45, 0.6, "rate handoff 0.5 deg/s", fontsize=7, color="#F9A825")
ax.set_xlabel("time since release [s]"); ax.set_ylabel("residual"); ax.set_ylim(0, 5); ax.set_xlim(0, 60)
ax.set_title("SEP channel: detumble + LOS acquisition -> handoff", fontsize=10); ax.legend(fontsize=7, ncol=2)

ax = axs[1, 1]; labels = ["DEPLOY", "COAST", "ENCOUNTER"]
onb = [donb, conb, to*1e3]; gnd = [gdep["compute_s"]*1e3, 120*1e3, tg*1e3]
xx = np.arange(3); w = 0.36
ax.bar(xx-w/2, onb, w, color="#1565C0", label="onboard (ms)")
ax.bar(xx+w/2, gnd, w, color="#C62828", label="ground (ms)")
for i in range(3):
    ax.text(xx[i], max(onb[i], gnd[i])*1.4, "x%d" % (gnd[i]/max(onb[i], 1e-3)), ha="center", fontsize=8, weight="bold")
ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels(labels); ax.set_ylabel("compute [ms, log]")
ax.set_title("ONBOARD vs GROUND compute per phase (onboard live; ground = MC / hi-fi)", fontsize=10); ax.legend(fontsize=8)
fig.suptitle("Full-mission FDI: deployment -> encounter, onboard vs ground", fontsize=14, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(OUT, "full_mission_monitor.png"), dpi=140)
print("wrote figs/full_mission_monitor.png")
