#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FORM/PEER demo figure + RL training/eval figures."""
import json, sys
from pathlib import Path
import numpy as np
import numpy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import use_style, C, SERIES, FULL_W, HALF_W
from fdi_core import FDICore, FDIState, FaultType
from channels import PeerConsensusBuffer
import exp_formpeer as FP
import run_fdi_nominal as RFN

R = Path(__file__).resolve().parent.parent / "results"
FIGS = Path(__file__).resolve().parent.parent / "figs"
use_style(serif=True)

def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf"); fig.savefig(FIGS / f"{name}.png", dpi=150)
    plt.close(fig); print("wrote", name)

def jload(fn):
    return json.load(open(R / fn))

# ── FORM/PEER example trial with tracing ─────────────────────────────────────
def traced_form_local(seed=11):
    fleet, t_tca = RFN.load_fleet(FP.DATA)
    events, miss = FP.precompute(fleet, t_tca)
    rng = np.random.default_rng(seed)
    cfg = FP.cfg_formpeer()
    n_dep = len(fleet)
    cores = {d: FDICore(cfg) for d in range(n_dep)}
    for d in range(n_dep):
        cores[d].reset(t0=events[0][0])
    peer = PeerConsensusBuffer(n_dep=n_dep, latency_s=0.0, mad_floor=0.25)
    focus = int(rng.integers(0, n_dep))
    t_fault = float(rng.uniform(t_tca - FP.FORM_WINDOW_S, t_tca - FP.MIN_T_GO_S))
    tr = {d: dict(t=[], rf=[], rp=[], Sf=[], Sp=[], st=[]) for d in range(n_dep)}
    for t_k, d, k in events:
        g = fleet[d]
        innov = g["ypre"][k]
        if (not np.all(np.isfinite(innov))) or bool(g["rejected"][k]):
            ik = Sk = None
        else:
            ik = innov; sg = np.maximum(g["sig"][k], 1e-6); Sk = np.diag(sg ** 2)
        dvc = g["dv"][k]; burn = bool(la.norm(dvc) > 0.0); de = Pvv = None
        if burn and k > 0:
            Pvv = g["P"][k, 3:6, 3:6]; de = (g["x"][k, 3:6] - g["x"][k - 1, 3:6]) - dvc
        rf = rp = None
        m = miss[d][k]
        if np.isfinite(m):
            frac = max(0.0, (t_k - t_fault)) / max(t_tca - t_fault, 1.0)
            if d == focus and t_k >= t_fault:
                m = m + FP.RAMP_M * frac
            ttg = max(t_tca - t_k, 0.0)
            w = 1.0 + 1.5 * np.clip(1.0 - ttg / FP.FORM_WINDOW_S, 0.0, 1.0)
            peer.push(d, t_k, m)
            scale = peer.fleet_scale(t_k, floor=FP.FORM_SCALE_M)
            rf = (m / scale) * w
            rp = peer.r_peer(d, t_k, own_value=m, scale=scale)
        snap = cores[d].step(t=t_k, innov=ik, S_innov=Sk, dv_est_H=de, P_vv=Pvv,
                             burn_flag=burn, sigma_BR=g["sBR"][k], omega_BR_B=g["oBR"][k],
                             hb_age=FP.HB, hb_nominal=FP.HB, r_form=rf, r_peer=rp)
        tr[d]["t"].append(t_k); tr[d]["rf"].append(rf if rf is not None else np.nan)
        tr[d]["rp"].append(rp if rp is not None else np.nan)
        tr[d]["Sf"].append(cores[d].S_form); tr[d]["Sp"].append(cores[d].S_peer)
        tr[d]["st"].append(snap.state.state.value)
    return tr, focus, t_fault, t_tca, cores

def fig_formpeer():
    tr, focus, t_fault, t_tca, cores = traced_form_local()
    lab = cores[focus].state.fault_type.name
    fig = plt.figure(figsize=(FULL_W, 2.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.14, wspace=0.26)
    healthy = 0 if focus != 0 else 1
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    for d, col, nm in ((focus, C["verm"], "faulted deputy"), (healthy, C["blue"], "healthy deputy")):
        tt = (np.array(tr[d]["t"]) - t_tca) / 60.0
        ax1.plot(tt, tr[d]["Sf"], lw=1.2, color=col, label=nm)
        ax2.plot(tt, tr[d]["Sp"], lw=1.2, color=col)
    ax1.axhline(30, color="0.35", ls="--", lw=0.9); ax1.set_ylabel("$S_{form}$")
    ax2.axhline(20, color="0.35", ls="--", lw=0.9); ax2.set_ylabel("$S_{peer}$")
    for ax in (ax1, ax2):
        ax.axvline((t_fault - t_tca) / 60.0, color=C["orange"], lw=1.0, ls=":")
    ax1.text((t_fault - t_tca) / 60.0 + 1.0, 120, "fault onset", fontsize=6.6, color=C["orange"])
    ax1.legend(fontsize=6.6, loc="upper left")
    ax1.tick_params(labelbottom=False)
    ax2.set_xlabel("time to closest approach [min]")
    ax1.set_title(f"(a) FORM_LOCAL example: settled label = {lab}", fontsize=8.3)

    axb = fig.add_subplot(gs[:, 1])
    d1 = jload("part_fp_FORM_LOCAL.json"); d2 = jload("part_fp_NAV_GLOBAL_ABSORBED.json")
    d3 = jload("part_fp_NAV_GLOBAL_STARVED.json")
    d4 = jload("part_fp_PEER_STRESS.json"); d5 = jload("part_fp_PEER_STRESS_f1e-09.json")
    cases = [("local fault\n$\\rightarrow$ FORM", d1["correct"] / d1["n"]),
             ("common mode\n$\\rightarrow$ absorbed", d2["healthy_trials"] / d2["n"]),
             ("common mode,\nISL out $\\rightarrow$ NAV", d3["correct"] / d3["n"]),
             ("homogeneous,\nMAD floor on", d4["healthy_trials"] / d4["n"]),
             ("homogeneous,\nno floor", d5["healthy_trials"] / d5["n"])]
    y = np.arange(len(cases))[::-1]
    vals = [c[1] for c in cases]
    cols = [C["green"]] * 4 + [C["verm"]]
    axb.barh(y, vals, 0.62, color=cols, edgecolor="none")
    for yi, v in zip(y, vals):
        axb.text(min(v + 0.02, 0.86), yi, f"{100*v:.0f}%", va="center", fontsize=7)
    axb.set_yticks(y); axb.set_yticklabels([c[0] for c in cases], fontsize=6.4)
    axb.set_xlim(0, 1.12); axb.set_xlabel("correct outcome rate", fontsize=8)
    axb.set_title("(b) campaign outcomes", fontsize=8.3)
    save(fig, "fig_formpeer")
fig_formpeer()

# ── RL training curves ────────────────────────────────────────────────────────
def fig_rl_training():
    df = pd.read_csv(R / "tb_scalars.csv")
    stages = [("PPO_1", "local pre-train", "0.0"),
              ("hpc_severe_0", "severe", None), ("hpc_nominal_0", "nominal", None),
              ("hpc_incipient_0", "incipient", None), ("hpc_adversarial_0", "adversarial", None),
              ("hpc_hard_class_0", "hard-class fine-tune", None)]
    tags = [("train/loss", "total loss"), ("train/value_loss", "value loss"),
            ("train/entropy_loss", "entropy loss"), ("train/explained_variance", "explained variance")]
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 3.7), sharex=True)
    bounds = {}
    for dn, _, _ in stages:
        sub = df[df.dir_name == dn]
        if len(sub):
            bounds[dn] = (sub.step.min() / 1e6, sub.step.max() / 1e6)
    for ax, (tag, name) in zip(axes.flat, tags):
        for i, (dn, lb, _) in enumerate(stages):
            sub = df[(df.dir_name == dn) & (df.tag == tag)].sort_values("step")
            if not len(sub):
                continue
            ax.plot(sub.step / 1e6, sub.value, lw=1.05, color=SERIES[i],
                    label=lb if tag == "train/loss" else None)
        ax.set_title(name, fontsize=8.3)
        if tag == "train/loss":
            ax.legend(fontsize=5.9, ncol=2, loc="upper right", handlelength=1.2)
    for ax in axes[1]:
        ax.set_xlabel("environment steps [$\\times 10^6$]")
    fig.subplots_adjust(hspace=0.24, wspace=0.20)
    save(fig, "fig_rl_training")
fig_rl_training()

# ── RL recovery ablation ─────────────────────────────────────────────────────
def fig_rl_recovery():
    runs = [("fdir_brain_hpc_metrics.json", "base (deployable)"),
            ("fdir_brain_hpc_hc_metrics.json", "fine-tune A (hot)"),
            ("fdir_brain_hpc_hc2_metrics.json", "fine-tune B (gentle)")]
    CHS = ["NAV", "ATT", "COMM", "PROP", "FORM", "PEER"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.5),
                                 gridspec_kw=dict(width_ratios=[1.6, 1.0]))
    x = np.arange(len(CHS)); w = 0.26
    inj = None
    for i, (fn, lb) in enumerate(runs):
        m = jload(fn)
        vals = [m["recovery"][ch] for ch in CHS]
        inj = m["injected"]
        a1.bar(x + (i - 1) * w, vals, w, color=SERIES[i], label=lb, edgecolor="none")
    a1.set_xticks(x)
    a1.set_xticklabels([f"{ch}\n({inj[ch]})" for ch in CHS], fontsize=6.8)
    a1.set_ylabel("recovery success rate"); a1.set_ylim(0, 1.12)
    a1.legend(fontsize=6.6, loc="upper right")
    a1.set_title("(a) per-class recovery, 100 held-out missions", fontsize=8.3)
    ups = [jload(fn)["unsafe_proposals"] for fn, _ in runs]
    rta = [jload(fn)["rta_violations"] for fn, _ in runs]
    xx = np.arange(3)
    a2.bar(xx - 0.17, ups, 0.34, color=C["orange"], label="unsafe proposals\n(all vetoed)")
    a2.bar(xx + 0.17, rta, 0.34, color=C["green"], label="RTA violations")
    for xi, v in zip(xx, ups):
        a2.text(xi - 0.17, v * 1.12, f"{v:,}", ha="center", fontsize=6.4)
    for xi in xx:
        a2.text(xi + 0.17, 120, "0", ha="center", fontsize=7, color=C["green"], weight="bold")
    a2.set_yscale("log"); a2.set_ylim(80, 3e4)
    a2.set_xticks(xx); a2.set_xticklabels(["base", "A", "B"], fontsize=7.5)
    a2.legend(fontsize=6.2, loc="upper left")
    a2.set_title("(b) safety invariance", fontsize=8.3)
    save(fig, "fig_rl_recovery")
fig_rl_recovery()
print("FP+RL FIGS DONE")
