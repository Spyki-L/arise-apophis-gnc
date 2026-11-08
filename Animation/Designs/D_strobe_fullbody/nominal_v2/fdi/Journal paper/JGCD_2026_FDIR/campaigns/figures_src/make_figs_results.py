#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Result figures: nav errors, FDI nominal, MC campaigns, Byzantine, regimes."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import use_style, C, SERIES, FULL_W, HALF_W

R = Path(__file__).resolve().parent.parent / "results"
FIGS = Path(__file__).resolve().parent.parent / "figs"
use_style(serif=True)

def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf"); fig.savefig(FIGS / f"{name}.png", dpi=150)
    plt.close(fig); print("wrote", name)

def jload(fn):
    return json.load(open(R / fn))

# ── 1. nav position errors vs 3-sigma (two 4-deputy figures) ────────────────
nav = np.load(R / "nav_errors.npz")
def fig_nav(group, name):
    fig, axes = plt.subplots(4, 3, figsize=(FULL_W, 5.6), sharex=True)
    lab = ["$R$ [m]", "$T$ [m]", "$N$ [m]"]
    for r, d in enumerate(group):
        t = nav[f"t_D{d}"]; th = (t - t[0]) / 3600.0
        err = nav[f"err_D{d}"]; s3 = nav[f"sig3_D{d}"]
        for cidx in range(3):
            ax = axes[r, cidx]
            ax.fill_between(th, -s3[:, cidx], s3[:, cidx], color=C["sky"], alpha=0.35, lw=0)
            ax.plot(th, err[:, cidx], color=C["blue"], lw=0.55)
            ax.set_ylim(-2.2, 2.2)
            if cidx == 0:
                ax.set_ylabel(f"D{d}\n{lab[0]}", fontsize=8)
            if r == 0:
                ax.set_title(["Radial", "Along-track", "Cross-track"][cidx], fontsize=9)
            if r == 3:
                ax.set_xlabel("time since arc start [h]")
    fig.align_ylabels()
    fig.subplots_adjust(hspace=0.14, wspace=0.22)
    save(fig, name)
fig_nav([0, 1, 2, 3], "fig_nav_errors_a")
fig_nav([4, 5, 6, 7], "fig_nav_errors_b")

# ── 2. FDI nominal overview for D0 ───────────────────────────────────────────
H = np.load(R / "fdi_nominal_hist.npz")
t_tca = float(H["t_tca"][0])
def fig_overview(d=0):
    chans = [("nav", 1.5, 15.0), ("att", 1.0, 8.0), ("prop", 1.0, 3.0),
             ("form", 3.0, 30.0), ("peer", 3.0, 20.0)]
    t = H[f"t_D{d}"]; th = (t - t[0]) / 3600.0; tca_h = (t_tca - t[0]) / 3600.0
    fig, axes = plt.subplots(6, 2, figsize=(FULL_W, 6.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 1, 1, 1, 0.45]))
    for i, (ch, nu, h) in enumerate(chans):
        r = H[f"r_{ch}_D{d}"]; S = H[f"S_{ch}_D{d}"]
        axL, axR = axes[i, 0], axes[i, 1]
        axL.plot(th, r, lw=0.55, color=C["blue"])
        axL.axhline(nu, color=C["verm"], lw=0.9, ls="--")
        axL.set_ylabel(ch.upper(), fontsize=8)
        axR.plot(th, S, lw=0.8, color=C["green"])
        axR.axhline(h, color=C["verm"], lw=0.9, ls="--")
        axR.set_ylim(-0.04 * h, 1.15 * h)
        if ch in ("form", "peer"):
            for ax in (axL, axR):
                ax.axvspan(tca_h - 1.0, tca_h, color=C["yellow"], alpha=0.25, lw=0)
        for ax in (axL, axR):
            ax.axvline(tca_h, color="0.35", lw=0.7, ls=":")
        if i == 0:
            axL.set_title("residual $r$ (dashed: reference $\\nu$)", fontsize=8.5)
            axR.set_title("leaky CUSUM $S$ (dashed: threshold $h$)", fontsize=8.5)
    axS = axes[5, 0]; st = H[f"state_D{d}"]
    axS.step(th, st, lw=1.0, color="black")
    axS.set_ylim(0.5, 4.5); axS.set_yticks([1, 2, 3, 4])
    axS.set_yticklabels(["HEALTHY", "SUSPECT", "FAULTED", "DEAD"], fontsize=6.4)
    axS.set_xlabel("time since arc start [h]"); axes[5, 1].axis("off")
    axS.axvline(tca_h, color="0.35", lw=0.7, ls=":")
    fig.subplots_adjust(hspace=0.16, wspace=0.18)
    save(fig, "fig_fdi_overview_D0")
fig_overview(0)

# ── 3. fleet health bars ─────────────────────────────────────────────────────
pk = jload("fleet_peaks.json")
def fig_fleet():
    chans = ["NAV", "ATT", "PROP", "FORM", "PEER"]
    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    x = np.arange(8); w = 0.16
    for i, ch in enumerate(chans):
        vals = [pk[str(d)][ch] for d in range(8)]
        ax.bar(x + (i - 2) * w, vals, w, label=ch, color=SERIES[i], edgecolor="none")
    ax.axhline(1.0, color=C["verm"], lw=1.0, ls="--")
    ax.text(7.45, 1.03, "fault threshold", fontsize=7, color=C["verm"], ha="right")
    ax.set_xticks(x); ax.set_xticklabels([f"D{d}" for d in range(8)])
    ax.set_ylabel("peak $S/h$ per channel"); ax.set_ylim(0, 1.12)
    ax.legend(ncol=5, loc="upper left", columnspacing=1.1, handlelength=1.4)
    save(fig, "fig_fleet_health")
fig_fleet()

# ── 4. detection summary: Pd bars + delay CDFs ──────────────────────────────
def fig_detection():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.55),
                                 gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    SC = ["NAV", "ATT", "PROP", "COMM"]
    pd_, lo_, hi_ = [], [], []
    for sc in SC:
        d = jload(f"part_det_{sc}.json")
        pd_.append(d["Pd"]); lo_.append(d["Pd"] - d["ci"][0]); hi_.append(d["ci"][1] - d["Pd"])
    x = np.arange(4)
    a1.bar(x, pd_, 0.55, color=[SERIES[i] for i in range(4)], edgecolor="none")
    a1.errorbar(x, pd_, yerr=[lo_, hi_], fmt="none", ecolor="black", lw=1.0, capsize=2.5)
    pfa = jload("part_det_PFA.json")
    a1.set_xticks(x); a1.set_xticklabels(SC)
    a1.set_ylabel("detection probability $P_d$"); a1.set_ylim(0.9, 1.005)
    a1.set_title(f"$P_{{fa}}$ = {pfa['Pfa']:.3f} [{pfa['ci'][0]:.3f}, {pfa['ci'][1]:.3f}]"
                 " per deputy per encounter", fontsize=8)
    for i, sc in enumerate(SC):
        dl = np.load(R / f"delays_{sc}.npz")["delays"]
        dl = np.sort(dl[np.isfinite(dl)])
        a2.plot(dl, np.arange(1, dl.size + 1) / dl.size, lw=1.3, color=SERIES[i],
                label=f"{sc} (median {np.median(dl):,.0f} s)")
    a2.set_xscale("log"); a2.set_xlabel("detection delay after onset [s]")
    a2.set_ylabel("empirical CDF"); a2.set_ylim(0, 1.02)
    a2.axvline(7200, color="0.4", lw=0.8, ls=":")
    a2.text(6300, 0.06, "2 h imaging\nwindow ", fontsize=6.6, color="0.3", ha="right")
    a2.legend(loc="lower right", fontsize=6.8)
    save(fig, "fig_mc_detection")
fig_detection()

# ── 5. isolation confusion before/after ──────────────────────────────────────
def fig_confusion():
    LABr = ["NAV", "ATT", "PROP", "COMM"]; LABc = ["NONE", "NAV", "ATT", "PROP", "COMM"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.35))
    for ax, mode, title in zip(axes, ["legacy", "fixed"],
                               ["(a) legacy classifier", "(b) fixed classifier (PROP first-class + latch)"]):
        d = jload(f"part_iso_{mode}.json")["confusion"]; n = jload(f"part_iso_{mode}.json")["n"]
        Mx = np.zeros((4, 5))
        for i, sc in enumerate(LABr):
            for j, l in enumerate(LABc):
                Mx[i, j] = 100.0 * d[sc].get(l, 0) / n
        im = ax.imshow(Mx, cmap="Blues", vmin=0, vmax=100)
        for i in range(4):
            for j in range(5):
                if Mx[i, j] > 0:
                    ax.text(j, i, f"{Mx[i,j]:.1f}", ha="center", va="center", fontsize=7,
                            color="white" if Mx[i, j] > 55 else "black")
        ax.set_xticks(range(5)); ax.set_xticklabels(LABc, fontsize=7)
        ax.set_yticks(range(4)); ax.set_yticklabels(LABr, fontsize=7)
        ax.set_xlabel("identified label", fontsize=8); ax.grid(False)
        if mode == "legacy":
            ax.set_ylabel("injected fault", fontsize=8)
        ax.set_title(title, fontsize=8.3)
    save(fig, "fig_confusion")
fig_confusion()

# ── 6. detectability curves ──────────────────────────────────────────────────
def fig_magnitude():
    mags = {k: jload(f"part_mag_{k}.json")["rows"] for k in ("NAV", "ATT", "PROP")}
    ref = jload("part_mag_refine.json")
    mags["NAV"] = sorted(mags["NAV"] + ref["NAV"], key=lambda r: r["mag"])
    mags["ATT"] = sorted(mags["ATT"] + ref["ATT"], key=lambda r: r["mag"])
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.3))
    xl = ["NAV innovation bias [$\\sigma$]", "ATT attitude bias [deg]", "PROP $\\Delta v$ mismatch [$\\sigma$]"]
    for ax, (ch, rows), xlab in zip(axes, mags.items(), xl):
        m = [r["mag"] for r in rows]; p = [r["Pd"] for r in rows]
        lo = [r["Pd"] - r["ci"][0] for r in rows]; hi = [r["ci"][1] - r["Pd"] for r in rows]
        ax.errorbar(m, p, yerr=[lo, hi], fmt="o-", ms=3, lw=1.2, color=C["blue"],
                    ecolor="0.55", elinewidth=0.7, capsize=1.8)
        ax.set_xlabel(xlab, fontsize=7.8); ax.set_ylim(-0.05, 1.08)
        if ch == "NAV":
            ax.set_ylabel("$P_d$")
        ax2 = ax.twinx()
        md = [r["lat_median"] for r in rows]
        ax2.plot(m, md, "s--", ms=2.6, lw=0.9, color=C["verm"])
        ax2.set_yscale("log"); ax2.grid(False)
        ax2.tick_params(axis="y", colors=C["verm"], labelsize=6.4)
        if ch == "PROP":
            ax2.set_ylabel("median delay [s]", color=C["verm"], fontsize=7.5)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig_magnitude")
fig_magnitude()

# ── 7. Byzantine sweep ───────────────────────────────────────────────────────
def fig_byz():
    data = {}
    for fn in ("byz_N4_6_8.json", "byz_N12_16.json", "byz_N24_32.json"):
        data.update(jload(fn)["res"])
    fig, ax = plt.subplots(figsize=(HALF_W + 0.6, 2.5))
    for i, (N, per_f) in enumerate(sorted(data.items(), key=lambda kv: int(kv[0]))):
        N = int(N)
        fs = sorted(int(f) for f in per_f)
        x = [f / N for f in fs]; y = [per_f[str(f)]["Pfa"] for f in fs]
        ax.plot(x, y, "o-", ms=3, lw=1.15, color=SERIES[i], label=f"$N$={N}", drawstyle="steps-post")
    ax.axvline(0.5, color="0.3", lw=1.0, ls="--")
    ax.text(0.505, 0.45, "$f=N/2$\n(median bound)", fontsize=7, ha="left")
    ax.set_xlabel("Byzantine fraction $f/N$")
    ax.set_ylabel("honest-deputy false-alarm rate")
    ax.set_xlim(-0.02, 0.62); ax.set_ylim(-0.04, 1.04)
    ax.legend(fontsize=6.8, ncol=2)
    save(fig, "fig_byzantine")
fig_byz()

# ── 8. cross-regime transfer + sensor sweep ─────────────────────────────────
def fig_regimes():
    reg = {k: jload(f"regime_{k}.json")["res"] for k in ("LEO", "GEO", "NRHO")}
    apo = {sc: jload(f"part_det_{sc}.json") for sc in ("NAV", "ATT", "PROP", "COMM")}
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.45), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1.0]))
    SC = ["NAV", "ATT", "PROP", "COMM"]
    names = ["Apophis", "LEO", "GEO", "NRHO"]
    a0, a1, a2 = axes
    x = np.arange(4); w = 0.19
    for i, nm in enumerate(names):
        if nm == "Apophis":
            pd_ = [apo[sc]["Pd"] for sc in SC]
        else:
            pd_ = [reg[nm][sc]["Pd"] for sc in SC]
        a0.bar(x + (i - 1.5) * w, pd_, w, color=SERIES[i], label=nm, edgecolor="none")
    a0.set_xticks(x); a0.set_xticklabels(SC, fontsize=7.5)
    a0.set_ylabel("$P_d$"); a0.set_ylim(0, 1.13)
    a0.legend(ncol=2, fontsize=6.4, loc="lower left")
    a0.set_title("(a) detection probability", fontsize=8.3)
    for i, nm in enumerate(names):
        if nm == "Apophis":
            lat = [apo[sc]["lat_median"] for sc in SC]
        else:
            lat = [reg[nm][sc]["lat_median"] for sc in SC]
        a1.bar(x + (i - 1.5) * w, lat, w, color=SERIES[i], edgecolor="none")
    a1.set_yscale("log"); a1.set_xticks(x); a1.set_xticklabels(SC, fontsize=7.5)
    a1.set_ylabel("median delay [s]")
    a1.set_title("(b) detection latency", fontsize=8.3)
    sen = jload("regime_SENSOR.json")["res"]
    ks = [1, 2, 5, 10]
    pd_ = [sen[f"x{k}"]["NAV"]["Pd"] for k in ks]
    lat = [sen[f"x{k}"]["NAV"]["lat_median"] for k in ks]
    a2.plot(ks, pd_, "o-", color=C["blue"], lw=1.2, ms=3.5, label="$P_d$")
    a2.set_xlabel("sensor noise scale"); a2.set_ylabel("$P_d$ (fixed-size NAV fault)")
    a2.set_ylim(-0.05, 1.1); a2.set_xscale("log"); a2.set_xticks(ks)
    a2.set_xticklabels([f"{k}x" for k in ks])
    a2.set_title("(c) sensor-grade sweep", fontsize=8.3)
    at = a2.twinx()
    at.plot(ks[:2], lat[:2], "s--", color=C["verm"], lw=1.0, ms=3)
    at.grid(False); at.tick_params(axis="y", colors=C["verm"], labelsize=6.4)
    at.set_ylabel("median delay [s]", color=C["verm"], fontsize=7.5)
    fig.subplots_adjust(wspace=0.46)
    save(fig, "fig_regimes")
fig_regimes()
print("ALL RESULT FIGS DONE")
