#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Architecture + concept figures in the ARISE block-diagram house style."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse, Circle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (use_style, draw_box, arrow, alabel, blank_axes,
                       SALMON, SALMON_EDGE, GREYBOX, BOX_EDGE, C, FULL_W)

FIGS = Path(__file__).resolve().parent.parent / "figs"
FIGS.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=170)
    plt.close(fig)
    print("wrote", name)


# ══════════════════════════════════════════════════════════════════════════
# Figure: deputy GNC stack with FDI taps (house style)
# ══════════════════════════════════════════════════════════════════════════
def fig_gnc_arch():
    use_style(serif=False)
    fig, ax = plt.subplots(figsize=(FULL_W, 4.35))
    blank_axes(ax, xlim=(0, 1), ylim=(0.14, 1.0))
    FS = 6.1     # module text
    FT = 6.6     # container titles
    FL = 5.6     # arrow labels

    # ══ high-rate bus container (top-left) ══
    draw_box(ax, 0.012, 0.615, 0.575, 0.375, "", fc="white", lw=1.2)
    alabel(ax, 0.030, 0.962, "HIGH-RATE (50–100 Hz) ACTUATION & PROPAGATION BUS",
           fs=FT, ha="left", va="center")
    draw_box(ax, 0.035, 0.795, 0.150, 0.105, "Dynamics core\nrigid-body prop.\n(RKF)", fc=GREYBOX, fs=FS)
    draw_box(ax, 0.245, 0.795, 0.175, 0.105, "Disturbance / env.\nmodel (gravity\nharmonics, SRP)", fc=GREYBOX, fs=FS)
    draw_box(ax, 0.035, 0.640, 0.150, 0.105, "Thruster bank\n+ RW assembly", fc=GREYBOX, fs=FS)
    draw_box(ax, 0.245, 0.640, 0.175, 0.105, "Prop-sequencer\n(burn timing, min\nimpulse, cooldowns)", fc=GREYBOX, fs=FS)
    draw_box(ax, 0.455, 0.640, 0.115, 0.260, "Ctrl / wrench\nallocator\n+ control\nmode gate", fc=SALMON, ec=SALMON_EDGE, fs=FS)
    arrow(ax, (0.185, 0.848), (0.245, 0.848))
    alabel(ax, 0.215, 0.856, "states", fs=FL)
    arrow(ax, (0.3325, 0.745), (0.3325, 0.795))
    alabel(ax, 0.340, 0.765, "ext. torque", fs=FL, ha="left")
    arrow(ax, (0.245, 0.6925), (0.185, 0.6925))
    alabel(ax, 0.215, 0.700, "thrust", fs=FL)
    arrow(ax, (0.455, 0.6925), (0.420, 0.6925))
    alabel(ax, 0.437, 0.700, "torque", fs=FL)
    arrow(ax, (0.420, 0.848), (0.455, 0.848))

    # ══ estimation bus container (bottom-left) ══
    draw_box(ax, 0.012, 0.175, 0.575, 0.385, "", fc=SALMON, ec=SALMON_EDGE, lw=1.2)
    alabel(ax, 0.030, 0.532, "MID-RATE ESTIMATION BUS (10 Hz)", fs=FT, ha="left", va="center")
    draw_box(ax, 0.030, 0.345, 0.150, 0.150, "Sensor suite\n(IMU, star tracker,\nsun vector, LOS,\nranging beacon)", fs=FS)
    draw_box(ax, 0.215, 0.370, 0.115, 0.100, "Sensor mixer\n(align, gate,\nscale)", fs=FS)
    draw_box(ax, 0.365, 0.370, 0.125, 0.100, "Hill-frame EKF\n(rel. nav +\nattitude est.)", fs=FS)
    draw_box(ax, 0.030, 0.195, 0.185, 0.115, "Attitude control\nstack (MRP PID\n+ torque LPF)", fs=FS)
    draw_box(ax, 0.315, 0.195, 0.255, 0.130,
             "FDI CORE (deputy-centric)\nNAV | ATT | PROP | COMM |\nFORM | PEER",
             fc=SALMON, ec=SALMON_EDGE, lw=1.35, fs=FS, weight="bold")
    arrow(ax, (0.180, 0.420), (0.215, 0.420))
    arrow(ax, (0.330, 0.420), (0.365, 0.420))
    arrow(ax, (0.4275, 0.370), (0.4275, 0.325))
    alabel(ax, 0.435, 0.347, "innovations $y,S$; burn recon.", fs=FL, ha="left")
    arrow(ax, (0.215, 0.2525), (0.315, 0.2525))
    alabel(ax, 0.265, 0.260, "$\\sigma_{BR},\\,\\omega_{BR}$", fs=FL)

    # bus-to-bus links
    arrow(ax, (0.100, 0.615), (0.100, 0.495), style="<|-|>")
    alabel(ax, 0.092, 0.555, "s/c states", fs=FL, rot=90, ha="center", va="center")
    arrow(ax, (0.5125, 0.560), (0.5125, 0.640), style="-|>")
    alabel(ax, 0.520, 0.598, "torque desired", fs=FL, rot=90, ha="left", va="center")

    # ══ guidance / planning container (top-right) ══
    draw_box(ax, 0.645, 0.615, 0.345, 0.375, "", fc=SALMON, ec=SALMON_EDGE, lw=1.2)
    alabel(ax, 0.8175, 0.962, "GUIDANCE / PLANNING LAYER (1 Hz)", fs=FT)
    draw_box(ax, 0.663, 0.790, 0.310, 0.140,
             "Phase manager (modes + guards)\nTransfer + relative planner (RH-IG)\n"
             "Stereo-geometry optimizer (GA+BO)\nFormation safety checker (CBF)",
             fs=FS, align="left")
    draw_box(ax, 0.663, 0.640, 0.310, 0.105,
             "Timeline broadcaster\n(time tags, cues, burn windows,\nFDIR flags to the fleet)",
             fc="white", fs=FS)
    arrow(ax, (0.8175, 0.790), (0.8175, 0.745))
    alabel(ax, 0.825, 0.766, "$\\Delta v$ table", fs=FL, ha="left")

    # ══ recovery container (bottom-right) ══
    draw_box(ax, 0.645, 0.175, 0.345, 0.385, "", fc="white", lw=1.2)
    alabel(ax, 0.8175, 0.532, "ONBOARD RECOVERY (CORTEX)", fs=FT)
    draw_box(ax, 0.663, 0.360, 0.310, 0.130,
             "RL recovery policy $\\pi_\\theta$ (PPO,\nshared MLP), 9 discrete\nrecovery actions",
             fc=SALMON, ec=SALMON_EDGE, fs=FS)
    draw_box(ax, 0.663, 0.195, 0.310, 0.115,
             "RTA safety gate (CBF check +\nVSC fallback simplex)\n+ executive",
             fs=FS)
    arrow(ax, (0.8175, 0.360), (0.8175, 0.310))
    alabel(ax, 0.825, 0.333, "proposed action", fs=FL, ha="left")

    # ══ cross links ══
    arrow(ax, (0.570, 0.260), (0.663, 0.260))
    alabel(ax, 0.616, 0.268, "health $h_i$", fs=FL)
    arrow(ax, (0.570, 0.290), (0.615, 0.290), style="-")
    arrow(ax, (0.615, 0.290), (0.615, 0.425), style="-")
    arrow(ax, (0.615, 0.425), (0.663, 0.425))
    arrow(ax, (0.8175, 0.560), (0.8175, 0.615))
    alabel(ax, 0.825, 0.585, "retask / replan", fs=FL, ha="left")
    arrow(ax, (0.645, 0.6925), (0.587, 0.6925), style="-|>")
    alabel(ax, 0.616, 0.700, "burn cues", fs=FL)
    save(fig, "fig_gnc_arch")


# ══════════════════════════════════════════════════════════════════════════
# Figure: FDI core pipeline (house style, vertical)
# ══════════════════════════════════════════════════════════════════════════
def fig_fdi_core():
    use_style(serif=False)
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    blank_axes(ax)

    draw_box(ax, 0.03, 0.025, 0.94, 0.955, "", fc=SALMON, ec=SALMON_EDGE, lw=1.4)
    draw_box(ax, 0.10, 0.915, 0.80, 0.052,
             "FDI CORE  (deputy-centric, filter-agnostic)   ~10 Hz",
             fc="white", fs=8.0, weight="bold")

    draw_box(ax, 0.10, 0.72, 0.80, 0.165,
             "Inputs (Estimation-bus + telemetry taps)\n"
             "NAV: innovations $y_k$, covariance $S_k$ (EKF-gated samples skipped)\n"
             "ATT: $\\sigma_{BR}$, $\\omega_{BR}$  ·  PROP: $\\Delta v_{cmd}$ vs $\\Delta\\hat{v}$, $P_{vv}$\n"
             "COMM: heartbeat age  ·  FORM: predicted terminal miss (CW)\n"
             "PEER: causal consensus outlier score (median/MAD, floored)",
             fc="white", fs=6.8, align="left")

    draw_box(ax, 0.10, 0.595, 0.80, 0.075,
             "Residual builder (normalize $\\rightarrow$ scalars)\n"
             "$r_{nav},\\ r_{att},\\ r_{prop},\\ r_{comm},\\ r_{form},\\ r_{peer}$",
             fc="white", fs=7.0)

    draw_box(ax, 0.10, 0.435, 0.80, 0.11,
             "Detector bank\n"
             "one-sided leaky CUSUM: NAV / ATT / COMM / FORM / PEER\n"
             "leaky burn-gated CUSUM: PROP (evidence persists between burns)",
             fc="white", fs=7.0)

    draw_box(ax, 0.10, 0.25, 0.80, 0.135,
             "Classifier (persistence state machine + per-episode latch)\n"
             "HEALTHY $\\rightarrow$ SUSPECT $\\rightarrow$ FAULTED  (DEAD on sustained comm loss)\n"
             "settled label = precedence over latched channel set:\n"
             "FORM$\\wedge$PEER$\\rightarrow$FORM  ·  PEER only$\\rightarrow$PEER  ·  FORM only$\\rightarrow$NAV",
             fc="white", fs=6.8)

    draw_box(ax, 0.10, 0.095, 0.80, 0.10,
             "Output: health vector $h_i$ = (state, fault label, confidence)\n"
             "+ burn-gate / replan hints",
             fc="white", fs=7.2)

    for y0, y1 in ((0.72, 0.67), (0.595, 0.545), (0.435, 0.385), (0.25, 0.195)):
        arrow(ax, (0.5, y0), (0.5, y1), lw=1.2)

    alabel(ax, 0.5, 0.052,
           "$h_i \\rightarrow$ guidance / recovery policy / timeline broadcaster (FDIR flags to fleet)",
           fs=6.8)
    save(fig, "fig_fdi_core")


# ══════════════════════════════════════════════════════════════════════════
# Figure: CORTEX recovery layer (RL + RTA simplex)
# ══════════════════════════════════════════════════════════════════════════
def fig_cortex():
    use_style(serif=False)
    fig, ax = plt.subplots(figsize=(FULL_W, 3.3))
    blank_axes(ax)
    FS = 6.4

    draw_box(ax, 0.015, 0.66, 0.205, 0.27,
             "FDI core output\n$h_i$ = (state, label,\nconf.) + channel $S/h$",
             fc=SALMON, ec=SALMON_EDGE, fs=FS)
    draw_box(ax, 0.015, 0.24, 0.205, 0.33,
             "Deputy telemetry\n(authority, time-to-go,\noff-station distance,\nneighbour health)",
             fs=FS)
    draw_box(ax, 0.25, 0.42, 0.15, 0.28,
             "Observation\nbuilder (28-dim\nper deputy)", fs=FS)
    draw_box(ax, 0.43, 0.42, 0.23, 0.30,
             "Recovery policy $\\pi_\\theta$\nshared PPO MLP 2$\\times$128\n($\\approx$42k params)\n9 discrete actions",
             fc=SALMON, ec=SALMON_EDGE, fs=FS)
    draw_box(ax, 0.755, 0.42, 0.235, 0.28,
             "Run-time assurance gate\nCBF admissibility check\naccept, else VSC fallback\n(simplex; policy advisory)",
             fs=FS)
    arrow(ax, (0.22, 0.78), (0.25, 0.62), connstyle="arc3,rad=-0.18")
    arrow(ax, (0.22, 0.40), (0.25, 0.50), connstyle="arc3,rad=0.14")
    arrow(ax, (0.40, 0.56), (0.43, 0.56))
    arrow(ax, (0.66, 0.56), (0.755, 0.56))
    alabel(ax, 0.7075, 0.585, "proposal", fs=5.8)
    arrow(ax, (0.8725, 0.42), (0.8725, 0.27))
    alabel(ax, 0.880, 0.345, "safe action", fs=5.8, ha="left")
    draw_box(ax, 0.755, 0.10, 0.235, 0.17,
             "Executive / behaviour tree\n$\\rightarrow$ planner + actuators", fs=FS)
    alabel(ax, 0.8725, 0.735, "0 RTA violations in every evaluation", fs=5.9)

    # offline training loop (dashed container)
    draw_box(ax, 0.245, 0.045, 0.475, 0.30, "", dashed=True, lw=1.0)
    alabel(ax, 0.4825, 0.310, "offline training loop (HPC)", fs=6.2)
    draw_box(ax, 0.263, 0.085, 0.205, 0.18,
             "Replay backend over\n200-draw Basilisk MC\n+ fault library", fs=5.9)
    draw_box(ax, 0.497, 0.085, 0.205, 0.18,
             "4-stage curriculum:\nsevere / nominal /\nincipient / adversarial", fs=5.9)
    arrow(ax, (0.468, 0.175), (0.497, 0.175))
    arrow(ax, (0.545, 0.345), (0.545, 0.42), dashed=True)
    alabel(ax, 0.555, 0.375, "20M steps, 88 workers, 11.5 h", fs=5.8, ha="left")
    save(fig, "fig_cortex")


# ══════════════════════════════════════════════════════════════════════════
# Figure: two-panel concept (geometry + detection idea)
# ══════════════════════════════════════════════════════════════════════════
def fig_concept():
    use_style(serif=True)
    fig = plt.figure(figsize=(FULL_W, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.14)

    # (a) geometry ---------------------------------------------------------
    ax = blank_axes(fig.add_subplot(gs[0]), xlim=(-1.34, 1.42), ylim=(-1.10, 1.04))
    th = np.linspace(-0.9, 0.75, 100)
    ax.plot(1.9 * np.sin(th) - 0.3, -0.75 + 0.3 * np.cosh(th), color="0.55", lw=1.4)
    ax.text(-1.05, -0.35, "chief trajectory", fontsize=7, color="0.4", rotation=-4)
    cx, cy = -0.3 + 1.9 * np.sin(0.12), -0.75 + 0.3 * np.cosh(0.12)
    angs = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for i, a in enumerate(angs):
        ex, ey = 0.52 * np.cos(a + 0.3), 0.33 * np.sin(a + 0.3)
        e = Ellipse((cx, cy), 2 * abs(ex) + 0.25, 2 * abs(ey) + 0.16,
                    angle=np.degrees(a) / 3.2, fill=False, ec="0.8", lw=0.55)
        ax.add_patch(e)
    dep_pos = []
    for i, a in enumerate(angs):
        px = cx + (0.42 + 0.06 * (i % 3)) * np.cos(a)
        py = cy + (0.28 + 0.05 * (i % 2)) * np.sin(a)
        dep_pos.append((px, py))
    for i, (px, py) in enumerate(dep_pos):
        col = C["verm"] if i == 5 else C["blue"]
        ax.add_patch(Circle((px, py), 0.028, fc=col, ec="none", zorder=5))
    ax.add_patch(Circle((cx, cy), 0.042, fc="black", zorder=6))
    ax.text(cx + 0.055, cy - 0.115, "$C$ (chief)", fontsize=8)
    fx, fy = dep_pos[5]
    arrow(ax, (fx, fy), (fx + 0.06, fy - 0.20), color=C["verm"], lw=1.3)
    ax.text(fx + 0.13, fy - 0.29, "faulted deputy\ndrifts off station", fontsize=7,
            color=C["verm"], ha="center")
    ax.text(cx - 0.06, cy + 0.72, "deputy relative orbits\n(hundreds of metres)",
            fontsize=7, ha="center", color="0.35")
    # Apophis
    ax.add_patch(Circle((1.08, 0.58), 0.085, fc="0.75", ec="0.4", lw=0.8))
    ax.text(1.08, 0.40, "(99942) Apophis", fontsize=7.5, ha="center")
    arrow(ax, (0.84, 0.82), (1.30, 0.62), color="black", lw=1.2, style="-|>")
    ax.text(1.02, 0.92, "$v_{rel}=8.76$ km/s", fontsize=7.5, ha="center")
    # LVLH triad
    ox, oy = -1.18, -0.94
    arrow(ax, (ox, oy), (ox + 0.22, oy), color=C["green"], lw=1.1)
    arrow(ax, (ox, oy), (ox, oy + 0.22), color=C["green"], lw=1.1)
    ax.text(ox + 0.26, oy - 0.01, "$\\hat{T}$", fontsize=8, color=C["green"])
    ax.text(ox - 0.02, oy + 0.26, "$\\hat{R}$", fontsize=8, color=C["green"])
    ax.text(-1.30, 0.92, "(a)  swarm at the close flyby", fontsize=8.5, weight="bold")

    # (b) detection idea ---------------------------------------------------
    ax = blank_axes(fig.add_subplot(gs[1]), xlim=(0, 1), ylim=(0, 1))
    ax.text(0.01, 0.965, "(b)  six-channel detection on every deputy",
            fontsize=8.5, weight="bold")
    ax.text(0.31, 0.895, "residual $r(t)$", fontsize=7, ha="center", color="0.25")
    ax.text(0.61, 0.895, "leaky CUSUM $S$ vs $h$", fontsize=7, ha="center", color="0.25")
    chans = ["NAV", "ATT", "PROP", "COMM", "FORM", "PEER"]
    y0 = 0.785
    step = 0.118
    rngb = np.random.default_rng(5)
    for i, ch in enumerate(chans):
        y = y0 - i * step
        ax.text(0.045, y, ch, fontsize=7.2, va="center", ha="left",
                color=SALMON_EDGE if ch in ("FORM", "PEER") else "black",
                weight="bold" if ch in ("FORM", "PEER") else "normal")
        xx = np.linspace(0.175, 0.45, 60)
        base = 0.0125 * rngb.standard_normal(60)
        if ch == "NAV":
            base[38:] += np.linspace(0, 0.032, 22)
        ax.plot(xx, y + base, lw=0.75, color=C["blue"])
        # CUSUM sparkline
        xx2 = np.linspace(0.52, 0.70, 60)
        s = np.zeros(60)
        drive = base * 2.2
        for k in range(1, 60):
            s[k] = max(0.0, 0.95 * s[k - 1] + drive[k])
        s = 0.42 * s / max(s.max(), 1e-6) * (0.075 if ch == "NAV" else 0.024)
        ax.plot(xx2, y - 0.020 + s, lw=0.85, color=C["verm"])
        ax.plot([0.52, 0.70], [y + 0.034] * 2, lw=0.6, ls=":", color="0.5")
    draw_box(ax, 0.785, 0.355, 0.205, 0.435,
             "FSM +\nepisode latch\n\nHEALTHY\n$\\downarrow$\nSUSPECT\n$\\downarrow$\nFAULTED",
             fc="white", ec=BOX_EDGE, fs=6.6)
    arrow(ax, (0.72, 0.572), (0.785, 0.572), lw=1.0)
    draw_box(ax, 0.785, 0.085, 0.205, 0.195,
             "$h_i$ = (state,\nlabel, conf.)", fc=SALMON, ec=SALMON_EDGE, fs=7.0)
    arrow(ax, (0.887, 0.355), (0.887, 0.28), lw=1.0)
    ax.text(0.315, 0.045, "FORM: predicted miss at fixed TCA\nPEER: causal fleet-consensus outlier",
            fontsize=6.8, ha="center", color=SALMON_EDGE)
    save(fig, "fig_concept")

if __name__ == "__main__":
    fig_gnc_arch()
    fig_fdi_core()
    fig_cortex()
    fig_concept()
