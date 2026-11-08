#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faultlab_loss.py — W9: deputy-loss compensation (ΔV + torque hand-off).

When a deputy is lost (comm-dead, or faulted out of imaging), the survivors take up a
BIT of its share — not fully. This window shows the live plan: which healthy deputies
nudge their CA station toward the gap (a ΔV) and re-point their camera (a torque / RW
momentum bite), and how much of the lost deputies' footprint that recovers. A schematic
shows the gap and the compensating nudges; bars show the per-helper ΔV and torque; the
summary shows the fleet budget spent and the coverage regained. matplotlib only.
"""
from __future__ import annotations
import numpy as np
import numpy.linalg as la

from cockpit_ops import STATE_COLORS, NEON, GREEN, RED, AMBER, TXT, AXIS_RGBA, _glow, style2d
from faultlab_encounter import compute_compensation


class LossCompensationWindow:
    """W9 — survivors' partial ΔV + torque hand-off for lost deputies."""

    def __init__(self, engine, refresh=6):
        import matplotlib.pyplot as plt
        self.plt = plt; self.engine = engine; self.N = engine.N; self.refresh = refresh
        self._cmp = None; self._last = -999; self._lost_key = None
        self.fig = plt.figure(figsize=(14.6, 8.5)); self.fig.patch.set_facecolor("black")
        self.ax_s = self.fig.add_axes([0.03, 0.07, 0.43, 0.80]); self.ax_s.axis("off")
        self.ax_dv = self.fig.add_axes([0.55, 0.55, 0.42, 0.30])
        self.ax_sum = self.fig.add_axes([0.55, 0.07, 0.42, 0.38]); self.ax_sum.axis("off")
        self.fig.suptitle("ARISE / APOPHIS  ·  DEPUTY-LOSS COMPENSATION  —  survivors' ΔV + torque hand-off",
                          color=NEON, fontsize=13.5, weight="bold", y=0.975)

    def _comp(self, st):
        lost = tuple(sorted(d for d in range(self.N) if st["health"][d] in ("DEAD", "FAULTED")))
        if self._cmp is None or lost != self._lost_key or (st["frame"] - self._last) >= self.refresh:
            stations = {d: st["ca_all"][d] for d in range(self.N) if st["health"][d] != "DEAD"}
            self._cmp = compute_compensation(self.engine.tr, stations, st["health"], self.engine.offset,
                                             self.engine.I_dep, st["t_go"]) if stations else None
            self._last = st["frame"]; self._lost_key = lost
        return self._cmp

    def _project(self, st):
        pts = np.array([st["ca_all"][d] for d in range(self.N)])
        c = pts.mean(0); X = pts - c
        try:
            _, _, Vt = la.svd(X)
            return X @ Vt[:2].T
        except la.LinAlgError:
            return X[:, :2]

    def update(self, st):
        plt = self.plt; cmp = self._comp(st); P = self._project(st)
        helpers = {r["helper"]: r for p in (cmp["plans"] if cmp else []) for r in p["helpers"]}
        # ── schematic ────────────────────────────────────────────────────────
        ax = self.ax_s; ax.clear(); ax.axis("off")
        lim = np.abs(P).max() * 1.25 + 1
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_title("FORMATION — gap & nudges", color=NEON, fontsize=9.5, loc="left")
        ax.scatter([0], [0], marker="*", s=240, color="#cc9966", ec="white", zorder=2)
        ax.text(0, -lim*0.07, "APOPHIS", color="#cc9966", fontsize=8, ha="center")
        lost = cmp["lost"] if cmp else []
        gap_c = P[lost].mean(0) if lost else np.zeros(2)
        for d in range(self.N):
            stt = st["health"][d]; col = STATE_COLORS[stt]; p = P[d]
            if d in lost:
                ax.scatter([p[0]], [p[1]], marker="x", s=160, color=RED, lw=3, zorder=4)
                ax.text(p[0], p[1] + lim*0.06, f"D{d+1} LOST", color=RED, fontsize=7, ha="center", weight="bold")
            else:
                ax.scatter([p[0]], [p[1]], marker="s", s=90, color=col, ec="white", lw=0.5, zorder=4)
                ax.text(p[0], p[1] - lim*0.07, f"D{d+1}", color=col, fontsize=6.5, ha="center")
            if d in helpers:                                  # nudge arrow toward the gap, length ~ ΔV
                r = helpers[d]; v = (gap_c - p); v = v / (la.norm(v) + 1e-9)
                L = lim * 0.18 * (0.4 + min(r["dv"] / 0.5, 1.0))
                ax.annotate("", xy=(p[0] + v[0]*L, p[1] + v[1]*L), xytext=(p[0], p[1]),
                            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.2), zorder=5)
                ax.text(p[0] + v[0]*L*1.2, p[1] + v[1]*L*1.2, f"{r['dv']:.2f} m/s",
                        color=GREEN, fontsize=6.8, ha="center")
        if lost:
            ax.scatter([gap_c[0]], [gap_c[1]], marker="o", s=260, facecolor="none", ec=RED, lw=1.5, ls=":", zorder=3)
            ax.text(gap_c[0], gap_c[1] + lim*0.10, "GAP", color=RED, fontsize=8, ha="center")
        # ── per-helper ΔV + torque ───────────────────────────────────────────
        ax = self.ax_dv; ax.clear(); style2d(ax, "", "", "")
        ax.set_title("HELPER ΔV / TORQUE", color=NEON, fontsize=8, loc="left", pad=2)
        if helpers:
            hs = sorted(helpers); x = np.arange(len(hs)); w = 0.38
            dv = [helpers[h]["dv"] for h in hs]; tq = [helpers[h]["torque"] * 1e6 for h in hs]
            ax.bar(x - w/2, dv, w, color=NEON, label="ΔV [m/s]")
            ax.bar(x + w/2, tq, w, color=ORANGE2, label="torque [µN·m]")
            ax.set_xticks(x); ax.set_xticklabels([f"D{h+1}" for h in hs], color=TXT, fontsize=6.5)
            ax.tick_params(axis="y", labelsize=6)
            ax.legend(fontsize=5.5, facecolor="black", labelcolor="white", edgecolor=AXIS_RGBA,
                      loc="upper left", framealpha=0.6)
            for i, h in enumerate(hs):
                ax.text(i, max(dv[i], tq[i]) * 1.02, f"{helpers[h]['dtheta_deg']:.0f}°", ha="center",
                        color=AMBER, fontsize=5.5, clip_on=True)
        else:
            ax.text(0.5, 0.5, "fleet nominal — no compensation needed", transform=ax.transAxes,
                    ha="center", color=(.6, .6, .6), fontsize=9, clip_on=True); ax.axis("off")
        # ── summary ──────────────────────────────────────────────────────────
        ax = self.ax_sum; ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if cmp:
            # compact 4-row summary sized for the short W9 tile (no stacked-text overlap)
            ax.text(0.03, 0.96, "COMPENSATION SUMMARY", color=NEON, fontsize=8, weight="bold", va="top")
            rows = [("lost", ", ".join(f"D{d+1}" for d in cmp["lost"]) or "—"),
                    ("helpers", str(cmp["n_helpers"])),
                    ("total ΔV", f"{cmp['total_dv']:.2f} m/s"),
                    ("horizon", f"{cmp['T_man']/60:.0f} min")]
            y = 0.80
            for lab, val in rows:
                ax.text(0.03, y, lab, color=TXT, fontsize=7, va="center")
                ax.text(0.97, y, val, color="white", fontsize=7.5, ha="right", va="center",
                        weight="bold", clip_on=True); y -= 0.145
            # gap coverage recovered bar
            b0 = cmp["gap_cov_before"]; b1 = cmp["gap_cov_after"]
            ax.add_patch(plt.Rectangle((0.03, 0.10), 0.94, 0.075, fill=False, ec=AXIS_RGBA, lw=1))
            ax.add_patch(plt.Rectangle((0.03, 0.10), 0.94 * b0, 0.075, color=(0.4, 0.4, 0.5)))
            ax.add_patch(plt.Rectangle((0.03 + 0.94 * b0, 0.10), 0.94 * max(b1 - b0, 0), 0.075, color=GREEN))
            ax.text(0.5, 0.025, f"lost-region coverage {b0*100:.0f}% → {b1*100:.0f}%",
                    ha="center", va="center", color=GREEN, fontsize=7.5, weight="bold")
        else:
            ax.text(0.5, 0.5, "FLEET NOMINAL\nno deputy lost — no compensation needed",
                    ha="center", va="center", color=GREEN, fontsize=10, weight="bold")


ORANGE2 = "#ff9a33"
