#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faultlab_innovations.py — W7: the LIVE innovation monitor for the NAV and ATT EKFs.

The FDI consumes only the onboard estimators' output. This window shows their
INNOVATION sequences (measurement − prediction, normalized by their own covariance √S,
so the consistency bound is always ±1σ / ±3σ). A fault drives the innovation out of
bounds — the true filter-level fault signature, seen before the CUSUM finishes.

  * NAV — relative-navigation innovation y_k (3 comp) + NIS, from the CW EKF.
  * ATT — attitude-error innovation σ_BR and body-rate innovation ω_BR, from the att EKF.

The estimators run at the mission's PHASE rate (separation & encounter sample fast,
coast slow) so the sample stream is dense in the critical phases — shown as phase bands
behind the traces, with the TCM burns marked. A truth-vs-estimate strip proves the
estimate tracks truth (small error) even while the TRUE state was driven off by the
fault — i.e. the FDI caught it from the estimate alone. matplotlib only.
"""
from __future__ import annotations
from collections import deque
import numpy as np

from cockpit_ops import STATE_COLORS, NEON, GREEN, RED, AMBER, TXT, AXIS_RGBA, _glow, style2d

COMP = ["#33bbff", "#00e0c6", "#ffcc00"]
NAV_LBL = ["y₁", "y₂", "y₃"]; SIG_LBL = ["roll", "pitch", "yaw"]; OM_LBL = ["p", "q", "r"]
PHASE_COL = {"SEPARATION": "#ff7a33", "COAST": "#2a4a6a", "APPROACH": "#9b6bff", "ENCOUNTER": "#ff4488"}
YLIM = 6.0; HIST = 320


class InnovationMonitorWindow:
    """W7 — live normalized NAV & ATT innovation sequences (phase-paced) with ±1σ/±3σ + validation."""

    def __init__(self, engine):
        import matplotlib.pyplot as plt
        self.plt = plt; self.engine = engine; self.N = engine.N
        # per-deputy stream of (nav3, st3, gy3, phase, frame, tcm)
        self.hist = {d: deque(maxlen=HIST) for d in range(self.N)}
        self.fig = plt.figure(figsize=(14.8, 8.8)); self.fig.patch.set_facecolor("black")
        self.ax_nav = self.fig.add_axes([0.065, 0.640, 0.62, 0.165])
        self.ax_ats = self.fig.add_axes([0.065, 0.430, 0.62, 0.170])
        self.ax_atw = self.fig.add_axes([0.065, 0.210, 0.62, 0.170])
        self.ax_ph = self.fig.add_axes([0.065, 0.085, 0.62, 0.080])
        self.ax_te = self.fig.add_axes([0.73, 0.50, 0.255, 0.41]); self.ax_te.axis("off")
        self.ax_fleet = self.fig.add_axes([0.73, 0.095, 0.255, 0.305]); self.ax_fleet.axis("off")
        self.ax_hdr = self.fig.add_axes([0.065, 0.878, 0.62, 0.047]); self.ax_hdr.axis("off")
        self.fig.suptitle("ARISE / APOPHIS  ·  INNOVATION MONITOR  —  NAV & ATT EKF innovations (σ units) · phase-paced · TCM-marked",
                          color=NEON, fontsize=13, weight="bold", y=0.992)

    def _focus(self, st):
        cand = []
        for di in range(self.N):
            if st["health"][di] == "DEAD":
                continue
            iv = st["innov"][di]
            m = max(float(np.max(np.abs(iv["nav"] / np.maximum(iv["nav_sig"], 1e-9)))),
                    float(np.max(np.abs(iv["att_s"] / np.maximum(iv["att_s_sig"], 1e-12)))),
                    float(np.max(np.abs(iv["att_w"] / np.maximum(iv["att_w_sig"], 1e-12)))))
            cand.append((m, di))
        return max(cand)[1] if (cand and max(cand)[0] > 2.0) else st["focus"]

    def _bands_and_tcm(self, ax, tags, n):
        """Shade phase bands + draw TCM marks behind a trace panel (x = sample-ago)."""
        if not tags:
            return
        xs = np.arange(-n, 0)
        # phase bands
        i = 0
        while i < n:
            ph = tags[i][0]; j = i
            while j < n and tags[j][0] == ph:
                j += 1
            ax.axvspan(xs[i], xs[j - 1] + 1, color=PHASE_COL.get(ph, "#222"), alpha=0.14, lw=0)
            i = j
        # TCM marks (first sample of each TCM frame)
        seen = set()
        for k in range(n):
            if tags[k][2] and tags[k][1] not in seen:
                seen.add(tags[k][1]); ax.axvline(xs[k], color=GREEN, lw=1.0, ls=(0, (2, 2)), alpha=0.8)

    def _panel(self, ax, series, tags, labels, title, n):
        plt = self.plt; ax.clear(); style2d(ax, title, "", "Innovation [σ]")
        ax.tick_params(labelbottom=False)      # shared x — the phase strip below carries it
        self._bands_and_tcm(ax, tags, n)
        ax.axhspan(-1, 1, color=(1, 1, 1, 0.07))
        for s in (3, -3):
            ax.axhline(s, color=RED, ls=(0, (4, 2)), lw=1.0, alpha=0.85)
        ax.axhline(0, color=(1, 1, 1, 0.25), lw=0.6)
        xs = np.arange(-n, 0)
        for i in range(3):
            ax.plot(xs, np.clip([series[k][i] for k in range(n)], -YLIM, YLIM), color=COMP[i], lw=1.3, label=labels[i])
        ax.set_ylim(-YLIM, YLIM); ax.set_xlim(-n, 0)
        ax.legend(fontsize=6.4, ncol=3, facecolor="black", labelcolor="white", edgecolor=AXIS_RGBA, loc="lower left")
        if n and any(abs(series[-1][i]) > 3.5 for i in range(3)):
            ax.text(0.99, 0.9, "OUT OF BOUNDS", transform=ax.transAxes, ha="right", color="white",
                    fontsize=8.5, weight="bold", path_effects=_glow(RED, 2.2))

    def update(self, st):
        plt = self.plt
        ph = st["phase"]; fr = st["frame"]; tcm = fr in set(st.get("tcm_marks", []))
        for d in range(self.N):
            for (nav, sts, gys) in st["innov"][d]["samples"]:
                self.hist[d].append((nav, sts, gys, ph, fr, tcm))
            if not st["innov"][d]["samples"]:                 # DEAD: keep the stream moving (flatline)
                self.hist[d].append((np.zeros(3), np.zeros(3), np.zeros(3), ph, fr, tcm))
        d = self._focus(st); H = list(self.hist[d]); n = len(H)
        fd = st["fdi"][d]; gt = st["gt_fault"].get(d); te = st["truth_est"][d]
        navser = [h[0] for h in H]; atsser = [h[1] for h in H]; atwser = [h[2] for h in H]
        tags = [(h[3], h[4], h[5]) for h in H]
        nis = float(np.sum(np.asarray(navser[-1]) ** 2)) if n else 0.0
        # header
        ax = self.ax_hdr; ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.0, 0.5, f"FOCUS D{d+1}", color=STATE_COLORS[fd["state"]], fontsize=9.5,
                weight="bold", va="center")
        ax.text(0.22, 0.5, f"{fd['state']} · iso {fd['fault']} · inj {gt or 'none'} · "
                           f"PHASE {ph} ×{st['cadence']}", color=TXT, fontsize=6.8,
                va="center", clip_on=True)
        ax.text(0.8, -0.20, f"NIS {nis:4.1f}", color=RED if nis > 7.8 else GREEN, fontsize=8.5,
                ha="right", va="center", weight="bold")   # 0.5 → 0.30 nudges it down within the strip
        # panels
        self._panel(self.ax_nav, navser, tags, NAV_LBL, f"NAV  y_k/σ · CW EKF innov · r={fd['r']['NAV']:.2f}", n)
        self._panel(self.ax_ats, atsser, tags, SIG_LBL, f"ATT  σ_BR/σ · Att-err innov · r={fd['r']['ATT']:.2f}", n)
        self._panel(self.ax_atw, atwser, tags, OM_LBL, "ATT  ω_BR/σ · Body-rate innov", n)
        # phase / TCM legend strip
        ax = self.ax_ph; ax.clear(); ax.set_facecolor("black"); ax.set_xlim(-n if n else -1, 0); ax.set_ylim(0, 1)
        ax.set_yticks([]); ax.tick_params(colors=TXT, labelsize=7)
        for s in ax.spines.values():
            s.set_color(AXIS_RGBA)
        self._bands_and_tcm(ax, tags, n); ax.set_xlabel("Samples ago", color=TXT, fontsize=8)
        xx = 0.0
        for nm, c in PHASE_COL.items():
            ax.add_patch(plt.Rectangle((xx, 1.25), 0.0, 0, color=c))  # no-op keep
        ax.text(0.005, 0.5, "  ".join(f"■{p}" for p in PHASE_COL), transform=ax.transAxes, fontsize=7,
                color=TXT, va="center")
        for p, c in PHASE_COL.items():
            pass
        ax.text(0.78, 0.5, "│ = TCM burn", transform=ax.transAxes, fontsize=7, color=GREEN, va="center")
        # truth-vs-estimate validation
        ax = self.ax_te; ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.97, "TRUTH  vs  ESTIMATE", ha="center", color=NEON, fontsize=10, weight="bold")
        ax.text(0.5, 0.90, "(Only Estimates are given to FDIR)", ha="center", color=(.6, .6, .6), fontsize=7.5)
        rows = [("NAV est error", f"{te['nav_err']:.1f} m", te['nav_err'] < 200),
                ("NAV true drift", f"{te['nav_dev']/1000:.2f} km", te['nav_dev'] < 1),
                ("ATT est error", f"{np.degrees(te['att_err']):.2f}°", np.degrees(te['att_err']) < 1),
                ("ATT true error", f"{te['att_true_deg']:.1f}°", te['att_true_deg'] < 1)]
        y = 0.78
        for lab, val, good in rows:
            ax.text(0.05, y, lab, color=TXT, fontsize=8.5)
            ax.text(0.95, y, val, color=GREEN if good else RED, fontsize=9, ha="right", weight="bold")
            y -= 0.085
        # single multi-line note with its own bbox — box always hugs the text exactly
        ax.text(0.5, 0.25, "Fault drives the TRUTH off \nestimator + FDI catch it\nfrom the innovation alone",
                ha="center", va="center", color=GREEN, fontsize=7.5, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.55", facecolor="black", edgecolor=AXIS_RGBA, lw=1))
        # fleet anomaly strip
        ax = self.ax_fleet; ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, self.N + 0.4)
        ax.text(0.5, self.N + 0.12, "FLEET |peak σ|   NAV / ATT", ha="center", color=NEON, fontsize=8.5, weight="bold")
        for di in range(self.N):
            iv = st["innov"][di]
            navm = float(np.max(np.abs(iv["nav"] / np.maximum(iv["nav_sig"], 1e-9))))
            attm = float(max(np.max(np.abs(iv["att_s"] / np.maximum(iv["att_s_sig"], 1e-12))),
                             np.max(np.abs(iv["att_w"] / np.maximum(iv["att_w_sig"], 1e-12)))))
            row = self.N - 1 - di
            ax.text(0.02, row + 0.5, f"D{di+1}", color=STATE_COLORS[st["health"][di]], fontsize=8, va="center", weight="bold")
            for x0, val, c in [(0.16, navm, COMP[0]), (0.58, attm, AMBER)]:
                ax.add_patch(plt.Rectangle((x0, row + 0.32), 0.32 * min(val / 8., 1), 0.16, color=RED if val > 3 else c))
                ax.plot([x0 + 0.32 * 3 / 8.] * 2, [row + 0.28, row + 0.56], color=(1, 1, 1, 0.4), lw=0.7)
