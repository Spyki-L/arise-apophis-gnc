#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faultlab_windows.py — the FDI-core-centric live windows for the fault lab:

  W2  ChannelScopeWindow   — the 8x6 live CUSUM detector matrix (every deputy x every
                             channel) vs thresholds, deputy state lamps, and an
                             injected-vs-isolated verification ticker. This is the
                             "is the FDI core actually working" view.
  W5  DeputyDeepDiveWindow — one deputy (the worst-off) in full: 6 CUSUM traces vs
                             thresholds, residual bars, RCS thruster rosette, ISL links,
                             confidence, and the live FDIR decision.

Black mission-ops aesthetic (shared with cockpit_ops). matplotlib only.
"""
from __future__ import annotations
from collections import deque
import numpy as np

from cockpit_ops import STATE_COLORS, NEON, GREEN, RED, AMBER, TXT, AXIS_RGBA, _glow, style2d
from faultlab_engine import CHANNELS, FAULT_COLOR

# The per-deputy DEPUTY/STATE lamp column in W2 duplicates W1's FLEET HEALTH panel, so it is
# hidden by default; the CUSUM matrix widens to reclaim the space. Set True to bring it back.
SHOW_DEPUTY_STATE = False


def _round_lamp(ax, x, y, w, h, color, label, sub, fs=8):
    import matplotlib.pyplot as plt
    ax.add_patch(plt.Rectangle((x, y), w, h, color=color, alpha=0.9, zorder=2))
    ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center", color="black",
            fontsize=fs + 1, weight="bold", zorder=3)
    ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center", color="black",
            fontsize=fs - 1.5, weight="bold", zorder=3)


class ChannelScopeWindow:
    """W2 — fleet x channel CUSUM detector matrix + state lamps + verification."""

    def __init__(self, engine):
        import matplotlib.pyplot as plt
        self.plt = plt; self.engine = engine; self.N = engine.N
        self.fig = plt.figure(figsize=(15, 8.5)); self.fig.patch.set_facecolor("black")
        if hasattr(self.fig, "number"):                 # standalone Figure only (skip when embedded in a subfigure)
            try:
                self.fig.canvas.manager.set_window_title("W2 · FDI CORE SCOPE — live CUSUM detectors")
            except Exception:
                pass
        if SHOW_DEPUTY_STATE:
            self.ax_lamp = self.fig.add_axes([0.015, 0.10, 0.135, 0.80]); self.ax_lamp.axis("off")
            _mat_left = 0.175
        else:
            self.ax_lamp = None
            _mat_left = 0.045
        self.ax_mat = self.fig.add_axes([_mat_left, 0.10, 0.875 - _mat_left, 0.80])
        self.ax_leg = self.fig.add_axes([0.885, 0.10, 0.105, 0.80]); self.ax_leg.axis("off")
        self.ax_tick = self.fig.add_axes([0.015, 0.015, 0.975, 0.06]); self.ax_tick.axis("off")
        self.fig.suptitle("ARISE / APOPHIS  ·  FDI CORE SCOPE  —  per-deputy × per-channel leaky-CUSUM detectors",
                          color=NEON, fontsize=14, weight="bold", y=0.975)

    def update(self, st):
        plt = self.plt; thr = st["thresholds"]
        # ── state lamps (hidden by default — duplicates W1 Fleet Health) ─────
        if SHOW_DEPUTY_STATE and self.ax_lamp is not None:
            ax = self.ax_lamp; ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, self.N)
            ax.text(0.5, self.N + 0.15, "DEPUTY / STATE", ha="center", color=NEON, fontsize=9, weight="bold")
            for d in range(self.N):
                fd = st["fdi"][d]; row = self.N - 1 - d
                col = STATE_COLORS[fd["state"]]
                _round_lamp(ax, 0.05, row + 0.12, 0.9, 0.76, col, f"D{d+1}  {fd['state'][:4]}",
                            f"iso:{fd['fault']}  c{fd['conf']:.2f}")
        # ── CUSUM matrix ─────────────────────────────────────────────────────
        ax = self.ax_mat; ax.clear(); ax.set_facecolor("black")
        ax.set_xlim(0, len(CHANNELS)); ax.set_ylim(0, self.N)
        for sp in ax.spines.values():
            sp.set_color(AXIS_RGBA)
        ax.set_xticks([]); ax.set_yticks([])
        for c, ch in enumerate(CHANNELS):
            ax.text(c + 0.5, self.N + 0.05, ch, ha="center", va="bottom", color=FAULT_COLOR[ch], fontsize=9, weight="bold")
            ax.text(c + 0.5, self.N + 0.30, f"h={thr[ch]:.0f}", ha="center", va="bottom", color=(.6, .6, .6), fontsize=6.0)
        n_trip = 0
        for d in range(self.N):
            fd = st["fdi"][d]; row = self.N - 1 - d
            for c, ch in enumerate(CHANNELS):
                S = fd["S"][ch]; h = thr[ch]; ratio = S / h if h > 0 else 0.0
                # cell frame
                ax.add_patch(plt.Rectangle((c + 0.04, row + 0.06), 0.92, 0.88, fill=False,
                                           edgecolor=(1, 1, 1, 0.10), lw=0.6))
                bar = min(ratio, 1.30) / 1.30 * 0.86
                tripped = ratio >= 1.0
                fcol = FAULT_COLOR[ch] if not tripped else RED
                ax.add_patch(plt.Rectangle((c + 0.18, row + 0.08), 0.64, max(bar, 0.006),
                                           color=fcol, alpha=0.92 if tripped else 0.7, zorder=2))
                # threshold line at ratio=1 within the cell
                yth = row + 0.08 + (1.0 / 1.30) * 0.86
                ax.plot([c + 0.12, c + 0.88], [yth, yth], color=(1, 1, 1, 0.35), lw=0.7, ls=(0, (3, 2)))
                if tripped:
                    n_trip += 1
                    ax.text(c + 0.5, row + 0.5, "TRIP", ha="center", va="center", color="white",
                            fontsize=6.8, weight="bold", path_effects=_glow(RED, 2.4), zorder=4)
            ax.text(-0.06, row + 0.5, f"D{d+1}", ha="right", va="center", color=TXT, fontsize=8)
        ax.text(len(CHANNELS) / 2, -0.06, f"bar = S / h  (fills to the dashed TRIP line at threshold)   ·   "
                f"frame {st['frame']}  ·  t-to-CA {st['t_go']/3600:.2f} h",
                ha="center", va="top", color=(.62, .62, .62), fontsize=6.5)
        # ── legend / counts ──────────────────────────────────────────────────
        ax = self.ax_leg; ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.97, "STATE", ha="center", color=NEON, fontsize=9, weight="bold")
        for i, (k, c) in enumerate(STATE_COLORS.items()):
            ax.add_patch(plt.Rectangle((0.04, 0.86 - i * 0.075), 0.16, 0.05, color=c))
            ax.text(0.26, 0.885 - i * 0.075, k, color=TXT, fontsize=7.5, va="center")
        hc = sum(v == "HEALTHY" for v in st["health"].values())
        fc = sum(v in ("SUSPECT", "FAULTED") for v in st["health"].values())
        dc = sum(v == "DEAD" for v in st["health"].values())
        ax.text(0.5, 0.50, "FLEET", ha="center", color=NEON, fontsize=9, weight="bold")
        ax.text(0.06, 0.42, f"Healthy  {hc}", color=GREEN, fontsize=8)
        ax.text(0.06, 0.36, f"Faulted  {fc}", color=RED, fontsize=8)
        ax.text(0.06, 0.30, f"Dead     {dc}", color=(.6, .6, .6), fontsize=8)
        ax.text(0.06, 0.20, f"Channels", color=TXT, fontsize=8)
        ax.text(0.06, 0.14, f"Tripped  {n_trip}", color=AMBER, fontsize=8)
        # ── verification ticker ──────────────────────────────────────────────
        ax = self.ax_tick; ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.005, 0.5, "INJECTED → ISOLATED:", color=NEON, fontsize=8.5, weight="bold", va="center")
        x = 0.14
        gt = st["gt_fault"]
        if not gt:
            ax.text(x, 0.5, "no active fault — fleet nominal", color=(.6, .6, .6), fontsize=8, va="center")
        for d, kind in sorted(gt.items()):
            S = st["fdi"][d]["S"][kind]; h = st["thresholds"][kind]; det = S >= h
            mark = "DETECTED" if det else "building"
            mcol = GREEN if det else AMBER
            txt = f"D{d+1} {kind} S/h={S/h:0.1f} [{mark}]"
            ax.text(x, 0.5, txt, color=mcol, fontsize=8, va="center", family="monospace")
            x += 0.010 * len(txt) + 0.016


class DeputyDeepDiveWindow:
    """W5 — one deputy in full depth (auto-focuses the worst-off deputy)."""

    def __init__(self, engine, history=140):
        import matplotlib.pyplot as plt
        self.plt = plt; self.engine = engine; self.N = engine.N; self.H = history
        self.hist = {d: {ch: deque(maxlen=history) for ch in CHANNELS} for d in range(self.N)}
        self.fhist = {d: deque(maxlen=history) for d in range(self.N)}
        self.fig = plt.figure(figsize=(14, 8.5)); self.fig.patch.set_facecolor("black")
        self.ax_lamp = self.fig.add_axes([0.015, 0.55, 0.20, 0.36]); self.ax_lamp.axis("off")
        self.ax_res = self.fig.add_axes([0.04, 0.10, 0.20, 0.36])
        self.ax_tr = self.fig.add_axes([0.265, 0.55, 0.46, 0.33])
        self.ax_rcs = self.fig.add_axes([0.265, 0.10, 0.21, 0.36])
        self.ax_isl = self.fig.add_axes([0.515, 0.10, 0.21, 0.36])
        self.ax_dec = self.fig.add_axes([0.755, 0.10, 0.235, 0.81]); self.ax_dec.axis("off")
        self.fig.suptitle("ARISE / APOPHIS  ·  DEPUTY DEEP-DIVE  —  full sensor / detector / actuator / decision state",
                          color=NEON, fontsize=13.5, weight="bold", y=0.975)

    def update(self, st):
        plt = self.plt; thr = st["thresholds"]
        for d in range(self.N):
            for ch in CHANNELS:
                self.hist[d][ch].append(st["fdi"][d]["S"][ch] / max(thr[ch], 1e-9))
            self.fhist[d].append(st["fdi"][d]["conf"])
        d = st["focus"]; fd = st["fdi"][d]
        # ── lamp + confidence ────────────────────────────────────────────────
        ax = self.ax_lamp; ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        col = STATE_COLORS[fd["state"]]
        ax.add_patch(plt.Rectangle((0.01, 0.50), 0.9, 0.5, color=col, alpha=0.92))
        ax.text(0.5, 0.78, f"D{d+1}", ha="center", color="black", fontsize=24, weight="bold")
        ax.text(0.5, 0.56, f"{fd['state']}", ha="center", color="black", fontsize=13, weight="bold")
        ax.text(0.5, 0.36, f"Isolated: {fd['fault']}", ha="center", color=TXT, fontsize=8,
                weight="bold", clip_on=True)
        ax.add_patch(plt.Rectangle((0.05, 0.18), 0.9, 0.10, fill=False, edgecolor=AXIS_RGBA, lw=1))
        ax.add_patch(plt.Rectangle((0.05, 0.18), 0.9 * fd["conf"], 0.10,
                                   color=RED if fd["conf"] > 0.5 else AMBER))
        ax.text(0.5, 0.075, f"Confidence {fd['conf']:.2f}", ha="center", color=TXT,
                fontsize=7, clip_on=True)
        # ── residual bars ────────────────────────────────────────────────────
        ax = self.ax_res; ax.clear(); style2d(ax, "RESIDUALS (Raw r)", "", "")
        vals = [fd["r"][ch] for ch in CHANNELS]
        ax.bar(range(len(CHANNELS)), vals, color=[FAULT_COLOR[c] for c in CHANNELS])
        ax.set_xticks(range(len(CHANNELS))); ax.set_xticklabels(CHANNELS, color=TXT, fontsize=7, rotation=30)
        ax.set_ylim(0, max(3.0, max(vals) * 1.2))
        # ── CUSUM traces vs threshold ────────────────────────────────────────
        ax = self.ax_tr; ax.clear(); style2d(ax, f"D{d+1} CUSUMs · S/h live", "frames ago", "S/h")
        xs = np.arange(-len(self.hist[d]["NAV"]), 0)
        for ch in CHANNELS:
            ax.plot(xs, np.clip(list(self.hist[d][ch]), 0, 2.15), color=FAULT_COLOR[ch], lw=1.6, label=ch)
        ax.axhline(1.0, color="white", ls=(0, (4, 2)), lw=1.0)
        ax.text(xs[0] if len(xs) else 0, 1.04, "TRIP threshold", color="white", fontsize=7)
        ax.set_ylim(0, 2.2); ax.legend(fontsize=6.5, ncol=3, facecolor="black", labelcolor="white",
                                       edgecolor=AXIS_RGBA, loc="upper left")
        # ── RCS rosette ──────────────────────────────────────────────────────
        ax = self.ax_rcs; ax.clear(); ax.set_facecolor("black"); ax.axis("off")
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.35); ax.set_aspect("equal")
        ax.set_title("RCS THRUSTERS", color=NEON, fontsize=9.5, weight="bold", loc="left")
        lv = st["rcs"][d]
        ax.add_patch(plt.Rectangle((-0.34, -0.34), 0.68, 0.68, color=(0.12, 0.12, 0.16), ec=NEON, lw=1))
        for i in range(8):
            a = i / 8 * 2 * np.pi; ux, uy = np.cos(a), np.sin(a)
            lvl = float(lv[i]); c = (1.0, 0.45 + 0.0 * lvl, 0.1) if lvl > 0.4 else (0.2, 0.6, 1.0)
            ax.plot([0.42 * ux, (0.55 + lvl) * ux], [0.42 * uy, (0.55 + lvl) * uy],
                    color=c, lw=2 + 5 * lvl, alpha=0.5 + 0.5 * lvl, solid_capstyle="round")
            ax.scatter([0.42 * ux], [0.42 * uy], s=14, color=NEON)
        if st["tcm"][d]["firing"]:
            ax.text(0, 1.16, "TCM MISFIRE" if st["tcm"][d]["misfire"] else "TCM BURN",
                    ha="center", color=RED if st["tcm"][d]["misfire"] else GREEN, fontsize=8.5, weight="bold")
        # ── ISL links from this deputy ───────────────────────────────────────
        ax = self.ax_isl; ax.clear(); ax.set_facecolor("black"); ax.axis("off")
        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.45, 1.15); ax.set_aspect("equal")
        ax.set_title("ISL LINKS", color=NEON, fontsize=9.5, weight="bold", loc="left")
        CY = -0.22; R = 0.92                                   # push star DOWN; ring radius
        ax.scatter([0], [CY], s=120, marker="s", color=STATE_COLORS[fd["state"]], ec="white", zorder=3)
        ax.text(0, CY - 0.20, f"D{d+1}", ha="center", color="white", fontsize=7, zorder=4)
        others = [p for p in range(self.N) if p != d]
        PHASE = np.pi/2 + np.pi/len(others)                   # gap at top-center for the header text
        for k, p in enumerate(others):
            a = PHASE + k / len(others) * 2 * np.pi
            px, py = R * np.cos(a), CY + R * np.sin(a)
            q = 0.0
            for (i, j, qq) in st["isl"]["links"]:
                if {i, j} == {d, p}:
                    q = qq; break
            dead = st["health"][p] == "DEAD"
            ax.plot([0, px], [CY, py], color=(0.4, 1.0, 0.6) if not dead else (0.4, 0.4, 0.4),
                    lw=0.5 + 2.5 * q, alpha=0.2 + 0.7 * q)
            ax.scatter([px], [py], s=60, marker="s", color=STATE_COLORS[st["health"][p]], zorder=3)
            lx, ly = 1.16 * R * np.cos(a), CY + 1.16 * R * np.sin(a)   # label just outside each node
            ax.text(lx, ly, f"D{p+1}", ha="center", va="center", color=TXT, fontsize=6)
        ax.text(-1.2, 1.05, f"Chief link q={st['isl']['chief'][d]:.2f}", ha="left", color=NEON, fontsize=7)
        # ── decision panel ───────────────────────────────────────────────────
        ax = self.ax_dec; ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.97, "FDIR DECISION LOG", ha="center", color=NEON, fontsize=10, weight="bold")
        # 10 rows, wider line pitch, clipped + truncated so nothing can spill into the
        # neighbouring panel or the FOCUS box below, even at large font scales
        for i, (fr, msg) in enumerate(self.engine.log[-10:][::-1]):
            focus_hit = f"D{d+1} " in msg
            ax.text(0.02, 0.905 - i * 0.074, f"f{fr:>3}", color=AMBER, fontsize=6.4,
                    family="monospace", clip_on=True)
            ax.text(0.26, 0.905 - i * 0.074, msg[:24], color=GREEN if not focus_hit else "white",
                    fontsize=6.8, family="monospace", weight="bold" if focus_hit else "normal",
                    clip_on=True)
        gt = st["gt_fault"].get(d)
        ax.add_patch(plt.Rectangle((0.02, 0.025), 0.96, 0.155, color=(0.1, 0.1, 0.14), ec=NEON, lw=1))
        ax.text(0.5, 0.138, f"FOCUS D{d+1} · inj: {gt or 'none'}", ha="center", va="center",
                color="white", fontsize=7.5, weight="bold", clip_on=True)
        ax.text(0.5, 0.062, f"FDI: {fd['fault']} · {fd['state']}", ha="center", va="center",
                color=GREEN if (gt is None or fd['fault'] == gt or (gt == 'FORM' and fd['fault'] in ('FORM', 'NAV'))) else AMBER,
                fontsize=7.2, clip_on=True)
