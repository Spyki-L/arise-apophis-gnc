#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faultlab_subsystems.py — the subsystem / comms live windows for the fault lab:

  W3  SubsystemsWindow  — per-deputy propulsion & attitude actuators: the 8-thruster
                          RCS rosette, scheduled TCM burns and MISFIRES, propellant,
                          attitude error and body rate. A TCM timeline runs along the
                          bottom. This is the "NAV/PROP — thrusters, RCS, TCM" view.
  W6  ISLNetworkWindow  — the inter-satellite-link comms network (chief + 8 deputies,
                          laid out by their true chief-frame geometry): ISL links with
                          live quality, the DATA-TO-CHIEF relays a misbehaving deputy's
                          science is routed through, downlinks, and the peer-imaging
                          hand-offs (which healthy deputy photographs which suspect).

Black mission-ops aesthetic. matplotlib only.
"""
from __future__ import annotations
import numpy as np
import numpy.linalg as la

from cockpit_ops import STATE_COLORS, NEON, GREEN, RED, AMBER, TXT, AXIS_RGBA, _glow, style2d
from faultlab_engine import CHANNELS, FAULT_COLOR


class SubsystemsWindow:
    """W3 — per-deputy RCS / TCM / propellant / attitude actuator tiles + TCM timeline."""

    def __init__(self, engine):
        import matplotlib.pyplot as plt
        self.plt = plt; self.engine = engine; self.N = engine.N
        self.fuel = {d: 100.0 for d in range(self.N)}
        self.fig = plt.figure(figsize=(15, 8.6)); self.fig.patch.set_facecolor("black")
        self.tiles = []
        xs = [0.015, 0.260, 0.505, 0.750]; ys = [0.535, 0.135]
        for r in range(2):
            for c in range(4):
                ax = self.fig.add_axes([xs[c], ys[r], 0.225, 0.375]); ax.axis("off")   # height 0.345 → 0.375
                self.tiles.append(ax)
        self.ax_tl = self.fig.add_axes([0.05, 0.04, 0.90, 0.072])
        self.fig.suptitle("ARISE / APOPHIS  ·  PROPULSION & ATTITUDE SUBSYSTEMS  —  RCS thrusters · TCM burns/misfires · propellant",
                          color=NEON, fontsize=13.5, weight="bold", y=0.975)

    def _tile(self, ax, d, st):
        plt = self.plt
        ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        state = st["health"][d]; col = STATE_COLORS[state]
        ax.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, ec=col, lw=2.0))
        ax.text(0.05, 0.90, f"D{d+1}", color=col, fontsize=11, weight="bold")
        ax.text(0.26, 0.905, state, color=col, fontsize=7.2, weight="bold", clip_on=True)
        ax.text(0.97, 0.905, f"iso:{st['fdi'][d]['fault']}", color=TXT, fontsize=6.2, ha="right")
        # RCS rosette (left)
        cx, cy, rr = 0.27, 0.45, 0.16; lv = st["rcs"][d]
        ax.add_patch(plt.Rectangle((cx - 0.09, cy - 0.09), 0.18, 0.18, color=(0.12, 0.12, 0.16), ec=NEON, lw=1))
        firing_any = False
        for i in range(8):
            a = i / 8 * 2 * np.pi; ux, uy = np.cos(a), np.sin(a) * 1.6
            lvl = float(lv[i]); hot = lvl > 0.4; firing_any = firing_any or hot
            c = (1.0, 0.45, 0.1) if hot else (0.25, 0.6, 1.0)
            ax.plot([cx + 0.10 * ux, cx + (0.12 + rr * lvl) * ux],
                    [cy + 0.10 * uy, cy + (0.12 + rr * lvl) * uy],
                    color=c, lw=1.5 + 4 * lvl, alpha=0.45 + 0.55 * lvl, solid_capstyle="round")
        ax.text(cx, cy - 0.30, "RCS", color=NEON, fontsize=7.5, ha="center", weight="bold")
        # TCM indicator
        tcm = st["tcm"][d]
        if tcm["misfire"]:
            ax.text(0.74, 0.80, "TCM MISFIRE", color=RED, fontsize=7.5, ha="center", weight="bold",
                    path_effects=_glow(RED, 2.2))
        elif tcm["firing"]:
            ax.text(0.74, 0.80, "TCM BURN", color=GREEN, fontsize=7.5, ha="center", weight="bold")
        # right-side bars: propellant, attitude error, body rate
        cfg = self.engine.cfg0
        r = st["fdi"][d]["r"]
        eatt = r["ATT"]; om = la.norm(st["rcs"][d]) * 0.0 + r["ATT"]   # att residual as proxy
        use = (np.sum(lv) * 0.015 + (0.35 if tcm["firing"] else 0) + (0.12 if state in ("SUSPECT", "FAULTED") else 0))
        if state != "DEAD":
            self.fuel[d] = max(0.0, self.fuel[d] - use)
        bx = 0.52
        def bar(y, frac, label, value, color):
            # label left / value right on one row ABOVE the bar — no text-on-bar overlap
            ax.text(bx, y + 0.062, label, color=TXT, fontsize=5.8, va="bottom", clip_on=True)
            ax.text(bx + 0.44, y + 0.062, value, color=color, fontsize=5.8, va="bottom",
                    ha="right", clip_on=True)
            ax.add_patch(plt.Rectangle((bx, y), 0.44, 0.05, fill=False, ec=AXIS_RGBA, lw=0.8))
            ax.add_patch(plt.Rectangle((bx, y), 0.44 * np.clip(frac, 0, 1), 0.05, color=color))
        bar(0.62, self.fuel[d] / 100., "prop", f"{self.fuel[d]:.0f}%",
            GREEN if self.fuel[d] > 25 else RED)
        bar(0.44, min(eatt / 6., 1), "att err", f"{eatt:.1f}", AMBER if eatt > 1 else NEON)
        bar(0.26, min(r['PROP'] / 4., 1), "resid", f"{r['PROP']:.1f}",
            FAULT_COLOR['PROP'] if r['PROP'] > 1 else NEON)
        ax.text(bx, 0.12, "ISL " + ("UP" if st["isl"]["chief"][d] > 0.05 else "DOWN"),
                color=GREEN if st["isl"]["chief"][d] > 0.05 else RED, fontsize=7, weight="bold")

    def update(self, st):
        plt = self.plt
        for d in range(self.N):
            self._tile(self.tiles[d], d, st)
        # TCM timeline — fully self-contained strip: title INSIDE the axes (a set_title
        # would draw above the strip, onto the deputy cards), no x tick labels (the
        # in-strip "TCM f###" markers carry the positions)
        ax = self.ax_tl; ax.clear(); style2d(ax, "", "", "")
        F = self.engine.tr["T"].size
        ax.set_xlim(0, F); ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xticks([])
        ax.axvspan(0, st["frame"], color=(0.2, 0.4, 0.8, 0.12))
        ax.axvline(st["frame"], color=NEON, lw=1.5)
        for _i, fr in enumerate(sorted(self.engine.tcm_frames)):
            ax.axvline(fr, color=GREEN, lw=1.2, ls=(0, (3, 2)))
            ax.text(fr, 0.80 if _i % 2 == 0 else 0.34, f"TCM f{fr}", color=GREEN,
                    fontsize=5.5, ha="center", va="center", clip_on=True)   # was 0.92/0.50, va="top"
        ax.text(0.005, 0.90, "TCM BURN TIMELINE", color=NEON, fontsize=6.2, weight="bold",
                va="top", transform=ax.transAxes, clip_on=True)


class ISLNetworkWindow:
    """W6 — ISL comms network: links, data-to-chief relays, downlink, peer-imaging hand-offs."""

    def __init__(self, engine):
        import matplotlib.pyplot as plt
        self.plt = plt; self.engine = engine; self.N = engine.N
        self.fig = plt.figure(figsize=(14.5, 8.6)); self.fig.patch.set_facecolor("black")
        self.ax = self.fig.add_axes([0.02, 0.08, 0.66, 0.85]);
        self.ax_log = self.fig.add_axes([0.70, 0.08, 0.29, 0.85]); self.ax_log.axis("off")
        self.fig.suptitle("ARISE / APOPHIS  ·  ISL COMMS NETWORK  —  links · data-to-chief relays · peer-imaging hand-offs",
                          color=NEON, fontsize=13.5, weight="bold", y=0.975)
        self.relay_log = []

    def _layout(self, st):
        """Node positions from the true chief-frame geometry (T-N plane), chief at origin."""
        H = st["isl"]["H"]; P = (H[:, [1, 2]].astype(float)).copy()  # (N,2) along-track / cross-track [m]
        sc = np.percentile(np.abs(P), 95) + 1.0
        P = P / sc                                                   # normalize ~[-1,1]
        for _ in range(80):                                          # declutter: keep close nodes/labels legible
            moved = False
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    dd = P[i] - P[j]; r = float(np.hypot(*dd))
                    if r < 0.24:
                        if r < 1e-6:
                            dd = np.array([0.02, 0.0]); r = 0.02
                        u = dd / r; push = (0.24 - r) / 2.0
                        P[i] += u * push; P[j] -= u * push; moved = True
            if not moved:
                break
        return P

    def update(self, st):
        plt = self.plt; ax = self.ax; ax.clear(); ax.set_facecolor("black")
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal"); ax.axis("off")
        P = self._layout(st)
        # ISL deputy-deputy links
        for (i, j, q) in st["isl"]["links"]:
            if q <= 0.02:
                continue
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color=(0.35, 0.9, 0.6),
                    lw=0.4 + 2.2 * q, alpha=0.10 + 0.45 * q, zorder=1)
        # chief at origin
        ax.scatter([0], [0], s=320, marker="*", color=NEON, ec="white", lw=1.2, zorder=5)
        ax.text(0, -0.12, "CHIEF", color=NEON, fontsize=9, ha="center", weight="bold", zorder=6)
        # downlinks deputy->chief
        for d in range(self.N):
            q = st["isl"]["chief"][d]
            if q > 0.02:
                ax.plot([P[d, 0], 0], [P[d, 1], 0], color=(0.3, 0.5, 1.0), lw=0.4 + 1.6 * q,
                        alpha=0.08 + 0.30 * q, zorder=1)
        # data-to-chief relays (animated packets along deputy -> relay -> chief)
        ph = (st["frame"] % 12) / 12.0
        for (d, relay) in st.get("data_to_chief", []):
            path = [P[d], P[relay], np.zeros(2)]
            for a in range(2):
                ax.plot([path[a][0], path[a + 1][0]], [path[a][1], path[a + 1][1]],
                        color="#ff5db1", lw=2.2, alpha=0.85, zorder=3)
            seg = int(ph * 2); fr = ph * 2 - seg
            pk = path[seg] * (1 - fr) + path[seg + 1] * fr
            ax.scatter([pk[0]], [pk[1]], s=70, marker="o", color="white", ec="#ff5db1", lw=1.5, zorder=6)
            ax.text(P[d, 0], P[d, 1] + 0.12, "DATA→CHIEF", color="#ff5db1", fontsize=6.5, ha="center", zorder=6)
        # peer-imaging hand-offs (imager photographs target)
        for (imager, target) in st.get("imaging", []):
            ax.annotate("", xy=(P[target, 0], P[target, 1]), xytext=(P[imager, 0], P[imager, 1]),
                        arrowprops=dict(arrowstyle="-|>", color="#33bbff", lw=2.0, alpha=0.9), zorder=4)
            mid = (P[imager] + P[target]) / 2
            ax.text(mid[0], mid[1] + 0.06, "IMG", color="#33bbff", fontsize=7, ha="center", weight="bold", zorder=6)
        # deputy nodes
        for d in range(self.N):
            stt = st["health"][d]; col = STATE_COLORS[stt]
            ax.scatter([P[d, 0]], [P[d, 1]], s=240, marker="s", color=col, ec="white", lw=0.8, zorder=5)
            ax.text(P[d, 0], P[d, 1] - 0.11, f"D{d+1}", color="white", fontsize=7.5, ha="center", zorder=6)
            if stt == "DEAD":
                ax.text(P[d, 0], P[d, 1] + 0.10, "LOST", color=(.7, .7, .7), fontsize=6.5, ha="center", zorder=6)
        ax.set_title(f"chief-frame · link width = ISL quality · frame {st['frame']}",
                     color=NEON, fontsize=8.5, loc="left", y=0.895)
        # ── side log ─────────────────────────────────────────────────────────
        # compact, tile-safe layout: fixed row pitch sized for the short W6 tile, list
        # entries capped + clipped, link-quality as a 2-column grid — no stacking overlap
        ax = self.ax_log; ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.985, "COMMS / RELAY STATUS", ha="center", va="top", color=NEON,
                fontsize=10, weight="bold")
        y = 0.855
        ax.text(0.03, y, "DATA→CHIEF RELAYS", color="#ff5db1", fontsize=8.5, weight="bold",
                va="center"); y -= 0.082
        d2c = st.get("data_to_chief", [])
        if not d2c:
            ax.text(0.06, y, "none — all deputies nominal", color=(.6, .6, .6), fontsize=7.5,
                    va="center"); y -= 0.082
        for (d, relay) in d2c[:2]:
            ax.text(0.06, y, f"D{d+1} → D{relay+1} → CHIEF", color="white", fontsize=7.5,
                    family="monospace", va="center", clip_on=True); y -= 0.082
        ax.text(0.03, y, "PEER IMAGING", color="#33bbff", fontsize=8.5, weight="bold",
                va="center"); y -= 0.082
        img = st.get("imaging", [])
        if not img:
            ax.text(0.06, y, "none active", color=(.6, .6, .6), fontsize=7.5, va="center"); y -= 0.082
        for (im, tg) in img[:2]:
            ax.text(0.06, y, f"D{im+1} images D{tg+1}", color="white", fontsize=7.5,
                    family="monospace", va="center", clip_on=True); y -= 0.082
        ax.text(0.03, y, "LINK QUALITY (to chief)", color=NEON, fontsize=8.5, weight="bold",
                va="center"); y -= 0.09
        rows = (self.N + 1) // 2
        for d in range(self.N):
            cxx = 0.03 if d < rows else 0.53
            yy = y - (d % rows) * 0.088
            q = st["isl"]["chief"][d]; c = GREEN if q > 0.4 else (AMBER if q > 0.05 else RED)
            ax.text(cxx, yy, f"D{d+1}", color=TXT, fontsize=7.5, va="center")
            ax.add_patch(self.plt.Rectangle((cxx + 0.10, yy - 0.017), 0.25 * max(q, 0.02), 0.034, color=c))
            ax.text(cxx + 0.37, yy, f"{q:.2f}", color=c, fontsize=7, va="center")
