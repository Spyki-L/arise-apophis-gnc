#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faultlab_brain.py — W4: the AUTONOMY BRAIN.

A live reasoning connectome of the onboard FDIR executive — how the autonomy "thinks":
it wires many independent information streams (the six FDI detectors + fleet consensus +
peer vision + the suspect's ISL self-report + collision-safety CBFs + the deadline-aware
coverage value), INTEGRATES them, fuses the evidence ROBUSTLY in a pulsing decision CORE,
forms a BELIEF (the fault hypothesis + confidence), and COMMITS an action — with the
active thought-path lit and signal packets flowing along the synapses, a live
chain-of-thought, and the fleet belief state. Graphically rich; matplotlib only.
"""
from __future__ import annotations
import numpy as np

from cockpit_ops import STATE_COLORS, NEON, GREEN, RED, AMBER, TXT, AXIS_RGBA, _glow
from faultlab_engine import CHANNELS, FAULT_COLOR

CORE = "#21e0ff"
IN_EXTRA = {"CONSENSUS": "#00e0c6", "VISION": "#33bbff", "ISL-LOG": "#ff5db1",
            "CBF-SAFE": "#9b6bff", "VALUE": "#ffd24a"}
HYP = ["NONE", "NAV", "ATT", "COMM", "PROP", "FORM", "PEER"]
HYP_COL = {"NONE": "#39507a", "NAV": FAULT_COLOR["NAV"], "ATT": FAULT_COLOR["ATT"], "COMM": FAULT_COLOR["COMM"],
           "PROP": FAULT_COLOR["PROP"], "FORM": FAULT_COLOR["FORM"], "PEER": FAULT_COLOR["PEER"]}
ACTS = ["REINIT", "RELOAD", "MODE_SWITCH", "SAFE_HOLD", "DISPOSE", "RE-OPTIMIZE", "RE-IMAGE", "RELAY"]
FAULT2ACT = {"NAV": "REINIT", "COMM": "RELOAD", "FORM": "MODE_SWITCH", "ATT": "SAFE_HOLD",
             "PROP": "SAFE_HOLD", "PEER": "SAFE_HOLD", "NONE": None}


def _curve(p0, p1, bend=0.05):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]; L = np.hypot(dx, dy) + 1e-9
    nx, ny = -dy / L, dx / L
    cx, cy = (p0[0] + p1[0]) / 2 + nx * bend, (p0[1] + p1[1]) / 2 + ny * bend
    t = np.linspace(0, 1, 18)
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * cx + t ** 2 * p1[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * cy + t ** 2 * p1[1]
    return x, y


class AutonomyBrainWindow:
    def __init__(self, engine):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle  # noqa
        self.plt = plt; self.engine = engine; self.N = engine.N
        self.fig = plt.figure(figsize=(16, 9)); self.fig.patch.set_facecolor("black")
        self.ax = self.fig.add_axes([0, 0, 1, 0.93]); self.ax.set_axis_off()
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        self.fig.suptitle("AUTONOMY", color=CORE, fontsize=13, weight="bold", y=0.965)
        # ── compact decision pipeline stages ─────────────────────────────────
        self.STAGES = ["SENSE", "INTEGRATE", "ISOLATE", "DECIDE", "COMMIT"]

    # ── glowing neuron ───────────────────────────────────────────────────────
    def _neuron(self, xy, color, act, r=0.013, pulse=0.0, ring=0.0):
        from matplotlib.patches import Circle
        x, y = xy; rr = r * (1 + 0.18 * pulse)
        a = 0.18 + 0.82 * float(np.clip(act, 0, 1))
        for i, gl in enumerate((3.4, 2.4, 1.7)):
            self.ax.add_patch(Circle((x, y), rr * gl, color=color, alpha=0.05 + 0.10 * a, lw=0, zorder=2))
        self.ax.add_patch(Circle((x, y), rr, color=color, alpha=a, ec="white", lw=0.6 * a, zorder=3))
        if ring > 0:
            th = np.linspace(0, 2 * np.pi * ring, 40)
            self.ax.plot(x + rr * 2.2 * np.cos(th), y + rr * 2.2 * np.sin(th), color="white", lw=1.6, zorder=4)

    def _syn(self, e, color, act, ph, base=0.05):
        x, y = _curve(self.P[e[0]], self.P[e[1]])
        self.ax.plot(x, y, color=color, lw=0.6 + 2.4 * act, alpha=base + 0.7 * act, zorder=1, solid_capstyle="round")
        if act > 0.2:
            k = int((ph % 1.0) * (len(x) - 1))
            self.ax.scatter([x[k]], [y[k]], s=18 + 26 * act, color="white", ec=color, lw=1.2, zorder=5)

    def _focus_action(self, d):
        for fr, msg in reversed(self.engine.log):
            if msg.startswith(f"D{d+1} ") and "isolated ->" in msg:
                return msg.split("isolated ->")[1].split(";")[0].strip()
        return None

    # ── flat rounded chip ────────────────────────────────────────────────────
    def _chip(self, cx, cy, w, h, label, facecolor, edgecolor, txtcolor, fs=8.0, weight="bold"):
        from matplotlib.patches import FancyBboxPatch
        self.ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle="round,pad=0.004,rounding_size=0.012",
                          facecolor=facecolor, edgecolor=edgecolor, lw=1.2, zorder=3, mutation_aspect=0.6))
        self.ax.text(cx, cy, label, ha="center", va="center", color=txtcolor,
                     fontsize=fs, weight=weight, zorder=4)

    def update(self, st):
        ax = self.ax; ax.clear(); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ph = (st["frame"] % 16) / 16.0; pulse = 0.5 + 0.5 * np.sin(st["frame"] * 0.5)
        d = st["focus"]; fd = st["fdi"][d]; thr = st["thresholds"]
        kind = st["gt_fault"].get(d); fault = fd["fault"]; conf = fd["conf"]
        active = fd["state"] in ("SUSPECT", "FAULTED", "DEAD")
        # ── input activations ────────────────────────────────────────────────
        actv = {}
        dom = "NAV"; domv = 0.0
        for ch in CHANNELS:
            v = float(np.clip(fd["S"][ch] / max(thr[ch], 1e-9), 0, 1))
            actv[("in", ch)] = v
            if v > domv:
                domv = v; dom = ch
        actv[("in", "CONSENSUS")] = conf if active else 0.05
        actv[("in", "VISION")] = 0.9 if active else 0.05
        actv[("in", "ISL-LOG")] = (0.2 if kind == "PEER" else 0.8) if active else 0.05   # PEER => lying/low trust
        actv[("in", "CBF-SAFE")] = 0.8 if active else 0.2
        actv[("in", "VALUE")] = float(np.clip(st.get("cov_onb", 0.6), 0, 1))
        # integrators
        actv[("int", "DETECT")] = domv
        actv[("int", "ISOLATE")] = domv if active else 0.05
        actv[("int", "FUSE")] = conf if active else 0.05
        actv[("int", "SAFETY")] = actv[("in", "CBF-SAFE")]
        actv[("int", "VALUE")] = actv[("in", "VALUE")]
        actv[("core", "X")] = (0.35 + 0.65 * conf) if active else 0.18
        # hypotheses (belief)
        for h in HYP:
            actv[("hyp", h)] = (conf if h == fault else 0.05) if active else (0.8 if h == "NONE" else 0.05)
        # actions
        act_name = self._focus_action(d) if active else None
        committed = None
        if act_name:
            for a in ACTS:
                if a.split()[0] in act_name or a.replace("_", " ") in act_name:
                    committed = a; break
            if committed is None and "RECOVERED" in act_name:
                committed = FAULT2ACT.get(fault)
            if committed is None and "relay" in act_name.lower():
                committed = "RELAY"
        for a in ACTS:
            on = (a == committed)
            base = 0.12
            if a in ("RE-OPTIMIZE", "RE-IMAGE", "RELAY") and active:
                base = 0.45
            actv[("act", a)] = 1.0 if on else base
        actv[("commit", "X")] = (0.4 + 0.6 * conf) if committed else 0.15

        # ── colors for this focus deputy ─────────────────────────────────────
        scol = STATE_COLORS[fd["state"]]
        ev_ratio = fd['S'][dom] / max(thr[dom], 1e-9)
        muted = (0.40, 0.46, 0.60)
        mut_face = (0.07, 0.09, 0.13)
        hi_face = (0.10, 0.14, 0.20)

        # ── focus header ─────────────────────────────────────────────────────
        ax.text(0.030, 0.905, f"D{d+1}", color=scol, fontsize=11, weight="bold",
                ha="left", va="center", zorder=5)
        ax.text(0.105, 0.905, fd["state"], color=scol, fontsize=8.0, weight="bold",
                ha="left", va="center", zorder=5)
        ax.text(0.970, 0.905, f"conf {conf:.2f}", color=(CORE if active else muted),
                fontsize=7.5, ha="right", va="center", zorder=5)

        # ── decision PIPELINE row (5 flat chips + thin arrows) ───────────────
        py = 0.715
        # active stage index: how far the reasoning has progressed
        if not active:
            act_idx = 0
        elif committed:
            act_idx = 4
        else:
            act_idx = 3
        xs = np.linspace(0.115, 0.885, len(self.STAGES))
        cw, chh = 0.155, 0.105
        for i, (sx, lab) in enumerate(zip(xs, self.STAGES)):
            on = active and i <= act_idx
            face = hi_face if on else mut_face
            ec = scol if on else muted
            tc = "white" if on else muted
            self._chip(sx, py, cw, chh, lab, face, ec, tc, fs=7.6)
            if i < len(self.STAGES) - 1:
                gap0 = sx + cw / 2 + 0.006; gap1 = xs[i + 1] - cw / 2 - 0.006
                arr_on = active and (i + 1) <= act_idx
                acol = scol if arr_on else muted
                ax.add_line(self.plt.Line2D([gap0, gap1], [py, py], color=acol,
                            lw=1.6 if arr_on else 1.0, zorder=2, solid_capstyle="round"))
                ax.add_line(self.plt.Line2D([gap1 - 0.010, gap1], [py + 0.013, py],
                            color=acol, lw=1.6 if arr_on else 1.0, zorder=2, solid_capstyle="round"))
                ax.add_line(self.plt.Line2D([gap1 - 0.010, gap1], [py - 0.013, py],
                            color=acol, lw=1.6 if arr_on else 1.0, zorder=2, solid_capstyle="round"))

        # ── reasoning line ───────────────────────────────────────────────────
        if active:
            fcol = HYP_COL.get(fault, CORE)
            act_lbl = (committed or (act_name.split(";")[0].strip() if act_name else "DECIDING"))
            ax.text(0.030, 0.495, f"D{d+1}", color=scol, fontsize=8.5, weight="bold",
                    ha="left", va="center", zorder=5)
            ax.text(0.105, 0.495, f"{dom}", color=fcol, fontsize=8.5, weight="bold",
                    ha="left", va="center", zorder=5)
            ax.text(0.195, 0.495, f"{ev_ratio:.1f}x thresh", color=(0.78, 0.82, 0.92),
                    fontsize=8.0, ha="left", va="center", zorder=5)
            ax.text(0.430, 0.495, f"->  isolate {fault}", color=fcol, fontsize=8.0,
                    weight="bold", ha="left", va="center", zorder=5)
            ax.text(0.730, 0.495, f"->  {act_lbl}", color=GREEN, fontsize=8.0,
                    weight="bold", ha="left", va="center", zorder=5)
        else:
            ax.text(0.030, 0.495, f"D{d+1}", color=scol, fontsize=8.5, weight="bold",
                    ha="left", va="center", zorder=5)
            ax.text(0.105, 0.495, "all channels within bounds  ->  NOMINAL", color=GREEN,
                    fontsize=8.0, ha="left", va="center", zorder=5)

        # ── FLEET BELIEFS strip (8 flat cells) ───────────────────────────────
        ax.text(0.030, 0.345, "FLEET BELIEFS", color=CORE, fontsize=7.5, weight="bold",
                ha="left", va="center", zorder=5)
        fxs = np.linspace(0.090, 0.940, self.N)
        fw = min(0.092, (fxs[1] - fxs[0]) * 0.86) if self.N > 1 else 0.092
        fy = 0.250; fh_cell = 0.080
        for di, fx in enumerate(fxs):
            fh = st["fdi"][di]
            healthy = st["health"][di] == "HEALTHY"
            c = GREEN if healthy else HYP_COL.get(fh["fault"], "#39507a")
            face = (0.06, 0.10, 0.075) if healthy else (0.14, 0.08, 0.09)
            self._chip(fx, fy, fw, fh_cell, f"D{di+1}", face, c,
                       c if not healthy else (0.85, 0.92, 0.85), fs=6.8)
            sub = "OK" if healthy else fh["fault"][:4]
            ax.text(fx, fy - 0.072, sub, ha="center", va="center", color=c, fontsize=6.2, zorder=4)

        # ── chain-of-thought (one truncated monospace line) ──────────────────
        if active:
            isl_txt = "self-report DISTRUSTED" if kind == "PEER" else "self-report OK"
            thought = (f"D{d+1}: {dom} {ev_ratio:.0f}x thr | isolate={fault} | "
                       f"consensus+vision {conf:.2f}, {isl_txt} -> {act_name or 'deciding'}")
        else:
            thought = f"D{d+1}: all detectors within bounds | belief=NOMINAL | monitoring {self.N} deputies"
        ax.add_patch(self.plt.Rectangle((0.022, 0.040), 0.956, 0.070,
                     facecolor=(0.05, 0.07, 0.11), ec=(CORE if active else AXIS_RGBA), lw=1.0, zorder=2))
        ax.text(0.036, 0.075, thought[:96], color=(GREEN if active else (0.70, 0.74, 0.80)),
                fontsize=6.8, family="monospace", ha="left", va="center", zorder=4)
