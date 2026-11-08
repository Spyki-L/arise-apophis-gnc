#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faultlab_blockdiagram.py — W4: a LIVE block diagram of the whole onboard FDIR
architecture, drawn as a closed control loop with animated data flow:

   SENSORS ─► CHANNEL DETECTORS ─► FUSION / ISOLATION ─► STATE MACHINE
                                                              │
   (plant) ◄─ ACTUATORS ◄─ RECOVERY · RE-OPT · ISL-SCHED ◄─ AUTONOMY EXECUTIVE

Every frame the blocks and edges that are actually carrying data light up: a
channel whose fleet-max CUSUM is climbing glows in its colour and pulses packets
into FUSION; a conviction pulses the EXECUTIVE and the chosen response block; the
actuators (RCS / TCM / camera / comms) light when they act. It mirrors the real
engine state exactly — it's the architecture you can watch think. matplotlib only.
"""
from __future__ import annotations
import numpy as np

from cockpit_ops import STATE_COLORS, NEON, GREEN, RED, AMBER, TXT, AXIS_RGBA, _glow
from faultlab_engine import CHANNELS, FAULT_COLOR

SENSOR_OF = {"NAV": "rel-NAV EKF", "ATT": "star-tracker + gyro", "COMM": "RF heartbeat",
             "PROP": "accel / Δv", "FORM": "range / coord", "PEER": "ISL peer reports"}


def _blend(base, activity):
    base = np.array(__import__("matplotlib").colors.to_rgb(base))
    return tuple(0.10 + (base - 0.10) * (0.25 + 0.75 * activity))


class BlockDiagramWindow:
    def __init__(self, engine):
        import matplotlib.pyplot as plt
        self.plt = plt; self.engine = engine; self.N = engine.N
        self.exec_glow = 0; self.last_action = ""
        self.fig = plt.figure(figsize=(16, 9)); self.fig.patch.set_facecolor("black")
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 0.93]); self.ax.set_axis_off()
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        self.fig.suptitle("ARISE / APOPHIS  ·  ONBOARD FDIR ARCHITECTURE  —  live data flow (the loop you can watch think)",
                          color=NEON, fontsize=14, weight="bold", y=0.975)
        # static block geometry: key -> (x, y, w, h)
        self.B = {}
        ys = np.linspace(0.86, 0.55, 6)
        for i, ch in enumerate(CHANNELS):
            self.B[f"S_{ch}"] = (0.015, ys[i] - 0.022, 0.135, 0.044)
            self.B[f"C_{ch}"] = (0.175, ys[i] - 0.022, 0.110, 0.044)
        self.B["FUSE"] = (0.330, 0.66, 0.115, 0.16)
        self.B["STATE"] = (0.490, 0.63, 0.150, 0.22)
        self.B["EXEC"] = (0.700, 0.66, 0.150, 0.16)
        self.B["RECOV"] = (0.700, 0.40, 0.150, 0.075)
        self.B["REOPT"] = (0.700, 0.30, 0.150, 0.075)
        self.B["ISLS"] = (0.700, 0.20, 0.150, 0.075)
        self.B["A_RCS"] = (0.470, 0.36, 0.120, 0.058)
        self.B["A_TCM"] = (0.470, 0.28, 0.120, 0.058)
        self.B["A_CAM"] = (0.470, 0.20, 0.120, 0.058)
        self.B["A_COM"] = (0.470, 0.12, 0.120, 0.058)
        self.B["PLANT"] = (0.250, 0.16, 0.150, 0.16)

    # ---- drawing helpers ----
    def _box(self, key, title, sub, color, active=0.0, glow=False, tcol="white"):
        plt = self.plt; x, y, w, h = self.B[key]
        ec = color if active > 0.05 or glow else (0.4, 0.4, 0.45)
        self.ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=_blend(color, active),
                                        edgecolor=ec, lw=2.2 if (active > 0.5 or glow) else 1.1,
                                        zorder=3, path_effects=_glow(color, 5) if glow else None))
        self.ax.text(x + w / 2, y + h * (0.66 if sub else 0.5), title, ha="center", va="center",
                     color=tcol, fontsize=8.6, weight="bold", zorder=4)
        if sub:
            self.ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                         color=(0.85, 0.85, 0.9), fontsize=6.6, zorder=4)

    def _anchor(self, key, side):
        x, y, w, h = self.B[key]
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2),
                "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]

    def _edge(self, k0, s0, k1, s1, color, active, phase, lw=1.4, bend=False):
        p0 = self._anchor(k0, s0); p1 = self._anchor(k1, s1)
        a = 0.12 + 0.7 * active
        if bend:
            xm = (p0[0] + p1[0]) / 2
            xs = [p0[0], xm, xm, p1[0]]; ys = [p0[1], p0[1], p1[1], p1[1]]
            self.ax.plot(xs, ys, color=color, lw=lw + 1.4 * active, alpha=a, zorder=1, solid_capstyle="round")
            pts = np.array(list(zip(xs, ys)))
        else:
            self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw + 1.4 * active,
                         alpha=a, zorder=1, solid_capstyle="round")
            pts = np.array([p0, p1])
        if active > 0.15:                                  # moving packet
            seglens = np.hypot(*np.diff(pts, axis=0).T); total = seglens.sum() + 1e-9
            t = (phase % 1.0) * total; acc = 0
            for s in range(len(pts) - 1):
                if acc + seglens[s] >= t:
                    f = (t - acc) / max(seglens[s], 1e-9)
                    pk = pts[s] * (1 - f) + pts[s + 1] * f
                    self.ax.scatter([pk[0]], [pk[1]], s=34, color="white", ec=color, lw=1.3, zorder=5)
                    break
                acc += seglens[s]

    def update(self, st):
        ax = self.ax; ax.clear(); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ph = (st["frame"] % 14) / 14.0
        thr = st["thresholds"]
        # channel activity = fleet-max S/h, clipped
        act = {ch: float(np.clip(max(st["fdi"][d]["S"][ch] / max(thr[ch], 1e-9)
                                     for d in range(self.N)), 0, 1.0)) for ch in CHANNELS}
        tripped = {ch: act[ch] >= 1.0 for ch in CHANNELS}
        if st["event"] is not None:
            self.exec_glow = 16
            self.last_action = st["log_tail"]
        eg = self.exec_glow / 16.0
        self.exec_glow = max(0, self.exec_glow - 1)

        # column headers
        for x, lab in [(0.082, "SENSORS"), (0.230, "CHANNEL DETECTORS"), (0.387, "FUSION"),
                       (0.565, "STATE MACHINE"), (0.775, "AUTONOMY EXEC")]:
            ax.text(x, 0.92, lab, ha="center", color=NEON, fontsize=9, weight="bold")
        ax.text(0.775, 0.50, "RESPONSE", ha="center", color=NEON, fontsize=9, weight="bold")
        ax.text(0.530, 0.45, "ACTUATORS", ha="center", color=NEON, fontsize=9, weight="bold")

        # sensors + channels + their edges
        for ch in CHANNELS:
            a = act[ch]
            self._box(f"S_{ch}", SENSOR_OF[ch], "", (0.45, 0.55, 0.7), 0.25 + 0.6 * a, tcol="white")
            self._box(f"C_{ch}", f"{ch} CUSUM", f"S/h {a:.2f}" + ("  TRIP" if tripped[ch] else ""),
                      FAULT_COLOR[ch], a, glow=tripped[ch])
            self._edge(f"S_{ch}", "r", f"C_{ch}", "l", (0.5, 0.6, 0.8), 0.3 + 0.6 * a, ph)
            self._edge(f"C_{ch}", "r", "FUSE", "l", FAULT_COLOR[ch], a, ph + 0.3)

        any_trip = any(tripped.values())
        self._box("FUSE", "FUSION /", "ISOLATION", NEON, 0.3 + 0.6 * any_trip, glow=any_trip)
        # fleet state machine
        cnt = {s: sum(v == s for v in st["health"].values()) for s in ("HEALTHY", "SUSPECT", "FAULTED", "DEAD")}
        x, y, w, h = self.B["STATE"]
        ax.add_patch(self.plt.Rectangle((x, y), w, h, facecolor=(0.08, 0.08, 0.11),
                                        edgecolor=NEON, lw=1.6, zorder=3))
        ax.text(x + w / 2, y + h - 0.022, "FDI STATE MACHINE", ha="center", color=NEON, fontsize=8.2, weight="bold", zorder=4)
        for i, s in enumerate(("HEALTHY", "SUSPECT", "FAULTED", "DEAD")):
            yy = y + h - 0.058 - i * 0.042
            ax.add_patch(self.plt.Rectangle((x + 0.012, yy - 0.016), 0.024, 0.028,
                                            color=STATE_COLORS[s], zorder=4))
            ax.text(x + 0.05, yy, f"{s}", color=STATE_COLORS[s], fontsize=7.6, va="center", zorder=4)
            ax.text(x + w - 0.02, yy, f"{cnt[s]}", color="white", fontsize=8.5, va="center", ha="right",
                    weight="bold", zorder=4)
        self._edge("FUSE", "r", "STATE", "l", NEON, 0.3 + 0.6 * any_trip, ph)
        self._edge("STATE", "r", "EXEC", "l", NEON, 0.3 + 0.7 * eg, ph)

        # executive + response blocks
        self._box("EXEC", "AUTONOMY", "deadline-aware value · RTA", NEON, 0.3 + 0.7 * eg, glow=eg > 0.3)
        recent = eg > 0.05
        ra = eg if ("reopt" in self.last_action or recent) else 0.0
        self._box("RECOV", "RECOVERY LADDER", "inspect·fuse·act", "#ff7a33", eg, glow=eg > 0.4 and "isolated" in self.last_action)
        self._box("REOPT", "RE-OPTIMIZER", "surrogate coverage", "#9b6bff", eg)
        self._box("ISLS", "ISL SCHEDULER", "links · relays", "#00e0c6", 0.3 + 0.5 * (len(st.get("data_to_chief", [])) > 0))
        self._edge("EXEC", "b", "RECOV", "r", "#ff7a33", eg, ph, bend=True)
        self._edge("EXEC", "b", "REOPT", "r", "#9b6bff", eg, ph + .2, bend=True)
        self._edge("EXEC", "b", "ISLS", "r", "#00e0c6", 0.3 + 0.5 * (len(st.get("data_to_chief", [])) > 0), ph + .4, bend=True)

        # actuators (lit by what is acting)
        rcs_on = float(any(st["fdi"][d]["fault"] == "ATT" and st["health"][d] in ("SUSPECT", "FAULTED")
                           for d in range(self.N)))
        tcm_on = float(st["is_tcm"]); cam_on = float(len(st.get("imaging", [])) > 0)
        com_on = float(len(st.get("data_to_chief", [])) > 0)
        self._box("A_RCS", "RCS THRUSTERS", "attitude / detumble", "#ff7a33", 0.25 + 0.7 * rcs_on, glow=rcs_on > 0)
        self._box("A_TCM", "TCM / Δv", "trajectory burns", GREEN, 0.25 + 0.7 * tcm_on, glow=tcm_on > 0)
        self._box("A_CAM", "CAMERA", "peer imaging", NEON, 0.25 + 0.7 * cam_on, glow=cam_on > 0)
        self._box("A_COM", "COMMS / ISL", "data → chief", "#ff5db1", 0.25 + 0.7 * com_on, glow=com_on > 0)
        for k, on in [("RECOV", eg), ("REOPT", eg), ("ISLS", com_on)]:
            self._edge(k, "l", "A_RCS", "r", (0.6, 0.6, 0.7), 0.2 + 0.5 * on, ph)
        self._edge("RECOV", "l", "A_CAM", "r", NEON, cam_on, ph + .3, bend=True)
        self._edge("ISLS", "l", "A_COM", "r", "#ff5db1", com_on, ph + .1, bend=True)

        # plant feedback loop: actuators -> plant -> sensors
        self._box("PLANT", "SPACECRAFT /", "FORMATION (plant)", (0.5, 0.5, 0.6), 0.4)
        self._edge("A_RCS", "l", "PLANT", "r", (0.6, 0.6, 0.7), 0.3, ph)
        self._edge("A_TCM", "l", "PLANT", "r", (0.6, 0.6, 0.7), 0.2 + 0.5 * tcm_on, ph + .2)
        # plant up to the first sensor (closes the loop)
        p0 = self._anchor("PLANT", "t"); p1 = self._anchor("S_PEER", "b")
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=(0.5, 0.6, 0.8),
                    lw=1.4, alpha=0.5, connectionstyle="arc3,rad=-0.25"), zorder=1)
        ax.text((p0[0] + p1[0]) / 2 - 0.02, (p0[1] + p1[1]) / 2 + 0.02, "observe", color=(0.5, 0.6, 0.8),
                fontsize=7, rotation=90)

        # banner: latest decision
        ax.add_patch(self.plt.Rectangle((0.02, 0.015), 0.96, 0.055, facecolor=(0.08, 0.08, 0.12),
                                        edgecolor=NEON if eg > 0 else AXIS_RGBA, lw=1.4, zorder=3))
        ax.text(0.03, 0.043, "LATEST FDIR DECISION:", color=NEON, fontsize=8.5, weight="bold", va="center", zorder=4)
        ax.text(0.20, 0.043, (self.last_action or "fleet nominal — monitoring")[:120],
                color=GREEN if eg > 0 else (0.6, 0.6, 0.6), fontsize=8.2, va="center", family="monospace", zorder=4)
        ax.text(0.985, 0.043, f"f{st['frame']}", color=AMBER, fontsize=8.5, va="center", ha="right", zorder=4)
