#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cockpit_ops.py — black mission-ops cockpit (FDI_anim aesthetic) for the live FDIR.

Black background, neon palette, a 3D chief-LVLH (Hill) view showing the COMPLETE
deputy trajectories as fading trails with D1..D8 labels, R-T and R-N projections,
and a telemetry stack (fleet health · predicted-CA coverage onboard vs ground ·
peer camera capture · onboard-vs-ground compute · autonomy event log). Driven by
the live MissionEngine. matplotlib only.
"""
from __future__ import annotations
import numpy as np
import numpy.linalg as la

STATE_COLORS = {"HEALTHY": "#00ff6a", "SUSPECT": "#ffcc00", "FAULTED": "#ff4444", "DEAD": "#888888"}
NEON = "#33bbff"; AMBER = "#ffcc00"; RED = "#ff4444"; GREEN = "#00ff6a"
GRID_RGBA = (1, 1, 1, 0.10); AXIS_RGBA = (1, 1, 1, 0.55); TXT = "white"


def _glow(c="black", lw=2.2):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=lw, foreground=c)]


def style2d(ax, title="", xl="", yl=""):
    ax.set_facecolor("black")
    ax.set_title(title, color=NEON, fontsize=10, weight="bold", loc="left")
    ax.set_xlabel(xl, color=TXT, fontsize=8); ax.set_ylabel(yl, color=TXT, fontsize=8)
    ax.tick_params(colors=TXT, labelsize=7)
    for s in ax.spines.values():
        s.set_color(AXIS_RGBA)
    ax.grid(True, color=GRID_RGBA)


def style3d(ax):
    ax.set_facecolor("black")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(TXT)
        try:
            axis.set_pane_color((0, 0, 0, 1.0)); axis._axinfo["grid"]["color"] = GRID_RGBA
        except Exception:
            pass
        axis.set_tick_params(labelsize=6, colors=(1, 1, 1, 0.6))


class OpsCockpit:
    def __init__(self, engine, truth, n_frames):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        self.plt = plt; self.engine = engine; self.tr = truth
        self.hist = {"t": [], "onb": [], "gt": [], "gnd": []}
        self.fig = plt.figure(figsize=(16, 9)); self.fig.patch.set_facecolor("black")
        self.ax3 = self.fig.add_axes([0.01, 0.34, 0.49, 0.56], projection="3d")
        self.ax_rt = self.fig.add_axes([0.045, 0.04, 0.20, 0.19])
        self.ax_rn = self.fig.add_axes([0.295, 0.04, 0.20, 0.19])
        self.ax_health = self.fig.add_axes([0.545, 0.71, 0.185, 0.19])
        self.ax_cov = self.fig.add_axes([0.805, 0.71, 0.185, 0.19])
        self.ax_cam = self.fig.add_axes([0.55, 0.38, 0.20, 0.24])
        self.ax_comp = self.fig.add_axes([0.785, 0.38, 0.205, 0.24])
        self.ax_evt = self.fig.add_axes([0.55, 0.05, 0.44, 0.26])
        # asteroid primitive for the capture inset
        from cockpit_viz import lumpy_ellipsoid, render_capture_mpl
        self.ast = lumpy_ellipsoid(); self._cap = render_capture_mpl
        # Hill-trajectory scale (km). The deputies dispense FROM the chief, so they
        # begin co-located (sep≈0) and fan out to a ~10-13 km imaging formation. A few
        # deputy tracks start a frame late on the common grid → np.interp holds a far
        # sample (pre-deploy artifact); clamp those frames back onto the chief so the
        # trails read as a clean expanding fan and the auto-scale isn't blown out.
        H = (truth["hill"] / 1000.).copy()
        sep = la.norm(H, axis=2)
        H[sep > 50.0] = 0.0                                   # pre-deploy / interp artifact → on chief
        self.Hkm = H; self.HA = truth["hill_A"] / 1000.
        self.lim = max(2.0, float(np.nanpercentile(la.norm(H, axis=2), 99.5)) * 1.25)
        self.fig.suptitle("ARISE / APOPHIS — ONBOARD FDIR MISSION OPS  ·  deploy → encounter",
                          color=NEON, fontsize=15, weight="bold", y=0.985)

    # ── 3D Hill + projections with complete fading trajectories ──────────────
    def _trails(self, st):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from matplotlib.collections import LineCollection
        fi = st["frame"]; ax = self.ax3; ax.clear(); style3d(ax)
        ax.scatter([0], [0], [0], c=NEON, s=40, marker="^", depthshade=False)   # chief
        ax.text(0, 0, self.lim*0.05, "CHIEF", color=NEON, fontsize=7, path_effects=_glow())
        # Apophis: in the Hill box at CA, an edge bearing during the long approach
        ap = self.HA[fi]; r_ap = float(la.norm(ap)); L = self.lim
        if r_ap <= L:
            ax.scatter([ap[1]], [ap[2]], [ap[0]], c="#cc9966", s=110, marker="o", depthshade=False,
                       edgecolor="white", linewidth=0.5)
            ax.text(ap[1], ap[2], ap[0]+L*0.05, "APOPHIS", color="#cc9966", fontsize=7.5,
                    weight="bold", path_effects=_glow())
            ap_draw = ap
        else:
            u = ap / r_ap * (L * 0.92)                        # bearing on the box edge
            ax.scatter([u[1]], [u[2]], [u[0]], c="#cc9966", s=60, marker=(3, 0, 0), depthshade=False)
            ax.text(u[1], u[2], u[0], f"  APOPHIS  {r_ap:,.0f} km", color="#cc9966", fontsize=7,
                    path_effects=_glow())
            ap_draw = u
        for d in range(self.engine.N):
            col = STATE_COLORS[st["health"][d]]
            P = self.Hkm[d, :fi+1]                                  # complete trajectory to now
            if len(P) > 1:
                seg = np.stack([P[:-1], P[1:]], 1)[:, :, [1, 2, 0]]  # (T,N,R) axis order
                a0 = np.linspace(0.05, 0.95, len(seg))               # fade old->new
                lc = Line3DCollection(seg, colors=[(*self.plt.matplotlib.colors.to_rgb(col), al) for al in a0],
                                      linewidths=1.8)
                ax.add_collection3d(lc)
            h = P[-1]; ax.scatter([h[1]], [h[2]], [h[0]], c=col, s=46, marker="s",
                                  edgecolor="white", linewidth=0.4, depthshade=False)
            ax.text(h[1], h[2], h[0]+self.lim*0.04, f"D{d+1}", color=col, fontsize=7.5,
                    weight="bold", path_effects=_glow())
            if st["capture"] and st["capture"]["deputy"] == d:
                ax.plot([h[1], ap_draw[1]], [h[2], ap_draw[2]], [h[0], ap_draw[0]],
                        color=NEON, lw=2.0, alpha=0.85, ls=(0, (4, 2)))
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L); ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("T (Along-track) [km]", fontsize=7, labelpad=-1)
        ax.set_ylabel("N (Cross-track) [km]", fontsize=7, labelpad=-1)
        ax.set_zlabel("R (Radial) [km]", fontsize=7, labelpad=-1)
        ax.set_title(f"Chief-LVLH Hill frame · t-to-CA {st['t_go']/3600:.2f} h",
                     color=NEON, fontsize=9, loc="left")
        # projections R-T and R-N
        for ax2, (a1, a2, xl, yl) in [(self.ax_rt, (1, 0, "T [km]", "R [km]")), (self.ax_rn, (2, 0, "N [km]", "R [km]"))]:
            ax2.clear(); style2d(ax2, f"Projection {yl[0]}-{xl[0]}", xl, yl)
            ax2.scatter([0], [0], c=NEON, s=25, marker="^")
            for d in range(self.engine.N):
                col = STATE_COLORS[st["health"][d]]; P = self.Hkm[d, :fi+1]
                ax2.plot(P[:, a1], P[:, a2], color=col, lw=1.0, alpha=0.5)
                ax2.plot(P[-1, a1], P[-1, a2], "s", color=col, ms=4)
                ax2.text(P[-1, a1], P[-1, a2], f" D{d+1}", color=col, fontsize=6)
            ax2.set_xlim(-L, L); ax2.set_ylim(-L, L); ax2.set_aspect("equal", "box")

    # ── telemetry panels ─────────────────────────────────────────────────────
    def _panels(self, st):
        plt = self.plt
        # health grid
        ax = self.ax_health; ax.clear(); ax.set_facecolor("black"); ax.axis("off")
        ax.set_title("FLEET HEALTH (FDI)", color=NEON, fontsize=10, weight="bold", loc="left")
        for d in range(self.engine.N):
            x, y = d % 4, 1 - d // 4; col = STATE_COLORS[st["health"][d]]
            ax.add_patch(plt.Rectangle((x*0.25+0.01, y*0.46+0.02), 0.22, 0.42, color=col, alpha=0.85))
            ax.text(x*0.25+0.12, y*0.46+0.23, f"D{d+1}\n{st['health'][d][:4]}", ha="center", va="center",
                    color="black", fontsize=7.5, weight="bold")
        # coverage
        ax = self.ax_cov; ax.clear(); style2d(ax, "PREDICTED COVERAGE", "frame", "stereo")
        if self.hist["t"]:
            ax.plot(self.hist["t"], self.hist["onb"], "-", color=NEON, lw=1.6, label="onboard")
        if self.hist["gt"]:
            ax.plot(self.hist["gt"], self.hist["gnd"], "s", color=RED, ms=6, label="ground")
        ax.set_ylim(0, 1); ax.legend(loc="lower right", fontsize=6, facecolor="black", labelcolor="white", edgecolor=AXIS_RGBA)
        # camera capture
        ax = self.ax_cam; ax.clear(); ax.set_facecolor("black"); ax.axis("off")
        if st["capture"]:
            ax.imshow(self._cap(self.ast, st["ca_all"][st["capture"]["deputy"]]), origin="lower", cmap="inferno")
            ax.set_title(f"D{st['capture']['deputy']+1} CAMERA CAPTURE", color=GREEN, fontsize=10, weight="bold", loc="left")
        else:
            ax.set_title("PEER CAMERA (idle)", color=(0.6, 0.6, 0.6), fontsize=10, loc="left")
        # compute
        ax = self.ax_comp; ax.clear(); style2d(ax, "ONBOARD vs GROUND COMPUTE", "", "ms (log)")
        if self.engine.compute:
            c = self.engine.compute[-1]
            ax.bar([0, 1], [c[1], c[2]], color=[NEON, RED], width=.6)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["onboard", "ground"], color=TXT, fontsize=8)
            ax.set_yscale("log"); ax.set_title(f"ONBOARD ×{c[2]/max(c[1],1e-3):.0f} faster",
                                               color=NEON, fontsize=10, weight="bold", loc="left")
        else:
            ax.text(0.5, 0.5, "awaiting event", color="0.5", ha="center", transform=ax.transAxes); ax.axis("off")
        # event log
        ax = self.ax_evt; ax.clear(); ax.set_facecolor("black"); ax.axis("off")
        ax.set_title("AUTONOMY EVENT LOG", color=NEON, fontsize=10, weight="bold", loc="left")
        for i, (fr, msg) in enumerate(self.engine.log[-7:]):
            ax.text(0.005, 0.86 - i*0.13, f"f{fr:>3}  {msg[:52]}", transform=ax.transAxes, color=GREEN,
                    fontsize=7.6, family="monospace", clip_on=True)

    def update(self, st):
        self.hist["t"].append(st["frame"]); self.hist["onb"].append(st["cov_onb"])
        if st["event"] and st["cov_gnd"] is not None:
            self.hist["gt"].append(st["frame"]); self.hist["gnd"].append(st["cov_gnd"])
        try:
            self._trails(st)
        except Exception:
            pass
        self._panels(st)        # always repaint panels (incl. the peer camera) even if the 3D trail fails
