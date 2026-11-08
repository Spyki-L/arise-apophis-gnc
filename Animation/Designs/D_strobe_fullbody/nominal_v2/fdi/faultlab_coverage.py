#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faultlab_coverage.py — W8: the live CLOSEST-APPROACH coverage & strobe map.

Highest-fidelity onboard visibility (incidence + range, near-side for the convex body)
for each ACTIVE deputy, mapped onto the asteroid two ways:
  * a 3-D asteroid coloured by stereo-coverage, with the Sun direction and the strobes
    lighting the SHADOWED (anti-sun) facets the deputies image — Design-D's dark-side
    imaging — and the per-deputy camera lines;
  * an unwrapped lat-lon map of the same coverage, with the strobe-lit dark side overlaid
    and the sub-solar / anti-solar points marked.

Plus per-deputy and overall coverage, and the fraction of the dark side the strobes light.
Recomputed at a light cadence; matplotlib only.
"""
from __future__ import annotations
import numpy as np
import numpy.linalg as la

from cockpit_ops import STATE_COLORS, NEON, GREEN, RED, AMBER, TXT, AXIS_RGBA, _glow, style2d, style3d
from faultlab_encounter import compute_cov_map

ORANGE = "#ff9a33"


def _facet_colors(cm):
    n = len(cm["count"]); col = np.zeros((n, 3))
    base_lit = np.array([0.32, 0.30, 0.27]); base_dark = np.array([0.07, 0.07, 0.10])
    col[:] = np.where(cm["lit"][:, None], base_lit, base_dark)
    col[cm["count"] == 1] = (0.20, 0.55, 1.0)            # mono-imaged = blue
    col[cm["stereo"]] = (0.10, 0.90, 0.45)              # stereo-imaged = green
    col[cm["strobe_lit"]] = (1.0, 0.60, 0.20)           # strobe lights the dark side = orange
    return col


class CoverageStrobeWindow:
    """W8 — 3-D + lat-lon CA coverage map with dark-side strobe illumination."""

    def __init__(self, engine, refresh=6):
        import matplotlib.pyplot as plt
        self.plt = plt; self.engine = engine; self.N = engine.N; self.refresh = refresh
        self._cm = None; self._last = -999
        self.fig = plt.figure(figsize=(15.2, 8.6)); self.fig.patch.set_facecolor("black")
        self.ax3 = self.fig.add_axes([0.005, 0.06, 0.47, 0.86], projection="3d")
        self.ax_map = self.fig.add_axes([0.545, 0.55, 0.44, 0.37])
        self.ax_bar = self.fig.add_axes([0.545, 0.085, 0.44, 0.34])
        self.fig.suptitle("ARISE / APOPHIS  ·  CA COVERAGE & STROBE MAP  —  per-deputy + overall · dark-side strobe imaging",
                          color=NEON, fontsize=13.5, weight="bold", y=0.975)

    def _covmap(self, st):
        if self._cm is None or (st["frame"] - self._last) >= self.refresh:
            stations = {d: st["ca_all"][d] for d in range(self.N) if st["health"][d] != "DEAD"}
            self._cm = compute_cov_map(self.engine.tr, stations) if stations else None
            self._last = st["frame"]
        return self._cm

    def update(self, st):
        plt = self.plt; cm = self._covmap(st)
        ax = self.ax3; ax.clear(); style3d(ax)
        if cm is None:
            ax.text2D(0.5, 0.5, "no active deputies", color=RED, transform=ax.transAxes, ha="center")
            return
        cent = cm["cent"]; col = _facet_colors(cm); R = np.percentile(la.norm(cent, axis=1), 95)
        ax.scatter(cent[:, 0], cent[:, 1], cent[:, 2], c=col, s=12, depthshade=True, edgecolors="none")
        # sun direction (light comes from +sun toward the body)
        sun = cm["sun"] * R * 1.9
        ax.quiver(sun[0], sun[1], sun[2], -sun[0]*0.4, -sun[1]*0.4, -sun[2]*0.4, color="#ffe08a", lw=2.5, arrow_length_ratio=0.25)
        ax.text(sun[0], sun[1], sun[2], "  SUN", color="#ffe08a", fontsize=8, weight="bold")
        # active deputies + their camera lines, and strobe beams onto the dark side
        dlit = cm["cent"][cm["strobe_lit"]]
        dark_centroid = dlit.mean(axis=0) if len(dlit) else np.zeros(3)
        for d in range(self.N):
            if st["health"][d] == "DEAD":
                continue
            p = st["ca_all"][d]; c = STATE_COLORS[st["health"][d]]
            ax.scatter([p[0]], [p[1]], [p[2]], c=c, s=44, marker="s", edgecolor="white", lw=0.4, depthshade=False)
            ax.text(p[0], p[1], p[2], f" D{d+1}", color=c, fontsize=7)
            ax.plot([p[0], 0], [p[1], 0], [p[2], 0], color=c, lw=0.5, alpha=0.35)
            if len(dlit) and (st["ca_all"][d] @ cm["sun"]) < 0:        # a deputy on the dark side strobes it
                ax.plot([p[0], dark_centroid[0]], [p[1], dark_centroid[1]], [p[2], dark_centroid[2]],
                        color=ORANGE, lw=1.4, alpha=0.5, ls=(0, (2, 2)))
        L = R * 2.0
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L); ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"Stereo coverage {cm['cov_overall']*100:.0f}%  ·  Dark side {cm['dark_lit_frac']*100:.0f}% strobe-lit",
                     color=NEON, fontsize=10, loc="left", y=0.965)
        # ── lat-lon coverage map ─────────────────────────────────────────────
        ax = self.ax_map; ax.clear(); style2d(ax, "UNWRAPPED COVERAGE MAP (lat-lon)", "Longitude [°]", "Latitude [°]")
        cval = np.clip(cm["count"], 0, 3)
        ax.scatter(cm["lon"], cm["lat"], c=cval, cmap="viridis", s=7, vmin=0, vmax=3)
        sl = cm["strobe_lit"]
        ax.scatter(cm["lon"][sl], cm["lat"][sl], c=ORANGE, s=8, label="Strobe-lit dark side")
        # sub-solar / anti-solar
        su = cm["sun"]; sub_lon = np.degrees(np.arctan2(su[1], su[0])); sub_lat = np.degrees(np.arcsin(np.clip(su[2], -1, 1)))
        ax.plot(sub_lon, sub_lat, "*", color="#ffe08a", ms=13, label="sub-solar")
        ax.plot(((sub_lon + 360) % 360) - 180, -sub_lat, "x", color="#88aaff", ms=8, label="anti-solar")
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.legend(fontsize=6.0, facecolor="black", labelcolor="white", edgecolor=AXIS_RGBA,
                  loc="lower center", ncol=3, framealpha=0.75, borderpad=0.3,
                  columnspacing=0.9, handletextpad=0.4)
        # ── per-deputy + overall bars ────────────────────────────────────────
        ax = self.ax_bar; ax.clear(); style2d(ax, "COVERAGE PER DEPUTY  +  OVERALL", "", "Facet fraction")
        ds = sorted(cm["cov_per_dep"]); vals = [cm["cov_per_dep"][d] for d in ds]
        cols = [STATE_COLORS[st["health"][d]] for d in ds]
        ax.bar([f"D{d+1}" for d in ds], vals, color=cols)
        ax.bar(["STEREO\nall"], [cm["cov_overall"]], color=GREEN)
        ax.bar(["DARK\nstrobe"], [cm["dark_lit_frac"]], color=ORANGE)
        ax.set_ylim(0, max(0.6, cm["cov_overall"] * 1.3, max(vals) * 1.3))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"Stereo {cm['cov_overall']*100:.0f}%  ·  Dark strobe-lit {cm['dark_lit_frac']*100:.0f}%  ·  "
                     f"{cm['n_facets']} facets", color=NEON, fontsize=9.5, loc="left")
