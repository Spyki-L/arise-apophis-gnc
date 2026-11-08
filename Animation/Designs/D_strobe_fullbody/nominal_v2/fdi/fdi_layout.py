#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fdi_layout.py — shared free-form tile layout for the FDIR fault lab.

This is the single source of truth for HOW the nine views are laid out as movable tiles and
how a saved design (`fdi_layout.json`) is read/written. It is imported by BOTH:

  • interactive_fault_lab.py   — the STATIC designer you arrange + save with, and
  • live_fault_lab.py          — the LIVE runner (LAYOUT="custom") that renders the real
                                  simulation using whatever you saved.

Because both use the same builder here, what you see in the designer is exactly what you get
in the live run: same positions, sizes, headers, hidden panels and font scale.

Design (validated): each tile is a sub-figure placed by slicing one shared GridSpec (the only
matplotlib mechanism that actually confines sub-figures). Inside each tile, a reserved header
band sits above a nested content sub-figure, so the panel's own long title is dropped and our
clean header never overlaps the content. Global font scaling multiplies every numeric font
size while panel CONTENT is (re)drawn, leaving headers/buttons on their own explicit sizes.
"""
from __future__ import annotations
import os
import json
import matplotlib.text as mtext

# ── tile-placement grid + figure-fraction extent the grid may use ────────────
GX, GY = 120, 60                      # fine grid → near-continuous placement / snap
ML, MR, MT, MB = 0.004, 0.997, 0.994, 0.075   # left, right, top, bottom of the tile area
CTRL_TOP = 0.075                      # tiles stay above this; controls (designer) live below
MINW, MINH = 0.10, 0.10               # smallest a tile may be dragged
ACCENT = "#39d9ff"
HEADER_BASE = 12.5                    # header font at scale 1.0

DEFAULT_ORDER = ["ops", "architecture", "scope", "innovations", "subsystems",
                 "coverage", "deputy", "isl", "loss"]
DEFAULT_TITLES = {
    "ops": "W1 · MISSION OPS",         "architecture": "W4 · AUTONOMY BRAIN",
    "scope": "W2 · FDI SCOPE",         "innovations": "W7 · INNOVATION MONITOR",
    "subsystems": "W3 · SUBSYSTEMS",   "coverage": "W8 · COVERAGE & STROBE",
    "deputy": "W5 · DEPUTY DEEP-DIVE", "isl": "W6 · ISL NETWORK",
    "loss": "W9 · LOSS COMPENSATION",
}


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ── global, scoped font scaling ──────────────────────────────────────────────
_SCALE_ON = [False]
_SCALE_VAL = [1.0]


def install_font_patch():
    """Multiply every numeric fontsize by the live scale — but only while content is drawn."""
    if getattr(mtext.Text, "_fdi_fontpatch", False):
        return
    orig = mtext.Text.set_fontsize

    def patched(self, s):
        if _SCALE_ON[0] and isinstance(s, (int, float)):
            s = s * _SCALE_VAL[0]
        return orig(self, s)

    mtext.Text.set_fontsize = patched
    mtext.Text._fdi_fontpatch = True


def set_scale(v):
    _SCALE_VAL[0] = float(v)


def scale_on():
    _SCALE_ON[0] = True


def scale_off():
    _SCALE_ON[0] = False


# ── layout ↔ grid cell conversions ───────────────────────────────────────────
def cells(rect):
    x, y, w, h = rect
    fx0 = (x - ML) / (MR - ML); fx1 = (x + w - ML) / (MR - ML)
    ryt = (MT - (y + h)) / (MT - MB); ryb = (MT - y) / (MT - MB)
    c0 = int(_clamp(round(fx0 * GX), 0, GX - 1)); c1 = int(_clamp(round(fx1 * GX), c0 + 1, GX))
    r0 = int(_clamp(round(ryt * GY), 0, GY - 1)); r1 = int(_clamp(round(ryb * GY), r0 + 1, GY))
    return r0, r1, c0, c1


def rect_from_cells(r0, r1, c0, c1):
    x = ML + c0 / GX * (MR - ML); x1 = ML + c1 / GX * (MR - ML)
    ytop = MT - r0 / GY * (MT - MB); ybot = MT - r1 / GY * (MT - MB)
    return [x, ybot, x1 - x, ytop - ybot]


def snap(rect):
    return [round(v, 4) for v in rect_from_cells(*cells(rect))]


def default_layout(order=DEFAULT_ORDER):
    """The familiar 3×3 grid as figure-fraction rects (x, y, w, h), y measured from the bottom."""
    cols, rows = 3, 3
    gx, gy = 0.006, 0.010
    x0, x1 = 0.004, 0.996
    ytop, ybot = 0.994, CTRL_TOP + 0.004
    tw = (x1 - x0 - (cols - 1) * gx) / cols
    th = (ytop - ybot - (rows - 1) * gy) / rows
    out = {}
    for i, n in enumerate(order):
        r, c = divmod(i, cols)
        x = x0 + c * (tw + gx)
        y = ytop - r * (th + gy) - th
        out[n] = [round(x, 4), round(y, 4), round(tw, 4), round(th, 4)]
    return out


def layout_path(here):
    return os.path.join(here, "fdi_layout.json")


def load(path, order=DEFAULT_ORDER):
    """Return dict(layout, titles, shown, font_scale, found). Missing/garbage → defaults."""
    base = dict(layout=default_layout(order), titles=dict(DEFAULT_TITLES),
                shown={n: True for n in order}, font_scale=1.0, found=False)
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except FileNotFoundError:
        return base
    except Exception as e:
        print(f"[fdi_layout] load failed ({e}); using defaults")
        return base
    lay = default_layout(order)
    for n, v in d.get("layout", {}).items():
        if n in order:
            try:
                lay[n] = [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
            except Exception:
                pass
    titles = {n: d.get("titles", {}).get(n, DEFAULT_TITLES[n]) for n in order}
    shown = {n: bool(d.get("shown", {}).get(n, True)) for n in order}
    try:
        fs = float(d.get("font_scale", 1.0))
    except Exception:
        fs = 1.0
    return dict(layout=lay, titles=titles, shown=shown, font_scale=fs, found=True)


def save(path, layout, titles, shown, font_scale):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(layout=layout, titles=titles, shown=shown,
                       font_scale=float(font_scale)), f, indent=2)


def representative_state(engine, n_frames, preview_frame):
    """Advance the engine to a mid-mission frame (cheap; no rendering) and return that state."""
    target = int(_clamp(preview_frame, 0, n_frames - 1))
    st = None
    for fi in range(target + 1):
        st = engine.step(fi)
    return st


class TileBuilder:
    """Builds the free-form tiles (sub-figures + headers + embedded windows) on a figure.

    The caller owns the figure (and any clearing / control widgets). `make_window(name, engine,
    truth)` is the window factory from live_fault_lab._make_window.
    """

    def __init__(self, fig, make_window, engine, truth, order=DEFAULT_ORDER):
        self.fig = fig
        self.make_window = make_window
        self.engine = engine
        self.truth = truth
        self.order = order
        self.tiles = {}
        self.wins = []
        self.gs = None

    def build(self, layout, titles, shown, font_scale):
        self.tiles = {}
        self.wins = []
        self.gs = self.fig.add_gridspec(GY, GX, left=ML, right=MR, top=MT, bottom=MB,
                                        wspace=0.12, hspace=0.34)
        for n in self.order:
            if shown.get(n, True):
                self._make_tile(n, layout[n], titles.get(n, n), font_scale)

    def _make_tile(self, n, rect, title, font_scale):
        r0, r1, c0, c1 = cells(rect)
        outer = self.fig.add_subfigure(self.gs[r0:r1, c0:c1])
        outer.patch.set_facecolor("black")
        win = self._embed(n, outer)              # embed at FULL region height (no header-strip squish)
        hdr = outer.text(0.5, 0.992, title, ha="center", va="top", color=ACCENT,
                         fontsize=HEADER_BASE * font_scale, weight="bold", zorder=6)
        self.tiles[n] = dict(outer=outer, inner=outer, header=hdr, win=win)
        self.wins.append((n, win))

    def _embed(self, n, subfig):
        import matplotlib.pyplot as plt
        real = plt.figure
        plt.figure = lambda *a, **k: subfig                # window draws into THIS sub-figure
        try:
            win = self.make_window(n, self.engine, self.truth)
        finally:
            plt.figure = real
        try:
            subfig.suptitle("")                            # drop the window's own long title
        except Exception:
            pass
        return win

    def refresh(self, st, titles, font_scale):
        """Redraw every panel's content at the given font scale; re-assert clean headers."""
        import matplotlib.pyplot as plt
        set_scale(font_scale)
        scale_on()
        old_fs = plt.rcParams.get("font.size", 10)
        plt.rcParams["font.size"] = 10 * font_scale        # scale default-sized text (ticks) too
        try:
            for n, w in self.wins:
                try:
                    w.update(st)
                except Exception as e:
                    print(f"[{n}] update error: {e}", flush=True)
        finally:
            scale_off()
            plt.rcParams["font.size"] = old_fs
        for n, t in self.tiles.items():
            try:
                t["inner"].suptitle("")
            except Exception:
                pass
            t["header"].set_text(titles.get(n, n))
            t["header"].set_fontsize(HEADER_BASE * font_scale)
