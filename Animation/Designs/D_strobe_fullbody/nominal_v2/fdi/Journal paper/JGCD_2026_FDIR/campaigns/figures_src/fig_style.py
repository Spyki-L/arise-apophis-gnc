#!/usr/bin/env python3
"""Shared journal figure style (JGCD-friendly): STIX serif, Okabe-Ito palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Okabe-Ito colourblind-safe palette
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00",
         purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", grey="#7F7F7F", black="#000000")
SERIES = [C["blue"], C["verm"], C["green"], C["orange"], C["purple"], C["sky"], C["grey"], C["yellow"]]

FULL_W = 6.5     # inches, \textwidth for new-aiaa journal
HALF_W = 3.35


def use_style(serif=True):
    plt.rcParams.update({
        "font.family": "STIXGeneral" if serif else "DejaVu Sans",
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.8,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.55,
        "grid.linestyle": ":",
        "lines.linewidth": 1.25,
        "axes.grid": True,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.75",
        "axes.axisbelow": True,
    })


# ── architecture-diagram helpers (match the ARISE block-diagram house style) ──
SALMON = "#F6CBC5"          # highlighted module fill
SALMON_EDGE = "#B03A2E"
GREYBOX = "#EFEFEF"
BOX_EDGE = "#2B2B2B"


def draw_box(ax, x, y, w, h, text, fc="white", ec=BOX_EDGE, lw=1.0, fs=7.6,
             weight="normal", dashed=False, align="center", pad=0.0, text_color="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.0,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw,
                                linestyle=(0, (4, 2.2)) if dashed else "solid",
                                mutation_aspect=1.0))
    ha = "center" if align == "center" else "left"
    tx = x + w / 2 if align == "center" else x + 0.015 + pad
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs,
            weight=weight, color=text_color, linespacing=1.25)


def arrow(ax, p0, p1, lw=1.0, color="black", style="-|>", ms=7.5, dashed=False,
          connstyle=None, shrinkA=1.0, shrinkB=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=color,
                                 linestyle=(0, (4, 2.4)) if dashed else "solid",
                                 connectionstyle=connstyle or "arc3,rad=0.0",
                                 shrinkA=shrinkA, shrinkB=shrinkB))


def alabel(ax, x, y, text, fs=6.8, ha="center", va="bottom", color="black", rot=0):
    ax.text(x, y, text, fontsize=fs, ha=ha, va=va, color=color, rotation=rot,
            linespacing=1.15)


def blank_axes(ax, xlim=(0, 1), ylim=(0, 1)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.axis("off"); ax.grid(False)
    return ax
