#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_impulses_fleet_live.py
---------------------------

Purpose
-------
Interactive “live” fleet impulse viewer for sets of `viz_inputs_D*.npz`
bundles. It produces:

1) A stem-plot figure of |Δv| vs time since deployment for all deputies.
2) A matching summary table where the first column shows a colored glyph
   whose color matches the plotted marker for that deputy.

The script is designed for quick inspection and interactive tweaking,
not for batch report generation.

Inputs and expected NPZ format
------------------------------
By default, the script searches for:

    viz_inputs_D*.npz

in the current working directory (or uses any explicit list of NPZ paths
passed on the command line). Each NPZ is expected to contain at least:

- `dv_times` : 1D array of burn times [s].
- `dv_list`  : (N,3) array of Δv vectors [m/s].
- `best_t`   : 1-element array with reference epoch [s] (optional).
- `t0`       : 1-element array with reference epoch [s] (fallback if
               `best_t` is missing).

The loader `_load(path)` returns:

- `t_min` : time since deployment in minutes (dv_times − best_t)/60.
- `mags`  : |Δv| magnitudes for each burn.
- `inj`   : magnitude of the first burn (“injection Δv”) [m/s].
- `tot`   : total |Δv| sum over all burns [m/s].
- `n`     : number of burns.

If a file contains no burns (`dv_times.size == 0`), it will still be
included in the summary table with zeros and omitted from the stem plot.

Plotting behavior
-----------------
Figure 1 – Impulse history (fleet overview)
- Uses the default Matplotlib style (no custom grid layout beyond
  global rcParams).
- Each deputy:
    * Assigned a unique marker from `MARKER_CODES`:
          ["o","s","^","v","D","P","X","8",">","<"].
    * Assigned a color from a predefined list (C0…C9, tab:cyan, tab:pink,
      tab:olive, tab:brown).
    * Plotted via `_safe_stem(ax, t, mag, colour=..., marker=...)` which
      wraps `ax.stem` and handles older Matplotlib versions that don’t
      support `use_line_collection`.
- x-axis: “Time since Deployment [min]”.
- y-axis: “|Δv| [m/s]”.
- Title: “Impulse History – Fleet Overview”.
- **No legend** is drawn on this figure, by design.

Figure 2 – Fleet Δv summary table
- Built via `build_table_figure(rows)` from the same rows used to construct
  the stem plot.
- Table layout:
    * Column 0: Unicode glyph matching the deputy’s plot marker, colored
      with the same Matplotlib color (e.g. ●, ■, ▲, ▼, ◆, ⬟, ✖, ⬢, ▶, ◀).
    * Column 1: “Deputy {i}”.
    * Column 2: injection Δv (first burn) [m/s], formatted to 3 decimals.
    * Column 3: total Δv (sum of |Δv|) [m/s], formatted to 3 decimals.
    * Column 4: integer number of burns.
- Visual details:
    * Column widths tunable via `TABLE_COLWIDTHS`.
    * Row height scaled via `TABLE_ROW_SCALE`.
    * Zebra striping on table body rows.
    * Header row with a blue background and white bold text.
    * Figure title: “Fleet Δv summary”.

Styling and configuration
-------------------------
User-adjustable knobs at the top:

- `BASE_FSIZE`      : base font size for axes, labels, and titles.
- `TICK_FSIZE`      : font size for tick labels.
- `TABLE_COLWIDTHS` : relative column widths for the summary table.
- `TABLE_ROW_SCALE` : vertical scaling factor for table rows.

Matplotlib style:

- `plt.style.use("default")` resets to the default style.
- `plt.rcParams` tweaks:
    * Large fonts for readability.
    * Grid enabled (`axes.grid=True`).
    * `figure.dpi=110` for reasonable on-screen clarity.

Helper functions
----------------
- `_vec_mag(v)`:
    * Tries to import `Visualize._vec_mag` if available.
    * Falls back to `np.linalg.norm(v, axis=1)` if not.

- `_load(path)`:
    * Loads the NPZ, extracts time and Δv magnitudes, and returns derived
      injection/total Δv and burn count.

- `_safe_stem(ax, x, y, colour, marker)`:
    * Calls `ax.stem` with the desired color/marker.
    * Includes a fallback path for older Matplotlib versions that lack
      `use_line_collection`.

CLI / interactive usage
-----------------------
The main entry point is:

    def main(argv: list[str] | None = None):

If called as a script:

    python plot_impulses_fleet_live.py

- If no paths are given on the command line, `main` uses:
      sorted(glob.glob("viz_inputs_D*.npz")).
- If no files match, it raises:
      RuntimeError("No viz_inputs_D*.npz found.").

Running as `__main__`:

- Enables interactive mode (`plt.ion()`).
- Calls `main(sys.argv[1:])` to get `(fig_imp, fig_tbl)`.
- Shows the figures non-blocking (`plt.show(block=False)`).
- Drops into a Python REPL via `code.interact`, with:
      fig_imp, fig_tbl, plt
  bound in the local namespace for manual inspection and tweaking.

This is intended for “live tuning”: you can, for example, change fonts,
colormap, or table styling interactively and redraw as needed.
"""


from __future__ import annotations
import sys, glob, os, code
import numpy as np
import matplotlib.pyplot as plt

# ────────── USER KNOBS ──────────
BASE_FSIZE       = 35
TICK_FSIZE       = 30
TABLE_COLWIDTHS  = [0.12, 0.22, 0.28, 0.28, 0.10]   # glyph | deputy | inj | tot | burns
TABLE_ROW_SCALE  = 2.5

# ────────── STYLE ──────────
plt.style.use("default")
plt.rcParams.update(
    {"font.size": BASE_FSIZE,
     "axes.titlesize": BASE_FSIZE,
     "axes.labelsize": BASE_FSIZE,
     "axes.grid": True,
     "figure.dpi": 110})

# ────────── |Δv| helper ──────────
try:
    from Visualize import _vec_mag          # type: ignore
except ImportError:
    def _vec_mag(v: np.ndarray) -> np.ndarray:
        return np.linalg.norm(v, axis=1)

# ────────── marker → glyph map ──────────
MARKER_CODES = ["o", "s", "^", "v", "D", "P", "X", "8", ">", "<"]
GLYPHS = {          # unicode look-alikes
    "o": "●",
    "s": "■",
    "^": "▲",
    "v": "▼",
    "D": "◆",
    "P": "⬟",
    "X": "✖",
    "8": "⬢",       # octagon
    ">": "▶",
    "<": "◀",
}

# ────────── loader & safe-stem ──────────
def _load(path: str):
    d = np.load(path, allow_pickle=True)
    dv_times, dv_list = d["dv_times"], d["dv_list"]
    if dv_times.size == 0:
        return np.array([]), np.array([]), 0, 0, 0
    best_t = float(d.get("best_t", d["t0"])[0])
    t_min  = (dv_times - best_t) / 60.0
    mags   = _vec_mag(dv_list)
    return t_min, mags, mags[0], mags.sum(), len(mags)

def _safe_stem(ax, x, y, *, colour, marker):
    try:
        ax.stem(x, y, linefmt=f"{colour}-", markerfmt=f"{colour}{marker}",
                basefmt=" ", use_line_collection=True,
                linewidth=1.6, markersize=6)
    except TypeError:                       # mpl ≤3.1
        cont = ax.stem(x, y, linefmt=f"{colour}-", markerfmt=f"{colour}{marker}",
                       basefmt=" ")
        plt.setp(cont.markerline, markersize=6, color=colour)
        plt.setp(cont.stemlines,  linewidth=1.6, color=colour)
        plt.setp(cont.baseline,   linewidth=0)

# ────────── build figures ──────────
def build_impulse_figure(paths):
    colours = ("C0 C1 C2 C3 C4 C5 C6 C7 C8 C9 tab:cyan tab:pink "
               "tab:olive tab:brown").split()
    labels  = [f"Deputy {i+1}" for i in range(len(paths))]
    fig, ax = plt.subplots(figsize=(16, 12))
    rows = []

    for k, p in enumerate(paths):
        marker = MARKER_CODES[k % len(MARKER_CODES)]
        colour = colours[k % len(colours)]
        t, mag, inj, tot, n = _load(p)
        rows.append((GLYPHS[marker], colour, labels[k], inj, tot, n))
        if t.size:
            _safe_stem(ax, t, mag, colour=colour, marker=marker)

    ax.set_xlabel("Time since Deployment [min]")
    ax.set_ylabel("|Δv|  [m/s]")
    ax.set_title("Impulse History – Fleet Overview")
    ax.tick_params(labelsize=TICK_FSIZE)
    fig.tight_layout()
    return fig, rows


def build_table_figure(rows):
    headers = ["", "Deputy", "Injection Δv (m/s)", "Total Δv (m/s)", "Burns"]
    fmt_rows = [
        [glyph, dep, f"{inj:.3f}", f"{tot:.3f}", f"{n:d}"]
        for glyph, _col, dep, inj, tot, n in rows
    ]

    fig_h = 0.9 + 0.6 * len(fmt_rows)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.axis("off")

    zebra = [["#f7f7f7" if i % 2 else "#ececec"] * len(headers)
             for i in range(len(fmt_rows))]

    tbl = ax.table(cellText=fmt_rows,
                   cellColours=zebra,
                   colLabels=headers,
                   colWidths=TABLE_COLWIDTHS,
                   colLoc="center", cellLoc="center",
                   loc="center")
    tbl.auto_set_font_size(False)

    # header
    for k in range(len(headers)):
        h = tbl[(0, k)]
        h.set_facecolor("#4f81bd")
        h.get_text().set(color="white", weight="bold", fontsize=BASE_FSIZE)

    # body font + colour first col
    for i, (_, colour, _, _, _, _) in enumerate(rows, start=1):
        for j in range(len(headers)):
            tbl[(i, j)].get_text().set_fontsize(TICK_FSIZE)
        tbl[(i, 0)].get_text().set_color(colour)

    tbl.scale(1, TABLE_ROW_SCALE)
    tbl.auto_set_column_width(col=range(len(headers)))
    ax.set_title("Fleet Δv summary",
                 pad=12, fontsize=BASE_FSIZE, fontweight="bold")
    fig.tight_layout()
    return fig

# ────────── main / CLI ──────────
def main(argv: list[str] | None = None):
    paths = argv or sorted(glob.glob("viz_inputs_D*.npz"))
    if not paths:
        raise RuntimeError("No viz_inputs_D*.npz found.")
    fig_imp, rows = build_impulse_figure(paths)
    fig_tbl = build_table_figure(rows)
    return fig_imp, fig_tbl

if __name__ == "__main__":
    plt.ion()
    fig_imp, fig_tbl = main(sys.argv[1:])
    plt.show(block=False)

    code.interact(local=dict(fig_imp=fig_imp, fig_tbl=fig_tbl, plt=plt))
