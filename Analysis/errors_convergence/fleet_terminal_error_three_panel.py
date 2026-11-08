#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fleet_terminal_error_three_panel.py

Purpose
-------
Generate a three-figure visualization of predicted terminal-separation error
for an Apophis-style deputy fleet, using compact `viz_inputs_D*.npz` bundles
as input. Each NPZ corresponds to a single deputy and contains a time history
of predicted terminal error norms in the arrival LVLH frame.

Inputs
------
The script searches the current directory for files matching:

    viz_inputs_D*.npz

using a pattern of the form `viz_inputs_D{idx}.npz`, where `{idx}` is a
zero-based deputy index. Each NPZ is expected to contain:

- t_hist   : 1D array of time samples [s].
- err_hist : 1D array of predicted terminal error norms [m] (same length as t_hist).
- best_t   : scalar or 1-element array giving the “deployment” or reference time [s].
- E_TRIGGER: scalar or 1-element array defining a trigger/error-threshold level [m].

High-level behavior
-------------------
1. Discover and load all `viz_inputs_D*.npz` files, keyed by deputy index.
2. For each deputy:
   - Convert time to “minutes since deployment”:
        t_rel_min = (t_hist − best_t) / 60
   - Plot the full error curve in a consistent color/marker.
   - Mark the final sample with an open marker and record that final ‖error‖
     value for later summarization.
   - Draw a single horizontal dashed line at `E_TRIGGER` as a visual threshold.

3. Figure 1 — Full curves (no legend)
   - X-axis: time since deployment [min], full window.
   - Y-axis: predicted terminal error norm ‖pred sep − target‖ [m].
   - All deputies overlaid with unique color/marker combinations.
   - No legend (kept clean; the legend is handled by Figure 3).

4. Figure 2 — Zoomed view
   - Re-plots the same curves and final markers as Figure 1.
   - X-axis limited to `ZOOM_XLIM_MIN` (e.g., 840–876 min ≈ 14.0–14.6 h).
   - Y-axis limited to `ZOOM_YLIM` (e.g., 0–650 m).
   - Used to highlight the late-time convergence/dispersion of the fleet.

5. Figure 3 — Summary table with colored glyph legend
   - Builds a table where each row corresponds to one deputy:
        [ colored glyph, "Deputy {idx+1}", final ‖error‖ (m) ]
   - Glyph column uses a set of Unicode symbols (●, ■, ▲, …) and colors 
     matched to the line colors in Figures 1 and 2.
   - Alternating row shading (“zebra” stripes) for readability.
   - Column headers: “”, “Deputy”, “Final ‖error‖”.
   - A bold title “Final terminal-error norms” at the top.

Styling and layout
------------------
- Global Matplotlib settings set large base fonts suitable for presentations:
  BASE_FONT, TICK_FONT, TABLE_FONT, HEADER_FONT, ICON_FONT.
- The color cycle uses `plt.cm.tab10` and a fixed sequence of marker shapes.
- Figures use `constrained_layout=True` to reduce manual spacing tweaks.

Usage
-----
Run in a directory that contains one or more `viz_inputs_D*.npz` files:

    python fleet_terminal_error_three_panel.py

The script will:
- Raise an error and exit cleanly if no matching NPZ files are found.
- Show three interactive figures in sequence:
  1) Full fleet curves.
  2) Zoomed-in fleet curves.
  3) Table summary / legend for final terminal-error norms.
"""


from pathlib import Path, PurePath, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ╔════════════════════════════  CONTROLS  ═══════════════════════════════╗
BASE_FONT     = 35
TICK_FONT     = 30
TABLE_FONT    = 35
HEADER_FONT   = 35
ICON_FONT     = 35
TABLE_COL_W   = [0.10, 0.20, 0.20]      # icon | Deputy | value
TABLE_ROW_H   = 2.5
ZOOM_XLIM_MIN = (840.0, 876.0)          # 14.0–14.6 h in minutes
ZOOM_YLIM     = (0.0, 650.0)
MARKERS       = ["o", "s", "^", "v", "D", "p", "X", "h"]
GLYPHS        = ["●", "■", "▲", "▼", "◆", "⬟", "✖", "⬣"]
COLORS        = list(plt.cm.tab10.colors)
# ╚═══════════════════════════════════════════════════════════════════════╝

# ── locate viz_inputs_D*.npz bundles ─────────────────────────────────────
pat = re.compile(r"viz_inputs_D(\d+)\.npz")
bundles = {int(m.group(1)): p for p in Path(".").glob("viz_inputs_D*.npz")
           if (m := pat.fullmatch(PurePath(p).name))}
if not bundles:
    raise SystemExit("✖  No viz_inputs_D*.npz bundles found.")

# ── global Matplotlib style ──────────────────────────────────────────────
mpl.rcParams.update({
    "font.size": BASE_FONT,
    "axes.prop_cycle": mpl.cycler(color=COLORS),
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": .5, "grid.alpha": .7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": TICK_FONT, "ytick.labelsize": TICK_FONT,
})

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return (d["t_hist"], d["err_hist"],
            float(d.get("best_t", [0.0])[0]),
            float(d.get("E_TRIGGER", [5.0])[0]))

# ── FIGURE 1 – full curves ───────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7.5, 3.5), constrained_layout=True)
rows, trigger_drawn = [], False

for idx, p in sorted(bundles.items()):
    th, eh, bt, etr = load_npz(p)
    tx_min = (th - bt) / 60.0                     # seconds → minutes
    col, mkr = COLORS[idx % len(COLORS)], MARKERS[idx % len(MARKERS)]

    ax1.plot(tx_min, eh, lw=2, color=col,
             marker=mkr, markevery=[-1], markerfacecolor="none")

    if not trigger_drawn:
        ax1.axhline(etr, ls="--", color="grey"); trigger_drawn = True

    rows.append((idx, eh[-1], col))

ax1.set_xlabel("Time since Deployment [min]")
ax1.set_ylabel("‖pred sep − target‖ [m]")
ax1.set_title("Predicted terminal error (arrival LVLH) — All Deputies")

# ── FIGURE 2 – zoomed view ───────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9.2, 4.1), constrained_layout=True)
for ln in ax1.get_lines():
    ax2.plot(ln.get_xdata(), ln.get_ydata(), lw=1.8, color=ln.get_color())
    ax2.plot(ln.get_xdata()[-1], ln.get_ydata()[-1],
             marker=ln.get_marker(), markersize=8,
             markeredgecolor=ln.get_color(), markerfacecolor="none")

xmin, xmax = ZOOM_XLIM_MIN
if xmin is None: xmin = ax2.get_xlim()[0]
if xmax is None: xmax = ax1.get_xlim()[1]
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(*ZOOM_YLIM)
ax2.set_xlabel("Time since Deployment [min]")
ax2.set_ylabel("‖pred sep − target‖ [m]")
ax2.set_title(f"Zoomed view  ({xmin:.0f}–{xmax:.0f} min, y ≤ {ZOOM_YLIM[1]:.0f} m)")

# ── FIGURE 3 – summary table  (Deputy 1 – Deputy 2 – …) ──────────────────
rows.sort(key=lambda r: r[0])
fig3 = plt.figure(figsize=(6.6, 0.6 + TABLE_ROW_H * 0.32 * len(rows)),
                  constrained_layout=True)
ax3 = fig3.add_subplot(111); ax3.axis("off")

table_txt = [
    [GLYPHS[i % len(GLYPHS)],
     f"Deputy {idx + 1}",          #  ← new one-based naming
     f"{err:.2f} m"]
    for i, (idx, err, _) in enumerate(rows)
]

tbl = ax3.table(cellText=table_txt,
                colLabels=["", "Deputy", "Final ‖error‖"],
                colWidths=TABLE_COL_W,
                colLoc="center", cellLoc="center",
                loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(TABLE_FONT)
tbl.scale(1.0, TABLE_ROW_H)

for (r, c), cell in tbl.get_celld().items():
    if r == 0:                   # header row
        cell.set_facecolor("#1f77b4")
        cell.get_text().set_color("white")
        cell.get_text().set_fontsize(HEADER_FONT)
        cell.get_text().set_weight("bold")
    else:
        if c == 0:               # glyph column
            colour = rows[r-1][2]
            cell.get_text().set_color(colour)
            cell.get_text().set_fontsize(ICON_FONT)
        if r % 2 == 0:
            cell.set_facecolor("#f2f2f2")  # zebra stripes

fig3.suptitle("Final terminal-error norms",
              fontsize=BASE_FONT + 3, weight="bold", y=0.99)

plt.show()
