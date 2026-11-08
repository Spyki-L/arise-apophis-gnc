#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fleet_quicklook_plots.py  –  3-column fleet dashboards

Purpose
-------
Generate a set of 3-column “quicklook” dashboards for a deputy fleet using
previously saved NPZ logs (e.g. from an ARISE/ESPA multi-deputy run). Each
figure shows all deputies in a grid, with:

- 3 columns per row (number of rows auto-computed from the number of deputies),
- A single y-axis label on the leftmost column,
- X-tick labels only on a chosen “anchor” subplot (Dep-6 by default) plus the
  bottom row,
- Shared layout parameters via a common `tight_layout(rect=..., w_pad=..., h_pad=...)`.

Inputs
------
Expects a directory containing per-deputy NPZ logs named:

    dep??_logs.npz

where “??” is a 2-digit index (e.g. dep01_logs.npz, dep06_logs.npz). Each NPZ
is assumed to contain at least:

- `t_err`      : time stamps [s] for attitude error signals,
- `sigma_BR`   : 3×N attitude error MRPs (body vs reference),
- `omega_BR`   : 3×N body-rate errors [rad/s],
- `t_torque`   : time stamps [s] for torque logs,
- `torque_B`   : 3×N body-frame commanded torques [N·m],
- `t_navT`     : time stamps [s] for navigation truth/estimate comparisons,
- `r_nav`      : 3×N nav-estimated position [m],
- `r_true`     : 3×N true position [m],
- `v_nav`      : 3×N nav-estimated velocity [m/s],
- `v_true`     : 3×N true velocity [m/s].

If no `dep??_logs.npz` are found in `LOG_DIR`, the script raises a
FileNotFoundError.

What it plots
-------------
For all deputies found in `LOG_DIR`:

1) σ components
   - Figure title: “Attitude-Tracking Error (σ components)”.
   - Each subplot:
       * Plots σ₁, σ₂, σ₃ vs time from `sigma_BR` cropped to t ≤ `T_CUTOFF_S`.
       * Adds a legend [σ₁, σ₂, σ₃] on the first subplot only.

2) ω components
   - Figure title: “Body-Rate Error (ω components)”.
   - Each subplot:
       * Plots ω₁, ω₂, ω₃ vs time from `omega_BR` cropped to `T_CUTOFF_S`.
       * Adds a legend [ω₁, ω₂, ω₃] on the first subplot.

3) Attitude-error norm ‖σ‖
   - Figure title: “Attitude-Error Norm ‖σ_BR‖”.
   - Each subplot:
       * Plots ‖σ_BR‖₂ vs time (norm over the 3 components).

4) Commanded torques
   - Figure title: “Commanded Torques”.
   - Each subplot:
       * Plots Lx, Ly, Lz vs time from `torque_B`.
       * Adds a legend [Lx, Ly, Lz] on the first subplot.

5) Position-error norm ‖Δr_nav‖
   - Figure title: “Position-Error Norm ‖Δr_nav‖”.
   - Each subplot:
       * Plots ‖r_nav − r_true‖₂ vs time.

6) Velocity-error norm ‖Δv_nav‖
   - Figure title: “Velocity-Error Norm ‖Δv_nav‖”.
   - Each subplot:
       * Plots ‖v_nav − v_true‖₂ vs time.

Layout and styling
------------------
- `COLS` controls the number of columns (default: 3).
- `ROWS` is computed from the number of deputies.
- `SHOW_X_AT` (0-based index) selects which subplot keeps x-tick labels and the
  single x-axis label (“Time [s]”) — by default the Dep-6 subplot.
- All figures share:
    * Global font sizing (`BASE_FONT`, `TICK_FONT`),
    * A single y-axis label per figure in the left column only,
    * `plt.tight_layout(rect=LAYOUT_RECT, w_pad=W_PAD, h_pad=H_PAD)` to enforce
      consistent spacing and outer margins.
- Each subplot title is shifted slightly upward using `TITLE_PAD`, and a
  figure-level suptitle is placed at `FIG_TITLE_Y`.

Output
------
- If `SAVE_PNGS` is False (default):
    * All figures are shown interactively at the end via `plt.show()`.

- If `SAVE_PNGS` is True:
    * The script saves:
        - "sigma_components.png"
        - "omega_components.png"
        - "sigma_norm.png"
        - "torques.png"
        - "pos_error.png"
        - "vel_error.png"
      in the current working directory, and prints a small status message.

Usage
-----
Set `LOG_DIR` to the folder containing `dep??_logs.npz`, adjust `T_CUTOFF_S`
and `COLS` as desired, and run:

    python fleet_quicklook_plots.py

The script will auto-detect how many deputies are present, report the grid
size (ROWS × COLS), then generate the six fleet dashboards either as PNGs or
interactive figures depending on the `SAVE_PNGS` flag.
"""


from pathlib import Path
import math, numpy as np, numpy.linalg as la
import matplotlib.pyplot as plt


# ╔═══════════════  USER CONTROLS  ═══════════════╗
LOG_DIR     = Path("fleet_logs")          # folder with dep??_logs.npz
COLS        = 3                           # sub-plots per row
T_CUTOFF_S  = 7_530.0                     # x-range limit
SAVE_PNGS   = False                       # True → save PNGs, False → show

BASE_FONT   = 35                          # axis / title font
TICK_FONT   = 30                          # tick-label font
LAYOUT_RECT = (-0.05,- 0.03, 0.98, 1.05)    # common paddings (l,b,r,t) 
W_PAD, H_PAD = 0.1, 0.1                   # ⇦ spacing between sub-plots
TITLE_PAD   = 1.02                       # how high to position each subplot title
FIG_TITLE_Y = 0.97                       # y-position for the figure suptitle
# ╚══════════════════════════════════════════════╝


# ── global font ----------------------------------------------------------------
plt.rcParams.update({
    "font.size": BASE_FONT,
    "axes.titlesize": BASE_FONT,
    "axes.labelsize": BASE_FONT,
    "xtick.labelsize": TICK_FONT,
    "ytick.labelsize": TICK_FONT,
    "legend.fontsize": BASE_FONT - 5,
})

# ── load bundles ─────────────────────────────────────────────────────────────
paths = sorted(LOG_DIR.glob("dep??_logs.npz"))
if not paths:
    raise FileNotFoundError(f"No dep??_logs.npz found in {LOG_DIR}")
bundles = [np.load(p) for p in paths]

ROWS      = math.ceil(len(paths) / COLS)
SHOW_X_AT = 5   # zero-based index for Dep-6 subplot

print(f"[INFO] plotting {len(bundles)} deputies → grid {ROWS}×{COLS}")

# ── helpers ──────────────────────────────────────────────────────────────────
def dep_name(p): return f"Dep {int(p.stem[3:5]) + 1}"
def crop(t, *arrs):
    keep = t <= T_CUTOFF_S
    return (t[keep],) + tuple(a[keep] for a in arrs)

def tidy(ax, row, col, ylabel=None, show_x=False):
    if col == 0 and ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
    if show_x or row == ROWS - 1:
        ax.set_xlabel("Time [s]")
    else:
        ax.set_xlabel(""); ax.set_xticklabels([])
    ax.grid(True)

def fig_and_axes(title):
    fig = plt.figure(figsize=(COLS*5, ROWS*4), facecolor="white")
    fig.suptitle(title, y=FIG_TITLE_Y)
    return fig

def finish(fig, fname=None):
    fig.tight_layout(rect=LAYOUT_RECT, w_pad=W_PAD, h_pad=H_PAD)
    if SAVE_PNGS and fname:
        fig.savefig(fname, dpi=150)

# ═══════════════════════════════════════════════════════════════════════════
# 1) σ components
# ═══════════════════════════════════════════════════════════════════════════
fig = fig_and_axes("Attitude-Tracking Error  (σ components)")
for k,(p,d) in enumerate(zip(paths,bundles)):
    r,c = divmod(k, COLS)
    ax  = fig.add_subplot(ROWS, COLS, k+1)
    t, σ = crop(d["t_err"], d["sigma_BR"])
    ax.plot(t, σ); ax.set_xlim(0, T_CUTOFF_S)

    ax.set_title(dep_name(p))
    ax.title.set_y(TITLE_PAD)

    tidy(ax, r, c, ylabel="σ", show_x=(k==SHOW_X_AT))
    if k == 0:
        ax.legend(["σ₁","σ₂","σ₃"], loc="upper right")
finish(fig, "sigma_components.png")

# 2) ω components -----------------------------------------------------------
fig = fig_and_axes("Body-Rate Error  (ω components)")
for k,(p,d) in enumerate(zip(paths,bundles)):
    r,c = divmod(k, COLS)
    ax  = fig.add_subplot(ROWS, COLS, k+1)
    t, ω = crop(d["t_err"], d["omega_BR"])
    ax.plot(t, ω); ax.set_xlim(0, T_CUTOFF_S)

    ax.set_title(dep_name(p))
    ax.title.set_y(TITLE_PAD)

    tidy(ax, r, c, ylabel="ω [rad s⁻¹]", show_x=(k==SHOW_X_AT))
    if k == 0:
        ax.legend(["ω₁","ω₂","ω₃"], loc="upper right")
finish(fig, "omega_components.png")

# 3) ‖σ‖ norm ---------------------------------------------------------------
fig = fig_and_axes("Attitude-Error Norm  ‖σ_BR‖")
for k,(p,d) in enumerate(zip(paths,bundles)):
    r,c = divmod(k, COLS)
    ax  = fig.add_subplot(ROWS, COLS, k+1)
    t, σ = crop(d["t_err"], d["sigma_BR"])
    ax.plot(t, la.norm(σ, axis=1)); ax.set_xlim(0, T_CUTOFF_S)

    ax.set_title(dep_name(p))
    ax.title.set_y(TITLE_PAD)

    tidy(ax, r, c, ylabel="‖σ‖", show_x=(k==SHOW_X_AT))
finish(fig, "sigma_norm.png")

# 4) torques ----------------------------------------------------------------
fig = fig_and_axes("Commanded Torques")
for k,(p,d) in enumerate(zip(paths,bundles)):
    r,c = divmod(k, COLS)
    ax  = fig.add_subplot(ROWS, COLS, k+1)
    t, τ = crop(d["t_torque"], d["torque_B"])
    ax.plot(t, τ); ax.set_xlim(0, T_CUTOFF_S)

    ax.set_title(dep_name(p))
    ax.title.set_y(TITLE_PAD)

    tidy(ax, r, c, ylabel="N m", show_x=(k==SHOW_X_AT))
    if k == 0:
        ax.legend(["Lx","Ly","Lz"], loc="upper right")
finish(fig, "torques.png")

# 5) position-error norm ----------------------------------------------------
fig = fig_and_axes("Position-Error Norm  ‖Δr_nav‖")
for k,(p,d) in enumerate(zip(paths,bundles)):
    r,c = divmod(k, COLS)
    ax  = fig.add_subplot(ROWS, COLS, k+1)
    t, r_nav, r_true = crop(d["t_navT"], d["r_nav"], d["r_true"])
    ax.plot(t, la.norm(r_nav - r_true, axis=1)); ax.set_xlim(0, T_CUTOFF_S)

    ax.set_title(dep_name(p))
    ax.title.set_y(TITLE_PAD)

    tidy(ax, r, c, ylabel="m", show_x=(k==SHOW_X_AT))
finish(fig, "pos_error.png")

# 6) velocity-error norm ----------------------------------------------------
fig = fig_and_axes("Velocity-Error Norm  ‖Δv_nav‖")
for k,(p,d) in enumerate(zip(paths,bundles)):
    r,c = divmod(k, COLS)
    ax  = fig.add_subplot(ROWS, COLS, k+1)
    t, v_nav, v_true = crop(d["t_navT"], d["v_nav"], d["v_true"])
    ax.plot(t, la.norm(v_nav - v_true, axis=1)); ax.set_xlim(0, T_CUTOFF_S)

    ax.set_title(dep_name(p))
    ax.title.set_y(TITLE_PAD)

    tidy(ax, r, c, ylabel="m s⁻¹", show_x=(k==SHOW_X_AT))
finish(fig, "vel_error.png")

# ── finalize ────────────────────────────────────────────────────────────────
if SAVE_PNGS:
    print("[INFO] PNGs saved.")
else:
    plt.show()