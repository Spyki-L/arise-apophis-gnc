#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fleet_quicklook_plots.py
------------------------

Simple "quicklook" dashboards for GEO fleet logs.

Reads dep??_logs.npz from a fleet_logs/ directory and generates:

  1) Attitude-tracking error components σ_BR
  2) Body-rate error components ω_BR
  3) Attitude-error norm ‖σ_BR‖
  4) Commanded torques L_B
  5) Position-error norm ‖Δr_nav‖ = ‖r_nav - r_true‖
  6) Velocity-error norm ‖Δv_nav‖ = ‖v_nav - v_true‖

All plots:
  - use the actual log time stamps (no resampling),
  - drop any NaN / Inf entries,
  - drop any rows where |value| > BAD_LIMIT (to remove Basilisk 1e35 sentinels).
"""

from pathlib import Path
import math

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# user knobs
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).resolve().parent
LOG_DIR     = SCRIPT_DIR / "fleet_logs"   # where dep??_logs.npz live

COLS        = 3          # subplots per row
T_CUTOFF_S  = None       # None → use full span; or set e.g. 7500.0
SAVE_PNGS   = False      # True → save PNGs instead of interactive

BAD_LIMIT   = 1.0e20     # drop anything with |value| > BAD_LIMIT

BASE_FONT   = 16
TICK_FONT   = 13

# ---------------------------------------------------------------------------
# matplotlib defaults
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": BASE_FONT,
    "axes.titlesize": BASE_FONT,
    "axes.labelsize": BASE_FONT,
    "xtick.labelsize": TICK_FONT,
    "ytick.labelsize": TICK_FONT,
    "legend.fontsize": BASE_FONT - 2,
})

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_fleet_logs(log_dir: Path):
    paths = sorted(log_dir.glob("dep??_logs.npz"))
    if not paths:
        raise FileNotFoundError(f"[quicklook] no dep??_logs.npz found in {log_dir}")
    bundles = [np.load(p) for p in paths]
    return paths, bundles


def dep_name(path: Path) -> str:
    """Convert dep00_logs.npz → 'Dep 1' etc."""
    stem = path.stem  # dep00_logs
    idx = int(stem[3:5])
    return f"Dep {idx + 1}"


def grid_shape(n: int, cols: int = COLS):
    rows = math.ceil(n / cols)
    return rows, cols


def crop_and_clean(t, *arrays, t_max=None, bad_limit=BAD_LIMIT):
    """
    1) apply time cutoff
    2) align lengths
    3) drop rows with NaN/Inf or |value| > bad_limit
    """
    t = np.asarray(t, float)
    arrs = [np.asarray(a) for a in arrays]

    if t_max is not None:
        keep_t = t <= t_max
        t = t[keep_t]
        arrs = [a[keep_t] for a in arrs]

    if not arrs:
        return (t,)

    # enforce common length
    n = min(len(t), *(len(a) for a in arrs))
    t = t[:n]
    arrs = [a[:n] for a in arrs]

    # flatten and stack to build a joint "good" mask
    flat_cols = [a.reshape(n, -1) for a in arrs]
    stacked = np.hstack(flat_cols)

    good = np.isfinite(stacked)
    if bad_limit is not None:
        good &= np.abs(stacked) < bad_limit
    good = np.all(good, axis=1)

    t = t[good]
    arrs = [a[good] for a in arrs]
    return (t,) + tuple(arrs)


def tidy_axes(ax, row, col, rows, ylabel: str | None):
    if col == 0 and ylabel:
        ax.set_ylabel(ylabel)
    if row == rows - 1:
        ax.set_xlabel("Time [s]")
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    ax.grid(True, alpha=0.4)


def new_figure(title: str, rows: int, cols: int):
    fig = plt.figure(
        figsize=(cols * 4.5, rows * 3.0),
        facecolor="white"
    )
    fig.suptitle(title, y=0.97)
    return fig


def finish_fig(fig, filename: str | None = None):
    fig.tight_layout(rect=(0.03, 0.03, 0.99, 0.94))
    if SAVE_PNGS and filename is not None:
        fig.savefig(filename, dpi=150)


# ---------------------------------------------------------------------------
# individual figure builders
# ---------------------------------------------------------------------------

def fig_sigma_components(paths, bundles, t_max):
    rows, cols = grid_shape(len(bundles))
    fig = new_figure("Attitude-Tracking Error  σ_BR components", rows, cols)

    for k, (p, d) in enumerate(zip(paths, bundles)):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(rows, cols, k + 1)

        t, sigma = crop_and_clean(d["t_err"], d["sigma_BR"], t_max=t_max)
        if len(t) == 0:
            ax.set_title(dep_name(p))
            continue

        ax.plot(t, sigma)
        ax.set_title(dep_name(p))
        tidy_axes(ax, r, c, rows, ylabel="σ [-]" if c == 0 else None)
        if k == 0:
            ax.legend([r"σ₁", r"σ₂", r"σ₃"], loc="upper right")

    finish_fig(fig, "sigma_components.png")


def fig_omega_components(paths, bundles, t_max):
    rows, cols = grid_shape(len(bundles))
    fig = new_figure("Body-Rate Error  ω_BR components", rows, cols)

    for k, (p, d) in enumerate(zip(paths, bundles)):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(rows, cols, k + 1)

        t, omega = crop_and_clean(d["t_err"], d["omega_BR"], t_max=t_max)
        if len(t) == 0:
            ax.set_title(dep_name(p))
            continue

        ax.plot(t, omega)
        ax.set_title(dep_name(p))
        tidy_axes(ax, r, c, rows, ylabel=r"ω [rad s$^{-1}$]" if c == 0 else None)
        if k == 0:
            ax.legend([r"ω₁", r"ω₂", r"ω₃"], loc="upper right")

    finish_fig(fig, "omega_components.png")


def fig_sigma_norm(paths, bundles, t_max):
    rows, cols = grid_shape(len(bundles))
    fig = new_figure("Attitude-Error Norm  ‖σ_BR‖", rows, cols)

    for k, (p, d) in enumerate(zip(paths, bundles)):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(rows, cols, k + 1)

        t, sigma = crop_and_clean(d["t_err"], d["sigma_BR"], t_max=t_max)
        if len(t) == 0:
            ax.set_title(dep_name(p))
            continue

        ax.plot(t, la.norm(sigma, axis=1))
        ax.set_title(dep_name(p))
        tidy_axes(ax, r, c, rows, ylabel="‖σ‖ [-]" if c == 0 else None)

    finish_fig(fig, "sigma_norm.png")


def fig_torques(paths, bundles, t_max):
    rows, cols = grid_shape(len(bundles))
    fig = new_figure("Commanded Torques", rows, cols)

    for k, (p, d) in enumerate(zip(paths, bundles)):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(rows, cols, k + 1)

        t, tau = crop_and_clean(d["t_torque"], d["torque_B"], t_max=t_max)
        if len(t) == 0:
            ax.set_title(dep_name(p))
            continue

        ax.plot(t, tau)
        ax.set_title(dep_name(p))
        tidy_axes(ax, r, c, rows, ylabel="Torque [N m]" if c == 0 else None)
        if k == 0:
            ax.legend(["Lx", "Ly", "Lz"], loc="upper right")

    finish_fig(fig, "torques.png")


def fig_pos_error(paths, bundles, t_max):
    rows, cols = grid_shape(len(bundles))
    fig = new_figure("Position-Error Norm  ‖Δr_nav‖", rows, cols)

    for k, (p, d) in enumerate(zip(paths, bundles)):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(rows, cols, k + 1)

        if "t_navT" not in d or "r_nav" not in d or "r_true" not in d:
            ax.set_title(dep_name(p) + " (no navT)")
            continue

        t, r_nav, r_true = crop_and_clean(
            d["t_navT"], d["r_nav"], d["r_true"], t_max=t_max
        )
        if len(t) == 0:
            ax.set_title(dep_name(p))
            continue

        dr = r_nav - r_true
        ax.plot(t, la.norm(dr, axis=1))
        ax.set_title(dep_name(p))
        tidy_axes(ax, r, c, rows, ylabel=r"‖Δr‖ [m]" if c == 0 else None)

    finish_fig(fig, "pos_error.png")


def fig_vel_error(paths, bundles, t_max):
    rows, cols = grid_shape(len(bundles))
    fig = new_figure("Velocity-Error Norm  ‖Δv_nav‖", rows, cols)

    for k, (p, d) in enumerate(zip(paths, bundles)):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(rows, cols, k + 1)

        if "t_navT" not in d or "v_nav" not in d or "v_true" not in d:
            ax.set_title(dep_name(p) + " (no navT)")
            continue

        t, v_nav, v_true = crop_and_clean(
            d["t_navT"], d["v_nav"], d["v_true"], t_max=t_max
        )
        if len(t) == 0:
            ax.set_title(dep_name(p))
            continue

        dv = v_nav - v_true
        ax.plot(t, la.norm(dv, axis=1))
        ax.set_title(dep_name(p))
        tidy_axes(ax, r, c, rows, ylabel=r"‖Δv‖ [m s$^{-1}$]" if c == 0 else None)

    finish_fig(fig, "vel_error.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    paths, bundles = load_fleet_logs(LOG_DIR)

    # use attitude-error time base for overall span
    t0 = min(float(b["t_err"][0]) for b in bundles)
    t1 = max(float(b["t_err"][-1]) for b in bundles)
    t_max = T_CUTOFF_S if T_CUTOFF_S is not None else t1

    print(f"[quicklook] plotting {len(bundles)} deputies "
          f"over {t0:.1f}–{t_max:.1f} s from {LOG_DIR}")

    fig_sigma_components(paths, bundles, t_max)
    fig_omega_components(paths, bundles, t_max)
    fig_sigma_norm(paths, bundles, t_max)
    fig_torques(paths, bundles, t_max)
    fig_pos_error(paths, bundles, t_max)
    fig_vel_error(paths, bundles, t_max)

    if SAVE_PNGS:
        print("[quicklook] PNGs saved in current directory.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
