#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_apophis_fleet_lvlh.py
==========================

Single 3D LVLH-style plot for the Apophis RHIG fleet with TCM markers.

Run from Scenario/Apophis/, where you have:

    apophis_rendezvous_full_D0.npz
    ...
    apophis_rendezvous_full_D7.npz

Each NPZ contains:
    t        : (N,)   global time [s since departure]
    rD_N     : (N,3)  deputy ECI position [m]
    rC_N     : (N,3)  chief ECI position [m]
    rA_N     : (N,3)  Apophis ECI position [m]
    impulses : (Nb,4) [t, dv_R, dv_T, dv_N] in Hill [m/s] (optional)
"""

import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)


# ---------- USER CONTROLS ----------

# LVLH center: "chief" or "apophis"
LVLH_CENTER = "chief"   # LVLH origin at chief; change to "apophis" if desired

# Downsampling for plotting (for huge N)
MAX_POINTS_PER_TRAJ = 4000

# Trajectory / TCM styles
TRAJ_ALPHA = 0.9
TRAJ_LINEWIDTH = 2.0
TCM_MARKER = "o"
TCM_SIZE = 55

# Font size controls (journal-figure sizes)
TITLE_FONTSIZE      = 20
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 15
LEGEND_FONTSIZE     = 15

# View angle for the 3D plot (play with these)
ELEV_ANGLE = 25.0   # degrees
AZIM_ANGLE = 35.0   # degrees

# Figure size
FIGSIZE = (10, 8)

# -----------------------------------


# Colour-blind-safe deputy palette (Okabe-Ito based), matched to the
# navigation and FDI figures so the whole paper shares one look.
DEPUTY_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#8B4513",  # brown
    "#444444",  # dark grey
]


def _setup_journal_style() -> str:
    """
    Apply the same journal-quality matplotlib style used by the ARISE
    navigation and FDI figures.  Prefers the ``scienceplots`` package and
    silently falls back to a self-contained rcParams block otherwise.
    """
    try:
        import scienceplots  # noqa: F401  - importing registers the styles
        plt.style.use(["science", "no-latex", "grid"])
        applied = "scienceplots"
    except Exception:
        plt.style.use("default")
        applied = "matplotlib-fallback"

    plt.rcParams.update({
        "figure.dpi":        120,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "font.family":       "serif",
        "mathtext.fontset":  "dejavuserif",
        "axes.titlesize":    TITLE_FONTSIZE,
        "axes.labelsize":    AXIS_LABEL_FONTSIZE,
        "xtick.labelsize":   TICK_LABEL_FONTSIZE,
        "ytick.labelsize":   TICK_LABEL_FONTSIZE,
        "legend.fontsize":   LEGEND_FONTSIZE,
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "grid.linewidth":    0.6,
        "axes.linewidth":    0.9,
        "axes.axisbelow":    True,
        "lines.linewidth":   1.3,
        "legend.frameon":    True,
        "legend.framealpha": 0.92,
        "legend.edgecolor":  "0.7",
    })
    return applied


# Apply the journal style once, at import time.
_FLEET_PLOT_STYLE = _setup_journal_style()


def make_equal_axis(ax, X, Y, Z):
    """Set 3D axes to equal scale."""
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    z_min, z_max = np.min(Z), np.max(Z)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    half = 0.5 * max_range

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)


def get_lvlh_positions(data, center: str):
    """
    Return deputy positions in LVLH-like frame [km], centered at 'chief' or 'apophis'.

    center = "chief"   -> rD_N - rC_N
           = "apophis" -> rD_N - rA_N
    """
    rD = data["rD_N"]  # [m]

    if center == "chief":
        r0 = data["rC_N"]
    elif center == "apophis":
        r0 = data["rA_N"]
    else:
        raise ValueError(f"Unknown LVLH center: {center}")

    r_rel = rD - r0  # [m]
    return r_rel / 1e3  # [km]


def pick_indices_for_impulses(t, t_imp):
    """For each impulse time, pick the closest sample index."""
    t = np.asarray(t)
    t_imp = np.asarray(t_imp)

    idxs = []
    for ti in t_imp:
        idx = np.argmin(np.abs(t - ti))
        idxs.append(idx)
    return np.array(idxs, dtype=int)


def add_dv_table(fig, data_bundles):
    """
    Overlay a per-deputy delta-v summary table (injection burn and total
    maneuvering budget) on the right of the figure, styled to match the
    journal figures.

    Injection delta-v is the magnitude of the first (time-sorted) impulse;
    total delta-v is the sum of all impulse magnitudes for that deputy.
    """
    rows = []
    for k, data in enumerate(data_bundles):
        imp = np.asarray(data["impulses"], float).reshape(-1, 4)
        if imp.size:
            imp = imp[np.argsort(imp[:, 0])]
            mags = np.linalg.norm(imp[:, 1:4], axis=1)
            inj, tot = float(mags[0]), float(np.sum(mags))
        else:
            inj, tot = 0.0, 0.0
        rows.append([f"Deputy {k}", f"{inj:.3f}", f"{tot:.3f}"])

    col_labels = ["Deputy",
                  r"Injection $\Delta v$ (m/s)",
                  r"Total $\Delta v$ (m/s)"]

    # dedicated, invisible axes on the right so tight_layout leaves it alone
    tax = fig.add_axes([0.605, 0.30, 0.385, 0.30])
    tax.axis("off")
    tbl = tax.table(cellText=rows, colLabels=col_labels,
                    cellLoc="center", loc="center",
                    bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(LEGEND_FONTSIZE)

    header_color = "#0072B2"   # journal-palette blue
    row_alt      = "#EEF3F8"   # light blue-grey for alternating rows
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.65")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("white" if (r % 2 == 1) else row_alt)
    return tbl


def main():
    # Load NPZ bundles
    npz_files = sorted(glob.glob("apophis_rendezvous_full_D*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            "No 'apophis_rendezvous_full_D*.npz' files found in current directory."
        )

    print(f"Found {len(npz_files)} rendezvous bundles:")
    data_bundles = []
    for f in npz_files:
        print("  -", os.path.basename(f))
        data_bundles.append(np.load(f))

    # Create single LVLH figure
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=ELEV_ANGLE, azim=AZIM_ANGLE)

    n_colors = len(DEPUTY_COLORS)
    all_X, all_Y, all_Z = [], [], []

    for k, data in enumerate(data_bundles):
        color = DEPUTY_COLORS[k % n_colors]
        t = data["t"]

        # LVLH positions (km)
        r = get_lvlh_positions(data, LVLH_CENTER)

        # Downsample for plotting
        N = r.shape[0]
        if N > MAX_POINTS_PER_TRAJ:
            step = int(np.ceil(N / MAX_POINTS_PER_TRAJ))
            r_plot = r[::step]
        else:
            r_plot = r

        x, y, z = r_plot[:, 0], r_plot[:, 1], r_plot[:, 2]
        all_X.append(x)
        all_Y.append(y)
        all_Z.append(z)

        ax.plot(
            x,
            y,
            z,
            label=f"Deputy {k}",
            color=color,
            linewidth=TRAJ_LINEWIDTH,
            alpha=TRAJ_ALPHA,
        )

        # TCM markers
        if "impulses" in data.files and data["impulses"].size > 0:
            impulses = data["impulses"]  # (Nb,4): [t, dv_R, dv_T, dv_N]
            t_imp = impulses[:, 0]
            idxs = pick_indices_for_impulses(t, t_imp)
            r_tcm = r[idxs]  # in LVLH frame

            ax.scatter(
                r_tcm[:, 0],
                r_tcm[:, 1],
                r_tcm[:, 2],
                s=TCM_SIZE,
                marker=TCM_MARKER,
                edgecolors="white",
                linewidths=0.7,
                facecolors=color,
                alpha=0.95,
                label="TCM burns" if k == 0 else None,
            )

    # Assemble arrays for scaling
    all_X = np.concatenate(all_X)
    all_Y = np.concatenate(all_Y)
    all_Z = np.concatenate(all_Z)

    # LVLH origin marker (center spacecraft / body)
    ax.scatter(
        0.0,
        0.0,
        0.0,
        s=90,
        marker="*",
        color="black",
        label=f"{LVLH_CENTER.capitalize()} LVLH origin",
    )

    make_equal_axis(ax, all_X, all_Y, all_Z)

    # Axis labels & title
    center_label = "Chief" if LVLH_CENTER == "chief" else "Apophis"
    ax.set_xlabel(f"X ({center_label} LVLH) [km]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(f"Y ({center_label} LVLH) [km]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_zlabel(f"Z ({center_label} LVLH) [km]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f"Apophis Fleet – {center_label} LVLH Frame", fontsize=TITLE_FONTSIZE)

    # Tick font size
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    # Subtle grid + legend
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()

    # per-deputy delta-v summary table (added after tight_layout so its
    # fixed-position axes is not disturbed)
    add_dv_table(fig, data_bundles)

    plt.show()


if __name__ == "__main__":
    main()
