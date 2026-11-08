#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_fdi_mc_plot.py
============================

Pretty plotting for the Apophis FDI Monte-Carlo campaign.

This script reads:

    fdi_mc_results.npz

produced by `apophis_swarm_fdi_mc.py` and generates:

  - fdi_mc_summary.png
        Pd & Pfa vs scenario (NAV / ATT / PROP / COMM)

  - fdi_mc_delay_combined.png
        combined detection-delay distributions (overlaid PDFs + boxplots)

  - fdi_mc_phase_<SC>.png
        fault-phase vs detection-delay scatter per scenario SC

Run from the same directory that contains `fdi_mc_results.npz`:

    python apophis_swarm_fdi_mc_plot.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

# ────────── global plotting controls ──────────

TITLE_FONTSIZE      = 20
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE     = 12

plt.rcParams.update({
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "xtick.labelsize": TICK_LABEL_FONTSIZE,
    "ytick.labelsize": TICK_LABEL_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})

# number of deputies used in the MC (needed for Pfa)
N_DEP = 8

# consistent colours by scenario / channel
SCENARIO_COLORS = {
    "NOMINAL": "0.6",
    "NAV":     "tab:blue",
    "ATT":     "tab:green",
    "PROP":    "tab:orange",
    "COMM":    "tab:purple",
}

# ────────── Scenario / Stats containers ──────────

class Scenario(Enum):
    NOMINAL = auto()
    NAV     = auto()
    ATT     = auto()
    PROP    = auto()
    COMM    = auto()


@dataclass
class ScenarioStats:
    name: str
    n_trials: int
    n_faulty_deps: int
    n_detected_correct: int
    n_missed: int
    n_false_alarm: int
    det_delays: np.ndarray           # all detection delays [s]
    inj_phase_detected: np.ndarray   # phase for detected faults
    det_delay_detected: np.ndarray   # delays for detected faults [s]
    inj_phase_missed: np.ndarray     # phase for missed faults
    Pd: float                        # detection probability
    Pfa: float                       # false alarm per healthy dep


# ────────── loading NPZ results ──────────

SCRIPT_DIR = Path(__file__).resolve().parent
NPZ_PATH   = SCRIPT_DIR / "fdi_mc_results.npz"


def load_stats_from_npz(path: Path) -> dict[Scenario, ScenarioStats]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot find {path}. Run apophis_swarm_fdi_mc.py first."
        )

    data = np.load(path, allow_pickle=True)

    stats: dict[Scenario, ScenarioStats] = {}

    for sc in Scenario:
        prefix = sc.name
        n_trials           = int(data[f"{prefix}_n_trials"])
        n_faulty_deps      = int(data[f"{prefix}_n_faulty_deps"])
        n_detected_correct = int(data[f"{prefix}_n_detected_correct"])
        n_missed           = int(data[f"{prefix}_n_missed"])
        n_false_alarm      = int(data[f"{prefix}_n_false_alarm"])

        det_delays         = np.asarray(data[f"{prefix}_det_delays"], dtype=float)
        inj_phase_det      = np.asarray(data[f"{prefix}_inj_phase_detected"], dtype=float)
        det_delay_det      = np.asarray(data[f"{prefix}_det_delay_detected"], dtype=float)
        inj_phase_missed   = np.asarray(data[f"{prefix}_inj_phase_missed"], dtype=float)

        if n_faulty_deps > 0:
            Pd = n_detected_correct / n_faulty_deps
        else:
            Pd = 0.0

        # approximate number of healthy deputies per trial:
        if sc is Scenario.NOMINAL:
            n_healthy = n_trials * N_DEP
        else:
            n_healthy = n_trials * (N_DEP - 1)

        Pfa = n_false_alarm / max(n_healthy, 1)

        stats[sc] = ScenarioStats(
            name=sc.name,
            n_trials=n_trials,
            n_faulty_deps=n_faulty_deps,
            n_detected_correct=n_detected_correct,
            n_missed=n_missed,
            n_false_alarm=n_false_alarm,
            det_delays=det_delays,
            inj_phase_detected=inj_phase_det,
            det_delay_detected=det_delay_det,
            inj_phase_missed=inj_phase_missed,
            Pd=Pd,
            Pfa=Pfa,
        )

    return stats


# ────────── plotting helpers ──────────

def plot_summary(stats: dict[Scenario, ScenarioStats]) -> None:
    """Pd & Pfa vs scenario (two stacked panels)."""
    scenarios = [Scenario.NAV, Scenario.ATT, Scenario.PROP, Scenario.COMM]
    labels = [sc.name for sc in scenarios]
    colors = [SCENARIO_COLORS[sc.name] for sc in scenarios]

    Pd_vals  = [stats[sc].Pd for sc in scenarios]
    Pfa_vals = [stats[sc].Pfa for sc in scenarios]

    x = np.arange(len(scenarios))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Pd
    ax1.bar(x, Pd_vals, color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_ylabel(r"Detection probability $P_d$")
    ax1.set_ylim(0, 1.05)
    for xi, val in zip(x, Pd_vals):
        ax1.text(xi, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax1.set_title("FDI Monte Carlo – Detection and False Alarm Performance")

    # Pfa
    ax2.bar(x, Pfa_vals, color=colors, edgecolor="black", linewidth=0.6)
    ax2.set_ylabel(r"False alarm rate $P_{fa}$\n(per healthy deputy)")
    ax2.set_xlabel("Fault scenario")
    y_off = max(Pfa_vals) * 0.05 if max(Pfa_vals) > 0 else 0.001
    for xi, val in zip(x, Pfa_vals):
        ax2.text(
            xi, val + y_off,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = SCRIPT_DIR / "fdi_mc_summary.png"
    fig.savefig(out, dpi=300)
    print(f"[plot] {out.name}")


def plot_delay_distributions_combined(stats: dict[Scenario, ScenarioStats]) -> None:
    """
    Combined detection-delay distributions:

      - Top: overlaid, semi-transparent PDFs (normalized histograms).
      - Bottom: boxplots per scenario (NAV / ATT / PROP only).
    """
    # 🔸 Only include NAV / ATT / PROP in the combined delay view
    scenarios = [Scenario.NAV, Scenario.ATT, Scenario.PROP]

    # collect delays (in minutes) for scenarios with detections
    delays_by_sc: dict[Scenario, np.ndarray] = {}
    for sc in scenarios:
        st = stats[sc]
        if st.det_delays.size == 0:
            continue
        delays_by_sc[sc] = st.det_delays / 60.0  # [min]

    if not delays_by_sc:
        print("[plot] No detections; skipping combined delay figure.")
        return

    max_delay = max(d.max() for d in delays_by_sc.values())
    bins = np.linspace(0, max_delay * 1.05, 30)

    fig, (ax_pdf, ax_box) = plt.subplots(2, 1, figsize=(8, 7))

    # --- top: PDFs (normalized histograms) ---
    medians = {}

    for sc in scenarios:
        if sc not in delays_by_sc:
            continue
        dmin = delays_by_sc[sc]
        color = SCENARIO_COLORS[sc.name]

        ax_pdf.hist(
            dmin,
            bins=bins,
            density=True,
            alpha=0.28,
            color=color,
            edgecolor="none",
            label=sc.name,
        )
        medians[sc] = np.median(dmin)

    # add vertical lines at medians after hist to get proper y-limits
    y_top = ax_pdf.get_ylim()[1]
    for sc, med in medians.items():
        color = SCENARIO_COLORS[sc.name]
        ax_pdf.axvline(
            med,
            0,
            y_top,
            color=color,
            linestyle="--",
            linewidth=1.2,
        )

    ax_pdf.set_ylabel("PDF (normalized)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_pdf.set_title("Detection delay distributions by fault scenario")
    ax_pdf.legend(loc="upper right")

    # annotate basic stats
    summary_lines = []
    for sc in scenarios:
        if sc not in delays_by_sc:
            continue
        dmin = delays_by_sc[sc]
        text = f"{sc.name}: med={np.median(dmin):.1f} min, 90%={np.percentile(dmin,90):.1f}"
        summary_lines.append(text)
    ax_pdf.text(
        0.01,
        0.98,
        "\n".join(summary_lines),
        transform=ax_pdf.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # --- bottom: boxplots per scenario ---
    selected = [sc for sc in scenarios if sc in delays_by_sc]
    box_data = [delays_by_sc[sc] for sc in selected]
    positions = np.arange(1, len(selected) + 1)

    bp = ax_box.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
        ),
    )

    for patch, sc in zip(bp["boxes"], selected):
        patch.set_facecolor(SCENARIO_COLORS[sc.name])
        patch.set_alpha(0.4)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    for median_line in bp["medians"]:
        median_line.set_color("black")
        median_line.set_linewidth(1.2)

    ax_box.set_xticks(positions)
    ax_box.set_xticklabels([sc.name for sc in selected])
    ax_box.set_ylabel("Detection delay [min]", fontsize=AXIS_LABEL_FONTSIZE)
    ax_box.set_xlabel("Fault scenario")

    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = SCRIPT_DIR / "fdi_mc_delay_combined.png"
    fig.savefig(out, dpi=300)
    print(f"[plot] {out.name}")


def plot_phase_vs_delay(stats: dict[Scenario, ScenarioStats]) -> None:
    """
    Fault injection phase vs detection delay.

    Detected faults: coloured dots (colour = delay).
    Slowest 10 % of detections: open circles.
    Missed faults: black crosses along a top "miss" band.
    """
    for sc in [Scenario.NAV, Scenario.ATT, Scenario.PROP, Scenario.COMM]:
        st = stats[sc]
        phases_det = st.inj_phase_detected
        delays_det = st.det_delay_detected
        phases_miss = st.inj_phase_missed

        if phases_det.size == 0 and phases_miss.size == 0:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))

        # convert to useful units
        x_det = phases_det * 100.0           # percent of nav arc
        y_det = delays_det / 60.0            # minutes

        # base scatter
        if x_det.size > 0:
            sca = ax.scatter(
                x_det,
                y_det,
                c=y_det,
                cmap="viridis",
                s=12,
                alpha=0.8,
                edgecolor="none",
            )
            cbar = fig.colorbar(sca, ax=ax)
            cbar.set_label("Detection delay [min]")

            y_max = float(y_det.max())
        else:
            y_max = 1.0

        # highlight slowest 10 %
        if y_det.size > 0:
            thr = np.percentile(y_det, 90.0)
            mask_crit = y_det >= thr
            if np.any(mask_crit):
                ax.scatter(
                    x_det[mask_crit],
                    y_det[mask_crit],
                    facecolors="none",
                    edgecolors="red",
                    s=80,
                    linewidths=1.2,
                    label="slowest 10%",
                )

        # misses as crosses along a band above max delay
        if phases_miss.size > 0:
            y_cross = np.full_like(phases_miss, y_max * 1.10)
            ax.scatter(
                phases_miss * 100.0,
                y_cross,
                marker="x",
                s=30,
                color="k",
                label="missed faults",
            )
            ax.set_ylim(0, y_max * 1.25)
        else:
            ax.set_ylim(0, y_max * 1.15)

        ax.set_xlabel("Fault injection phase in nav arc [%]")
        ax.set_ylabel("Detection delay [min]")
        ax.set_title(f"Fault phase vs detection delay – {sc.name} faults")
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right")

        # annotate counts
        text = (
            f"Detected: {phases_det.size}\n"
            f"Missed:   {phases_miss.size}"
        )
        ax.text(
            0.02, 0.98,
            text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        fig.tight_layout()
        out = SCRIPT_DIR / f"fdi_mc_phase_{sc.name}.png"
        fig.savefig(out, dpi=300)
        print(f"[plot] {out.name}")


# ────────── main entry ──────────

def main() -> None:
    print(f"Loading MC results from {NPZ_PATH} ...")
    stats = load_stats_from_npz(NPZ_PATH)

    plot_summary(stats)
    plot_delay_distributions_combined(stats)
    plot_phase_vs_delay(stats)

    print("MC plotting complete.")
    plt.show()


if __name__ == "__main__":
    main()
