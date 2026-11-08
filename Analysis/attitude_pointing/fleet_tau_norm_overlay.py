#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fleet_tau_norm_overlay.py
=========================

Purpose
-------
Create a **single, shared-axis** plot of torque norm ‖τ‖ for all deputies in an
ESPA + 8×XB6 scenario, using NPZ outputs from the fleet SE(3) controller runs.
Each deputy’s time-history appears as a separate colored curve, making it easy
to compare torque usage and engagement behavior at a glance.

Key behaviors
-------------
- Loads torque histories for all deputies from either:
    * A combined NPZ (e.g. `espa_xb6_8deps_*.npz`), or
    * Per-deputy NPZs (e.g. `*_D1.npz`, `*_D2.npz`, …).

- For **each deputy**, it picks the first available 3×N torque vector from:
    1. `tau_apply` (applied torque),
    2. `tau_cmd`   (commanded torque),
    3. `tau_des`   (desired torque).

- Uses the corresponding time array `t` (Dk_t or shared `t`) and plots:

      ‖τ(t)‖₂ = sqrt(τ_x² + τ_y² + τ_z²)

  on a single set of axes with one colored line per deputy.

- Optionally draws a vertical engagement-time marker, e.g. at `t = 10.3 s`.

- Produces a vertical legend (one column) placed **outside the axes** on the
  right-hand side, so it doesn’t occlude data.

- Prints a short report listing which τ-source was used per deputy.

I/O and data sources
--------------------
The script expects either:

1. A combined NPZ:

    --combined espa_xb6_8deps_YYYYMMDD_HHMMSS.npz

   where the file contains keys like:

      D1_t, D1_tau_apply, D1_tau_cmd, D1_tau_des, ...
      D2_t, D2_tau_apply, ...

   plus global keys for carrier, sched, etc.

   The number of deputies is inferred from keys matching:

      "Dk_t" for k ∈ {1,2,...}.

2. A set of per-deputy NPZs:

    --per-deputy "espa_xb6_8deps_*_D*.npz"

   The script wraps those files with a small `Proxy` accessor that allows
   uniform access using keys like `"D1_tau_apply"`, `"D2_t"`, etc.

If neither `--combined` nor `--per-deputy` is provided, the script will
attempt to auto-detect a combined file:

- Finds the latest `espa_xb6_8deps_*.npz` that does **not** end with `_D#.npz`.

CLI options
-----------
I/O:
- `--combined <path>`:
    Path to a combined NPZ (preferred).
- `--per-deputy <glob ...>`:
    One or more paths or glob patterns to per-deputy NPZs.

Visuals:
- `--engage <float>`:
    Engagement-time marker in seconds (default: 10.3 s).
    If provided, a vertical dashed line is drawn at this time.
- `--label-prefix <str>`:
    Prefix for legend labels (default: "Deputy ").
    Labels become e.g. "Deputy 1", "Deputy 2", etc.

Font sizes:
- `--fs-title <int>`   : Axes title font size (default: 35).
- `--fs-label <int>`   : Axes label font size (default: 30).
- `--fs-tick <int>`    : Tick-label font size (default: 30).
- `--fs-legend <int>`  : Legend font size (default: 30).

Global font settings are applied via `plt.rcParams.update(...)`.

Plot details
------------
- The script creates a figure (≈ 14×8 inches) and a single axes.
- For each deputy i ∈ {1..N}:
    * Attempts to load `t` from:
          "D{i}_t" or "t"
    * Attempts to load a torque vector from the first that exists among:
          "D{i}_tau_apply", "D{i}_tau_cmd", "D{i}_tau_des"
      or their un-prefixed equivalents ("tau_apply", etc. for single-dep files).
    * If found, plots ‖τ‖₂ vs t and records which field was used.

- The legend:
    * Title: "Deputy"
    * Vertical (ncol=1)
    * Placed outside on the right:
          bbox_to_anchor=(1.02, 0.5), loc="center left"
    * Requires `fig.subplots_adjust(right=0.8)` to leave space.

- Axes labels:
    * Title:            "Torque Norm ‖τ‖"
    * x-label:          "time [s]"
    * y-label:          "‖τ‖ [N·m]"
    * Grid with mild transparency.

Console output
--------------
After plotting, the script prints:

- A summary of τ-series used per deputy:

      [τ series used per deputy]
        Deputy 1: tau_apply
        Deputy 2: tau_cmd
        ...

If a deputy is missing a time array or torque vector, a warning is printed
and that deputy is skipped in the overlay.

Usage examples
--------------
Combined NPZ only:

    python fleet_tau_norm_overlay.py \
        --combined espa_xb6_8deps_20250926_011530.npz \
        --engage 10.3 \
        --label-prefix "XB6 " \
        --fs-title 40 --fs-label 35 --fs-tick 32 --fs-legend 30

Per-deputy NPZs:

    python fleet_tau_norm_overlay.py \
        --per-deputy "espa_xb6_8deps_*_D*.npz"

The script will:

- Load the requested data,
- Build a single shared plot of ‖τ‖ vs time for all deputies,
- Draw an engagement-time marker if requested,
- Display the figure,
- Print which tensor (tau_apply / tau_cmd / tau_des) was used for each deputy.
"""


from __future__ import annotations
import argparse, os, re, glob
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# -------------------- v7-style loader --------------------
def auto_find_combined():
    cands = sorted(glob.glob("espa_xb6_8deps_*.npz"))
    cands = [p for p in cands if not re.search(r"_D\d+\.npz$", p)]
    return cands[-1] if cands else None

def find_num_deps_from_keys(keys) -> int:
    patt = re.compile(r"^D(\d+)_t$")
    idx = [int(m.group(1)) for k in keys if (m := patt.match(k))]
    return max(idx) if idx else 0

def load_combined_npz(path: str):
    data = np.load(path, allow_pickle=False)
    num_deps = find_num_deps_from_keys(data.files)
    has_rafti = "sched_dv_cmd" in data.files
    return data, num_deps, has_rafti

def load_per_deputy_npz(paths):
    return [np.load(p, allow_pickle=False) for p in sorted(paths)]

# -------------------- accessor / getters --------------------
def make_proxy(deps):
    class Proxy:
        def __init__(self, deps):
            self.deps = deps
            self.files = set().union(*[set(d.files) for d in deps])
        def __getitem__(self, k):
            m = re.match(r"D(\d+)_(.+)", k)
            if not m:
                return self.deps[0][k]
            i = int(m.group(1)) - 1
            nm = m.group(2)
            return self.deps[i][nm]
    return Proxy(deps)

def pick_key(data_src, i_dep: int, base: str) -> str | None:
    k1 = f"D{i_dep}_{base}"
    if k1 in data_src.files: return k1
    if base in data_src.files: return base
    return None

def get_arr(data_src, i_dep: int, base: str) -> np.ndarray | None:
    k = pick_key(data_src, i_dep, base)
    return None if k is None else np.array(data_src[k], float)

def get_vec3_first(data_src, i_dep: int, candidates: list[str]):
    """Return (array, name) for first candidate present as (N,3)."""
    for nm in candidates:
        A = get_arr(data_src, i_dep, nm)
        if isinstance(A, np.ndarray) and A.ndim == 2 and A.shape[1] == 3:
            return A, nm
    return None, None

def norm_rows(X: np.ndarray, n_expected: int | None = None) -> np.ndarray:
    if isinstance(X, np.ndarray) and X.ndim == 2 and (n_expected is None or X.shape[0] == n_expected):
        return la.norm(X, axis=1)
    return np.array([])

# -------------------- font helpers --------------------
def apply_font_rc(title:int, label:int, tick:int, legend:int):
    plt.rcParams.update({
        "axes.titlesize": title,
        "axes.labelsize": label,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": legend,
    })

# -------------------- plot --------------------
def plot_tau_norm_all(accessor, N: int, engage_t: float | None, label_prefix:str,
                      fs_title:int, fs_legend:int):
    """
    Overlay ‖τ‖ for all deputies on one axes.
    Legend is vertical (ncol=1) and placed outside on the right.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    used_key = {}

    for i in range(1, N+1):
        t = get_arr(accessor, i, "t")
        if t is None or t.size == 0:
            print(f"[WARN] {label_prefix}{i}: missing time array 't'; skipping")
            continue

        tau, which = get_vec3_first(accessor, i, ["tau_apply", "tau_cmd", "tau_des"])
        if tau is None:
            print(f"[WARN] {label_prefix}{i}: no τ vector found (tau_apply/cmd/des); skipping")
            continue

        used_key[i] = which
        ax.plot(t, norm_rows(tau, n_expected=t.size), label=f"{label_prefix}{i}")

    if engage_t is not None:
        ax.axvline(engage_t, ls=":", alpha=0.6)

    ax.set_title("Torque Norm ‖τ‖", fontsize=fs_title)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("‖τ‖ [N·m]")
    ax.grid(alpha=0.25)

    # Vertical legend on the right (one column)
    # Reserve space on the right so the legend never overlaps the plot.
    fig.subplots_adjust(right=0.8)
    ax.legend(
        ncol=1,
        title="Deputy",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True
    )

    fig.tight_layout()
    plt.show()

    if used_key:
        print("\n[τ series used per deputy]")
        for i in sorted(used_key):
            print(f"  {label_prefix}{i}: {used_key[i]}")
        print()

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Plot ‖τ‖ overlay (all deputies) with vertical legend.")
    # I/O
    ap.add_argument("--combined",   type=str, default=None, help="Path to combined NPZ (preferred).")
    ap.add_argument("--per-deputy", type=str, nargs="*", default=None, help="Paths or globs to per-deputy NPZs.")
    # Visuals
    ap.add_argument("--engage",       type=float, default=10.3, help="Optional engagement time marker (seconds).")
    ap.add_argument("--label-prefix", type=str,   default="Deputy ", help="Deputy label prefix for titles/legend.")
    # Font sizes
    ap.add_argument("--fs-title",   type=int, default=35, help="Axes title font size.")
    ap.add_argument("--fs-label",   type=int, default=30, help="Axes label font size.")
    ap.add_argument("--fs-tick",    type=int, default=30, help="Tick label font size.")
    ap.add_argument("--fs-legend",  type=int, default=30, help="Legend font size.")
    args = ap.parse_args()

    # Apply font sizes globally
    apply_font_rc(args.fs_title, args.fs_label, args.fs_tick, args.fs_legend)

    combined_path = args.combined
    per_paths = args.per_deputy

    # Expand globs for per-deputy
    if per_paths:
        expanded = []
        for p in per_paths:
            expanded += glob.glob(p)
        per_paths = sorted(set(expanded))

    data_combined = None
    N = 0
    accessor = None

    if combined_path is None and not per_paths:
        auto = auto_find_combined()
        if auto:
            combined_path = auto

    if combined_path:
        if not os.path.exists(combined_path):
            raise FileNotFoundError(f"Combined NPZ not found: {combined_path}")
        data_combined, num_deps, _ = load_combined_npz(combined_path)
        N = num_deps
        accessor = data_combined
        print(f"[LOAD] Combined: {combined_path}  (deps={N})")

    if accessor is None and per_paths:
        deps = load_per_deputy_npz(per_paths)
        accessor = make_proxy(deps)
        N = len(deps)
        print(f"[LOAD] Per-deputy files: {N}")

    if accessor is None or N <= 0:
        raise SystemExit("No data found or could not infer number of deputies.")

    # Plot torque norm with vertical legend
    plot_tau_norm_all(
        accessor, N, engage_t=args.engage, label_prefix=args.label_prefix,
        fs_title=args.fs_title, fs_legend=args.fs_legend
    )

    print("[DONE] Torque plot displayed.")

if __name__ == "__main__":
    main()
