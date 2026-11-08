#!/usr/bin/env python3
"""
basilik_deputy_mroe_plots.py

Purpose
-------
End-to-end, “resilient” Mean-ROE plotting tool for deputy trajectories saved
as NPZ bundles. It reconstructs osculating chief/deputy states from NPZ files,
converts them to Classical Orbital Elements (COE), computes Mean Relative
Orbital Elements (M-ROE) using `mroe_tools` / `MROE`, and then compares:

- “Truth” M-ROE computed directly from the time-varying COE, and
- A simple linear M-ROE propagator driven by the chief’s mean motion.

For each matching NPZ bundle, it generates a 3×2 figure of the six mean ROE
components vs time and saves a PNG to the chosen output directory.

Inputs and dependencies
-----------------------
Run this script from a directory that contains:

- Deputy NPZ bundles, typically one per deputy, e.g.
    * `apophis_rendezvous_full_D*.npz`, or
    * `viz_inputs_D*.npz`, or any NPZ set matching a user-specified pattern.

- An M-ROE utility module:
    * `mroe_tools.py` (preferred), or
    * `MROE.py` (legacy).

The script:

- Attempts to import `mroe_tools` first, then falls back to `MROE`. It expects
  the module to define:
    * `mroe(...)`
    * `propagate_mroe(...)`
    * `MU_EARTH` (gravitational parameter).

- Uses Basilisk’s `orbitalMotion` (`Basilisk.utilities.orbitalMotion as om`)
  for COE handling:
    * Prefers `om.rv2coe` when available.
    * Falls back to `om.rv2elem`.
    * If neither exists, a minimal internal `rv_to_coe` routine computes a
      ClassicElements-like struct from r, v.

Bundle expectations
-------------------
Each NPZ bundle is expected to contain at least:

- A chief state:
    * Position/velocity arrays with keys of the form:
          rC_N, vC_N   or
          rC_hist, vC_hist   or
          rC, vC
      The helper `state_pair(z, "rC")` searches over the following suffix
      combinations:
          ("_N", "_hist", "").

- A deputy state:
    * Similarly, with keys like:
          rD_N, vD_N   or
          rD_hist, vD_hist   or
          rD, vD.

- A time array:
    * `t_hist`, or
    * `t`, or
    * `t0` (scalar start time), in which case the script synthesizes a simple
      0…N−1 time index and subtracts the first value so that t starts at 0.

All states are coerced to numeric (N,3) arrays using `coerce_state`, which
handles ragged object arrays and raises explicit `ValueError`s if malformed.

M-ROE computation
-----------------
For each bundle:

1. Extracts chief and deputy r(t), v(t) histories and a time vector t:

   - `mroe_series` computes:
       * `roe0` = mroe(chief_COE(0), deputy_COE(0), MU, dt=0),
       * chief mean motion `n_c` from a_c,
       * `linear[k]` = `propagate_mroe(roe0, n_c, t[k])`,
       * `truth[k]` = mroe(chief_COE(k), deputy_COE(k), MU, dt=0).

   Here, `rv_to_coe` converts each (r, v) sample into a ClassicElements-like
   structure, compatible with `mroe()` and `propagate_mroe()`.

2. Produces “truth” and linearized M-ROE time series of shape (N,6).

3. Converts angle-like components to degrees:
   - δλ, δi_x, δi_y are converted from radians to degrees for plotting.

Plotting and output
-------------------
For each bundle, `save_plot`:

- Creates a 3×2 figure (`subplots(3,2)`) with the 6 mean ROE components:

    Row 1: δa/a, δλ [deg]
    Row 2: δe_x, δe_y
    Row 3: δi_x [deg], δi_y [deg]

- Plots:
    * “truth” (solid line, lw=2), and
    * “linear” (dashed line, lw=1.3)
  on the same axes.

- Adds appropriate y-labels using LaTeX-friendly labels:

    LABELS = [
        r"$\delta a/a$", r"$\delta\lambda\,[^\circ]$",
        r"$\delta e_x$", r"$\delta e_y$",
        r"$\delta i_x\,[^\circ]$", r"$\delta i_y\,[^\circ]$"
    ]

- Uses “time since deploy [s]” as the common x-axis label.
- Sets the figure title to:

    "Deputy {dep_idx} – Mean ROE"

- Saves each plot as:

    mroe_deputy_D{dep_idx}.png

  in the specified output directory, and optionally shows it interactively.

CLI
---
Command-line arguments:

- `--data-dir PATH`   : directory containing NPZ bundles (default: current dir)
- `--pattern GLOB`    : glob for NPZ bundles (default: "apophis*_D*.npz")
- `--out-dir PATH`    : directory to write PNGs (default: same as data-dir)
- `--show`            : if provided, show each figure with `plt.show()` in
                        addition to saving it to disk.

Example:

    python basilik_deputy_mroe_plots.py \
        --data-dir results \
        --pattern "apophis*_D*.npz" \
        --out-dir results/mroe_plots \
        --show

For each matching file, the script prints either:

- “✓ mroe_deputy_D{idx}.png” on success, or
- A short diagnostic line (with exception text) if required arrays are
  missing or malformed.

Notes
-----
- All angles are in radians internally, but δλ, δi_x, and δi_y are plotted in
  degrees for easier interpretation.
- The script is designed to be resilient to small structural differences
  between NPZ bundles (e.g., `t_hist` vs `t`, `_N` vs `_hist` suffixes).
- It does not depend on Basilisk’s simulation architecture; only
  `orbitalMotion` is required for COE conversions.
"""


from __future__ import annotations
import argparse, importlib
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from Basilisk.utilities import orbitalMotion as om

# ────────────── import utilities (mroe_tools or MROE) ──────────────
for _modname in ("mroe_tools", "MROE"):
    try:
        mroemod = importlib.import_module(_modname)
        break
    except ModuleNotFoundError:
        continue
else:
    raise ImportError("Neither mroe_tools.py nor MROE.py is importable")

mroe, propagate_mroe = mroemod.mroe, mroemod.propagate_mroe
MU = mroemod.MU_EARTH

try:                                    # old Basilisk type (optional)
    from Basilisk.utilities.orbitalMotion import ClassicElements
except ImportError:                     # newer Basilisk doesn’t expose this
    ClassicElements = None

# ────────────── rv → COE helper (robust) ──────────────
def rv_to_coe(r: np.ndarray, v: np.ndarray):
    if hasattr(om, "rv2coe"):
        return om.rv2coe(r, v, MU)
    if hasattr(om, "rv2elem"):
        return om.rv2elem(r, v, MU)

    # minimal fallback
    h = np.cross(r, v); h_norm = np.linalg.norm(h)
    n = np.cross([0, 0, 1], h); n_norm = np.linalg.norm(n)
    e_vec = np.cross(v, h) / MU - r / np.linalg.norm(r); ecc = np.linalg.norm(e_vec)

    i = np.arccos(h[2] / h_norm)
    Ω = np.arctan2(n[1], n[0]) if n_norm > 1e-12 else 0.0
    ω = (np.arctan2(np.dot(np.cross(n, e_vec), h) / h_norm, np.dot(n, e_vec))
         if n_norm > 1e-12 else 0.0)
    f = np.arctan2(np.dot(np.cross(e_vec, r), h) / h_norm, np.dot(e_vec, r))
    a = 1 / (2/np.linalg.norm(r) - np.linalg.norm(v)**2 / MU)

    ce = ClassicElements() if ClassicElements else type("CE", (), {})()
    ce.a, ce.e, ce.i = a, ecc, i
    ce.Omega, ce.omega, ce.f = Ω % (2*np.pi), ω % (2*np.pi), f % (2*np.pi)
    return ce

# ────────────── bundle helpers ──────────────
def coerce_state(arr: Any, key: str) -> np.ndarray:
    """
    Ensure numeric (N,3).  Raises ValueError with explicit key if malformed.
    """
    a = np.asarray(arr, dtype=object)

    if a.dtype == object:          # ragged object array
        try:
            a = np.vstack(a)
        except ValueError:
            raise ValueError(f"{key}: ragged / non-vector rows")

    if a.ndim == 1 and a.size == 3:
        a = a.reshape(1, 3)

    if a.ndim != 2 or a.shape[1] != 3 or a.shape[0] < 1:
        raise ValueError(f"{key}: expected (N,3) got {a.shape}")

    return a.astype(float)

def state_pair(z, base: str) -> Tuple[np.ndarray, np.ndarray]:
    for suf in ("_N", "_hist", ""):
        rk, vk = f"{base}{suf}", f"v{base[1:]}{suf}"
        if rk in z and vk in z:
            return coerce_state(z[rk], rk), coerce_state(z[vk], vk)
    raise KeyError(f"{base}: state arrays not found")

def timeline(z, rows: int) -> np.ndarray:
    if "t_hist" in z:   t = z["t_hist"]
    elif "t" in z:      t = z["t"]
    elif "t0" in z:     t = float(z["t0"][0]) + np.arange(rows)
    else:               raise KeyError("no time array")
    if t.size != rows:
        raise ValueError("timeline length ≠ state rows")
    return t - t[0]

# ────────────── M-ROE series ──────────────
def mroe_series(rC, vC, rD, vD, t):
    N = len(t)
    truth = np.empty((N, 6))

    ce_c0, ce_d0 = rv_to_coe(rC[0], vC[0]), rv_to_coe(rD[0], vD[0])
    n_c = (MU / ce_c0.a**3)**0.5
    roe0 = mroe(ce_c0, ce_d0, MU, 0.0)
    linear = np.array([propagate_mroe(roe0, n_c, float(ti)) for ti in t])

    for k in range(N):
        truth[k] = mroe(rv_to_coe(rC[k], vC[k]),
                        rv_to_coe(rD[k], vD[k]), MU, 0.0)
    return truth, linear

# ────────────── plotting ──────────────
LABELS = [
    r"$\delta a/a$", r"$\delta\lambda\,[^\circ]$",
    r"$\delta e_x$", r"$\delta e_y$",
    r"$\delta i_x\,[^\circ]$", r"$\delta i_y\,[^\circ]$"
]

def save_plot(t, truth, linear, dep_idx: int, path: Path, show=False):
    truth = truth.copy(); linear = linear.copy()
    truth[:,[1,4,5]] *= 180/np.pi; linear[:,[1,4,5]] *= 180/np.pi

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()
    for k, ax in enumerate(axes):
        ax.plot(t, truth[:, k], label="truth", lw=2)
        ax.plot(t, linear[:, k], "--", label="linear", lw=1.3)
        ax.set_ylabel(LABELS[k]); ax.grid(alpha=.3)
        if k == 0:
            ax.legend()
    axes[-1].set_xlabel("time since deploy [s]")
    fig.suptitle(f"Deputy {dep_idx} – Mean ROE", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    print(f"✓ {path.name}")
    if show:
        plt.show()
    plt.close(fig)

# ────────────── CLI ──────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="bundle directory")
    parser.add_argument("--pattern", default="apophis*_D*.npz", help="glob pattern")
    parser.add_argument("--out-dir", type=Path, default=None, help="output PNG directory")
    parser.add_argument("--show", action="store_true", help="show figures interactively")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = (args.out_dir or data_dir).resolve()
    out_dir.mkdir(exist_ok=True)

    bundles = sorted(data_dir.glob(args.pattern))
    if not bundles:
        print("No bundles matched pattern — nothing to do.")
        return

    for bundle in bundles:
        dep_idx = int(bundle.stem.split("_D")[-1]) if "_D" in bundle.stem else -1
        try:
            with np.load(bundle, allow_pickle=True) as z:
                rC, vC = state_pair(z, "rC")
                rD, vD = state_pair(z, "rD")

                if rC.shape[0] < 2:
                    print(f"– {bundle.name}: only {rC.shape[0]} sample — skipped")
                    continue

                t = timeline(z, rC.shape[0])

            truth, linear = mroe_series(rC, vC, rD, vD, t)
            save_plot(t, truth, linear, dep_idx,
                      out_dir / f"mroe_deputy_D{dep_idx}.png",
                      show=args.show)

        except (KeyError, ValueError) as err:
            print(f"✗ {bundle.name}: {err}")

    print("Done.")

if __name__ == "__main__":
    main()
