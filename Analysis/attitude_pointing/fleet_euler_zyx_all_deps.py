#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fleet_euler_zyx_all_deps.py

Purpose
-------
Plot Euler ZYX attitude histories (φ, θ, ψ) for a fleet of up to eight deputies
on a single multi-panel figure, with achieved vs desired angles clearly
overlaid. Deputy 1 is read from a stand-alone PH_SE3 log, while the remaining
deputies are read from either a combined NPZ or per-deputy NPZ files.

Data model
----------
- Deputy 1:
  * Loaded from a dedicated file (default: `deputy1_PH_SE3.npz`).
  * Expected keys:
      - t              : time array [s]
      - roll, pitch, yaw     : achieved Euler angles [rad]
      - roll_d, pitch_d, yaw_d (optional) : desired Euler angles [rad]
    Alternatively, the file may contain packed arrays:
      - euler_wrapped (N×3) : achieved [φ, θ, ψ]
      - euler_des     (N×3) : desired [φ_d, θ_d, ψ_d]

- Other deputies (D2..D8) can be supplied in two ways:
  1) **Combined NPZ** (`--combined`):
     - Single file containing keys such as:
         D1_t, D1_euler_wrapped, D1_euler_des,
         D2_t, D2_euler_wrapped, ...
     - Or separate per-axis keys (e.g., D3_roll, D3_pitch, D3_yaw, etc.).
  2) **Per-deputy NPZs** (`--per-deputy` globs):
     - One NPZ per deputy; loaded and accessed via a `Proxy` that supports
       both packed Euler arrays and separate roll/pitch/yaw keys.

If neither `--combined` nor `--per-deputy` is given, the script will try to
auto-detect the latest `espa_xb6_8deps_*.npz` that is *not* a per-deputy
split file (no `_D#.npz` suffix).

What gets plotted
-----------------
- A grid of subplots arranged as up to 4 columns × 2 rows (for 8 deputies):
    * Each subplot corresponds to one deputy, labeled "Deputy D{i}".
    * Solid lines: achieved Euler components (φ, θ, ψ).
    * Dashed lines: desired Euler components, when available.
- A vertical dashed-dot line at `engage_t` (default 10.3 s) is drawn in each
  subplot to mark controller engagement / RAFTI push completion if desired.

- A single, shared legend at the bottom of the figure:
    * φ_d, θ_d, ψ_d → dashed line entries.
    * φ, θ, ψ       → solid line entries.

Styling
-------
- Large fonts for titles, labels, ticks, and legend for use in slides or
  paper figures.
- Each axis is labeled:
    * x-label: `t [s]`
    * y-label: `[rad]`
- Unused subplots (if fewer than 8 deputies) are turned off.

Usage
-----
Typical command lines:

  # Combined fleet file with all deputies
  python fleet_euler_zyx_all_deps.py --combined espa_xb6_8deps_20250926_011530.npz

  # Per-deputy split files
  python fleet_euler_zyx_all_deps.py --per-deputy "espa_xb6_8deps_*_D*.npz"

  # Custom Deputy 1 source and custom engage marker
  python fleet_euler_zyx_all_deps.py --dep1 deputy1_PH_SE3.npz --engage 10.3

If neither `--combined` nor `--per-deputy` is supplied, the script attempts to
auto-find a suitable `espa_xb6_8deps_*.npz` combined file in the current
directory.
"""


from __future__ import annotations
import argparse, glob, math, os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------- Plot style ----------
plt.rcParams.update({
    "axes.titlesize": 40,
    "axes.labelsize": 34,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 28,
})

# ---------- Helpers ----------
def auto_find_combined():
    cands = sorted(glob.glob("espa_xb6_8deps_*.npz"))
    cands = [p for p in cands if not re.search(r"_D\d+\.npz$", p)]
    return cands[-1] if cands else None

def load_dep1_npz(path: str):
    """
    Expect keys like: t, roll, pitch, yaw (and optionally roll_d, pitch_d, yaw_d).
    Also accepts packed arrays 'euler_wrapped' (Nx3) and 'euler_des' (Nx3).
    """
    A = np.load(path, allow_pickle=False)
    # time
    t = np.array(A["t"], float) if "t" in A.files else None

    # achieved Euler
    if "euler_wrapped" in A.files:
        eul = np.array(A["euler_wrapped"], float)
        roll, pitch, yaw = eul[:,0], eul[:,1], eul[:,2]
    else:
        roll  = np.array(A["roll"],  float) if "roll"  in A.files else None
        pitch = np.array(A["pitch"], float) if "pitch" in A.files else None
        yaw   = np.array(A["yaw"],   float) if "yaw"   in A.files else None

    # desired Euler (optional)
    if "euler_des" in A.files:
        des = np.array(A["euler_des"], float)
        roll_d, pitch_d, yaw_d = des[:,0], des[:,1], des[:,2]
    else:
        roll_d  = np.array(A["roll_d"],  float) if "roll_d"  in A.files else None
        pitch_d = np.array(A["pitch_d"], float) if "pitch_d" in A.files else None
        yaw_d   = np.array(A["yaw_d"],   float) if "yaw_d"   in A.files else None

    return dict(t=t, roll=roll, pitch=pitch, yaw=yaw,
                roll_d=roll_d, pitch_d=pitch_d, yaw_d=yaw_d)

def load_combined_npz(path: str):
    return np.load(path, allow_pickle=False)

def load_per_deputy_npz(paths):
    return [np.load(p, allow_pickle=False) for p in sorted(paths)]

class Proxy:
    """
    Unifies access to per-deputy split files so we can address keys like 'D3_t', 'D3_euler_wrapped'.
    Falls back to base keys ('t', 'euler_wrapped') when present per-file.
    """
    def __init__(self, deps):
        self.deps = deps
        self.files = set().union(*[set(d.files) for d in deps])

    def __getitem__(self, k):
        m = re.match(r"D(\d+)_(.+)", k)
        if not m:
            # if asking a base key, serve from D1 file
            return self.deps[0][k]
        i = int(m.group(1)) - 1
        nm = m.group(2)
        return self.deps[i][nm]

def num_deps_from_combined(files) -> int:
    idx = []
    for f in files:
        m = re.match(r"D(\d+)_t$", f)
        if m: idx.append(int(m.group(1)))
    return max(idx) if idx else 0

def get_deputy_euler(accessor, i_dep: int):
    """
    Extract (t, φ, θ, ψ, φ_d, θ_d, ψ_d) for deputy i_dep from a combined/per-dep accessor.
    Accepts either packed arrays (euler_wrapped/euler_des) or separate roll/pitch/yaw keys.
    """
    # Keys by deputy
    t_key    = f"D{i_dep}_t"                 if f"D{i_dep}_t"                 in accessor.files else "t"
    wrap_key = f"D{i_dep}_euler_wrapped"     if f"D{i_dep}_euler_wrapped"     in accessor.files else "euler_wrapped"
    des_key  = f"D{i_dep}_euler_des"         if f"D{i_dep}_euler_des"         in accessor.files else "euler_des"

    t = np.array(accessor[t_key], float) if t_key in accessor else None

    if wrap_key in accessor.files:
        E = np.array(accessor[wrap_key], float)
        roll, pitch, yaw = E[:,0], E[:,1], E[:,2]
    else:
        # separate axes fallbacks (names occasionally vary)
        def pick(name_options):
            for nm in name_options:
                k = f"D{i_dep}_{nm}" if f"D{i_dep}_{nm}" in accessor.files else nm
                if k in accessor.files:
                    return np.array(accessor[k], float)
            return None
        roll  = pick(["roll", "phi",  "euler_x"])
        pitch = pick(["pitch","theta","euler_y"])
        yaw   = pick(["yaw",  "psi",  "euler_z"])

    if des_key in accessor.files:
        D = np.array(accessor[des_key], float)
        roll_d, pitch_d, yaw_d = D[:,0], D[:,1], D[:,2]
    else:
        def pickd(name_options):
            for nm in name_options:
                k = f"D{i_dep}_{nm}" if f"D{i_dep}_{nm}" in accessor.files else nm
                if k in accessor.files:
                    return np.array(accessor[k], float)
            return None
        roll_d  = pickd(["roll_d",  "phi_d",  "euler_x_d"])
        pitch_d = pickd(["pitch_d", "theta_d","euler_y_d"])
        yaw_d   = pickd(["yaw_d",   "psi_d",  "euler_z_d"])

    return dict(t=t, roll=roll, pitch=pitch, yaw=yaw,
                roll_d=roll_d, pitch_d=pitch_d, yaw_d=yaw_d)

# ---------- Main plotting ----------
def plot_all(dep1_path: str, combined_path: str|None, per_globs: list[str]|None, engage_t: float):
    # Deputy 1 (stand-alone file)
    if not os.path.exists(dep1_path):
        raise FileNotFoundError(f"dep1 file not found: {dep1_path}")
    dep1 = load_dep1_npz(dep1_path)

    # Others (from combined or per-dep)
    accessor = None
    N = 8  # expect up to 8 deputies by requirement
    if combined_path:
        if not os.path.exists(combined_path):
            raise FileNotFoundError(f"Combined NPZ not found: {combined_path}")
        accessor = load_combined_npz(combined_path)
        N = max(N, num_deps_from_combined(accessor.files) or 8)
    elif per_globs:
        paths = []
        for g in per_globs:
            paths += glob.glob(g)
        paths = sorted(set(paths))
        if not paths:
            raise SystemExit("No per-deputy files matched the given globs.")
        accessor = Proxy(load_per_deputy_npz(paths))
        N = max(N, len(accessor.deps))
    else:
        # nothing provided for others — still plot dep1 only
        N = 1

    # 4×2 grid
    ncols = 4
    nrows = math.ceil(N / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols+8, 5*nrows+3), sharex=False)
    axes = np.atleast_2d(axes)

    # build common legend (dashed = desired, solid = achieved)
    handles = [
        Line2D([0],[0], linestyle='--'), Line2D([0],[0], linestyle='--'), Line2D([0],[0], linestyle='--'),
        Line2D([0],[0], linestyle='-'),  Line2D([0],[0], linestyle='-'),  Line2D([0],[0], linestyle='-')
    ]
    labels  = ["φ_d","θ_d","ψ_d","φ","θ","ψ"]

    for i in range(1, N+1):
        r, c = (i-1)//ncols, (i-1)%ncols
        ax = axes[r, c]

        if i == 1:
            D = dep1
        else:
            if accessor is None:
                ax.set_title(f"D{i} — (no data)"); ax.axis("off"); continue
            D = get_deputy_euler(accessor, i)

        t = D["t"]
        if t is None or t.size == 0 or D["roll"] is None or D["pitch"] is None or D["yaw"] is None:
            ax.set_title(f"D{i} — missing Euler"); ax.axis("off"); continue

        # desired (if present)
        if D["roll_d"] is not None and D["pitch_d"] is not None and D["yaw_d"] is not None:
            ax.plot(t, D["roll_d"],  "k--", lw=1.8)
            ax.plot(t, D["pitch_d"], "k--", lw=1.8)
            ax.plot(t, D["yaw_d"],   "k--", lw=1.8)

        # achieved
        ax.plot(t, D["roll"],  lw=2.2)
        ax.plot(t, D["pitch"], lw=2.2)
        ax.plot(t, D["yaw"],   lw=2.2)

        if engage_t is not None:
            ax.axvline(engage_t, ls=":", alpha=0.5)

        ax.set_title(f"Deputy D{i}")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("[rad]")
        ax.grid(True)

    # hide unused axes
    total = nrows*ncols
    for j in range(N+1, total+1):
        axes[(j-1)//ncols, (j-1)%ncols].axis("off")

    fig.suptitle("Euler ZYX with SE(3) based pH Controller", fontsize=44)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.04), fontsize=24, framealpha=0.2)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.show()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Plot Euler ZYX for all deputies (D1 from deputy1_PH_SE3.npz).")
    ap.add_argument("--dep1",      type=str, default="deputy1_PH_SE3.npz", help="Path to deputy1_PH_SE3.npz")
    ap.add_argument("--combined",  type=str, default=None, help="Combined NPZ for multi-dep data")
    ap.add_argument("--per-deputy",type=str, nargs="*", default=None, help="Glob(s) for per-deputy NPZs")
    ap.add_argument("--engage",    type=float, default=10.3, help="Engagement time vertical marker")
    args = ap.parse_args()

    # If neither combined nor per-deputy given, try to auto-find combined
    combined = args.combined
    if combined is None and (not args.per_deputy):
        combined = auto_find_combined()

    plot_all(args.dep1, combined, args.per_deputy, args.engage)

if __name__ == "__main__":
    main()

