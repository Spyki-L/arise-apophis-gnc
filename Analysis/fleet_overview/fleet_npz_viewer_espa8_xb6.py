#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fleet_npz_viewer_espa8_xb6.py

Purpose
-------
Interactive NPZ viewer for an ESPA + 8×XB6 separation campaign. This script
loads either a combined multi-deputy NPZ or per-deputy NPZ files, then produces:

- Aggregate Δv and pointing plots across the full fleet.
- A combined H_SE3,d energy plot for all deputies.
- An 8-panel Euler ZYX (desired vs achieved) grid with a single shared legend.
- Per-deputy “3-row stack” plots showing ‖τ‖, ω components, and ‖h‖ (or ‖ω‖
  as a proxy), plus zoomed-in versions and printed error metrics at a chosen
  reference time.

It is meant for quick visual inspection, slide-ready figures, and console
summaries rather than archival post-processing.

Inputs and data model
---------------------
The script can ingest data in two ways:

1) **Combined NPZ** (`--combined`):
   - A single file (e.g. `espa_xb6_8deps_20250926_011530.npz`) containing keys like:
        D1_t, D1_euler_wrapped, D1_euler_des, D1_dv_cum, D1_Hd, ...
        D2_t, ...
     plus optional RAFTI schedule arrays such as:
        sched_dv_cmd
   - `find_num_deps_from_keys()` uses keys like `Dk_t` to infer the number of
     deputies present.

2) **Per-deputy NPZs** (`--per-deputy`):
   - One NPZ per deputy (e.g. `..._D1.npz`, `..._D2.npz`, …), passed via one
     or more glob patterns.
   - A small `Proxy` class lets the rest of the script access them with
     combined-style keys like `D3_t`, `D3_euler_wrapped`, etc.

If neither `--combined` nor `--per-deputy` is provided explicitly, the script
tries to auto-detect a combined file of the form:

    espa_xb6_8deps_*.npz

that does *not* end with `_D#.npz`.

Aggregate plots
---------------
1. **Δv summary (per deputy)**
   - If RAFTI schedule info is available:
        * Bar chart comparing RAFTI commanded Δv (`sched_dv_cmd`) vs control
          Δv (`Dk_dv_cum`) for all deputies.
   - If RAFTI info is absent:
        * Single bar per deputy: control Δv only.

2. **Pointing metrics (all deputies)**
   - Boresight error vs time:
        * `Dk_boresight_deg` when present, else NaNs.
        * Horizontal 1° line for a loose pointing tolerance.
   - Roll-about-LOS error vs time:
        * `Dk_roll_about_LOS_deg` when present.
        * Horizontal 2° line for roll-constraint visualization.

3. **Energy (H_SE3,d) overlay**
   - One figure showing H_SE3,d time histories for all deputies using keys:
        Dk_t, Dk_Hd (or shared `t`, `Hd` as fallback).
   - A vertical line at `engage_t` (default 10.3 s) marks controller engagement.

4. **Euler ZYX grid (desired vs achieved)**
   - 4×2 grid (up to 8 deputies) where each subplot plots:
        * Achieved Euler ZYX (φ, θ, ψ) from `euler_wrapped`.
        * Desired Euler ZYX (φ_d, θ_d, ψ_d) from `euler_des`, when available.
   - The script prints Euler errors at a specific time (default 250 s) for each
     deputy when desired angles exist.
   - A single figure-level legend at the bottom:
        {φ_d, θ_d, ψ_d} as dashed lines, {φ, θ, ψ} as solid lines.

Per-deputy detailed plots
-------------------------
For each deputy k:

1. **Full-range 3-row stack**
   - Row 1: ‖τ‖ norms
        * |τ_d| from `tau_des`
        * |τ_cmd| from `tau_cmd`
        * |τ_appl| from `tau_apply`
   - Row 2: ω components (ω_x, ω_y, ω_z) from `omega_B` (or similar).
   - Row 3: ‖h‖ (angular momentum norm) using any of:
        `h_B`, `angMom_B`, `H_B`, `mom_B`, `rw_h_B`, `ang_momentum_B`.
     If no momentum vector is found, the script falls back to ‖ω‖ and notes
     this in the console.

2. **Zoomed 3-row stack + error print**
   - Same rows as above, but restricted to `zoom_x = (t_min, t_max)` and
     `zoom_y = (y_min, y_max)` (configurable).
   - Relative torque errors are computed with `rel_err`:
        * rel τ(d→cmd)
        * rel τ(d→appl)
     and the values at a specific time (default 250 s) are printed to the console.

Styling and configuration
-------------------------
- Global font sizes tuned for large, clear figures:
    axes titles, labels, ticks, and legends scaled for slide/paper use.
- CLI arguments:
    --combined    : path to combined NPZ (optional)
    --per-deputy  : one or more paths/globs for per-deputy NPZs
    --engage      : engagement time marker for vertical lines (default: 10.3 s)
    --zoom-x      : zoom x-range, e.g. `--zoom-x 150 250`
    --zoom-y      : zoom y-range, e.g. `--zoom-y -1 1`

Usage
-----
Examples:

  # Combined file only (auto-deputy detection)
  python fleet_npz_viewer_espa8_xb6.py --combined espa_xb6_8deps_20250926_011530.npz

  # Custom zoom window
  python fleet_npz_viewer_espa8_xb6.py --combined espa_xb6_8deps_*.npz --zoom-x 150 250 --zoom-y -1 1

  # Per-deputy split logs
  python fleet_npz_viewer_espa8_xb6.py --per-deputy "espa_xb6_8deps_*_D*.npz"

The script then loads the data, prints torque/Euler error summaries at the
specified reference time, and displays all aggregate and per-deputy plots.
"""


from __future__ import annotations
import argparse, os, re, glob, math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------- Global plot sizing ----------
plt.rcParams.update({
    "axes.titlesize": 45,
    "axes.labelsize": 40,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 28,
})

# ---------- Utilities ----------
def rel_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return np.zeros(0)
    if a.ndim == 1:
        a = a.reshape(-1, 1); b = b.reshape(-1, 1)
    den = la.norm(a, axis=1)
    ref = max(float(np.max(den)) if den.size else 1.0, 1.0)
    num = la.norm(a - b, axis=1)
    mask = den > 1e-3 * ref
    out = np.zeros_like(den)
    out[mask] = num[mask] / np.maximum(den[mask], 1e-12)
    return out

def value_at_time(t: np.ndarray, y: np.ndarray, t_query: float) -> np.ndarray:
    if t is None or y is None or t.size == 0:
        return np.array([])
    idx = int(np.clip(np.argmin(np.abs(t - t_query)), 0, t.size - 1))
    return y[idx]

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

def auto_find_combined():
    cands = sorted(glob.glob("espa_xb6_8deps_*.npz"))
    cands = [p for p in cands if not re.search(r"_D\d+\.npz$", p)]
    return cands[-1] if cands else None

# ---------- Aggregate (Δv/pointing kept; plus energy + RPY grid) ----------
def plot_aggregate_dv(data_combined, num_deps, has_rafti: bool):
    idx = np.arange(num_deps); w = 0.35
    dv_ctl = np.array([data_combined[f"D{i+1}_dv_cum"] for i in range(num_deps)], float)
    plt.figure(figsize=(12, 7))
    if has_rafti:
        dv_cmd = np.array(data_combined["sched_dv_cmd"], float)[:num_deps]
        plt.bar(idx - w/2, dv_cmd, width=w, label="RAFTI Δv_cmd")
        plt.bar(idx + w/2, dv_ctl, width=w, label="Control Δv")
    else:
        plt.bar(idx, dv_ctl, width=w, label="Control Δv")
    plt.xticks(idx, [f"D{i+1}" for i in range(num_deps)])
    plt.ylabel("Δv [m/s]")
    plt.title("Separation Δv vs Control Δv — all deputies" if has_rafti else "Control Δv — all deputies")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_aggregate_pointing(data_combined, num_deps):
    # boresight
    plt.figure(figsize=(12, 8))
    for i in range(num_deps):
        t = np.array(data_combined[f"D{i+1}_t"], float)
        k = f"D{i+1}_boresight_deg"
        bores = np.array(data_combined[k], float) if k in data_combined.files else np.full_like(t, np.nan)
        if t.size: plt.plot(t, bores, label=f"D{i+1}")
    plt.axhline(1.0, ls="--", alpha=0.3)
    plt.xlabel("time [s]"); plt.ylabel("boresight [deg]")
    plt.title("Boresight error vs time (all deputies)")
    plt.legend(ncol=4); plt.tight_layout(); plt.show()

    # roll-about-LOS
    plt.figure(figsize=(12, 8))
    for i in range(num_deps):
        t = np.array(data_combined[f"D{i+1}_t"], float)
        k = f"D{i+1}_roll_about_LOS_deg"
        rollE = np.array(data_combined[k], float) if k in data_combined.files else np.full_like(t, np.nan)
        if t.size: plt.plot(t, rollE, label=f"D{i+1}")
    plt.axhline(2.0, ls="--", alpha=0.3)
    plt.xlabel("time [s]"); plt.ylabel("roll about LOS [deg]")
    plt.title("Roll-about-LOS error vs time (all deputies)")
    plt.legend(ncol=4); plt.tight_layout(); plt.show()

def plot_aggregate_energy_all_deps(accessor, num_deps, engage_t: float, label_prefix: str = "XB6 D"):
    """One figure: H_SE3,d for all deputies in different colors."""
    plt.figure(figsize=(13, 8))
    for i in range(1, num_deps+1):
        t_key  = f"D{i}_t"  if f"D{i}_t"  in accessor.files else "t"
        Hd_key = f"D{i}_Hd" if f"D{i}_Hd" in accessor.files else "Hd"
        if t_key not in accessor.files or Hd_key not in accessor.files: continue
        t  = np.array(accessor[t_key], float)
        Hd = np.array(accessor[Hd_key], float)
        plt.plot(t, Hd, label=f"{label_prefix}{i}")
    plt.axvline(engage_t, ls="--", alpha=0.5)
    plt.xlabel("time [s]"); plt.ylabel("H_SE3,d")
    plt.title("H_SE3,d — all deputies")
    plt.legend(ncol=4)
    plt.tight_layout(); plt.show()

def plot_rpy_grid_all_deps(accessor, num_deps, engage_t: float,
                           label_prefix: str = "XB6 D",
                           print_euler_errors_at: float = 250.0):
    """
    8-subplot grid (4×2). Each subplot shows R/P/Y (desired vs achieved) for a deputy.
    No axes legends; instead a single **figure-level legend** is added.
    """
    ncols = 4; nrows = math.ceil(num_deps / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols+8, 5*nrows+4), sharex=False)
    axes = np.atleast_2d(axes)

    # Track whether any "desired" data exists to condition the common legend.
    any_desired = False

    for i in range(1, num_deps+1):
        r = (i-1)//ncols; c = (i-1)%ncols; ax = axes[r, c]
        t_key = f"D{i}_t" if f"D{i}_t" in accessor.files else "t"
        ewk   = f"D{i}_euler_wrapped" if f"D{i}_euler_wrapped" in accessor.files else "euler_wrapped"
        edk   = f"D{i}_euler_des"     if f"D{i}_euler_des"     in accessor.files else "euler_des"
        if t_key not in accessor.files or ewk not in accessor.files:
            ax.set_title(f"{label_prefix}{i} — no Euler"); ax.axis("off"); continue

        t = np.array(accessor[t_key], float)
        eul_wr  = np.array(accessor[ewk], float)
        eul_des = np.array(accessor[edk], float) if edk in accessor.files else None

        if eul_wr.ndim != 2 or eul_wr.shape[0] != t.size or eul_wr.shape[1] != 3:
            ax.set_title(f"{label_prefix}{i} — bad Euler shape"); ax.axis("off"); continue

        if eul_des is not None and eul_des.shape == eul_wr.shape:
            any_desired = True
            ax.plot(t, eul_des[:,0], "--")
            ax.plot(t, eul_des[:,1], "--")
            ax.plot(t, eul_des[:,2], "--")
            # Print Euler errors at 250 s to console
            err_250 = value_at_time(t, eul_des - eul_wr, print_euler_errors_at)
            if err_250.size == 3:
                print(f"[Euler @ {print_euler_errors_at:.1f}s] {label_prefix}{i}: "
                      f"|Δφ|={abs(err_250[0]):.6g}, |Δθ|={abs(err_250[1]):.6g}, |Δψ|={abs(err_250[2]):.6g}")
        else:
            print(f"[Euler] {label_prefix}{i}: desired angles not available; skipping error print.")

        # Achieved traces
        ax.plot(t, eul_wr[:,0])
        ax.plot(t, eul_wr[:,1])
        ax.plot(t, eul_wr[:,2])

        ax.axvline(engage_t, ls=":", alpha=0.6)
        ax.set_xlabel("t [s]"); ax.set_ylabel("[rad]"); ax.set_title(f"{label_prefix}{i}")

    # Hide unused axes
    total = nrows*ncols
    for j in range(num_deps+1, total+1):
        r = (j-1)//ncols; c = (j-1)%ncols
        axes[r, c].axis("off")

    # Build a single, clean, figure-level legend.
    handles = []
    if any_desired:
        handles += [
            Line2D([0], [0], linestyle='--'),  # φ_d
            Line2D([0], [0], linestyle='--'),  # θ_d
            Line2D([0], [0], linestyle='--'),  # ψ_d
        ]
    handles += [
        Line2D([0], [0], linestyle='-'),  # φ
        Line2D([0], [0], linestyle='-'),  # θ
        Line2D([0], [0], linestyle='-'),  # ψ
    ]
    labels = (["φ_d", "θ_d", "ψ_d"] if any_desired else []) + ["φ", "θ", "ψ"]

    fig.suptitle("Euler ZYX (desired vs achieved) — all deputies", fontsize=45)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.04), fontsize=24, framealpha=0.2)

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])  # leave space for legend & title
    plt.show()

# ---------- Per-deputy detail (3 rows: ‖τ‖ ; ω ; ‖h‖) ----------
def plot_per_deputy_detail(data_src, i_dep: int, engage_t: float,
                           zoom_x: tuple[float,float], zoom_y: tuple[float,float],
                           label_prefix: str = "XB6 D",
                           print_tau_errors_at: float = 250.0):
    # helpers
    def pick_key(name: str) -> str | None:
        k1 = f"D{i_dep}_{name}"
        if k1 in data_src.files: return k1
        if name in data_src.files: return name
        return None
    def get_arr(name: str) -> np.ndarray | None:
        k = pick_key(name); 
        return None if k is None else np.array(data_src[k], float)
    def get_vec3(*candidates: str) -> np.ndarray | None:
        for nm in candidates:
            A = get_arr(nm)
            if A is not None and A.ndim == 2 and A.shape[1] == 3:
                return A
        return None

    # time base
    t = get_arr("t")
    if t is None:
        print(f"[WARN] Deputy {i_dep}: no time array 't' found; skipping."); return
    t = t.astype(float)

    # series
    tau_d = get_arr("tau_des");   tau_d = np.zeros((t.size,3))   if tau_d is None else tau_d
    tau_cmd = get_arr("tau_cmd"); tau_cmd = np.zeros((t.size,3)) if tau_cmd is None else tau_cmd
    tau_app = get_arr("tau_apply"); tau_app = np.zeros((t.size,3)) if tau_app is None else tau_app
    omega = get_vec3("omega_B");  omega = np.zeros((t.size,3))   if omega is None else omega

    # h-vector (fallback to ω)
    h_vec = get_vec3("h_B", "angMom_B", "H_B", "mom_B", "rw_h_B", "ang_momentum_B")
    use_proxy_for_h = False
    if h_vec is None:
        h_vec = omega; use_proxy_for_h = True
        print(f"[WARN] No angular-momentum vector for {label_prefix}{i_dep}; using ‖ω‖ as proxy for ‖h‖.")

    # norms
    def nrm(X: np.ndarray) -> np.ndarray:
        return la.norm(X, axis=1) if (isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[0] == t.size) else np.zeros_like(t)

    # ---- full-range 3-row stack (H removed) ----
    fig, axs = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    axs[0].plot(t, nrm(tau_d),   label="|τ_d|")
    axs[0].plot(t, nrm(tau_cmd), label="|τ_cmd|")
    axs[0].plot(t, nrm(tau_app), label="|τ_appl|")
    axs[0].legend(loc="upper right")
    axs[0].set_ylabel("Torque norm")
    axs[0].set_title(f"{label_prefix}{i_dep} — histories")

    axs[1].plot(t, omega[:,0], label="ω_x")
    axs[1].plot(t, omega[:,1], label="ω_y")
    axs[1].plot(t, omega[:,2], label="ω_z")
    axs[1].legend(loc="upper right")
    axs[1].set_ylabel("ω [rad/s]")

    h_label = "‖h‖" if not use_proxy_for_h else "‖ω‖ (proxy for ‖h‖)"
    axs[2].plot(t, nrm(h_vec), label=h_label)
    axs[2].legend(loc="upper right")
    axs[2].set_ylabel(h_label)
    axs[2].set_xlabel("time [s]")

    fig.tight_layout(); plt.show()

    # ---- Zoom (same 3 rows) & print τ errors at 250 s to console ----
    e_tau_cmd = rel_err(tau_d, tau_cmd); e_tau_app = rel_err(tau_d, tau_app)
    if e_tau_cmd.size == t.size:
        v = value_at_time(t, e_tau_cmd, print_tau_errors_at)
        print(f"[τ err @ {print_tau_errors_at:.1f}s] {label_prefix}{i_dep}: rel τ(d→cmd) = {float(v):.6g}")
    if e_tau_app.size == t.size:
        v = value_at_time(t, e_tau_app, print_tau_errors_at)
        print(f"[τ err @ {print_tau_errors_at:.1f}s] {label_prefix}{i_dep}: rel τ(d→appl) = {float(v):.6g}")

    figz, axz = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axz[0].plot(t, nrm(tau_d)); axz[0].plot(t, nrm(tau_cmd)); axz[0].plot(t, nrm(tau_app)); axz[0].set_ylabel("Torque norm")
    axz[1].plot(t, omega[:,0]); axz[1].plot(t, omega[:,1]); axz[1].plot(t, omega[:,2]); axz[1].set_ylabel("ω [rad/s]")
    axz[2].plot(t, nrm(h_vec)); axz[2].set_ylabel(h_label); axz[2].set_xlabel("time [s]")
    for a in axz:
        a.set_xlim(*zoom_x); a.set_ylim(*zoom_y)
    figz.suptitle(f"{label_prefix}{i_dep} — zoom {zoom_x[0]}–{zoom_x[1]} s (‖τ‖, ω, ‖h‖)", fontsize=45)
    figz.tight_layout(rect=[0, 0.02, 1, 0.94]); plt.show()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Plot ESPA+XB6 NPZ logs (aggregate and per-deputy).")
    ap.add_argument("--combined",   type=str, default=None, help="Path to combined NPZ (with carrier & sched).")
    ap.add_argument("--per-deputy", type=str, nargs="*", default=None, help="Paths or glob to per-deputy NPZs.")
    ap.add_argument("--engage",     type=float, default=10.3, help="Engagement time marker (default: 10.3 s).")
    ap.add_argument("--zoom-x",     type=float, nargs=2, default=(150.0, 250.0), help="Zoom x-limits (start end).")
    ap.add_argument("--zoom-y",     type=float, nargs=2, default=(-1.0, 1.0),   help="Zoom y-limits (min max).")
    args = ap.parse_args()

    combined_path = args.combined; per_paths = args.per_deputy

    # expand globs
    if per_paths:
        expanded = []
        for p in per_paths: expanded += glob.glob(p)
        per_paths = sorted(set(expanded))

    data_combined = None; num_deps = 0; has_rafti = False
    if combined_path is None and not per_paths:
        auto = auto_find_combined()
        if auto: combined_path = auto

    if combined_path:
        if not os.path.exists(combined_path): raise FileNotFoundError(f"Combined NPZ not found: {combined_path}")
        data_combined, num_deps, has_rafti = load_combined_npz(combined_path)
        print(f"[LOAD] Combined: {combined_path}  (deps={num_deps}, RAFTI={'yes' if has_rafti else 'no'})")

    deps_per = []
    if per_paths:
        deps_per = load_per_deputy_npz(per_paths)
        print(f"[LOAD] Per-deputy files: {len(deps_per)}")

    # unified accessor for aggregates when only per-deputy files exist
    class Proxy:
        def __init__(self, deps):
            self.deps = deps
            self.files = set().union(*[set(d.files) for d in deps])
        def __getitem__(self, k):
            m = re.match(r"D(\d+)_(.+)", k)
            if not m: return self.deps[0][k]
            i = int(m.group(1)) - 1; nm = m.group(2)
            return self.deps[i][nm]

    if data_combined is not None:
        N = num_deps; accessor = data_combined
    elif deps_per:
        N = len(deps_per); accessor = Proxy(deps_per)
    else:
        raise SystemExit("No data found. Provide --combined NPZ or --per-deputy files.")

    # Classic aggregates when combined file exists
    if data_combined is not None:
        plot_aggregate_dv(data_combined, N, has_rafti)
        plot_aggregate_pointing(data_combined, N)

    # New aggregates
    plot_aggregate_energy_all_deps(accessor, N, args.engage, label_prefix="XB6 D")
    plot_rpy_grid_all_deps(accessor, N, args.engage, label_prefix="XB6 D", print_euler_errors_at=250.0)

    # Per-deputy stacks + zooms
    for i in range(1, N+1):
        plot_per_deputy_detail(accessor, i, args.engage, tuple(args.zoom_x), tuple(args.zoom_y),
                               label_prefix="XB6 D", print_tau_errors_at=250.0)

    print("[DONE] Plots displayed.")

if __name__ == "__main__":
    main()




