"""
Visualize.py — fleet-ready Apophis rendezvous plotting toolkit
==============================================================

Purpose
-------
Provide a reusable set of Matplotlib-based plotting routines for Apophis
rendezvous / station-keeping simulations, especially those that export
“rich” NPZ bundles such as:

    apophis_rendezvous_full_D{k}.npz

The module assumes a `final_res` dict-style payload (from a single deputy),
containing arrays like:

- t_hist        : (N,)      time stamps [s] since some reference (e.g. departure)
- rD_hist       : (N,3)     deputy geocentric positions [m]
- vD_hist       : (N,3)     deputy geocentric velocities [m/s]
- rA_hist       : (N,3)     Apophis geocentric positions [m]
- vA_hist       : (N,3)     Apophis geocentric velocities [m/s]
- rC_hist       : (N,3)     chief geocentric positions [m]
- vC_hist       : (N,3)     chief geocentric velocities [m/s]
- C_A_N_hist    : (N,3,3)   Apophis LVLH←ECI DCMs (rows are RTN axes)
- sepA_hist     : (N,3)     deputy–Apophis LVLH separations [m] (optional;
                             recomputed from C_A_N & rD−rA if missing)
- err_hist      : (N,)      predicted terminal error norm [m]
- dv_times      : (Nb,)     burn times [s]
- dv_list       : (Nb,3)    burn Δv vectors in LVLH/RTN [m/s]
- impulses      : list of (t, dv) pairs (for burn markers)
- rNavD_N, vNavD_N (optional): nav outputs for deputy [m], [m/s]

It also expects scalar constants provided to `plot_all`:

- best_t         : deployment epoch [s]
- DT_DYN         : dynamics step size [s]
- APOPHIS_TARGET : target LVLH separation [m]
- TARGET_MARGIN_M: margin for LVLH bands [m]
- E_TRIGGER      : trigger threshold for error norm [m]
- V_MIN, V_MAX   : per-burn Δv min/max [m/s]
- DV_BUDGET_MS   : optional ΔV budget [m/s] (for impulse plots)
- R_EARTH_KM     : Earth radius [km]

Style & layout
--------------
- Global styling is configured via `_configure_style(font_size, tick_size)`:
    * Uses `plt.style.use("bmh")`.
    * Sets axes/title/label/tick/legend font sizes for large, slide-ready
      figures.
- `_maybe_save(fig, filename, save_dir)` will:
    * Save a PNG to `save_dir/filename` if both are provided,
    * Otherwise, do nothing (pure “show-only” mode).

Vector helpers
--------------
- `_eci_to_rtn_series(C_A_N, vN_series)`:
    * Applies a batch transform from ECI to RTN using C_A_N_hist (LVLH←N DCMs).
- `_vec_mag(v)`:
    * Returns ‖v‖₂ along axis 1.

Plot functions
--------------
Each plot builder returns the figure it created and optionally saves it via
`_maybe_save`.

1. `plot_geo_xy(final_res, best_t, DT_DYN, save_dir=None, title_prefix="")`
   - Geocentric XY view of deputy vs Apophis.
   - Marks:
       * Deploy (first point),
       * Arrival (last point),
       * Any burns from `final_res["impulses"]` (red points).

2. `plot_lvlh_components(final_res, best_t, APOPHIS_TARGET, TARGET_MARGIN_M, save_dir=None, title_prefix="")`
   - Stacked 3×1 plot of LVLH R/T/N components vs time since deploy [h].
   - Draws:
       * Target line per component,
       * Shaded ±margin band around the target.

3. `plot_rtn_combined(final_res, best_t, APOPHIS_TARGET=None, save_dir=None, title_prefix="")`
   - Combined RTN separation on a single axes.
   - Uses `sepA_hist` if present, otherwise reconstructs via C_A_N & rD−rA.
   - Optionally draws target lines per component.

4. `plot_error_norm(final_res, best_t, E_TRIGGER, save_dir=None, title_prefix="")`
   - Predicted terminal-error norm vs time since deploy [h].
   - Horizontal line at `E_TRIGGER`.

5. `plot_impulse_history(final_res, best_t, V_MIN, V_MAX, DV_BUDGET_MS=None, save_dir=None, title_prefix="")`
   - Two-panel figure:
       * Top: stems of Δv components (R/T/N) in mm/s vs time [min].
         Shows ±V_MAX and ±V_MIN bands.
       * Bottom: stems of |Δv| plus a cumulative ΔV [m/s] line, with optional
         budget line at `DV_BUDGET_MS`.

6. `plot_3d_trajectories(final_res, R_EARTH_KM, best_t, DT_DYN, save_dir=None, title_prefix="")`
   - 3D Earth-centred view showing:
       * Earth sphere of radius R_EARTH_KM,
       * Chief, deputy, and Apophis tracks,
       * Deploy, arrival, and burn locations (if any).

7. `plot_nav_errors(final_res, best_t, npz_path_for_nav=None, save_dir=None, title_prefix="")`
   - If nav outputs are available (either in `final_res` or in a separate NPZ)
     plots four figures:
       * ECI position-error magnitude vs time [h],
       * ECI velocity-error magnitude vs time [h],
       * RTN position errors (R/T/N components) vs time [h],
       * RTN velocity errors vs time [h].

Master driver
-------------
`plot_all(final_res, *, best_t, DT_DYN, APOPHIS_TARGET, TARGET_MARGIN_M, E_TRIGGER, V_MIN, V_MAX, DV_BUDGET_MS, R_EARTH_KM, save_dir=None, font_size=50, tick_size=45, npz_path_for_nav=None, title_prefix="", show=True)`

- Calls all the plot builders in a standard sequence:

    1. `plot_geo_xy`
    2. `plot_lvlh_components`
    3. `plot_rtn_combined`
    4. `plot_error_norm`
    5. `plot_impulse_history`
    6. `plot_3d_trajectories`
    7. `plot_nav_errors` (if nav data present)

- Applies `_configure_style(font_size, tick_size)` before plotting.
- Uses `title_prefix` for multi-deputy runs (e.g., "Dep 3: ").
- Optionally saves each figure to `save_dir` if provided.
- If `show=True`, calls `plt.show()` at the end.

Usage
-----
Typical calling pattern inside a rendezvous driver:

    import Visualize as viz

    viz.plot_all(
        final_res,
        best_t=best_t,
        DT_DYN=DT_DYN,
        APOPHIS_TARGET=target_vec_m,
        TARGET_MARGIN_M=1.0,
        E_TRIGGER=E_TRIGGER,
        V_MIN=V_MIN,
        V_MAX=V_MAX,
        DV_BUDGET_MS=None,
        R_EARTH_KM=R_EARTH_KM,
        save_dir="plots_D0",          # or None
        font_size=50,
        tick_size=45,
        npz_path_for_nav=None,        # or path to nav NPZ
        title_prefix="Dep 0: ",
        show=True
    )

All functions are fleet-ready and make no assumptions about index numbering
beyond what is encoded in the `final_res` dict.
"""


from __future__ import annotations
import os
import numpy as np
import numpy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  keep 3-D import

# ───────────────────── figure-style helper ────────────────────────
def _configure_style(font_size: int = 50, tick_size: int = 45):
    """Global Matplotlib style."""
    plt.style.use("bmh")
    mpl.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": tick_size,
        "figure.titlesize": font_size,
    })

def _maybe_save(fig: plt.Figure, filename: str | None, save_dir: str | None):
    """Save `fig` to `save_dir/filename` if asked."""
    if save_dir and filename:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, filename),
                    dpi=150, bbox_inches="tight")

# ───────────────────── simple vector helpers ──────────────────────
def _eci_to_rtn_series(C_A_N: np.ndarray, vN_series: np.ndarray) -> np.ndarray:
    """ECI → RTN transform for a timeseries."""
    return np.einsum("kij,kj->ki", C_A_N, vN_series)

def _vec_mag(v: np.ndarray) -> np.ndarray:
    return la.norm(v, axis=1)

# ───────────────────── individual plot builders ───────────────────
def plot_geo_xy(final_res: dict, best_t: float, DT_DYN: float,
                save_dir: str | None = None,
                title_prefix: str = "") -> plt.Figure:
    """Geocentric XY deputy & Apophis."""
    rA, rD = final_res["rA_hist"], final_res["rD_hist"]
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)
    ax.plot(rA[:, 0]/1e3, rA[:, 1]/1e3, label="Apophis (geo)", lw=2)
    ax.plot(rD[:, 0]/1e3, rD[:, 1]/1e3, label="Deputy (geo)",   lw=2)

    # burns
    if final_res["impulses"]:
        idxs = [min(int((t-best_t)/DT_DYN), len(rD)-1)
                for t, _ in final_res["impulses"]]
        ax.scatter(rD[idxs, 0]/1e3, rD[idxs, 1]/1e3,
                   c="r", s=40, label="Burns")

    # deploy / arrival
    ax.scatter(rD[0, 0]/1e3,  rD[0, 1]/1e3,  c="g", s=50, label="Deploy")
    ax.scatter(rD[-1, 0]/1e3, rD[-1, 1]/1e3, c="k", s=50, label="Arrival")

    ax.axis("equal")
    ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]")
    ax.set_title(f"{title_prefix}Geocentric XY: Deputy vs Apophis")
    ax.legend(loc="best")
    _maybe_save(fig, "xy_deputy_apophis.png", save_dir)
    return fig


def plot_lvlh_components(final_res: dict, best_t: float,
                         APOPHIS_TARGET: np.ndarray,
                         TARGET_MARGIN_M: float,
                         save_dir: str | None = None,
                         title_prefix: str = "") -> plt.Figure:
    """R/T/N separation (stacked)."""
    t_hr = (final_res["t_hist"] - best_t) / 3600.0
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    lab = ["R", "T", "N"]
    for i in range(3):
        axs[i].plot(t_hr, final_res["sepA_hist"][:, i]/1e3, lw=2, label=lab[i])
        targ = APOPHIS_TARGET[i]/1e3
        axs[i].axhline(targ, ls="--", color="k", label="target" if i == 0 else None)
        m = TARGET_MARGIN_M/1e3
        axs[i].fill_between(t_hr, targ-m, targ+m, color="gray", alpha=0.2)
        axs[i].set_ylabel(f"{lab[i]} [km]")
    axs[-1].set_xlabel("Time since deploy [h]")
    axs[0].set_title(f"{title_prefix}Deputy–Apophis separation (LVLH)")
    if axs[0].get_legend_handles_labels()[0]:
        axs[0].legend(loc="best")
    _maybe_save(fig, "lvlh_components_with_margin.png", save_dir)
    return fig


def plot_rtn_combined(final_res: dict, best_t: float,
                      APOPHIS_TARGET: np.ndarray | None = None,
                      save_dir: str | None = None,
                      title_prefix: str = "") -> plt.Figure:
    """Combined R/T/N separation one-axis."""
    t_hr = (final_res["t_hist"] - best_t) / 3600.0
    if "sepA_hist" in final_res:
        sepA = final_res["sepA_hist"]
    else:
        C = final_res["C_A_N_hist"]
        sepA = _eci_to_rtn_series(C, final_res["rD_hist"] - final_res["rA_hist"])
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.plot(t_hr, sepA[:, 0]/1e3, lw=4, label="R")
    ax.plot(t_hr, sepA[:, 1]/1e3, lw=4, label="T")
    ax.plot(t_hr, sepA[:, 2]/1e3, lw=4, label="N")
    if APOPHIS_TARGET is not None:
        ax.axhline(APOPHIS_TARGET[0]/1e3, ls="--", color="C0", alpha=.6)
        ax.axhline(APOPHIS_TARGET[1]/1e3, ls="--", color="C1", alpha=.6)
        ax.axhline(APOPHIS_TARGET[2]/1e3, ls="--", color="C2", alpha=.6)
    ax.set_title(f"{title_prefix}Combined RTN Separation (LVLH)")
    ax.set_xlabel("Time since deploy [h]"); ax.set_ylabel("Separation [km]")
    ax.grid(True); ax.legend(loc="best")
    _maybe_save(fig, "combined_RTN_separation.png", save_dir)
    return fig


def plot_error_norm(final_res: dict, best_t: float, E_TRIGGER: float,
                    save_dir: str | None = None,
                    title_prefix: str = "") -> plt.Figure:
    """Terminal-error norm prediction."""
    t_hr = (final_res["t_hist"] - best_t) / 3600.0
    fig = plt.figure(figsize=(9, 4)); ax = fig.add_subplot(111)
    ax.plot(t_hr, final_res["err_hist"], lw=2)
    ax.axhline(E_TRIGGER, ls="--", color="gray", label="trigger")
    ax.set_xlabel("Time since deploy [h]")
    ax.set_ylabel("‖pred sep − target‖ [m]")
    ax.set_title(f"{title_prefix}Predicted terminal error (arrival LVLH)")
    ax.legend(loc="best")
    _maybe_save(fig, "predicted_error_norm.png", save_dir)
    return fig


def plot_impulse_history(final_res: dict, best_t: float,
                         V_MIN: float, V_MAX: float,
                         DV_BUDGET_MS: float | None = None,
                         save_dir: str | None = None,
                         title_prefix: str = "") -> plt.Figure:
    """Impulse stems & cumulative ΔV."""
    dv_t = (final_res["dv_times"] - best_t) if final_res["dv_times"].size else np.array([])
    dv_rtn = final_res["dv_list"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if dv_t.size:
        axs[0].stem(dv_t/60, dv_rtn[:, 0]*1e3, linefmt="C0-", markerfmt="C0o",
                    basefmt=" ", label="R")
        axs[0].stem(dv_t/60, dv_rtn[:, 1]*1e3, linefmt="C1-", markerfmt="C1s",
                    basefmt=" ", label="T")
        axs[0].stem(dv_t/60, dv_rtn[:, 2]*1e3, linefmt="C2-", markerfmt="C2^",
                    basefmt=" ", label="N")
        axs[0].axhline(V_MAX*1e3,  ls="--", color="gray", alpha=.6)
        axs[0].axhline(-V_MAX*1e3, ls="--", color="gray", alpha=.6, label="±V_MAX")
        axs[0].axhline(V_MIN*1e3,  ls=":", color="0.3")
        axs[0].axhline(-V_MIN*1e3, ls=":", color="0.3", label="±V_MIN")
        axs[0].set_ylabel("Δv comp [mm/s]"); axs[0].legend(loc="best")

        mag = _vec_mag(dv_rtn); cum = np.cumsum(mag)
        axs[1].stem(dv_t/60, mag*1e3, linefmt="C3-", markerfmt="C3o",
                    basefmt=" ", label="|Δv|")
        axs[1].plot(dv_t/60, cum, lw=2, label="Cumulative ΔV [m/s]")
        if DV_BUDGET_MS is not None:
            axs[1].axhline(DV_BUDGET_MS, ls="--", color="k",
                           label=f"Budget {DV_BUDGET_MS:.2f} m/s")
        axs[1].set_xlabel("Time since deploy [min]")
        axs[1].set_ylabel("mm/s (stems) & m/s (line)")
        axs[1].set_title(f"{title_prefix}Impulse history & ΔV budget")
        axs[1].legend(loc="best")
    else:
        for ax in axs:
            ax.text(0.5, 0.5, "No impulses", ha="center", transform=ax.transAxes)
    _maybe_save(fig, "impulse_history.png", save_dir)
    return fig


def plot_3d_trajectories(final_res: dict, R_EARTH_KM: float,
                         best_t: float, DT_DYN: float,
                         save_dir: str | None = None,
                         title_prefix: str = "") -> plt.Figure:
    """3-D chief / deputy / Apophis."""
    rC = final_res["rC_hist"]/1e3
    rD = final_res["rD_hist"]/1e3
    rA = final_res["rA_hist"]/1e3
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    # Earth
    u,v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(R_EARTH_KM*np.cos(u)*np.sin(v),
                    R_EARTH_KM*np.sin(u)*np.sin(v),
                    R_EARTH_KM*np.cos(v),
                    color="lightsteelblue", alpha=.4, linewidth=0)
    # paths
    ax.plot(rC[:,0], rC[:,1], rC[:,2], "C0", lw=2, label="Chief")
    ax.plot(rD[:,0], rD[:,1], rD[:,2], "C1", lw=2, label="Deputy")
    ax.plot(rA[:,0], rA[:,1], rA[:,2], "C2", lw=2, label="Apophis")
    # marks
    ax.scatter(rD[0,0], rD[0,1], rD[0,2], c="g", s=50, label="Deploy")
    ax.scatter(rD[-1,0],rD[-1,1],rD[-1,2], c="k", s=50, label="Arrival")
    if final_res["impulses"]:
        idxs = [min(int((t-best_t)/DT_DYN), len(rD)-1)
                for t,_ in final_res["impulses"]]
        ax.scatter(rD[idxs,0], rD[idxs,1], rD[idxs,2],
                   c="r", s=18, label="Burns")
    rng = np.max(np.abs(np.vstack((rC,rD,rA))))
    ax.set_xlim([-rng,rng]); ax.set_ylim([-rng,rng]); ax.set_zlim([-rng,rng])
    ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
    ax.set_title(f"{title_prefix}3-D Trajectories")
    ax.legend(loc="upper right")
    _maybe_save(fig, "trajectories_3d.png", save_dir)
    return fig


def plot_nav_errors(final_res: dict, best_t: float,
                    npz_path_for_nav: str | None = None,
                    save_dir: str | None = None,
                    title_prefix: str = "") -> list[plt.Figure]:
    """SimpleNav error plots (4 figs) if nav data present."""
    rNav,vNav = None,None
    if npz_path_for_nav:
        try:
            d = np.load(npz_path_for_nav, allow_pickle=True)
            if "rNavD_N" in d and "vNavD_N" in d:
                rNav, vNav = d["rNavD_N"], d["vNavD_N"]
        except Exception as e:
            print(f"[Visualize] nav read fail: {e}")
    if (rNav is None or vNav is None) and \
       ("rNavD_N" in final_res and "vNavD_N" in final_res):
        rNav, vNav = final_res["rNavD_N"], final_res["vNavD_N"]
    if rNav is None or vNav is None:
        return []   # nothing to do

    rD, vD = final_res["rD_hist"], final_res["vD_hist"]
    C      = final_res["C_A_N_hist"]
    e_rN   = rNav - rD;  e_vN = vNav - vD
    e_rRTN = _eci_to_rtn_series(C, e_rN)
    e_vRTN = _eci_to_rtn_series(C, e_vN)
    t_hr   = (final_res["t_hist"] - best_t) / 3600.0
    figs=[]
    # mag pos
    f,ax = plt.subplots(figsize=(14,10)); ax.plot(t_hr,_vec_mag(e_rN),lw=4)
    ax.set_title(f"{title_prefix}Nav Position Error Mag (ECI)")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Error [m]"); ax.grid()
    _maybe_save(f,"nav_position_error_mag.png",save_dir); figs.append(f)
    # mag vel
    f,ax = plt.subplots(figsize=(14,10)); ax.plot(t_hr,_vec_mag(e_vN),lw=4)
    ax.set_title(f"{title_prefix}Nav Velocity Error Mag (ECI)")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Error [m/s]"); ax.grid()
    _maybe_save(f,"nav_velocity_error_mag.png",save_dir); figs.append(f)
    # pos RTN
    f,ax = plt.subplots(figsize=(14,10))
    ax.plot(t_hr,e_rRTN[:,0],lw=4,label="e_R"); ax.plot(t_hr,e_rRTN[:,1],lw=4,label="e_T")
    ax.plot(t_hr,e_rRTN[:,2],lw=4,label="e_N")
    ax.set_title(f"{title_prefix}Nav Position Error (RTN)")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Error [m]"); ax.grid(); ax.legend()
    _maybe_save(f,"nav_position_error_RTN.png",save_dir); figs.append(f)
    # vel RTN
    f,ax = plt.subplots(figsize=(14,10))
    ax.plot(t_hr,e_vRTN[:,0],lw=4,label="ė_R"); ax.plot(t_hr,e_vRTN[:,1],lw=4,label="ė_T")
    ax.plot(t_hr,e_vRTN[:,2],lw=4,label="ė_N")
    ax.set_title(f"{title_prefix}Nav Velocity Error (RTN)")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Error [m/s]"); ax.grid(); ax.legend()
    _maybe_save(f,"nav_velocity_error_RTN.png",save_dir); figs.append(f)
    return figs

# ─────────────────────────── master driver ────────────────────────
def plot_all(final_res: dict, *,
             best_t: float,
             DT_DYN: float,
             APOPHIS_TARGET: np.ndarray,
             TARGET_MARGIN_M: float,
             E_TRIGGER: float,
             V_MIN: float,
             V_MAX: float,
             DV_BUDGET_MS: float | None,
             R_EARTH_KM: float,
             save_dir: str | None = None,
             font_size: int = 50,
             tick_size: int = 45,
             npz_path_for_nav: str | None = None,
             title_prefix: str = "",
             show: bool = True) -> None:
    """
    Build the full figure set.

    Parameters
    ----------
    title_prefix : str
        Prepended to every figure title (e.g. "Dep 3: ").
    show : bool, default True
        Call ``plt.show()`` if True.  Set False when running many deputies
        head-lessly.
    """
    _configure_style(font_size, tick_size)

    plot_geo_xy(final_res, best_t, DT_DYN, save_dir, title_prefix)
    plot_lvlh_components(final_res, best_t, APOPHIS_TARGET, TARGET_MARGIN_M,
                         save_dir, title_prefix)
    plot_rtn_combined(final_res, best_t, APOPHIS_TARGET, save_dir, title_prefix)
    plot_error_norm(final_res, best_t, E_TRIGGER, save_dir, title_prefix)
    plot_impulse_history(final_res, best_t, V_MIN, V_MAX, DV_BUDGET_MS,
                         save_dir, title_prefix)
    plot_3d_trajectories(final_res, R_EARTH_KM, best_t, DT_DYN,
                         save_dir, title_prefix)
    plot_nav_errors(final_res, best_t, npz_path_for_nav, save_dir, title_prefix)

    plt.tight_layout()
    if show:
        plt.show()
