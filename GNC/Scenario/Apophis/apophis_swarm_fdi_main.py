#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_fdi_main.py
=========================

Offline FDI run for the Apophis fleet.

Per deputy Dk this script consumes:

  - apophis_rendezvous_full_Dk.npz   : RHIG guidance bundle (for formation/task)
  - apophis_swarm_FDI_rgazel_Dk.npz  : rel-nav EKF outputs in Hill
  - dep{k:02d}_logs.npz              : 6-DOF attitude tracking logs

and produces:

  - apophis_swarm_FDI_results_Dk.npz

containing FDI residuals, CUSUM histories, and state machine history over
the rel-nav timeline.

This version also builds a single, publication-style figure for one chosen
deputy, showing NAV / ATT / FORM+PEER residuals and the FDI state track.

Rel-nav upgrade
---------------
The FDI now consumes the range/azimuth/elevation rel-nav bundles produced by
``apophis_swarm_ckf_ekf_range_az_el.py`` (camera LOS + ranging beacon, with a
5 m chief-GPS frame).  That EKF reaches ~0.1-0.6 m per-axis accuracy — about
two orders of magnitude tighter than the old LOS-only nav (25-68 m) — and is
statistically consistent (mean NIS/channel ≈ 1.0).  The FDI is re-tuned to
exploit that consistency rather than the de-tuned thresholds the coarse nav
needed:

  * NAV  : NAV_R_SCALE = 1.0 (no artificial covariance inflation); nu_nav /
           leak_nav / h_nav returned to principled, fault-sensitive values;
           EKF-rejected outliers are skipped instead of false-tripping the
           CUSUM (outlier-aware NAV channel).
  * FORM : the |e_form| normalization floor is relaxed now that the
           fleet-median scale is trustworthy.
  * PEER : the fleet MAD is floored so a homogeneous fleet no longer makes
           r_peer explode (the old code false-faulted D2/D5 this way).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# ─────────── journal-quality plotting style ───────────

# Font sizes (points) tuned for a double-column journal figure.
TITLE_FONTSIZE      = 13
AXIS_LABEL_FONTSIZE = 11
TICK_LABEL_FONTSIZE = 9
LEGEND_FONTSIZE     = 9

# Colour-blind-safe palette (Okabe-Ito) for the four FDI channels, plus a
# distinct colour reserved for fault thresholds and fault shading.
CHANNEL_COLORS = {
    "NAV":  "#0072B2",   # blue
    "ATT":  "#009E73",   # bluish green
    "FORM": "#CC79A7",   # reddish purple
    "PEER": "#E69F00",   # orange
}
THRESHOLD_COLOR = "#D55E00"   # vermillion - fault threshold h / fault shading


def _setup_journal_style() -> str:
    """
    Apply a journal-quality matplotlib style.

    Prefers the ``scienceplots`` package (IEEE / Nature-grade presets).  If it
    is not installed the function silently falls back to a self-contained
    publication rcParams block, so the script always runs.  Returns the name
    of the style that was actually applied.

    Install the full styling with:  pip install scienceplots
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


# Apply the style once, at import time.
_FDI_PLOT_STYLE = _setup_journal_style()

# Which deputy gets the detailed per-deputy figure
PAPER_DEPUTY_IDX = 0  # 0..7

# How much of the tail of the ATT residual to hide from the plot (minutes).
# Retained for backward compatibility only; the ATT terminal artifact is now
# handled upstream (see ATT_TERMINAL_GUARD_S) so this is no longer used.
ATT_CROP_LAST_MIN = 0.2


# ───────── locate project root / import FDI core ─────────

SCRIPT_DIR = Path(__file__).resolve().parent

ROOT = None
for parent in SCRIPT_DIR.parents:
    if (parent / "GNC").is_dir():
        ROOT = parent
        break
if ROOT is None:
    ROOT = SCRIPT_DIR.parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNC.Navigation.fdi_core import FDICore, FDIConfig, FDIState, FaultType  # type: ignore[import]

# ───────── config knobs ─────────

N_DEP = 8   # deputies 0..7

# ── rel-nav input bundle ─────────────────────────────────────────────
# Filename stem of the rel-nav EKF bundles consumed by the FDI.  These are
# the range/azimuth/elevation outputs of apophis_swarm_ckf_ekf_range_az_el.py
# (keys: t, x_true, x_est, P, y_pre, y_post, sigma_meas, dv_seq_H).
NAV_BUNDLE_PREFIX = "apophis_swarm_FDI_rgazel_D"   # + "{d}.npz"

# Formation residual window: active only in the last W seconds before flyby
FORM_WINDOW_DURATION_S = 3600.0   # [s] last hour before flyby

# Lower bound on normalization scale for |e_form| (meters).
# With the consistent rgazel nav the fleet-median |e_form| is a trustworthy
# scale, so this floor only needs to guard against a degenerate (near-zero)
# median.  It was 2500 m as a crutch for the noisy LOS-only nav, which made
# FORM insensitive whenever the fleet was genuinely tight; now relaxed.
FORM_SIGMA_M = 250.0              # used as floor; actual scale = max(median, this)

# Use CW-based predictor to arrival (True) or instantaneous separation (False)
FORM_USE_CW_PREDICTOR = True

# NAV innovation covariance scaling for FDI (larger → smaller r_nav).
# The rgazel EKF is statistically consistent, so sigma_meas already
# calibrates the innovation covariance honestly and no artificial inflation
# is required — r_nav becomes a true per-channel NIS RMS (≈1 nominal).
# (The old LOS-only nav used 3.0 to suppress false NAV alarms.)
NAV_R_SCALE = 1.0

# Peer-consistency MAD floor.  r_peer divides a deputy's deviation from the
# fleet median by the fleet MAD of r_form; when the fleet is homogeneous
# that MAD collapses toward zero and r_peer explodes for any deputy with a
# tiny, benign geometric difference (the old code false-faulted D2/D5 this
# way).  Flooring the MAD keeps a nominal deputy's r_peer ≈ O(1) while a
# genuine fleet outlier still scores far above nu_peer.
PEER_MAD_FLOOR = 0.25

# Attitude-log terminal artifact guard.  The 6-DOF attitude logs end with a
# 1-2 sample transient at the sim boundary (r_att jumps to ~20-30 on the very
# last sample) that is NOT a real attitude fault.  The ATT channel holds the
# last valid attitude residual through the final ATT_TERMINAL_GUARD_S seconds
# so this artifact never enters the ATT CUSUM (it previously pushed S_att to
# ~25-32 against h_att=8, surviving only by lack of persistence).
ATT_TERMINAL_GUARD_S = 60.0


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Basic helpers: interpolation
# ╚══════════════════════════════════════════════════════════════════════════╝

def _interp_vec_series(t_grid: np.ndarray,
                       data: np.ndarray,
                       t_query: float) -> np.ndarray:
    """Linear interpolation of a (N,3) vector series at scalar time t_query."""
    t = np.asarray(t_grid, float).reshape(-1)
    arr = np.asarray(data,   float).reshape(-1, data.shape[-1])

    if t_query <= t[0]:
        return arr[0]
    if t_query >= t[-1]:
        return arr[-1]

    j = np.searchsorted(t, t_query) - 1
    dt = t[j+1] - t[j]
    if dt <= 0.0:
        return arr[j]

    f = (t_query - t[j]) / dt
    return (1.0 - f) * arr[j] + f * arr[j+1]


def _interp_scalar_series(t_grid: np.ndarray,
                          data: np.ndarray,
                          t_query: float) -> float:
    """Linear interpolation of a (N,) scalar series at scalar time t_query."""
    t = np.asarray(t_grid, float).reshape(-1)
    arr = np.asarray(data, float).reshape(-1)

    if t_query <= t[0]:
        return float(arr[0])
    if t_query >= t[-1]:
        return float(arr[-1])

    j = np.searchsorted(t, t_query) - 1
    dt = t[j+1] - t[j]
    if dt <= 0.0:
        return float(arr[j])

    f = (t_query - t[j]) / dt
    return float((1.0 - f) * arr[j] + f * arr[j+1])


def _find_global_t0_abs(root: Path) -> float:
    """Earliest time across RHIG bundles (for aligning attitude logs)."""
    t0s = []
    for d in range(N_DEP):
        npz = np.load(root / f"apophis_rendezvous_full_D{d}.npz")
        try:
            t0s.append(float(npz["t"][0]))
        finally:
            npz.close()
    return min(t0s)


# ╔══════════════════════════════════════════════════════════════════════════╗
#  LVLH / CW utilities
# ╚══════════════════════════════════════════════════════════════════════════╝

def dcm_lvlh(r_N: np.ndarray, v_N: np.ndarray) -> np.ndarray:
    """
    Hill (LVLH)-from-ECI DCM.

    Rows are unit vectors of R,T,N expressed in ECI.
    """
    r = np.asarray(r_N, float)
    v = np.asarray(v_N, float)
    i = r / la.norm(r)
    k = np.cross(r, v)
    k = k / la.norm(k)
    j = np.cross(k, i)
    return np.vstack((i, j, k))


def hill_rate(r_N: np.ndarray, v_N: np.ndarray) -> float:
    """Mean motion n from chief position/velocity."""
    h = np.cross(r_N, v_N)
    return la.norm(h) / (la.norm(r_N) ** 2)


def cw_blocks(n: float, dt: float):
    """Clohessy–Wiltshire state transition sub-blocks for mean motion n, step dt."""
    c = np.cos(n * dt)
    s = np.sin(n * dt)
    nt = n * dt

    Phi_rr = np.array([
        [4.0 - 3.0 * c, 0.0, 0.0],
        [6.0 * (s - nt), 1.0, 0.0],
        [0.0, 0.0, c],
    ])

    Phi_rv = np.array([
        [s / n, 2.0 * (1.0 - c) / n, 0.0],
        [-2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * nt) / n, 0.0],
        [0.0, 0.0, s / n],
    ])

    Phi_vr = np.array([
        [3.0 * n * s, 0.0, 0.0],
        [6.0 * n * (c - 1.0), 0.0, 0.0],
        [0.0, 0.0, -n * s],
    ])

    Phi_vv = np.array([
        [c, 2.0 * s, 0.0],
        [-2.0 * s, 4.0 * c - 3.0, 0.0],
        [0.0, 0.0, c],
    ])

    return Phi_rr, Phi_rv, Phi_vr, Phi_vv


def _predict_sepA_at_arrival(
    t_now: float,
    t_arr: float,
    rho_now: np.ndarray,
    rhod_now: np.ndarray,
    t_full: np.ndarray,
    rC_full: np.ndarray,
    vC_full: np.ndarray,
    rA_full: np.ndarray,
    vA_full: np.ndarray,
) -> np.ndarray:
    """
    Predict Apophis-LVLH separation at fixed arrival time t_arr from a Hill
    state (rho, rhod) at time t_now using CW dynamics around the chief.
    """
    dt_go = max(t_arr - t_now, 0.0)

    # chief & Apophis at now and arrival
    rC_now = _interp_vec_series(t_full, rC_full, t_now)
    vC_now = _interp_vec_series(t_full, vC_full, t_now)

    rC_arr = _interp_vec_series(t_full, rC_full, t_arr)
    vC_arr = _interp_vec_series(t_full, vC_full, t_arr)
    rA_arr = _interp_vec_series(t_full, rA_full, t_arr)
    vA_arr = _interp_vec_series(t_full, vA_full, t_arr)

    # CW propagation in chief LVLH
    n = hill_rate(rC_now, vC_now)
    if n <= 0.0 or dt_go == 0.0:
        rho_f = np.asarray(rho_now, float)
    else:
        Phi_rr, Phi_rv, _, _ = cw_blocks(n, dt_go)
        rho_f = Phi_rr @ rho_now + Phi_rv @ rhod_now

    # map to ECI at arrival then into Apophis LVLH
    C_C_arr = dcm_lvlh(rC_arr, vC_arr)   # Hill-from-ECI at chief
    C_A_arr = dcm_lvlh(rA_arr, vA_arr)   # Apophis-LVLH-from-ECI

    rD_arr = rC_arr + C_C_arr.T @ rho_f
    sepA_arr = C_A_arr @ (rD_arr - rA_arr)
    return sepA_arr


def _instantaneous_sepA(
    t_now: float,
    rho_now: np.ndarray,
    t_full: np.ndarray,
    rC_full: np.ndarray,
    vC_full: np.ndarray,
    rA_full: np.ndarray,
    vA_full: np.ndarray,
) -> np.ndarray:
    """
    Instantaneous Apophis-LVLH separation at t_now from a Hill state rho_now.
    """
    rC = _interp_vec_series(t_full, rC_full, t_now)
    vC = _interp_vec_series(t_full, vC_full, t_now)
    rA = _interp_vec_series(t_full, rA_full, t_now)
    vA = _interp_vec_series(t_full, vA_full, t_now)

    C_C = dcm_lvlh(rC, vC)
    C_A = dcm_lvlh(rA, vA)

    rD = rC + C_C.T @ rho_now
    sepA = C_A @ (rD - rA)
    return sepA


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Fleet data assembly
# ╚══════════════════════════════════════════════════════════════════════════╝

def _load_fleet_data(root: Path):
    """
    Load nav, RHIG, and attitude logs for all deputies.
    """
    fleet = []

    for d in range(N_DEP):
        nav_npz_path  = root / f"{NAV_BUNDLE_PREFIX}{d}.npz"
        full_npz_path = root / f"apophis_rendezvous_full_D{d}.npz"
        att_npz_path  = root / f"dep{d:02d}_logs.npz"

        if not nav_npz_path.is_file():
            raise FileNotFoundError(nav_npz_path)
        if not full_npz_path.is_file():
            raise FileNotFoundError(full_npz_path)
        if not att_npz_path.is_file():
            raise FileNotFoundError(att_npz_path)

        try:
            nav = np.load(nav_npz_path)
        except Exception as exc:   # e.g. truncated / corrupt .npz
            raise RuntimeError(
                f"Could not read rel-nav bundle {nav_npz_path.name}: {exc}. "
                f"Regenerate it by re-running apophis_swarm_ckf_ekf_range_az_el.py "
                f"for deputy D{d}."
            ) from exc
        full = np.load(full_npz_path)
        att  = np.load(att_npz_path)

        # rel-nav bundle
        t_nav      = nav["t"].astype(float)          # (N_nav,)
        x_est      = nav["x_est"].astype(float)      # (N_nav,6)
        P_hist     = nav["P"].astype(float)          # (N_nav,6,6)
        y_pre      = nav["y_pre"].astype(float)      # (N_nav,3)
        y_post     = nav["y_post"].astype(float)     # (N_nav,3)
        sigma_meas = nav["sigma_meas"].astype(float) # (N_nav,3)
        dv_seq_H   = nav["dv_seq_H"].astype(float)   # (N_nav,3)

        # Outlier-aware NAV mask.  The rgazel EKF runs a Mahalanobis gate;
        # when it rejects a measurement it holds its prediction and writes
        # the (large) pre-fit innovation into BOTH y_pre and y_post.  So a
        # finite sample with y_pre exactly equal to y_post is an EKF-rejected
        # outlier — a sensor glitch the filter already absorbed, not a
        # navigation fault.  The NAV channel skips these (see _run_fdi_*).
        finite_row   = np.isfinite(y_pre).all(axis=1)
        nav_rejected = finite_row & np.all(y_pre == y_post, axis=1)

        # full RHIG bundle (truth + guidance)
        t_full = full["t"].astype(float)             # (N_full,)
        rD_N   = full["rD_N"].astype(float)          # (N_full,3)
        rC_N   = full["rC_N"].astype(float)          # (N_full,3)
        vC_N   = full["vC_N"].astype(float)          # (N_full,3)
        rA_N   = full["rA_N"].astype(float)          # (N_full,3)
        vA_N   = full["vA_N"].astype(float)          # (N_full,3)
        target = full["APOPHIS_TARGET"].astype(float).reshape(3)

        # attitude logs
        t_err      = att["t_err"].astype(float)      # (Na,)
        sigma_BR   = att["sigma_BR"].astype(float)   # (Na,3)
        omega_BR_B = att["omega_BR"].astype(float)   # (Na,3)

        fleet.append(
            dict(
                t_nav=t_nav,
                x_est=x_est,
                P_hist=P_hist,
                y_pre=y_pre,
                y_post=y_post,
                nav_rejected=nav_rejected,
                sigma_meas=sigma_meas,
                dv_seq_H=dv_seq_H,
                t_full=t_full,
                rD_N=rD_N,
                rC_N=rC_N,
                vC_N=vC_N,
                rA_N=rA_N,
                vA_N=vA_N,
                target=target,
                t_err=t_err,
                sigma_BR=sigma_BR,
                omega_BR_B=omega_BR_B,
            )
        )

    return fleet


def _estimate_global_arrival_time(fleet):
    """
    Estimate a common flyby time t_arr_global as the median time of minimum
    D–Apophis separation across all deputies (using truth).
    """
    t_arr_list = []

    for d in range(N_DEP):
        t_full = fleet[d]["t_full"]
        rD_N   = fleet[d]["rD_N"]
        rA_N   = fleet[d]["rA_N"]

        sep = la.norm(rD_N - rA_N, axis=1)
        idx_min = int(np.argmin(sep))
        t_arr_list.append(float(t_full[idx_min]))

    t_arr_arr = np.array(t_arr_list, float)
    t_arr_global = float(np.median(t_arr_arr))
    return t_arr_global, t_arr_arr


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Formation & peer residuals
# ╚══════════════════════════════════════════════════════════════════════════╝

def _compute_formation_residuals(
    fleet,
    t_arr_global: float,
    window_duration_s: float = FORM_WINDOW_DURATION_S,
    sigma_form: float = FORM_SIGMA_M,
    use_cw_predictor: bool = FORM_USE_CW_PREDICTOR,
):
    """
    Build time-gated, *normalized* formation residual r_form for each deputy.
    """
    all_e_raw = []

    # first pass: compute raw error norms and weights per deputy
    for d in range(N_DEP):
        t_nav   = fleet[d]["t_nav"]
        x_est   = fleet[d]["x_est"]
        t_full  = fleet[d]["t_full"]
        rC_N    = fleet[d]["rC_N"]
        vC_N    = fleet[d]["vC_N"]
        rA_N    = fleet[d]["rA_N"]
        vA_N    = fleet[d]["vA_N"]
        target  = fleet[d]["target"]

        N_nav = t_nav.size

        e_raw = np.zeros(N_nav)
        w_vec = np.zeros(N_nav)
        active = np.zeros(N_nav, dtype=bool)

        t_start_window = t_arr_global - window_duration_s

        for k in range(N_nav):
            t_k = float(t_nav[k])

            # Outside active window or after flyby → no formation residual
            if t_k < t_start_window or t_k >= t_arr_global:
                continue

            active[k] = True

            rho  = x_est[k, 0:3]
            rhod = x_est[k, 3:6]

            if use_cw_predictor:
                sepA = _predict_sepA_at_arrival(
                    t_k, t_arr_global,
                    rho, rhod,
                    t_full, rC_N, vC_N, rA_N, vA_N
                )
            else:
                sepA = _instantaneous_sepA(
                    t_k, rho,
                    t_full, rC_N, vC_N, rA_N, vA_N
                )

            e_form = sepA - target
            e_norm = la.norm(e_form)

            e_raw[k] = e_norm
            all_e_raw.append(e_norm)

            # time-to-go weighting: 1.0 at start of window → 2.5 at flyby
            t_go = max(t_arr_global - t_k, 0.0)
            alpha = 1.0 - t_go / max(window_duration_s, 1e-6)
            alpha = np.clip(alpha, 0.0, 1.0)
            w = 1.0 + 1.5 * alpha   # 1 → 2.5
            w_vec[k] = w

        fleet[d]["e_form_raw"] = e_raw
        fleet[d]["w_form"] = w_vec
        fleet[d]["form_active"] = active
        fleet[d]["t_arr_global"] = t_arr_global

    # global normalization scale
    all_e_raw = np.asarray(all_e_raw, float)
    if all_e_raw.size == 0:
        med = 0.0
        scale = sigma_form
    else:
        med = float(np.median(all_e_raw))
        scale = max(med, sigma_form)

    print(
        f"Formation normalization: median |e_form| = {med:.1f} m, "
        f"scale = {scale:.1f} m"
    )

    # second pass: build r_form using common scale
    for d in range(N_DEP):
        e_raw   = fleet[d]["e_form_raw"]
        w_vec   = fleet[d]["w_form"]
        active  = fleet[d]["form_active"]

        r_form = np.zeros_like(e_raw)
        if np.any(active):
            r_form[active] = (e_raw[active] / max(scale, 1e-6)) * w_vec[active]

        fleet[d]["r_form"] = r_form


def _compute_peer_residuals(fleet):
    """
    For each deputy and each of its nav timestamps, compute a peer-consistency
    metric r_peer based on r_form across the fleet.
    """
    t_nav_list  = [f["t_nav"]  for f in fleet]
    r_form_list = [f["r_form"] for f in fleet]

    for d in range(N_DEP):
        t_nav_d   = t_nav_list[d]
        r_form_d  = r_form_list[d]
        N_nav     = t_nav_d.size
        r_peer_d  = np.zeros(N_nav)

        for k in range(N_nav):
            t_k = float(t_nav_d[k])
            vals = []

            for j in range(N_DEP):
                t_j = t_nav_list[j]
                if t_k < t_j[0] or t_k > t_j[-1]:
                    continue  # deputy j not active at this time
                r_j = _interp_scalar_series(t_j, r_form_list[j], t_k)
                vals.append(r_j)

            if len(vals) < 2:
                r_peer_d[k] = 0.0
                continue

            vals = np.asarray(vals, float)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            # Floor the MAD.  A homogeneous fleet drives the raw MAD toward
            # zero, which would make r_peer explode for any deputy with a
            # tiny, benign geometric difference (this is what false-faulted
            # D2/D5 under the old code).  PEER_MAD_FLOOR keeps a nominal
            # deputy near O(1) while a genuine outlier still scores high.
            mad_eff = max(mad, PEER_MAD_FLOOR)
            r_peer_d[k] = abs(r_form_d[k] - med) / mad_eff

        fleet[d]["r_peer"] = r_peer_d


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Paper-style plotting for one deputy
# ╚══════════════════════════════════════════════════════════════════════════╝

def _plot_deputy_paper_figure(
    dep_idx: int,
    dep_data: dict,
    t_fdi: np.ndarray,
    r_nav: np.ndarray,
    r_att: np.ndarray,
    r_form: np.ndarray,
    r_peer: np.ndarray,
    S_nav: np.ndarray,
    S_att: np.ndarray,
    S_form: np.ndarray,
    S_peer: np.ndarray,
    state_code: np.ndarray,
    fault_code: np.ndarray,
    cfg: FDIConfig,
) -> None:
    """
    Journal-quality FDI visualization for one deputy.

    Each FDI channel gets two panels side by side:

      * left  (DIAGNOSTIC) - the per-step residual r with its CUSUM
        reference nu.  Crossing nu is normal; nu is only the value
        subtracted from the leaky CUSUM each step, not a fault line.
      * right (DECISION)   - the leaky-CUSUM detector S with its fault
        threshold h.  The FDI flags the channel only when S exceeds h.
        The annotation reports peak S and the fraction of h used.

    A fifth row shows the FDI state track.  Plotting S against h makes the
    fault margin explicit: a healthy deputy keeps every S well below h.
    """
    t_fdi = np.asarray(t_fdi, float).reshape(-1)
    t0 = t_fdi[0]
    t_hr = (t_fdi - t0) / 3600.0

    t_arr_global = float(dep_data["t_arr_global"])
    t_arr_hr   = (t_arr_global - t0) / 3600.0
    t_start_hr = (t_arr_global - FORM_WINDOW_DURATION_S - t0) / 3600.0
    form_window = (t_start_hr, t_arr_hr)

    channels = [
        ("NAV",  np.asarray(r_nav, float),  np.asarray(S_nav, float),
         cfg.nu_nav,  cfg.h_nav,  cfg.leak_nav,  CHANNEL_COLORS["NAV"],  None),
        ("ATT",  np.asarray(r_att, float),  np.asarray(S_att, float),
         cfg.nu_att,  cfg.h_att,  cfg.leak_att,  CHANNEL_COLORS["ATT"],  None),
        ("FORM", np.asarray(r_form, float), np.asarray(S_form, float),
         cfg.nu_form, cfg.h_form, cfg.leak_form, CHANNEL_COLORS["FORM"], form_window),
        ("PEER", np.asarray(r_peer, float), np.asarray(S_peer, float),
         cfg.nu_peer, cfg.h_peer, cfg.leak_peer, CHANNEL_COLORS["PEER"], form_window),
    ]

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 0.62],
                          hspace=0.62, wspace=0.18,
                          left=0.075, right=0.975, top=0.900, bottom=0.070)

    for i, (name, r, S, nu, h, leak, color, window) in enumerate(channels):
        ax_r = fig.add_subplot(gs[i, 0])
        ax_s = fig.add_subplot(gs[i, 1])

        if window is not None:
            for ax in (ax_r, ax_s):
                ax.axvspan(window[0], window[1], color="0.87",
                           alpha=0.7, lw=0, zorder=0)

        # --- left: per-step residual vs CUSUM reference nu ---
        finite_r = np.isfinite(r)
        ax_r.plot(t_hr, r, color=color, linewidth=1.1)
        ax_r.axhline(nu, color="0.35", linestyle="--", linewidth=1.0)
        ax_r.fill_between(t_hr, nu, r, where=finite_r & (r > nu),
                          color=color, alpha=0.16, interpolate=True)
        ax_r.set_ylabel(rf"$r_\mathrm{{{name}}}$")
        ax_r.set_ylim(bottom=0.0)
        ax_r.margins(x=0.01)
        ax_r.set_title(rf"{name}  $\cdot$  per-step residual   "
                       rf"(dashed $\nu={nu:g}$: CUSUM reference, not a fault line)",
                       loc="left", fontsize=10)

        # --- right: leaky-CUSUM detector S vs fault threshold h ---
        finite_S = np.isfinite(S)
        ax_s.plot(t_hr, S, color=color, linewidth=1.4)
        ax_s.axhline(h, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.4)
        ax_s.fill_between(t_hr, h, S, where=finite_S & (S > h),
                          color=THRESHOLD_COLOR, alpha=0.25, interpolate=True)
        peak = float(np.nanmax(S)) if np.any(finite_S) else 0.0
        frac = peak / h if h > 0 else 0.0
        tripped = peak > h
        ax_s.set_ylabel(rf"$S_\mathrm{{{name}}}$")
        ax_s.set_ylim(0.0, max(h * 1.20, peak * 1.25, 1.0))
        ax_s.margins(x=0.01)
        ax_s.set_title(rf"{name}  $\cdot$  leaky-CUSUM detector   "
                       rf"($\lambda={leak:g}$, fault threshold $h={h:g}$)",
                       loc="left", fontsize=10)
        ax_s.text(0.975, 0.90,
                  f"peak $S$ = {peak:.2f}   ({frac:.0%} of $h$)   "
                  + ("FAULT" if tripped else "healthy margin"),
                  transform=ax_s.transAxes, ha="right", va="top", fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.32",
                            fc=("#fadbd8" if tripped else "#e3f1e3"),
                            ec=("#c0392b" if tripped else "#5a8f5a"), lw=0.9))

        if i == len(channels) - 1:
            ax_r.set_xlabel("Time since nav start  [h]")
            ax_s.set_xlabel("Time since nav start  [h]")

    # --- FDI state track (spans both columns) ---
    ax_state = fig.add_subplot(gs[4, :])
    sc = np.asarray(state_code, int)
    ax_state.step(t_hr, sc, where="post", color="0.22", linewidth=1.6)
    fault_mask = (sc == FDIState.FAULTED.value)
    if np.any(fault_mask):
        ax_state.fill_between(t_hr, sc.min() - 0.2, sc.max() + 0.2,
                              where=fault_mask, color=THRESHOLD_COLOR,
                              alpha=0.13, step="post")
    ax_state.axvline(t_arr_hr, color="0.35", linestyle=":", linewidth=1.3)
    uniq = np.unique(sc)
    labels = []
    for s in uniq:
        try:
            labels.append(FDIState(int(s)).name)
        except ValueError:
            labels.append(str(s))
    ax_state.set_yticks(uniq)
    ax_state.set_yticklabels(labels)
    ax_state.set_ylim(uniq.min() - 0.5, uniq.max() + 0.5)
    ax_state.set_ylabel("FDI state")
    ax_state.set_xlabel("Time since nav start  [h]")
    ax_state.margins(x=0.01)
    ax_state.set_title(r"FDI state machine   (dotted line = Apophis flyby)",
                       loc="left", fontsize=10)

    # --- title, reading guide, column headers, result line ---
    fig.suptitle(f"FDI overview – Deputy {dep_idx}",
                 fontsize=TITLE_FONTSIZE, y=0.987)
    fig.text(0.5, 0.957,
             "Left column = per-step residual (noisy; crossing the dashed "
             "reference is expected).   Right column = the leaky-CUSUM "
             "detector – a fault is raised only if it crosses threshold h.",
             ha="center", fontsize=9, style="italic", color="0.25")
    fig.text(0.275, 0.927, "DIAGNOSTIC  ·  per-step residual $r$",
             ha="center", fontsize=10.5, weight="bold", color="0.15")
    fig.text(0.737, 0.927, "DECISION  ·  CUSUM detector $S$ vs threshold $h$",
             ha="center", fontsize=10.5, weight="bold", color="0.15")

    final_state_name = FDIState(int(sc[-1])).name
    final_fault_name = FaultType(int(np.asarray(fault_code, int)[-1])).name
    healthy = (final_state_name == "HEALTHY")
    fig.text(0.5, 0.020,
             f"Result:   final state = {final_state_name}      "
             f"identified fault = {final_fault_name}      "
             f"formation window = last {FORM_WINDOW_DURATION_S/60:.0f} min "
             f"before flyby (shaded)",
             ha="center", fontsize=10, weight="bold",
             color=("#1d6f1d" if healthy else "#b3261e"))


def _plot_fleet_summary(summaries, cfg: FDIConfig) -> None:
    """
    Journal-quality fleet-wide FDI health overview.

    Grouped bar chart: for every deputy and every channel the bar height is
    peak_S / h - the fraction of that channel's fault threshold the leaky
    CUSUM actually reached.  A deputy is healthy when every bar stays well
    below the 1.0 line (= threshold h).  This is the at-a-glance
    "is anyone close to faulting?" view for the whole swarm.
    """
    if not summaries:
        return

    channels = [
        ("NAV",  cfg.h_nav,  CHANNEL_COLORS["NAV"]),
        ("ATT",  cfg.h_att,  CHANNEL_COLORS["ATT"]),
        ("FORM", cfg.h_form, CHANNEL_COLORS["FORM"]),
        ("PEER", cfg.h_peer, CHANNEL_COLORS["PEER"]),
    ]
    deps = [int(s["dep"]) for s in summaries]
    n = len(deps)
    x = np.arange(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6))

    worst = 0.0
    for j, (name, h, color) in enumerate(channels):
        fracs = []
        for s in summaries:
            pk = float(s["peak_S"].get(name, 0.0))
            fracs.append(pk / h if h > 0 else 0.0)
        worst = max(worst, max(fracs) if fracs else 0.0)
        bars = ax.bar(x + (j - 1.5) * width, fracs, width,
                      label=name, color=color, edgecolor="0.25",
                      linewidth=0.5)
        for b, f in zip(bars, fracs):
            ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.018,
                    f"{f:.0%}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(1.0, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.7)
    ax.text(n - 0.45, 1.03, r"fault threshold $h$  ($S/h = 1$)",
            color=THRESHOLD_COLOR, ha="right", va="bottom",
            fontsize=9, weight="bold")

    state_labels = []
    for s in summaries:
        try:
            state_labels.append(FDIState(int(s["state"])).name)
        except ValueError:
            state_labels.append(str(s["state"]))
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{d}\n{st}" for d, st in zip(deps, state_labels)])
    ax.set_ylabel(r"peak CUSUM  $S / h$   (fraction of fault threshold)")
    ax.set_ylim(0.0, max(1.18, worst * 1.3))
    ax.set_title("FDI fleet health – peak CUSUM margin per deputy",
                 fontsize=TITLE_FONTSIZE)
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(False, axis="x")
    ax.legend(ncol=4, loc="upper left", title="channel")
    fig.text(0.5, 0.005,
             "Each bar = fraction of a channel's fault threshold reached by "
             "its leaky CUSUM.  Bars far below the red line = wide fault "
             "margin = healthy.",
             ha="center", fontsize=8, style="italic", color="0.3")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Core per-deputy driver
# ╚══════════════════════════════════════════════════════════════════════════╝

def _run_fdi_for_deputy(
    dep_idx: int,
    root: Path,
    cfg: FDIConfig,
    t0_abs: float,
    dep_data: dict,
) -> dict:
    """
    Run FDI for one deputy, save results NPZ, and optionally make the
    paper-style figure for PAPER_DEPUTY_IDX.
    """
    t_nav      = dep_data["t_nav"]
    x_est      = dep_data["x_est"]
    P_hist     = dep_data["P_hist"]
    y_pre      = dep_data["y_pre"]
    sigma_meas = dep_data["sigma_meas"]
    dv_seq_H   = dep_data["dv_seq_H"]
    t_err      = dep_data["t_err"]
    sigma_BR   = dep_data["sigma_BR"]
    omega_BR_B = dep_data["omega_BR_B"]
    r_form     = dep_data["r_form"]
    r_peer     = dep_data["r_peer"]
    nav_rejected = dep_data["nav_rejected"]

    # map rel-nav absolute times to attitude sim relative times:
    # t_att_rel = t_abs - t0_abs  (attitude sim ran with t_rel=0 at t0_abs)
    t_att_rel = t_nav - t0_abs

    N = t_nav.size

    # FDI core
    fdi = FDICore(cfg)
    fdi.reset(t0=float(t_nav[0]))

    # pre-allocate outputs
    r_nav_hist   = np.zeros(N)
    r_prop_hist  = np.zeros(N)
    r_att_hist   = np.zeros(N)
    r_comm_hist  = np.zeros(N)
    r_form_hist  = np.zeros(N)
    r_peer_hist  = np.zeros(N)
    state_code   = np.zeros(N, dtype=int)
    fault_code   = np.zeros(N, dtype=int)
    confidence   = np.zeros(N)
    S_nav_hist   = np.zeros(N)
    S_att_hist   = np.zeros(N)
    S_comm_hist  = np.zeros(N)
    S_form_hist  = np.zeros(N)
    S_peer_hist  = np.zeros(N)

    # COMM placeholder: heartbeat nominal + age (no real comm model yet)
    hb_nominal = 60.0  # [s] arbitrary; r_comm ≈ 1.0 with no outage

    # outlier-aware NAV bookkeeping
    n_nav_outlier = 0   # EKF-rejected measurements skipped by the NAV channel

    # main loop over rel-nav samples
    for k in range(N):
        t_k    = float(t_nav[k])
        t_relk = float(t_att_rel[k])

        # NAV channel: innovation and S (~S = R only).
        # Skip both dropouts (NaN innovations) and EKF-rejected outliers:
        # each means "no trustworthy NAV measurement this step", so the
        # NAV CUSUM simply leaks instead of reacting to a sensor glitch
        # the rel-nav EKF already gated out.
        innov = y_pre[k]
        if (not np.all(np.isfinite(innov))) or bool(nav_rejected[k]):
            innov_k   = None
            S_innov_k = None
            if bool(nav_rejected[k]):
                n_nav_outlier += 1
        else:
            innov_k = innov
            sigma_k = np.maximum(sigma_meas[k], 1e-3)
            sigma_k_eff = NAV_R_SCALE * sigma_k
            R_k = np.diag(sigma_k_eff**2)
            S_innov_k = R_k

        # PROP channel: only active when a burn is commanded
        dv_cmd    = dv_seq_H[k]
        burn_flag = bool(la.norm(dv_cmd) > 0.0)

        dv_est_H = None
        P_vv     = None
        if burn_flag and k > 0:
            v_prev = x_est[k-1, 3:6]
            v_now  = x_est[k,   3:6]
            dv_est = (v_now - v_prev) - dv_cmd
            dv_est_H = dv_est
            P_vv     = P_hist[k, 3:6, 3:6]

        # ATT channel: interpolate σ_BR and ω_BR_B from attitude logs.
        # Clamp the query time off the attitude-log terminal artifact so the
        # ATT residual holds its last valid value instead of spiking on the
        # sim-boundary transient (see ATT_TERMINAL_GUARD_S).
        t_att_q = min(t_relk, float(t_err[-1]) - ATT_TERMINAL_GUARD_S)
        sig_BR_k = _interp_vec_series(t_err, sigma_BR,   t_att_q)
        omg_BR_k = _interp_vec_series(t_err, omega_BR_B, t_att_q)

        # COMM channel: hold heartbeat at nominal (no outage model yet)
        hb_age_k = hb_nominal

        # formation/task + peer residuals (already computed on nav grid)
        r_form_k = float(r_form[k])
        r_peer_k = float(r_peer[k])

        snap = fdi.step(
            t=t_k,
            innov=innov_k,
            S_innov=S_innov_k,
            dv_est_H=dv_est_H,
            P_vv=P_vv,
            burn_flag=burn_flag,
            sigma_BR=sig_BR_k,
            omega_BR_B=omg_BR_k,
            hb_age=hb_age_k,
            hb_nominal=hb_nominal,
            r_form=r_form_k,
            r_peer=r_peer_k,
        )

        r_nav_hist[k]   = snap.r_nav
        r_prop_hist[k]  = snap.r_prop
        r_att_hist[k]   = snap.r_att
        r_comm_hist[k]  = snap.r_comm
        r_form_hist[k]  = snap.r_form
        r_peer_hist[k]  = snap.r_peer
        state_code[k]   = snap.state.state.value
        fault_code[k]   = snap.state.fault_type.value
        confidence[k]   = snap.state.confidence

        # CUSUMs from the core
        S_nav_hist[k]   = fdi.S_nav
        S_att_hist[k]   = fdi.S_att
        S_comm_hist[k]  = fdi.S_comm
        S_form_hist[k]  = fdi.S_form
        S_peer_hist[k]  = fdi.S_peer

    # store results
    out_path = root / f"apophis_swarm_FDI_results_D{dep_idx}.npz"
    np.savez_compressed(
        out_path,
        t_fdi=t_nav,
        r_nav=r_nav_hist,
        r_prop=r_prop_hist,
        r_att=r_att_hist,
        r_comm=r_comm_hist,
        r_form=r_form_hist,
        r_peer=r_peer_hist,
        state_code=state_code,
        fault_code=fault_code,
        confidence=confidence,
        S_nav=S_nav_hist,
        S_att=S_att_hist,
        S_comm=S_comm_hist,
        S_form=S_form_hist,
        S_peer=S_peer_hist,
    )
    print(
        f"[dep{dep_idx}] FDI results -> {out_path.name}  "
        f"(final state={FDIState(state_code[-1]).name}, "
        f"fault={FaultType(fault_code[-1]).name}, "
        f"max r_nav={r_nav_hist.max():.2f}, "
        f"max r_form={r_form_hist.max():.2f}, "
        f"max r_peer={r_peer_hist.max():.2f}, "
        f"NAV outliers skipped={n_nav_outlier})"
    )

    # per-channel peak CUSUM summary (consumed by the fleet-health figure)
    summary = dict(
        dep=dep_idx,
        state=int(state_code[-1]),
        fault=int(fault_code[-1]),
        peak_S=dict(
            NAV=float(S_nav_hist.max()),
            ATT=float(S_att_hist.max()),
            FORM=float(S_form_hist.max()),
            PEER=float(S_peer_hist.max()),
        ),
    )

    # Build the detailed per-deputy figure only for the chosen deputy
    if dep_idx == PAPER_DEPUTY_IDX:
        _plot_deputy_paper_figure(
            dep_idx=dep_idx,
            dep_data=dep_data,
            t_fdi=t_nav,
            r_nav=r_nav_hist,
            r_att=r_att_hist,
            r_form=r_form_hist,
            r_peer=r_peer_hist,
            S_nav=S_nav_hist,
            S_att=S_att_hist,
            S_form=S_form_hist,
            S_peer=S_peer_hist,
            state_code=state_code,
            fault_code=fault_code,
            cfg=cfg,
        )

    return summary


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Entry point
# ╚══════════════════════════════════════════════════════════════════════════╝

def main(data_dir: str | Path | None = None) -> None:
    """
    Run FDI sweep.

    If data_dir is None, use the directory where this script lives
    (Scenario/Apophis), which is where all the NPZ bundles are.
    """
    if data_dir is None:
        root = SCRIPT_DIR
    else:
        root = Path(data_dir).resolve()

    print(f"\nFDI sweep over Apophis fleet in {root}")

    t0_abs = _find_global_t0_abs(root)
    print(f"Global t0_abs (earliest RHIG sample) = {t0_abs:.1f} s")

    # Load all deputies' data
    fleet = _load_fleet_data(root)

    # Estimate common flyby time from truth
    t_arr_global, t_arrs = _estimate_global_arrival_time(fleet)
    print(f"Estimated global flyby time t_arr_global = {t_arr_global:.1f} s")
    print(
        "Per-deputy min-range times (s): "
        + ", ".join(f"{t:.1f}" for t in t_arrs)
    )

    # Construct formation + peer residuals
    _compute_formation_residuals(
        fleet,
        t_arr_global=t_arr_global,
        window_duration_s=FORM_WINDOW_DURATION_S,
        sigma_form=FORM_SIGMA_M,
        use_cw_predictor=FORM_USE_CW_PREDICTOR,
    )
    _compute_peer_residuals(fleet)

    # FDI configuration:
    cfg = FDIConfig(
        use_nav_channel=True,
        use_att_channel=True,
        use_comm_channel=False,
        use_form_channel=True,
        use_peer_channel=True,
        # FORM: a healthy fleet's weighted r_form peaks ≈2.4 (time-to-go
        # weight 1→2.5 × e_form/scale ≈ 1).  nu_form sits clear above that
        # so a healthy fleet never accumulates S_form; h_form trips for a
        # deputy whose predicted terminal miss runs well past the fleet.
        nu_form=3.0,
        leak_form=0.02,
        h_form=30.0,
        # PEER: with PEER_MAD_FLOOR a nominal deputy scores r_peer = O(1);
        # a genuine fleet outlier scores many times nu_peer.
        nu_peer=3.0,
        leak_peer=0.05,
        h_peer=20.0,
    )

    # NAV thresholds.  The rgazel EKF is statistically consistent and
    # NAV_R_SCALE = 1.0, so r_nav is a true per-channel NIS RMS: a healthy
    # run has median r_nav ≈ 0.38 and p99 ≈ 1.4.  nu_nav just above that
    # band keeps a healthy run from accumulating S_nav, while a sustained
    # navigation fault (r_nav ≳ 2.3) drives S_nav past h_nav within the
    # persistence window.  These replace the de-tuned 5.0 / 0.02 / 60.0
    # values that the coarse LOS-only nav required.
    cfg.nu_nav = 1.5
    cfg.leak_nav = 0.05
    cfg.h_nav = 15.0

    summaries = []
    for d in range(N_DEP):
        summaries.append(_run_fdi_for_deputy(d, root, cfg, t0_abs, fleet[d]))

    # fleet-wide health overview across all deputies
    _plot_fleet_summary(summaries, cfg)

    print("\nFDI sweep complete.")
    plt.show()


if __name__ == "__main__":
    # no args → uses SCRIPT_DIR = Scenario/Apophis
    main()
