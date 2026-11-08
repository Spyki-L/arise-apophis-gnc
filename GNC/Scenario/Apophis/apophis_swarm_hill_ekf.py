#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_ckf_ekf_demo.py
=============================

Multi-deputy CKF + EKF demo for the Apophis rendezvous case, but with
SDEKF-style **range-only** measurements:

  z_k = || r_D(t_k) - r_A(t_k) || + bias_k + noise_k

State is deputy Hill (chief LVLH):

  x = [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N]^T.

Inputs
------
Script expects to sit in the same directory as the RHIG fleet bundles:

    Scenario/Apophis/
        apophis_rendezvous_full_D0.npz
        ...
        apophis_rendezvous_full_D7.npz

Each full NPZ is assumed to contain (as written by the RHIG fleet script):

    t        : (N,)   global time [s since departure]
    rD_N     : (N,3)  deputy ECI position [m]
    vD_N     : (N,3)  deputy ECI velocity [m/s]
    rC_N     : (N,3)  chief ECI position [m]
    vC_N     : (N,3)  chief ECI velocity [m/s]
    rA_N     : (N,3)  Apophis ECI position [m]
    vA_N     : (N,3)  Apophis ECI velocity [m/s]
    impulses : (Nb,4) [t, dv_R, dv_T, dv_N] in Hill [m/s] (optional)

Per deputy, we:

  * Take its time grid (starting at that deputy's deploy time).
  * Crop a bit after deployment.
  * Decimate by MEAS_DECIM.
  * Convert truth ECI → chief Hill.
  * Build **range-only** measurements to Apophis with realistic noise.
  * Run N_CKF passes to refine the initial state.
  * Run a final EKF pass.
  * Plot position/velocity errors vs 3σ and range residuals.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# ───────── locate ARISE root ─────────

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

from GNC.Navigation.hill_ekf import (
    HillEKF, HillEKFConfig,
    dcm_lvlh, hill_rate
)

# ───────── configuration ─────────

N_DEP      = 8      # deputies D0..D7 from the fleet script
MEAS_DECIM = 10     # decimate 1 Hz truth to this cadence for nav
N_CKF      = 2      # CKF passes before final EKF

# "True" range accuracy envelope (1σ)
RANGE_HI_TRUE = 200.0   # [m] early
RANGE_LO_TRUE =  50.0   # [m] late
TAU_HOURS     = 2.5     # time constant for decay

# Filter's *assumed* σ_r (mis-model vs truth)
RANGE_SCALE_ASSUMED = 0.8  # EKF thinks range is ~20% better

# Bias / dropout / outlier
BIAS_RANGE_STD   = 50.0     # [m] 1σ of slow bias
BIAS_TAU_HOURS   = 4.0      # correlation time
P_DROPOUT        = 0.05     # probability of no range at an epoch
P_OUTLIER        = 0.01     # outlier probability
OUTLIER_SCALE    = 5.0      # outlier magnitude multiplier


# ───────── helper: load a deputy's bundle ─────────

def load_apophis_bundle(path: Path) -> Dict[str, np.ndarray]:
    """
    Load fields needed for navigation from a full Apophis bundle.
    """
    wanted = (
        "t", "rD_N", "vD_N", "rC_N", "vC_N",
        "rA_N", "vA_N", "impulses"
    )
    out: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as npz:
        for k in wanted:
            if k in npz:
                out[k] = npz[k]
    if "impulses" not in out:
        out["impulses"] = np.zeros((0, 4), dtype=float)
    return out


# ───────── interpolators from bundle ─────────

def make_state_fn_from_bundle(
    data: Dict[str, np.ndarray],
    r_key: str,
    v_key: str,
):
    """
    Build ECI state interpolator from a single bundle:

        fn(t) -> (r_N(t), v_N(t))
    """
    t_arr = np.asarray(data["t"], float).reshape(-1)
    r = np.asarray(data[r_key], float).reshape(-1, 3)
    v = np.asarray(data[v_key], float).reshape(-1, 3)

    def interp_vec(t: float, arr: np.ndarray) -> np.ndarray:
        if t <= t_arr[0]:
            return arr[0]
        if t >= t_arr[-1]:
            return arr[-1]
        j = np.searchsorted(t_arr, t) - 1
        f = (t - t_arr[j]) / (t_arr[j+1] - t_arr[j] + 1e-15)
        return (1.0 - f) * arr[j] + f * arr[j+1]

    def state_fn(t: float) -> Tuple[np.ndarray, np.ndarray]:
        return interp_vec(t, r), interp_vec(t, v)

    return state_fn


# ───────── burns mapped onto a time grid ─────────

def build_burn_sequence(impulses: np.ndarray,
                        t_arr: np.ndarray,
                        dt: float) -> np.ndarray:
    """
    Map impulses [t, dv_R, dv_T, dv_N] into Δv_H (N,3) aligned with t_arr.

    Any impulse with time t_imp is applied to the sample whose time is within
    ±0.5*dt of t_imp.
    """
    impulses = np.asarray(impulses, float).reshape(-1, 4)
    dv_seq = np.zeros((len(t_arr), 3))
    if impulses.size == 0:
        return dv_seq

    impulses = impulses[np.argsort(impulses[:, 0])]
    idx_imp = 0
    for k, t in enumerate(t_arr):
        while idx_imp < impulses.shape[0] and impulses[idx_imp, 0] < t - 0.5*dt:
            idx_imp += 1
        if idx_imp < impulses.shape[0] and abs(impulses[idx_imp, 0] - t) <= 0.5*dt:
            dv_seq[k, :] = impulses[idx_imp, 1:4]
            idx_imp += 1
    return dv_seq


# ───────── ECI → Hill truth conversion ─────────

def eci_to_hill_traj(
    rD_N: np.ndarray,
    vD_N: np.ndarray,
    rC_N: np.ndarray,
    vC_N: np.ndarray
) -> np.ndarray:
    """
    Convert deputy ECI truth + chief ECI truth into Hill states:

        x = [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N]^T
    """
    rD_N = np.asarray(rD_N, float)
    vD_N = np.asarray(vD_N, float)
    rC_N = np.asarray(rC_N, float)
    vC_N = np.asarray(vC_N, float)

    N = rD_N.shape[0]
    x_true = np.zeros((N, 6))

    for k in range(N):
        rC = rC_N[k]
        vC = vC_N[k]
        rD = rD_N[k]
        vD = vD_N[k]

        C_HN = dcm_lvlh(rC, vC)               # Hill-from-ECI
        r_rel_N = rD - rC
        v_rel_N = vD - vC

        rho = C_HN @ r_rel_N
        omega_HN = np.array([0.0, 0.0, hill_rate(rC, vC)])
        v_rot = C_HN @ v_rel_N
        rhod = v_rot - np.cross(omega_HN, rho)

        x_true[k, 0:3] = rho
        x_true[k, 3:6] = rhod

    return x_true


# ───────── range-only measurement generator ─────────

def build_range_meas(
    rng: np.random.Generator,
    t_arr: np.ndarray,
    rD_N: np.ndarray,
    rA_N: np.ndarray,
    range_hi_true: float = RANGE_HI_TRUE,
    range_lo_true: float = RANGE_LO_TRUE,
    tau_hours: float = TAU_HOURS,
    bias_std: float = BIAS_RANGE_STD,
    bias_tau_hours: float = BIAS_TAU_HOURS,
    p_dropout: float = P_DROPOUT,
    p_outlier: float = P_OUTLIER,
    outlier_scale: float = OUTLIER_SCALE,
):
    """
    Range-only sensor model:

        z_k = ||rD_k - rA_k|| + b_k + w_k

      - w_k: zero-mean Gaussian with time-varying σ (true).
      - b_k: slowly varying bias (AR(1) random walk).
      - Dropouts and outliers are injected.

    Returns
    -------
    z_meas   : (N,)  noisy ranges
    sigma_assumed : (N,) EKF-assumed 1σ range (per sample)
    valid    : (N,) bool mask, True => measurement present
    """
    t_arr = np.asarray(t_arr, float).reshape(-1)
    rD_N = np.asarray(rD_N, float).reshape(-1, 3)
    rA_N = np.asarray(rA_N, float).reshape(-1, 3)

    N = len(t_arr)
    if N < 2:
        dt_eff = 1.0
    else:
        dt_eff = float(np.median(np.diff(t_arr)))

    # time-varying true σ
    t_rel = t_arr - t_arr[0]
    tau = tau_hours * 3600.0
    w = np.exp(-t_rel / tau)
    sigma_true = range_lo_true + (range_hi_true - range_lo_true) * w

    # bias AR(1)
    bias_tau_sec = bias_tau_hours * 3600.0
    rho_bias = np.exp(-dt_eff / max(bias_tau_sec, 1e-3))
    q_bias = bias_std * np.sqrt(1.0 - rho_bias**2)

    b = 0.0
    z_meas = np.zeros(N)
    valid = np.ones(N, dtype=bool)

    for k in range(N):
        if k > 0:
            b = rho_bias * b + q_bias * rng.normal()

        r_rel = rD_N[k] - rA_N[k]
        r_true = la.norm(r_rel)

        noise = rng.normal() * sigma_true[k]
        if rng.random() < p_outlier:
            noise *= outlier_scale

        z_meas[k] = r_true + b + noise

        if k == 0:
            valid[k] = True   # always keep first measurement
        else:
            valid[k] = (rng.random() >= p_dropout)

    sigma_assumed = RANGE_SCALE_ASSUMED * sigma_true
    return z_meas, sigma_assumed, valid


# ───────── range prediction & Jacobian in Hill coordinates ─────────

def range_pred_and_jacobian(
    x_H: np.ndarray,
    t: float,
    chief_state_fn,
    apophis_state_fn,
) -> Tuple[float, np.ndarray]:
    """
    Given Hill state x_H and time t, predict range to Apophis and
    compute Jacobian H = ∂h/∂x in Hill coordinates.

    x_H = [ρ, ρ̇]^T (6,)

    Steps:
      - chief ECI state (rC, vC) from chief_state_fn
      - Apophis ECI rA from apophis_state_fn
      - C_HN = Hill-from-ECI (chief)
      - rD = rC + C_HN^T ρ
      - h = || rD - rA ||

      gradient wrt ρ:
        u = (rD - rA)/||rD - rA||
        ∂h/∂ρ = (C_HN u)^T
      gradient wrt ρ̇ is zero.
    """
    x_H = np.asarray(x_H, float).reshape(6)
    rho = x_H[0:3]

    rC, vC = chief_state_fn(t)
    rA, _  = apophis_state_fn(t)

    C_HN = dcm_lvlh(rC, vC)          # Hill-from-ECI
    rD = rC + C_HN.T @ rho
    r_rel = rD - rA
    r_norm = la.norm(r_rel)

    if r_norm < 1e-6:
        u = np.zeros(3)
    else:
        u = r_rel / r_norm

    grad_rho = (C_HN @ u).T          # 1x3

    H = np.zeros((1, 6))
    H[0, 0:3] = grad_rho
    # H[:,3:6] = 0

    return float(r_norm), H


# ───────── generic EKF update for scalar range ─────────

def ekf_update_range(
    ekf: HillEKF,
    t: float,
    z: float,
    R: float,
    chief_state_fn,
    apophis_state_fn,
):
    """
    Nonlinear EKF update with scalar range measurement:

        z = h(x, t) + v,   v ~ N(0, R)

    where h(x,t) is range to Apophis.
    """
    # predicted measurement & Jacobian at x_pred
    z_hat, H = range_pred_and_jacobian(ekf.x_pred, t, chief_state_fn, apophis_state_fn)

    z = float(z)
    R_mat = np.array([[float(R)]])
    H = np.asarray(H, float).reshape(1, ekf.x.size)
    I = np.eye(ekf.x.size)

    y = np.array([z - z_hat])          # innovation (1,)
    S = H @ ekf.P_pred @ H.T + R_mat   # (1,1)
    K = ekf.P_pred @ H.T @ la.inv(S)   # (6,1)

    x_new = ekf.x_pred + (K @ y).reshape(-1)

    if ekf.cfg.use_joseph:
        KH = K @ H
        ekf.P = (I - KH) @ ekf.P_pred @ (I - KH).T + K @ R_mat @ K.T
    else:
        ekf.P = (I - K @ H) @ ekf.P_pred

    ekf.x = x_new

    # store residuals (scalar)
    ekf.innov = y.copy()
    ekf.S = S.copy()

    # pre-fit innovation
    ekf.y_pre = y.copy()

    # post-fit residual (recompute at updated state)
    z_hat_post, _ = range_pred_and_jacobian(ekf.x, t, chief_state_fn, apophis_state_fn)
    ekf.y_post = np.array([z - z_hat_post])

    ekf.H_last = H.copy()


# ───────── single-deputy CKF+EKF helper ─────────

def run_ckf_ekf_for_deputy(
    dep_idx: int,
    t_arr: np.ndarray,
    x_true: np.ndarray,
    rD_sel: np.ndarray,
    rA_sel: np.ndarray,
    dv_seq_H: np.ndarray,
    chief_state_fn,
    apophis_state_fn,
    N_ckf: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    CKF + EKF for one deputy with **range-only** measurements.

    Returns
    -------
    x_est_hist : (N,6)   EKF state history
    P_hist     : (N,6,6) EKF covariance history
    y_pre_hist : (N,)    scalar pre-fit innovation (range)
    y_post_hist: (N,)    scalar post-fit residual (range)
    sigma_r_assumed_final : (N,) EKF-assumed 1σ range at each epoch
    """
    dt = float(np.median(np.diff(t_arr)))

    cfg = HillEKFConfig(
        dt=dt,
        sigma_proc_pos=1.0,
        sigma_proc_vel=1e-3,
        sigma_meas_pos=20.0,   # not used directly now
        sigma_meas_vel=0.01,
        sigma_init_pos=1500.0,
        sigma_init_vel=0.5,
        sigma_dv=0.05,
        use_joseph=True,
    )
    ekf = HillEKF(cfg, chief_state_fn=chief_state_fn)

    # Initial state bias
    x0_bias = np.array([300.0, -250.0, 150.0, 0.12, -0.09, 0.05])
    x0_guess = x_true[0] + x0_bias

    p_pos0 = cfg.sigma_init_pos**2
    p_vel0 = cfg.sigma_init_vel**2
    P0_big = np.diag([p_pos0, p_pos0, p_pos0, p_vel0, p_vel0, p_vel0])

    print(f"[dep{dep_idx}] INIT ||x0_guess - x_true(0)|| = "
          f"{la.norm(x0_guess - x_true[0]):.3f}, Tr(P0) = {np.trace(P0_big):.3e}")

    # CKF passes: refine initial state only (still using range-only measurements)
    if N_ckf > 0:
        print(f"[dep{dep_idx}] CKF {N_ckf} passes...")
    for it in range(N_ckf):
        ekf.reset(x0_guess, P0_big)

        rng_pass = np.random.default_rng(1000 + dep_idx * 10 + it)
        z_pass, sigma_assumed_pass, valid_pass = build_range_meas(
            rng_pass, t_arr, rD_sel, rA_sel,
        )

        ekf.x_pred = ekf.x.copy()
        ekf.P_pred = ekf.P.copy()
        ekf.t_last = t_arr[0]

        # first measurement (forced valid)
        R0 = sigma_assumed_pass[0]**2
        ekf_update_range(ekf, t_arr[0], z_pass[0], R0,
                         chief_state_fn, apophis_state_fn)

        for k in range(1, len(t_arr)):
            Rk = sigma_assumed_pass[k]**2

            ekf.predict(t_arr[k], burn_dv_H=dv_seq_H[k])

            if valid_pass[k]:
                ekf_update_range(ekf, t_arr[k], z_pass[k], Rk,
                                 chief_state_fn, apophis_state_fn)
            else:
                # prediction only
                ekf.x = ekf.x_pred.copy()
                ekf.P = ekf.P_pred.copy()
                ekf.y_pre = np.array([0.0])
                ekf.y_post = np.array([0.0])

        err_end = la.norm(ekf.x - x_true[-1])
        print(f"    pass {it+1}: ||x_end - x_true_end|| = {err_end:.3f}, "
              f"Tr(P_end) = {np.trace(ekf.P):.3e}")
        x0_guess = ekf.x.copy()

    # Final EKF pass with fresh measurements
    print(f"[dep{dep_idx}] EKF final pass...")
    ekf.reset(x0_guess, P0_big)

    rng_final = np.random.default_rng(2000 + dep_idx)
    z_meas, sigma_assumed_final, valid_final = build_range_meas(
        rng_final, t_arr, rD_sel, rA_sel,
    )

    N = len(t_arr)
    x_est_hist = np.zeros((N, 6))
    P_hist = np.zeros((N, 6, 6))
    y_pre_hist = np.full(N, np.nan)
    y_post_hist = np.full(N, np.nan)

    ekf.x_pred = ekf.x.copy()
    ekf.P_pred = ekf.P.copy()
    ekf.t_last = t_arr[0]

    R0 = sigma_assumed_final[0]**2
    ekf_update_range(ekf, t_arr[0], z_meas[0], R0,
                     chief_state_fn, apophis_state_fn)
    x_est_hist[0] = ekf.x
    P_hist[0] = ekf.P
    y_pre_hist[0] = ekf.y_pre[0]
    y_post_hist[0] = ekf.y_post[0]

    for k in range(1, N):
        Rk = sigma_assumed_final[k]**2
        ekf.predict(t_arr[k], burn_dv_H=dv_seq_H[k])

        if valid_final[k]:
            ekf_update_range(ekf, t_arr[k], z_meas[k], Rk,
                             chief_state_fn, apophis_state_fn)
            y_pre_hist[k] = ekf.y_pre[0]
            y_post_hist[k] = ekf.y_post[0]
        else:
            ekf.x = ekf.x_pred.copy()
            ekf.P = ekf.P_pred.copy()
            # leave y_* as NaN

        x_est_hist[k] = ekf.x
        P_hist[k] = ekf.P

    return x_est_hist, P_hist, y_pre_hist, y_post_hist, sigma_assumed_final


# ───────── main swarm demo ─────────

def apophis_swarm_ckf_ekf_demo(
    data_dir: Path,
    N_dep: int = N_DEP,
    meas_decim: int = MEAS_DECIM,
    N_ckf: int = N_CKF,
):
    """
    For each apophis_rendezvous_full_Dk.npz in `data_dir`, run CKF+EKF and
    produce plots of position/velocity errors vs 3σ and range residuals.
    """
    data_dir = Path(data_dir)

    x_true_swarm = []
    x_est_swarm  = []
    P_swarm      = []
    y_pre_swarm  = []
    y_post_swarm = []
    sigma_r_swarm = []
    t_swarm      = []

    for d in range(N_dep):
        bundle_path = data_dir / f"apophis_rendezvous_full_D{d}.npz"
        if not bundle_path.is_file():
            raise FileNotFoundError(f"Missing bundle: {bundle_path}")

        print(f"\n===== Deputy {d} – loading {bundle_path.name} =====")
        data = load_apophis_bundle(bundle_path)

        t_full  = np.asarray(data["t"], float).reshape(-1)
        rD_full = np.asarray(data["rD_N"], float).reshape(-1, 3)
        vD_full = np.asarray(data["vD_N"], float).reshape(-1, 3)
        rC_full = np.asarray(data["rC_N"], float).reshape(-1, 3)
        vC_full = np.asarray(data["vC_N"], float).reshape(-1, 3)
        rA_full = np.asarray(data["rA_N"], float).reshape(-1, 3)
        vA_full = np.asarray(data["vA_N"], float).reshape(-1, 3)
        impulses = np.asarray(data["impulses"], float).reshape(-1, 4)

        # crop to skip the first ~0.5 h after deploy (like GEO)
        t0_full = t_full[0]
        t_crop_start = t0_full + 0.5 * 3600.0
        idx_crop = int(np.searchsorted(t_full, t_crop_start))
        t_crop_arr = t_full[idx_crop:]

        idx_decim_local = np.arange(0, len(t_crop_arr), meas_decim)
        idx_sel = idx_crop + idx_decim_local
        t_arr = t_full[idx_sel]

        dt_full = float(np.median(np.diff(t_full)))
        dv_full = build_burn_sequence(impulses, t_full, dt_full)
        dv_seq_H = dv_full[idx_sel]

        rD_sel = rD_full[idx_sel]
        vD_sel = vD_full[idx_sel]
        rC_sel = rC_full[idx_sel]
        vC_sel = vC_full[idx_sel]
        rA_sel = rA_full[idx_sel]
        vA_sel = vA_full[idx_sel]

        # Hill truth for this deputy on the decimated grid
        x_true = eci_to_hill_traj(rD_sel, vD_sel, rC_sel, vC_sel)

        # chief & Apophis interpolators (use full histories)
        chief_state_fn   = make_state_fn_from_bundle(data, "rC_N", "vC_N")
        apophis_state_fn = make_state_fn_from_bundle(data, "rA_N", "vA_N")

        # run CKF+EKF for this deputy
        (x_est_d, P_d,
         y_pre_d, y_post_d,
         sigma_r_final) = run_ckf_ekf_for_deputy(
            dep_idx=d,
            t_arr=t_arr,
            x_true=x_true,
            rD_sel=rD_sel,
            rA_sel=rA_sel,
            dv_seq_H=dv_seq_H,
            chief_state_fn=chief_state_fn,
            apophis_state_fn=apophis_state_fn,
            N_ckf=N_ckf,
        )

        x_true_swarm.append(x_true)
        x_est_swarm.append(x_est_d)
        P_swarm.append(P_d)
        y_pre_swarm.append(y_pre_d)
        y_post_swarm.append(y_post_d)
        sigma_r_swarm.append(sigma_r_final)
        t_swarm.append(t_arr)

    # ─── diagnostics / plots ───

    labels = ["R", "T", "N"]

    for d in range(N_dep):
        print(f"\n=== Deputy {d} EKF results (range-only) ===")
        t_arr = t_swarm[d]
        t_rel = t_arr - t_arr[0]
        t_hr = t_rel / 3600.0

        x_true_d = x_true_swarm[d]
        x_est_d  = x_est_swarm[d]
        err_d    = x_est_d - x_true_d

        # skip initial tiny segment
        mask = t_hr >= 0.1
        t_hr_plot = t_hr[mask]

        err_plot = err_d[mask]
        P_diag_plot = np.diagonal(P_swarm[d][mask], axis1=1, axis2=2)
        three_sigma = 3.0 * np.sqrt(P_diag_plot)

        # stats
        for i, lbl in enumerate(labels):
            e = err_plot[:, i]
            print(f"[POS] {lbl}: mean={np.mean(e):.3f} m, "
                  f"RMS={np.sqrt(np.mean(e**2)):.3f} m")

        # 1) Position error vs 3σ
        fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            ax = axes1[i]
            e = err_plot[:, i]
            s = three_sigma[:, i]
            ax.scatter(t_hr_plot, e, s=6, alpha=0.4,
                       label=f"dep{d} error ρ_{labels[i]}")
            ax.plot(t_hr_plot, s, "r--", linewidth=1.0,
                    label="+3σ" if i == 0 else None)
            ax.plot(t_hr_plot, -s, "r--", linewidth=1.0,
                    label="-3σ" if i == 0 else None)
            ax.set_ylabel(f"{labels[i]} [m]")
            ax.grid(True)
        axes1[-1].set_xlabel("Time since deploy [h]")
        axes1[0].legend()
        fig1.suptitle(f"Apophis rendezvous – Position error vs 3σ – Deputy {d}")
        fig1.tight_layout()

        # 2) Velocity error vs 3σ
        fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            ax = axes2[i]
            e = err_plot[:, 3+i]
            s = three_sigma[:, 3+i]
            ax.scatter(t_hr_plot, e, s=6, alpha=0.4,
                       label=f"dep{d} error ρ̇_{labels[i]}")
            ax.plot(t_hr_plot, s, "r--", linewidth=1.0,
                    label="+3σ" if i == 0 else None)
            ax.plot(t_hr_plot, -s, "r--", linewidth=1.0,
                    label="-3σ" if i == 0 else None)
            ax.set_ylabel(f"{labels[i]} [m/s]")
            ax.grid(True)
        axes2[-1].set_xlabel("Time since deploy [h]")
        axes2[0].legend()
        fig2.suptitle(f"Apophis rendezvous – Velocity error vs 3σ – Deputy {d}")
        fig2.tight_layout()

        # 3) Range residuals (scalar) pre- vs post-fit
        y_pre_d  = y_pre_swarm[d][mask]
        y_post_d = y_post_swarm[d][mask]
        sigma_r  = sigma_r_swarm[d][mask]
        three_sigma_r = 3.0 * sigma_r

        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        ax3.scatter(t_hr_plot, y_pre_d, s=4, alpha=0.4,
                    label="pre-fit range")
        ax3.scatter(t_hr_plot, y_post_d, s=4, alpha=0.4,
                    label="post-fit range")
        ax3.plot(t_hr_plot, three_sigma_r, "r--", linewidth=1.0,
                 label="+3σ (range)")
        ax3.plot(t_hr_plot, -three_sigma_r, "r--", linewidth=1.0,
                 label="-3σ (range)")
        ax3.set_ylabel("range residual [m]")
        ax3.set_xlabel("Time since deploy [h]")
        ax3.grid(True)
        ax3.legend()
        fig3.suptitle(f"Apophis rendezvous – Range residuals – Deputy {d}")
        fig3.tight_layout()

    plt.show()


# ───────── entry point ─────────

if __name__ == "__main__":
    # Assume we are sitting in Scenario/Apophis alongside the RHIG bundles
    dep0_path = SCRIPT_DIR / "apophis_rendezvous_full_D0.npz"
    if not dep0_path.is_file():
        raise FileNotFoundError(
            f"Expected Apophis bundle not found: {dep0_path}\n"
            f"Run `basilik_rhig_apophis_rendezvous_fleet.py` first."
        )
    apophis_swarm_ckf_ekf_demo(SCRIPT_DIR)
