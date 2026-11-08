#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_relnav_sigma_sweep.py
=============================

Sigma sweep for Apophis relative-navigation CKF/EKF.

Purpose
-------
Vary a single process-noise tuning parameter σ_proc over a log-spaced range.
For each σ_proc:

  * run the Hill CKF+EKF on one deputy (D0 by default),
  * compute:
      - RMS of 3D relative-position post-fit residuals (measurement space),
      - 3D RMS position error,
      - 3D RMS velocity error,
  * then plot:

      Top panel:
         left  y-axis: RMS measurement residual (3D) vs σ_proc
         right y-axis: 3D RMS position error vs σ_proc

      Bottom panel:
         left  y-axis: RMS measurement residual (3D) vs σ_proc
         right y-axis: 3D RMS velocity error vs σ_proc

and find an "optimal" σ_proc at the approximate intersection of
log10(residual RMS) and log10(3D position RMS) (normalized).

Assumptions
-----------
- Uses the same Apophis NPZ bundles as apophis_swarm_ckf_ekf_demo.py
- Uses the same relative-position measurement model:
      z_k = Δr_H^{(D-A)} + bias + noise
- Only runs on a single "representative" deputy (D0 by default) for the sweep.

Tuning knob
-----------
We map each scalar σ_proc (units arbitrary here; think "process noise level")
into HillEKFConfig via:

    sigma_proc_pos = σ_proc
    sigma_proc_vel = σ_proc

You can refine that mapping to match your continuous-time SNC model if needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# ───────── plotting font controls ─────────

TITLE_FONTSIZE      = 30
AXIS_LABEL_FONTSIZE = 25
TICK_LABEL_FONTSIZE = 22
LEGEND_FONTSIZE     = 22

plt.rcParams.update({
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "xtick.labelsize": TICK_LABEL_FONTSIZE,
    "ytick.labelsize": TICK_LABEL_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
})

# ───────── locate ARISE root / imports ─────────

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

from GNC.Navigation.hill_ekf import (  # type: ignore[import]
    HillEKF,
    HillEKFConfig,
    dcm_lvlh,
    hill_rate,
)

# ───────── configuration ─────────

DEP_IDX        = 0     # which deputy to use for sweep (D0 by default)
MEAS_DECIM     = 10    # decimate 1 Hz truth to this cadence for nav
N_CKF_PASSES   = 2     # number of CKF passes before final EKF

# Sigma sweep range (process noise tuning)
# Adjust to taste: this is the *filter*'s σ_proc, not a physical m/s^2 directly.
SIGMA_PROC_VEC = np.logspace(-6, 0, 60)   # e.g. 1e-4 ... 1, 25 points

# Measurement model parameters (same flavor as your CKF/EKF demo)
POS_HI_TRUE     = 150.0    # [m] early in arc, per-axis 1σ for rel-pos
POS_LO_TRUE     =  30.0    # [m] late in arc
TAU_HOURS       = 2.5      # exponential decay time constant

POS_SCALE_ASSUMED = 0.8    # EKF thinks meas are ~20% better

BIAS_POS_STD     = 30.0    # [m] 1σ for slow 3D bias
BIAS_TAU_HOURS   = 4.0     # [h] AR(1) bias correlation time
P_DROPOUT        = 0.05    # dropout probability per epoch
P_OUTLIER        = 0.01    # outlier probability per axis
OUTLIER_SCALE    = 5.0     # outlier magnitude scale

# Time after which to compute RMS metrics (skip early transient) [hours]
METRIC_START_HOURS = 0.1


# ───────── helpers: load bundle, interpolate, burns ─────────

def load_apophis_bundle(path: Path) -> Dict[str, np.ndarray]:
    wanted = (
        "t", "rD_N", "vD_N", "rC_N", "vC_N",
        "rA_N", "vA_N", "impulses",
    )
    out: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as npz:
        for k in wanted:
            if k in npz:
                out[k] = npz[k]
    if "impulses" not in out:
        out["impulses"] = np.zeros((0, 4), dtype=float)
    return out


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
        f = (t - t_arr[j]) / (t_arr[j + 1] - t_arr[j] + 1e-15)
        return (1.0 - f) * arr[j] + f * arr[j + 1]

    def state_fn(t: float) -> Tuple[np.ndarray, np.ndarray]:
        return interp_vec(t, r), interp_vec(t, v)

    return state_fn


def build_burn_sequence(
    impulses: np.ndarray,
    t_arr: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Map discrete impulses [t, dv_R, dv_T, dv_N] to a time grid t_arr.
    """
    impulses = np.asarray(impulses, float).reshape(-1, 4)
    dv_seq = np.zeros((len(t_arr), 3))
    if impulses.size == 0:
        return dv_seq

    impulses = impulses[np.argsort(impulses[:, 0])]
    idx_imp = 0
    for k, t in enumerate(t_arr):
        while idx_imp < impulses.shape[0] and impulses[idx_imp, 0] < t - 0.5 * dt:
            idx_imp += 1
        if idx_imp < impulses.shape[0] and abs(impulses[idx_imp, 0] - t) <= 0.5 * dt:
            dv_seq[k, :] = impulses[idx_imp, 1:4]
            idx_imp += 1
    return dv_seq


# ───────── ECI → Hill truth ─────────

def eci_to_hill_traj(
    rD_N: np.ndarray,
    vD_N: np.ndarray,
    rC_N: np.ndarray,
    vC_N: np.ndarray,
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

        C_HN = dcm_lvlh(rC, vC)  # Hill-from-ECI
        r_rel_N = rD - rC
        v_rel_N = vD - vC

        rho = C_HN @ r_rel_N
        omega_HN = np.array([0.0, 0.0, hill_rate(rC, vC)])
        v_rot = C_HN @ v_rel_N
        rhod = v_rot - np.cross(omega_HN, rho)

        x_true[k, 0:3] = rho
        x_true[k, 3:6] = rhod

    return x_true


# ───────── measurement generation: rel-pos in chief Hill ─────────

def build_relpos_meas(
    rng: np.random.Generator,
    t_arr: np.ndarray,
    rD_N: np.ndarray,
    rC_N: np.ndarray,
    vC_N: np.ndarray,
    rA_N: np.ndarray,
    pos_hi_true: float = POS_HI_TRUE,
    pos_lo_true: float = POS_LO_TRUE,
    tau_hours: float = TAU_HOURS,
    bias_std: float = BIAS_POS_STD,
    bias_tau_hours: float = BIAS_TAU_HOURS,
    p_dropout: float = P_DROPOUT,
    p_outlier: float = P_OUTLIER,
    outlier_scale: float = OUTLIER_SCALE,
):
    """
    Build noisy relative position measurements in chief Hill frame:

        z_k = Δr_H^{(D-A)}(t_k) + b_k + w_k     (3x1)

    where Δr_H^{(D-A)} = C_HN (rD - rA).
    """
    t_arr = np.asarray(t_arr, float).reshape(-1)
    rD_N = np.asarray(rD_N, float).reshape(-1, 3)
    rC_N = np.asarray(rC_N, float).reshape(-1, 3)
    vC_N = np.asarray(vC_N, float).reshape(-1, 3)
    rA_N = np.asarray(rA_N, float).reshape(-1, 3)

    N = len(t_arr)
    if N < 2:
        dt_eff = 1.0
    else:
        dt_eff = float(np.median(np.diff(t_arr)))

    # time-varying true σ per axis
    t_rel = t_arr - t_arr[0]
    tau = tau_hours * 3600.0
    w = np.exp(-t_rel / tau)
    sigma_true_scalar = pos_lo_true + (pos_hi_true - pos_lo_true) * w
    sigma_true = np.repeat(sigma_true_scalar[:, None], 3, axis=1)

    # AR(1) bias
    bias_tau_sec = bias_tau_hours * 3600.0
    rho_b = np.exp(-dt_eff / max(bias_tau_sec, 1e-3))
    q_b = bias_std * np.sqrt(1.0 - rho_b**2)

    b = np.zeros(3)
    z_meas = np.zeros((N, 3))
    valid = np.ones(N, dtype=bool)

    for k in range(N):
        if k > 0:
            b = rho_b * b + q_b * rng.normal(size=3)

        rC = rC_N[k]
        vC = vC_N[k]
        rD = rD_N[k]
        rA = rA_N[k]

        C_HN = dcm_lvlh(rC, vC)          # Hill-from-ECI
        r_rel_H_true = C_HN @ (rD - rA)  # deputy − Apophis in Hill

        noise = rng.normal(size=3) * sigma_true[k]

        # occasional per-axis outliers
        for j in range(3):
            if rng.random() < p_outlier:
                noise[j] *= outlier_scale

        z_meas[k] = r_rel_H_true + b + noise

        if k > 0 and rng.random() < p_dropout:
            valid[k] = False

    sigma_assumed = POS_SCALE_ASSUMED * sigma_true
    return z_meas, sigma_assumed, valid


# ───────── measurement model: prediction & Jacobian ─────────

def relpos_pred_and_jacobian(
    x_H: np.ndarray,
    t: float,
    chief_state_fn,
    apophis_state_fn,
):
    """
    Predict relative position to Apophis in chief Hill frame and 3x6 Jacobian.

    x_H = [ρ_cd, ρ̇_cd] (6,)

      h(x,t) = ρ_cd + C_HN(t) (rC(t) - rA(t))
      H      = [ I_3  0_3 ].
    """
    x_H = np.asarray(x_H, float).reshape(6)
    rho_cd = x_H[0:3]

    rC, vC = chief_state_fn(t)
    rA, _  = apophis_state_fn(t)

    C_HN = dcm_lvlh(rC, vC)
    offset = C_HN @ (rC - rA)

    z_hat = rho_cd + offset

    H = np.zeros((3, 6))
    H[:, 0:3] = np.eye(3)

    return z_hat, H


def ekf_update_relpos(
    ekf: HillEKF,
    t: float,
    z_vec: np.ndarray,
    sigma_vec: np.ndarray,
    chief_state_fn,
    apophis_state_fn,
):
    """
    EKF update with 3D relative position measurement in Hill frame.
    """
    z_hat, H = relpos_pred_and_jacobian(
        ekf.x_pred, t, chief_state_fn, apophis_state_fn,
    )

    z = np.asarray(z_vec, float).reshape(3)
    sigma_vec = np.asarray(sigma_vec, float).reshape(3)

    y = z - z_hat                       # innovation (3,)
    R = np.diag(sigma_vec**2)
    H = np.asarray(H, float).reshape(3, ekf.x.size)
    I = np.eye(ekf.x.size)

    S = H @ ekf.P_pred @ H.T + R
    K = ekf.P_pred @ H.T @ la.inv(S)

    x_new = ekf.x_pred + (K @ y).reshape(-1)

    if ekf.cfg.use_joseph:
        KH = K @ H
        ekf.P = (I - KH) @ ekf.P_pred @ (I - KH).T + K @ R @ K.T
    else:
        ekf.P = (I - K @ H) @ ekf.P_pred

    ekf.x = x_new

    ekf.innov = y.copy()
    ekf.S = S.copy()
    ekf.y_pre = y.copy()

    # post-fit residual
    z_hat_post, _ = relpos_pred_and_jacobian(
        ekf.x, t, chief_state_fn, apophis_state_fn,
    )
    ekf.y_post = z - z_hat_post
    ekf.H_last = H.copy()


# ───────── filter run for a single σ_proc ─────────

def run_rel_nav_for_sigma(
    sigma_proc: float,
    data: Dict[str, np.ndarray],
    dep_idx: int = DEP_IDX,
    meas_decim: int = MEAS_DECIM,
    N_ckf_passes: int = N_CKF_PASSES,
):
    """
    Run CKF+EKF for a single deputy and a single process-noise tuning sigma_proc.

    Returns:
        t_arr            : (N,) times [s] on decimated grid
        x_true           : (N,6) Hill truth
        x_est            : (N,6) Hill EKF estimate
        y_post           : (N,3) post-fit residuals in rel-pos measurement space
    """
    t_full  = np.asarray(data["t"], float).reshape(-1)
    rD_full = np.asarray(data["rD_N"], float).reshape(-1, 3)
    vD_full = np.asarray(data["vD_N"], float).reshape(-1, 3)
    rC_full = np.asarray(data["rC_N"], float).reshape(-1, 3)
    vC_full = np.asarray(data["vC_N"], float).reshape(-1, 3)
    rA_full = np.asarray(data["rA_N"], float).reshape(-1, 3)
    vA_full = np.asarray(data["vA_N"], float).reshape(-1, 3)
    impulses = np.asarray(data["impulses"], float).reshape(-1, 4)

    # crop to skip the first ~0.5 h after deploy
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

    # Hill truth on decimated grid
    x_true = eci_to_hill_traj(rD_sel, vD_sel, rC_sel, vC_sel)

    # full-history interpolators for chief & Apophis
    chief_state_fn   = make_state_fn_from_bundle(data, "rC_N", "vC_N")
    apophis_state_fn = make_state_fn_from_bundle(data, "rA_N", "vA_N")

    # EKF setup for this sigma_proc
    dt = float(np.median(np.diff(t_arr)))
    cfg = HillEKFConfig(
        dt=dt,
        sigma_proc_pos=sigma_proc,
        sigma_proc_vel=sigma_proc,
        sigma_meas_pos=20.0,
        sigma_meas_vel=0.02,
        sigma_init_pos=500.0,
        sigma_init_vel=0.1,
        sigma_dv=0.02,
        use_joseph=True,
    )
    ekf = HillEKF(cfg, chief_state_fn=chief_state_fn)

    # initial guess
    x0_bias = np.array([300.0, -250.0, 150.0, 0.05, -0.04, 0.02])
    x0_guess = x_true[0] + x0_bias

    p_pos0 = cfg.sigma_init_pos**2
    p_vel0 = cfg.sigma_init_vel**2
    P0_big = np.diag([p_pos0, p_pos0, p_pos0, p_vel0, p_vel0, p_vel0])

    # CKF-like passes: refine initial state a bit
    for it in range(N_ckf_passes):
        ekf.reset(x0_guess, P0_big)

        rng_pass = np.random.default_rng(1000 + dep_idx * 10 + it)
        z_pass, sigma_pass, valid_pass = build_relpos_meas(
            rng_pass, t_arr, rD_sel, rC_sel, vC_sel, rA_sel,
        )

        ekf.x_pred = ekf.x.copy()
        ekf.P_pred = ekf.P.copy()
        ekf.t_last = t_arr[0]

        ekf_update_relpos(
            ekf,
            t_arr[0],
            z_pass[0],
            sigma_pass[0],
            chief_state_fn,
            apophis_state_fn,
        )

        for k in range(1, len(t_arr)):
            ekf.predict(t_arr[k], burn_dv_H=dv_seq_H[k])

            if valid_pass[k]:
                ekf_update_relpos(
                    ekf,
                    t_arr[k],
                    z_pass[k],
                    sigma_pass[k],
                    chief_state_fn,
                    apophis_state_fn,
                )
            else:
                ekf.x = ekf.x_pred.copy()
                ekf.P = ekf.P_pred.copy()

        x0_guess = ekf.x.copy()

    # Final EKF pass for metrics
    ekf.reset(x0_guess, P0_big)

    rng_final = np.random.default_rng(2000 + dep_idx)
    z_meas, sigma_final, valid_final = build_relpos_meas(
        rng_final, t_arr, rD_sel, rC_sel, vC_sel, rA_sel,
    )

    N = len(t_arr)
    x_est_hist = np.zeros((N, 6))
    y_post_hist = np.full((N, 3), np.nan)

    ekf.x_pred = ekf.x.copy()
    ekf.P_pred = ekf.P.copy()
    ekf.t_last = t_arr[0]

    ekf_update_relpos(
        ekf,
        t_arr[0],
        z_meas[0],
        sigma_final[0],
        chief_state_fn,
        apophis_state_fn,
    )
    x_est_hist[0] = ekf.x
    y_post_hist[0] = ekf.y_post

    for k in range(1, N):
        ekf.predict(t_arr[k], burn_dv_H=dv_seq_H[k])

        if valid_final[k]:
            ekf_update_relpos(
                ekf,
                t_arr[k],
                z_meas[k],
                sigma_final[k],
                chief_state_fn,
                apophis_state_fn,
            )
        else:
            ekf.x = ekf.x_pred.copy()
            ekf.P = ekf.P_pred.copy()
            ekf.y_post = np.zeros(3)

        x_est_hist[k] = ekf.x
        y_post_hist[k] = ekf.y_post

    return t_arr, x_true, x_est_hist, y_post_hist


# ───────── metrics vs sigma_proc ─────────

def sigma_sweep_rel_nav():
    """
    Run the sigma sweep, compute metrics, find intersection-based sigma_opt,
    and produce plotyy-style plots.
    """
    dep_path = SCRIPT_DIR / f"apophis_rendezvous_full_D{DEP_IDX}.npz"
    if not dep_path.is_file():
        raise FileNotFoundError(
            f"Expected Apophis bundle not found: {dep_path}\n"
            f"Run `basilik_rhig_apophis_rendezvous_fleet.py` first."
        )

    print(f"Using deputy D{DEP_IDX} from {dep_path.name} for sigma sweep.")
    data = load_apophis_bundle(dep_path)

    RMS_meas = np.zeros_like(SIGMA_PROC_VEC)
    RMS_pos3D = np.zeros_like(SIGMA_PROC_VEC)
    RMS_vel3D = np.zeros_like(SIGMA_PROC_VEC)

    for k, sigma_proc in enumerate(SIGMA_PROC_VEC):
        print(f"\n--- σ_proc[{k+1}/{len(SIGMA_PROC_VEC)}] = {sigma_proc:g} ---")

        t_arr, x_true, x_est, y_post = run_rel_nav_for_sigma(
            sigma_proc,
            data,
            dep_idx=DEP_IDX,
        )

        t_rel = t_arr - t_arr[0]
        t_hr = t_rel / 3600.0

        mask = t_hr >= METRIC_START_HOURS
        if not np.any(mask):
            mask[:] = True  # just in case

        # 3D position & velocity error
        err = x_est - x_true
        r_err = err[mask, 0:3]
        v_err = err[mask, 3:6]

        RMS_pos3D[k] = np.sqrt(np.mean(np.sum(r_err**2, axis=1)))
        RMS_vel3D[k] = np.sqrt(np.mean(np.sum(v_err**2, axis=1)))

        # 3D measurement residual RMS (post-fit)
        y = y_post[mask, :]
        RMS_meas[k] = np.sqrt(np.mean(np.sum(y**2, axis=1)))

        print(f"  RMS(meas) = {RMS_meas[k]:.3f} m, "
              f"RMS(3D pos) = {RMS_pos3D[k]:.3f} m, "
              f"RMS(3D vel) = {RMS_vel3D[k]:.3f} m/s")

    # Intersection-based sigma_opt using normalized logs
    logRho = np.log10(RMS_meas)
    logPos = np.log10(RMS_pos3D)

    rho_norm = (logRho - logRho.min()) / (logRho.max() - logRho.min())
    pos_norm = (logPos - logPos.min()) / (logPos.max() - logPos.min())

    idx_opt = np.argmin(np.abs(rho_norm - pos_norm))
    sigma_opt = SIGMA_PROC_VEC[idx_opt]

    print(f"\nApprox. intersection-based sigma_proc ≈ {sigma_opt:g}")

    # ───── plotyy-style plots ─────
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # ---------- Top subplot: meas residual vs 3D pos RMS ----------
    ax1_left = ax_top
    ax1_right = ax1_left.twinx()

    ax1_left.semilogx(
        SIGMA_PROC_VEC, RMS_meas,
        "b-o", linewidth=2, markersize=5,
        label="RMS rel-pos residual (3D)",
    )
    ax1_right.semilogx(
        SIGMA_PROC_VEC, RMS_pos3D,
        "g-s", linewidth=2, markersize=5,
        label="3D RMS position error",
    )

    ax1_left.set_ylabel("RMS rel-pos residual (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax1_right.set_ylabel("3D RMS position error (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax1_left.grid(True, which="both", linestyle="--", alpha=0.6)
    ax1_left.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax1_right.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    ax1_left.set_title(
        "Relative Navigation Performance vs σ_proc (Deputy D{})".format(DEP_IDX),
        fontsize=TITLE_FONTSIZE,
    )

    # Mark sigma_opt
    ax1_right.plot(
        sigma_opt, RMS_pos3D[idx_opt],
        "ro", markersize=9, linewidth=2,
        label="σ_opt (pos)",
    )

    # Build a joint legend
    lines_left, labels_left = ax1_left.get_legend_handles_labels()
    lines_right, labels_right = ax1_right.get_legend_handles_labels()
    ax1_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="best",
        fontsize=LEGEND_FONTSIZE,
    )

    # ---------- Bottom subplot: meas residual vs 3D vel RMS ----------
    ax2_left = ax_bot
    ax2_right = ax2_left.twinx()

    ax2_left.semilogx(
        SIGMA_PROC_VEC, RMS_meas,
        "b-o", linewidth=2, markersize=5,
        label="RMS rel-pos residual (3D)",
    )
    ax2_right.semilogx(
        SIGMA_PROC_VEC, RMS_vel3D,
        "g-s", linewidth=2, markersize=5,
        label="3D RMS velocity error",
    )

    ax2_left.set_xlabel(r"$\sigma_{\mathrm{proc}}$ (filter tuning)", fontsize=AXIS_LABEL_FONTSIZE)
    ax2_left.set_ylabel("RMS rel-pos residual (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax2_right.set_ylabel("3D RMS velocity error (m/s)", fontsize=AXIS_LABEL_FONTSIZE)

    ax2_left.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2_left.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax2_right.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    # Mark sigma_opt on bottom as well (on velocity curve)
    ax2_right.plot(
        sigma_opt, RMS_vel3D[idx_opt],
        "ro", markersize=9, linewidth=2,
        label="σ_opt (from top)",
    )

    lines_left2, labels_left2 = ax2_left.get_legend_handles_labels()
    lines_right2, labels_right2 = ax2_right.get_legend_handles_labels()
    ax2_left.legend(
        lines_left2 + lines_right2,
        labels_left2 + labels_right2,
        loc="best",
        fontsize=LEGEND_FONTSIZE,
    )

    fig.tight_layout()
    plt.show()


# ───────── entry point ─────────

if __name__ == "__main__":
    sigma_sweep_rel_nav()

