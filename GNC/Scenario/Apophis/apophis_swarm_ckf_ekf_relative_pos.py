#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_ckf_ekf_demo.py
=============================

Multi-deputy CKF + EKF demo for the Apophis rendezvous case, now with
**relative position measurements** to Apophis in the chief Hill (LVLH) frame.

Measurement model
-----------------
For each deputy:

    z_k = Δr_H^{(D-A)}(t_k) + b_k + w_k        (3x1, in [R,T,N])

where
    Δr_H^{(D-A)} = C_HN (r_D - r_A),
    C_HN         = Hill-from-ECI DCM at the chief,
    b_k          = slowly varying 3D bias (AR(1) random walk),
    w_k          = zero-mean noise with time-varying σ, plus dropouts/outliers.

State
-----
Deputy state is Hill wrt chief:

    x = [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N]^T.

Measurement function in Hill coordinates is

    h(x, t) = ρ(t) + C_HN(t) (r_C(t) - r_A(t))

which is linear in x, with Jacobian

    H = [ I_3  0_3 ]  (3x6)

Inputs
------
Script expects to sit in the same directory as the RHIG fleet bundles:

    Scenario/Apophis/
        apophis_rendezvous_full_D0.npz
        ...
        apophis_rendezvous_full_D7.npz

Each NPZ should contain:

    t        : (N,)   global time [s since departure]
    rD_N     : (N,3)  deputy ECI position [m]
    vD_N     : (N,3)  deputy ECI velocity [m/s]
    rC_N     : (N,3)  chief ECI position [m]
    vC_N     : (N,3)  chief ECI velocity [m/s]
    rA_N     : (N,3)  Apophis ECI position [m]
    vA_N     : (N,3)  Apophis ECI velocity [m/s]
    impulses : (Nb,4) [t, dv_R, dv_T, dv_N] in Hill [m/s] (optional)
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

# Apply global matplotlib defaults
plt.rcParams.update({
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "xtick.labelsize": TICK_LABEL_FONTSIZE,
    "ytick.labelsize": TICK_LABEL_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
})

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

from GNC.Navigation.hill_ekf import (  # type: ignore[import]
    HillEKF,
    HillEKFConfig,
    dcm_lvlh,
    hill_rate,
)

# ───────── configuration ─────────

N_DEP      = 8      # deputies D0..D7 from the fleet script
MEAS_DECIM = 10     # decimate 1 Hz truth to this cadence for nav
N_CKF      = 2      # CKF passes before final EKF

# True relative position accuracy envelope (per axis, 1σ)
POS_HI_TRUE = 150.0   # [m] early in the arc
POS_LO_TRUE =  30.0   # [m] late in the arc
TAU_HOURS   = 2.5     # time constant for decay

# Filter's *assumed* σ (mis-model vs truth)
POS_SCALE_ASSUMED = 0.8  # EKF thinks measurements are ~20% better

# Bias / dropout / outlier parameters
BIAS_POS_STD   = 30.0     # [m] 1σ of slow 3D bias
BIAS_TAU_HOURS = 4.0      # [h] correlation time
P_DROPOUT      = 0.05     # probability of dropping measurement at an epoch
P_OUTLIER      = 0.01     # outlier probability (per axis)
OUTLIER_SCALE  = 5.0      # outlier magnitude multiplier


# ───────── helper: load a deputy's bundle ─────────

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
        f = (t - t_arr[j]) / (t_arr[j + 1] - t_arr[j] + 1e-15)
        return (1.0 - f) * arr[j] + f * arr[j + 1]

    def state_fn(t: float) -> Tuple[np.ndarray, np.ndarray]:
        return interp_vec(t, r), interp_vec(t, v)

    return state_fn


# ───────── burns mapped onto a time grid ─────────

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


# ───────── ECI → Hill truth conversion ─────────

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


# ───────── relative position measurement generator ─────────

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

    Geometry:

      ρ_cd      = deputy wrt chief in Hill
      rC, rA    = chief and Apophis in ECI
      C_HN      = Hill-from-ECI at chief

      Δr_H^{(D-A)} = C_HN (rD - rA)
                   = C_HN (rC + C_HN^T ρ_cd - rA)
                   = C_HN (rC - rA) + ρ_cd

    So

      h(x,t) = ρ_cd + offset(t),   offset(t) = C_HN (rC - rA)
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


# ───────── EKF update for relative position ─────────

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


# ───────── single-deputy CKF+EKF helper ─────────

def run_ckf_ekf_for_deputy(
    dep_idx: int,
    t_arr: np.ndarray,
    x_true: np.ndarray,
    rD_sel: np.ndarray,
    rC_sel: np.ndarray,
    vC_sel: np.ndarray,
    rA_sel: np.ndarray,
    dv_seq_H: np.ndarray,
    chief_state_fn,
    apophis_state_fn,
    N_ckf: int,
):
    """
    CKF + EKF for one deputy with relative position measurements.

    Returns
    -------
    x_est_hist : (N,6)
    P_hist     : (N,6,6)
    y_pre_hist : (N,3)  pre-fit innovations [ΔR,ΔT,ΔN]
    y_post_hist: (N,3)  post-fit residuals   [ΔR,ΔT,ΔN]
    sigma_assumed_final : (N,3)  EKF-assumed 1σ per axis
    """
    dt = float(np.median(np.diff(t_arr)))

    cfg = HillEKFConfig(
        dt=dt,
        sigma_proc_pos=0.5,     # [m]
        sigma_proc_vel=1e-3,    # [m/s]
        sigma_meas_pos=20.0,
        sigma_meas_vel=0.02,
        sigma_init_pos=500.0,   # [m]
        sigma_init_vel=0.1,     # [m/s]
        sigma_dv=0.02,          # [m/s]
        use_joseph=True,
    )
    ekf = HillEKF(cfg, chief_state_fn=chief_state_fn)

    # Initial state bias
    x0_bias = np.array([
        300.0, -250.0, 150.0,
        0.05, -0.04, 0.02,
    ])
    x0_guess = x_true[0] + x0_bias

    p_pos0 = cfg.sigma_init_pos**2
    p_vel0 = cfg.sigma_init_vel**2
    P0_big = np.diag([p_pos0, p_pos0, p_pos0, p_vel0, p_vel0, p_vel0])

    print(
        f"[dep{dep_idx}] INIT ||x0_guess - x_true(0)|| = "
        f"{la.norm(x0_guess - x_true[0]):.3f}, Tr(P0) = {np.trace(P0_big):.3e}",
    )

    # CKF passes: refine initial state only
    if N_ckf > 0:
        print(f"[dep{dep_idx}] CKF {N_ckf} passes...")

    for it in range(N_ckf):
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
                ekf.y_pre = np.zeros(3)
                ekf.y_post = np.zeros(3)

        err_end = la.norm(ekf.x - x_true[-1])
        print(
            f"    pass {it + 1}: ||x_end - x_true_end|| = {err_end:.3f}, "
            f"Tr(P_end) = {np.trace(ekf.P):.3e}",
        )
        x0_guess = ekf.x.copy()

    # Final EKF pass with fresh measurements
    print(f"[dep{dep_idx}] EKF final pass...")
    ekf.reset(x0_guess, P0_big)

    rng_final = np.random.default_rng(2000 + dep_idx)
    z_meas, sigma_final, valid_final = build_relpos_meas(
        rng_final, t_arr, rD_sel, rC_sel, vC_sel, rA_sel,
    )

    N = len(t_arr)
    x_est_hist = np.zeros((N, 6))
    P_hist = np.zeros((N, 6, 6))
    y_pre_hist = np.full((N, 3), np.nan)
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
    P_hist[0] = ekf.P
    y_pre_hist[0] = ekf.y_pre
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
            y_pre_hist[k] = ekf.y_pre
            y_post_hist[k] = ekf.y_post
        else:
            ekf.x = ekf.x_pred.copy()
            ekf.P = ekf.P_pred.copy()

        x_est_hist[k] = ekf.x
        P_hist[k] = ekf.P

    return x_est_hist, P_hist, y_pre_hist, y_post_hist, sigma_final


# ───────── main swarm demo ─────────

def apophis_swarm_ckf_ekf_demo(
    data_dir: Path,
    N_dep: int = N_DEP,
    meas_decim: int = MEAS_DECIM,
    N_ckf: int = N_CKF,
):
    data_dir = Path(data_dir)

    x_true_swarm = []
    x_est_swarm  = []
    P_swarm      = []
    y_pre_swarm  = []
    y_post_swarm = []
    sigma_swarm  = []
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

        # Hill truth for this deputy on the decimated grid
        x_true = eci_to_hill_traj(rD_sel, vD_sel, rC_sel, vC_sel)

        # chief & Apophis interpolators (use full histories)
        chief_state_fn   = make_state_fn_from_bundle(data, "rC_N", "vC_N")
        apophis_state_fn = make_state_fn_from_bundle(data, "rA_N", "vA_N")

        (x_est_d, P_d,
         y_pre_d, y_post_d,
         sigma_final) = run_ckf_ekf_for_deputy(
            dep_idx=d,
            t_arr=t_arr,
            x_true=x_true,
            rD_sel=rD_sel,
            rC_sel=rC_sel,
            vC_sel=vC_sel,
            rA_sel=rA_sel,
            dv_seq_H=dv_seq_H,
            chief_state_fn=chief_state_fn,
            apophis_state_fn=apophis_state_fn,
            N_ckf=N_ckf,
        )

        # --- save FDI-core bundle for this deputy ---
        fdi_path = data_dir / f"apophis_swarm_FDI_D{d}.npz"
        np.savez_compressed(
            fdi_path,
            t=t_arr,
            x_true=x_true,
            x_est=x_est_d,
            P=P_d,
            y_pre=y_pre_d,
            y_post=y_post_d,
            sigma_meas=sigma_final,
            dv_seq_H=dv_seq_H,
        )
        print(f"[dep{d}] Saved FDI bundle -> {fdi_path.name}")

        x_true_swarm.append(x_true)
        x_est_swarm.append(x_est_d)
        P_swarm.append(P_d)
        y_pre_swarm.append(y_pre_d)
        y_post_swarm.append(y_post_d)
        sigma_swarm.append(sigma_final)
        t_swarm.append(t_arr)

    # ─── diagnostics / plots ───

    labels = ["R", "T", "N"]

    for d in range(N_dep):
        print(f"\n=== Deputy {d} EKF results (relative position) ===")

        t_arr = t_swarm[d]
        t_rel = t_arr - t_arr[0]
        t_hr = t_rel / 3600.0

        x_true_d = x_true_swarm[d]
        x_est_d  = x_est_swarm[d]
        err_d    = x_est_d - x_true_d

        # ignore very early transient for RMS stats / plots
        mask = t_hr >= 0.1
        t_hr_plot = t_hr[mask]

        err_plot = err_d[mask]
        P_diag_plot = np.diagonal(P_swarm[d][mask], axis1=1, axis2=2)
        three_sigma = 3.0 * np.sqrt(P_diag_plot)

        # stats
        for i, lbl in enumerate(labels):
            e = err_plot[:, i]
            print(
                f"[POS] {lbl}: mean={np.mean(e):.3f} m, "
                f"RMS={np.sqrt(np.mean(e**2)):.3f} m",
            )

        # 1) Position error vs 3σ
        fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            ax = axes1[i]
            e = err_plot[:, i]
            s = three_sigma[:, i]

            ax.scatter(
                t_hr_plot, e, s=6, alpha=0.4,
                label=f"dep{d} error ρ_{labels[i]}",
            )
            ax.plot(
                t_hr_plot, s, "r--", linewidth=1.0,
                label="+3σ" if i == 0 else None,
            )
            ax.plot(
                t_hr_plot, -s, "r--", linewidth=1.0,
                label="-3σ" if i == 0 else None,
            )
            ax.set_ylabel(f"{labels[i]} [m]", fontsize=AXIS_LABEL_FONTSIZE)
            ax.grid(True)
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

        axes1[-1].set_xlabel(
            "Time since deploy [h]", fontsize=AXIS_LABEL_FONTSIZE,
        )
        axes1[0].legend(fontsize=LEGEND_FONTSIZE)
        fig1.suptitle(
            f"Position Error vs 3σ – Deputy {d}",
            fontsize=TITLE_FONTSIZE,
        )
        fig1.tight_layout()

        # 2) Velocity error vs 3σ
        fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            ax = axes2[i]
            e = err_plot[:, 3 + i]
            s = three_sigma[:, 3 + i]

            ax.scatter(
                t_hr_plot, e, s=6, alpha=0.4,
                label=f"dep{d} error ρ̇_{labels[i]}",
            )
            ax.plot(
                t_hr_plot, s, "r--", linewidth=1.0,
                label="+3σ" if i == 0 else None,
            )
            ax.plot(
                t_hr_plot, -s, "r--", linewidth=1.0,
                label="-3σ" if i == 0 else None,
            )
            ax.set_ylabel(f"{labels[i]} [m/s]", fontsize=AXIS_LABEL_FONTSIZE)
            ax.grid(True)
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

        axes2[-1].set_xlabel(
            "Time since deploy [h]", fontsize=AXIS_LABEL_FONTSIZE,
        )
        axes2[0].legend(fontsize=LEGEND_FONTSIZE)
        fig2.suptitle(
            f"Velocity Error vs 3σ – Deputy {d}",
            fontsize=TITLE_FONTSIZE,
        )
        fig2.tight_layout()

        # 3) Relative position residuals (measurement space) vs 3σ
        y_pre_d  = y_pre_swarm[d][mask]
        y_post_d = y_post_swarm[d][mask]
        sigma_d  = sigma_swarm[d][mask]
        three_sigma_meas = 3.0 * sigma_d

        fig3, axes3 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            ax = axes3[i]
            ax.scatter(
                t_hr_plot, y_pre_d[:, i], s=4, alpha=0.4,
                label=f"pre-fit Δ{labels[i]}",
            )
            ax.scatter(
                t_hr_plot, y_post_d[:, i], s=4, alpha=0.4,
                label=f"post-fit Δ{labels[i]}",
            )
            ax.plot(
                t_hr_plot, three_sigma_meas[:, i], "r--",
                linewidth=1.0, label="+3σ" if i == 0 else None,
            )
            ax.plot(
                t_hr_plot, -three_sigma_meas[:, i], "r--",
                linewidth=1.0, label="-3σ" if i == 0 else None,
            )
            ax.set_ylabel(f"Δ{labels[i]} [m]", fontsize=AXIS_LABEL_FONTSIZE)
            ax.grid(True)
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

        axes3[-1].set_xlabel(
            "Time since deploy [h]", fontsize=AXIS_LABEL_FONTSIZE,
        )
        axes3[0].legend(fontsize=LEGEND_FONTSIZE)
        fig3.suptitle(
            f"Rel-pos Residuals – Deputy {d}",
            fontsize=TITLE_FONTSIZE,
        )
        fig3.tight_layout()

    plt.show()


# ───────── entry point ─────────

if __name__ == "__main__":
    dep0_path = SCRIPT_DIR / "apophis_rendezvous_full_D0.npz"
    if not dep0_path.is_file():
        raise FileNotFoundError(
            f"Expected Apophis bundle not found: {dep0_path}\n"
            f"Run `basilik_rhig_apophis_rendezvous_fleet.py` first.",
        )
    apophis_swarm_ckf_ekf_demo(SCRIPT_DIR)
