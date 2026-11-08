#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_ckf_ekf_demo.py
=============================

Multi-deputy CKF + EKF demo for the Apophis flyby case, aligned with
geo_swarm_ckf_ekf_demo.py.

State (per deputy, chief LVLH):
    x_d = [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N]^T  [m, m/s]

Chief:
    ESPA / carrier spacecraft from apophis_rendezvous_export.npz
Deputies:
    Synthetic Hill initial conditions around the chief (8 deputies).
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
    dcm_lvlh, hill_rate, cw_state_transition
)

# Chief ephemeris bundle (Apophis flyby)
APOPHIS_EXPORT = (SCRIPT_DIR / "apophis_rendezvous_export.npz").resolve()
# Use dep0 bundle only to steal its burn log (if present)
APOPHIS_DEP0   = (SCRIPT_DIR / "apophis_rendezvous_full_D0.npz").resolve()

N_DEP      = 8       # number of deputies in the swarm
MEAS_DECIM = 10      # measurement decimation factor
N_CKF      = 2       # CKF passes per deputy before final EKF


# ───────── helpers: load only the fields we need ─────────

def load_apophis_chief_bundle(path: Path) -> Dict[str, np.ndarray]:
    """
    Load chief ephemeris from apophis_rendezvous_export.npz.

    Expected fields:
        t      : time [s]
        rC_N   : chief ECI position [m]
        vC_N   : chief ECI velocity [m/s]
    """
    wanted = ("t", "rC_N", "vC_N")
    out: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as npz:
        for k in wanted:
            if k in npz:
                out[k] = npz[k]
    return out


def load_apophis_burns_from_dep0(path: Path) -> np.ndarray:
    """
    Load burn impulses from apophis_rendezvous_full_D0.npz, if present.

    Expected format if present:
        impulses : (N_imp, 4) [t, dv_R, dv_T, dv_N] in Hill frame.

    Returns
    -------
    impulses : (N_imp, 4) or empty (0,4) if not found.
    """
    if not path.is_file():
        return np.zeros((0, 4))
    with np.load(path, allow_pickle=True) as npz:
        if "impulses" in npz:
            return np.asarray(npz["impulses"], float).reshape(-1, 4)
    return np.zeros((0, 4))


# ───────── chief, burns, CW truth, measurements ─────────

def make_chief_state_fn_from_npz(data: dict):
    """
    Build chief state interpolator from Apophis export NPZ:

        t -> (rC_N(t), vC_N(t))
    """
    t_arr = np.asarray(data["t"], float).reshape(-1)
    rC = np.asarray(data["rC_N"], float).reshape(-1, 3)
    vC = np.asarray(data["vC_N"], float).reshape(-1, 3)

    def interp_vec(t: float, arr: np.ndarray) -> np.ndarray:
        if t <= t_arr[0]:
            return arr[0]
        if t >= t_arr[-1]:
            return arr[-1]
        j = np.searchsorted(t_arr, t) - 1
        f = (t - t_arr[j]) / (t_arr[j+1] - t_arr[j] + 1e-15)
        return (1.0 - f) * arr[j] + f * arr[j+1]

    def chief_state_fn(t: float) -> Tuple[np.ndarray, np.ndarray]:
        return interp_vec(t, rC), interp_vec(t, vC)

    return chief_state_fn


def build_burn_sequence(impulses: np.ndarray,
                        t_arr: np.ndarray,
                        dt: float) -> np.ndarray:
    """
    Map discrete impulses [t, dv_R, dv_T, dv_N] into a per-step Δv_H sequence
    aligned with t_arr.
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


def generate_hill_truth_cw(
    t_arr: np.ndarray,
    chief_state_fn,
    x0_true: np.ndarray,
    dv_seq_H: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Propagate a deputy Hill state with CW dynamics around the time-varying chief.
    """
    N = len(t_arr)
    x_true = np.zeros((N, 6))
    x_true[0] = np.asarray(x0_true, float).reshape(6)

    for k in range(N - 1):
        t_mid = 0.5 * (t_arr[k] + t_arr[k+1])
        r_c, v_c = chief_state_fn(t_mid)
        n = hill_rate(r_c, v_c)
        Phi = cw_state_transition(n, dt)

        x_next = Phi @ x_true[k]
        dv = dv_seq_H[k+1]
        if np.any(dv != 0.0):
            x_next[3:6] += dv

        x_true[k+1] = x_next

    return x_true


def build_time_varying_meas(
    rng: np.random.Generator,
    t_arr: np.ndarray,
    x_true: np.ndarray,
    pos_hi: float = 80.0,
    pos_lo: float = 20.0,
    vel_hi: float = 0.08,
    vel_lo: float = 0.02,
    tau_hours: float = 1.5,
):
    """
    Construct time-varying measurement noise envelopes and noisy z_k:

        z_k = x_true_k + v_k

    where σ decays from (pos_hi,vel_hi) to (pos_lo,vel_lo) with time constant τ.
    """
    N = len(t_arr)
    t_rel = t_arr - t_arr[0]
    tau = tau_hours * 3600.0
    w = np.exp(-t_rel / tau)

    sigma_pos = pos_lo + (pos_hi - pos_lo) * w
    sigma_vel = vel_lo + (vel_hi - vel_lo) * w

    z_meas = np.empty_like(x_true)
    z_meas[:, :3] = x_true[:, :3] + rng.normal(0.0, 1.0, size=(N, 3)) * sigma_pos[:, None]
    z_meas[:, 3:] = x_true[:, 3:] + rng.normal(0.0, 1.0, size=(N, 3)) * sigma_vel[:, None]

    return z_meas, sigma_pos, sigma_vel


def update_from_hill_measurement(ekf: HillEKF, z: np.ndarray):
    """
    Single EKF measurement update with full 6D Hill state measurement.
    Mirrors geo_ckf_ekf_demo.
    """
    z = np.asarray(z, float).reshape(6)
    H = np.eye(6)
    I6 = np.eye(6)

    y = z - H @ ekf.x_pred
    S = H @ ekf.P_pred @ H.T + ekf.R
    K = ekf.P_pred @ H.T @ la.inv(S)

    x_new = ekf.x_pred + K @ y

    if ekf.cfg.use_joseph:
        KH = K @ H
        ekf.P = (I6 - KH) @ ekf.P_pred @ (I6 - KH).T + K @ ekf.R @ K.T
    else:
        ekf.P = (I6 - K @ H) @ ekf.P_pred

    ekf.x = x_new
    ekf.innov = y
    ekf.S = S
    ekf.y_pre = y
    ekf.y_post = z - H @ ekf.x
    ekf.H_last = H.copy()

    return ekf.x.copy(), ekf.P.copy()


# ───────── single-deputy CKF+EKF helper ─────────

def run_ckf_ekf_for_deputy(
    dep_idx: int,
    t_arr: np.ndarray,
    x_true: np.ndarray,
    dv_seq_H: np.ndarray,
    sigma_pos_arr: np.ndarray,
    sigma_vel_arr: np.ndarray,
    chief_state_fn,
    N_ckf: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Clone of the GEO helper, but for Apophis chief.

    Returns
    -------
    x_est_hist : (N,6)
    P_hist     : (N,6,6)
    y_pre_hist, y_post_hist : (N,6)
    """
    dt = float(np.median(np.diff(t_arr)))

    cfg = HillEKFConfig(
        dt=dt,
        sigma_proc_pos=1.0,
        sigma_proc_vel=2e-3,
        sigma_meas_pos=20.0,
        sigma_meas_vel=0.02,
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

    # CKF passes: refine initial state only
    if N_ckf > 0:
        print(f"[dep{dep_idx}] CKF {N_ckf} passes...")
    for it in range(N_ckf):
        ekf.reset(x0_guess, P0_big)

        rng_pass = np.random.default_rng(1000 + dep_idx * 10 + it)
        z_pass, _, _ = build_time_varying_meas(
            rng_pass, t_arr, x_true,
            pos_hi=sigma_pos_arr[0], pos_lo=sigma_pos_arr[-1],
            vel_hi=sigma_vel_arr[0], vel_lo=sigma_vel_arr[-1],
            tau_hours=1.5,
        )

        ekf.x_pred = ekf.x.copy()
        ekf.P_pred = ekf.P.copy()
        ekf.t_last = t_arr[0]
        sig_p0 = sigma_pos_arr[0]
        sig_v0 = sigma_vel_arr[0]
        ekf.R = np.diag([sig_p0**2, sig_p0**2, sig_p0**2,
                         sig_v0**2, sig_v0**2, sig_v0**2])
        update_from_hill_measurement(ekf, z_pass[0])

        for k in range(1, len(t_arr)):
            sig_p = sigma_pos_arr[k]
            sig_v = sigma_vel_arr[k]
            ekf.R = np.diag([sig_p**2, sig_p**2, sig_p**2,
                             sig_v**2, sig_v**2, sig_v**2])
            ekf.predict(t_arr[k], burn_dv_H=dv_seq_H[k])
            update_from_hill_measurement(ekf, z_pass[k])

        err_end = la.norm(ekf.x - x_true[-1])
        print(f"    pass {it+1}: ||x_end - x_true_end|| = {err_end:.3f}, "
              f"Tr(P_end) = {np.trace(ekf.P):.3e}")
        x0_guess = ekf.x.copy()

    # Final EKF pass with fresh measurements
    print(f"[dep{dep_idx}] EKF final pass...")
    ekf.reset(x0_guess, P0_big)

    rng_final = np.random.default_rng(2000 + dep_idx)
    z_meas, _, _ = build_time_varying_meas(
        rng_final, t_arr, x_true,
        pos_hi=sigma_pos_arr[0], pos_lo=sigma_pos_arr[-1],
        vel_hi=sigma_vel_arr[0], vel_lo=sigma_vel_arr[-1],
        tau_hours=1.5,
    )

    N = len(t_arr)
    x_est_hist = np.zeros((N, 6))
    P_hist = np.zeros((N, 6, 6))
    y_pre_hist = np.zeros((N, 6))
    y_post_hist = np.zeros((N, 6))

    ekf.x_pred = ekf.x.copy()
    ekf.P_pred = ekf.P.copy()
    ekf.t_last = t_arr[0]
    sig_p0 = sigma_pos_arr[0]
    sig_v0 = sigma_vel_arr[0]
    ekf.R = np.diag([sig_p0**2, sig_p0**2, sig_p0**2,
                     sig_v0**2, sig_v0**2, sig_v0**2])
    update_from_hill_measurement(ekf, z_meas[0])
    x_est_hist[0] = ekf.x
    P_hist[0] = ekf.P
    y_pre_hist[0] = ekf.y_pre
    y_post_hist[0] = ekf.y_post

    for k in range(1, N):
        sig_p = sigma_pos_arr[k]
        sig_v = sigma_vel_arr[k]
        ekf.R = np.diag([sig_p**2, sig_p**2, sig_p**2,
                         sig_v**2, sig_v**2, sig_v**2])
        ekf.predict(t_arr[k], burn_dv_H=dv_seq_H[k])
        update_from_hill_measurement(ekf, z_meas[k])

        x_est_hist[k] = ekf.x
        P_hist[k] = ekf.P
        y_pre_hist[k] = ekf.y_pre
        y_post_hist[k] = ekf.y_post

    return x_est_hist, P_hist, y_pre_hist, y_post_hist


# ───────── main swarm demo ─────────

def apophis_swarm_ckf_ekf_demo(chief_npz: Path,
                               dep0_npz: Path,
                               N_ckf: int = N_CKF):
    """
    Run CKF+EKF for an 8-deputy swarm in Apophis LVLH, using the chief
    ephemeris from `chief_npz` and burn sequence from `dep0_npz`.
    """
    chief_data = load_apophis_chief_bundle(chief_npz)
    t_full = np.asarray(chief_data["t"], float).reshape(-1)

    impulses = load_apophis_burns_from_dep0(dep0_npz)

    # crop and decimate (same flavor as GEO)
    t0_full = t_full[0]
    t_crop_start = t0_full + 0.5 * 3600.0
    idx_crop = int(np.searchsorted(t_full, t_crop_start))
    t_crop_arr = t_full[idx_crop:]

    idx_decim = np.arange(0, len(t_crop_arr), MEAS_DECIM)
    t_arr = t_crop_arr[idx_decim]
    dt = float(np.median(np.diff(t_arr)))
    print(f"[SWARM] crop @ 0.50 h, decim={MEAS_DECIM}, dt={dt:.1f} s, N={len(t_arr)}")

    chief_state_fn = make_chief_state_fn_from_npz(chief_data)

    dt_full = float(np.median(np.diff(t_crop_arr)))
    dv_full = build_burn_sequence(impulses, t_crop_arr, dt_full)
    dv_seq_H = dv_full[idx_decim]

    # base Apophis LVLH offsets (km → m) for 8 deputies
    base_targets_km = np.array([
        [  1.0,  1.0,  0.5],
        [ -1.0,  1.0, -0.5],
        [  1.0, -1.0,  0.5],
        [ -1.0, -1.0, -0.5],
        [  2.0,  0.0,  0.0],
        [ -2.0,  0.0,  0.0],
        [  0.0,  2.0,  0.0],
        [  0.0, -2.0,  0.0],
    ])
    base_targets_m = base_targets_km * 1e3

    # build a dummy sigma schedule (same for all deps)
    dummy_x0 = np.array([1e3, -800.0, 600.0, 0.05, -0.03, 0.02])
    dummy_true = generate_hill_truth_cw(t_arr, chief_state_fn,
                                        dummy_x0, dv_seq_H, dt)
    rng_dummy = np.random.default_rng(42)
    _, sigma_pos_arr, sigma_vel_arr = build_time_varying_meas(
        rng_dummy, t_arr, dummy_true
    )

    # storage
    x_true_swarm = np.zeros((N_DEP, len(t_arr), 6))
    x_est_swarm  = np.zeros_like(x_true_swarm)
    P_swarm      = np.zeros((N_DEP, len(t_arr), 6, 6))
    y_pre_swarm  = np.zeros_like(x_true_swarm)
    y_post_swarm = np.zeros_like(x_true_swarm)

    for d in range(N_DEP):
        x0_true_d = np.zeros(6)
        x0_true_d[:3] = base_targets_m[d % base_targets_m.shape[0]]
        x0_true_d[3:] = 0.0
        x_true_d = generate_hill_truth_cw(t_arr, chief_state_fn,
                                          x0_true_d, dv_seq_H, dt)
        x_true_swarm[d] = x_true_d

        x_est_d, P_d, y_pre_d, y_post_d = run_ckf_ekf_for_deputy(
            dep_idx=d,
            t_arr=t_arr,
            x_true=x_true_d,
            dv_seq_H=dv_seq_H,
            sigma_pos_arr=sigma_pos_arr,
            sigma_vel_arr=sigma_vel_arr,
            chief_state_fn=chief_state_fn,
            N_ckf=N_ckf,
        )

        x_est_swarm[d]  = x_est_d
        P_swarm[d]      = P_d
        y_pre_swarm[d]  = y_pre_d
        y_post_swarm[d] = y_post_d

    # ─── diagnostics / plots ───

    t_rel = t_arr - t_arr[0]
    t_hr = t_rel / 3600.0
    mask = t_hr >= 0.1
    t_hr_plot = t_hr[mask]

    labels = ["R", "T", "N"]

    for d in range(N_DEP):
        print(f"\n=== Deputy {d} ===")
        x_true_d = x_true_swarm[d]
        x_est_d  = x_est_swarm[d]
        err_d    = x_est_d - x_true_d

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
        axes1[-1].set_xlabel("Time since cropped start [h]")
        axes1[0].legend()
        fig1.suptitle(f"Apophis swarm – Position error vs 3σ – Deputy {d}")
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
        axes2[-1].set_xlabel("Time since cropped start [h]")
        axes2[0].legend()
        fig2.suptitle(f"Apophis swarm – Velocity error vs 3σ – Deputy {d}")
        fig2.tight_layout()

        # Residuals
        y_pre_d  = y_pre_swarm[d][mask]
        y_post_d = y_post_swarm[d][mask]

        # 3) Pre- vs post-fit residuals (position)
        fig3, axes3 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            ax = axes3[i]
            ax.scatter(t_hr_plot, y_pre_d[:, i], s=4, alpha=0.4,
                       label=f"pre-fit ρ_{labels[i]}")
            ax.scatter(t_hr_plot, y_post_d[:, i], s=4, alpha=0.4,
                       label=f"post-fit ρ_{labels[i]}")
            ax.set_ylabel(f"{labels[i]} [m]")
            ax.grid(True)
        axes3[-1].set_xlabel("Time since cropped start [h]")
        axes3[0].legend()
        fig3.suptitle(f"Apophis swarm – Pre- vs post-fit residuals (pos) – Deputy {d}")
        fig3.tight_layout()

        # 4) Pre- vs post-fit residuals (velocity)
        fig4, axes4 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            ax = axes4[i]
            ax.scatter(t_hr_plot, y_pre_d[:, 3+i], s=4, alpha=0.4,
                       label=f"pre-fit ρ̇_{labels[i]}")
            ax.scatter(t_hr_plot, y_post_d[:, 3+i], s=4, alpha=0.4,
                       label=f"post-fit ρ̇_{labels[i]}")
            ax.set_ylabel(f"{labels[i]} [m/s]")
            ax.grid(True)
        axes4[-1].set_xlabel("Time since cropped start [h]")
        axes4[0].legend()
        fig4.suptitle(f"Apophis swarm – Pre- vs post-fit residuals (vel) – Deputy {d}")
        fig4.tight_layout()

    plt.show()


if __name__ == "__main__":
    if not APOPHIS_EXPORT.is_file():
        raise FileNotFoundError(f"Apophis chief export not found at {APOPHIS_EXPORT}")
    if not APOPHIS_DEP0.is_file():
        print(f"[WARN] apophis_rendezvous_full_D0.npz not found at {APOPHIS_DEP0}, "
              f"using zero impulses.")
    apophis_swarm_ckf_ekf_demo(APOPHIS_EXPORT, APOPHIS_DEP0)
