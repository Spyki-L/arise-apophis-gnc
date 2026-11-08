#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swarm_autonomous_nav_demo.py
============================

Centralized Swarm EKF demo for the Apophis rendezvous case.

State per deputy i (chief LVLH):
    x_i = [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N]^T

Swarm state:
    x = [x_0; x_1; ...; x_{N_dep-1}]

Measurements (this demo)
------------------------
Each deputy supplies *relative position to Apophis* in the chief Hill frame:

    z_i(t_k) = Δr_H^{(D-A)}(t_k) + b_k + w_k       (3x1)

where
    Δr_H^{(D-A)} = C_HN(t_k) [ r_D(t_k) - r_A(t_k) ]

and
    C_HN : Hill-from-ECI DCM at the chief,
    b_k  : slowly + quickly varying bias,
    w_k  : zero-mean noise with anisotropic σ (R much worse than T,N),
           plus outliers and dropouts.

We do NOT assume we get such a measurement every nav cycle:
  - only every MEAS_STRIDE step,
  - plus additional random dropouts.

Measurement equation is linear in ρ_i:

    Δr_H^{(D-A)} = ρ_i + C_HN (r_C - r_A)

so we implement:

    z_eff = z_i - C_HN (r_C - r_A) = ρ_i + noise

and update the centralized SwarmHillEKF with

    H = [I_3  0_3] on the i-th deputy block.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
#  Locate ARISE root and import SwarmHillEKF
# --------------------------------------------------------------------------- #

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

from GNC.Navigation.hill_ekf import dcm_lvlh, hill_rate
from GNC.Navigation.swarm_hill_ekf import SwarmHillEKF, SwarmHillEKFConfig

# --------------------------------------------------------------------------- #
#  Global configuration
# --------------------------------------------------------------------------- #

N_DEP = 8

# nav grid
DT_NAV            = 1.0         # [s] nav / EKF step
CROP_AFTER_DEPLOY = 0 * 3600.0 # start 0.5 h after each deputy's deploy time

# true 1σ envelope for T/N components [m]
POS_HI_TRUE   = 150.0   # early
POS_LO_TRUE   =  30.0   # late
TAU_HOURS     = 2.5     # hours

# radial is much worse than T/N (camera-like geometry)
RADIAL_FACTOR = 20.0     # σ_R ≈ RADIAL_FACTOR * σ_TN

# how often we actually have a measurement
MEAS_STRIDE   = 2       # 1 ⇒ every nav step, 2 ⇒ every second step, ...

# bias / dropout / outlier model
BIAS_POS_STD    = 30.0  # [m] slow 3D bias magnitude
BIAS_TAU_HOURS  = 4.0   # [h] slow bias correlation time
P_DROPOUT       = 0.05  # extra random dropouts (on top of stride)
P_OUTLIER       = 0.01  # outlier probability per axis
OUTLIER_SCALE   = 8.0   # outliers ~ this times σ

# EKF thinks measurements are a bit better than they are
POS_SCALE_ASSUMED = 0.8

# Swarm EKF tuning (Q, initial P, etc.)
CFG = SwarmHillEKFConfig(
    dt=DT_NAV,
    # process noise (unmodelled drift, mismodelling, etc.)
    sigma_proc_pos=.0,     # [m]
    sigma_proc_vel=8e-3,    # [m/s]
    # measurement templates (not directly used, but kept reasonable)
    sigma_meas_abs_pos=50.0,
    sigma_meas_abs_vel=0.05,
    sigma_meas_rel_pos=20.0,
    # initial covariance (per deputy)
    sigma_init_pos=1000.0,
    sigma_init_vel=0.5,
    # Δv uncertainty
    sigma_dv=0.08,
    use_joseph=True,
)

# --------------------------------------------------------------------------- #
#  NPZ helpers and ECI→Hill conversion
# --------------------------------------------------------------------------- #

def load_apophis_bundle(path: Path) -> Dict[str, np.ndarray]:
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


def make_state_fn_from_bundle(
    data: Dict[str, np.ndarray],
    r_key: str,
    v_key: str,
):
    """state_fn(t) -> (r_N(t), v_N(t)) using linear interpolation."""
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


def build_burn_sequence(impulses: np.ndarray,
                        t_arr: np.ndarray,
                        dt: float) -> np.ndarray:
    """Map impulses [t, dv_R, dv_T, dv_N] to Δv_H (len(t_arr),3)."""
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


def eci_to_hill_traj(
    t_arr: np.ndarray,
    rD_N: np.ndarray,
    vD_N: np.ndarray,
    rC_N: np.ndarray,
    vC_N: np.ndarray,
) -> np.ndarray:
    """ECI → Hill trajectory x = [ρ, ρ̇] for one deputy wrt chief."""
    rD_N = np.asarray(rD_N, float)
    vD_N = np.asarray(vD_N, float)
    rC_N = np.asarray(rC_N, float)
    vC_N = np.asarray(vC_N, float)
    N = len(t_arr)

    x_true = np.zeros((N, 6))
    for k in range(N):
        rC = rC_N[k]
        vC = vC_N[k]
        rD = rD_N[k]
        vD = vD_N[k]

        C_HN = dcm_lvlh(rC, vC)   # Hill-from-ECI
        r_rel_N = rD - rC
        v_rel_N = vD - vC

        rho = C_HN @ r_rel_N
        omega_HN = np.array([0.0, 0.0, hill_rate(rC, vC)])
        v_rot = C_HN @ v_rel_N
        rhod = v_rot - np.cross(omega_HN, rho)

        x_true[k, 0:3] = rho
        x_true[k, 3:6] = rhod

    return x_true


# --------------------------------------------------------------------------- #
#  Relative-position measurement generator (reduced information)
# --------------------------------------------------------------------------- #

def build_relpos_meas(
    rng: np.random.Generator,
    t_arr: np.ndarray,
    rD_N: np.ndarray,
    rC_N: np.ndarray,
    vC_N: np.ndarray,
    rA_N: np.ndarray,
):
    """
    Build noisy, *partial* relative position measurements in chief Hill:

        z_k = Δr_H^{(D-A)}(t_k) + b_slow_k + b_fast_k + w_k

    where Δr_H^{(D-A)} = C_HN (rD - rA).

    Information reduction:
      - R component has RADIAL_FACTOR times larger σ than T,N.
      - Only every MEAS_STRIDE step carries a valid measurement
        (valid[k] = True); other steps are prediction-only.
    """
    t_arr = np.asarray(t_arr, float).reshape(-1)
    rD_N = np.asarray(rD_N, float).reshape(-1, 3)
    rC_N = np.asarray(rC_N, float).reshape(-1, 3)
    vC_N = np.asarray(vC_N, float).reshape(-1, 3)
    rA_N = np.asarray(rA_N, float).reshape(-1, 3)

    N = len(t_arr)
    if N < 2:
        dt_eff = DT_NAV
    else:
        dt_eff = float(np.median(np.diff(t_arr)))

    # 1) time-varying base σ for T/N
    t_rel = t_arr - t_arr[0]
    tau_env = TAU_HOURS * 3600.0
    w_env = np.exp(-t_rel / tau_env)  # 1 → early, 0 → late

    sigma_base = POS_LO_TRUE + (POS_HI_TRUE - POS_LO_TRUE) * w_env  # (N,)

    sigma_true = np.zeros((N, 3))
    sigma_true[:, 0] = RADIAL_FACTOR * sigma_base   # R much worse
    sigma_true[:, 1] = sigma_base                   # T
    sigma_true[:, 2] = sigma_base                   # N

    # 2) slow AR(1) bias
    bias_tau_sec = BIAS_TAU_HOURS * 3600.0
    rho_slow = np.exp(-dt_eff / max(bias_tau_sec, 1e-3))
    q_slow = BIAS_POS_STD * np.sqrt(1.0 - rho_slow**2)
    b_slow = np.zeros(3)

    # 3) fast jitter (few minutes)
    tau_fast_sec = 5.0 * 60.0
    rho_fast = np.exp(-dt_eff / tau_fast_sec)
    q_fast = 0.3 * POS_LO_TRUE * np.sqrt(1.0 - rho_fast**2)
    b_fast = np.zeros(3)

    # 4) assemble measurements
    z_meas = np.zeros((N, 3))
    valid = np.ones(N, dtype=bool)

    for k in range(N):
        # duty-cycle: only measure every MEAS_STRIDE step
        if (k % MEAS_STRIDE) != 0:
            valid[k] = False
            continue

        if k > 0:
            b_slow = rho_slow * b_slow + q_slow * rng.normal(size=3)
            b_fast = rho_fast * b_fast + q_fast * rng.normal(size=3)

        rC = rC_N[k]
        vC = vC_N[k]
        rD = rD_N[k]
        rA = rA_N[k]

        C_HN = dcm_lvlh(rC, vC)
        r_rel_H_true = C_HN @ (rD - rA)  # deputy − Apophis in Hill

        noise = rng.normal(size=3) * sigma_true[k]
        for j in range(3):
            if rng.random() < P_OUTLIER:
                noise[j] *= OUTLIER_SCALE

        z_meas[k] = r_rel_H_true + b_slow + b_fast + noise

        # additional random dropouts
        if k > 0 and rng.random() < P_DROPOUT:
            valid[k] = False

    sigma_assumed = POS_SCALE_ASSUMED * sigma_true
    return z_meas, sigma_assumed, valid


def update_dep_relpos(
    ekf: SwarmHillEKF,
    t: float,
    dep_idx: int,
    z_rel_H: np.ndarray,
    sigma_vec: np.ndarray,
    chief_state_fn,
    apophis_state_fn,
):
    """
    EKF update for deputy dep_idx using relative pos to Apophis in Hill.

    Measurement model:

        z_rel_H = ρ_i + C_HN (r_C - r_A) + noise

    We convert to a pseudo-measurement of ρ_i:

        z_eff = z_rel_H - C_HN (r_C - r_A) = ρ_i + noise

    and call the SwarmHillEKF's generic linear update with H selecting ρ_i.
    """
    z_rel_H = np.asarray(z_rel_H, float).reshape(3)
    sigma_vec = np.asarray(sigma_vec, float).reshape(3)

    rC, vC = chief_state_fn(t)
    rA, _  = apophis_state_fn(t)
    C_HN = dcm_lvlh(rC, vC)
    offset = C_HN @ (rC - rA)

    z_eff = z_rel_H - offset
    H = np.zeros((3, ekf.nx))
    H[:, 6 * dep_idx:6 * dep_idx + 3] = np.eye(3)
    R = np.diag(sigma_vec**2)

    ekf._update_generic(
        t=t,
        H=H,
        z=z_eff,
        R=R,
        meas_type="abs",
        idx=(dep_idx,),
    )


# --------------------------------------------------------------------------- #
#  Main demo
# --------------------------------------------------------------------------- #

def main():
    np.random.seed(1)

    # Load all deputy bundles
    bundles: List[Dict[str, np.ndarray]] = []
    t_start_list = []
    t_end_list = []

    for d in range(N_DEP):
        path = SCRIPT_DIR / f"apophis_rendezvous_full_D{d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundle: {path}")
        print(f"Loading {path.name}")
        data = load_apophis_bundle(path)
        bundles.append(data)

        t_full = np.asarray(data["t"], float).reshape(-1)
        t_start_list.append(t_full[0] + CROP_AFTER_DEPLOY)
        t_end_list.append(t_full[-1])

    # Global nav grid where all deputies have data
    t_nav_start = max(t_start_list)
    t_nav_end   = min(t_end_list)
    t_nav = np.arange(t_nav_start, t_nav_end + DT_NAV * 0.5, DT_NAV)
    N_nav = len(t_nav)

    print(f"Nav grid: {N_nav} steps, dt_nav = {DT_NAV:.1f} s, "
          f"t ∈ [{t_nav[0]:.1f}, {t_nav[-1]:.1f}]")

    # Chief & Apophis interpolators (use deputy 0 bundle – chief/Apophis are shared)
    chief_state_fn   = make_state_fn_from_bundle(bundles[0], "rC_N", "vC_N")
    apophis_state_fn = make_state_fn_from_bundle(bundles[0], "rA_N", "vA_N")

    # Precompute chief & Apophis on nav grid (for measurement offsets)
    rC_nav = np.zeros((N_nav, 3))
    vC_nav = np.zeros((N_nav, 3))
    rA_nav = np.zeros((N_nav, 3))
    vA_nav = np.zeros((N_nav, 3))
    for k, t_k in enumerate(t_nav):
        rC_nav[k], vC_nav[k] = chief_state_fn(t_k)
        rA_nav[k], vA_nav[k] = apophis_state_fn(t_k)

    # Interpolate each deputy's ECI state onto nav grid, build truth Hill, burns, and measurements
    x_true_swarm = np.zeros((N_DEP, N_nav, 6))
    dv_nav_swarm = np.zeros((N_DEP, N_nav, 3))
    z_swarm      = np.zeros((N_DEP, N_nav, 3))
    sigma_swarm  = np.zeros((N_DEP, N_nav, 3))
    valid_swarm  = np.ones((N_DEP, N_nav), dtype=bool)

    rng_meas = np.random.default_rng(12345)

    for d in range(N_DEP):
        data = bundles[d]
        t_full  = np.asarray(data["t"], float).reshape(-1)
        rD_full = np.asarray(data["rD_N"], float).reshape(-1, 3)
        vD_full = np.asarray(data["vD_N"], float).reshape(-1, 3)
        rC_full = np.asarray(data["rC_N"], float).reshape(-1, 3)
        vC_full = np.asarray(data["vC_N"], float).reshape(-1, 3)
        rA_full = np.asarray(data["rA_N"], float).reshape(-1, 3)
        impulses = np.asarray(data["impulses"], float).reshape(-1, 4)

        def interp_vec(t_grid, arr_src_t, arr):
            out = np.empty((len(t_grid), 3))
            for k in range(3):
                out[:, k] = np.interp(t_grid, arr_src_t, arr[:, k])
            return out

        rD_nav = interp_vec(t_nav, t_full, rD_full)
        vD_nav = interp_vec(t_nav, t_full, vD_full)
        rC_d_nav = interp_vec(t_nav, t_full, rC_full)
        vC_d_nav = interp_vec(t_nav, t_full, vC_full)
        rA_d_nav = interp_vec(t_nav, t_full, rA_full)

        x_true_swarm[d] = eci_to_hill_traj(t_nav, rD_nav, vD_nav, rC_d_nav, vC_d_nav)
        dv_nav_swarm[d] = build_burn_sequence(impulses, t_nav, DT_NAV)

        rng_i = np.random.default_rng(rng_meas.integers(0, 2**31 - 1))
        z_d, sigma_d, valid_d = build_relpos_meas(
            rng_i, t_nav, rD_nav, rC_d_nav, vC_d_nav, rA_d_nav
        )
        z_swarm[d] = z_d
        sigma_swarm[d] = sigma_d
        valid_swarm[d] = valid_d

    # Centralized Swarm EKF
    ekf = SwarmHillEKF(N_dep=N_DEP, cfg=CFG, chief_state_fn=chief_state_fn)

    # Initial guess: truth + common bias on all deputies (pos & vel)
    bias_pos = np.array([500.0, -400.0, 250.0])
    bias_vel = np.array([0.15, -0.10, 0.06])

    x0_true_flat = x_true_swarm[:, 0, :].reshape(-1)
    x0_guess = x0_true_flat.copy()
    for d in range(N_DEP):
        idx = 6 * d
        x0_guess[idx:idx+3]   += bias_pos
        x0_guess[idx+3:idx+6] += bias_vel

    ekf.reset(x0_guess)  # uses CFG.P_init

    # Storage
    x_est_swarm = np.zeros((N_DEP, N_nav, 6))
    P_est_swarm = np.zeros((N_DEP, N_nav, 6, 6))

    # Run EKF on nav grid
    for k, t_k in enumerate(t_nav):
        burns_k = dv_nav_swarm[:, k, :]  # (N_DEP,3)
        ekf.predict(t_k, burns_dv_H=burns_k)

        for d in range(N_DEP):
            if not valid_swarm[d, k]:
                continue
            update_dep_relpos(
                ekf,
                t_k,
                d,
                z_swarm[d, k],
                sigma_swarm[d, k],
                chief_state_fn,
                apophis_state_fn,
            )

        X_flat = ekf.x.reshape(N_DEP, 6)
        for d in range(N_DEP):
            x_est_swarm[d, k] = X_flat[d]
            P_est_swarm[d, k] = ekf.P[6*d:6*d+6, 6*d:6*d+6]

    # ------------------------------------------------------------------ plots --

    t_rel_hr = (t_nav - t_nav[0]) / 3600.0
    labels = ["R", "T", "N"]

    for d in range(N_DEP):
        x_true_d = x_true_swarm[d]
        x_est_d  = x_est_swarm[d]
        err = x_est_d - x_true_d
        err_pos = err[:, 0:3]
        rms = np.sqrt(np.mean(np.sum(err_pos**2, axis=1)))
        print(f"Dep {d}: RMS pos error = {rms:.1f} m")

        sigma_d = np.sqrt(np.diagonal(P_est_swarm[d], axis1=1, axis2=2))
        three_sigma = 3.0 * sigma_d

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            ax = axes[i]
            e = err[:, i]
            s = three_sigma[:, i]
            ax.scatter(t_rel_hr, e, s=8, alpha=0.5,
                       label=f"dep{d} error ρ_{labels[i]}")
            ax.plot(t_rel_hr, s, "r--", linewidth=1.0,
                    label="+3σ" if i == 0 else None)
            ax.plot(t_rel_hr, -s, "r--", linewidth=1.0,
                    label="-3σ" if i == 0 else None)
            ax.set_ylabel(f"{labels[i]} [m]")
            ax.grid(True)
        axes[-1].set_xlabel("Time since nav start [h]")
        axes[0].legend()
        fig.suptitle(f"Swarm autonomous EKF – Deputy {d}")
        fig.tight_layout()

    plt.show()


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    dep0_path = SCRIPT_DIR / "apophis_rendezvous_full_D0.npz"
    if not dep0_path.is_file():
        raise FileNotFoundError(
            f"Expected Apophis bundle not found: {dep0_path}\n"
            f"Run `basilik_rhig_apophis_rendezvous_fleet.py` first."
        )
    main()
