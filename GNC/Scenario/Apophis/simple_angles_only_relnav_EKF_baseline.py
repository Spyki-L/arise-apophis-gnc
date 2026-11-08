#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_angles_only_relnav_EKF_baseline.py
==========================================

EKF-only baseline of the angles-only relative navigation between a
single chief and a single deputy in low-Earth orbit, run entirely in
the ECI frame.  This is the "BEFORE" version — kept around so the
upgraded filter (UKF + 4-D Formation-Identity Manifold Constraint +
chief-as-extended-object + URTSS smoother in
``simple_angles_only_relnav.py``) has something to be compared against.

Differences from the upgraded filter
------------------------------------
* Deputy filter is a textbook EKF (analytic Jacobian on LOS), NOT a
  UKF.  The Jacobian is linearised about the current mean, which biases
  the filter when the angles-only along-LOS uncertainty is large.
* Deputy update uses ONLY the (α, δ) line-of-sight measurement.  No
  FIMC pseudo-measurement, no chief-size monocular range.
* No backward smoother: outputs are the raw forward EKF estimate.

Everything else is identical to the upgraded script — same orbits,
same initial biases, same process noise, same RNG seed — so the
performance gap reflects ONLY the algorithmic differences.

Outputs
-------
* Console:   per-minute snapshot of true vs estimated states.
* Disk:      simple_angles_only_results_EKF_baseline.npz with the full
             per-step time series (truth, estimate, P-diagonals,
             innovations, accept flags).
* Plots:     residuals vs ±3σ for chief and deputy; 3-D trajectories;
             LOS innovation history; |a-priori − truth| convergence.

Author: Harish Vernekar (2026)
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)

# ──────────────────────────────────────────────────────────────────────────────
#                              CONSTANTS / CONFIG
# ──────────────────────────────────────────────────────────────────────────────

# Earth gravity
MU_EARTH    = 3.986004418e14    # [m^3 / s^2]
R_EARTH     = 6378.137e3        # [m]

# Orbit
ALTITUDE    = 700e3             # [m]
R_ORBIT     = R_EARTH + ALTITUDE
V_CIRC      = np.sqrt(MU_EARTH / R_ORBIT)
N_ORBIT     = V_CIRC / R_ORBIT  # mean motion [rad/s]
T_ORBIT     = 2 * np.pi / N_ORBIT

# Formation
BASELINE    = 2000.0            # [m]  along-track separation at t=0

# Run length — KEEP IN SYNC with the upgraded script
DT_NAV      = 1.0                       # [s]   nav step
DURATION_S  = 15 * 3600.0                # [s]
N_STEPS     = int(round(DURATION_S / DT_NAV)) + 1

# Truth integrator sub-stepping
N_RK4_SUBSTEPS = 5

# Chief GPS
SIGMA_GPS_POS = 5.0             # [m]  1-σ at 1 Hz

# Deputy camera
CAMERA_FOV_DEG       = 27.0
CAMERA_RES_M_AT_2KM  = 0.15
CAMERA_NOMINAL_RANGE = 2000.0
CAMERA_RAW_HZ        = 30.0
SIGMA_LOS_RAW        = CAMERA_RES_M_AT_2KM / CAMERA_NOMINAL_RANGE
SIGMA_LOS_EFF        = SIGMA_LOS_RAW / np.sqrt(CAMERA_RAW_HZ / DT_NAV)

# Initial estimate biases (deputy starts close to truth — angles-only
# EKF cannot recover from a large initial velocity bias)
SIGMA_INIT_POS_DEP  = 50.0      # [m]
SIGMA_INIT_VEL_DEP  = 0.05      # [m/s]
SIGMA_INIT_POS_CHF  = 50.0      # [m]
SIGMA_INIT_VEL_CHF  = 0.5       # [m/s]
INIT_BIAS_POS_DEP   = np.array([20.0, -15.0, 10.0])       # [m]
INIT_BIAS_VEL_DEP   = np.array([0.01, -0.008, 0.005])     # [m/s]

# Process noise — IDENTICAL to the upgraded script so the comparison is
# apples-to-apples.  In the EKF-only setup these values are still
# defensible; they just no longer have FIMC/chief-size to anchor the
# along-LOS axis, so the filter has to live with weaker observability.
SIGMA_PROC_POS = 0.05           # [m / √s]
SIGMA_PROC_VEL = 5e-4           # [m/s² / √s]

# NIS gating thresholds
NIS_GATE_GPS_POS = 100.0
NIS_GATE_LOS     = 100.0

# Output cadence
PRINT_EVERY_S = 60.0
PRINT_EVERY_K = int(round(PRINT_EVERY_S / DT_NAV))

# Output paths — separate from the upgraded run so they don't collide
OUT_DIR  = Path(__file__).resolve().parent
NPZ_PATH = OUT_DIR / "simple_angles_only_results_EKF_baseline.npz"
SAVE_FIGS = False
FIG_DIR  = OUT_DIR / "simple_angles_only_figs_EKF_baseline"

RNG_SEED = 1234   # match the upgraded run exactly


# ──────────────────────────────────────────────────────────────────────────────
#                         dynamics:  ECI two-body
# ──────────────────────────────────────────────────────────────────────────────

def two_body_accel(r: np.ndarray) -> np.ndarray:
    rn = la.norm(r)
    return -MU_EARTH * r / (rn ** 3)


def rk4_step(state6: np.ndarray, dt: float) -> np.ndarray:
    def f(s):
        return np.hstack([s[3:6], two_body_accel(s[:3])])
    k1 = f(state6)
    k2 = f(state6 + 0.5 * dt * k1)
    k3 = f(state6 + 0.5 * dt * k2)
    k4 = f(state6 +       dt * k3)
    return state6 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def propagate_truth(state6: np.ndarray, dt_total: float,
                    n_substeps: int = N_RK4_SUBSTEPS) -> np.ndarray:
    dt = dt_total / n_substeps
    s = state6.copy()
    for _ in range(n_substeps):
        s = rk4_step(s, dt)
    return s


def two_body_state_transition(r: np.ndarray, v: np.ndarray,
                              dt: float) -> np.ndarray:
    """ Φ ≈ I + F·dt + ½(F·dt)²   for 2-body dynamics. """
    rn = la.norm(r)
    rhat = r / rn
    G = -(MU_EARTH / rn**3) * (np.eye(3) - 3.0 * np.outer(rhat, rhat))
    F = np.zeros((6, 6))
    F[0:3, 3:6] = np.eye(3)
    F[3:6, 0:3] = G
    Fdt = F * dt
    return np.eye(6) + Fdt + 0.5 * Fdt @ Fdt


# ──────────────────────────────────────────────────────────────────────────────
#                  measurement models
# ──────────────────────────────────────────────────────────────────────────────

def gps_position_measurement(r_truth: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
    return r_truth + SIGMA_GPS_POS * rng.normal(size=3)


def los_angles(r_observer: np.ndarray, r_target: np.ndarray
               ) -> Tuple[float, float, np.ndarray]:
    """ Right ascension α and declination δ of the LOS. """
    dr = r_target - r_observer
    dn = la.norm(dr)
    alpha = np.arctan2(dr[1], dr[0])
    delta = np.arcsin(np.clip(dr[2] / dn, -1.0, 1.0))
    return alpha, delta, dr


def los_jacobian_wrt_observer(dr: np.ndarray) -> np.ndarray:
    """ H_obs = ∂[α, δ] / ∂r_observer    (2×3) """
    dx, dy, dz = dr
    sxy2 = dx*dx + dy*dy
    n2 = sxy2 + dz*dz
    n  = np.sqrt(n2)
    sxy = np.sqrt(sxy2)

    H = np.zeros((2, 3))
    if sxy2 > 1e-12:
        H_alpha_dD = np.array([-dy, dx, 0.0]) / sxy2
        H[0, :] = -H_alpha_dD
    if sxy > 1e-9 and n > 1e-9:
        H_delta_dD = np.array([-dx * dz, -dy * dz, sxy2]) / (sxy * n2)
        H[1, :] = -H_delta_dD
    return H


def generate_los_measurement(r_dep_truth: np.ndarray,
                              r_chief_truth: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
    a, d, _ = los_angles(r_dep_truth, r_chief_truth)
    return np.array([
        a + SIGMA_LOS_EFF * rng.normal(),
        d + SIGMA_LOS_EFF * rng.normal(),
    ])


# ──────────────────────────────────────────────────────────────────────────────
#                                EKF (6-D, ECI)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EKFConfig:
    sigma_proc_pos: float = SIGMA_PROC_POS
    sigma_proc_vel: float = SIGMA_PROC_VEL
    sigma_init_pos: float = 100.0
    sigma_init_vel: float = 0.5
    use_joseph:     bool  = True


class TwoBodyEKF:
    """
    Generic 6-state (r, v) EKF in ECI with 2-body process model.
    Measurement model is supplied per call (z, ẑ, H, R, NIS-gate, label).
    """

    def __init__(self, name: str, cfg: EKFConfig,
                 x0: np.ndarray, P0: Optional[np.ndarray] = None):
        self.name = name
        self.cfg = cfg
        self.x = np.asarray(x0, float).reshape(6).copy()
        if P0 is None:
            pp2 = cfg.sigma_init_pos ** 2
            pv2 = cfg.sigma_init_vel ** 2
            self.P = np.diag([pp2]*3 + [pv2]*3)
        else:
            self.P = np.asarray(P0, float).reshape(6, 6).copy()
        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()
        self.t_last: Optional[float] = None

        self.n_predict = 0
        self.n_update_accept = 0
        self.n_update_reject = 0

    def predict(self, t_abs: float):
        if self.t_last is None:
            self.t_last = t_abs - DT_NAV
        dt = t_abs - self.t_last
        self.x_pred = propagate_truth(self.x, dt, n_substeps=N_RK4_SUBSTEPS)
        Phi = two_body_state_transition(self.x[:3], self.x[3:6], dt)
        Q = self._Q_for_dt(dt)
        self.P_pred = Phi @ self.P @ Phi.T + Q
        self.t_last = t_abs
        self.n_predict += 1

    def _Q_for_dt(self, dt: float) -> np.ndarray:
        qp = (self.cfg.sigma_proc_pos ** 2) * dt
        qv = (self.cfg.sigma_proc_vel ** 2) * dt
        return np.diag([qp]*3 + [qv]*3)

    def update(self, z: np.ndarray, zhat: np.ndarray, H: np.ndarray,
               R: np.ndarray, nis_gate: float, label: str = "") -> dict:
        z = np.asarray(z, float).reshape(-1)
        zhat = np.asarray(zhat, float).reshape(-1)
        H = np.asarray(H, float)
        R = np.asarray(R, float)
        I6 = np.eye(6)

        y = z - zhat
        if label.startswith("LOS"):
            y[0] = (y[0] + np.pi) % (2*np.pi) - np.pi

        S = H @ self.P_pred @ H.T + R
        try:
            nis = float(y @ la.solve(S, y))
        except la.LinAlgError:
            nis = float(y @ y)

        if nis > nis_gate:
            self.x = self.x_pred.copy(); self.P = self.P_pred.copy()
            self.n_update_reject += 1
            return dict(accepted=False, nis=nis, y=y, S=S)

        K = self.P_pred @ H.T @ la.inv(S)
        self.x = self.x_pred + K @ y
        if self.cfg.use_joseph:
            KH = K @ H
            self.P = (I6 - KH) @ self.P_pred @ (I6 - KH).T + K @ R @ K.T
        else:
            self.P = (I6 - K @ H) @ self.P_pred
        self.P = 0.5 * (self.P + self.P.T)
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        self.n_update_accept += 1
        return dict(accepted=True, nis=nis, y=y, S=S)


# ──────────────────────────────────────────────────────────────────────────────
#                          truth + sim driver
# ──────────────────────────────────────────────────────────────────────────────

def initial_chief_state() -> np.ndarray:
    return np.array([R_ORBIT, 0.0, 0.0,    0.0, V_CIRC, 0.0])


def initial_deputy_state() -> np.ndarray:
    """
    Deputy on a 3-D Hill ellipse (A = B = BASELINE).  Identical IC to the
    upgraded script so the comparison is fair.
    """
    A   = BASELINE
    rT0 = 0.0
    B   = BASELINE
    rho_H    = np.array([-0.5 * A,  rT0,  B])
    rhodot_H = np.array([ 0.0,      N_ORBIT * A,  0.0])
    omega = np.array([0.0, 0.0, N_ORBIT])
    chief = initial_chief_state()
    r_dep = chief[:3] + rho_H
    v_dep = chief[3:] + rhodot_H + np.cross(omega, rho_H)
    return np.hstack([r_dep, v_dep])


def run_sim(verbose: bool = True):
    rng = np.random.default_rng(RNG_SEED)

    # ── truth ICs ──
    x_chief_truth = initial_chief_state()
    x_dep_truth   = initial_deputy_state()

    # ── filter ICs (biased; consistent with P_init) ──
    chief_cfg = EKFConfig(sigma_init_pos=SIGMA_INIT_POS_CHF,
                          sigma_init_vel=SIGMA_INIT_VEL_CHF)
    dep_cfg   = EKFConfig(sigma_init_pos=SIGMA_INIT_POS_DEP,
                          sigma_init_vel=SIGMA_INIT_VEL_DEP)

    x0_chief = x_chief_truth + np.hstack([
        SIGMA_INIT_POS_CHF * rng.normal(size=3),
        SIGMA_INIT_VEL_CHF * rng.normal(size=3),
    ])
    x0_dep = x_dep_truth + np.hstack([INIT_BIAS_POS_DEP, INIT_BIAS_VEL_DEP])

    # BOTH filters are EKFs in this baseline.
    chief_ekf = TwoBodyEKF("CHIEF",  chief_cfg, x0_chief)
    dep_ekf   = TwoBodyEKF("DEPUTY", dep_cfg,   x0_dep)

    # ── allocate histories ──
    t_arr   = np.linspace(0.0, (N_STEPS - 1) * DT_NAV, N_STEPS)
    truth_chief = np.zeros((N_STEPS, 6))
    truth_dep   = np.zeros((N_STEPS, 6))
    est_chief   = np.zeros((N_STEPS, 6))
    est_dep     = np.zeros((N_STEPS, 6))
    P_chief     = np.zeros((N_STEPS, 6))
    P_dep       = np.zeros((N_STEPS, 6))
    gps_innov   = np.full((N_STEPS, 3), np.nan)
    los_innov   = np.full((N_STEPS, 2), np.nan)
    los_alpha   = np.full(N_STEPS, np.nan)
    los_delta   = np.full(N_STEPS, np.nan)
    accept_chief = np.zeros(N_STEPS, dtype=bool)
    accept_dep   = np.zeros(N_STEPS, dtype=bool)

    if verbose:
        print("=" * 100)
        print("  SIMPLE ANGLES-ONLY RELATIVE NAVIGATION — EKF BASELINE (no FIMC, no smoother)")
        print("=" * 100)
        print(f"  Chief altitude    : {ALTITUDE/1e3:.1f} km  (R = {R_ORBIT:.6f} m)")
        print(f"  Orbit period      : {T_ORBIT:.3f} s   (mean motion n = {N_ORBIT:.9f} rad/s)")
        print(f"  Baseline          : {BASELINE:.1f} m along-track + cross-track 3-D HCW")
        print(f"  Run length        : {DURATION_S/3600.0:.2f} h "
              f"({DURATION_S/T_ORBIT:.2f} orbits)  "
              f"=> {N_STEPS} nav steps at {1.0/DT_NAV:.1f} Hz")
        print(f"  Camera σ_LOS_eff  : {SIGMA_LOS_EFF*1e6:.3f} µrad ⇒ "
              f"{SIGMA_LOS_EFF*BASELINE*1e3:.3f} mm at {BASELINE/1e3:.1f} km")
        print(f"  GPS σ             : {SIGMA_GPS_POS:.1f} m   (chief only)")
        print(f"  σ_proc_pos        : {SIGMA_PROC_POS:.4f} m / √s")
        print(f"  σ_proc_vel        : {SIGMA_PROC_VEL:.6f} m/s² / √s")
        print()

    # ── main loop ──
    for k in range(N_STEPS):
        t = k * DT_NAV

        # Propagate truth
        if k > 0:
            x_chief_truth = propagate_truth(x_chief_truth, DT_NAV)
            x_dep_truth   = propagate_truth(x_dep_truth,   DT_NAV)

        # ── chief: predict + GPS update ──
        chief_ekf.predict(t)
        z_gps = gps_position_measurement(x_chief_truth[:3], rng)
        H_gps = np.zeros((3, 6)); H_gps[:, :3] = np.eye(3)
        zhat_gps = chief_ekf.x_pred[:3]
        R_gps = (SIGMA_GPS_POS**2) * np.eye(3)
        info_c = chief_ekf.update(z_gps, zhat_gps, H_gps, R_gps,
                                  nis_gate=NIS_GATE_GPS_POS, label="GPS")
        accept_chief[k] = info_c["accepted"]
        gps_innov[k] = info_c["y"] if info_c["accepted"] else np.nan

        # ── deputy: predict ──
        dep_ekf.predict(t)

        # Generate LOS measurement (truth-based)
        z_los = generate_los_measurement(x_dep_truth[:3], x_chief_truth[:3], rng)
        los_alpha[k] = z_los[0]; los_delta[k] = z_los[1]

        # Predicted LOS using deputy x_pred and chief estimate
        r_chief_est = chief_ekf.x[:3]
        a_p, d_p, dr_pred = los_angles(dep_ekf.x_pred[:3], r_chief_est)
        zhat_los = np.array([a_p, d_p])

        # Jacobian: H = [∂[α,δ]/∂r_dep | 0_3]
        H_obs = los_jacobian_wrt_observer(dr_pred)
        H_los = np.hstack([H_obs, np.zeros((2, 3))])

        # Schmidt-augmented R: include chief's position uncertainty along
        # ∂[α,δ]/∂r_chief = -∂[α,δ]/∂r_dep (since dr = r_chief - r_dep).
        H_chief = -H_obs
        R_los = (SIGMA_LOS_EFF**2) * np.eye(2) + \
                H_chief @ chief_ekf.P[:3, :3] @ H_chief.T

        info_d = dep_ekf.update(z_los, zhat_los, H_los, R_los,
                                nis_gate=NIS_GATE_LOS, label="LOS")
        accept_dep[k] = info_d["accepted"]
        los_innov[k]  = info_d["y"] if info_d["accepted"] else np.nan

        # ── record ──
        truth_chief[k] = x_chief_truth
        truth_dep[k]   = x_dep_truth
        est_chief[k]   = chief_ekf.x
        est_dep[k]     = dep_ekf.x
        P_chief[k]     = np.diag(chief_ekf.P)
        P_dep[k]       = np.diag(dep_ekf.P)

        if verbose and (k % PRINT_EVERY_K == 0 or k == N_STEPS - 1):
            err_c = est_chief[k] - truth_chief[k]
            err_d = est_dep[k]   - truth_dep[k]
            range_now = la.norm(x_chief_truth[:3] - x_dep_truth[:3])
            print(f"t = {t:7.1f} s   ({t/T_ORBIT:5.2f} orb)")
            print(f"  CHIEF  err pos = {la.norm(err_c[:3]):8.3f} m   "
                  f"err vel = {la.norm(err_c[3:6])*1e3:8.3f} mm/s")
            print(f"  DEPUTY err pos = {la.norm(err_d[:3]):8.3f} m   "
                  f"err vel = {la.norm(err_d[3:6])*1e3:8.3f} mm/s   "
                  f"(range_truth = {range_now:.1f} m)")
            print()

    # ── summary ──
    if verbose:
        i0 = int(0.5 * T_ORBIT / DT_NAV)
        print("\n" + "=" * 100)
        print("  RESULTS SUMMARY — EKF BASELINE  (after 0.5-orbit transient)")
        print("=" * 100)

        for ekf, truth, est, P_diag, name in [
            (chief_ekf, truth_chief, est_chief, P_chief, "CHIEF"),
            (dep_ekf,   truth_dep,   est_dep,   P_dep,   "DEPUTY"),
        ]:
            err = est - truth
            rms_pos = np.sqrt(np.mean(np.sum(err[i0:, :3]**2, axis=1)))
            rms_vel = np.sqrt(np.mean(np.sum(err[i0:, 3:6]**2, axis=1)))
            rms_pos_axis = np.sqrt(np.mean(err[i0:, :3]**2, axis=0))
            rms_vel_axis = np.sqrt(np.mean(err[i0:, 3:6]**2, axis=0))
            sigma_f = np.sqrt(P_diag[-1])
            print(f"\n  {name}")
            print(f"    n_predict        : {ekf.n_predict}")
            print(f"    n_update_accept  : {ekf.n_update_accept}")
            print(f"    n_update_reject  : {ekf.n_update_reject}")
            print(f"    RMS pos (3-D)    : {rms_pos:.6f} m")
            print(f"    RMS pos per axis : "
                  f"[{rms_pos_axis[0]:.6f}, {rms_pos_axis[1]:.6f}, "
                  f"{rms_pos_axis[2]:.6f}] m")
            print(f"    RMS vel (3-D)    : {rms_vel:.9f} m/s")
            print(f"    RMS vel per axis : "
                  f"[{rms_vel_axis[0]:.9f}, {rms_vel_axis[1]:.9f}, "
                  f"{rms_vel_axis[2]:.9f}] m/s")
            print(f"    Final 1σ (pos)   : "
                  f"[{sigma_f[0]:.6f}, {sigma_f[1]:.6f}, "
                  f"{sigma_f[2]:.6f}] m")
            print(f"    Final 1σ (vel)   : "
                  f"[{sigma_f[3]:.9f}, {sigma_f[4]:.9f}, "
                  f"{sigma_f[5]:.9f}] m/s")
        print()

    # ── save raw arrays ──
    np.savez(NPZ_PATH,
             t=t_arr,
             truth_chief=truth_chief, est_chief=est_chief, P_chief_diag=P_chief,
             truth_dep=truth_dep,     est_dep=est_dep,     P_dep_diag=P_dep,
             gps_innov=gps_innov, los_innov=los_innov,
             los_alpha=los_alpha, los_delta=los_delta,
             accept_chief=accept_chief, accept_dep=accept_dep)
    if verbose:
        print(f"  Saved per-step time series to {NPZ_PATH}\n")

    return dict(
        t=t_arr,
        truth_chief=truth_chief, est_chief=est_chief, P_chief=P_chief,
        truth_dep=truth_dep,     est_dep=est_dep,     P_dep=P_dep,
        gps_innov=gps_innov, los_innov=los_innov,
        accept_chief=accept_chief, accept_dep=accept_dep,
    )


# ──────────────────────────────────────────────────────────────────────────────
#                                 plotting
# ──────────────────────────────────────────────────────────────────────────────

def _residual_panel(ax, t_h, err, three_sigma, ylabel, title):
    ax.plot(t_h, err, lw=0.6, alpha=0.8)
    ax.plot(t_h,  three_sigma, "r--", lw=1.0, alpha=0.7)
    ax.plot(t_h, -three_sigma, "r--", lw=1.0, alpha=0.7)
    ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3); ax.set_title(title)


def plot_results(res):
    if SAVE_FIGS:
        FIG_DIR.mkdir(parents=True, exist_ok=True)

    t = res["t"]
    t_h = t / 3600.0
    labels = ["x", "y", "z"]

    # ── 1) chief residuals vs 3σ ──
    err_c = res["est_chief"] - res["truth_chief"]
    three_sigma_c = 3.0 * np.sqrt(np.maximum(res["P_chief"], 0.0))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    fig.suptitle("CHIEF residuals — EKF baseline", fontsize=14)
    for j in range(3):
        _residual_panel(axes[0, j], t_h, err_c[:, j], three_sigma_c[:, j],
                        f"r_{labels[j]} err [m]", f"r_{labels[j]}")
        _residual_panel(axes[1, j], t_h, err_c[:, 3+j], three_sigma_c[:, 3+j],
                        f"v_{labels[j]} err [m/s]", f"v_{labels[j]}")
    for ax in axes[1]:
        ax.set_xlabel("time [h]")
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(FIG_DIR / "chief_residuals.png", dpi=150)

    # ── 2) deputy residuals vs 3σ ──
    err_d = res["est_dep"] - res["truth_dep"]
    three_sigma_d = 3.0 * np.sqrt(np.maximum(res["P_dep"], 0.0))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    fig.suptitle("DEPUTY residuals — EKF baseline (LOS-only)",
                 fontsize=14)
    for j in range(3):
        _residual_panel(axes[0, j], t_h, err_d[:, j], three_sigma_d[:, j],
                        f"r_{labels[j]} err [m]", f"r_{labels[j]}")
        _residual_panel(axes[1, j], t_h, err_d[:, 3+j], three_sigma_d[:, 3+j],
                        f"v_{labels[j]} err [m/s]", f"v_{labels[j]}")
    for ax in axes[1]:
        ax.set_xlabel("time [h]")
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(FIG_DIR / "deputy_residuals.png", dpi=150)

    # ── 3) 3-D trajectory truth vs est ──
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*res["truth_chief"][:, :3].T, lw=0.6, label="chief truth")
    ax.plot(*res["est_chief"][:, :3].T,   lw=0.6, label="chief est",  alpha=0.7)
    ax.plot(*res["truth_dep"][:, :3].T,   lw=0.6, label="deputy truth")
    ax.plot(*res["est_dep"][:, :3].T,     lw=0.6, label="deputy est", alpha=0.7)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("3-D ECI trajectory — EKF baseline"); ax.legend()
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(FIG_DIR / "trajectory.png", dpi=150)

    # ── 4) LOS innovation ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("LOS innovations — EKF baseline", fontsize=13)
    axes[0].plot(t_h, res["los_innov"][:, 0]*1e6, lw=0.6)
    axes[0].set_ylabel("α innov [µrad]"); axes[0].set_xlabel("time [h]")
    axes[0].grid(True, alpha=0.3); axes[0].set_title("right ascension")
    axes[1].plot(t_h, res["los_innov"][:, 1]*1e6, lw=0.6)
    axes[1].set_ylabel("δ innov [µrad]"); axes[1].set_xlabel("time [h]")
    axes[1].grid(True, alpha=0.3); axes[1].set_title("declination")
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(FIG_DIR / "los_innovation.png", dpi=150)

    # ── 5) per-step deputy position error ──
    fig, ax = plt.subplots(figsize=(10, 4.5))
    err_norm = la.norm(res["est_dep"][:, :3] - res["truth_dep"][:, :3], axis=1)
    ax.semilogy(t_h, err_norm, lw=0.6)
    ax.set_xlabel("time [h]"); ax.set_ylabel("|Δr_deputy| [m]")
    ax.set_title("Deputy position error norm — EKF baseline"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(FIG_DIR / "deputy_error_norm.png", dpi=150)


if __name__ == "__main__":
    res = run_sim(verbose=True)
    plot_results(res)
    plt.show()
