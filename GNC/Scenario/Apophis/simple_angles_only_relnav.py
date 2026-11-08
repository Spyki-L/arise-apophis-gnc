#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_angles_only_relnav.py
============================

Minimal angles-only relative navigation between a single chief and a
single deputy in low-Earth orbit, run entirely in the ECI frame.

Setup
-----
* Chief: 700 km circular LEO, equatorial.  Carries a GPS receiver.
        z_gps = r_chief + N(0, σ_gps² · I_3),  σ_gps = 5 m, 1 Hz.
* Deputy: 2 km along-track from the chief.  Carries ONLY a CCD camera.
        FOV = 27°, GSD-equivalent angular resolution σ ≈ 0.15 m / 2000 m
        ≈ 7.5e-5 rad per axis at 30 Hz.  No ranging beacon — range is
        UNOBSERVABLE from a single bearing measurement; it converges
        only because orbital dynamics change the line-of-sight over time.
* Both spacecraft propagate by simple ECI 2-body gravity.  No burns.
* Run for 6 h at a 1 Hz navigation rate.  Camera measurements at 30 Hz
  are averaged into one effective measurement per nav step, so the
  effective LOS noise becomes σ_los_eff = 7.5e-5 / √30 ≈ 1.37e-5 rad.

State (per spacecraft, ECI):
    x = [r_x, r_y, r_z, v_x, v_y, v_z]^T            (6 elements, m and m/s)

Filters:
    chief_EKF   — 2-body propagation + GPS update (H = [I_3 | 0_3])
    deputy_EKF  — 2-body propagation + LOS angle update (α, δ),
                  with Schmidt augmentation of R using the chief's
                  covariance broadcast.

Outputs:
    * Console:   per-minute snapshot of true vs estimated states with
                 full decimal precision.
    * Disk:      simple_angles_only_results.npz with full per-step time
                 series of truth, estimate, covariance, and residuals.
    * Plots:     position and velocity residuals vs ±3σ for both chief
                 and deputy; 3-D trajectory truth vs estimate; LOS
                 innovation history; |a-priori − truth| convergence.

Author: Harish Vernekar (2026)
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Callable

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

# Run length
DT_NAV      = 1.0                       # [s]   nav step
DURATION_S  = 6 * 3600.0                # [s]   6 hours
N_STEPS     = int(round(DURATION_S / DT_NAV)) + 1

# Truth integrator sub-stepping (for accurate 2-body propagation)
N_RK4_SUBSTEPS = 5  # 0.2-s sub-steps per 1-s nav step

# Chief GPS
SIGMA_GPS_POS = 5.0             # [m]  1-σ at 1 Hz

# Deputy camera
CAMERA_FOV_DEG       = 27.0
CAMERA_RES_M_AT_2KM  = 0.15
CAMERA_NOMINAL_RANGE = 2000.0
CAMERA_RAW_HZ        = 30.0
SIGMA_LOS_RAW        = CAMERA_RES_M_AT_2KM / CAMERA_NOMINAL_RANGE
# When 30 Hz raw measurements are averaged inside one 1-Hz nav step:
SIGMA_LOS_EFF        = SIGMA_LOS_RAW / np.sqrt(CAMERA_RAW_HZ / DT_NAV)

# Initial estimate biases (deputy starts close to truth — angles-only
# EKF cannot recover from a large initial velocity bias because a
# 0.1 m/s velocity error produces secular along-track drift of tens of
# kilometres over a 6-h run, faster than angles-only can correct).
# A real deputy with chief-ranging during initial deploy could anchor
# itself to ~10 m / cm-s.  We model that here.
SIGMA_INIT_POS_DEP  = 50.0      # [m]    deputy's pos uncertainty at t=0
SIGMA_INIT_VEL_DEP  = 0.05      # [m/s]  deputy's vel uncertainty at t=0
SIGMA_INIT_POS_CHF  = 50.0      # [m]    chief converges fast on GPS
SIGMA_INIT_VEL_CHF  = 0.5       # [m/s]
INIT_BIAS_POS_DEP   = np.array([20.0, -15.0, 10.0])       # [m]
INIT_BIAS_VEL_DEP   = np.array([0.01, -0.008, 0.005])     # [m/s]

# Process noise.  Truth IS pure 2-body, but for angles-only navigation
# the *along-LOS* direction is poorly observable until the orbital
# geometry rotates substantially (multiple orbits), and the filter
# WILL collapse σ along that direction far below the actual residual
# bias if Q is too small — looking confident while being wrong.
#
# We size Q to keep the σ-floor near the residual angles-only
# observability floor, ≈ 10 m / few cm/s for a 2 km, 1-orbit baseline.
# This is "consistent" rather than "tight": the filter honestly
# reports its weak along-LOS observability instead of pretending.
#
# UPDATE (post-FIMC-4D + chief-size + smoother): the residuals stagnated
# at ~6 m because Q was sized for the angles-only-only era and now
# starves the filter of resolution.  With three independent information
# sources (LOS, chief-size, 4-D FIMC) the σ-floor is no longer at risk
# of collapse, so Q can return to a tight value reflecting the truth's
# pure 2-body dynamics.
SIGMA_PROC_POS = 0.05           # [m / √s]    (was 0.5 — too generous now)
SIGMA_PROC_VEL = 5e-4           # [m/s² / √s] (was 5e-3 — too generous now)

# NIS gating thresholds.  Loose by design: the simulation has no
# outliers, so gating can only HURT a clean run.
NIS_GATE_GPS_POS = 100.0
NIS_GATE_LOS     = 100.0
NIS_GATE_SIZE    = 100.0
NIS_GATE_FIMC    = 100.0

# ── A. Chief as an extended optical target ─────────────────────────────
# The deputy's existing camera resolves the chief as ≈ chord/range
# radians.  For chord = 5 m at 2 km, that's 2.5 mrad — about 33 pixels
# at the camera's 7.5e-5 rad/pixel resolution.  Treating chief as an
# extended object turns "angles only" into "angles + monocular range"
# without adding any hardware.
CHIEF_CHORD_M       = 5.0      # [m]   assumed chief body characteristic length
# Sub-pixel centroiding on the chief's ~33-pixel disk: centroid-of-mass
# on a resolved extended object yields ≈ 0.2 px (5× tighter than 1 px).
# This is a pure software upgrade — same camera, same exposures.
SIGMA_SIZE_RAD      = 1.5e-5   # 0.2-pixel angular resolution per nav step

# ── B. Formation-Identity Manifold Constraint (FIMC, novel) ───────────
# UPGRADE: 4-D version.  A co-orbital deputy shares the chief's FULL
# integrals-of-motion vector — same orbital plane, same shape, same
# semi-major axis — and only the true anomaly is free.
#
#     Specific energy        ε  = ½ |v|² − μ/|r|         (1 scalar)
#     Specific ang. mom. h    h  = r × v                  (3-vector)
#
# Constraining (ε, h) reduces the deputy's 6-D state to a 1-D manifold
# (true anomaly only); the chief-size measurement then anchors that
# remaining DoF.  This is the rigorous formulation — energy alone was
# the timid 1-D version.
#
# σ for each component must honor the chief's own broadcast uncertainty
# (≈ 5 m, 5 mm/s on chief state) AND the formation control tolerance.
#
#     δε ≈ |v|·δv + (μ/r²)·δr  ≈ 7600·5e-3 + 8.4·5 ≈ 80 m²/s²
#     δh ≈ |r|·δv + |v|·δr     ≈ 7e6·5e-3 + 7600·5 ≈ 7.3e4 m²/s
#
FIMC_DELTA_A_TOL    = 12.0     # [m]    tolerated δa between deputy and chief
SIGMA_FIMC_EPS      = 80.0     # ≈ chief-broadcast uncertainty in ε  [m²/s²]
SIGMA_FIMC_H        = 1.0e5    # ≈ chief-broadcast uncertainty in h  [m²/s]
# Backward-compat alias (old 1-D code path used SIGMA_FIMC):
SIGMA_FIMC          = SIGMA_FIMC_EPS

# Output cadence
PRINT_EVERY_S = 60.0
PRINT_EVERY_K = int(round(PRINT_EVERY_S / DT_NAV))

# Output paths
OUT_DIR = Path(__file__).resolve().parent
NPZ_PATH = OUT_DIR / "simple_angles_only_results.npz"
SAVE_FIGS = False
FIG_DIR  = OUT_DIR / "simple_angles_only_figs"

RNG_SEED = 1234


# ──────────────────────────────────────────────────────────────────────────────
#                         dynamics:  ECI two-body
# ──────────────────────────────────────────────────────────────────────────────

def two_body_accel(r: np.ndarray) -> np.ndarray:
    """ a = -μ r / |r|^3 """
    rn = la.norm(r)
    return -MU_EARTH * r / (rn ** 3)


def rk4_step(state6: np.ndarray, dt: float) -> np.ndarray:
    """One RK4 step of the 2-body equations on a 6-state [r, v]."""
    def f(s):
        return np.hstack([s[3:6], two_body_accel(s[:3])])
    k1 = f(state6)
    k2 = f(state6 + 0.5 * dt * k1)
    k3 = f(state6 + 0.5 * dt * k2)
    k4 = f(state6 +       dt * k3)
    return state6 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def propagate_truth(state6: np.ndarray, dt_total: float,
                    n_substeps: int = N_RK4_SUBSTEPS) -> np.ndarray:
    """Multi-step RK4 propagation."""
    dt = dt_total / n_substeps
    s = state6.copy()
    for _ in range(n_substeps):
        s = rk4_step(s, dt)
    return s


def two_body_state_transition(r: np.ndarray, v: np.ndarray,
                              dt: float) -> np.ndarray:
    """
    Approximate 6×6 state-transition matrix for 2-body dynamics about
    the current state (r, v).  Uses Φ ≈ I + F·dt + ½(F·dt)².

    F = [[0_3,        I_3 ],
         [G(r),       0_3 ]]

    where G(r) = -μ/r^3 · (I_3 - 3 r̂ r̂^T) is the gravity-gradient matrix.
    """
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

def gps_position_measurement(r_truth: np.ndarray, rng: np.random.Generator
                              ) -> np.ndarray:
    return r_truth + SIGMA_GPS_POS * rng.normal(size=3)


def los_angles(r_observer: np.ndarray, r_target: np.ndarray
               ) -> Tuple[float, float, np.ndarray]:
    """
    Right ascension (α) and declination (δ) of the line-of-sight from
    observer to target, in the ECI frame.

        Δ = r_target - r_observer
        α = atan2(Δy, Δx)
        δ = asin(Δz / |Δ|)

    Returns (alpha, delta, Δ).
    """
    dr = r_target - r_observer
    dn = la.norm(dr)
    alpha = np.arctan2(dr[1], dr[0])
    delta = np.arcsin(np.clip(dr[2] / dn, -1.0, 1.0))
    return alpha, delta, dr


def los_jacobian_wrt_observer(dr: np.ndarray) -> np.ndarray:
    """
    H_obs = ∂[α, δ] / ∂r_observer    (2×3)

    Note: ∂h/∂r_observer = -∂h/∂Δ  because Δ = r_target - r_observer.
    """
    dx, dy, dz = dr
    sxy2 = dx*dx + dy*dy
    n2 = sxy2 + dz*dz
    n  = np.sqrt(n2)
    sxy = np.sqrt(sxy2)

    H = np.zeros((2, 3))
    if sxy2 > 1e-12:
        # ∂α/∂Δ = [-dy, dx, 0] / sxy²
        H_alpha_dD = np.array([-dy, dx, 0.0]) / sxy2
        H[0, :] = -H_alpha_dD
    if sxy > 1e-9 and n > 1e-9:
        # ∂δ/∂Δ = (1/sxy/n²) · [-dx·dz, -dy·dz, sxy²]
        H_delta_dD = np.array([-dx * dz, -dy * dz, sxy2]) / (sxy * n2)
        H[1, :] = -H_delta_dD
    return H


def generate_los_measurement(r_dep_truth: np.ndarray, r_chief_truth: np.ndarray,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Generate a noisy (alpha, delta) measurement.  The per-axis noise
    SIGMA_LOS_EFF already incorporates the √30 averaging of raw 30-Hz
    measurements over the 1-Hz nav step.
    """
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
    Measurement model is supplied per call (Z, H, R, NIS-gate, label).
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

        # diagnostics
        self.n_predict = 0
        self.n_update_accept = 0
        self.n_update_reject = 0

    # ─────────────────── predict ───────────────────

    def predict(self, t_abs: float):
        if self.t_last is None:
            self.t_last = t_abs - DT_NAV
        dt = t_abs - self.t_last
        # Propagate state via RK4 (truth-grade)
        self.x_pred = propagate_truth(self.x, dt, n_substeps=N_RK4_SUBSTEPS)
        # Linearise about the current state for covariance
        Phi = two_body_state_transition(self.x[:3], self.x[3:6], dt)
        Q = self._Q_for_dt(dt)
        self.P_pred = Phi @ self.P @ Phi.T + Q
        self.t_last = t_abs
        self.n_predict += 1

    def _Q_for_dt(self, dt: float) -> np.ndarray:
        qp = (self.cfg.sigma_proc_pos ** 2) * dt
        qv = (self.cfg.sigma_proc_vel ** 2) * dt
        return np.diag([qp]*3 + [qv]*3)

    # ─────────────────── generic measurement update ───────────────────

    def update(self, z: np.ndarray, zhat: np.ndarray, H: np.ndarray,
               R: np.ndarray, nis_gate: float, label: str = "") -> dict:
        z = np.asarray(z, float).reshape(-1)
        zhat = np.asarray(zhat, float).reshape(-1)
        H = np.asarray(H, float)
        R = np.asarray(R, float)
        I6 = np.eye(6)

        y = z - zhat
        # Wrap angle innovations to [-π, π] when they look like angles:
        if label.startswith("LOS"):
            y[0] = (y[0] + np.pi) % (2*np.pi) - np.pi
            # δ ∈ (-π/2, π/2) — no wrap needed

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
#                                Unscented KF (6-D, ECI)
# ──────────────────────────────────────────────────────────────────────────────

class TwoBodyUKF:
    """
    6-state Unscented Kalman Filter for the same 2-body ECI problem.

    Why UKF here: the LOS measurement h(x_dep, r_chief) = (atan2, asin)
    is highly non-linear in deputy position, especially when the
    estimate-to-truth bias is comparable to the chief–deputy range.
    The EKF's tangent-plane Jacobian is then a poor local model and
    drives the filter to overconfident, biased posteriors.  The UKF
    deterministically propagates 2n+1 sigma points through the FULL
    non-linear dynamics and FULL non-linear measurement function and
    recovers an empirical mean/covariance — capturing curvature that
    the Jacobian throws away.

    Interface:
        ukf.predict(t)
            advances state through 2-body dynamics; caches the
            propagated sigma points in self.sp_pred.
        ukf.update(z, h_fn, R, nis_gate, label, aux)
            applies a non-linear measurement update where h_fn(x, aux)
            returns the predicted measurement vector.
    """

    def __init__(self, name: str, cfg: EKFConfig,
                 x0: np.ndarray, P0: Optional[np.ndarray] = None,
                 alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.name = name
        self.cfg = cfg
        self.n = 6
        self.x = np.asarray(x0, float).reshape(6).copy()
        if P0 is None:
            pp2 = cfg.sigma_init_pos ** 2
            pv2 = cfg.sigma_init_vel ** 2
            self.P = np.diag([pp2]*3 + [pv2]*3)
        else:
            self.P = np.asarray(P0, float).reshape(6, 6).copy()
        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()
        self.sp_pred: Optional[np.ndarray] = None  # cached propagated sigma points

        # Unscented transform parameters (Wan & van der Merwe, 2000)
        self.alpha_ut = alpha
        self.beta_ut  = beta
        self.kappa    = kappa
        self.lam      = alpha**2 * (self.n + kappa) - self.n
        self.gamma    = np.sqrt(self.n + self.lam)

        # Mean and covariance weights (length 2n+1)
        n2p1 = 2 * self.n + 1
        self.Wm = np.full(n2p1, 1.0 / (2.0 * (self.n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.Wm[0] + (1.0 - alpha**2 + beta)

        self.t_last: Optional[float] = None
        self.n_predict = 0
        self.n_update_accept = 0
        self.n_update_reject = 0

    # ─────────────────── sigma points ───────────────────

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """2n+1 sigma points around (x, P) per Julier–Uhlmann."""
        n = self.n
        scale = (n + self.lam)
        try:
            S = np.linalg.cholesky(scale * P)
        except np.linalg.LinAlgError:
            # Add jitter to ensure positive-definiteness
            jitter = 1e-9 * (np.max(np.abs(np.diag(P))) + 1.0)
            S = np.linalg.cholesky(scale * (P + jitter * np.eye(n)))
        sp = np.zeros((2*n + 1, n))
        sp[0] = x
        for i in range(n):
            sp[i + 1]     = x + S[:, i]
            sp[i + 1 + n] = x - S[:, i]
        return sp

    # ─────────────────── predict ───────────────────

    def predict(self, t_abs: float):
        if self.t_last is None:
            self.t_last = t_abs - DT_NAV
        dt = t_abs - self.t_last

        # Sample the prior with sigma points
        sp = self._sigma_points(self.x, self.P)

        # Push every sigma point through the full nonlinear 2-body dynamics
        sp_pred = np.empty_like(sp)
        for i in range(sp.shape[0]):
            sp_pred[i] = propagate_truth(sp[i], dt, n_substeps=N_RK4_SUBSTEPS)

        # Empirical mean and covariance from propagated sigma points
        x_pred = (self.Wm[:, None] * sp_pred).sum(axis=0)
        P_pred = np.zeros((self.n, self.n))
        # Cross-covariance between current posterior and next-step prior.
        # Needed for the Unscented RTS smoother (URTSS) backward pass.
        P_cross = np.zeros((self.n, self.n))
        x_post = self.x.copy()  # x_post_{k-1} entering this predict
        for i in range(sp.shape[0]):
            d_pred = sp_pred[i] - x_pred
            P_pred  += self.Wc[i] * np.outer(d_pred, d_pred)
            P_cross += self.Wc[i] * np.outer(sp[i] - x_post, d_pred)
        # Add process noise
        qp = (self.cfg.sigma_proc_pos ** 2) * dt
        qv = (self.cfg.sigma_proc_vel ** 2) * dt
        P_pred += np.diag([qp]*3 + [qv]*3)
        P_pred = 0.5 * (P_pred + P_pred.T)

        self.x_pred  = x_pred
        self.P_pred  = P_pred
        self.sp_pred = sp_pred
        self.P_cross = P_cross   # smoother needs P_cross between x_post_{k-1} and x_pred_k
        self.t_last  = t_abs
        self.n_predict += 1

    # ─────────────────── nonlinear measurement update ───────────────────

    def update(self, z: np.ndarray, h_fn: Callable, R: np.ndarray,
               nis_gate: float = 1e10, label: str = "",
               aux=None) -> dict:
        """
        Generic non-linear measurement update.

            z       : measurement vector (n_z,)
            h_fn    : callable h_fn(state_6, aux) -> predicted_meas (n_z,)
            R       : measurement noise covariance (n_z, n_z)
            label   : if it starts with "LOS", the first measurement
                      component is wrapped to [-π, π] when subtracting.
            aux     : passed through to h_fn (e.g. the chief's estimated
                      position for LOS h_fn)
        """
        z = np.asarray(z, float).reshape(-1)
        n_z = z.size

        # Pass each PROPAGATED sigma point through h
        Z = np.empty((2 * self.n + 1, n_z))
        for i in range(2 * self.n + 1):
            Z[i] = h_fn(self.sp_pred[i], aux)

        # Predicted measurement (mean)
        z_pred = (self.Wm[:, None] * Z).sum(axis=0)
        if label.startswith("LOS"):
            # Wrap predicted azimuth — important for the mean computation
            # when sigma points straddle the ±π discontinuity.  Done by
            # de-wrapping each Z[:, 0] relative to z_pred.
            for i in range(2 * self.n + 1):
                d = Z[i, 0] - z_pred[0]
                if d > np.pi:
                    Z[i, 0] -= 2 * np.pi
                elif d < -np.pi:
                    Z[i, 0] += 2 * np.pi
            z_pred = (self.Wm[:, None] * Z).sum(axis=0)

        # Innovation covariance and cross-covariance
        S   = np.array(R, dtype=float).copy()
        Pxz = np.zeros((self.n, n_z))
        for i in range(2 * self.n + 1):
            dz = Z[i] - z_pred
            dx = self.sp_pred[i] - self.x_pred
            S   += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        # Innovation
        y = z - z_pred
        if label.startswith("LOS"):
            y[0] = (y[0] + np.pi) % (2 * np.pi) - np.pi

        # NIS gate
        try:
            nis = float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            nis = float(y @ y)
        if nis > nis_gate:
            self.x = self.x_pred.copy(); self.P = self.P_pred.copy()
            self.n_update_reject += 1
            return dict(accepted=False, nis=nis, y=y, S=S)

        # Kalman gain and posterior
        try:
            K = Pxz @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.x = self.x_pred.copy(); self.P = self.P_pred.copy()
            self.n_update_reject += 1
            return dict(accepted=False, nis=nis, y=y, S=S)

        self.x = self.x_pred + K @ y
        self.P = self.P_pred - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Reset prediction state and refresh sigma points around the
        # posterior so any subsequent update at the same time step
        # sees the updated geometry.
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        self.sp_pred = self._sigma_points(self.x, self.P)

        self.n_update_accept += 1
        return dict(accepted=True, nis=nis, y=y, S=S)


# ──────────────────────────────────────────────────────────────────────────────
#                          truth + sim driver
# ──────────────────────────────────────────────────────────────────────────────

def initial_chief_state() -> np.ndarray:
    """Chief at perigee equivalent, equatorial circular orbit, +X position."""
    return np.array([R_ORBIT, 0.0, 0.0,    0.0, V_CIRC, 0.0])


def initial_deputy_state() -> np.ndarray:
    """
    Deputy on a stable HCW relative orbit around the chief.

    With a co-orbital "same orbit, phase offset" formation the relative
    geometry is essentially frozen and angles-only navigation has no
    range observability.  Instead we put the deputy on a closed Hill
    ellipse with non-trivial radial oscillation:

        ρ_R(t)  = -A/2 cos(n t + φ)         radial    (oscillates)
        ρ_T(t)  =  A   sin(n t + φ) + ρ_T0  tangential (offset + osc.)
        ρ_N(t)  =  B   cos(n t + ψ)         cross-track

    For φ=0, ψ=0 this gives initial Hill state:
        ρ      = (-A/2,  ρ_T0,  B)
        ρ̇      = ( 0,    n A,    0)         in Hill frame

    Choose A = BASELINE = 2 km in-plane semi-major axis,
           ρ_T0 = 0      (centre of ellipse aligned with chief),
           B = 0         (planar formation — keep the demo flat).

    The deputy's range to chief swings ≈1.5–2.5 km over each orbit,
    which is exactly the geometric diversity angles-only navigation
    needs to break range ambiguity.
    """
    A   = BASELINE        # in-plane HCW amplitude [m]
    rT0 = 0.0             # tangential offset of ellipse centre
    # Cross-track amplitude — set EQUAL to in-plane amplitude so the
    # relative orbit is genuinely three-dimensional and α and δ
    # measurements carry comparable information content.
    B   = BASELINE        # cross-track amplitude [m]  (= 2 km)

    # Hill-frame state at t=0
    # ρ_R(0) = -A/2,  ρ_T(0) = ρ_T0,  ρ_N(0) = B  (peak of cross-track)
    # ρ̇_R(0) = 0,    ρ̇_T(0) = n A,    ρ̇_N(0) = 0  (cross-track at peak ⇒ velocity = 0)
    rho_H    = np.array([-0.5 * A,  rT0,  B])
    rhodot_H = np.array([ 0.0,      N_ORBIT * A,  0.0])

    # Convert Hill -> ECI offset relative to chief.
    # At t=0 the chief sits at +X with velocity in +Y, so its
    # LVLH (R, T, N) axes coincide with ECI (x, y, z).  Hence
    # rho_inertial == rho_H.  The Hill velocity needs to be
    # corrected for the rotating-frame Coriolis term:
    #     v_inertial = v_chief + v_H + Ω × ρ
    # with Ω = n ẑ for a circular chief in the equatorial plane.
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

    # Chief: linear GPS measurement, EKF is fine and very accurate.
    chief_ekf = TwoBodyEKF("CHIEF", chief_cfg, x0_chief)
    # Deputy: highly non-linear LOS measurement; UKF handles this without
    # the Jacobian-induced bias the EKF accumulates over orbital periods.
    dep_ekf   = TwoBodyUKF("DEPUTY", dep_cfg, x0_dep)

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

    # Smoother histories: full posterior, prior, and cross-cov from each step
    # so we can run a backward URTSS pass after the forward filter completes.
    sm_x_post   = np.zeros((N_STEPS, 6))
    sm_P_post   = np.zeros((N_STEPS, 6, 6))
    sm_x_pred   = np.zeros((N_STEPS, 6))   # predicted state at step k from k-1
    sm_P_pred   = np.zeros((N_STEPS, 6, 6))
    sm_P_cross  = np.zeros((N_STEPS, 6, 6))

    # initial step
    truth_chief[0] = x_chief_truth
    truth_dep[0]   = x_dep_truth
    est_chief[0]   = chief_ekf.x
    est_dep[0]     = dep_ekf.x
    P_chief[0]     = np.diag(chief_ekf.P)
    P_dep[0]       = np.diag(dep_ekf.P)
    sm_x_post[0]   = dep_ekf.x.copy()
    sm_P_post[0]   = dep_ekf.P.copy()
    sm_x_pred[0]   = dep_ekf.x.copy()
    sm_P_pred[0]   = dep_ekf.P.copy()
    sm_P_cross[0]  = np.eye(6) * 1e-9

    if verbose:
        print(f"\nSimple angles-only relative navigation")
        print(f"  Orbit period      : {T_ORBIT:.3f} s "
              f"({T_ORBIT/60:.3f} min)")
        print(f"  V_circ            : {V_CIRC:.6f} m/s")
        print(f"  Mean motion n     : {N_ORBIT:.9e} rad/s")
        print(f"  Baseline (||Δr||₀) : {BASELINE:.3f} m")
        print(f"  σ_GPS             : {SIGMA_GPS_POS:.3f} m")
        print(f"  σ_LOS raw         : {SIGMA_LOS_RAW:.6e} rad "
              f"(≈ {np.degrees(SIGMA_LOS_RAW)*3600:.3f} arcsec)")
        print(f"  σ_LOS effective   : {SIGMA_LOS_EFF:.6e} rad "
              f"(≈ {np.degrees(SIGMA_LOS_EFF)*3600:.3f} arcsec) "
              f"after √{int(CAMERA_RAW_HZ)} averaging")
        print(f"  Run length        : {DURATION_S:.0f} s "
              f"({DURATION_S/3600:.3f} h, {DURATION_S/T_ORBIT:.3f} orbits)")
        print(f"  N_steps           : {N_STEPS}")
        print(f"  Print cadence     : every {PRINT_EVERY_S:.0f} s "
              f"(=every {PRINT_EVERY_K} steps)")
        print()

    # ── main loop ──
    for k in range(1, N_STEPS):
        t = t_arr[k]

        # Truth propagation
        x_chief_truth = propagate_truth(x_chief_truth, DT_NAV)
        x_dep_truth   = propagate_truth(x_dep_truth,   DT_NAV)

        # EKF predict (both)
        chief_ekf.predict(t)
        dep_ekf.predict(t)

        # ── Chief: GPS update ──
        z_gps = gps_position_measurement(x_chief_truth[:3], rng)
        H_gps = np.zeros((3, 6)); H_gps[:, :3] = np.eye(3)
        R_gps = (SIGMA_GPS_POS**2) * np.eye(3)
        zhat_gps = chief_ekf.x_pred[:3]
        info = chief_ekf.update(z_gps, zhat_gps, H_gps, R_gps,
                                NIS_GATE_GPS_POS, label="GPS")
        accept_chief[k] = info["accepted"]
        gps_innov[k] = info["y"] if info["accepted"] else np.nan

        # ── Save smoother histories: prior at step k (right after predict) ──
        sm_x_pred[k]  = dep_ekf.x_pred.copy()
        sm_P_pred[k]  = dep_ekf.P_pred.copy()
        sm_P_cross[k] = dep_ekf.P_cross.copy()

        # ── Deputy: LOS update via UKF ──
        z_los = generate_los_measurement(x_dep_truth[:3],
                                          x_chief_truth[:3], rng)
        los_alpha[k], los_delta[k] = z_los

        r_chief_est = chief_ekf.x[:3]
        _, _, dr_pred = los_angles(dep_ekf.x_pred[:3], r_chief_est)
        H_obs_pos    = los_jacobian_wrt_observer(dr_pred)
        H_chief_pos  = -H_obs_pos
        R_consider   = H_chief_pos @ chief_ekf.P[:3, :3] @ H_chief_pos.T
        R_los        = (SIGMA_LOS_EFF**2) * np.eye(2) + R_consider

        def _los_h(x_dep, r_chief):
            a, d, _ = los_angles(x_dep[:3], r_chief)
            return np.array([a, d])

        info = dep_ekf.update(z_los, _los_h, R_los,
                              nis_gate=NIS_GATE_LOS, label="LOS",
                              aux=r_chief_est)
        accept_dep[k] = info["accepted"]
        los_innov[k]  = info["y"] if info["accepted"] else np.nan

        # ── A. Chief-as-extended-object: angular size → range ──
        # Truth: chord / range.  The deputy's camera resolves the chief's
        # body (≈ CHIEF_CHORD_M metres) as ≈ chord/range radians; one-pixel
        # resolution σ_size_rad gives a few-meter range precision per
        # measurement.  Same camera, no new hardware.
        true_size = CHIEF_CHORD_M / la.norm(x_chief_truth[:3] - x_dep_truth[:3])
        z_size = np.array([true_size + SIGMA_SIZE_RAD * rng.normal()])
        R_size = np.array([[SIGMA_SIZE_RAD**2 +
                            # also consider chief uncertainty contribution
                            (CHIEF_CHORD_M / la.norm(dr_pred)**2)**2 *
                            np.trace(chief_ekf.P[:3, :3]) / 3]])

        def _size_h(x_dep, r_chief):
            return np.array([CHIEF_CHORD_M / la.norm(r_chief - x_dep[:3])])

        dep_ekf.update(z_size, _size_h, R_size,
                       nis_gate=NIS_GATE_SIZE, label="SIZE",
                       aux=r_chief_est)

        # ── B. Formation-Identity Manifold Constraint (NOVEL, 4-D) ──
        # Synthetic pseudo-measurement that the deputy's full integrals
        # of motion equal the chief's, by mission-design assumption.
        #
        #     ε  = ½|v|² − μ/|r|       (specific orbital energy)
        #     h  = r × v               (specific angular momentum 3-vector)
        #
        # Together (ε, h) collapse the deputy's 6-D state to a 1-D
        # manifold parameterised by true anomaly; the chief-size
        # measurement then anchors that final DoF.  Information source:
        # the chief's broadcast on the formation uplink — no new
        # hardware on the deputy.  Effect: kills BOTH the along-LOS
        # ambiguity AND any cross-track plane-mismatch drift, which
        # the 1-D ε-only version missed.
        r_c = chief_ekf.x[:3]
        v_c = chief_ekf.x[3:]
        eps_chief = 0.5 * np.dot(v_c, v_c) - MU_EARTH / la.norm(r_c)
        h_chief   = np.cross(r_c, v_c)
        z_fimc = np.hstack([[eps_chief], h_chief])
        R_fimc = np.diag([SIGMA_FIMC_EPS**2,
                          SIGMA_FIMC_H**2,
                          SIGMA_FIMC_H**2,
                          SIGMA_FIMC_H**2])

        def _fimc_h(x_dep, _aux):
            r = x_dep[:3]; v = x_dep[3:]
            eps = 0.5 * np.dot(v, v) - MU_EARTH / la.norm(r)
            h   = np.cross(r, v)
            return np.hstack([[eps], h])

        dep_ekf.update(z_fimc, _fimc_h, R_fimc,
                       nis_gate=NIS_GATE_FIMC, label="FIMC",
                       aux=None)

        # ── Save smoother histories: posterior at step k ──
        sm_x_post[k] = dep_ekf.x.copy()
        sm_P_post[k] = dep_ekf.P.copy()

        # ── record ──
        truth_chief[k] = x_chief_truth
        truth_dep[k]   = x_dep_truth
        est_chief[k]   = chief_ekf.x
        est_dep[k]     = dep_ekf.x
        P_chief[k]     = np.diag(chief_ekf.P)
        P_dep[k]       = np.diag(dep_ekf.P)

        # ── periodic printout (full decimal precision) ──
        if verbose and (k % PRINT_EVERY_K == 0 or k == N_STEPS - 1):
            err_c = est_chief[k] - truth_chief[k]
            err_d = est_dep[k]   - truth_dep[k]
            sigma_c = np.sqrt(P_chief[k])
            sigma_d = np.sqrt(P_dep[k])
            range_now = la.norm(truth_dep[k, :3] - truth_chief[k, :3])
            print(f"t = {t:11.6f} s   ({t/60:8.4f} min,  "
                  f"orbit-phase = {(t/T_ORBIT)%1.0:.6f})")
            print(f"  CHIEF  truth r  = "
                  f"[{truth_chief[k,0]:.6f}, {truth_chief[k,1]:.6f}, {truth_chief[k,2]:.6f}] m")
            print(f"         est   r  = "
                  f"[{est_chief[k,0]:.6f}, {est_chief[k,1]:.6f}, {est_chief[k,2]:.6f}] m")
            print(f"         err   r  = "
                  f"[{err_c[0]:+.6f}, {err_c[1]:+.6f}, {err_c[2]:+.6f}] m   "
                  f"|err| = {la.norm(err_c[:3]):.6f} m   "
                  f"(σ_pos = [{sigma_c[0]:.3f}, {sigma_c[1]:.3f}, {sigma_c[2]:.3f}] m)")
            print(f"         err   v  = "
                  f"[{err_c[3]:+.9f}, {err_c[4]:+.9f}, {err_c[5]:+.9f}] m/s   "
                  f"|err| = {la.norm(err_c[3:]):.9f} m/s")
            print(f"  DEPUTY truth r  = "
                  f"[{truth_dep[k,0]:.6f}, {truth_dep[k,1]:.6f}, {truth_dep[k,2]:.6f}] m")
            print(f"         est   r  = "
                  f"[{est_dep[k,0]:.6f}, {est_dep[k,1]:.6f}, {est_dep[k,2]:.6f}] m")
            print(f"         err   r  = "
                  f"[{err_d[0]:+.6f}, {err_d[1]:+.6f}, {err_d[2]:+.6f}] m   "
                  f"|err| = {la.norm(err_d[:3]):.6f} m   "
                  f"(σ_pos = [{sigma_d[0]:.3f}, {sigma_d[1]:.3f}, {sigma_d[2]:.3f}] m)")
            print(f"         err   v  = "
                  f"[{err_d[3]:+.9f}, {err_d[4]:+.9f}, {err_d[5]:+.9f}] m/s   "
                  f"|err| = {la.norm(err_d[3:]):.9f} m/s")
            print(f"  Range chief↔deputy (truth) = {range_now:.6f} m   "
                  f"LOS innov = ({los_innov[k,0]*1e6:+.3f}, "
                  f"{los_innov[k,1]*1e6:+.3f}) µrad" if accept_dep[k]
                  else f"  Range chief↔deputy (truth) = {range_now:.6f} m   "
                       f"LOS innov = (REJECTED)")
            print()

    # ── URTSS backward smoother on the deputy state ──
    # Standard Unscented Rauch–Tung–Striebel smoother: at each k,
    #     A_k = P_cross_{k+1} · P_pred_{k+1}^{-1}
    #     x_smooth_k = x_post_k + A_k (x_smooth_{k+1} - x_pred_{k+1})
    #     P_smooth_k = P_post_k + A_k (P_smooth_{k+1} - P_pred_{k+1}) A_k^T
    smooth_x = sm_x_post.copy()
    smooth_P = sm_P_post.copy()
    for k in range(N_STEPS - 2, -1, -1):
        try:
            A = sm_P_cross[k + 1] @ la.inv(sm_P_pred[k + 1] +
                                            1e-12 * np.eye(6))
        except la.LinAlgError:
            continue
        smooth_x[k] = sm_x_post[k] + A @ (smooth_x[k + 1] - sm_x_pred[k + 1])
        smooth_P[k] = sm_P_post[k] + A @ (smooth_P[k + 1] -
                                           sm_P_pred[k + 1]) @ A.T
        smooth_P[k] = 0.5 * (smooth_P[k] + smooth_P[k].T)

    if verbose:
        print("\n" + "=" * 100)
        print("  URTSS SMOOTHER  (forward UKF + backward pass)")
        print("=" * 100)
        i0 = int(0.5 * T_ORBIT / DT_NAV)
        err_filt = est_dep - truth_dep
        err_smth = smooth_x - truth_dep
        rms_filt_pos = np.sqrt(np.mean(np.sum(err_filt[i0:, :3]**2, axis=1)))
        rms_smth_pos = np.sqrt(np.mean(np.sum(err_smth[i0:, :3]**2, axis=1)))
        rms_filt_vel = np.sqrt(np.mean(np.sum(err_filt[i0:, 3:6]**2, axis=1)))
        rms_smth_vel = np.sqrt(np.mean(np.sum(err_smth[i0:, 3:6]**2, axis=1)))
        print(f"  Deputy filter     RMS pos = {rms_filt_pos:14.6f} m   "
              f"RMS vel = {rms_filt_vel:.9f} m/s")
        print(f"  Deputy smoother   RMS pos = {rms_smth_pos:14.6f} m   "
              f"RMS vel = {rms_smth_vel:.9f} m/s")
        if rms_filt_pos > 0:
            print(f"  Smoother improvement (pos): "
                  f"{(1.0 - rms_smth_pos/rms_filt_pos)*100:+.2f} %")
            print(f"  Smoother improvement (vel): "
                  f"{(1.0 - rms_smth_vel/rms_filt_vel)*100:+.2f} %")

    # ── summary ──
    if verbose:
        # transient cutoff: skip first 0.5 orbit
        i0 = int(0.5 * T_ORBIT / DT_NAV)
        print("\n" + "=" * 100)
        print("  RESULTS SUMMARY  (after 0.5-orbit transient)")
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
        smooth_x=smooth_x, smooth_P_diag=np.array([np.diag(p) for p in smooth_P]),
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
    fig.suptitle("CHIEF (GPS-aided) — residuals vs 3σ", fontsize=14)
    for i in range(3):
        _residual_panel(axes[0, i], t_h, err_c[:, i], three_sigma_c[:, i],
                        f"r_{labels[i]} [m]", f"Position {labels[i]}")
    for i in range(3):
        _residual_panel(axes[1, i], t_h, err_c[:, 3+i], three_sigma_c[:, 3+i],
                        f"v_{labels[i]} [m/s]", f"Velocity {labels[i]}")
    axes[1, 0].set_xlabel("Time [h]"); axes[1, 1].set_xlabel("Time [h]")
    axes[1, 2].set_xlabel("Time [h]")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "chief_residuals.png", dpi=150)
    plt.show()

    # ── 2) deputy residuals vs 3σ ──
    err_d = res["est_dep"] - res["truth_dep"]
    three_sigma_d = 3.0 * np.sqrt(np.maximum(res["P_dep"], 0.0))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    fig.suptitle("DEPUTY (angles-only camera, no ranging) — residuals vs 3σ",
                 fontsize=14)
    for i in range(3):
        _residual_panel(axes[0, i], t_h, err_d[:, i], three_sigma_d[:, i],
                        f"r_{labels[i]} [m]", f"Position {labels[i]}")
    for i in range(3):
        _residual_panel(axes[1, i], t_h, err_d[:, 3+i], three_sigma_d[:, 3+i],
                        f"v_{labels[i]} [m/s]", f"Velocity {labels[i]}")
    axes[1, 0].set_xlabel("Time [h]"); axes[1, 1].set_xlabel("Time [h]")
    axes[1, 2].set_xlabel("Time [h]")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "deputy_residuals.png", dpi=150)
    plt.show()

    # ── 3) deputy |error| vs |3σ| (log scale) — convergence curve ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DEPUTY error magnitude convergence (log scale)", fontsize=14)
    err_pos_norm = la.norm(err_d[:, :3], axis=1)
    err_vel_norm = la.norm(err_d[:, 3:6], axis=1)
    sigma_pos_norm = np.sqrt(np.sum(np.maximum(res["P_dep"][:, :3], 0.0), axis=1))
    sigma_vel_norm = np.sqrt(np.sum(np.maximum(res["P_dep"][:, 3:6], 0.0), axis=1))
    ax1.plot(t_h, err_pos_norm, lw=0.7, label="filter |error|", color="C0")
    ax1.plot(t_h, 3 * sigma_pos_norm, "r--", lw=1, label="filter 3σ")
    if "smooth_x" in res:
        err_smooth = res["smooth_x"] - res["truth_dep"]
        sm_norm = la.norm(err_smooth[:, :3], axis=1)
        sm_sigma = np.sqrt(np.sum(np.maximum(res["smooth_P_diag"][:, :3], 0.0), axis=1))
        ax1.plot(t_h, sm_norm, lw=1.0, color="C2", label="smoother |error|")
        ax1.plot(t_h, 3 * sm_sigma, "g--", lw=1, alpha=0.6, label="smoother 3σ")
    ax1.set_yscale("log"); ax1.set_ylabel("|r_error| [m]")
    ax1.set_xlabel("Time [h]"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax1.set_title("Position — filter vs URTSS smoother")
    ax2.plot(t_h, err_vel_norm, lw=0.7, label="filter |error|", color="C0")
    ax2.plot(t_h, 3 * sigma_vel_norm, "r--", lw=1, label="filter 3σ")
    if "smooth_x" in res:
        sm_v = la.norm(err_smooth[:, 3:], axis=1)
        sm_v_sigma = np.sqrt(np.sum(np.maximum(res["smooth_P_diag"][:, 3:], 0.0), axis=1))
        ax2.plot(t_h, sm_v, lw=1.0, color="C2", label="smoother |error|")
        ax2.plot(t_h, 3 * sm_v_sigma, "g--", lw=1, alpha=0.6, label="smoother 3σ")
    ax2.set_yscale("log"); ax2.set_ylabel("|v_error| [m/s]")
    ax2.set_xlabel("Time [h]"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    ax2.set_title("Velocity — filter vs URTSS smoother")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "deputy_convergence.png", dpi=150)
    plt.show()

    # ── 4) LOS innovation ──
    los = res["los_innov"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle("LOS innovation (rad) — angles-only deputy update", fontsize=14)
    axes[0].plot(t_h, los[:, 0] * 1e6, lw=0.5, alpha=0.7)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].axhline(+3 * SIGMA_LOS_EFF * 1e6, color="r", ls="--", lw=1)
    axes[0].axhline(-3 * SIGMA_LOS_EFF * 1e6, color="r", ls="--", lw=1)
    axes[0].set_ylabel("α innovation [µrad]"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_h, los[:, 1] * 1e6, lw=0.5, alpha=0.7)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].axhline(+3 * SIGMA_LOS_EFF * 1e6, color="r", ls="--", lw=1)
    axes[1].axhline(-3 * SIGMA_LOS_EFF * 1e6, color="r", ls="--", lw=1)
    axes[1].set_ylabel("δ innovation [µrad]"); axes[1].set_xlabel("Time [h]")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "los_innov.png", dpi=150)
    plt.show()

    # ── 5) 3-D ECI trajectory: truth (solid) vs estimate (dashed) ──
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("ECI trajectories — truth (solid) vs estimate (dashed)",
                 fontsize=14)
    decim = max(1, len(t) // 4000)
    tc = res["truth_chief"][::decim]; ec = res["est_chief"][::decim]
    td = res["truth_dep"][::decim];   ed = res["est_dep"][::decim]
    ax.plot(tc[:, 0], tc[:, 1], tc[:, 2], "b-",  lw=1.2, label="chief truth")
    ax.plot(ec[:, 0], ec[:, 1], ec[:, 2], "b--", lw=0.8, alpha=0.7,
            label="chief est")
    ax.plot(td[:, 0], td[:, 1], td[:, 2], "g-",  lw=1.2, label="deputy truth")
    ax.plot(ed[:, 0], ed[:, 1], ed[:, 2], "g--", lw=0.8, alpha=0.7,
            label="deputy est")
    ax.scatter(0, 0, 0, color="k", s=140, marker="X", label="Earth centre")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "eci_trajectories.png", dpi=150)
    plt.show()

    # ── 6) Deputy-only zoomed 3-D (relative to chief truth) ──
    rel = res["truth_dep"][:, :3] - res["truth_chief"][:, :3]
    rel_est = res["est_dep"][:, :3] - res["est_chief"][:, :3]
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("DEPUTY relative to CHIEF — truth vs estimate",
                 fontsize=14)
    ax.plot(rel[::decim, 0], rel[::decim, 1], rel[::decim, 2],
            "g-", lw=1.5, label="truth Δr")
    ax.plot(rel_est[::decim, 0], rel_est[::decim, 1], rel_est[::decim, 2],
            "r--", lw=1.0, label="est Δr")
    ax.scatter(0, 0, 0, color="k", s=80, marker="X", label="chief")
    ax.set_xlabel("Δx [m]"); ax.set_ylabel("Δy [m]"); ax.set_zlabel("Δz [m]")
    ax.legend(fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "relative_3d.png", dpi=150)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
#                                  main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.set_printoptions(precision=9, suppress=False, linewidth=200)
    res = run_sim(verbose=True)
    plot_results(res)
