#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_angles_only_relnav_v3_basilisk.py
=========================================

High-fidelity twin of the v2 angles-only relative-navigation experiment.
The TRUTH side is now propagated by Basilisk:

    * Two BSK ``spacecraft.Spacecraft`` objects (chief, deputy) integrated
      synchronously inside one BSK process.
    * Earth gravity via full spherical harmonics (GGM03S, default 70x70)
      — captures J2, J3, J4, … all the way through whatever ``MAX_SH_DEG``
      is set to.  This is a step up from the v2 numpy 2-body+J2 model.
    * Optional Sun and Moon third-body gravity, driven by SPICE
      ephemerides anchored to a real UTC epoch.
    * Optional atmospheric drag via ``exponentialAtmosphere`` +
      ``DragDynamicEffector`` (one effector per spacecraft, both
      subscribed to the same atmosphere model).
    * Both spacecraft share one ``gravBodyFactory`` so the perturbing
      environment is identical — the filter never has to wonder whether
      the deputy is feeling a different planet.

The FILTER side is unchanged from v2:

    Chief filter:    EKF + GPS  (linear position update)
    Deputy filter:   UKF + sigma-point propagation
        per step    1.  IPLF LOS update
                    2.  Chief-as-extended-object angular size update
                    3.  7-D LRL-extended FIMC update (ε, h, e)
        post-pass   4.  URTSS backward smoother

The filter's internal dynamics are still 2-body + J2 (the v2 model).
The TRUTH-vs-FILTER mismatch (J3+, third-body, drag) is absorbed by the
process-noise covariance Q.  This is the realistic case: every operational
GNC system fights its own model error.

Outputs
-------
    Disk:   simple_angles_only_results_v3_basilisk.npz with the full
            per-step time series of truth (from BSK), estimate, P-diag,
            innovations, IPLF iter counts, and smoother.
    Plots:  residuals vs ±3σ for chief and deputy, log-scale convergence
            of filter and smoother, IPLF iteration count, 3-D trajectories,
            relative position residual, and the deputy's relative motion
            in the chief's HILL/LVLH frame (computed via Basilisk's
            ``orbitalMotion.rv2hill``).

Author: Harish Vernekar (2026)
"""

from __future__ import annotations
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Callable

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ──────────────────────────────────────────────────────────────────────────────
#                       Basilisk imports (with friendly fallback)
# ──────────────────────────────────────────────────────────────────────────────

BSK_AVAILABLE = False
try:
    from Basilisk import __path__ as bsk_pkg_path
    from Basilisk.simulation import (
        spacecraft,
        gravityEffector,
    )
    from Basilisk.utilities import (
        SimulationBaseClass,
        macros,
        orbitalMotion as bsk_orbital,
        simIncludeGravBody,
        unitTestSupport,
    )
    # Optional perturbations — wrapped because some BSK builds omit them.
    try:
        from Basilisk.simulation import (
            dragDynamicEffector,
            exponentialAtmosphere,
        )
        from Basilisk.utilities import simSetPlanetEnvironment
        BSK_DRAG_AVAILABLE = True
    except Exception:
        BSK_DRAG_AVAILABLE = False

    BSK_AVAILABLE = True
    BSK_PATH = bsk_pkg_path[0]
except Exception as exc:        # pragma: no cover  (import failure path)
    BSK_AVAILABLE = False
    BSK_DRAG_AVAILABLE = False
    BSK_PATH = ""
    warnings.warn(
        "Basilisk is not importable in this Python environment "
        f"({exc!r}).  Falling back to numpy 2-body+J2 truth — install "
        "Basilisk and re-run for the high-fidelity case.",
        RuntimeWarning,
    )


# ──────────────────────────────────────────────────────────────────────────────
#                              CONSTANTS / CONFIG
# ──────────────────────────────────────────────────────────────────────────────

# Earth gravity
MU_EARTH    = 3.986004418e14    # [m^3 / s^2]
R_EARTH     = 6378.137e3        # [m]

# Orbit
ALTITUDE    = 700e3
R_ORBIT     = R_EARTH + ALTITUDE
V_CIRC      = np.sqrt(MU_EARTH / R_ORBIT)
N_ORBIT     = V_CIRC / R_ORBIT
T_ORBIT     = 2 * np.pi / N_ORBIT

# Formation
BASELINE    = 2000.0

# Run length
DT_NAV         = 1.0                           # [s]   nav step  (= filter rate)
DT_BSK_INT     = 0.5                           # [s]   BSK truth integration step
DURATION_S     = 6 * 3600.0
N_STEPS        = int(round(DURATION_S / DT_NAV)) + 1
N_PERIOD       = int(round(T_ORBIT / DT_NAV))

# Filter side fractional period offset (carried over from v2 — still needed
# if APPLY_PERIODIC is ever turned on).
DT_PERIOD_OFFSET = T_ORBIT - N_PERIOD * DT_NAV   # ≈ 0.379 s
APPLY_PERIODIC   = False

# ── BSK truth-side toggles ────────────────────────────────────────────────────
USE_BASILISK            = BSK_AVAILABLE      # set False to force numpy fallback
MAX_SH_DEG              = 70                 # spherical-harmonics max degree
INCLUDE_THIRD_BODIES    = True               # Sun + Moon as point masses
INCLUDE_DRAG            = True              # exponential atmosphere drag
DRAG_AREA_M2            = 1.0
DRAG_COEFF              = 2.2
SC_MASS_KG              = 100.0
EPOCH_UTC               = "2026 May 09 00:00:00"   # SPICE init epoch
SH_GRAV_FILE            = (
    BSK_PATH + "/supportData/LocalGravData/GGM03S.txt"
    if BSK_AVAILABLE else ""
)

# ── Numpy filter-side gravity (used for the FILTER's internal dynamics) ──────
# Filter knows only 2-body + J2; the gap to BSK truth is the realistic
# unmodelled-physics budget that Q has to absorb.
J2_COEFF    = 1.08262668e-3
INCLUDE_J2_FILTER = True

# Truth integrator sub-stepping (used only by the numpy fallback)
N_RK4_SUBSTEPS = 5

# Chief GPS
SIGMA_GPS_POS = 5.0

# Deputy camera
CAMERA_FOV_DEG       = 27.0
CAMERA_RES_M_AT_2KM  = 0.15
CAMERA_NOMINAL_RANGE = 2000.0
CAMERA_RAW_HZ        = 30.0
SIGMA_LOS_RAW        = CAMERA_RES_M_AT_2KM / CAMERA_NOMINAL_RANGE
SIGMA_LOS_EFF        = SIGMA_LOS_RAW / np.sqrt(CAMERA_RAW_HZ / DT_NAV)

# Initial estimate biases
SIGMA_INIT_POS_DEP  = 50.0
SIGMA_INIT_VEL_DEP  = 0.05
SIGMA_INIT_POS_CHF  = 50.0
SIGMA_INIT_VEL_CHF  = 0.5
INIT_BIAS_POS_DEP   = np.array([20.0, -15.0, 10.0])
INIT_BIAS_VEL_DEP   = np.array([0.01, -0.008, 0.005])

# Process noise — slightly bumped relative to v2 because the high-fidelity
# truth has J3+, third-body, and (optionally) drag perturbations that the
# filter's 2-body+J2 model does not see.
SIGMA_PROC_POS = 0.10           # [m / √s]      (was 0.05 in v2)
SIGMA_PROC_VEL = 1e-3           # [m/s² / √s]   (was 5e-4 in v2)

# NIS gating thresholds
NIS_GATE_GPS_POS = 100.0
NIS_GATE_LOS     = 100.0
NIS_GATE_SIZE    = 100.0
NIS_GATE_FIMC    = 1e6
NIS_GATE_PERIOD  = 1e6

# Chief-as-extended-object measurement
CHIEF_CHORD_M       = 5.0
SIGMA_SIZE_RAD      = 1.5e-5

# 7-D FIMC σ values
SIGMA_FIMC_EPS      = 80.0
SIGMA_FIMC_H        = 2.0e7     # bumped for J2/J3 RAAN precession
SIGMA_FIMC_LRL      = 5.0e-4

# Periodic constraint σ values (only used if APPLY_PERIODIC = True)
SIGMA_PERIOD_POS    = 8.0
SIGMA_PERIOD_VEL    = 0.012

# IPLF iteration parameters
IPLF_MAX_ITER       = 5
IPLF_TOL_POS        = 1e-3
IPLF_TOL_VEL        = 1e-6

# Output cadence
PRINT_EVERY_S = 60.0
PRINT_EVERY_K = int(round(PRINT_EVERY_S / DT_NAV))

OUT_DIR  = Path(__file__).resolve().parent
NPZ_PATH = OUT_DIR / "simple_angles_only_results_v3_basilisk.npz"
SAVE_FIGS = False
FIG_DIR  = OUT_DIR / "simple_angles_only_figs_v3_basilisk"

RNG_SEED = 1234


# ──────────────────────────────────────────────────────────────────────────────
#         FILTER-SIDE dynamics  (2-body + J2 only — the model the filter
#         uses internally; the truth-vs-model gap is what Q absorbs)
# ──────────────────────────────────────────────────────────────────────────────

def two_body_accel(r: np.ndarray) -> np.ndarray:
    rn = la.norm(r)
    return -MU_EARTH * r / (rn ** 3)


def j2_accel(r: np.ndarray) -> np.ndarray:
    """Earth J2 oblateness perturbation in ECI (z = spin axis)."""
    x, y, z = r
    rn = la.norm(r)
    if rn < 1e-6:
        return np.zeros(3)
    rn5 = rn ** 5
    coeff = -1.5 * J2_COEFF * MU_EARTH * (R_EARTH ** 2) / rn5
    s = (z / rn) ** 2
    return coeff * np.array([
        x * (1.0 - 5.0 * s),
        y * (1.0 - 5.0 * s),
        z * (3.0 - 5.0 * s),
    ])


def filter_accel(r: np.ndarray) -> np.ndarray:
    """Acceleration model used by the FILTER (2-body + optional J2)."""
    if INCLUDE_J2_FILTER:
        return two_body_accel(r) + j2_accel(r)
    return two_body_accel(r)


def rk4_step(state6: np.ndarray, dt: float) -> np.ndarray:
    def f(s):
        return np.hstack([s[3:6], filter_accel(s[:3])])
    k1 = f(state6)
    k2 = f(state6 + 0.5 * dt * k1)
    k3 = f(state6 + 0.5 * dt * k2)
    k4 = f(state6 +       dt * k3)
    return state6 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def propagate_filter(state6: np.ndarray, dt_total: float,
                     n_substeps: int = N_RK4_SUBSTEPS) -> np.ndarray:
    """Multi-step RK4 propagation under the FILTER's internal dynamics."""
    dt = dt_total / n_substeps
    s = state6.copy()
    for _ in range(n_substeps):
        s = rk4_step(s, dt)
    return s


def two_body_state_transition(r: np.ndarray, v: np.ndarray,
                              dt: float) -> np.ndarray:
    """ Φ ≈ I + F·dt + ½(F·dt)²   (used only by the chief EKF). """
    rn = la.norm(r)
    rhat = r / rn
    G = -(MU_EARTH / rn**3) * (np.eye(3) - 3.0 * np.outer(rhat, rhat))
    F = np.zeros((6, 6))
    F[0:3, 3:6] = np.eye(3)
    F[3:6, 0:3] = G
    Fdt = F * dt
    return np.eye(6) + Fdt + 0.5 * Fdt @ Fdt


# ──────────────────────────────────────────────────────────────────────────────
#                    Keplerian integrals of motion (FIMC)
# ──────────────────────────────────────────────────────────────────────────────

def specific_energy(r, v):
    return 0.5 * np.dot(v, v) - MU_EARTH / la.norm(r)

def specific_ang_momentum(r, v):
    return np.cross(r, v)

def laplace_runge_lenz(r, v):
    h = np.cross(r, v)
    return np.cross(v, h) / MU_EARTH - r / la.norm(r)

def kepler_invariants(state6: np.ndarray) -> np.ndarray:
    r = state6[:3]; v = state6[3:]
    return np.hstack([[specific_energy(r, v)],
                      specific_ang_momentum(r, v),
                      laplace_runge_lenz(r, v)])


# ──────────────────────────────────────────────────────────────────────────────
#                             measurement models
# ──────────────────────────────────────────────────────────────────────────────

def gps_position_measurement(r_truth, rng):
    return r_truth + SIGMA_GPS_POS * rng.normal(size=3)


def los_angles(r_observer, r_target):
    dr = r_target - r_observer
    dn = la.norm(dr)
    alpha = np.arctan2(dr[1], dr[0])
    delta = np.arcsin(np.clip(dr[2] / dn, -1.0, 1.0))
    return alpha, delta, dr


def los_jacobian_wrt_observer(dr):
    dx, dy, dz = dr
    sxy2 = dx*dx + dy*dy
    n2 = sxy2 + dz*dz
    n  = np.sqrt(n2)
    sxy = np.sqrt(sxy2)
    H = np.zeros((2, 3))
    if sxy2 > 1e-12:
        H[0, :] = -np.array([-dy, dx, 0.0]) / sxy2
    if sxy > 1e-9 and n > 1e-9:
        H[1, :] = -np.array([-dx*dz, -dy*dz, sxy2]) / (sxy * n2)
    return H


def generate_los_measurement(r_dep_truth, r_chief_truth, rng):
    a, d, _ = los_angles(r_dep_truth, r_chief_truth)
    return np.array([
        a + SIGMA_LOS_EFF * rng.normal(),
        d + SIGMA_LOS_EFF * rng.normal(),
    ])


# ──────────────────────────────────────────────────────────────────────────────
#                       EKF (chief, linear GPS update)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EKFConfig:
    sigma_proc_pos: float = SIGMA_PROC_POS
    sigma_proc_vel: float = SIGMA_PROC_VEL
    sigma_init_pos: float = 100.0
    sigma_init_vel: float = 0.5
    use_joseph:     bool  = True


class TwoBodyEKF:
    def __init__(self, name, cfg, x0, P0=None):
        self.name = name; self.cfg = cfg
        self.x = np.asarray(x0, float).reshape(6).copy()
        if P0 is None:
            self.P = np.diag([cfg.sigma_init_pos**2]*3 + [cfg.sigma_init_vel**2]*3)
        else:
            self.P = np.asarray(P0, float).reshape(6, 6).copy()
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        self.t_last = None
        self.n_predict = self.n_update_accept = self.n_update_reject = 0

    def predict(self, t_abs):
        if self.t_last is None: self.t_last = t_abs - DT_NAV
        dt = t_abs - self.t_last
        self.x_pred = propagate_filter(self.x, dt, n_substeps=N_RK4_SUBSTEPS)
        Phi = two_body_state_transition(self.x[:3], self.x[3:], dt)
        qp = (self.cfg.sigma_proc_pos**2)*dt
        qv = (self.cfg.sigma_proc_vel**2)*dt
        Q = np.diag([qp]*3 + [qv]*3)
        self.P_pred = Phi @ self.P @ Phi.T + Q
        self.t_last = t_abs; self.n_predict += 1

    def update(self, z, zhat, H, R, nis_gate, label=""):
        z = np.asarray(z, float).reshape(-1)
        zhat = np.asarray(zhat, float).reshape(-1)
        H = np.asarray(H, float); R = np.asarray(R, float)
        I6 = np.eye(6)
        y = z - zhat
        if label.startswith("LOS"):
            y[0] = (y[0] + np.pi) % (2*np.pi) - np.pi
        S = H @ self.P_pred @ H.T + R
        try: nis = float(y @ la.solve(S, y))
        except la.LinAlgError: nis = float(y @ y)
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
#                  UKF + IPLF (deputy)  — same as v2
# ──────────────────────────────────────────────────────────────────────────────

class TwoBodyUKF_IPLF:
    """UKF (one-shot) + IPLF (iterated posterior linearisation) on top."""

    def __init__(self, name, cfg, x0, P0=None,
                 alpha=1e-3, beta=2.0, kappa=0.0):
        self.name = name; self.cfg = cfg; self.n = 6
        self.x = np.asarray(x0, float).reshape(6).copy()
        if P0 is None:
            self.P = np.diag([cfg.sigma_init_pos**2]*3 + [cfg.sigma_init_vel**2]*3)
        else:
            self.P = np.asarray(P0, float).reshape(6, 6).copy()
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        self.sp_pred = None
        self.P_cross = np.zeros((6, 6))
        self.alpha_ut = alpha; self.beta_ut = beta; self.kappa = kappa
        self.lam = alpha**2 * (self.n + kappa) - self.n
        n2p1 = 2 * self.n + 1
        self.Wm = np.full(n2p1, 1.0 / (2.0 * (self.n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.Wm[0] + (1.0 - alpha**2 + beta)
        self.t_last = None
        self.n_predict = self.n_update_accept = self.n_update_reject = 0
        self.n_iplf_iters = 0

    def _sigma_points(self, x, P):
        scale = (self.n + self.lam)
        try:
            S = np.linalg.cholesky(scale * P)
        except np.linalg.LinAlgError:
            jitter = 1e-9 * (np.max(np.abs(np.diag(P))) + 1.0)
            S = np.linalg.cholesky(scale * (P + jitter * np.eye(self.n)))
        sp = np.zeros((2*self.n + 1, self.n)); sp[0] = x
        for i in range(self.n):
            sp[i+1]          = x + S[:, i]
            sp[i+1+self.n]   = x - S[:, i]
        return sp

    def predict(self, t_abs):
        if self.t_last is None: self.t_last = t_abs - DT_NAV
        dt = t_abs - self.t_last
        sp = self._sigma_points(self.x, self.P)
        sp_pred = np.empty_like(sp)
        for i in range(sp.shape[0]):
            sp_pred[i] = propagate_filter(sp[i], dt, n_substeps=N_RK4_SUBSTEPS)
        x_pred  = (self.Wm[:, None] * sp_pred).sum(axis=0)
        P_pred  = np.zeros((self.n, self.n))
        P_cross = np.zeros((self.n, self.n))
        x_post = self.x.copy()
        for i in range(sp.shape[0]):
            d_pred = sp_pred[i] - x_pred
            P_pred  += self.Wc[i] * np.outer(d_pred, d_pred)
            P_cross += self.Wc[i] * np.outer(sp[i] - x_post, d_pred)
        qp = (self.cfg.sigma_proc_pos**2) * dt
        qv = (self.cfg.sigma_proc_vel**2) * dt
        P_pred += np.diag([qp]*3 + [qv]*3)
        P_pred = 0.5 * (P_pred + P_pred.T)
        self.x_pred = x_pred; self.P_pred = P_pred
        self.sp_pred = sp_pred; self.P_cross = P_cross
        self.t_last = t_abs; self.n_predict += 1

    def update(self, z, h_fn, R, nis_gate=1e10, label="", aux=None):
        """Standard one-shot UKF measurement update."""
        z = np.asarray(z, float).reshape(-1); n_z = z.size
        Z = np.empty((2*self.n + 1, n_z))
        for i in range(2*self.n + 1):
            Z[i] = h_fn(self.sp_pred[i], aux)
        z_pred = (self.Wm[:, None] * Z).sum(axis=0)
        if label.startswith("LOS"):
            for i in range(2*self.n + 1):
                d = Z[i, 0] - z_pred[0]
                if d > np.pi:   Z[i, 0] -= 2*np.pi
                elif d < -np.pi: Z[i, 0] += 2*np.pi
            z_pred = (self.Wm[:, None] * Z).sum(axis=0)
        S = np.array(R, float).copy()
        Pxz = np.zeros((self.n, n_z))
        for i in range(2*self.n + 1):
            dz = Z[i] - z_pred; dx = self.sp_pred[i] - self.x_pred
            S   += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)
        y = z - z_pred
        if label.startswith("LOS"):
            y[0] = (y[0] + np.pi) % (2*np.pi) - np.pi
        try: nis = float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError: nis = float(y @ y)
        if nis > nis_gate:
            self.x = self.x_pred.copy(); self.P = self.P_pred.copy()
            self.n_update_reject += 1
            return dict(accepted=False, nis=nis, y=y, S=S)
        try: K = Pxz @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.x = self.x_pred.copy(); self.P = self.P_pred.copy()
            self.n_update_reject += 1
            return dict(accepted=False, nis=nis, y=y, S=S)
        self.x = self.x_pred + K @ y
        self.P = self.P_pred - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        self.sp_pred = self._sigma_points(self.x, self.P)
        self.n_update_accept += 1
        return dict(accepted=True, nis=nis, y=y, S=S)

    def update_iplf(self, z, h_fn, R, nis_gate=1e10, label="", aux=None,
                    max_iter=IPLF_MAX_ITER,
                    tol_pos=IPLF_TOL_POS, tol_vel=IPLF_TOL_VEL):
        """Iterated Posterior Linearisation Filter measurement update."""
        z = np.asarray(z, float).reshape(-1); n_z = z.size
        I6 = np.eye(self.n)
        x_iter = self.x_pred.copy(); P_iter = self.P_pred.copy()

        def _wrap_minus(zz, zp):
            d = zz - zp
            if label.startswith("LOS"):
                d[0] = (d[0] + np.pi) % (2*np.pi) - np.pi
            return d

        x_new = x_iter; P_new = P_iter
        last_S = R.copy(); last_y = np.zeros(n_z); last_nis = 0.0
        n_iter_done = 0
        for it in range(max_iter):
            n_iter_done = it + 1
            sp_iter = self._sigma_points(x_iter, P_iter)
            Z = np.empty((2*self.n + 1, n_z))
            for i in range(2*self.n + 1):
                Z[i] = h_fn(sp_iter[i], aux)
            z_pred = (self.Wm[:, None] * Z).sum(axis=0)
            if label.startswith("LOS"):
                for i in range(2*self.n + 1):
                    d = Z[i, 0] - z_pred[0]
                    if d > np.pi:   Z[i, 0] -= 2*np.pi
                    elif d < -np.pi: Z[i, 0] += 2*np.pi
                z_pred = (self.Wm[:, None] * Z).sum(axis=0)
            Pzz = np.array(R, float).copy()
            Pxz = np.zeros((self.n, n_z))
            for i in range(2*self.n + 1):
                dz = Z[i] - z_pred; dx = sp_iter[i] - x_iter
                Pzz += self.Wc[i] * np.outer(dz, dz)
                Pxz += self.Wc[i] * np.outer(dx, dz)
            try:
                P_iter_inv = np.linalg.inv(P_iter + 1e-12*np.eye(self.n))
            except np.linalg.LinAlgError:
                break
            A = Pxz.T @ P_iter_inv
            b = z_pred - A @ x_iter
            Omega = Pzz - A @ P_iter @ A.T
            Omega = 0.5 * (Omega + Omega.T)
            S = Omega + A @ self.P_pred @ A.T
            S = 0.5 * (S + S.T)
            try: K = self.P_pred @ A.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError: break
            innov = _wrap_minus(z, b + A @ self.x_pred)
            x_new = self.x_pred + K @ innov
            KA = K @ A
            P_new = ((I6 - KA) @ self.P_pred @ (I6 - KA).T
                     + K @ Omega @ K.T)
            P_new = 0.5 * (P_new + P_new.T)
            try: last_nis = float(innov @ np.linalg.solve(S, innov))
            except np.linalg.LinAlgError: last_nis = float(innov @ innov)
            last_S = S; last_y = innov
            d_pos = la.norm(x_new[:3] - x_iter[:3])
            d_vel = la.norm(x_new[3:] - x_iter[3:])
            x_iter, P_iter = x_new, P_new
            if d_pos < tol_pos and d_vel < tol_vel:
                break
        self.n_iplf_iters += n_iter_done
        if last_nis > nis_gate:
            self.x = self.x_pred.copy(); self.P = self.P_pred.copy()
            self.n_update_reject += 1
            return dict(accepted=False, nis=last_nis, y=last_y, S=last_S,
                        n_iter=n_iter_done)
        self.x = x_new; self.P = P_new
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        self.sp_pred = self._sigma_points(self.x, self.P)
        self.n_update_accept += 1
        return dict(accepted=True, nis=last_nis, y=last_y, S=last_S,
                    n_iter=n_iter_done)


# ──────────────────────────────────────────────────────────────────────────────
#                          truth ICs (numpy + BSK share)
# ──────────────────────────────────────────────────────────────────────────────

def initial_chief_state() -> np.ndarray:
    return np.array([R_ORBIT, 0.0, 0.0,    0.0, V_CIRC, 0.0])


def initial_deputy_state() -> np.ndarray:
    """Deputy on a 3-D Hill ellipse — A = B = BASELINE = 2 km."""
    A   = BASELINE; B = BASELINE
    rho_H    = np.array([-0.5 * A,  0.0,       B])
    rhodot_H = np.array([ 0.0,      N_ORBIT*A, 0.0])
    omega = np.array([0.0, 0.0, N_ORBIT])
    chief = initial_chief_state()
    r_dep = chief[:3] + rho_H
    v_dep = chief[3:] + rhodot_H + np.cross(omega, rho_H)
    return np.hstack([r_dep, v_dep])


# ──────────────────────────────────────────────────────────────────────────────
#                BASILISK truth-side propagation
# ──────────────────────────────────────────────────────────────────────────────

def run_basilisk_truth(verbose: bool = True):
    """
    Build a full-physics Basilisk simulation of the chief and deputy and
    return the per-nav-step ECI truth time series.

    Returns
    -------
    t_arr        : (N_STEPS,) array of seconds since start
    truth_chief  : (N_STEPS, 6) ECI position+velocity time series
    truth_dep    : (N_STEPS, 6) ECI position+velocity time series
    bsk_meta     : dict with diagnostic info (gravity model, perturbations)
    """
    if not USE_BASILISK:
        raise RuntimeError("Basilisk not available — caller should fall back")

    # ── simulation skeleton ──
    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(True)
    procName = "relnavTruthProc"
    taskName = "relnavTruthTask"
    dynProcess = scSim.CreateNewProcess(procName)
    dt_int_ns = macros.sec2nano(DT_BSK_INT)
    dynProcess.addTask(scSim.CreateNewTask(taskName, dt_int_ns))

    # ── two spacecraft ──
    scChief = spacecraft.Spacecraft(); scChief.ModelTag  = "Chief"
    scDeputy = spacecraft.Spacecraft(); scDeputy.ModelTag = "Deputy"
    scChief.hub.mHub  = SC_MASS_KG
    scDeputy.hub.mHub = SC_MASS_KG
    # share the same ODE integrator (lock-step propagation)
    scChief.syncDynamicsIntegration(scDeputy)
    scSim.AddModelToTask(taskName, scChief)
    scSim.AddModelToTask(taskName, scDeputy)

    # ── gravity ──
    gravFactory = simIncludeGravBody.gravBodyFactory()
    body_list = ["earth"]
    if INCLUDE_THIRD_BODIES:
        body_list += ["sun", "moon"]
    gravBodies = gravFactory.createBodies(*body_list)
    gravBodies["earth"].isCentralBody = True

    sh_loaded = False
    try:
        gravBodies["earth"].useSphericalHarmonicsGravityModel(
            SH_GRAV_FILE, MAX_SH_DEG)
        sh_loaded = True
    except Exception as exc:
        warnings.warn(
            f"Could not load spherical harmonics from {SH_GRAV_FILE!r} "
            f"({exc!r}); falling back to point-mass gravity for Earth.",
            RuntimeWarning)

    gravFactory.addBodiesTo(scChief)
    gravFactory.addBodiesTo(scDeputy)

    # ── SPICE ephemeris (only useful if we asked for third bodies) ──
    spice_enabled = False
    if INCLUDE_THIRD_BODIES:
        try:
            spiceObj = gravFactory.createSpiceInterface(
                time=EPOCH_UTC, epochInMsg=True)
            scSim.AddModelToTask(taskName, spiceObj)
            # Anchor the inertial frame at Earth so that scStateOutMsg
            # gives us positions w.r.t. Earth, matching our v2 convention.
            spiceObj.zeroBase = "earth"
            spice_enabled = True
        except Exception as exc:
            warnings.warn(
                "Could not initialise SPICE interface "
                f"({exc!r}); third bodies will sit at zero — disable "
                "INCLUDE_THIRD_BODIES if this is undesired.",
                RuntimeWarning)

    # ── atmospheric drag ──
    drag_enabled = False
    if INCLUDE_DRAG and BSK_DRAG_AVAILABLE:
        atmo = exponentialAtmosphere.ExponentialAtmosphere()
        atmo.ModelTag = "ExpAtmo"
        try:
            simSetPlanetEnvironment.exponentialAtmosphere(atmo, "earth")
        except Exception:
            # Fall back to manually setting the canonical Earth params.
            atmo.planetRadius = R_EARTH
            atmo.baseDensity  = 1.217   # kg/m³ at sea level
            atmo.scaleHeight  = 8500.0  # m
        scSim.AddModelToTask(taskName, atmo)
        atmo.addSpacecraftToModel(scChief.scStateOutMsg)
        atmo.addSpacecraftToModel(scDeputy.scStateOutMsg)

        dragChief = dragDynamicEffector.DragDynamicEffector()
        dragChief.ModelTag = "DragChief"
        dragChief.coreParams.projectedArea = DRAG_AREA_M2
        dragChief.coreParams.dragCoeff     = DRAG_COEFF
        scChief.addDynamicEffector(dragChief)
        dragChief.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[0])
        scSim.AddModelToTask(taskName, dragChief)

        dragDeputy = dragDynamicEffector.DragDynamicEffector()
        dragDeputy.ModelTag = "DragDeputy"
        dragDeputy.coreParams.projectedArea = DRAG_AREA_M2
        dragDeputy.coreParams.dragCoeff     = DRAG_COEFF
        scDeputy.addDynamicEffector(dragDeputy)
        dragDeputy.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[1])
        scSim.AddModelToTask(taskName, dragDeputy)
        drag_enabled = True

    # ── initial conditions: identical to the v2 numpy run ──
    x_chief_0 = initial_chief_state()
    x_dep_0   = initial_deputy_state()
    scChief.hub.r_CN_NInit  = x_chief_0[:3].reshape(3, 1).tolist()
    scChief.hub.v_CN_NInit  = x_chief_0[3:].reshape(3, 1).tolist()
    scDeputy.hub.r_CN_NInit = x_dep_0[:3].reshape(3, 1).tolist()
    scDeputy.hub.v_CN_NInit = x_dep_0[3:].reshape(3, 1).tolist()

    # ── recorders sampled at the nav rate ──
    samp_ns = macros.sec2nano(DT_NAV)
    chiefRec  = scChief.scStateOutMsg.recorder(samp_ns)
    deputyRec = scDeputy.scStateOutMsg.recorder(samp_ns)
    scSim.AddModelToTask(taskName, chiefRec)
    scSim.AddModelToTask(taskName, deputyRec)

    # ── run ──
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(macros.sec2nano(DURATION_S))
    if verbose:
        print(f"[BSK] propagating {DURATION_S/3600:.2f} h "
              f"@ dt_int = {DT_BSK_INT:.3f} s "
              f"(SH deg = {MAX_SH_DEG if sh_loaded else 'point-mass'}, "
              f"third-bodies = {spice_enabled}, drag = {drag_enabled}) …")
    scSim.ExecuteSimulation()

    # Unload SPICE kernels to leave the process clean for any caller
    # who runs multiple BSK simulations back-to-back.
    try:
        gravFactory.unloadSpiceKernels()
    except Exception:
        pass

    # ── extract + align to N_STEPS ──
    pos_c = np.asarray(chiefRec.r_BN_N)
    vel_c = np.asarray(chiefRec.v_BN_N)
    pos_d = np.asarray(deputyRec.r_BN_N)
    vel_d = np.asarray(deputyRec.v_BN_N)
    times = np.asarray(chiefRec.times()) * macros.NANO2SEC
    # Trim/pad to exactly N_STEPS (BSK returns one sample at t=0 and one
    # per sampling period thereafter; with DT_NAV sampling we expect
    # N_STEPS exactly, but guard against off-by-one).
    n_avail = min(len(pos_c), len(pos_d), len(times), N_STEPS)
    truth_chief = np.zeros((N_STEPS, 6))
    truth_dep   = np.zeros((N_STEPS, 6))
    t_arr       = np.linspace(0.0, (N_STEPS - 1) * DT_NAV, N_STEPS)
    truth_chief[:n_avail] = np.hstack([pos_c[:n_avail], vel_c[:n_avail]])
    truth_dep[:n_avail]   = np.hstack([pos_d[:n_avail], vel_d[:n_avail]])
    if n_avail < N_STEPS:
        # If BSK gave fewer samples (e.g. duration not exactly an
        # integer number of nav steps), hold the last value.
        truth_chief[n_avail:] = truth_chief[n_avail - 1]
        truth_dep[n_avail:]   = truth_dep[n_avail - 1]

    bsk_meta = dict(
        sh_loaded=sh_loaded, sh_deg=(MAX_SH_DEG if sh_loaded else 0),
        spice_enabled=spice_enabled, drag_enabled=drag_enabled,
        epoch_utc=EPOCH_UTC,
    )
    return t_arr, truth_chief, truth_dep, bsk_meta


# ──────────────────────────────────────────────────────────────────────────────
#                Numpy fallback truth (2-body + J2)  — same as v2
# ──────────────────────────────────────────────────────────────────────────────

def run_numpy_truth(verbose: bool = True):
    """Fallback truth propagator using filter-side dynamics (2-body + J2).
    Useful as a sanity baseline when Basilisk is not installed."""
    if verbose:
        print("[NUMPY-FALLBACK] propagating truth with 2-body + J2 RK4 …")
    t_arr = np.linspace(0.0, (N_STEPS - 1) * DT_NAV, N_STEPS)
    truth_chief = np.zeros((N_STEPS, 6))
    truth_dep   = np.zeros((N_STEPS, 6))
    truth_chief[0] = initial_chief_state()
    truth_dep[0]   = initial_deputy_state()
    for k in range(1, N_STEPS):
        truth_chief[k] = propagate_filter(truth_chief[k-1], DT_NAV)
        truth_dep[k]   = propagate_filter(truth_dep[k-1],   DT_NAV)
    bsk_meta = dict(sh_loaded=False, sh_deg=0, spice_enabled=False,
                    drag_enabled=False, epoch_utc="")
    return t_arr, truth_chief, truth_dep, bsk_meta


# ──────────────────────────────────────────────────────────────────────────────
#                              MAIN simulation
# ──────────────────────────────────────────────────────────────────────────────

def run_sim(verbose: bool = True):
    rng = np.random.default_rng(RNG_SEED)

    # ── propagate truth (BSK if available, numpy otherwise) ──
    if USE_BASILISK:
        t_arr, truth_chief, truth_dep, bsk_meta = run_basilisk_truth(verbose)
    else:
        t_arr, truth_chief, truth_dep, bsk_meta = run_numpy_truth(verbose)

    # ── filter setup ──
    chief_cfg = EKFConfig(sigma_init_pos=SIGMA_INIT_POS_CHF,
                          sigma_init_vel=SIGMA_INIT_VEL_CHF)
    dep_cfg   = EKFConfig(sigma_init_pos=SIGMA_INIT_POS_DEP,
                          sigma_init_vel=SIGMA_INIT_VEL_DEP)

    x_chief_truth_0 = truth_chief[0]
    x_dep_truth_0   = truth_dep[0]
    x0_chief = x_chief_truth_0 + np.hstack([
        SIGMA_INIT_POS_CHF * rng.normal(size=3),
        SIGMA_INIT_VEL_CHF * rng.normal(size=3),
    ])
    x0_dep = x_dep_truth_0 + np.hstack([INIT_BIAS_POS_DEP, INIT_BIAS_VEL_DEP])

    chief_ekf = TwoBodyEKF("CHIEF", chief_cfg, x0_chief)
    dep_ekf   = TwoBodyUKF_IPLF("DEPUTY", dep_cfg, x0_dep)

    # ── histories ──
    est_chief = np.zeros((N_STEPS, 6)); est_dep = np.zeros((N_STEPS, 6))
    P_chief   = np.zeros((N_STEPS, 6)); P_dep   = np.zeros((N_STEPS, 6))
    gps_innov = np.full((N_STEPS, 3), np.nan)
    los_innov = np.full((N_STEPS, 2), np.nan)
    los_alpha = np.full(N_STEPS, np.nan); los_delta = np.full(N_STEPS, np.nan)
    los_iters = np.zeros(N_STEPS, dtype=int)
    accept_chief = np.zeros(N_STEPS, dtype=bool)
    accept_dep   = np.zeros(N_STEPS, dtype=bool)
    period_active = np.zeros(N_STEPS, dtype=bool)

    sm_x_post = np.zeros((N_STEPS, 6)); sm_P_post = np.zeros((N_STEPS, 6, 6))
    sm_x_pred = np.zeros((N_STEPS, 6)); sm_P_pred = np.zeros((N_STEPS, 6, 6))
    sm_P_cross = np.zeros((N_STEPS, 6, 6))
    periodic_buf = np.zeros((N_PERIOD, 6))

    est_chief[0] = chief_ekf.x; est_dep[0] = dep_ekf.x
    P_chief[0]   = np.diag(chief_ekf.P); P_dep[0] = np.diag(dep_ekf.P)
    sm_x_post[0] = dep_ekf.x.copy(); sm_P_post[0] = dep_ekf.P.copy()
    sm_x_pred[0] = dep_ekf.x.copy(); sm_P_pred[0] = dep_ekf.P.copy()
    sm_P_cross[0] = np.eye(6) * 1e-9
    periodic_buf[0] = dep_ekf.x.copy()

    if verbose:
        print("=" * 100)
        print("  ANGLES-ONLY RELNAV  v3 :  Basilisk truth + LRL-FIMC + IPLF")
        print("=" * 100)
        if USE_BASILISK:
            print(f"  Truth source      : Basilisk")
            print(f"    Spherical harm.  : {'GGM03S deg ' + str(bsk_meta['sh_deg']) if bsk_meta['sh_loaded'] else 'point-mass'}")
            print(f"    Third bodies     : {'Sun + Moon (SPICE)' if bsk_meta['spice_enabled'] else 'off'}")
            print(f"    Atmospheric drag : {'on (Cd='+str(DRAG_COEFF)+', A='+str(DRAG_AREA_M2)+' m²)' if bsk_meta['drag_enabled'] else 'off'}")
            print(f"    Epoch (UTC)      : {bsk_meta['epoch_utc']}")
        else:
            print(f"  Truth source      : numpy fallback (2-body + J2)")
        print(f"  Filter model      : 2-body + J2 (Q absorbs the gap)")
        print(f"  Periodicity ON?   : {APPLY_PERIODIC}")
        print(f"  Chief altitude    : {ALTITUDE/1e3:.1f} km")
        print(f"  Orbit period      : {T_ORBIT:.3f} s ({T_ORBIT/60:.3f} min)")
        print(f"  Run length        : {DURATION_S/3600:.2f} h "
              f"({DURATION_S/T_ORBIT:.2f} orbits, {N_STEPS} nav steps)")
        print(f"  σ_LOS effective   : {SIGMA_LOS_EFF:.3e} rad ({SIGMA_LOS_EFF*1e6:.3f} µrad)")
        print(f"  σ_FIMC_eps / h / LRL : {SIGMA_FIMC_EPS:.1f} / "
              f"{SIGMA_FIMC_H:.2e} / {SIGMA_FIMC_LRL:.2e}")
        print(f"  σ_proc_pos / vel  : {SIGMA_PROC_POS:.3f} / {SIGMA_PROC_VEL:.4f}")
        print(f"  IPLF max iters    : {IPLF_MAX_ITER}")
        print()

    # ── main loop ──
    for k in range(1, N_STEPS):
        t = t_arr[k]
        x_chief_truth = truth_chief[k]
        x_dep_truth   = truth_dep[k]

        chief_ekf.predict(t)
        dep_ekf.predict(t)

        sm_x_pred[k]  = dep_ekf.x_pred.copy()
        sm_P_pred[k]  = dep_ekf.P_pred.copy()
        sm_P_cross[k] = dep_ekf.P_cross.copy()

        # Chief: GPS update
        z_gps = gps_position_measurement(x_chief_truth[:3], rng)
        H_gps = np.zeros((3, 6)); H_gps[:, :3] = np.eye(3)
        R_gps = (SIGMA_GPS_POS**2) * np.eye(3)
        zhat_gps = chief_ekf.x_pred[:3]
        info = chief_ekf.update(z_gps, zhat_gps, H_gps, R_gps,
                                NIS_GATE_GPS_POS, label="GPS")
        accept_chief[k] = info["accepted"]
        gps_innov[k] = info["y"] if info["accepted"] else np.nan

        # Deputy: LOS update via IPLF
        z_los = generate_los_measurement(x_dep_truth[:3],
                                          x_chief_truth[:3], rng)
        los_alpha[k], los_delta[k] = z_los

        r_chief_est = chief_ekf.x[:3]
        _, _, dr_pred = los_angles(dep_ekf.x_pred[:3], r_chief_est)
        H_chief_pos = -los_jacobian_wrt_observer(dr_pred)
        R_consider  = H_chief_pos @ chief_ekf.P[:3, :3] @ H_chief_pos.T
        R_los       = (SIGMA_LOS_EFF**2) * np.eye(2) + R_consider

        def _los_h(x_dep, r_chief):
            a, d, _ = los_angles(x_dep[:3], r_chief)
            return np.array([a, d])

        info = dep_ekf.update_iplf(z_los, _los_h, R_los,
                                    nis_gate=NIS_GATE_LOS, label="LOS",
                                    aux=r_chief_est)
        accept_dep[k] = info["accepted"]
        los_innov[k]  = info["y"] if info["accepted"] else np.nan
        los_iters[k]  = info.get("n_iter", 0)

        # Chief-as-extended-object
        true_size = CHIEF_CHORD_M / la.norm(x_chief_truth[:3] - x_dep_truth[:3])
        z_size = np.array([true_size + SIGMA_SIZE_RAD * rng.normal()])
        R_size = np.array([[SIGMA_SIZE_RAD**2 +
                            (CHIEF_CHORD_M / la.norm(dr_pred)**2)**2 *
                            np.trace(chief_ekf.P[:3, :3]) / 3]])

        def _size_h(x_dep, r_chief):
            return np.array([CHIEF_CHORD_M / la.norm(r_chief - x_dep[:3])])

        dep_ekf.update(z_size, _size_h, R_size,
                       nis_gate=NIS_GATE_SIZE, label="SIZE",
                       aux=r_chief_est)

        # 7-D LRL-FIMC vs chief
        z_fimc = kepler_invariants(chief_ekf.x)
        R_fimc = np.diag([
            SIGMA_FIMC_EPS**2,
            SIGMA_FIMC_H**2, SIGMA_FIMC_H**2, SIGMA_FIMC_H**2,
            SIGMA_FIMC_LRL**2, SIGMA_FIMC_LRL**2, SIGMA_FIMC_LRL**2,
        ])

        def _fimc_h(x_dep, _aux):
            return kepler_invariants(x_dep)

        dep_ekf.update(z_fimc, _fimc_h, R_fimc,
                       nis_gate=NIS_GATE_FIMC, label="FIMC", aux=None)

        # Optional periodicity
        if APPLY_PERIODIC and k >= N_PERIOD:
            x_old = periodic_buf[k % N_PERIOD]
            if abs(DT_PERIOD_OFFSET) > 1e-9:
                x_period = propagate_filter(x_old, -DT_PERIOD_OFFSET,
                                             n_substeps=max(1, N_RK4_SUBSTEPS // 5))
            else:
                x_period = x_old.copy()
            R_period = np.diag([SIGMA_PERIOD_POS**2]*3 +
                                [SIGMA_PERIOD_VEL**2]*3)

            def _period_h(x_dep, _aux):
                return x_dep.copy()

            dep_ekf.update(x_period, _period_h, R_period,
                           nis_gate=NIS_GATE_PERIOD, label="PERIOD", aux=None)
            period_active[k] = True

        periodic_buf[k % N_PERIOD] = dep_ekf.x.copy()
        sm_x_post[k] = dep_ekf.x.copy()
        sm_P_post[k] = dep_ekf.P.copy()

        est_chief[k] = chief_ekf.x; est_dep[k] = dep_ekf.x
        P_chief[k]   = np.diag(chief_ekf.P); P_dep[k] = np.diag(dep_ekf.P)

        if verbose and (k % PRINT_EVERY_K == 0 or k == N_STEPS - 1):
            err_c = est_chief[k] - truth_chief[k]
            err_d = est_dep[k]   - truth_dep[k]
            sigma_d = np.sqrt(P_dep[k])
            range_now = la.norm(truth_dep[k, :3] - truth_chief[k, :3])
            print(f"t = {t:11.3f} s ({t/T_ORBIT:.3f} orb)")
            print(f"  CHIEF  |Δr| = {la.norm(err_c[:3]):8.3f} m   "
                  f"|Δv| = {la.norm(err_c[3:6])*1e3:8.3f} mm/s")
            print(f"  DEPUTY |Δr| = {la.norm(err_d[:3]):8.3f} m   "
                  f"|Δv| = {la.norm(err_d[3:6])*1e3:8.3f} mm/s   "
                  f"σ_pos = [{sigma_d[0]:.3f}, {sigma_d[1]:.3f}, "
                  f"{sigma_d[2]:.3f}]   IPLF iters = {los_iters[k]}")
            print(f"  Range chief↔deputy (truth) = {range_now:.3f} m")
            print()

    # ── URTSS smoother ──
    smooth_x = sm_x_post.copy(); smooth_P = sm_P_post.copy()
    for k in range(N_STEPS - 2, -1, -1):
        try:
            A = sm_P_cross[k+1] @ la.inv(sm_P_pred[k+1] + 1e-12*np.eye(6))
        except la.LinAlgError:
            continue
        smooth_x[k] = sm_x_post[k] + A @ (smooth_x[k+1] - sm_x_pred[k+1])
        smooth_P[k] = sm_P_post[k] + A @ (smooth_P[k+1] - sm_P_pred[k+1]) @ A.T
        smooth_P[k] = 0.5 * (smooth_P[k] + smooth_P[k].T)

    # ── summary ──
    if verbose:
        i0 = int(0.5 * T_ORBIT / DT_NAV)
        print("\n" + "=" * 100)
        print("  RESULTS SUMMARY  v3-basilisk  (after 0.5-orbit transient)")
        print("=" * 100)
        for ekf, truth, est, P_diag, name in [
            (chief_ekf, truth_chief, est_chief, P_chief, "CHIEF"),
            (dep_ekf,   truth_dep,   est_dep,   P_dep,   "DEPUTY"),
        ]:
            err = est - truth
            rms_pos = np.sqrt(np.mean(np.sum(err[i0:, :3]**2, axis=1)))
            rms_vel = np.sqrt(np.mean(np.sum(err[i0:, 3:6]**2, axis=1)))
            print(f"\n  {name}")
            print(f"    n_predict       : {ekf.n_predict}")
            print(f"    n_update_accept : {ekf.n_update_accept}")
            print(f"    n_update_reject : {ekf.n_update_reject}")
            if hasattr(ekf, "n_iplf_iters"):
                avg = ekf.n_iplf_iters / max(1, np.count_nonzero(los_iters))
                print(f"    avg IPLF iters  : {avg:.2f}")
            print(f"    RMS pos (3-D)   : {rms_pos:.6f} m")
            print(f"    RMS vel (3-D)   : {rms_vel:.9f} m/s")

        err_filt = est_dep - truth_dep
        err_smth = smooth_x - truth_dep
        rms_filt_pos = np.sqrt(np.mean(np.sum(err_filt[i0:, :3]**2, axis=1)))
        rms_smth_pos = np.sqrt(np.mean(np.sum(err_smth[i0:, :3]**2, axis=1)))
        rms_filt_vel = np.sqrt(np.mean(np.sum(err_filt[i0:, 3:6]**2, axis=1)))
        rms_smth_vel = np.sqrt(np.mean(np.sum(err_smth[i0:, 3:6]**2, axis=1)))
        print("\n  URTSS SMOOTHER")
        print(f"    Deputy filter   RMS pos = {rms_filt_pos:.6f} m   "
              f"vel = {rms_filt_vel:.9f} m/s")
        print(f"    Deputy smoother RMS pos = {rms_smth_pos:.6f} m   "
              f"vel = {rms_smth_vel:.9f} m/s")
        if rms_filt_pos > 0:
            print(f"    Smoother improvement: pos {(1.0 - rms_smth_pos/rms_filt_pos)*100:+.2f} %, "
                  f"vel {(1.0 - rms_smth_vel/rms_filt_vel)*100:+.2f} %")

    np.savez(NPZ_PATH,
             t=t_arr,
             truth_chief=truth_chief, est_chief=est_chief, P_chief_diag=P_chief,
             truth_dep=truth_dep,     est_dep=est_dep,     P_dep_diag=P_dep,
             gps_innov=gps_innov, los_innov=los_innov,
             los_alpha=los_alpha, los_delta=los_delta, los_iters=los_iters,
             accept_chief=accept_chief, accept_dep=accept_dep,
             period_active=period_active,
             smooth_x=smooth_x,
             smooth_P_diag=np.array([np.diag(p) for p in smooth_P]),
             bsk_meta=np.array([str(bsk_meta)], dtype=object))
    if verbose:
        print(f"\n  Saved per-step time series to {NPZ_PATH}\n")

    return dict(
        t=t_arr,
        truth_chief=truth_chief, est_chief=est_chief, P_chief=P_chief,
        truth_dep=truth_dep,     est_dep=est_dep,     P_dep=P_dep,
        smooth_x=smooth_x,
        smooth_P_diag=np.array([np.diag(p) for p in smooth_P]),
        gps_innov=gps_innov, los_innov=los_innov, los_iters=los_iters,
        accept_chief=accept_chief, accept_dep=accept_dep,
        period_active=period_active,
        bsk_meta=bsk_meta,
    )


# ──────────────────────────────────────────────────────────────────────────────
#                                 plotting
# ──────────────────────────────────────────────────────────────────────────────

def _residual_panel(ax, t_h, err, three_sigma, ylabel, title):
    ax.plot(t_h, err, lw=0.6, alpha=0.8)
    ax.plot(t_h,  three_sigma, "r--", lw=1.0, alpha=0.7)
    ax.plot(t_h, -three_sigma, "r--", lw=1.0, alpha=0.7)
    ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3); ax.set_title(title)


def _hill_relative_state(truth_chief, truth_dep):
    """
    Compute the deputy's relative state in the chief's HILL/LVLH frame.
    Falls back to a manual implementation if Basilisk isn't importable.
    """
    rho_H = np.zeros((len(truth_chief), 3))
    if BSK_AVAILABLE:
        for k in range(len(truth_chief)):
            rho, _ = bsk_orbital.rv2hill(
                truth_chief[k, :3], truth_chief[k, 3:],
                truth_dep[k, :3],   truth_dep[k, 3:])
            rho_H[k] = np.asarray(rho).reshape(3)
    else:
        # Manual Hill DCM
        for k in range(len(truth_chief)):
            r_c = truth_chief[k, :3]; v_c = truth_chief[k, 3:]
            ir = r_c / la.norm(r_c)
            h  = np.cross(r_c, v_c)
            ih = h / la.norm(h)
            it = np.cross(ih, ir)
            HN = np.array([ir, it, ih])
            rho_H[k] = HN @ (truth_dep[k, :3] - r_c)
    return rho_H


def plot_results(res):
    if SAVE_FIGS:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
    t = res["t"]; t_h = t / 3600.0
    labels = ["x", "y", "z"]

    err_c = res["est_chief"] - res["truth_chief"]
    err_d = res["est_dep"]   - res["truth_dep"]
    three_sigma_c = 3.0 * np.sqrt(np.maximum(res["P_chief"], 0.0))
    three_sigma_d = 3.0 * np.sqrt(np.maximum(res["P_dep"], 0.0))

    # ── 1) chief residuals ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    fig.suptitle("CHIEF residuals — v3 (BSK truth)", fontsize=14)
    for i in range(3):
        _residual_panel(axes[0, i], t_h, err_c[:, i], three_sigma_c[:, i],
                        f"r_{labels[i]} [m]", f"Position {labels[i]}")
        _residual_panel(axes[1, i], t_h, err_c[:, 3+i], three_sigma_c[:, 3+i],
                        f"v_{labels[i]} [m/s]", f"Velocity {labels[i]}")
    for ax in axes[1]: ax.set_xlabel("Time [h]")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "chief_residuals.png", dpi=150)

    # ── 2) deputy residuals ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    fig.suptitle("DEPUTY residuals — v3 (BSK truth, LRL-FIMC + IPLF)",
                 fontsize=14)
    for i in range(3):
        _residual_panel(axes[0, i], t_h, err_d[:, i], three_sigma_d[:, i],
                        f"r_{labels[i]} [m]", f"Position {labels[i]}")
        _residual_panel(axes[1, i], t_h, err_d[:, 3+i], three_sigma_d[:, 3+i],
                        f"v_{labels[i]} [m/s]", f"Velocity {labels[i]}")
    for ax in axes[1]: ax.set_xlabel("Time [h]")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "deputy_residuals.png", dpi=150)

    # ── 3) deputy convergence (filter vs smoother) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DEPUTY error magnitude — v3 (BSK truth)", fontsize=14)
    err_pos_norm = la.norm(err_d[:, :3], axis=1)
    err_vel_norm = la.norm(err_d[:, 3:6], axis=1)
    sigma_pos_norm = np.sqrt(np.sum(np.maximum(res["P_dep"][:, :3], 0.0), axis=1))
    sigma_vel_norm = np.sqrt(np.sum(np.maximum(res["P_dep"][:, 3:6], 0.0), axis=1))
    err_smooth = res["smooth_x"] - res["truth_dep"]
    sm_norm   = la.norm(err_smooth[:, :3], axis=1)
    sm_v      = la.norm(err_smooth[:, 3:], axis=1)
    sm_sigma  = np.sqrt(np.sum(np.maximum(res["smooth_P_diag"][:, :3], 0.0), axis=1))
    sm_v_sig  = np.sqrt(np.sum(np.maximum(res["smooth_P_diag"][:, 3:], 0.0), axis=1))

    ax1.plot(t_h, err_pos_norm,     lw=0.7, label="filter |error|", color="C0")
    ax1.plot(t_h, 3*sigma_pos_norm, "r--", lw=1, label="filter 3σ")
    ax1.plot(t_h, sm_norm,          lw=1.0, color="C2", label="smoother |error|")
    ax1.plot(t_h, 3*sm_sigma,       "g--", lw=1, alpha=0.6, label="smoother 3σ")
    ax1.set_yscale("log"); ax1.set_ylabel("|r_error| [m]")
    ax1.set_xlabel("Time [h]"); ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3); ax1.set_title("Position")

    ax2.plot(t_h, err_vel_norm,     lw=0.7, label="filter |error|", color="C0")
    ax2.plot(t_h, 3*sigma_vel_norm, "r--", lw=1, label="filter 3σ")
    ax2.plot(t_h, sm_v,             lw=1.0, color="C2", label="smoother |error|")
    ax2.plot(t_h, 3*sm_v_sig,       "g--", lw=1, alpha=0.6, label="smoother 3σ")
    ax2.set_yscale("log"); ax2.set_ylabel("|v_error| [m/s]")
    ax2.set_xlabel("Time [h]"); ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3); ax2.set_title("Velocity")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "deputy_convergence.png", dpi=150)

    # ── 4) IPLF iter count ──
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.step(t_h, res["los_iters"], where="post", lw=0.8)
    ax.set_xlabel("Time [h]"); ax.set_ylabel("IPLF iterations on LOS update")
    ax.set_title("IPLF iteration count per nav step (v3)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if SAVE_FIGS: fig.savefig(FIG_DIR / "iplf_iters.png", dpi=150)

    # ── 5) Relative position residual ──
    rel_est   = res["est_chief"][:, :3]   - res["est_dep"][:, :3]
    rel_truth = res["truth_chief"][:, :3] - res["truth_dep"][:, :3]
    rel_res   = rel_est - rel_truth
    rel_res_norm   = la.norm(rel_res,   axis=1)
    rel_truth_norm = la.norm(rel_truth, axis=1)
    rel_est_norm   = la.norm(rel_est,   axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("DEPUTY relative position estimate — convergence (v3)",
                 fontsize=14)
    for j, lab in enumerate(["x", "y", "z"]):
        axes[0, 0].plot(t_h, rel_res[:, j], lw=0.6, alpha=0.85,
                        label=f"Δr_{lab} residual")
    axes[0, 0].axhline(0, color="k", lw=0.5)
    axes[0, 0].set_xlabel("Time [h]"); axes[0, 0].set_ylabel("Δr_est − Δr_truth [m]")
    axes[0, 0].set_title("Relative position residual per axis")
    axes[0, 0].legend(fontsize=9); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(t_h, np.maximum(rel_res_norm, 1e-3), lw=0.7, color="C3")
    axes[0, 1].set_xlabel("Time [h]"); axes[0, 1].set_ylabel("|Δr_est − Δr_truth| [m]")
    axes[0, 1].set_title("Relative position residual norm (LOG)")
    axes[0, 1].grid(True, which="both", alpha=0.3)

    axes[1, 0].plot(t_h, rel_truth_norm, lw=0.7, label="|Δr| truth")
    axes[1, 0].plot(t_h, rel_est_norm,   lw=0.7, label="|Δr| estimate", alpha=0.75)
    axes[1, 0].set_xlabel("Time [h]"); axes[1, 0].set_ylabel("|Δr| [m]")
    axes[1, 0].set_title("Chief↔Deputy range: truth vs estimate")
    axes[1, 0].legend(fontsize=9); axes[1, 0].grid(True, alpha=0.3)

    cmin = np.minimum.accumulate(rel_res_norm)
    axes[1, 1].plot(t_h, rel_res_norm, lw=0.6, alpha=0.7, label="instantaneous")
    axes[1, 1].plot(t_h, cmin, lw=1.2, color="C2", label="running minimum (envelope)")
    axes[1, 1].set_xlabel("Time [h]"); axes[1, 1].set_ylabel("|Δr_est − Δr_truth| [m]")
    axes[1, 1].set_title("Relative position residual (linear)")
    axes[1, 1].legend(fontsize=9); axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "relative_residual.png", dpi=150)

    # ── 6) Hill / LVLH frame relative motion (NEW) ──
    rho_H_truth = _hill_relative_state(res["truth_chief"], res["truth_dep"])
    rho_H_est   = _hill_relative_state(res["est_chief"],   res["est_dep"])
    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")
    fig.suptitle("Deputy in CHIEF's Hill/LVLH frame — truth (solid) vs est. (dashed)",
                 fontsize=13)
    decim = max(1, len(t) // 4000)
    ax.plot(rho_H_truth[::decim, 0], rho_H_truth[::decim, 1],
            rho_H_truth[::decim, 2], "g-",  lw=1.4, label="ρ truth")
    ax.plot(rho_H_est[::decim, 0],   rho_H_est[::decim, 1],
            rho_H_est[::decim, 2],   "r--", lw=1.0, alpha=0.7,
            label="ρ estimate")
    ax.scatter(0, 0, 0, color="k", s=100, marker="X", label="chief")
    ax.set_xlabel("R̂ (radial) [m]")
    ax.set_ylabel("Θ̂ (along-track) [m]")
    ax.set_zlabel("Ĥ (cross-track) [m]")
    ax.legend(fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "hill_frame.png", dpi=150)

    # ── 7) 3-D ECI trajectories ──
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("ECI trajectories — v3 (BSK truth) — truth vs estimate",
                 fontsize=14)
    decim = max(1, len(t) // 4000)
    tc = res["truth_chief"][::decim]; ec = res["est_chief"][::decim]
    td = res["truth_dep"][::decim];   ed = res["est_dep"][::decim]
    ax.plot(tc[:, 0], tc[:, 1], tc[:, 2], "b-",  lw=1.2, label="chief truth")
    ax.plot(ec[:, 0], ec[:, 1], ec[:, 2], "b--", lw=0.8, alpha=0.7, label="chief est")
    ax.plot(td[:, 0], td[:, 1], td[:, 2], "g-",  lw=1.2, label="deputy truth")
    ax.plot(ed[:, 0], ed[:, 1], ed[:, 2], "g--", lw=0.8, alpha=0.7, label="deputy est")
    ax.scatter(0, 0, 0, color="k", s=140, marker="X", label="Earth centre")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "eci_trajectories.png", dpi=150)


if __name__ == "__main__":
    np.set_printoptions(precision=9, suppress=False, linewidth=200)
    res = run_sim(verbose=True)
    plot_results(res)
    plt.show()
