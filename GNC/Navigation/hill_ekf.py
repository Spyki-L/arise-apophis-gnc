#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hill_ekf.py
===========

Hill-frame Kalman filter for deputy relative navigation in the chief LVLH frame.

State
-----
x = [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N]^T

* ρ    : relative position deputy - chief in LVLH/Hill frame [m]
* ρ̇    : relative velocity in LVLH/Hill frame [m/s]

Process model
-------------
Discrete-time Hill/Clohessy–Wiltshire dynamics:

    x_{k+1}^- = Φ(n_k, dt) x_k^+ + w_k

where Φ is built from the same `cw_blocks` used in RHIG.

Measurements
------------
At each step, we use an inertial nav state from SimpleNav:

    r_nav_N, v_nav_N : deputy in ECI [m], [m/s]
    r_c_N,   v_c_N   : chief in ECI [m], [m/s]

We form "measured" Hill states:

    ρ_meas    = C_HN (r_nav_N - r_c_N)
    ρ̇_meas   = C_HN (v_nav_N - v_c_N) - Ω_H × ρ_meas

with C_HN = dcm_lvlh(r_c_N, v_c_N), Ω_H = [0, 0, n]^T.

Measurement model:

    z_k = H x_k + v_k,   H = I_6

Config
------
HillEKFConfig:
    dt            : filter timestep [s]
    sigma_proc_*  : process noise std
    sigma_meas_*  : measurement noise std
    sigma_init_*  : initial covariance std
    sigma_dv      : process noise bump on velocity at burns
    use_joseph    : if True, use Joseph covariance update

Usage
-----
See geo_ckf_ekf_demo.py for an example offline CKF→EKF application.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import numpy.linalg as la


# ───────────────────── HCW / Hill helpers ───────────────────────────────────

def dcm_lvlh(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    LVLH<-ECI DCM with rows = [R; T; N] in ECI components.

    Parameters
    ----------
    r, v : (3,) ndarray
        Chief position and velocity in ECI [m], [m/s].

    Returns
    -------
    C : (3,3) ndarray
        DCM mapping v_N -> v_LVLH (rows = RTN axes expressed in ECI).
    """
    r = np.asarray(r, float).reshape(3)
    v = np.asarray(v, float).reshape(3)
    ih = r / la.norm(r)
    k  = np.cross(r, v); k /= la.norm(k)
    j  = np.cross(k, ih)
    return np.vstack((ih, j, k))


def hill_rate(r: np.ndarray, v: np.ndarray) -> float:
    """Mean motion / Hill rate n = |h| / r^2."""
    h_vec = np.cross(r, v)
    return float(la.norm(h_vec) / (la.norm(r)**2 + 1e-12))


def cw_blocks(n: float, dt: float):
    """
    HCW state transition sub-blocks.

    Returns
    -------
    Phi_rr, Phi_rv, Phi_vr, Phi_vv : each (3,3)
    """
    c  = np.cos(n * dt)
    s  = np.sin(n * dt)
    nt = n * dt

    Phi_rr = np.array([[4 - 3 * c, 0,         0],
                       [6 * (s - nt), 1,      0],
                       [0,           0,       c]])

    Phi_rv = np.array([[s / n,        2 * (1 - c) / n, 0],
                       [-2 * (1 - c) / n, (4 * s - 3 * nt) / n, 0],
                       [0,              0,                   s / n]])

    Phi_vr = np.array([[3 * n * s, 0, 0],
                       [6 * n * (c - 1), 0, 0],
                       [0, 0, -n * s]])

    Phi_vv = np.array([[c, 2 * s, 0],
                       [-2 * s, 4 * c - 3, 0],
                       [0, 0, c]])

    return Phi_rr, Phi_rv, Phi_vr, Phi_vv


def cw_state_transition(n: float, dt: float) -> np.ndarray:
    """Full 6×6 Φ from the HCW blocks."""
    Phi_rr, Phi_rv, Phi_vr, Phi_vv = cw_blocks(n, dt)
    Phi = np.block([[Phi_rr, Phi_rv],
                    [Phi_vr, Phi_vv]])
    return Phi


# ───────────────────── EKF config & state snapshot ──────────────────────────

@dataclass
class HillEKFConfig:
    dt: float = 0.1  # [s] filter step (mid-rate bus)

    # Process noise (1σ) on each component
    sigma_proc_pos: float = 5.0    # [m]
    sigma_proc_vel: float = 5e-3   # [m/s]

    # Measurement noise (1σ)
    sigma_meas_pos: float = 50.0   # [m]
    sigma_meas_vel: float = 0.05   # [m/s]

    # Initial covariance (1σ)
    sigma_init_pos: float = 100.0  # [m]
    sigma_init_vel: float = 0.1    # [m/s]

    # Additional covariance bump on velocities at each burn (ΔV uncertainty)
    sigma_dv: float = 0.01         # [m/s]

    # Use Joseph form or simple covariance update
    use_joseph: bool = True


@dataclass
class HillEKFSnapshot:
    t: float
    x: np.ndarray        # (6,) posterior state
    P: np.ndarray        # (6,6) posterior covariance
    innov: np.ndarray    # (6,) pre-fit innovation y_k
    S: np.ndarray        # (6,6) innovation covariance


# ───────────────────── main EKF class ────────────────────────────────────────

class HillEKF:
    """
    Linear KF for deputy–chief relative motion in Hill/LVLH frame.

    chief_state_fn: Callable mapping t_abs [s] -> (r_c_N, v_c_N) [m, m/s].
    """

    def __init__(
        self,
        cfg: HillEKFConfig,
        chief_state_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]],
    ):
        self.cfg = cfg
        self.chief_state_fn = chief_state_fn

        # Process and measurement covariances
        q_pos2 = cfg.sigma_proc_pos**2
        q_vel2 = cfg.sigma_proc_vel**2
        self.Q = np.diag([q_pos2, q_pos2, q_pos2,
                          q_vel2, q_vel2, q_vel2])

        r_pos2 = cfg.sigma_meas_pos**2
        r_vel2 = cfg.sigma_meas_vel**2
        self.R = np.diag([r_pos2, r_pos2, r_pos2,
                          r_vel2, r_vel2, r_vel2])

        p_pos2 = cfg.sigma_init_pos**2
        p_vel2 = cfg.sigma_init_vel**2
        self.P = np.diag([p_pos2, p_pos2, p_pos2,
                          p_vel2, p_vel2, p_vel2])

        self.Q_dv = (cfg.sigma_dv**2) * np.eye(3)

        # Filter state
        self.x = np.zeros(6)   # posterior state at last step
        self.t_last: Optional[float] = None

        # For diagnostics
        self.innov = np.zeros(6)    # pre-fit residual
        self.S = np.eye(6)
        self.x_pred = np.zeros(6)
        self.P_pred = self.P.copy()

        self.y_pre = np.zeros(6)    # pre-fit residual
        self.y_post = np.zeros(6)   # post-fit residual
        self.H_last = np.eye(6)

    # ---------------------------------------------------------------- reset --

    def reset(self,
              x_init: Optional[np.ndarray] = None,
              P_init: Optional[np.ndarray] = None):
        """
        Reset filter state and covariance (for multi-pass CKF usage).
        """
        if x_init is None:
            self.x = np.zeros(6)
        else:
            self.x = np.asarray(x_init, float).reshape(6)
        if P_init is None:
            self.P = self.P.copy()  # keep whatever was configured
        else:
            self.P = np.asarray(P_init, float).reshape(6, 6)
        self.t_last = None

    # ---------------------------------------------------------------- utils --

    def _compute_meas(self, t_abs: float,
                      r_nav_N: np.ndarray,
                      v_nav_N: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build z_k and H_k given nav and chief state.

        Returns
        -------
        z : (6,) ndarray
        H : (6,6) identity
        """
        r_nav_N = np.asarray(r_nav_N, float).reshape(3)
        v_nav_N = np.asarray(v_nav_N, float).reshape(3)

        r_c_N, v_c_N = self.chief_state_fn(t_abs)
        r_c_N = np.asarray(r_c_N, float).reshape(3)
        v_c_N = np.asarray(v_c_N, float).reshape(3)

        C_HN = dcm_lvlh(r_c_N, v_c_N)
        n = hill_rate(r_c_N, v_c_N)

        rho_meas = C_HN @ (r_nav_N - r_c_N)
        rho_dot_meas = C_HN @ (v_nav_N - v_c_N) - np.cross(
            np.array([0.0, 0.0, n]), rho_meas
        )

        z = np.hstack((rho_meas, rho_dot_meas))
        H = np.eye(6)
        return z, H

    # ---------------------------------------------------------------- predict

    def predict(self, t_abs: float, burn_dv_H: Optional[np.ndarray] = None):
        """
        Time update using HCW transition; applies optional instantaneous Δv
        to the velocity components.
        """
        dt = self.cfg.dt
        if self.t_last is None:
            self.t_last = t_abs - dt

        # Use chief state at mid-step for n
        t_mid = 0.5 * (self.t_last + t_abs)
        r_c_N, v_c_N = self.chief_state_fn(t_mid)
        n = hill_rate(r_c_N, v_c_N)
        Phi = cw_state_transition(n, dt)

        self.x_pred = Phi @ self.x
        self.P_pred = Phi @ self.P @ Phi.T + self.Q

        # Inject Δv in Hill if we had a burn on this step
        if burn_dv_H is not None:
            burn_dv_H = np.asarray(burn_dv_H, float).reshape(3)
            self.x_pred[3:6] += burn_dv_H
            self.P_pred[3:6, 3:6] += self.Q_dv

    # ----------------------------------------------------------------- update

    def update(self, t_abs: float,
               r_nav_N: np.ndarray,
               v_nav_N: np.ndarray) -> HillEKFSnapshot:
        """
        Measurement update using SimpleNav inertial state.
        """
        z, H = self._compute_meas(t_abs, r_nav_N, v_nav_N)
        I6 = np.eye(6)

        # pre-fit innovation
        y = z - H @ self.x_pred
        S = H @ self.P_pred @ H.T + self.R
        K = self.P_pred @ H.T @ la.inv(S)

        x_new = self.x_pred + K @ y

        if self.cfg.use_joseph:
            # Joseph form
            KH = K @ H
            self.P = (I6 - KH) @ self.P_pred @ (I6 - KH).T + K @ self.R @ K.T
        else:
            # simple form
            self.P = (I6 - K @ H) @ self.P_pred

        self.x = x_new
        self.t_last = t_abs

        # Diagnostics
        self.innov = y
        self.S = S
        self.y_pre = y
        self.y_post = z - H @ self.x
        self.H_last = H.copy()

        snap = HillEKFSnapshot(
            t=t_abs,
            x=self.x.copy(),
            P=self.P.copy(),
            innov=self.innov.copy(),
            S=self.S.copy(),
        )
        return snap

    # ----------------------------------------------------------------- NEW: update from Hill measurement directly

    def update_from_hill_measurement(self, z_H: np.ndarray) -> HillEKFSnapshot:
        """
        Measurement update when the measurement is already a Hill state:

            z_H = [ρ_meas, ρ̇_meas] in RTN.

        This skips the inertial -> Hill conversion and just uses H = I.
        """
        z = np.asarray(z_H, float).reshape(6)
        H = np.eye(6)
        I6 = np.eye(6)

        # innovation using predicted state
        y = z - H @ self.x_pred
        S = H @ self.P_pred @ H.T + self.R
        K = self.P_pred @ H.T @ la.inv(S)

        x_new = self.x_pred + K @ y

        if self.cfg.use_joseph:
            KH = K @ H
            self.P = (I6 - KH) @ self.P_pred @ (I6 - KH).T + K @ self.R @ K.T
        else:
            self.P = (I6 - K @ H) @ self.P_pred

        self.x = x_new

        # Do not change t_last here; it is updated in predict()
        self.innov = y
        self.S = S
        self.y_pre = y
        self.y_post = z - H @ self.x
        self.H_last = H.copy()

        snap = HillEKFSnapshot(
            t=self.t_last if self.t_last is not None else 0.0,
            x=self.x.copy(),
            P=self.P.copy(),
            innov=self.innov.copy(),
            S=self.S.copy(),
        )
        return snap

    # ----------------------------------------------------------------- step

    def step(
        self,
        t_abs: float,
        r_nav_N: np.ndarray,
        v_nav_N: np.ndarray,
        burn_dv_H: Optional[np.ndarray] = None,
    ) -> HillEKFSnapshot:
        """
        Combined predict + update in one call using inertial nav states.
        """
        self.predict(t_abs, burn_dv_H=burn_dv_H)
        snap = self.update(t_abs, r_nav_N, v_nav_N)
        return snap

