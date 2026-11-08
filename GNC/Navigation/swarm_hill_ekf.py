#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swarm_hill_ekf.py
=================

Multi-deputy Hill-frame EKF with cross-link measurements.

State
-----
x = [x_0; x_1; ...; x_{N-1}],  each x_i = [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N]^T

Process model
-------------
Per deputy, CW dynamics around common chief:

    x_{i,k+1} = Φ(n_k, dt) x_{i,k} + w_{i,k}

Whole swarm:

    x_{k+1} = (I_N ⊗ Φ) x_k + w_k

Measurements
------------
1) Absolute Hill measurement for deputy i:

       z_i = x_i + v_i  (6x1)

   H_i selects block i.

2) Relative cross-link between i and j (position only):

       z_ij = ρ_j - ρ_i + v_ij  (3x1)

   H_ij compares R/T/N position parts of blocks i and j.

Innovation logging
------------------
Each measurement update records:
    - raw innovation y
    - innovation covariance S
    - whitened innovation nu = L^{-1} y  (S = LLᵀ)
along with timestamp, measurement type, and index.
"""

from __future__ import annotations
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Literal, Dict

import numpy as np
import numpy.linalg as la

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


MeasType = Literal["abs", "rel"]


@dataclass
class SwarmHillEKFConfig:
    dt: float = 1.0

    # Process noise (1σ)
    sigma_proc_pos: float = 1.0     # [m]
    sigma_proc_vel: float = 2e-3    # [m/s]

    # Absolute measurement noise (1σ)
    sigma_meas_abs_pos: float = 20.0   # [m]
    sigma_meas_abs_vel: float = 0.02   # [m/s]

    # Relative cross-link measurement noise (1σ)
    sigma_meas_rel_pos: float = 5.0    # [m]

    # Initial covariance (1σ)
    sigma_init_pos: float = 1500.0     # [m]
    sigma_init_vel: float = 0.5        # [m/s]

    # Additional covariance bump on velocities at burns (Δv uncertainty)
    sigma_dv: float = 0.05             # [m/s]

    use_joseph: bool = True


@dataclass
class InnovationRecord:
    t: float
    meas_type: MeasType          # "abs" or "rel"
    idx: Tuple[int, ...]         # (i,) for abs, (i,j) for rel
    y: np.ndarray                # raw innovation
    nu: np.ndarray               # whitened innovation
    S: np.ndarray                # innovation covariance


class SwarmHillEKF:
    """
    Centralized EKF for N_dep deputies in chief LVLH frame.

    chief_state_fn: t_abs -> (r_c_N, v_c_N) in ECI [m, m/s]
    """

    def __init__(
        self,
        N_dep: int,
        cfg: SwarmHillEKFConfig,
        chief_state_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]],
    ):
        self.N_dep = int(N_dep)
        self.cfg = cfg
        self.chief_state_fn = chief_state_fn

        self.nx = 6 * self.N_dep

        # Single-deputy process covariance Q_single
        q_pos2 = cfg.sigma_proc_pos**2
        q_vel2 = cfg.sigma_proc_vel**2
        Q_single = np.diag([q_pos2, q_pos2, q_pos2,
                            q_vel2, q_vel2, q_vel2])
        # Swarm Q
        self.Q = np.kron(np.eye(self.N_dep), Q_single)

        # Absolute R template
        r_pos2 = cfg.sigma_meas_abs_pos**2
        r_vel2 = cfg.sigma_meas_abs_vel**2
        self.R_abs = np.diag([r_pos2, r_pos2, r_pos2,
                              r_vel2, r_vel2, r_vel2])

        # Relative R template (3x3)
        r_rel2 = cfg.sigma_meas_rel_pos**2
        self.R_rel = np.eye(3) * r_rel2

        # Initial covariance
        p_pos2 = cfg.sigma_init_pos**2
        p_vel2 = cfg.sigma_init_vel**2
        P_single = np.diag([p_pos2, p_pos2, p_pos2,
                            p_vel2, p_vel2, p_vel2])
        self.P_init = np.kron(np.eye(self.N_dep), P_single)

        # Δv process bump (velocity sub-blocks only)
        self.Q_dv_single = (cfg.sigma_dv**2) * np.eye(3)

        # Filter state
        self.x = np.zeros(self.nx)
        self.P = self.P_init.copy()
        self.x_pred = np.zeros_like(self.x)
        self.P_pred = self.P.copy()

        self.t_last: Optional[float] = None

        # Innovation history
        self.innovations: List[InnovationRecord] = []

    # ---------------------------------------------------------------- reset --

    def reset(self,
              x_init: Optional[np.ndarray] = None,
              P_init: Optional[np.ndarray] = None):
        if x_init is None:
            self.x = np.zeros(self.nx)
        else:
            self.x = np.asarray(x_init, float).reshape(self.nx)

        if P_init is None:
            self.P = self.P_init.copy()
        else:
            self.P = np.asarray(P_init, float).reshape(self.nx, self.nx)

        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()
        self.t_last = None
        self.innovations.clear()

    # ---------------------------------------------------------------- predict --

    def predict(self,
                t_abs: float,
                burns_dv_H: Optional[np.ndarray] = None):
        """
        Time update step.

        burns_dv_H: optional (N_dep, 3) array of Δv in Hill for each deputy.
        """
        dt = self.cfg.dt
        if self.t_last is None:
            self.t_last = t_abs - dt

        # chief dynamics at mid-step
        t_mid = 0.5 * (self.t_last + t_abs)
        r_c, v_c = self.chief_state_fn(t_mid)
        n = hill_rate(r_c, v_c)
        Phi_single = cw_state_transition(n, dt)

        # Build block-diagonal Φ_big = I_N ⊗ Φ_single
        Phi_big = np.kron(np.eye(self.N_dep), Phi_single)

        self.x_pred = Phi_big @ self.x
        self.P_pred = Phi_big @ self.P @ Phi_big.T + self.Q

        # burns: add Δv to velocities & process bump to corresponding sub-blocks
        if burns_dv_H is not None:
            dv = np.asarray(burns_dv_H, float).reshape(self.N_dep, 3)
            for i in range(self.N_dep):
                if not np.any(dv[i]):
                    continue
                idx_v = slice(6*i + 3, 6*i + 6)
                self.x_pred[idx_v] += dv[i]
                # bump velocity-velocity covariance block
                self.P_pred[idx_v, idx_v] += self.Q_dv_single

        self.t_last = t_abs

    # ---------------------------------------------------------------- utility --

    def _update_generic(self,
                        t: float,
                        H: np.ndarray,
                        z: np.ndarray,
                        R: np.ndarray,
                        meas_type: MeasType,
                        idx: Tuple[int, ...]):
        """
        Generic EKF measurement update. Also logs innovation (raw + whitened).
        """
        z = np.asarray(z, float).reshape(-1)
        H = np.asarray(H, float)
        R = np.asarray(R, float)

        I = np.eye(self.nx)

        y = z - H @ self.x_pred
        S = H @ self.P_pred @ H.T + R
        K = self.P_pred @ H.T @ la.inv(S)

        self.x = self.x_pred + K @ y

        if self.cfg.use_joseph:
            KH = K @ H
            self.P = (I - KH) @ self.P_pred @ (I - KH).T + K @ R @ K.T
        else:
            self.P = (I - K @ H) @ self.P_pred

        # whiten innovation
        try:
            L = la.cholesky(S)
            nu = la.solve(L, y)
        except la.LinAlgError:
            # fall back to pseudo-inverse whitening if S not PD
            Sinv_sqrt = la.sqrtm(la.pinv(S))
            nu = Sinv_sqrt @ y

        rec = InnovationRecord(
            t=t,
            meas_type=meas_type,
            idx=idx,
            y=y.copy(),
            nu=nu.copy(),
            S=S.copy(),
        )
        self.innovations.append(rec)

    # -------------------------------------------------------------- update abs --

    def update_abs(self,
                   t_abs: float,
                   dep_idx: int,
                   z_abs: np.ndarray,
                   R_override: Optional[np.ndarray] = None):
        """
        Absolute measurement update for deputy dep_idx::

            z_abs = x_dep + v,  dim 6

        If R_override is given, it is used instead of the template R_abs.
        """
        i = int(dep_idx)
        assert 0 <= i < self.N_dep

        H = np.zeros((6, self.nx))
        H[:, 6*i:6*i+6] = np.eye(6)

        R = self.R_abs if R_override is None else np.asarray(R_override, float)

        self._update_generic(t_abs, H, z_abs, R, "abs", (i,))

    # -------------------------------------------------------------- update rel --

    def update_rel(self,
                   t_abs: float,
                   i: int,
                   j: int,
                   z_rel: np.ndarray,
                   R_override: Optional[np.ndarray] = None):
        """
        Relative cross-link measurement between deputies i and j::

            z_rel = ρ_j - ρ_i + v,  dim 3

        Only position components are used.
        """
        i = int(i)
        j = int(j)
        assert 0 <= i < self.N_dep
        assert 0 <= j < self.N_dep
        assert i != j

        H = np.zeros((3, self.nx))
        # -I on position block of i
        H[:, 6*i:6*i+3] = -np.eye(3)
        # +I on position block of j
        H[:, 6*j:6*j+3] = np.eye(3)

        R = self.R_rel if R_override is None else np.asarray(R_override, float)

        self._update_generic(t_abs, H, z_rel, R, "rel", (i, j))

