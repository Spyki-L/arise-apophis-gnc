#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_peer_relnav_v4.py
================================

Cascade-Anchored MOS-CI with **Dynamic Model Compensation (DMC)**

Builds on v3 by augmenting the per-deputy state from 6 to 9 dimensions:

    x = [ ρ_R, ρ_T, ρ_N,    position    (Hill / chief LVLH)
          ρ̇_R, ρ̇_T, ρ̇_N,    velocity
          a_R, a_T, a_N ]    unmodelled Hill-frame acceleration

The acceleration triple is a 3-D first-order Gauss-Markov process

        ȧ(t) = -a(t) / τ_a  +  w(t),    w ~ N(0, q_a · I_3)

so the filter explicitly absorbs the residual deep-space dynamics that
CW cannot represent: solar radiation pressure, third-body Earth and
Sun, and Apophis irregular-gravity perturbations.

This is the classic Cruz / Tapley / Ingram-Tapley DMC formulation
(Mars approach navigation, 1970s; widely used on asteroid rendezvous).

Δ from v3
---------
* State 6 -> 9.  Φ_aug is block:

      [ Φ_CW (6×6)        Γ_xa (6×3) ]
      [ 0   (3×6)         β · I (3×3) ]   β = exp(-dt/τ_a)

  with Γ_xa = [½ dt² · I; dt · I]  (constant-accel over interval).

* Process noise:
      top 6×6 = small CW Q,
      bottom 3×3 = (σ_a² (1 − e^{-2 dt/τ_a})) · I_3   (steady-state OU).

* Measurement Jacobians extend with zero columns for the 3 accel states.

* MOS-CI: position marginal still 3-D.  Conditional update now propagates
  the position improvement to the 6-D (velocity ∪ acceleration) tail via
  a 6×3 cross-covariance gain.

* Burn aggregation (from v3 patch): every impulse is summed into the
  sim-step window that contains it -> no burns dropped under MEAS_DECIM>>1.

* Diagnostics: estimated DMC acceleration printed and plotted per deputy.

Author: Harish Vernekar (2026)
"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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

from GNC.Navigation.hill_ekf import dcm_lvlh, hill_rate, cw_state_transition
from GNC.Navigation.crosslink_scheduler import CrosslinkScheduler
from GNC.Scenario.Apophis.apophis_swarm_peer_relnav_v2 import (
    polar_to_cartesian_drho,
    polar_meas_covariance_in_cartesian,
    peer_range_pred_and_jac as _v2_range_jac,
    peer_bearing_pred_and_jac as _v2_bearing_jac,
    generate_peer_measurement,
    load_apophis_bundle,
    make_state_fn,
    eci_to_hill_traj,
    PeerNavSnapshot,
)
from GNC.Scenario.Apophis.apophis_swarm_peer_relnav_v3 import (
    _find_chief_reference, load_chief_reference,
    CHIEF_REFERENCE_NAME,
)

# ──────────────────────────────────────────────────────────────────────────────
#                              CONFIG  (v4 defaults)
# ──────────────────────────────────────────────────────────────────────────────

N_DEP       = 8
MEAS_DECIM  = 20
CROP_HR     = 0.0

# Peer sensor noise
RANGE_SIGMA   = 5.0
AZ_SIGMA_DEG  = 0.3
EL_SIGMA_DEG  = 0.3

# Chief observation
CHIEF_OBS_SIGMA_POS = 5.0
CHIEF_OBS_DROPOUT   = 0.05

# Bootstrap CI burst
BOOTSTRAP_CI_BURST_N = 4
BOOTSTRAP_NO_DROPOUT = True

# Peer link
P_PEER_DROPOUT = 0.05
P_PEER_OUTLIER = 0.01
OUTLIER_SCALE  = 6.0

NIS_GATE_PEER  = 12.84   # chi2(3) @ 99.5%
NIS_GATE_CHIEF = 11.34   # chi2(3) @ 99.0%

# MOS-CI
USE_CI_FUSION       = True
CI_EVERY_N_STEPS    = 1
CI_NEES_TARGET      = 3.0
CI_NEES_PENALTY     = 0.10
CI_REJECT_DET_RATIO = 1.05

# Event-triggered ISL broadcast
ISL_TRIGGER_MAHAL  = 0.3
ISL_TRIGGER_TR_REL = 0.05

# DMC parameters
DMC_TAU      = 900.0     # [s]    correlation time of the acceleration process
DMC_SIGMA_A  = 1e-4      # [m/s²] steady-state RMS of unmodelled accel
                         #        (SRP at 1 AU ~ 1e-7..1e-6 m/s², third-body Earth
                         #         passing close ~ 1e-5..1e-4 m/s²; pick high end)
DMC_INIT_A_SIGMA = 1e-4  # [m/s²] initial 1σ on the accel state

# Plotting
SAVE_FIGS = False
FIG_DIR   = SCRIPT_DIR / "peer_relnav_v4_figs"


# ──────────────────────────────────────────────────────────────────────────────
#                       9-D state-transition + Jacobians
# ──────────────────────────────────────────────────────────────────────────────

def cw_dmc_state_transition(n: float, dt: float, tau_a: float) -> np.ndarray:
    """
    9×9 augmented state-transition for [pos(3); vel(3); accel(3)] in Hill
    frame.

    Top 6×6  = Φ_CW (Clohessy–Wiltshire),  built by cw_state_transition().
    Top 6×3  = Γ_xa, the "constant-acceleration over [0, dt]" influence on
               (pos, vel).  We use the standard zero-order-hold approximation:
                   Γ_pos = ½ dt² · I_3
                   Γ_vel = dt    · I_3
    Bot 3×3  = β · I_3, where β = exp(-dt / τ_a) is the OU decay factor.
    Bot 3×6  = 0.
    """
    Phi_cw = cw_state_transition(n, dt)                      # 6×6
    Phi = np.zeros((9, 9))
    Phi[:6, :6] = Phi_cw
    Phi[:3, 6:9] = 0.5 * dt * dt * np.eye(3)
    Phi[3:6, 6:9] = dt * np.eye(3)
    beta = np.exp(-dt / max(tau_a, 1e-9))
    Phi[6:9, 6:9] = beta * np.eye(3)
    return Phi


def dmc_process_noise(dt: float, sigma_a: float, tau_a: float,
                      sigma_proc_pos: float = 0.0,
                      sigma_proc_vel: float = 0.0) -> np.ndarray:
    """
    Per-step 9×9 Q for the augmented system.  The acceleration block is
    the analytic stationary OU increment

        Q_acc = σ_a² · (1 − e^{−2 dt / τ_a}) · I_3.

    The pos/vel blocks carry only residual numerical noise — DMC absorbs
    the bulk of the unmodelled dynamics, so we keep them small.
    """
    qp = (sigma_proc_pos ** 2) * dt
    qv = (sigma_proc_vel ** 2) * dt
    qa = (sigma_a ** 2) * (1.0 - np.exp(-2.0 * dt / max(tau_a, 1e-9)))
    Q = np.zeros((9, 9))
    Q[0, 0] = Q[1, 1] = Q[2, 2] = qp
    Q[3, 3] = Q[4, 4] = Q[5, 5] = qv
    Q[6, 6] = Q[7, 7] = Q[8, 8] = qa
    return Q


def peer_range_pred_and_jac9(x_i9, x_j_hat9):
    """Range Jacobian extended to 9-D (zero acceleration columns)."""
    s, H6 = _v2_range_jac(x_i9[:6], x_j_hat9[:6])     # 1×6
    H = np.zeros((1, 9)); H[:, :6] = H6
    return s, H


def peer_bearing_pred_and_jac9(x_i9, x_j_hat9):
    z, H6 = _v2_bearing_jac(x_i9[:6], x_j_hat9[:6])    # 2×6
    H = np.zeros((2, 9)); H[:, :6] = H6
    return z, H


# ──────────────────────────────────────────────────────────────────────────────
#                      9-D MOS-CI (DMC-aware)
# ──────────────────────────────────────────────────────────────────────────────

def oif_ci_fuse_dmc(
    x_i: np.ndarray, P_i: np.ndarray,
    z_polar: np.ndarray,
    R_polar_diag: Tuple[float, float, float],
    x_j_hat: np.ndarray, P_j: np.ndarray,
    nees_target: float = CI_NEES_TARGET,
    nees_weight: float = 0.0,
    x_true_for_nees: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """
    9-D MOS-CI.  Position marginal stays 3-D; the conditional re-embed
    now writes back to the 6-D *velocity ∪ acceleration* tail via a
    6×3 cross-covariance gain.
    """
    sr, saz, sel = R_polar_diag
    drho   = polar_to_cartesian_drho(z_polar)                     # (3,)
    R_cart = polar_meas_covariance_in_cartesian(z_polar, sr, saz, sel)
    rho_ind  = x_j_hat[:3] - drho                                  # (3,)
    Sigma_ind = P_j[:3, :3] + R_cart                               # (3,3)

    P_a_pos = P_i[:3, :3].copy()
    x_a_pos = x_i[:3].copy()
    Y_a_pos = la.inv(P_a_pos + 1e-12 * np.eye(3))
    Y_b_pos = la.inv(Sigma_ind + 1e-12 * np.eye(3))

    lo, hi = 1e-3, 1.0 - 1e-3
    gr = (np.sqrt(5.0) + 1.0) / 2.0
    def _trace_inv(w):
        Y = w * Y_a_pos + (1 - w) * Y_b_pos
        try:    return float(np.trace(la.inv(Y + 1e-15 * np.eye(3))))
        except la.LinAlgError: return 1e30
    def cost(w):
        c = _trace_inv(w)
        if nees_weight > 0.0 and x_true_for_nees is not None:
            Y = w * Y_a_pos + (1 - w) * Y_b_pos
            try:    Pp = la.inv(Y + 1e-15 * np.eye(3))
            except la.LinAlgError: return 1e30
            xp = Pp @ (w * Y_a_pos @ x_a_pos + (1 - w) * Y_b_pos @ rho_ind)
            e = xp - x_true_for_nees[:3]
            try:    nees = float(e @ la.solve(Pp, e))
            except la.LinAlgError: nees = 1e30
            c += nees_weight * (nees - nees_target) ** 2
        return c
    for _ in range(60):
        c = hi - (hi - lo) / gr; d = lo + (hi - lo) / gr
        if cost(c) < cost(d): hi = d
        else: lo = c
    omega_star = 0.5 * (lo + hi)

    # Boundary skip: peer adds nothing -> return prior unchanged
    if omega_star > 1.0 - 5e-3:
        return x_i.copy(), P_i.copy(), 1.0, dict(
            omega=1.0, skipped="peer_uninformative",
            tr_P_pre_pos=float(np.trace(P_i[:3, :3])),
            tr_P_post_pos=float(np.trace(P_i[:3, :3])),
            tr_P_pre_vel=float(np.trace(P_i[3:6, 3:6])),
            tr_P_post_vel=float(np.trace(P_i[3:6, 3:6])),
            tr_P_pre_acc=float(np.trace(P_i[6:9, 6:9])),
            tr_P_post_acc=float(np.trace(P_i[6:9, 6:9])),
            det_P_pre=float(la.det(P_i + 1e-12 * np.eye(9))),
            det_P_post=float(la.det(P_i + 1e-12 * np.eye(9))),
        )

    Y_f_pos = omega_star * Y_a_pos + (1 - omega_star) * Y_b_pos
    P_f_pos = la.inv(Y_f_pos + 1e-15 * np.eye(3))
    P_f_pos = 0.5 * (P_f_pos + P_f_pos.T)
    x_f_pos = P_f_pos @ (omega_star * Y_a_pos @ x_a_pos +
                          (1 - omega_star) * Y_b_pos @ rho_ind)

    # Conditional re-embed: tail = velocity (3) + acceleration (3) = 6 vars
    P_pp = P_i[:3, :3]
    P_pt = P_i[:3, 3:9]      # (3,6)  cross-cov pos -> (vel,accel)
    P_tt = P_i[3:9, 3:9]     # (6,6)
    P_pp_inv = la.inv(P_pp + 1e-12 * np.eye(3))
    K_cond = P_pt.T @ P_pp_inv                  # (6,3)

    x_f = np.empty(9)
    x_f[:3]  = x_f_pos
    x_f[3:9] = x_i[3:9] + K_cond @ (x_f_pos - x_a_pos)
    P_f = np.zeros((9, 9))
    P_f[:3, :3]  = P_f_pos
    P_f[:3, 3:9] = P_f_pos @ K_cond.T
    P_f[3:9, :3] = P_f[:3, 3:9].T
    P_f[3:9, 3:9] = P_tt - K_cond @ (P_pp - P_f_pos) @ K_cond.T
    P_f = 0.5 * (P_f + P_f.T)

    return x_f, P_f, float(omega_star), dict(
        omega=float(omega_star), skipped="",
        tr_P_pre_pos=float(np.trace(P_i[:3, :3])),
        tr_P_post_pos=float(np.trace(P_f[:3, :3])),
        tr_P_pre_vel=float(np.trace(P_i[3:6, 3:6])),
        tr_P_post_vel=float(np.trace(P_f[3:6, 3:6])),
        tr_P_pre_acc=float(np.trace(P_i[6:9, 6:9])),
        tr_P_post_acc=float(np.trace(P_f[6:9, 6:9])),
        det_P_pre=float(la.det(P_i + 1e-12 * np.eye(9))),
        det_P_post=float(la.det(P_f + 1e-12 * np.eye(9))),
    )


# ──────────────────────────────────────────────────────────────────────────────
#                              EKF (9-D, DMC)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PeerRelNavConfigV4:
    dt: float = 1.0
    sigma_proc_pos: float = 0.5      # very small; DMC handles bulk noise
    sigma_proc_vel: float = 5e-3
    sigma_range:    float = RANGE_SIGMA
    sigma_az:       float = np.deg2rad(AZ_SIGMA_DEG)
    sigma_el:       float = np.deg2rad(EL_SIGMA_DEG)
    sigma_chief_obs: float = CHIEF_OBS_SIGMA_POS
    sigma_init_pos: float = 1500.0
    sigma_init_vel: float = 0.5
    sigma_init_acc: float = DMC_INIT_A_SIGMA
    sigma_dv:       float = 0.05
    sigma_a_dmc:    float = DMC_SIGMA_A
    tau_a_dmc:      float = DMC_TAU
    use_joseph:     bool  = True


class PeerRelNavEKFV4:
    """9-D Cascade-Anchored EKF with DMC."""

    def __init__(self, dep_id, cfg, chief_state_fn):
        self.dep_id = dep_id
        self.cfg = cfg
        self.chief_state_fn = chief_state_fn

        pp2 = cfg.sigma_init_pos ** 2
        pv2 = cfg.sigma_init_vel ** 2
        pa2 = cfg.sigma_init_acc ** 2
        self.P_init = np.diag([pp2]*3 + [pv2]*3 + [pa2]*3)

        self.R_peer = np.diag([cfg.sigma_range**2,
                               cfg.sigma_az**2,
                               cfg.sigma_el**2])

        self.Q_dv = (cfg.sigma_dv ** 2) * np.eye(3)

        self.x = np.zeros(9)
        self.P = self.P_init.copy()
        self.x_pred = np.zeros(9)
        self.P_pred = self.P.copy()
        self.t_last: Optional[float] = None

        self.history: List[PeerNavSnapshot] = []
        self.peer_update_count = 0
        self.chief_obs_count   = 0
        self.ci_count          = 0
        self.ci_reject_count   = 0
        self.broadcast_count   = 0
        self.bootstrap_done    = False
        self.has_been_updated  = False
        self.deploy_t = None
        self.deploy_step = None

        self.x_last_broadcast: Optional[np.ndarray] = None
        self.P_last_broadcast: Optional[np.ndarray] = None
        self.t_last_broadcast: Optional[float] = None

        self.nees_history: List[float] = []
        self.omega_history: List[float] = []

    def reset(self, x0_6, P0=None):
        # x0_6 is 6-D (pos + vel from upstream); accel block init = 0
        self.x = np.zeros(9)
        self.x[:6] = np.asarray(x0_6, float).reshape(6)
        self.P = self.P_init.copy() if P0 is None else np.asarray(P0, float).reshape(9, 9)
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        self.t_last = None
        self.history.clear()
        self.peer_update_count = 0; self.chief_obs_count = 0
        self.ci_count = 0; self.ci_reject_count = 0
        self.broadcast_count = 0
        self.bootstrap_done = False; self.has_been_updated = False
        self.x_last_broadcast = None; self.P_last_broadcast = None
        self.t_last_broadcast = None
        self.nees_history.clear(); self.omega_history.clear()

    # ─────────────────── predict (DMC-augmented) ───────────────────

    def predict(self, t_abs, burn_dv_H=None):
        dt = self.cfg.dt
        if self.t_last is None:
            self.t_last = t_abs - dt
        t_mid = 0.5 * (self.t_last + t_abs)
        rC, vC = self.chief_state_fn(t_mid)
        n = hill_rate(rC, vC)

        Phi = cw_dmc_state_transition(n, dt, self.cfg.tau_a_dmc)
        Q   = dmc_process_noise(dt, self.cfg.sigma_a_dmc, self.cfg.tau_a_dmc,
                                 self.cfg.sigma_proc_pos, self.cfg.sigma_proc_vel)

        self.x_pred = Phi @ self.x
        self.P_pred = Phi @ self.P @ Phi.T + Q

        if burn_dv_H is not None:
            dv = np.asarray(burn_dv_H, float).reshape(3)
            if la.norm(dv) > 0.0:
                self.x_pred[3:6] += dv
                self.P_pred[3:6, 3:6] += self.Q_dv

        self.t_last = t_abs

    # ─────────────────── chief observation (3-D pos meas) ───────────────────

    def chief_observation_update(self, t, z_chief_pos, x_true_for_nees=None):
        z = np.asarray(z_chief_pos, float).reshape(3)
        H = np.zeros((3, 9)); H[:, :3] = np.eye(3)
        R = (self.cfg.sigma_chief_obs ** 2) * np.eye(3)

        zhat = self.x_pred[:3]
        y = z - zhat
        S = H @ self.P_pred @ H.T + R

        try:    nis = float(y @ la.solve(S, y))
        except la.LinAlgError: nis = float(y @ y)
        if nis > NIS_GATE_CHIEF:
            self.x = self.x_pred.copy(); self.P = self.P_pred.copy()
            return

        K = self.P_pred @ H.T @ la.inv(S)
        I9 = np.eye(9)
        self.x = self.x_pred + K @ y
        KH = K @ H
        if self.cfg.use_joseph:
            self.P = (I9 - KH) @ self.P_pred @ (I9 - KH).T + K @ R @ K.T
        else:
            self.P = (I9 - KH) @ self.P_pred
        self.P = 0.5 * (self.P + self.P.T)
        self.chief_obs_count += 1
        self.has_been_updated = True

        nees = None
        if x_true_for_nees is not None:
            e = self.x[:6] - x_true_for_nees[:6]
            try:    nees = float(e @ la.solve(self.P[:6, :6] + 1e-12*np.eye(6), e))
            except la.LinAlgError: nees = float("nan")
            self.nees_history.append(nees)

        self.history.append(PeerNavSnapshot(
            t=t, dep_id=self.dep_id, x=self.x.copy(), P=self.P.copy(),
            innov=y.copy(), S=S.copy(), nees=nees, meas_type="chief_obs"))
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()

    # ─────────────────── peer (range + bearing) update ───────────────────

    def peer_update(self, t, z_meas, x_j_hat9, P_j9, j_id,
                    x_true_for_nees=None):
        z_meas   = np.asarray(z_meas, float).reshape(3)
        x_j_hat9 = np.asarray(x_j_hat9, float).reshape(9)
        P_j9     = np.asarray(P_j9, float).reshape(9, 9)

        r_pred, H_r = peer_range_pred_and_jac9(self.x_pred, x_j_hat9)
        b_pred, H_b = peer_bearing_pred_and_jac9(self.x_pred, x_j_hat9)
        zhat = np.array([r_pred, b_pred[0], b_pred[1]])
        H = np.vstack((H_r, H_b))                 # (3, 9)

        H_pos = H[:, :3]
        R_consider = H_pos @ P_j9[:3, :3] @ H_pos.T
        R_eff = self.R_peer + R_consider

        y_raw = z_meas - zhat
        y_raw[1] = (y_raw[1] + np.pi) % (2 * np.pi) - np.pi

        S = H @ self.P_pred @ H.T + R_eff
        try:    nis = float(y_raw @ la.solve(S, y_raw))
        except la.LinAlgError: nis = float(y_raw @ y_raw)
        if nis > NIS_GATE_PEER:
            self.x = self.x_pred.copy(); self.P = self.P_pred.copy()
            return y_raw, S

        K = self.P_pred @ H.T @ la.inv(S)
        I9 = np.eye(9)
        self.x = self.x_pred + K @ y_raw
        KH = K @ H
        if self.cfg.use_joseph:
            self.P = (I9 - KH) @ self.P_pred @ (I9 - KH).T + K @ R_eff @ K.T
        else:
            self.P = (I9 - KH) @ self.P_pred
        self.P = 0.5 * (self.P + self.P.T)
        self.peer_update_count += 1
        self.has_been_updated = True

        nees = None
        if x_true_for_nees is not None:
            e = self.x[:6] - x_true_for_nees[:6]
            try:    nees = float(e @ la.solve(self.P[:6, :6] + 1e-12*np.eye(6), e))
            except la.LinAlgError: nees = float("nan")
            self.nees_history.append(nees)

        self.history.append(PeerNavSnapshot(
            t=t, dep_id=self.dep_id, x=self.x.copy(), P=self.P.copy(),
            innov=y_raw.copy(), S=S.copy(), nees=nees,
            meas_type=f"peer_{self.dep_id}_{j_id}"))
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        return y_raw, S

    # ─────────────────── MOS-CI (9-D) ───────────────────

    def oif_ci_fuse(self, t, z_polar, x_j_hat9, P_j9, j_id,
                    x_true_for_nees=None, nees_weight=0.0):
        x_pre = self.x.copy(); P_pre = self.P.copy()
        x_f, P_f, omega, diag = oif_ci_fuse_dmc(
            x_i=self.x, P_i=self.P, z_polar=z_polar,
            R_polar_diag=(self.cfg.sigma_range, self.cfg.sigma_az, self.cfg.sigma_el),
            x_j_hat=x_j_hat9, P_j=P_j9,
            nees_target=CI_NEES_TARGET, nees_weight=nees_weight,
            x_true_for_nees=x_true_for_nees)

        det_pre  = la.det(P_pre + 1e-12 * np.eye(9))
        det_post = la.det(P_f   + 1e-12 * np.eye(9))
        if det_pre > 0 and det_post / det_pre > CI_REJECT_DET_RATIO:
            self.ci_reject_count += 1
            return

        self.x = x_f; self.P = 0.5 * (P_f + P_f.T)
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        self.ci_count += 1
        self.omega_history.append(omega)

        nees = None
        if x_true_for_nees is not None:
            e = self.x[:6] - x_true_for_nees[:6]
            try:    nees = float(e @ la.solve(self.P[:6, :6] + 1e-12*np.eye(6), e))
            except la.LinAlgError: nees = float("nan")
        self.history.append(PeerNavSnapshot(
            t=t, dep_id=self.dep_id, x=self.x.copy(), P=self.P.copy(),
            omega=omega, nees=nees, meas_type=f"ci_{j_id}"))

    # ─────────────────── event-trigger broadcast ───────────────────

    def should_broadcast(self):
        if self.x_last_broadcast is None:
            return True
        dx = self.x - self.x_last_broadcast
        try:
            mahal = float(dx @ la.solve(
                self.P_last_broadcast + 1e-12 * np.eye(9), dx))
        except la.LinAlgError:
            return True
        if mahal > ISL_TRIGGER_MAHAL ** 2 * 9:
            return True
        tr_now  = np.trace(self.P[:3, :3])
        tr_last = np.trace(self.P_last_broadcast[:3, :3])
        if tr_last > 0 and abs(tr_now - tr_last) / tr_last > ISL_TRIGGER_TR_REL:
            return True
        return False

    def mark_broadcast(self, t):
        self.x_last_broadcast = self.x.copy()
        self.P_last_broadcast = self.P.copy()
        self.t_last_broadcast = t
        self.broadcast_count += 1


# ──────────────────────────────────────────────────────────────────────────────
#                            simulation driver
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate_burns(impulses: np.ndarray, t_arr: np.ndarray,
                     dt_nav: float, N: int) -> np.ndarray:
    """Sum every impulse into the sim-step window that contains it."""
    dv = np.zeros((N, 3))
    if impulses.size == 0:
        return dv
    half = 0.5 * dt_nav
    for b in range(impulses.shape[0]):
        tb = float(impulses[b, 0])
        k = int(np.searchsorted(t_arr, tb + half)) - 1
        if 0 <= k < N and abs(t_arr[k] - tb) <= half:
            dv[k] += impulses[b, 1:4]
    return dv


def run_peer_relnav_v4_sim(data_dir: Path,
                            N_dep: int = N_DEP,
                            meas_decim: int = MEAS_DECIM,
                            chief_ref_path: Optional[Path] = None
                            ) -> Dict[str, np.ndarray]:
    data_dir = Path(data_dir)

    # ── chief truth ──
    if chief_ref_path is None:
        chief_ref_path = _find_chief_reference()
    if chief_ref_path is None:
        raise FileNotFoundError(
            f"Could not find {CHIEF_REFERENCE_NAME} in any standard location.")
    print(f"\n[v4] chief truth source: {chief_ref_path}")
    chief_ref = load_chief_reference(chief_ref_path)

    # ── deputy bundles ──
    bundles = []
    for d in range(N_dep):
        p = data_dir / f"apophis_rendezvous_full_D{d}.npz"
        if not p.is_file():
            raise FileNotFoundError(f"Missing: {p}")
        bundles.append(load_apophis_bundle(p))

    # ── deployment times ──
    t_deploy     = np.array([float(bd["t"][0])  for bd in bundles])
    t_deploy_end = np.array([float(bd["t"][-1]) for bd in bundles])
    deploy_order = np.argsort(t_deploy).tolist()
    t_sim_start = float(t_deploy[deploy_order[0]])
    t_sim_end   = float(min(t_deploy_end))
    if CROP_HR > 0:
        t_sim_start += CROP_HR * 3600.0

    data_ref = bundles[deploy_order[0]]
    t_full = np.asarray(data_ref["t"], float).reshape(-1)
    idx_start = int(np.searchsorted(t_full, t_sim_start))
    idx_end   = int(np.searchsorted(t_full, t_sim_end, side="right")) - 1
    idx_sel = np.arange(idx_start, idx_end + 1, meas_decim)
    t_arr = t_full[idx_sel]
    N = len(t_arr)
    dt_nav = float(np.median(np.diff(t_arr)))

    deploy_step = np.clip(np.array(
        [int(np.searchsorted(t_arr, t_deploy[d])) for d in range(N_dep)],
        dtype=int), 0, N - 1)

    print(f"\n[Cascade MOS-CI v4 + DMC]")
    print(f"  Sim window: [{t_sim_start:.0f}, {t_sim_end:.0f}] s "
          f"({(t_sim_end - t_sim_start)/3600:.2f} h)")
    print(f"  Nav grid:   {N} steps, dt = {dt_nav:.1f} s")
    print(f"  DMC: tau_a = {DMC_TAU:.0f} s, sigma_a = {DMC_SIGMA_A:.1e} m/s^2")
    print(f"  Deployment order (earliest first):")
    for d in deploy_order:
        dt_rel = (t_deploy[d] - t_sim_start) / 60.0
        print(f"    D{d}:  t+{dt_rel:6.2f} min   (step k={deploy_step[d]})")

    # chief truth on sim grid
    chief_state_fn = make_state_fn(chief_ref, "rC_N", "vC_N")
    rC_grid = np.zeros((N, 3)); vC_grid = np.zeros((N, 3))
    for k in range(N):
        rC_grid[k], vC_grid[k] = chief_state_fn(float(t_arr[k]))

    # deputy truth + aggregated burns
    print("\n  Burn aggregation:")
    x_true_all = np.zeros((N_dep, N, 6))
    dv_all     = np.zeros((N_dep, N, 3))
    for d in range(N_dep):
        bd = bundles[d]
        t_full_d = np.asarray(bd["t"], float).reshape(-1)
        rD = np.asarray(bd["rD_N"], float).reshape(-1, 3)
        vD = np.asarray(bd["vD_N"], float).reshape(-1, 3)
        idx_d = np.clip(np.searchsorted(t_full_d, t_arr), 0, len(t_full_d) - 1)
        x_true_all[d] = eci_to_hill_traj(rD[idx_d], vD[idx_d], rC_grid, vC_grid)
        imp = np.asarray(bd["impulses"], float).reshape(-1, 4)
        dv_all[d] = _aggregate_burns(imp, t_arr, dt_nav, N)
        n_burns  = int(imp.shape[0])
        n_caught = int(np.sum(np.linalg.norm(dv_all[d], axis=1) > 0))
        if n_burns > 0:
            print(f"    D{d}: {n_burns:3d} burns  ->  {n_caught:3d} sim windows  "
                  f"(captured ΣΔV = {np.linalg.norm(dv_all[d].sum(axis=0)):.4f} m/s)")

    cfg = PeerRelNavConfigV4(dt=dt_nav)
    ekfs: List[PeerRelNavEKFV4] = [
        PeerRelNavEKFV4(d, cfg, chief_state_fn) for d in range(N_dep)
    ]
    initial_x = []
    for d in range(N_dep):
        ks = deploy_step[d]
        bias = np.array([200.0, -150.0, 80.0, 0.06, -0.04, 0.02])
        initial_x.append(x_true_all[d, ks] + bias * (1.0 + 0.1 * d))

    sched = CrosslinkScheduler(N_dep)
    rng = np.random.default_rng(42)
    sigma_az_rad = np.deg2rad(AZ_SIGMA_DEG)
    sigma_el_rad = np.deg2rad(EL_SIGMA_DEG)

    active = np.zeros(N_dep, dtype=bool)
    omega_hist: List[Tuple[float, int, int, float]] = []
    x_est_hist  = np.full((N_dep, N, 9), np.nan)
    P_diag_hist = np.full((N_dep, N, 9), np.nan)
    nees_hist   = np.full((N_dep, N), np.nan)
    accel_hist  = np.full((N_dep, N, 3), np.nan)
    active_hist = np.zeros((N_dep, N), dtype=bool)

    print(f"\n[Cascade + DMC] Running ...")
    for k in range(N):
        t = float(t_arr[k])

        # 1) deploy + bootstrap
        for d in range(N_dep):
            if not active[d] and k >= deploy_step[d]:
                ekfs[d].reset(initial_x[d])
                ekfs[d].deploy_t = t; ekfs[d].deploy_step = k
                active[d] = True
                seniors = [s for s in range(N_dep) if active[s] and s != d]
                seniors.sort(key=lambda s: np.trace(ekfs[s].P[:3, :3]))
                used = []
                for s in seniors[:BOOTSTRAP_CI_BURST_N]:
                    z_polar, ok = generate_peer_measurement(
                        rng, x_true_all[d, k], x_true_all[s, k],
                        RANGE_SIGMA, sigma_az_rad, sigma_el_rad,
                        p_dropout=(0.0 if BOOTSTRAP_NO_DROPOUT else P_PEER_DROPOUT),
                        p_outlier=0.0)
                    if not ok:
                        continue
                    ekfs[d].x_pred = ekfs[d].x.copy()
                    ekfs[d].P_pred = ekfs[d].P.copy()
                    ekfs[d].oif_ci_fuse(t, z_polar,
                                        ekfs[s].x.copy(), ekfs[s].P.copy(), s,
                                        x_true_for_nees=x_true_all[d, k],
                                        nees_weight=CI_NEES_PENALTY)
                    used.append(s)
                ekfs[d].bootstrap_done = True
                trP = np.sqrt(np.trace(ekfs[d].P[:3, :3]))
                print(f"  [t+{(t-t_arr[0])/60:6.2f} min] D{d} DEPLOYED  "
                      f"bootstrap seniors={used}  -> sigma_pos~{trP:.1f} m")

        # 2) predict
        for d in range(N_dep):
            if not active[d]:
                continue
            if not ekfs[d].has_been_updated and ekfs[d].peer_update_count == 0 \
               and ekfs[d].chief_obs_count == 0 and ekfs[d].ci_count == 0:
                ekfs[d].x_pred = ekfs[d].x.copy()
                ekfs[d].P_pred = ekfs[d].P.copy()
                ekfs[d].t_last = t
            else:
                ekfs[d].predict(t, burn_dv_H=dv_all[d, k])

        # 3) chief observation
        for d in range(N_dep):
            if not active[d]:
                continue
            if rng.random() < CHIEF_OBS_DROPOUT:
                continue
            z = x_true_all[d, k, :3] + CHIEF_OBS_SIGMA_POS * rng.normal(size=3)
            ekfs[d].chief_observation_update(t, z, x_true_for_nees=x_true_all[d, k])

        # 4) peer measurements + MOS-CI
        pairs = sched.pairs_for_step(k)
        for (i, j) in pairs:
            if not (active[i] and active[j]):
                continue
            z_ij, ok_ij = generate_peer_measurement(
                rng, x_true_all[i, k], x_true_all[j, k],
                RANGE_SIGMA, sigma_az_rad, sigma_el_rad)
            z_ji, ok_ji = generate_peer_measurement(
                rng, x_true_all[j, k], x_true_all[i, k],
                RANGE_SIGMA, sigma_az_rad, sigma_el_rad)

            if ekfs[i].should_broadcast(): ekfs[i].mark_broadcast(t)
            if ekfs[j].should_broadcast(): ekfs[j].mark_broadcast(t)
            xj_b = (ekfs[j].x_last_broadcast.copy()
                    if ekfs[j].x_last_broadcast is not None else ekfs[j].x.copy())
            Pj_b = (ekfs[j].P_last_broadcast.copy()
                    if ekfs[j].P_last_broadcast is not None else ekfs[j].P.copy())
            xi_b = (ekfs[i].x_last_broadcast.copy()
                    if ekfs[i].x_last_broadcast is not None else ekfs[i].x.copy())
            Pi_b = (ekfs[i].P_last_broadcast.copy()
                    if ekfs[i].P_last_broadcast is not None else ekfs[i].P.copy())

            if ok_ij:
                ekfs[i].peer_update(t, z_ij, xj_b, Pj_b, j,
                                    x_true_for_nees=x_true_all[i, k])
            if ok_ji:
                ekfs[j].peer_update(t, z_ji, xi_b, Pi_b, i,
                                    x_true_for_nees=x_true_all[j, k])

            if USE_CI_FUSION and (k % CI_EVERY_N_STEPS == 0):
                if ok_ij:
                    ekfs[i].oif_ci_fuse(t, z_ij, xj_b, Pj_b, j,
                                        x_true_for_nees=x_true_all[i, k],
                                        nees_weight=CI_NEES_PENALTY)
                    if ekfs[i].omega_history:
                        omega_hist.append((t, i, j, ekfs[i].omega_history[-1]))
                if ok_ji:
                    ekfs[j].oif_ci_fuse(t, z_ji, xi_b, Pi_b, i,
                                        x_true_for_nees=x_true_all[j, k],
                                        nees_weight=CI_NEES_PENALTY)
                    if ekfs[j].omega_history:
                        omega_hist.append((t, j, i, ekfs[j].omega_history[-1]))

        # 5) record
        for d in range(N_dep):
            active_hist[d, k] = active[d]
            if active[d]:
                x_est_hist[d, k]  = ekfs[d].x.copy()
                P_diag_hist[d, k] = np.diag(ekfs[d].P)
                accel_hist[d, k]  = ekfs[d].x[6:9].copy()
                e = ekfs[d].x[:6] - x_true_all[d, k]
                try:
                    nees_hist[d, k] = float(e @ la.solve(
                        ekfs[d].P[:6, :6] + 1e-12*np.eye(6), e))
                except la.LinAlgError:
                    pass

    t_hr = (t_arr - t_arr[0]) / 3600.0

    print("\n" + "=" * 100)
    print("  CASCADE-ANCHORED MOS-CI + DMC  —  RESULTS")
    print("=" * 100)
    print(f"\n┌─── Per-Deputy Cascade+DMC Summary ─────────────────────────────────────────────────────────────┐")
    print(f"│ {'Dep':>3} │ {'deploy@':>8} │ {'RMS pos':>8} │ {'RMS vel':>8} │ "
          f"{'ChiefObs':>8} │ {'Peers':>5} │ {'CI':>4} │ {'CIrej':>5} │ "
          f"{'<NEES>':>7} │ {'<ω>':>5} │ {'σ_pos_f':>8} │ {'|a|_f':>9} │")
    for d in range(N_dep):
        ks = deploy_step[d]
        m = active_hist[d].copy()
        m[:ks + max(1, int(180.0 / dt_nav))] = False
        if not m.any():
            m = active_hist[d]
        err = x_est_hist[d, :, :6] - x_true_all[d]
        rms_pos = float(np.sqrt(np.nanmean(np.sum(err[m, :3]**2, axis=1))))
        rms_vel = float(np.sqrt(np.nanmean(np.sum(err[m, 3:6]**2, axis=1))))
        nees_avg = float(np.nanmean(nees_hist[d, m]))
        omega_avg = (float(np.mean(ekfs[d].omega_history))
                     if ekfs[d].omega_history else float("nan"))
        sigma_pos_f = float(np.sqrt(np.nansum(P_diag_hist[d, -1, :3])))
        a_final = ekfs[d].x[6:9]
        deploy_min = (t_deploy[d] - t_sim_start) / 60.0
        print(f"│  D{d} │ {deploy_min:7.2f}m │ {rms_pos:8.2f} │ {rms_vel:8.4f} │ "
              f"{ekfs[d].chief_obs_count:8d} │ {ekfs[d].peer_update_count:5d} │ "
              f"{ekfs[d].ci_count:4d} │ {ekfs[d].ci_reject_count:5d} │ "
              f"{nees_avg:7.2f} │ {omega_avg:5.2f} │ {sigma_pos_f:8.2f} │ "
              f"{la.norm(a_final):8.2e}m/s² │")
    print(f"└────────────────────────────────────────────────────────────────────────────────────────────────┘")

    return dict(
        t_arr=t_arr, t_hr=t_hr,
        x_true_all=x_true_all, x_est_hist=x_est_hist,
        P_diag_hist=P_diag_hist, nees_hist=nees_hist,
        accel_hist=accel_hist, omega_hist=omega_hist,
        active_hist=active_hist, deploy_step=deploy_step,
        t_deploy=t_deploy, N_dep=N_dep, ekfs=ekfs,
    )


# ──────────────────────────────────────────────────────────────────────────────
#                                 plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_results_v4(res):
    t_hr        = res["t_hr"]
    x_true_all  = res["x_true_all"]
    x_est_hist  = res["x_est_hist"][..., :6]
    P_diag      = res["P_diag_hist"][..., :6]
    nees_hist   = res["nees_hist"]
    accel_hist  = res["accel_hist"]
    omega_hist  = res["omega_hist"]
    active_hist = res["active_hist"]
    deploy_step = res["deploy_step"]
    N_dep       = res["N_dep"]

    if SAVE_FIGS:
        FIG_DIR.mkdir(parents=True, exist_ok=True)

    colors = plt.cm.tab10(np.linspace(0, 1, N_dep))
    labels = ["R", "T", "N"]

    # 1) Position covariance trace
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Cascade MOS-CI + DMC: position covariance trace", fontsize=14)
    for d in range(N_dep):
        m = active_hist[d]
        ax.plot(t_hr[m], np.sqrt(np.nansum(P_diag[d, m, :3], axis=1)),
                lw=0.9, color=colors[d], label=f"D{d}", alpha=0.85)
        ax.axvline(t_hr[deploy_step[d]], color=colors[d], ls=":", lw=0.6, alpha=0.5)
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    ax.set_ylabel("sqrt(tr(P_pos)) [m]"); ax.set_xlabel("Time [h]")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "cov_trace.png", dpi=150)
    plt.show()

    # 2) per-deputy errors vs 3sigma
    for d in range(N_dep):
        m = active_hist[d]
        if not m.any():
            continue
        err = x_est_hist[d] - x_true_all[d]
        three_sigma = 3.0 * np.sqrt(np.maximum(P_diag[d], 0.0))
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        fig.suptitle(f"v4+DMC — D{d} errors vs 3sigma", fontsize=14)
        tt = t_hr[m]
        for i in range(3):
            ax = axes[0, i]
            ax.scatter(tt, err[m, i], s=4, alpha=0.5)
            ax.plot(tt,  three_sigma[m, i], "r--", lw=1)
            ax.plot(tt, -three_sigma[m, i], "r--", lw=1)
            ax.set_ylabel(f"ρ_{labels[i]} [m]"); ax.grid(True, alpha=0.3)
            ax.set_title(f"Position {labels[i]}")
        for i in range(3):
            ax = axes[1, i]
            ax.scatter(tt, err[m, 3+i], s=4, alpha=0.5)
            ax.plot(tt,  three_sigma[m, 3+i], "r--", lw=1)
            ax.plot(tt, -three_sigma[m, 3+i], "r--", lw=1)
            ax.set_ylabel(f"ρ̇_{labels[i]} [m/s]"); ax.set_xlabel("Time [h]")
            ax.grid(True, alpha=0.3); ax.set_title(f"Velocity {labels[i]}")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if SAVE_FIGS: fig.savefig(FIG_DIR / f"d{d}_err.png", dpi=150)
        plt.show()

    # 3) NEES (target ~ 6 for the 6-D pos+vel sub-state)
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("v4+DMC: NEES per deputy (pos+vel sub-state, target ~ 6)",
                 fontsize=14)
    for d in range(N_dep):
        m = active_hist[d]
        ax.plot(t_hr[m], nees_hist[d, m], lw=0.7, color=colors[d],
                label=f"D{d}", alpha=0.7)
    ax.axhline(6.0, color="k", ls=":", lw=1, label="dof = 6")
    ax.axhline(12.6, color="r", ls=":", lw=1, label="chi2(6) 95%")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    ax.set_ylabel("NEES"); ax.set_xlabel("Time [h]")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "nees.png", dpi=150)
    plt.show()

    # 4) DMC acceleration estimates
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("v4+DMC: estimated unmodelled acceleration per axis",
                 fontsize=14)
    for ax_i, axis in enumerate(["a_R", "a_T", "a_N"]):
        ax = axes[ax_i]
        for d in range(N_dep):
            m = active_hist[d]
            ax.plot(t_hr[m], accel_hist[d, m, ax_i],
                    lw=0.7, color=colors[d], label=f"D{d}" if ax_i == 0 else None,
                    alpha=0.7)
        ax.set_ylabel(f"{axis} [m/s²]"); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [h]")
    if N_dep > 0: axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "accel.png", dpi=150)
    plt.show()

    # 5) omega histogram
    if omega_hist:
        ws = np.array([w for (_, _, _, w) in omega_hist])
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle("v4+DMC: omega* distribution")
        ax.hist(ws, bins=30, edgecolor="k", alpha=0.7)
        ax.axvline(np.mean(ws), color="r", ls="--", lw=1,
                   label=f"mean = {np.mean(ws):.3f}")
        ax.set_xlabel("omega*"); ax.set_ylabel("count"); ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        if SAVE_FIGS: fig.savefig(FIG_DIR / "omega.png", dpi=150)
        plt.show()

    # 6) Combined 3-D truth vs estimate
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("v4+DMC: TRUTH vs ESTIMATE (Hill frame)", fontsize=14)
    for d in range(N_dep):
        m = active_hist[d]
        if not m.any():
            continue
        xt = x_true_all[d, m]; xe = x_est_hist[d, m]
        ax.plot(xt[:, 0], xt[:, 1], xt[:, 2], "-", color=colors[d],
                lw=1.3, alpha=0.85)
        ax.plot(xe[:, 0], xe[:, 1], xe[:, 2], "--", color=colors[d],
                lw=1.0, alpha=0.6)
        ax.scatter(xt[0, 0],  xt[0, 1],  xt[0, 2],
                   color=colors[d], s=40, marker="o", edgecolor="k")
        ax.scatter(xt[-1, 0], xt[-1, 1], xt[-1, 2],
                   color=colors[d], s=80, marker="*", edgecolor="k",
                   label=f"D{d}")
    ax.scatter(0, 0, 0, color="k", s=140, marker="X",
               label="Chief (origin)")
    ax.set_xlabel("R [m]"); ax.set_ylabel("T [m]"); ax.set_zlabel("N [m]")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "all_3d.png", dpi=150)
    plt.show()

    # 7) per-deputy 3-D panels
    rows, cols = 2, 4
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("v4+DMC: per-deputy 3-D truth (solid) vs estimate (dashed)",
                 fontsize=14)
    for d in range(N_dep):
        ax = fig.add_subplot(rows, cols, d + 1, projection="3d")
        m = active_hist[d]
        if not m.any():
            ax.set_title(f"D{d} (no data)"); continue
        xt = x_true_all[d, m]; xe = x_est_hist[d, m]
        ax.plot(xt[:, 0], xt[:, 1], xt[:, 2], "b-",  lw=1.3, label="truth")
        ax.plot(xe[:, 0], xe[:, 1], xe[:, 2], "r--", lw=1.0, label="est")
        ax.scatter(0, 0, 0, color="k", s=60, marker="X")
        ax.set_xlabel("R [m]"); ax.set_ylabel("T [m]"); ax.set_zlabel("N [m]")
        ax.set_title(f"D{d}  (deploys k={deploy_step[d]})")
        ax.legend(fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "per_dep_3d.png", dpi=150)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
#                                 entry point
# ──────────────────────────────────────────────────────────────────────────────

def _find_data_dir() -> Path:
    candidates = [SCRIPT_DIR, ROOT.parent if ROOT else SCRIPT_DIR]
    for d in candidates:
        if (d / "apophis_rendezvous_full_D0.npz").is_file():
            return d
    raise FileNotFoundError(
        "Cannot find apophis_rendezvous_full_D0.npz in any known location.")


if __name__ == "__main__":
    data_dir = _find_data_dir()
    print(f"Using data from: {data_dir}")
    results = run_peer_relnav_v4_sim(data_dir)
    plot_results_v4(results)
