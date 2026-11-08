#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_peer_relnav.py
============================

**Peer-to-Peer Decentralised Relative Navigation with Covariance Intersection**

Novel navigation architecture where each deputy estimates its own Hill state
by observing *other deputies* (not a ground station, not the chief directly).
Deputies share their local estimates over inter-satellite links, and fuse the
indirect information through Covariance Intersection (CI) — a conservative
but consistent fusion method that requires no cross-covariance bookkeeping.

Architecture
------------

Each deputy *i* runs a **local EKF** with state:

    x_i = [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N]^T    (Hill / chief LVLH)

Process model: same CW dynamics as HillEKF.

**Peer measurements** (deputy i observes deputy j):

    1) Inter-deputy range:
           z_range = ||ρ_j - ρ_i||  + v_r

    2) Inter-deputy LOS bearing (2 angles):
           z_az  = atan2(Δρ_T, Δρ_R)  + v_az
           z_el  = asin(Δρ_N / ||Δρ||) + v_el

    where Δρ = ρ_j - ρ_i (relative position in Hill frame).

    The Jacobians H_range, H_bearing are computed analytically with respect
    to x_i (own state) given j's communicated estimate x̂_j.

**Covariance Intersection (CI) fusion step**:

    After deputy i incorporates its own peer measurements, deputy j shares
    (x̂_j, P_j). Deputy i can infer an *indirect* estimate of its own state:

        x̃_i|j = x̂_j - z_rel_ij    (indirect self-estimate from j's viewpoint)
        P̃_i|j = P_j + R_rel         (inflated by relative measurement noise)

    The local and indirect estimates are fused via CI:

        P_fused^{-1} = ω P_i^{-1} + (1 - ω) P̃_{i|j}^{-1}
        x_fused = P_fused (ω P_i^{-1} x_i + (1 - ω) P̃_{i|j}^{-1} x̃_{i|j})

    where ω ∈ [0, 1] minimises det(P_fused) or tr(P_fused).

Scheduling
----------
Uses CrosslinkScheduler (round-robin 1-factorisation of K_N) to decide
which pairs communicate at each step.

Expected data
-------------
Apophis rendezvous bundles: apophis_rendezvous_full_D0..D7.npz

"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
#                          locate ARISE root
# ──────────────────────────────────────────────────────────────────────────────

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
    HillEKFConfig, dcm_lvlh, hill_rate, cw_state_transition,
)
from GNC.Navigation.crosslink_scheduler import CrosslinkScheduler

# ──────────────────────────────────────────────────────────────────────────────
#                              TOGGLES
# ──────────────────────────────────────────────────────────────────────────────

N_DEP       = 8
MEAS_DECIM  = 100        # decimate truth to nav cadence (was 100 — more steps)
CROP_HR     = 0.5       # skip first 0.5 h after deploy

# Peer sensor noise (1σ)
RANGE_SIGMA   = 5.0     # [m]  inter-deputy range
AZ_SIGMA_DEG  = 0.3     # [deg] azimuth bearing
EL_SIGMA_DEG  = 0.3     # [deg] elevation bearing

# Absolute GPS baseline (every deputy, every step — noisy but keeps absolute track)
# Peer measurements then refine BEYOND what GPS alone can achieve.
USE_ANCHOR_GPS      = True
GPS_SIGMA_POS       = 50.0   # [m]  degraded GPS (deep space proxy)
GPS_SIGMA_VEL       = 0.05   # [m/s]
GPS_ALL_DEPUTIES    = True   # True = every deputy gets GPS every step (baseline)
                             # False = round-robin one deputy at a time
ANCHOR_CADENCE      = 1      # every N steps (only used if GPS_ALL_DEPUTIES=False)

# Dropout / outlier on peer link
P_PEER_DROPOUT  = 0.05
P_PEER_OUTLIER  = 0.01
OUTLIER_SCALE   = 6.0

# NIS gating threshold (chi2 dof=3 at 99.5%)
NIS_GATE_PEER = 12.84

# CI fusion toggle
USE_CI_FUSION = True
CI_EVERY_N_STEPS = 20   # only CI-fuse every N steps (avoid rumor propagation)

# Covariance floor — prevents P from collapsing below physical limits
COV_FLOOR_POS = 10.0    # [m]   minimum 1σ position
COV_FLOOR_VEL = 0.01    # [m/s] minimum 1σ velocity

# Plotting
SAVE_FIGS      = False
FIG_DIR        = SCRIPT_DIR / "peer_relnav_figs"

# ──────────────────────────────────────────────────────────────────────────────
#                       data loading (same pattern as your existing scripts)
# ──────────────────────────────────────────────────────────────────────────────

def load_apophis_bundle(path: Path) -> Dict[str, np.ndarray]:
    wanted = ("t", "rD_N", "vD_N", "rC_N", "vC_N", "rA_N", "vA_N", "impulses")
    out: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as npz:
        for k in wanted:
            if k in npz:
                out[k] = npz[k]
    if "impulses" not in out:
        out["impulses"] = np.zeros((0, 4), dtype=float)
    return out


def make_state_fn(data: Dict[str, np.ndarray], r_key: str, v_key: str):
    t_arr = np.asarray(data["t"], float).reshape(-1)
    r = np.asarray(data[r_key], float).reshape(-1, 3)
    v = np.asarray(data[v_key], float).reshape(-1, 3)

    def interp_vec(t: float, arr: np.ndarray) -> np.ndarray:
        if t <= t_arr[0]:
            return arr[0].copy()
        if t >= t_arr[-1]:
            return arr[-1].copy()
        j = np.searchsorted(t_arr, t) - 1
        f = (t - t_arr[j]) / (t_arr[j + 1] - t_arr[j] + 1e-15)
        return (1.0 - f) * arr[j] + f * arr[j + 1]

    def fn(t: float) -> Tuple[np.ndarray, np.ndarray]:
        return interp_vec(t, r), interp_vec(t, v)
    return fn


def hill_state_from_eci(rD, vD, rC, vC) -> np.ndarray:
    rC = np.asarray(rC, float).reshape(3)
    vC = np.asarray(vC, float).reshape(3)
    rD = np.asarray(rD, float).reshape(3)
    vD = np.asarray(vD, float).reshape(3)
    C_HN = dcm_lvlh(rC, vC)
    n = hill_rate(rC, vC)
    rho = C_HN @ (rD - rC)
    rhod = C_HN @ (vD - vC) - np.cross([0.0, 0.0, n], rho)
    return np.hstack((rho, rhod))


def eci_to_hill_traj(rD, vD, rC, vC) -> np.ndarray:
    N = rD.shape[0]
    out = np.zeros((N, 6))
    for k in range(N):
        out[k] = hill_state_from_eci(rD[k], vD[k], rC[k], vC[k])
    return out


def build_burn_sequence(impulses, t_arr, dt):
    impulses = np.asarray(impulses, float).reshape(-1, 4)
    dv = np.zeros((len(t_arr), 3))
    if impulses.size == 0:
        return dv
    impulses = impulses[np.argsort(impulses[:, 0])]
    idx = 0
    for k, t in enumerate(t_arr):
        while idx < impulses.shape[0] and impulses[idx, 0] < t - 0.5 * dt:
            idx += 1
        if idx < impulses.shape[0] and abs(impulses[idx, 0] - t) <= 0.5 * dt:
            dv[k, :] = impulses[idx, 1:4]
            idx += 1
    return dv


# ──────────────────────────────────────────────────────────────────────────────
#                  Peer Measurement Model (range + bearing in Hill)
# ──────────────────────────────────────────────────────────────────────────────

def peer_range_pred_and_jac(
    x_i: np.ndarray,
    x_j_hat: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Predicted inter-deputy range and Jacobian wrt x_i.

        h = ||ρ_j - ρ_i||

        ∂h/∂x_i = [-û^T, 0^T]   (1×6)

    where û = (ρ_j - ρ_i) / ||ρ_j - ρ_i||, using x̂_j as a known constant.
    """
    rho_i = x_i[:3]
    rho_j = x_j_hat[:3]
    d = rho_j - rho_i
    s = la.norm(d)

    if s < 1e-9:
        return 0.0, np.zeros((1, 6))

    u = d / s
    H = np.zeros((1, 6))
    H[0, :3] = -u  # ∂||ρj - ρi|| / ∂ρi = -û
    return float(s), H


def peer_bearing_pred_and_jac(
    x_i: np.ndarray,
    x_j_hat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicted azimuth and elevation of deputy j as seen from deputy i,
    both in the chief Hill frame.

        Δρ = ρ_j - ρ_i
        az  = atan2(Δρ_T, Δρ_R)
        el  = asin(Δρ_N / ||Δρ||)

    Jacobians are wrt x_i (treating x̂_j as constant).

    Returns
    -------
    z_pred : (2,)  [az, el]  [rad]
    H      : (2,6) Jacobian
    """
    rho_i = x_i[:3]
    rho_j = x_j_hat[:3]
    d = rho_j - rho_i  # [R, T, N]
    s = la.norm(d)

    if s < 1e-9:
        return np.zeros(2), np.zeros((2, 6))

    dR, dT, dN = d
    s_xy = np.sqrt(dR**2 + dT**2)

    az = np.arctan2(dT, dR)
    el = np.arcsin(np.clip(dN / s, -1.0, 1.0))

    H = np.zeros((2, 6))

    # ∂az/∂(Δρ) = [-Δρ_T, Δρ_R, 0] / (Δρ_R² + Δρ_T²)
    # ∂az/∂ρ_i = -∂az/∂(Δρ)
    if s_xy > 1e-12:
        daz_dD = np.array([-dT, dR, 0.0]) / (s_xy**2)
        H[0, :3] = -daz_dD  # ∂az/∂ρ_i = -∂az/∂Δρ

    # ∂el/∂(Δρ):
    #   el = asin(dN / s)
    #   ∂el/∂dR = -dN*dR / (s^2 * s_xy), etc.
    cos_el = np.cos(el)
    if abs(cos_el) > 1e-12 and s > 1e-9:
        # d(dN/s)/d(dR) = -dN*dR / s^3
        # d(dN/s)/d(dT) = -dN*dT / s^3
        # d(dN/s)/d(dN) = (s^2 - dN^2) / s^3 = (dR^2+dT^2)/s^3
        s3 = s**3
        del_dD = np.array([
            -dN * dR / s3,
            -dN * dT / s3,
            (dR**2 + dT**2) / s3,
        ]) / cos_el
        H[1, :3] = -del_dD  # ∂el/∂ρ_i = -∂el/∂Δρ

    z_pred = np.array([az, el])
    return z_pred, H


def generate_peer_measurement(
    rng: np.random.Generator,
    x_true_i: np.ndarray,
    x_true_j: np.ndarray,
    sigma_r: float,
    sigma_az: float,
    sigma_el: float,
    p_dropout: float = P_PEER_DROPOUT,
    p_outlier: float = P_PEER_OUTLIER,
    outlier_scale: float = OUTLIER_SCALE,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Simulate a peer measurement from deputy i observing deputy j.

    Returns
    -------
    z : (3,) [range, az, el] or None if dropout
    valid : bool
    """
    if rng.random() < p_dropout:
        return None, False

    d = x_true_j[:3] - x_true_i[:3]
    s = la.norm(d)
    if s < 1e-6:
        return None, False

    dR, dT, dN = d
    az_true = np.arctan2(dT, dR)
    el_true = np.arcsin(np.clip(dN / s, -1.0, 1.0))

    nr = sigma_r * rng.normal()
    naz = sigma_az * rng.normal()
    nel = sigma_el * rng.normal()

    if rng.random() < p_outlier:
        nr *= outlier_scale
        naz *= outlier_scale
        nel *= outlier_scale

    z = np.array([s + nr, az_true + naz, el_true + nel])
    return z, True


# ──────────────────────────────────────────────────────────────────────────────
#                    Covariance Intersection (CI) Fusion
# ──────────────────────────────────────────────────────────────────────────────

def _ci_cost(omega: float,
             P_a_inv: np.ndarray,
             P_b_inv: np.ndarray) -> float:
    """
    Cost = log det( ω P_a^{-1} + (1-ω) P_b^{-1} )^{-1}
         = -log det( ω P_a^{-1} + (1-ω) P_b^{-1} )

    We minimise this (i.e. minimise the fused covariance determinant).
    """
    P_fused_inv = omega * P_a_inv + (1.0 - omega) * P_b_inv
    sign, logdet = la.slogdet(P_fused_inv)
    if sign <= 0:
        return 1e30
    return -logdet


def covariance_intersection(
    x_a: np.ndarray, P_a: np.ndarray,
    x_b: np.ndarray, P_b: np.ndarray,
    n_search: int = 11,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Covariance Intersection: fuse two independent estimates (x_a, P_a) and
    (x_b, P_b) without knowing their cross-covariance.

    Finds ω* ∈ [0, 1] minimising det(P_fused), then returns:

        P_f^{-1} = ω P_a^{-1} + (1-ω) P_b^{-1}
        x_f      = P_f ( ω P_a^{-1} x_a + (1-ω) P_b^{-1} x_b )

    Parameters
    ----------
    n_search : int
        Number of grid points for ω search (then golden-section refinement).

    Returns
    -------
    x_f, P_f : fused state and covariance
    """
    x_a = np.asarray(x_a, float).reshape(-1)
    x_b = np.asarray(x_b, float).reshape(-1)
    P_a = np.asarray(P_a, float)
    P_b = np.asarray(P_b, float)

    # Regularise tiny eigenvalues
    eps = 1e-10
    P_a_reg = P_a + eps * np.eye(P_a.shape[0])
    P_b_reg = P_b + eps * np.eye(P_b.shape[0])

    try:
        P_a_inv = la.inv(P_a_reg)
        P_b_inv = la.inv(P_b_reg)
    except la.LinAlgError:
        # If inversion fails, return the tighter of the two
        if la.det(P_a) < la.det(P_b):
            return x_a.copy(), P_a.copy()
        else:
            return x_b.copy(), P_b.copy()

    # Grid search for optimal ω
    omegas = np.linspace(0.01, 0.99, n_search)
    costs = [_ci_cost(w, P_a_inv, P_b_inv) for w in omegas]
    best_idx = int(np.argmin(costs))

    # Golden-section refinement around the best grid point
    lo = omegas[max(best_idx - 1, 0)]
    hi = omegas[min(best_idx + 1, n_search - 1)]
    gr = (np.sqrt(5.0) + 1.0) / 2.0

    for _ in range(20):
        c = hi - (hi - lo) / gr
        d = lo + (hi - lo) / gr
        if _ci_cost(c, P_a_inv, P_b_inv) < _ci_cost(d, P_a_inv, P_b_inv):
            hi = d
        else:
            lo = c

    omega_star = 0.5 * (lo + hi)

    P_f_inv = omega_star * P_a_inv + (1.0 - omega_star) * P_b_inv
    try:
        P_f = la.inv(P_f_inv)
    except la.LinAlgError:
        P_f = P_a.copy()
        return x_a.copy(), P_f

    x_f = P_f @ (omega_star * P_a_inv @ x_a +
                  (1.0 - omega_star) * P_b_inv @ x_b)

    # Symmetrise
    P_f = 0.5 * (P_f + P_f.T)

    return x_f, P_f


# ──────────────────────────────────────────────────────────────────────────────
#                     PeerRelNavEKF (per-deputy local filter)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PeerRelNavConfig:
    dt: float = 1.0

    # CW process noise (reflects unmodelled dynamics: J2, SRP, nonlinear gravity)
    sigma_proc_pos: float = 2.0      # [m]
    sigma_proc_vel: float = 5e-3     # [m/s]

    # Peer measurement noise (filter-assumed)
    sigma_range: float = RANGE_SIGMA       # [m]
    sigma_az:    float = np.deg2rad(AZ_SIGMA_DEG)   # [rad]
    sigma_el:    float = np.deg2rad(EL_SIGMA_DEG)   # [rad]

    # GPS (degraded) measurement noise
    sigma_gps_pos: float = GPS_SIGMA_POS   # [m]
    sigma_gps_vel: float = GPS_SIGMA_VEL   # [m/s]

    # Initial covariance
    sigma_init_pos: float = 1500.0   # [m]
    sigma_init_vel: float = 0.5      # [m/s]

    # ΔV uncertainty
    sigma_dv: float = 0.05           # [m/s]

    use_joseph: bool = True


@dataclass
class PeerNavSnapshot:
    t: float
    dep_id: int
    x: np.ndarray
    P: np.ndarray
    innov: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None
    meas_type: str = ""              # "peer_i_j", "gps", "ci_j"


class PeerRelNavEKF:
    """
    Local EKF for a single deputy in the peer-to-peer architecture.

    State: x = [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N]^T in chief Hill frame.

    This filter is updated by:
      1. peer_update():  range + bearing to another deputy
      2. gps_update():   sparse absolute GPS fix (anchor measurements)
      3. ci_fuse():      covariance intersection with indirect estimate from peer
    """

    def __init__(
        self,
        dep_id: int,
        cfg: PeerRelNavConfig,
        chief_state_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]],
    ):
        self.dep_id = dep_id
        self.cfg = cfg
        self.chief_state_fn = chief_state_fn

        # Process noise
        qp2 = cfg.sigma_proc_pos**2
        qv2 = cfg.sigma_proc_vel**2
        self.Q = np.diag([qp2, qp2, qp2, qv2, qv2, qv2])

        # Peer R template (range + az + el)
        self.R_peer = np.diag([
            cfg.sigma_range**2,
            cfg.sigma_az**2,
            cfg.sigma_el**2,
        ])

        # GPS R template
        rp2 = cfg.sigma_gps_pos**2
        rv2 = cfg.sigma_gps_vel**2
        self.R_gps = np.diag([rp2, rp2, rp2, rv2, rv2, rv2])

        # ΔV bump
        self.Q_dv = (cfg.sigma_dv**2) * np.eye(3)

        # State
        pp2 = cfg.sigma_init_pos**2
        pv2 = cfg.sigma_init_vel**2
        self.P_init = np.diag([pp2, pp2, pp2, pv2, pv2, pv2])

        self.x = np.zeros(6)
        self.P = self.P_init.copy()
        self.x_pred = np.zeros(6)
        self.P_pred = self.P.copy()

        self.t_last: Optional[float] = None

        # Diagnostics
        self.history: List[PeerNavSnapshot] = []
        self.ci_count = 0
        self.peer_update_count = 0

    def reset(self, x0: np.ndarray, P0: Optional[np.ndarray] = None):
        self.x = np.asarray(x0, float).reshape(6)
        self.P = np.asarray(P0, float).reshape(6, 6) if P0 is not None else self.P_init.copy()
        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()
        self.t_last = None
        self.history.clear()
        self.ci_count = 0
        self.peer_update_count = 0

    # ──────────────── predict ────────────────

    def predict(self, t_abs: float, burn_dv_H: Optional[np.ndarray] = None):
        dt = self.cfg.dt
        if self.t_last is None:
            self.t_last = t_abs - dt

        t_mid = 0.5 * (self.t_last + t_abs)
        rC, vC = self.chief_state_fn(t_mid)
        n = hill_rate(rC, vC)
        Phi = cw_state_transition(n, dt)

        self.x_pred = Phi @ self.x
        self.P_pred = Phi @ self.P @ Phi.T + self.Q

        if burn_dv_H is not None:
            dv = np.asarray(burn_dv_H, float).reshape(3)
            self.x_pred[3:6] += dv
            self.P_pred[3:6, 3:6] += self.Q_dv

        self.t_last = t_abs

    # ──────────────── covariance floor ────────────────

    def _enforce_cov_floor(self):
        """
        Prevent covariance collapse by enforcing a minimum diagonal.

        This guards against the "rumor propagation" failure mode where
        repeated CI fusions with correlated peers shrink P below the
        actual estimation accuracy, causing the filter to reject all
        future corrective measurements via NIS gating.
        """
        floor = np.array([
            COV_FLOOR_POS**2, COV_FLOOR_POS**2, COV_FLOOR_POS**2,
            COV_FLOOR_VEL**2, COV_FLOOR_VEL**2, COV_FLOOR_VEL**2,
        ])
        diag = np.diag(self.P)
        needs_fix = diag < floor
        if np.any(needs_fix):
            for i in range(6):
                if needs_fix[i]:
                    self.P[i, i] = floor[i]

    # ──────────────── generic EKF update ────────────────

    def _ekf_update(self, z, zhat, H, R, label: str, t: float):
        z = np.asarray(z, float).reshape(-1)
        zhat = np.asarray(zhat, float).reshape(-1)
        H = np.asarray(H, float)
        R = np.asarray(R, float)
        I6 = np.eye(6)

        y = z - zhat
        S = H @ self.P_pred @ H.T + R
        K = self.P_pred @ H.T @ la.inv(S)

        self.x = self.x_pred + K @ y

        if self.cfg.use_joseph:
            KH = K @ H
            self.P = (I6 - KH) @ self.P_pred @ (I6 - KH).T + K @ R @ K.T
        else:
            self.P = (I6 - K @ H) @ self.P_pred

        # Symmetrise
        self.P = 0.5 * (self.P + self.P.T)

        # Enforce covariance floor
        self._enforce_cov_floor()

        snap = PeerNavSnapshot(
            t=t, dep_id=self.dep_id,
            x=self.x.copy(), P=self.P.copy(),
            innov=y.copy(), S=S.copy(),
            meas_type=label,
        )
        self.history.append(snap)
        return y, S

    # ──────────────── peer update (range + bearing) ────────────────

    def peer_update(
        self,
        t: float,
        z_meas: np.ndarray,
        x_j_hat: np.ndarray,
        P_j: np.ndarray,
        j_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update using range + bearing measurement to deputy j.

        z_meas  = [range, az, el]  (3,)
        x_j_hat = communicated state estimate from deputy j  (6,)
        P_j     = communicated covariance from deputy j  (6,6)

        The measurement model uses x_pred_i and treats x̂_j as a noisy
        reference.  To account for the uncertainty in x̂_j we inflate R:

            R_eff = R_sensor + H_j @ P_j_pos @ H_j^T

        where H_j is the Jacobian of h(x_i, x_j) with respect to x_j
        (which has the same magnitude but opposite sign to H wrt x_i for
        range, and likewise for bearing).  Since ∂h/∂x_j = -∂h/∂x_i for
        the position-block (the measurement only depends on Δρ = ρ_j - ρ_i),
        we get:

            R_eff = R_sensor + H @ P_j_pos_mapped @ H^T

        This is the "consider covariance" / "Schmidt–Kalman" approach:
        the peer's position uncertainty is projected into measurement
        space and added to R so the filter doesn't over-trust noisy peers.
        """
        z_meas = np.asarray(z_meas, float).reshape(3)
        x_j_hat = np.asarray(x_j_hat, float).reshape(6)
        P_j = np.asarray(P_j, float).reshape(6, 6)

        # Predicted range
        r_pred, H_r = peer_range_pred_and_jac(self.x_pred, x_j_hat)

        # Predicted bearing
        b_pred, H_b = peer_bearing_pred_and_jac(self.x_pred, x_j_hat)

        # Stack
        zhat = np.array([r_pred, b_pred[0], b_pred[1]])
        H = np.vstack((H_r, H_b))  # (3, 6)

        # ── R inflation: "consider" the peer's covariance ──
        # Since ∂h/∂ρ_j = -∂h/∂ρ_i, the Jacobian wrt x_j position has
        # the same H[:, :3] columns but with opposite sign.  Mapping P_j
        # through (-H) gives the same quadratic form as H @ P_j @ H^T.
        # We only use the position-block of P_j (bearing/range don't
        # depend on the peer's velocity).
        P_j_pos = P_j[:3, :3]
        H_pos = H[:, :3]  # (3, 3) — position columns of Jacobian
        R_consider = H_pos @ P_j_pos @ H_pos.T   # (3, 3) peer uncertainty in meas space

        R_eff = self.R_peer + R_consider

        # Wrap azimuth innovation to [-π, π]
        y_raw = z_meas - zhat
        y_raw[1] = (y_raw[1] + np.pi) % (2 * np.pi) - np.pi

        # NIS gating (using inflated R)
        S = H @ self.P_pred @ H.T + R_eff
        try:
            nis = float(y_raw @ la.solve(S, y_raw))
        except la.LinAlgError:
            nis = float(y_raw @ y_raw)

        if nis > NIS_GATE_PEER:
            # Rejected — keep prediction
            self.x = self.x_pred.copy()
            self.P = self.P_pred.copy()
            return y_raw, S

        y, S = self._ekf_update(z_meas, zhat, H, R_eff,
                                 f"peer_{self.dep_id}_{j_id}", t)
        self.peer_update_count += 1

        # After own update, re-set pred for any subsequent CI fusion
        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()

        return y, S

    # ──────────────── GPS update (sparse anchor) ────────────────

    def gps_update(self, t: float, z_gps: np.ndarray):
        """
        Absolute GPS-like measurement: z = x_true + noise, H = I_6.
        Used sparingly as an anchor to prevent drift of the entire swarm.
        """
        z_gps = np.asarray(z_gps, float).reshape(6)
        H = np.eye(6)
        zhat = self.x_pred.copy()
        self._ekf_update(z_gps, zhat, H, self.R_gps, "gps", t)

        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()

    # ──────────────── CI fusion ────────────────

    def ci_fuse(
        self,
        t: float,
        x_j_hat: np.ndarray,
        P_j: np.ndarray,
        z_rel_ij: np.ndarray,
        R_rel: np.ndarray,
        j_id: int,
    ):
        """
        Covariance Intersection with an indirect estimate from peer j.

        Deputy j says "I think I'm at (x̂_j, P_j)".
        Deputy i measured Δρ_ij ≈ ρ_j - ρ_i, so an indirect estimate of i is:

            x̃_i|j = x̂_j - [Δρ, 0]   (position only, velocity left at x̂_j)
            P̃_i|j = P_j + R_rel

        Then CI fuses (x_i, P_i) with (x̃_i|j, P̃_i|j).
        """
        x_j_hat = np.asarray(x_j_hat, float).reshape(6)
        P_j = np.asarray(P_j, float).reshape(6, 6)
        z_rel = np.asarray(z_rel_ij, float).reshape(3)

        # Build indirect estimate of self from peer's viewpoint
        x_indirect = x_j_hat.copy()
        x_indirect[:3] -= z_rel  # ρ_i ≈ ρ_j - Δρ_ij

        # Velocity: use peer's velocity estimate minus a rough relative velocity
        # (conservative: just inflate P for velocity)
        P_indirect = P_j + np.asarray(R_rel, float).reshape(6, 6)

        x_f, P_f = covariance_intersection(
            self.x, self.P,
            x_indirect, P_indirect,
        )

        self.x = x_f
        self.P = P_f
        self._enforce_cov_floor()
        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()
        self.ci_count += 1

        snap = PeerNavSnapshot(
            t=t, dep_id=self.dep_id,
            x=self.x.copy(), P=self.P.copy(),
            meas_type=f"ci_{j_id}",
        )
        self.history.append(snap)


# ──────────────────────────────────────────────────────────────────────────────
#                         main simulation driver
# ──────────────────────────────────────────────────────────────────────────────

def run_peer_relnav_sim(
    data_dir: Path,
    N_dep: int = N_DEP,
    meas_decim: int = MEAS_DECIM,
) -> Dict[str, np.ndarray]:
    """
    Full swarm peer-to-peer relative navigation simulation.

    Returns dict of results for each deputy (truth, estimates, errors, etc.)
    """
    data_dir = Path(data_dir)

    # ── load all bundles ──
    bundles = []
    for d in range(N_dep):
        p = data_dir / f"apophis_rendezvous_full_D{d}.npz"
        if not p.is_file():
            raise FileNotFoundError(f"Missing: {p}")
        bundles.append(load_apophis_bundle(p))

    # ── build common time grid from INTERSECTION of all deputies ──
    # Find the latest start and earliest end across all bundles so
    # no deputy ever runs past its actual data.
    t_starts = []
    t_ends = []
    for bd in bundles:
        t_d = np.asarray(bd["t"], float).reshape(-1)
        t_starts.append(float(t_d[0]))
        t_ends.append(float(t_d[-1]))

    t_common_start = max(t_starts)          # latest deploy
    t_common_end   = min(t_ends)            # earliest bundle end
    t_crop = t_common_start + CROP_HR * 3600.0

    if t_crop >= t_common_end:
        raise RuntimeError(
            f"After cropping {CROP_HR} h, no valid data remains. "
            f"Common window: [{t_common_start:.0f}, {t_common_end:.0f}] s"
        )

    # Use D0 as the reference grid, but only within the common window
    data0 = bundles[0]
    t_full = np.asarray(data0["t"], float).reshape(-1)
    dt_full = float(np.median(np.diff(t_full)))

    idx_crop = int(np.searchsorted(t_full, t_crop))
    idx_end  = int(np.searchsorted(t_full, t_common_end, side="right")) - 1
    idx_sel = np.arange(idx_crop, min(idx_end + 1, len(t_full)), meas_decim)
    t_arr = t_full[idx_sel]
    N = len(t_arr)

    dt_nav = float(np.median(np.diff(t_arr)))

    print(f"Common window: [{t_common_start:.0f}, {t_common_end:.0f}] s  "
          f"({(t_common_end - t_common_start)/3600:.2f} h)")
    print(f"Nav grid: {N} steps, dt = {dt_nav:.1f} s, "
          f"span = {(t_arr[-1] - t_arr[0])/3600:.2f} h")

    # ── per-deputy data ──
    chief_state_fn = make_state_fn(data0, "rC_N", "vC_N")

    # Truth in Hill + burn sequences
    x_true_all = np.zeros((N_dep, N, 6))
    dv_all     = np.zeros((N_dep, N, 3))

    for d in range(N_dep):
        bd = bundles[d]
        t_full_d  = np.asarray(bd["t"], float).reshape(-1)
        rD_full_d = np.asarray(bd["rD_N"], float).reshape(-1, 3)
        vD_full_d = np.asarray(bd["vD_N"], float).reshape(-1, 3)
        rC_full_d = np.asarray(bd["rC_N"], float).reshape(-1, 3)
        vC_full_d = np.asarray(bd["vC_N"], float).reshape(-1, 3)

        # Map common time grid into this deputy's index space
        idx_sel_d = np.searchsorted(t_full_d, t_arr)
        idx_sel_d = np.clip(idx_sel_d, 0, len(t_full_d) - 1)

        rD = rD_full_d[idx_sel_d]
        vD = vD_full_d[idx_sel_d]
        rC = rC_full_d[idx_sel_d]
        vC = vC_full_d[idx_sel_d]
        x_true_all[d] = eci_to_hill_traj(rD, vD, rC, vC)

        imp = np.asarray(bd["impulses"], float).reshape(-1, 4)
        dt_d = float(np.median(np.diff(t_full_d)))
        dv_full_d = build_burn_sequence(imp, t_full_d, dt_d)
        idx_dv = np.clip(idx_sel_d, 0, len(dv_full_d) - 1)
        dv_all[d] = dv_full_d[idx_dv]

    # ── create per-deputy EKFs ──
    cfg = PeerRelNavConfig(dt=dt_nav)
    ekfs: List[PeerRelNavEKF] = []
    for d in range(N_dep):
        ekf = PeerRelNavEKF(d, cfg, chief_state_fn)
        # Init with biased guess
        bias = np.array([200.0, -150.0, 80.0, 0.06, -0.04, 0.02])
        x0 = x_true_all[d, 0] + bias * (1.0 + 0.1 * d)
        ekf.reset(x0)
        ekfs.append(ekf)

    # ── crosslink scheduler ──
    sched = CrosslinkScheduler(N_dep)
    rng = np.random.default_rng(42)

    sigma_az_rad = np.deg2rad(AZ_SIGMA_DEG)
    sigma_el_rad = np.deg2rad(EL_SIGMA_DEG)

    # ── storage ──
    x_est_hist = np.zeros((N_dep, N, 6))
    P_diag_hist = np.zeros((N_dep, N, 6))

    # ── main loop ──
    for k in range(N):
        t = float(t_arr[k])

        if k % 200 == 0:
            elapsed_hr = (t - t_arr[0]) / 3600.0
            print(f"  step {k:5d}/{N}  t = {elapsed_hr:.3f} h")

        # 1) Predict all deputies
        for d in range(N_dep):
            if k == 0:
                ekfs[d].x_pred = ekfs[d].x.copy()
                ekfs[d].P_pred = ekfs[d].P.copy()
                ekfs[d].t_last = t
            else:
                ekfs[d].predict(t, burn_dv_H=dv_all[d, k])

        # 2) GPS updates — absolute position reference
        if USE_ANCHOR_GPS:
            if GPS_ALL_DEPUTIES:
                # Every deputy gets a (noisy) GPS fix every step
                for d in range(N_dep):
                    z_gps = x_true_all[d, k].copy()
                    z_gps[:3] += GPS_SIGMA_POS * rng.normal(size=3)
                    z_gps[3:] += GPS_SIGMA_VEL * rng.normal(size=3)
                    ekfs[d].gps_update(t, z_gps)
            elif k % ANCHOR_CADENCE == 0:
                # Round-robin: one deputy per ANCHOR_CADENCE steps
                anchor_dep = k // ANCHOR_CADENCE % N_dep
                z_gps = x_true_all[anchor_dep, k].copy()
                z_gps[:3] += GPS_SIGMA_POS * rng.normal(size=3)
                z_gps[3:] += GPS_SIGMA_VEL * rng.normal(size=3)
                ekfs[anchor_dep].gps_update(t, z_gps)

        # 3) Peer measurements according to crosslink schedule
        pairs = sched.pairs_for_step(k)
        for (i, j) in pairs:
            # Deputy i observes j
            z_ij, valid_ij = generate_peer_measurement(
                rng, x_true_all[i, k], x_true_all[j, k],
                RANGE_SIGMA, sigma_az_rad, sigma_el_rad,
            )
            # Deputy j observes i
            z_ji, valid_ji = generate_peer_measurement(
                rng, x_true_all[j, k], x_true_all[i, k],
                RANGE_SIGMA, sigma_az_rad, sigma_el_rad,
            )

            # i updates from observing j (using j's communicated estimate + cov)
            if valid_ij:
                ekfs[i].peer_update(t, z_ij, ekfs[j].x.copy(),
                                    ekfs[j].P.copy(), j)

            # j updates from observing i
            if valid_ji:
                ekfs[j].peer_update(t, z_ji, ekfs[i].x.copy(),
                                    ekfs[i].P.copy(), i)

            # 4) CI fusion: i fuses indirect self-estimate from j
            #    Rate-limited to avoid rumor propagation (double-counting)
            #    Only fuse when both peers have reasonable covariance
            ci_trP_gate = 500.0**2 * 3  # only CI-fuse if √tr(P_pos) < 500 m
            i_trP = np.trace(ekfs[i].P[:3, :3])
            j_trP = np.trace(ekfs[j].P[:3, :3])
            ci_ready = (i_trP < ci_trP_gate) and (j_trP < ci_trP_gate)
            ci_time_ok = (k % CI_EVERY_N_STEPS == 0)
            if USE_CI_FUSION and valid_ij and valid_ji and ci_ready and ci_time_ok:
                # Relative position measurement (i→j)
                d_pos_ij = x_true_all[j, k, :3] - x_true_all[i, k, :3]
                d_pos_ij += RANGE_SIGMA * rng.normal(size=3)  # noisy

                # R_rel for CI (inflated)
                R_ci = np.diag([
                    (2 * RANGE_SIGMA)**2, (2 * RANGE_SIGMA)**2,
                    (2 * RANGE_SIGMA)**2,
                    (3 * cfg.sigma_gps_vel)**2, (3 * cfg.sigma_gps_vel)**2,
                    (3 * cfg.sigma_gps_vel)**2,
                ])

                ekfs[i].ci_fuse(t, ekfs[j].x.copy(), ekfs[j].P.copy(),
                                d_pos_ij, R_ci, j)
                ekfs[j].ci_fuse(t, ekfs[i].x.copy(), ekfs[i].P.copy(),
                                -d_pos_ij, R_ci, i)

        # 5) Record
        for d in range(N_dep):
            x_est_hist[d, k] = ekfs[d].x.copy()
            P_diag_hist[d, k] = np.diag(ekfs[d].P)

    # ── metrics ──
    t_hr = (t_arr - t_arr[0]) / 3600.0
    mask = t_hr >= 0.2  # skip initial transient

    # ────────────────────────────────────────────────────────────────────
    #                  PRINT ALL VALUES
    # ────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("  PEER-TO-PEER RELATIVE NAVIGATION — FULL RESULTS")
    print("=" * 90)

    # --- Configuration summary ---
    print("\n┌─── Configuration ───────────────────────────────────────────────┐")
    print(f"│  Deputies          : {N_dep}")
    print(f"│  Nav grid steps    : {N}")
    print(f"│  Nav dt            : {dt_nav:.1f} s")
    print(f"│  Time span         : {(t_arr[-1] - t_arr[0])/3600:.3f} h")
    print(f"│  Meas decimation   : {meas_decim}")
    print(f"│  Crop after deploy : {CROP_HR} h")
    print(f"│  Peer range σ      : {RANGE_SIGMA} m")
    print(f"│  Peer az σ         : {AZ_SIGMA_DEG}°")
    print(f"│  Peer el σ         : {EL_SIGMA_DEG}°")
    gps_mode = "OFF"
    if USE_ANCHOR_GPS:
        gps_mode = "ALL deputies every step" if GPS_ALL_DEPUTIES else f"round-robin every {ANCHOR_CADENCE} steps"
    print(f"│  GPS anchor        : {gps_mode}")
    print(f"│  GPS pos σ         : {GPS_SIGMA_POS} m")
    print(f"│  GPS vel σ         : {GPS_SIGMA_VEL} m/s")
    print(f"│  CI fusion         : {'ON' if USE_CI_FUSION else 'OFF'}")
    print(f"│  Peer dropout prob : {P_PEER_DROPOUT}")
    print(f"│  Peer outlier prob : {P_PEER_OUTLIER}")
    print(f"│  NIS gate (peer)   : {NIS_GATE_PEER}")
    print(f"│  Process σ_pos     : {cfg.sigma_proc_pos} m")
    print(f"│  Process σ_vel     : {cfg.sigma_proc_vel} m/s")
    print(f"│  Init σ_pos        : {cfg.sigma_init_pos} m")
    print(f"│  Init σ_vel        : {cfg.sigma_init_vel} m/s")
    print(f"│  ΔV σ              : {cfg.sigma_dv} m/s")
    print(f"│  Crosslink rounds  : {sched.n_rounds}")
    print(f"└─────────────────────────────────────────────────────────────────┘")

    # --- Per-deputy summary table ---
    print("\n┌─── Per-Deputy Summary ──────────────────────────────────────────────────────────────────┐")
    print(f"│ {'Dep':>3} │ {'RMS pos [m]':>11} │ {'RMS_R':>8} │ {'RMS_T':>8} │ {'RMS_N':>8} │"
          f" {'RMS vel':>10} │ {'Peers':>5} │ {'CI':>4} │ {'Final err':>10} │ {'√tr(P_f)':>9} │")
    print(f"│{'─'*3:>3}─┼─{'─'*11}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─"
          f"{'─'*10}─┼─{'─'*5}─┼─{'─'*4}─┼─{'─'*10}─┼─{'─'*9}─│")

    for d in range(N_dep):
        err = x_est_hist[d] - x_true_all[d]
        pos_err = err[mask, :3]
        vel_err = err[mask, 3:]
        rms_pos = np.sqrt(np.mean(np.sum(pos_err**2, axis=1)))
        rms_vel = np.sqrt(np.mean(np.sum(vel_err**2, axis=1)))
        rms_ax = np.sqrt(np.mean(pos_err**2, axis=0))
        final_pos_err = la.norm(err[-1, :3])
        final_trP = np.sqrt(np.sum(P_diag_hist[d, -1, :3]))
        print(f"│  D{d} │ {rms_pos:11.2f} │ {rms_ax[0]:8.2f} │ {rms_ax[1]:8.2f} │ {rms_ax[2]:8.2f} │"
              f" {rms_vel:10.4f} │ {ekfs[d].peer_update_count:5d} │ {ekfs[d].ci_count:4d} │"
              f" {final_pos_err:10.2f} │ {final_trP:9.2f} │")

    print(f"└───────────────────────────────────────────────────────────────────────────────────────────┘")

    # --- Per-deputy detailed time histories ---
    for d in range(N_dep):
        err = x_est_hist[d] - x_true_all[d]
        print(f"\n{'─'*90}")
        print(f"  DEPUTY D{d} — Detailed Time History")
        print(f"{'─'*90}")
        print(f"  {'t [h]':>8}  │ {'ρR_true':>10} {'ρT_true':>10} {'ρN_true':>10} │"
              f" {'ρR_est':>10} {'ρT_est':>10} {'ρN_est':>10} │"
              f" {'errR':>8} {'errT':>8} {'errN':>8} │ {'||err||':>8} │"
              f" {'√trP_pos':>9}")

        for k in range(N):
            t_h = t_hr[k]
            xt = x_true_all[d, k]
            xe = x_est_hist[d, k]
            e = xe - xt
            enorm = la.norm(e[:3])
            trP = np.sqrt(np.sum(P_diag_hist[d, k, :3]))
            print(f"  {t_h:8.4f}  │ {xt[0]:10.2f} {xt[1]:10.2f} {xt[2]:10.2f} │"
                  f" {xe[0]:10.2f} {xe[1]:10.2f} {xe[2]:10.2f} │"
                  f" {e[0]:8.2f} {e[1]:8.2f} {e[2]:8.2f} │ {enorm:8.2f} │"
                  f" {trP:9.2f}")

        # Velocity details
        print(f"\n  {'t [h]':>8}  │ {'v̇R_true':>10} {'v̇T_true':>10} {'v̇N_true':>10} │"
              f" {'v̇R_est':>10} {'v̇T_est':>10} {'v̇N_est':>10} │"
              f" {'errVR':>8} {'errVT':>8} {'errVN':>8} │ {'||Verr||':>8} │"
              f" {'√trP_vel':>9}")

        for k in range(N):
            t_h = t_hr[k]
            xt = x_true_all[d, k]
            xe = x_est_hist[d, k]
            e = xe - xt
            venorm = la.norm(e[3:])
            trPv = np.sqrt(np.sum(P_diag_hist[d, k, 3:]))
            print(f"  {t_h:8.4f}  │ {xt[3]:10.5f} {xt[4]:10.5f} {xt[5]:10.5f} │"
                  f" {xe[3]:10.5f} {xe[4]:10.5f} {xe[5]:10.5f} │"
                  f" {e[3]:8.5f} {e[4]:8.5f} {e[5]:8.5f} │ {venorm:8.5f} │"
                  f" {trPv:9.5f}")

    # --- Covariance diagonal at final step ---
    print(f"\n{'='*90}")
    print(f"  FINAL COVARIANCE DIAGONALS (1σ)")
    print(f"{'='*90}")
    print(f"  {'Dep':>3} │ {'σ_R [m]':>10} {'σ_T [m]':>10} {'σ_N [m]':>10} │"
          f" {'σ_vR [m/s]':>12} {'σ_vT [m/s]':>12} {'σ_vN [m/s]':>12}")
    for d in range(N_dep):
        P_final = P_diag_hist[d, -1]
        sigma_final = np.sqrt(np.maximum(P_final, 0.0))
        print(f"   D{d} │ {sigma_final[0]:10.3f} {sigma_final[1]:10.3f} {sigma_final[2]:10.3f} │"
              f" {sigma_final[3]:12.6f} {sigma_final[4]:12.6f} {sigma_final[5]:12.6f}")

    # --- Crosslink schedule info ---
    print(f"\n{'='*90}")
    print(f"  CROSSLINK SCHEDULE (Round-Robin Pairs)")
    print(f"{'='*90}")
    for r in range(sched.n_rounds):
        pairs = sched.pairs_for_step(r)
        pair_str = ", ".join(f"(D{i}↔D{j})" for i, j in pairs)
        print(f"  Round {r}: {pair_str}")

    print(f"\n{'='*90}")
    print(f"  END OF RESULTS")
    print(f"{'='*90}")

    return dict(
        t_arr=t_arr,
        t_hr=t_hr,
        x_true_all=x_true_all,
        x_est_hist=x_est_hist,
        P_diag_hist=P_diag_hist,
        mask=mask,
        N_dep=N_dep,
    )


# ──────────────────────────────────────────────────────────────────────────────
#                              plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(res: Dict[str, np.ndarray]):
    t_hr       = res["t_hr"]
    x_true_all = res["x_true_all"]
    x_est_hist = res["x_est_hist"]
    P_diag     = res["P_diag_hist"]
    mask       = res["mask"]
    N_dep      = res["N_dep"]

    labels = ["R", "T", "N"]
    tt = t_hr[mask]

    if SAVE_FIGS:
        FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1) Per-deputy position error vs 3σ ──
    for d in range(N_dep):
        err = x_est_hist[d] - x_true_all[d]
        three_sigma = 3.0 * np.sqrt(np.maximum(P_diag[d], 0.0))

        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        fig.suptitle(f"Peer RelNav — Deputy D{d}: errors vs 3σ", fontsize=14)

        for i in range(3):
            ax = axes[0, i]
            ax.scatter(tt, err[mask, i], s=4, alpha=0.4)
            ax.plot(tt,  three_sigma[mask, i], "r--", lw=1)
            ax.plot(tt, -three_sigma[mask, i], "r--", lw=1)
            ax.set_ylabel(f"ρ_{labels[i]} [m]")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Position {labels[i]}")

        for i in range(3):
            ax = axes[1, i]
            ax.scatter(tt, err[mask, 3 + i], s=4, alpha=0.4)
            ax.plot(tt,  three_sigma[mask, 3 + i], "r--", lw=1)
            ax.plot(tt, -three_sigma[mask, 3 + i], "r--", lw=1)
            ax.set_ylabel(f"ρ̇_{labels[i]} [m/s]")
            ax.set_xlabel("Time [h]")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Velocity {labels[i]}")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if SAVE_FIGS:
            fig.savefig(FIG_DIR / f"dep{d}_error_3sigma.png", dpi=150)
        plt.show()

    # ── 2) Swarm-level RMS comparison ──
    fig2, (ax_p, ax_v) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig2.suptitle("Peer RelNav — Swarm position & velocity RMS over time",
                  fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, N_dep))
    for d in range(N_dep):
        err = x_est_hist[d] - x_true_all[d]
        pos_rms_running = np.sqrt(np.sum(err[:, :3]**2, axis=1))
        vel_rms_running = np.sqrt(np.sum(err[:, 3:]**2, axis=1))
        ax_p.plot(t_hr, pos_rms_running, lw=0.8, color=colors[d],
                  label=f"D{d}", alpha=0.7)
        ax_v.plot(t_hr, vel_rms_running, lw=0.8, color=colors[d],
                  alpha=0.7)

    ax_p.set_ylabel("Position error [m]")
    ax_p.legend(ncol=4, fontsize=8)
    ax_p.grid(True, alpha=0.3)
    ax_p.set_yscale("log")

    ax_v.set_ylabel("Velocity error [m/s]")
    ax_v.set_xlabel("Time [h]")
    ax_v.grid(True, alpha=0.3)
    ax_v.set_yscale("log")

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS:
        fig2.savefig(FIG_DIR / "swarm_rms_timeseries.png", dpi=150)
    plt.show()

    # ── 3) Covariance trace evolution ──
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    fig3.suptitle("Peer RelNav — Position covariance trace evolution", fontsize=14)

    for d in range(N_dep):
        tr_pos = np.sum(P_diag[d, :, :3], axis=1)
        ax3.plot(t_hr, np.sqrt(tr_pos), lw=0.8, color=colors[d],
                 label=f"D{d}", alpha=0.7)

    ax3.set_ylabel("√tr(P_pos) [m]")
    ax3.set_xlabel("Time [h]")
    ax3.legend(ncol=4, fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")
    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS:
        fig3.savefig(FIG_DIR / "cov_trace_evolution.png", dpi=150)
    plt.show()

    # ── 4) 3D Hill trajectory for D0 ──
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, projection="3d")
    d = 0
    ax4.plot(x_true_all[d, :, 0], x_true_all[d, :, 1], x_true_all[d, :, 2],
             lw=1.0, label="truth")
    ax4.plot(x_est_hist[d, :, 0], x_est_hist[d, :, 1], x_est_hist[d, :, 2],
             lw=1.0, label="estimate")
    ax4.set_xlabel("R [m]")
    ax4.set_ylabel("T [m]")
    ax4.set_zlabel("N [m]")
    ax4.set_title("Deputy D0: Hill trajectory — truth vs peer-nav estimate")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    if SAVE_FIGS:
        fig4.savefig(FIG_DIR / "d0_3d_trajectory.png", dpi=150)
    plt.show()

    if SAVE_FIGS:
        print(f"\nFigures saved to: {FIG_DIR}/")
    else:
        print("\nAll figures displayed interactively.")


# ──────────────────────────────────────────────────────────────────────────────
#                              entry point
# ──────────────────────────────────────────────────────────────────────────────

def _find_data_dir() -> Path:
    """
    Search for apophis_rendezvous_full_D0.npz in common locations:
      1) SCRIPT_DIR (alongside this file)
      2) Repository root (where RHIG fleet script typically writes)
    """
    candidates = [
        SCRIPT_DIR,
        ROOT.parent if ROOT else SCRIPT_DIR,  # repo root
    ]
    for d in candidates:
        if (d / "apophis_rendezvous_full_D0.npz").is_file():
            return d
    raise FileNotFoundError(
        "Cannot find apophis_rendezvous_full_D0.npz in any known location.\n"
        "Searched: " + ", ".join(str(c) for c in candidates) + "\n"
        "Run `basilik_rhig_apophis_rendezvous_fleet.py` first."
    )


if __name__ == "__main__":
    data_dir = _find_data_dir()
    print(f"Using data from: {data_dir}")

    results = run_peer_relnav_sim(data_dir)
    plot_results(results)
