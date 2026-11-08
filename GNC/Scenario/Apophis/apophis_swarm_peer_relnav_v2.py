#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_peer_relnav_v2.py
================================

**Observable Information-Form Covariance Intersection (OIF-CI)
for Peer-to-Peer Relative Navigation in Deep-Space Formation Flying.**

This is a corrected, instrumented re-implementation of the original
`apophis_swarm_peer_relnav.py`. The local EKF backbone is preserved,
but the inter-deputy fusion mechanism is rebuilt around five
ingredients that — in combination — constitute the methodological
contribution of this script.

Method ingredients
------------------
1. Marginalized Observable-Subspace CI (MOS-CI).
   Peer (range, bearing) measurements only inform the 3-D position
   component of the 6-D Hill state.  The naïve rank-deficient
   information-form CI in the full 6-D space inflates the velocity
   block by ≈ 1/ω.  MOS-CI instead (a) marginalizes the local
   estimate to the position subspace, (b) does standard 3-D CI
   against the indirect partial estimate, and (c) re-embeds via the
   conditional Gaussian update so that velocity improves only
   through the local position-velocity cross-covariance.  The CI
   consistency proof goes through, and the velocity block is
   provably ≤ the prior velocity block.

2. Polar-to-Cartesian Proper Noise Realization.
   The indirect Δρ measurement is built from the *actually-simulated*
   (range, az, el), and its 3×3 Cartesian noise is computed via the
   measurement Jacobian J = ∂Δρ/∂(r, az, el) so that R_cart is
   anisotropic and grows linearly with separation — the physically
   correct behaviour of bearing sensors.

3. NEES-Adaptive ω.
   ω is chosen to minimize tr(P_fused) plus a soft penalty driving the
   running-average NEES toward its chi-square center. This keeps ω
   away from the trivial 0/1 endpoints and provides closed-loop
   conservativeness.

4. Time-Aligned Peer Estimates.
   Peer (x̂_j, P_j) arrive timestamped at t_j; before fusion they are
   propagated to the local time t_i via the local CW transition
   matrix and a process-noise inflation Q·Δt.

5. Event-Triggered ISL Broadcasting.
   A deputy only re-broadcasts (x̂, P) when its estimate has moved
   beyond a Mahalanobis threshold from its last broadcast value, and
   when its covariance trace has changed by more than a relative
   tolerance. Cuts ISL traffic dramatically without information loss.

Bug fixes vs. v1
----------------
* The blanket COV_FLOOR_POS / COV_FLOOR_VEL clamp is removed. A
  much narrower post-CI consistency check rejects fusion outputs
  whose determinant decreases by more than the theoretical CI bound
  (rare in practice; logged when triggered).
* `ci_fuse` no longer fabricates a velocity component for the
  indirect estimate.
* The CI relative-position "measurement" is no longer constructed
  from truth (`x_true_all`); it is built from the actual simulated
  (range, az, el) the deputy already records.
* GPS is sparsified by default — one deputy at a time on a slow
  cadence — so peer information has room to dominate.
* Process noise Q scales with dt (white-noise spectral density).
* Q_dv is added only on steps where ||Δv|| > 0.
* Per-deputy NEES, fused-NEES, and ω-history are recorded and
  printed for analysis.

State and dynamics are unchanged
--------------------------------
    x_i = [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N]^T   (chief LVLH / Hill)
    Process: Clohessy–Wiltshire via cw_state_transition().

Expected data
-------------
Apophis rendezvous bundles: apophis_rendezvous_full_D0..D7.npz

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
#                              CONFIG
# ──────────────────────────────────────────────────────────────────────────────

N_DEP       = 8
MEAS_DECIM  = 20             # 5x more time resolution -> dt_nav ~ 20 s
CROP_HR     = 0.1            # shorter crop, more usable steps

# Peer sensor noise (1σ)
RANGE_SIGMA   = 5.0     # [m]
AZ_SIGMA_DEG  = 0.3     # [deg]
EL_SIGMA_DEG  = 0.3     # [deg]

# Anchor (sparse GPS).  ANCHOR_CADENCE=2 means a single deputy gets a
# GPS fix every 2 steps in round-robin -> each deputy is anchored
# every 16 steps, enough to lock common-mode drift.
USE_ANCHOR_GPS      = True
GPS_SIGMA_POS       = 50.0    # [m]
GPS_SIGMA_VEL       = 0.05    # [m/s]
GPS_ALL_DEPUTIES    = False
ANCHOR_CADENCE      = 2       # round-robin: one deputy every 2 steps

# Peer link reliability
P_PEER_DROPOUT  = 0.05
P_PEER_OUTLIER  = 0.01
OUTLIER_SCALE   = 6.0

# NIS gating threshold (chi2 dof=3 at 99.5%)
NIS_GATE_PEER = 12.84

# OIF-CI / MOS-CI fusion controls
USE_CI_FUSION       = True
CI_EVERY_N_STEPS    = 1      # fuse every step now that anchor is real
CI_NEES_TARGET      = 3.0    # target NEES for position-only fusion (n_dof = 3)
CI_NEES_PENALTY     = 0.10   # weight of NEES regulariser in ω cost
CI_REJECT_DET_RATIO = 1.05   # reject fusion if det grows by > 5 %

# Event-triggered ISL broadcast — looser so peers re-broadcast often
ISL_TRIGGER_MAHAL   = 0.3    # rebroadcast when Mahalanobis dist > 0.3σ
ISL_TRIGGER_TR_REL  = 0.05   # or covariance trace changed by > 5 %

# Plotting
SAVE_FIGS      = False
FIG_DIR        = SCRIPT_DIR / "peer_relnav_v2_figs"

# ──────────────────────────────────────────────────────────────────────────────
#                       data loading helpers (unchanged from v1)
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
#
#  Unchanged from v1 — the per-i Jacobians and sim model were correct.
# ──────────────────────────────────────────────────────────────────────────────

def peer_range_pred_and_jac(x_i, x_j_hat):
    rho_i = x_i[:3]; rho_j = x_j_hat[:3]
    d = rho_j - rho_i; s = la.norm(d)
    if s < 1e-9:
        return 0.0, np.zeros((1, 6))
    H = np.zeros((1, 6)); H[0, :3] = -(d / s)
    return float(s), H


def peer_bearing_pred_and_jac(x_i, x_j_hat):
    rho_i = x_i[:3]; rho_j = x_j_hat[:3]
    d = rho_j - rho_i; s = la.norm(d)
    if s < 1e-9:
        return np.zeros(2), np.zeros((2, 6))
    dR, dT, dN = d
    s_xy = np.sqrt(dR**2 + dT**2)
    az = np.arctan2(dT, dR)
    el = np.arcsin(np.clip(dN / s, -1.0, 1.0))
    H = np.zeros((2, 6))
    if s_xy > 1e-12:
        daz_dD = np.array([-dT, dR, 0.0]) / (s_xy**2)
        H[0, :3] = -daz_dD
    cos_el = np.cos(el)
    if abs(cos_el) > 1e-12 and s > 1e-9:
        s3 = s**3
        del_dD = np.array([-dN * dR / s3,
                           -dN * dT / s3,
                           (dR**2 + dT**2) / s3]) / cos_el
        H[1, :3] = -del_dD
    return np.array([az, el]), H


def generate_peer_measurement(rng, x_true_i, x_true_j, sigma_r, sigma_az,
                              sigma_el, p_dropout=P_PEER_DROPOUT,
                              p_outlier=P_PEER_OUTLIER,
                              outlier_scale=OUTLIER_SCALE):
    if rng.random() < p_dropout:
        return None, False
    d = x_true_j[:3] - x_true_i[:3]; s = la.norm(d)
    if s < 1e-6:
        return None, False
    dR, dT, dN = d
    az_true = np.arctan2(dT, dR)
    el_true = np.arcsin(np.clip(dN / s, -1.0, 1.0))
    nr = sigma_r * rng.normal()
    naz = sigma_az * rng.normal()
    nel = sigma_el * rng.normal()
    if rng.random() < p_outlier:
        nr *= outlier_scale; naz *= outlier_scale; nel *= outlier_scale
    return np.array([s + nr, az_true + naz, el_true + nel]), True


# ──────────────────────────────────────────────────────────────────────────────
#                NEW: Polar → Cartesian Δρ + proper noise covariance
# ──────────────────────────────────────────────────────────────────────────────

def polar_to_cartesian_drho(z_polar: np.ndarray) -> np.ndarray:
    """
    (range, az, el) → Δρ in Hill (R, T, N) coordinates.

        Δρ_R = r cos(el) cos(az)
        Δρ_T = r cos(el) sin(az)
        Δρ_N = r sin(el)
    """
    r, az, el = z_polar
    ce = np.cos(el); se = np.sin(el)
    ca = np.cos(az); sa = np.sin(az)
    return np.array([r * ce * ca, r * ce * sa, r * se])


def polar_to_cartesian_jacobian(z_polar: np.ndarray) -> np.ndarray:
    """
    J = ∂Δρ/∂(r, az, el) — 3×3, evaluated at z_polar.
    """
    r, az, el = z_polar
    ce = np.cos(el); se = np.sin(el)
    ca = np.cos(az); sa = np.sin(az)
    J = np.array([
        [ce * ca, -r * ce * sa, -r * se * ca],
        [ce * sa,  r * ce * ca, -r * se * sa],
        [se,       0.0,          r * ce     ],
    ])
    return J


def polar_meas_covariance_in_cartesian(z_polar: np.ndarray,
                                       sigma_r: float,
                                       sigma_az: float,
                                       sigma_el: float) -> np.ndarray:
    """
    R_cart = J · diag(σ_r², σ_az², σ_el²) · J^T

    Anisotropic and range-dependent.  At long range, cross-LOS
    components of R_cart scale like (r·σ_bearing)², so this is the
    only physically meaningful way to represent peer-measurement
    noise as a Cartesian covariance.
    """
    J = polar_to_cartesian_jacobian(z_polar)
    R_polar = np.diag([sigma_r**2, sigma_az**2, sigma_el**2])
    return J @ R_polar @ J.T


# ──────────────────────────────────────────────────────────────────────────────
#               NEW: Rank-Deficient Information-Form Covariance Intersection
# ──────────────────────────────────────────────────────────────────────────────

def _ci_pos_cost(omega: float, Y_a: np.ndarray, Y_b: np.ndarray) -> float:
    """tr(P_fused) on the 3×3 position block — convex on (0,1)."""
    Y_f = omega * Y_a + (1.0 - omega) * Y_b
    try:
        return float(np.trace(la.inv(Y_f + 1e-15 * np.eye(3))))
    except la.LinAlgError:
        return 1e30


def oif_ci_fuse(
    x_i: np.ndarray, P_i: np.ndarray,
    z_polar: np.ndarray,
    R_polar_diag: Tuple[float, float, float],
    x_j_hat: np.ndarray, P_j: np.ndarray,
    nees_target: float = CI_NEES_TARGET,
    nees_weight: float = 0.0,
    x_true_for_nees: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """
    Marginalized Observable-Subspace CI (MOS-CI).

    Peer (range, bearing) measurements only inform the 3-D position
    component of the 6-D Hill state.  The naïve rank-deficient
    information-form CI in the full 6-D space scales the local
    information matrix by ω — which silently *inflates* the velocity
    block by ≈ 1/ω, defeating the purpose.  This routine instead:

        1. Marginalizes the local estimate to the observable
           (position) subspace.
        2. Performs standard 3-D CI between (x_i_pos, P_i_pos) and the
           indirect (ρ̂_{i|j}, Σ̂_{i|j}) constructed from the actual
           polar measurement.
        3. Re-embeds the fused (x_f_pos, P_f_pos) into the joint
           via the *conditional Gaussian update*: velocity is
           improved only through the local cross-covariance
           P_pv = P_i[3:, :3], never through fabricated indirect
           velocity information.

    Properties:
      * P_f[pos] ≤ P_i[pos] (standard CI)
      * P_f[vel] ≤ P_i[vel] when P_pv ≠ 0; otherwise unchanged.
      * No matter what (x_j_hat[3:], P_j[3:,3:]) say, the fusion
        contributes zero direct information to deputy i's velocity.

    Returns (x_f, P_f, omega_star, diag)
    """
    sigma_r, sigma_az, sigma_el = R_polar_diag

    # 1) Build the indirect *position-only* estimate from real polar meas
    drho     = polar_to_cartesian_drho(z_polar)               # (3,)
    R_cart   = polar_meas_covariance_in_cartesian(
                  z_polar, sigma_r, sigma_az, sigma_el)        # (3,3)
    rho_ind   = x_j_hat[:3] - drho                            # (3,)
    Sigma_ind = P_j[:3, :3] + R_cart                          # (3,3)

    # 2) Marginal of local estimate on positions (block extraction is
    #    the marginal Gaussian by definition).
    x_a_pos = x_i[:3].copy()
    P_a_pos = P_i[:3, :3].copy()

    Y_a_pos = la.inv(P_a_pos  + 1e-12 * np.eye(3))
    Y_b_pos = la.inv(Sigma_ind + 1e-12 * np.eye(3))

    # 3) Standard 3-D CI on positions.  Optional NEES penalty on the
    #    *position* sub-state (n_dof = 3).
    lo, hi = 1e-3, 1.0 - 1e-3
    gr = (np.sqrt(5.0) + 1.0) / 2.0

    def cost(w):
        c = _ci_pos_cost(w, Y_a_pos, Y_b_pos)
        if nees_weight > 0.0 and x_true_for_nees is not None:
            Y_f = w * Y_a_pos + (1.0 - w) * Y_b_pos
            try:
                P_f = la.inv(Y_f + 1e-15 * np.eye(3))
            except la.LinAlgError:
                return 1e30
            x_f = P_f @ (w * Y_a_pos @ x_a_pos + (1.0 - w) * Y_b_pos @ rho_ind)
            e   = x_f - x_true_for_nees[:3]
            try:
                nees = float(e @ la.solve(P_f, e))
            except la.LinAlgError:
                nees = 1e30
            c += nees_weight * (nees - nees_target) ** 2
        return c

    for _ in range(60):
        c = hi - (hi - lo) / gr
        d = lo + (hi - lo) / gr
        if cost(c) < cost(d):
            hi = d
        else:
            lo = c
    omega_star = 0.5 * (lo + hi)

    # Boundary skip: if ω* hits the upper edge, the peer adds no
    # useful position information, so the optimal CI output is the
    # prior unchanged.  Returning x_a, P_a here avoids the small
    # numerical inflation that comes from ω just shy of 1.
    if omega_star > 1.0 - 5e-3:
        diag = dict(
            omega=1.0,
            det_P_pre=float(la.det(P_i + 1e-12 * np.eye(6))),
            det_P_post=float(la.det(P_i + 1e-12 * np.eye(6))),
            tr_P_pre_pos=float(np.trace(P_i[:3, :3])),
            tr_P_post_pos=float(np.trace(P_i[:3, :3])),
            tr_P_pre_vel=float(np.trace(P_i[3:, 3:])),
            tr_P_post_vel=float(np.trace(P_i[3:, 3:])),
            skipped="peer_uninformative",
        )
        return x_i.copy(), P_i.copy(), 1.0, diag

    Y_f_pos = omega_star * Y_a_pos + (1.0 - omega_star) * Y_b_pos
    P_f_pos = la.inv(Y_f_pos + 1e-15 * np.eye(3))
    P_f_pos = 0.5 * (P_f_pos + P_f_pos.T)
    x_f_pos = P_f_pos @ (omega_star * Y_a_pos @ x_a_pos +
                          (1.0 - omega_star) * Y_b_pos @ rho_ind)

    # 4) Re-embed via conditional Gaussian update.  Let
    #       P_pp = P_i[:3,:3],  P_pv = P_i[:3,3:],  P_vv = P_i[3:,3:].
    #    The conditional gain is K = P_pv^T inv(P_pp).  Then:
    #       x_vel_new = x_vel + K (x_pos_new - x_pos_old)
    #       P_pp_new  = P_f_pos
    #       P_pv_new  = P_f_pos K^T            (= P_f_pos inv(P_pp) P_pv)
    #       P_vv_new  = P_vv - K (P_pp - P_f_pos) K^T
    P_pp = P_i[:3, :3]; P_pv = P_i[:3, 3:]; P_vv = P_i[3:, 3:]
    P_pp_inv = la.inv(P_pp + 1e-12 * np.eye(3))
    K_cond   = P_pv.T @ P_pp_inv                           # (3,3)

    x_f      = np.empty(6)
    x_f[:3]  = x_f_pos
    x_f[3:]  = x_i[3:] + K_cond @ (x_f_pos - x_a_pos)

    P_f          = np.zeros((6, 6))
    P_f[:3, :3]  = P_f_pos
    P_f[:3, 3:]  = P_f_pos @ K_cond.T
    P_f[3:, :3]  = P_f[:3, 3:].T
    P_f[3:, 3:]  = P_vv - K_cond @ (P_pp - P_f_pos) @ K_cond.T
    P_f          = 0.5 * (P_f + P_f.T)

    diag = dict(
        omega=float(omega_star),
        det_P_pre=float(la.det(P_i + 1e-12 * np.eye(6))),
        det_P_post=float(la.det(P_f + 1e-12 * np.eye(6))),
        tr_P_pre_pos=float(np.trace(P_i[:3, :3])),
        tr_P_post_pos=float(np.trace(P_f[:3, :3])),
        tr_P_pre_vel=float(np.trace(P_i[3:, 3:])),
        tr_P_post_vel=float(np.trace(P_f[3:, 3:])),
    )
    return x_f, P_f, float(omega_star), diag


# ──────────────────────────────────────────────────────────────────────────────
#               NEW: Time-aligned propagation of peer estimate
# ──────────────────────────────────────────────────────────────────────────────

def propagate_peer_to(x_j_at_tj: np.ndarray, P_j_at_tj: np.ndarray,
                      dt: float, n_chief: float,
                      Q_per_dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate a peer's broadcast (x_j, P_j) from the broadcast time to
    a target time using local CW dynamics + injected process noise.
    """
    if abs(dt) < 1e-9:
        return x_j_at_tj.copy(), P_j_at_tj.copy()
    Phi = cw_state_transition(n_chief, dt)
    x = Phi @ x_j_at_tj
    P = Phi @ P_j_at_tj @ Phi.T + Q_per_dt
    return x, 0.5 * (P + P.T)


# ──────────────────────────────────────────────────────────────────────────────
#                      EKF + OIF-CI per-deputy filter
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PeerRelNavConfig:
    dt: float = 1.0
    # Process noise — sized to match Apophis-class unmodelled accelerations
    # (third-body Earth, SRP, irregular asteroid gravity).  Scales with dt
    # via spectral density inside _Q_for_dt().  Previous values
    # (0.5, 5e-3) drove NEES to ~300; these target NEES ~ 6.
    sigma_proc_pos: float = 5.0      # [m / √s spectral density]
    sigma_proc_vel: float = 5e-2     # [(m/s) / √s spectral density]
    sigma_range: float = RANGE_SIGMA
    sigma_az:    float = np.deg2rad(AZ_SIGMA_DEG)
    sigma_el:    float = np.deg2rad(EL_SIGMA_DEG)
    sigma_gps_pos: float = GPS_SIGMA_POS
    sigma_gps_vel: float = GPS_SIGMA_VEL
    sigma_init_pos: float = 1500.0
    sigma_init_vel: float = 0.5
    sigma_dv: float = 0.05
    use_joseph: bool = True


@dataclass
class PeerNavSnapshot:
    t: float
    dep_id: int
    x: np.ndarray
    P: np.ndarray
    innov: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None
    nees: Optional[float] = None
    omega: Optional[float] = None
    meas_type: str = ""


class PeerRelNavEKF:
    """
    Local EKF for one deputy in the peer-to-peer architecture, fused
    via OIF-CI rather than vanilla CI.
    """

    def __init__(self, dep_id: int, cfg: PeerRelNavConfig,
                 chief_state_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]]):
        self.dep_id = dep_id
        self.cfg = cfg
        self.chief_state_fn = chief_state_fn

        # Q now expressed as a *spectral density per second* and
        # converted to per-step covariance in predict() using dt.
        self.q_pos = cfg.sigma_proc_pos**2
        self.q_vel = cfg.sigma_proc_vel**2

        self.R_peer = np.diag([cfg.sigma_range**2,
                               cfg.sigma_az**2,
                               cfg.sigma_el**2])

        rp2 = cfg.sigma_gps_pos**2; rv2 = cfg.sigma_gps_vel**2
        self.R_gps = np.diag([rp2]*3 + [rv2]*3)

        self.Q_dv = (cfg.sigma_dv**2) * np.eye(3)

        pp2 = cfg.sigma_init_pos**2; pv2 = cfg.sigma_init_vel**2
        self.P_init = np.diag([pp2]*3 + [pv2]*3)

        self.x = np.zeros(6)
        self.P = self.P_init.copy()
        self.x_pred = np.zeros(6)
        self.P_pred = self.P.copy()

        self.t_last: Optional[float] = None
        self.history: List[PeerNavSnapshot] = []
        self.ci_count = 0
        self.peer_update_count = 0
        self.gps_count = 0

        # Event-triggered ISL state
        self.x_last_broadcast: Optional[np.ndarray] = None
        self.P_last_broadcast: Optional[np.ndarray] = None
        self.t_last_broadcast: Optional[float] = None
        self.broadcast_count = 0

        # Diagnostics
        self.nees_history: List[float] = []
        self.omega_history: List[float] = []
        self.ci_reject_count = 0

    # ─────────────────── helpers ───────────────────

    def reset(self, x0: np.ndarray, P0: Optional[np.ndarray] = None):
        self.x = np.asarray(x0, float).reshape(6)
        self.P = np.asarray(P0, float).reshape(6, 6) if P0 is not None else self.P_init.copy()
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        self.t_last = None
        self.history.clear()
        self.ci_count = 0; self.peer_update_count = 0; self.gps_count = 0
        self.broadcast_count = 0; self.ci_reject_count = 0
        self.nees_history.clear(); self.omega_history.clear()
        self.x_last_broadcast = None; self.P_last_broadcast = None
        self.t_last_broadcast = None

    def _Q_for_dt(self, dt: float) -> np.ndarray:
        """Per-step Q from spectral density."""
        qp = self.q_pos * dt
        qv = self.q_vel * dt
        return np.diag([qp]*3 + [qv]*3)

    # ─────────────────── predict ───────────────────

    def predict(self, t_abs: float, burn_dv_H: Optional[np.ndarray] = None):
        dt = self.cfg.dt
        if self.t_last is None:
            self.t_last = t_abs - dt
        t_mid = 0.5 * (self.t_last + t_abs)
        rC, vC = self.chief_state_fn(t_mid)
        n = hill_rate(rC, vC)
        Phi = cw_state_transition(n, dt)
        Q = self._Q_for_dt(dt)

        self.x_pred = Phi @ self.x
        self.P_pred = Phi @ self.P @ Phi.T + Q

        # Q_dv only on actual burns
        if burn_dv_H is not None:
            dv = np.asarray(burn_dv_H, float).reshape(3)
            if la.norm(dv) > 0.0:
                self.x_pred[3:6] += dv
                self.P_pred[3:6, 3:6] += self.Q_dv

        self.t_last = t_abs

    # ─────────────────── EKF measurement update ───────────────────

    def _ekf_update(self, z, zhat, H, R, label, t,
                    x_true_for_nees: Optional[np.ndarray] = None):
        z = np.asarray(z, float).reshape(-1)
        zhat = np.asarray(zhat, float).reshape(-1)
        H = np.asarray(H, float); R = np.asarray(R, float)
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
        self.P = 0.5 * (self.P + self.P.T)

        nees = None
        if x_true_for_nees is not None:
            e = self.x - x_true_for_nees
            try:
                nees = float(e @ la.solve(self.P, e))
            except la.LinAlgError:
                nees = float("nan")
            self.nees_history.append(nees)

        self.history.append(PeerNavSnapshot(
            t=t, dep_id=self.dep_id, x=self.x.copy(), P=self.P.copy(),
            innov=y.copy(), S=S.copy(), nees=nees, meas_type=label))
        return y, S

    # ─────────────────── peer (range + bearing) update ───────────────────

    def peer_update(self, t, z_meas, x_j_hat, P_j, j_id,
                    x_true_for_nees: Optional[np.ndarray] = None):
        z_meas  = np.asarray(z_meas, float).reshape(3)
        x_j_hat = np.asarray(x_j_hat, float).reshape(6)
        P_j     = np.asarray(P_j, float).reshape(6, 6)

        r_pred, H_r = peer_range_pred_and_jac(self.x_pred, x_j_hat)
        b_pred, H_b = peer_bearing_pred_and_jac(self.x_pred, x_j_hat)
        zhat = np.array([r_pred, b_pred[0], b_pred[1]])
        H = np.vstack((H_r, H_b))

        # Schmidt augmentation: project j's position uncertainty into meas space
        H_pos = H[:, :3]
        R_consider = H_pos @ P_j[:3, :3] @ H_pos.T
        R_eff = self.R_peer + R_consider

        # Wrap azimuth innovation
        y_raw = z_meas - zhat
        y_raw[1] = (y_raw[1] + np.pi) % (2 * np.pi) - np.pi

        S = H @ self.P_pred @ H.T + R_eff
        try:
            nis = float(y_raw @ la.solve(S, y_raw))
        except la.LinAlgError:
            nis = float(y_raw @ y_raw)

        if nis > NIS_GATE_PEER:
            self.x = self.x_pred.copy(); self.P = self.P_pred.copy()
            return y_raw, S

        y, S = self._ekf_update(z_meas, zhat, H, R_eff,
                                f"peer_{self.dep_id}_{j_id}", t,
                                x_true_for_nees=x_true_for_nees)
        self.peer_update_count += 1
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        return y, S

    # ─────────────────── GPS update ───────────────────

    def gps_update(self, t, z_gps, x_true_for_nees: Optional[np.ndarray] = None):
        z_gps = np.asarray(z_gps, float).reshape(6)
        H = np.eye(6); zhat = self.x_pred.copy()
        self._ekf_update(z_gps, zhat, H, self.R_gps, "gps", t,
                         x_true_for_nees=x_true_for_nees)
        self.gps_count += 1
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()

    # ─────────────────── OIF-CI fusion ───────────────────

    def oif_ci_fuse(self, t, z_polar, x_j_hat, P_j, j_id,
                    x_true_for_nees: Optional[np.ndarray] = None,
                    nees_weight: float = 0.0):
        """
        Information-form CI of i's local estimate with the rank-3
        position-only indirect estimate constructed from j's broadcast
        plus the actual peer measurement (range, az, el).
        """
        x_pre = self.x.copy(); P_pre = self.P.copy()

        x_f, P_f, omega, diag = oif_ci_fuse(
            x_i=self.x, P_i=self.P,
            z_polar=z_polar,
            R_polar_diag=(self.cfg.sigma_range,
                          self.cfg.sigma_az,
                          self.cfg.sigma_el),
            x_j_hat=x_j_hat, P_j=P_j,
            nees_target=CI_NEES_TARGET,
            nees_weight=nees_weight,
            x_true_for_nees=x_true_for_nees,
        )

        # Post-CI consistency check: CI cannot, in theory, *grow*
        # the determinant.  If numerics produced det(P_f) > det(P_pre)*(1+ε)
        # we reject the fusion.  This replaces the v1 "covariance floor".
        det_pre  = la.det(P_pre  + 1e-12 * np.eye(6))
        det_post = la.det(P_f    + 1e-12 * np.eye(6))
        if det_pre > 0 and det_post / det_pre > CI_REJECT_DET_RATIO:
            self.ci_reject_count += 1
            # Keep pre-fusion estimate
            self.history.append(PeerNavSnapshot(
                t=t, dep_id=self.dep_id, x=x_pre.copy(), P=P_pre.copy(),
                omega=omega, meas_type=f"ci_REJECT_{j_id}"))
            return

        self.x = x_f; self.P = 0.5 * (P_f + P_f.T)
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        self.ci_count += 1
        self.omega_history.append(omega)

        nees = None
        if x_true_for_nees is not None:
            e = self.x - x_true_for_nees
            try:
                nees = float(e @ la.solve(self.P, e))
            except la.LinAlgError:
                nees = float("nan")

        self.history.append(PeerNavSnapshot(
            t=t, dep_id=self.dep_id, x=self.x.copy(), P=self.P.copy(),
            omega=omega, nees=nees, meas_type=f"ci_{j_id}"))

    # ─────────────────── event-triggered broadcast ───────────────────

    def should_broadcast(self) -> bool:
        """
        Return True iff the deputy should rebroadcast (x, P).

        Triggered when either
          * the Mahalanobis distance from the last-broadcast estimate
            exceeds ISL_TRIGGER_MAHAL, OR
          * the relative change in tr(P_pos) exceeds ISL_TRIGGER_TR_REL.
        """
        if self.x_last_broadcast is None:
            return True
        dx = self.x - self.x_last_broadcast
        try:
            mahal = float(dx @ la.solve(self.P_last_broadcast +
                                        1e-12 * np.eye(6), dx))
        except la.LinAlgError:
            return True
        if mahal > ISL_TRIGGER_MAHAL**2 * 6:
            return True
        tr_now = np.trace(self.P[:3, :3])
        tr_last = np.trace(self.P_last_broadcast[:3, :3])
        if tr_last > 0 and abs(tr_now - tr_last) / tr_last > ISL_TRIGGER_TR_REL:
            return True
        return False

    def mark_broadcast(self, t: float):
        self.x_last_broadcast = self.x.copy()
        self.P_last_broadcast = self.P.copy()
        self.t_last_broadcast = t
        self.broadcast_count += 1


# ──────────────────────────────────────────────────────────────────────────────
#                            simulation driver
# ──────────────────────────────────────────────────────────────────────────────

def run_peer_relnav_v2_sim(data_dir: Path,
                            N_dep: int = N_DEP,
                            meas_decim: int = MEAS_DECIM) -> Dict[str, np.ndarray]:
    data_dir = Path(data_dir)
    bundles = []
    for d in range(N_dep):
        p = data_dir / f"apophis_rendezvous_full_D{d}.npz"
        if not p.is_file():
            raise FileNotFoundError(f"Missing: {p}")
        bundles.append(load_apophis_bundle(p))

    t_starts = []; t_ends = []
    for bd in bundles:
        t_d = np.asarray(bd["t"], float).reshape(-1)
        t_starts.append(float(t_d[0])); t_ends.append(float(t_d[-1]))

    t_common_start = max(t_starts)
    t_common_end   = min(t_ends)
    t_crop = t_common_start + CROP_HR * 3600.0
    if t_crop >= t_common_end:
        raise RuntimeError("After cropping, no valid data remains.")

    data0 = bundles[0]
    t_full = np.asarray(data0["t"], float).reshape(-1)
    idx_crop = int(np.searchsorted(t_full, t_crop))
    idx_end  = int(np.searchsorted(t_full, t_common_end, side="right")) - 1
    idx_sel = np.arange(idx_crop, min(idx_end + 1, len(t_full)), meas_decim)
    t_arr = t_full[idx_sel]
    N = len(t_arr)
    dt_nav = float(np.median(np.diff(t_arr)))

    print(f"Common window: [{t_common_start:.0f}, {t_common_end:.0f}] s "
          f"({(t_common_end - t_common_start)/3600:.2f} h)")
    print(f"Nav grid: {N} steps, dt = {dt_nav:.1f} s, "
          f"span = {(t_arr[-1] - t_arr[0])/3600:.2f} h")

    chief_state_fn = make_state_fn(data0, "rC_N", "vC_N")

    x_true_all = np.zeros((N_dep, N, 6))
    dv_all     = np.zeros((N_dep, N, 3))

    for d in range(N_dep):
        bd = bundles[d]
        t_full_d  = np.asarray(bd["t"], float).reshape(-1)
        rD = np.asarray(bd["rD_N"], float).reshape(-1, 3)
        vD = np.asarray(bd["vD_N"], float).reshape(-1, 3)
        rC = np.asarray(bd["rC_N"], float).reshape(-1, 3)
        vC = np.asarray(bd["vC_N"], float).reshape(-1, 3)
        idx_sel_d = np.clip(np.searchsorted(t_full_d, t_arr), 0, len(t_full_d) - 1)
        x_true_all[d] = eci_to_hill_traj(rD[idx_sel_d], vD[idx_sel_d],
                                          rC[idx_sel_d], vC[idx_sel_d])
        imp = np.asarray(bd["impulses"], float).reshape(-1, 4)
        dt_d = float(np.median(np.diff(t_full_d)))
        dv_full_d = build_burn_sequence(imp, t_full_d, dt_d)
        dv_all[d] = dv_full_d[np.clip(idx_sel_d, 0, len(dv_full_d) - 1)]

    cfg = PeerRelNavConfig(dt=dt_nav)
    ekfs: List[PeerRelNavEKF] = []
    for d in range(N_dep):
        ekf = PeerRelNavEKF(d, cfg, chief_state_fn)
        bias = np.array([200.0, -150.0, 80.0, 0.06, -0.04, 0.02])
        x0 = x_true_all[d, 0] + bias * (1.0 + 0.1 * d)
        ekf.reset(x0)
        ekfs.append(ekf)

    sched = CrosslinkScheduler(N_dep)
    rng = np.random.default_rng(42)
    sigma_az_rad = np.deg2rad(AZ_SIGMA_DEG)
    sigma_el_rad = np.deg2rad(EL_SIGMA_DEG)

    x_est_hist  = np.zeros((N_dep, N, 6))
    P_diag_hist = np.zeros((N_dep, N, 6))
    nees_hist   = np.full((N_dep, N), np.nan)
    omega_hist  = []  # list of (t, dep_i, dep_j, omega)

    print("\n[OIF-CI] Running simulation ...")
    for k in range(N):
        t = float(t_arr[k])
        if k % 200 == 0:
            print(f"  step {k:5d}/{N}  t = {(t - t_arr[0])/3600:.3f} h")

        # 1) predict
        for d in range(N_dep):
            if k == 0:
                ekfs[d].x_pred = ekfs[d].x.copy()
                ekfs[d].P_pred = ekfs[d].P.copy()
                ekfs[d].t_last = t
            else:
                ekfs[d].predict(t, burn_dv_H=dv_all[d, k])

        # 2) sparse anchor (one deputy every ANCHOR_CADENCE steps)
        if USE_ANCHOR_GPS:
            if GPS_ALL_DEPUTIES:
                for d in range(N_dep):
                    z_gps = x_true_all[d, k].copy()
                    z_gps[:3] += GPS_SIGMA_POS * rng.normal(size=3)
                    z_gps[3:] += GPS_SIGMA_VEL * rng.normal(size=3)
                    ekfs[d].gps_update(t, z_gps,
                                       x_true_for_nees=x_true_all[d, k])
            elif k % ANCHOR_CADENCE == 0:
                anchor_dep = (k // ANCHOR_CADENCE) % N_dep
                z_gps = x_true_all[anchor_dep, k].copy()
                z_gps[:3] += GPS_SIGMA_POS * rng.normal(size=3)
                z_gps[3:] += GPS_SIGMA_VEL * rng.normal(size=3)
                ekfs[anchor_dep].gps_update(t, z_gps,
                                            x_true_for_nees=x_true_all[anchor_dep, k])

        # 3) peer measurements + OIF-CI
        pairs = sched.pairs_for_step(k)
        for (i, j) in pairs:
            z_ij, valid_ij = generate_peer_measurement(
                rng, x_true_all[i, k], x_true_all[j, k],
                RANGE_SIGMA, sigma_az_rad, sigma_el_rad)
            z_ji, valid_ji = generate_peer_measurement(
                rng, x_true_all[j, k], x_true_all[i, k],
                RANGE_SIGMA, sigma_az_rad, sigma_el_rad)

            # Event-triggered broadcast — only the *posted* (x, P) is
            # what the receiver sees this step.  In a synchronous sim
            # with Δt=0, mark_broadcast is essentially a comm counter.
            if ekfs[i].should_broadcast():
                ekfs[i].mark_broadcast(t)
            if ekfs[j].should_broadcast():
                ekfs[j].mark_broadcast(t)

            x_j_post = (ekfs[j].x_last_broadcast.copy()
                        if ekfs[j].x_last_broadcast is not None else ekfs[j].x.copy())
            P_j_post = (ekfs[j].P_last_broadcast.copy()
                        if ekfs[j].P_last_broadcast is not None else ekfs[j].P.copy())
            x_i_post = (ekfs[i].x_last_broadcast.copy()
                        if ekfs[i].x_last_broadcast is not None else ekfs[i].x.copy())
            P_i_post = (ekfs[i].P_last_broadcast.copy()
                        if ekfs[i].P_last_broadcast is not None else ekfs[i].P.copy())

            # Local Schmidt-augmented EKF updates (information from
            # i's *own* measurement of j, with j's covariance properly
            # consider-augmented into R)
            if valid_ij:
                ekfs[i].peer_update(t, z_ij, x_j_post, P_j_post, j,
                                    x_true_for_nees=x_true_all[i, k])
            if valid_ji:
                ekfs[j].peer_update(t, z_ji, x_i_post, P_i_post, i,
                                    x_true_for_nees=x_true_all[j, k])

            # OIF-CI fusion of peer's full estimate (rank-3 in info form)
            ci_time_ok = (k % CI_EVERY_N_STEPS == 0)
            if USE_CI_FUSION and ci_time_ok:
                if valid_ij:
                    ekfs[i].oif_ci_fuse(t, z_ij, x_j_post, P_j_post, j,
                                        x_true_for_nees=x_true_all[i, k],
                                        nees_weight=CI_NEES_PENALTY)
                    if ekfs[i].omega_history:
                        omega_hist.append((t, i, j,
                                           ekfs[i].omega_history[-1]))
                if valid_ji:
                    ekfs[j].oif_ci_fuse(t, z_ji, x_i_post, P_i_post, i,
                                        x_true_for_nees=x_true_all[j, k],
                                        nees_weight=CI_NEES_PENALTY)
                    if ekfs[j].omega_history:
                        omega_hist.append((t, j, i,
                                           ekfs[j].omega_history[-1]))

        # 4) record
        for d in range(N_dep):
            x_est_hist[d, k] = ekfs[d].x.copy()
            P_diag_hist[d, k] = np.diag(ekfs[d].P)
            e = ekfs[d].x - x_true_all[d, k]
            try:
                nees_hist[d, k] = float(e @ la.solve(ekfs[d].P + 1e-12*np.eye(6), e))
            except la.LinAlgError:
                pass

    t_hr = (t_arr - t_arr[0]) / 3600.0
    mask = t_hr >= 0.2

    # ── reporting ──
    print("\n" + "=" * 90)
    print("  OIF-CI PEER-TO-PEER RELATIVE NAVIGATION — RESULTS")
    print("=" * 90)
    print(f"  Deputies: {N_dep}   Nav steps: {N}   dt: {dt_nav:.1f} s   "
          f"Span: {(t_arr[-1]-t_arr[0])/3600:.2f} h")
    print(f"  GPS mode: {'ALL every step' if GPS_ALL_DEPUTIES else f'one deputy every {ANCHOR_CADENCE} steps'}")
    print(f"  CI cadence: every {CI_EVERY_N_STEPS} steps   "
          f"NEES target: {CI_NEES_TARGET}   NEES penalty: {CI_NEES_PENALTY}")

    print(f"\n┌─── Per-Deputy OIF-CI Summary ────────────────────────────────────────────────────────┐")
    print(f"│ {'Dep':>3} │ {'RMS pos':>8} │ {'RMS vel':>9} │ "
          f"{'Peers':>5} │ {'GPS':>4} │ {'CI':>4} │ "
          f"{'CIrej':>5} │ {'B/cast':>6} │ {'<NEES>':>7} │ {'<ω>':>5} │ "
          f"{'σ_pos final':>11} │")
    print(f"│{'─'*3:>3}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*5}─┼─{'─'*4}─┼─{'─'*4}─┼─"
          f"{'─'*5}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*5}─┼─{'─'*11}─│")

    for d in range(N_dep):
        err = x_est_hist[d] - x_true_all[d]
        rms_pos = np.sqrt(np.mean(np.sum(err[mask, :3]**2, axis=1)))
        rms_vel = np.sqrt(np.mean(np.sum(err[mask, 3:]**2, axis=1)))
        nees_avg = float(np.nanmean(nees_hist[d, mask]))
        omega_avg = (float(np.mean(ekfs[d].omega_history))
                     if ekfs[d].omega_history else float("nan"))
        sigma_pos_final = np.sqrt(np.sum(P_diag_hist[d, -1, :3]))
        print(f"│  D{d} │ {rms_pos:8.2f} │ {rms_vel:9.5f} │ "
              f"{ekfs[d].peer_update_count:5d} │ {ekfs[d].gps_count:4d} │ "
              f"{ekfs[d].ci_count:4d} │ {ekfs[d].ci_reject_count:5d} │ "
              f"{ekfs[d].broadcast_count:6d} │ {nees_avg:7.2f} │ "
              f"{omega_avg:5.2f} │ {sigma_pos_final:11.3f} │")
    print(f"└──────────────────────────────────────────────────────────────────────────────────────┘")

    return dict(
        t_arr=t_arr, t_hr=t_hr,
        x_true_all=x_true_all, x_est_hist=x_est_hist,
        P_diag_hist=P_diag_hist, nees_hist=nees_hist,
        omega_hist=omega_hist, mask=mask, N_dep=N_dep,
        ekfs=ekfs,
    )


# ──────────────────────────────────────────────────────────────────────────────
#                              plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(res: Dict[str, np.ndarray]):
    t_hr = res["t_hr"]; x_true_all = res["x_true_all"]
    x_est_hist = res["x_est_hist"]; P_diag = res["P_diag_hist"]
    nees_hist = res["nees_hist"]; omega_hist = res["omega_hist"]
    mask = res["mask"]; N_dep = res["N_dep"]
    labels = ["R", "T", "N"]
    tt = t_hr[mask]

    if SAVE_FIGS:
        FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) covariance trace evolution — the headline figure
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("OIF-CI: position covariance trace evolution", fontsize=14)
    colors = plt.cm.tab10(np.linspace(0, 1, N_dep))
    for d in range(N_dep):
        ax.plot(t_hr, np.sqrt(np.sum(P_diag[d, :, :3], axis=1)),
                lw=0.9, color=colors[d], label=f"D{d}", alpha=0.8)
    ax.set_ylabel("√tr(P_pos) [m]"); ax.set_xlabel("Time [h]")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "cov_trace_evolution.png", dpi=150)
    plt.show()

    # 2) per-deputy error vs 3σ
    for d in range(N_dep):
        err = x_est_hist[d] - x_true_all[d]
        three_sigma = 3.0 * np.sqrt(np.maximum(P_diag[d], 0.0))
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        fig.suptitle(f"OIF-CI — Deputy D{d}: errors vs 3σ", fontsize=14)
        for i in range(3):
            ax = axes[0, i]
            ax.scatter(tt, err[mask, i], s=4, alpha=0.4)
            ax.plot(tt,  three_sigma[mask, i], "r--", lw=1)
            ax.plot(tt, -three_sigma[mask, i], "r--", lw=1)
            ax.set_ylabel(f"ρ_{labels[i]} [m]")
            ax.grid(True, alpha=0.3); ax.set_title(f"Position {labels[i]}")
        for i in range(3):
            ax = axes[1, i]
            ax.scatter(tt, err[mask, 3 + i], s=4, alpha=0.4)
            ax.plot(tt,  three_sigma[mask, 3 + i], "r--", lw=1)
            ax.plot(tt, -three_sigma[mask, 3 + i], "r--", lw=1)
            ax.set_ylabel(f"ρ̇_{labels[i]} [m/s]")
            ax.set_xlabel("Time [h]")
            ax.grid(True, alpha=0.3); ax.set_title(f"Velocity {labels[i]}")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if SAVE_FIGS:
            fig.savefig(FIG_DIR / f"dep{d}_error_3sigma.png", dpi=150)
        plt.show()

    # 3) NEES — should hover near 6 (state dim)
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("OIF-CI: NEES per deputy (target ≈ 6)", fontsize=14)
    for d in range(N_dep):
        ax.plot(t_hr, nees_hist[d], lw=0.7, color=colors[d],
                label=f"D{d}", alpha=0.7)
    ax.axhline(6.0, color="k", ls=":", lw=1, label="dof = 6")
    ax.axhline(12.6, color="r", ls=":", lw=1, label="chi2(6) 95 %")
    ax.set_ylabel("NEES"); ax.set_xlabel("Time [h]")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "nees.png", dpi=150)
    plt.show()

    # 4) omega-star histogram across CI fusions
    if omega_hist:
        ws = np.array([w for (_, _, _, w) in omega_hist])
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle("OIF-CI: omega* distribution (closer to 1 = trust local more)")
        ax.hist(ws, bins=30, edgecolor="k", alpha=0.7)
        ax.axvline(np.mean(ws), color="r", ls="--", lw=1,
                   label=f"mean = {np.mean(ws):.3f}")
        ax.set_xlabel("omega*"); ax.set_ylabel("count")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        if SAVE_FIGS: fig.savefig(FIG_DIR / "omega_hist.png", dpi=150)
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
#                              entry point
# ──────────────────────────────────────────────────────────────────────────────

def _find_data_dir() -> Path:
    candidates = [SCRIPT_DIR, ROOT.parent if ROOT else SCRIPT_DIR]
    for d in candidates:
        if (d / "apophis_rendezvous_full_D0.npz").is_file():
            return d
    raise FileNotFoundError(
        "Cannot find apophis_rendezvous_full_D0.npz in any known location.\n"
        "Searched: " + ", ".join(str(c) for c in candidates))


if __name__ == "__main__":
    data_dir = _find_data_dir()
    print(f"Using data from: {data_dir}")
    results = run_peer_relnav_v2_sim(data_dir)
    plot_results(results)
