#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_angles_only_relnav_v2.py
================================

Next-generation angles-only relative navigation between a single chief
and a single deputy in low-Earth orbit.  Builds on
``simple_angles_only_relnav.py`` (UKF + 4-D Formation-Identity Manifold
Constraint + chief-as-extended-object + URTSS smoother) and adds three
mathematically novel pieces:

    (1)  LAPLACE-RUNGE-LENZ EXTENSION OF FIMC  →  7-D
         The classical two-body problem has SEVEN integrals of motion:
             ε              specific orbital energy           (1 scalar)
             h = r × v      specific angular momentum         (3-vector)
             e = (v×h)/μ − r̂   Laplace-Runge-Lenz eccentricity (3-vector)
         Of these seven, six are independent (the seventh is a
         constraint).  The previous filter constrained only (ε, h) — the
         orbital plane and shape, but NOT the orientation of perigee in
         the plane.  Adding e collapses the deputy's 6-D state to a
         strictly 1-D manifold: true anomaly only.  All other DoF are
         pinned by the chief's broadcast.  σ values are sized to honour
         BOTH chief broadcast uncertainty AND the natural deputy/chief
         orbit-element offset induced by the bounded relative motion.

    (5)  ITERATED POSTERIOR LINEARISATION FILTER (IPLF)
         The standard UKF computes the statistical Jacobian
         A = Pxz^T P^-1 once, around the PRIOR mean.  For the highly
         non-linear LOS measurement h(x) = (atan2, asin), this leaves
         a non-trivial linearisation bias when the angles-only along-LOS
         uncertainty is large.  IPLF (García-Fernández et al. 2015)
         RE-LINEARISES around the current posterior, then re-runs the
         KF update against the original prior, and iterates to
         convergence.  Each iteration tightens the linearisation
         centroid, cutting the per-update bias by 1-2 orders of
         magnitude versus a single-pass UKF.

    (4)  TOPOLOGY-AWARE PERIODIC-ORBIT CONSTRAINT
         A deputy on a bounded co-orbital relative orbit traces a CLOSED
         loop in the chief's LVLH frame with period T_orbit.  In the
         inertial ECI frame, with a circular chief, both spacecraft
         satisfy x(t + T) = x(t) exactly.  We exploit this as a global
         topology prior the per-step Kalman filter would otherwise
         ignore: maintain a rolling buffer of past deputy state
         estimates and inject a 6-D pseudo-measurement that
             x_dep(k)  ≈  x_dep(k − N_T)
         with σ sized for one orbit's accumulated process noise.  This
         is independent of the chief broadcast (the comparison is
         deputy-to-itself across one period), so it carries
         information ORTHOGONAL to FIMC.

Algorithm summary
-----------------
    Gravity model:   2-body central + J2 oblateness  (toggle INCLUDE_J2)
    Chief filter:    EKF + GPS  (unchanged from baseline)
    Deputy filter:   UKF + sigma-point propagation
        per step    1.  IPLF LOS update                              (#5)
                    2.  Chief-as-extended-object angular size update
                    3.  7-D FIMC update  (ε, h, e)                   (#1)
                    4.  6-D periodic-orbit update if k ≥ N_T          (#4)
        post-pass   5.  URTSS backward smoother

J2 implications
---------------
J2 is a CONSERVATIVE, AXISYMMETRIC perturbation, so:
    ε       conserved exactly             →  FIMC-energy unchanged
    |h|, h_z  conserved exactly           →  FIMC-h still useful
    h_x, h_y  oscillate (RAAN precession) →  σ_FIMC_H bumped to 2e7
    |e|     conserved exactly             →  FIMC-LRL still useful
    e direction precesses (perigee)       →  σ_FIMC_LRL already loose enough
    ECI periodicity broken                →  another reason APPLY_PERIODIC
                                              defaults to False

Outputs
-------
    Console:   per-minute snapshot of true vs estimated states
    Disk:      simple_angles_only_results_v2.npz with the full per-step
               time series (truth, estimate, P-diag, innovations,
               accept flags, smoother).
    Plots:     residuals vs ±3σ for chief and deputy, 3-D trajectories,
               LOS innovations, |error| vs |3σ| for filter and smoother.

Author: Harish Vernekar (2026)
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from collections import deque
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

# ── J2 oblateness perturbation ───────────────────────────────────────────────
# J2 is the dominant zonal harmonic of Earth's gravity field — Earth's
# oblate shape squashes the geoid by ~21 km between equator and pole.
# At 700 km LEO, J2-induced acceleration is ~1e-2 m/s² (≈ 0.13 % of the
# 8 m/s² central gravity), and its dynamical effects over a few orbits
# can dominate the GNC error budget compared with sensor noise.
#
# J2 has nice mathematical structure:
#     - It is a CONSERVATIVE force (gradient of a static potential), so
#       specific orbital energy ε is still exactly conserved.
#     - It is AXISYMMETRIC about Earth's spin axis (z), so the angular
#       momentum component h_z and the magnitude |h| are also conserved.
#     - The transverse h components (h_x, h_y) are NOT conserved — the
#       orbit plane precesses (RAAN drift), at rate
#           dΩ/dt ≈ -1.5 · n · J2 · (R_E/p)² · cos(i)
#       which is ~7°/day for an i=0 LEO at 700 km.
#     - The Laplace-Runge-Lenz eccentricity vector PRECESSES (argument
#       of perigee drifts), at rate
#           dω/dt ≈ 1.5 · n · J2 · (R_E/p)² · (5cos²i − 1)/2
#       which is ~3.4°/day for an i=0 LEO at 700 km.
#
# Implications for our filter:
#     - FIMC-energy:  unchanged, ε conserved exactly under J2.
#     - FIMC-h:       |h| and h_z are conserved, but h_xy can swing by
#                     |h|·sin(Δi) between deputy and chief if their
#                     inclinations differ.  σ_FIMC_H is bumped to 2e7
#                     to absorb this.
#     - FIMC-LRL:     |e| conserved, e direction precesses.  σ_FIMC_LRL
#                     is large enough already.
#     - Periodicity:  ECI orbit is no longer strictly periodic under J2
#                     (RAAN/perigee precession), so this is yet another
#                     reason APPLY_PERIODIC defaults to False.
#
J2_COEFF    = 1.08262668e-3     # Earth J2 coefficient (Vallado, EGM96)
INCLUDE_J2  = True              # set False to fall back to pure 2-body

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
DURATION_S  = 6 * 3600.0                # [s]
N_STEPS     = int(round(DURATION_S / DT_NAV)) + 1
N_PERIOD    = int(round(T_ORBIT / DT_NAV))   # number of nav steps per orbit

# CRITICAL: N_PERIOD * DT_NAV ≠ T_ORBIT exactly (off by ~0.379 s for our
# 700-km LEO setup).  Without compensation, the periodicity constraint
# compares x_dep(t) to x_dep(t − N_PERIOD·DT_NAV) which is OFFSET from
# x_dep(t − T_orbit) by ~0.379 s of real motion → ~|v|·0.379 ≈ 2.8 km of
# position error.  This is exactly the failure mode observed in the
# first run of v2 (the deputy locked onto a multi-km wrong attractor as
# soon as the constraint activated at t ≈ T_orbit).  Fix: propagate the
# stored "one-orbit-ago" estimate BACKWARD by this fractional offset to
# recover x_dep(t − T_orbit) from x_dep(t − N_PERIOD·DT_NAV).
DT_PERIOD_OFFSET = T_ORBIT - N_PERIOD * DT_NAV   # [s]  ≈ 0.379

# Master switch for the periodicity pseudo-measurement.  Default OFF
# until you've confirmed v2 converges with LRL-FIMC + IPLF alone, since
# the constraint adds little information in pure 2-body dynamics (it's
# largely self-fulfilling: x_dep(t-T) propagated forward by T equals the
# filter's own propagation, which the filter already does).  Set to True
# to experiment with it after the discretization fix.
APPLY_PERIODIC = False

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

# Initial estimate biases
SIGMA_INIT_POS_DEP  = 50.0      # [m]
SIGMA_INIT_VEL_DEP  = 0.05      # [m/s]
SIGMA_INIT_POS_CHF  = 50.0      # [m]
SIGMA_INIT_VEL_CHF  = 0.5       # [m/s]
INIT_BIAS_POS_DEP   = np.array([20.0, -15.0, 10.0])
INIT_BIAS_VEL_DEP   = np.array([0.01, -0.008, 0.005])

# Process noise — slightly tighter than v1 because the periodic
# constraint pins long-horizon behaviour better than chief broadcasts.
SIGMA_PROC_POS = 0.05           # [m / √s]
SIGMA_PROC_VEL = 5e-4           # [m/s² / √s]

# NIS gating thresholds
NIS_GATE_GPS_POS = 100.0
NIS_GATE_LOS     = 100.0
NIS_GATE_SIZE    = 100.0
NIS_GATE_FIMC    = 1e6          # 7-D FIMC: keep loose during transient
NIS_GATE_PERIOD  = 1e6          # 6-D periodic constraint: keep loose

# Chief-as-extended-object measurement
CHIEF_CHORD_M       = 5.0       # [m]
SIGMA_SIZE_RAD      = 1.5e-5    # 0.2-pixel sub-pixel centroiding

# ── 7-D FIMC: integrals-of-motion comparison to chief broadcast ──────────────
# σ values are the larger of:
#     (a) the chief broadcast uncertainty for that integral, and
#     (b) the natural deputy-vs-chief offset for the design formation.
#
# Numerical sizing for our setup (700 km circ chief, A=B=2 km Hill ellipse):
#     |h| ≈ 5.3e10 m²/s,   chief δh ≈ 7e4 m²/s
#     |e_chief| = 0 (circular),   |e_dep| ≈ A/(2 R) ≈ 1.4e-4 (small ecc.
#         induced by Hill radial oscillation)
#     |i_dep − i_chief| ≈ B/R ≈ 2.9e-4 rad ⇒ δh ≈ |h|·sin(i) ≈ 1.5e7
#
# To keep ALL three pieces (ε, h, e) honestly informative, we size σ at
# the deputy/chief offset level — looser than chief broadcast but still
# tight enough that a wandering deputy estimate gets pulled back.
SIGMA_FIMC_EPS      = 80.0      # [m²/s²]   ε exact under J2 — keep tight
SIGMA_FIMC_H        = 2.0e7     # [m²/s]    bumped for J2 RAAN precession
                                #           |h_x|, |h_y| ≲ |h|·sin(i) ≈ 1.5e7
SIGMA_FIMC_LRL      = 5.0e-4    # dimensionless, honours deputy ecc + chief δe
                                #           plus J2-driven argument-of-perigee
                                #           drift (~3.4°/day at our orbit)

# ── Periodic-orbit pseudo-measurement: x_dep(k) ≈ x_dep(k − N_T) ──────────────
# σ should reflect the accumulated process noise over one orbital period
# PLUS the filter's own posterior uncertainty from one orbit ago.  We
# use a generous fixed value so the constraint is informative without
# locking the filter into a self-fulfilling bias.
SIGMA_PERIOD_POS    = 8.0       # [m]     ≈ √(2 T_orbit)·σ_proc_pos + history σ
SIGMA_PERIOD_VEL    = 0.012     # [m/s]   ≈ √(2 T_orbit)·σ_proc_vel + history σ

# ── IPLF iteration parameters ────────────────────────────────────────────────
IPLF_MAX_ITER       = 5         # typically converges in 2-3 iterations
IPLF_TOL_POS        = 1e-3      # [m]    convergence tolerance on Δx_pos
IPLF_TOL_VEL        = 1e-6      # [m/s]  convergence tolerance on Δx_vel

# Output cadence
PRINT_EVERY_S = 60.0
PRINT_EVERY_K = int(round(PRINT_EVERY_S / DT_NAV))

# Output paths
OUT_DIR  = Path(__file__).resolve().parent
NPZ_PATH = OUT_DIR / "simple_angles_only_results_v2.npz"
SAVE_FIGS = False
FIG_DIR  = OUT_DIR / "simple_angles_only_figs_v2"

RNG_SEED = 1234


# ──────────────────────────────────────────────────────────────────────────────
#                  dynamics:  ECI two-body  +  J2 oblateness
# ──────────────────────────────────────────────────────────────────────────────

def two_body_accel(r: np.ndarray) -> np.ndarray:
    """Central two-body acceleration:  a = −μ r / |r|³ ."""
    rn = la.norm(r)
    return -MU_EARTH * r / (rn ** 3)


def j2_accel(r: np.ndarray) -> np.ndarray:
    """
    J2 oblateness perturbation acceleration in ECI (z = Earth spin axis).

        a_J2 = −(3/2) · μ · J2 · R_E² / r⁵ ·
               [ x · (1 − 5·(z/r)²),
                 y · (1 − 5·(z/r)²),
                 z · (3 − 5·(z/r)²) ]

    For 700-km LEO, |a_J2| ≈ 1.5e-2 m/s², about 0.13 % of central gravity.
    """
    x, y, z = r
    rn = la.norm(r)
    if rn < 1e-6:
        return np.zeros(3)
    rn2 = rn * rn
    rn5 = rn2 * rn2 * rn
    coeff = -1.5 * J2_COEFF * MU_EARTH * (R_EARTH ** 2) / rn5
    s = (z / rn) ** 2
    return coeff * np.array([
        x * (1.0 - 5.0 * s),
        y * (1.0 - 5.0 * s),
        z * (3.0 - 5.0 * s),
    ])


def gravity_accel(r: np.ndarray) -> np.ndarray:
    """Total gravitational acceleration: 2-body + (optionally) J2."""
    if INCLUDE_J2:
        return two_body_accel(r) + j2_accel(r)
    return two_body_accel(r)


def rk4_step(state6: np.ndarray, dt: float) -> np.ndarray:
    """One RK4 step on a 6-state [r, v] under the chosen gravity model."""
    def f(s):
        return np.hstack([s[3:6], gravity_accel(s[:3])])
    k1 = f(state6)
    k2 = f(state6 + 0.5 * dt * k1)
    k3 = f(state6 + 0.5 * dt * k2)
    k4 = f(state6 +       dt * k3)
    return state6 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def propagate_truth(state6: np.ndarray, dt_total: float,
                    n_substeps: int = N_RK4_SUBSTEPS) -> np.ndarray:
    """Multi-step RK4 propagation under the chosen gravity model."""
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
#                       Keplerian integrals of motion
# ──────────────────────────────────────────────────────────────────────────────

def specific_energy(r: np.ndarray, v: np.ndarray) -> float:
    """ ε = ½|v|² − μ/|r| """
    return 0.5 * np.dot(v, v) - MU_EARTH / la.norm(r)


def specific_ang_momentum(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """ h = r × v """
    return np.cross(r, v)


def laplace_runge_lenz(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Laplace-Runge-Lenz (eccentricity) vector.

        e = (v × h)/μ − r̂

    For a Keplerian orbit, e points from the focus toward perigee, with
    magnitude equal to the orbital eccentricity.  e is a CONSERVED
    integral of motion in the pure two-body problem.
    """
    h = np.cross(r, v)
    rn = la.norm(r)
    return np.cross(v, h) / MU_EARTH - r / rn


def kepler_invariants(state6: np.ndarray) -> np.ndarray:
    """
    Pack the seven Keplerian integrals into a 7-vector:
        [ε,  h_x, h_y, h_z,  e_x, e_y, e_z]

    All seven are conserved exactly under pure two-body dynamics.
    """
    r = state6[:3]
    v = state6[3:]
    eps = specific_energy(r, v)
    h   = specific_ang_momentum(r, v)
    e   = laplace_runge_lenz(r, v)
    return np.hstack([[eps], h, e])


# ──────────────────────────────────────────────────────────────────────────────
#                  measurement models
# ──────────────────────────────────────────────────────────────────────────────

def gps_position_measurement(r_truth: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
    return r_truth + SIGMA_GPS_POS * rng.normal(size=3)


def los_angles(r_observer: np.ndarray, r_target: np.ndarray
               ) -> Tuple[float, float, np.ndarray]:
    dr = r_target - r_observer
    dn = la.norm(dr)
    alpha = np.arctan2(dr[1], dr[0])
    delta = np.arcsin(np.clip(dr[2] / dn, -1.0, 1.0))
    return alpha, delta, dr


def los_jacobian_wrt_observer(dr: np.ndarray) -> np.ndarray:
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
#                                EKF (6-D, ECI)  —  used for the chief
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EKFConfig:
    sigma_proc_pos: float = SIGMA_PROC_POS
    sigma_proc_vel: float = SIGMA_PROC_VEL
    sigma_init_pos: float = 100.0
    sigma_init_vel: float = 0.5
    use_joseph:     bool  = True


class TwoBodyEKF:
    """ 6-state EKF in ECI; chief uses this with a linear GPS update. """

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
#                     UKF + IPLF (6-D, ECI)  —  used for the deputy
# ──────────────────────────────────────────────────────────────────────────────

class TwoBodyUKF_IPLF:
    """
    6-state Unscented Kalman Filter for the 2-body ECI problem, with an
    Iterated Posterior Linearisation Filter (IPLF) measurement update
    available alongside the standard one-shot UKF update.

    IPLF (García-Fernández et al., 2015) replaces the single statistical
    linearisation around the prior with an inner loop that re-linearises
    around the current posterior estimate, then re-runs the KF update
    against the original prior, and iterates to convergence.  The
    statistical Jacobian
        A^(i) = Pxz^(i)·(P^(i))^-1
    and offset
        b^(i) = ẑ^(i) − A^(i)·x^(i)
    are recomputed at every iteration.  For non-linear measurements
    like atan2/asin (LOS angles), this cuts the per-update bias by 1-2
    orders of magnitude over a single-pass UKF.
    """

    def __init__(self, name: str, cfg: EKFConfig,
                 x0: np.ndarray, P0: Optional[np.ndarray] = None,
                 alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.name = name
        self.cfg  = cfg
        self.n    = 6
        self.x = np.asarray(x0, float).reshape(6).copy()
        if P0 is None:
            pp2 = cfg.sigma_init_pos ** 2
            pv2 = cfg.sigma_init_vel ** 2
            self.P = np.diag([pp2]*3 + [pv2]*3)
        else:
            self.P = np.asarray(P0, float).reshape(6, 6).copy()
        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()
        self.sp_pred: Optional[np.ndarray] = None
        self.P_cross = np.zeros((6, 6))

        # Unscented transform parameters (Wan & van der Merwe, 2000)
        self.alpha_ut = alpha
        self.beta_ut  = beta
        self.kappa    = kappa
        self.lam      = alpha**2 * (self.n + kappa) - self.n
        n2p1 = 2 * self.n + 1
        self.Wm = np.full(n2p1, 1.0 / (2.0 * (self.n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.Wm[0] + (1.0 - alpha**2 + beta)

        self.t_last: Optional[float] = None
        self.n_predict = 0
        self.n_update_accept = 0
        self.n_update_reject = 0
        self.n_iplf_iters    = 0

    # ──────────────── sigma points ────────────────
    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        scale = (self.n + self.lam)
        try:
            S = np.linalg.cholesky(scale * P)
        except np.linalg.LinAlgError:
            jitter = 1e-9 * (np.max(np.abs(np.diag(P))) + 1.0)
            S = np.linalg.cholesky(scale * (P + jitter * np.eye(self.n)))
        sp = np.zeros((2*self.n + 1, self.n))
        sp[0] = x
        for i in range(self.n):
            sp[i + 1]          = x + S[:, i]
            sp[i + 1 + self.n] = x - S[:, i]
        return sp

    # ──────────────── predict ────────────────
    def predict(self, t_abs: float):
        if self.t_last is None:
            self.t_last = t_abs - DT_NAV
        dt = t_abs - self.t_last

        sp = self._sigma_points(self.x, self.P)
        sp_pred = np.empty_like(sp)
        for i in range(sp.shape[0]):
            sp_pred[i] = propagate_truth(sp[i], dt, n_substeps=N_RK4_SUBSTEPS)

        x_pred = (self.Wm[:, None] * sp_pred).sum(axis=0)
        P_pred  = np.zeros((self.n, self.n))
        P_cross = np.zeros((self.n, self.n))
        x_post  = self.x.copy()
        for i in range(sp.shape[0]):
            d_pred = sp_pred[i] - x_pred
            P_pred  += self.Wc[i] * np.outer(d_pred, d_pred)
            P_cross += self.Wc[i] * np.outer(sp[i] - x_post, d_pred)

        qp = (self.cfg.sigma_proc_pos ** 2) * dt
        qv = (self.cfg.sigma_proc_vel ** 2) * dt
        P_pred += np.diag([qp]*3 + [qv]*3)
        P_pred = 0.5 * (P_pred + P_pred.T)

        self.x_pred  = x_pred
        self.P_pred  = P_pred
        self.sp_pred = sp_pred
        self.P_cross = P_cross
        self.t_last  = t_abs
        self.n_predict += 1

    # ──────────────── one-shot UKF measurement update ────────────────
    def update(self, z: np.ndarray, h_fn: Callable, R: np.ndarray,
               nis_gate: float = 1e10, label: str = "",
               aux=None) -> dict:
        z = np.asarray(z, float).reshape(-1)
        n_z = z.size

        Z = np.empty((2 * self.n + 1, n_z))
        for i in range(2 * self.n + 1):
            Z[i] = h_fn(self.sp_pred[i], aux)

        z_pred = (self.Wm[:, None] * Z).sum(axis=0)
        if label.startswith("LOS"):
            for i in range(2 * self.n + 1):
                d = Z[i, 0] - z_pred[0]
                if d > np.pi:
                    Z[i, 0] -= 2 * np.pi
                elif d < -np.pi:
                    Z[i, 0] += 2 * np.pi
            z_pred = (self.Wm[:, None] * Z).sum(axis=0)

        S   = np.array(R, dtype=float).copy()
        Pxz = np.zeros((self.n, n_z))
        for i in range(2 * self.n + 1):
            dz = Z[i] - z_pred
            dx = self.sp_pred[i] - self.x_pred
            S   += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        y = z - z_pred
        if label.startswith("LOS"):
            y[0] = (y[0] + np.pi) % (2 * np.pi) - np.pi

        try:
            nis = float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            nis = float(y @ y)
        if nis > nis_gate:
            self.x = self.x_pred.copy(); self.P = self.P_pred.copy()
            self.n_update_reject += 1
            return dict(accepted=False, nis=nis, y=y, S=S)

        try:
            K = Pxz @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.x = self.x_pred.copy(); self.P = self.P_pred.copy()
            self.n_update_reject += 1
            return dict(accepted=False, nis=nis, y=y, S=S)

        self.x = self.x_pred + K @ y
        self.P = self.P_pred - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Refresh prediction state for any subsequent updates this step
        self.x_pred  = self.x.copy(); self.P_pred = self.P.copy()
        self.sp_pred = self._sigma_points(self.x, self.P)
        self.n_update_accept += 1
        return dict(accepted=True, nis=nis, y=y, S=S)

    # ──────────────── IPLF measurement update (NOVEL) ────────────────
    def update_iplf(self, z: np.ndarray, h_fn: Callable, R: np.ndarray,
                    nis_gate: float = 1e10, label: str = "",
                    aux=None,
                    max_iter: int = IPLF_MAX_ITER,
                    tol_pos: float = IPLF_TOL_POS,
                    tol_vel: float = IPLF_TOL_VEL) -> dict:
        """
        Iterated Posterior Linearisation Filter measurement update.

        Algorithm (García-Fernández et al., 2015, "Posterior linearisation
        filter: principles and implementation using sigma points",
        IEEE TSP 63(20)):

        Set x^(0) = x_pred, P^(0) = P_pred.
        For i = 0, 1, 2, ... until convergence:
            1.  Compute sigma points {χ_j^(i)} around (x^(i), P^(i))
            2.  Pass them through h:           Z_j^(i) = h(χ_j^(i), aux)
            3.  ẑ^(i)  = Σ Wm_j · Z_j^(i)
            4.  Pzz^(i) = R + Σ Wc_j (Z_j^(i) − ẑ^(i))(Z_j^(i) − ẑ^(i))^T
                Pxz^(i) = Σ Wc_j (χ_j^(i) − x^(i))(Z_j^(i) − ẑ^(i))^T
            5.  A^(i) = Pxz^(i)^T · (P^(i))^-1     (statistical Jacobian)
                b^(i) = ẑ^(i) − A^(i) · x^(i)      (offset)
                Ω^(i) = Pzz^(i) − A^(i) · P^(i) · A^(i)^T  (residual var)
            6.  KF update around the ORIGINAL prior with linear (A,b,Ω):
                    S^(i)    = Ω^(i) + A^(i) · P_pred · A^(i)^T
                    K^(i)    = P_pred · A^(i)^T · (S^(i))^-1
                    x^(i+1)  = x_pred + K^(i) · (z − b^(i) − A^(i)·x_pred)
                    P^(i+1)  = (I − K^(i)·A^(i)) · P_pred · (...)^T
                                + K^(i)·Ω^(i)·K^(i)^T          [Joseph form]
            7.  Stop when ||x^(i+1) − x^(i)|| < tol.
        """
        z = np.asarray(z, float).reshape(-1)
        n_z = z.size
        I6 = np.eye(self.n)

        # Initialise iterate at the prior
        x_iter = self.x_pred.copy()
        P_iter = self.P_pred.copy()

        # Unwrap helpers for angle innovations
        def _wrap_minus(z, z_pred):
            d = z - z_pred
            if label.startswith("LOS"):
                d[0] = (d[0] + np.pi) % (2 * np.pi) - np.pi
            return d

        x_new = x_iter
        P_new = P_iter
        last_S = R.copy()
        last_y = np.zeros(n_z)
        last_nis = 0.0
        n_iter_done = 0

        for it in range(max_iter):
            n_iter_done = it + 1

            # 1-2: Sigma points around current iterate, push through h
            sp_iter = self._sigma_points(x_iter, P_iter)
            Z = np.empty((2 * self.n + 1, n_z))
            for i in range(2 * self.n + 1):
                Z[i] = h_fn(sp_iter[i], aux)

            # 3: Predicted measurement (with optional angle unwrap)
            z_pred = (self.Wm[:, None] * Z).sum(axis=0)
            if label.startswith("LOS"):
                for i in range(2 * self.n + 1):
                    d = Z[i, 0] - z_pred[0]
                    if d > np.pi:
                        Z[i, 0] -= 2 * np.pi
                    elif d < -np.pi:
                        Z[i, 0] += 2 * np.pi
                z_pred = (self.Wm[:, None] * Z).sum(axis=0)

            # 4: Pxz, Pzz (statistical moments)
            Pzz = np.array(R, dtype=float).copy()
            Pxz = np.zeros((self.n, n_z))
            for i in range(2 * self.n + 1):
                dz = Z[i] - z_pred
                dx = sp_iter[i] - x_iter
                Pzz += self.Wc[i] * np.outer(dz, dz)
                Pxz += self.Wc[i] * np.outer(dx, dz)

            # 5: Statistical linearisation (A, b, Ω)
            try:
                P_iter_inv = np.linalg.inv(P_iter +
                                           1e-12 * np.eye(self.n))
            except np.linalg.LinAlgError:
                break
            A = Pxz.T @ P_iter_inv             # (n_z × n)
            b = z_pred - A @ x_iter            # (n_z,)
            Omega = Pzz - A @ P_iter @ A.T     # residual measurement noise
            Omega = 0.5 * (Omega + Omega.T)

            # 6: KF update around the ORIGINAL prior with the linearised
            #    measurement model z ≈ A·x + b + N(0, Ω).
            S = Omega + A @ self.P_pred @ A.T
            S = 0.5 * (S + S.T)
            try:
                K = self.P_pred @ A.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                break

            innov = _wrap_minus(z, b + A @ self.x_pred)
            x_new = self.x_pred + K @ innov
            KA = K @ A
            P_new = ((I6 - KA) @ self.P_pred @ (I6 - KA).T
                     + K @ Omega @ K.T)
            P_new = 0.5 * (P_new + P_new.T)

            # NIS computed against the linearised innovation covariance
            try:
                last_nis = float(innov @ np.linalg.solve(S, innov))
            except np.linalg.LinAlgError:
                last_nis = float(innov @ innov)
            last_S = S
            last_y = innov

            # 7: Convergence check on the iterate
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

        self.x = x_new
        self.P = P_new
        self.x_pred  = self.x.copy(); self.P_pred = self.P.copy()
        self.sp_pred = self._sigma_points(self.x, self.P)
        self.n_update_accept += 1
        return dict(accepted=True, nis=last_nis, y=last_y, S=last_S,
                    n_iter=n_iter_done)


# ──────────────────────────────────────────────────────────────────────────────
#                          truth + sim driver
# ──────────────────────────────────────────────────────────────────────────────

def initial_chief_state() -> np.ndarray:
    return np.array([R_ORBIT, 0.0, 0.0,    0.0, V_CIRC, 0.0])


def initial_deputy_state() -> np.ndarray:
    """ Deputy on a 3-D Hill ellipse — identical to v1. """
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

    chief_ekf = TwoBodyEKF("CHIEF", chief_cfg, x0_chief)
    dep_ekf   = TwoBodyUKF_IPLF("DEPUTY", dep_cfg, x0_dep)

    # ── histories ──
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
    los_iters   = np.zeros(N_STEPS, dtype=int)
    accept_chief = np.zeros(N_STEPS, dtype=bool)
    accept_dep   = np.zeros(N_STEPS, dtype=bool)
    period_active = np.zeros(N_STEPS, dtype=bool)

    # Smoother histories
    sm_x_post   = np.zeros((N_STEPS, 6))
    sm_P_post   = np.zeros((N_STEPS, 6, 6))
    sm_x_pred   = np.zeros((N_STEPS, 6))
    sm_P_pred   = np.zeros((N_STEPS, 6, 6))
    sm_P_cross  = np.zeros((N_STEPS, 6, 6))

    # Periodic-orbit buffer: ring buffer of past deputy state estimates
    # of length N_PERIOD so we can read state from one orbit ago.
    periodic_buf = np.zeros((N_PERIOD, 6))

    # Initialise step 0
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
    periodic_buf[0] = dep_ekf.x.copy()

    if verbose:
        print("=" * 100)
        print("  ANGLES-ONLY RELNAV  v2 :  LRL-extended FIMC + IPLF + topological periodicity")
        print("=" * 100)
        if INCLUDE_J2:
            print(f"  Gravity model     : 2-body + J2   (J2 = {J2_COEFF:.6e})")
        else:
            print(f"  Gravity model     : 2-body only")
        print(f"  Periodicity ON?   : {APPLY_PERIODIC}")
        print(f"  Chief altitude    : {ALTITUDE/1e3:.1f} km")
        print(f"  Orbit period      : {T_ORBIT:.3f} s   ({T_ORBIT/60:.3f} min)")
        print(f"  Mean motion n     : {N_ORBIT:.6e} rad/s")
        print(f"  N_PERIOD          : {N_PERIOD} nav steps per orbit")
        print(f"  Run length        : {DURATION_S/3600:.2f} h ({DURATION_S/T_ORBIT:.2f} orbits, "
              f"{N_STEPS} nav steps)")
        print(f"  σ_LOS effective   : {SIGMA_LOS_EFF:.3e} rad ({SIGMA_LOS_EFF*1e6:.3f} µrad)")
        print(f"  σ_FIMC_eps        : {SIGMA_FIMC_EPS:.3f} m²/s²")
        print(f"  σ_FIMC_h          : {SIGMA_FIMC_H:.3e} m²/s")
        print(f"  σ_FIMC_LRL        : {SIGMA_FIMC_LRL:.3e} (dimensionless)")
        print(f"  σ_periodic        : {SIGMA_PERIOD_POS:.3f} m / {SIGMA_PERIOD_VEL:.4f} m/s")
        print(f"  IPLF max iters    : {IPLF_MAX_ITER}")
        print()

    # ── main loop ──
    for k in range(1, N_STEPS):
        t = t_arr[k]

        # Truth propagation
        x_chief_truth = propagate_truth(x_chief_truth, DT_NAV)
        x_dep_truth   = propagate_truth(x_dep_truth,   DT_NAV)

        # Predict both filters
        chief_ekf.predict(t)
        dep_ekf.predict(t)

        # Save smoother prior histories (right after predict, before any update)
        sm_x_pred[k]  = dep_ekf.x_pred.copy()
        sm_P_pred[k]  = dep_ekf.P_pred.copy()
        sm_P_cross[k] = dep_ekf.P_cross.copy()

        # ── Chief: GPS update ──
        z_gps = gps_position_measurement(x_chief_truth[:3], rng)
        H_gps = np.zeros((3, 6)); H_gps[:, :3] = np.eye(3)
        R_gps = (SIGMA_GPS_POS**2) * np.eye(3)
        zhat_gps = chief_ekf.x_pred[:3]
        info = chief_ekf.update(z_gps, zhat_gps, H_gps, R_gps,
                                NIS_GATE_GPS_POS, label="GPS")
        accept_chief[k] = info["accepted"]
        gps_innov[k] = info["y"] if info["accepted"] else np.nan

        # ── Deputy: LOS update via IPLF ──  (#5)
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

        # ── Chief-as-extended-object: monocular range from chord/range ──
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

        # ── 7-D FIMC: ε + h + e(LRL)  vs  chief ──  (#1)
        # Chief broadcasts its full 7-element invariant vector (1 scalar
        # + 3 + 3).  Deputy enforces match with σ values that allow the
        # natural co-orbital offset for our Hill-ellipse formation.
        z_fimc = kepler_invariants(chief_ekf.x)
        R_fimc = np.diag([
            SIGMA_FIMC_EPS**2,
            SIGMA_FIMC_H**2, SIGMA_FIMC_H**2, SIGMA_FIMC_H**2,
            SIGMA_FIMC_LRL**2, SIGMA_FIMC_LRL**2, SIGMA_FIMC_LRL**2,
        ])

        def _fimc_h(x_dep, _aux):
            return kepler_invariants(x_dep)

        dep_ekf.update(z_fimc, _fimc_h, R_fimc,
                       nis_gate=NIS_GATE_FIMC, label="FIMC",
                       aux=None)

        # ── Topology-aware periodic-orbit pseudo-measurement ──  (#4)
        # For a co-orbital deputy (same orbital period as chief), the
        # ECI state satisfies x(t + T) = x(t) up to process-noise drift.
        # Use the deputy's own state estimate from one orbit ago as a
        # 6-D pseudo-measurement on the current state.  Information is
        # ORTHOGONAL to FIMC (which compares to chief, not to self).
        if APPLY_PERIODIC and k >= N_PERIOD:
            x_old = periodic_buf[k % N_PERIOD]
            # Discretization fix: x_old corresponds to t_old = t − N_PERIOD·DT_NAV,
            # which differs from t − T_orbit by DT_PERIOD_OFFSET ≈ 0.379 s.
            # By periodicity, x_dep(t) = x_dep(t − T_orbit) = x_dep(t_old − DT_PERIOD_OFFSET)
            # = x_old propagated BACKWARD by DT_PERIOD_OFFSET seconds.
            if abs(DT_PERIOD_OFFSET) > 1e-9:
                x_period = propagate_truth(x_old, -DT_PERIOD_OFFSET,
                                            n_substeps=max(1, N_RK4_SUBSTEPS // 5))
            else:
                x_period = x_old.copy()
            z_period = x_period
            R_period = np.diag([SIGMA_PERIOD_POS**2]*3 +
                                [SIGMA_PERIOD_VEL**2]*3)

            def _period_h(x_dep, _aux):
                return x_dep.copy()

            dep_ekf.update(z_period, _period_h, R_period,
                           nis_gate=NIS_GATE_PERIOD, label="PERIOD",
                           aux=None)
            period_active[k] = True

        # Store CURRENT estimate into the ring buffer
        periodic_buf[k % N_PERIOD] = dep_ekf.x.copy()

        # Save smoother posterior histories
        sm_x_post[k] = dep_ekf.x.copy()
        sm_P_post[k] = dep_ekf.P.copy()

        # Record
        truth_chief[k] = x_chief_truth
        truth_dep[k]   = x_dep_truth
        est_chief[k]   = chief_ekf.x
        est_dep[k]     = dep_ekf.x
        P_chief[k]     = np.diag(chief_ekf.P)
        P_dep[k]       = np.diag(dep_ekf.P)

        # Periodic console printout
        if verbose and (k % PRINT_EVERY_K == 0 or k == N_STEPS - 1):
            err_c = est_chief[k] - truth_chief[k]
            err_d = est_dep[k]   - truth_dep[k]
            sigma_d = np.sqrt(P_dep[k])
            range_now = la.norm(truth_dep[k, :3] - truth_chief[k, :3])
            tag_p = " +PERIOD" if period_active[k] else ""
            print(f"t = {t:11.3f} s   ({t/60:7.2f} min,  orbit {t/T_ORBIT:.3f}){tag_p}")
            print(f"  CHIEF  |Δr| = {la.norm(err_c[:3]):8.3f} m   "
                  f"|Δv| = {la.norm(err_c[3:6])*1e3:8.3f} mm/s")
            print(f"  DEPUTY |Δr| = {la.norm(err_d[:3]):8.3f} m   "
                  f"|Δv| = {la.norm(err_d[3:6])*1e3:8.3f} mm/s   "
                  f"(σ_pos = [{sigma_d[0]:.3f}, {sigma_d[1]:.3f}, "
                  f"{sigma_d[2]:.3f}] m)   "
                  f"IPLF iters = {los_iters[k]}")
            print(f"  Range chief↔deputy (truth) = {range_now:.3f} m")
            print()

    # ── URTSS backward smoother ──
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

    # ── summary ──
    if verbose:
        i0 = int(0.5 * T_ORBIT / DT_NAV)
        print("\n" + "=" * 100)
        print("  RESULTS SUMMARY  v2  (after 0.5-orbit transient)")
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
            if hasattr(ekf, "n_iplf_iters"):
                avg_iters = ekf.n_iplf_iters / max(1, np.count_nonzero(los_iters))
                print(f"    avg IPLF iters   : {avg_iters:.2f}")
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

        # URTSS smoother
        print("\n  URTSS SMOOTHER  (forward IPLF/UKF + backward pass)")
        err_filt = est_dep - truth_dep
        err_smth = smooth_x - truth_dep
        rms_filt_pos = np.sqrt(np.mean(np.sum(err_filt[i0:, :3]**2, axis=1)))
        rms_smth_pos = np.sqrt(np.mean(np.sum(err_smth[i0:, :3]**2, axis=1)))
        rms_filt_vel = np.sqrt(np.mean(np.sum(err_filt[i0:, 3:6]**2, axis=1)))
        rms_smth_vel = np.sqrt(np.mean(np.sum(err_smth[i0:, 3:6]**2, axis=1)))
        print(f"    Deputy filter   RMS pos = {rms_filt_pos:.6f} m   "
              f"RMS vel = {rms_filt_vel:.9f} m/s")
        print(f"    Deputy smoother RMS pos = {rms_smth_pos:.6f} m   "
              f"RMS vel = {rms_smth_vel:.9f} m/s")
        if rms_filt_pos > 0:
            print(f"    Smoother improvement (pos): "
                  f"{(1.0 - rms_smth_pos/rms_filt_pos)*100:+.2f} %")
            print(f"    Smoother improvement (vel): "
                  f"{(1.0 - rms_smth_vel/rms_filt_vel)*100:+.2f} %")
        print()

    # ── save ──
    np.savez(NPZ_PATH,
             t=t_arr,
             truth_chief=truth_chief, est_chief=est_chief, P_chief_diag=P_chief,
             truth_dep=truth_dep,     est_dep=est_dep,     P_dep_diag=P_dep,
             gps_innov=gps_innov, los_innov=los_innov,
             los_alpha=los_alpha, los_delta=los_delta,
             los_iters=los_iters,
             accept_chief=accept_chief, accept_dep=accept_dep,
             period_active=period_active,
             smooth_x=smooth_x,
             smooth_P_diag=np.array([np.diag(p) for p in smooth_P]))
    if verbose:
        print(f"  Saved per-step time series to {NPZ_PATH}\n")

    return dict(
        t=t_arr,
        truth_chief=truth_chief, est_chief=est_chief, P_chief=P_chief,
        truth_dep=truth_dep,     est_dep=est_dep,     P_dep=P_dep,
        smooth_x=smooth_x,
        smooth_P_diag=np.array([np.diag(p) for p in smooth_P]),
        gps_innov=gps_innov, los_innov=los_innov, los_iters=los_iters,
        accept_chief=accept_chief, accept_dep=accept_dep,
        period_active=period_active,
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
    fig.suptitle("CHIEF (GPS-aided)  v2  — residuals vs 3σ", fontsize=14)
    for i in range(3):
        _residual_panel(axes[0, i], t_h, err_c[:, i], three_sigma_c[:, i],
                        f"r_{labels[i]} [m]", f"Position {labels[i]}")
        _residual_panel(axes[1, i], t_h, err_c[:, 3+i], three_sigma_c[:, 3+i],
                        f"v_{labels[i]} [m/s]", f"Velocity {labels[i]}")
    for ax in axes[1]: ax.set_xlabel("Time [h]")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "chief_residuals.png", dpi=150)

    # ── 2) deputy residuals vs 3σ ──
    err_d = res["est_dep"] - res["truth_dep"]
    three_sigma_d = 3.0 * np.sqrt(np.maximum(res["P_dep"], 0.0))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    fig.suptitle("DEPUTY  v2  (LRL-FIMC + IPLF + periodicity) — residuals vs 3σ",
                 fontsize=14)
    for i in range(3):
        _residual_panel(axes[0, i], t_h, err_d[:, i], three_sigma_d[:, i],
                        f"r_{labels[i]} [m]", f"Position {labels[i]}")
        _residual_panel(axes[1, i], t_h, err_d[:, 3+i], three_sigma_d[:, 3+i],
                        f"v_{labels[i]} [m/s]", f"Velocity {labels[i]}")
    for ax in axes[1]: ax.set_xlabel("Time [h]")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "deputy_residuals.png", dpi=150)

    # ── 3) deputy convergence: filter vs smoother (log scale) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DEPUTY error magnitude convergence — v2", fontsize=14)
    err_pos_norm = la.norm(err_d[:, :3], axis=1)
    err_vel_norm = la.norm(err_d[:, 3:6], axis=1)
    sigma_pos_norm = np.sqrt(np.sum(np.maximum(res["P_dep"][:, :3], 0.0), axis=1))
    sigma_vel_norm = np.sqrt(np.sum(np.maximum(res["P_dep"][:, 3:6], 0.0), axis=1))
    ax1.plot(t_h, err_pos_norm, lw=0.7, label="filter |error|", color="C0")
    ax1.plot(t_h, 3 * sigma_pos_norm, "r--", lw=1, label="filter 3σ")
    err_smooth = res["smooth_x"] - res["truth_dep"]
    sm_norm = la.norm(err_smooth[:, :3], axis=1)
    sm_sigma = np.sqrt(np.sum(np.maximum(res["smooth_P_diag"][:, :3], 0.0),
                              axis=1))
    ax1.plot(t_h, sm_norm, lw=1.0, color="C2", label="smoother |error|")
    ax1.plot(t_h, 3 * sm_sigma, "g--", lw=1, alpha=0.6, label="smoother 3σ")
    ax1.set_yscale("log"); ax1.set_ylabel("|r_error| [m]")
    ax1.set_xlabel("Time [h]"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax1.set_title("Position — filter vs URTSS smoother")
    ax2.plot(t_h, err_vel_norm, lw=0.7, label="filter |error|", color="C0")
    ax2.plot(t_h, 3 * sigma_vel_norm, "r--", lw=1, label="filter 3σ")
    sm_v = la.norm(err_smooth[:, 3:], axis=1)
    sm_v_sigma = np.sqrt(np.sum(np.maximum(res["smooth_P_diag"][:, 3:], 0.0),
                                 axis=1))
    ax2.plot(t_h, sm_v, lw=1.0, color="C2", label="smoother |error|")
    ax2.plot(t_h, 3 * sm_v_sigma, "g--", lw=1, alpha=0.6, label="smoother 3σ")
    ax2.set_yscale("log"); ax2.set_ylabel("|v_error| [m/s]")
    ax2.set_xlabel("Time [h]"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    ax2.set_title("Velocity — filter vs URTSS smoother")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "deputy_convergence.png", dpi=150)

    # ── 4) IPLF iteration count history ──
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.step(t_h, res["los_iters"], where="post", lw=0.8)
    ax.set_xlabel("Time [h]"); ax.set_ylabel("IPLF iterations on LOS update")
    ax.set_title("IPLF iteration count per nav step (v2)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if SAVE_FIGS: fig.savefig(FIG_DIR / "iplf_iters.png", dpi=150)

    # ── 4b) Relative position residual: (Δr_est) − (Δr_truth) ─────────────
    # The user-requested check on monotonic convergence of the deputy's
    # relative position estimate.
    #
    #     Δr_est    = r_chief_est   − r_dep_est        (relative pos estimate)
    #     Δr_truth  = r_chief_truth − r_dep_truth      (relative pos truth)
    #     residual  = Δr_est − Δr_truth
    #              = (r_chief_est − r_chief_truth) − (r_dep_est − r_dep_truth)
    #              = chief_pos_error − deputy_pos_error
    #
    # If the chief is well-anchored by GPS (~5 m), this ≈ −deputy_pos_error.
    # We plot per-axis time series, the norm on log scale, and the truth
    # vs estimate range histories side by side.
    rel_est   = res["est_chief"][:, :3]   - res["est_dep"][:, :3]
    rel_truth = res["truth_chief"][:, :3] - res["truth_dep"][:, :3]
    rel_res   = rel_est - rel_truth
    rel_res_norm   = la.norm(rel_res,   axis=1)
    rel_truth_norm = la.norm(rel_truth, axis=1)
    rel_est_norm   = la.norm(rel_est,   axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("DEPUTY relative position estimate — convergence diagnostics",
                 fontsize=14)

    # Top-left: 3-component time series of relative position residual
    for j, lab in enumerate(["x", "y", "z"]):
        axes[0, 0].plot(t_h, rel_res[:, j], lw=0.6, alpha=0.85,
                        label=f"Δr_{lab} residual")
    axes[0, 0].axhline(0, color="k", lw=0.5)
    axes[0, 0].set_xlabel("Time [h]")
    axes[0, 0].set_ylabel("Δr_est − Δr_truth  [m]")
    axes[0, 0].set_title("Relative position residual per axis")
    axes[0, 0].legend(fontsize=9); axes[0, 0].grid(True, alpha=0.3)

    # Top-right: norm of residual on log scale (the headline plot for
    # checking monotonic convergence).
    axes[0, 1].semilogy(t_h, np.maximum(rel_res_norm, 1e-3), lw=0.7, color="C3")
    axes[0, 1].set_xlabel("Time [h]")
    axes[0, 1].set_ylabel("|Δr_est − Δr_truth|  [m]")
    axes[0, 1].set_title("Relative position residual norm  (LOG scale)")
    axes[0, 1].grid(True, which="both", alpha=0.3)

    # Bottom-left: truth vs estimate range magnitudes, overlaid
    axes[1, 0].plot(t_h, rel_truth_norm, lw=0.7, label="|Δr| truth")
    axes[1, 0].plot(t_h, rel_est_norm,   lw=0.7, label="|Δr| estimate",
                    alpha=0.75)
    axes[1, 0].set_xlabel("Time [h]")
    axes[1, 0].set_ylabel("|Δr|  [m]")
    axes[1, 0].set_title("Chief↔Deputy range: truth vs estimate")
    axes[1, 0].legend(fontsize=9); axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: residual norm on linear scale, with a running
    # cumulative-min envelope to make monotonic improvement visible.
    cmin = np.minimum.accumulate(rel_res_norm)
    axes[1, 1].plot(t_h, rel_res_norm, lw=0.6, alpha=0.7, label="instantaneous")
    axes[1, 1].plot(t_h, cmin, lw=1.2, color="C2",
                    label="running minimum (envelope)")
    axes[1, 1].set_xlabel("Time [h]")
    axes[1, 1].set_ylabel("|Δr_est − Δr_truth|  [m]")
    axes[1, 1].set_title("Relative position residual  (linear scale)")
    axes[1, 1].legend(fontsize=9); axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "relative_residual.png", dpi=150)

    # ── 5) 3-D ECI trajectories ──
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("ECI trajectories — v2 — truth (solid) vs estimate (dashed)",
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


if __name__ == "__main__":
    np.set_printoptions(precision=9, suppress=False, linewidth=200)
    res = run_sim(verbose=True)
    plot_results(res)
    plt.show()
