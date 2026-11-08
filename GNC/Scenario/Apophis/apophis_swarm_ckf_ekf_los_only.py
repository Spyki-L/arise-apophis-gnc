#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_ckf_ekf_los_only.py
=================================

Multi-deputy CKF + EKF demo for the Apophis rendezvous case in a
**pure line-of-sight (camera-only)** navigation configuration.

Sensor concept (per deputy)
---------------------------
Each deputy carries ONLY a body-mounted camera (with star-tracker–assisted
attitude knowledge).  It continuously points at and tracks the chief
mothership and reports the bearing to the chief in the chief Hill (LVLH)
frame:

    z = [ azimuth, elevation ]^T          (2x1 — NO range channel)

    azimuth   = atan2( ρ_T, ρ_R )
    elevation = atan2( ρ_N, sqrt(ρ_R² + ρ_T²) )

with ρ = (deputy − chief) in the chief Hill frame.  There is no ranging
beacon, no GNSS pseudorange, nothing else on the deputy — bearings only.

Why this is harder than range+az+el
-----------------------------------
A bearing measurement is invariant to the deputy–chief *distance*: scaling
ρ by any positive factor leaves (az, el) unchanged.  So range is NOT
instantaneously observable.  It becomes observable only indirectly:

  * the Hill / Clohessy–Wiltshire relative dynamics curve the trajectory,
    and a wrong range predicts a bearing history that slowly drifts off
    the measurements;
  * the known RHIG impulsive burns (Δv in Hill) inject a hard length
    scale the filter can latch onto.

Consequently the EKF error is strongly anisotropic: cross-LOS components
collapse almost immediately to the camera noise floor, while the
along-LOS (range) component converges slowly and to a much looser value.
This script reports that behaviour honestly — there is no tuning target.

Chief navigation
----------------
The chief still broadcasts its GPS position over the ISL (≈5 m raw,
smoothed to ≈1 m before broadcast) so the deputies can build the chief
Hill frame.  Only the deputy-side sensor suite is reduced to camera-only.

State (per deputy)
------------------
    x_H = [ ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N ]^T     (Hill / LVLH wrt chief)

Inputs
------
Deputy trajectory bundles:
    Scenario/Apophis/apophis_rendezvous_full_D0.npz ... D7.npz
Chief truth reference (positions in km, velocities in km/s):
    Scenario/Apophis/chief_reference.npz

Outputs
-------
For each deputy d:
    apophis_swarm_FDI_losonly_D{d}.npz
        t, x_true, x_est, P, y_pre, y_post, sigma_meas (in [rad, rad]),
        dv_seq_H
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# ───────── plotting font controls ─────────

TITLE_FONTSIZE      = 30
AXIS_LABEL_FONTSIZE = 25
TICK_LABEL_FONTSIZE = 22
LEGEND_FONTSIZE     = 22

plt.rcParams.update({
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "xtick.labelsize": TICK_LABEL_FONTSIZE,
    "ytick.labelsize": TICK_LABEL_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    "figure.max_open_warning": 50,
})

# ───────── locate ARISE root ─────────

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

from GNC.Navigation.hill_ekf import (  # type: ignore[import]
    HillEKF,
    HillEKFConfig,
    dcm_lvlh,
    hill_rate,
)

# ───────── configuration ─────────

N_DEP      = 8        # deputies D0..D7
N_CKF      = 0        # CKF pre-pass (0 ⇒ rely on bootstrap + EKF)

# Mid-rate Estimation bus cadence (paper's spec for in-flight filter)
FILTER_RATE_HZ = 0.1
FILTER_DT      = 1.0 / FILTER_RATE_HZ

# Decimate saved FDI bundles to this rate to keep file sizes small.
SAVE_RATE_HZ = 1.0
SAVE_STRIDE  = max(1, int(round(FILTER_RATE_HZ / SAVE_RATE_HZ)))

# --- Camera-only sensor model (deputy side) ---------------------------
# Angular 1σ values; the camera tightens slightly near closest approach
# where SNR improves.  NO range channel exists in this configuration.
ANG_SIGMA_HI_TRUE = np.deg2rad(0.008)   # [rad]  ~0.008°  far end of arc
ANG_SIGMA_LO_TRUE = np.deg2rad(0.003)   # [rad]  ~0.003°  near closest app.
TAU_HOURS         = 2.5                  # noise-decay time constant [h]

# --- Chief GPS antenna noise (5 m raw → ≈1 m broadcast) ---------------
CHIEF_GPS_RAW_SIGMA    = 5.0        # [m] documentary only — raw GPS σ
CHIEF_GPS_SIGMA_TRUE   = 1.0        # [m] post-smoothing broadcast bias σ
CHIEF_GPS_SIGMA_FILTER = 0.3        # [m] filter-assumed white jitter

# --- Bias / dropout / outlier (mild) ---------------------------------
ANG_BIAS_STD     = np.deg2rad(0.001)
BIAS_TAU_HOURS   = 4.0
P_DROPOUT        = 0.02             # 2% epoch dropout
P_OUTLIER        = 0.005            # 0.5% per-channel outlier
OUTLIER_SCALE    = 5.0

# Filter pessimism factor on assumed sensor σ  (>1 ⇒ more conservative)
MEAS_SIGMA_INFLATE = 1.2

# Mahalanobis-distance gate for outlier rejection.
# For a 2-d.o.f. measurement, χ²(p=0.997, df=2) ≈ 11.83.
MAHALANOBIS_GATE = 11.83

# --- Range-bootstrap parameters (angles-only has no range channel) ----
# The camera cannot measure distance, so the EKF is started at a nominal
# stand-off range along the measured bearing, with a large along-LOS
# covariance.  The dynamics + burns then make range observable.
NOMINAL_RANGE_M       = 900.0       # [m] nominal deputy stand-off guess
RANGE_INIT_SIGMA_M    = 700.0       # [m] 1σ on the unknown initial range
CROSS_LOS_INIT_SIGMA  = 25.0        # [m] 1σ across-LOS (well-observed)
VEL_INIT_SIGMA        = 0.5         # [m/s] 1σ on the (unknown) velocity

# --- Physical velocity prior (the key bearings-only stabilizer) -------
# A bearing measurement is invariant to the deputy–chief distance, so a
# Cartesian EKF cannot tell "far + fast" from "near + slow": both fit the
# observed bearing-rate equally.  Left unconstrained, the range estimate
# runs away to tens of km.  But ARISE deputies are RHIG-station-kept in a
# bounded formation — total maneuvering Δv is ≈1 m/s per deputy and the
# Hill-frame relative SPEED stays well under ~1 m/s for the whole arc.
# Enforcing that physical bound on the velocity magnitude breaks the
# range/speed ambiguity and keeps the filter stable.
VEL_CLAMP_MAX  = 1.0    # [m/s] hard cap on |ρ̇| (physical formation bound)
LOS_VAR_FLOOR  = 50.0   # [m]  min 1σ kept along the LOS (range) direction

# Physical range band.  The radial velocity is ALSO unobservable from
# bearings, so over a long arc the range estimate would still creep.
# ARISE keeps every deputy inside a bounded imaging formation, so the
# deputy–chief range is physically confined to this band.  Clamping the
# range estimate to it makes the filter stable for arcs of any length.
RANGE_CLAMP_MIN = 120.0     # [m] physical formation inner bound
RANGE_CLAMP_MAX = 6000.0    # [m] physical formation outer bound

# --- FIMC: orbital-invariant constraint (the range-observability fix) -
# Woffinden & Geller (2007) proved angles-only relative navigation is
# unobservable under linear CW dynamics — exactly the range runaway seen
# without this term.  Observability is RESTORED by the nonlinear orbital
# structure.  We exploit it by constraining the deputy's two-body
# invariants (energy ε, angular momentum h, Laplace-Runge-Lenz vector e)
# to the chief's broadcast values.  Because ε, h, e depend on |ρ|, this
# pseudo-measurement makes range observable from a bearings-only filter.
#
#   * ENERGY is the powerful term: a bounded (non-drifting) formation has
#     matched semi-major axes ⇒ δε ≈ 0.  σ_ε is sized from a small
#     formation-keeping δa tolerance — this is what pins the range.
#   * h and e differences are NOT zero — they encode the relative orbit.
#     Their σ are sized to the genuine relative-orbit offset so the
#     constraint rejects gross range errors without forcing the deputy
#     onto the chief's exact orbit.
MU_EARTH      = 3.986004418e14   # [m³/s²] geocentric gravitational param.
# IMPORTANT — σ sizing measured from the actual ARISE Apophis bundles.
# Unlike a quiescent LEO station-kept formation (δa ≈ 0), these deputies
# are on an ACTIVE RENDEZVOUS: the measured deputy-vs-chief offsets are
#     |δε| ≈ 400–950 m²/s²  (⇔ δa ≈ 5–11 km),
#     |δh|/|h| ≈ 1.0–1.8e-4,  |δe_LRL| ≈ 1.0–2.5e-4.
# σ are sized to those REAL offsets so the constraint never biases the
# filter.  (For a true bounded formation δa→0 and σ_ε would be ~100×
# tighter — that is where FIMC delivers its large range gains.)
FIMC_DA       = 11000.0          # [m] sizes σ_ε from the real δa scale
FIMC_DH_FRAC  = 2.5e-4           # relative angular-momentum offset frac.
FIMC_DE       = 3.5e-4           # relative LRL (eccentricity) offset
FIMC_N_ITER   = 1                # iterated-update passes (IPLF spirit)
# The invariants vary slowly, so the (numerically-Jacobian'd) FIMC update
# is applied only every FIMC_STRIDE filter steps — same information,
# ~FIMC_STRIDE× cheaper.
FIMC_STRIDE   = 10


# ───────── helper: load a deputy bundle ─────────

def load_apophis_bundle(path: Path) -> Dict[str, np.ndarray]:
    wanted = (
        "t", "rD_N", "vD_N", "rC_N", "vC_N",
        "rA_N", "vA_N", "impulses",
    )
    out: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as npz:
        for k in wanted:
            if k in npz:
                out[k] = npz[k]
    if "impulses" not in out:
        out["impulses"] = np.zeros((0, 4), dtype=float)
    return out


# ───────── helper: chief truth fn from chief_reference.npz ─────────

def load_chief_reference(path: Path):
    """
    Load chief_reference.npz → (t, rC_N [m], vC_N [m/s], interpolator fn).
    The file stores positions in km and velocities in km/s.
    """
    with np.load(path, allow_pickle=True) as npz:
        t_arr = np.asarray(npz["times"], float).reshape(-1)
        r_km  = np.asarray(npz["positions"], float).reshape(-1, 3)
        v_kms = np.asarray(npz["velocities"], float).reshape(-1, 3)

    rC_N = r_km * 1000.0
    vC_N = v_kms * 1000.0

    def chief_state_fn(t: float) -> Tuple[np.ndarray, np.ndarray]:
        if t <= t_arr[0]:
            return rC_N[0].copy(), vC_N[0].copy()
        if t >= t_arr[-1]:
            return rC_N[-1].copy(), vC_N[-1].copy()
        j = int(np.searchsorted(t_arr, t)) - 1
        f = (t - t_arr[j]) / (t_arr[j + 1] - t_arr[j] + 1e-15)
        r = (1.0 - f) * rC_N[j] + f * rC_N[j + 1]
        v = (1.0 - f) * vC_N[j] + f * vC_N[j + 1]
        return r, v

    return t_arr, rC_N, vC_N, chief_state_fn


# ───────── burns on the time grid ─────────

def build_burn_sequence(
    impulses: np.ndarray,
    t_arr: np.ndarray,
    dt: float,
) -> np.ndarray:
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


# ───────── ECI → Hill truth conversion ─────────

def eci_to_hill_traj(
    rD_N: np.ndarray, vD_N: np.ndarray,
    rC_N: np.ndarray, vC_N: np.ndarray,
) -> np.ndarray:
    rD_N = np.asarray(rD_N, float)
    vD_N = np.asarray(vD_N, float)
    rC_N = np.asarray(rC_N, float)
    vC_N = np.asarray(vC_N, float)

    N = rD_N.shape[0]
    x_true = np.zeros((N, 6))
    for k in range(N):
        rC = rC_N[k]; vC = vC_N[k]
        rD = rD_N[k]; vD = vD_N[k]
        C_HN = dcm_lvlh(rC, vC)
        rho = C_HN @ (rD - rC)
        omega_HN = np.array([0.0, 0.0, hill_rate(rC, vC)])
        rhod = C_HN @ (vD - vC) - np.cross(omega_HN, rho)
        x_true[k, 0:3] = rho
        x_true[k, 3:6] = rhod
    return x_true


# ───────── LOS (azimuth / elevation) helpers ─────────

def los_from_rho(rho: np.ndarray) -> np.ndarray:
    """
    Convert a Hill-frame relative vector ρ = [R, T, N] into bearing
    (azimuth, elevation).  Range information is intentionally discarded.
    """
    x, y, z = rho
    r_xy = np.sqrt(max(x * x + y * y, 1e-30))
    az = np.arctan2(y, x)
    el = np.arctan2(z, r_xy)
    return np.array([az, el])


def los_jacobian(rho: np.ndarray) -> np.ndarray:
    """
    2x6 Jacobian of (az, el) w.r.t. the 6-state x_H = [ρ, ρ̇].
    Velocity columns are zero; range is unobservable from one sample
    (both rows are orthogonal to the LOS direction).
    """
    x, y, z = rho
    r2_xy = max(x * x + y * y, 1e-30)
    r_xy  = np.sqrt(r2_xy)
    r2    = max(r2_xy + z * z, 1e-30)

    daz_drho = np.array([-y / r2_xy, x / r2_xy, 0.0])
    dEl_drho = np.array([-x * z / (r2 * r_xy),
                         -y * z / (r2 * r_xy),
                          r_xy / r2])

    H = np.zeros((2, 6))
    H[0, 0:3] = daz_drho
    H[1, 0:3] = dEl_drho
    return H


def rho_from_los_range(az: float, el: float, rng: float) -> np.ndarray:
    """Build a Hill vector from a bearing and an assumed range."""
    ce = np.cos(el)
    return np.array([
        rng * ce * np.cos(az),
        rng * ce * np.sin(az),
        rng * np.sin(el),
    ])


def wrap_pi(a):
    """Wrap angle(s) to [-π, π]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ───────── measurement generator (bearings only) ─────────

def build_los_meas(
    rng: np.random.Generator,
    t_arr: np.ndarray,
    rD_N: np.ndarray,
    rC_N_true: np.ndarray,
    vC_N_true: np.ndarray,
    ang_sigma_hi: float = ANG_SIGMA_HI_TRUE,
    ang_sigma_lo: float = ANG_SIGMA_LO_TRUE,
    tau_hours:    float = TAU_HOURS,
    ang_bias_std: float = ANG_BIAS_STD,
    bias_tau_hours: float = BIAS_TAU_HOURS,
    p_dropout:    float = P_DROPOUT,
    p_outlier:    float = P_OUTLIER,
    outlier_scale: float = OUTLIER_SCALE,
    gps_sigma_true: float = CHIEF_GPS_SIGMA_TRUE,
):
    """
    Generate noisy (az, el) bearing measurements per epoch.

    Returns
    -------
    z_meas        : (N,2)   [rad, rad]
    sigma_assumed : (N,2)   EKF-assumed 1σ per channel
    valid         : (N,)    bool dropout mask
    rC_N_meas     : (N,3)   chief inertial position (GPS-jittered)
    vC_N_frame    : (N,3)   chief inertial velocity used by the EKF
    """
    N = len(t_arr)
    t_rel = t_arr - t_arr[0]
    tau   = tau_hours * 3600.0
    w     = np.exp(-t_rel / tau)
    sigma_ang_true = ang_sigma_lo + (ang_sigma_hi - ang_sigma_lo) * w

    dt_eff = float(np.median(np.diff(t_arr))) if N > 1 else 1.0
    bias_tau_sec = bias_tau_hours * 3600.0
    rho_b = np.exp(-dt_eff / max(bias_tau_sec, 1e-3))
    qa_b  = ang_bias_std * np.sqrt(1.0 - rho_b ** 2)
    b_az, b_el = 0.0, 0.0

    z_meas     = np.zeros((N, 2))
    valid      = np.ones(N, dtype=bool)
    rC_N_meas  = np.zeros((N, 3))
    vC_N_frame = np.zeros((N, 3))

    for k in range(N):
        if k > 0:
            b_az = rho_b * b_az + qa_b * rng.normal()
            b_el = rho_b * b_el + qa_b * rng.normal()

        rC = rC_N_true[k]; vC = vC_N_true[k]
        rD = rD_N[k]

        rC_meas = rC + gps_sigma_true * rng.normal(size=3)
        rC_N_meas[k]  = rC_meas
        vC_N_frame[k] = vC

        C_HN_true = dcm_lvlh(rC, vC)
        rho_true  = C_HN_true @ (rD - rC)
        z_true = los_from_rho(rho_true)               # [az, el]

        noise = np.array([
            sigma_ang_true[k] * rng.normal(),
            sigma_ang_true[k] * rng.normal(),
        ])
        for j in range(2):
            if rng.random() < p_outlier:
                noise[j] *= outlier_scale

        z_meas[k, 0] = wrap_pi(z_true[0] + b_az + noise[0])
        z_meas[k, 1] = wrap_pi(z_true[1] + b_el + noise[1])

        if k > 0 and rng.random() < p_dropout:
            valid[k] = False

    sigma_assumed = MEAS_SIGMA_INFLATE * np.column_stack(
        (sigma_ang_true, sigma_ang_true)
    )
    return z_meas, sigma_assumed, valid, rC_N_meas, vC_N_frame


# ───────── measurement model: predict & Jacobian ─────────

def los_pred_and_jacobian(x_H: np.ndarray):
    """
    Predict (az, el) from the deputy Hill state and return the 2x6
    Jacobian.  Bearing depends only on the direction of ρ.
    """
    rho = np.asarray(x_H, float).reshape(6)[0:3]
    z_hat = los_from_rho(rho)
    H = los_jacobian(rho)
    return z_hat, H


# ───────── EKF update for bearings-only ─────────

def ekf_update_los(
    ekf: HillEKF,
    z_vec: np.ndarray,
    sigma_vec: np.ndarray,
    gps_sigma_filter: float = CHIEF_GPS_SIGMA_FILTER,
    mahalanobis_gate: float = MAHALANOBIS_GATE,
) -> bool:
    """
    EKF update with a 2-channel [az, el] measurement.

    R = R_sens + R_gps, with R_gps = σ_gps² (Hp · Hpᵀ) capturing the
    chief-GPS jitter mapped through the bearing Jacobian.

    A 2-d.o.f. Mahalanobis gate rejects gross outliers.

    Returns
    -------
    accepted : bool
    """
    z_hat, H = los_pred_and_jacobian(ekf.x_pred)

    z = np.asarray(z_vec, float).reshape(2)
    sigma_vec = np.asarray(sigma_vec, float).reshape(2)

    y = np.array([
        wrap_pi(z[0] - z_hat[0]),
        wrap_pi(z[1] - z_hat[1]),
    ])

    R_sens = np.diag(sigma_vec ** 2)
    Hp = H[:, 0:3]
    R_gps = (gps_sigma_filter ** 2) * (Hp @ Hp.T)
    R_eff = R_sens + R_gps

    S = H @ ekf.P_pred @ H.T + R_eff
    S_inv = la.inv(S)

    d2 = float(y @ S_inv @ y)
    if d2 > mahalanobis_gate:
        ekf.x = ekf.x_pred.copy()
        ekf.P = ekf.P_pred.copy()
        ekf.innov = y.copy()
        ekf.S = S.copy()
        ekf.y_pre = y.copy()
        ekf.y_post = y.copy()
        ekf.H_last = H.copy()
        return False

    K = ekf.P_pred @ H.T @ S_inv
    x_new = ekf.x_pred + (K @ y).reshape(-1)

    # ── confine the POSITION correction to the cross-LOS plane ────────
    # A bearing measurement carries no range information, so the filter
    # must not move the state along the line of sight.  Spurious
    # along-LOS↔cross-LOS covariance cross-terms otherwise let the
    # bearing update jerk the range, and the estimate runs away to tens
    # of km.  We project out the along-LOS component of the position
    # update; the range coordinate then evolves ONLY through the HCW
    # dynamics in the predict step (bounded by the clamped relative
    # velocity).  Velocity stays fully updated — bearings do constrain
    # the cross-LOS speed.
    u_los = ekf.x_pred[0:3]
    nrm = float(la.norm(u_los))
    if nrm > 1e-6:
        u_los = u_los / nrm
        dpos = x_new[0:3] - ekf.x_pred[0:3]
        dpos = dpos - (u_los @ dpos) * u_los
        x_new[0:3] = ekf.x_pred[0:3] + dpos

    I = np.eye(ekf.x.size)
    if ekf.cfg.use_joseph:
        KH = K @ H
        ekf.P = (I - KH) @ ekf.P_pred @ (I - KH).T + K @ R_eff @ K.T
    else:
        ekf.P = (I - K @ H) @ ekf.P_pred

    ekf.x = x_new

    ekf.innov = y.copy()
    ekf.S = S.copy()
    ekf.y_pre = y.copy()

    z_hat_post, _ = los_pred_and_jacobian(ekf.x)
    ekf.y_post = np.array([
        wrap_pi(z[0] - z_hat_post[0]),
        wrap_pi(z[1] - z_hat_post[1]),
    ])
    ekf.H_last = H.copy()
    return True


# ───────── bearings-only stabilizer ─────────

def apply_los_stabilizer(
    ekf: HillEKF,
    vel_clamp: float = VEL_CLAMP_MAX,
    los_var_floor: float = LOS_VAR_FLOOR,
):
    """
    Keep a Cartesian bearings-only EKF physically consistent.

    1. Velocity magnitude clamp.  A bearing is scale-invariant, so the
       filter cannot separate range from speed.  ARISE deputies are
       RHIG-station-kept, so |ρ̇| is physically bounded — enforcing that
       bound is what stops the range estimate running away.

    2. Along-LOS covariance floor.  Range is only weakly observable
       (through the dynamics + burns), so the filter must not be allowed
       to claim false range precision.  We keep at least `los_var_floor`
       of 1σ uncertainty along the current line of sight, which keeps
       the filter consistent (NIS ≈ 1) and the gate honest.
    """
    # 1) physical velocity-magnitude prior
    v = ekf.x[3:6]
    vmag = float(la.norm(v))
    if vmag > vel_clamp:
        ekf.x[3:6] = v * (vel_clamp / vmag)

    # 2) physical range band.  Range and radial velocity are both
    #    unobservable from bearings; on a long arc the range estimate
    #    would creep out of any physical bound.  ARISE keeps the deputy
    #    inside a bounded imaging formation, so we clamp |ρ| to the
    #    physical band (rescaling along the well-observed LOS direction).
    rho = ekf.x[0:3]
    rmag = float(la.norm(rho))
    if rmag > 1e-6:
        r_clamped = min(max(rmag, RANGE_CLAMP_MIN), RANGE_CLAMP_MAX)
        if r_clamped != rmag:
            ekf.x[0:3] = rho * (r_clamped / rmag)
            rmag = r_clamped

    # 3) floor the position covariance along the current LOS direction
    if rmag > 1e-6:
        u = ekf.x[0:3] / rmag
        Ppos = ekf.P[0:3, 0:3]
        var_los = float(u @ Ppos @ u)
        if var_los < los_var_floor ** 2:
            ekf.P[0:3, 0:3] = Ppos + (los_var_floor ** 2 - var_los) * np.outer(u, u)


# ───────── FIMC: orbital-invariant range-observability constraint ─────

def hill_to_inertial(x_H: np.ndarray, rC: np.ndarray, vC: np.ndarray):
    """
    Convert a deputy Hill state x_H = [ρ, ρ̇] back to its inertial
    (rD, vD), given the chief inertial state.  Inverse of the mapping in
    eci_to_hill_traj().
    """
    rho  = np.asarray(x_H, float).reshape(6)[0:3]
    rhod = np.asarray(x_H, float).reshape(6)[3:6]
    C_HN = dcm_lvlh(rC, vC)             # Hill-from-ECI
    C_NH = C_HN.T
    omega = np.array([0.0, 0.0, hill_rate(rC, vC)])
    rD = rC + C_NH @ rho
    vD = vC + C_NH @ (rhod + np.cross(omega, rho))
    return rD, vD


def orbital_invariants(r: np.ndarray, v: np.ndarray,
                       mu: float = MU_EARTH) -> np.ndarray:
    """
    The two-body integrals of motion as a 7-vector [ε, h(3), e(3)]:

        ε = |v|²/2 − μ/|r|              specific orbital energy
        h = r × v                       specific angular momentum
        e = (v×h)/μ − r/|r|             Laplace-Runge-Lenz eccentricity
    """
    r = np.asarray(r, float).reshape(3)
    v = np.asarray(v, float).reshape(3)
    rn = float(la.norm(r))
    eps = 0.5 * float(v @ v) - mu / rn
    h = np.cross(r, v)
    e = np.cross(v, h) / mu - r / rn
    return np.concatenate(([eps], h, e))


def fimc_residual_and_jacobian(
    x_H: np.ndarray, rC: np.ndarray, vC: np.ndarray, mu: float = MU_EARTH,
):
    """
    FIMC residual  z = invariants(deputy) − invariants(chief)  and its
    7×6 Jacobian w.r.t. the deputy Hill state (central finite difference).

    For a bounded co-orbital formation the deputy and chief share the
    same two-body invariants up to the small offset induced by the
    relative motion, so z is small.  Crucially z depends on |ρ| — that
    dependence is what makes range observable.
    """
    inv_C = orbital_invariants(rC, vC, mu)

    def inv_dep(xh):
        rD, vD = hill_to_inertial(xh, rC, vC)
        return orbital_invariants(rD, vD, mu)

    x_H = np.asarray(x_H, float).reshape(6)
    z0 = inv_dep(x_H) - inv_C

    H = np.zeros((7, 6))
    step = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3])
    for j in range(6):
        xp = x_H.copy(); xp[j] += step[j]
        xm = x_H.copy(); xm[j] -= step[j]
        H[:, j] = (inv_dep(xp) - inv_dep(xm)) / (2.0 * step[j])
    return z0, H


def fimc_R(rC: np.ndarray, vC: np.ndarray, mu: float = MU_EARTH):
    """
    Build the 7×7 measurement covariance for the FIMC pseudo-measurement.

    σ_ε  — from the formation-keeping semi-major-axis tolerance FIMC_DA:
           a bounded formation has δa ≈ 0, and δε = μ·δa /(2a²).  This
           is the tight term that pins the range.
    σ_h  — a fraction of |h_chief| covering the genuine relative-orbit
           angular-momentum offset.
    σ_e  — an absolute LRL offset covering the relative eccentricity.
    """
    inv_C = orbital_invariants(rC, vC, mu)
    a = -mu / (2.0 * inv_C[0])                     # chief semi-major axis
    sig_eps = mu * FIMC_DA / (2.0 * a * a)
    sig_h   = FIMC_DH_FRAC * max(float(la.norm(inv_C[1:4])), 1.0)
    sig_e   = FIMC_DE
    return np.diag([sig_eps ** 2,
                    sig_h ** 2, sig_h ** 2, sig_h ** 2,
                    sig_e ** 2, sig_e ** 2, sig_e ** 2])


def ekf_update_fimc(
    ekf: HillEKF,
    rC: np.ndarray,
    vC: np.ndarray,
    n_iter: int = FIMC_N_ITER,
    mu: float = MU_EARTH,
):
    """
    Iterated EKF update against the FIMC orbital-invariant pseudo-
    measurement.  Applied AFTER the bearing update each step: the bearing
    pins the cross-LOS direction, then FIMC pins the range by requiring
    the deputy's two-body invariants to match the chief's broadcast.

    The 'measurement' value is 0 (invariants should match); R = fimc_R()
    is sized for the genuine relative-orbit offset.  The update is
    iterated (re-linearised about the posterior — the IPLF idea) because
    the invariant map is strongly non-linear in the range coordinate.
    """
    x_prior = ekf.x.copy()
    P_prior = ekf.P.copy()
    R = fimc_R(rC, vC, mu)
    I = np.eye(6)

    x_lin = x_prior.copy()
    for _ in range(max(1, n_iter)):
        z0, H = fimc_residual_and_jacobian(x_lin, rC, vC, mu)
        # iterated-EKF innovation: target residual is 0
        y = -z0 - H @ (x_prior - x_lin)
        S = H @ P_prior @ H.T + R
        K = P_prior @ H.T @ la.inv(S)
        x_new = x_prior + K @ y
        if la.norm(x_new - x_lin) < 1e-2:
            x_lin = x_new
            break
        x_lin = x_new

    # covariance update (Joseph form) at the converged linearisation
    _, H = fimc_residual_and_jacobian(x_lin, rC, vC, mu)
    S = H @ P_prior @ H.T + R
    K = P_prior @ H.T @ la.inv(S)
    KH = K @ H
    ekf.P = (I - KH) @ P_prior @ (I - KH).T + K @ R @ K.T
    ekf.x = x_lin


# ───────── single-deputy CKF + EKF helper ─────────

def run_ckf_ekf_for_deputy(
    dep_idx: int,
    t_arr: np.ndarray,
    x_true: np.ndarray,
    rD_sel: np.ndarray,
    rC_sel: np.ndarray,
    vC_sel: np.ndarray,
    dv_seq_H: np.ndarray,
    chief_state_fn,
    N_ckf: int,
):
    """
    CKF + EKF for one deputy using bearings-only (az, el) measurements.

    Returns
    -------
    x_est_hist : (N,6)
    P_hist     : (N,6,6)
    y_pre_hist : (N,2) pre-fit innovations [rad, rad]
    y_post_hist: (N,2)
    sigma_final: (N,2) assumed 1σ per channel
    """
    dt = float(np.median(np.diff(t_arr)))

    cfg = HillEKFConfig(
        dt=dt,
        sigma_proc_pos=0.05,    # [m]    per step
        sigma_proc_vel=2e-4,    # [m/s]  per step — looser than range+az+el
        sigma_meas_pos=2.0,     # placeholder; LOS update builds its own R
        sigma_meas_vel=0.01,
        sigma_init_pos=RANGE_INIT_SIGMA_M,   # large — range unknown
        sigma_init_vel=VEL_INIT_SIGMA,
        sigma_dv=0.01,
        use_joseph=True,
    )
    ekf = HillEKF(cfg, chief_state_fn=chief_state_fn)

    # ──────────────────────────────────────────────────────────────
    # Bootstrap.  A camera cannot measure range, so:
    #   * bearing  → the LOS direction is taken straight from z₀;
    #   * range    → set to NOMINAL_RANGE_M (a nominal stand-off guess);
    #   * P₀       → built ANISOTROPICALLY: tight (CROSS_LOS_INIT_SIGMA)
    #                across the LOS where bearings constrain the state,
    #                very loose (RANGE_INIT_SIGMA_M) along the LOS where
    #                only the dynamics + burns will resolve scale.
    # ──────────────────────────────────────────────────────────────
    rng_boot = np.random.default_rng(3000 + dep_idx)
    z_boot, _, _, _, _ = build_los_meas(
        rng_boot, t_arr[:2], rD_sel[:2], rC_sel[:2], vC_sel[:2],
    )
    az0, el0 = float(z_boot[0, 0]), float(z_boot[0, 1])
    u_los = rho_from_los_range(az0, el0, 1.0)          # unit LOS direction
    rho0_hat = NOMINAL_RANGE_M * u_los
    x0_guess = np.concatenate([rho0_hat, np.zeros(3)])  # velocity unknown

    # Anisotropic P₀: variance RANGE_INIT_SIGMA² along u_los, plus an
    # isotropic CROSS_LOS_INIT_SIGMA² floor on every position axis.
    P_pos = (CROSS_LOS_INIT_SIGMA ** 2) * np.eye(3)
    P_pos += (RANGE_INIT_SIGMA_M ** 2) * np.outer(u_los, u_los)
    p_vel0 = cfg.sigma_init_vel ** 2
    P0_big = np.zeros((6, 6))
    P0_big[0:3, 0:3] = P_pos
    P0_big[3:6, 3:6] = p_vel0 * np.eye(3)

    print(
        f"[dep{dep_idx}] BOOT-INIT (LOS-only) ||x0_guess - x_true(0)|| = "
        f"{la.norm(x0_guess - x_true[0]):.3f}  "
        f"(range guess {NOMINAL_RANGE_M:.0f} m vs true "
        f"{la.norm(x_true[0, 0:3]):.0f} m), Tr(P0) = {np.trace(P0_big):.3e}",
    )

    # Optional CKF refinement passes (start-state carried forward).
    if N_ckf > 0:
        print(f"[dep{dep_idx}] CKF {N_ckf} passes...")

    for it in range(N_ckf):
        ekf.reset(x0_guess, P0_big)

        rng_pass = np.random.default_rng(1000 + dep_idx * 10 + it)
        z_pass, sigma_pass, valid_pass, _, _ = build_los_meas(
            rng_pass, t_arr, rD_sel, rC_sel, vC_sel,
        )

        ekf.x_pred = ekf.x.copy()
        ekf.P_pred = ekf.P.copy()
        ekf.t_last = t_arr[0]
        ekf_update_los(ekf, z_pass[0], sigma_pass[0])
        ekf_update_fimc(ekf, rC_sel[0], vC_sel[0])
        apply_los_stabilizer(ekf)
        x_at_start = ekf.x.copy()

        for k in range(1, len(t_arr)):
            ekf.predict(t_arr[k], burn_dv_H=dv_seq_H[k])
            if valid_pass[k]:
                ekf_update_los(ekf, z_pass[k], sigma_pass[k])
                if k % FIMC_STRIDE == 0:
                    ekf_update_fimc(ekf, rC_sel[k], vC_sel[k])
                apply_los_stabilizer(ekf)
            else:
                ekf.x = ekf.x_pred.copy()
                ekf.P = ekf.P_pred.copy()
                ekf.y_pre = np.zeros(2)
                ekf.y_post = np.zeros(2)

        err_end = la.norm(ekf.x - x_true[-1])
        print(
            f"    pass {it + 1}: ||x_end - x_true_end|| = {err_end:.3f}, "
            f"Tr(P_end) = {np.trace(ekf.P):.3e}",
        )
        x0_guess = x_at_start

    # Final EKF pass with fresh measurements
    print(f"[dep{dep_idx}] EKF final pass...")
    ekf.reset(x0_guess, P0_big)

    rng_final = np.random.default_rng(2000 + dep_idx)
    z_meas, sigma_final, valid_final, _, _ = build_los_meas(
        rng_final, t_arr, rD_sel, rC_sel, vC_sel,
    )

    N = len(t_arr)
    x_est_hist  = np.zeros((N, 6))
    P_hist      = np.zeros((N, 6, 6))
    y_pre_hist  = np.full((N, 2), np.nan)
    y_post_hist = np.full((N, 2), np.nan)

    ekf.x_pred = ekf.x.copy()
    ekf.P_pred = ekf.P.copy()
    ekf.t_last = t_arr[0]

    n_reject = 0
    if not ekf_update_los(ekf, z_meas[0], sigma_final[0]):
        n_reject += 1
    ekf_update_fimc(ekf, rC_sel[0], vC_sel[0])
    apply_los_stabilizer(ekf)
    x_est_hist[0] = ekf.x
    P_hist[0]     = ekf.P
    y_pre_hist[0] = ekf.y_pre
    y_post_hist[0] = ekf.y_post

    for k in range(1, N):
        ekf.predict(t_arr[k], burn_dv_H=dv_seq_H[k])
        if valid_final[k]:
            if not ekf_update_los(ekf, z_meas[k], sigma_final[k]):
                n_reject += 1
            if k % FIMC_STRIDE == 0:
                ekf_update_fimc(ekf, rC_sel[k], vC_sel[k])
            apply_los_stabilizer(ekf)
            y_pre_hist[k] = ekf.y_pre
            y_post_hist[k] = ekf.y_post
        else:
            ekf.x = ekf.x_pred.copy()
            ekf.P = ekf.P_pred.copy()

        x_est_hist[k] = ekf.x
        P_hist[k]     = ekf.P

    print(
        f"[dep{dep_idx}] outlier gate: rejected {n_reject}/{N} measurements "
        f"({100.0 * n_reject / max(N, 1):.2f}%)"
    )

    return x_est_hist, P_hist, y_pre_hist, y_post_hist, sigma_final


# ───────── main swarm demo ─────────

def apophis_swarm_ckf_ekf_los_demo(
    data_dir: Path,
    N_dep: int = N_DEP,
    filter_dt: float = FILTER_DT,
    N_ckf: int = N_CKF,
    save_stride: int = SAVE_STRIDE,
):
    data_dir = Path(data_dir)

    chief_ref_path = data_dir / "chief_reference.npz"
    if not chief_ref_path.is_file():
        raise FileNotFoundError(f"Missing chief reference: {chief_ref_path}")

    t_ref, rC_ref, vC_ref, chief_state_fn = load_chief_reference(chief_ref_path)
    print(
        f"[chief-ref] {chief_ref_path.name}: "
        f"{len(t_ref)} samples, t∈[{t_ref[0]:.1f}, {t_ref[-1]:.1f}] s",
    )

    x_true_swarm = []
    x_est_swarm  = []
    P_swarm      = []
    y_pre_swarm  = []
    y_post_swarm = []
    sigma_swarm  = []
    t_swarm      = []

    for d in range(N_dep):
        bundle_path = data_dir / f"apophis_rendezvous_full_D{d}.npz"
        if not bundle_path.is_file():
            raise FileNotFoundError(f"Missing bundle: {bundle_path}")

        print(f"\n===== Deputy {d} – loading {bundle_path.name} =====")
        data = load_apophis_bundle(bundle_path)

        t_full  = np.asarray(data["t"], float).reshape(-1)
        rD_full = np.asarray(data["rD_N"], float).reshape(-1, 3)
        vD_full = np.asarray(data["vD_N"], float).reshape(-1, 3)
        impulses = np.asarray(data["impulses"], float).reshape(-1, 4)

        # Upsample the 1 Hz truth bundle to the filter cadence.
        t_crop_start = t_full[0] + 0.5 * 3600.0
        t_arr = np.arange(t_crop_start, t_full[-1] - 1e-9, filter_dt)

        def _interp3(arr_native):
            return np.column_stack([
                np.interp(t_arr, t_full, arr_native[:, i]) for i in range(3)
            ])

        rD_sel = _interp3(rD_full)
        vD_sel = _interp3(vD_full)
        rC_sel = np.zeros((len(t_arr), 3))
        vC_sel = np.zeros((len(t_arr), 3))
        for k, t in enumerate(t_arr):
            rC_sel[k], vC_sel[k] = chief_state_fn(float(t))

        dv_seq_H = build_burn_sequence(impulses, t_arr, filter_dt)
        x_true = eci_to_hill_traj(rD_sel, vD_sel, rC_sel, vC_sel)

        print(
            f"[dep{d}] grid: {len(t_arr)} steps @ {1.0/filter_dt:.1f} Hz, "
            f"arc = {(t_arr[-1] - t_arr[0])/3600:.2f} h"
        )

        (x_est_d, P_d,
         y_pre_d, y_post_d,
         sigma_final) = run_ckf_ekf_for_deputy(
            dep_idx=d,
            t_arr=t_arr,
            x_true=x_true,
            rD_sel=rD_sel,
            rC_sel=rC_sel,
            vC_sel=vC_sel,
            dv_seq_H=dv_seq_H,
            chief_state_fn=chief_state_fn,
            N_ckf=N_ckf,
        )

        s = save_stride
        fdi_path = data_dir / f"apophis_swarm_FDI_losonly_D{d}.npz"
        np.savez_compressed(
            fdi_path,
            t=t_arr[::s],
            x_true=x_true[::s],
            x_est=x_est_d[::s],
            P=P_d[::s],
            y_pre=y_pre_d[::s],
            y_post=y_post_d[::s],
            sigma_meas=sigma_final[::s],
            dv_seq_H=dv_seq_H[::s],
        )
        print(f"[dep{d}] Saved FDI bundle -> {fdi_path.name}")

        x_true_swarm.append(x_true)
        x_est_swarm.append(x_est_d)
        P_swarm.append(P_d)
        y_pre_swarm.append(y_pre_d)
        y_post_swarm.append(y_post_d)
        sigma_swarm.append(sigma_final)
        t_swarm.append(t_arr)

    # ─── diagnostics / plots ───
    labels = ["R", "T", "N"]

    for d in range(N_dep):
        print(f"\n=== Deputy {d} EKF results (LOS-only / camera-only) ===")

        t_arr = t_swarm[d]
        t_hr = (t_arr - t_arr[0]) / 3600.0

        x_true_d = x_true_swarm[d]
        x_est_d  = x_est_swarm[d]
        err_d    = x_est_d - x_true_d

        mask = t_hr >= 0.1
        t_hr_plot = t_hr[mask]
        err_plot = err_d[mask]
        P_diag_plot = np.diagonal(P_swarm[d][mask], axis1=1, axis2=2)
        three_sigma = 3.0 * np.sqrt(np.maximum(P_diag_plot, 0.0))

        # Decompose the position error into along-LOS and cross-LOS parts.
        rho_true = x_true_d[mask, 0:3]
        u = rho_true / np.maximum(
            np.linalg.norm(rho_true, axis=1, keepdims=True), 1e-9)
        err_pos = err_plot[:, 0:3]
        along = np.sum(err_pos * u, axis=1)               # signed range err
        cross = np.linalg.norm(err_pos - along[:, None] * u, axis=1)

        for i, lbl in enumerate(labels):
            e = err_plot[:, i]
            print(
                f"[POS] {lbl}: mean={np.mean(e):+.3f} m, "
                f"RMS={np.sqrt(np.mean(e**2)):.3f} m",
            )
        print(
            f"[LOS] along-LOS (range) RMS = {np.sqrt(np.mean(along**2)):.3f} m"
        )
        print(
            f"[LOS] cross-LOS (bearing) RMS = "
            f"{np.sqrt(np.mean(cross**2)):.3f} m"
        )

        # 1) Position error vs 3σ
        fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            ax = axes1[i]
            ax.scatter(t_hr_plot, err_plot[:, i], s=6, alpha=0.4,
                       label=f"dep{d} error ρ_{labels[i]}")
            ax.plot(t_hr_plot,  three_sigma[:, i], "r--", linewidth=1.0,
                    label="+3σ" if i == 0 else None)
            ax.plot(t_hr_plot, -three_sigma[:, i], "r--", linewidth=1.0,
                    label="-3σ" if i == 0 else None)
            ax.set_ylabel(f"{labels[i]} [m]", fontsize=AXIS_LABEL_FONTSIZE)
            ax.grid(True)
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
        axes1[-1].set_xlabel("Time since deploy [h]",
                             fontsize=AXIS_LABEL_FONTSIZE)
        axes1[0].legend(fontsize=LEGEND_FONTSIZE)
        fig1.suptitle(f"Position Error vs 3σ – Deputy {d} (LOS-only)",
                      fontsize=TITLE_FONTSIZE)
        fig1.tight_layout()

        # 2) Velocity error vs 3σ
        fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            ax = axes2[i]
            ax.scatter(t_hr_plot, err_plot[:, 3 + i], s=6, alpha=0.4,
                       label=f"dep{d} error ρ̇_{labels[i]}")
            ax.plot(t_hr_plot,  three_sigma[:, 3 + i], "r--", linewidth=1.0,
                    label="+3σ" if i == 0 else None)
            ax.plot(t_hr_plot, -three_sigma[:, 3 + i], "r--", linewidth=1.0,
                    label="-3σ" if i == 0 else None)
            ax.set_ylabel(f"{labels[i]} [m/s]", fontsize=AXIS_LABEL_FONTSIZE)
            ax.grid(True)
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
        axes2[-1].set_xlabel("Time since deploy [h]",
                             fontsize=AXIS_LABEL_FONTSIZE)
        axes2[0].legend(fontsize=LEGEND_FONTSIZE)
        fig2.suptitle(f"Velocity Error vs 3σ – Deputy {d} (LOS-only)",
                      fontsize=TITLE_FONTSIZE)
        fig2.tight_layout()

        # 3) Bearing residuals (measurement space)
        y_pre_d  = y_pre_swarm[d][mask]
        y_post_d = y_post_swarm[d][mask]
        sigma_d  = sigma_swarm[d][mask]
        three_sigma_meas = 3.0 * sigma_d
        meas_labels = ["az [rad]", "el [rad]"]

        fig3, axes3 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for i in range(2):
            ax = axes3[i]
            ax.scatter(t_hr_plot, y_pre_d[:, i],  s=4, alpha=0.4,
                       label=f"pre-fit {meas_labels[i]}")
            ax.scatter(t_hr_plot, y_post_d[:, i], s=4, alpha=0.4,
                       label=f"post-fit {meas_labels[i]}")
            ax.plot(t_hr_plot,  three_sigma_meas[:, i], "r--",
                    linewidth=1.0, label="+3sigma" if i == 0 else None)
            ax.plot(t_hr_plot, -three_sigma_meas[:, i], "r--",
                    linewidth=1.0, label="-3sigma" if i == 0 else None)
            ax.set_ylabel(meas_labels[i], fontsize=AXIS_LABEL_FONTSIZE)
            ax.grid(True)
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
        axes3[-1].set_xlabel("Time since deploy [h]",
                             fontsize=AXIS_LABEL_FONTSIZE)
        axes3[0].legend(fontsize=LEGEND_FONTSIZE)
        fig3.suptitle(f"Bearing Residuals - Deputy {d} (LOS-only)",
                      fontsize=TITLE_FONTSIZE)
        fig3.tight_layout()

    # swarm-wide overview plots
    fig_pos, axes_pos = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig_vel, axes_vel = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for d in range(N_dep):
        t_arr = t_swarm[d]
        t_hr = (t_arr - t_arr[0]) / 3600.0
        mask = t_hr >= 0.1
        err_d = x_est_swarm[d] - x_true_swarm[d]
        for i in range(3):
            axes_pos[i].plot(t_hr[mask], err_d[mask, i], lw=0.9,
                             alpha=0.8, label=f"D{d}")
            axes_vel[i].plot(t_hr[mask], err_d[mask, 3 + i], lw=0.9,
                             alpha=0.8, label=f"D{d}")
    for i, lbl in enumerate(labels):
        axes_pos[i].set_ylabel(f"rho_{lbl} err [m]",
                               fontsize=AXIS_LABEL_FONTSIZE)
        axes_pos[i].grid(True)
        axes_pos[i].tick_params(labelsize=TICK_LABEL_FONTSIZE)
        axes_vel[i].set_ylabel(f"rhodot_{lbl} err [m/s]",
                               fontsize=AXIS_LABEL_FONTSIZE)
        axes_vel[i].grid(True)
        axes_vel[i].tick_params(labelsize=TICK_LABEL_FONTSIZE)
    axes_pos[-1].set_xlabel("Time since deploy [h]",
                            fontsize=AXIS_LABEL_FONTSIZE)
    axes_vel[-1].set_xlabel("Time since deploy [h]",
                            fontsize=AXIS_LABEL_FONTSIZE)
    axes_pos[0].legend(fontsize=LEGEND_FONTSIZE, ncol=4, loc="upper right")
    axes_vel[0].legend(fontsize=LEGEND_FONTSIZE, ncol=4, loc="upper right")
    fig_pos.suptitle("Swarm position error - LOS-only + FIMC",
                     fontsize=TITLE_FONTSIZE)
    fig_vel.suptitle("Swarm velocity error - LOS-only + FIMC",
                     fontsize=TITLE_FONTSIZE)
    fig_pos.tight_layout()
    fig_vel.tight_layout()

    # along-LOS vs cross-LOS RMS bar chart
    fig_rms, ax_rms = plt.subplots(figsize=(12, 6))
    along_tab = np.zeros(N_dep)
    cross_tab = np.zeros(N_dep)
    for d in range(N_dep):
        t_arr = t_swarm[d]
        t_hr  = (t_arr - t_arr[0]) / 3600.0
        mask  = t_hr >= 0.1
        err_pos = (x_est_swarm[d][mask] - x_true_swarm[d][mask])[:, 0:3]
        rho_true = x_true_swarm[d][mask, 0:3]
        u = rho_true / np.maximum(
            np.linalg.norm(rho_true, axis=1, keepdims=True), 1e-9)
        along = np.sum(err_pos * u, axis=1)
        cross = np.linalg.norm(err_pos - along[:, None] * u, axis=1)
        along_tab[d] = np.sqrt(np.mean(along ** 2))
        cross_tab[d] = np.sqrt(np.mean(cross ** 2))
    xpos = np.arange(N_dep)
    width = 0.38
    ax_rms.bar(xpos - width / 2, along_tab, width, label="along-LOS (range)")
    ax_rms.bar(xpos + width / 2, cross_tab, width, label="cross-LOS (bearing)")
    ax_rms.set_xticks(xpos)
    ax_rms.set_xticklabels([f"D{d}" for d in range(N_dep)],
                           fontsize=TICK_LABEL_FONTSIZE)
    ax_rms.set_ylabel("Position RMS [m]", fontsize=AXIS_LABEL_FONTSIZE)
    ax_rms.set_title("LOS-only + FIMC: along-LOS vs cross-LOS RMS",
                     fontsize=TITLE_FONTSIZE)
    ax_rms.grid(True, axis="y", alpha=0.4)
    ax_rms.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax_rms.legend(fontsize=LEGEND_FONTSIZE)
    fig_rms.tight_layout()

    # swarm 3-D Hill trajectory (truth vs EKF)
    fig_xyz = plt.figure(figsize=(11, 9))
    ax_xyz = fig_xyz.add_subplot(111, projection="3d")
    for d in range(N_dep):
        x_t = x_true_swarm[d]
        x_e = x_est_swarm[d]
        ax_xyz.plot(x_t[:, 0], x_t[:, 1], x_t[:, 2],
                    lw=1.2, alpha=0.9, label=f"D{d} truth")
        ax_xyz.plot(x_e[:, 0], x_e[:, 1], x_e[:, 2],
                    lw=0.8, ls="--", alpha=0.7)
    ax_xyz.set_xlabel("rho_R [m]", fontsize=AXIS_LABEL_FONTSIZE)
    ax_xyz.set_ylabel("rho_T [m]", fontsize=AXIS_LABEL_FONTSIZE)
    ax_xyz.set_zlabel("rho_N [m]", fontsize=AXIS_LABEL_FONTSIZE)
    ax_xyz.set_title("Hill-frame trajectories - LOS-only + FIMC (truth solid, EKF dashed)",
                     fontsize=TITLE_FONTSIZE)
    ax_xyz.legend(fontsize=LEGEND_FONTSIZE * 0.7, ncol=2, loc="upper left")
    fig_xyz.tight_layout()

    # NIS consistency check
    fig_nis, ax_nis = plt.subplots(figsize=(12, 6))
    for d in range(N_dep):
        t_arr = t_swarm[d]
        t_hr = (t_arr - t_arr[0]) / 3600.0
        sigma_d = sigma_swarm[d]
        y_pre_d = y_pre_swarm[d]
        good = np.isfinite(y_pre_d).all(axis=1) & (t_hr >= 0.1)
        yk = y_pre_d[good]
        sk = sigma_d[good]
        nis = np.sum((yk / sk) ** 2, axis=1) / 2.0
        ax_nis.plot(t_hr[good], nis, lw=0.8, alpha=0.7, label=f"D{d}")
    ax_nis.axhline(1.0, color="k", lw=0.8, ls="--", label="unity (target)")
    ax_nis.set_yscale("log")
    ax_nis.set_xlabel("Time since deploy [h]", fontsize=AXIS_LABEL_FONTSIZE)
    ax_nis.set_ylabel("mean NIS per channel", fontsize=AXIS_LABEL_FONTSIZE)
    ax_nis.set_title("Normalized innovation squared - LOS-only + FIMC",
                     fontsize=TITLE_FONTSIZE)
    ax_nis.grid(True, which="both", alpha=0.4)
    ax_nis.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax_nis.legend(fontsize=LEGEND_FONTSIZE * 0.7, ncol=4)
    fig_nis.tight_layout()

    plt.show()


# entry point

if __name__ == "__main__":
    dep0_path = SCRIPT_DIR / "apophis_rendezvous_full_D0.npz"
    chief_ref = SCRIPT_DIR / "chief_reference.npz"
    if not dep0_path.is_file():
        raise FileNotFoundError(
            f"Expected Apophis bundle not found: {dep0_path}"
        )
    if not chief_ref.is_file():
        raise FileNotFoundError(
            f"Expected chief reference not found: {chief_ref}"
        )

    apophis_swarm_ckf_ekf_los_demo(SCRIPT_DIR)
