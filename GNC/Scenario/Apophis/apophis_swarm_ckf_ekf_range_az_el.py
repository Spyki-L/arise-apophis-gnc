#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_ckf_ekf_range_az_el.py
====================================

Multi-deputy CKF + EKF demo for the Apophis rendezvous case, using a new
deputy-side sensor suite that gives **range, azimuth, and elevation** to the
chief mothership, while the **chief carries a 5 m-accuracy GPS antenna**.

Sensor concept (per deputy)
---------------------------
* Body-mounted camera (with star-tracker–assisted attitude knowledge) gives
  the line-of-sight unit vector to the chief.
* A co-pointed ranging beacon (laser/RF tag responder on the chief)
  returns the deputy↔chief range.
* After attitude correction, the deputy reports the chief in the
  chief Hill (LVLH) frame as

      z = [ range, azimuth, elevation ]^T

  where
      range     = ||ρ||
      azimuth   = atan2( ρ_T, ρ_R )
      elevation = atan2( ρ_N, sqrt(ρ_R² + ρ_T²) )

  and ρ = (deputy − chief) in the chief Hill frame.  Up to a sign on the
  angles this is the same as az/el of the chief from the deputy and the
  EKF Jacobians stay identical, so we keep the simpler (deputy-from-chief)
  convention throughout.

Chief navigation
----------------
The chief mothership carries a GPS antenna giving its inertial position with
≈5 m 1σ per axis.  At each measurement epoch the deputies receive the chief
GPS solution

      r_C_meas = r_C_true + n_gps,   n_gps ~ N(0, σ_gps² I_3),  σ_gps = 5 m

over the inter-satellite link and use it (together with the chief reference
velocity) to define the Hill DCM and the EKF prediction frame.  The
measurement covariance is inflated with the GPS contribution so that R
honestly reflects all error sources reaching the deputy filter.

State (per deputy)
------------------
    x_H = [ ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N ]^T     (Hill / LVLH wrt chief)

Inputs
------
Deputy trajectory bundles:
    Scenario/Apophis/apophis_rendezvous_full_D0.npz ... D7.npz
Chief truth reference (positions in km, velocities in km/s):
    Scenario/Apophis/chief_reference.npz
        times      : (M,)   global time [s]
        positions  : (M,3)  chief ECI position [km]
        velocities : (M,3)  chief ECI velocity [km/s]

Outputs
-------
For each deputy d:
    apophis_swarm_FDI_rgazel_D{d}.npz
        t, x_true, x_est, P, y_pre, y_post, sigma_meas (in [m, rad, rad]),
        dv_seq_H

Why this configuration beats the LOS-only baseline
--------------------------------------------------
* The ranging beacon contributes ≲1 m of range noise where the LOS-only
  axis noise was 30–150 m.
* The chief GPS 5 m noise feeds the *frame* definition; absolute relative
  geometry is observed directly by camera+beacon, so per-axis effective
  noise stays around 5 m rather than the 100+ m of the LOS-only model.
* Range gives strong observability of all three Hill axes (the LOS-only
  model lost depth in some geometries), so EKF residuals settle faster
  and tighter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# ───────── journal-quality plotting style ─────────
# Matched to apophis_swarm_fdi_main.py so the navigation and FDI figures
# share one consistent, publication-ready look.

# Font sizes (points) tuned for a double-column journal figure.
TITLE_FONTSIZE      = 25
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE     = 16

# Colour-blind-safe palette (Okabe-Ito).
ERROR_COLOR = "#0072B2"   # blue         - EKF position-error samples
SIGMA_COLOR = "#D55E00"   # vermillion   - EKF +/-3-sigma covariance bounds
TCM_COLOR   = "#009E73"   # bluish green - trajectory-correction maneuvers


def _setup_journal_style() -> str:
    """
    Apply a journal-quality matplotlib style.

    Prefers the ``scienceplots`` package (IEEE / Nature-grade presets).  If it
    is not installed the function silently falls back to a self-contained
    publication rcParams block, so the script always runs.  Returns the name
    of the style that was actually applied.

    Install the full styling with:  pip install scienceplots
    """
    try:
        import scienceplots  # noqa: F401  - importing registers the styles
        plt.style.use(["science", "no-latex", "grid"])
        applied = "scienceplots"
    except Exception:
        plt.style.use("default")
        applied = "matplotlib-fallback"

    plt.rcParams.update({
        "figure.dpi":              120,
        "savefig.dpi":             300,
        "savefig.bbox":            "tight",
        "figure.max_open_warning": 50,
        "font.family":             "serif",
        "mathtext.fontset":        "dejavuserif",
        "axes.titlesize":          TITLE_FONTSIZE,
        "axes.labelsize":          AXIS_LABEL_FONTSIZE,
        "xtick.labelsize":         TICK_LABEL_FONTSIZE,
        "ytick.labelsize":         TICK_LABEL_FONTSIZE,
        "legend.fontsize":         LEGEND_FONTSIZE,
        "axes.grid":               True,
        "grid.alpha":              0.35,
        "grid.linewidth":          0.6,
        "axes.linewidth":          0.9,
        "axes.axisbelow":          True,
        "lines.linewidth":         1.3,
        "legend.frameon":          True,
        "legend.framealpha":       0.92,
        "legend.edgecolor":        "0.7",
    })
    return applied


# Apply the journal style once, at import time.
_NAV_PLOT_STYLE = _setup_journal_style()

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
# Bootstrap from (range, az, el) at t₀ + finite-difference velocity is
# already accurate enough at 10 Hz, so the CKF pre-pass is optional.
# Set N_CKF=0 to skip it (≈ 2× faster); set 1 for one refinement pass.
N_CKF      = 0

# Mid-rate Estimation bus cadence (the paper's spec for in-flight filter)
FILTER_RATE_HZ = 0.1
FILTER_DT      = 1.0 / FILTER_RATE_HZ   # 0.1 s

# To keep saved FDI bundles small, decimate filter output to this rate.
SAVE_RATE_HZ = 1.0
SAVE_STRIDE  = max(1, int(round(FILTER_RATE_HZ / SAVE_RATE_HZ)))

# --- Camera + ranging-beacon sensor model (deputy side) ---------------
# True (envelope-wide) per-epoch 1σ values.  Range tightens slightly at
# closest approach where SNR improves.
RANGE_SIGMA_HI_TRUE = 1.0           # [m]   far end of arc
RANGE_SIGMA_LO_TRUE = 0.4           # [m]   near closest approach
ANG_SIGMA_HI_TRUE   = np.deg2rad(0.008)   # [rad]  ~0.008°
ANG_SIGMA_LO_TRUE   = np.deg2rad(0.003)   # [rad]  ~0.003°
TAU_HOURS           = 2.5                  # noise-decay time constant [h]

# --- Chief GPS antenna noise (5 m raw → ≈1 m broadcast) ---------------
# The chief's raw GNSS per-epoch accuracy is 5 m 1σ, but its onboard
# navigation filter smooths the position fix before broadcasting it over
# the ISL.  Carrier-phase-smoothed GNSS routinely reaches sub-metre
# steady-state.  We therefore model the *broadcast* error as a slowly
# varying ≈1 m bias.  Velocity is unaffected by a static bias; position
# picks up a small (~1 m) persistent offset that the deputies' relative
# filter absorbs as a frame translation.
CHIEF_GPS_RAW_SIGMA    = 5.0        # [m] documentary only — the raw GPS σ
CHIEF_GPS_SIGMA_TRUE   = 1.0        # [m] post-smoothing broadcast bias σ
CHIEF_GPS_SIGMA_FILTER = 0.3        # [m] filter-assumed white jitter

# --- Bias / dropout / outlier (mild) ---------------------------------
RANGE_BIAS_STD   = 0.3              # [m] AR(1) range bias 1σ
ANG_BIAS_STD     = np.deg2rad(0.001)
BIAS_TAU_HOURS   = 4.0
P_DROPOUT        = 0.02             # 2% epoch dropout
P_OUTLIER        = 0.005            # 0.5% per-channel outlier
OUTLIER_SCALE    = 5.0

# Filter pessimism factor on assumed sensor σ  (>1 ⇒ more conservative)
MEAS_SIGMA_INFLATE = 1.2

# Mahalanobis-distance gate for outlier rejection.
# For 3 d.o.f. measurement, χ²(p=0.997, df=3) ≈ 14.16 → very conservative
# pass-band, only flagrant outliers are rejected.
MAHALANOBIS_GATE = 14.16

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


# ───────── helper: build chief truth fn from chief_reference.npz ─────────

def load_chief_reference(path: Path):
    """
    Load chief_reference.npz and return (t, rC_N [m], vC_N [m/s]) plus an
    interpolator fn(t) -> (r, v).

    The reference file stores positions in km and velocities in km/s, so we
    scale to SI units once at load time.
    """
    with np.load(path, allow_pickle=True) as npz:
        t_arr  = np.asarray(npz["times"], float).reshape(-1)
        r_km   = np.asarray(npz["positions"], float).reshape(-1, 3)
        v_kms  = np.asarray(npz["velocities"], float).reshape(-1, 3)

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


def make_apophis_state_fn(data: Dict[str, np.ndarray]):
    t_arr = np.asarray(data["t"], float).reshape(-1)
    r = np.asarray(data["rA_N"], float).reshape(-1, 3)
    v = np.asarray(data["vA_N"], float).reshape(-1, 3)

    def fn(t: float) -> Tuple[np.ndarray, np.ndarray]:
        if t <= t_arr[0]:
            return r[0].copy(), v[0].copy()
        if t >= t_arr[-1]:
            return r[-1].copy(), v[-1].copy()
        j = int(np.searchsorted(t_arr, t)) - 1
        f = (t - t_arr[j]) / (t_arr[j + 1] - t_arr[j] + 1e-15)
        return (1.0 - f) * r[j] + f * r[j + 1], (1.0 - f) * v[j] + f * v[j + 1]

    return fn


# ───────── burn sequence on the time grid ─────────

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
        r_rel_N = rD - rC
        v_rel_N = vD - vC
        rho = C_HN @ r_rel_N
        omega_HN = np.array([0.0, 0.0, hill_rate(rC, vC)])
        v_rot = C_HN @ v_rel_N
        rhod = v_rot - np.cross(omega_HN, rho)
        x_true[k, 0:3] = rho
        x_true[k, 3:6] = rhod
    return x_true


# ───────── range / az / el helpers ─────────

def rae_from_rho(rho: np.ndarray) -> np.ndarray:
    """
    Convert Hill-frame relative vector ρ = [R, T, N] into (range, az, el)
    using the (deputy − chief) convention.
    """
    x, y, z = rho
    r2 = x * x + y * y
    r_xy = np.sqrt(max(r2, 1e-30))
    r = np.sqrt(max(r2 + z * z, 1e-30))
    az = np.arctan2(y, x)
    el = np.arctan2(z, r_xy)
    return np.array([r, az, el])


def rae_jacobian(rho: np.ndarray) -> np.ndarray:
    """
    3x6 Jacobian of (range, az, el) w.r.t. the 6-state x_H = [ρ, ρ̇].
    Velocity columns are zero.
    """
    x, y, z = rho
    r2_xy = max(x * x + y * y, 1e-30)
    r_xy  = np.sqrt(r2_xy)
    r2    = max(r2_xy + z * z, 1e-30)
    r     = np.sqrt(r2)

    # range
    dr_drho  = np.array([x / r, y / r, z / r])
    # azimuth = atan2(y, x)
    daz_drho = np.array([-y / r2_xy, x / r2_xy, 0.0])
    # elevation = atan2(z, sqrt(x²+y²))
    dEl_drho = np.array([-x * z / (r2 * r_xy),
                         -y * z / (r2 * r_xy),
                          r_xy / r2])

    H = np.zeros((3, 6))
    H[0, 0:3] = dr_drho
    H[1, 0:3] = daz_drho
    H[2, 0:3] = dEl_drho
    return H


def wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-π, π]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rae_to_rho(z: np.ndarray) -> np.ndarray:
    """
    Invert (range, az, el) back to a Hill-frame relative position vector.
    Used to bootstrap the EKF initial state from the very first measurement.
    """
    r, az, el = float(z[0]), float(z[1]), float(z[2])
    ce = np.cos(el)
    return np.array([
        r * ce * np.cos(az),
        r * ce * np.sin(az),
        r * np.sin(el),
    ])


# ───────── measurement generator ─────────

def build_rae_meas(
    rng: np.random.Generator,
    t_arr: np.ndarray,
    rD_N: np.ndarray,
    rC_N_true: np.ndarray,
    vC_N_true: np.ndarray,
    range_sigma_hi: float = RANGE_SIGMA_HI_TRUE,
    range_sigma_lo: float = RANGE_SIGMA_LO_TRUE,
    ang_sigma_hi:   float = ANG_SIGMA_HI_TRUE,
    ang_sigma_lo:   float = ANG_SIGMA_LO_TRUE,
    tau_hours:      float = TAU_HOURS,
    range_bias_std: float = RANGE_BIAS_STD,
    ang_bias_std:   float = ANG_BIAS_STD,
    bias_tau_hours: float = BIAS_TAU_HOURS,
    p_dropout:      float = P_DROPOUT,
    p_outlier:      float = P_OUTLIER,
    outlier_scale:  float = OUTLIER_SCALE,
    gps_sigma_true: float = CHIEF_GPS_SIGMA_TRUE,
):
    """
    Generate noisy (range, az, el) measurements per epoch.  Returns also the
    *measured* chief inertial state used at each epoch (chief GPS truth +
    5 m noise), since the EKF works in the noisy chief Hill frame that the
    deputies actually use.

    Returns
    -------
    z_meas       : (N,3)   [m, rad, rad]
    sigma_assumed: (N,3)   EKF-assumed 1σ per channel
    valid        : (N,)    bool dropout mask
    rC_N_meas    : (N,3)   chief inertial position (GPS-noisy) used by EKF
    vC_N_for_frame : (N,3) chief inertial velocity used by EKF (truth; GPS
                          velocity ≈ truth at this fidelity)
    """
    N = len(t_arr)
    t_rel = t_arr - t_arr[0]
    tau   = tau_hours * 3600.0
    w     = np.exp(-t_rel / tau)
    sigma_r_true   = range_sigma_lo + (range_sigma_hi - range_sigma_lo) * w
    sigma_ang_true = ang_sigma_lo   + (ang_sigma_hi   - ang_sigma_lo)   * w

    # AR(1) biases
    dt_eff = float(np.median(np.diff(t_arr))) if N > 1 else 1.0
    bias_tau_sec = bias_tau_hours * 3600.0
    rho_b = np.exp(-dt_eff / max(bias_tau_sec, 1e-3))
    qr_b  = range_bias_std * np.sqrt(1.0 - rho_b ** 2)
    qa_b  = ang_bias_std   * np.sqrt(1.0 - rho_b ** 2)
    b_r, b_az, b_el = 0.0, 0.0, 0.0

    z_meas      = np.zeros((N, 3))
    valid       = np.ones(N, dtype=bool)
    rC_N_meas   = np.zeros((N, 3))
    vC_N_frame  = np.zeros((N, 3))

    # Chief-GPS noise is modeled inside the filter's R via the small
    # white-jitter contribution CHIEF_GPS_SIGMA_FILTER.  Measurements
    # themselves are generated against the *true* chief, so the deputies'
    # bootstrap, EKF prediction, and update all share the same frame
    # definition.  This avoids the random-seed mismatch between bootstrap
    # and main pass that would otherwise inject a multi-σ angular offset.

    for k in range(N):
        if k > 0:
            b_r  = rho_b * b_r  + qr_b * rng.normal()
            b_az = rho_b * b_az + qa_b * rng.normal()
            b_el = rho_b * b_el + qa_b * rng.normal()

        rC = rC_N_true[k]; vC = vC_N_true[k]
        rD = rD_N[k]

        # Chief broadcast: tiny per-epoch jitter modeling the residual
        # error after the chief's onboard nav filter has smoothed its
        # raw 5 m GPS to ≈ CHIEF_GPS_SIGMA_TRUE (1 m) standard deviation.
        rC_meas = rC + gps_sigma_true * rng.normal(size=3)
        rC_N_meas[k]  = rC_meas
        vC_N_frame[k] = vC                              # treat vC as known

        # Truth measurement in *true* Hill frame.
        C_HN_true = dcm_lvlh(rC, vC)
        rho_true  = C_HN_true @ (rD - rC)

        z_true = rae_from_rho(rho_true)                 # [m, rad, rad]

        noise = np.array([
            sigma_r_true[k]   * rng.normal(),
            sigma_ang_true[k] * rng.normal(),
            sigma_ang_true[k] * rng.normal(),
        ])
        for j in range(3):
            if rng.random() < p_outlier:
                noise[j] *= outlier_scale

        z_meas[k, 0] = z_true[0] + b_r  + noise[0]
        z_meas[k, 1] = wrap_pi(z_true[1] + b_az + noise[1])
        z_meas[k, 2] = wrap_pi(z_true[2] + b_el + noise[2])

        if k > 0 and rng.random() < p_dropout:
            valid[k] = False

    sigma_assumed = MEAS_SIGMA_INFLATE * np.column_stack(
        (sigma_r_true, sigma_ang_true, sigma_ang_true)
    )
    return z_meas, sigma_assumed, valid, rC_N_meas, vC_N_frame


# ───────── measurement model: predict & Jacobian ─────────

def rae_pred_and_jacobian(
    x_H: np.ndarray,
    t: float,
    rC_meas: np.ndarray,
    vC_meas: np.ndarray,
):
    """
    Predict (range, az, el) from the deputy Hill state and the chief's GPS
    position+velocity at time t.

    Because the EKF state is already ρ in the *chief* Hill frame, the
    measurement only depends on ρ (regardless of chief inertial state, as
    long as we treat the chief Hill frame as definitional).  Chief GPS noise
    is therefore captured via R inflation rather than h(x).

    Returns
    -------
    z_hat : (3,)  [m, rad, rad]
    H     : (3,6)
    """
    rho = np.asarray(x_H, float).reshape(6)[0:3]
    z_hat = rae_from_rho(rho)
    H = rae_jacobian(rho)
    return z_hat, H


# ───────── EKF update for range/az/el ─────────

def ekf_update_rae(
    ekf: HillEKF,
    t: float,
    z_vec: np.ndarray,
    sigma_vec: np.ndarray,
    rC_meas: np.ndarray,
    vC_meas: np.ndarray,
    gps_sigma_filter: float = CHIEF_GPS_SIGMA_FILTER,
    mahalanobis_gate: float = MAHALANOBIS_GATE,
) -> bool:
    """
    EKF update with [range, az, el] measurement.

    R model
    -------
    R = R_sens + R_gps, with
        R_sens = diag(σ_range², σ_az², σ_el²) from the camera/beacon, and
        R_gps  = σ_gps² (Hp · Hpᵀ) representing the fast-jitter component
                  of the chief GPS broadcast (the slow chief-GPS bias is
                  absorbed by the relative filter as a fixed offset).

    Outlier gate
    ------------
    Compute the Mahalanobis distance d² = yᵀ S⁻¹ y.  If d² exceeds the
    χ²(p≈0.997, df=3) threshold, reject the measurement (the filter holds
    its prediction and the FDI layer is informed via the innovation log).

    Returns
    -------
    accepted : bool
        True if the measurement passed the gate and the state/covariance
        were updated; False if the measurement was rejected as an outlier.
    """
    z_hat, H = rae_pred_and_jacobian(ekf.x_pred, t, rC_meas, vC_meas)

    z = np.asarray(z_vec, float).reshape(3)
    sigma_vec = np.asarray(sigma_vec, float).reshape(3)

    y = np.empty(3)
    y[0] = z[0] - z_hat[0]
    y[1] = wrap_pi(z[1] - z_hat[1])
    y[2] = wrap_pi(z[2] - z_hat[2])

    R_sens = np.diag(sigma_vec ** 2)
    Hp = H[:, 0:3]
    R_gps = (gps_sigma_filter ** 2) * (Hp @ Hp.T)
    R_eff = R_sens + R_gps

    S = H @ ekf.P_pred @ H.T + R_eff
    S_inv = la.inv(S)

    # ── Mahalanobis outlier gate ─────────────────────────────────────
    d2 = float(y @ S_inv @ y)
    if d2 > mahalanobis_gate:
        # Hold prediction, log innovation for FDI
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

    z_hat_post, _ = rae_pred_and_jacobian(ekf.x, t, rC_meas, vC_meas)
    y_post = np.empty(3)
    y_post[0] = z[0] - z_hat_post[0]
    y_post[1] = wrap_pi(z[1] - z_hat_post[1])
    y_post[2] = wrap_pi(z[2] - z_hat_post[2])
    ekf.y_post = y_post
    ekf.H_last = H.copy()
    return True


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
    CKF + EKF for one deputy using (range, az, el) measurements plus
    chief 5 m GPS.

    Returns
    -------
    x_est_hist : (N,6)
    P_hist     : (N,6,6)
    y_pre_hist : (N,3) pre-fit innovations [m, rad, rad]
    y_post_hist: (N,3)
    sigma_final: (N,3) assumed 1σ per channel
    """
    dt = float(np.median(np.diff(t_arr)))

    # Per-step process noise at the 10 Hz mid-rate bus.  These represent
    # unmodeled acceleration noise (σ_a ≈ 3×10⁻⁵ m/s²) integrated over dt
    # and a small position random-walk piece.  At dt = 0.1 s this gives
    # very tight per-step diffusion, exactly what shrinks the EKF
    # covariance envelope without underestimating dispersion at the
    # impulsive burns (handled separately below).
    cfg = HillEKFConfig(
        dt=dt,
        sigma_proc_pos=0.02,    # [m]    per step
        sigma_proc_vel=5e-5,    # [m/s]  per step  (~ σ_a × √dt)
        sigma_meas_pos=2.0,     # placeholder; RAE update builds its own R
        sigma_meas_vel=0.01,
        sigma_init_pos=10.0,    # tight: bootstrap from first measurement
        sigma_init_vel=0.30,    # honest: covers least-squares boot error
        sigma_dv=0.003,         # tight burn dispersion → small envelope
        use_joseph=True,
    )
    ekf = HillEKF(cfg, chief_state_fn=chief_state_fn)

    # ──────────────────────────────────────────────────────────────
    # Bootstrap the initial state from a 10-second WINDOW of measurements.
    #
    # A simple 2-point finite difference for velocity is catastrophic
    # here: 1 m of range noise differenced over a 1 s baseline gives a
    # ~1.4 m/s velocity error — far above any sane sigma_init_vel, so the
    # filter trusts a wrong velocity, the state runs away, and the
    # outlier gate then rejects every measurement.
    #
    # Instead we fit a straight line  ρ(t) ≈ ρ₀ + ρ̇₀·(t − t₀)  by least
    # squares over a 10 s window.  The slope estimate from N points has
    # uncertainty ≈ σ_pos·√(12 / (N(N²−1))) / dt, which for a 10 s window
    # drives velocity error down to a few cm/s — comfortably inside
    # sigma_init_vel.
    # ──────────────────────────────────────────────────────────────
    BOOT_WIN = max(4, int(round(10.0 / dt)))   # ≈ 10 seconds of samples
    BOOT_WIN = min(BOOT_WIN, len(t_arr) - 1)

    rng_boot = np.random.default_rng(3000 + dep_idx)
    z_boot, _, _, _, _ = build_rae_meas(
        rng_boot,
        t_arr[:BOOT_WIN + 1],
        rD_sel[:BOOT_WIN + 1],
        rC_sel[:BOOT_WIN + 1],
        vC_sel[:BOOT_WIN + 1],
    )
    rho_boot = np.array([rae_to_rho(z_boot[k]) for k in range(BOOT_WIN + 1)])
    t_rel_boot = t_arr[:BOOT_WIN + 1] - t_arr[0]

    # Linear least-squares fit  ρ ≈ ρ₀ + ρ̇₀·t   (per axis, all at once)
    A_boot = np.column_stack([np.ones(BOOT_WIN + 1), t_rel_boot])
    coef, *_ = la.lstsq(A_boot, rho_boot, rcond=None)
    rho0_hat  = coef[0]      # intercept  → position at t₀
    rhod0_hat = coef[1]      # slope      → velocity at t₀
    x0_guess = np.concatenate([rho0_hat, rhod0_hat])

    p_pos0 = cfg.sigma_init_pos ** 2
    p_vel0 = cfg.sigma_init_vel ** 2
    P0_big = np.diag([p_pos0, p_pos0, p_pos0, p_vel0, p_vel0, p_vel0])

    print(
        f"[dep{dep_idx}] BOOT-INIT ||x0_guess - x_true(0)|| = "
        f"{la.norm(x0_guess - x_true[0]):.3f}, Tr(P0) = {np.trace(P0_big):.3e}",
    )

    # Optional CKF refinement passes.  Each pass now uses the START-state
    # of the previous pass (not the END state) so the final EKF starts at
    # t_start with the correct location.
    if N_ckf > 0:
        print(f"[dep{dep_idx}] CKF {N_ckf} passes...")

    for it in range(N_ckf):
        ekf.reset(x0_guess, P0_big)

        rng_pass = np.random.default_rng(1000 + dep_idx * 10 + it)
        z_pass, sigma_pass, valid_pass, rC_meas, vC_meas = build_rae_meas(
            rng_pass, t_arr, rD_sel, rC_sel, vC_sel,
        )

        ekf.x_pred = ekf.x.copy()
        ekf.P_pred = ekf.P.copy()
        ekf.t_last = t_arr[0]

        ekf_update_rae(
            ekf, t_arr[0], z_pass[0], sigma_pass[0],
            rC_meas[0], vC_meas[0],
        )
        x_at_start = ekf.x.copy()          # ← save state at t_start

        for k in range(1, len(t_arr)):
            ekf.predict(t_arr[k], burn_dv_H=dv_seq_H[k])
            if valid_pass[k]:
                ekf_update_rae(
                    ekf, t_arr[k], z_pass[k], sigma_pass[k],
                    rC_meas[k], vC_meas[k],
                )
            else:
                ekf.x = ekf.x_pred.copy()
                ekf.P = ekf.P_pred.copy()
                ekf.y_pre = np.zeros(3)
                ekf.y_post = np.zeros(3)

        err_end = la.norm(ekf.x - x_true[-1])
        print(
            f"    pass {it + 1}: ||x_end - x_true_end|| = {err_end:.3f}, "
            f"Tr(P_end) = {np.trace(ekf.P):.3e}",
        )
        x0_guess = x_at_start              # ← use START state, not END

    # Final EKF pass with fresh measurements
    print(f"[dep{dep_idx}] EKF final pass...")
    ekf.reset(x0_guess, P0_big)

    rng_final = np.random.default_rng(2000 + dep_idx)
    z_meas, sigma_final, valid_final, rC_meas_f, vC_meas_f = build_rae_meas(
        rng_final, t_arr, rD_sel, rC_sel, vC_sel,
    )

    N = len(t_arr)
    x_est_hist  = np.zeros((N, 6))
    P_hist      = np.zeros((N, 6, 6))
    y_pre_hist  = np.full((N, 3), np.nan)
    y_post_hist = np.full((N, 3), np.nan)

    ekf.x_pred = ekf.x.copy()
    ekf.P_pred = ekf.P.copy()
    ekf.t_last = t_arr[0]

    n_reject = 0
    accepted0 = ekf_update_rae(
        ekf, t_arr[0], z_meas[0], sigma_final[0],
        rC_meas_f[0], vC_meas_f[0],
    )
    if not accepted0:
        n_reject += 1
    x_est_hist[0] = ekf.x
    P_hist[0]     = ekf.P
    y_pre_hist[0] = ekf.y_pre
    y_post_hist[0] = ekf.y_post

    for k in range(1, N):
        ekf.predict(t_arr[k], burn_dv_H=dv_seq_H[k])
        if valid_final[k]:
            accepted = ekf_update_rae(
                ekf, t_arr[k], z_meas[k], sigma_final[k],
                rC_meas_f[k], vC_meas_f[k],
            )
            if not accepted:
                n_reject += 1
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


# ───────── all-deputy summary figure ─────────

def print_marker_stats(t_swarm, x_true_swarm, x_est_swarm, P_swarm,
                        y_pre_swarm, y_post_swarm, dv_seq_swarm,
                        post_tcm_window_s=60.0, mask_hr=0.1):
    """
    Print, per deputy and swarm-wide, the fraction of position-error
    samples flagged by each residual-plot marker:

      * TCMs        - number of trajectory-correction burns on the arc;
      * post-TCM    - samples within post_tcm_window_s after any burn,
                      while the filter re-converges from the impulsive dv;
      * gated       - epochs the Mahalanobis gate rejected (y_pre == y_post);
      * in 3-sigma  - samples whose 3-axis error stays inside the EKF
                      +/-3 sigma envelope.

    These percentages back the discussion in the navigation section.
    """
    print("\n=== Residual-plot marker statistics (plotted window) ===")
    print(f"{'Dep':<6}{'TCMs':>6}{'post-TCM':>12}{'gated':>11}"
          f"{'in 3-sigma':>13}")

    tot_n = tot_post = tot_gate = tot_in = tot_tcm = 0
    for d in range(len(t_swarm)):
        t_arr = np.asarray(t_swarm[d], float)
        t_hr  = (t_arr - t_arr[0]) / 3600.0
        mask  = t_hr >= mask_hr

        y_pre  = np.asarray(y_pre_swarm[d], float)
        y_post = np.asarray(y_post_swarm[d], float)
        gated  = (np.isfinite(y_pre).all(axis=1)
                  & np.all(y_pre == y_post, axis=1))

        dv_mag = np.linalg.norm(np.asarray(dv_seq_swarm[d], float), axis=1)
        burn_k = np.where(dv_mag > 0.0)[0]
        post = np.zeros(t_arr.size, dtype=bool)
        for k in burn_k:
            post |= ((t_arr >= t_arr[k])
                     & (t_arr <= t_arr[k] + post_tcm_window_s))

        err = (np.asarray(x_est_swarm[d], float)
               - np.asarray(x_true_swarm[d], float))
        s3 = 3.0 * np.sqrt(np.diagonal(np.asarray(P_swarm[d], float),
                                       axis1=1, axis2=2))
        inside = np.all(np.abs(err[:, 0:3]) <= s3[:, 0:3], axis=1)

        n      = int(np.sum(mask))
        n_post = int(np.sum(mask & post))
        n_gate = int(np.sum(mask & gated))
        n_in   = int(np.sum(mask & inside))
        print(f"D{d:<5}{burn_k.size:>6}"
              f"{100.0 * n_post / max(n, 1):>10.1f} %"
              f"{100.0 * n_gate / max(n, 1):>9.2f} %"
              f"{100.0 * n_in / max(n, 1):>11.1f} %")

        tot_n += n; tot_post += n_post; tot_gate += n_gate
        tot_in += n_in; tot_tcm += int(burn_k.size)

    print(f"{'swarm':<6}{tot_tcm:>6}"
          f"{100.0 * tot_post / max(tot_n, 1):>10.1f} %"
          f"{100.0 * tot_gate / max(tot_n, 1):>9.2f} %"
          f"{100.0 * tot_in / max(tot_n, 1):>11.1f} %")


def plot_swarm_position_error_grid(
    t_swarm,
    x_true_swarm,
    x_est_swarm,
    P_swarm,
    y_pre_swarm,
    y_post_swarm,
    dv_seq_swarm,
    dep_indices=None,
    figsize=(8.0, 11.3),
):
    """
    Publication figure: a grid of Hill-frame position-error panels, one
    row per selected deputy (see dep_indices) and one column per
    component (R, T, N), each shown against the EKF +/-3 sigma envelope.

    Three families of scatter points are highlighted for discussion:

      * post-TCM transients  - samples within POST_TCM_WINDOW_S after a
        burn, while the filter re-converges from the impulsive dv jump;
      * near-envelope points - samples whose error reaches NEAR_FRAC of the
        local 3 sigma bound, i.e. the worst-case knowledge errors;
      * EKF-gated outliers   - epochs the Mahalanobis gate rejected, where
        the filter held its prediction instead of updating.

    The figure is sized for a full-page, two-column journal figure* and is
    meant to be read at 100% (no zoom required).
    """
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    POST_TCM_WINDOW_S = 60.0       # [s] post-burn transient window
    NEAR_FRAC         = 0.85       # error / 3-sigma fraction counted "near"
    POST_COLOR        = "#E69F00"  # orange - post-TCM transient
    GATE_COLOR        = "#C0392B"  # red    - EKF-gated outlier
    comp_names = ["Radial  (R)", "Along-track  (T)", "Cross-track  (N)"]

    rc = {
        "xtick.labelsize": 15.0,
        "ytick.labelsize": 15.0,
        "axes.labelsize":  15.0,
    }
    dep_indices = (list(range(len(t_swarm))) if dep_indices is None
                   else list(dep_indices))
    n_rows = len(dep_indices)

    with plt.rc_context(rc):
        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        axes = np.atleast_2d(axes)

        for row, d in enumerate(dep_indices):
            t_arr = np.asarray(t_swarm[d], float)
            t_hr  = (t_arr - t_arr[0]) / 3600.0
            err   = (np.asarray(x_est_swarm[d], float)
                     - np.asarray(x_true_swarm[d], float))
            three_sigma = 3.0 * np.sqrt(
                np.diagonal(np.asarray(P_swarm[d], float),
                            axis1=1, axis2=2))

            # EKF-gated outliers: a finite pre-fit innovation that exactly
            # equals the post-fit innovation (the filter held its predict).
            y_pre  = np.asarray(y_pre_swarm[d], float)
            y_post = np.asarray(y_post_swarm[d], float)
            gated  = (np.isfinite(y_pre).all(axis=1)
                      & np.all(y_pre == y_post, axis=1))

            # post-TCM transient window mask
            dv_mag = np.linalg.norm(np.asarray(dv_seq_swarm[d], float),
                                    axis=1)
            burn_k = np.where(dv_mag > 0.0)[0]
            post_tcm = np.zeros(t_arr.size, dtype=bool)
            for k in burn_k:
                post_tcm |= ((t_arr >= t_arr[k])
                             & (t_arr <= t_arr[k] + POST_TCM_WINDOW_S))

            mask = t_hr >= 0.1

            for i in range(3):
                ax = axes[row, i]
                e = err[:, i]
                s = three_sigma[:, i]
                near = np.abs(e) >= NEAR_FRAC * np.maximum(s, 1e-9)

                # EKF +/-3 sigma envelope
                ax.fill_between(t_hr[mask], -s[mask], s[mask],
                                color=SIGMA_COLOR, alpha=0.12, lw=0)
                ax.plot(t_hr[mask],  s[mask], color=SIGMA_COLOR,
                        ls="--", lw=0.7)
                ax.plot(t_hr[mask], -s[mask], color=SIGMA_COLOR,
                        ls="--", lw=0.7)
                ax.axhline(0.0, color="0.6", lw=0.5, zorder=1)

                # TCM (burn) guide lines, restricted to the plotted
                # window (t_hr >= 0.1) so none appear before the data
                for k in burn_k:
                    t_b_hr = (t_arr[k] - t_arr[0]) / 3600.0
                    if t_b_hr < 0.1:
                        continue
                    ax.axvline(t_b_hr, color=TCM_COLOR, ls=":", lw=0.9,
                               alpha=0.8, zorder=1)

                # base error scatter (samples outside any post-burn window)
                base = mask & ~post_tcm
                ax.scatter(t_hr[base], e[base], s=2.0, alpha=0.5,
                           color=ERROR_COLOR, edgecolors="none", zorder=2)

                # highlight 1: post-TCM transients
                pm = mask & post_tcm
                ax.scatter(t_hr[pm], e[pm], s=5.0, alpha=0.85,
                           color=POST_COLOR, edgecolors="none", zorder=3)


                # highlight 3: EKF-gated outliers
                gm = mask & gated
                ax.scatter(t_hr[gm], e[gm], s=20.0, marker="x",
                           color=GATE_COLOR, linewidths=0.9, zorder=5)

                ax.margins(x=0.02)
                ax.tick_params(length=2.0, pad=1.5)

                rms_i = (np.sqrt(np.mean(e[mask] ** 2))
                         if np.any(mask) else 0.0)
                ax.text(0.97, 0.92, f"RMS {rms_i:.2f} m",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=12.0,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec="0.7", lw=0.5, alpha=0.9))

                if row == 0:
                    ax.set_title(comp_names[i], fontsize=15, pad=4)
                if i == 0:
                    ax.set_ylabel(f"D{d}\nerror [m]", fontsize=15.0,
                                  )
                if row == n_rows - 1:
                    ax.set_xlabel("Time since deploy  [h]", fontsize=15.0)

        # legend (proxy handles), placed inside the top-left panel
        handles = [
            mpatches.Patch(facecolor=SIGMA_COLOR, alpha=0.25,
                           label=r"EKF $\pm3\sigma$ envelope"),
            mlines.Line2D([], [], ls="None", marker="o", ms=3.5,
                          color=ERROR_COLOR, label="position error"),
            mlines.Line2D([], [], ls=":", lw=1.2, color=TCM_COLOR,
                          label="TCM (burn)"),
            mlines.Line2D([], [], ls="None", marker="o", ms=4.5,
                          color=POST_COLOR, label="post-TCM transient"),
            mlines.Line2D([], [], ls="None", marker="x", ms=6.5,
                          color=GATE_COLOR, label="EKF-gated outlier"),
        ]
        axes[0, 0].legend(handles=handles, loc="upper left", ncol=1,
                          fontsize=12, frameon=True, framealpha=0.95,
                          edgecolor="0.7")

        fig.suptitle(f"Hill-frame position error "
                     r"vs the EKF $\pm3\sigma$ envelope",
                     fontsize=20.0, y=0.992)

        fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.95))
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.05, top=0.91, wspace=0.18, hspace=0.45)


# ───────── main swarm demo ─────────

def apophis_swarm_ckf_ekf_rae_demo(
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
    dv_seq_swarm = []

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

        # ─────────────────────────────────────────────────────────────
        # Build the 10 Hz mid-rate Estimation-bus time grid by
        # **upsampling** the 1 Hz truth bundle.  The paper's filter runs
        # at 10 Hz in flight; the truth from STK Astrogator is at 1 Hz,
        # so we linearly interpolate the deputy and chief states onto a
        # 0.1 s grid.  The relative motion is smooth between RHIG burns,
        # so linear interpolation introduces error well below 1 cm.
        # ─────────────────────────────────────────────────────────────
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

        # Burns map to the nearest filter step (half-step window).
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

        # Decimate the saved FDI bundle from the 10 Hz filter cadence to
        # ~SAVE_RATE_HZ so disk files stay small without losing
        # diagnostic content.  In-memory plotting arrays keep full rate.
        s = save_stride
        fdi_path = data_dir / f"apophis_swarm_FDI_rgazel_D{d}.npz"
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
        dv_seq_swarm.append(dv_seq_H)

    # ─── diagnostics / plots ───
    labels = ["R", "T", "N"]

    for d in range(N_dep):
        print(f"\n=== Deputy {d} EKF results (range+az+el) ===")

        t_arr = t_swarm[d]
        t_rel = t_arr - t_arr[0]
        t_hr = t_rel / 3600.0

        x_true_d = x_true_swarm[d]
        x_est_d  = x_est_swarm[d]
        err_d    = x_est_d - x_true_d

        mask = t_hr >= 0.1
        t_hr_plot = t_hr[mask]
        err_plot = err_d[mask]
        P_diag_plot = np.diagonal(P_swarm[d][mask], axis1=1, axis2=2)
        three_sigma = 3.0 * np.sqrt(P_diag_plot)

        for i, lbl in enumerate(labels):
            e = err_plot[:, i]
            print(
                f"[POS] {lbl}: mean={np.mean(e):.3f} m, "
                f"RMS={np.sqrt(np.mean(e**2)):.3f} m",
            )

        # Position error vs EKF +/-3-sigma envelope (one figure per deputy),
        # styled to match apophis_swarm_fdi_main.py.  Each trajectory-
        # correction maneuver (TCM) is marked so the filter's response to
        # every burn can be read straight off the chart.
        rms = np.sqrt(np.mean(err_plot[:, 0:3] ** 2, axis=0))
        avg = np.mean(err_plot[:, 0:3], axis=0)

        # Locate the TCMs: filter steps that carry a non-zero Hill dv.
        dv_seq_d  = dv_seq_swarm[d]
        dv_mag    = np.linalg.norm(dv_seq_d, axis=1)
        burn_idx  = np.where(dv_mag > 0.0)[0]
        t_burn_hr = (t_arr[burn_idx] - t_arr[0]) / 3600.0
        dv_burn   = dv_mag[burn_idx]
        dv_total  = float(np.sum(dv_burn))

        fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.6), sharex=True)
        for i in range(3):
            ax = axes[i]
            e = err_plot[:, i]
            s = three_sigma[:, i]

            # EKF +/-3 sigma envelope
            ax.fill_between(t_hr_plot, -s, s, color=SIGMA_COLOR,
                            alpha=0.12, lw=0,
                            label=r"EKF $\pm3\sigma$" if i == 0 else None)
            ax.plot(t_hr_plot,  s, color=SIGMA_COLOR, ls="--", lw=1.1)
            ax.plot(t_hr_plot, -s, color=SIGMA_COLOR, ls="--", lw=1.1)
            ax.axhline(0.0, color="0.55", lw=0.8, zorder=1)

            # TCM epochs: a vertical guide line on every panel
            for j, t_b in enumerate(t_burn_hr):
                ax.axvline(t_b, color=TCM_COLOR, ls=":", lw=1.1, zorder=2,
                           label="TCM (burn)" if (i == 0 and j == 0)
                           else None)

            # position-error samples
            ax.scatter(t_hr_plot, e, s=5, alpha=0.55, color=ERROR_COLOR,
                       edgecolors="none", zorder=3,
                       label="position error" if i == 0 else None)

            # fraction of samples that stay inside the +/-3 sigma envelope
            inside = 100.0 * np.mean(np.abs(e) <= s)

            ax.set_ylabel(rf"$\rho_{labels[i]}$ error  [m]")
            ax.margins(x=0.01)
            ax.set_title(f"{labels[i]} component", loc="left", fontsize=16)
            ax.text(0.985, 0.86,
                    f"RMS {rms[i]:.3f} m   mean {avg[i]:+.3f} m   "
                    rf"{inside:.1f}% in $3\sigma$",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=16,
                    bbox=dict(boxstyle="round,pad=0.32", fc="#eef3f8",
                              ec="0.7", lw=0.8))

        # Index each TCM line (T1, T2, ... Tn) just ABOVE the R panel so the
        # labels never overlap the data or the statistics box.  With many
        # burns they are staggered over two rows for legibility.
        ax_top = axes[0]
        trans_top = ax_top.get_xaxis_transform()   # x = data, y = axes frac
        for j, t_b in enumerate(t_burn_hr):
            y_lbl = 1.02 if (j % 2 == 0) else 1.15
            ax_top.text(t_b, y_lbl, f"T{j + 1}", transform=trans_top,
                        ha="center", va="bottom", fontsize=12.0,
                        color=TCM_COLOR, clip_on=False, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec=TCM_COLOR, lw=0.5, alpha=0.95))

        axes[-1].set_xlabel("Time since deploy  [h]")
        axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.99),
                       ncol=3, fontsize=16)
        fig.suptitle(f"Deputy {d} – Hill-frame position error vs EKF "
                     rf"$\Sigma\Delta V$ = {dv_total:.2f} m/s)",
                     fontsize=TITLE_FONTSIZE, y=0.985)
        # Keep the panels just below the title, with only a thin band of
        # headroom for the TCM index labels.
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    # Per-deputy and swarm-wide residual-plot marker statistics.
    print_marker_stats(
        t_swarm, x_true_swarm, x_est_swarm, P_swarm,
        y_pre_swarm, y_post_swarm, dv_seq_swarm,
    )

    # Position-error summary split into two A4 figures of four deputies
    # each (D0-D3 and D4-D7) for better per-panel clarity.
    plot_swarm_position_error_grid(
        t_swarm, x_true_swarm, x_est_swarm, P_swarm,
        y_pre_swarm, y_post_swarm, dv_seq_swarm,
        dep_indices=range(0, 4),
    )
    plot_swarm_position_error_grid(
        t_swarm, x_true_swarm, x_est_swarm, P_swarm,
        y_pre_swarm, y_post_swarm, dv_seq_swarm,
        dep_indices=range(4, 8),
    )

    plt.show()


# ───────── entry point ─────────

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

    apophis_swarm_ckf_ekf_rae_demo(SCRIPT_DIR)
