#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geo_swarm_realistic_nav_demo.py
===============================

“Realistic GEO-style” navigation demo for a chief + multiple deputies over a
few-hour propagation window, using measurement sources you’d actually expect:

Chief OD (absolute orbit determination):
  - Ground-station tracking arcs (range + Doppler/range-rate) during scheduled contacts
  - Optional GEO GNSS / “GPS-like” onboard nav (intermittent availability)

Deputy relative navigation (relative to chief in Hill/LVLH):
  - Crosslink ranging (range-only or range+range-rate) between deputy and chief
  - Optional deputy GNSS / “GPS-like” relative state measurement

You get:
  - Chief OD plots (errors vs 3σ, range/Doppler innovations, contact shading)
  - Per-deputy plots (3D relative traj + covariance ellipsoids, errors vs 3σ, crosslink residuals)

Assumptions (reasonable for a demo):
  - Uses truth from NPZ bundles (same structure as your Apophis fleet bundles)
  - Earth rotation is modeled (simple spin about +Z)
  - Stations are WGS84 geodetic → ECEF → ECI
  - No light-time, troposphere/ionosphere models (captured via bias + noise instead)
  - Chief EKF dynamics = two-body + process noise acceleration
  - Deputy EKF dynamics = your existing HillEKF propagation

File structure expected (default):
  Scenario/GEO/  (or wherever you run the script)
    apophis_rendezvous_full_D0.npz
    ...
    apophis_rendezvous_full_D7.npz

Change BUNDLE_TEMPLATE if your GEO files have a different name.

"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
#                                  TOGGLES
# ──────────────────────────────────────────────────────────────────────────────

# --- bundle naming ---
N_DEP = 8
BUNDLE_TEMPLATE = "apophis_rendezvous_full_D{idx}.npz"   # <-- change for your GEO scenario
DATA_DIR = None  # None => SCRIPT_DIR; else Path("Scenario/GEO") etc.

# --- time handling / decimation ---
CROP_AFTER_DEPLOY_HR = 0.5

# Pick ONE:
NAV_RATE_HZ = 1.0              # desired nav propagation/update grid (recommended for few-hour OD)
NAV_DECIM_OVERRIDE = None      # set to int to force decimation factor (overrides NAV_RATE_HZ)

# --- measurement mode toggles ---
# Chief absolute OD measurements:
CHIEF_MODE = "ground"          # "ground" | "gnss" | "ground+gnss"
GROUND_USE_RANGE = True
GROUND_USE_DOPPLER = True      # Doppler approximated as instantaneous range-rate
GROUND_USE_ELEV_MASK = True
ELEV_MASK_DEG = 10.0

# Deputy relative measurements (relative to chief in Hill):
#  "gps"       -> GPS-like relative state only
#  "r"         -> crosslink range only
#  "rr"        -> crosslink range + range-rate only
#  "gps+r"     -> GPS + crosslink range
#  "gps+rr"    -> GPS + crosslink range + range-rate
DEPUTY_MODE = "gps+rr"

# --- plotting toggles ---
PLOT_CHIEF = True
PLOT_DEPUTIES = True
PLOT_ELLIPSOIDS = True
ELLIPSOID_NSTD = 3.0
ELLIPSOID_EVERY = 60          # draw ellipsoid every N samples
ELLIPSOID_NPTS = 12

# If True, does per-deputy figures (separate windows). If False, you’ll still get a single plt.show().
PLOT_SEPARATE_PER_DEPUTY = True

# --- Earth / OD model constants ---
MU_EARTH = 3.986004418e14        # [m^3/s^2]
OMEGA_EARTH = 7.2921150e-5       # [rad/s]

# WGS84 (for station placement)
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# --- ground tracking arc schedule (fits “few hours”) ---
# periodic arcs: start offset, then repeat every period, each arc lasts duration
GROUND_ARC_START_MIN = 10.0
GROUND_ARC_PERIOD_MIN = 60.0
GROUND_ARC_DURATION_MIN = 12.0

GROUND_MEAS_RATE_HZ = 0.2        # e.g. 0.2 Hz => one measurement every 5 seconds during contact
GROUND_RANGE_SIGMA_M = 5.0       # “good” two-way range-ish
GROUND_DOPP_SIGMA_MPS = 0.005    # 5 mm/s-ish equivalent instantaneous range-rate (tunable)

# elevation-dependent scaling: sigma *= 1/sin(elev) (clamped at elev mask)
GROUND_SCALE_WITH_ELEV = True

# ground measurement realism knobs
GROUND_P_DROPOUT = 0.02
GROUND_P_OUTLIER = 0.005
GROUND_OUTLIER_SCALE = 8.0

# slowly varying range bias (troposphere/instrument) AR(1)
GROUND_RANGE_BIAS_STD_M = 10.0
GROUND_RANGE_BIAS_TAU_HR = 2.0

# filter assumed noise scaling (mis-model)
GROUND_SIGMA_ASSUMED_SCALE = 1.0

# --- chief GNSS (optional) ---
CHIEF_GNSS_RATE_HZ = 1.0
CHIEF_GNSS_AVAIL_PROB = 0.6       # GEO GNSS can be intermittent; tune this
CHIEF_GNSS_AVAIL_TAU_MIN = 10.0   # availability burstiness
CHIEF_GNSS_SIGMA_POS_M = 20.0
CHIEF_GNSS_SIGMA_VEL_MPS = 0.05
CHIEF_GNSS_SIGMA_ASSUMED_SCALE = 1.0

# --- chief EKF tuning ---
CHIEF_INIT_POS_SIGMA_M = 2000.0
CHIEF_INIT_VEL_SIGMA_MPS = 1.0
CHIEF_INIT_BIAS = np.array([500.0, -300.0, 200.0, 0.1, -0.08, 0.05])  # initial state error guess (m, m/s)

CHIEF_SIGMA_ACCEL_PROC = 3e-7      # [m/s^2] process accel noise (tune)
CHIEF_RK4_SUBSTEPS = 1             # increase if your nav dt is large (e.g. 10s)

# NIS gates (optional robustness)
CHIEF_USE_NIS_GATING = True
NIS_GATE_RANGE = 10.83    # chi2_1 99.9%
NIS_GATE_RANGEDOT = 10.83 # chi2_1 99.9%
NIS_GATE_RANGE_DOPP_2D = 13.82  # chi2_2 99.9%

# --- crosslink (deputy<->chief) ---
CROSSLINK_MEAS_RATE_HZ = 1.0
CROSSLINK_RANGE_SIGMA_M = 2.0
CROSSLINK_RR_SIGMA_MPS = 0.02

CROSSLINK_P_DROPOUT = 0.05
CROSSLINK_P_OUTLIER = 0.01
CROSSLINK_OUTLIER_SCALE = 6.0

CROSSLINK_RANGE_BIAS_STD_M = 2.0
CROSSLINK_RANGE_BIAS_TAU_HR = 4.0

CROSSLINK_SIGMA_ASSUMED_SCALE = 1.0

# Optionally inflate crosslink R using chief OD covariance projected on LOS
INCLUDE_CHIEF_UNCERT_IN_CROSSLINK_R = True

# --- deputy “GPS-like” relative state measurement ---
DEP_GNSS_RATE_HZ = 1.0
DEP_GNSS_AVAIL_PROB = 0.5
DEP_GNSS_AVAIL_TAU_MIN = 10.0
DEP_GNSS_SIGMA_POS_M = 10.0
DEP_GNSS_SIGMA_VEL_MPS = 0.05
DEP_GNSS_SIGMA_ASSUMED_SCALE = 1.0

# --- deputy HillEKF tuning (propagation only; measurement update done here) ---
DEPUTY_SIGMA_INIT_POS_M = 300.0
DEPUTY_SIGMA_INIT_VEL_MPS = 0.3
DEPUTY_SIGMA_PROC_POS_M = 0.2
DEPUTY_SIGMA_PROC_VEL_MPS = 5e-4
DEPUTY_SIGMA_DV_MPS = 0.02

DEPUTY_INIT_BIAS = np.array([200.0, -150.0, 100.0, 0.05, -0.04, 0.02])

DEPUTY_USE_NIS_GATING = True
NIS_GATE_XLINK_R = 10.83
NIS_GATE_XLINK_RR = 13.82

# --- ground station network ---
# For “real” GEO ops you’d have a network; for a short demo 1-2 stations is plenty.
@dataclass
class GroundStation:
    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float

GROUND_STATIONS: List[GroundStation] = [
    GroundStation("GS_AZ", lat_deg=33.0, lon_deg=-111.0, alt_m=500.0),
    GroundStation("GS_EU", lat_deg=40.0, lon_deg=10.0, alt_m=100.0),
]

# ──────────────────────────────────────────────────────────────────────────────
#                         locate ARISE root / HillEKF
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR if DATA_DIR is None else Path(DATA_DIR)

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
    dcm_lvlh, hill_rate
)

# ──────────────────────────────────────────────────────────────────────────────
#                                   utilities
# ──────────────────────────────────────────────────────────────────────────────

def rot_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def geodetic_to_ecef_wgs84(lat_rad: float, lon_rad: float, alt_m: float) -> np.ndarray:
    """
    WGS84 geodetic to ECEF.
    """
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def station_state_eci(st: GroundStation, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Station ECI state from geodetic coords using simple Earth rotation.

    Assumes at t=0: ECI and ECEF aligned (GMST=0).
    """
    lat = np.deg2rad(st.lat_deg)
    lon = np.deg2rad(st.lon_deg)
    r_ecef = geodetic_to_ecef_wgs84(lat, lon, st.alt_m)

    theta = OMEGA_EARTH * t
    R = rot_z(theta)
    r_eci = R @ r_ecef

    omega = np.array([0.0, 0.0, OMEGA_EARTH])
    v_eci = np.cross(omega, r_eci)

    return r_eci, v_eci, r_ecef, theta


def elevation_deg(r_sat_eci: np.ndarray, st: GroundStation, t: float) -> float:
    """
    Compute elevation angle in degrees using ECEF ENU geometry.
    """
    r_st_eci, _v_st_eci, r_st_ecef, theta = station_state_eci(st, t)

    # ECI -> ECEF
    r_sat_ecef = rot_z(-theta) @ np.asarray(r_sat_eci, float).reshape(3)
    rho_ecef = r_sat_ecef - r_st_ecef
    rho_norm = la.norm(rho_ecef)
    if rho_norm < 1e-9:
        return -90.0

    rho_hat = rho_ecef / rho_norm

    lat = np.deg2rad(st.lat_deg)
    lon = np.deg2rad(st.lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # ENU basis in ECEF
    east  = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up    = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])

    u = float(np.dot(rho_hat, up))
    elev = np.arcsin(np.clip(u, -1.0, 1.0))
    return float(np.rad2deg(elev))


def build_periodic_arc_mask(t_arr: np.ndarray,
                            start_offset_s: float,
                            period_s: float,
                            duration_s: float) -> np.ndarray:
    """
    Periodic contact arcs on a given time grid.
    """
    t0 = float(t_arr[0])
    t_end = float(t_arr[-1])
    mask = np.zeros(len(t_arr), dtype=bool)

    if duration_s <= 0.0 or period_s <= 0.0:
        return mask

    start = t0 + start_offset_s
    while start <= t_end:
        end = start + duration_s
        mask |= (t_arr >= start) & (t_arr <= end)
        start += period_s

    return mask


def meas_decim_from_rates(nav_rate_hz: float, meas_rate_hz: float) -> int:
    if meas_rate_hz <= 0.0:
        return 10**9
    if nav_rate_hz <= 0.0:
        return 1
    d = int(round(nav_rate_hz / meas_rate_hz))
    return max(1, d)


def make_linear_interp_fn(t_grid: np.ndarray, y_grid: np.ndarray):
    """
    y_grid can be (N,3) or (N,6) or (N,6,6). Returns a function y(t).
    Clamps at ends.
    """
    t_grid = np.asarray(t_grid, float).reshape(-1)
    y_grid = np.asarray(y_grid)

    def interp(t: float):
        t = float(t)
        if t <= t_grid[0]:
            return y_grid[0]
        if t >= t_grid[-1]:
            return y_grid[-1]
        j = int(np.searchsorted(t_grid, t) - 1)
        t0 = t_grid[j]
        t1 = t_grid[j + 1]
        f = (t - t0) / (t1 - t0 + 1e-15)
        return (1.0 - f) * y_grid[j] + f * y_grid[j + 1]

    return interp


def intervals_from_mask(t_hr: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    """
    Convert boolean mask on a time vector into contiguous intervals (in hours).
    """
    mask = np.asarray(mask, bool)
    t_hr = np.asarray(t_hr, float)
    intervals = []
    in_seg = False
    t_start = None

    for k in range(len(mask)):
        if mask[k] and not in_seg:
            in_seg = True
            t_start = t_hr[k]
        if in_seg and (not mask[k] or k == len(mask) - 1):
            t_end = t_hr[k] if mask[k] else t_hr[k - 1]
            intervals.append((float(t_start), float(t_end)))
            in_seg = False

    return intervals


# ──────────────────────────────────────────────────────────────────────────────
#                                   bundle I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_bundle(path: Path) -> Dict[str, np.ndarray]:
    wanted = ("t", "rD_N", "vD_N", "rC_N", "vC_N", "impulses")
    out: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as npz:
        for k in wanted:
            if k in npz:
                out[k] = npz[k]
    if "impulses" not in out:
        out["impulses"] = np.zeros((0, 4), dtype=float)
    return out


def build_burn_sequence(impulses: np.ndarray, t_arr: np.ndarray, dt: float) -> np.ndarray:
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


def decimate_indices(t_full: np.ndarray, nav_rate_hz: float, decim_override: Optional[int]) -> int:
    t_full = np.asarray(t_full, float).reshape(-1)
    dt_truth = float(np.median(np.diff(t_full))) if len(t_full) > 1 else 1.0

    if decim_override is not None:
        return max(1, int(decim_override))

    # choose decimation factor that gives roughly nav dt = 1/nav_rate_hz
    dt_nav = 1.0 / max(nav_rate_hz, 1e-9)
    decim = int(round(dt_nav / dt_truth))
    return max(1, decim)


def hill_state_from_eci(rD_N, vD_N, rC_N, vC_N) -> np.ndarray:
    rC = np.asarray(rC_N, float).reshape(3)
    vC = np.asarray(vC_N, float).reshape(3)
    rD = np.asarray(rD_N, float).reshape(3)
    vD = np.asarray(vD_N, float).reshape(3)

    C_HN = dcm_lvlh(rC, vC)
    r_rel_N = rD - rC
    v_rel_N = vD - vC

    rho = C_HN @ r_rel_N
    omega_H = np.array([0.0, 0.0, hill_rate(rC, vC)])
    v_rot = C_HN @ v_rel_N
    rhod = v_rot - np.cross(omega_H, rho)

    x = np.zeros(6)
    x[0:3] = rho
    x[3:6] = rhod
    return x


def eci_to_hill_traj(rD_N, vD_N, rC_N, vC_N) -> np.ndarray:
    N = np.asarray(rD_N).shape[0]
    x_true = np.zeros((N, 6))
    for k in range(N):
        x_true[k] = hill_state_from_eci(rD_N[k], vD_N[k], rC_N[k], vC_N[k])
    return x_true


# ──────────────────────────────────────────────────────────────────────────────
#                             chief EKF (ECI) model
# ──────────────────────────────────────────────────────────────────────────────

def twobody_accel(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, float).reshape(3)
    rn = la.norm(r)
    if rn < 1e-9:
        return np.zeros(3)
    return -MU_EARTH * r / (rn**3)


def twobody_dadr(r: np.ndarray) -> np.ndarray:
    """
    ∂a/∂r for a = -μ r / r^3
    """
    r = np.asarray(r, float).reshape(3)
    rn = la.norm(r)
    if rn < 1e-9:
        return np.zeros((3, 3))
    I = np.eye(3)
    rrT = np.outer(r, r)
    return -MU_EARTH * (I / rn**3 - 3.0 * rrT / rn**5)


def rk4_step_rv(r: np.ndarray, v: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    RK4 propagate two-body for one step dt.
    """
    r = np.asarray(r, float).reshape(3)
    v = np.asarray(v, float).reshape(3)
    dt = float(dt)

    def f(state):
        rr = state[0:3]
        vv = state[3:6]
        aa = twobody_accel(rr)
        return np.hstack([vv, aa])

    x = np.hstack([r, v])
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return x_next[0:3], x_next[3:6]


def propagate_chief_state(r: np.ndarray, v: np.ndarray, dt: float, substeps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    RK4 with optional substeps.
    """
    substeps = max(1, int(substeps))
    h = float(dt) / substeps
    rr = np.asarray(r, float).reshape(3)
    vv = np.asarray(v, float).reshape(3)
    for _ in range(substeps):
        rr, vv = rk4_step_rv(rr, vv, h)
    return rr, vv


def chief_predict(x: np.ndarray, P: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    EKF predict step for chief (two-body + accel process noise).
    x: (6,) [r;v] in ECI
    """
    x = np.asarray(x, float).reshape(6)
    P = np.asarray(P, float).reshape(6, 6)
    dt = float(dt)

    r = x[0:3]
    v = x[3:6]

    # state propagation (RK4)
    r_next, v_next = propagate_chief_state(r, v, dt, CHIEF_RK4_SUBSTEPS)
    x_pred = np.hstack([r_next, v_next])

    # covariance propagation
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = twobody_dadr(r)

    Phi = np.eye(6) + A * dt  # OK for small dt; tune if you go big dt

    # white accel noise -> discrete Q
    q = float(CHIEF_SIGMA_ACCEL_PROC)**2
    I3 = np.eye(3)
    Q = np.block([
        [ (dt**3)/3 * q * I3, (dt**2)/2 * q * I3 ],
        [ (dt**2)/2 * q * I3, (dt)     * q * I3 ],
    ])

    P_pred = Phi @ P @ Phi.T + Q
    return x_pred, P_pred


def ekf_update_generic(x_pred: np.ndarray, P_pred: np.ndarray,
                       z: np.ndarray, zhat: np.ndarray,
                       H: np.ndarray, R: np.ndarray,
                       use_joseph: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic EKF update. Returns (x, P, y, S).
    """
    x_pred = np.asarray(x_pred, float).reshape(-1)
    P_pred = np.asarray(P_pred, float)
    z = np.asarray(z, float).reshape(-1)
    zhat = np.asarray(zhat, float).reshape(-1)
    H = np.asarray(H, float)
    R = np.asarray(R, float)

    y = (z - zhat).reshape(-1, 1)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ la.inv(S)
    x = x_pred + (K @ y).reshape(-1)

    I = np.eye(len(x_pred))
    if use_joseph:
        KH = K @ H
        P = (I - KH) @ P_pred @ (I - KH).T + K @ R @ K.T
    else:
        P = (I - K @ H) @ P_pred

    return x, P, y.reshape(-1), S


def nis_value(y: np.ndarray, S: np.ndarray) -> float:
    y = np.asarray(y, float).reshape(-1, 1)
    S = np.asarray(S, float)
    return float((y.T @ la.inv(S) @ y).reshape(()))


# ──────────────────────────────────────────────────────────────────────────────
#                       ground tracking measurement model
# ──────────────────────────────────────────────────────────────────────────────

def ground_range_dopp_model(x: np.ndarray, st: GroundStation, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Measurement model:
      z = [range, range_rate]^T
    """
    x = np.asarray(x, float).reshape(6)
    r = x[0:3]
    v = x[3:6]

    r_st, v_st, _r_ecef, _theta = station_state_eci(st, t)

    rho = r - r_st
    rhon = la.norm(rho)
    if rhon < 1e-9:
        u = np.zeros(3)
    else:
        u = rho / rhon

    v_rel = v - v_st
    rdot = float(np.dot(u, v_rel))
    rng = float(rhon)

    # Jacobian H (2x6)
    H = np.zeros((2, 6))
    # range
    H[0, 0:3] = u
    # range-rate
    if rhon >= 1e-9:
        I3 = np.eye(3)
        grad_r = (I3 - np.outer(u, u)) @ v_rel / rhon
    else:
        grad_r = np.zeros(3)
    H[1, 0:3] = grad_r
    H[1, 3:6] = u

    zhat = np.array([rng, rdot], dtype=float)
    return zhat, H


def simulate_ground_tracking(
    rng: np.random.Generator,
    t_arr: np.ndarray,
    r_true: np.ndarray,
    v_true: np.ndarray,
    stations: List[GroundStation],
    nav_rate_hz: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate ground tracking measurements (range + Doppler) per station on nav grid.
    Returns dict keyed by station name with fields:
      valid, elev_deg, z (N,2), sigma (N,2), contact (N,)
    """
    t_arr = np.asarray(t_arr, float).reshape(-1)
    r_true = np.asarray(r_true, float).reshape(-1, 3)
    v_true = np.asarray(v_true, float).reshape(-1, 3)

    N = len(t_arr)

    # scheduled arcs (same for all stations, but you can customize per station if you want)
    arc_mask_sched = build_periodic_arc_mask(
        t_arr,
        start_offset_s=GROUND_ARC_START_MIN * 60.0,
        period_s=GROUND_ARC_PERIOD_MIN * 60.0,
        duration_s=GROUND_ARC_DURATION_MIN * 60.0,
    )

    meas_decim = meas_decim_from_rates(nav_rate_hz, GROUND_MEAS_RATE_HZ)

    out: Dict[str, Dict[str, np.ndarray]] = {}

    dt = float(np.median(np.diff(t_arr))) if N > 1 else 1.0
    # AR(1) bias setup
    tau = max(GROUND_RANGE_BIAS_TAU_HR * 3600.0, 1e-6)
    rho_b = np.exp(-dt / tau)
    q_b = GROUND_RANGE_BIAS_STD_M * np.sqrt(max(1.0 - rho_b**2, 0.0))

    for st in stations:
        z = np.full((N, 2), np.nan)
        valid = np.zeros(N, dtype=bool)
        elev = np.full(N, np.nan)
        sigma_true = np.zeros((N, 2))
        sigma_assumed = np.zeros((N, 2))
        contact = np.zeros(N, dtype=bool)

        b = 0.0
        for k in range(N):
            t = float(t_arr[k])

            # bias evolves continuously
            if k > 0:
                b = rho_b * b + q_b * rng.normal()

            # compute elevation from truth
            elev_k = elevation_deg(r_true[k], st, t)
            elev[k] = elev_k

            # contact logic
            in_sched = bool(arc_mask_sched[k])
            in_view = (elev_k >= ELEV_MASK_DEG) if GROUND_USE_ELEV_MASK else True
            in_contact = in_sched and in_view
            contact[k] = in_contact

            # measurement cadence
            if (k % meas_decim) != 0:
                continue

            if not in_contact:
                continue

            # dropout
            if rng.random() < GROUND_P_DROPOUT and k != 0:
                continue

            # true measurement
            xk_true = np.hstack([r_true[k], v_true[k]])
            z_true, _H = ground_range_dopp_model(xk_true, st, t)

            # noise scaling with elevation
            elev_rad = np.deg2rad(max(elev_k, ELEV_MASK_DEG))
            scale = (1.0 / np.sin(elev_rad)) if GROUND_SCALE_WITH_ELEV else 1.0

            sig_r = GROUND_RANGE_SIGMA_M * scale
            sig_d = GROUND_DOPP_SIGMA_MPS * scale

            sigma_true[k, 0] = sig_r
            sigma_true[k, 1] = sig_d
            sigma_assumed[k, 0] = GROUND_SIGMA_ASSUMED_SCALE * sig_r
            sigma_assumed[k, 1] = GROUND_SIGMA_ASSUMED_SCALE * sig_d

            # outliers
            nr = rng.normal() * sig_r
            nd = rng.normal() * sig_d
            if rng.random() < GROUND_P_OUTLIER:
                nr *= GROUND_OUTLIER_SCALE
                nd *= GROUND_OUTLIER_SCALE

            # apply bias only to range (typical)
            z[k, 0] = z_true[0] + b + nr
            z[k, 1] = z_true[1] + nd

            valid[k] = True

        out[st.name] = dict(
            z=z,
            valid=valid,
            elev_deg=elev,
            sigma_true=sigma_true,
            sigma_assumed=sigma_assumed,
            contact=contact,
            arc_sched=arc_mask_sched.copy(),
        )

    return out


# ──────────────────────────────────────────────────────────────────────────────
#                         GNSS availability / measurements
# ──────────────────────────────────────────────────────────────────────────────

def simulate_markov_availability(rng: np.random.Generator,
                                N: int,
                                dt: float,
                                p_on: float,
                                tau_min: float) -> np.ndarray:
    """
    2-state Markov chain with stationary P(on)=p_on and approx correlation time tau_min.
    """
    tau = max(tau_min * 60.0, 1e-6)
    p_change = 1.0 - np.exp(-dt / tau)
    a = (1.0 - p_on) * p_change  # on->off
    b = p_on * p_change          # off->on

    avail = np.zeros(N, dtype=bool)
    state = (rng.random() < p_on)
    avail[0] = state
    for k in range(1, N):
        if state:
            if rng.random() < a:
                state = False
        else:
            if rng.random() < b:
                state = True
        avail[k] = state
    return avail


def simulate_gnss_eci(
    rng: np.random.Generator,
    t_arr: np.ndarray,
    r_true: np.ndarray,
    v_true: np.ndarray,
    nav_rate_hz: float,
    meas_rate_hz: float,
    avail_prob: float,
    avail_tau_min: float,
    sigma_pos: float,
    sigma_vel: float,
    sigma_assumed_scale: float,
) -> Dict[str, np.ndarray]:
    """
    GNSS PVT-like measurement for ECI state:
      z = [r; v] + noise
    with intermittent availability.
    """
    t_arr = np.asarray(t_arr, float).reshape(-1)
    r_true = np.asarray(r_true, float).reshape(-1, 3)
    v_true = np.asarray(v_true, float).reshape(-1, 3)
    N = len(t_arr)
    dt = float(np.median(np.diff(t_arr))) if N > 1 else 1.0

    z = np.full((N, 6), np.nan)
    valid = np.zeros(N, dtype=bool)

    sigma_true = np.zeros((N, 6))
    sigma_assumed = np.zeros((N, 6))

    avail = simulate_markov_availability(rng, N, dt, avail_prob, avail_tau_min)
    meas_decim = meas_decim_from_rates(nav_rate_hz, meas_rate_hz)

    for k in range(N):
        if (k % meas_decim) != 0:
            continue
        if not avail[k]:
            continue

        noise_r = sigma_pos * rng.normal(size=3)
        noise_v = sigma_vel * rng.normal(size=3)

        z[k, 0:3] = r_true[k] + noise_r
        z[k, 3:6] = v_true[k] + noise_v
        valid[k] = True

        sigma_true[k, 0:3] = sigma_pos
        sigma_true[k, 3:6] = sigma_vel
        sigma_assumed[k, 0:3] = sigma_assumed_scale * sigma_pos
        sigma_assumed[k, 3:6] = sigma_assumed_scale * sigma_vel

    return dict(z=z, valid=valid, sigma_true=sigma_true, sigma_assumed=sigma_assumed, avail=avail)


def simulate_gnss_hill_rel(
    rng: np.random.Generator,
    t_arr: np.ndarray,
    x_true_hill: np.ndarray,
    nav_rate_hz: float,
    meas_rate_hz: float,
    avail_prob: float,
    avail_tau_min: float,
    sigma_pos: float,
    sigma_vel: float,
    sigma_assumed_scale: float,
) -> Dict[str, np.ndarray]:
    """
    “GPS-like” measurement directly on Hill relative state:
      z = x_true_hill + noise
    Intermittent availability.
    """
    t_arr = np.asarray(t_arr, float).reshape(-1)
    x_true_hill = np.asarray(x_true_hill, float).reshape(-1, 6)
    N = len(t_arr)
    dt = float(np.median(np.diff(t_arr))) if N > 1 else 1.0

    z = np.full((N, 6), np.nan)
    valid = np.zeros(N, dtype=bool)

    sigma_true = np.zeros((N, 6))
    sigma_assumed = np.zeros((N, 6))

    avail = simulate_markov_availability(rng, N, dt, avail_prob, avail_tau_min)
    meas_decim = meas_decim_from_rates(nav_rate_hz, meas_rate_hz)

    for k in range(N):
        if (k % meas_decim) != 0:
            continue
        if not avail[k]:
            continue

        noise = np.hstack([
            sigma_pos * rng.normal(size=3),
            sigma_vel * rng.normal(size=3),
        ])
        z[k] = x_true_hill[k] + noise
        valid[k] = True

        sigma_true[k, 0:3] = sigma_pos
        sigma_true[k, 3:6] = sigma_vel
        sigma_assumed[k, 0:3] = sigma_assumed_scale * sigma_pos
        sigma_assumed[k, 3:6] = sigma_assumed_scale * sigma_vel

    return dict(z=z, valid=valid, sigma_true=sigma_true, sigma_assumed=sigma_assumed, avail=avail)


# ──────────────────────────────────────────────────────────────────────────────
#                          crosslink measurement model
# ──────────────────────────────────────────────────────────────────────────────

def simulate_crosslink(
    rng: np.random.Generator,
    t_arr: np.ndarray,
    rD_true: np.ndarray,
    vD_true: np.ndarray,
    rC_true: np.ndarray,
    vC_true: np.ndarray,
    nav_rate_hz: float,
) -> Dict[str, np.ndarray]:
    """
    Simulate crosslink measurements between deputy and chief:
      z = [range, range_rate] + bias + noise
    (bias applied to range only)
    """
    t_arr = np.asarray(t_arr, float).reshape(-1)
    rD_true = np.asarray(rD_true, float).reshape(-1, 3)
    vD_true = np.asarray(vD_true, float).reshape(-1, 3)
    rC_true = np.asarray(rC_true, float).reshape(-1, 3)
    vC_true = np.asarray(vC_true, float).reshape(-1, 3)
    N = len(t_arr)
    dt = float(np.median(np.diff(t_arr))) if N > 1 else 1.0

    meas_decim = meas_decim_from_rates(nav_rate_hz, CROSSLINK_MEAS_RATE_HZ)

    # bias AR(1)
    tau = max(CROSSLINK_RANGE_BIAS_TAU_HR * 3600.0, 1e-6)
    rho_b = np.exp(-dt / tau)
    q_b = CROSSLINK_RANGE_BIAS_STD_M * np.sqrt(max(1.0 - rho_b**2, 0.0))
    b = 0.0

    z = np.full((N, 2), np.nan)
    valid = np.zeros(N, dtype=bool)
    sigma_true = np.zeros((N, 2))
    sigma_assumed = np.zeros((N, 2))

    for k in range(N):
        if k > 0:
            b = rho_b * b + q_b * rng.normal()

        if (k % meas_decim) != 0:
            continue

        # dropout
        if rng.random() < CROSSLINK_P_DROPOUT and k != 0:
            continue

        r_rel = rD_true[k] - rC_true[k]
        v_rel = vD_true[k] - vC_true[k]
        s = la.norm(r_rel)
        if s < 1e-9:
            rr_true = 0.0
        else:
            rr_true = float(np.dot(r_rel, v_rel) / s)
        rng_true = float(s)

        sig_r = CROSSLINK_RANGE_SIGMA_M
        sig_rr = CROSSLINK_RR_SIGMA_MPS

        sigma_true[k, 0] = sig_r
        sigma_true[k, 1] = sig_rr
        sigma_assumed[k, 0] = CROSSLINK_SIGMA_ASSUMED_SCALE * sig_r
        sigma_assumed[k, 1] = CROSSLINK_SIGMA_ASSUMED_SCALE * sig_rr

        nr = rng.normal() * sig_r
        nrr = rng.normal() * sig_rr
        if rng.random() < CROSSLINK_P_OUTLIER:
            nr *= CROSSLINK_OUTLIER_SCALE
            nrr *= CROSSLINK_OUTLIER_SCALE

        z[k, 0] = rng_true + b + nr
        z[k, 1] = rr_true + nrr
        valid[k] = True

    return dict(z=z, valid=valid, sigma_true=sigma_true, sigma_assumed=sigma_assumed)


def xlink_pred_and_H_range(x_hill: np.ndarray) -> Tuple[float, np.ndarray]:
    x_hill = np.asarray(x_hill, float).reshape(6)
    rho = x_hill[0:3]
    s = la.norm(rho)
    u = np.zeros(3) if s < 1e-9 else (rho / s)
    H = np.zeros((1, 6))
    H[0, 0:3] = u
    return float(s), H


def xlink_pred_and_H_rangerate(x_hill: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_hill = np.asarray(x_hill, float).reshape(6)
    rho = x_hill[0:3]
    rhod = x_hill[3:6]
    s = la.norm(rho)
    if s < 1e-9:
        u = np.zeros(3)
        rr = 0.0
        grad_r = np.zeros(3)
    else:
        u = rho / s
        rr = float(np.dot(u, rhod))
        I3 = np.eye(3)
        grad_r = (I3 - np.outer(u, u)) @ rhod / s

    zhat = np.array([float(s), rr], dtype=float)
    H = np.zeros((2, 6))
    H[0, 0:3] = u
    H[1, 0:3] = grad_r
    H[1, 3:6] = u
    return zhat, H


def projected_var(P_pos_eci: np.ndarray, u_eci: np.ndarray) -> float:
    P_pos_eci = np.asarray(P_pos_eci, float).reshape(3, 3)
    u_eci = np.asarray(u_eci, float).reshape(3)
    return float(u_eci.T @ P_pos_eci @ u_eci)


# ──────────────────────────────────────────────────────────────────────────────
#                           plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_cov_ellipsoid(ax, mean: np.ndarray, cov: np.ndarray, n_std: float = 3.0, n_pts: int = 12):
    mean = np.asarray(mean, float).reshape(3)
    cov = np.asarray(cov, float).reshape(3, 3)

    w, V = la.eigh(cov)
    w = np.maximum(w, 0.0)
    radii = n_std * np.sqrt(w)

    u = np.linspace(0.0, 2.0 * np.pi, n_pts)
    v = np.linspace(0.0, np.pi, n_pts)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack([xs, ys, zs], axis=0).reshape(3, -1)
    ell = (V @ (radii.reshape(3, 1) * sphere)).reshape(3, xs.shape[0], xs.shape[1])

    X = ell[0] + mean[0]
    Y = ell[1] + mean[1]
    Z = ell[2] + mean[2]
    ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, linewidth=0.4)


def shade_contacts(ax, t_hr: np.ndarray, contact_mask: np.ndarray, label: Optional[str] = None):
    intervals = intervals_from_mask(t_hr, contact_mask)
    for i, (a, b) in enumerate(intervals):
        ax.axvspan(a, b, alpha=0.08, label=label if (label is not None and i == 0) else None)


# ──────────────────────────────────────────────────────────────────────────────
#                           CHIEF OD (absolute EKF)
# ──────────────────────────────────────────────────────────────────────────────

def run_chief_od_from_bundle(bundle_path: Path) -> Dict[str, np.ndarray]:
    data = load_bundle(bundle_path)

    t_full = np.asarray(data["t"], float).reshape(-1)
    rC_full = np.asarray(data["rC_N"], float).reshape(-1, 3)
    vC_full = np.asarray(data["vC_N"], float).reshape(-1, 3)
    impulses = np.asarray(data["impulses"], float).reshape(-1, 4)

    # crop after deploy
    t0 = float(t_full[0])
    t_crop_start = t0 + CROP_AFTER_DEPLOY_HR * 3600.0
    idx_crop = int(np.searchsorted(t_full, t_crop_start))
    t_crop = t_full[idx_crop:]

    # decimate to nav grid
    decim = decimate_indices(t_full, NAV_RATE_HZ, NAV_DECIM_OVERRIDE)
    idx_decim = np.arange(0, len(t_crop), decim)
    idx_sel = idx_crop + idx_decim

    t = t_full[idx_sel]
    r_true = rC_full[idx_sel]
    v_true = vC_full[idx_sel]

    # burns mapped to time grid (chief impulses usually none, but keep)
    dt_truth = float(np.median(np.diff(t_full))) if len(t_full) > 1 else 1.0
    dv_full = build_burn_sequence(impulses, t_full, dt_truth)
    dv_seq = dv_full[idx_sel]  # (N,3) in Hill in your bundle; we won't apply to ECI chief EKF here.

    nav_dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    nav_rate_hz = 1.0 / nav_dt

    # simulate measurements
    ground = simulate_ground_tracking(
        rng=np.random.default_rng(100),
        t_arr=t,
        r_true=r_true,
        v_true=v_true,
        stations=GROUND_STATIONS,
        nav_rate_hz=nav_rate_hz,
    )

    gnss = simulate_gnss_eci(
        rng=np.random.default_rng(200),
        t_arr=t,
        r_true=r_true,
        v_true=v_true,
        nav_rate_hz=nav_rate_hz,
        meas_rate_hz=CHIEF_GNSS_RATE_HZ,
        avail_prob=CHIEF_GNSS_AVAIL_PROB,
        avail_tau_min=CHIEF_GNSS_AVAIL_TAU_MIN,
        sigma_pos=CHIEF_GNSS_SIGMA_POS_M,
        sigma_vel=CHIEF_GNSS_SIGMA_VEL_MPS,
        sigma_assumed_scale=CHIEF_GNSS_SIGMA_ASSUMED_SCALE,
    )

    # initial conditions
    x_true0 = np.hstack([r_true[0], v_true[0]])
    x = x_true0 + CHIEF_INIT_BIAS

    P = np.diag([
        CHIEF_INIT_POS_SIGMA_M**2, CHIEF_INIT_POS_SIGMA_M**2, CHIEF_INIT_POS_SIGMA_M**2,
        CHIEF_INIT_VEL_SIGMA_MPS**2, CHIEF_INIT_VEL_SIGMA_MPS**2, CHIEF_INIT_VEL_SIGMA_MPS**2,
    ])

    N = len(t)
    x_hist = np.zeros((N, 6))
    P_hist = np.zeros((N, 6, 6))

    # innovation logs (per station)
    y_range = {st.name: np.full(N, np.nan) for st in GROUND_STATIONS}
    y_dopp  = {st.name: np.full(N, np.nan) for st in GROUND_STATIONS}
    contact_any = np.zeros(N, dtype=bool)

    # GNSS innovation (norms)
    y_gnss_pos = np.full(N, np.nan)
    y_gnss_vel = np.full(N, np.nan)

    x_hist[0] = x
    P_hist[0] = P

    for k in range(1, N):
        dt = float(t[k] - t[k - 1])

        # predict
        x_pred, P_pred = chief_predict(x, P, dt)

        # (optional) GNSS update
        if CHIEF_MODE in ("gnss", "ground+gnss") and gnss["valid"][k]:
            z = gnss["z"][k]
            # H = I
            H = np.eye(6)
            zhat = x_pred.copy()

            sig = gnss["sigma_assumed"][k]
            R = np.diag(sig**2)

            x_upd, P_upd, y, S = ekf_update_generic(x_pred, P_pred, z, zhat, H, R, use_joseph=True)
            y_gnss_pos[k] = la.norm(y[0:3])
            y_gnss_vel[k] = la.norm(y[3:6])
            x_pred, P_pred = x_upd, P_upd

        # ground updates (sequential for each station that has a measurement)
        if CHIEF_MODE in ("ground", "ground+gnss"):
            for st in GROUND_STATIONS:
                g = ground[st.name]
                if g["valid"][k]:
                    contact_any[k] = True
                    z_meas = g["z"][k].copy()

                    # choose which components are used
                    use_rng = bool(GROUND_USE_RANGE)
                    use_dop = bool(GROUND_USE_DOPPLER)

                    zhat_full, H_full = ground_range_dopp_model(x_pred, st, float(t[k]))

                    # build measurement selection
                    if use_rng and use_dop:
                        z_use = z_meas
                        zhat = zhat_full
                        H = H_full
                        sig = g["sigma_assumed"][k]
                        R = np.diag(sig**2)

                        # NIS gate (2D)
                        if CHIEF_USE_NIS_GATING:
                            y_tmp = (z_use - zhat).reshape(-1, 1)
                            S_tmp = H @ P_pred @ H.T + R
                            if nis_value(y_tmp, S_tmp) > NIS_GATE_RANGE_DOPP_2D:
                                continue

                        x_upd, P_upd, y, S = ekf_update_generic(x_pred, P_pred, z_use, zhat, H, R, use_joseph=True)
                        y_range[st.name][k] = y[0]
                        y_dopp[st.name][k]  = y[1]
                        x_pred, P_pred = x_upd, P_upd

                    elif use_rng and (not use_dop):
                        z_use = np.array([z_meas[0]])
                        zhat = np.array([zhat_full[0]])
                        H = H_full[0:1, :]
                        sig = float(g["sigma_assumed"][k, 0])
                        R = np.array([[sig**2]])

                        if CHIEF_USE_NIS_GATING:
                            y_tmp = (z_use - zhat).reshape(-1, 1)
                            S_tmp = H @ P_pred @ H.T + R
                            if nis_value(y_tmp, S_tmp) > NIS_GATE_RANGE:
                                continue

                        x_upd, P_upd, y, S = ekf_update_generic(x_pred, P_pred, z_use, zhat, H, R, use_joseph=True)
                        y_range[st.name][k] = y[0]
                        x_pred, P_pred = x_upd, P_upd

                    elif (not use_rng) and use_dop:
                        z_use = np.array([z_meas[1]])
                        zhat = np.array([zhat_full[1]])
                        H = H_full[1:2, :]
                        sig = float(g["sigma_assumed"][k, 1])
                        R = np.array([[sig**2]])

                        if CHIEF_USE_NIS_GATING:
                            y_tmp = (z_use - zhat).reshape(-1, 1)
                            S_tmp = H @ P_pred @ H.T + R
                            if nis_value(y_tmp, S_tmp) > NIS_GATE_RANGEDOT:
                                continue

                        x_upd, P_upd, y, S = ekf_update_generic(x_pred, P_pred, z_use, zhat, H, R, use_joseph=True)
                        y_dopp[st.name][k] = y[0]
                        x_pred, P_pred = x_upd, P_upd

        # commit
        x, P = x_pred, P_pred
        x_hist[k] = x
        P_hist[k] = P

    err = x_hist - np.hstack([r_true, v_true])

    return dict(
        t=t,
        t_hr=(t - t[0]) / 3600.0,
        x_true=np.hstack([r_true, v_true]),
        x_est=x_hist,
        P=P_hist,
        err=err,
        ground=ground,
        y_range=y_range,
        y_dopp=y_dopp,
        contact_any=contact_any,
        gnss=gnss,
        y_gnss_pos=y_gnss_pos,
        y_gnss_vel=y_gnss_vel,
        nav_rate_hz=(1.0 / float(np.median(np.diff(t))) if len(t) > 1 else 1.0),
    )


# ──────────────────────────────────────────────────────────────────────────────
#                           Deputy relative EKF (Hill)
# ──────────────────────────────────────────────────────────────────────────────

def run_deputy_relative(
    dep_idx: int,
    bundle_path: Path,
    chief_est_fn,          # fn(t)->(rC_hat, vC_hat)
    chief_P_fn=None,       # fn(t)->P_C (6,6) optional
) -> Dict[str, np.ndarray]:

    data = load_bundle(bundle_path)

    t_full  = np.asarray(data["t"], float).reshape(-1)
    rD_full = np.asarray(data["rD_N"], float).reshape(-1, 3)
    vD_full = np.asarray(data["vD_N"], float).reshape(-1, 3)
    rC_full = np.asarray(data["rC_N"], float).reshape(-1, 3)
    vC_full = np.asarray(data["vC_N"], float).reshape(-1, 3)
    impulses = np.asarray(data["impulses"], float).reshape(-1, 4)

    # crop
    t0 = float(t_full[0])
    t_crop_start = t0 + CROP_AFTER_DEPLOY_HR * 3600.0
    idx_crop = int(np.searchsorted(t_full, t_crop_start))
    t_crop = t_full[idx_crop:]

    # decimate
    decim = decimate_indices(t_full, NAV_RATE_HZ, NAV_DECIM_OVERRIDE)
    idx_decim = np.arange(0, len(t_crop), decim)
    idx_sel = idx_crop + idx_decim

    t = t_full[idx_sel]
    rD = rD_full[idx_sel]
    vD = vD_full[idx_sel]
    rC = rC_full[idx_sel]
    vC = vC_full[idx_sel]

    dt_truth = float(np.median(np.diff(t_full))) if len(t_full) > 1 else 1.0
    dv_full = build_burn_sequence(impulses, t_full, dt_truth)
    dv_seq_H = dv_full[idx_sel]   # impulses are in Hill in your bundle

    x_true = eci_to_hill_traj(rD, vD, rC, vC)

    nav_dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    nav_rate_hz = 1.0 / nav_dt

    # simulate measurements
    # crosslink always simulated; mode decides if we use it
    xlink = simulate_crosslink(
        rng=np.random.default_rng(1000 + dep_idx),
        t_arr=t,
        rD_true=rD,
        vD_true=vD,
        rC_true=rC,
        vC_true=vC,
        nav_rate_hz=nav_rate_hz,
    )

    # gps-like relative measurement
    gps_rel = simulate_gnss_hill_rel(
        rng=np.random.default_rng(2000 + dep_idx),
        t_arr=t,
        x_true_hill=x_true,
        nav_rate_hz=nav_rate_hz,
        meas_rate_hz=DEP_GNSS_RATE_HZ,
        avail_prob=DEP_GNSS_AVAIL_PROB,
        avail_tau_min=DEP_GNSS_AVAIL_TAU_MIN,
        sigma_pos=DEP_GNSS_SIGMA_POS_M,
        sigma_vel=DEP_GNSS_SIGMA_VEL_MPS,
        sigma_assumed_scale=DEP_GNSS_SIGMA_ASSUMED_SCALE,
    )

    # HillEKF config (prop only)
    cfg = HillEKFConfig(
        dt=nav_dt,
        sigma_proc_pos=DEPUTY_SIGMA_PROC_POS_M,
        sigma_proc_vel=DEPUTY_SIGMA_PROC_VEL_MPS,
        sigma_meas_pos=DEP_GNSS_SIGMA_POS_M,
        sigma_meas_vel=DEP_GNSS_SIGMA_VEL_MPS,
        sigma_init_pos=DEPUTY_SIGMA_INIT_POS_M,
        sigma_init_vel=DEPUTY_SIGMA_INIT_VEL_MPS,
        sigma_dv=DEPUTY_SIGMA_DV_MPS,
        use_joseph=True,
    )

    ekf = HillEKF(cfg, chief_state_fn=chief_est_fn)

    # init
    x0 = x_true[0] + DEPUTY_INIT_BIAS
    P0 = np.diag([
        DEPUTY_SIGMA_INIT_POS_M**2, DEPUTY_SIGMA_INIT_POS_M**2, DEPUTY_SIGMA_INIT_POS_M**2,
        DEPUTY_SIGMA_INIT_VEL_MPS**2, DEPUTY_SIGMA_INIT_VEL_MPS**2, DEPUTY_SIGMA_INIT_VEL_MPS**2,
    ])
    ekf.reset(x0, P0)

    N = len(t)
    x_est = np.zeros((N, 6))
    P_hist = np.zeros((N, 6, 6))

    y_xlink_r = np.full(N, np.nan)
    y_xlink_rr = np.full((N, 2), np.nan)

    y_gps_pos = np.full(N, np.nan)
    y_gps_vel = np.full(N, np.nan)

    # init pred containers
    ekf.x_pred = ekf.x.copy()
    ekf.P_pred = ekf.P.copy()
    ekf.t_last = float(t[0])

    for k in range(N):
        tk = float(t[k])

        if k > 0:
            ekf.predict(tk, burn_dv_H=dv_seq_H[k])
        else:
            ekf.x_pred = ekf.x.copy()
            ekf.P_pred = ekf.P.copy()

        # --- GPS-like update (direct state)
        if "gps" in DEPUTY_MODE and gps_rel["valid"][k]:
            z = gps_rel["z"][k]
            zhat = ekf.x_pred.copy()
            H = np.eye(6)
            sig = gps_rel["sigma_assumed"][k]
            R = np.diag(sig**2)

            # update in-place using HillEKF containers
            x_upd, P_upd, y, S = ekf_update_generic(ekf.x_pred, ekf.P_pred, z, zhat, H, R, use_joseph=True)
            ekf.x = x_upd
            ekf.P = P_upd
            y_gps_pos[k] = la.norm(y[0:3])
            y_gps_vel[k] = la.norm(y[3:6])

        else:
            ekf.x = ekf.x_pred.copy()
            ekf.P = ekf.P_pred.copy()

        # prepare for subsequent measurement updates
        ekf.x_pred = ekf.x.copy()
        ekf.P_pred = ekf.P.copy()

        # --- Crosslink range-only update
        if DEPUTY_MODE in ("r", "gps+r") and xlink["valid"][k]:
            z = np.array([xlink["z"][k, 0]])
            zhat_r, H = xlink_pred_and_H_range(ekf.x_pred)
            zhat = np.array([zhat_r])

            # base measurement variance
            sig_r = float(xlink["sigma_assumed"][k, 0])
            Rvar = (sig_r**2)

            # optionally inflate with chief OD covariance projected onto LOS
            if INCLUDE_CHIEF_UNCERT_IN_CROSSLINK_R and (chief_P_fn is not None):
                # LOS direction in Hill and then ECI
                rho = ekf.x_pred[0:3]
                s = la.norm(rho)
                if s > 1e-9:
                    u_h = rho / s
                    rC_hat, vC_hat = chief_est_fn(tk)
                    C_HN = dcm_lvlh(rC_hat, vC_hat)
                    u_eci = C_HN.T @ u_h
                    P_C = chief_P_fn(tk)
                    P_C_pos = P_C[0:3, 0:3]
                    Rvar += projected_var(P_C_pos, u_eci)

            R = np.array([[Rvar]])

            # NIS gate
            if DEPUTY_USE_NIS_GATING:
                y_tmp = (z - zhat).reshape(-1, 1)
                S_tmp = H @ ekf.P_pred @ H.T + R
                if nis_value(y_tmp, S_tmp) <= NIS_GATE_XLINK_R:
                    x_upd, P_upd, y, S = ekf_update_generic(ekf.x_pred, ekf.P_pred, z, zhat, H, R, use_joseph=True)
                    ekf.x = x_upd
                    ekf.P = P_upd
                    y_xlink_r[k] = y[0]
            else:
                x_upd, P_upd, y, S = ekf_update_generic(ekf.x_pred, ekf.P_pred, z, zhat, H, R, use_joseph=True)
                ekf.x = x_upd
                ekf.P = P_upd
                y_xlink_r[k] = y[0]

        # --- Crosslink range + range-rate update
        if DEPUTY_MODE in ("rr", "gps+rr") and xlink["valid"][k]:
            z = xlink["z"][k].copy()
            zhat, H = xlink_pred_and_H_rangerate(ekf.x_pred)

            sig = xlink["sigma_assumed"][k].copy()
            R = np.diag(sig**2)

            # inflate if chief uncertainty is enabled
            if INCLUDE_CHIEF_UNCERT_IN_CROSSLINK_R and (chief_P_fn is not None):
                rho = ekf.x_pred[0:3]
                s = la.norm(rho)
                if s > 1e-9:
                    u_h = rho / s
                    rC_hat, vC_hat = chief_est_fn(tk)
                    C_HN = dcm_lvlh(rC_hat, vC_hat)
                    u_eci = C_HN.T @ u_h
                    P_C = chief_P_fn(tk)
                    P_C_pos = P_C[0:3, 0:3]
                    P_C_vel = P_C[3:6, 3:6]

                    R[0, 0] += projected_var(P_C_pos, u_eci)
                    R[1, 1] += projected_var(P_C_vel, u_eci)

            # NIS gate (2D)
            if DEPUTY_USE_NIS_GATING:
                y_tmp = (z - zhat).reshape(-1, 1)
                S_tmp = H @ ekf.P_pred @ H.T + R
                if nis_value(y_tmp, S_tmp) <= NIS_GATE_XLINK_RR:
                    x_upd, P_upd, y, S = ekf_update_generic(ekf.x_pred, ekf.P_pred, z, zhat, H, R, use_joseph=True)
                    ekf.x = x_upd
                    ekf.P = P_upd
                    y_xlink_rr[k, :] = y
            else:
                x_upd, P_upd, y, S = ekf_update_generic(ekf.x_pred, ekf.P_pred, z, zhat, H, R, use_joseph=True)
                ekf.x = x_upd
                ekf.P = P_upd
                y_xlink_rr[k, :] = y

        # store
        x_est[k] = ekf.x
        P_hist[k] = ekf.P

    err = x_est - x_true
    t_hr = (t - t[0]) / 3600.0
    mask = t_hr >= 0.1
    rms_pos = float(np.sqrt(np.mean(np.sum(err[mask, 0:3]**2, axis=1)))) if np.any(mask) else float("nan")

    return dict(
        dep_idx=dep_idx,
        t=t,
        t_hr=t_hr,
        x_true=x_true,
        x_est=x_est,
        P=P_hist,
        err=err,
        xlink=xlink,
        gps=gps_rel,
        y_xlink_r=y_xlink_r,
        y_xlink_rr=y_xlink_rr,
        y_gps_pos=y_gps_pos,
        y_gps_vel=y_gps_vel,
        rms_pos=np.array([rms_pos], dtype=float),
    )


# ──────────────────────────────────────────────────────────────────────────────
#                                     main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # sanity check
    dep0_path = DATA_DIR / BUNDLE_TEMPLATE.format(idx=0)
    if not dep0_path.is_file():
        raise FileNotFoundError(
            f"Expected bundle not found: {dep0_path}\n"
            f"Set DATA_DIR/BUNDLE_TEMPLATE to point at your GEO truth bundles."
        )

    # 1) Chief OD (run once)
    chief = run_chief_od_from_bundle(dep0_path)

    t_chief = chief["t"]
    xC_est = chief["x_est"]
    PC = chief["P"]

    # build interpolators for chief estimated state and covariance (for deputies)
    rC_fn = make_linear_interp_fn(t_chief, xC_est[:, 0:3])
    vC_fn = make_linear_interp_fn(t_chief, xC_est[:, 3:6])
    PC_fn = make_linear_interp_fn(t_chief, PC)

    def chief_est_fn(t: float) -> Tuple[np.ndarray, np.ndarray]:
        return np.asarray(rC_fn(t), float).reshape(3), np.asarray(vC_fn(t), float).reshape(3)

    def chief_P_fn(t: float) -> np.ndarray:
        return np.asarray(PC_fn(t), float).reshape(6, 6)

    # 2) Deputies relative nav
    deputies: List[Dict[str, np.ndarray]] = []
    for d in range(N_DEP):
        path = DATA_DIR / BUNDLE_TEMPLATE.format(idx=d)
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundle: {path}")

        print(f"\n----- Deputy D{d} relative nav (MODE={DEPUTY_MODE}) -----")
        res = run_deputy_relative(d, path, chief_est_fn, chief_P_fn)
        deputies.append(res)
        print(f"D{d} RMS ||pos err|| (Hill) ≈ {float(res['rms_pos'][0]):.3f} m")

    # 3) PLOTS
    if PLOT_CHIEF:
        t_hr = chief["t_hr"]
        err = chief["err"]
        P = chief["P"]

        mask = t_hr >= 0.1
        tt = t_hr[mask]

        Pdiag = np.diagonal(P[mask], axis1=1, axis2=2)
        three_sigma = 3.0 * np.sqrt(np.maximum(Pdiag, 0.0))

        # Chief position errors vs 3σ (scatter)
        figC1, axC1 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        labels = ["x", "y", "z"]
        for i in range(3):
            axC1[i].scatter(tt, err[mask, i], s=6, alpha=0.45)
            axC1[i].plot(tt,  three_sigma[:, i], linestyle="--", linewidth=1.0)
            axC1[i].plot(tt, -three_sigma[:, i], linestyle="--", linewidth=1.0)
            axC1[i].set_ylabel(f"{labels[i]} [m]")
            axC1[i].grid(True)

        axC1[-1].set_xlabel("Time since OD start [h]")
        figC1.suptitle(f"Chief OD: position error vs 3σ (CHIEF_MODE={CHIEF_MODE})")
        figC1.tight_layout(rect=[0, 0, 1, 0.95])

        # Chief velocity errors vs 3σ (scatter)
        figC2, axC2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for i in range(3):
            axC2[i].scatter(tt, err[mask, 3 + i], s=6, alpha=0.45)
            axC2[i].plot(tt,  three_sigma[:, 3 + i], linestyle="--", linewidth=1.0)
            axC2[i].plot(tt, -three_sigma[:, 3 + i], linestyle="--", linewidth=1.0)
            axC2[i].set_ylabel(f"{labels[i]} [m/s]")
            axC2[i].grid(True)

        axC2[-1].set_xlabel("Time since OD start [h]")
        figC2.suptitle(f"Chief OD: velocity error vs 3σ (CHIEF_MODE={CHIEF_MODE})")
        figC2.tight_layout(rect=[0, 0, 1, 0.95])

        # Ground measurement innovations (range + doppler), with contact shading
        figC3, axC3 = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        figC3.suptitle("Chief OD: ground tracking innovations (pre-fit)")

        # contact shading for any station scheduled contact (union)
        # (we can approximate by shading if ANY station had scheduled contact)
        # Here we build union of scheduled arcs (same schedule), but also you can use actual contact if desired.
        arc_union = np.zeros_like(chief["contact_any"])
        for st in GROUND_STATIONS:
            arc_union |= chief["ground"][st.name]["arc_sched"]

        shade_contacts(axC3[0], t_hr, arc_union, label="scheduled arcs")
        shade_contacts(axC3[1], t_hr, arc_union)

        # plot each station innovations
        for st in GROUND_STATIONS:
            yR = chief["y_range"][st.name]
            yD = chief["y_dopp"][st.name]
            axC3[0].scatter(t_hr, yR, s=6, alpha=0.45, label=f"{st.name} range")
            axC3[1].scatter(t_hr, yD, s=6, alpha=0.45, label=f"{st.name} doppler")

        axC3[0].axhline( 3.0 * GROUND_RANGE_SIGMA_M, linestyle="--", linewidth=1.0)
        axC3[0].axhline(-3.0 * GROUND_RANGE_SIGMA_M, linestyle="--", linewidth=1.0)
        axC3[0].set_ylabel("Range innov [m]")
        axC3[0].grid(True)
        axC3[0].legend(ncol=2, fontsize=9)

        axC3[1].axhline( 3.0 * GROUND_DOPP_SIGMA_MPS, linestyle="--", linewidth=1.0)
        axC3[1].axhline(-3.0 * GROUND_DOPP_SIGMA_MPS, linestyle="--", linewidth=1.0)
        axC3[1].set_ylabel("Doppler innov [m/s]")
        axC3[1].set_xlabel("Time since OD start [h]")
        axC3[1].grid(True)
        axC3[1].legend(ncol=2, fontsize=9)

        figC3.tight_layout(rect=[0, 0, 1, 0.94])

    if PLOT_DEPUTIES:
        labels_h = ["R", "T", "N"]

        for res in deputies:
            d = int(res["dep_idx"])
            t_hr = res["t_hr"]
            x_true = res["x_true"]
            x_est = res["x_est"]
            P = res["P"]
            err = res["err"]

            mask = t_hr >= 0.1
            tt = t_hr[mask]

            Pdiag = np.diagonal(P[mask], axis1=1, axis2=2)
            three_sigma = 3.0 * np.sqrt(np.maximum(Pdiag, 0.0))

            # Fig 1: 3D Hill traj + ellipsoids
            fig1 = plt.figure(figsize=(10, 8))
            ax1 = fig1.add_subplot(111, projection="3d")
            ax1.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], linewidth=1.0, label="truth ρ")
            ax1.plot(x_est[:, 0],  x_est[:, 1],  x_est[:, 2],  linewidth=1.0, label="est ρ")

            if PLOT_ELLIPSOIDS and ELLIPSOID_EVERY > 0:
                for k in range(0, len(t_hr), ELLIPSOID_EVERY):
                    plot_cov_ellipsoid(ax1, x_est[k, 0:3], P[k, 0:3, 0:3],
                                       n_std=ELLIPSOID_NSTD, n_pts=ELLIPSOID_NPTS)

            ax1.set_xlabel("R [m]")
            ax1.set_ylabel("T [m]")
            ax1.set_zlabel("N [m]")
            ax1.set_title(f"Deputy D{d}: Hill trajectory + {ELLIPSOID_NSTD:.0f}σ ellipsoids (DEPUTY_MODE={DEPUTY_MODE})")
            ax1.grid(True)
            ax1.legend()

            # Fig 2: errors vs 3σ (scatter)
            fig2, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
            fig2.suptitle(f"Deputy D{d}: Errors vs 3σ (Hill frame)")

            for i in range(3):
                axp = axes[0, i]
                axp.scatter(tt, err[mask, i], s=6, alpha=0.45)
                axp.plot(tt,  three_sigma[:, i], linestyle="--", linewidth=1.0)
                axp.plot(tt, -three_sigma[:, i], linestyle="--", linewidth=1.0)
                axp.set_title(f"ρ_{labels_h[i]}")
                axp.set_ylabel("m")
                axp.grid(True)

            for i in range(3):
                axv = axes[1, i]
                axv.scatter(tt, err[mask, 3 + i], s=6, alpha=0.45)
                axv.plot(tt,  three_sigma[:, 3 + i], linestyle="--", linewidth=1.0)
                axv.plot(tt, -three_sigma[:, 3 + i], linestyle="--", linewidth=1.0)
                axv.set_title(f"ρ̇_{labels_h[i]}")
                axv.set_ylabel("m/s")
                axv.set_xlabel("Time since rel-nav start [h]")
                axv.grid(True)

            fig2.tight_layout(rect=[0, 0, 1, 0.94])

            # Fig 3: measurement innovations (only those relevant to mode)
            if DEPUTY_MODE in ("r", "gps+r", "rr", "gps+rr"):
                fig3, ax3 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                fig3.suptitle(f"Deputy D{d}: Crosslink innovations (accepted updates)")

                if DEPUTY_MODE in ("r", "gps+r"):
                    yR = res["y_xlink_r"]
                    ax3[0].scatter(tt, yR[mask], s=6, alpha=0.45)
                    ax3[0].axhline( 3.0 * CROSSLINK_RANGE_SIGMA_M, linestyle="--", linewidth=1.0)
                    ax3[0].axhline(-3.0 * CROSSLINK_RANGE_SIGMA_M, linestyle="--", linewidth=1.0)
                    ax3[0].set_ylabel("Range innov [m]")
                    ax3[0].grid(True)
                    ax3[1].axis("off")
                else:
                    yRR = res["y_xlink_rr"]
                    ax3[0].scatter(tt, yRR[mask, 0], s=6, alpha=0.45)
                    ax3[0].axhline( 3.0 * CROSSLINK_RANGE_SIGMA_M, linestyle="--", linewidth=1.0)
                    ax3[0].axhline(-3.0 * CROSSLINK_RANGE_SIGMA_M, linestyle="--", linewidth=1.0)
                    ax3[0].set_ylabel("Range innov [m]")
                    ax3[0].grid(True)

                    ax3[1].scatter(tt, yRR[mask, 1], s=6, alpha=0.45)
                    ax3[1].axhline( 3.0 * CROSSLINK_RR_SIGMA_MPS, linestyle="--", linewidth=1.0)
                    ax3[1].axhline(-3.0 * CROSSLINK_RR_SIGMA_MPS, linestyle="--", linewidth=1.0)
                    ax3[1].set_ylabel("Range-rate innov [m/s]")
                    ax3[1].set_xlabel("Time since rel-nav start [h]")
                    ax3[1].grid(True)

                fig3.tight_layout(rect=[0, 0, 1, 0.94])

    plt.show()


if __name__ == "__main__":
    main()

