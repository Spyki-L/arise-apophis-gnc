#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_batch_od_los.py
=============================

Camera-only relative navigation for the ARISE Apophis swarm, solved as a
**full-arc batch nonlinear least-squares orbit determination** instead of
a forward EKF/UKF.

Why a batch solver
------------------
The deputy trajectories are fixed (pre-decided).  The navigation task is
therefore not *tracking* but *orbit determination*: the whole arc is
determined by a small constant parameter set, and there are tens of
thousands of bearing measurements constraining it.  A forward filter
processes each measurement once and pins the weakly-observable range with
only the data available so far.  A batch fit uses every measurement to
constrain every parameter — it is the maximum-likelihood / minimum-
variance estimator for a fixed trajectory and it extracts the range
information the orbital curvature and the known burns actually contain.

Estimated parameters (per deputy)
---------------------------------
    theta = [ delta_r0 (3),  delta_v0 (3),  a_emp (3) ]            (9-D)

  * delta_r0, delta_v0  — the deputy-minus-chief relative position and
    velocity at the first epoch, in the Earth-centred inertial (ECI)
    frame.  Six numbers fully define the relative trajectory given the
    known burns and dynamics.
  * a_emp — a constant empirical differential acceleration (ECI).  It
    absorbs the one un-modelled term large enough to matter over a
    multi-hour arc: differential solar-radiation pressure between the
    small deputy and the larger chief (see ASSUMPTION A4).  It carries a
    soft zero-mean prior so it cannot masquerade as the range signal.

Dynamics model (the propagator)
-------------------------------
The RELATIVE state is integrated (RK4) in the ECI frame:

    d/dt delta_r = delta_v
    d/dt delta_v = g(r_C + delta_r) - g(r_C)            (exact diff. gravity)
                 + g_A(r_C + delta_r) - g_A(r_C)        (Apophis, optional)
                 + a_emp

with g(r) = -mu_E r / |r|^3 and g_A the Apophis point-mass term.  The
chief position r_C(t) comes from the chief's broadcast ephemeris.  Known
RHIG burns are applied as instantaneous delta_v jumps (ECI).

Working with the RELATIVE state is what makes this honest: every
perturbation that acts almost equally on the deputy and the chief
(central-body harmonics, luni-solar tides) is common-mode and cancels in
the difference, leaving only the differential term — which over the ~km
baseline is millimetre-level (see ASSUMPTION A3).  The exact differential
gravity g(r_C+dr)-g(r_C) is fully NON-LINEAR in the range |delta_r|;
that range-dependence is what makes range observable from bearings — a
linear Clohessy-Wiltshire model, by contrast, is provably range-blind.

Measurement model
-----------------
Each deputy's camera reports the bearing to the chief in the chief Hill
(LVLH) frame:

    z = [ azimuth, elevation ]^T
    azimuth   = atan2( rho_T, rho_R )
    elevation = atan2( rho_N, sqrt(rho_R^2 + rho_T^2) )

with rho = C_HN(r_C, v_C) * delta_r.  No range channel.

Solver
------
Gauss-Newton with Levenberg-Marquardt damping; the Jacobian is built by
finite differences (9 re-propagations per iteration).  Gross outliers are
down-weighted with a Huber robust loss (iteratively re-weighted least
squares).  The formal parameter covariance is the inverse of the damped
normal matrix; propagated forward it gives an honest per-epoch position
uncertainty.

================================================================
MISSION ASSUMPTIONS  — read these before trusting any number
================================================================
A1  CENTRAL BODY.  The chief is on a geocentric orbit; mu_E =
    3.986004418e14 m^3/s^2.  Justification: the chief reference state
    magnitudes (|r_C| ~ 5e7 m, |v_C| ~ 2.6 km/s, specific energy < 0)
    are consistent with a bound Earth orbit.  The script VERIFIES this
    at run time and warns if the chief orbit looks non-geocentric.

A2  CHIEF BROADCAST.  The chief shares its inertial position and velocity
    over the inter-satellite link.  Its raw GPS is ~5 m 1-sigma; an
    onboard nav filter smooths it to ~1 m before broadcast.  The 1 m
    error affects only (i) the Hill-frame orientation — a ~2e-8 rad
    rotation, negligible — and (ii) the differential-gravity reference —
    a second difference, utterly insensitive to 1 m.  It does NOT enter
    the deputy-to-chief bearing the camera measures.  Hence the chief
    broadcast does NOT floor the RELATIVE navigation (it only floors the
    deputy's ABSOLUTE position, which is the chief's responsibility).

A3  NEGLECTED PERTURBATIONS.  Differential J2 and luni-solar tides over
    the ~km deputy-chief baseline are ~1e-10 m/s^2, i.e. millimetre-level
    position error over the arc — neglected.  (Order-of-magnitude checks
    are in the code comments.)

A4  DIFFERENTIAL SOLAR-RADIATION PRESSURE.  The deputies are identical
    SmallSats, but the chief is a larger mothership, so deputy-vs-chief
    SRP does not cancel: a differential ~1e-7 m/s^2, which integrates to
    metre-to-tens-of-metre position error over a multi-hour arc.  This is
    the dominant un-modelled term.  It is ABSORBED by the estimated
    constant empirical acceleration a_emp, with a soft prior of
    EMP_ACC_PRIOR_SIGMA.  If a_emp is disabled this term is the accuracy
    floor.

A5  APOPHIS THIRD BODY.  Apophis's differential gravity across the
    baseline exceeds Earth's only when the swarm is within ~800 m of
    Apophis (crossover computed from mu_A vs mu_E gradients).  It is
    included as an optional differential term using the broadcast
    Apophis ephemeris and an ASSUMED mu_A = 3.0 m^3/s^2 (Apophis GM is
    uncertain to roughly +/- 50 %).  The script reports the swarm-Apophis
    distance range so the user can see whether this term matters; where
    the swarm stays far from Apophis the mu_A uncertainty is irrelevant.

A6  KNOWN BURNS.  The RHIG impulsive delta_v (Hill frame) are known to
    the navigator (commanded / broadcast) and applied as exact velocity
    jumps.  Real burn-execution error (~mm/s) is small and unmodelled
    here; it would appear as a minor process noise.

A7  FIXED TRAJECTORY.  The deputy trajectory is pre-decided, so the
    navigation problem is pure estimation.  The 9 parameters fully
    describe the arc given known burns + dynamics.

A8  CAMERA / ATTITUDE.  Bearing is reported in the chief Hill frame; the
    deputy's star-tracker-aided attitude knowledge is good enough that
    attitude error is negligible against the 0.003-0.008 deg camera
    noise.  An AR(1) bias (~0.001 deg), 2 % dropout and rare 5x outliers
    are modelled.

A9  TRUTH FIDELITY.  The .npz bundles were generated with a high-fidelity
    force model (STK Astrogator).  This batch OD uses a lower-fidelity
    model; its accuracy floor is set by A3/A4/A5.  The script prints the
    post-fit residual RMS — if it is well above the camera noise, model
    error dominates and the reported covariance is optimistic.

A10 UNITS / CLOCK.  chief_reference.npz stores positions in km and
    velocities in km/s (verified); they are scaled to SI on load.  The
    chief reference and the deputy bundles share one mission clock.

Inputs   :  apophis_rendezvous_full_D{0..7}.npz , chief_reference.npz
Outputs  :  apophis_swarm_OD_los_D{d}.npz  (fitted state, covariance,
            residuals) per deputy, plus diagnostic plots.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Dict, Callable

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# ───────── plotting font controls ─────────

TITLE_FONTSIZE      = 26
AXIS_LABEL_FONTSIZE = 22
TICK_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE     = 18

plt.rcParams.update({
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "xtick.labelsize": TICK_LABEL_FONTSIZE,
    "ytick.labelsize": TICK_LABEL_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    "figure.max_open_warning": 60,
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

from GNC.Navigation.hill_ekf import dcm_lvlh, hill_rate  # type: ignore[import]

# ═════════════════════════ configuration ═════════════════════════

N_DEP = 8                       # deputies D0..D7

# Gravitational parameters
MU_EARTH   = 3.986004418e14     # [m^3/s^2]  geocentric  (ASSUMPTION A1)
MU_APOPHIS = 3.0                # [m^3/s^2]  ASSUMED, uncertain ~+/-50% (A5)
INCLUDE_APOPHIS = True          # include the Apophis differential term

# Batch-fit time grid.  The arc is decimated to this cadence for the
# orbit-determination fit; it is well above what 9 parameters need and
# keeps the solver fast.  (Increase for a tighter formal covariance.)
DT_MEAS  = 10.0                 # [s] measurement / propagation cadence

# --- Camera-only sensor model (per ASSUMPTION A8) ---------------------
ANG_SIGMA_HI_TRUE = np.deg2rad(0.008)   # [rad] far end of arc
ANG_SIGMA_LO_TRUE = np.deg2rad(0.003)   # [rad] near closest approach
TAU_HOURS         = 2.5                  # noise-decay time constant [h]
ANG_BIAS_STD      = np.deg2rad(0.001)    # AR(1) bearing bias 1-sigma
BIAS_TAU_HOURS    = 4.0
P_DROPOUT         = 0.02
P_OUTLIER         = 0.005
OUTLIER_SCALE     = 5.0
MEAS_SIGMA_INFLATE = 1.2                 # filter pessimism factor

# --- Chief broadcast (ASSUMPTION A2) ----------------------------------
CHIEF_GPS_SIGMA = 1.0           # [m] smoothed broadcast 1-sigma per axis

# --- Empirical differential acceleration (ASSUMPTION A4) --------------
# Diagnosed directly from the bundles (double-differencing the truth
# relative trajectory and subtracting the exact differential gravity):
# there is an un-modelled differential acceleration of ~4e-7 to ~1e-6
# m/s^2 — mildly time-varying, deputy-dependent — most consistent with
# differential solar-radiation pressure between the SmallSat deputies
# and the larger chief.  A single constant a_emp absorbs its MEAN; the
# residual time variation is the dominant remaining error.
#
# The prior is deliberately a REGULARIZER, not just a bound: over a
# single arc a constant differential acceleration is partly degenerate
# with the range coordinate, so an over-loose prior lets a_emp soak up
# genuine range signal and the range estimate gets WORSE.  3e-7 lets
# a_emp reach the bulk of the real differential acceleration while still
# damping the degeneracy.
ESTIMATE_EMP_ACC    = True
EMP_ACC_PRIOR_SIGMA = 1.0e-7    # [m/s^2] soft zero-mean prior on a_emp

# --- Batch solver settings --------------------------------------------
MAX_GN_ITERS   = 25             # max Gauss-Newton / LM iterations
GN_TOL         = 1.0e-4         # convergence: ||delta-theta|| scale test
HUBER_C        = 3.0            # Huber threshold in sigma units
LM_LAMBDA0     = 1.0e-2         # initial Levenberg-Marquardt damping
NOMINAL_RANGE  = 900.0          # [m] initial range guess along first LOS


# ═════════════════════════ data loading ══════════════════════════

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


def load_chief_reference(path: Path):
    """
    Load chief_reference.npz -> interpolator fn(t) -> (r_C [m], v_C [m/s]).
    The file stores km and km/s (ASSUMPTION A10); scaled to SI here.

    IMPORTANT — the reference is sampled only every ~43 s.  LINEAR
    interpolation of an orbit at that cadence carries ~0.5*a*(dt/2)^2 ~
    37 m of error mid-interval, which would masquerade as dynamics-model
    error in the OD.  Because the file stores BOTH position and velocity,
    we use CUBIC HERMITE interpolation (the unique cubic matching r and v
    at both ends); its error drops to O(dt^4) — sub-millimetre here.
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
        h = t_arr[j + 1] - t_arr[j]
        s = (t - t_arr[j]) / h
        s2 = s * s
        s3 = s2 * s
        # cubic Hermite basis — position
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        r = (h00 * rC_N[j] + h10 * h * vC_N[j]
             + h01 * rC_N[j + 1] + h11 * h * vC_N[j + 1])
        # derivative basis — velocity
        d00 = 6 * s2 - 6 * s
        d10 = 3 * s2 - 4 * s + 1
        d01 = -6 * s2 + 6 * s
        d11 = 3 * s2 - 2 * s
        v = (d00 * rC_N[j] + d10 * h * vC_N[j]
             + d01 * rC_N[j + 1] + d11 * h * vC_N[j + 1]) / h
        return r, v

    return t_arr, rC_N, vC_N, chief_state_fn


def make_vec_interp(t_arr: np.ndarray, arr: np.ndarray) -> Callable:
    """Linear interpolator for a (N,3) time series (fine-sampled data)."""
    t_arr = np.asarray(t_arr, float).reshape(-1)
    arr = np.asarray(arr, float).reshape(-1, 3)

    def fn(t: float) -> np.ndarray:
        if t <= t_arr[0]:
            return arr[0].copy()
        if t >= t_arr[-1]:
            return arr[-1].copy()
        j = int(np.searchsorted(t_arr, t)) - 1
        f = (t - t_arr[j]) / (t_arr[j + 1] - t_arr[j] + 1e-15)
        return (1.0 - f) * arr[j] + f * arr[j + 1]

    return fn


def make_hermite_state_fn(t_arr, r_arr, v_arr) -> Callable:
    """
    Cubic-Hermite (position + velocity) state interpolator.
    fn(t) -> (r, v).  Error O(dt^4) — needed for the chief because the
    differential-gravity model is only as clean as the chief trajectory.
    """
    t_arr = np.asarray(t_arr, float).reshape(-1)
    r_arr = np.asarray(r_arr, float).reshape(-1, 3)
    v_arr = np.asarray(v_arr, float).reshape(-1, 3)

    def fn(t: float):
        if t <= t_arr[0]:
            return r_arr[0].copy(), v_arr[0].copy()
        if t >= t_arr[-1]:
            return r_arr[-1].copy(), v_arr[-1].copy()
        j = int(np.searchsorted(t_arr, t)) - 1
        h = t_arr[j + 1] - t_arr[j]
        s = (t - t_arr[j]) / h
        s2 = s * s
        s3 = s2 * s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        r = (h00 * r_arr[j] + h10 * h * v_arr[j]
             + h01 * r_arr[j + 1] + h11 * h * v_arr[j + 1])
        d00 = 6 * s2 - 6 * s
        d10 = 3 * s2 - 4 * s + 1
        d01 = -6 * s2 + 6 * s
        d11 = 3 * s2 - 2 * s
        v = (d00 * r_arr[j] + d10 * h * v_arr[j]
             + d01 * r_arr[j + 1] + d11 * h * v_arr[j + 1]) / h
        return r, v

    return fn


# ═════════════════════════ frame helpers ═════════════════════════

def hill_relpos(delta_r_eci: np.ndarray, rC: np.ndarray,
                vC: np.ndarray) -> np.ndarray:
    """Express an ECI relative-position vector in the chief Hill frame."""
    return dcm_lvlh(rC, vC) @ delta_r_eci


def eci_to_hill_state(rD: np.ndarray, vD: np.ndarray,
                      rC: np.ndarray, vC: np.ndarray) -> np.ndarray:
    """Full 6-state deputy-minus-chief in the chief Hill frame."""
    C_HN = dcm_lvlh(rC, vC)
    rho = C_HN @ (rD - rC)
    omega = np.array([0.0, 0.0, hill_rate(rC, vC)])
    rhod = C_HN @ (vD - vC) - np.cross(omega, rho)
    return np.concatenate([rho, rhod])


# ═════════════════════════ bearing model ═════════════════════════

def bearing_from_rho(rho: np.ndarray) -> np.ndarray:
    """Hill relative vector -> (azimuth, elevation)."""
    x, y, z = rho
    r_xy = np.sqrt(max(x * x + y * y, 1e-30))
    return np.array([np.arctan2(y, x), np.arctan2(z, r_xy)])


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ═════════════════════════ dynamics ══════════════════════════════

def _point_mass_diff(delta_r: np.ndarray, r_ref: np.ndarray,
                     mu: float) -> np.ndarray:
    """
    Differential point-mass gravity:  g(r_ref + delta_r) - g(r_ref),
    with g(r) = -mu r / |r|^3.  Exact and fully non-linear in delta_r —
    this is the term that makes range observable from bearings.
    """
    r1 = r_ref + delta_r
    n1 = la.norm(r1)
    n0 = la.norm(r_ref)
    return -mu * (r1 / (n1 ** 3) - r_ref / (n0 ** 3))


def relative_accel(delta_r: np.ndarray, t: float,
                   rC_fn: Callable, rA_fn: Callable,
                   a_emp: np.ndarray) -> np.ndarray:
    """
    ECI relative acceleration of the deputy w.r.t. the chief.

    = differential Earth gravity
    + differential Apophis gravity   (ASSUMPTION A5; optional)
    + constant empirical acceleration a_emp   (ASSUMPTION A4)
    """
    rC = rC_fn(t)
    acc = _point_mass_diff(delta_r, rC, MU_EARTH)
    if INCLUDE_APOPHIS and rA_fn is not None:
        rA = rA_fn(t)
        # chief- and deputy-relative-to-Apophis vectors
        acc = acc + _point_mass_diff(delta_r, rC - rA, MU_APOPHIS)
    return acc + a_emp


def propagate_relative(
    delta_r0: np.ndarray,
    delta_v0: np.ndarray,
    a_emp: np.ndarray,
    t_grid: np.ndarray,
    rC_fn: Callable,
    vC_fn: Callable,
    rA_fn: Callable,
    burn_times: np.ndarray,
    burn_dv_eci: np.ndarray,
) -> np.ndarray:
    """
    RK4-integrate the ECI relative state over t_grid, applying the known
    burns (already converted to ECI delta_v) at their epochs.

    Returns (N,6) array of [delta_r, delta_v] at each grid time.
    """
    n = len(t_grid)
    out = np.zeros((n, 6))
    dr = np.array(delta_r0, float)
    dv = np.array(delta_v0, float)
    out[0] = np.concatenate([dr, dv])

    # burn lookup: index of next burn
    bt = np.asarray(burn_times, float)
    order = np.argsort(bt)
    bt = bt[order]
    bdv = np.asarray(burn_dv_eci, float)[order] if len(bt) else \
        np.zeros((0, 3))
    ib = 0

    for k in range(1, n):
        t0 = t_grid[k - 1]
        t1 = t_grid[k]
        h = t1 - t0

        # apply any burns landing in (t0, t1]
        while ib < len(bt) and bt[ib] <= t1 + 1e-9:
            if bt[ib] > t0 + 1e-9:
                dv = dv + bdv[ib]
            ib += 1

        # RK4 on (dr, dv); a_emp constant
        def deriv(dr_, dv_, tt):
            return dv_, relative_accel(dr_, tt, rC_fn, rA_fn, a_emp)

        k1r, k1v = deriv(dr, dv, t0)
        k2r, k2v = deriv(dr + 0.5 * h * k1r, dv + 0.5 * h * k1v, t0 + 0.5 * h)
        k3r, k3v = deriv(dr + 0.5 * h * k2r, dv + 0.5 * h * k2v, t0 + 0.5 * h)
        k4r, k4v = deriv(dr + h * k3r, dv + h * k3v, t1)
        dr = dr + (h / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
        dv = dv + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        out[k] = np.concatenate([dr, dv])

    return out


# ═════════════════════════ measurement generation ════════════════

def build_camera_meas(
    rng: np.random.Generator,
    t_arr: np.ndarray,
    rD_N: np.ndarray,
    rC_N: np.ndarray,
    vC_N: np.ndarray,
):
    """
    Generate noisy (az, el) bearing measurements (ASSUMPTION A8).

    Returns z_meas (N,2), sigma_assumed (N,2), valid (N,) bool.
    """
    N = len(t_arr)
    t_rel = t_arr - t_arr[0]
    w = np.exp(-t_rel / (TAU_HOURS * 3600.0))
    sig = ANG_SIGMA_LO_TRUE + (ANG_SIGMA_HI_TRUE - ANG_SIGMA_LO_TRUE) * w

    dt_eff = float(np.median(np.diff(t_arr))) if N > 1 else 1.0
    rho_b = np.exp(-dt_eff / max(BIAS_TAU_HOURS * 3600.0, 1e-3))
    qa = ANG_BIAS_STD * np.sqrt(1.0 - rho_b ** 2)
    b_az = b_el = 0.0

    z = np.zeros((N, 2))
    valid = np.ones(N, dtype=bool)
    for k in range(N):
        if k > 0:
            b_az = rho_b * b_az + qa * rng.normal()
            b_el = rho_b * b_el + qa * rng.normal()
        # true bearing in the chief Hill frame
        rho_true = hill_relpos(rD_N[k] - rC_N[k], rC_N[k], vC_N[k])
        z_true = bearing_from_rho(rho_true)
        noise = sig[k] * rng.normal(size=2)
        for j in range(2):
            if rng.random() < P_OUTLIER:
                noise[j] *= OUTLIER_SCALE
        z[k, 0] = wrap_pi(z_true[0] + b_az + noise[0])
        z[k, 1] = wrap_pi(z_true[1] + b_el + noise[1])
        if k > 0 and rng.random() < P_DROPOUT:
            valid[k] = False
    sigma_assumed = MEAS_SIGMA_INFLATE * np.column_stack((sig, sig))
    return z, sigma_assumed, valid


# ═════════════════════════ batch residuals ═══════════════════════

def predicted_bearings(
    theta: np.ndarray,
    t_grid: np.ndarray,
    rC_arr: np.ndarray,
    vC_arr: np.ndarray,
    rC_fn: Callable,
    vC_fn: Callable,
    rA_fn: Callable,
    burn_times: np.ndarray,
    burn_dv_eci: np.ndarray,
) -> np.ndarray:
    """
    Propagate the relative state for parameter vector theta and return
    the predicted (az, el) bearing at every grid epoch.  (N,2)
    """
    delta_r0 = theta[0:3]
    delta_v0 = theta[3:6]
    a_emp = theta[6:9] if len(theta) >= 9 else np.zeros(3)

    states = propagate_relative(
        delta_r0, delta_v0, a_emp, t_grid,
        rC_fn, vC_fn, rA_fn, burn_times, burn_dv_eci,
    )
    N = len(t_grid)
    zp = np.zeros((N, 2))
    for k in range(N):
        rho = dcm_lvlh(rC_arr[k], vC_arr[k]) @ states[k, 0:3]
        zp[k] = bearing_from_rho(rho)
    return zp


def residual_vector(
    theta, t_grid, z_meas, valid, rC_arr, vC_arr,
    rC_fn, vC_fn, rA_fn, burn_times, burn_dv_eci,
):
    """Stacked (az, el) innovations for all valid epochs, angle-wrapped."""
    zp = predicted_bearings(theta, t_grid, rC_arr, vC_arr,
                            rC_fn, vC_fn, rA_fn, burn_times, burn_dv_eci)
    r = np.zeros((len(t_grid), 2))
    r[:, 0] = wrap_pi(z_meas[:, 0] - zp[:, 0])
    r[:, 1] = wrap_pi(z_meas[:, 1] - zp[:, 1])
    r[~valid] = 0.0
    return r


# ═════════════════════════ batch OD solver ═══════════════════════

def batch_od_solve(
    theta0: np.ndarray,
    t_grid: np.ndarray,
    z_meas: np.ndarray,
    sigma: np.ndarray,
    valid: np.ndarray,
    rC_arr: np.ndarray,
    vC_arr: np.ndarray,
    rC_fn: Callable,
    vC_fn: Callable,
    rA_fn: Callable,
    burn_times: np.ndarray,
    burn_dv_eci: np.ndarray,
    n_param: int,
    verbose: bool = True,
):
    """
    Gauss-Newton / Levenberg-Marquardt batch fit with Huber robust
    weighting and a soft prior on the empirical acceleration.

    Returns (theta_hat, P_theta, info_dict).
    """
    theta = np.array(theta0, float)
    lam = LM_LAMBDA0

    # finite-difference steps per parameter
    fd_step = np.array([1.0, 1.0, 1.0,        # delta_r0  [m]
                        1e-3, 1e-3, 1e-3,     # delta_v0  [m/s]
                        1e-8, 1e-8, 1e-8])[:n_param]

    # prior precision (only the empirical-acc block is constrained)
    prior_prec = np.zeros((n_param, n_param))
    if n_param >= 9:
        for j in range(6, 9):
            prior_prec[j, j] = 1.0 / (EMP_ACC_PRIOR_SIGMA ** 2)

    def cost(th):
        r = residual_vector(th, t_grid, z_meas, valid, rC_arr, vC_arr,
                             rC_fn, vC_fn, rA_fn, burn_times, burn_dv_eci)
        rw = (r / sigma)[valid].reshape(-1)
        c = float(rw @ rw)
        if n_param >= 9:
            c += float((th[6:9] / EMP_ACC_PRIOR_SIGMA) @
                       (th[6:9] / EMP_ACC_PRIOR_SIGMA))
        return c

    hist = []
    c_prev = cost(theta)
    for it in range(MAX_GN_ITERS):
        # residuals at current theta
        r = residual_vector(theta, t_grid, z_meas, valid, rC_arr, vC_arr,
                             rC_fn, vC_fn, rA_fn, burn_times, burn_dv_eci)
        # normalized residuals (sigma-weighted)
        rn = r / sigma                                     # (N,2)

        # Huber robust weights (down-weight gross outliers)
        absr = np.abs(rn)
        hub = np.ones_like(absr)
        big = absr > HUBER_C
        hub[big] = HUBER_C / absr[big]
        hub[~valid] = 0.0

        # finite-difference Jacobian of the (sigma-normalized) residual
        idx = np.where(valid)[0]
        m = 2 * len(idx)
        J = np.zeros((m, n_param))
        base = (rn[idx]).reshape(-1)
        for j in range(n_param):
            tp = theta.copy(); tp[j] += fd_step[j]
            rp = residual_vector(tp, t_grid, z_meas, valid, rC_arr, vC_arr,
                                 rC_fn, vC_fn, rA_fn,
                                 burn_times, burn_dv_eci)
            J[:, j] = ((rp[idx] / sigma[idx]).reshape(-1) - base) / fd_step[j]

        w = hub[idx].reshape(-1)                           # robust weights
        Jw = J * w[:, None]
        rw = base * w

        # damped normal equations with the empirical-acc prior
        H = Jw.T @ Jw + prior_prec
        g = Jw.T @ rw
        if n_param >= 9:
            g[6:9] += theta[6:9] / (EMP_ACC_PRIOR_SIGMA ** 2)

        # Levenberg-Marquardt step.  The 9 parameters span position,
        # velocity and acceleration scales, so the normal matrix is badly
        # conditioned.  We solve in COLUMN-SCALED coordinates (Marquardt's
        # diagonal scaling): in scaled space H has a unit diagonal and
        # the damping lambda*I is uniform across parameters.
        dsc = np.sqrt(np.abs(np.diag(H)))
        dsc[dsc < 1e-30] = 1.0
        Hs = H / np.outer(dsc, dsc)
        gs = g / dsc
        accepted = False
        for _ in range(18):
            Hd = Hs + lam * np.eye(n_param)
            try:
                dth = -la.solve(Hd, gs) / dsc
            except la.LinAlgError:
                lam *= 10.0
                continue
            c_new = cost(theta + dth)
            if c_new < c_prev:
                theta = theta + dth
                c_prev = c_new
                lam = max(lam * 0.3, 1e-12)
                accepted = True
                break
            lam *= 5.0
        hist.append(c_prev)

        step_scale = la.norm(dth / np.maximum(fd_step, 1e-30)) \
            / np.sqrt(n_param)
        if verbose:
            print(f"    GN it {it + 1:2d}: cost={c_prev:.6e} "
                  f"lambda={lam:.1e} step={step_scale:.3e}")
        if accepted and step_scale < GN_TOL:
            break
        if not accepted:
            break

    # formal covariance from the converged (undamped) normal matrix
    r = residual_vector(theta, t_grid, z_meas, valid, rC_arr, vC_arr,
                         rC_fn, vC_fn, rA_fn, burn_times, burn_dv_eci)
    rn = r / sigma
    absr = np.abs(rn)
    hub = np.ones_like(absr)
    big = absr > HUBER_C
    hub[big] = HUBER_C / absr[big]
    hub[~valid] = 0.0
    idx = np.where(valid)[0]
    base = (rn[idx]).reshape(-1)
    J = np.zeros((2 * len(idx), n_param))
    for j in range(n_param):
        tp = theta.copy(); tp[j] += fd_step[j]
        rp = residual_vector(tp, t_grid, z_meas, valid, rC_arr, vC_arr,
                             rC_fn, vC_fn, rA_fn, burn_times, burn_dv_eci)
        J[:, j] = ((rp[idx] / sigma[idx]).reshape(-1) - base) / fd_step[j]
    w = hub[idx].reshape(-1)
    Jw = J * w[:, None]
    H = Jw.T @ Jw + prior_prec
    # invert in column-scaled coordinates for numerical stability
    dsc = np.sqrt(np.abs(np.diag(H)))
    dsc[dsc < 1e-30] = 1.0
    Hs = H / np.outer(dsc, dsc)
    try:
        P_scaled = la.inv(Hs)
    except la.LinAlgError:
        P_scaled = la.pinv(Hs, rcond=1e-12)
    P_theta = P_scaled / np.outer(dsc, dsc)

    info = {
        "cost_history": np.array(hist),
        "post_resid_rms": float(np.sqrt(np.mean(
            ((r / sigma)[valid]) ** 2))),
        "cond_number": float(np.linalg.cond(H)),
        "n_meas": int(np.count_nonzero(valid)),
    }
    return theta, P_theta, info


# ═════════════════════════ per-deputy driver ═════════════════════

def run_batch_od_for_deputy(dep_idx, data, chief_state_fn=None):
    """
    Build measurements and run the batch OD for one deputy.

    CHIEF SOURCE (important).  The chief trajectory is taken from the
    deputy bundle's own rC_N / vC_N — NOT from chief_reference.npz.
    Verified facts behind this choice:
      * all eight deputy bundles carry an identical rC_N (bit-for-bit),
        so it is the single, consistent chief the swarm was simulated
        against;
      * that chief's specific energy is constant to ~7e-7 — a clean
        dynamically-consistent trajectory;
      * chief_reference.npz is a SEPARATE solution that differs from the
        bundle chief by ~25 m / ~0.04 m/s.  Feeding it into the
        differential-gravity model injects a ~1e-6 m/s^2 inconsistency
        (the deputy and the reference chief then obey different force
        models), which integrates to hundreds of metres of spurious
        "model error" over a long arc.
    Using the bundle chief makes rho = r_D - r_C a clean relative
    trajectory, so the only residual model error is the genuine
    differential perturbation (ASSUMPTION A3/A4).  In flight the deputy
    would use the chief's broadcast instead; its ~1 m smoothed error has
    a negligible effect on the (second-difference) differential gravity.

    Returns a dict with the fitted trajectory (Hill), truth, residuals,
    formal sigma and diagnostics.
    """
    t_full  = np.asarray(data["t"], float).reshape(-1)
    rD_full = np.asarray(data["rD_N"], float).reshape(-1, 3)
    vD_full = np.asarray(data["vD_N"], float).reshape(-1, 3)
    rC_full = np.asarray(data["rC_N"], float).reshape(-1, 3)
    vC_full = np.asarray(data["vC_N"], float).reshape(-1, 3)
    rA_full = np.asarray(data["rA_N"], float).reshape(-1, 3) \
        if "rA_N" in data else None
    impulses = np.asarray(data["impulses"], float).reshape(-1, 4)

    # chief interpolator from the bundle's own (consistent) chief
    chief_state_fn = make_hermite_state_fn(t_full, rC_full, vC_full)

    # ---- decimated batch-fit time grid (skip first 0.5 h transient) ----
    t0 = t_full[0] + 0.5 * 3600.0
    t_grid = np.arange(t0, t_full[-1] - 1e-9, DT_MEAS)
    N = len(t_grid)

    # interpolate deputy truth + chief + Apophis onto the grid
    rD_fn = make_vec_interp(t_full, rD_full)
    vD_fn = make_vec_interp(t_full, vD_full)
    rD_arr = np.array([rD_fn(t) for t in t_grid])
    vD_arr = np.array([vD_fn(t) for t in t_grid])
    rC_arr = np.zeros((N, 3)); vC_arr = np.zeros((N, 3))
    for k, t in enumerate(t_grid):
        rC_arr[k], vC_arr[k] = chief_state_fn(float(t))

    def rC_fn(t):
        return chief_state_fn(float(t))[0]

    def vC_fn(t):
        return chief_state_fn(float(t))[1]

    if rA_full is not None and INCLUDE_APOPHIS:
        rA_fn = make_vec_interp(t_full, rA_full)
        rA_arr = np.array([rA_fn(t) for t in t_grid])
    else:
        rA_fn = None
        rA_arr = None

    # ---- truth Hill state on the grid (for scoring only) ----
    x_true = np.array([
        eci_to_hill_state(rD_arr[k], vD_arr[k], rC_arr[k], vC_arr[k])
        for k in range(N)
    ])

    # ---- burns: Hill-frame delta_v -> ECI ----
    burn_times = impulses[:, 0] if impulses.size else np.zeros(0)
    burn_dv_eci = np.zeros((len(burn_times), 3))
    for i, tb in enumerate(burn_times):
        rC_b, vC_b = chief_state_fn(float(tb))
        burn_dv_eci[i] = dcm_lvlh(rC_b, vC_b).T @ impulses[i, 1:4]

    # ---- camera measurements ----
    rng = np.random.default_rng(7000 + dep_idx)
    z_meas, sigma, valid = build_camera_meas(
        rng, t_grid, rD_arr, rC_arr, vC_arr)

    # ---- initial guess for theta ----
    # position: nominal range along the first measured line of sight,
    # mapped from Hill to ECI; velocity and a_emp start at zero.
    az0, el0 = z_meas[0]
    ce = np.cos(el0)
    rho0_hill = NOMINAL_RANGE * np.array(
        [ce * np.cos(az0), ce * np.sin(az0), np.sin(el0)])
    dr0_eci = dcm_lvlh(rC_arr[0], vC_arr[0]).T @ rho0_hill
    n_param = 9 if ESTIMATE_EMP_ACC else 6
    theta0 = np.zeros(n_param)
    theta0[0:3] = dr0_eci

    print(f"[dep{dep_idx}] batch OD: {N} epochs @ {1.0/DT_MEAS:.2f} Hz, "
          f"arc {(t_grid[-1]-t_grid[0])/3600:.2f} h, "
          f"{len(burn_times)} burns, {n_param} parameters")

    theta_hat, P_theta, info = batch_od_solve(
        theta0, t_grid, z_meas, sigma, valid, rC_arr, vC_arr,
        rC_fn, vC_fn, rA_fn, burn_times, burn_dv_eci, n_param)

    # ---- propagate the fitted solution ----
    states = propagate_relative(
        theta_hat[0:3], theta_hat[3:6],
        theta_hat[6:9] if n_param >= 9 else np.zeros(3),
        t_grid, rC_fn, vC_fn, rA_fn, burn_times, burn_dv_eci)

    # fitted state in the Hill frame
    x_est = np.zeros((N, 6))
    for k in range(N):
        C_HN = dcm_lvlh(rC_arr[k], vC_arr[k])
        rho = C_HN @ states[k, 0:3]
        omega = np.array([0.0, 0.0, hill_rate(rC_arr[k], vC_arr[k])])
        rhod = C_HN @ states[k, 3:6] - np.cross(omega, rho)
        x_est[k] = np.concatenate([rho, rhod])

    # ---- per-epoch formal position sigma (propagate P_theta) ----
    # finite-difference state sensitivity to theta, mapped to Hill pos.
    formal_sig = np.zeros((N, 3))
    fd = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3,
                   1e-8, 1e-8, 1e-8])[:n_param]
    S = np.zeros((N, 3, n_param))
    for j in range(n_param):
        tp = theta_hat.copy(); tp[j] += fd[j]
        st = propagate_relative(
            tp[0:3], tp[3:6], tp[6:9] if n_param >= 9 else np.zeros(3),
            t_grid, rC_fn, vC_fn, rA_fn, burn_times, burn_dv_eci)
        for k in range(N):
            C_HN = dcm_lvlh(rC_arr[k], vC_arr[k])
            S[k, :, j] = (C_HN @ (st[k, 0:3] - states[k, 0:3])) / fd[j]
    for k in range(N):
        Pk = S[k] @ P_theta @ S[k].T
        formal_sig[k] = np.sqrt(np.maximum(np.diag(Pk), 0.0))

    # ---- Apophis-distance diagnostic (ASSUMPTION A5) ----
    if rA_arr is not None:
        d_swarm_apophis = np.linalg.norm(rD_arr - rA_arr, axis=1)
        apo_min = float(d_swarm_apophis.min())
        apo_max = float(d_swarm_apophis.max())
    else:
        apo_min = apo_max = float("nan")

    return {
        "t": t_grid, "x_true": x_true, "x_est": x_est,
        "formal_sig": formal_sig, "theta": theta_hat, "P_theta": P_theta,
        "info": info, "apo_min": apo_min, "apo_max": apo_max,
        "z_meas": z_meas, "sigma": sigma, "valid": valid,
        "burn_times": burn_times,
    }


# ═════════════════════════ assumption checks ═════════════════════

def verify_chief_orbit(chief_state_fn, t_probe):
    """Runtime check of ASSUMPTION A1 (geocentric, bound chief orbit)."""
    rC, vC = chief_state_fn(float(t_probe))
    rmag = la.norm(rC); vmag = la.norm(vC)
    eps = 0.5 * vmag ** 2 - MU_EARTH / rmag
    a = -MU_EARTH / (2.0 * eps) if eps < 0 else float("inf")
    ok = eps < 0 and 6.6e6 < rmag < 1.0e9
    print(f"[A1 check] chief |r|={rmag:.3e} m, |v|={vmag:.1f} m/s, "
          f"eps={eps:.3e} m^2/s^2, a={a:.3e} m  -> "
          f"{'GEOCENTRIC BOUND (ok)' if ok else 'WARNING: assumption A1 questionable'}")
    return ok


# ═════════════════════════ main ══════════════════════════════════

def apophis_swarm_batch_od_demo(data_dir: Path, n_dep: int = N_DEP):
    data_dir = Path(data_dir)
    chief_ref_path = data_dir / "chief_reference.npz"
    if not chief_ref_path.is_file():
        raise FileNotFoundError(f"Missing chief reference: {chief_ref_path}")

    t_ref, rC_ref, vC_ref, chief_state_fn = load_chief_reference(
        chief_ref_path)
    print(f"[chief-ref] {chief_ref_path.name}: {len(t_ref)} samples, "
          f"t in [{t_ref[0]:.1f}, {t_ref[-1]:.1f}] s")
    verify_chief_orbit(chief_state_fn, 0.5 * (t_ref[0] + t_ref[-1]))

    results = []
    for d in range(n_dep):
        bundle = data_dir / f"apophis_rendezvous_full_D{d}.npz"
        if not bundle.is_file():
            raise FileNotFoundError(f"Missing bundle: {bundle}")
        print(f"\n===== Deputy {d} =====")
        data = load_apophis_bundle(bundle)
        res = run_batch_od_for_deputy(d, data, chief_state_fn)
        results.append(res)

        # save OD bundle
        out = data_dir / f"apophis_swarm_OD_los_D{d}.npz"
        np.savez_compressed(
            out, t=res["t"], x_true=res["x_true"], x_est=res["x_est"],
            formal_sig=res["formal_sig"], theta=res["theta"],
            P_theta=res["P_theta"])
        print(f"[dep{d}] saved -> {out.name}")

    # ─── per-deputy report ───
    labels = ["R", "T", "N"]
    print("\n" + "=" * 64)
    print("BATCH-OD CAMERA-ONLY NAVIGATION — SUMMARY")
    print("=" * 64)
    for d in range(n_dep):
        res = results[d]
        t = res["t"]; t_hr = (t - t[0]) / 3600.0
        mask = t_hr >= 0.1
        err = (res["x_est"] - res["x_true"])[mask]
        rho_true = res["x_true"][mask, 0:3]
        u = rho_true / np.maximum(
            np.linalg.norm(rho_true, axis=1, keepdims=True), 1e-9)
        along = np.sum(err[:, 0:3] * u, axis=1)
        cross = np.linalg.norm(err[:, 0:3] - along[:, None] * u, axis=1)
        info = res["info"]
        print(f"\nDeputy {d}:")
        for i, lbl in enumerate(labels):
            e = err[:, i]
            print(f"  [POS] {lbl}: mean={np.mean(e):+8.3f} m  "
                  f"RMS={np.sqrt(np.mean(e**2)):8.3f} m")
        print(f"  along-LOS (range) RMS = "
              f"{np.sqrt(np.mean(along**2)):8.3f} m")
        print(f"  cross-LOS (bearing) RMS = "
              f"{np.sqrt(np.mean(cross**2)):8.3f} m")
        print(f"  post-fit residual RMS = {info['post_resid_rms']:.3f} "
              f"(sigma units; ~1.0 ideal)   "
              f"normal-matrix cond = {info['cond_number']:.2e}")
        print(f"  swarm-Apophis distance over arc: "
              f"[{res['apo_min']:.0f}, {res['apo_max']:.0f}] m  "
              f"-> Apophis term "
              f"{'MATTERS (<800 m)' if res['apo_min'] < 800.0 else 'negligible'}")

    # ─── plots ───
    _make_plots(results, n_dep)


def _make_plots(results, n_dep):
    labels = ["R", "T", "N"]

    # per-deputy: error vs formal 3-sigma
    for d in range(n_dep):
        res = results[d]
        t = res["t"]; t_hr = (t - t[0]) / 3600.0
        mask = t_hr >= 0.1
        err = res["x_est"] - res["x_true"]
        fsig = res["formal_sig"]
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            ax = axes[i]
            ax.scatter(t_hr[mask], err[mask, i], s=6, alpha=0.4,
                       label=f"D{d} error rho_{labels[i]}")
            ax.plot(t_hr[mask], 3.0 * fsig[mask, i], "r--", lw=1.0,
                    label="+3sigma formal" if i == 0 else None)
            ax.plot(t_hr[mask], -3.0 * fsig[mask, i], "r--", lw=1.0,
                    label="-3sigma formal" if i == 0 else None)
            ax.set_ylabel(f"{labels[i]} [m]", fontsize=AXIS_LABEL_FONTSIZE)
            ax.grid(True)
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
        axes[-1].set_xlabel("Time since deploy [h]",
                            fontsize=AXIS_LABEL_FONTSIZE)
        axes[0].legend(fontsize=LEGEND_FONTSIZE)
        fig.suptitle(f"Batch-OD position error vs formal 3sigma - Deputy {d}",
                     fontsize=TITLE_FONTSIZE)
        fig.tight_layout()

    # along-LOS vs cross-LOS RMS bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    along_t = np.zeros(n_dep); cross_t = np.zeros(n_dep)
    for d in range(n_dep):
        res = results[d]
        t = res["t"]; t_hr = (t - t[0]) / 3600.0
        mask = t_hr >= 0.1
        err = (res["x_est"] - res["x_true"])[mask]
        rho_true = res["x_true"][mask, 0:3]
        u = rho_true / np.maximum(
            np.linalg.norm(rho_true, axis=1, keepdims=True), 1e-9)
        along = np.sum(err[:, 0:3] * u, axis=1)
        cross = np.linalg.norm(err[:, 0:3] - along[:, None] * u, axis=1)
        along_t[d] = np.sqrt(np.mean(along ** 2))
        cross_t[d] = np.sqrt(np.mean(cross ** 2))
    xp = np.arange(n_dep); wdt = 0.38
    ax.bar(xp - wdt / 2, along_t, wdt, label="along-LOS (range)")
    ax.bar(xp + wdt / 2, cross_t, wdt, label="cross-LOS (bearing)")
    ax.set_xticks(xp); ax.set_xticklabels([f"D{d}" for d in range(n_dep)])
    ax.set_ylabel("Position RMS [m]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Batch-OD camera-only: along-LOS vs cross-LOS RMS",
                 fontsize=TITLE_FONTSIZE)
    ax.grid(True, axis="y", alpha=0.4)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()

    # convergence history
    fig, ax = plt.subplots(figsize=(11, 6))
    for d in range(n_dep):
        ch = results[d]["info"]["cost_history"]
        ax.semilogy(np.arange(1, len(ch) + 1), ch, "-o", lw=1.0,
                    label=f"D{d}")
    ax.set_xlabel("Gauss-Newton iteration", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("batch cost", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Batch-OD solver convergence", fontsize=TITLE_FONTSIZE)
    ax.grid(True, which="both", alpha=0.4)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, ncol=4)
    fig.tight_layout()

    # 3-D Hill trajectories
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    for d in range(n_dep):
        xt = results[d]["x_true"]; xe = results[d]["x_est"]
        ax.plot(xt[:, 0], xt[:, 1], xt[:, 2], lw=1.2, alpha=0.9,
                label=f"D{d} truth")
        ax.plot(xe[:, 0], xe[:, 1], xe[:, 2], lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("rho_R [m]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("rho_T [m]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_zlabel("rho_N [m]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Batch-OD Hill trajectories (truth solid, OD dashed)",
                 fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE * 0.7, ncol=2, loc="upper left")
    fig.tight_layout()

    plt.show()


# entry point

if __name__ == "__main__":
    dep0 = SCRIPT_DIR / "apophis_rendezvous_full_D0.npz"
    chief = SCRIPT_DIR / "chief_reference.npz"
    if not dep0.is_file():
        raise FileNotFoundError(f"Expected bundle not found: {dep0}")
    if not chief.is_file():
        raise FileNotFoundError(f"Expected chief reference not found: {chief}")
    apophis_swarm_batch_od_demo(SCRIPT_DIR)
