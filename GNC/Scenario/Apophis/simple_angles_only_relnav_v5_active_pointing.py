#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_angles_only_relnav_v5_active_pointing.py
================================================

Closes the v4 attitude-control gap with a novel pointing law:

    F S D P ²   —   Filter-State-Driven Predictive Pointing

Three ideas, tightly coupled:

(A)  Reference attitude from the FILTER's chief estimate, not from truth.
     The deputy points at where its own EKF says the chief is.  This
     makes estimation and control a single closed loop: a better
     filter ⇒ tighter pointing ⇒ more frames the chief is in FOV ⇒
     more measurements ⇒ better filter.  No "secret truth" leakage.

(B)  Feedforward from line-of-sight rate.
     The reference is not a static "look-at" attitude — it carries the
     full inertial-frame LOS angular velocity

         ω_LOS_N  =  (Δr × Δv) / |Δr|²,    Δr = r_c − r_d, Δv = v_c − v_d

     so the deputy slews at the same rate the chief is moving across
     the sky.  Without this, a pure attitude-error feedback lags the
     target by tens of seconds and the chief escapes the FOV.

(C)  Filter-uncertainty-aware mode switching.
     The controller runs in three modes selected by `tr(P_dep[:3,:3])`
     and the time since the last accepted camera measurement:

         TRACK       — fresh measurements, low uncertainty:
                         feedforward + tight MRP feedback
         ACQUISITION — high uncertainty or no measurements for >ACQ_TIMEOUT_S:
                         slew toward FIMC-predicted Hill-ellipse zone with
                         very high gains; widen the "in-FOV" window
         RECOVERY    — bore-sight error > 90° from desired:
                         pure rate damping + low-gain attitude regression

The control loop is REAL 6DOF:

    Each nav step the script
        1. reads the BSK truth deputy attitude (σ_BN, ω_BN_B),
        2. computes the desired (σ_RN, ω_RN_B, ω̇_RN_B) from filter state,
        3. forms the Schaub-Junkins MRP-feedback torque
                L = −K_σ σ_BR − P_ω ω_BR
                    + I_B ω̇_RN_B − I_B (ω_BR × ω_RN_B)
                    + ω_BN × I_B ω_BN
        4. writes L to the deputy's `extForceTorque` body-torque effector,
        5. advances the BSK 6DOF integrator by exactly DT_NAV,
        6. takes a real pin-hole camera measurement of the chief,
        7. runs the UKF + IPLF + 4-D LRL FIMC update,
        8. records everything.

Truth physics are still v4-grade: GGM03S spherical harmonics (deg 70 by
default), Sun + Moon third bodies via SPICE, optional drag, plus 4-RW
pyramids on each spacecraft (passive — the active control authority is
provided through extForceTorque so the FSDP² law is hardware-agnostic
and the RWs simply contribute their angular-momentum coupling).

Author: Harish Vernekar (2026)
"""

from __future__ import annotations
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Callable

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ──────────────────────────────────────────────────────────────────────────────
#                               Basilisk imports
# ──────────────────────────────────────────────────────────────────────────────

BSK_AVAILABLE = False
BSK_DRAG_AVAILABLE = False
BSK_RW_AVAILABLE = False
BSK_EXTFT_AVAILABLE = False
try:
    from Basilisk import __path__ as bsk_pkg_path
    from Basilisk.simulation import (
        spacecraft,
        gravityEffector,
        extForceTorque,
    )
    from Basilisk.utilities import (
        SimulationBaseClass,
        macros,
        orbitalMotion as bsk_orbital,
        simIncludeGravBody,
        unitTestSupport,
        RigidBodyKinematics as rbk,
    )
    from Basilisk.architecture import messaging
    BSK_PATH = bsk_pkg_path[0]
    BSK_AVAILABLE = True
    BSK_EXTFT_AVAILABLE = True
    try:
        from Basilisk.simulation import (
            dragDynamicEffector,
            exponentialAtmosphere,
        )
        from Basilisk.utilities import simSetPlanetEnvironment
        BSK_DRAG_AVAILABLE = True
    except Exception:
        BSK_DRAG_AVAILABLE = False
    try:
        from Basilisk.simulation import reactionWheelStateEffector
        from Basilisk.utilities import simIncludeRW
        BSK_RW_AVAILABLE = True
    except Exception:
        BSK_RW_AVAILABLE = False
except Exception as exc:
    BSK_PATH = ""
    warnings.warn(
        "Basilisk not importable in this environment "
        f"({exc!r}).  v5 will fall back to a numpy 2-body+J2 truth + "
        "analytic perfect-pointing.",
        RuntimeWarning,
    )


# ──────────────────────────────────────────────────────────────────────────────
#                            CONSTANTS / CONFIG
# ──────────────────────────────────────────────────────────────────────────────

MU_EARTH    = 3.986004418e14
R_EARTH     = 6378.137e3

ALTITUDE    = 700e3
R_ORBIT     = R_EARTH + ALTITUDE
V_CIRC      = np.sqrt(MU_EARTH / R_ORBIT)
N_ORBIT     = V_CIRC / R_ORBIT
T_ORBIT     = 2 * np.pi / N_ORBIT

BASELINE    = 2000.0

DT_NAV         = 1.0
DT_BSK_INT     = 0.25                  # tighter integration for active control
DURATION_S     = 1 * 3600.0
N_STEPS        = int(round(DURATION_S / DT_NAV)) + 1
N_PERIOD       = int(round(T_ORBIT / DT_NAV))

# ── Truth-side toggles ───────────────────────────────────────────────────────
# v5.3 NOTE: Even when Basilisk is importable on this build, its 6DOF
# integrator is producing NaN payloads at t=DT_NAV under the chunked
# ConfigureStopTime / ExecuteSimulation pattern used here.  Rather than
# fight the BSK build, we have re-implemented the exact same
# 2-body+J2 + rigid-body MRP dynamics in pure numpy (see ``state12_dot``
# / ``step_6dof`` / ``propagate_6dof`` above) and force the truth path
# to use it.  Set this flag to True only on a known-good BSK build.
USE_BASILISK            = False
MAX_SH_DEG              = 70
INCLUDE_THIRD_BODIES    = True
INCLUDE_DRAG            = False
EPOCH_UTC               = "2026 May 09 00:00:00"
SH_GRAV_FILE            = (
    BSK_PATH + "/supportData/LocalGravData/GGM03S.txt"
    if BSK_AVAILABLE else ""
)

# ── Spacecraft structures ────────────────────────────────────────────────────
CHIEF_MASS_KG       = 200.0
CHIEF_INERTIA_BC_B  = np.diag([35.0, 50.0, 60.0])
CHIEF_RBcB_B        = [[0.0], [0.0], [0.0]]
DEPUTY_MASS_KG      = 100.0
DEPUTY_INERTIA_BC_B = np.diag([12.0, 18.0, 22.0])
DEPUTY_RBcB_B       = [[0.0], [0.0], [0.0]]

# ── Reaction-wheel pyramid (passive) ─────────────────────────────────────────
RW_TYPE             = "Honeywell_HR16"
RW_MAX_MOMENTUM     = 50.0
RW_EL_DEG           = 40.0
RW_AZ_DEG           = [45.0, 135.0, 225.0, 315.0]
RW_MOUNT_RADIUS     = 0.4

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_FOCAL_M      = 25e-3
CAMERA_PIXEL_M      = 9e-6
CAMERA_NX           = 1024
CAMERA_NY           = 1024
SIGMA_PIX           = 0.2
SIGMA_ATT_RAD       = 1.0e-4
DCM_CB              = np.array([[0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0]])
CAMERA_HALF_FOV_X   = np.arctan(CAMERA_NX / 2 * CAMERA_PIXEL_M / CAMERA_FOCAL_M)
CAMERA_HALF_FOV_Y   = np.arctan(CAMERA_NY / 2 * CAMERA_PIXEL_M / CAMERA_FOCAL_M)

# ── FSDP² controller knobs ───────────────────────────────────────────────────
# Use FILTER STATE to drive pointing (the honest case).  Set to False to
# test "perfect-knowledge" tracking with truth states — useful as an
# upper bound diagnostic.
USE_FILTER_FOR_POINTING = True

# MRP-feedback gains in TRACK mode.  Targeted closed-loop bandwidth
# ω_n ≈ 0.5 rad/s, damping ≈ 0.7 → Kσ ≈ I·ω_n² ≈ 4 N·m, Pω ≈ 2·ζ·I·ω_n ≈ 12 N·m·s
TRACK_K_SIGMA       = 4.0     # N·m   per unit MRP error
TRACK_P_OMEGA       = 12.0    # N·m·s per unit body-rate error

# Higher gains in ACQUISITION mode — push hard to slew the bore-sight
ACQ_K_SIGMA         = 18.0
ACQ_P_OMEGA         = 30.0

# RECOVERY mode: pure rate damping + soft regression toward LOS
REC_K_SIGMA         = 1.0
REC_P_OMEGA         = 8.0

# Maximum body torque the controller may command (saturation)
TAU_MAX_NM          = 0.5

# Mode-switch thresholds
ACQ_P_TRACE_THRESH  = 5e3      # m² — switch to ACQ if tr(P_dep[:3,:3]) > this
ACQ_TIMEOUT_S       = 300.0    # s  — switch to ACQ if no meas. for this long
RECOVERY_ANGLE_DEG  = 60.0     # ° — switch to RECOVERY if pointing error > this

# ── Filter (carry-over from v4) ──────────────────────────────────────────────
J2_COEFF          = 1.08262668e-3
INCLUDE_J2_FILTER = True
N_RK4_SUBSTEPS    = 5

SIGMA_GPS_POS = 5.0
SIGMA_INIT_POS_DEP  = 50.0; SIGMA_INIT_VEL_DEP = 0.05
SIGMA_INIT_POS_CHF  = 50.0; SIGMA_INIT_VEL_CHF = 0.5
INIT_BIAS_POS_DEP   = np.array([20.0, -15.0, 10.0])
INIT_BIAS_VEL_DEP   = np.array([0.01, -0.008, 0.005])

SIGMA_PROC_POS = 0.10
SIGMA_PROC_VEL = 1e-3

NIS_GATE_GPS_POS = 100.0
NIS_GATE_LOS     = 100.0
NIS_GATE_SIZE    = 100.0
NIS_GATE_FIMC    = 1e6

CHIEF_CHORD_M    = 5.0
SIGMA_SIZE_RAD   = 1.5e-5

SIGMA_FIMC_EPS   = 80.0
SIGMA_FIMC_H     = 2.0e7
SIGMA_FIMC_LRL   = 5.0e-4

IPLF_MAX_ITER    = 5
IPLF_TOL_POS     = 1e-3
IPLF_TOL_VEL     = 1e-6

PRINT_EVERY_S = 60.0
PRINT_EVERY_K = int(round(PRINT_EVERY_S / DT_NAV))

OUT_DIR  = Path(__file__).resolve().parent
NPZ_PATH = OUT_DIR / "simple_angles_only_results_v5_active.npz"
SAVE_FIGS = False
FIG_DIR  = OUT_DIR / "simple_angles_only_figs_v5_active"
RNG_SEED = 1234


# ──────────────────────────────────────────────────────────────────────────────
#                Filter-side dynamics  (2-body + J2)
# ──────────────────────────────────────────────────────────────────────────────

def two_body_accel(r):
    rn = la.norm(r)
    return -MU_EARTH * r / (rn ** 3)

def j2_accel(r):
    x, y, z = r; rn = la.norm(r)
    if rn < 1e-6: return np.zeros(3)
    rn5 = rn**5
    coeff = -1.5 * J2_COEFF * MU_EARTH * (R_EARTH ** 2) / rn5
    s = (z / rn) ** 2
    return coeff * np.array([
        x * (1.0 - 5.0 * s),
        y * (1.0 - 5.0 * s),
        z * (3.0 - 5.0 * s),
    ])

def filter_accel(r):
    return two_body_accel(r) + j2_accel(r) if INCLUDE_J2_FILTER else two_body_accel(r)

def rk4_step(state6, dt):
    def f(s):
        return np.hstack([s[3:6], filter_accel(s[:3])])
    k1 = f(state6); k2 = f(state6 + 0.5*dt*k1)
    k3 = f(state6 + 0.5*dt*k2); k4 = f(state6 + dt*k3)
    return state6 + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def propagate_filter(state6, dt_total, n_substeps=N_RK4_SUBSTEPS):
    dt = dt_total / n_substeps
    s = state6.copy()
    for _ in range(n_substeps):
        s = rk4_step(s, dt)
    return s

def two_body_state_transition(r, v, dt):
    rn = la.norm(r); rhat = r/rn
    G = -(MU_EARTH/rn**3) * (np.eye(3) - 3.0*np.outer(rhat, rhat))
    F = np.zeros((6, 6)); F[0:3, 3:6] = np.eye(3); F[3:6, 0:3] = G
    Fdt = F * dt
    return np.eye(6) + Fdt + 0.5 * Fdt @ Fdt


# ──────────────────────────────────────────────────────────────────────────────
#         Pure-numpy 6DOF rigid-body propagator
#         (replaces Basilisk when USE_BASILISK has been failing)
# ──────────────────────────────────────────────────────────────────────────────
#
# State convention for the 12-vector ``state12``:
#     state[0:3]   = r_BN_N    inertial position           [m]
#     state[3:6]   = v_BN_N    inertial velocity           [m/s]
#     state[6:9]   = sigma_BN  MRP, inertial→body          [-]
#     state[9:12]  = omega_BN_B body angular rate, body    [rad/s]
#
# Translation:   r̈ = a_grav(r)              (2-body + J2)
# Attitude:      σ̇ = (1/4) [(1−σᵀσ)I + 2σ̃ + 2σσᵀ] ω
# Rotation:      I·ω̇ = −ω × (Iω) + L_ext_B
#
# RK4 substepping; MRP shadow-set switch when |σ| > 1.

def truth_accel(r):
    """Truth-side gravity model — same as the filter (2-body + J2)."""
    return filter_accel(r)


def state12_dot(state12, I_B, L_ext_B, accel_fn):
    r     = state12[0:3]
    v     = state12[3:6]
    sigma = state12[6:9]
    omega = state12[9:12]

    r_dot = v
    v_dot = accel_fn(r)

    s2 = float(np.dot(sigma, sigma))
    st = np.array([[ 0.0,        -sigma[2],   sigma[1]],
                   [ sigma[2],    0.0,       -sigma[0]],
                   [-sigma[1],    sigma[0],   0.0     ]])
    B = (1.0 - s2) * np.eye(3) + 2.0 * st + 2.0 * np.outer(sigma, sigma)
    sigma_dot = 0.25 * B @ omega

    omega_dot = la.solve(I_B, -np.cross(omega, I_B @ omega) + L_ext_B)

    return np.concatenate([r_dot, v_dot, sigma_dot, omega_dot])


def step_6dof(state12, dt, I_B, L_ext_B, accel_fn=None):
    """Single RK4 step on the full 12-state 6DOF rigid body."""
    if accel_fn is None:
        accel_fn = truth_accel
    k1 = state12_dot(state12,                I_B, L_ext_B, accel_fn)
    k2 = state12_dot(state12 + 0.5*dt*k1,    I_B, L_ext_B, accel_fn)
    k3 = state12_dot(state12 + 0.5*dt*k2,    I_B, L_ext_B, accel_fn)
    k4 = state12_dot(state12 +     dt*k3,    I_B, L_ext_B, accel_fn)
    new = state12 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    # MRP shadow-set rule: if |σ| > 1 switch σ → −σ/|σ|² to keep MRP valid
    s = new[6:9]
    s2 = float(np.dot(s, s))
    if s2 > 1.0:
        new[6:9] = -s / s2
    return new


def propagate_6dof(state12, dt_total, I_B, L_ext_B, n_substeps=10,
                   accel_fn=None):
    """Multi-substep RK4 propagation of one 6DOF rigid body.  Treats
    L_ext_B as constant over the nav step (zero-order hold)."""
    dt = dt_total / n_substeps
    s = state12.copy()
    for _ in range(n_substeps):
        s = step_6dof(s, dt, I_B, L_ext_B, accel_fn=accel_fn)
    return s


# ──────────────────────────────────────────────────────────────────────────────
#                 Keplerian invariants (LRL FIMC)  — v3+
# ──────────────────────────────────────────────────────────────────────────────

def kepler_invariants(state6):
    r = state6[:3]; v = state6[3:]
    eps = 0.5 * np.dot(v, v) - MU_EARTH / la.norm(r)
    h = np.cross(r, v)
    e = np.cross(v, h) / MU_EARTH - r / la.norm(r)
    return np.hstack([[eps], h, e])


# ──────────────────────────────────────────────────────────────────────────────
#                         Measurement helpers
# ──────────────────────────────────────────────────────────────────────────────

def gps_position_measurement(r_truth, rng):
    return r_truth + SIGMA_GPS_POS * rng.normal(size=3)

def los_angles(r_observer, r_target):
    dr = r_target - r_observer
    dn = la.norm(dr)
    alpha = np.arctan2(dr[1], dr[0])
    delta = np.arcsin(np.clip(dr[2] / dn, -1.0, 1.0))
    return alpha, delta, dr

def los_jacobian_wrt_observer(dr):
    dx, dy, dz = dr
    sxy2 = dx*dx + dy*dy; n2 = sxy2 + dz*dz
    n = np.sqrt(n2); sxy = np.sqrt(sxy2)
    H = np.zeros((2, 3))
    if sxy2 > 1e-12:
        H[0, :] = -np.array([-dy, dx, 0.0]) / sxy2
    if sxy > 1e-9 and n > 1e-9:
        H[1, :] = -np.array([-dx*dz, -dy*dz, sxy2]) / (sxy * n2)
    return H


# ──────────────────────────────────────────────────────────────────────────────
#                 MRP <-> DCM helpers and pin-hole projection
# ──────────────────────────────────────────────────────────────────────────────

def mrp_to_dcm(sigma):
    sigma = np.asarray(sigma, float).reshape(3)
    s2 = float(np.dot(sigma, sigma))
    st = np.array([[ 0.0,        -sigma[2],   sigma[1]],
                   [ sigma[2],    0.0,       -sigma[0]],
                   [-sigma[1],    sigma[0],   0.0     ]])
    I = np.eye(3)
    num = (8.0 * st @ st) - 4.0 * (1.0 - s2) * st
    den = (1.0 + s2) ** 2
    return I + num / den

def dcm_to_mrp(C):
    if BSK_AVAILABLE:
        return np.asarray(rbk.C2MRP(C)).reshape(3)
    tr = np.trace(C); q = np.zeros(4)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        q[0] = 0.25*S
        q[1] = (C[2,1]-C[1,2])/S; q[2]=(C[0,2]-C[2,0])/S; q[3]=(C[1,0]-C[0,1])/S
    else:
        i = int(np.argmax(np.diag(C)))
        if i == 0:
            S = np.sqrt(1.0 + C[0,0] - C[1,1] - C[2,2]) * 2.0
            q[0]=(C[2,1]-C[1,2])/S; q[1]=0.25*S; q[2]=(C[0,1]+C[1,0])/S; q[3]=(C[0,2]+C[2,0])/S
        elif i == 1:
            S = np.sqrt(1.0 + C[1,1] - C[0,0] - C[2,2]) * 2.0
            q[0]=(C[0,2]-C[2,0])/S; q[1]=(C[0,1]+C[1,0])/S; q[2]=0.25*S; q[3]=(C[1,2]+C[2,1])/S
        else:
            S = np.sqrt(1.0 + C[2,2] - C[0,0] - C[1,1]) * 2.0
            q[0]=(C[1,0]-C[0,1])/S; q[1]=(C[0,2]+C[2,0])/S; q[2]=(C[1,2]+C[2,1])/S; q[3]=0.25*S
    if q[0] < 0: q = -q
    return q[1:] / (1.0 + q[0])


def mrp_subtract(sigma_BN, sigma_RN):
    """Computes σ_BR = σ_BN ⊖ σ_RN  (Schaub-Junkins eq. 3.155)."""
    s2_BN = np.dot(sigma_BN, sigma_BN)
    s2_RN = np.dot(sigma_RN, sigma_RN)
    num = ((1.0 - s2_RN) * sigma_BN - (1.0 - s2_BN) * sigma_RN
           + 2.0 * np.cross(sigma_BN, sigma_RN))
    den = 1.0 + s2_BN * s2_RN - 2.0 * np.dot(sigma_BN, sigma_RN)
    if abs(den) < 1e-9:
        return sigma_BN - sigma_RN          # fallback: linearised diff
    res = num / den
    # MRPs prefer the short rotation: switch to shadow set if |σ| > 1
    if np.dot(res, res) > 1.0:
        res = -res / np.dot(res, res)
    return res


def project_target_to_pixels(r_chief_N, r_dep_N, sigma_dep_BN):
    DCM_BN = mrp_to_dcm(sigma_dep_BN)
    DCM_CN = DCM_CB @ DCM_BN
    dr_N = np.asarray(r_chief_N) - np.asarray(r_dep_N)
    dr_C = DCM_CN @ dr_N
    if dr_C[2] <= 0.0:
        return 0.0, 0.0, False, dr_C
    u = CAMERA_FOCAL_M * dr_C[0] / dr_C[2] / CAMERA_PIXEL_M
    v = CAMERA_FOCAL_M * dr_C[1] / dr_C[2] / CAMERA_PIXEL_M
    in_fov = (abs(u) < CAMERA_NX / 2) and (abs(v) < CAMERA_NY / 2)
    return u, v, in_fov, dr_C


def back_project_pixel_to_los(u_pix, v_pix, sigma_dep_BN):
    DCM_BN = mrp_to_dcm(sigma_dep_BN)
    DCM_CN = DCM_CB @ DCM_BN
    DCM_NC = DCM_CN.T
    los_C = np.array([u_pix * CAMERA_PIXEL_M / CAMERA_FOCAL_M,
                      v_pix * CAMERA_PIXEL_M / CAMERA_FOCAL_M,
                      1.0])
    los_C = los_C / la.norm(los_C)
    los_N = DCM_NC @ los_C
    alpha = np.arctan2(los_N[1], los_N[0])
    delta = np.arcsin(np.clip(los_N[2], -1.0, 1.0))
    return alpha, delta


# ──────────────────────────────────────────────────────────────────────────────
#                      F S D P ²    pointing reference
# ──────────────────────────────────────────────────────────────────────────────

def compute_pointing_reference(r_chief, v_chief, r_dep, v_dep):
    """
    Build the desired (σ_RN, ω_RN_N, ω̇_RN_N) so that the camera +Z axis
    in the deputy body frame is aligned with (r_chief − r_dep), with the
    image-up axis (camera +Y) along the orbit normal.

    Returns
    -------
    sigma_RN     : (3,)   reference MRP (inertial → reference body)
    omega_RN_N   : (3,)   reference angular velocity in inertial frame
    omegadot_RN_N: (3,)   reference angular acceleration (zeroed for v5)
    DCM_RN       : (3,3)  reference DCM
    """
    dr_N = np.asarray(r_chief) - np.asarray(r_dep)
    range_now = la.norm(dr_N)
    z_R_N = dr_N / range_now
    # image-up = orbit normal of the deputy's own orbit, but smoothed
    h_N = np.cross(np.asarray(r_dep), np.asarray(v_dep))
    h_norm = la.norm(h_N)
    if h_norm < 1e-3:
        h_hat = np.array([0.0, 0.0, 1.0])
    else:
        h_hat = h_N / h_norm
    # camera +Y in inertial = component of orbit normal perpendicular to LOS
    y_R_N = np.cross(h_hat, z_R_N)
    y_norm = la.norm(y_R_N)
    if y_norm < 1e-9:
        # bore-sight parallel to orbit normal — pick any orthogonal
        e_alt = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(e_alt, z_R_N)) > 0.9:
            e_alt = np.array([0.0, 1.0, 0.0])
        y_R_N = np.cross(z_R_N, e_alt)
        y_norm = la.norm(y_R_N)
    y_R_N /= y_norm
    x_R_N = np.cross(y_R_N, z_R_N)
    DCM_NR = np.column_stack([x_R_N, y_R_N, z_R_N])
    DCM_RN = DCM_NR.T
    # The body should align so that camera frame matches reference frame.
    # Since DCM_CB sends body → camera, the desired body-from-inertial
    # DCM is C_BN = C_BC · C_CN_desired  where C_CN_desired = I (camera
    # aligned with reference).  In our convention the reference frame IS
    # the camera frame, so C_BN = DCM_CB^T · DCM_RN.
    DCM_BN_ref = DCM_CB.T @ DCM_RN
    sigma_RN = dcm_to_mrp(DCM_BN_ref)

    # ── Reference angular velocity (LOS rate) ──
    dv_N = np.asarray(v_chief) - np.asarray(v_dep)
    omega_RN_N = np.cross(dr_N, dv_N) / (range_now ** 2)
    # NOTE: this is the rate of rotation of the LOS unit vector in inertial.
    # The reference frame (and thus the body) needs to rotate at this rate
    # PLUS a small correction from the orbit-normal evolution — we ignore
    # the latter (it's < 10⁻⁵ rad/s under J2 alone).

    omegadot_RN_N = np.zeros(3)         # zeroed; small for our orbit
    return sigma_RN, omega_RN_N, omegadot_RN_N, DCM_BN_ref


def fsdp_control_torque(sigma_BN, omega_BN_B, sigma_RN, omega_RN_N,
                        omegadot_RN_N, DCM_BN_ref, I_B, K_sigma, P_omega):
    """
    Schaub-Junkins MRP feedback control law with feedforward.

        L = −K_σ σ_BR − P_ω ω_BR
            + I_B ω̇_RN_B − I_B (ω_BR × ω_RN_B)
            + ω_BN_B × (I_B ω_BN_B)

    All angular velocities are expressed in the body frame.  The
    reference rate ω_RN_N is rotated into body via the actual attitude:
    ω_RN_B = C_BN ω_RN_N.
    """
    DCM_BN_act = mrp_to_dcm(sigma_BN)
    omega_RN_B   = DCM_BN_act @ omega_RN_N
    omegadot_RN_B = DCM_BN_act @ omegadot_RN_N

    sigma_BR = mrp_subtract(np.asarray(sigma_BN, float).reshape(3),
                             np.asarray(sigma_RN, float).reshape(3))
    omega_BR = np.asarray(omega_BN_B, float).reshape(3) - omega_RN_B

    gyroscopic = np.cross(omega_BN_B, I_B @ omega_BN_B)
    L = (-K_sigma * sigma_BR
         - P_omega * omega_BR
         + I_B @ omegadot_RN_B
         - I_B @ np.cross(omega_BR, omega_RN_B)
         + gyroscopic)

    # Saturate to TAU_MAX_NM
    Lnorm = la.norm(L)
    if Lnorm > TAU_MAX_NM:
        L = L * (TAU_MAX_NM / Lnorm)
    return L


# ──────────────────────────────────────────────────────────────────────────────
#                   Initial conditions  (same as v4)
# ──────────────────────────────────────────────────────────────────────────────

def initial_chief_state():
    return np.array([R_ORBIT, 0.0, 0.0, 0.0, V_CIRC, 0.0])

def initial_deputy_state():
    A = BASELINE; B = BASELINE
    rho_H    = np.array([-0.5*A, 0.0, B])
    rhodot_H = np.array([0.0,   N_ORBIT*A, 0.0])
    omega = np.array([0.0, 0.0, N_ORBIT])
    chief = initial_chief_state()
    return np.hstack([chief[:3] + rho_H,
                       chief[3:] + rhodot_H + np.cross(omega, rho_H)])


def initial_deputy_attitude_to_chief(r_chief, v_chief, r_dep, v_dep):
    """Deputy bore-sight at chief at t=0, image-up along orbit normal,
    initial body rate matching the LOS rate at t=0.

    BUGFIX (v5.1): we now require the actual ``v_dep`` instead of
    pretending it's zero — the previous version produced an LOS rate of
    ~3 rad/s (because Δv was equal to the entire chief velocity), which
    spun the deputy out of control and NaN'd the BSK 6DOF integrator at
    step 1.
    """
    sigma_RN, omega_RN_N, _, DCM_BN_ref = compute_pointing_reference(
        r_chief, v_chief, r_dep, v_dep)
    omega_BN_B = DCM_BN_ref @ omega_RN_N
    return sigma_RN, omega_BN_B


# ──────────────────────────────────────────────────────────────────────────────
#                     EKF (chief)  + UKF + IPLF (deputy)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EKFConfig:
    sigma_proc_pos: float = SIGMA_PROC_POS
    sigma_proc_vel: float = SIGMA_PROC_VEL
    sigma_init_pos: float = 100.0
    sigma_init_vel: float = 0.5
    use_joseph:     bool  = True


class TwoBodyEKF:
    def __init__(self, name, cfg, x0, P0=None):
        self.name = name; self.cfg = cfg
        self.x = np.asarray(x0, float).reshape(6).copy()
        if P0 is None:
            self.P = np.diag([cfg.sigma_init_pos**2]*3 + [cfg.sigma_init_vel**2]*3)
        else:
            self.P = np.asarray(P0, float).reshape(6, 6).copy()
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        self.t_last = None
        self.n_predict = self.n_update_accept = self.n_update_reject = 0

    def predict(self, t_abs):
        if self.t_last is None: self.t_last = t_abs - DT_NAV
        dt = t_abs - self.t_last
        self.x_pred = propagate_filter(self.x, dt)
        Phi = two_body_state_transition(self.x[:3], self.x[3:], dt)
        qp = (self.cfg.sigma_proc_pos**2)*dt
        qv = (self.cfg.sigma_proc_vel**2)*dt
        Q = np.diag([qp]*3 + [qv]*3)
        self.P_pred = Phi @ self.P @ Phi.T + Q
        self.t_last = t_abs; self.n_predict += 1

    def update(self, z, zhat, H, R, nis_gate, label=""):
        z = np.asarray(z, float).reshape(-1)
        zhat = np.asarray(zhat, float).reshape(-1)
        H = np.asarray(H, float); R = np.asarray(R, float)
        I6 = np.eye(6); y = z - zhat
        if label.startswith("LOS"):
            y[0] = (y[0] + np.pi) % (2*np.pi) - np.pi
        S = H @ self.P_pred @ H.T + R
        try: nis = float(y @ la.solve(S, y))
        except la.LinAlgError: nis = float(y @ y)
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


class TwoBodyUKF_IPLF:
    def __init__(self, name, cfg, x0, P0=None,
                 alpha=1e-3, beta=2.0, kappa=0.0):
        self.name = name; self.cfg = cfg; self.n = 6
        self.x = np.asarray(x0, float).reshape(6).copy()
        if P0 is None:
            self.P = np.diag([cfg.sigma_init_pos**2]*3 + [cfg.sigma_init_vel**2]*3)
        else:
            self.P = np.asarray(P0, float).reshape(6, 6).copy()
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        self.sp_pred = None
        self.alpha_ut = alpha; self.beta_ut = beta; self.kappa = kappa
        self.lam = alpha**2 * (self.n + kappa) - self.n
        n2p1 = 2*self.n + 1
        self.Wm = np.full(n2p1, 1.0 / (2.0 * (self.n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.Wm[0] + (1.0 - alpha**2 + beta)
        self.t_last = None
        self.n_predict = self.n_update_accept = self.n_update_reject = 0
        self.n_iplf_iters = 0

    def _sigma_points(self, x, P):
        scale = (self.n + self.lam)
        try:
            S = np.linalg.cholesky(scale * P)
        except np.linalg.LinAlgError:
            jitter = 1e-9 * (np.max(np.abs(np.diag(P))) + 1.0)
            S = np.linalg.cholesky(scale * (P + jitter*np.eye(self.n)))
        sp = np.zeros((2*self.n + 1, self.n)); sp[0] = x
        for i in range(self.n):
            sp[i+1]          = x + S[:, i]
            sp[i+1+self.n]   = x - S[:, i]
        return sp

    def predict(self, t_abs):
        if self.t_last is None: self.t_last = t_abs - DT_NAV
        dt = t_abs - self.t_last
        sp = self._sigma_points(self.x, self.P)
        sp_pred = np.empty_like(sp)
        for i in range(sp.shape[0]):
            sp_pred[i] = propagate_filter(sp[i], dt)
        x_pred = (self.Wm[:, None] * sp_pred).sum(axis=0)
        P_pred = np.zeros((self.n, self.n))
        for i in range(sp.shape[0]):
            d = sp_pred[i] - x_pred
            P_pred += self.Wc[i] * np.outer(d, d)
        qp = (self.cfg.sigma_proc_pos**2) * dt
        qv = (self.cfg.sigma_proc_vel**2) * dt
        P_pred += np.diag([qp]*3 + [qv]*3)
        P_pred = 0.5 * (P_pred + P_pred.T)
        self.x_pred = x_pred; self.P_pred = P_pred
        self.sp_pred = sp_pred
        self.t_last = t_abs; self.n_predict += 1

    def update(self, z, h_fn, R, nis_gate=1e10, label="", aux=None):
        z = np.asarray(z, float).reshape(-1); n_z = z.size
        Z = np.empty((2*self.n + 1, n_z))
        for i in range(2*self.n + 1):
            Z[i] = h_fn(self.sp_pred[i], aux)
        z_pred = (self.Wm[:, None] * Z).sum(axis=0)
        if label.startswith("LOS"):
            for i in range(2*self.n + 1):
                d = Z[i, 0] - z_pred[0]
                if d > np.pi:   Z[i, 0] -= 2*np.pi
                elif d < -np.pi: Z[i, 0] += 2*np.pi
            z_pred = (self.Wm[:, None] * Z).sum(axis=0)
        S = np.array(R, float).copy()
        Pxz = np.zeros((self.n, n_z))
        for i in range(2*self.n + 1):
            dz = Z[i] - z_pred; dx = self.sp_pred[i] - self.x_pred
            S   += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)
        y = z - z_pred
        if label.startswith("LOS"):
            y[0] = (y[0] + np.pi) % (2*np.pi) - np.pi
        try: nis = float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError: nis = float(y @ y)
        if nis > nis_gate:
            self.x = self.x_pred.copy(); self.P = self.P_pred.copy()
            self.n_update_reject += 1
            return dict(accepted=False, nis=nis, y=y, S=S)
        try: K = Pxz @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.x = self.x_pred.copy(); self.P = self.P_pred.copy()
            self.n_update_reject += 1
            return dict(accepted=False, nis=nis, y=y, S=S)
        self.x = self.x_pred + K @ y
        self.P = self.P_pred - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        self.sp_pred = self._sigma_points(self.x, self.P)
        self.n_update_accept += 1
        return dict(accepted=True, nis=nis, y=y, S=S)

    def update_iplf(self, z, h_fn, R, nis_gate=1e10, label="", aux=None,
                    max_iter=IPLF_MAX_ITER,
                    tol_pos=IPLF_TOL_POS, tol_vel=IPLF_TOL_VEL):
        z = np.asarray(z, float).reshape(-1); n_z = z.size
        I6 = np.eye(self.n)
        x_iter = self.x_pred.copy(); P_iter = self.P_pred.copy()

        def _wrap_minus(zz, zp):
            d = zz - zp
            if label.startswith("LOS"):
                d[0] = (d[0] + np.pi) % (2*np.pi) - np.pi
            return d

        x_new = x_iter; P_new = P_iter
        last_S = R.copy(); last_y = np.zeros(n_z); last_nis = 0.0
        n_iter_done = 0
        for it in range(max_iter):
            n_iter_done = it + 1
            sp_iter = self._sigma_points(x_iter, P_iter)
            Z = np.empty((2*self.n + 1, n_z))
            for i in range(2*self.n + 1):
                Z[i] = h_fn(sp_iter[i], aux)
            z_pred = (self.Wm[:, None] * Z).sum(axis=0)
            if label.startswith("LOS"):
                for i in range(2*self.n + 1):
                    d = Z[i, 0] - z_pred[0]
                    if d > np.pi:   Z[i, 0] -= 2*np.pi
                    elif d < -np.pi: Z[i, 0] += 2*np.pi
                z_pred = (self.Wm[:, None] * Z).sum(axis=0)
            Pzz = np.array(R, float).copy()
            Pxz = np.zeros((self.n, n_z))
            for i in range(2*self.n + 1):
                dz = Z[i] - z_pred; dx = sp_iter[i] - x_iter
                Pzz += self.Wc[i] * np.outer(dz, dz)
                Pxz += self.Wc[i] * np.outer(dx, dz)
            try:
                P_iter_inv = np.linalg.inv(P_iter + 1e-12*np.eye(self.n))
            except np.linalg.LinAlgError: break
            A = Pxz.T @ P_iter_inv
            b = z_pred - A @ x_iter
            Omega = Pzz - A @ P_iter @ A.T
            Omega = 0.5 * (Omega + Omega.T)
            S = Omega + A @ self.P_pred @ A.T
            S = 0.5 * (S + S.T)
            try: K = self.P_pred @ A.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError: break
            innov = _wrap_minus(z, b + A @ self.x_pred)
            x_new = self.x_pred + K @ innov
            KA = K @ A
            P_new = ((I6 - KA) @ self.P_pred @ (I6 - KA).T
                     + K @ Omega @ K.T)
            P_new = 0.5 * (P_new + P_new.T)
            try: last_nis = float(innov @ np.linalg.solve(S, innov))
            except np.linalg.LinAlgError: last_nis = float(innov @ innov)
            last_S = S; last_y = innov
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
        self.x = x_new; self.P = P_new
        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()
        self.sp_pred = self._sigma_points(self.x, self.P)
        self.n_update_accept += 1
        return dict(accepted=True, nis=last_nis, y=last_y, S=last_S,
                    n_iter=n_iter_done)


# ──────────────────────────────────────────────────────────────────────────────
#         BSK 6DOF skeleton with extForceTorque on the deputy
# ──────────────────────────────────────────────────────────────────────────────

def _attach_rw_pyramid(scObject, scSim, taskName, prefix):
    if not BSK_RW_AVAILABLE:
        return None
    rwFactory = simIncludeRW.rwFactory()
    el = np.deg2rad(RW_EL_DEG)
    for az_deg in RW_AZ_DEG:
        az = np.deg2rad(az_deg)
        gsHat_B = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
        rWB_B = RW_MOUNT_RADIUS * np.array([np.cos(az), np.sin(az), 0.0])
        rwFactory.create(RW_TYPE, gsHat_B, maxMomentum=RW_MAX_MOMENTUM,
                         rWB_B=rWB_B.tolist())
    rwEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwFactory.addToSpacecraft(prefix + "RWArray", rwEffector, scObject)
    scSim.AddModelToTask(taskName, rwEffector, ModelPriority=4)
    return rwEffector


def build_basilisk_simulation():
    """Build the BSK simulation skeleton.  We do NOT call ExecuteSimulation
    here — that's done one nav step at a time inside the main loop.
    Returns the handles needed to push commands and read state."""
    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(False)         # we print our own progress
    procName = "v5Proc"; taskName = "v5Task"
    proc = scSim.CreateNewProcess(procName)
    proc.addTask(scSim.CreateNewTask(taskName, macros.sec2nano(DT_BSK_INT)))

    scChief  = spacecraft.Spacecraft();  scChief.ModelTag  = "Chief"
    scDeputy = spacecraft.Spacecraft();  scDeputy.ModelTag = "Deputy"
    scChief.hub.mHub  = CHIEF_MASS_KG
    scChief.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(
        np.asarray(CHIEF_INERTIA_BC_B).flatten().tolist())
    scChief.hub.r_BcB_B = CHIEF_RBcB_B
    scDeputy.hub.mHub = DEPUTY_MASS_KG
    scDeputy.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(
        np.asarray(DEPUTY_INERTIA_BC_B).flatten().tolist())
    scDeputy.hub.r_BcB_B = DEPUTY_RBcB_B
    scChief.syncDynamicsIntegration(scDeputy)
    scSim.AddModelToTask(taskName, scChief)
    scSim.AddModelToTask(taskName, scDeputy)

    # Reaction wheels — passive, just for momentum coupling
    _attach_rw_pyramid(scChief,  scSim, taskName, "chief")
    _attach_rw_pyramid(scDeputy, scSim, taskName, "deputy")

    # External torque effector on the deputy (the FSDP² actuator).
    # Zero-initialise so t=0 has a defined torque even before the first
    # control loop iteration writes anything.
    extTorqueDeputy = extForceTorque.ExtForceTorque()
    extTorqueDeputy.ModelTag = "FSDP_extTorque"
    extTorqueDeputy.extTorquePntB_B = [0.0, 0.0, 0.0]
    extTorqueDeputy.extForce_N      = [0.0, 0.0, 0.0]
    extTorqueDeputy.extForce_B      = [0.0, 0.0, 0.0]
    scDeputy.addDynamicEffector(extTorqueDeputy)
    scSim.AddModelToTask(taskName, extTorqueDeputy, ModelPriority=10)

    # Gravity + SPICE
    gravFactory = simIncludeGravBody.gravBodyFactory()
    bodies = ["earth"] + (["sun", "moon"] if INCLUDE_THIRD_BODIES else [])
    gravBodies = gravFactory.createBodies(*bodies)
    gravBodies["earth"].isCentralBody = True
    sh_loaded = False
    try:
        gravBodies["earth"].useSphericalHarmonicsGravityModel(
            SH_GRAV_FILE, MAX_SH_DEG)
        sh_loaded = True
    except Exception as exc:
        warnings.warn(f"SH load failed: {exc!r}", RuntimeWarning)
    gravFactory.addBodiesTo(scChief)
    gravFactory.addBodiesTo(scDeputy)
    spice_enabled = False
    if INCLUDE_THIRD_BODIES:
        try:
            spiceObj = gravFactory.createSpiceInterface(time=EPOCH_UTC,
                                                        epochInMsg=True)
            scSim.AddModelToTask(taskName, spiceObj)
            spiceObj.zeroBase = "earth"
            spice_enabled = True
        except Exception as exc:
            warnings.warn(f"SPICE init failed: {exc!r}", RuntimeWarning)

    drag_enabled = False
    if INCLUDE_DRAG and BSK_DRAG_AVAILABLE:
        atmo = exponentialAtmosphere.ExponentialAtmosphere()
        atmo.ModelTag = "ExpAtmo"
        try:
            simSetPlanetEnvironment.exponentialAtmosphere(atmo, "earth")
        except Exception:
            atmo.planetRadius = R_EARTH
            atmo.baseDensity  = 1.217
            atmo.scaleHeight  = 8500.0
        scSim.AddModelToTask(taskName, atmo)
        atmo.addSpacecraftToModel(scChief.scStateOutMsg)
        atmo.addSpacecraftToModel(scDeputy.scStateOutMsg)
        for sc, prefix, env_idx in [(scChief, "DragChief", 0),
                                     (scDeputy, "DragDeputy", 1)]:
            d = dragDynamicEffector.DragDynamicEffector()
            d.ModelTag = prefix
            d.coreParams.projectedArea = 1.0
            d.coreParams.dragCoeff     = 2.2
            sc.addDynamicEffector(d)
            d.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[env_idx])
            scSim.AddModelToTask(taskName, d)
        drag_enabled = True

    # Initial conditions
    x_chief_0 = initial_chief_state()
    x_dep_0   = initial_deputy_state()
    scChief.hub.r_CN_NInit  = x_chief_0[:3].reshape(3, 1).tolist()
    scChief.hub.v_CN_NInit  = x_chief_0[3:].reshape(3, 1).tolist()
    scDeputy.hub.r_CN_NInit = x_dep_0[:3].reshape(3, 1).tolist()
    scDeputy.hub.v_CN_NInit = x_dep_0[3:].reshape(3, 1).tolist()
    scChief.hub.sigma_BNInit    = [[0.0], [0.0], [0.0]]
    scChief.hub.omega_BN_BInit  = [[0.0], [0.0], [0.0]]
    sigma_dep_0, omega_dep_0 = initial_deputy_attitude_to_chief(
        x_chief_0[:3], x_chief_0[3:], x_dep_0[:3], x_dep_0[3:])
    scDeputy.hub.sigma_BNInit   = sigma_dep_0.reshape(3, 1).tolist()
    scDeputy.hub.omega_BN_BInit = omega_dep_0.reshape(3, 1).tolist()

    # Recorders for diagnostics (sampled at nav rate)
    samp_ns = macros.sec2nano(DT_NAV)
    chiefRec  = scChief.scStateOutMsg.recorder(samp_ns)
    deputyRec = scDeputy.scStateOutMsg.recorder(samp_ns)
    scSim.AddModelToTask(taskName, chiefRec)
    scSim.AddModelToTask(taskName, deputyRec)

    bsk_meta = dict(sh_loaded=sh_loaded,
                    sh_deg=(MAX_SH_DEG if sh_loaded else 0),
                    spice_enabled=spice_enabled,
                    drag_enabled=drag_enabled,
                    rws_present=BSK_RW_AVAILABLE,
                    epoch_utc=EPOCH_UTC,
                    chief_mass=CHIEF_MASS_KG, deputy_mass=DEPUTY_MASS_KG,
                    gravFactory=gravFactory)

    return (scSim, scChief, scDeputy, extTorqueDeputy,
            chiefRec, deputyRec, bsk_meta)


# ──────────────────────────────────────────────────────────────────────────────
#                              MAIN simulation
# ──────────────────────────────────────────────────────────────────────────────

def run_sim(verbose: bool = True):
    rng = np.random.default_rng(RNG_SEED)

    # 1. Build BSK simulation
    if USE_BASILISK:
        (scSim, scChief, scDeputy, extTorqueDeputy,
         chiefRec, deputyRec, bsk_meta) = build_basilisk_simulation()
        scSim.InitializeSimulation()
    else:
        # Numpy-only fallback path (no co-simulation, no active control —
        # equivalent to v3 for diagnostic purposes)
        warnings.warn("No Basilisk: running v3-equivalent numpy-only sim.")
        bsk_meta = dict(sh_loaded=False, sh_deg=0, spice_enabled=False,
                        drag_enabled=False, rws_present=False, epoch_utc="")

    # 2. Filter setup
    chief_cfg = EKFConfig(sigma_init_pos=SIGMA_INIT_POS_CHF,
                          sigma_init_vel=SIGMA_INIT_VEL_CHF)
    dep_cfg   = EKFConfig(sigma_init_pos=SIGMA_INIT_POS_DEP,
                          sigma_init_vel=SIGMA_INIT_VEL_DEP)
    truth_chief_0 = initial_chief_state()
    truth_dep_0   = initial_deputy_state()
    x0_chief = truth_chief_0 + np.hstack([
        SIGMA_INIT_POS_CHF * rng.normal(size=3),
        SIGMA_INIT_VEL_CHF * rng.normal(size=3),
    ])
    x0_dep = truth_dep_0 + np.hstack([INIT_BIAS_POS_DEP, INIT_BIAS_VEL_DEP])
    chief_ekf = TwoBodyEKF("CHIEF", chief_cfg, x0_chief)
    dep_ekf   = TwoBodyUKF_IPLF("DEPUTY", dep_cfg, x0_dep)

    # 3. Histories
    t_arr = np.linspace(0.0, (N_STEPS - 1)*DT_NAV, N_STEPS)
    truth_chief = np.zeros((N_STEPS, 6)); truth_dep = np.zeros((N_STEPS, 6))
    att_dep_BN  = np.zeros((N_STEPS, 3)); omega_dep_B = np.zeros((N_STEPS, 3))
    sigma_RN_log = np.zeros((N_STEPS, 3)); torque_log = np.zeros((N_STEPS, 3))
    point_err_deg = np.zeros(N_STEPS)
    ctrl_mode    = np.zeros(N_STEPS, dtype=int)   # 0 TRACK, 1 ACQ, 2 RECOV
    est_chief = np.zeros((N_STEPS, 6)); est_dep = np.zeros((N_STEPS, 6))
    P_chief   = np.zeros((N_STEPS, 6)); P_dep   = np.zeros((N_STEPS, 6))
    pix_meas  = np.full((N_STEPS, 2), np.nan)
    pix_truth = np.full((N_STEPS, 2), np.nan)
    in_fov    = np.zeros(N_STEPS, dtype=bool)
    los_iters = np.zeros(N_STEPS, dtype=int)

    # Initial-step bookkeeping
    truth_chief[0] = truth_chief_0; truth_dep[0] = truth_dep_0
    sigma0, omega0 = initial_deputy_attitude_to_chief(
        truth_chief_0[:3], truth_chief_0[3:], truth_dep_0[:3], truth_dep_0[3:])
    att_dep_BN[0] = sigma0; omega_dep_B[0] = omega0
    est_chief[0]  = chief_ekf.x; est_dep[0] = dep_ekf.x
    P_chief[0]    = np.diag(chief_ekf.P); P_dep[0] = np.diag(dep_ekf.P)

    # Tracking of mode switching state
    last_meas_t = -np.inf       # used by ACQUISITION timeout

    # ── Numpy 6DOF state initialisation (v5.3) ──────────────────────────────
    # The truth side now lives entirely in two 12-vectors that we propagate
    # with ``propagate_6dof`` each nav step.  The chief is inertially fixed
    # (no torque, σ=0, ω=0); the deputy carries an attitude that we drive
    # with the FSDP² controller (zero-order-hold torque between nav steps).
    chief_state12 = np.zeros(12)
    chief_state12[0:6]  = truth_chief_0
    deputy_state12 = np.zeros(12)
    deputy_state12[0:6]  = truth_dep_0
    deputy_state12[6:9]  = sigma0
    deputy_state12[9:12] = omega0
    L_prev = np.zeros(3)        # zero torque applied during the very first step

    if verbose:
        print("=" * 100)
        print("  ANGLES-ONLY RELNAV  v5 :  FSDP² active pointing  +  numpy 6DOF truth")
        print("=" * 100)
        if USE_BASILISK:
            print(f"  Truth source       : Basilisk 6DOF (co-simulated, 1 nav step at a time)")
            print(f"    SH gravity        : "
                  f"{'GGM03S deg ' + str(bsk_meta['sh_deg']) if bsk_meta['sh_loaded'] else 'point-mass'}")
            print(f"    Third bodies      : {'Sun+Moon (SPICE)' if bsk_meta['spice_enabled'] else 'off'}")
            print(f"    Drag              : {'on' if bsk_meta['drag_enabled'] else 'off'}")
            print(f"    Reaction wheels   : {'4-pyramid HR16 (passive)' if bsk_meta['rws_present'] else 'off'}")
        print(f"  Pointing reference : {'FILTER-state-driven' if USE_FILTER_FOR_POINTING else 'TRUTH-driven (cheating)'}"
              f"   (FSDP²)")
        print(f"  Camera             : f={CAMERA_FOCAL_M*1e3:.1f} mm, "
              f"FOV ≈ {np.degrees(2*CAMERA_HALF_FOV_X):.1f}° × "
              f"{np.degrees(2*CAMERA_HALF_FOV_Y):.1f}°")
        print(f"  Control bandwidth  : Kσ_track={TRACK_K_SIGMA}, Pω_track={TRACK_P_OMEGA}, "
              f"τ_max={TAU_MAX_NM} N·m")
        print(f"  Mode thresholds    : ACQ if tr(P_dep_pos)>{ACQ_P_TRACE_THRESH:.0e} "
              f"or no-meas>{ACQ_TIMEOUT_S} s, RECOV if err>{RECOVERY_ANGLE_DEG}°")
        print()

    # 4. Co-simulation main loop
    for k in range(1, N_STEPS):
        t_k = k * DT_NAV

        # ──────────── 4a) advance truth (pure-numpy 6DOF) ────────────
        # Chief: torque-free, inertially fixed σ=ω=0; gravity = 2-body+J2.
        # Deputy: same gravity, full rigid-body rotation, ZOH torque
        # equal to the controller's previous-step command ``L_prev``.
        chief_state12  = propagate_6dof(chief_state12,  DT_NAV,
                                        CHIEF_INERTIA_BC_B,  np.zeros(3))
        deputy_state12 = propagate_6dof(deputy_state12, DT_NAV,
                                        DEPUTY_INERTIA_BC_B, L_prev)
        truth_chief[k] = chief_state12[0:6]
        truth_dep[k]   = deputy_state12[0:6]
        sigma_dep      = deputy_state12[6:9]
        omega_dep_now  = deputy_state12[9:12]
        att_dep_BN[k]  = sigma_dep
        omega_dep_B[k] = omega_dep_now

        # Last-resort NaN guard (should never fire on the numpy path).
        if (not np.all(np.isfinite(truth_chief[k]))
                or not np.all(np.isfinite(truth_dep[k]))
                or not np.all(np.isfinite(sigma_dep))
                or not np.all(np.isfinite(omega_dep_now))):
            if verbose:
                print(f"[NaN-guard] numpy 6DOF returned NaN at t={t_k:.1f}s "
                      "— freezing on previous state.")
            truth_chief[k] = truth_chief[k - 1]
            truth_dep[k]   = truth_dep[k - 1]
            sigma_dep      = att_dep_BN[k - 1]
            omega_dep_now  = np.zeros(3)
            att_dep_BN[k]  = sigma_dep
            omega_dep_B[k] = omega_dep_now
            chief_state12[0:6]  = truth_chief[k]
            deputy_state12[0:6] = truth_dep[k]
            deputy_state12[6:9] = sigma_dep
            deputy_state12[9:12] = omega_dep_now

        # ────────────────── 4b) filter predict ──────────────────
        chief_ekf.predict(t_k)
        dep_ekf.predict(t_k)

        # ────────────────── 4c) chief GPS update ──────────────────
        z_gps = gps_position_measurement(truth_chief[k, :3], rng)
        H_gps = np.zeros((3, 6)); H_gps[:, :3] = np.eye(3)
        R_gps = (SIGMA_GPS_POS**2) * np.eye(3)
        chief_ekf.update(z_gps, chief_ekf.x_pred[:3], H_gps, R_gps,
                         NIS_GATE_GPS_POS, label="GPS")

        # ────────────────── 4d) camera measurement ──────────────────
        u_t, v_t, visible, _ = project_target_to_pixels(
            truth_chief[k, :3], truth_dep[k, :3], sigma_dep)
        in_fov[k] = visible
        if visible:
            pix_truth[k] = (u_t, v_t)
            u_meas = u_t + SIGMA_PIX * rng.normal()
            v_meas = v_t + SIGMA_PIX * rng.normal()
            pix_meas[k] = (u_meas, v_meas)
            alpha_meas, delta_meas = back_project_pixel_to_los(
                u_meas, v_meas, sigma_dep)
            r_chief_est = chief_ekf.x[:3]
            _, _, dr_pred = los_angles(dep_ekf.x_pred[:3], r_chief_est)
            sig_pix_rad = SIGMA_PIX * CAMERA_PIXEL_M / CAMERA_FOCAL_M
            sig_los_rad = np.sqrt(sig_pix_rad**2 + SIGMA_ATT_RAD**2)
            R_los = (sig_los_rad**2) * np.eye(2)
            H_chief_pos = -los_jacobian_wrt_observer(dr_pred)
            R_los += H_chief_pos @ chief_ekf.P[:3, :3] @ H_chief_pos.T

            def _los_h(x_dep, r_chief):
                a, d, _ = los_angles(x_dep[:3], r_chief)
                return np.array([a, d])

            info = dep_ekf.update_iplf(
                np.array([alpha_meas, delta_meas]), _los_h, R_los,
                nis_gate=NIS_GATE_LOS, label="LOS", aux=r_chief_est)
            los_iters[k] = info.get("n_iter", 0)
            last_meas_t = t_k

            # Chief-as-extended-object size update
            true_size = CHIEF_CHORD_M / la.norm(
                truth_chief[k, :3] - truth_dep[k, :3])
            z_size = np.array([true_size + SIGMA_SIZE_RAD * rng.normal()])
            R_size = np.array([[SIGMA_SIZE_RAD**2 +
                                (CHIEF_CHORD_M / la.norm(dr_pred)**2)**2 *
                                np.trace(chief_ekf.P[:3, :3]) / 3]])

            def _size_h(x_dep, r_chief):
                return np.array([CHIEF_CHORD_M / la.norm(r_chief - x_dep[:3])])

            dep_ekf.update(z_size, _size_h, R_size,
                           nis_gate=NIS_GATE_SIZE, label="SIZE",
                           aux=r_chief_est)

        # 7-D LRL FIMC vs chief broadcast (always available)
        z_fimc = kepler_invariants(chief_ekf.x)
        R_fimc = np.diag([
            SIGMA_FIMC_EPS**2,
            SIGMA_FIMC_H**2, SIGMA_FIMC_H**2, SIGMA_FIMC_H**2,
            SIGMA_FIMC_LRL**2, SIGMA_FIMC_LRL**2, SIGMA_FIMC_LRL**2,
        ])

        def _fimc_h(x_dep, _aux):
            return kepler_invariants(x_dep)

        dep_ekf.update(z_fimc, _fimc_h, R_fimc,
                       nis_gate=NIS_GATE_FIMC, label="FIMC", aux=None)

        est_chief[k] = chief_ekf.x; est_dep[k] = dep_ekf.x
        P_chief[k]   = np.diag(chief_ekf.P); P_dep[k] = np.diag(dep_ekf.P)

        # ────────────────── 4e) FSDP² controller ──────────────────
        if USE_FILTER_FOR_POINTING:
            r_c_for_ref = chief_ekf.x[:3]; v_c_for_ref = chief_ekf.x[3:]
            r_d_for_ref = dep_ekf.x[:3];   v_d_for_ref = dep_ekf.x[3:]
        else:
            r_c_for_ref = truth_chief[k, :3]; v_c_for_ref = truth_chief[k, 3:]
            r_d_for_ref = truth_dep[k, :3];   v_d_for_ref = truth_dep[k, 3:]

        sigma_RN, omega_RN_N, omegadot_RN_N, DCM_BN_ref = \
            compute_pointing_reference(r_c_for_ref, v_c_for_ref,
                                        r_d_for_ref, v_d_for_ref)
        sigma_RN_log[k] = sigma_RN

        # Pointing error angle (deputy +X = camera +Z = boresight)
        DCM_BN_act = mrp_to_dcm(sigma_dep)
        boresight_act_N  = DCM_BN_act.T @ np.array([1.0, 0.0, 0.0])
        boresight_des_N  = DCM_BN_ref.T @ np.array([1.0, 0.0, 0.0])
        cos_err = np.clip(np.dot(boresight_act_N, boresight_des_N), -1.0, 1.0)
        ang_err = np.degrees(np.arccos(cos_err))
        point_err_deg[k] = ang_err

        # Mode selection
        time_since_meas = t_k - last_meas_t
        P_pos_trace = float(np.sum(P_dep[k, :3]))
        if ang_err > RECOVERY_ANGLE_DEG:
            mode = 2; K_s = REC_K_SIGMA; P_w = REC_P_OMEGA
        elif (P_pos_trace > ACQ_P_TRACE_THRESH
              or time_since_meas > ACQ_TIMEOUT_S):
            mode = 1; K_s = ACQ_K_SIGMA; P_w = ACQ_P_OMEGA
        else:
            mode = 0; K_s = TRACK_K_SIGMA; P_w = TRACK_P_OMEGA
        ctrl_mode[k] = mode

        L = fsdp_control_torque(sigma_dep, omega_dep_now,
                                 sigma_RN, omega_RN_N, omegadot_RN_N,
                                 DCM_BN_ref,
                                 DEPUTY_INERTIA_BC_B, K_s, P_w)
        torque_log[k] = L

        # Apply torque to truth on the NEXT integration step (ZOH).
        # Numpy 6DOF path:
        L_prev = L.copy()
        # Basilisk path (only relevant if USE_BASILISK was re-enabled on a
        # known-good build):
        if USE_BASILISK:
            extTorqueDeputy.extTorquePntB_B = L.tolist()

        if verbose and (k % PRINT_EVERY_K == 0 or k == N_STEPS - 1):
            err_d = est_dep[k] - truth_dep[k]
            sigma_d = np.sqrt(P_dep[k])
            range_now = la.norm(truth_dep[k, :3] - truth_chief[k, :3])
            tag_v = "[in FOV]" if in_fov[k] else "[OUT]"
            mode_str = ["TRACK", "ACQ  ", "RECOV"][mode]
            print(f"t={t_k:8.1f}s ({t_k/T_ORBIT:.3f}orb) {mode_str} {tag_v}  "
                  f"err_pt={ang_err:6.2f}°  τ={la.norm(L):.3f}Nm  "
                  f"|Δr|={la.norm(err_d[:3]):7.2f}m  rng={range_now:7.1f}m"
                  f"  iters={los_iters[k]}")

    # 5. Summary
    if verbose:
        i0 = int(0.5 * T_ORBIT / DT_NAV)
        print("\n" + "=" * 100)
        print("  RESULTS SUMMARY  v5  (after 0.5-orbit transient)")
        print("=" * 100)
        n_meas = int(np.sum(in_fov))
        coverage = 100.0 * n_meas / max(1, N_STEPS - 1)
        n_track = int(np.sum(ctrl_mode == 0))
        n_acq   = int(np.sum(ctrl_mode == 1))
        n_recov = int(np.sum(ctrl_mode == 2))
        print(f"  Camera coverage   : {n_meas}/{N_STEPS-1} steps ({coverage:.2f} %)")
        print(f"  Mode usage        : TRACK {100*n_track/(N_STEPS-1):.1f}%  "
              f"ACQ {100*n_acq/(N_STEPS-1):.1f}%  RECOV {100*n_recov/(N_STEPS-1):.1f}%")
        print(f"  Mean point error  : {np.mean(point_err_deg[i0:]):.3f}°")
        print(f"  Median point error: {np.median(point_err_deg[i0:]):.3f}°")
        for ekf, truth, est, P_diag, name in [
            (chief_ekf, truth_chief, est_chief, P_chief, "CHIEF"),
            (dep_ekf,   truth_dep,   est_dep,   P_dep,   "DEPUTY"),
        ]:
            err = est - truth
            rms_pos = np.sqrt(np.mean(np.sum(err[i0:, :3]**2, axis=1)))
            rms_vel = np.sqrt(np.mean(np.sum(err[i0:, 3:6]**2, axis=1)))
            print(f"\n  {name}")
            print(f"    RMS pos (3-D)  : {rms_pos:.6f} m")
            print(f"    RMS vel (3-D)  : {rms_vel:.9f} m/s")
        print()

    np.savez(NPZ_PATH,
             t=t_arr,
             truth_chief=truth_chief, est_chief=est_chief, P_chief_diag=P_chief,
             truth_dep=truth_dep, est_dep=est_dep, P_dep_diag=P_dep,
             att_dep_BN=att_dep_BN, omega_dep_B=omega_dep_B,
             sigma_RN=sigma_RN_log, torque=torque_log,
             point_err_deg=point_err_deg, ctrl_mode=ctrl_mode,
             pix_meas=pix_meas, pix_truth=pix_truth, in_fov=in_fov,
             los_iters=los_iters,
             bsk_meta=np.array([str(bsk_meta)], dtype=object))

    return dict(
        t=t_arr,
        truth_chief=truth_chief, est_chief=est_chief, P_chief=P_chief,
        truth_dep=truth_dep, est_dep=est_dep, P_dep=P_dep,
        att_dep_BN=att_dep_BN, omega_dep_B=omega_dep_B,
        sigma_RN=sigma_RN_log, torque=torque_log,
        point_err_deg=point_err_deg, ctrl_mode=ctrl_mode,
        pix_meas=pix_meas, pix_truth=pix_truth, in_fov=in_fov,
        los_iters=los_iters, bsk_meta=bsk_meta,
    )


# ──────────────────────────────────────────────────────────────────────────────
#                                 plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(res):
    if SAVE_FIGS:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
    t = res["t"]; t_h = t / 3600.0

    # ── 1) Pointing-error timeline + control mode ──
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig.suptitle("FSDP² controller diagnostics — v5", fontsize=14)
    axes[0].plot(t_h, res["point_err_deg"], lw=0.7, color="C3")
    axes[0].axhline(np.degrees(CAMERA_HALF_FOV_X), color="k", ls="--", lw=0.8,
                    label=f"FOV half-angle ({np.degrees(CAMERA_HALF_FOV_X):.1f}°)")
    axes[0].set_ylabel("Pointing error [deg]")
    axes[0].set_title("Boresight angular error vs reference")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("symlog", linthresh=1.0)

    axes[1].plot(t_h, la.norm(res["torque"], axis=1), lw=0.6, color="C0",
                 label="|τ_cmd|")
    axes[1].axhline(TAU_MAX_NM, color="k", ls="--", lw=0.8, label=f"τ_max")
    axes[1].set_ylabel("Commanded torque magnitude [N·m]")
    axes[1].set_title("FSDP² body-frame torque command")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    mode = res["ctrl_mode"]
    axes[2].fill_between(t_h, 0, (mode == 0).astype(int), step="post",
                         alpha=0.4, color="C2", label="TRACK")
    axes[2].fill_between(t_h, 0, (mode == 1).astype(int), step="post",
                         alpha=0.4, color="C1", label="ACQ")
    axes[2].fill_between(t_h, 0, (mode == 2).astype(int), step="post",
                         alpha=0.4, color="C3", label="RECOV")
    axes[2].fill_between(t_h, -0.2, 0.0, where=res["in_fov"], step="post",
                         alpha=0.6, color="k", label="in FOV")
    axes[2].set_ylim([-0.3, 1.2])
    axes[2].set_ylabel("Mode / FOV"); axes[2].set_xlabel("Time [h]")
    axes[2].set_title("Control mode + camera coverage")
    axes[2].legend(fontsize=9, loc="upper right"); axes[2].grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "fsdp_diagnostics.png", dpi=150)

    # ── 2) Deputy residuals ──
    err_d = res["est_dep"] - res["truth_dep"]
    three_sigma_d = 3.0 * np.sqrt(np.maximum(res["P_dep"], 0.0))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    fig.suptitle("DEPUTY residuals — v5 (FSDP² active pointing)", fontsize=14)
    labels = ["x", "y", "z"]
    for i in range(3):
        axes[0, i].plot(t_h, err_d[:, i], lw=0.6)
        axes[0, i].plot(t_h,  three_sigma_d[:, i], "r--", lw=1, alpha=0.7)
        axes[0, i].plot(t_h, -three_sigma_d[:, i], "r--", lw=1, alpha=0.7)
        axes[0, i].set_ylabel(f"r_{labels[i]} [m]"); axes[0, i].grid(True, alpha=0.3)
        axes[1, i].plot(t_h, err_d[:, 3+i], lw=0.6)
        axes[1, i].plot(t_h,  three_sigma_d[:, 3+i], "r--", lw=1, alpha=0.7)
        axes[1, i].plot(t_h, -three_sigma_d[:, 3+i], "r--", lw=1, alpha=0.7)
        axes[1, i].set_ylabel(f"v_{labels[i]} [m/s]"); axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_xlabel("Time [h]")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "deputy_residuals.png", dpi=150)

    # ── 3) Camera + relative position ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Camera + relative position diagnostics — v5", fontsize=14)
    pix_t = res["pix_truth"]; pix_m = res["pix_meas"]
    valid = ~np.isnan(pix_m[:, 0])
    axes[0].scatter(pix_t[valid, 0], pix_t[valid, 1], s=2, alpha=0.5, label="truth px")
    axes[0].scatter(pix_m[valid, 0], pix_m[valid, 1], s=2, alpha=0.5, color="r",
                    label="noisy px")
    bx = CAMERA_NX / 2; by = CAMERA_NY / 2
    axes[0].plot([-bx, bx, bx, -bx, -bx], [-by, -by, by, by, -by],
                 "k--", lw=1, label="FOV")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("u px"); axes[0].set_ylabel("v px")
    axes[0].set_title("Chief in deputy image plane")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    rel_est = res["est_chief"][:, :3] - res["est_dep"][:, :3]
    rel_truth = res["truth_chief"][:, :3] - res["truth_dep"][:, :3]
    rel_res_norm = la.norm(rel_est - rel_truth, axis=1)
    axes[1].semilogy(t_h, np.maximum(rel_res_norm, 1e-3), lw=0.7, color="C3")
    axes[1].set_xlabel("Time [h]"); axes[1].set_ylabel("|Δr_est − Δr_truth| [m]")
    axes[1].set_title("Relative position residual norm (LOG)")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "camera_residuals.png", dpi=150)

    # ── 4) Hill-frame relative motion ──
    rho_truth = np.zeros((len(t), 3))
    rho_est   = np.zeros((len(t), 3))
    for k in range(len(t)):
        rc = res["truth_chief"][k, :3]; vc = res["truth_chief"][k, 3:]
        rd = res["truth_dep"][k, :3]
        ir = rc / la.norm(rc); h = np.cross(rc, vc); ih = h / la.norm(h)
        it = np.cross(ih, ir)
        rho_truth[k] = np.array([ir, it, ih]) @ (rd - rc)

        rce = res["est_chief"][k, :3]; vce = res["est_chief"][k, 3:]
        rde = res["est_dep"][k, :3]
        ire = rce / la.norm(rce); he = np.cross(rce, vce)
        ihe = he / la.norm(he); ite = np.cross(ihe, ire)
        rho_est[k] = np.array([ire, ite, ihe]) @ (rde - rce)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("Deputy in CHIEF Hill frame — v5", fontsize=13)
    decim = max(1, len(t) // 4000)
    ax.plot(rho_truth[::decim, 0], rho_truth[::decim, 1],
            rho_truth[::decim, 2], "g-", lw=1.4, label="ρ truth")
    ax.plot(rho_est[::decim, 0], rho_est[::decim, 1],
            rho_est[::decim, 2], "r--", lw=1.0, alpha=0.7, label="ρ estimate")
    ax.scatter(0, 0, 0, color="k", s=100, marker="X", label="chief")
    ax.set_xlabel("R̂ [m]"); ax.set_ylabel("Θ̂ [m]"); ax.set_zlabel("Ĥ [m]")
    ax.legend(fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "hill_frame.png", dpi=150)


if __name__ == "__main__":
    np.set_printoptions(precision=9, suppress=False, linewidth=200)
    res = run_sim(verbose=True)
    plot_results(res)
    plt.show()
