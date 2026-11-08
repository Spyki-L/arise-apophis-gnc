#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_angles_only_relnav_v4_6dof.py
=====================================

Full-6DOF, structured-spacecraft, real-camera elevation of v3.

What's new compared to v3
-------------------------
1.  CHIEF and DEPUTY are now structured rigid bodies (not point masses).
    They have realistic masses (chief = 200 kg, deputy = 100 kg) and
    inertia tensors derived from a simple box-bus + solar-panel model.
2.  Each spacecraft carries a 4-wheel pyramid of reaction wheels
    (Honeywell HR16) — modelled as ``ReactionWheelStateEffector``s, so
    their angular momentum does enter the rigid-body dynamics.  No
    torque is commanded in this v4 (controllers are a v5 add-on); the
    wheels are present but free.
3.  The deputy carries a real camera with pin-hole optics:
        focal length 25 mm, 1024×1024 pixels, 9 µm pixel pitch
        ⇒ FOV ≈ 27° diagonal, ~21° edge-to-edge
    The camera body-to-camera DCM places the bore-sight along the
    deputy +X body axis.
4.  The deputy's initial attitude is computed analytically so the
    camera bore-sight points at the chief at t = 0; its initial body
    rate is set to ≈ −ω_orbit·ẑ (orbit-normal), so the deputy "spins
    with the orbit" and keeps the chief approximately in the frame
    over multi-orbit runs without active attitude control.
5.  The measurement pipeline is now PIXEL-LEVEL.  Each nav step:
        a.  Use BSK truth (r_chief, r_dep, σ_dep) to project the
            chief's centre into the deputy camera frame.
        b.  If the chief is inside the field of view AND in front of
            the bore-sight, emit a pixel coordinate (u, v) corrupted
            with Gaussian pixel noise.
        c.  The filter back-projects (u, v) through the deputy
            attitude truth (with a small attitude-knowledge noise
            contribution) into an inertial LOS unit vector and feeds
            the resulting (α, δ) pair to the existing UKF + LRL-FIMC +
            IPLF — the filter itself is unchanged from v3.
    Frames where the chief is OUTSIDE the FOV produce NO measurement,
    just like a real spacecraft that has lost target lock.
6.  Truth physics from v3 are kept: full GGM03S spherical harmonics
    (degree 70 by default), Sun + Moon as third bodies via SPICE, and
    optional exponential-atmosphere drag.

Why we still keep v3's filter
-----------------------------
The filter knows ONLY 2-body + J2 in its internal dynamics.  All the
new fidelity (J3+, third-body, drag, attitude couplings) becomes
unmodelled physics for the filter — exactly the operational story:
real GNC always fights its own model error, and process-noise Q has to
absorb the gap.

Outputs
-------
    Disk:   simple_angles_only_results_v4_6dof.npz with truth, estimate,
            pixel measurements, FOV-visibility mask, IPLF iters, etc.
    Plots:  v3's panels PLUS:
              - pixel-measurement scatter inside the camera frame
              - FOV-visibility timeline
              - Hill-frame relative trajectory

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
#                            Basilisk imports
# ──────────────────────────────────────────────────────────────────────────────

BSK_AVAILABLE = False
BSK_DRAG_AVAILABLE = False
BSK_RW_AVAILABLE = False
try:
    from Basilisk import __path__ as bsk_pkg_path
    from Basilisk.simulation import (
        spacecraft,
        gravityEffector,
    )
    from Basilisk.utilities import (
        SimulationBaseClass,
        macros,
        orbitalMotion as bsk_orbital,
        simIncludeGravBody,
        unitTestSupport,
        RigidBodyKinematics as rbk,
    )
    BSK_PATH = bsk_pkg_path[0]
    BSK_AVAILABLE = True

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
    BSK_AVAILABLE = False
    BSK_DRAG_AVAILABLE = False
    BSK_RW_AVAILABLE = False
    BSK_PATH = ""
    warnings.warn(
        "Basilisk is not importable in this Python environment "
        f"({exc!r}).  v4 will fall back to a numpy 2-body+J2 truth and "
        "a perfect-attitude camera so the script still runs end-to-end.",
        RuntimeWarning,
    )


# ──────────────────────────────────────────────────────────────────────────────
#                           CONSTANTS / CONFIG
# ──────────────────────────────────────────────────────────────────────────────

# Earth gravity
MU_EARTH    = 3.986004418e14
R_EARTH     = 6378.137e3

# Orbit
ALTITUDE    = 700e3
R_ORBIT     = R_EARTH + ALTITUDE
V_CIRC      = np.sqrt(MU_EARTH / R_ORBIT)
N_ORBIT     = V_CIRC / R_ORBIT
T_ORBIT     = 2 * np.pi / N_ORBIT

# Formation
BASELINE    = 2000.0

# Run length
DT_NAV         = 1.0
DT_BSK_INT     = 0.5
DURATION_S     = 6 * 3600.0
N_STEPS        = int(round(DURATION_S / DT_NAV)) + 1
N_PERIOD       = int(round(T_ORBIT / DT_NAV))

DT_PERIOD_OFFSET = T_ORBIT - N_PERIOD * DT_NAV
APPLY_PERIODIC   = False

# ── Truth-side toggles ───────────────────────────────────────────────────────
USE_BASILISK            = BSK_AVAILABLE
MAX_SH_DEG              = 70
INCLUDE_THIRD_BODIES    = True
INCLUDE_DRAG            = False
EPOCH_UTC               = "2026 May 09 00:00:00"
SH_GRAV_FILE            = (
    BSK_PATH + "/supportData/LocalGravData/GGM03S.txt"
    if BSK_AVAILABLE else ""
)

# ── Spacecraft structures (chief & deputy as real rigid bodies) ──────────────
# Chief: 1.0 × 1.0 × 1.0 m bus + a single 2 m × 1 m solar panel.
# Deputy: 0.6 × 0.6 × 1.0 m bus + a 1.5 m × 0.8 m solar panel + camera bay.
# Inertias are estimated about Bc (centre of mass) in body frame.
CHIEF_MASS_KG      = 200.0
CHIEF_INERTIA_BC_B = np.diag([35.0, 50.0, 60.0]).tolist()    # kg·m²
CHIEF_RBcB_B       = [[0.0], [0.0], [0.0]]

DEPUTY_MASS_KG      = 100.0
DEPUTY_INERTIA_BC_B = np.diag([12.0, 18.0, 22.0]).tolist()   # kg·m²
DEPUTY_RBcB_B       = [[0.0], [0.0], [0.0]]

# ── Reaction-wheel pyramid (Honeywell HR16) on each spacecraft ───────────────
RW_TYPE             = "Honeywell_HR16"
RW_MAX_MOMENTUM     = 50.0       # Nms
RW_EL_DEG           = 40.0       # elevation above body XY plane
RW_AZ_DEG           = [45.0, 135.0, 225.0, 315.0]
RW_MOUNT_RADIUS     = 0.4        # m, distance from body origin

# ── Camera (deputy) ──────────────────────────────────────────────────────────
# Pin-hole optics with body-to-camera DCM that places the bore-sight along
# the deputy +X body axis.  Camera frame convention: +Z = bore-sight,
# +X = image-row direction (right), +Y = image-column direction (up).
CAMERA_FOCAL_M      = 25e-3
CAMERA_PIXEL_M      = 9e-6
CAMERA_NX           = 1024
CAMERA_NY           = 1024
SIGMA_PIX           = 0.2        # [pixels] sub-pixel centroiding noise
SIGMA_ATT_RAD       = 1.0e-4     # [rad] deputy-attitude knowledge error (1σ)
# DCM from body to camera frame: body +X → camera +Z (bore-sight),
# body +Y → camera +X, body +Z → camera +Y.
DCM_CB              = np.array([[0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0]])
# Derived camera angular size (full FOV, edge to edge):
CAMERA_HALF_FOV_X   = np.arctan(CAMERA_NX / 2 * CAMERA_PIXEL_M / CAMERA_FOCAL_M)
CAMERA_HALF_FOV_Y   = np.arctan(CAMERA_NY / 2 * CAMERA_PIXEL_M / CAMERA_FOCAL_M)

# Filter-side gravity model
J2_COEFF          = 1.08262668e-3
INCLUDE_J2_FILTER = True
N_RK4_SUBSTEPS    = 5

# Chief GPS
SIGMA_GPS_POS = 5.0

# Initial estimate biases
SIGMA_INIT_POS_DEP  = 50.0
SIGMA_INIT_VEL_DEP  = 0.05
SIGMA_INIT_POS_CHF  = 50.0
SIGMA_INIT_VEL_CHF  = 0.5
INIT_BIAS_POS_DEP   = np.array([20.0, -15.0, 10.0])
INIT_BIAS_VEL_DEP   = np.array([0.01, -0.008, 0.005])

# Process noise
SIGMA_PROC_POS = 0.10
SIGMA_PROC_VEL = 1e-3

# NIS gates
NIS_GATE_GPS_POS = 100.0
NIS_GATE_LOS     = 100.0
NIS_GATE_SIZE    = 100.0
NIS_GATE_FIMC    = 1e6
NIS_GATE_PERIOD  = 1e6

# Chief-as-extended-object measurement
CHIEF_CHORD_M       = 5.0
SIGMA_SIZE_RAD      = 1.5e-5

# 7-D FIMC σ values
SIGMA_FIMC_EPS      = 80.0
SIGMA_FIMC_H        = 2.0e7
SIGMA_FIMC_LRL      = 5.0e-4

# IPLF iteration parameters
IPLF_MAX_ITER       = 5
IPLF_TOL_POS        = 1e-3
IPLF_TOL_VEL        = 1e-6

# Output cadence
PRINT_EVERY_S = 60.0
PRINT_EVERY_K = int(round(PRINT_EVERY_S / DT_NAV))

OUT_DIR  = Path(__file__).resolve().parent
NPZ_PATH = OUT_DIR / "simple_angles_only_results_v4_6dof.npz"
SAVE_FIGS = False
FIG_DIR  = OUT_DIR / "simple_angles_only_figs_v4_6dof"

RNG_SEED = 1234


# ──────────────────────────────────────────────────────────────────────────────
#                  Filter-side dynamics (2-body + J2)
# ──────────────────────────────────────────────────────────────────────────────

def two_body_accel(r):
    rn = la.norm(r)
    return -MU_EARTH * r / (rn ** 3)

def j2_accel(r):
    x, y, z = r
    rn = la.norm(r)
    if rn < 1e-6:
        return np.zeros(3)
    rn5 = rn**5
    coeff = -1.5 * J2_COEFF * MU_EARTH * (R_EARTH ** 2) / rn5
    s = (z / rn) ** 2
    return coeff * np.array([
        x * (1.0 - 5.0 * s),
        y * (1.0 - 5.0 * s),
        z * (3.0 - 5.0 * s),
    ])

def filter_accel(r):
    if INCLUDE_J2_FILTER:
        return two_body_accel(r) + j2_accel(r)
    return two_body_accel(r)

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
#                Keplerian invariants for FIMC measurement
# ──────────────────────────────────────────────────────────────────────────────

def kepler_invariants(state6):
    r = state6[:3]; v = state6[3:]
    eps = 0.5 * np.dot(v, v) - MU_EARTH / la.norm(r)
    h = np.cross(r, v)
    e = np.cross(v, h) / MU_EARTH - r / la.norm(r)
    return np.hstack([[eps], h, e])


# ──────────────────────────────────────────────────────────────────────────────
#                         Measurement models
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
#                     Pin-hole camera projection (Python)
# ──────────────────────────────────────────────────────────────────────────────

def mrp_to_dcm(sigma):
    """MRP → DCM (body-from-inertial), Schaub & Junkins convention."""
    sigma = np.asarray(sigma, float).reshape(3)
    s2 = float(np.dot(sigma, sigma))
    s_tilde = np.array([[ 0.0,        -sigma[2],   sigma[1]],
                        [ sigma[2],    0.0,       -sigma[0]],
                        [-sigma[1],    sigma[0],   0.0     ]])
    I = np.eye(3)
    num = (8.0 * s_tilde @ s_tilde) - 4.0 * (1.0 - s2) * s_tilde
    den = (1.0 + s2) ** 2
    return I + num / den

def dcm_to_mrp(C):
    """DCM (body-from-inertial) → MRP, short rotation."""
    if BSK_AVAILABLE:
        return np.asarray(rbk.C2MRP(C)).reshape(3)
    # Fallback: from quaternion shortest path then to MRP
    tr = np.trace(C)
    q = np.zeros(4)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        q[0] = 0.25 * S
        q[1] = (C[2, 1] - C[1, 2]) / S
        q[2] = (C[0, 2] - C[2, 0]) / S
        q[3] = (C[1, 0] - C[0, 1]) / S
    else:
        i = np.argmax(np.diag(C))
        if i == 0:
            S = np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2]) * 2.0
            q[0] = (C[2, 1] - C[1, 2]) / S
            q[1] = 0.25 * S
            q[2] = (C[0, 1] + C[1, 0]) / S
            q[3] = (C[0, 2] + C[2, 0]) / S
        elif i == 1:
            S = np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2]) * 2.0
            q[0] = (C[0, 2] - C[2, 0]) / S
            q[1] = (C[0, 1] + C[1, 0]) / S
            q[2] = 0.25 * S
            q[3] = (C[1, 2] + C[2, 1]) / S
        else:
            S = np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1]) * 2.0
            q[0] = (C[1, 0] - C[0, 1]) / S
            q[1] = (C[0, 2] + C[2, 0]) / S
            q[2] = (C[1, 2] + C[2, 1]) / S
            q[3] = 0.25 * S
    if q[0] < 0:
        q = -q
    return q[1:] / (1.0 + q[0])

def project_target_to_pixels(r_chief_N, r_dep_N, sigma_dep_BN):
    """
    Pin-hole projection of the chief into the deputy's camera frame.
    Returns (u, v, in_fov, dr_C) where (u, v) are pixel coordinates with
    (0, 0) at the principal point and in_fov is True iff the chief is
    within both the image extents AND in front of the bore-sight.
    """
    DCM_BN = mrp_to_dcm(sigma_dep_BN)
    DCM_CN = DCM_CB @ DCM_BN
    dr_N = np.asarray(r_chief_N) - np.asarray(r_dep_N)
    dr_C = DCM_CN @ dr_N
    if dr_C[2] <= 0.0:
        return 0.0, 0.0, False, dr_C
    u = CAMERA_FOCAL_M * dr_C[0] / dr_C[2] / CAMERA_PIXEL_M   # [pixels]
    v = CAMERA_FOCAL_M * dr_C[1] / dr_C[2] / CAMERA_PIXEL_M
    in_fov = (abs(u) < CAMERA_NX / 2) and (abs(v) < CAMERA_NY / 2)
    return u, v, in_fov, dr_C


def back_project_pixel_to_los(u_pix, v_pix, sigma_dep_BN):
    """
    Convert a (u, v) pixel measurement back to an inertial LOS unit
    vector using the deputy's attitude.  Returns the (α, δ) angle pair
    that the existing UKF LOS measurement function expects.
    """
    DCM_BN = mrp_to_dcm(sigma_dep_BN)
    DCM_CN = DCM_CB @ DCM_BN
    DCM_NC = DCM_CN.T
    # LOS in camera frame
    los_C = np.array([u_pix * CAMERA_PIXEL_M / CAMERA_FOCAL_M,
                      v_pix * CAMERA_PIXEL_M / CAMERA_FOCAL_M,
                      1.0])
    los_C = los_C / la.norm(los_C)
    los_N = DCM_NC @ los_C
    alpha = np.arctan2(los_N[1], los_N[0])
    delta = np.arcsin(np.clip(los_N[2], -1.0, 1.0))
    return alpha, delta


def initial_deputy_attitude_to_chief(r_chief, r_dep, v_chief):
    """
    Build the deputy MRP that places the camera bore-sight along
    (r_chief − r_dep), with image rows roughly cross-track.

    Returns
    -------
    sigma_BN_dep    : MRP (3-vector) that takes inertial → deputy body
    omega_BN_B_dep  : initial body rate that approximately tracks the
                      chief's apparent motion (≈ −ω_orbit·ẑ_body)
    """
    z_c_N = np.asarray(r_chief) - np.asarray(r_dep)
    z_c_N = z_c_N / la.norm(z_c_N)
    # Pick image-up (camera +Y) as orbit-normal — this fixes the rotation
    # about the bore-sight.
    h_N = np.cross(np.asarray(r_chief), np.asarray(v_chief))
    h_N = h_N / la.norm(h_N)
    y_c_N = np.cross(h_N, z_c_N)
    if la.norm(y_c_N) < 1e-9:
        # Edge case: bore-sight parallel to orbit normal — pick anything.
        y_c_N = np.cross(np.array([0.0, 0.0, 1.0]), z_c_N)
    y_c_N = y_c_N / la.norm(y_c_N)
    x_c_N = np.cross(y_c_N, z_c_N)
    DCM_NC = np.column_stack([x_c_N, y_c_N, z_c_N])
    DCM_CN = DCM_NC.T
    DCM_BN = DCM_CB.T @ DCM_CN
    sigma_BN = dcm_to_mrp(DCM_BN)

    # Initial body rate: orbit angular rate, expressed in body frame.
    omega_orbit_N = -N_ORBIT * h_N
    omega_BN_B    = DCM_BN @ omega_orbit_N
    return sigma_BN, omega_BN_B


# ──────────────────────────────────────────────────────────────────────────────
#                  EKF (chief) and UKF + IPLF (deputy)  — same as v3
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
        self.sp_pred = None; self.P_cross = np.zeros((6, 6))
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
            sp[i+1] = x + S[:, i]
            sp[i+1+self.n] = x - S[:, i]
        return sp

    def predict(self, t_abs):
        if self.t_last is None: self.t_last = t_abs - DT_NAV
        dt = t_abs - self.t_last
        sp = self._sigma_points(self.x, self.P)
        sp_pred = np.empty_like(sp)
        for i in range(sp.shape[0]):
            sp_pred[i] = propagate_filter(sp[i], dt)
        x_pred = (self.Wm[:, None] * sp_pred).sum(axis=0)
        P_pred  = np.zeros((self.n, self.n))
        P_cross = np.zeros((self.n, self.n))
        x_post = self.x.copy()
        for i in range(sp.shape[0]):
            d_pred = sp_pred[i] - x_pred
            P_pred  += self.Wc[i] * np.outer(d_pred, d_pred)
            P_cross += self.Wc[i] * np.outer(sp[i] - x_post, d_pred)
        qp = (self.cfg.sigma_proc_pos**2) * dt
        qv = (self.cfg.sigma_proc_vel**2) * dt
        P_pred += np.diag([qp]*3 + [qv]*3)
        P_pred = 0.5 * (P_pred + P_pred.T)
        self.x_pred = x_pred; self.P_pred = P_pred
        self.sp_pred = sp_pred; self.P_cross = P_cross
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
            except np.linalg.LinAlgError:
                break
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
#                          Initial conditions
# ──────────────────────────────────────────────────────────────────────────────

def initial_chief_state():
    return np.array([R_ORBIT, 0.0, 0.0,    0.0, V_CIRC, 0.0])

def initial_deputy_state():
    A = BASELINE; B = BASELINE
    rho_H    = np.array([-0.5*A, 0.0, B])
    rhodot_H = np.array([0.0,   N_ORBIT*A, 0.0])
    omega = np.array([0.0, 0.0, N_ORBIT])
    chief = initial_chief_state()
    r_dep = chief[:3] + rho_H
    v_dep = chief[3:] + rhodot_H + np.cross(omega, rho_H)
    return np.hstack([r_dep, v_dep])


# ──────────────────────────────────────────────────────────────────────────────
#                  BSK 6DOF truth simulation (chief + deputy)
# ──────────────────────────────────────────────────────────────────────────────

def _attach_rw_pyramid(scObject, scSim, taskName, prefix):
    """Attach a 4-RW pyramid (Honeywell HR16) to a spacecraft.  Wheels
    are passive in v4 (no motor torque commands); they still couple
    angular momentum into the rigid-body dynamics."""
    if not BSK_RW_AVAILABLE:
        return None, None
    rwFactory = simIncludeRW.rwFactory()
    el = np.deg2rad(RW_EL_DEG)
    for az_deg in RW_AZ_DEG:
        az = np.deg2rad(az_deg)
        gsHat_B = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ])
        rWB_B = RW_MOUNT_RADIUS * np.array([np.cos(az), np.sin(az), 0.0])
        rwFactory.create(RW_TYPE,
                         gsHat_B,
                         maxMomentum=RW_MAX_MOMENTUM,
                         rWB_B=rWB_B.tolist())
    rwEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwFactory.addToSpacecraft(prefix + "RWArray", rwEffector, scObject)
    scSim.AddModelToTask(taskName, rwEffector, ModelPriority=4)
    return rwFactory, rwEffector


def run_basilisk_truth_6dof(verbose: bool = True):
    """
    Build a full-6DOF BSK simulation of chief + deputy and return:

        t_arr           : (N_STEPS,)         seconds since start
        truth_chief     : (N_STEPS, 6)       ECI position+velocity
        truth_dep       : (N_STEPS, 6)       ECI position+velocity
        att_chief_BN    : (N_STEPS, 3)       chief MRP (inertial→body)
        att_dep_BN      : (N_STEPS, 3)       deputy MRP (inertial→body)
        omega_dep_B     : (N_STEPS, 3)       deputy body-frame rate
        bsk_meta        : dict
    """
    if not USE_BASILISK:
        raise RuntimeError("Basilisk not available")

    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(True)
    procName = "v4TruthProc"; taskName = "v4TruthTask"
    dynProcess = scSim.CreateNewProcess(procName)
    dt_int_ns = macros.sec2nano(DT_BSK_INT)
    dynProcess.addTask(scSim.CreateNewTask(taskName, dt_int_ns))

    # ── chief 6DOF ──
    scChief = spacecraft.Spacecraft()
    scChief.ModelTag = "Chief"
    scChief.hub.mHub = CHIEF_MASS_KG
    scChief.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(
        np.asarray(CHIEF_INERTIA_BC_B).flatten().tolist())
    scChief.hub.r_BcB_B = CHIEF_RBcB_B

    # ── deputy 6DOF ──
    scDeputy = spacecraft.Spacecraft()
    scDeputy.ModelTag = "Deputy"
    scDeputy.hub.mHub = DEPUTY_MASS_KG
    scDeputy.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(
        np.asarray(DEPUTY_INERTIA_BC_B).flatten().tolist())
    scDeputy.hub.r_BcB_B = DEPUTY_RBcB_B

    scChief.syncDynamicsIntegration(scDeputy)
    scSim.AddModelToTask(taskName, scChief)
    scSim.AddModelToTask(taskName, scDeputy)

    # ── reaction wheels (passive — no torque commanded) ──
    rwFacChief, rwEffChief   = _attach_rw_pyramid(scChief,  scSim, taskName, "chief")
    rwFacDeputy, rwEffDeputy = _attach_rw_pyramid(scDeputy, scSim, taskName, "deputy")

    # ── gravity ──
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
        warnings.warn(f"SH load failed: {exc!r}; using point-mass.",
                      RuntimeWarning)
    gravFactory.addBodiesTo(scChief)
    gravFactory.addBodiesTo(scDeputy)

    # ── SPICE ephemeris ──
    spice_enabled = False
    if INCLUDE_THIRD_BODIES:
        try:
            spiceObj = gravFactory.createSpiceInterface(
                time=EPOCH_UTC, epochInMsg=True)
            scSim.AddModelToTask(taskName, spiceObj)
            spiceObj.zeroBase = "earth"
            spice_enabled = True
        except Exception as exc:
            warnings.warn(f"SPICE init failed: {exc!r}", RuntimeWarning)

    # ── drag (optional) ──
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

    # ── initial conditions ──
    x_chief_0 = initial_chief_state()
    x_dep_0   = initial_deputy_state()

    # Translational ICs
    scChief.hub.r_CN_NInit  = x_chief_0[:3].reshape(3, 1).tolist()
    scChief.hub.v_CN_NInit  = x_chief_0[3:].reshape(3, 1).tolist()
    scDeputy.hub.r_CN_NInit = x_dep_0[:3].reshape(3, 1).tolist()
    scDeputy.hub.v_CN_NInit = x_dep_0[3:].reshape(3, 1).tolist()

    # Chief attitude: inertially fixed, zero rate.
    scChief.hub.sigma_BNInit    = [[0.0], [0.0], [0.0]]
    scChief.hub.omega_BN_BInit  = [[0.0], [0.0], [0.0]]

    # Deputy attitude: bore-sight at chief at t=0, body rate ≈ −ω_orbit·ẑ
    sigma_dep_0, omega_dep_0 = initial_deputy_attitude_to_chief(
        x_chief_0[:3], x_dep_0[:3], x_chief_0[3:])
    scDeputy.hub.sigma_BNInit    = sigma_dep_0.reshape(3, 1).tolist()
    scDeputy.hub.omega_BN_BInit  = omega_dep_0.reshape(3, 1).tolist()

    # ── recorders sampled at the nav rate ──
    samp_ns = macros.sec2nano(DT_NAV)
    chiefRec  = scChief.scStateOutMsg.recorder(samp_ns)
    deputyRec = scDeputy.scStateOutMsg.recorder(samp_ns)
    scSim.AddModelToTask(taskName, chiefRec)
    scSim.AddModelToTask(taskName, deputyRec)

    # ── run ──
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(macros.sec2nano(DURATION_S))
    if verbose:
        print(f"[BSK 6DOF] propagating {DURATION_S/3600:.2f} h "
              f"@ dt_int={DT_BSK_INT:.3f} s "
              f"(SH deg={MAX_SH_DEG if sh_loaded else 'point-mass'}, "
              f"third-bodies={spice_enabled}, drag={drag_enabled}, "
              f"RWs={'on' if BSK_RW_AVAILABLE else 'off'}) …")
    scSim.ExecuteSimulation()

    try:
        gravFactory.unloadSpiceKernels()
    except Exception:
        pass

    # ── extract ──
    pos_c = np.asarray(chiefRec.r_BN_N);  vel_c = np.asarray(chiefRec.v_BN_N)
    pos_d = np.asarray(deputyRec.r_BN_N); vel_d = np.asarray(deputyRec.v_BN_N)
    sig_c = np.asarray(chiefRec.sigma_BN)
    sig_d = np.asarray(deputyRec.sigma_BN)
    om_d  = np.asarray(deputyRec.omega_BN_B)

    n_avail = min(len(pos_c), len(pos_d), N_STEPS)
    t_arr = np.linspace(0.0, (N_STEPS - 1)*DT_NAV, N_STEPS)
    truth_chief = np.zeros((N_STEPS, 6))
    truth_dep   = np.zeros((N_STEPS, 6))
    att_chief_BN = np.zeros((N_STEPS, 3))
    att_dep_BN   = np.zeros((N_STEPS, 3))
    omega_dep_B  = np.zeros((N_STEPS, 3))
    truth_chief[:n_avail] = np.hstack([pos_c[:n_avail], vel_c[:n_avail]])
    truth_dep[:n_avail]   = np.hstack([pos_d[:n_avail], vel_d[:n_avail]])
    att_chief_BN[:n_avail] = sig_c[:n_avail]
    att_dep_BN[:n_avail]   = sig_d[:n_avail]
    omega_dep_B[:n_avail]  = om_d[:n_avail]
    if n_avail < N_STEPS:
        truth_chief[n_avail:]  = truth_chief[n_avail-1]
        truth_dep[n_avail:]    = truth_dep[n_avail-1]
        att_chief_BN[n_avail:] = att_chief_BN[n_avail-1]
        att_dep_BN[n_avail:]   = att_dep_BN[n_avail-1]
        omega_dep_B[n_avail:]  = omega_dep_B[n_avail-1]

    bsk_meta = dict(
        sh_loaded=sh_loaded, sh_deg=(MAX_SH_DEG if sh_loaded else 0),
        spice_enabled=spice_enabled, drag_enabled=drag_enabled,
        rws_present=BSK_RW_AVAILABLE, epoch_utc=EPOCH_UTC,
        chief_mass=CHIEF_MASS_KG, deputy_mass=DEPUTY_MASS_KG,
    )
    return (t_arr, truth_chief, truth_dep,
            att_chief_BN, att_dep_BN, omega_dep_B, bsk_meta)


# ──────────────────────────────────────────────────────────────────────────────
#         Numpy-fallback truth (used only if BSK is not available)
# ──────────────────────────────────────────────────────────────────────────────

def run_numpy_truth_6dof(verbose: bool = True):
    """Fallback: numpy 2-body+J2 translation + analytic perfect-pointing
    deputy attitude.  Loses 6DOF realism but keeps the script runnable."""
    if verbose:
        print("[NUMPY-FALLBACK] propagating truth with 2-body+J2 RK4 + "
              "analytic deputy pointing …")
    t_arr = np.linspace(0.0, (N_STEPS - 1) * DT_NAV, N_STEPS)
    truth_chief = np.zeros((N_STEPS, 6))
    truth_dep   = np.zeros((N_STEPS, 6))
    truth_chief[0] = initial_chief_state()
    truth_dep[0]   = initial_deputy_state()
    for k in range(1, N_STEPS):
        truth_chief[k] = propagate_filter(truth_chief[k-1], DT_NAV)
        truth_dep[k]   = propagate_filter(truth_dep[k-1],   DT_NAV)

    # Analytic deputy attitude: perfect chief-pointing every step.
    att_chief_BN = np.zeros((N_STEPS, 3))
    att_dep_BN   = np.zeros((N_STEPS, 3))
    omega_dep_B  = np.zeros((N_STEPS, 3))
    for k in range(N_STEPS):
        sig, om = initial_deputy_attitude_to_chief(
            truth_chief[k, :3], truth_dep[k, :3], truth_chief[k, 3:])
        att_dep_BN[k] = sig
        omega_dep_B[k] = om
    bsk_meta = dict(sh_loaded=False, sh_deg=0, spice_enabled=False,
                    drag_enabled=False, rws_present=False, epoch_utc="",
                    chief_mass=CHIEF_MASS_KG, deputy_mass=DEPUTY_MASS_KG)
    return (t_arr, truth_chief, truth_dep,
            att_chief_BN, att_dep_BN, omega_dep_B, bsk_meta)


# ──────────────────────────────────────────────────────────────────────────────
#                              MAIN simulation
# ──────────────────────────────────────────────────────────────────────────────

def run_sim(verbose: bool = True):
    rng = np.random.default_rng(RNG_SEED)

    # 1) Truth (BSK 6DOF if available)
    if USE_BASILISK:
        (t_arr, truth_chief, truth_dep,
         att_chief_BN, att_dep_BN, omega_dep_B, bsk_meta) = \
            run_basilisk_truth_6dof(verbose)
    else:
        (t_arr, truth_chief, truth_dep,
         att_chief_BN, att_dep_BN, omega_dep_B, bsk_meta) = \
            run_numpy_truth_6dof(verbose)

    # 2) Filter setup
    chief_cfg = EKFConfig(sigma_init_pos=SIGMA_INIT_POS_CHF,
                          sigma_init_vel=SIGMA_INIT_VEL_CHF)
    dep_cfg   = EKFConfig(sigma_init_pos=SIGMA_INIT_POS_DEP,
                          sigma_init_vel=SIGMA_INIT_VEL_DEP)
    x0_chief = truth_chief[0] + np.hstack([
        SIGMA_INIT_POS_CHF * rng.normal(size=3),
        SIGMA_INIT_VEL_CHF * rng.normal(size=3),
    ])
    x0_dep = truth_dep[0] + np.hstack([INIT_BIAS_POS_DEP, INIT_BIAS_VEL_DEP])
    chief_ekf = TwoBodyEKF("CHIEF", chief_cfg, x0_chief)
    dep_ekf   = TwoBodyUKF_IPLF("DEPUTY", dep_cfg, x0_dep)

    # 3) Histories
    est_chief = np.zeros((N_STEPS, 6)); est_dep = np.zeros((N_STEPS, 6))
    P_chief   = np.zeros((N_STEPS, 6)); P_dep   = np.zeros((N_STEPS, 6))
    gps_innov = np.full((N_STEPS, 3), np.nan)
    los_innov = np.full((N_STEPS, 2), np.nan)
    pix_meas  = np.full((N_STEPS, 2), np.nan)
    pix_truth = np.full((N_STEPS, 2), np.nan)
    in_fov    = np.zeros(N_STEPS, dtype=bool)
    los_iters = np.zeros(N_STEPS, dtype=int)
    accept_chief = np.zeros(N_STEPS, dtype=bool)
    accept_dep   = np.zeros(N_STEPS, dtype=bool)

    sm_x_post = np.zeros((N_STEPS, 6)); sm_P_post = np.zeros((N_STEPS, 6, 6))
    sm_x_pred = np.zeros((N_STEPS, 6)); sm_P_pred = np.zeros((N_STEPS, 6, 6))
    sm_P_cross = np.zeros((N_STEPS, 6, 6))

    est_chief[0] = chief_ekf.x; est_dep[0] = dep_ekf.x
    P_chief[0]   = np.diag(chief_ekf.P); P_dep[0] = np.diag(dep_ekf.P)
    sm_x_post[0] = dep_ekf.x.copy(); sm_P_post[0] = dep_ekf.P.copy()
    sm_x_pred[0] = dep_ekf.x.copy(); sm_P_pred[0] = dep_ekf.P.copy()
    sm_P_cross[0] = np.eye(6) * 1e-9

    if verbose:
        print("=" * 100)
        print("  ANGLES-ONLY RELNAV  v4 :  6DOF + structured spacecraft + real camera")
        print("=" * 100)
        if USE_BASILISK:
            print(f"  Truth source         : Basilisk 6DOF")
            print(f"    SH gravity         : "
                  f"{'GGM03S deg ' + str(bsk_meta['sh_deg']) if bsk_meta['sh_loaded'] else 'point-mass'}")
            print(f"    Third bodies       : {'Sun + Moon (SPICE)' if bsk_meta['spice_enabled'] else 'off'}")
            print(f"    Drag               : {'on' if bsk_meta['drag_enabled'] else 'off'}")
            print(f"    Reaction wheels    : {'4-pyramid HR16 on each SC (passive)' if bsk_meta['rws_present'] else 'off'}")
            print(f"    Epoch (UTC)        : {bsk_meta['epoch_utc']}")
        else:
            print(f"  Truth source         : numpy fallback (2-body+J2, perfect attitude)")
        print(f"  Chief mass / inertia : {CHIEF_MASS_KG:.1f} kg / diag(35,50,60) kg·m²")
        print(f"  Deputy mass / inertia: {DEPUTY_MASS_KG:.1f} kg / diag(12,18,22) kg·m²")
        print(f"  Camera: f={CAMERA_FOCAL_M*1e3:.1f} mm, pix={CAMERA_PIXEL_M*1e6:.1f} µm, "
              f"{CAMERA_NX}×{CAMERA_NY} px,  FOV ≈ "
              f"{np.degrees(2*CAMERA_HALF_FOV_X):.2f}° × "
              f"{np.degrees(2*CAMERA_HALF_FOV_Y):.2f}°")
        print(f"  σ_pix / σ_attitude   : {SIGMA_PIX:.3f} px / "
              f"{SIGMA_ATT_RAD*1e6:.1f} µrad")
        print(f"  Filter model         : 2-body + J2 (Q absorbs the gap)")
        print(f"  σ_proc_pos / vel     : {SIGMA_PROC_POS:.3f} / {SIGMA_PROC_VEL:.4f}")
        print()

    # 4) Main loop
    for k in range(1, N_STEPS):
        t = t_arr[k]
        x_chief_truth = truth_chief[k]
        x_dep_truth   = truth_dep[k]
        sigma_dep     = att_dep_BN[k]

        chief_ekf.predict(t)
        dep_ekf.predict(t)
        sm_x_pred[k]  = dep_ekf.x_pred.copy()
        sm_P_pred[k]  = dep_ekf.P_pred.copy()
        sm_P_cross[k] = dep_ekf.P_cross.copy()

        # Chief: GPS update
        z_gps = gps_position_measurement(x_chief_truth[:3], rng)
        H_gps = np.zeros((3, 6)); H_gps[:, :3] = np.eye(3)
        R_gps = (SIGMA_GPS_POS**2) * np.eye(3)
        zhat_gps = chief_ekf.x_pred[:3]
        info = chief_ekf.update(z_gps, zhat_gps, H_gps, R_gps,
                                NIS_GATE_GPS_POS, label="GPS")
        accept_chief[k] = info["accepted"]
        gps_innov[k] = info["y"] if info["accepted"] else np.nan

        # ── CAMERA: project chief into deputy's image ─────────────────
        u_t, v_t, visible, dr_C = project_target_to_pixels(
            x_chief_truth[:3], x_dep_truth[:3], sigma_dep)
        in_fov[k] = visible
        if visible:
            pix_truth[k] = (u_t, v_t)
            # pixel noise
            u_meas = u_t + SIGMA_PIX * rng.normal()
            v_meas = v_t + SIGMA_PIX * rng.normal()
            pix_meas[k] = (u_meas, v_meas)

            # ── Back-project to inertial LOS using deputy attitude ──
            # For the back-projection we use the truth attitude (this is
            # the perfect attitude-knowledge case).  An imperfect attitude
            # estimate is folded in by inflating R_los below.
            alpha_meas, delta_meas = back_project_pixel_to_los(
                u_meas, v_meas, sigma_dep)

            r_chief_est = chief_ekf.x[:3]
            _, _, dr_pred = los_angles(dep_ekf.x_pred[:3], r_chief_est)

            # σ_pix in radians:  σ_pix · (CAMERA_PIXEL_M / CAMERA_FOCAL_M)
            sig_pix_rad = SIGMA_PIX * CAMERA_PIXEL_M / CAMERA_FOCAL_M
            # Total measurement σ: pixel + attitude knowledge
            sig_los_rad = np.sqrt(sig_pix_rad**2 + SIGMA_ATT_RAD**2)
            R_los = (sig_los_rad**2) * np.eye(2)
            # Schmidt: also consider chief-pos uncertainty
            H_chief_pos = -los_jacobian_wrt_observer(dr_pred)
            R_los += H_chief_pos @ chief_ekf.P[:3, :3] @ H_chief_pos.T

            def _los_h(x_dep, r_chief):
                a, d, _ = los_angles(x_dep[:3], r_chief)
                return np.array([a, d])

            info = dep_ekf.update_iplf(
                np.array([alpha_meas, delta_meas]), _los_h, R_los,
                nis_gate=NIS_GATE_LOS, label="LOS", aux=r_chief_est)
            accept_dep[k] = info["accepted"]
            los_innov[k]  = info["y"] if info["accepted"] else np.nan
            los_iters[k]  = info.get("n_iter", 0)

            # ── Chief-as-extended-object size measurement ──
            true_size = CHIEF_CHORD_M / la.norm(
                x_chief_truth[:3] - x_dep_truth[:3])
            z_size = np.array([true_size + SIGMA_SIZE_RAD * rng.normal()])
            R_size = np.array([[SIGMA_SIZE_RAD**2 +
                                (CHIEF_CHORD_M / la.norm(dr_pred)**2)**2 *
                                np.trace(chief_ekf.P[:3, :3]) / 3]])

            def _size_h(x_dep, r_chief):
                return np.array([CHIEF_CHORD_M / la.norm(r_chief - x_dep[:3])])

            dep_ekf.update(z_size, _size_h, R_size,
                           nis_gate=NIS_GATE_SIZE, label="SIZE",
                           aux=r_chief_est)
        # else: no measurement this step — filter only propagates.

        # 7-D LRL-FIMC vs chief (always available — uses chief broadcast)
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

        sm_x_post[k] = dep_ekf.x.copy()
        sm_P_post[k] = dep_ekf.P.copy()

        est_chief[k] = chief_ekf.x; est_dep[k] = dep_ekf.x
        P_chief[k]   = np.diag(chief_ekf.P); P_dep[k] = np.diag(dep_ekf.P)

        if verbose and (k % PRINT_EVERY_K == 0 or k == N_STEPS - 1):
            err_d = est_dep[k] - truth_dep[k]
            sigma_d = np.sqrt(P_dep[k])
            range_now = la.norm(truth_dep[k, :3] - truth_chief[k, :3])
            tag_v = "  [in FOV]" if in_fov[k] else "  [OUT OF FOV]"
            print(f"t = {t:11.3f} s ({t/T_ORBIT:.3f} orb){tag_v}")
            print(f"  DEPUTY |Δr| = {la.norm(err_d[:3]):8.3f} m   "
                  f"|Δv| = {la.norm(err_d[3:6])*1e3:8.3f} mm/s   "
                  f"σ_pos = [{sigma_d[0]:.3f}, {sigma_d[1]:.3f}, "
                  f"{sigma_d[2]:.3f}]")
            if in_fov[k]:
                print(f"  Pixel meas  = ({pix_meas[k,0]:+8.2f}, "
                      f"{pix_meas[k,1]:+8.2f}) px  "
                      f"IPLF iters = {los_iters[k]}")
            print(f"  Range chief↔deputy (truth) = {range_now:.3f} m")
            print()

    # 5) URTSS smoother
    smooth_x = sm_x_post.copy(); smooth_P = sm_P_post.copy()
    for k in range(N_STEPS - 2, -1, -1):
        try:
            A = sm_P_cross[k+1] @ la.inv(sm_P_pred[k+1] + 1e-12*np.eye(6))
        except la.LinAlgError:
            continue
        smooth_x[k] = sm_x_post[k] + A @ (smooth_x[k+1] - sm_x_pred[k+1])
        smooth_P[k] = sm_P_post[k] + A @ (smooth_P[k+1] - sm_P_pred[k+1]) @ A.T
        smooth_P[k] = 0.5 * (smooth_P[k] + smooth_P[k].T)

    # 6) Summary
    if verbose:
        i0 = int(0.5 * T_ORBIT / DT_NAV)
        print("\n" + "=" * 100)
        print("  RESULTS SUMMARY  v4-6dof")
        print("=" * 100)
        n_meas = int(np.sum(in_fov))
        coverage = 100.0 * n_meas / max(1, N_STEPS - 1)
        print(f"  Camera coverage   : {n_meas}/{N_STEPS-1} steps ({coverage:.2f} %)")
        print(f"  Measurements taken: {int(np.sum(accept_dep))}")
        for ekf, truth, est, P_diag, name in [
            (chief_ekf, truth_chief, est_chief, P_chief, "CHIEF"),
            (dep_ekf,   truth_dep,   est_dep,   P_dep,   "DEPUTY"),
        ]:
            err = est - truth
            rms_pos = np.sqrt(np.mean(np.sum(err[i0:, :3]**2, axis=1)))
            rms_vel = np.sqrt(np.mean(np.sum(err[i0:, 3:6]**2, axis=1)))
            print(f"\n  {name}")
            print(f"    n_predict       : {ekf.n_predict}")
            print(f"    n_update_accept : {ekf.n_update_accept}")
            print(f"    n_update_reject : {ekf.n_update_reject}")
            if hasattr(ekf, "n_iplf_iters"):
                avg = ekf.n_iplf_iters / max(1, np.count_nonzero(los_iters))
                print(f"    avg IPLF iters  : {avg:.2f}")
            print(f"    RMS pos (3-D)   : {rms_pos:.6f} m")
            print(f"    RMS vel (3-D)   : {rms_vel:.9f} m/s")

    np.savez(NPZ_PATH,
             t=t_arr,
             truth_chief=truth_chief, est_chief=est_chief, P_chief_diag=P_chief,
             truth_dep=truth_dep,     est_dep=est_dep,     P_dep_diag=P_dep,
             att_chief_BN=att_chief_BN, att_dep_BN=att_dep_BN,
             omega_dep_B=omega_dep_B,
             gps_innov=gps_innov, los_innov=los_innov,
             pix_meas=pix_meas, pix_truth=pix_truth, in_fov=in_fov,
             los_iters=los_iters,
             accept_chief=accept_chief, accept_dep=accept_dep,
             smooth_x=smooth_x,
             smooth_P_diag=np.array([np.diag(p) for p in smooth_P]),
             bsk_meta=np.array([str(bsk_meta)], dtype=object))
    if verbose:
        print(f"\n  Saved per-step time series to {NPZ_PATH}\n")

    return dict(
        t=t_arr,
        truth_chief=truth_chief, est_chief=est_chief, P_chief=P_chief,
        truth_dep=truth_dep,     est_dep=est_dep,     P_dep=P_dep,
        att_chief_BN=att_chief_BN, att_dep_BN=att_dep_BN,
        omega_dep_B=omega_dep_B,
        smooth_x=smooth_x,
        smooth_P_diag=np.array([np.diag(p) for p in smooth_P]),
        gps_innov=gps_innov, los_innov=los_innov,
        pix_meas=pix_meas, pix_truth=pix_truth, in_fov=in_fov,
        los_iters=los_iters,
        accept_chief=accept_chief, accept_dep=accept_dep,
        bsk_meta=bsk_meta,
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
    t = res["t"]; t_h = t / 3600.0
    labels = ["x", "y", "z"]

    err_d = res["est_dep"] - res["truth_dep"]
    three_sigma_d = 3.0 * np.sqrt(np.maximum(res["P_dep"], 0.0))

    # ── 1) deputy residuals ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    fig.suptitle("DEPUTY residuals — v4 (6DOF + camera)", fontsize=14)
    for i in range(3):
        _residual_panel(axes[0, i], t_h, err_d[:, i], three_sigma_d[:, i],
                        f"r_{labels[i]} [m]", f"Position {labels[i]}")
        _residual_panel(axes[1, i], t_h, err_d[:, 3+i], three_sigma_d[:, 3+i],
                        f"v_{labels[i]} [m/s]", f"Velocity {labels[i]}")
    for ax in axes[1]: ax.set_xlabel("Time [h]")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "deputy_residuals.png", dpi=150)

    # ── 2) deputy convergence ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DEPUTY error magnitude — v4", fontsize=14)
    err_pos_norm = la.norm(err_d[:, :3], axis=1)
    err_vel_norm = la.norm(err_d[:, 3:6], axis=1)
    sigma_pos_norm = np.sqrt(np.sum(np.maximum(res["P_dep"][:, :3], 0.0), axis=1))
    sigma_vel_norm = np.sqrt(np.sum(np.maximum(res["P_dep"][:, 3:6], 0.0), axis=1))
    err_smooth = res["smooth_x"] - res["truth_dep"]
    sm_norm = la.norm(err_smooth[:, :3], axis=1)
    sm_v    = la.norm(err_smooth[:, 3:], axis=1)
    sm_sigma = np.sqrt(np.sum(np.maximum(res["smooth_P_diag"][:, :3], 0.0), axis=1))
    sm_v_sig = np.sqrt(np.sum(np.maximum(res["smooth_P_diag"][:, 3:], 0.0), axis=1))
    ax1.plot(t_h, err_pos_norm, lw=0.7, label="filter |error|", color="C0")
    ax1.plot(t_h, 3*sigma_pos_norm, "r--", lw=1, label="filter 3σ")
    ax1.plot(t_h, sm_norm, lw=1.0, color="C2", label="smoother |error|")
    ax1.plot(t_h, 3*sm_sigma, "g--", lw=1, alpha=0.6, label="smoother 3σ")
    ax1.set_yscale("log"); ax1.set_ylabel("|r_error| [m]")
    ax1.set_xlabel("Time [h]"); ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3); ax1.set_title("Position")
    ax2.plot(t_h, err_vel_norm, lw=0.7, label="filter |error|", color="C0")
    ax2.plot(t_h, 3*sigma_vel_norm, "r--", lw=1, label="filter 3σ")
    ax2.plot(t_h, sm_v, lw=1.0, color="C2", label="smoother |error|")
    ax2.plot(t_h, 3*sm_v_sig, "g--", lw=1, alpha=0.6, label="smoother 3σ")
    ax2.set_yscale("log"); ax2.set_ylabel("|v_error| [m/s]")
    ax2.set_xlabel("Time [h]"); ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3); ax2.set_title("Velocity")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "deputy_convergence.png", dpi=150)

    # ── 3) Camera scatter (chief in deputy's image) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Camera measurement diagnostics (v4)", fontsize=14)
    pix_t = res["pix_truth"]; pix_m = res["pix_meas"]
    in_fov = res["in_fov"].astype(bool)
    valid = ~np.isnan(pix_m[:, 0])
    axes[0].scatter(pix_t[valid, 0], pix_t[valid, 1], s=3, alpha=0.5,
                    label="truth pixel")
    axes[0].scatter(pix_m[valid, 0], pix_m[valid, 1], s=3, alpha=0.5,
                    color="r", label="noisy meas.")
    # Draw FOV box (centered at 0)
    box_x = CAMERA_NX / 2; box_y = CAMERA_NY / 2
    axes[0].plot([-box_x, box_x, box_x, -box_x, -box_x],
                 [-box_y, -box_y, box_y, box_y, -box_y],
                 "k--", lw=1, label="FOV bounds")
    axes[0].set_xlabel("u (pixels, 0 = principal point)")
    axes[0].set_ylabel("v (pixels, 0 = principal point)")
    axes[0].set_title("Chief in deputy's image plane")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(t_h, 0, in_fov.astype(int), step="post",
                         alpha=0.4, color="C2", label="in FOV")
    axes[1].set_xlabel("Time [h]"); axes[1].set_ylabel("Visibility (0/1)")
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].set_title("FOV-visibility timeline")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "camera_diagnostics.png", dpi=150)

    # ── 4) Relative position residual (carry-over from v3) ──
    rel_est   = res["est_chief"][:, :3]   - res["est_dep"][:, :3]
    rel_truth = res["truth_chief"][:, :3] - res["truth_dep"][:, :3]
    rel_res   = rel_est - rel_truth
    rel_res_norm = la.norm(rel_res, axis=1)
    rel_truth_norm = la.norm(rel_truth, axis=1)
    rel_est_norm   = la.norm(rel_est, axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Relative position estimate convergence — v4", fontsize=14)
    for j, lab in enumerate(["x", "y", "z"]):
        axes[0, 0].plot(t_h, rel_res[:, j], lw=0.6, alpha=0.85,
                        label=f"Δr_{lab}")
    axes[0, 0].axhline(0, color="k", lw=0.5)
    axes[0, 0].set_xlabel("Time [h]"); axes[0, 0].set_ylabel("Δr_est − Δr_truth [m]")
    axes[0, 0].set_title("Relative position residual per axis")
    axes[0, 0].legend(fontsize=9); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].semilogy(t_h, np.maximum(rel_res_norm, 1e-3), lw=0.7, color="C3")
    axes[0, 1].set_xlabel("Time [h]"); axes[0, 1].set_ylabel("|Δr_est − Δr_truth| [m]")
    axes[0, 1].set_title("Relative position residual norm (LOG)")
    axes[0, 1].grid(True, which="both", alpha=0.3)
    axes[1, 0].plot(t_h, rel_truth_norm, lw=0.7, label="|Δr| truth")
    axes[1, 0].plot(t_h, rel_est_norm,   lw=0.7, label="|Δr| estimate", alpha=0.75)
    axes[1, 0].set_xlabel("Time [h]"); axes[1, 0].set_ylabel("|Δr| [m]")
    axes[1, 0].set_title("Chief↔Deputy range: truth vs est.")
    axes[1, 0].legend(fontsize=9); axes[1, 0].grid(True, alpha=0.3)
    cmin = np.minimum.accumulate(rel_res_norm)
    axes[1, 1].plot(t_h, rel_res_norm, lw=0.6, alpha=0.7, label="instantaneous")
    axes[1, 1].plot(t_h, cmin, lw=1.2, color="C2", label="running minimum")
    axes[1, 1].set_xlabel("Time [h]"); axes[1, 1].set_ylabel("|Δr_est − Δr_truth| [m]")
    axes[1, 1].set_title("Relative position residual (linear)")
    axes[1, 1].legend(fontsize=9); axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "relative_residual.png", dpi=150)

    # ── 5) Hill-frame relative trajectory ──
    rho_truth = np.zeros((len(t), 3))
    rho_est   = np.zeros((len(t), 3))
    for k in range(len(t)):
        rc = res["truth_chief"][k, :3]; vc = res["truth_chief"][k, 3:]
        rd = res["truth_dep"][k, :3]
        ir = rc / la.norm(rc); h = np.cross(rc, vc); ih = h / la.norm(h)
        it = np.cross(ih, ir)
        HN = np.array([ir, it, ih])
        rho_truth[k] = HN @ (rd - rc)

        rc_e = res["est_chief"][k, :3]; vc_e = res["est_chief"][k, 3:]
        rd_e = res["est_dep"][k, :3]
        ir_e = rc_e / la.norm(rc_e); h_e = np.cross(rc_e, vc_e)
        ih_e = h_e / la.norm(h_e); it_e = np.cross(ih_e, ir_e)
        HN_e = np.array([ir_e, it_e, ih_e])
        rho_est[k] = HN_e @ (rd_e - rc_e)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("Deputy in CHIEF's Hill/LVLH frame — truth vs estimate (v4)",
                 fontsize=13)
    decim = max(1, len(t) // 4000)
    ax.plot(rho_truth[::decim, 0], rho_truth[::decim, 1], rho_truth[::decim, 2],
            "g-", lw=1.4, label="ρ truth")
    ax.plot(rho_est[::decim, 0], rho_est[::decim, 1], rho_est[::decim, 2],
            "r--", lw=1.0, alpha=0.7, label="ρ estimate")
    ax.scatter(0, 0, 0, color="k", s=100, marker="X", label="chief")
    ax.set_xlabel("R̂ (radial) [m]")
    ax.set_ylabel("Θ̂ (along-track) [m]")
    ax.set_zlabel("Ĥ (cross-track) [m]")
    ax.legend(fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "hill_frame.png", dpi=150)


if __name__ == "__main__":
    np.set_printoptions(precision=9, suppress=False, linewidth=200)
    res = run_sim(verbose=True)
    plot_results(res)
    plt.show()
