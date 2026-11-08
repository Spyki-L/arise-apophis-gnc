# -*- coding: utf-8 -*-
"""
attitude.py - rigid-body attitude dynamics for 6-DOF RAFTI docking.

Modified Rodrigues Parameters (MRP) sigma + body rate omega, matching the ARISE
convention (sigma_BN / omega_B, N.m torques) used by the SE(3) separation
controller.  Here attitude is tracked relative to the ESPA-centred LVLH frame
(sigma = body <- LVLH), since the docking geometry lives in LVLH.

    kinematics :  sigma_dot = 1/4 B(sigma) omega            (Schaub & Junkins)
    dynamics   :  J omega_dot = tau - omega x (J omega)      (Euler's equation)

C(sigma) is the DCM [B/L] mapping LVLH -> body, so the deputy's docking face
(+x_body) expressed in LVLH is  C(sigma)^T @ [1,0,0].  MRP shadow-set switching
keeps |sigma| <= 1 (no kinematic singularity).  Deputy inertia defaults to the
real 6U value from the separation run:  J = diag(0.042, 0.204, 0.228) kg m^2.
"""
from __future__ import annotations
import numpy as np

J_DEPUTY = np.array([0.04218398, 0.20445541, 0.22767195])   # principal inertia [kg m^2]


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def mrp_to_dcm(sigma):
    """DCM [B/L] (maps LVLH -> body) from MRP sigma (body wrt LVLH)."""
    s = np.asarray(sigma, float); s2 = s @ s
    sk = _skew(s)
    return np.eye(3) + (8 * sk @ sk - 4 * (1 - s2) * sk) / (1 + s2) ** 2


def dcm_to_mrp(C):
    """MRP from DCM [B/L]."""
    z = np.sqrt(max(np.trace(C) + 1.0, 1e-12))
    return np.array([C[1, 2] - C[2, 1], C[2, 0] - C[0, 2], C[0, 1] - C[1, 0]]) / (z * (z + 2))


def mrp_shadow(sigma):
    """Switch to the shadow MRP set when |sigma| > 1 (keeps it bounded)."""
    s = np.asarray(sigma, float); s2 = s @ s
    return -s / s2 if s2 > 1.0 else s


def _B(sigma):
    s = np.asarray(sigma, float); s2 = s @ s
    return (1 - s2) * np.eye(3) + 2 * _skew(s) + 2 * np.outer(s, s)


def sigma_dot(sigma, omega):
    return 0.25 * _B(sigma) @ omega


def omega_dot(omega, tau, J):
    Jw = J * omega
    return (tau - np.cross(omega, Jw)) / J


def rk4_attitude(sigma, omega, tau, dt, J=J_DEPUTY, substeps=8):
    """Propagate (sigma, omega) under constant body torque tau over dt (RK4)."""
    s = np.asarray(sigma, float).copy(); w = np.asarray(omega, float).copy()
    h = dt / substeps
    for _ in range(substeps):
        def d(sv, wv):
            return sigma_dot(sv, wv), omega_dot(wv, tau, J)
        k1s, k1w = d(s, w)
        k2s, k2w = d(s + 0.5 * h * k1s, w + 0.5 * h * k1w)
        k3s, k3w = d(s + 0.5 * h * k2s, w + 0.5 * h * k2w)
        k4s, k4w = d(s + h * k3s, w + h * k3w)
        s = s + (h / 6) * (k1s + 2 * k2s + 2 * k3s + k4s)
        w = w + (h / 6) * (k1w + 2 * k2w + 2 * k3w + k4w)
        s = mrp_shadow(s)
    return s, w


def body_axis_in_lvlh(sigma, axis=(1.0, 0.0, 0.0)):
    """A body-fixed axis (default +x_body = dock face) expressed in LVLH."""
    return mrp_to_dcm(sigma).T @ np.asarray(axis, float)


def pointing_error_deg(sigma, desired_axis_lvlh, body_axis=(1.0, 0.0, 0.0)):
    """Angle [deg] between the deputy dock face and a desired LVLH direction."""
    a = body_axis_in_lvlh(sigma, body_axis)
    d = np.asarray(desired_axis_lvlh, float); d = d / (np.linalg.norm(d) + 1e-15)
    return float(np.degrees(np.arccos(np.clip(a @ d, -1, 1))))


def mrp_error_to_point(sigma, desired_axis_lvlh, body_axis=(1.0, 0.0, 0.0)):
    """MRP-like 3-vector error whose rotation brings the dock face onto the target
    (used for control + as a compact attitude observation).  Axis-angle / 4 form."""
    a = body_axis_in_lvlh(sigma, body_axis)
    d = np.asarray(desired_axis_lvlh, float); d = d / (np.linalg.norm(d) + 1e-15)
    v = np.cross(a, d); s = np.linalg.norm(v); c = np.clip(a @ d, -1, 1)
    if s < 1e-9:
        return np.zeros(3) if c > 0 else np.array([0, 0, 1.0]) * (np.pi / 4)  # 180 deg
    ang = np.arctan2(s, c)
    return (v / s) * np.tan(ang / 4.0)     # MRP of the error rotation


if __name__ == "__main__":
    # 1) torque-free symmetric top conserves |omega| and energy
    s, w = np.array([0.1, -0.2, 0.05]), np.array([0.02, -0.01, 0.03])
    J = J_DEPUTY
    E0 = 0.5 * (J * w) @ w
    for _ in range(200):
        s, w = rk4_attitude(s, w, np.zeros(3), 1.0, J)
    E1 = 0.5 * (J * w) @ w
    print(f"torque-free energy drift: {abs(E1-E0)/E0:.2e}")
    assert abs(E1 - E0) / E0 < 1e-3
    # 2) a constant torque slews the dock face toward a target, then damp to rest
    sigma = np.zeros(3); omega = np.zeros(3)
    target = np.array([-1.0, 0.0, 0.0])          # want +x_body -> -R (point "inward")
    print("initial pointing err:", round(pointing_error_deg(sigma, target), 1), "deg")
    # simple PD on the MRP pointing error
    for k in range(60):
        err = mrp_error_to_point(sigma, target)
        tau = np.clip(0.02 * err - 0.05 * omega, -5e-3, 5e-3)
        sigma, omega = rk4_attitude(sigma, omega, tau, 1.0, J)
    print("after slew pointing err:", round(pointing_error_deg(sigma, target), 2),
          "deg | rate:", round(np.linalg.norm(omega), 4), "rad/s")
    assert pointing_error_deg(sigma, target) < 5.0
    print("attitude self-test OK")


def mrp_error(sigma, sigma_ref):
    """Relative MRP of body wrt a reference attitude (Schaub & Junkins, eq. for MRP
    difference).  Drives to zero when sigma == sigma_ref.  Body-frame error vector."""
    s = np.asarray(sigma, float); sr = np.asarray(sigma_ref, float)
    s2 = s @ s; sr2 = sr @ sr
    den = 1 + sr2 * s2 + 2 * (sr @ s)
    num = (1 - sr2) * s - (1 - s2) * sr + 2 * np.cross(s, sr)
    e = num / (den + 1e-15)
    return mrp_shadow(e)


def desired_dock_dcm(port_outward_normal, ref=(0.0, 0.0, 1.0)):
    """DCM [B/L] of the desired docking attitude: +x_body -> INWARD (into the port,
    = -outward normal), roll fixed by `ref`.  Returned as a DCM (NOT an MRP) because
    the docking attitude is ~180 deg from identity - the MRP singularity."""
    x = -np.asarray(port_outward_normal, float); x /= np.linalg.norm(x) + 1e-15
    z = np.asarray(ref, float)
    if abs(x @ z) > 0.95:
        z = np.array([0.0, 1.0, 0.0])
    y = np.cross(z, x); y /= np.linalg.norm(y) + 1e-15
    z = np.cross(x, y)
    R_LB = np.column_stack([x, y, z])          # body axes in LVLH (columns)
    return R_LB.T                               # [B/L]


def dock_attitude_error(sigma, port_outward_normal, ref=(0.0, 0.0, 1.0)):
    """Body-frame 3-DOF attitude ERROR MRP for docking, computed via DCMs so it is
    well-defined (the error is near identity when aligned, never the 180-deg MRP
    singularity).  Drives to zero when the dock face points into the port with the
    reference roll.  Full 3-DOF -> stabilises roll (fixes the intermediate-axis drift)."""
    C_des = desired_dock_dcm(port_outward_normal, ref)
    C_cur = mrp_to_dcm(sigma)
    dC = C_cur @ C_des.T                         # error rotation, ~identity when aligned
    return mrp_shadow(dcm_to_mrp(dC))

