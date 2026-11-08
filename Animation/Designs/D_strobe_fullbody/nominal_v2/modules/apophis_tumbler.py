"""
apophis_tumbler.py  -  ARISE V2 true rigid-body NPA attitude (A1)
=================================================================
Replaces V1's kinematic precession-cone with a PHYSICALLY CORRECT torque-free rigid-body
tumbler: integrate Euler's equations + quaternion kinematics with the radar/Pravec inertia
tensor.  This is the single biggest fidelity gain because it (a) gives the correct instantaneous
attitude evolution across the encounter window and (b) tightens the dominant uncertainty (the
spin/precession phase at CA).

Inertia (Pravec et al. 2014, normalized to I3=max):
    I1/I3 = 0.61, I2/I3 = 0.965, I3 = 1     (so I1 < I2 < I3, z_body = max-inertia axis)
    E/E0  = 1.024  (slight excess over principal-axis spin -> tumbling)
Periods: P_rot ~ 27.45 h, P_prec ~ 265.7 h ; nutation amplitude ~ up to 55 deg.

EFFICIENT PROPAGATION.  The 16.31 yr from the radar zero-epoch (2012-12-23) to CA is NOT
stepped: the period uncertainties wrap the phase many times over 16 yr, so the rotation and
precession phases at CA are sampled (near-uniform under the period error), exactly as V1 found.
We then build a true free-tumble initial state at the START of the encounter window from those
phases + the sampled pole/inertia, and integrate Euler's equations over only the short window
(seconds-minutes) with RK4.  Within the window the attitude follows real tumbling dynamics.

Output: a callable body_R(t) -> 3x3 DCM mapping body-frame vectors to inertial (J2000), valid
over the window, plus omega(t).  The coverage scorer rotates facet normals by body_R(t).
"""
from __future__ import annotations
import os, sys
from dataclasses import dataclass, field
import numpy as np

ZERO_EPOCH = "2012-12-23T04:14"          # radar light-curve zero epoch (Pravec)
DT_TO_CA_YR = 16.31                      # zero-epoch -> 2029-04-13 CA


# ---------------------------------------------------------------- quaternion / DCM utils
def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_norm(q):
    return q / np.linalg.norm(q)


def quat_to_dcm(q):
    """Body->inertial DCM from quaternion [w,x,y,z]."""
    w, x, y, z = quat_norm(q)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)]])


def axis_angle_quat(axis, angle):
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    return np.array([np.cos(angle/2), *(np.sin(angle/2) * axis)])


def dcm_from_lonlat(lon_deg, lat_deg):
    """Unit vector (inertial) of a pole given ecliptic/equatorial lon,lat."""
    lo, la = np.radians(lon_deg), np.radians(lat_deg)
    return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])


# ---------------------------------------------------------------- tumbler state
@dataclass
class TumblerParams:
    I: np.ndarray = field(default_factory=lambda: np.array([0.61, 0.965, 1.0]))  # I1,I2,I3 (I3 max)
    P_rot_h: float = 27.45
    P_prec_h: float = 265.7
    nutation_deg: float = 55.0           # obliquity of max-inertia axis from L
    pole_lon_deg: float = 246.8          # angular-momentum (L) direction in inertial frame
    pole_lat_deg: float = -59.3
    phase_rot_deg: float = 0.0           # sampled rotation phase at window start
    phase_prec_deg: float = 0.0          # sampled precession phase at window start


def _initial_state(p: TumblerParams):
    """Build a true free-tumble (omega0, q0) at the window start from the sampled phases.
    Construction: angular momentum L along the sampled pole; the max-inertia (z) body axis is
    tilted from L by the nutation obliquity, spun about L by the precession phase, and the body
    rotated about its own z by the rotation phase.  omega0 = I^-1 (R^T L), |omega| set by P_rot."""
    L_hat = dcm_from_lonlat(p.pole_lon_deg, p.pole_lat_deg)
    # reference inertial triad with z' = L_hat
    z = L_hat
    x = np.cross([0, 0, 1.0], z); 
    if np.linalg.norm(x) < 1e-6: x = np.array([1.0, 0, 0])
    x = x / np.linalg.norm(x); y = np.cross(z, x)
    # orientation: precession about L, then tilt by nutation, then spin about body z
    q_prec = axis_angle_quat(z, np.radians(p.phase_prec_deg))
    tilt_axis = x  # tilt direction (rotated by precession below via composition)
    q_tilt = axis_angle_quat(tilt_axis, np.radians(p.nutation_deg))
    q_spin = axis_angle_quat([0, 0, 1.0], np.radians(p.phase_rot_deg))
    q0 = quat_norm(quat_mul(quat_mul(q_prec, q_tilt), q_spin))
    R = quat_to_dcm(q0)                    # body->inertial
    # body-frame angular momentum, scaled so mean spin rate matches P_rot
    h_body = R.T @ L_hat                    # direction of L in body frame
    omega_dir = h_body / p.I               # I^-1 h  (component-wise, principal axes)
    omega_dir = omega_dir / np.linalg.norm(omega_dir)
    rate = 2*np.pi / (p.P_rot_h * 3600.0)  # rad/s
    omega0 = rate * omega_dir
    return omega0, q0


def _derivs(state, I):
    w = state[:3]; q = state[3:]
    I1, I2, I3 = I
    dw = np.array([(I2-I3)*w[1]*w[2]/I1,
                   (I3-I1)*w[2]*w[0]/I2,
                   (I1-I2)*w[0]*w[1]/I3])
    wq = np.array([0.0, *w])
    dq = 0.5 * quat_mul(q, wq)
    return np.concatenate([dw, dq])


def integrate_window(p: TumblerParams, t_grid):
    """RK4 integrate Euler's eqs over the encounter window. t_grid in seconds, t=0 at window
    start.  Returns omega[N,3] (body) and a list of body->inertial DCMs at each t."""
    omega0, q0 = _initial_state(p)
    state = np.concatenate([omega0, q0])
    I = p.I
    states = [state.copy()]
    for k in range(1, len(t_grid)):
        h = t_grid[k] - t_grid[k-1]
        k1 = _derivs(state, I)
        k2 = _derivs(state + 0.5*h*k1, I)
        k3 = _derivs(state + 0.5*h*k2, I)
        k4 = _derivs(state + h*k3, I)
        state = state + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        state[3:] = quat_norm(state[3:])
        states.append(state.copy())
    states = np.array(states)
    omegas = states[:, :3]
    dcms = np.array([quat_to_dcm(s[3:]) for s in states])
    return omegas, dcms


def sample_tumbler(rng) -> TumblerParams:
    """Sample inertia/periods/pole/phases within published bars for the MC."""
    I1 = np.clip(rng.normal(0.61, 0.03), 0.50, 0.72)
    I2 = np.clip(rng.normal(0.965, 0.015), 0.90, 0.99)
    return TumblerParams(
        I=np.array([I1, I2, 1.0]),
        P_rot_h=float(rng.normal(27.45, 0.02)),
        P_prec_h=float(rng.normal(265.7, 6.0)),
        nutation_deg=float(np.clip(rng.normal(37.0, 12.0), 5.0, 60.0)),  # Pravec ellipse spread
        # pole within Brozovic/Pravec ellipse (27 x 14 deg)
        pole_lon_deg=float(rng.normal(246.8, 14.0)),
        pole_lat_deg=float(np.clip(rng.normal(-59.3, 7.0), -90, -30)),
        # phases near-uniform at CA (period error wraps over 16.31 yr)
        phase_rot_deg=float(rng.uniform(0, 360)),
        phase_prec_deg=float(rng.uniform(0, 360)))


def conservation_check(p: TumblerParams, window_s=120.0, n=2000):
    """Verify the integrator conserves |L| and rotational energy (free body)."""
    t = np.linspace(0, window_s, n)
    omegas, dcms = integrate_window(p, t)
    Es = 0.5 * (p.I[0]*omegas[:,0]**2 + p.I[1]*omegas[:,1]**2 + p.I[2]*omegas[:,2]**2)
    # inertial angular momentum magnitude
    Ls = []
    for k in range(n):
        h_body = p.I * omegas[k]
        Ls.append(np.linalg.norm(dcms[k] @ h_body))
    Ls = np.array(Ls)
    return dict(E_rel_drift=float((Es.max()-Es.min())/Es.mean()),
                L_rel_drift=float((Ls.max()-Ls.min())/Ls.mean()),
                E0=float(Es[0]), L0=float(Ls[0]))


if __name__ == "__main__":
    print("ARISE V2 Apophis tumbler (true Euler-equation NPA attitude)")
    p = TumblerParams(phase_rot_deg=40.0, phase_prec_deg=120.0)
    chk = conservation_check(p, window_s=120.0, n=4000)
    print("  conservation over 120 s window (RK4):")
    print("    energy   relative drift: %.2e" % chk['E_rel_drift'])
    print("    |L|      relative drift: %.2e" % chk['L_rel_drift'])
    # report instantaneous attitude change across a typical imaging window
    t = np.linspace(0, 4.0, 41)
    _, dcms = integrate_window(p, t)
    z0 = dcms[0][:, 2]; z1 = dcms[-1][:, 2]
    ang = np.degrees(np.arccos(np.clip(z0 @ z1, -1, 1)))
    print("  body z-axis moves %.4f deg over a 4 s imaging window" % ang)
    rng = np.random.default_rng(0)
    ps = sample_tumbler(rng)
    print("  sample draw: I=%s  P_rot=%.2fh  nut=%.1fdeg  pole=(%.1f,%.1f)  phases=(%.0f,%.0f)"
          % (np.round(ps.I,3), ps.P_rot_h, ps.nutation_deg, ps.pole_lon_deg, ps.pole_lat_deg,
             ps.phase_rot_deg, ps.phase_prec_deg))
