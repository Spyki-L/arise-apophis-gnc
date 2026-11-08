#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faultlab_ekf.py — the per-deputy onboard estimators that produce the values the FDI
actually consumes.  The FDI NEVER sees truth: it sees only these filters' estimates
and innovations.  Truth is used only to (a) generate the measurements (with faults
injected as unmodeled accel/torque or sensor errors) and (b) validate the estimate.

  NavEKF — 6-state Clohessy–Wiltshire relative-orbit EKF.  State x=[r,v] in the chief
           Hill frame; relative-position measurements (ISL ranging).  Innovation y_k =
           z − H x_pred with covariance S = H P⁻ Hᵀ + R is exactly the NAV channel input.

  AttEKF — 6-state attitude-error EKF.  State x=[θ (small-angle MRP error), ω (body
           rate)]; star-tracker (θ) and gyro (ω) measurements.  The estimated θ,ω feed
           the ATT channel; the star-tracker/gyro innovations drive the innovation
           monitor.  An unmodeled torque on the truth makes the innovation leave its
           ±σ bound — the true fault signature.

Pure numpy.  Filter-agnostic, O(1) per step (6×6 algebra), flight-software friendly.
"""
from __future__ import annotations
import numpy as np
import numpy.linalg as la

I3 = np.eye(3)
I6 = np.eye(6)


class NavEKF:
    """6-state Clohessy–Wiltshire relative-orbit EKF (x = [r(3), v(3)] in chief Hill frame)."""

    def __init__(self, n, r_meas=14.0, q_accel=3.0e-4):
        self.n = float(n)                      # chief mean motion [rad/s]
        self.R = I3 * r_meas ** 2              # ranging-measurement covariance [m²]
        self.q = float(q_accel)                # process-noise accel density [m/s²]
        self.x = np.zeros(6)
        self.P = np.diag([1e4, 1e4, 1e4, 4.0, 4.0, 4.0])
        self.y = np.zeros(3)                   # last innovation
        self.S = self.R.copy()
        self.initd = False

    def _Phi(self, dt):
        """Closed-form CW state-transition matrix (→ double-integrator as n·dt → 0)."""
        n, nt = self.n, self.n * dt
        if abs(nt) < 1e-7:
            Phi = I6.copy(); Phi[:3, 3:] = I3 * dt; return Phi
        s, c = np.sin(nt), np.cos(nt)
        Phi = np.zeros((6, 6))
        Phi[0, 0] = 4 - 3 * c;            Phi[1, 0] = 6 * (s - nt);     Phi[2, 2] = c
        Phi[1, 1] = 1.0
        Phi[0, 3] = s / n;                Phi[0, 4] = 2 * (1 - c) / n
        Phi[1, 3] = -2 * (1 - c) / n;     Phi[1, 4] = (4 * s - 3 * nt) / n;   Phi[2, 5] = s / n
        Phi[3, 0] = 3 * n * s;            Phi[4, 0] = -6 * n * (1 - c); Phi[5, 2] = -n * s
        Phi[3, 3] = c;                    Phi[3, 4] = 2 * s
        Phi[4, 3] = -2 * s;               Phi[4, 4] = 4 * c - 3;        Phi[5, 5] = c
        return Phi

    def init(self, r0, v0=None):
        self.x = np.concatenate([np.asarray(r0, float), np.zeros(3) if v0 is None else np.asarray(v0, float)])
        self.initd = True

    def _Q(self, dt):
        a = self.q
        return np.diag([(0.5 * a * dt * dt) ** 2] * 3 + [(a * dt) ** 2] * 3) + 1e-6 * I6

    def predict_only(self, dt):
        self.x = self._Phi(dt) @ self.x

    def step(self, dt, z, u_accel=None):
        """Predict with CW (+ known nominal accel feed-forward), update with a relative-position
        measurement z. The fault is the UNMODELED accel not in u_accel. Returns (y, S)."""
        if not self.initd:
            self.init(z)
        Phi = self._Phi(dt)
        xp = Phi @ self.x
        if u_accel is not None:                       # feed-forward the planned (non-CW) dynamics
            u = np.asarray(u_accel, float)
            xp[:3] = xp[:3] + 0.5 * dt * dt * u
            xp[3:] = xp[3:] + dt * u
        Pp = Phi @ self.P @ Phi.T + self._Q(dt)
        H = np.zeros((3, 6)); H[:, :3] = I3
        self.y = np.asarray(z, float) - H @ xp
        self.S = H @ Pp @ H.T + self.R
        K = Pp @ H.T @ la.inv(self.S)
        self.x = xp + K @ self.y
        self.P = (I6 - K @ H) @ Pp
        return self.y, self.S

    @property
    def pos(self):
        return self.x[:3]

    @property
    def vel(self):
        return self.x[3:]


class AttEKF:
    """6-state attitude-error EKF (x = [θ small-angle error (3), ω body rate (3)])."""

    def __init__(self, st_noise=np.deg2rad(0.05), gy_noise=None, q_torque=np.deg2rad(0.004)):
        # gy_noise = onboard gyro rate-measurement 1-sigma [rad/s].  IDENTITY-WHEN-ABSENT: the flat
        # 0.003 deg/s heritage value, UNLESS the hi-fi IMU gate has exported ARISE_GYRO_SIGMA_RADPS
        # (imu_gyro.py, nav-grade MEMS Allan model +/- event-camera rate aid).  Explicit arg overrides.
        if gy_noise is None:
            import os
            gy_noise = float(os.getenv("ARISE_GYRO_SIGMA_RADPS", np.deg2rad(0.003)))
        self.Rst = I3 * st_noise ** 2
        self.Rgy = I3 * gy_noise ** 2
        self.q = float(q_torque)
        self.st_n = float(st_noise); self.gy_n = float(gy_noise)
        self.x = np.zeros(6)
        self.P = np.diag([1e-4] * 3 + [1e-6] * 3)
        self.y_st = np.zeros(3); self.y_gy = np.zeros(3)
        self.S_st = self.Rst.copy(); self.S_gy = self.Rgy.copy()

    def _Q(self, dt):
        a = self.q
        return np.diag([(0.5 * a * dt * dt) ** 2] * 3 + [(a * dt) ** 2] * 3) + 1e-12 * I6

    def step(self, dt, z_theta, z_omega):
        """Predict (θ += ω dt), update star-tracker then gyro. Returns (x, y_st, y_gy)."""
        F = I6.copy(); F[:3, 3:] = I3 * dt
        xp = F @ self.x
        Pp = F @ self.P @ F.T + self._Q(dt)
        # star-tracker update (measures θ)
        H1 = np.zeros((3, 6)); H1[:, :3] = I3
        self.y_st = np.asarray(z_theta, float) - H1 @ xp
        S1 = H1 @ Pp @ H1.T + self.Rst; self.S_st = S1
        K1 = Pp @ H1.T @ la.inv(S1)
        x1 = xp + K1 @ self.y_st
        P1 = (I6 - K1 @ H1) @ Pp
        # gyro update (measures ω)
        H2 = np.zeros((3, 6)); H2[:, 3:] = I3
        self.y_gy = np.asarray(z_omega, float) - H2 @ x1
        S2 = H2 @ P1 @ H2.T + self.Rgy; self.S_gy = S2
        K2 = P1 @ H2.T @ la.inv(S2)
        self.x = x1 + K2 @ self.y_gy
        self.P = (I6 - K2 @ H2) @ P1
        return self.x, self.y_st, self.y_gy

    @property
    def theta(self):
        return self.x[:3]

    @property
    def omega(self):
        return self.x[3:]


def chief_mean_motion(rC, vC):
    """Instantaneous orbital rate of the chief about the primary: n = |r×v| / |r|²."""
    rC = np.asarray(rC, float); vC = np.asarray(vC, float)
    return float(la.norm(np.cross(rC, vC)) / max(la.norm(rC) ** 2, 1e-9))
