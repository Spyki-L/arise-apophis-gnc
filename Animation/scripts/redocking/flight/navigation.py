# -*- coding: utf-8 -*-
"""
navigation.py - onboard relative navigation.

RelativeNavEKF : relative-nav EKF in the LVLH frame.  The state is [r_rel, v_rel]
                 (6) OPTIONALLY augmented with a 3-state LVLH acceleration bias
                 [a_bias] (9) that ESTIMATES the un-modelled relative acceleration.
                 Time update uses the Clohessy-Wiltshire STM (+ impulsive Delta-v
                 feed-forward + the estimated accel bias forcing [r,v]); measurement
                 update ingests VBN bearing (az,el) and, when available, range, with
                 a chi-square innovation gate, plus a direct docking-VBN position fix.

                 Rationale for the bias state: the flight loop propagates truth in
                 Basilisk (deputy carries full J2 / third-body / SRP relative to a
                 Keplerian chief).  A pure CW predict is then a WRONG model and the
                 filter, if overconfident (tiny Q), flies to a biased estimate it
                 trusts -> 0% dock.  Estimating a_bias captures that un-modelled
                 acceleration so the estimate tracks truth.  On the HCW surrogate
                 a_bias -> 0 and the filter reduces to the CW EKF.

AttitudeEstimator : MRP attitude + gyro-bias complementary filter (gyro propagation
                 corrected by the star tracker).

Feeds the ESTIMATED state to guidance/control (truth is never used in the loop).
`self.x` stays the 6-state [r,v] guidance interface; the bias is internal (self.b).
"""
from __future__ import annotations
import numpy as np
from dynamics.hcw import hcw_stm
from dynamics.attitude import sigma_dot, mrp_shadow, mrp_error, mrp_to_dcm
from flight import config_flight as F


def _h(r):
    """Measurement model: relative position -> [az, el, range]."""
    rho = np.linalg.norm(r) + 1e-12
    return np.array([np.arctan2(r[1], r[0]), np.arcsin(np.clip(r[2] / rho, -1, 1)), rho])


def _Hjac(r, eps=1e-4):
    """Numerical Jacobian d[az,el,range]/d[r,v] (velocity cols are zero)."""
    H = np.zeros((3, 6)); h0 = _h(r)
    for j in range(3):
        dr = np.zeros(3); dr[j] = eps
        H[:, j] = (_h(r + dr) - h0) / eps
    return H, h0


class RelativeNavEKF:
    """LVLH relative-nav EKF; 6-state [r,v] optionally augmented with a 3-state LVLH
    acceleration bias (9-state) estimating the un-modelled relative acceleration."""
    def __init__(self, x0, P0=None, omega=None):
        cfg = getattr(F, "NAV", {})
        self.estimate_bias = bool(cfg.get("estimate_accel_bias", True))
        self.q_pos = float(cfg.get("q_pos", 1e-4))
        self.q_vel = float(cfg.get("q_vel", 1e-6 if self.estimate_bias else 1e-5))
        self.q_bias = float(cfg.get("q_accel_bias", 3e-6))
        b0s = float(cfg.get("accel_bias0_sigma", 3e-2))
        self.x = np.asarray(x0, float).copy()               # [r(3), v(3)] - guidance interface
        self.b = np.zeros(3)                                 # LVLH accel-bias estimate
        self.w = float(omega if omega is not None else 1.027e-3)
        self.gate2 = (F.FDIR["nav_innov_sigma_gate"] + 2.0) ** 2   # a touch wider during acquisition
        p6 = np.diag([100.0, 100.0, 100.0, 0.04, 0.04, 0.04]) if P0 is None else np.asarray(P0, float)
        self.P = np.zeros((9, 9)); self.P[:6, :6] = p6
        self.P[6:, 6:] = np.diag([b0s ** 2] * 3) if self.estimate_bias else np.zeros((3, 3))

    def _phi_aug(self, dt):
        """9x9 augmented STM: CW on [r,v], accel-bias forces [r,v] (0.5dt^2, dt), bias = random walk."""
        A = np.eye(9); A[:6, :6] = hcw_stm(dt, self.w)
        if self.estimate_bias:
            G = np.zeros((6, 3)); G[:3, :] = 0.5 * dt * dt * np.eye(3); G[3:, :] = dt * np.eye(3)
            A[:6, 6:] = G
        return A

    def predict(self, dt, dv_applied=None):
        xa = np.concatenate([self.x, self.b])
        if dv_applied is not None:
            xa[3:6] += np.asarray(dv_applied, float)         # feed-forward the commanded burn
        A = self._phi_aug(dt)
        xa = A @ xa
        qb = self.q_bias if self.estimate_bias else 0.0
        Q = np.diag([self.q_pos] * 3 + [self.q_vel] * 3 + [qb] * 3) * dt
        self.P = A @ self.P @ A.T + Q
        self.x = xa[:6].copy(); self.b = xa[6:].copy()

    def _update(self, H6, y, R, gate2):
        """EKF update: H6 is (m x 6) w.r.t. [r,v]; bias columns are zero (unobserved directly)."""
        m = H6.shape[0]
        H = np.zeros((m, 9)); H[:, :6] = H6
        S = H @ self.P @ H.T + R
        if float(y @ np.linalg.solve(S, y)) > gate2:         # innovation gate -> reject outlier
            return "rejected"
        K = self.P @ H.T @ np.linalg.inv(S)
        xa = np.concatenate([self.x, self.b]) + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P
        self.x = xa[:6].copy(); self.b = xa[6:].copy()
        return "ok"

    def update(self, meas):
        """meas: dict from VBNSensor (az, el, range or None, R_bearing, R_range)."""
        if meas is None:
            return "dropout"
        r = self.x[:3]; H3, h0 = _Hjac(r)
        if meas["range"] is not None:
            z = np.array([meas["az"], meas["el"], meas["range"]]); H6 = H3
            R = np.diag([meas["R_bearing"], meas["R_bearing"], meas["R_range"]])
        else:
            z = np.array([meas["az"], meas["el"]]); H6 = H3[:2]; h0 = h0[:2]
            R = np.diag([meas["R_bearing"], meas["R_bearing"]])
        y = z - h0
        y[0] = (y[0] + np.pi) % (2 * np.pi) - np.pi           # wrap az innovation
        return self._update(H6, y, R, self.gate2)

    def update_position(self, pos_meas_lvlh, R):
        """Direct relative-position fix from the docking VBN/LIDAR (LVLH frame)."""
        H6 = np.zeros((3, 6)); H6[:, :3] = np.eye(3)
        y = np.asarray(pos_meas_lvlh, float) - self.x[:3]
        return self._update(H6, y, np.asarray(R, float), self.gate2 + 25.0)

    @property
    def pos_sigma(self):
        return float(np.sqrt(np.trace(self.P[:3, :3]) / 3))

    def diverged(self):
        return self.pos_sigma > F.FDIR["nav_cov_diverge_m"]


class AttitudeEstimator:
    def __init__(self, sigma0=None, k_att=0.3, k_bias=0.02):
        self.sigma = np.zeros(3) if sigma0 is None else np.asarray(sigma0, float).copy()
        self.bias = np.zeros(3); self.k_att = k_att; self.k_bias = k_bias; self.st_ok = True

    def step(self, gyro_meas, dt, st_meas=None, sun_meas_body=None, sun_ref=None,
             k_sun=0.25):
        w = np.asarray(gyro_meas, float) - self.bias
        pred = mrp_shadow(self.sigma + sigma_dot(self.sigma, w) * dt)         # gyro propagate
        if st_meas is not None:
            self.bias -= self.k_bias * mrp_error(np.asarray(st_meas, float), pred) / dt
            self.sigma = np.asarray(st_meas, float)
            self.st_ok = True
        elif sun_meas_body is not None and sun_ref is not None:
            C = mrp_to_dcm(pred)                       # ref -> body
            s_pred = C @ (np.asarray(sun_ref, float) / (np.linalg.norm(sun_ref) + 1e-15))
            s_meas = np.asarray(sun_meas_body, float)
            s_meas = s_meas / (np.linalg.norm(s_meas) + 1e-15)
            dtheta = 0.5 * np.cross(s_meas, s_pred)    # body correction pred->true (perp to Sun)
            self.sigma = mrp_shadow(pred + k_sun * dtheta)
            self.st_ok = False
        else:
            self.sigma = pred
            self.st_ok = False
        return self.sigma, w


if __name__ == "__main__":
    # A/B self-test: truth = CW propagation + a CONSTANT un-modelled LVLH accel, sparse
    # noisy position fixes.  The plain CW EKF cannot model the accel and drifts; the
    # accel-bias-augmented EKF estimates it and tracks.
    import importlib
    w = 1.027e-3; dt = 1.0
    Phi = hcw_stm(dt, w)
    G = np.vstack([0.5 * dt * dt * np.eye(3), dt * np.eye(3)])
    a_true = np.array([0.006, -0.009, 0.004])          # m/s^2 un-modelled (perturbation scale)

    def run(estimate_bias):
        F.NAV = dict(estimate_accel_bias=estimate_bias, q_pos=1e-4, q_vel=1e-6,
                     q_accel_bias=3e-6, accel_bias0_sigma=3e-2)
        rng = np.random.default_rng(0)
        xt = np.array([80., -40., 20., -0.05, 0.03, -0.01])
        ekf = RelativeNavEKF(x0=xt + np.concatenate([rng.normal(scale=8, size=3),
                                                     rng.normal(scale=0.05, size=3)]))
        perr = verr = 0.0
        for k in range(400):
            xt = Phi @ xt + G @ a_true                 # truth carries the un-modelled accel
            ekf.predict(dt)
            if k % 8 == 0:                             # sparse position fixes (so the MODEL matters)
                z = xt[:3] + rng.normal(scale=0.5, size=3)
                ekf.update_position(z, np.diag([0.25] * 3))
            perr = np.linalg.norm(ekf.x[:3] - xt[:3]); verr = np.linalg.norm(ekf.x[3:6] - xt[3:6])
        return perr, verr, ekf

    p6, v6, e6 = run(False)
    p9, v9, e9 = run(True)
    print(f"[6-state CW ]  pos_err={p6:8.2f} m  vel_err={v6:7.3f} m/s  reported_sigma={e6.pos_sigma:.2f} m")
    print(f"[9-state a_b]  pos_err={p9:8.2f} m  vel_err={v9:7.3f} m/s  reported_sigma={e9.pos_sigma:.2f} m  "
          f"a_bias_est={np.round(e9.b,4)} (true {a_true})")
    assert p9 < 0.2 * p6 and p9 < 5.0, (p6, p9)
    print("EKF accel-bias augmentation OK -> tracks the un-modelled acceleration the CW EKF cannot")
