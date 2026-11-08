# -*- coding: utf-8 -*-
"""
sensors.py - flight sensor models feeding the navigation filter.

  VBNSensor    vision-based relative navigation: line-of-sight BEARING (az,el) to
               the ESPA always, plus RANGE when within range-channel validity.
               Noise + bias + latency + random dropout.
  StarTracker  absolute attitude (MRP body<-LVLH) with noise + dropout.
  RateGyro     body angular rate with white noise + slowly-varying bias (Gauss-Markov).

All models are deterministic given a seeded numpy Generator (flight reproducibility).
Measurements are returned in engineering units; the EKF (navigation.py) owns the
measurement models and reconstructs the relative state.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from flight import config_flight as F
from flight import hardware_bridge as HWB


class _Latency:
    def __init__(self, n):
        self.buf = deque(maxlen=max(1, n + 1)); self.n = n
    def push_pop(self, item):
        self.buf.append(item)
        return self.buf[0] if len(self.buf) > self.n else None


class VBNSensor:
    def __init__(self, rng):
        s = F.SENSORS; self.rng = rng
        self.b_sig = s["vbn_bearing_sigma_rad"]; self.b_bias = s["vbn_bearing_bias_rad"] * rng.normal(size=2)
        self.r_sig = s["vbn_range_sigma_frac"]; self.r_floor = s["vbn_range_floor_m"]
        self.r_max = s["vbn_range_max_m"]; self.drop = s["vbn_dropout_prob"]
        self.lat = _Latency(s["vbn_latency_steps"])

    def measure(self, p_rel_lvlh):
        """p_rel_lvlh: true deputy position relative to the ESPA (LVLH). Returns dict
        {az,el,range or None, R_bearing, R_range} delayed by latency, or None on dropout."""
        rho = float(np.linalg.norm(p_rel_lvlh))
        u = p_rel_lvlh / (rho + 1e-12)
        az = np.arctan2(u[1], u[0]); el = np.arcsin(np.clip(u[2], -1, 1))
        az_m = az + self.b_bias[0] + self.b_sig * self.rng.normal()
        el_m = el + self.b_bias[1] + self.b_sig * self.rng.normal()
        rng_m = None
        if rho <= self.r_max:
            rng_m = rho * (1 + self.r_sig * self.rng.normal()) + self.r_floor * self.rng.normal()
        meas = dict(az=az_m, el=el_m, range=rng_m,
                    R_bearing=self.b_sig**2, R_range=(self.r_sig * max(rho, 1.0))**2 + self.r_floor**2)
        out = self.lat.push_pop(meas)
        if out is not None and self.rng.random() < self.drop:
            return None
        return out


class StarTracker:
    """XACT-50 Gen-3 star tracker (nominal_v2) - rate-dependent per-axis error incl.
    streak degradation; falls back to a flat 1-sigma model."""
    def __init__(self, rng):
        self.rng = rng; self.hw = HWB.AVAILABLE and HWB.ST is not None
        self.sig = F.SENSORS["st_sigma_rad"]; self.drop = F.SENSORS["st_dropout_prob"]

    def measure(self, sigma_true, omega_true=None):
        if self.rng.random() < self.drop:
            return None
        if self.hw:
            w = 0.0 if omega_true is None else float(np.linalg.norm(omega_true))
            err = HWB.ST.measure_error(w, self.rng)          # per-axis angle error [rad]
            if not np.all(np.isfinite(err)):                 # tracker lost (too fast) -> dropout
                return None
            return np.asarray(sigma_true, float) + err / 4.0  # small-angle -> MRP
        return np.asarray(sigma_true, float) + (self.sig / 4.0) * self.rng.normal(size=3)


class RateGyro:
    """Nav-grade IMU gyro (nominal_v2): ARW + bias-instability (Gauss-Markov) + RRW +
    scale-factor/misalignment/quantization; falls back to a white+GM model."""
    def __init__(self, rng):
        self.rng = rng; self.hw = HWB.AVAILABLE and HWB.GYRO is not None; self._state = None
        s = F.SENSORS; self.sig = s["gyro_sigma_radps"]
        self.bias = s["gyro_bias_radps"] * rng.normal(size=3); self.tau = s["gyro_bias_tau_s"]

    def measure(self, omega_true, dt):
        if self.hw:
            wm, self._state = HWB.GYRO.measure(np.asarray(omega_true, float), self.rng, self._state, dt)
            return np.asarray(wm, float)
        self.bias += (-self.bias / self.tau) * dt + self.sig * 0.1 * np.sqrt(dt) * self.rng.normal(size=3)
        return np.asarray(omega_true, float) + self.bias + self.sig * self.rng.normal(size=3)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    vbn, st, gy = VBNSensor(rng), StarTracker(rng), RateGyro(rng)
    p = np.array([50.0, -30.0, 10.0])
    m = None
    for _ in range(3):
        m = vbn.measure(p)                       # first calls buffered by latency
    print("VBN:", None if m is None else {k: (round(v, 4) if np.isscalar(v) and v is not None else v)
                                          for k, v in m.items() if k in ("az", "el", "range")})
    print("ST err [rad]:", np.round(st.measure(np.array([0.1, -0.2, 0.05])) - np.array([0.1, -0.2, 0.05]), 5))
    print("gyro:", np.round(gy.measure(np.array([0.01, 0, -0.02]), 1.0), 5))
    print("sensors OK")
