# -*- coding: utf-8 -*-
"""
avionics.py - the rest of the deputy avionics wired from nominal_v2 for completeness.

  SunSensorHead  coarse Sun-vector knowledge (ARISE CSS, ~0.09 deg) - a safe-mode /
                 star-tracker-dropout attitude backup (6 heads tile 4-pi).
  OnboardClock   CSAC time reference (Microsemi SA.45s class): time error grows with the
                 Allan/aging model; provides the onboard timestamp + fleet time-sync sigma.
  ISLLink        optical inter-satellite link: physical link budget -> ranging sigma,
                 latency, and outage probability for peer relative-nav aiding, fleet
                 collision-avoidance coordination, and comm-health (FDIR) realism.

All wrap the real nominal_v2 models when reachable (hardware_bridge.AVAILABLE), else
fall back to representative constants so the flight package always runs.
"""
from __future__ import annotations
import os, sys
import numpy as np
from flight import hardware_bridge as HWB

_SS = _PNT = _ISL = None
if HWB.AVAILABLE:
    try:
        import sun_sensor as _SS      # noqa
        import pnt_timing as _PNT     # noqa
        import isl_comms as _ISL      # noqa
    except Exception:
        _SS = _PNT = _ISL = None


class SunSensorHead:
    """Coarse Sun-vector attitude reference (safe-mode / ST-dropout backup)."""
    def __init__(self, rng):
        self.rng = rng
        self.sigma_rad = _SS.sun_vector_sigma_rad() if _SS else np.radians(0.094)
        self.name = _SS.active().name if _SS else "ARISE CSS (fallback 0.094 deg)"

    def measure(self, sun_hat_body):
        if _SS:
            return _SS.measure(np.asarray(sun_hat_body, float), rng=self.rng)
        s = np.asarray(sun_hat_body, float); s = s / (np.linalg.norm(s) + 1e-15)
        n = s + self.sigma_rad * self.rng.normal(size=3)
        return n / (np.linalg.norm(n) + 1e-15)


class OnboardClock:
    """CSAC onboard time reference with drift; provides timestamp + fleet-sync sigma."""
    def __init__(self, rng, dt=1.0):
        self.rng = rng; self.dt = dt; self.t = 0.0; self.x = 0.0    # time error [s]
        self.clk = _PNT.CSAC if _PNT else None
        # per-step time-error increment 1-sigma (Allan at the step)
        self.step_sigma = (self.clk.time_error_sigma_s(dt) if self.clk else 1e-9)
        self.sync_sigma = (_PNT.deputy_deputy_sync_s() if _PNT else 5e-9)   # inter-deputy TWTT
        self.name = self.clk.name if self.clk else "CSAC (fallback)"

    def tick(self):
        self.x += self.step_sigma * self.rng.normal()      # random-walk time error
        self.t += self.dt
        return self.t + self.x                              # onboard (biased) timestamp

    @property
    def time_error_s(self):
        return self.x


class ISLLink:
    """Optical ISL: peer ranging sigma, latency, outage from the physical link budget."""
    def __init__(self, rng):
        self.rng = rng
        self.p = _ISL.DEFAULT if _ISL else None
        self.name = "ARISE optical ISL" if _ISL else "ISL (fallback)"

    def peer_range(self, true_range_m):
        """Noisy inter-deputy range + link health at this separation.  Returns
        (range_meas or None on outage, dict(sigma_m, latency_s, outage_p, available))."""
        if self.p is not None:
            sig = float(self.p.ranging_sigma_m(true_range_m))
            lat = float(_ISL.latency_s(true_range_m))
            margin = float(self.p.link_margin_dB(true_range_m))
            outp = float(_ISL.outage_probability(margin, m=1.0)) if margin < 20 else 0.0
        else:
            sig = 0.01 + 1e-4 * true_range_m; lat = true_range_m / 3e8 + 1e-3; outp = 0.001
        if self.rng.random() < outp:
            return None, dict(sigma_m=sig, latency_s=lat, outage_p=outp, available=False)
        rm = true_range_m + sig * self.rng.normal()
        return rm, dict(sigma_m=sig, latency_s=lat, outage_p=outp, available=True)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ss, clk, isl = SunSensorHead(rng), OnboardClock(rng), ISLLink(rng)
    print("sun sensor :", ss.name, "-> sigma", round(np.degrees(ss.sigma_rad), 3), "deg")
    print("clock      :", clk.name, "-> step σ", f"{clk.step_sigma:.2e} s, sync σ {clk.sync_sigma:.2e} s")
    for _ in range(100): t = clk.tick()
    print("            onboard t after 100s:", round(t, 6), "err", f"{clk.time_error_s*1e9:.2f} ns")
    print("ISL        :", isl.name)
    for R in (100.0, 1000.0, 5000.0):
        rm, h = isl.peer_range(R)
        print(f"            peer @ {R:5.0f} m -> σ {h['sigma_m']*100:.2f} cm, latency {h['latency_s']*1e3:.2f} ms, outage {h['outage_p']:.1e}")
    print("avionics OK")
