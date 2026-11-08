"""
timing.py  -  ARISE V2 timing fidelity (C3)
===========================================
At 8.76 km/s with a usable close-range window of only ~0.3-0.9 s, timing decides whether a
PLANNED stereo pair actually lands in-band.  This models:

  * TCA jitter        - along-track trajectory uncertainty -> uncertainty in WHEN closest
                        approach occurs (sigma_t = sigma_alongtrack / v_flyby)
  * trigger timing    - exposure-command latency + jitter relative to the planned CA transit
  * inter-deputy clock sync - relative clock offsets between deputies; sets how coincident two
                        deputies' frames are (needed to co-register a spatial stereo pair while
                        the scene sweeps fast)

The body itself is effectively frozen over seconds (apophis_tumbler: ~0.015 deg / 4 s), so clock
sync matters for SCENE coincidence under fast relative motion, not for body phase.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

V_FLYBY = 8760.0


@dataclass(frozen=True)
class TimingModel:
    alongtrack_sigma_m: float = 600.0     # along-track position 1-sigma (Set III B-plane scale)
    trigger_latency_s: float = 0.002      # ASSUMED: mean command->exposure latency 2 ms
    trigger_jitter_s: float = 0.001       # ASSUMED: 1 ms trigger jitter 1-sigma
    clock_sync_sigma_s: float = 0.001     # ASSUMED: inter-deputy clock sync 1 ms 1-sigma (GPS-disc.)
    usable_window_s: float = 0.6          # close-range usable imaging window (~0.3-0.9 s)

    def tca_sigma_s(self):
        return self.alongtrack_sigma_m / V_FLYBY


def sample_timing(n_deputies, rng, m: TimingModel = TimingModel()):
    """Sample one encounter's timing errors."""
    tca_offset = float(rng.normal(0.0, m.tca_sigma_s()))               # common TCA shift
    trig = m.trigger_latency_s + rng.normal(0.0, m.trigger_jitter_s, n_deputies)
    clock = rng.normal(0.0, m.clock_sync_sigma_s, n_deputies)          # per-deputy offsets
    return dict(tca_offset_s=tca_offset, trigger_s=trig, clock_offset_s=clock)


def effective_frame_times(planned_times, deputy_idx, timing):
    """Actual frame times for a deputy = planned + common TCA shift + its trigger + its clock."""
    pt = np.asarray(planned_times, float)
    return pt + timing["tca_offset_s"] + timing["trigger_s"][deputy_idx] + timing["clock_offset_s"][deputy_idx]


def pair_coincident(t_a, t_b, tol_s):
    """Two frames can co-register for a spatial stereo pair if within tol_s."""
    return abs(t_a - t_b) <= tol_s


def in_band_probability(m: TimingModel = TimingModel(), n_trials=20000, rng=None,
                        coincidence_tol_s=0.01):
    """Monte-Carlo: probability a planned center-of-window stereo pair (two deputies) both land
    inside the usable window AND are coincident enough to co-register."""
    rng = rng or np.random.default_rng(0)
    half = 0.5 * m.usable_window_s
    ok = 0
    for _ in range(n_trials):
        t = sample_timing(2, rng, m)
        ta = effective_frame_times([0.0], 0, t)[0]
        tb = effective_frame_times([0.0], 1, t)[0]
        in_win = (abs(ta) <= half) and (abs(tb) <= half)
        if in_win and pair_coincident(ta, tb, coincidence_tol_s):
            ok += 1
    return ok / n_trials


if __name__ == "__main__":
    m = TimingModel()
    rng = np.random.default_rng(0)
    print("ARISE V2 timing fidelity sanity")
    print("  TCA jitter sigma         : %.1f ms  (=%.0f m along-track / %.0f m/s)"
          % (m.tca_sigma_s()*1e3, m.alongtrack_sigma_m, V_FLYBY))
    print("  trigger latency/jitter   : %.0f / %.0f ms" % (m.trigger_latency_s*1e3, m.trigger_jitter_s*1e3))
    print("  inter-deputy clock sync  : %.0f ms 1-sigma" % (m.clock_sync_sigma_s*1e3))
    print("  usable close-range window: %.2f s" % m.usable_window_s)
    t = sample_timing(8, rng, m)
    print("  sample: TCA offset %.1f ms, trigger spread %.1f ms, clock spread %.1f ms"
          % (t["tca_offset_s"]*1e3, (t["trigger_s"].max()-t["trigger_s"].min())*1e3,
             (t["clock_offset_s"].max()-t["clock_offset_s"].min())*1e3))
    for tol in (0.005, 0.01, 0.05):
        p = in_band_probability(m, coincidence_tol_s=tol)
        print("  P(planned pair in-band & coincident, tol %3.0f ms) = %.3f" % (tol*1e3, p))
