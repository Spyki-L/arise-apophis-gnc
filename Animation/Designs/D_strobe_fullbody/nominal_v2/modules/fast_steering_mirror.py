"""
fast_steering_mirror.py  -  ARISE V2 high-fidelity fast-steering-mirror (FSM) model
==============================================================================
Replaces the flat "FSM_THROW_RAD = 2 deg saturation" in guidance_v2 with a PHYSICS-BASED
2-axis FSM model, built from:

  [1] Z. Liu et al., "Modeling and Analysis of a Fast Steering Mirror Based on the Transfer
      Matrix Method," IEEE Access 14 (2026).  -> the PLANT: a four-actuator (voice-coil)
      symmetric tip/tilt mirror; first structural natural frequency ~660 Hz (TMM 661.8 /
      FEM 637), mechanical stroke ~5-8.4 mrad (optical = 2x = ~0.5-1.0 deg), <0.5% static error.

Model contents:
  * PLANT       : 2nd-order per axis (tip/tilt), natural freq f_n, structural damping z_s
                  (beam angle = 2 x mirror angle).
  * CONTROL     : closed-loop tip/tilt loop, bandwidth f_c, damping z_c -> command tracking
                  T(f) and disturbance-rejection sensitivity S(f).
  * DISTURBANCE : total LOS disturbance transmission T_dist(f) = |S(f)| . |H_struct(f)| --
                  rejects low-frequency body pointing error / slew lag, but the mirror's own
                  ~660 Hz structural resonance can amplify near-resonance jitter.
  * LIMITS      : optical stroke +/- throw, slew-rate limit, sub-urad resolution/noise floor.

Mission use:
  1. Reject the reaction-wheel microvibration jitter + the body pointing error/slew-lag ->
     the STABILIZED camera line-of-sight is far better than the bus.
  2. STEP-STARE imaging: during a short imaging burst the FSM fine-tracks the LOS within its
     throw, so the BODY can settle to a low rate.  The (body-fixed) star tracker then sees that
     LOW rate -> unlocks the Gen-3 NST's ~1" static (which is otherwise wasted at 10 deg/s slew).

Run `python fast_steering_mirror.py` for the self-test (prints PASS/FAIL).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

ARCSEC = np.radians(1.0 / 3600.0)
MRAD = 1e-3


@dataclass(frozen=True)
class FSMParams:
    name: str = "ARISE FSM (4-actuator VCM, per Liu 2026 TMM)"
    # --- plant (paper [1]) ---
    natural_freq_hz: float = 660.0        # first structural natural frequency (TMM 661.8 / FEM 637) [1]
    struct_damping: float = 0.02          # flexure structural damping (Q ~ 1/(2z) ~ 25)
    stroke_mech_mrad: float = 8.4         # max mechanical stroke (paper 5.04-8.39 mrad)             [1]
    reflection_factor: float = 2.0        # optical beam angle = 2 x mirror rotation
    n_actuators: int = 4                  # four-actuator symmetric VCM drive                        [1]
    # --- control loop (design; not in the plant-only paper) ---
    ctrl_bandwidth_hz: float = 220.0      # closed-loop bandwidth (~f_n/3 for a 660 Hz mode)
    ctrl_damping: float = 0.7             # closed-loop damping
    # --- limits ---
    resolution_urad: float = 0.5          # commanded-angle resolution / noise floor (optical)
    slew_rate_limit_deg_s: float = 500.0  # max optical steering rate within the loop

    # ---- derived ----
    @property
    def stroke_optical_rad(self) -> float:
        return self.reflection_factor * self.stroke_mech_mrad * MRAD     # optical half-throw [rad]
    @property
    def stroke_optical_deg(self) -> float:
        return np.degrees(self.stroke_optical_rad)
    @property
    def struct_Q(self) -> float:
        return 1.0 / (2.0 * self.struct_damping)

    # ---- transfer functions (f in Hz) ----
    def complementary_T(self, f):
        """Command-tracking complementary sensitivity |T(f)| (2nd-order closed loop)."""
        wc = 2 * np.pi * self.ctrl_bandwidth_hz; s = 2j * np.pi * np.asarray(f, float)
        T = wc**2 / (s**2 + 2 * self.ctrl_damping * wc * s + wc**2)
        return np.abs(T)

    def sensitivity_S(self, f):
        """Disturbance-rejection sensitivity |S(f)| = |1 - T(f)|:  <<1 below bandwidth (rejects),
        ->1 at/above bandwidth."""
        wc = 2 * np.pi * self.ctrl_bandwidth_hz; s = 2j * np.pi * np.asarray(f, float)
        S = (s**2 + 2 * self.ctrl_damping * wc * s) / (s**2 + 2 * self.ctrl_damping * wc * s + wc**2)
        return np.abs(S)

    def _struct_mag(self, f):
        """Plant structural transfer |H(f)| (2nd-order): ~1 below f_n, peaks to Q at f_n, rolls off."""
        r = np.asarray(f, float) / self.natural_freq_hz
        return 1.0 / np.sqrt((1.0 - r**2)**2 + (r / self.struct_Q)**2)

    def disturbance_transmission(self, f):
        """Total LOS disturbance transmission: control rejection x structural amplification."""
        return self.sensitivity_S(f) * self._struct_mag(f)

    # ---- jitter rejection ----
    def residual_jitter_arcsec(self, freqs_hz, amps_arcsec):
        """Residual LOS jitter [arcsec] after the FSM, given disturbance tones (freqs, amplitudes)."""
        f = np.atleast_1d(np.asarray(freqs_hz, float)); a = np.atleast_1d(np.asarray(amps_arcsec, float))
        t = self.disturbance_transmission(f)
        floor = self.resolution_urad * 1e-6 / ARCSEC / np.sqrt(2)     # FSM own noise floor
        return float(np.sqrt(np.sum((a * t)**2) + floor**2))

    def reject_body_error_arcsec(self, body_err_arcsec, freq_hz=0.02):
        """Residual after the FSM cancels a quasi-static body pointing error / slew lag (low-freq,
        strongly rejected within the loop)."""
        return float(body_err_arcsec * self.disturbance_transmission(freq_hz))

    # ---- STEP-STARE: body-rate reduction that unlocks the Gen-3 star tracker ----
    def fsm_assisted_body_rate_deg_s(self, los_rate_deg_s, burst_s, body_max_deg_s=10.0):
        """During a short imaging burst the FSM fine-tracks the LOS within its throw, so the body
        only has to supply the part the FSM cannot cover.  The FSM can absorb up to
        (optical throw / burst) of LOS rate; the residual must be carried by the body.
        Returns the reduced body rate the (body-fixed) star tracker then sees."""
        fsm_cover = self.stroke_optical_rad / max(burst_s, 1e-6) / np.radians(1.0)   # deg/s the FSM absorbs
        residual = max(0.0, float(los_rate_deg_s) - fsm_cover)
        return min(residual, body_max_deg_s)

    def effective_startracker_rate_deg_s(self, body_cap_deg_s=10.0, exposure_s=0.05, floor_deg_s=0.5):
        """Reduced body rate the (body-fixed) star tracker sees during FSM step-stare imaging.
        The FSM absorbs LOS motion up to (throw / exposure), letting the body settle; the floor
        accounts for imperfect settling + the brief exact-CA peak where LOS rate exceeds the
        FSM's absorb capacity.  This is what unlocks the Gen-3 NST's ~1" static during imaging."""
        absorbed = self.stroke_optical_rad / exposure_s / np.radians(1.0)     # deg/s the FSM absorbs
        residual = max(0.0, float(body_cap_deg_s) - absorbed)
        return max(residual, floor_deg_s)


FSM = FSMParams()


# ======================================================================================
# SELF-TEST
# ======================================================================================
def _selftest():
    fails = []
    def check(name, cond, detail=""):
        print(("  PASS " if cond else "  FAIL ") + name + ("" if not detail else "  -> " + detail))
        if not cond: fails.append(name)

    print("=" * 78)
    print("FAST-STEERING-MIRROR SELF-TEST  (Liu 2026 TMM plant + closed-loop control)")
    print("=" * 78)
    p = FSM
    print(f"\n[plant] f_n={p.natural_freq_hz:.0f} Hz (Q~{p.struct_Q:.0f}) | optical throw +/-{p.stroke_optical_deg:.2f} deg "
          f"| control BW {p.ctrl_bandwidth_hz:.0f} Hz")

    # 1. Plant vs paper
    print("\n1. Plant (paper [1]):")
    check("first natural frequency ~660 Hz (TMM 661.8 / FEM 637)", 630 < p.natural_freq_hz < 665)
    check("optical stroke in 0.5-1.1 deg (2x mech 5-8.4 mrad)", 0.5 < p.stroke_optical_deg < 1.1, f"{p.stroke_optical_deg:.2f} deg")

    # 2. Disturbance rejection: strong at low f, ~1 at BW, resonance peak at f_n
    print("\n2. Disturbance rejection (freq -> transmission):")
    for f in [0.1, 1.0, 10.0, 100.0, 220.0, 660.0, 2000.0]:
        print(f"     {f:7.1f} Hz -> S={p.sensitivity_S(f):.3f}  T_dist={p.disturbance_transmission(f):.3f}")
    check("low-freq body error strongly rejected (S(1Hz) < 0.02)", p.sensitivity_S(1.0) < 0.02, f"{p.sensitivity_S(1.0):.4f}")
    check("passes near/above bandwidth (S(220Hz) > 0.5)", p.sensitivity_S(220.0) > 0.5)
    check("structural resonance amplifies at f_n (T_dist(660) > 3)", p.disturbance_transmission(660.0) > 3.0,
          f"{p.disturbance_transmission(660.0):.1f}")

    # 3. Body pointing-error rejection (the near-CA slew lag)
    print("\n3. Body pointing error / slew-lag rejection:")
    for be in [50.0, 200.0, 1000.0]:
        print(f"     body {be:6.0f}\" -> residual LOS {p.reject_body_error_arcsec(be):.3f}\"")
    check("quasi-static body error rejected >100x", p.reject_body_error_arcsec(1000.0) < 10.0)

    # 4. RW microvibration jitter rejection
    print("\n4. Reaction-wheel microvibration jitter rejection:")
    # RW fundamental jitter tones near CA (few tens of Hz) + a resonance spike tone
    freqs = np.array([30.0, 60.0, 88.0, 176.0]); amps = np.array([1.0, 0.6, 0.5, 0.3])
    raw = float(np.sqrt(np.sum(amps**2))); res = p.residual_jitter_arcsec(freqs, amps)
    print(f"     raw RSS {raw:.2f}\" -> post-FSM {res:.2f}\"  ({100*(1-res/raw):.0f}% rejected)")
    check("FSM reduces RW jitter", res < raw)

    # 5. STEP-STARE: FSM lets the body settle -> unlocks Gen-3 star tracker
    print("\n5. Step-stare body-rate reduction (FSM fine-tracks within its throw):")
    for los, burst in [(5.0, 0.05), (15.0, 0.05), (66.0, 0.02)]:
        wb = p.fsm_assisted_body_rate_deg_s(los, burst)
        cover = p.stroke_optical_rad/burst/np.radians(1.0)
        print(f"     LOS {los:4.0f} deg/s, burst {burst*1e3:.0f} ms -> FSM absorbs {cover:.0f} deg/s -> body {wb:.1f} deg/s")
    check("moderate-rate LOS fully absorbed -> body settles (~0 deg/s)", p.fsm_assisted_body_rate_deg_s(15.0, 0.05) < 1.0)

    print("\n" + "=" * 78)
    print(f"RESULT: {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}  ({len(fails)} failure(s))")
    print("=" * 78)
    return len(fails) == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
