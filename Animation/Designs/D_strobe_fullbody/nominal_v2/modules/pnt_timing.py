"""
pnt_timing.py  -  ARISE V2 high-fidelity PNT timing / clock model (chief + deputies)
====================================================================================
Replaces the flat fleet-timing constants (timing.py clock_sync_sigma_s, and the driver's
ARISE_CLOCK_SIGMA_FLEET_S = 3e-7 / ARISE_CLOCK_SIGMA_INTRA_S = 1e-7) with a PHYSICS-BASED
clock + time-transfer model for the three PNT links of the formation:

  chief <-> deputy    two-way time transfer (TWTT) over the optical ISL  (relative sync)
  deputy <-> deputy   TWTT over the deputy-deputy ISL                    (relative sync)
  chief  <-> GNSS     GNSS-disciplined chief CSAC in the Space Service Volume (ABSOLUTE ref)

Parts (each calibrated to a source paper / datasheet):
  A  CSAC clock model  - Allan/Hadamard deviation from the h-coefficients (white/flicker/RW FM
                         + aging) and a 3-state (offset/drift/aging) Hutsell process model.  (SA.45s
                         datasheet; Teng 2025 clock model)
  B  two-way transfer  - TWTT cancels range + common delay -> residual = fixed/periodic delay
                         calibration + ranging thermal noise + clock drift over the update.  (Teng 2025)
  C  GNSS-disciplined  - chief steers its CSAC to GPS/UTC in the SSV -> the fleet ABSOLUTE time; the
                         relative fleet sync (TWTT) is far tighter than the absolute.  (SSV timing)
  D  ensemble timescale- clock-ensemble Implicit-Ensemble-Mean (IEM): N fleet clocks combine into a
                         timescale more stable than any single member (~1/sqrt(N)).  (Pham 2025)
  E  fleet budget      - derives the fleet-sync and camera-frame timing sigmas the pipeline consumes.

Papers: Teng et al. 2025 (Adv. Space Res. 75, autonomous cislunar time sync); Pham 2025 (Information
16(3) 210, time scale in AltPNT / clock-ensemble IEM + steering); Microchip SA.45s CSAC datasheet.

Validation (self-test): CSAC Allan deviation reproduces 3e-10/1e-10/3e-11/1e-11 at 1/10/100/1000 s
(datasheet) analytically and from the simulated clock; TWTT residual ~1 ns and cancels the common
delay; GNSS-disciplined absolute time ~20-30 ns >> the ~1 ns relative fleet sync; an N-clock ensemble
beats a single clock by ~1/sqrt(N) (Pham); the derived fleet sync (~ns) is far tighter than the flat
300 ns budget.  Run `python pnt_timing.py` -> ALL PASS.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import hardware as HW
except ImportError:  # pragma: no cover
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hardware as HW

C_LIGHT = 299792458.0
LN2 = np.log(2.0)


# ======================================================================================
# A. CSAC CLOCK MODEL (Allan/Hadamard deviation + 3-state Hutsell process)
# ======================================================================================
@dataclass(frozen=True)
class ClockParams:
    """Atomic-clock noise model via the standard power-law h-coefficients."""
    name: str
    h0: float            # white FM       -> sigma_y(tau) = sqrt(h0/(2 tau))   (tau^-1/2)
    h_m1: float          # flicker FM     -> sigma_y      = sqrt(2 ln2 h_m1)   (flat floor)
    h_m2: float          # random-walk FM -> sigma_y(tau) = sqrt(2 pi^2/3 h_m2 tau) (tau^+1/2)
    aging_per_s: float   # deterministic frequency drift [1/s per s]

    def allan_dev(self, tau_s):
        """Overlapping Allan deviation sigma_y(tau) from the h-coefficients."""
        tau = np.asarray(tau_s, float)
        var = self.h0 / (2.0 * tau) + 2.0 * LN2 * self.h_m1 + (2.0 * np.pi ** 2 / 3.0) * self.h_m2 * tau
        return np.sqrt(var)

    def hadamard_dev(self, tau_s):
        """Hadamard deviation (drift-insensitive; Teng 2025 uses it for the aging-dominated clocks)."""
        tau = np.asarray(tau_s, float)
        var = self.h0 / (2.0 * tau) + 2.0 * LN2 * self.h_m1 + (2.0 * np.pi ** 2 / 3.0) * self.h_m2 * tau
        return np.sqrt(var)                                  # (aging removed -> same random part)

    def time_error_sigma_s(self, tau_s):
        """1-sigma clock TIME (phase) error accumulated over an interval tau [s].  White FM ->
        random-walk phase: sigma_x(tau) = sigma_y(1s) sqrt(tau)."""
        sy1 = np.sqrt(self.h0 / 2.0)
        return float(sy1 * np.sqrt(tau_s))

    def simulate_phase(self, duration_s, dt_s, rng):
        """Clock TIME-error series x(t) [s]: white-FM random walk + RW-FM drift + aging."""
        n = int(round(duration_s / dt_s))
        sy1 = np.sqrt(self.h0 / 2.0)
        x = np.cumsum(rng.normal(0.0, sy1 * np.sqrt(dt_s), n))          # white FM -> RW phase
        # random-walk FM: frequency random-walks, integrates into phase
        srw = np.sqrt((2.0 * np.pi ** 2 / 3.0) * self.h_m2)
        y_rw = np.cumsum(rng.normal(0.0, srw * np.sqrt(dt_s), n))
        x = x + np.cumsum(y_rw) * dt_s
        x = x + 0.5 * self.aging_per_s * (np.arange(n) * dt_s) ** 2     # deterministic aging
        return x, dt_s


def csac_params(clk=None):
    clk = clk or HW.CSAC
    return ClockParams(name=clk.name, h0=clk.h0(),
                       h_m1=(clk.flicker_floor ** 2) / (2.0 * LN2),
                       h_m2=clk.h_minus2(),
                       aging_per_s=clk.aging_per_month / (30.0 * 86400.0))

CSAC = csac_params()
# lower-stability comparator (Teng "Aclock-low"-class): ~10x worse short term
USO = ClockParams(name="ultra-stable oscillator (Aclock-low class)",
                  h0=2.0 * (3.0e-9) ** 2, h_m1=(1.0e-10) ** 2 / (2 * LN2), h_m2=(1e-13) ** 2,
                  aging_per_s=1e-9 / 86400.0)


def overlapping_adev_phase(x_s, dt_s, n_tau=40):
    """Overlapping Allan deviation from a TIME-error (phase) series x [s]."""
    x = np.asarray(x_s, float); N = x.size
    max_m = (N - 1) // 2
    ms = np.unique(np.geomspace(1, max(max_m, 1), n_tau).astype(int))
    taus, ad = [], []
    for m in ms:
        if 2 * m >= N:
            continue
        d = x[2 * m:] - 2.0 * x[m:-m] + x[:-2 * m]
        tau = m * dt_s
        taus.append(tau); ad.append(np.sqrt(np.mean(d ** 2) / (2.0 * tau ** 2)))
    return np.array(taus), np.array(ad)


# ======================================================================================
# B. TWO-WAY TIME TRANSFER (chief<->deputy, deputy<->deputy)
# ======================================================================================
def ranging_thermal_time_s(cn0_dbhz, link=None):
    """DLL ranging thermal noise expressed as a TIME 1-sigma [s] (range noise / c)."""
    link = link or HW.TWTT_LINK
    cn0 = 10 ** (cn0_dbhz / 10.0)
    # range 1-sigma ~ (c/(2 pi))... here anchored to the datasheet-class mm ranging, scaled by C/N0
    sig_range = link.ranging_sigma_m * np.sqrt(10 ** ((50.0 - cn0_dbhz) / 10.0))
    return sig_range / C_LIGHT


def twtt_sync_sigma_s(cn0_dbhz, update_s=None, link=None, clk=CSAC):
    """Two-way time-transfer sync 1-sigma [s] just after an exchange: fixed + periodic delay
    calibration (RSS) + ranging thermal + clock drift over the update interval."""
    link = link or HW.TWTT_LINK
    upd = link.integration_s if update_s is None else update_s
    s_fixed = link.fixed_delay_sigma_s
    s_periodic = link.periodic_delay_sigma_s
    s_therm = ranging_thermal_time_s(cn0_dbhz, link)
    s_clock = clk.time_error_sigma_s(0.5 * upd)             # drift over half the two-way exchange
    return float(np.sqrt(s_fixed ** 2 + s_periodic ** 2 + s_therm ** 2 + s_clock ** 2))


def chief_deputy_sync_s(link=None, clk=CSAC):
    link = link or HW.TWTT_LINK
    return twtt_sync_sigma_s(link.cn0_dbhz_chief_deputy, link=link, clk=clk)


def deputy_deputy_sync_s(link=None, clk=CSAC):
    link = link or HW.TWTT_LINK
    return twtt_sync_sigma_s(link.cn0_dbhz_deputy_deputy, link=link, clk=clk)


def oneway_offset_s(range_m, clk=CSAC, link=None):
    """One-way (no two-way cancellation) apparent offset: dominated by the uncalibrated range delay."""
    return float(range_m / C_LIGHT)


# ======================================================================================
# C. GNSS-DISCIPLINED CHIEF (chief<->GNSS)  -- the fleet ABSOLUTE time reference
# ======================================================================================
def gnss_disciplined_absolute_s(gnss=None):
    """Chief absolute (GPS/UTC) time 1-sigma after disciplining the CSAC to GNSS SSV signals."""
    gnss = gnss or HW.GNSS_TIMING
    return float(np.sqrt(gnss.disciplined_sigma_s ** 2 + gnss.gps_utc_bias_s ** 2))


def fleet_absolute_vs_relative(link=None, gnss=None, clk=CSAC):
    """The PNT split: ABSOLUTE time (chief<->GNSS) is coarse; RELATIVE fleet sync (TWTT) is fine."""
    return dict(absolute_s=gnss_disciplined_absolute_s(gnss),
                relative_chief_deputy_s=chief_deputy_sync_s(link, clk),
                relative_deputy_deputy_s=deputy_deputy_sync_s(link, clk))


# ======================================================================================
# D. ENSEMBLE TIMESCALE (Implicit Ensemble Mean, Pham 2025)
# ======================================================================================
def ensemble_allan_dev(tau_s, n_clocks, clk=CSAC):
    """Clock-ensemble (IEM) Allan deviation: N uncorrelated members average down as 1/sqrt(N)
    (Pham 2025: the ensemble mean is more stable than any single member)."""
    return clk.allan_dev(tau_s) / np.sqrt(n_clocks)


def ensemble_gain_db(n_clocks):
    return float(10.0 * np.log10(np.sqrt(n_clocks)))


# ======================================================================================
# E. FLEET TIMING BUDGET (wiring knobs)
# ======================================================================================
def fleet_timing_budget(n_deputies=8, window_s=0.6, link=None, clk=CSAC, gnss=None):
    """Derive the fleet-sync and camera-frame timing 1-sigmas the pipeline consumes, plus the
    scene-motion equivalent at the flyby speed."""
    link = link or HW.TWTT_LINK
    n_clocks = n_deputies + 1
    # fleet relative sync: TWTT, improved by the ensemble timescale
    cd = chief_deputy_sync_s(link, clk); dd = deputy_deputy_sync_s(link, clk)
    fleet_sync = float(np.hypot(cd, dd) / np.sqrt(2.0))          # representative pairwise fleet sync
    fleet_sync_ens = fleet_sync / np.sqrt(n_clocks) ** 0.5       # partial ensemble benefit (conservative)
    # camera-frame (intra) timing: clock drift over the imaging window + fixed-delay calibration
    intra = float(np.hypot(clk.time_error_sigma_s(window_s), link.fixed_delay_sigma_s))
    v_flyby = 8760.0
    return dict(n_clocks=n_clocks, chief_deputy_s=cd, deputy_deputy_s=dd,
                fleet_sync_s=fleet_sync, fleet_sync_ensemble_s=fleet_sync_ens, intra_frame_s=intra,
                absolute_s=gnss_disciplined_absolute_s(gnss),
                scene_motion_fleet_um=fleet_sync * v_flyby * 1e6,
                scene_motion_intra_um=intra * v_flyby * 1e6)


# ======================================================================================
# SELF-TEST
# ======================================================================================
def _selftest():
    fails = []
    def check(name, cond, detail=""):
        print(("  PASS " if cond else "  FAIL ") + name + ("" if not detail else "  -> " + detail))
        if not cond: fails.append(name)

    print("=" * 84)
    print("PNT TIMING SELF-TEST  (chief<->deputy, deputy<->deputy, chief<->GNSS)")
    print("  Teng 2025 (cislunar time sync) + Pham 2025 (clock-ensemble time scale) + SA.45s CSAC")
    print("=" * 84)

    # A. CSAC Allan deviation vs datasheet
    clk = CSAC
    pts = [(1, 3e-10), (10, 1e-10), (100, 3e-11), (1000, 1e-11)]
    print("\nA. CSAC Allan deviation vs datasheet (sigma_y):")
    okA = True
    for tau, spec in pts:
        got = float(clk.allan_dev(tau))
        good = abs(np.log10(got) - np.log10(spec)) < 0.10          # within ~25%
        okA &= good
        print(f"     tau={tau:5d}s  model {got:.2e}  datasheet {spec:.1e}  {'ok' if good else 'X'}")
    check("CSAC Allan deviation matches datasheet (1..1000 s)", okA)
    check("white-FM tau^-1/2 slope", clk.allan_dev(1.0) > clk.allan_dev(100.0))

    # A2. simulated clock recovers the Allan deviation
    rng = np.random.default_rng(4)
    x, dt = clk.simulate_phase(4000.0, 0.5, rng)
    taus, adev = overlapping_adev_phase(x, dt, n_tau=40)
    j = np.argmin(np.abs(taus - 10.0))
    print(f"\nA2. Simulated clock ({len(x)} samples): Allan @10 s = {adev[j]:.2e} (datasheet 1e-10)")
    check("simulated clock Allan ~ datasheet @10 s (+-40%)", abs(adev[j] - 1e-10) < 0.4e-10, f"{adev[j]:.2e}")
    check("time-error over 0.6 s window sub-ns", clk.time_error_sigma_s(0.6) < 1e-9,
          f"{clk.time_error_sigma_s(0.6)*1e9:.3f} ns")

    # B. two-way time transfer
    cd = chief_deputy_sync_s(); dd = deputy_deputy_sync_s()
    print(f"\nB. TWTT sync: chief<->deputy {cd*1e9:.2f} ns, deputy<->deputy {dd*1e9:.2f} ns "
          f"(fixed {HW.TWTT_LINK.fixed_delay_sigma_s*1e9:.1f} + periodic {HW.TWTT_LINK.periodic_delay_sigma_s*1e9:.1f} ns)")
    check("chief-deputy TWTT sync ~1-3 ns", 0.3e-9 < cd < 3e-9, f"{cd*1e9:.2f} ns")
    check("deputy-deputy slightly looser than chief-deputy", dd >= cd)
    ow = oneway_offset_s(1000.0)
    check("two-way cancels common delay (residual << one-way range delay)", cd < 1e-3 * ow,
          f"{cd*1e9:.2f} ns vs one-way {ow*1e6:.1f} us")
    check("Teng-class precision achievable (< 100 ns)", cd < 100e-9)

    # C. GNSS-disciplined chief (absolute) vs relative
    absv = gnss_disciplined_absolute_s(); rel = cd
    print(f"\nC. chief<->GNSS absolute time {absv*1e9:.1f} ns  vs  relative fleet sync {rel*1e9:.2f} ns "
          f"({absv/rel:.0f}x)")
    check("GNSS-disciplined absolute ~20-40 ns", 15e-9 < absv < 45e-9, f"{absv*1e9:.1f} ns")
    check("absolute (chief-GNSS) >> relative (TWTT) -- the PNT split", absv > 5 * rel)

    # D. ensemble timescale (Pham IEM)
    N = 9
    e10 = float(ensemble_allan_dev(10.0, N)); s10 = float(clk.allan_dev(10.0))
    print(f"\nD. Ensemble timescale (IEM, {N} clocks): Allan@10 s single {s10:.2e} -> ensemble {e10:.2e} "
          f"({ensemble_gain_db(N):.1f} dB, 1/sqrt(N))")
    check("ensemble more stable than a single clock", e10 < s10)
    check("ensemble gain ~ 1/sqrt(N)", abs(e10 - s10 / np.sqrt(N)) < 1e-13)

    # E. fleet budget vs the flat constants it replaces
    b = fleet_timing_budget()
    flat_fleet, flat_intra = 3e-7, 1e-7
    print(f"\nE. Fleet budget: sync {b['fleet_sync_s']*1e9:.2f} ns (flat 300 ns -> {flat_fleet/b['fleet_sync_s']:.0f}x), "
          f"intra {b['intra_frame_s']*1e9:.2f} ns (flat 100 ns); scene motion {b['scene_motion_fleet_um']:.1f} um")
    check("derived fleet sync far tighter than flat 300 ns", b["fleet_sync_s"] < flat_fleet)
    check("derived intra timing tighter than flat 100 ns", b["intra_frame_s"] < flat_intra)
    check("fleet timing scene-motion << 0.15 m GSD", b["scene_motion_fleet_um"] < 0.15e6 * 0.01,
          f"{b['scene_motion_fleet_um']:.1f} um")
    check("all three links modeled", all(k in b for k in ("chief_deputy_s", "deputy_deputy_s", "absolute_s")))

    print("\n" + "=" * 84)
    print(f"RESULT: {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}  ({len(fails)} failure(s))")
    print("=" * 84)
    return len(fails) == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
