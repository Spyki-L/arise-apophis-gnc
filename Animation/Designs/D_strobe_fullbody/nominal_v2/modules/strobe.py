"""
strobe.py  -  ARISE V2 dark-side strobe model (B2) [HI-FI: flash-source + link + noise]
==============================================================================
The dark hemisphere is imaged by co-boresighted strobes (near-zero phase -> opposition surge
helps).  This models the strobe realistically and sizes it to the MINIMUM energy that reaches a
usable SNR with the fastest aperture (f/1.8), then checks it against the XB6 battery budget from
hardware.py (per your direction: dark side = lowest SNR consistent with battery capacity).

Per pulse:
  fluence at surface     F_surf = E_pulse / (Omega_beam * R^2) * vignette   [J/m^2]
  reflected radiant flux L_flu  = F_surf * brdf.radiance_over_Esun(i,e,~0)  [J/m^2/sr]
  collected electrons    Ne     = L_flu * A_aper * Omega_pix / E_photon * QE * opt
  SNR                    Ne / sqrt(Ne + read^2 + dark*tgate + dsnu^2 + q^2 + (prnu*Ne)^2)

Beam: co-boresighted, half-angle defaults to the body's angular radius at range (light only what
you image).  Recharge duty: capacitor recharge from the battery PSU limits the pulse rate near CA.

HI-FI upgrade (Hines, Pierrottet & Amzajerdian, "High-Fidelity Flash Lidar Model Development",
NASA LaRC / SPIE): the flash imaging chain modeled as  s = PSF(pulse) (*) object + noise, adding
  * TEMPORAL pulse profile (Gaussian, width tau) -> peak power + motion SMEAR-FREEZE at CA,
  * GAUSSIAN spatial beam (divergence theta_div) -> on-axis concentration + spill efficiency
    (vs the flat cone), so the fluence varies across the illuminated body,
  * FULL detector noise budget: shot + dark + read + DSNU + PRNU + quantization + Johnson +
    BACKGROUND THERMAL photon flux (Planck at the scene temperature over the imaging band).
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardware as HW
import brdf as BRDF
import detector as DET

DARK_SNR_MIN = 30.0          # ASSUMED: minimum per-pulse SNR for feature matching on the dark side
PSU_AVG_POWER_W = 30.0       # ASSUMED: avg electrical power available to recharge the strobe PSU
BODY_RADIUS_M = 205.0        # ~410 m mean diameter -> 205 m radius (radar shape)
# item #5: the battery recharges at ~0.1 Hz (>>500 Hz burst), so a PRE-CHARGED CAPACITOR BANK
# fires the close-approach burst.  The bank energy caps the dark-side pulses per deputy per pass.
STROBE_MAX_PULSES = int(os.getenv("ARISE_STROBE_MAX_PULSES", "12"))   # ~1.7 kg HV cap bank @ ~290 J/pulse elec
CAP_ENERGY_DENSITY_J_PER_G = 2.0   # ASSUMED HV pulse-cap specific energy


from dataclasses import dataclass

H_PLANCK = 6.62607e-34; C_LIGHT = 2.998e8; K_BOLTZ = 1.380649e-23


@dataclass(frozen=True)
class StrobeParams:
    """Flash-source + scene parameters for the hi-fi model (Hines et al.)."""
    pulse_width_s: float = 1.0e-5        # strobe temporal pulse FWHM (<=16 us freezes CA smear)
    beam_div_half_rad: float = 0.0       # Gaussian 1/e^2 half-angle; 0 -> matched to body at range
    wavelength_m: float = 0.65e-6        # imaging band centre (CIS120 visible)
    band_m: tuple = (0.40e-6, 1.00e-6)   # detector band for the thermal-background integral
    scene_temp_K: float = 300.0          # illuminated-optics/target temp for background thermal


def pulse_peak_power_W(E_pulse_opt, pw: StrobeParams = StrobeParams()):
    """Peak optical power of a Gaussian pulse of energy E and FWHM tau_p."""
    sigma_t = pw.pulse_width_s / 2.3548                       # FWHM -> sigma
    return E_pulse_opt / (sigma_t * np.sqrt(2.0 * np.pi))


def smear_pixels(v_perp_ms, range_m, pw: StrobeParams = StrobeParams(), ifov=HW.IFOV_RAD):
    """Image smear [px] accumulated during the strobe pulse: a short pulse FREEZES motion.
    (The flash acts as the effective exposure for the dark-side snapshot.)"""
    return abs(v_perp_ms) * pw.pulse_width_s / max(range_m, 1.0) / ifov


def gaussian_beam(E_pulse_opt, range_m, off_axis_rad=0.0, beam_half_rad=None,
                  body_radius_m=BODY_RADIUS_M, vignette=0.8):
    """Gaussian co-boresighted beam.  Returns (fluence[J/m^2] at off_axis_rad, on-body efficiency).
    Peak on-axis fluence is ~2x the flat-cone value; energy in the 1/e^2 body cone is 1-e^-2."""
    if beam_half_rad is None:
        beam_half_rad = body_half_angle(range_m, body_radius_m)          # match 1/e^2 radius to body
    w = beam_half_rad * range_m                                          # beam 1/e^2 radius at range [m]
    r = off_axis_rad * range_m
    F0 = 2.0 * E_pulse_opt / (np.pi * w**2) * vignette                   # on-axis fluence [J/m^2]
    F = F0 * np.exp(-2.0 * (r / max(w, 1e-9))**2)
    eta_body = 1.0 - np.exp(-2.0 * (body_radius_m / max(w, 1e-9))**2)    # fraction of energy on body
    return float(F), float(eta_body)


def background_thermal_electrons(t_gate_s, pw: StrobeParams = StrobeParams(),
                                 aperture_m=HW.APERTURE_MAX_M, det=None, focal_m=HW.FOCAL_M):
    """Background thermal (Planck) photo-electrons collected in a pixel over the gate (Hines Eq for
    M(lambda)).  Negligible at the visible band for a cold body -- included for completeness / NIR."""
    det = det or HW.CIS120
    la, lb = pw.band_m; T = pw.scene_temp_K
    lam = np.linspace(la, lb, 40)
    Lp = 2.0 * C_LIGHT / lam**4 / (np.exp(H_PLANCK * C_LIGHT / (lam * K_BOLTZ * T)) - 1.0)   # photons/s/m^2/sr/m
    A = np.pi * (aperture_m / 2.0)**2; Omega_pix = (det.pixel_m / focal_m)**2
    flux = np.trapz(Lp, lam) * A * Omega_pix                             # photons/s into a pixel
    return float(flux * det.qe_at_band * det.optical_eff * t_gate_s)


def snr_pulse_hifi(Ne, v_perp_ms=0.0, range_m=HW.DESIGN_STANDOFF_M, pw: StrobeParams = StrobeParams(),
                   det=None):
    """Per-pulse SNR with the FULL noise budget (shot+dark+read+DSNU+PRNU+quant+Johnson+background
    thermal), and a smear-dilution factor (a long pulse spreads the signal over >1 px)."""
    det = det or HW.CIS120
    Ne = float(Ne); t_gate = pw.pulse_width_s
    dark = det.dark_current_e_s() * t_gate
    q = (det.full_well_e / 2 ** det.adc_bits) / np.sqrt(12.0)
    bg = background_thermal_electrons(t_gate, pw, det=det)
    johnson = np.sqrt(4.0 * K_BOLTZ * pw.scene_temp_K * 1.0 / 1e6) / 1.6e-19 * 0.0   # ~subsumed in read for CMOS
    smear = max(smear_pixels(v_perp_ms, range_m, pw), 1.0)              # signal spread over 'smear' px
    Ne_eff = Ne / smear
    var = Ne_eff + dark + bg + det.read_noise_e**2 + det.dsnu_e**2 + q**2 + (det.prnu_frac * Ne_eff)**2 + johnson**2
    return Ne_eff / np.sqrt(var)


def cap_bank_budget(n_burst=STROBE_MAX_PULSES, target_snr=DARK_SNR_MIN, range_m=HW.DESIGN_STANDOFF_M,
                    bus: HW.XB6Bus = None):
    """Capacitor-bank + thermal budget for an N-pulse close-approach burst (item #5).
    Returns the bank energy/mass, the per-burst waste heat + source temperature rise, and the
    battery recharge time to refill the bank (long approach, not the sub-second burst)."""
    bus = bus or HW.XB6
    E_pulse, _, _ = energy_for_snr(target_snr, range_m)
    E_elec = E_pulse / bus.strobe_psu_eff
    E_cap = n_burst * E_elec
    heat = n_burst * E_pulse * (1.0 / bus.strobe_psu_eff - 1.0)     # electrical - optical -> waste heat
    return dict(n_burst=n_burst, E_pulse_opt_J=E_pulse, E_pulse_elec_J=E_elec,
                cap_energy_J=E_cap, cap_mass_kg=E_cap / (CAP_ENERGY_DENSITY_J_PER_G * 1e3),
                burst_heat_J=heat, recharge_time_s=E_cap / PSU_AVG_POWER_W,
                energy_pulses_total=int(bus.strobe_optical_energy_budget_J() / max(E_pulse, 1e-9)))


def body_half_angle(range_m, body_radius_m=BODY_RADIUS_M):
    return np.arctan2(body_radius_m, max(range_m, 1.0))


def beam_solid_angle(half_angle_rad):
    return 2.0 * np.pi * (1.0 - np.cos(half_angle_rad))


def electrons_per_pulse(E_pulse_opt, range_m, aperture_m, beam_half_rad,
                        i_deg=0.0, e_deg=0.0, alpha_deg=0.0, model="hapke",
                        vignette=0.8, det=None, focal_m=HW.FOCAL_M):
    det = det or HW.CIS120
    Omega_beam = beam_solid_angle(beam_half_rad)
    F_surf = E_pulse_opt / (Omega_beam * range_m**2) * vignette          # J/m^2 at surface
    Lf = BRDF.radiance_over_Esun(np.radians(i_deg), np.radians(e_deg), np.radians(alpha_deg), model=model)
    L_flu = F_surf * Lf                                                  # J/m^2/sr
    A = np.pi * (aperture_m / 2.0) ** 2
    Omega_pix = (det.pixel_m / focal_m) ** 2
    photons = L_flu * A * Omega_pix / DET.PHOTON_ENERGY_J
    return photons * det.qe_at_band * det.optical_eff


def snr_pulse(Ne, t_gate_s=1e-5, det=None):
    det = det or HW.CIS120
    Ne = float(Ne)
    dark = det.dark_current_e_s() * t_gate_s
    q = (det.full_well_e / 2 ** det.adc_bits) / np.sqrt(12.0)
    var = Ne + dark + det.read_noise_e**2 + det.dsnu_e**2 + q**2 + (det.prnu_frac * Ne)**2
    return Ne / np.sqrt(var)


def energy_for_snr(target_snr=DARK_SNR_MIN, range_m=HW.DESIGN_STANDOFF_M,
                   aperture_m=HW.APERTURE_MAX_M, beam_half_rad=None, model="hapke",
                   vignette=0.8, det=None):
    """Optical energy per pulse [J] to reach target_snr at range_m (opposition geometry)."""
    det = det or HW.CIS120
    if beam_half_rad is None:
        beam_half_rad = body_half_angle(range_m)
    # invert SNR(Ne) ~ for shot+read: Ne = (s^2 + sqrt(s^4 + 4 s^2 rn^2))/2
    rn2 = det.read_noise_e**2 + det.dsnu_e**2
    s2 = target_snr**2
    Ne = (s2 + np.sqrt(s2**2 + 4.0 * s2 * rn2)) / 2.0
    # back out E_pulse from electrons_per_pulse linear scaling
    Ne_at_1J = electrons_per_pulse(1.0, range_m, aperture_m, beam_half_rad, 0, 0, 0, model, vignette, det)
    E_pulse = Ne / max(Ne_at_1J, 1e-30)
    return float(E_pulse), float(Ne), float(beam_half_rad)


def dark_side_budget(target_snr=DARK_SNR_MIN, range_m=HW.DESIGN_STANDOFF_M, n_pulses=12,
                     aperture_m=HW.APERTURE_MAX_M, model="hapke", bus: HW.XB6Bus = None):
    """Full dark-side energy budget vs the XB6 battery, with recharge-duty pulse-rate limit."""
    bus = bus or HW.XB6
    E_pulse, Ne, bhalf = energy_for_snr(target_snr, range_m, aperture_m, None, model)
    E_total = E_pulse * n_pulses
    budget = bus.strobe_optical_energy_budget_J()
    # recharge duty: electrical energy per pulse / available power -> min spacing between pulses
    E_pulse_elec = E_pulse / bus.strobe_psu_eff
    pulse_period_s = E_pulse_elec / PSU_AVG_POWER_W
    max_rate_hz = 1.0 / pulse_period_s
    return dict(target_snr=target_snr, range_m=range_m, aperture_cm=aperture_m*1e2,
                beam_half_deg=float(np.degrees(bhalf)), Ne_per_pulse=Ne,
                E_pulse_opt_J=E_pulse, n_pulses=n_pulses, E_total_opt_J=E_total,
                budget_opt_J=budget, within_budget=bool(E_total <= budget),
                frac_of_budget=float(E_total / budget),
                recharge_period_s=float(pulse_period_s), max_pulse_rate_hz=float(max_rate_hz))


if __name__ == "__main__":
    print("ARISE V2 dark-side strobe sanity (CIS120 f/1.8, opposition geometry)")
    print("  XB6 optical budget: %.0f J/deputy" % HW.XB6.strobe_optical_energy_budget_J())
    for snr_t in (20.0, 30.0, 50.0):
        b = dark_side_budget(target_snr=snr_t, n_pulses=12)
        print("\n  [target dark SNR = %.0f, 12 pulses, R=750 m, beam half %.1f deg]"
              % (snr_t, b['beam_half_deg']))
        print("    Ne/pulse %.0f e-   E_pulse %.1f J opt   E_total %.0f J  (%.1f%% of budget) %s"
              % (b['Ne_per_pulse'], b['E_pulse_opt_J'], b['E_total_opt_J'],
                 100*b['frac_of_budget'], "OK" if b['within_budget'] else "OVER BUDGET"))
        print("    recharge: %.2f s/pulse  -> max %.1f Hz pulse rate" % (b['recharge_period_s'], b['max_pulse_rate_hz']))
    # range sensitivity at fixed SNR=30
    print("\n  range sensitivity (SNR=30, single pulse):")
    for R in (300., 750., 1500., 3000., 5000.):
        E, Ne, bh = energy_for_snr(30.0, range_m=R)
        print("    R=%5.0f m  beam half %4.1f deg  E_pulse %7.1f J opt" % (R, np.degrees(bh), E))

    print("\n  === HI-FI checks (flash source + beam + noise; Hines et al.) ===")
    fails=[]
    def chk(n,c,d=""):
        print(("  PASS " if c else "  FAIL ")+n+(""if not d else"  -> "+d)); (fails.append(n) if not c else None)
    pw = StrobeParams()
    # 1. temporal pulse -> smear-freeze at CA (v_perp ~8.76 km/s)
    sm16 = smear_pixels(8760.0, 700.0, StrobeParams(pulse_width_s=16e-6))
    sm1  = smear_pixels(8760.0, 700.0, StrobeParams(pulse_width_s=1e-6))
    print("  smear @700m, v=8.76km/s: 16us pulse -> %.2f px ; 1us -> %.2f px" % (sm16, sm1))
    chk("<=16us pulse freezes CA smear to ~1 px (DESIGN_D)", sm16 < 1.3)
    chk("shorter pulse freezes harder", sm1 < sm16)
    chk("pulse peak power = E/tau scale", pulse_peak_power_W(40.0, pw) > 40.0/pw.pulse_width_s*0.3)
    # 2. Gaussian beam: on-axis brighter than flat cone; edge dimmer; ~86% energy on matched body
    F0, eta = gaussian_beam(40.0, 750.0)
    Ffar,_ = gaussian_beam(40.0, 750.0, off_axis_rad=body_half_angle(750.0))
    F_unif = 40.0/(beam_solid_angle(body_half_angle(750.0))*750.0**2)*0.8
    print("  Gaussian on-axis %.3e vs flat %.3e J/m^2 ; body-edge %.3e ; on-body eff %.0f%%" % (F0, F_unif, Ffar, 100*eta))
    chk("Gaussian on-axis concentrates vs flat cone", F0 > F_unif)
    chk("Gaussian body-edge dimmer than centre", Ffar < F0)
    chk("energy on 1/e^2-matched body ~86%", 0.80 < eta < 0.90)
    # 3. background thermal negligible at visible band / cold body (vs a 1.5um LIDAR)
    bg_vis = background_thermal_electrons(1e-5, StrobeParams(scene_temp_K=300))
    bg_hot = background_thermal_electrons(1e-5, StrobeParams(scene_temp_K=300, band_m=(1.0e-6,2.0e-6)))
    print("  background thermal e- (300K): visible %.2e ; NIR 1-2um %.2e" % (bg_vis, bg_hot))
    chk("visible-band thermal background negligible (<1 e-)", bg_vis < 1.0)
    # 4. full-noise SNR sane + smear dilutes SNR
    ne = 2000.0
    snr_frozen = snr_pulse_hifi(ne, v_perp_ms=8760, range_m=700, pw=StrobeParams(pulse_width_s=1e-6))
    snr_smear  = snr_pulse_hifi(ne, v_perp_ms=8760, range_m=700, pw=StrobeParams(pulse_width_s=100e-6))
    print("  hi-fi SNR (Ne=2000): frozen(1us) %.1f ; smeared(100us) %.1f" % (snr_frozen, snr_smear))
    chk("full-noise SNR positive & smear dilutes it", snr_frozen > snr_smear > 0)
    # 5. energy budget within the (upgraded) battery
    b = dark_side_budget(target_snr=30.0, n_pulses=12)
    print("  dark-side 12-pulse budget: %.0f J of %.0f J (%.1f%%) %s" % (b['E_total_opt_J'], b['budget_opt_J'], 100*b['frac_of_budget'], "OK" if b['within_budget'] else "OVER"))
    chk("dark-side burst within the battery budget", b['within_budget'])
    print("\n  RESULT:", "ALL PASS" if not fails else "FAILED: "+",".join(fails))
    import sys; sys.exit(0 if not fails else 1)
