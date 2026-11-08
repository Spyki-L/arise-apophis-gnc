"""
detector.py  -  ARISE V2 full detector + optics chain (B1)
==========================================================
Turns the binary "SNR > 300" gate into a realistic radiometric chain and a smooth detection
probability, for the Teledyne e2v CIS120 focal plane defined in hardware.py.  Models:

  signal   : etendue radiometry, resolved extended source (range-independent per pixel while
             resolved); L = E_sun * radiance_factor(i,e,alpha) from brdf.py
  noise     : shot (signal+dark) + read + DSNU + PRNU*signal + ADC quantization
  saturation: full-well clip
  optics    : MTF/PSF Gaussian blur (sigma_psf) + motion smear -> effective contrast & a
              resolved-feature test
  detection : P_det(SNR) smooth logistic/erf instead of a hard threshold

Plus the V2 design driver:
  size_sunlit_aperture(...) -> choose the variable stop + exposure (+ co-add) so the sunlit
  SNR lands in [300,400] at the smear-limited exposure, honoring f/# >= 1.8.

Radiometry convention: brdf.radiance_over_Esun already includes cos(incidence); do NOT
re-apply mu0 here.
"""
from __future__ import annotations
import os, sys
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardware as HW
import brdf as BRDF

# Physical constants
H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
EFF_WAVELENGTH_M = 600e-9                      # peak-QE wavelength (CIS120 BSI)
PHOTON_ENERGY_J = H_PLANCK * C_LIGHT / EFF_WAVELENGTH_M   # ~3.31e-19 J
SOLAR_CONST_1AU = 1361.0                       # W/m^2 at 1 AU
AU_M = 1.495978707e11

# Detection-probability shape
SNR_DETECT_THRESH = 5.0                        # ASSUMED: SNR for ~50% single-frame detection
SNR_DETECT_WIDTH = 1.5                         # ASSUMED: logistic width


def solar_irradiance(helio_dist_au: float = 1.0) -> float:
    return SOLAR_CONST_1AU / helio_dist_au**2


def pixel_solid_angle(focal_m: float = HW.FOCAL_M, pixel_m: float = None) -> float:
    pixel_m = HW.CIS120.pixel_m if pixel_m is None else pixel_m
    return (pixel_m / focal_m) ** 2            # IFOV^2 [sr], = Omega subtended by a pixel


def collected_electrons(L_over_Esun, E_sun, aperture_m, t_exp_s,
                        det: HW.CIS120Detector = None, focal_m=HW.FOCAL_M):
    """Signal electrons per pixel from a resolved, illuminated surface patch."""
    det = det or HW.CIS120
    A = np.pi * (aperture_m / 2.0) ** 2
    Omega = pixel_solid_angle(focal_m, det.pixel_m)
    L = E_sun * np.asarray(L_over_Esun, float)            # scene radiance [W/m^2/sr]
    P_pix = L * A * Omega                                  # power on a pixel [W]
    photons = P_pix / PHOTON_ENERGY_J * t_exp_s
    return photons * det.qe_at_band * det.optical_eff


def smear_pixels(t_exp_s, range_m, v_perp_ms, ifov=HW.IFOV_RAD):
    """Image smear in pixels from transverse relative motion during the exposure."""
    omega = v_perp_ms / max(range_m, 1.0)                  # LOS angular rate [rad/s]
    return omega * t_exp_s / ifov


def snr(Ne, t_exp_s, n_coadd=1, det: HW.CIS120Detector = None, temp_K=None):
    """Per-pixel SNR for n_coadd co-added frames (independent read/shot noise)."""
    det = det or HW.CIS120
    Ne = np.asarray(Ne, float)
    # identity-when-absent hi-fi detector knobs (exported by detector_hifi.py under ARISE_DETECTOR_HIFI);
    # unset -> exact flat CIS120 constants.
    dark_e_s = float(os.getenv("ARISE_DET_DARK_E_S", det.dark_current_e_s(temp_K)))
    read_e = float(os.getenv("ARISE_DET_READ_E", det.read_noise_e))
    dsnu_e = float(os.getenv("ARISE_DET_DSNU_E", det.dsnu_e))
    prnu = float(os.getenv("ARISE_DET_PRNU_FRAC", det.prnu_frac))
    dark = dark_e_s * t_exp_s
    q = (det.full_well_e / 2 ** det.adc_bits) / np.sqrt(12.0)     # ADC quantization [e-]
    var_frame = (Ne + dark) + read_e**2 + dsnu_e**2 + q**2 + (prnu * Ne)**2
    snr_frame = Ne / np.sqrt(var_frame)
    return snr_frame * np.sqrt(n_coadd)


def saturated(Ne, det: HW.CIS120Detector = None):
    det = det or HW.CIS120
    return np.asarray(Ne, float) >= det.full_well_e


def detection_probability(snr_value):
    """Smooth single-frame detection probability from SNR (erf logistic)."""
    from math import erf
    z = (np.asarray(snr_value, float) - SNR_DETECT_THRESH) / (np.sqrt(2.0) * SNR_DETECT_WIDTH)
    vec = np.vectorize(lambda x: 0.5 * (1.0 + erf(x)))
    return vec(z)


def effective_resolution_px(t_exp_s, range_m, v_perp_ms, sigma_psf_px=None, ifov=HW.IFOV_RAD):
    """Combined PSF + smear blur [px]; a feature is 'resolved' if blur <~ 1.5 px.  sigma_psf_px None ->
    ARISE_DET_SIGMA_PSF_PX (hi-fi charge-diffusion+IPC PSF) else the flat 0.7 px optics value."""
    if sigma_psf_px is None:
        sigma_psf_px = float(os.getenv("ARISE_DET_SIGMA_PSF_PX", 0.7))
    sm = smear_pixels(t_exp_s, range_m, v_perp_ms, ifov)
    return np.sqrt(sigma_psf_px**2 + (sm / np.sqrt(12.0))**2), sm


# ── system MTF + stray light (item #3: detector & optics realism) ─────────────────────────────
ABERR_WFE_RMS_WAVES = 0.07     # ASSUMED rms wavefront error [waves @600nm] (good, not perfect optics)
GLARE_FRACTION      = 1.0e-4   # ASSUMED veiling-glare / PST: fraction of an in-field BRIGHT-source
                               #   signal scattered onto dark-side pixels (sunlit-limb stray light).
                               #   1e-4 = good baffle/coatings; 1e-3 poorly baffled; 1e-5 excellent.
MTF_RESOLVE_MIN     = 0.10     # below this contrast at Nyquist, GSD-scale features are NOT resolved


def system_mtf_nyquist(fnum, smear_px=1.0, pixel_m=None, wfe_rms_waves=ABERR_WFE_RMS_WAVES,
                       wavelength_m=EFF_WAVELENGTH_M, det: HW.CIS120Detector = None):
    """System MTF at the detector Nyquist frequency = diffraction x pixel-aperture x motion-smear x
    aberration.  This is the CONTRAST actually available for sub-pixel feature matching at the GSD;
    geometric GSD alone overstates the resolution.  Returns (mtf, components)."""
    det = det or HW.CIS120
    pixel_m = det.pixel_m if pixel_m is None else pixel_m
    f_ny = 1.0 / (2.0 * pixel_m)                          # pixel Nyquist [cyc/m]
    f_cut = 1.0 / (wavelength_m * max(fnum, 1e-6))        # incoherent diffraction cutoff [cyc/m]
    nu = min(f_ny / f_cut, 1.0)
    mtf_diff = (2.0 / np.pi) * (np.arccos(nu) - nu * np.sqrt(max(1.0 - nu * nu, 0.0)))
    mtf_pix = abs(np.sinc(0.5))                           # pixel aperture: sin(pi/2)/(pi/2)=0.637
    # motion smear over `smear_px` pixels.  sinc has a null at the Nyquist period (smear=2 px) and
    # phase-reversed sidelobes beyond it -> clamp to 0 past the first null (feature unresolved).
    mtf_smear = abs(np.sinc(0.5 * max(smear_px, 1e-6))) if smear_px < 2.0 else 0.0
    mtf_aberr = float(np.exp(-(2.0 * np.pi * wfe_rms_waves) ** 2))   # Marechal wavefront factor
    mtf = float(mtf_diff * mtf_pix * mtf_smear * mtf_aberr)
    return mtf, dict(diffraction=float(mtf_diff), pixel=float(mtf_pix),
                     smear=float(mtf_smear), aberration=mtf_aberr, nyquist_cyc_per_m=float(f_ny))


def stray_light_background_e(bright_signal_e, glare_frac=GLARE_FRACTION):
    """Veiling-glare pedestal [e-] on a dark-side pixel from a bright in-field source (the sunlit
    limb, or strobe back-scatter): PST x the bright-source per-pixel signal.  This adds shot noise
    and a non-removable background that cuts dark-side contrast."""
    return float(glare_frac) * np.asarray(bright_signal_e, float)


def size_sunlit_aperture(range_m=HW.DESIGN_STANDOFF_M, v_perp_ms=8760.0,
                         i_deg=30.0, e_deg=10.0, alpha_deg=25.0,
                         helio_dist_au=1.0, smear_budget_px=1.0,
                         band=HW.SNR_LIT_BAND, model="hapke", det=None):
    """Pick the variable stop + exposure (+ co-add) so sunlit SNR lands in `band`.
    Strategy: set exposure to the smear budget (<=1 px), open to f/1.8; if a single frame
    overshoots `band`, stop the aperture down; if it undershoots, co-add frames."""
    det = det or HW.CIS120
    E_sun = solar_irradiance(helio_dist_au)
    Lf = BRDF.radiance_over_Esun(np.radians(i_deg), np.radians(e_deg), np.radians(alpha_deg), model=model)
    # smear-limited exposure for 1 px (cap at sensible max)
    t_exp = min(smear_budget_px * HW.IFOV_RAD * range_m / max(v_perp_ms, 1.0), 5e-4)
    lo, hi = band
    D = HW.APERTURE_MAX_M                                  # start fastest (f/1.8)
    Ne = collected_electrons(Lf, E_sun, D, t_exp, det)
    s1 = float(snr(Ne, t_exp, 1, det))
    n_coadd = 1
    note = ""
    if Ne >= det.full_well_e:
        # would saturate -> stop down so signal ~ 0.8 full well
        D = D * np.sqrt(0.8 * det.full_well_e / Ne)
        Ne = collected_electrons(Lf, E_sun, D, t_exp, det); s1 = float(snr(Ne, t_exp, 1, det))
        note = "stopped down to avoid saturation; "
    if s1 > hi:
        # shot-limited SNR ~ D ; scale aperture down to hit band centre
        target = 0.5 * (lo + hi)
        D = max(D * target / s1, 1e-3)
        Ne = collected_electrons(Lf, E_sun, D, t_exp, det); s1 = float(snr(Ne, t_exp, 1, det))
        note += "stopped down to land in band"
    elif s1 < lo:
        n_coadd = int(np.ceil((lo / max(s1, 1e-6)) ** 2))
        note += "f/1.8 photon-limited; co-add %d frames to reach band" % n_coadd
    s_tot = float(snr(Ne, t_exp, n_coadd, det))
    fnum = HW.FOCAL_M / D
    sm = float(smear_pixels(t_exp, range_m, v_perp_ms))
    return dict(aperture_m=float(D), fnum=float(fnum), fnum_ok=bool(fnum >= HW.FNUM_MIN - 1e-9),
                t_exp_s=float(t_exp), smear_px=sm, Ne_per_frame=float(Ne),
                snr_single=s1, n_coadd=int(n_coadd), snr_total=s_tot,
                saturated=bool(Ne >= det.full_well_e), note=note, model=model,
                E_sun_Wm2=float(E_sun), radiance_factor=float(Lf))


if __name__ == "__main__":
    print("ARISE V2 detector chain sanity (CIS120, focal 50mm)")
    print("  photon energy @600nm : %.3e J   solar @1AU : %.0f W/m^2" % (PHOTON_ENERGY_J, solar_irradiance()))
    print("  pixel solid angle    : %.3e sr  (IFOV^2)" % pixel_solid_angle())
    for model in ("lambert", "hapke", "ls"):
        r = size_sunlit_aperture(model=model)
        print("\n  [sunlit sizing, %s BRDF]" % model)
        print("    aperture %.2f cm  f/%.2f %s   t_exp %.1f us   smear %.2f px"
              % (r['aperture_m']*1e2, r['fnum'], '' if r['fnum_ok'] else '(!f/1.8)',
                 r['t_exp_s']*1e6, r['smear_px']))
        print("    Ne/frame %.0f e-  SNR_single %.0f  -> n_coadd %d  SNR_total %.0f"
              % (r['Ne_per_frame'], r['snr_single'], r['n_coadd'], r['snr_total']))
        print("    note: %s" % r['note'])
    # detection-probability curve
    ss = np.array([2, 4, 5, 7, 10, 20])
    print("\n  P_det(SNR) :", dict(zip(ss.tolist(), np.round(detection_probability(ss), 3).tolist())))
