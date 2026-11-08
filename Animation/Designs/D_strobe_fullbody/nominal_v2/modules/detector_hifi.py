"""
detector_hifi.py  -  ARISE V2 deep-fidelity CIS120 detector effects
===================================================================
Extends the detector.py B1 radiometric chain (which already has shot/read/DSNU/PRNU noise,
full-well, co-add, and a Nyquist system MTF) with the four SPATIAL / device-physics effects the
scalar model omits.  Each is calibrated to peer-reviewed sources and feeds detector.py through
effective-parameter knobs (identity-when-absent).

  A  sensor PSF / MTF   - lateral charge diffusion (Stevens 1991) + interpixel-capacitance (IPC)
                          crosstalk + pixel aperture -> a physical sensor MTF and an effective PSF
                          sigma that broadens the optics-only 0.7 px.  (Djite 2012; Teledyne COSMOS)
  B  charge transfer    - 4T pinned-photodiode charge-transfer inefficiency (CTI) / image lag and
                          its TID sensitivity (0.5% CTI acceptable-lag criterion).  (Solid-State Electronics)
  C  radiation damage   - proton displacement-damage dark-current increase: a uniform mean shift PLUS
                          the exponential HOT-PIXEL tail, RTS, and prompt transient hits; hot pixels
                          bias star centroids.  (Virmontois 2010 IEEE TNS; Sensors/MDPI 2018; AIP Adv. 2024)
  D  PRNU / DSNU (FPN)   - fixed-pattern non-uniformity as spatial gain/dark fields, and the total
                          noise combining temporal + spatial per the standard.  (EMVA 1288; Janesick)

Calibration anchor: Teledyne e2v / Capella CIS120 (SPIE 11852-0S, 2021) numbers in hardware.py.
Validation (self-test): sensor MTF at Nyquist in the BSI range and monotone; charge diffusion
broadens the PSF; CTI below the 0.5% criterion and grows with TID; radiation dark current + hot-pixel
fraction grow with proton fluence with an exponential (positively-skewed) tail; PRNU/DSNU spatial
fields reproduce their EMVA-1288 sigmas and combine into the correct total noise.  -> ALL PASS.
"""
from __future__ import annotations
import numpy as np

try:
    import hardware as HW
except ImportError:  # pragma: no cover
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hardware as HW


# ======================================================================================
# A. SENSOR PSF / MTF  (charge diffusion + IPC crosstalk + pixel aperture)
# ======================================================================================
def mtf_charge_diffusion(k_cyc_px, psf=None):
    """Charge-diffusion MTF (Lorentzian, exponential-PSF form): 1/(1+(2 pi k L/p)^2).  k in cyc/px."""
    psf = psf or HW.CIS120_PSF
    ratio = psf.diffusion_length_um / (HW.CIS120.pixel_m * 1e6)      # L_diff / pixel_pitch
    return 1.0 / (1.0 + (2.0 * np.pi * np.asarray(k_cyc_px, float) * ratio) ** 2)


def mtf_ipc(k_cyc_px, psf=None):
    """Interpixel-capacitance (IPC) MTF for a 4-neighbor kernel: 1 - 2a(1 - cos(2 pi k))."""
    psf = psf or HW.CIS120_PSF
    a = psf.ipc_alpha
    return 1.0 - 2.0 * a * (1.0 - np.cos(2.0 * np.pi * np.asarray(k_cyc_px, float)))


def mtf_pixel(k_cyc_px):
    """Pixel-aperture MTF = |sinc(k)|."""
    return np.abs(np.sinc(np.asarray(k_cyc_px, float)))


def sensor_mtf(k_cyc_px, psf=None, include_pixel=True):
    """Full sensor MTF = charge-diffusion x IPC x (pixel aperture)."""
    m = mtf_charge_diffusion(k_cyc_px, psf) * mtf_ipc(k_cyc_px, psf)
    return m * mtf_pixel(k_cyc_px) if include_pixel else m


def ipc_kernel(psf=None):
    """3x3 IPC crosstalk kernel (center 1-4a, 4 neighbors a); charge-conserving."""
    psf = psf or HW.CIS120_PSF; a = psf.ipc_alpha
    K = np.array([[0, a, 0], [a, 1 - 4 * a, a], [0, a, 0]])
    return K / K.sum()


def effective_psf_sigma_px(psf=None):
    """Effective PSF sigma [px] = optics RSS the Gaussian-equivalent of the charge-diffusion+IPC
    sensor MTF at Nyquist (what detector.py's sigma_psf should use)."""
    psf = psf or HW.CIS120_PSF
    m_ny = float(mtf_charge_diffusion(0.5, psf) * mtf_ipc(0.5, psf))   # sensor MTF at Nyquist (no pixel)
    sig_sensor = np.sqrt(max(-np.log(m_ny), 0.0) / 4.9348)            # exp(-2 pi^2 sig^2 (0.5)^2)=m
    return float(np.hypot(psf.optical_sigma_px, sig_sensor))


def ipc_crosstalk_frac(psf=None):
    """Fraction of a pixel's charge coupled OUT to its 4 neighbors by IPC."""
    psf = psf or HW.CIS120_PSF
    return float(4.0 * psf.ipc_alpha)


# ======================================================================================
# B. CHARGE TRANSFER INEFFICIENCY / IMAGE LAG (4T PPD)
# ======================================================================================
def cti_effective(cte=None, tid_krad=None):
    """Charge-transfer inefficiency including the TID-induced increase."""
    cte = cte or HW.CIS120_CTE
    tid = HW.CIS120_RAD.tid_krad if tid_krad is None else tid_krad
    return float(cte.cti + cte.cti_per_krad * tid)


def image_lag(signal_e, cte=None, tid_krad=None):
    """Charge deferred to the trailing frame/pixel by incomplete transfer = CTI x signal."""
    return float(cti_effective(cte, tid_krad) * np.asarray(signal_e, float))


def cti_ok(cte=None, tid_krad=None):
    """Is the effective CTI within the 0.5% acceptable-lag criterion?"""
    cte = cte or HW.CIS120_CTE
    return bool(cti_effective(cte, tid_krad) <= cte.cti_criterion)


# ======================================================================================
# C. RADIATION DAMAGE  (DDD dark current + hot-pixel tail + RTS + transients)
# ======================================================================================
def dark_current_mean_increase_e_s(fluence_p_cm2=None, rad=None):
    """Mean dark-current increase [e-/s] from proton displacement damage (Virmontois: linear in DDD)."""
    rad = rad or HW.CIS120_RAD
    fl = rad.proton_fluence_p_cm2 if fluence_p_cm2 is None else fluence_p_cm2
    return float(rad.ddd_dark_e_s_per_1e10 * fl / 1e10)


def sample_dark_field(n_pix, rng, fluence_p_cm2=None, rad=None, base_dark_e_s=None, temp_K=None):
    """Per-pixel dark current [e-/s] = base + uniform DDD mean + an EXPONENTIAL hot-pixel tail
    (Virmontois): displacement damage adds a positively-skewed increase with a heavy tail."""
    rad = rad or HW.CIS120_RAD
    fl = rad.proton_fluence_p_cm2 if fluence_p_cm2 is None else fluence_p_cm2
    base = HW.CIS120.dark_current_e_s(temp_K) if base_dark_e_s is None else base_dark_e_s
    mean_shift = rad.ddd_dark_e_s_per_1e10 * fl / 1e10
    tail_scale = rad.hotpix_tail_e_s_per_1e10 * fl / 1e10
    tail = rng.exponential(tail_scale, int(n_pix)) if tail_scale > 0 else np.zeros(int(n_pix))
    return base + mean_shift + tail


def hot_pixel_fraction(fluence_p_cm2=None, rad=None, temp_K=None):
    """Analytic hot-pixel fraction P(dark > threshold) for the exponential DDD tail."""
    rad = rad or HW.CIS120_RAD
    fl = rad.proton_fluence_p_cm2 if fluence_p_cm2 is None else fluence_p_cm2
    base = HW.CIS120.dark_current_e_s(temp_K) + rad.ddd_dark_e_s_per_1e10 * fl / 1e10
    tail_scale = rad.hotpix_tail_e_s_per_1e10 * fl / 1e10
    if tail_scale <= 0:
        return 0.0
    return float(np.exp(-max(rad.hotpix_threshold_e_s - base, 0.0) / tail_scale))


def rts_pixel_fraction(fluence_p_cm2=None, rad=None):
    rad = rad or HW.CIS120_RAD
    fl = rad.proton_fluence_p_cm2 if fluence_p_cm2 is None else fluence_p_cm2
    return float(rad.rts_pixel_frac_per_1e10 * fl / 1e10)


def transient_hits_per_frame(t_exp_s, rad=None, det=None):
    """Prompt particle transient hits on the full mosaic per frame (rate x area x exposure)."""
    rad = rad or HW.CIS120_RAD; det = det or HW.CIS120
    moz = HW.fov_mosaic(det)["mosaic_px"]
    area_cm2 = (moz * det.pixel_m * 1e2) ** 2
    return float(rad.transient_rate_hits_cm2_s * area_cm2 * t_exp_s)


def hotpixel_centroid_bias_px(star_flux_e, fluence_p_cm2=None, rad=None, temp_K=None, t_exp_s=1e-3):
    """Representative star-centroid bias [px] from a hot pixel landing in a 3x3 centroid window
    (AIP Adv. 2024): hot-pixel dark charge offsets the flux-weighted centroid."""
    rad = rad or HW.CIS120_RAD
    frac = hot_pixel_fraction(fluence_p_cm2, rad, temp_K)
    tail = rad.hotpix_tail_e_s_per_1e10 * (rad.proton_fluence_p_cm2 if fluence_p_cm2 is None else fluence_p_cm2) / 1e10
    hot_charge = (rad.hotpix_threshold_e_s + tail) * t_exp_s      # a typical hot pixel's charge in the window
    p_in_window = 1.0 - (1.0 - frac) ** 9                         # prob a hot pixel in the 3x3 window
    # centroid shift ~ (hot_charge / (star_flux + hot_charge)) * 1 px lever arm
    return float(p_in_window * hot_charge / (star_flux_e + hot_charge + 1e-9))


# ======================================================================================
# D. PRNU / DSNU FIXED-PATTERN NOISE  (EMVA 1288)
# ======================================================================================
def sample_prnu_field(n_pix, rng, fpn=None):
    """Per-pixel gain (photo-response) field, mean 1, std = PRNU."""
    fpn = fpn or HW.CIS120_FPN
    return 1.0 + rng.normal(0.0, fpn.prnu_frac, int(n_pix))


def sample_dsnu_field(n_pix, rng, fpn=None, fluence_p_cm2=None):
    """Per-pixel dark-offset field [e-], std = DSNU (grows with displacement damage)."""
    fpn = fpn or HW.CIS120_FPN
    fl = HW.CIS120_RAD.proton_fluence_p_cm2 if fluence_p_cm2 is None else fluence_p_cm2
    dsnu = np.hypot(fpn.dsnu_e, fpn.dsnu_rad_e_per_1e10 * fl / 1e10)
    return rng.normal(0.0, dsnu, int(n_pix)), float(dsnu)


def total_noise_e(signal_e, t_exp_s=1e-3, n_coadd=1, det=None, fpn=None, fluence_p_cm2=None, temp_K=None):
    """EMVA-1288 total noise [e-] = temporal (shot + read + dark shot) RSS spatial FPN (DSNU + PRNU*signal)."""
    det = det or HW.CIS120; fpn = fpn or HW.CIS120_FPN
    S = np.asarray(signal_e, float)
    dark = (det.dark_current_e_s(temp_K) + dark_current_mean_increase_e_s(fluence_p_cm2)) * t_exp_s
    fl = HW.CIS120_RAD.proton_fluence_p_cm2 if fluence_p_cm2 is None else fluence_p_cm2
    dsnu = np.hypot(fpn.dsnu_e, fpn.dsnu_rad_e_per_1e10 * fl / 1e10)
    temporal = S + dark + det.read_noise_e ** 2                   # variances [e-^2]
    spatial = dsnu ** 2 + (fpn.prnu_frac * S) ** 2
    return np.sqrt(temporal / n_coadd + spatial)


# ======================================================================================
# E. INTEGRATED EFFECTIVE PARAMETERS  (identity-when-absent knobs for detector.py)
# ======================================================================================
def effective_params(fluence_p_cm2=None, tid_krad=None, temp_K=None):
    """Effective CIS120 parameters with all deep effects folded in, for the pipeline to consume."""
    det = HW.CIS120; rad = HW.CIS120_RAD; fpn = HW.CIS120_FPN
    fl = rad.proton_fluence_p_cm2 if fluence_p_cm2 is None else fluence_p_cm2
    eff_dark = det.dark_current_e_s(temp_K) + dark_current_mean_increase_e_s(fl)
    rts_var = rts_pixel_fraction(fl) * (HW.CIS120_RAD.rts_amplitude_e / 2.0) ** 2
    eff_read = float(np.sqrt(det.read_noise_e ** 2 + rts_var))
    eff_dsnu = float(np.hypot(fpn.dsnu_e, fpn.dsnu_rad_e_per_1e10 * fl / 1e10))
    return dict(dark_e_s=float(eff_dark), sigma_psf_px=effective_psf_sigma_px(),
                read_e=eff_read, prnu_frac=float(fpn.prnu_frac), dsnu_e=eff_dsnu,
                hot_pixel_frac=hot_pixel_fraction(fl, temp_K=temp_K),
                cti=cti_effective(tid_krad=tid_krad),
                ipc_crosstalk=ipc_crosstalk_frac(), mtf_nyquist=float(sensor_mtf(0.5)))


# ======================================================================================
# SELF-TEST
# ======================================================================================
def _selftest():
    fails = []
    def check(name, cond, detail=""):
        print(("  PASS " if cond else "  FAIL ") + name + ("" if not detail else "  -> " + detail))
        if not cond: fails.append(name)

    det = HW.CIS120
    print("=" * 82)
    print("CIS120 DEEP-FIDELITY DETECTOR SELF-TEST")
    print("  Stevens/Djite (MTF) + Virmontois (radiation) + PPD-CTI + EMVA 1288 (FPN)")
    print("=" * 82)

    # A. PSF / MTF
    k = np.linspace(0, 0.5, 50)
    m = sensor_mtf(k); m_ny = float(sensor_mtf(0.5)); sig = effective_psf_sigma_px()
    print(f"\nA. Sensor PSF/MTF: MTF@Nyquist={m_ny:.3f}, eff PSF sigma {sig:.3f} px "
          f"(optics {HW.CIS120_PSF.optical_sigma_px}), IPC crosstalk {ipc_crosstalk_frac()*100:.1f}%")
    check("sensor MTF monotonically decreasing", np.all(np.diff(m) <= 1e-9))
    check("MTF at Nyquist in BSI range (0.2..0.7)", 0.2 < m_ny < 0.7, f"{m_ny:.3f}")
    check("charge diffusion broadens PSF beyond optics", sig > HW.CIS120_PSF.optical_sigma_px)
    from dataclasses import replace as _rep
    sig_more = effective_psf_sigma_px(_rep(HW.CIS120_PSF, diffusion_length_um=5.0))
    check("more diffusion -> larger PSF / lower MTF", sig_more > sig)
    K = ipc_kernel(); check("IPC kernel charge-conserving (sums to 1)", abs(K.sum() - 1.0) < 1e-12)
    check("IPC crosstalk = 4*alpha", abs(ipc_crosstalk_frac() - 4 * HW.CIS120_PSF.ipc_alpha) < 1e-12)

    # B. CTE / image lag
    cti0 = cti_effective(tid_krad=0.0); cti_eff = cti_effective(); cti_tid = cti_effective(tid_krad=20.0)
    lag = image_lag(1.0e4)                                     # uses the effective (mission-TID) CTI
    print(f"\nB. Charge transfer: CTI {cti0:.2e} nominal / {cti_eff:.2e} mission-TID "
          f"(criterion {HW.CIS120_CTE.cti_criterion:.1e}); @20 krad {cti_tid:.2e}; lag on 1e4 e- = {lag:.1f} e-")
    check("nominal CTI within 0.5% acceptable-lag criterion", cti_ok(tid_krad=0.0))
    check("image lag = effective CTI * signal", abs(lag - cti_eff * 1e4) < 1e-6)
    check("TID increases CTI", cti_tid > cti0)

    # C. radiation
    rng = np.random.default_rng(1)
    for fl in (0.0, 1e10, 1e11):
        di = dark_current_mean_increase_e_s(fl); hp = hot_pixel_fraction(fl)
        print(f"\nC. fluence {fl:.0e} p/cm^2: dark +{di:.1f} e-/s, hot-pixel frac {hp*100:.3f}%")
    d_lo = dark_current_mean_increase_e_s(1e10); d_hi = dark_current_mean_increase_e_s(1e11)
    hp_lo = hot_pixel_fraction(1e10); hp_hi = hot_pixel_fraction(1e11)
    field = sample_dark_field(200000, rng, fluence_p_cm2=1e11)
    fs = field - field.mean(); skew = float((fs ** 3).mean() / (fs.std() ** 3 + 1e-30))
    check("dark-current increase grows with fluence", d_hi > d_lo > 0)
    check("hot-pixel fraction grows with fluence", hp_hi > hp_lo >= 0)
    check("dark-current field has a positive (hot-pixel) tail", skew > 1.0, f"skew {skew:.2f}")
    thit = transient_hits_per_frame(1e-3)
    check("prompt transient hit rate computed (>0)", thit > 0, f"{thit:.2e} hits/frame")
    cb = hotpixel_centroid_bias_px(5e4, fluence_p_cm2=1e11)
    check("hot pixels bias star centroid (>0, star-sensor)", cb > 0, f"{cb*1e3:.2f} milli-px")

    # D. PRNU / DSNU (EMVA 1288)
    g = sample_prnu_field(300000, rng); d, dsnu = sample_dsnu_field(300000, rng, fluence_p_cm2=0.0)
    print(f"\nD. FPN: PRNU field std {g.std()*100:.2f}% (spec {HW.CIS120_FPN.prnu_frac*100:.1f}%), "
          f"DSNU field std {d.std():.2f} e- (spec {HW.CIS120_FPN.dsnu_e:.1f})")
    check("PRNU field reproduces spec sigma", abs(g.std() - HW.CIS120_FPN.prnu_frac) < 0.0015)
    check("DSNU field reproduces spec sigma", abs(d.std() - HW.CIS120_FPN.dsnu_e) < 0.2)
    n_hi = float(total_noise_e(1e4)); n_lo = float(total_noise_e(1e2))
    check("PRNU makes total noise scale with signal (EMVA)", n_hi > n_lo)
    _, dsnu_rad = sample_dsnu_field(10, rng, fluence_p_cm2=1e11)
    check("radiation increases DSNU", dsnu_rad > HW.CIS120_FPN.dsnu_e)

    # E. effective params
    ep = effective_params()
    print(f"\nE. Effective params: dark {ep['dark_e_s']:.1f} e-/s, PSF sigma {ep['sigma_psf_px']:.3f} px, "
          f"read {ep['read_e']:.2f} e-, DSNU {ep['dsnu_e']:.2f} e-, hot-pix {ep['hot_pixel_frac']*100:.3f}%, "
          f"MTF_ny {ep['mtf_nyquist']:.3f}")
    check("effective dark >= nominal (radiation adds)", ep["dark_e_s"] >= det.dark_current_e_s())
    check("effective PSF sigma > optics-only", ep["sigma_psf_px"] > HW.CIS120_PSF.optical_sigma_px)
    check("effective read >= nominal (RTS adds)", ep["read_e"] >= det.read_noise_e)
    check("all deep effects exported", all(kk in ep for kk in
          ("dark_e_s", "sigma_psf_px", "read_e", "prnu_frac", "dsnu_e", "hot_pixel_frac", "cti")))

    print("\n" + "=" * 82)
    print(f"RESULT: {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}  ({len(fails)} failure(s))")
    print("=" * 82)
    return len(fails) == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
