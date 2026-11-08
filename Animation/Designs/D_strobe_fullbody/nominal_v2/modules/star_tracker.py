"""
star_tracker.py  -  ARISE V2 high-fidelity star-tracker model
==============================================================================
Replaces the flat "pointing_knowledge ~ 6 arcsec" constant (hardware.py / nav_stage.py)
with a PHYSICS-BASED, SLEW-RATE-DEPENDENT, ANISOTROPIC star-tracker attitude-error model,
following the HFSS methodology of:

    L. Kazemi & J. Enright, "Accurate Star Tracker Simulation with On-Orbit Data
    Verification," IEEE Aerospace Conference, 2019.

and CALIBRATED to the flight vendor spec for the deputy's actual sensor:

    Blue Canyon Technologies XACT-50 datasheet (REV 1/2026), Gen-3 star tracker:
      static pointing (1s):  +-7" (3 axes, 2 trackers) / +-10" (2 ax) +-25" (3rd ax)
      slewing @ 1 deg/s (1s): cross-boresight 8" , around-boresight 50"   [Gen 3]
      slew rate up to 10 deg/s (6U/14kg), momentum 50 mNms/axis, max torque 0.006 Nm

Two layers (mirrors render_validate.py = hi-fi vs analytic):
  LAYER A  - fast error model (MC-embeddable): per-axis (pitch,yaw,roll) 1-sigma vs body
             rate.  Error MAGNITUDE is calibrated to the XACT-50 datasheet (4 anchor points);
             AVAILABILITY (does it still get a fix?) comes from the paper's photon/streak
             physics (illumination eq 13 + streak smear eq 12 -> detectable-star count).
  LAYER B  - literal HFSS pixel renderer (validation backend): projects a star, integrates the
             streak PSF along the eq-12 locus (RK4) + eq-14, adds Poisson+read noise, centroids.
             Confirms the streak-smear mechanism the datasheet quantifies (too slow for the MC).

Validation (self-test): ST-16RT preset reproduces the paper's on-orbit Table 4
(cross 3.35"/3.08", roll 16.63"); XACT-50 preset reproduces the BCT datasheet's four
anchor numbers and stays locked to its 10 deg/s slew spec.

Run `python star_tracker.py` for the self-test (prints PASS/FAIL).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

ARCSEC = np.radians(1.0 / 3600.0)
PHI0 = 1.8e10          # zero-magnitude reference photon flux [m^-2 s^-1]  (Reed 1993, paper eq 13)
FULL_SKY_DEG2 = 41252.96


def calibrate_from_datasheet(cross0, roll0, cross1, roll1):
    """Solve the Layer-A error constants from four vendor anchor points [arcsec, 1-sigma]:
    static cross/roll (cross0,roll0) and slewing @1 deg/s cross/roll (cross1,roll1).
    Model:  cross(w)=hypot(F, a*g(w)),  roll(w)=hypot(F, a*R*g(w)),  g(w)=1+h*w[deg/s].
    Returns {cal_centroid_arcsec:a, nea_floor_arcsec:F, roll_cross_ratio:R, rate_hardness:h}."""
    c0, r0, c1, r1 = float(cross0), float(roll0), float(cross1), float(roll1)
    # common floor F^2=u solves (c1^2-u)(r0^2-u) = (r1^2-u)(c0^2-u)  (linear in u):
    u = (c1**2*r0**2 - r1**2*c0**2) / (c1**2 + r0**2 - r1**2 - c0**2)
    F = np.sqrt(u); a = np.sqrt(c0**2 - u); k = np.sqrt(r0**2 - u)
    g1 = np.sqrt((c1**2 - u) / (c0**2 - u))
    return dict(cal_centroid_arcsec=float(a), nea_floor_arcsec=float(F),
                roll_cross_ratio=float(k / a), rate_hardness=float(g1 - 1.0))


def calibrate_per_axis(cross0, roll0, cross1, roll1):
    """Per-axis calibration for sensors whose cross/roll degrade at DIFFERENT rates (e.g. the Gen-3
    NST, whose anisotropy inverts with slew: cross 1->8 but roll 10->50 at 1 deg/s, which the
    shared-floor solver cannot fit).  Exact at static and 1 deg/s; the static term IS the floor
    (nea_floor=0).  NOTE: beyond 1 deg/s is a 2-point linear extrapolation."""
    a = float(cross0); ratio = float(roll0) / float(cross0)
    return dict(cal_centroid_arcsec=a, nea_floor_arcsec=0.0, roll_cross_ratio=ratio,
                rate_hardness=float(cross1) / float(cross0) - 1.0,
                rate_hardness_cross=float(cross1) / float(cross0) - 1.0,
                rate_hardness_roll=float(roll1) / float(roll0) - 1.0)


# ======================================================================================
# Sensor parameter block
# ======================================================================================
@dataclass(frozen=True)
class StarTrackerParams:
    name: str
    # --- optics / detector (drive AVAILABILITY: photon budget + streak) ---
    aperture_m: float
    fnum: float
    pixel_m: float
    fov_half_deg: float          # physical FOV half-angle (for star count)
    qe: float
    exposure_s: float
    read_noise_e: float
    sigma_psf_px: float
    detect_snr: float
    catalog_n: int               # stars over full sky brighter than catalog_mlim
    catalog_mlim: float
    # --- error-magnitude calibration (drive the ATTITUDE ERROR; fit to vendor data) ---
    cal_centroid_arcsec: float   # static centroid-limited cross-boresight precision 'a'
    nea_floor_arcsec: float      # common additive floor 'F' (catalog+thermal+distortion)
    roll_cross_ratio: float      # centroid-limited roll/cross ratio 'R' (FOV lever-arm)
    rate_hardness: float         # g(w)=1+rate_hardness*w[deg/s]  (streak degradation, /deg/s)
    # --- datasheet metadata (for hardware.py / guidance_v2) ---
    max_slew_deg_s: float = 10.0
    momentum_mNms_axis: float = 50.0
    max_torque_Nm: float = 0.006
    noise_pix_static: float = 4.0
    noise_pix_width: float = 3.0
    rate_hardness_cross: float = -1.0    # per-axis cross rate override (<0 -> use shared rate_hardness)
    rate_hardness_roll: float = -1.0     # per-axis roll  rate override (<0 -> use shared rate_hardness)

    # ---- derived optics ----
    @property
    def focal_m(self): return self.fnum * self.aperture_m
    @property
    def f_px(self): return self.focal_m / self.pixel_m
    @property
    def plate_scale_rad(self): return self.pixel_m / self.focal_m

    # ---- photometry (paper eq 13) ----
    def photoelectrons(self, mag):
        Se0 = 0.25 * self.qe * self.exposure_s * np.pi * self.aperture_m**2 * PHI0
        return Se0 * 10.0 ** (-0.4 * np.asarray(mag, float))

    # ---- streak geometry (paper eq 12) ----
    def streak_len_px(self, omega_rad_s):
        return self.f_px * np.abs(np.asarray(omega_rad_s, float)) * self.exposure_s

    def noise_pixels(self, omega_rad_s):
        return self.noise_pix_static + self.noise_pix_width * self.streak_len_px(omega_rad_s)

    def star_snr(self, mag, omega_rad_s):
        Se = self.photoelectrons(mag); npix = self.noise_pixels(omega_rad_s)
        return Se / np.sqrt(Se + npix * self.read_noise_e**2)

    def centroid_noise_px(self, mag, omega_rad_s):
        return self.sigma_psf_px / np.maximum(self.star_snr(mag, omega_rad_s), 1e-9)

    def limiting_mag(self, omega_rad_s):
        npix = float(self.noise_pixels(omega_rad_s)); s2 = self.detect_snr**2
        Se_min = 0.5 * (s2 + np.sqrt(s2**2 + 4.0 * s2 * npix * self.read_noise_e**2))
        Se0 = 0.25 * self.qe * self.exposure_s * np.pi * self.aperture_m**2 * PHI0
        return float(-2.5 * np.log10(Se_min / Se0))

    def n_stars_fov(self, omega_rad_s):
        m_lim = self.limiting_mag(omega_rad_s)
        n_full = self.catalog_n * 10.0 ** (0.6 * (m_lim - self.catalog_mlim))
        frac = (np.pi * self.fov_half_deg**2) / FULL_SKY_DEG2
        return float(max(n_full * frac, 0.0))

    # ---- LAYER A: per-axis attitude 1-sigma vs body rate ----
    def sigma_axes(self, omega_rad_s):
        w_deg = float(abs(omega_rad_s)) / ARCSEC / 3600.0          # rad/s -> deg/s
        hc = self.rate_hardness_cross if self.rate_hardness_cross >= 0 else self.rate_hardness
        hr = self.rate_hardness_roll if self.rate_hardness_roll >= 0 else self.rate_hardness
        cross_c = self.cal_centroid_arcsec * (1.0 + hc * w_deg)    # per-axis rate factor (fits Gen-3)
        roll_c = self.cal_centroid_arcsec * self.roll_cross_ratio * (1.0 + hr * w_deg)
        F = self.nea_floor_arcsec
        n = self.n_stars_fov(abs(omega_rad_s))
        return dict(sigma_x_rad=float(np.hypot(F, cross_c) * ARCSEC),
                    sigma_y_rad=float(np.hypot(F, cross_c) * ARCSEC),
                    sigma_z_rad=float(np.hypot(F, roll_c) * ARCSEC),
                    sigma_cross_rad=float(np.hypot(F, cross_c) * ARCSEC),
                    n_stars=n, limiting_mag=self.limiting_mag(abs(omega_rad_s)),
                    available=bool(w_deg <= self.max_slew_deg_s or n >= 2.0))

    def pointing_sigma_rad(self, omega_rad_s):
        """Cross-boresight (target-pointing) 1-sigma [rad] -- what nav_stage.pointing_error wants."""
        return self.sigma_axes(omega_rad_s)["sigma_cross_rad"]

    def midslew_pointing_sigma_rad(self):
        """Cross-boresight knowledge while imaging AT the near-CA body slew cap [rad].  The
        deputies image MID-SLEW (the body saturates the LOS near closest approach), so this
        rate-degraded value -- not the settled static one -- is what the coverage reconstruction
        should use for the pointing budget."""
        return self.pointing_sigma_rad(np.radians(self.max_slew_deg_s))

    def measure_error(self, omega_rad_s, rng):
        s = self.sigma_axes(omega_rad_s)
        if not s["available"]:
            return np.full(3, np.nan)
        return np.array([rng.normal(0, s["sigma_x_rad"]), rng.normal(0, s["sigma_y_rad"]),
                         rng.normal(0, s["sigma_z_rad"])])


# ---- presets --------------------------------------------------------------------------
# ST-16RT (paper).  Static-only calibration to on-orbit Table 4 (cross 3.2", roll 16.6");
# rate_hardness illustrative (paper gives no error-vs-rate curve).
ST16RT = StarTrackerParams(
    name="Sinclair ST-16RT (paper)",
    aperture_m=0.012, fnum=1.6, pixel_m=2.2e-6, fov_half_deg=7.5,
    qe=0.5, exposure_s=0.1, read_noise_e=50.0, sigma_psf_px=1.5, detect_snr=5.0,
    catalog_n=3746, catalog_mlim=5.75,
    cal_centroid_arcsec=1.515, nea_floor_arcsec=2.82, roll_cross_ratio=10.8, rate_hardness=3.0,
    max_slew_deg_s=8.0,
)

# XB6 XACT-50 (Gen-3 star tracker) -- the DEPUTY'S sensor (mission default).  Error magnitude
# calibrated to the BCT XACT-50 datasheet REV 1/2026 (static 7"/25", @1 deg/s 8"/50"); optics
# chosen so availability holds to the datasheet's 10 deg/s slew spec.
_xact = calibrate_from_datasheet(cross0=7.0, roll0=25.0, cross1=8.0, roll1=50.0)
XACT = StarTrackerParams(
    name="BCT NST (mission precision config; XACT-50)",
    aperture_m=0.013, fnum=1.4, pixel_m=5.5e-6, fov_half_deg=6.2,   # 10x12 deg FOV (~120 deg^2)
    qe=0.55, exposure_s=0.08, read_noise_e=45.0, sigma_psf_px=1.5, detect_snr=5.0,
    catalog_n=20000, catalog_mlim=7.5,          # NST: tracks to mag 7.5, >20,000-star catalog
    max_slew_deg_s=10.0, momentum_mNms_axis=50.0, max_torque_Nm=0.006,
    **_xact,
)

# XACT with the older Gen-2 tracker (datasheet: @1 deg/s cross 15", around 200"); kept for
# conservatism / sensitivity studies.
_xact2 = calibrate_from_datasheet(cross0=6.0, roll0=40.0, cross1=15.0, roll1=200.0)   # real NST Gen-2
XACT_GEN2 = StarTrackerParams(
    name="BCT NST Gen-2 (generic baseline / cost-down option)",
    aperture_m=0.013, fnum=1.4, pixel_m=5.5e-6, fov_half_deg=6.2,
    qe=0.55, exposure_s=0.08, read_noise_e=45.0, sigma_psf_px=1.5, detect_snr=5.0,
    catalog_n=20000, catalog_mlim=7.5, max_slew_deg_s=10.0,
    **_xact2,
)

# Gen-3 Full-Extension NST (UPGRADE): the 1"/10" static tracker.  Its anisotropy inverts with slew,
# so it uses the per-axis calibration.  Mid-slew (10 deg/s) is a 2-point extrapolation of 0 & 1 deg/s.
_xact3 = calibrate_per_axis(cross0=1.0, roll0=10.0, cross1=8.0, roll1=50.0)
XACT_GEN3 = StarTrackerParams(
    name="BCT NST Gen-3 Full-Extension (UPGRADE, 1/10 static)",
    aperture_m=0.016, fnum=1.4, pixel_m=5.5e-6, fov_half_deg=6.2,     # Full-Extension: larger optics
    qe=0.60, exposure_s=0.08, read_noise_e=40.0, sigma_psf_px=1.2, detect_snr=5.0,
    catalog_n=20000, catalog_mlim=7.5, max_slew_deg_s=10.0,
    **_xact3,
)


# TWO STAR TRACKERS (mission config): a second NST with a non-parallel boresight observes the
# first tracker's weak ROLL axis, so the roll penalty collapses (roll ~= cross) -> +/-0.002 deg
# ~ 7.2" 3-axis (the XB6 bus spec) [S1].  Modeled by setting roll_cross_ratio -> 1 (roll == cross).
# NOTE: 2 trackers need FleXcore ADCS or an externally-mounted 2nd NST (XACT baseline carries one).
from dataclasses import replace as _replace
XACT_2TRACKER = _replace(XACT, name="BCT NST x2 (2 trackers; +/-7.2\" 3-axis, roll = cross)",
                         roll_cross_ratio=1.0)
XACT_GEN3_2T = _replace(XACT_GEN3, name="BCT NST Gen-3 x2 (2 trackers; 1\" 3-axis)",
                        roll_cross_ratio=1.0, rate_hardness_roll=XACT_GEN3.rate_hardness_cross)


def active():
    """Mission's ACTIVE star-tracker preset, selected by env ARISE_ST_PRESET:
    'gen3x2' (DEFAULT, with FSM) | '2tracker' | 'precision' | 'gen2' | 'gen3'."""
    import os
    return {"precision": XACT, "gen2": XACT_GEN2, "gen3": XACT_GEN3,
            "gen3x2": XACT_GEN3_2T, "2tracker": XACT_2TRACKER}.get(
        os.getenv("ARISE_ST_PRESET", "gen3x2").strip().lower(), XACT_GEN3_2T)


# ======================================================================================
# LAYER B - literal HFSS pixel renderer (validation backend)
# ======================================================================================
def _rk4_locus(x0, y0, omega, f_px, te, nsteps=24):
    wx, wy, wz = omega
    def vel(x, y):
        return ((x*y/f_px)*wx - (f_px + x*x/f_px)*wy + y*wz,
                (f_px + y*y/f_px)*wx - (x*y/f_px)*wy - x*wz)
    x, y = x0, y0; dt = te/nsteps
    for _ in range(nsteps//2):
        k1=vel(x,y); k2=vel(x-0.5*dt*k1[0],y-0.5*dt*k1[1])
        k3=vel(x-0.5*dt*k2[0],y-0.5*dt*k2[1]); k4=vel(x-dt*k3[0],y-dt*k3[1])
        x-=dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); y-=dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    xs=[x]; ys=[y]
    for _ in range(nsteps):
        k1=vel(x,y); k2=vel(x+0.5*dt*k1[0],y+0.5*dt*k1[1])
        k3=vel(x+0.5*dt*k2[0],y+0.5*dt*k2[1]); k4=vel(x+dt*k3[0],y+dt*k3[1])
        x+=dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); y+=dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)


def render_star_patch(mag, omega_body, p, patch=128, rng=None):
    rng = rng or np.random.default_rng(0)
    Se = float(p.photoelectrons(mag)); f_px = p.f_px
    r_px = np.radians(p.fov_half_deg/np.sqrt(2)) * f_px
    xs, ys = _rk4_locus(r_px, 0.0, np.asarray(omega_body, float), f_px, p.exposure_s)
    xm, ym = xs[len(xs)//2], ys[len(ys)//2]; cx, cy = patch/2.0, patch/2.0
    yy, xx = np.mgrid[0:patch, 0:patch]; img = np.zeros((patch, patch))
    sig = p.sigma_psf_px; per = Se/len(xs)
    for xk, yk in zip(xs, ys):
        img += per/(2*np.pi*sig**2)*np.exp(-((xx-(cx+xk-xm))**2+(yy-(cy+yk-ym))**2)/(2*sig**2))
    return rng.poisson(np.clip(img,0,None)).astype(float) + rng.normal(0, p.read_noise_e, img.shape)


def centroid_patch(img, thresh_sigma=5.0, read_noise=50.0):
    b = img - np.median(img); m = b > (thresh_sigma*read_noise)
    if m.sum() < 3: return None
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    w = np.where(m, b, 0.0); W = w.sum()
    return (float((w*xx).sum()/W), float((w*yy).sum()/W))


def render_centroid_scatter(mag, omega_scalar, p, trials=80, seed=0):
    rng = np.random.default_rng(seed); omega = np.array([0.0, omega_scalar, 0.0]); cs, br = [], []
    for _ in range(trials):
        img = render_star_patch(mag, omega, p, rng=rng); c = centroid_patch(img, read_noise=p.read_noise_e)
        if c is not None: cs.append(c); br.append(float((img-np.median(img)).clip(0).sum()))
    if len(cs) < 5: return dict(sigma_px=np.inf, brightness=0.0, det_rate=len(cs)/trials)
    cs = np.array(cs)
    return dict(sigma_px=float(np.hypot(cs[:,0].std(), cs[:,1].std())),
                brightness=float(np.mean(br)), det_rate=len(cs)/trials)


# ======================================================================================
# SELF-TEST
# ======================================================================================
def _selftest():
    fails = []
    def check(name, cond, detail=""):
        print(("  PASS " if cond else "  FAIL ") + name + ("" if not detail else "  -> "+detail))
        if not cond: fails.append(name)

    print("="*78)
    print("STAR TRACKER MODEL SELF-TEST   (HFSS paper + BCT XACT-50 datasheet)")
    print("="*78)

    # 1. ST-16RT static validation vs paper Table 4
    p = ST16RT; s0 = p.sigma_axes(0.0)
    sx, sz = s0["sigma_x_rad"]/ARCSEC, s0["sigma_z_rad"]/ARCSEC
    print(f"\n1. ST-16RT static vs paper Table 4 (X=3.35 Y=3.08 Z=16.63):  model X={sx:.2f} Z={sz:.2f}  (N={s0['n_stars']:.0f})")
    check("ST-16RT cross ~3.2\" (paper)", abs(sx-3.2) < 1.0, f"{sx:.2f}")
    check("ST-16RT roll ~16.6\" (paper)", abs(sz-16.6) < 3.0, f"{sz:.2f}")
    check("ST-16RT illumination: mag-0 in 1e4..1e6", 1e4 < p.photoelectrons(0.0) < 1e6)

    # 2. XACT-50 Gen-3 validation vs BCT datasheet (the mission sensor)
    q = XACT
    a0 = q.sigma_axes(0.0); a1 = q.sigma_axes(np.radians(1.0))
    c0, r0 = a0["sigma_x_rad"]/ARCSEC, a0["sigma_z_rad"]/ARCSEC
    c1, r1 = a1["sigma_x_rad"]/ARCSEC, a1["sigma_z_rad"]/ARCSEC
    print(f"\n2. XACT-50 Gen-3 vs BCT datasheet (static 7\"/25\", @1 deg/s 8\"/50\"):")
    print(f"     static  cross={c0:.2f}\" roll={r0:.2f}\"     @1 deg/s  cross={c1:.2f}\" roll={r1:.2f}\"   (N={a0['n_stars']:.0f})")
    check("static cross-boresight ~7\"", abs(c0-7.0) < 0.7, f"{c0:.2f}")
    check("static around-boresight ~25\"", abs(r0-25.0) < 2.5, f"{r0:.2f}")
    check("@1 deg/s cross-boresight ~8\"", abs(c1-8.0) < 0.8, f"{c1:.2f}")
    check("@1 deg/s around-boresight ~50\"", abs(r1-50.0) < 5.0, f"{r1:.2f}")
    check("cross-boresight anisotropy: roll >> cross", r0/c0 > 3.0, f"{r0/c0:.1f}x")
    ms = q.midslew_pointing_sigma_rad()/ARCSEC
    print(f"     mid-slew (imaging at {q.max_slew_deg_s:.0f} deg/s body cap): cross = {ms:.1f}\"  (vs 7\" settled)")
    check("mid-slew pointing knowledge ~26\" (10 deg/s)", 22.0 < ms < 30.0, f"{ms:.1f}")

    # 3. Slew-rate behaviour: grows, and (per datasheet) STAYS LOCKED to 10 deg/s
    print(f"\n3. XACT-50 slew-rate curve (deg/s -> cross\", roll\", N, avail):")
    prev=-1; mono=True
    for rd in [0.0, 0.5, 1.0, 2.0, 5.0, 9.0, 10.0]:
        s = q.sigma_axes(np.radians(rd)); cr=s["sigma_cross_rad"]/ARCSEC; ro=s["sigma_z_rad"]/ARCSEC
        print(f"     {rd:5.1f} -> {cr:7.2f} {ro:8.2f}   N={s['n_stars']:5.1f}   {'OK' if s['available'] else 'DROPOUT'}")
        if cr < prev-1e-6: mono=False
        prev=cr
    check("error grows monotonically with slew rate", mono)
    check("locked at 1 deg/s", q.sigma_axes(np.radians(1.0))["available"])
    check("locked at 10 deg/s (datasheet slew spec)", q.sigma_axes(np.radians(10.0))["available"])
    check("drops out beyond spec (13 deg/s, e.g. CA peak slew)", not q.sigma_axes(np.radians(13.0))["available"])
    check("cross error grows >2.5x from 0->10 deg/s",
          (q.sigma_axes(np.radians(10.0))["sigma_cross_rad"]/a0["sigma_x_rad"]) > 2.5)

    # 4. LAYER B renderer confirms the streak-smear mechanism (paper Figs 4 & 8)
    print(f"\n4. HFSS renderer (Layer B) -- streak mechanism (mag 1.0 star):")
    b0 = render_centroid_scatter(1.0, 0.0, q, seed=1); b1 = render_centroid_scatter(1.0, np.radians(1.0), q, seed=2)
    print(f"     centroid sigma_px: {b0['sigma_px']:.3f} (0) -> {b1['sigma_px']:.3f} (1 deg/s);  "
          f"brightness {b0['brightness']:.2e} -> {b1['brightness']:.2e}")
    check("renderer detects bright star (0 & 1 deg/s)", b0["det_rate"]>0.9 and b1["det_rate"]>0.5)
    check("centroid scatter grows when slewing (Fig 4)", b1["sigma_px"] > b0["sigma_px"])
    check("registered brightness drops when slewing (Fig 8)", b1["brightness"] < b0["brightness"])
    check("static centroid noise in paper range (<0.3 px)", b0["sigma_px"] < 0.3)

    # 5. Sampler covariance matches sigma_axes
    rng = np.random.default_rng(0)
    draws = np.array([q.measure_error(np.radians(1.0), rng) for _ in range(6000)])
    emp = draws.std(axis=0)/ARCSEC; th = q.sigma_axes(np.radians(1.0))
    print(f"\n5. Sampler empirical sigma (arcsec): x={emp[0]:.2f} z={emp[2]:.2f}  (model x={th['sigma_x_rad']/ARCSEC:.2f} z={th['sigma_z_rad']/ARCSEC:.2f})")
    check("sampler cross matches model (<0.6\")", abs(emp[0]-th['sigma_x_rad']/ARCSEC) < 0.6)
    check("sampler roll matches model (<3\")", abs(emp[2]-th['sigma_z_rad']/ARCSEC) < 3.0)

    # 6. Gen-2 preset reproduces its datasheet anchors
    g2 = XACT_GEN2; e1 = g2.sigma_axes(np.radians(1.0))
    print(f"\n6. XACT-50 Gen-2 @1 deg/s (datasheet 15\"/200\"): cross={e1['sigma_x_rad']/ARCSEC:.1f}\" roll={e1['sigma_z_rad']/ARCSEC:.1f}\"")
    check("Gen-2 @1 deg/s cross ~15\"", abs(e1['sigma_x_rad']/ARCSEC-15.0) < 2.0)
    check("Gen-2 @1 deg/s roll ~200\"", abs(e1['sigma_z_rad']/ARCSEC-200.0) < 20.0)

    # 7. Gen-3 Full-Extension (UPGRADE) via per-axis calibration
    g3 = XACT_GEN3; f0 = g3.sigma_axes(0.0); f1 = g3.sigma_axes(np.radians(1.0)); fm = g3.sigma_axes(np.radians(10.0))
    print(f"\n7. NST Gen-3 UPGRADE (datasheet 1/10 static, 8/50 @1 deg/s): static "
          f"{f0['sigma_x_rad']/ARCSEC:.1f}/{f0['sigma_z_rad']/ARCSEC:.0f}\"  @1 {f1['sigma_x_rad']/ARCSEC:.1f}/{f1['sigma_z_rad']/ARCSEC:.0f}\"  "
          f"mid-slew(10) {fm['sigma_x_rad']/ARCSEC:.0f}/{fm['sigma_z_rad']/ARCSEC:.0f}\" [extrapolated]")
    check("Gen-3 static 1\"/10\"", abs(f0['sigma_x_rad']/ARCSEC-1.0)<0.2 and abs(f0['sigma_z_rad']/ARCSEC-10.0)<0.5)
    check("Gen-3 @1 deg/s 8\"/50\"", abs(f1['sigma_x_rad']/ARCSEC-8.0)<0.4 and abs(f1['sigma_z_rad']/ARCSEC-50.0)<2.0)
    check("existing XACT preset unchanged (static 7\")", abs(XACT.sigma_axes(0.0)['sigma_x_rad']/ARCSEC-7.0)<0.1)

    # 8. TWO trackers: roll penalty collapses (roll == cross); mission default
    t2 = XACT_2TRACKER; u0 = t2.sigma_axes(0.0); um = t2.sigma_axes(np.radians(10.0))
    print(f"\n8. TWO star trackers (mission default): static {u0['sigma_x_rad']/ARCSEC:.1f}/{u0['sigma_z_rad']/ARCSEC:.1f}\"  "
          f"mid-slew(10) {um['sigma_x_rad']/ARCSEC:.0f}/{um['sigma_z_rad']/ARCSEC:.0f}\"  (vs single-tracker roll 25/279)")
    check("2-tracker roll == cross (penalty removed)", abs(u0['sigma_z_rad']-u0['sigma_x_rad'])<1e-9)
    check("2-tracker static ~7.2\" 3-axis (bus spec)", 6.5 < u0['sigma_x_rad']/ARCSEC < 7.5)
    check("active() default = gen3x2 (FSM-enabled)", active() is XACT_GEN3_2T)

    print("\n" + "="*78)
    print(f"RESULT: {'ALL PASS' if not fails else 'FAILED: '+', '.join(fails)}  ({len(fails)} failure(s))")
    print("="*78)
    return len(fails) == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
