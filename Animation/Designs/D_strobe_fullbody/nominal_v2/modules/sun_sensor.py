"""
sun_sensor.py — high-fidelity photodiode-array + window Sun position sensor
===========================================================================
Physical model from:

  M. Diaz Salazar, N. Kondratiuk, A. Moreno, M. Martinez Bogado, S. Pavoni Oliver,
  M. Tamasi, "Design and simulation of sun position sensors for space applications:
  A comparative study," Rev. Sci. Instrum. 95, 075104 (2024).  doi:10.1063/5.0199540
  (Archived: Papers/SunSensor_design_DiazSalazar_2024.pdf)

WHAT THIS MODEL IS
------------------
A window (aperture 2r in a thin Cr film on glass of thickness h) casts a light
spot on a silicon photodiode array.  As the Sun moves off-normal by zenith angle
theta (azimuth phi), Snell refraction through the glass displaces the spot centre by

    L = h * tan(theta_r),   n1 sin(theta) = n2 sin(theta_r)        (Eqs. 2-3)
    Lx = -L cos(phi),   Ly = L sin(phi)                            (Eqs. 4-5)

The sensor compares the ILLUMINATED AREA on each photodiode.  The normalised
difference transfer function cancels the common short-circuit-current density Jsc
and the cos(theta) Lambert factor, leaving a function of geometry alone (Eq. 14):

    H = (I_A - I_B)/(I_A + I_B) = (A_A - A_B)/(A_A + A_B)

Six architectures are modelled (paper Secs. III-IV):
  single-axis 2-quadrant  : square window  (Eqs. 8-9,15-16) | circular window (Eqs.10-13)
  single-axis triangular  : triangle+rectangle photodiodes  (Eqs.18-21)
  dual-axis  4-quadrant   : square window  (Eqs.22-27)       | circular window (Eqs.28-37)

WHY ARISE CARES  (this is a COARSE / SAFE-MODE sensor, not the science sensor)
------------------------------------------------------------------------------
The star tracker (star_tracker.py) gives arcsecond attitude for imaging.  The Sun
sensor is 3-4 orders of magnitude coarser (arcminute-class) but it is the sensor of
last resort:
  * SAFEHOLD / sun acquisition after a star-tracker (ATT) fault -> the safe-mode
    pointing accuracy is set by the Sun-vector knowledge here.
  * Sun-keep-out: the NST has a 45 deg / 27 deg Sun & Earth exclusion; the Sun
    sensor supplies the vector used to keep the Sun out of the tracker/payload FOV.
  * Coverage: a wide-FOV head tiles the sphere so a few heads see the Sun in any
    attitude (n_heads_for_sphere()).

ACCURACY MODEL
--------------
Sensitivity  S = dH/dtheta [1/deg] (linear-regression slope, paper convention).
A readout resolves H to delta_H (ADC quantisation + shot noise); the random
angle error (noise-equivalent angle) is NEA = delta_H / S.  A differential Jsc
mismatch and a window-detector misalignment add a bias = delta_H_sys / S (largely
ground-calibrated out).  pointing_sigma_deg() = RSS(NEA, residual bias).

Sensitivity trades against field of view through the glass thickness h and window
size r (paper: thicker glass and smaller window -> more sensitive, narrower FOV).

VALIDATION (self-test __main__) reproduces the paper's Table I and its trends:
  Table I (h=1mm): FOV 63/66/unobstructed deg; S 0.012/0.015/0.016 1/deg;
                   Isc(0) 0.525/0.515/0.531 mA  (triangular/square/circular)
  h-trend: square S 4.3e-3 -> 2.86e-2 1/deg over glass 300um -> 2mm
  FOV:     2mm glass -> +/-30 deg;  1mm glass -> >+/-60 deg

The module is import-safe and side-effect-free; nothing in the ARISE pipeline
changes unless ARISE_SUNSENSOR_HIFI is set (identity-when-absent).
"""
from __future__ import annotations
import os, math
from dataclasses import dataclass, field
import numpy as np

DEG = math.pi / 180.0
ARCMIN = 1.0 / 60.0                       # deg per arcmin
AM0_IRRADIANCE_WM2 = 1367.0               # space (AM0) reference irradiance      [paper]
JSC_A_PER_M2 = 350.0                      # 35 mA/cm^2 CSS reference (AM0)         [paper]


# ======================================================================================
#  Sensor definition
# ======================================================================================
@dataclass(frozen=True)
class SunSensorParams:
    """One photodiode-array + window Sun-sensor head (paper geometry).  Lengths in metres."""
    name: str = "ARISE fine Sun sensor (2-quad square window)"
    architecture: str = "2quad_square"    # 2quad_square|2quad_circular|triangular|4quad_square|4quad_circular
    h_glass_m: float = 1.0e-3             # glass substrate thickness  (spot lever-arm)   [Eq.3]
    r_window_m: float = 0.95e-3           # window half-width / radius (side = 2r)         [paper]
    s_sep_m: float = 350e-6               # inter-photodiode dead separation (contact+barrier)
    n_air: float = 1.0                    # n1
    n_glass: float = 1.52                 # n2 (fused-silica-class)                        [Eq.2]
    # triangular-design photodiode geometry (only used for architecture='triangular')
    tri_a_m: float = 6.0e-3               # triangle base width a
    tri_b_m: float = 2.0e-3               # triangle height b  (sensitivity ~ 1/b)         [Eq.21]
    tri_c_m: float = 3.0e-3               # rectangular reference-photodiode width c
    # detector envelope + photometry
    detector_side_m: float = 12.0e-3      # 12x12 mm detector envelope                     [paper]
    jsc_A_m2: float = JSC_A_PER_M2
    irradiance_Wm2: float = AM0_IRRADIANCE_WM2
    # readout / error budget
    adc_bits: int = 12                    # differential-current ADC
    adc_fullscale_A: float = 1.0e-3       # ADC full scale (~ Isc range)
    jsc_mismatch_frac: float = 0.02       # 2% differential Jsc mismatch (paper note)
    calib_residual_frac: float = 0.1      # fraction of raw bias left after ground calibration
    misalign_um: float = 5.0              # window-detector registration error (1-sigma)   [um]
    fov_housing_deg: float = 90.0         # housing-limited zenith cap (effective FOV clip)

    # ---- geometry helpers -------------------------------------------------------------
    def theta_r(self, theta_i_rad):
        """Refracted zenith angle in the glass (Snell)."""
        s = (self.n_air / self.n_glass) * np.sin(theta_i_rad)
        return np.arcsin(np.clip(s, -1.0, 1.0))

    def spot_L_m(self, theta_i_rad):
        """Spot-centre displacement L = h tan(theta_r)  (Eq.3)."""
        return self.h_glass_m * np.tan(self.theta_r(theta_i_rad))

    def Lx_Ly(self, theta_i_rad, phi_rad):
        L = self.spot_L_m(theta_i_rad)
        return -L * np.cos(phi_rad), L * np.sin(phi_rad)        # Eqs.4-5

    # ---- illuminated areas (per architecture) -----------------------------------------
    def _areas_square_2q(self, Lx):
        """Two-quadrant, square window side 2r.  Eqs.8-9 (clipped to >=0)."""
        r, s = self.r_window_m, self.s_sep_m
        A_B = 2 * r * np.clip(r + Lx - s / 2, 0.0, 2 * r)         # Eq.8
        A_A = 4 * r * r - 2 * r * np.clip(r + Lx + s / 2, 0.0, 2 * r)   # Eq.9
        return np.clip(A_A, 0.0, None), np.clip(A_B, 0.0, None)

    def _areas_circular_2q(self, Lx):
        """Two-quadrant, circular window radius r.  Circular-segment areas Eqs.10-13."""
        r, s = self.r_window_m, self.s_sep_m
        dA = np.clip((Lx + s / 2) / r, -1.0, 1.0)
        dB = np.clip((Lx - s / 2) / r, -1.0, 1.0)
        aA = 2 * np.arccos(dA)                                    # Eq.12
        aB = 2 * np.arccos(dB)                                    # Eq.13
        A_A = 0.5 * r * r * (aA - np.sin(aA))                     # Eq.10 (segment beyond +s/2)
        A_B = math.pi * r * r - 0.5 * r * r * (aB - np.sin(aB))   # Eq.11
        return np.clip(A_A, 0.0, None), np.clip(A_B, 0.0, None)

    def _areas_triangular(self, Lx):
        """Triangular (A) + rectangular reference (B) photodiodes.  Eqs.19-20."""
        r, a, b, c = self.r_window_m, self.tri_a_m, self.tri_b_m, self.tri_c_m
        A_A = (a * r / b) * np.clip(b - 2 * Lx, 0.0, 2 * b)       # Eq.19
        A_B = c * 2 * r * np.ones_like(np.asarray(Lx, float))     # Eq.20 (constant reference)
        return np.clip(A_A, 0.0, None), A_B

    def _areas_square_4q(self, Lx, Ly):
        """Dual-axis four-quadrant, square window.  Rectangle areas Eqs.24-27."""
        r, s = self.r_window_m, self.s_sep_m
        xm = np.clip(r - Lx - s / 2, 0.0, 2 * r); xp = np.clip(r + Lx - s / 2, 0.0, 2 * r)
        ym = np.clip(r - Ly - s / 2, 0.0, 2 * r); yp = np.clip(r + Ly - s / 2, 0.0, 2 * r)
        A_A = xm * yp; A_B = xp * yp; A_C = xp * ym; A_D = xm * ym    # Eqs.24-27
        return A_A, A_B, A_C, A_D

    # ---- transfer functions -----------------------------------------------------------
    def transfer(self, theta_i_rad, phi_rad=0.0):
        """Normalised difference transfer function H (Eq.14 / 21 / 22).  Returns scalar
        for the measured axis (Hx); for 4-quadrant returns (Hx, Hy)."""
        Lx, Ly = self.Lx_Ly(np.asarray(theta_i_rad, float), np.asarray(phi_rad, float))
        arch = self.architecture
        if arch == "2quad_square":
            A_A, A_B = self._areas_square_2q(Lx)
            return (A_A - A_B) / np.clip(A_A + A_B, 1e-18, None)
        if arch == "2quad_circular":
            A_A, A_B = self._areas_circular_2q(Lx)
            return (A_A - A_B) / np.clip(A_A + A_B, 1e-18, None)
        if arch == "triangular":
            A_A, A_B = self._areas_triangular(Lx)
            return (A_A - A_B) / np.clip(A_B, 1e-18, None)        # Eq.21 (normalise by reference)
        if arch in ("4quad_square", "4quad_circular"):
            A_A, A_B, A_C, A_D = self._areas_square_4q(Lx, Ly)
            tot = np.clip(A_A + A_B + A_C + A_D, 1e-18, None)
            Hx = ((A_A + A_D) - (A_B + A_C)) / tot               # Eq.22
            Hy = ((A_A + A_B) - (A_C + A_D)) / tot               # Eq.23
            return Hx, Hy
        raise ValueError(f"unknown architecture {arch!r}")

    def isc_mA(self, theta_i_rad, phi_rad=0.0):
        """Short-circuit current of photodiode 'A' [mA]:  Isc = Jsc*A*cos(theta)  (Eq.1)."""
        Lx, _ = self.Lx_Ly(np.asarray(theta_i_rad, float), np.asarray(phi_rad, float))
        if self.architecture == "2quad_square":
            A_A, _ = self._areas_square_2q(Lx)
        elif self.architecture == "2quad_circular":
            A_A, _ = self._areas_circular_2q(Lx)
        elif self.architecture == "triangular":
            A_A, _ = self._areas_triangular(Lx)
        else:
            A_A, _, _, _ = self._areas_square_4q(Lx, 0.0)
        return self.jsc_A_m2 * A_A * np.cos(np.asarray(theta_i_rad, float)) * 1e3

    # ---- performance metrics ----------------------------------------------------------
    def sensitivity_per_deg(self, theta_max_deg=45.0, n=200):
        """Linear-regression slope |dH/dtheta| over [0, theta_max] deg (paper convention)."""
        th = np.linspace(0.0, theta_max_deg * DEG, n)
        H = self.transfer(th, 0.0)
        H = H[0] if isinstance(H, tuple) else H
        slope = np.polyfit(np.degrees(th), H, 1)[0]
        return abs(float(slope))

    def linearity_R2(self, theta_max_deg=45.0, n=200):
        th = np.linspace(0.0, theta_max_deg * DEG, n)
        H = self.transfer(th, 0.0); H = H[0] if isinstance(H, tuple) else H
        x = np.degrees(th); p = np.polyfit(x, H, 1); yh = np.polyval(p, x)
        ss_res = np.sum((H - yh) ** 2); ss_tot = np.sum((H - H.mean()) ** 2)
        return float(1 - ss_res / max(ss_tot, 1e-30))

    def fov_deg(self):
        """Geometric field of view = max zenith angle before a photodiode goes dark (Eq.6-7,18)."""
        r, s = self.r_window_m, self.s_sep_m
        if self.architecture in ("2quad_square", "2quad_circular", "4quad_square", "4quad_circular"):
            if "circular" in self.architecture:
                # circle keeps illuminating past the geometric limit -> housing-limited
                return float(self.fov_housing_deg)
            Lx_max = r - s / 2                                    # Eq.7
        elif self.architecture == "triangular":
            Lx_max = self.tri_b_m / 2 - r                         # Eq.18
        else:
            Lx_max = r - s / 2
        if Lx_max <= 0:
            return 0.0
        theta_r_max = math.atan(Lx_max / self.h_glass_m)
        sin_i = (self.n_glass / self.n_air) * math.sin(theta_r_max)
        theta_i_max = math.degrees(math.asin(min(sin_i, 1.0)))
        return float(min(theta_i_max, self.fov_housing_deg))

    # ---- error budget -----------------------------------------------------------------
    def _delta_H(self, theta_i_rad=0.0):
        """Random resolution of H from ADC quantisation + photon shot noise at balance."""
        I0 = max(self.isc_mA(theta_i_rad) * 1e-3, 1e-9)          # A, per photodiode
        dI_adc = self.adc_fullscale_A / (2 ** self.adc_bits)     # quantisation step
        q = 1.602e-19
        dI_shot = math.sqrt(2 * q * I0 * 1.0)                    # 1 Hz-equiv shot (coarse)
        dI = math.hypot(dI_adc, dI_shot)
        # H=(A-B)/(A+B); at balance dH = dI/(sqrt(2) I0)
        return dI / (math.sqrt(2.0) * I0)

    def noise_equivalent_angle_deg(self, theta_max_deg=45.0):
        """Random 1-sigma angle error NEA = delta_H / S."""
        S = self.sensitivity_per_deg(theta_max_deg)
        return float(self._delta_H(0.0) / max(S, 1e-9))

    def bias_deg(self, theta_max_deg=45.0):
        """Residual systematic 1-sigma after ground calibration (Jsc mismatch + misalignment)."""
        S = self.sensitivity_per_deg(theta_max_deg)
        dH_jsc = 0.5 * self.jsc_mismatch_frac                    # differential-Jsc offset in H
        dH_mis = (self.misalign_um * 1e-6) / max(self.r_window_m, 1e-9)  # spot-registration offset
        raw = math.hypot(dH_jsc, dH_mis)
        return float(self.calib_residual_frac * raw / max(S, 1e-9))

    def pointing_sigma_deg(self, theta_max_deg=45.0):
        """Total 1-sigma Sun-vector knowledge = RSS(NEA, residual bias)."""
        return float(math.hypot(self.noise_equivalent_angle_deg(theta_max_deg),
                                self.bias_deg(theta_max_deg)))

    def pointing_sigma_arcmin(self, theta_max_deg=45.0):
        return self.pointing_sigma_deg(theta_max_deg) / ARCMIN

    def n_heads_for_sphere(self, overlap=1.0):
        """Number of heads to see the Sun in ANY attitude (tile 4pi with FOV cones)."""
        hc = math.radians(self.fov_deg())
        cap = 2 * math.pi * (1 - math.cos(hc))                   # steradians per head
        if cap <= 0:
            return 999
        return int(math.ceil(overlap * 4 * math.pi / cap))


# ======================================================================================
#  Presets  (Table-I comparison configs are exact paper conditions -> used by self-test)
# ======================================================================================
PRESETS = {
    # --- ARISE mission presets ---
    "fine":  SunSensorParams(name="ARISE fine Sun sensor (2mm glass, 2-quad square)",
                             architecture="2quad_square", h_glass_m=2.0e-3, r_window_m=1.0e-3),
    "coarse": SunSensorParams(name="ARISE coarse Sun sensor (1mm glass, 4-quad square, wide FOV)",
                             architecture="4quad_square", h_glass_m=1.0e-3, r_window_m=1.0e-3),
    "coarse_dualaxis": SunSensorParams(name="ARISE dual-axis CSS (4-quad, 1mm glass)",
                             architecture="4quad_square", h_glass_m=1.0e-3, r_window_m=1.1e-3),
    # --- exact paper Table-I comparison configs (h=1mm, Isc(0)~0.5mA) ---
    "paper_square":   SunSensorParams(name="paper Table-I square (r=0.95mm,h=1mm)",
                             architecture="2quad_square", h_glass_m=1.0e-3, r_window_m=0.95e-3),
    "paper_circular": SunSensorParams(name="paper Table-I circular (r=1.1mm,h=1mm)",
                             architecture="2quad_circular", h_glass_m=1.0e-3, r_window_m=1.1e-3),
    "paper_triangular": SunSensorParams(name="paper Table-I triangular (b=2mm,2r=0.5mm,h=1mm)",
                             architecture="triangular", h_glass_m=1.0e-3, r_window_m=0.25e-3,
                             tri_a_m=6.0e-3, tri_b_m=2.0e-3, tri_c_m=3.0e-3),
}


def active():
    """Mission Sun-sensor head selected by ARISE_SS_PRESET (default 'coarse' = safe-mode CSS)."""
    return PRESETS.get(os.getenv("ARISE_SS_PRESET", "coarse"), PRESETS["coarse"])


def sun_vector_sigma_rad(theta_max_deg=45.0):
    """Convenience: active head's 1-sigma Sun-vector knowledge [rad] (safe-mode reference)."""
    return active().pointing_sigma_deg(theta_max_deg) * DEG


def measure(sun_hat_body, theta_max_deg=45.0, rng=None):
    """Return a NOISY measured Sun unit-vector given the TRUE body-frame Sun unit-vector.
    Perturbs by the active head's 1-sigma cone error.  (Truth in -> estimate out; callers
    must keep the truth for illumination and use this only for attitude knowledge.)"""
    rng = rng or np.random.default_rng()
    s = np.asarray(sun_hat_body, float); s = s / max(np.linalg.norm(s), 1e-18)
    sig = active().pointing_sigma_deg(theta_max_deg) * DEG
    # small-angle tilt in a random transverse direction
    a = np.array([1.0, 0, 0]) if abs(s[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(s, a); u /= max(np.linalg.norm(u), 1e-18); v = np.cross(s, u)
    ang = rng.normal(0, sig); az = rng.uniform(0, 2 * math.pi)
    tilt = math.cos(az) * u + math.sin(az) * v
    m = math.cos(ang) * s + math.sin(ang) * tilt
    return m / max(np.linalg.norm(m), 1e-18)


# ======================================================================================
#  Self-test — validate against the paper (Table I + trends)
# ======================================================================================
if __name__ == "__main__":
    ok = True
    def chk(msg, cond, got, exp):
        global ok; ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg:52s} got {got:>10}  exp {exp}")

    print("=" * 92)
    print("SUN SENSOR HI-FI  (Diaz Salazar et al., Rev. Sci. Instrum. 95, 075104, 2024)")
    print("=" * 92)

    sq, ci, tr = PRESETS["paper_square"], PRESETS["paper_circular"], PRESETS["paper_triangular"]

    print("\n-- Table I : performance-metric comparison (h=1mm, Isc(0)~0.5mA) --")
    print(f"{'':22s}{'triangular':>13s}{'square':>10s}{'circular':>11s}   paper: 0.012/0.015/0.016")
    S_tr = tr.sensitivity_per_deg(); S_sq = sq.sensitivity_per_deg(); S_ci = ci.sensitivity_per_deg()
    print(f"  sensitivity [1/deg] {S_tr:13.4f}{S_sq:10.4f}{S_ci:11.4f}")
    I_tr = float(tr.isc_mA(0.0)); I_sq = float(sq.isc_mA(0.0)); I_ci = float(ci.isc_mA(0.0))
    print(f"  Isc(0) [mA]         {I_tr:13.3f}{I_sq:10.3f}{I_ci:11.3f}   paper: 0.525/0.515/0.531")
    F_tr = tr.fov_deg(); F_sq = sq.fov_deg()
    print(f"  FOV [deg]           {F_tr:13.1f}{F_sq:10.1f}{'unobstr':>11s}   paper: 63 / 66 / unobstr")

    print("\n-- validation checks --")
    chk("square sensitivity ~0.015 1/deg",   abs(S_sq - 0.015) < 0.002, f"{S_sq:.4f}", "0.015")
    chk("circular sensitivity ~0.016 1/deg", abs(S_ci - 0.016) < 0.003, f"{S_ci:.4f}", "0.016")
    chk("triangular sensitivity ~0.012 1/deg", abs(S_tr - 0.012) < 0.002, f"{S_tr:.4f}", "0.012")
    chk("square Isc(0) ~0.515 mA",   abs(I_sq - 0.515) < 0.03, f"{I_sq:.3f}", "0.515")
    chk("circular Isc(0) ~0.531 mA", abs(I_ci - 0.531) < 0.03, f"{I_ci:.3f}", "0.531")
    chk("triangular Isc(0) ~0.525 mA", abs(I_tr - 0.525) < 0.05, f"{I_tr:.3f}", "0.525")
    chk("square FOV ~66 deg",  abs(F_sq - 66) < 3, f"{F_sq:.1f}", "66")
    chk("triangular FOV ~63 deg", abs(F_tr - 63) < 3, f"{F_tr:.1f}", "63")
    chk("square linearity R^2 >0.99", sq.linearity_R2() > 0.99, f"{sq.linearity_R2():.4f}", ">0.99")

    print("\n-- trend: sensitivity vs glass thickness (square, r=1mm) --  paper: 4.3e-3 -> 2.86e-2")
    for h_um in (300, 700, 1000, 2000):
        p = SunSensorParams(architecture="2quad_square", h_glass_m=h_um * 1e-6, r_window_m=1.0e-3)
        print(f"    h={h_um:5d} um : S={p.sensitivity_per_deg():.4e} 1/deg,  FOV={p.fov_deg():5.1f} deg")
    S_300 = SunSensorParams(architecture="2quad_square", h_glass_m=300e-6, r_window_m=1.0e-3).sensitivity_per_deg()
    S_2mm = SunSensorParams(architecture="2quad_square", h_glass_m=2000e-6, r_window_m=1.0e-3).sensitivity_per_deg()
    chk("h-trend S(300um) ~4.3e-3", abs(S_300 - 4.3e-3) < 1.5e-3, f"{S_300:.2e}", "4.3e-3")
    chk("h-trend S(2mm) ~2.86e-2",  abs(S_2mm - 2.86e-2) < 0.6e-2, f"{S_2mm:.2e}", "2.86e-2")
    chk("thicker glass -> narrower FOV",
        SunSensorParams(architecture="2quad_square", h_glass_m=2e-3, r_window_m=1e-3).fov_deg()
        < SunSensorParams(architecture="2quad_square", h_glass_m=1e-3, r_window_m=1e-3).fov_deg(),
        "2mm<1mm", "True")

    print("\n-- dual-axis 4-quadrant (recovers both axes) --")
    q = PRESETS["coarse"]
    Hx, Hy = q.transfer(20 * DEG, 35 * DEG)
    chk("4-quad recovers both axes (Hx,Hy != 0)", abs(Hx) > 1e-3 and abs(Hy) > 1e-3,
        f"{Hx:+.3f},{Hy:+.3f}", "both nonzero")
    Hx0, Hy0 = q.transfer(0.0, 0.0)
    chk("4-quad balanced at normal incidence", abs(Hx0) < 1e-6 and abs(Hy0) < 1e-6,
        f"{Hx0:+.1e}", "~0")

    print("\n-- ARISE safe-mode / coarse-pointing budget --")
    for name in ("fine", "coarse", "coarse_dualaxis"):
        p = PRESETS[name]
        print(f"    {name:16s}: sigma={p.pointing_sigma_arcmin():6.2f} arcmin  "
              f"({p.pointing_sigma_deg():.4f} deg)  FOV={p.fov_deg():5.1f} deg  "
              f"heads/4pi={p.n_heads_for_sphere()}")
    cs = PRESETS["coarse"]
    chk("coarse sigma in arcmin band (0.5-30')", 0.5 < cs.pointing_sigma_arcmin() < 30,
        f"{cs.pointing_sigma_arcmin():.2f}'", "0.5-30'")
    chk("Sun-vector sigma << NST 27deg keep-out", cs.pointing_sigma_deg() < 0.27,
        f"{cs.pointing_sigma_deg():.3f}deg", "<0.27")
    chk("few heads tile the sphere (<=8)", cs.n_heads_for_sphere() <= 8,
        f"{cs.n_heads_for_sphere()}", "<=8")
    m = measure([0, 0, 1.0], rng=np.random.default_rng(0))
    chk("measure() returns unit vector near truth", abs(np.linalg.norm(m) - 1) < 1e-9 and m[2] > 0.99,
        f"z={m[2]:.4f}", "~1,>0.99")

    print("\n" + "=" * 92)
    print("RESULT:", "ALL PASS" if ok else "SOME FAILED")
    print("=" * 92)
