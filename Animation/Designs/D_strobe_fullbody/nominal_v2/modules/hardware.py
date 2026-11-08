"""
hardware.py  -  ARISE V2 single source of truth for FLIGHT HARDWARE parameters
===============================================================================
Every V2 module imports its hardware numbers from here so the whole Monte Carlo is internally
consistent and traceable to real parts:

    Deputy bus     : Blue Canyon Technologies XB6 (6U-class), XACT ADCS
    Chief adapter  : ESPA-Grande, Ø24-inch port
    Imager         : Teledyne e2v CIS120 back-illuminated CMOS (mosaic focal plane)
    Propulsion     : Aerojet Rocketdyne MPS-130 (AF-M315E green monopropellant)

Values are cited inline.  Items behind a request-only datasheet or vendor-"TBD" are tagged
ASSUMED with a defensible value + justification so they are easy to replace later.

OPTICS DIRECTION (locked for V2):
  * Hold 0.15 m GSD @750 m  -> IFOV fixed -> with 10 um pixels, FOCAL length is FIXED (50 mm).
  * Hold 27 deg half-angle FOV -> focal-plane extent fixed -> realized by a CIS120 MOSAIC.
  * APERTURE is a VARIABLE STOP:
      - SUNLIT : stop down so SNR lands in [300,400] (and below full-well).
      - DARK   : open to the fastest allowed aperture (f/1.8) so the strobe energy / XB6
                 battery draw needed to reach a minimum detectable SNR is MINIMIZED.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ======================================================================================
# 1. DEPUTY BUS — Blue Canyon XB6 (6U), XACT ADCS
#    Refs: bluecanyontech.com/spacecraft/6u-cubesat ; XACT heritage specs.
# ======================================================================================
@dataclass(frozen=True)
class XB6Bus:
    name: str = "Blue Canyon XB6 (6U)"
    form_factor_U: float = 6.0
    # ADCS performance (1-sigma) — BCT XB6 / XACT (values reconciled to XB6_Technical_Profile):
    pointing_accuracy_arcsec: float = 7.2    # bus control accuracy ~7.2" (NASA RSDO)  [S5]
    pointing_knowledge_arcsec: float = 8.0   # bus knowledge ~8" (NASA RSDO)  [S5]  (star_tracker.py supersedes)
    pointing_jitter_arcsec: float = 1.0      # jitter 1 arcsec over 1 s (BCT REV 1/2026)  [S1]  (reaction_wheels.py supersedes)
    slew_rate_deg_s: float = 6.0             # XB6 bus agility > 6 deg/s (BCT); XACT-50 module >= 10 deg/s  [S1]
    # Mass / volume envelope
    dry_mass_kg: float = 10.0                # bus dry mass 10 kg (NASA RSDO)  [S5]  (was 10.5 ASSUMED)
    wet_mass_kg: float = 12.2                # ASSUMED: dry + MPS-130 wet (1.7 kg); BCT wet mass not published
    payload_mass_max_kg: float = 6.0         # max payload mass 6 kg (NASA RSDO)  [S5]
    payload_volume_U: float = 4.0            # 4U typical payload volume (BCT/RSDO)  [S1][S5]  (was 1.5 — WRONG)
    solar_array_W: float = 100.0             # total generation 92-108 W (BCT 2026)  [S1]
    payload_power_W: float = 39.0            # orbit-avg EOL payload power (NASA RSDO)  [S5]
    # Electrical — 12 V main bus; bounds the dark-side strobe energy per flash / per encounter.
    #   CURRENT baseline 6.8 Ah; UPGRADE up to 20.4 Ah (BCT REV 1/2026).  Strobe budget scales:
    #   82 Wh -> 53 kJ optical (baseline) ; 245 Wh -> 159 kJ (upgrade)  vs the old 32 kJ.
    battery_capacity_Wh: float = 244.8       # UPGRADE APPLIED (default): 20.4 Ah x 12 V (BCT max option)  [S1]
    battery_baseline_Wh: float = 81.6        # un-upgraded baseline: 6.8 Ah x 12 V (6U std)  [S1][S5]
    battery_usable_frac: float = 0.6         # ASSUMED: usable depth-of-discharge for the burst
    strobe_psu_eff: float = 0.30             # ASSUMED: electrical->optical efficiency of the strobe

    def inertia_diag(self) -> np.ndarray:
        """Principal moments [kg m^2], ASSUMED uniform 6U box 36x24x11 cm at wet mass."""
        m = self.wet_mass_kg
        a, b, c = 0.36, 0.24, 0.11   # m
        Ixx = m / 12.0 * (b**2 + c**2)
        Iyy = m / 12.0 * (a**2 + c**2)
        Izz = m / 12.0 * (a**2 + b**2)
        return np.array([Ixx, Iyy, Izz])

    def strobe_optical_energy_budget_J(self) -> float:
        """Total OPTICAL strobe energy a deputy can deliver from its battery during the
        encounter burst = usable battery energy x electrical->optical efficiency."""
        wh = self.battery_capacity_Wh * self.battery_usable_frac
        return wh * 3600.0 * self.strobe_psu_eff   # Wh -> J optical


# ======================================================================================
# 2. CHIEF ADAPTER — ESPA-Grande Ø24-inch port
#    Refs: Moog ESPA users guide ; ESPA-Grande qualification (USU SmallSat).
# ======================================================================================
@dataclass(frozen=True)
class ESPAChief:
    name: str = "ESPA-Grande (Ø24-in port)"
    port_mass_limit_kg: float = 700.0        # Ø24" port qualified <= 700 kg  [cited]
    ring_height_in: float = 42.0             # [cited]
    cg_for_max_cap_cm: float = 50.8          # CG <= 20 in for max capacity   [cited]
    chief_mass_kg: float = 300.0             # ASSUMED: ARISE chief ~300 kg (well inside 700)
    deploy_dv_sigma_mps: float = 0.05        # ASSUMED: deputy separation tip-off ~5 cm/s 1-sigma


# ======================================================================================
# 3. IMAGER — Teledyne e2v CIS120 back-illuminated CMOS
#    Refs: Capella/CIS120 SPIE 11852-0S (2021); Teledyne e2v CIS120 release (2019).
# ======================================================================================
@dataclass(frozen=True)
class CIS120Detector:
    name: str = "Teledyne e2v CIS120 BSI CMOS"
    nx: int = 2048
    ny: int = 2048
    pixel_m: float = 10.0e-6                 # 10 um pixel pitch  [cited]
    full_well_e: float = 80.0e3              # 80 ke- full-well option  [cited] (45 ke- alt)
    peak_qe: float = 0.90                    # ~90% peak QE @600 nm (BSI)  [cited]
    qe_at_band: float = 0.80                 # ASSUMED: ~0.80 averaged over the imaging band
    read_noise_e: float = 7.0                # ASSUMED: ~7 e- median (Capella CIS120 class)
    dark_current_e_s_at_293K: float = 125.0  # ASSUMED: ~125 e-/px/s @20C (BSI CMOS typical)
    dark_doubling_K: float = 6.5             # dark current doubles ~ every 6.5 K
    operating_temp_K: float = 273.0          # ASSUMED: detector cooled to ~0 C
    prnu_frac: float = 0.01                  # ASSUMED: 1% photo-response non-uniformity (post-flat)
    dsnu_e: float = 2.0                      # ASSUMED: 2 e- dark-signal non-uniformity
    adc_bits: int = 12                       # 12-bit global-shutter mode  [cited]
    global_shutter_fps: float = 20.0         # 20 fps @12-bit global shutter  [cited]
    optical_eff: float = 0.85                # ASSUMED: optics throughput (coatings, obscuration)

    def dark_current_e_s(self, temp_K: float | None = None) -> float:
        T = self.operating_temp_K if temp_K is None else temp_K
        return self.dark_current_e_s_at_293K * 2.0 ** ((T - 293.0) / self.dark_doubling_K)


# ======================================================================================
# 4. PROPULSION — Aerojet Rocketdyne MPS-130 (AF-M315E green monopropellant)
#    Refs: Aerojet/NASA MPS-130 (2016); AF-M315E GPIM (NTRS 20140012587).
# ======================================================================================
@dataclass(frozen=True)
class MPS130Thruster:
    name: str = "Aerojet Rocketdyne MPS-130 (AF-M315E)"
    total_impulse_Ns_1U: float = 1130.0      # >1130 N-s (1U)  [cited]
    wet_mass_kg: float = 1.7                 # 1U wet mass  [cited]
    dry_mass_kg: float = 1.1                 # 1U dry mass  [cited]
    isp_s: float = 245.0                     # ASSUMED: AF-M315E vac Isp ~245 s (5% > hydrazine)
    thrust_N: float = 0.5                    # ASSUMED: per-thruster ~0.5 N BOL (vendor TBD)
    min_impulse_bit_Ns: float = 0.010        # ASSUMED: MIB ~10 mN-s
    misalignment_deg_sigma: float = 0.5      # ASSUMED: thrust-vector misalignment 0.5 deg 1-sigma
    mib_quantization_Ns: float = 0.010       # impulses quantized to the MIB
    finite_burn_loss_frac: float = 0.02      # ASSUMED: ~2% cosine/finite-burn loss on TCMs

    def dv_capacity_mps(self, wet_mass_kg: float) -> float:
        m0 = wet_mass_kg
        mp = self.wet_mass_kg - self.dry_mass_kg     # propellant mass
        m1 = m0 - mp
        g0 = 9.80665
        return float(self.isp_s * g0 * np.log(m0 / m1))


# ======================================================================================
# 5. OPTICS — FIXED focal + FOV, VARIABLE aperture (per your V2 direction)
# ======================================================================================
HALF_ANGLE_FOV_DEG = 27.0     # required half-angle (diagonal) FOV
GSD_GOAL_M = 0.15             # GSD goal at the 750 m design standoff
DESIGN_STANDOFF_M = 750.0
FNUM_MIN = 1.8                # f/# >= 1.8 (hard floor) -> sets the FASTEST (largest) aperture
GSD_LIMIT_M = 1.0             # resolution *requirement* -> max imaging range
SNR_LIT_BAND = (300.0, 400.0) # sunlit target SNR band

IFOV_RAD = GSD_GOAL_M / DESIGN_STANDOFF_M            # = 2.0e-4 rad
FOCAL_M = CIS120Detector().pixel_m / IFOV_RAD        # = 0.050 m (50 mm), FIXED
APERTURE_MAX_M = FOCAL_M / FNUM_MIN                  # fastest allowed = f/1.8 -> 27.8 mm (dark side)


def fov_mosaic(det: "CIS120Detector | None" = None,
               half_fov_deg: float = HALF_ANGLE_FOV_DEG, focal_m: float = FOCAL_M):
    """Focal-plane sizing to cover the required (diagonal) FOV at the fixed focal length.
    Returns the needed pixel count and the CIS120 NxN mosaic that covers it."""
    det = det or CIS120Detector()
    half_diag_extent_m = focal_m * np.tan(np.radians(half_fov_deg))
    diag_px_needed = 2.0 * half_diag_extent_m / det.pixel_m
    side_px_needed = diag_px_needed / np.sqrt(2.0)
    n_per_side = int(np.ceil(side_px_needed / det.nx))   # CIS120s per mosaic side
    mosaic_px = n_per_side * det.nx
    realized_half_diag_m = 0.5 * np.sqrt(2.0) * mosaic_px * det.pixel_m
    realized_half_fov = np.degrees(np.arctan2(realized_half_diag_m, focal_m))
    return dict(side_px_needed=float(side_px_needed), n_per_side=n_per_side,
                n_sensors=n_per_side**2, mosaic_px=mosaic_px,
                realized_half_fov_deg=float(realized_half_fov))


def aperture_for_fnum(fnum: float, focal_m: float = FOCAL_M) -> float:
    return focal_m / fnum


def optics_prescription(aperture_m: float, det: "CIS120Detector | None" = None):
    """Optical prescription at a chosen aperture (the variable stop), focal + FOV fixed.
    SNR-driven aperture sizing for the sunlit/dark regimes lives in detector.py / strobe.py."""
    det = det or CIS120Detector()
    fnum = FOCAL_M / aperture_m
    moz = fov_mosaic(det)
    aperture_area = np.pi * (aperture_m / 2.0) ** 2
    etendue_per_pix = aperture_area * (det.pixel_m / FOCAL_M) ** 2   # A * Omega_pix [m^2 sr]
    return dict(focal_m=FOCAL_M, aperture_m=float(aperture_m), fnum=float(fnum),
                fnum_ok=bool(fnum >= FNUM_MIN - 1e-9), ifov_rad=IFOV_RAD,
                gsd_750_m=float(IFOV_RAD * DESIGN_STANDOFF_M),
                max_range_m=float(GSD_LIMIT_M / IFOV_RAD),
                aperture_area_m2=float(aperture_area), etendue_per_pix_m2sr=float(etendue_per_pix),
                half_fov_deg=moz["realized_half_fov_deg"], mosaic=moz)


# ======================================================================================
# 6. STAR TRACKER — XB6 XACT-50 integrated tracker (Blue Canyon XACT-50 datasheet, REV 1/2026)
#    Single source of truth for the star-tracker attitude-determination error ANCHORS.
#    The rate-dependent, anisotropic error MODEL is in star_tracker.py (HFSS methodology,
#    Kazemi & Enright 2019), calibrated to these datasheet numbers.
# ======================================================================================
@dataclass(frozen=True)
class StarTracker:
    """BCT Nano Star Tracker (NST) family, one per XACT [S10][S11].  Tracks to mag 7.5
    (>20,000-star catalog), >99% sky, lost-in-space <4 s, 5 Hz, FOV 10x12 deg.  The mission
    config (below) is the precision case (2-tracker-fused / Gen-3-class); star_tracker.py is
    calibrated to it.  Real single-tracker datasheet variants for current/upgrade studies:
      Gen-2 (current generic baseline) : 6" cross / 40" roll static
      Gen-3 Full-Extension (UPGRADE)   : 1" cross / 10" roll static  (~10x better)
      two trackers      (UPGRADE)      : +/-7.2" 3-axis (the +/-0.002 deg bus spec)"""
    name: str = "BCT NST x2 (2-tracker mission config; +/-7.2\" 3-axis)"
    # mission-config 1-sigma determination error [arcsec]:
    static_cross_arcsec: float = 7.0      # cross-boresight (2-tracker-fused / Gen-3-class)
    static_roll_arcsec: float = 25.0      # around-boresight
    slew1_cross_arcsec: float = 8.0       # @ 1 deg/s (Gen-3)                                                 [S11]
    slew1_roll_arcsec: float = 50.0       # @ 1 deg/s (Gen-3)                                                 [S11]
    # --- real NST single-tracker datasheet variants (current / upgrade) [S10][S11] ---
    gen2_static_cross_arcsec: float = 6.0   ; gen2_static_roll_arcsec: float = 40.0    # current generic baseline
    gen2_slew1_cross_arcsec: float = 15.0   ; gen2_slew1_roll_arcsec: float = 200.0
    gen3_static_cross_arcsec: float = 1.0   ; gen3_static_roll_arcsec: float = 10.0    # UPGRADE: Full-Extension Gen-3
    gen3_slew1_cross_arcsec: float = 8.0    ; gen3_slew1_roll_arcsec: float = 50.0
    two_tracker_3axis_arcsec: float = 7.2   # UPGRADE: 2 trackers -> +/-0.002 deg 3-axis                     [S1]
    # --- optics / operations (NST datasheet) [S10] ---
    fov_deg: tuple = (10.0, 12.0)
    limiting_mag: float = 7.5
    solution_rate_hz: float = 5.0
    n_trackers: int = 2                      # MISSION USES 2 TRACKERS -> roll = cross (+/-0.002 deg 3-axis) [S1]
    mass_kg: float = 0.35                    # per tracker; Standard NST (Mid-Ext 0.45, Full-Ext 0.85 kg) — x2 for the pair
    sun_earth_excl_deg: tuple = (45.0, 27.0) # Standard NST (Full-Ext 17.5 / 12)
    # --- XACT-50 ADCS envelope ---
    max_slew_deg_s: float = 10.0          # attitude solution maintained to 10 deg/s                          [S11]
    momentum_mNms_per_axis: float = 50.0  # XACT-50 (3x RWp050)                                               [S11]
    max_torque_Nm: float = 0.006          # XACT-50 system max torque                                         [S11]


# ======================================================================================
# 7. REACTION WHEELS — XACT-50-class RWA (4-wheel pyramid).  Single source of truth for the
#    wheel envelope + microvibration anchors.  The rate-dependent jitter + motor MODEL is in
#    reaction_wheels.py (Longato 2025 microvibration + Figueiredo 2013 motor), calibrated here.
# ======================================================================================
@dataclass(frozen=True)
class ReactionWheel:
    """BCT RWp-line wheels in the XACT.  CURRENT = 3x RWp050 (XACT-50), optional 4th (modeled as a
    4-wheel pyramid).  UPGRADES: RWp100 (0.100 Nms) for agility, or RWp500/RW1 (0.5/1.0 Nms) for a
    larger bus [S11][S12].  Microvibration+motor MODEL in reaction_wheels.py (calibrated to the XB6
    1"/1s jitter spec)."""
    name: str = "BCT RWp050 (XACT-50), 3 wheels + optional 4th"
    n_wheels: int = 4                        # 3 baseline + optional 4th (modeled as 4-wheel pyramid) [S11]
    I_wheel_kgm2: float = 7.96e-5            # RWp050 rotor inertia (0.050 Nms @ ~6000 rpm)
    max_momentum_Nms: float = 0.050          # RWp050: 0.050 Nms/wheel                              [S12]
    max_torque_Nm: float = 0.007             # RWp050 max torque (XACT-50 system 0.006 Nm)          [S12]
    max_speed_rpm: float = 6000.0            # ~6000 rpm flywheel                                   [ASSUMED]
    mass_kg: float = 0.24                     # RWp050 mass                                          [S12]
    power_W_max: float = 0.81                 # RWp050 power @ max torque                            [S12]
    pointing_jitter_arcsec: float = 1.0      # XB6 jitter spec: 1 arcsec over 1 s                   [S1]
    static_imbalance_kg_m: float = 5.0e-6    # U_s static imbalance (representative)                [paper[1]]
    dynamic_imbalance_kg_m2: float = 1.0e-7  # U_d dynamic imbalance (representative)               [paper[1]]
    whirl_modes_hz: tuple = (88.0, 220.0, 510.0)  # rotor whirl/structural resonances (representative)
    upgrade_RWp100_Nms: float = 0.100         # UPGRADE: more momentum/agility                       [S12]
    upgrade_RW1_Nms: float = 1.0              # UPGRADE: larger-bus wheel (1.0 Nms)                  [S12]


@dataclass(frozen=True)
class FastSteeringMirror:
    """Fine-pointing FSM (4-actuator voice-coil tip/tilt).  Plant per Liu 2026 (Transfer-Matrix
    Method): first structural mode ~660 Hz, mechanical stroke 5-8.4 mrad -> optical ~+/-1 deg.
    The dynamics + control + step-stare MODEL is in fast_steering_mirror.py.  Its role: reject
    the body pointing error / RW jitter, and (step-stare) settle the body during imaging so the
    Gen-3 NST reaches its ~1" static."""
    name: str = "ARISE FSM (4-actuator VCM; Liu 2026 TMM plant)"
    natural_freq_hz: float = 660.0        # first structural mode                          [Liu 2026]
    stroke_optical_deg: float = 0.96      # optical half-throw (2 x 8.4 mrad mechanical)   [Liu 2026]
    ctrl_bandwidth_hz: float = 220.0      # closed-loop bandwidth (design)
    resolution_urad: float = 0.5          # sub-urad optical resolution


# ======================================================================================
# 9. SUN SENSOR — photodiode-array + window Sun position sensor (COARSE / SAFE-MODE ref).
#    Physical MODEL in sun_sensor.py (Diaz Salazar 2024, RSI 95 075104): a window casts a
#    spot on a Si photodiode array; the normalised illuminated-area difference gives the Sun
#    vector.  Sensor of last resort (SAFEHOLD / sun acquisition after an ATT/star-tracker
#    fault) and the source of the vector for the NST Sun-keep-out.  NOT the science sensor.
# ======================================================================================
@dataclass(frozen=True)
class SunSensor:
    """ARISE coarse Sun-sensor suite (dual-axis 4-quadrant, 1 mm glass).  Arcminute-class,
    wide FOV; a few heads tile 4pi so the Sun is seen in any attitude.  Calibrated in
    sun_sensor.py to the paper's Table I.  UPGRADE: a 'fine' 2 mm-glass head (~2.8') for
    tighter sun-relative pointing at the cost of FOV."""
    name: str = "ARISE CSS x6 (4-quadrant, 1 mm glass, ~0.09 deg)"
    architecture: str = "4quad_square"    # dual-axis (recovers both zenith & azimuth)
    h_glass_mm: float = 1.0               # glass thickness (spot lever-arm)
    r_window_mm: float = 1.0              # window half-width (side 2r)
    sep_um: float = 350.0                 # inter-photodiode dead separation
    detector_side_mm: float = 12.0        # 12x12 mm Si detector envelope                  [paper]
    fov_deg: float = 75.0                 # geometric FOV (1 mm glass)                     [model]
    accuracy_deg: float = 0.094           # 1-sigma Sun-vector knowledge (~5.6 arcmin)     [model]
    n_heads: int = 6                      # heads to tile the full sphere (cube-face layout)
    jsc_mA_cm2: float = 35.0              # AM0 short-circuit-current density (CSS heritage)[paper]
    upgrade_fine_glass_mm: float = 2.0    # UPGRADE: fine head, ~2.8' but +/-30 deg FOV
    upgrade_fine_accuracy_deg: float = 0.046



# ======================================================================================
# 10. RATE GYRO / IMU — navigation-grade MEMS (attitude-rate knowledge through the CA slew)
#     Physical MODEL in imu_gyro.py (Zhao et al. 2021, Micromachines 12, 151 — the flight
#     Lobster-Eye / Einstein-Probe unit): an Allan-variance error chain (turn-on bias, ARW,
#     Gauss-Markov bias instability, rate random walk, scale-factor, misalignment, quant).
#     Companion to star_tracker.py: the gyro PROPAGATES attitude between star-tracker fixes
#     and CARRIES rate knowledge past the star tracker's ~13 deg/s dropout at closest approach.
# ======================================================================================
@dataclass(frozen=True)
class IMUGyro:
    """ARISE rate gyro / IMU — navigation-grade MEMS, calibrated to the flight-proven unit of
    Zhao et al. 2021 (Micromachines 12, 151; flown on the Lobster-Eye X-ray / Einstein-Probe
    satellite, 2020).  Paper Table 3 headline: range +/-20 deg/s, bias stability 0.02 deg/h
    (1-sigma), bias instability 0.006 deg/h, ARW 0.003 deg/root-h, bandwidth 12 Hz, 210 g,
    <1.5 W.  Precision is bought by trading measurement range and bandwidth (6-gyro redundant,
    self-calibrated).  The Allan-variance error MODEL + tactical/heritage comparator presets
    (HG4930, STIM210) live in imu_gyro.py."""
    name: str = "Nav-grade MEMS IMU (Zhao 2021; 6-gyro redundant)"
    # --- Allan-variance noise coefficients (paper Table 1/3) ---
    arw_deg_rth: float = 0.003          # angle random walk [deg/root-hour]                    [paper T3]
    bias_instab_deg_h: float = 0.006    # bias instability (flicker floor) [deg/h]             [paper T3]
    bias_stability_deg_h: float = 0.02  # 1-sigma bias stability over 24 h [deg/h]             [paper T3]
    turnon_bias_deg_h: float = 11.0     # 1-sigma turn-on bias spread (Table 1 offsets -13..+11) [paper T1]
    rrw_deg_h15: float = 0.003          # ASSUMED: rate random walk [deg/h^1.5] (nav-grade long-term)
    # --- deterministic errors (perturb KNOWLEDGE only, never truth) ---
    scale_factor_ppm: float = 300.0     # ASSUMED: residual scale-factor error [ppm] (post self-cal)
    misalign_arcsec: float = 20.0       # ASSUMED: axis-misalignment 1-sigma [arcsec] (post-calibration)
    # --- range / bandwidth / sampling (paper) ---
    range_deg_s: float = 20.0           # +/-20 deg/s measurement range (traded for precision)  [paper]
    bandwidth_hz: float = 12.0          # 12 Hz bandwidth (traded for precision)                [paper]
    sample_rate_hz: float = 100.0       # ASSUMED: 100 Hz output data rate (>= a few x the 12 Hz BW)
    bias_gm_tau_s: float = 100.0        # ASSUMED: bias-instability Gauss-Markov correlation time [s]
    quant_deg_s: float = 1.0e-5         # ASSUMED: rate quantization LSB [deg/s]
    n_gyro: int = 6                     # 6-gyro redundant configuration (2 per axis)           [paper]
    mass_g: float = 210.0               # unit mass [g]                                         [paper]
    power_W: float = 1.5                # unit power [W]                                         [paper]
    # --- heritage comparators (imu_gyro.py builds full presets; paper Introduction) ---
    hg4930_arw_deg_rth: float = 0.04        # Honeywell HG4930 ARW                              [paper intro]
    hg4930_bias_instab_deg_h: float = 0.25  # Honeywell HG4930 bias instability (0.25 deg/h)    [paper intro]
    stim210_bias_instab_deg_h: float = 0.5  # Sensonor STIM210 (~0.5 deg/h class)               [paper intro]


# ======================================================================================
# 11. EVENT-BASED RATE CAMERA — neuromorphic, drift-free angular-rate sensing from star motion
#     Physical MODEL (feature-sensitivity-matrix de-rotation + least squares + orthogonal
#     dual-camera fusion) in imu_gyro.py, calibrated to Franzese & El Hariry 2026 (Adv. Space
#     Res. 77, 7801).  Complements the gyro: no drift, and it keeps working through the fast CA
#     slew where a frame-based star tracker motion-blurs and drops out.
# ======================================================================================
@dataclass(frozen=True)
class EventRateCam:
    """ARISE event-based (neuromorphic) rate camera, calibrated to Franzese & El Hariry 2026
    (Adv. Space Res. 77, 7801-7810).  Estimates angular rate from the temporal distribution of
    brightness events triggered by apparent star motion.  Orthogonal dual-camera total RMS
    0.0275 deg/s (single camera 0.307 deg/s, roll-axis limited); beats miniaturized IMUs
    (0.05-0.5 deg/s) and competes with frame star trackers (0.01-0.1 deg/s), with the unique
    advantage of surviving fast slews and high-dynamic-range scenes."""
    name: str = "Event-based rate camera x2 (orthogonal; Franzese 2026)"
    fov_deg: float = 10.0             # ~10 deg FOV (star-tracker-like star observability)      [paper]
    n_pix: int = 1000                 # 1000 x 1000 sensor (paper simulation array)             [paper]
    contrast_thresh: float = 0.15     # ASSUMED: log-intensity event threshold C (~15%)         [paper eq16]
    window_s: float = 0.1             # 0.1 s measurement window (constant-rate assumption)     [paper]
    max_rate_deg_s: float = 30.0      # method validated to +/-30 deg/s per axis                [paper]
    n_cams: int = 2                   # orthogonal dual-camera configuration (roll recovery)    [paper]
    rms_single_deg_s: float = 0.307   # validation anchor: single-camera total RMS             [paper T1]
    rms_dual_deg_s: float = 0.0275    # validation anchor: orthogonal dual-camera total RMS    [paper T1]
    mass_g: float = 40.0              # as little as 40 g per camera                            [paper]
    power_W: float = 1.0              # ~1 W peak power                                         [paper]
    px_noise_frac: float = 0.5        # ASSUMED: pixel-location noise (largest error source)   [paper T2/T3]



# ======================================================================================
# 12. PROPELLANT — AF-M315E / ASCENT (HAN-based green monopropellant)
#     Physical props feed the catalytic-decomposition + performance MODEL in thruster_hifi.py.
#     Refs: AFRL ASCENT fact sheet; GPIM (NTRS 20140012587). Reduced-order decomposition
#     kinetics: Robledo Asencio 2020 (Arrhenius 2-step) + Kerr 2021 (1D UQ catalyst bed).
# ======================================================================================
@dataclass(frozen=True)
class AFM315E:
    """AF-M315E (ASCENT): HAN/water ionic-liquid green monopropellant.  ~50% higher density-Isp
    than hydrazine, negligible vapor pressure.  Decomposition products expanded through the nozzle."""
    name: str = "AF-M315E / ASCENT (HAN-based green monopropellant)"
    density_kg_m3: float = 1470.0        # 1.47 g/cc                                        [AFRL]
    isp_vac_s: float = 250.0             # ~250 s vac Isp                                    [AFRL]
    flame_temp_K: float = 2100.0         # adiabatic decomposition flame temp ~2100 K        [AFRL]
    gamma: float = 1.25                  # ASSUMED: product-gas ratio of specific heats
    M_products_g_mol: float = 20.0       # ASSUMED: mean molar mass of decomposition products
    cstar_efficiency: float = 0.963      # c* efficiency 0.963 +/- 0.004 (1-sigma)   [sub-N thesis]
    cstar_efficiency_sigma: float = 0.004
    catalytic_thermal_eff: float = 0.929 # max equilibrium catalytic thermal eff 0.929+/-0.014 [thesis]
    # --- reduced-order global decomposition kinetics (Kerr 2021 form; ASSUMED for HAN) ---
    activation_energy_J_mol: float = 1.0e5   # ASSUMED: global Arrhenius Ea (HAN ~80-120 kJ/mol)
    prefactor_1_s: float = 5.0e6             # ASSUMED: global Arrhenius prefactor (tuned to eta_cat)
    preheat_temp_K: float = 398.0            # catalyst-bed preheat >= 125 C before firing  [ops]
    # --- hydrazine reference (Robledo 2020 Table 1; for model cross-check) ---
    hydrazine_flame_temp_K: float = 1200.0
    hydrazine_isp_s: float = 230.0
    hydrazine_Ta_N2H4_homog_K: float = 18333.0   # activation temperature (Ea = Ta*R)   [Kesten/Robledo]
    hydrazine_Ta_N2H4_het_K: float = 2777.0
    hydrazine_Ta_NH3_het_K: float = 27778.0

    def cstar_ideal_m_s(self, eff: bool = False) -> float:
        """Ideal characteristic velocity c* = sqrt(R_s Tc) / Gamma(gamma) [m/s]; eff=True applies
        the measured c* efficiency."""
        Ru = 8.314462; Rs = Ru / (self.M_products_g_mol * 1e-3); g = self.gamma
        Gamma = np.sqrt(g) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
        c = np.sqrt(Rs * self.flame_temp_K) / Gamma
        return float(c * (self.cstar_efficiency if eff else 1.0))


# ======================================================================================
# 13. FEED SYSTEM — blowdown tank + feed line + solenoid valve + injector + catalyst bed
#     Water-hammer (Joukowsky/MOC), valve transient, chugging feed-coupling, and blowdown
#     Pc/thrust decay MODEL in thruster_hifi.py.  Refs: sub-Newton thesis (blowdown, chug,
#     c*); water-hammer (Joukowsky; Lecourt/Ounougha satellite feed-line studies).
# ======================================================================================
@dataclass(frozen=True)
class MPS130Feed:
    """MPS-130 propellant line: passive-blowdown tank -> feed line -> solenoid valve -> injector
    -> catalyst bed -> chamber/nozzle.  Numbers ASSUMED/representative for a 1U green-monoprop unit."""
    name: str = "MPS-130 blowdown feed line (AF-M315E)"
    # tank / blowdown
    tank_volume_m3: float = 1.16e-3      # ASSUMED: ~1.16 L (1.7 kg AF-M315E at 1.47 g/cc + ullage)
    ullage_frac0: float = 0.10           # ASSUMED: initial ullage fraction (pressurant)
    tank_press0_Pa: float = 2.4e6        # ASSUMED: 24 bar BOL tank pressure
    blowdown_ratio: float = 4.0          # ASSUMED: BOL/EOL pressure ratio over tank life
    pressurant_gamma: float = 1.4        # GHe/GN2 ullage (isentropic bound; ~1 isothermal)
    # chamber / nozzle
    chamber_press_nom_Pa: float = 1.0e6  # ~10 bar nominal Pc (thesis test point)             [thesis]
    throat_area_m2: float = 2.57e-14 * 1e6  # placeholder recomputed in module; keep >0
    cd_nozzle: float = 0.96              # discharge coefficient                              [thesis]
    # feed line (for water-hammer)
    line_length_m: float = 0.20          # ASSUMED
    line_diameter_m: float = 1.0e-3      # ASSUMED: 1 mm ID
    line_wall_thick_m: float = 2.0e-4    # ASSUMED: 0.2 mm wall
    line_wall_E_Pa: float = 200.0e9      # steel/Ti wall modulus
    prop_bulk_modulus_Pa: float = 2.0e9  # ASSUMED: AF-M315E bulk modulus
    # solenoid valve
    valve_open_s: float = 5.0e-3         # ASSUMED: 5 ms opening
    valve_close_s: float = 3.0e-3        # ASSUMED: 3 ms closing (sets water-hammer severity)
    valve_flow_velocity_ms: float = 2.0  # ASSUMED: steady line velocity (priming surge driver)
    # injector (chug stability)
    injector_stiffness: float = 0.25     # ASSUMED: dP_inj / Pc (>=~0.2 stiff/stable)         [thesis]
    chug_stiffness_threshold: float = 0.15  # onset of feed-coupled 'chugging' below this      [thesis]
    # catalyst bed
    bed_length_m: float = 0.020          # ASSUMED
    bed_diameter_m: float = 0.010        # ASSUMED
    pellet_diameter_m: float = 5.0e-4    # ASSUMED: 0.5 mm pellets (Shell-405 / Ir-Al2O3 class)
    bed_void_frac: float = 0.40          # ASSUMED: packed-bed voidage
    bed_loading_kg_m2_s: float = 30.0    # ASSUMED: mass flux through the bed


# ======================================================================================
# 14. PROPELLANT SLOSH — Abramson/Dodge equivalent-mechanical (pendulum) analog
#     Disturbance force/torque on the bus during a burn; MODEL in thruster_hifi.py.
#     Ref: Abramson (NASA SP-106) / Dodge "Dynamic Behavior of Liquids in Moving Containers".
# ======================================================================================
@dataclass(frozen=True)
class PropellantSlosh:
    """First lateral slosh mode of the (cylindrical) tank as an equivalent pendulum: frequency,
    slosh mass fraction, and damping.  Frequency rises with axial accel, falls with tank radius."""
    name: str = "AF-M315E slosh (Abramson/Dodge equivalent pendulum)"
    tank_radius_m: float = 0.05          # ASSUMED: 5 cm tank radius
    fill_fraction: float = 0.5           # liquid fill (varies over life)
    damping_ratio: float = 0.005         # ASSUMED: 0.5% (smooth-wall, no baffles)
    first_mode_root: float = 1.841       # first root of J1' (cylindrical tank first sloshing mode)


# ======================================================================================
# 15. THRUSTER-LINE SENSORS — health/telemetry (perturb KNOWLEDGE, never truth)
#     Pressure transducers (tank/feed/chamber), catalyst-bed thermocouple, flow, valve position.
#     Refs: aerospace pressure transducers (~0.1-0.25% FS); K-type thermocouples (catalyst bed).
# ======================================================================================
@dataclass(frozen=True)
class ThrusterLineSensors:
    """Thruster-line instrumentation error models.  measured = truth + bias + noise; truth intact."""
    name: str = "MPS-130 thruster-line sensor suite"
    press_pct_fs: float = 0.25           # pressure transducer 1-sigma [% of full scale]     [aero PT]
    press_fs_bar: float = 40.0           # transducer full-scale [bar]
    press_bias_pct_fs: float = 0.1       # ASSUMED: fixed bias [% FS]
    tc_sigma_K: float = 2.2              # K-type thermocouple 1-sigma [K] (catalyst bed)
    tc_bias_K: float = 1.0               # ASSUMED: thermocouple fixed bias [K]
    tc_max_K: float = 2573.0             # K-type usable to ~2300 C                            [space TC]
    flow_pct: float = 1.0                # ASSUMED: flow-meter 1-sigma [% of reading]
    valve_pos_sigma: float = 0.01        # ASSUMED: valve-position 1-sigma [fraction]
    sample_rate_hz: float = 100.0        # ASSUMED: health-telemetry rate



# ======================================================================================
# 16. CLOCK — Microchip SA.45s Chip-Scale Atomic Clock (CSAC), the fleet time reference
#     Allan-variance clock MODEL (3-state Hutsell; white/flicker/random-walk FM + aging) in
#     pnt_timing.py.  Refs: Microchip CSAC SA.45s datasheet; Teng 2025 (clock model / Hadamard).
# ======================================================================================
@dataclass(frozen=True)
class CSACClock:
    """SA.45s CSAC: <35 g, <120 mW, ~17 cc.  Short-term Allan deviation sigma_y(tau) = 3e-10/sqrt(tau)
    (white-FM dominated): 3e-10 @1 s, 1e-10 @10 s, 3e-11 @100 s, 1e-11 @1000 s.  Aging <9e-10/month."""
    name: str = "Microchip SA.45s CSAC"
    allan_1s: float = 3.0e-10            # sigma_y at tau=1 s (white FM)                     [datasheet]
    allan_10s: float = 1.0e-10           # sigma_y at tau=10 s                               [datasheet]
    allan_100s: float = 3.0e-11          # sigma_y at tau=100 s                              [datasheet]
    allan_1000s: float = 1.0e-11         # sigma_y at tau=1000 s                             [datasheet]
    aging_per_month: float = 9.0e-10     # frequency aging (drift) per month                 [datasheet]
    flicker_floor: float = 3.0e-12       # ASSUMED: flicker-FM floor (below the white-FM tail; plateaus >1000 s)
    mass_g: float = 35.0                 # <35 g                                             [datasheet]
    power_W: float = 0.12                # <120 mW                                           [datasheet]

    def h0(self) -> float:
        """White-FM PSD coefficient h0 from sigma_y(1 s): sigma_y^2 = h0/(2 tau) -> h0 = 2 sigma_y(1)^2."""
        return 2.0 * self.allan_1s ** 2
    def h_minus2(self) -> float:
        """Random-walk-FM coefficient from aging (per-second frequency-drift diffusion, ASSUMED)."""
        drift_per_s = self.aging_per_month / (30.0 * 86400.0)
        return (drift_per_s ** 2)        # representative RW-FM level


# ======================================================================================
# 17. TIME-TRANSFER LINK — inter-satellite two-way time transfer (TWTT) over the optical ISL
#     Cancels range + common delay; residual = fixed/periodic delay calibration + thermal noise
#     + clock drift over the update interval.  Ref: Teng 2025 (delays, DLL thermal noise).
# ======================================================================================
@dataclass(frozen=True)
class TimeTransferLink:
    """Two-way time-transfer error budget for a chief<->deputy or deputy<->deputy ISL."""
    name: str = "ARISE inter-satellite TWTT (optical ISL)"
    fixed_delay_sigma_s: float = 0.9e-9      # fixed hardware-delay calibration residual [s]  [Teng]
    periodic_delay_sigma_s: float = 0.5e-9   # thermal/orbital periodic-delay residual [s]    [Teng]
    cn0_dbhz_chief_deputy: float = 50.0      # ASSUMED: strong intra-swarm optical link C/N0
    cn0_dbhz_deputy_deputy: float = 45.0     # ASSUMED: slightly weaker deputy-deputy link
    integration_s: float = 1.0               # TWTT exchange / update interval [s]
    loop_bw_hz: float = 1.0                  # tracking-loop bandwidth [Hz]
    ranging_sigma_m: float = 0.005           # ASSUMED: optical ISL ranging ~5 mm (isl_comms class)


# ======================================================================================
# 18. GNSS TIMING — chief's GNSS-disciplined reference (chief<->GNSS), the fleet's ABSOLUTE time
#     ARISE images Apophis at ~38,000 km geocentric -> GNSS Space Service Volume (SSV, above the
#     GPS shell) -> degraded but usable timing.  Disciplines the chief CSAC to GPS/UTC.
# ======================================================================================
@dataclass(frozen=True)
class GNSSTiming:
    """Chief GNSS timing receiver operating in the GPS Space Service Volume near the flyby."""
    name: str = "chief GNSS-disciplined CSAC (SSV timing)"
    ssv_timing_sigma_s: float = 3.0e-8       # ASSUMED: ~30 ns SSV one-way timing (degraded geometry)
    disciplined_sigma_s: float = 2.0e-8      # ASSUMED: ~20 ns after clock steering/filtering
    availability: float = 0.7                # ASSUMED: SSV signal availability fraction near CA
    gps_utc_bias_s: float = 1.0e-8           # ASSUMED: GPS-time to UTC realization residual



# ======================================================================================
# 19. DETECTOR DEEP-FIDELITY — CIS120 spatial PSF, charge transfer, radiation damage, FPN
#     Physical MODEL in detector_hifi.py.  Refs: Stevens 1991 / Djite 2012 (charge-diffusion +
#     IPC MTF); Virmontois 2010 + Sensors/MDPI 2018 (proton DDD dark current + hot-pixel tail);
#     PPD charge-transfer-inefficiency (Solid-State Electronics); EMVA 1288 (PRNU/DSNU); Capella
#     CIS120 SPIE 11852-0S (calibration anchor).  Deep effects beyond the detector.py B1 chain.
# ======================================================================================
@dataclass(frozen=True)
class CIS120PSF:
    """Sensor point-spread beyond the optics: lateral charge diffusion in the BSI epi + interpixel
    capacitance (IPC) crosstalk.  Broadens the effective PSF / lowers the sensor MTF."""
    name: str = "CIS120 sensor PSF (charge diffusion + IPC crosstalk)"
    diffusion_length_um: float = 2.5     # lateral charge-diffusion length in the BSI epi [um]
    epi_thickness_um: float = 10.0       # epitaxial (collection) thickness [um]
    ipc_alpha: float = 0.010             # IPC coupling to EACH of the 4 nearest neighbors [fraction]
    optical_sigma_px: float = 0.7        # optics-only PSF sigma [px] (the current detector.py value)


@dataclass(frozen=True)
class CIS120CTE:
    """4T pinned-photodiode charge-transfer inefficiency (CTI) / image lag, and its TID sensitivity."""
    name: str = "CIS120 charge transfer / image lag (4T PPD)"
    cti: float = 1.0e-3                  # charge-transfer inefficiency (nominal)
    cti_criterion: float = 5.0e-3        # 0.5% CTI acceptable-lag criterion (Solid-State Electronics)
    lag_frac: float = 1.0e-3             # first-frame image-lag fraction (~ CTI)
    cti_per_krad: float = 3.0e-4         # TID-induced CTI increase per krad(Si)


@dataclass(frozen=True)
class CIS120Radiation:
    """Radiation environment + displacement-damage/TID response: mean dark increase + hot-pixel
    exponential tail (Virmontois), RTS, and prompt transient hits.  Fluence is 10-MeV-equivalent."""
    name: str = "CIS120 radiation damage (proton DDD + TID)"
    proton_fluence_p_cm2: float = 1.0e10     # mission 10-MeV-equiv proton fluence [p/cm^2]
    tid_krad: float = 5.0                    # total ionizing dose [krad(Si)]
    ddd_dark_e_s_per_1e10: float = 5.0       # MEAN dark-current increase [e-/s] per 1e10 p/cm^2  [Virmontois]
    hotpix_tail_e_s_per_1e10: float = 30.0   # exponential-tail scale of the dark increase [e-/s] per 1e10
    hotpix_threshold_e_s: float = 100.0      # hot-pixel definition [e-/s]
    rts_pixel_frac_per_1e10: float = 0.002   # fraction of pixels showing RTS per 1e10 p/cm^2
    rts_amplitude_e: float = 3.0             # RTS peak-to-peak amplitude [e-]
    transient_rate_hits_cm2_s: float = 5.0   # prompt particle (cosmic-ray/proton) transient rate [hits/cm^2/s]
    transient_charge_e: float = 2.0e3        # ASSUMED: mean charge deposited by a transient hit [e-]


@dataclass(frozen=True)
class CIS120FPN:
    """Fixed-pattern non-uniformity per EMVA 1288 (spatial std of the gain and dark fields).
    Anchored to the CIS120Detector scalars; the spatial-field MODEL lives in detector_hifi.py."""
    name: str = "CIS120 PRNU/DSNU fixed-pattern noise (EMVA 1288)"
    prnu_frac: float = 0.010             # photo-response non-uniformity (1-sigma, fraction of signal)
    dsnu_e: float = 2.0                  # dark-signal non-uniformity (1-sigma) [e-]
    dsnu_rad_e_per_1e10: float = 1.5     # DSNU growth from displacement damage [e-] per 1e10 p/cm^2



# ======================================================================================
# 20. ILLUMINATION GEOMETRY — finite-Sun soft shadows + surface self-illumination (ATPM)
#     Feeds the radiative occlusion MODEL in occlusion_bvh.py.  Ref: Rozitis & Green 2011
#     (MNRAS 415, 2042, Advanced Thermophysical Model): facet-mesh partial shadowing + sunlight
#     scattering + self-heating via view factors.  Apophis albedo: Binzel 2009 (Sq-type, pV~0.30).
# ======================================================================================
@dataclass(frozen=True)
class IlluminationGeom:
    """Illumination-geometry constants for soft shadows (finite Sun angular size) and single-bounce
    self-illumination (scattered sunlight between interfacing facets)."""
    name: str = "Apophis illumination geometry (ATPM)"
    sun_angular_radius_deg_1au: float = 0.2666   # Sun's angular radius at 1 AU (physical)
    helio_dist_au: float = 1.0                   # Apophis heliocentric distance at 2029 CA (~1 AU)
    bond_albedo: float = 0.12                    # ASSUMED: Bond albedo (Sq-type, pV~0.30)   [Binzel 2009]
    scatter_albedo: float = 0.30                 # single-scatter reflectance for self-illumination (~pV)

    def sun_angular_radius_deg(self) -> float:
        return self.sun_angular_radius_deg_1au / self.helio_dist_au



# ======================================================================================
# 21. PHOTOGRAMMETRY / SPC — reconstruction-accuracy model (stereophotoclinometry)
#     Physical MODEL in photogram.py.  Refs: Gaskell/Barnouin 2020 + Palmer 2024 (PSJ, SPC accuracy
#     for OSIRIS-REx/Bennu, ~pixel-scale / ~0.4% mean radius); Debella-Gilo & Kaeaeb 2011 (RSE) +
#     Forstner (Cramer-Rao subpixel matching); Horn 1990 / Kirk 1987 (photoclinometry / shape-from-shading).
# ======================================================================================
@dataclass(frozen=True)
class PhotogramModel:
    """Constants for the hi-fi reconstruction metric: texture-limited (Cramer-Rao) subpixel matching,
    multi-ray triangulation covariance, and photoclinometry (SFS) that recovers mono/low-parallax facets."""
    name: str = "ARISE SPC reconstruction model"
    match_window_px: int = 25            # NxN correlation window (N^2 pixels) for the CRLB
    texture_energy: float = 0.5          # ASSUMED: normalized image gradient energy (scene texture, 0..1)
    crlb_const: float = 0.5              # Cramer-Rao proportionality (calibrated to ~0.1 px floor at high SNR)
    sigma_disp_floor_px: float = 0.1     # best achievable subpixel disparity (well-textured, high SNR)
    pc_slope_noise_ref: float = 1.0      # photoclinometry radiometric->slope reference (dI/dslope scale)
    pc_incidence_min_deg: float = 5.0    # below this incidence SFS is ill-posed (near-normal, dI/dslope->0)
    apophis_mean_radius_m: float = 170.0 # Apophis mean radius (for the %-radius accuracy benchmark)
    spc_pixel_accuracy_frac: float = 1.0 # SPC delivers ~1 pixel (GSD) accuracy on well-observed facets [Palmer 2024]
    pc_systematic_frac: float = 0.2      # photoclinometry systematic floor (BRDF/albedo) as a fraction of GSD [Palmer 2024]



# ======================================================================================
# 22. NAVIGATION — consider parameters + dynamic model compensation (Schmidt-Kalman)
#     Physical MODEL in nav_stage.py.  Refs: Tapley, Schutz & Born "Statistical Orbit
#     Determination" (consider filter + DMC); Woodburn/Tanygin (recursive consider filter);
#     OSIRIS-REx landmark opnav (Gaskell/Getzandanner).  The un-estimated systematic biases
#     (which do NOT average down) set the achievable nav floor that a plain EKF misses.
# ======================================================================================
@dataclass(frozen=True)
class NavConsider:
    """Un-estimated 'consider' parameter uncertainties + DMC (Gauss-Markov unmodeled acceleration)."""
    name: str = "ARISE nav consider / DMC parameters"
    meas_range_bias_m: float = 1.0        # ASSUMED: systematic LIDAR range bias [m] (does not average down)
    meas_angle_bias_arcsec: float = 5.0   # ASSUMED: systematic camera/bearing bias [arcsec]
    chief_ephem_bias_m: float = 15.0      # chief-broadcast Apophis-ephemeris bias [m] (fleet common-mode)
    gm_uncertainty_frac: float = 0.01     # ASSUMED: Apophis GM 1-sigma fractional uncertainty
    dmc_tau_s: float = 300.0              # DMC Gauss-Markov acceleration correlation time [s]
    dmc_sigma_accel_ms2: float = 1.0e-7   # DMC unmodeled-acceleration 1-sigma (SRP class) [m/s^2]


# Convenience singletons
XB6 = XB6Bus()
ESPA = ESPAChief()
CIS120 = CIS120Detector()
MPS130 = MPS130Thruster()
STAR_TRACKER = StarTracker()
REACTION_WHEEL = ReactionWheel()
FAST_STEERING_MIRROR = FastSteeringMirror()
SUN_SENSOR = SunSensor()
IMU_GYRO = IMUGyro()
EVENT_RATE_CAM = EventRateCam()
AFM315E_PROP = AFM315E()
MPS130_FEED = MPS130Feed()
PROP_SLOSH = PropellantSlosh()
THR_SENSORS = ThrusterLineSensors()
CSAC = CSACClock()
TWTT_LINK = TimeTransferLink()
GNSS_TIMING = GNSSTiming()
CIS120_PSF = CIS120PSF()
CIS120_CTE = CIS120CTE()
CIS120_RAD = CIS120Radiation()
CIS120_FPN = CIS120FPN()
ILLUM_GEOM = IlluminationGeom()
PHOTOGRAM = PhotogramModel()
NAV_CONSIDER = NavConsider()


if __name__ == "__main__":
    print("ARISE V2 hardware sanity")
    print("  XB6 inertia diag [kg m^2] :", np.round(XB6.inertia_diag(), 4))
    print("  XB6 dV capacity on MPS-130:", round(MPS130.dv_capacity_mps(XB6.wet_mass_kg), 1), "m/s")
    print("  XB6 strobe optical budget :", round(XB6.strobe_optical_energy_budget_J(), 1), "J/deputy")
    print("  CIS120 dark @0C  [e-/px/s]:", round(CIS120.dark_current_e_s(), 2))
    print("  FIXED focal length        :", round(FOCAL_M * 1e3, 2), "mm   IFOV", IFOV_RAD, "rad")
    print("  GSD @750 m                :", round(IFOV_RAD * DESIGN_STANDOFF_M, 3), "m")
    print("  fastest aperture (f/1.8)  :", round(APERTURE_MAX_M * 1e2, 2), "cm  (dark-side stop)")
