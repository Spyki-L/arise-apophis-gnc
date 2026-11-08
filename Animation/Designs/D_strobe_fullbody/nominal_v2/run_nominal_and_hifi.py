#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_nominal_and_hifi.py  —  ARISE Design D FULL high-fidelity pipeline (one click, no PowerShell)
================================================================================================
Open this file in Visual Studio and RUN it (green |>).  No command-line arguments.

Runs the COMPLETE mission-grade Design D nominal case END-TO-END, FROM DEPLOYMENT, at the
highest fidelity only (the old low-fidelity CASE-1 baseline scoring pass has been REMOVED -- this
is now a single hi-fi pipeline), then regenerates every report figure + statistics table:

  STAGE A  DEPLOYMENT SYNC (no Basilisk here -- just the interlink seed).
           If the chief-deputy SEPARATION stage has been run (../chief deputy seperation/
           run_separation.py), this consumes its two products so the whole pipeline starts from
           the REAL RAFTI separation, not an idealized impulse:
             * separation_seed.npz     -> ARISE_SEPARATION_SEED_NPZ  (deploy state + deployment dV)
             * handoff_summary.json     -> ARISE_HANDOFF_SUMMARY_JSON (SE(3) pH -> MRP_FB handoff)
           A startup self-check verifies both are present, mutually consistent (same deputy count)
           and finite BEFORE the ~1.5 h run, so a stale/half-written interlink is caught early.

  STAGE B  HIGH-FIDELITY MISSION (Basilisk), the single nominal draw run FRESH through the full
           V2 chain via MC_v2_run.py --stages guidance,nav,6dof,score:
             1. GUIDANCE  basilik_hifi_apophis_rendezvous_fleet.py -- deploys each deputy from the
                          post-RAFTI seed state (deployment dV in sync) -> per-deputy trajectories.
             2. NAV       run_nav_on_new_traj.py -- per-deputy EKF -> nav_uncertainty.npz.
             3. 6-DOF     build_6dof_anim_data.py -- the ~10.8 h MRP_FB coast to CA, SEEDED at each
                          deputy's SE(3) pH -> MRP_FB handoff residual (handoff_summary.json), i.e.
                          MRP_FB ACQUIRES from the separation hand-off, not a fictitious aligned start.
             4. SCORE     full V2 imaging chain (occlusion ON, Hapke BRDF, real 24 deg SPICE Sun,
                          strobe cap-bank, NAC+WAC tiers, photogrammetry), over M spin phases.
           Every per-draw product (trajectories / attitude / control / nav / camera / coverage /
           apophis / anim) is written under data/mc/draw_0000/ AND the run inputs (seed + handoff)
           are copied in + a run_manifest.json is written, so ALL data needed to recreate any plot
           is saved with the run.  ~1.5 h.

  STAGE C  make_report_figures.py -- reads the STAGE-B products and writes ALL report figures
           (coverage, recon, nav EKF residuals, guidance convergence, attitude/rate/torque near CA,
           per-deputy FOV cones) + the stat tables (report_tables.txt / report_values.json /
           report_tables.tex) into outputs/.  Pure numpy+matplotlib, ~1 min.

Order of operation: run "chief deputy seperation/run_separation.py" FIRST (produces the seed +
handoff), then run THIS.  If you re-run the separation, re-run this so the deployment dV + handoff
stay in sync (the startup self-check will flag a mismatch).
All defaults are set below.  Edit only PY311 / DIST3 if your install differs, or M / NFRAME.
"""
import os, sys, subprocess, json, time

# ====== Basilisk environment (edit ONLY if your install paths differ) ====================
PY311 = r"C:\Program Files\Python311\python.exe"                                 # the Python that has Basilisk
DIST3 = r"C:\Basilisk\basilisk-develop\dist3"      # Basilisk source build (PYTHONPATH)

os.environ["PYTHONPATH"] = DIST3 + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
os.environ["PYTHONUTF8"] = "1"          # so Unicode prints don't crash when redirected on Windows
os.environ["MPLBACKEND"] = "Agg"

# Use the Basilisk Python 3.11 for ALL the heavy work.  subprocess quotes the space in
# "C:\Program Files\..." correctly (os.execv does NOT -- that caused the earlier split error).
# This parent script itself can run under any interpreter; it only orchestrates subprocesses.
PY = PY311 if os.path.exists(PY311) else sys.executable

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN = os.path.dirname(HERE)                  # .../D_strobe_fullbody
MC = os.path.join(HERE, "MC_v2_run.py")
FIGS = os.path.join(HERE, "make_report_figures.py")   # all report figures + tables, in one shot
# Mission-proven stage scripts (run by MC_v2_run.py per draw; compiled in the preflight).
SIXDOF = os.path.join(DESIGN, "scripts", "sim", "build_6dof_anim_data.py")        # ~10.8 h MRP_FB coast
NAVSCR = os.path.join(DESIGN, "scripts", "navigation", "run_nav_on_new_traj.py")  # per-deputy EKF
GUID   = os.path.join(DESIGN, "scripts", "guidance", "basilik_hifi_apophis_rendezvous_fleet.py")

# ===== Re-fly stations + NAC imaging architecture (CASE-2) ================================
# Point the guidance at the collision-safe CMA-optimized stations; the scorer adds the NAC
# 0.15 m tier, earthshine dark-side fill, and the visual-hull shape tier automatically.
# Swap to refly_targets_1500m_nac.npz for the safer (1295 m clearance) standoff option.
REFLY_TARGETS = os.path.join(HERE, "refly_targets_cma.npz")
if os.path.exists(REFLY_TARGETS):
    os.environ["ARISE_DEPUTY_TARGETS_NPZ"] = REFLY_TARGETS
    print(f"[refly] guidance targets : {os.path.basename(REFLY_TARGETS)}  (NAC 0.15 m @3 km + earthshine + tiered scoring)")
else:
    print("[refly] refly_targets_cma.npz not found -> CASE 2 will fly the original 3 km stations")

# ===== Near-CA trajectory sampling: 500 Hz so the capture gate is not data-rate-limited =======
# The body is within imaging range (<=5 km) only ~0.6 s before CA; 100 Hz under-samples the
# close-deputy bursts (~10-20 ms windows -> only 1-2 frames).  Sampling the final tail at 500 Hz
# resolves those bursts (~5-10 frames), directly strengthening the thin 0.15 m tier.  The tail
# need only cover the sub-second imaging window; 30 s at 500 Hz = 15k samples/deputy (~45x margin,
# modest file/runtime cost).  (8 min would be 4,200 km out of range -> 240k samples for nothing.)
os.environ["ARISE_EXPORT_FINE_DT"] = "0.002"          # 500 Hz dense near-CA export (was 0.01 = 100 Hz)
os.environ["ARISE_EXPORT_TAIL_S"]  = "30.0"           # last 30 s before CA at the fine rate (>> ~0.6 s window)
print(f"[sample] near-CA export : {1.0/float(os.environ['ARISE_EXPORT_FINE_DT']):.0f} Hz over last "
      f"{os.environ['ARISE_EXPORT_TAIL_S']} s  (resolves the capture-gate bursts incl. the 0.15 m close passes)")

# ===== Relative-nav + timing KNOWLEDGE budget (item #2): PREMIUM suite ========================
# Laser inter-satellite ranging (~cm relative position) + star-tracker bearing (~5 arcsec) +
# chip-scale atomic clock with two-way time transfer (~0.3 us fleet sync, ~0.1 us camera frame).
# run_nav_on_new_traj.py writes _nav_stage/nav_uncertainty.npz from these; the scorer inflates
# sigma_z for CROSS-deputy stereo only (self/motion-stereo errors are common-mode over the burst).
# At this suite cross-deputy stereo reaches ~0.15 m, so it also contributes fine reconstruction.
# laser ISL inter-deputy relative-position knowledge [m]: opt-in physical link-budget model, else 0.03
if os.getenv("ARISE_ISL_HIFI", "0") not in ("0", "", "false", "False"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
    import isl_comms as _ISL
    _isl_range_m = float(os.getenv("ARISE_ISL_RANGE_M", "1000"))     # representative inter-deputy range
    _relnav = _ISL.DEFAULT.ranging_sigma_m(_isl_range_m)
    os.environ["ARISE_RELNAV_SIGMA_M"] = f"{_relnav:.4f}"
    _cs = _ISL.link_summary(_isl_range_m)
    print(f"[nav-uq] ISL HI-FI ON: optical link margin {_cs['link_margin_dB']:.0f} dB, ranging sigma {_relnav*1e3:.1f} mm, "
          f"max {_cs['max_data_rate_Gbps']:.0f} Gbps, state {_cs['state']} (vs flat 30 mm)")
else:
    os.environ["ARISE_RELNAV_SIGMA_M"]      = "0.03"    # laser ISL inter-deputy relative-position knowledge [m]
# star-tracker bearing/pointing knowledge [arcsec]: opt-in hi-fi XACT-50 model, else 5.0 default
if os.getenv("ARISE_STARTRACKER_HIFI", "0") not in ("0", "", "false", "False"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
    import star_tracker as _ST
    # imaging is mid-slew; a hi-fi FSM (step-stare) settles the body -> much lower star-tracker rate
    if os.getenv("ARISE_FSM_HIFI", "0") not in ("0", "", "false", "False"):
        import fast_steering_mirror as _FSM
        _eff = _FSM.FSM.effective_startracker_rate_deg_s()
        _bear = _ST.active().pointing_sigma_rad(_FSM.np.radians(_eff)) / _ST.ARCSEC
        print(f"[nav-uq] STAR-TRACKER+FSM HI-FI: FSM step-stare -> {_eff:.1f} deg/s -> pointing knowledge = {_bear:.2f} arcsec ({_ST.active().name[:34]})")
    else:
        _bear = _ST.active().midslew_pointing_sigma_rad() / _ST.ARCSEC
        print(f"[nav-uq] STAR-TRACKER HI-FI ON: mid-slew pointing knowledge = {_bear:.2f} arcsec ({_ST.active().name[:34]} at {_ST.active().max_slew_deg_s:.0f} deg/s; no FSM)")
    os.environ["ARISE_BEARING_SIGMA_ARCSEC"] = f"{_bear:.2f}"
else:
    os.environ["ARISE_BEARING_SIGMA_ARCSEC"] = "5.0"    # star-tracker bearing/pointing knowledge [arcsec]

# reaction-wheel microvibration (opt-in): RSS the near-CA imaging jitter into the pointing budget
if os.getenv("ARISE_RW_HIFI", "0") not in ("0", "", "false", "False"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
    import reaction_wheels as _RW, hardware as _HW, numpy as _np
    _rwj = _RW.RWA.operating_jitter_arcsec(_np.radians(_HW.STAR_TRACKER.max_slew_deg_s))
    _cur = float(os.environ.get("ARISE_BEARING_SIGMA_ARCSEC", "5.0"))
    _tot = (_cur ** 2 + _rwj ** 2) ** 0.5
    os.environ["ARISE_BEARING_SIGMA_ARCSEC"] = f"{_tot:.2f}"
    print(f"[nav-uq] REACTION-WHEEL HI-FI ON: microvibration jitter {_rwj:.2f} arcsec -> pointing budget {_cur:.1f} -> {_tot:.2f} arcsec")

# rate gyro / IMU (opt-in): nav-grade MEMS onboard RATE knowledge (Zhao 2021) for the AttEKF gyro
# update + attitude propagation across the star-tracker dropout (>~13 deg/s) at closest approach,
# optionally FUSED with the drift-free event-camera rate aid (Franzese 2026).  Sets the ATT-channel
# gyro sigma only; it does NOT touch the star-tracker pointing budget above.  OFF -> env stays unset
# -> AttEKF keeps its exact flat 0.003 deg/s heritage value (identity-when-absent).
if os.getenv("ARISE_IMU_HIFI", "0") not in ("0", "", "false", "False"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
    import imu_gyro as _IMU
    _g = _IMU.active(); _sig = _g.ekf_rate_sigma_rad_s(); _src = _g.name[:30]
    if os.getenv("ARISE_RATEAID_EVENTCAM", "0") not in ("0", "", "false", "False"):
        _evrms = _IMU.HW.EVENT_RATE_CAM.rms_dual_deg_s * _IMU.DEG        # drift-free event-cam dual RMS
        _, _sf = _IMU.fuse_rate(_IMU.np.zeros(3), _sig, _IMU.np.zeros(3), _evrms)
        _sig = float(_IMU.np.mean(_sf)); _src += "+eventcam"
    os.environ["ARISE_GYRO_SIGMA_RADPS"] = f"{_sig:.4e}"
    _flat = 0.003 * _IMU.DEG
    print(f"[adcs] IMU HI-FI ON: gyro rate knowledge {_sig/_IMU.DEG*1e3:.3f} mdeg/s "
          f"(ARW {_g.arw_deg_rth:.3f} deg/rt-h, BI {_g.bias_instab_deg_h:.3f} deg/h, range +/-{_g.range_deg_s:.0f} deg/s) "
          f"-> {_flat/_sig:.0f}x tighter than the flat 0.003 deg/s default  ({_src})")
# else: leave ARISE_GYRO_SIGMA_RADPS unset -> AttEKF uses its exact np.deg2rad(0.003) default (identity)

# thruster line (opt-in): hi-fi MPS-130 / AF-M315E execution -- c*-efficiency + blowdown Isp,
# water-hammer MIB inflation, chugging penalty (thruster_hifi.py; Robledo 2020 + Kerr 2021 + the
# sub-Newton thesis).  Exports the effective execution knobs the MC dV tail (thruster.py) reads.
# OFF -> env stays unset -> thruster.py uses the flat MPS-130 constants (identity-when-absent).
if os.getenv("ARISE_THRUSTER_HIFI", "0") not in ("0", "", "false", "False"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
    import thruster_hifi as _THX
    _ep = _THX.effective_params(); _wh = _THX.water_hammer(); _cg = _THX.chug_state()
    os.environ["ARISE_THRUSTER_ISP_S"]   = f"{_ep['isp_s']:.2f}"
    os.environ["ARISE_THRUSTER_MIB_NS"]  = f"{_ep['mib_Ns']:.5f}"
    os.environ["ARISE_THRUSTER_LOSS_FRAC"] = f"{_ep['loss_frac']:.4f}"
    print(f"[prop] THRUSTER HI-FI ON: AF-M315E c*-eff Isp {_ep['isp_s']:.1f} s (vs flat 245), "
          f"MIB {_ep['mib_Ns']*1e3:.2f} mN-s (water-hammer +{_THX.mib_inflation_frac()*100:.0f}%, "
          f"surge {_wh['dP_effective_Pa']/1e6:.2f} MPa), chugging={_cg['chugging']}, loss {_ep['loss_frac']*100:.1f}%")
# else: leave ARISE_THRUSTER_* unset -> thruster.py uses the flat MPS-130 constants (identity)

# detector deep-fidelity (opt-in): CIS120 spatial PSF (charge diffusion + IPC crosstalk), charge-
# transfer/image-lag, proton-radiation dark current + hot pixels + RTS, and PRNU/DSNU fields
# (detector_hifi.py; Stevens/Djite + Virmontois + PPD-CTI + EMVA 1288).  Exports the effective
# detector params detector.py reads.  OFF -> unset -> flat CIS120 constants (identity-when-absent).
if os.getenv("ARISE_DETECTOR_HIFI", "0") not in ("0", "", "false", "False"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
    import detector_hifi as _DET
    _dp = _DET.effective_params()
    os.environ["ARISE_DET_DARK_E_S"]     = f"{_dp['dark_e_s']:.4f}"
    os.environ["ARISE_DET_SIGMA_PSF_PX"] = f"{_dp['sigma_psf_px']:.4f}"
    os.environ["ARISE_DET_READ_E"]       = f"{_dp['read_e']:.4f}"
    os.environ["ARISE_DET_PRNU_FRAC"]    = f"{_dp['prnu_frac']:.5f}"
    os.environ["ARISE_DET_DSNU_E"]       = f"{_dp['dsnu_e']:.4f}"
    print(f"[det] DETECTOR HI-FI ON: dark {_dp['dark_e_s']:.1f} e-/s (radiation +DDD), PSF sigma "
          f"{_dp['sigma_psf_px']:.3f} px (charge-diff+IPC {_dp['ipc_crosstalk']*100:.0f}%, MTF_ny {_dp['mtf_nyquist']:.2f}), "
          f"hot-pix {_dp['hot_pixel_frac']*100:.2f}%, CTI {_dp['cti']:.1e}, DSNU {_dp['dsnu_e']:.1f} e-")
# else: leave ARISE_DET_* unset -> detector.py uses the flat CIS120 constants (identity)

# occlusion radiative fidelity (opt-in): the scorer's lit/visible tests use finite-Sun SOFT SHADOWS
# (penumbra) and PARTIAL (fractional) visibility instead of a hard binary ray at the facet centroid;
# occlusion_bvh.py also exposes self-illumination (view factors) + watertight intersection (Rozitis &
# Green 2011 ATPM; Woop 2013).  OFF -> the scorer's gated hooks fall back to the exact binary tests.
if os.getenv("ARISE_OCCLUSION_HIFI", "0") not in ("0", "", "false", "False"):
    print("[shape] OCCLUSION HI-FI ON: soft-shadow illumination fraction (finite Sun) + partial-visibility "
          "coverage; self-illumination (view factors) + watertight ray/triangle available (ATPM / Woop)")
# else: ARISE_OCCLUSION_HIFI unset -> occlusion_bvh gated hooks return the exact binary lit/visible (identity)

# photogrammetry / SPC (opt-in): the reconstruction metric uses stereophotoclinometry -- texture-limited
# (Cramer-Rao) subpixel matching, the full multi-ray triangulation covariance (bundle adjustment), and
# photoclinometry / shape-from-shading that recovers mono / low-parallax facets (Gaskell/Palmer 2024
# OSIRIS-REx; Debella-Gilo 2011; Horn/Kirk).  OFF -> reconstruct() uses the legacy best-pair triangulation.
if os.getenv("ARISE_PHOTOGRAM_HIFI", "0") not in ("0", "", "false", "False"):
    print("[recon] PHOTOGRAM HI-FI ON: stereophotoclinometry (multi-ray covariance + photoclinometry SFS); "
          "mono/low-parallax facets recovered by shape-from-shading (SPC ~ pixel-scale, OSIRIS-REx class)")
# else: ARISE_PHOTOGRAM_HIFI unset -> reconstruct() uses the legacy best-pair triangulation (identity)

# navigation (opt-in): the relative-nav covariance uses a Schmidt-Kalman CONSIDER filter + dynamic
# model compensation -- the un-estimated systematic biases (range/bearing, chief ephemeris common-mode,
# GM) that do NOT average down set the honest achievable nav floor a plain EKF misses (Tapley/Schutz/Born;
# OSIRIS-REx opnav).  OFF -> run_ekf() is the optimistic EKF (identity-when-absent).
if os.getenv("ARISE_NAV_HIFI", "0") not in ("0", "", "false", "False"):
    print("[nav] NAV HI-FI ON: Schmidt-Kalman consider covariance + DMC -- absolute nav floored by the chief "
          "ephemeris common-mode (~15 m), relative/deputy-deputy baseline floored by measurement biases (~m); "
          "the optimistic sub-meter EKF is the un-considered lower bound")
# else: ARISE_NAV_HIFI unset -> run_ekf() returns the optimistic EKF covariance (identity)

# Sun sensor (opt-in): sets the COARSE / SAFE-MODE Sun-vector knowledge (SAFEHOLD after an ATT/
# star-tracker fault) and the Sun-keep-out margin for the NST.  It does NOT touch the science
# pointing budget above -- the Sun sensor is the coarse sensor of last resort, not the imager ref.
if os.getenv("ARISE_SUNSENSOR_HIFI", "0") not in ("0", "", "false", "False"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
    import sun_sensor as _SS, hardware as _HWs
    _ss = _SS.active()
    _ss_deg = _ss.pointing_sigma_deg(); _fov = _ss.fov_deg(); _nh = _ss.n_heads_for_sphere()
    os.environ["ARISE_SAFEHOLD_SUN_SIGMA_DEG"] = f"{_ss_deg:.4f}"
    _excl = min(_HWs.STAR_TRACKER.sun_earth_excl_deg)     # NST Sun/Earth keep-out (tighter of the two)
    print(f"[safe-mode] SUN-SENSOR HI-FI ON: coarse Sun-vector knowledge {_ss_deg:.3f} deg "
          f"({_ss.pointing_sigma_arcmin():.1f}'), FOV +/-{_fov:.0f} deg, {_nh} heads tile 4pi "
          f"({_ss.name[:30]})")
    print(f"[safe-mode] Sun-keep-out: {_ss_deg:.3f} deg knowledge << {_excl:.0f} deg NST exclusion "
          f"-> {_excl/max(_ss_deg,1e-6):.0f}x margin to protect tracker/payload")
else:
    os.environ["ARISE_SAFEHOLD_SUN_SIGMA_DEG"] = "0.5"   # generic coarse-sun-sensor safe-mode default [deg]
# PNT timing (opt-in): CSAC + two-way time transfer physics for the three formation links
# (chief<->deputy, deputy<->deputy relative sync; chief<->GNSS absolute) via pnt_timing.py
# (Teng 2025 + Pham 2025 + SA.45s datasheet).  Derives the fleet-sync + camera-frame clock sigmas.
# OFF -> the exact flat 0.3 us / 0.1 us budget (identity-when-absent).
if os.getenv("ARISE_TIMING_HIFI", "0") not in ("0", "", "false", "False"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
    import pnt_timing as _PNT
    _tb = _PNT.fleet_timing_budget(n_deputies=8)
    os.environ["ARISE_CLOCK_SIGMA_FLEET_S"] = f"{_tb['fleet_sync_s']:.3e}"
    os.environ["ARISE_CLOCK_SIGMA_INTRA_S"] = f"{_tb['intra_frame_s']:.3e}"
    print(f"[pnt] TIMING HI-FI ON: CSAC+TWTT fleet sync {_tb['fleet_sync_s']*1e9:.2f} ns "
          f"(chief-deputy {_tb['chief_deputy_s']*1e9:.2f}, deputy-deputy {_tb['deputy_deputy_s']*1e9:.2f} ns), "
          f"camera-frame {_tb['intra_frame_s']*1e9:.2f} ns, GNSS absolute {_tb['absolute_s']*1e9:.0f} ns "
          f"(vs flat 300/100 ns; scene motion {_tb['scene_motion_fleet_um']:.1f} um << 0.15 m GSD)")
else:
    os.environ["ARISE_CLOCK_SIGMA_FLEET_S"]  = "3e-7"   # 0.3 us inter-deputy time sync (flat; identity)
    os.environ["ARISE_CLOCK_SIGMA_INTRA_S"]  = "1e-7"   # 0.1 us camera frame clock (flat; identity)
print(f"[nav-uq] knowledge budget : laser-ISL suite -> rel-nav {os.environ['ARISE_RELNAV_SIGMA_M']} m, "
      f"bearing {os.environ['ARISE_BEARING_SIGMA_ARCSEC']} arcsec, clock fleet "
      f"{float(os.environ['ARISE_CLOCK_SIGMA_FLEET_S'])*1e6:.1f} us")

# ===== Strobe operational realism (item #5) ===================================================
# The battery recharges the strobe at ~0.1 Hz (<< the ~500 Hz burst), so a PRE-CHARGED capacitor
# bank fires the close-approach burst and the battery refills it over the long approach (~2 min).
# The bank size caps the dark-side pulses per deputy per pass; the scorer fires the strobe only on
# the closest STROBE_MAX_PULSES frames.  Eclipse: Apophis is sunlit throughout the flyby (it clears
# Earth's umbra by ~24,900 km -> the naked-eye-visible 2029 pass), so the sunlit baseline is intact.
os.environ["ARISE_STROBE_MAX_PULSES"] = "12"
print(f"[strobe] cap-bank burst  : <= {os.environ['ARISE_STROBE_MAX_PULSES']} pulses/deputy/pass "
      f"(~1.7 kg HV cap bank, ~16 K/burst thermal spike, ~2 min recharge) | eclipse: none (sunlit)")

# ===== Chief-deputy SEPARATION interlink (deployment dV seed + SE(3)->MRP_FB handoff) ==========
# If the separation stage has been run (../chief deputy seperation/scripts/run_separation.py, which
# runs the RAFTI SE(3) separation then separation_to_guidance.py + make_separation_figures.py), feed
# its TWO products into the pipeline so the whole mission starts from the REAL deployment:
#   * separation_seed.npz  -> ARISE_SEPARATION_SEED_NPZ : each deputy deploys from the post-RAFTI
#       state (port offset + separation Δv) instead of an idealized impulse  [guidance stage].
#   * handoff_summary.json -> ARISE_HANDOFF_SUMMARY_JSON: each deputy's SE(3) pH -> MRP_FB handoff
#       residual (pointing + rate) seeds the 6-DOF coast, so MRP_FB ACQUIRES from the separation
#       hand-off rather than from a fictitious perfectly-aligned start            [6-DOF stage].
# Both are env-gated + identity when absent (idealized deploy / aligned MRP_FB start).
SEP_SEED = os.path.join(DESIGN, "data", "separation_seed.npz")
SEP_HANDOFF = os.path.join(DESIGN, "data", "handoff_summary.json")
if os.path.exists(SEP_SEED):
    os.environ["ARISE_SEPARATION_SEED_NPZ"] = SEP_SEED
    print(f"[separation] deploy seed : {os.path.basename(SEP_SEED)}  (deploy from post-RAFTI state; deployment dV in sync)")
else:
    print("[separation] no deploy seed -> idealized impulsive deploy (run run_separation.py to add it)")
if os.path.exists(SEP_HANDOFF):
    os.environ["ARISE_HANDOFF_SUMMARY_JSON"] = SEP_HANDOFF
    print(f"[separation] handoff seed: {os.path.basename(SEP_HANDOFF)}  (6-DOF starts at SE(3) pH -> MRP_FB handoff)")
else:
    print("[separation] no handoff summary -> 6-DOF MRP_FB starts from aligned state (run run_separation.py to add it)")

def _verify_interlink():
    """Startup self-check of the deployment interlink BEFORE the ~1.5 h run: confirm the seed and
    handoff are present, structurally valid, finite, and describe the SAME number of deputies (so a
    stale or half-written separation product is caught here, not 80 min in).  WARN-only: never
    aborts (a missing/old interlink simply degrades to the documented identity behaviour)."""
    if not (os.path.exists(SEP_SEED) or os.path.exists(SEP_HANDOFF)):
        print("[interlink] none present -> idealized deploy + aligned MRP_FB (separation not run yet).")
        return
    probe = (
        "import json,os,numpy as np\n"
        f"seed=r'''{SEP_SEED}'''; ho=r'''{SEP_HANDOFF}'''\n"
        "o={}\n"
        "try:\n"
        " if os.path.exists(seed):\n"
        "  d=np.load(seed); o['seed_keys']=list(d.files)\n"
        "  o['seed_n']=int(np.asarray(d['dr_lvlh_m']).shape[0]) if 'dr_lvlh_m' in d.files else None\n"
        "  o['seed_dv_sigma']=float(d['dv_sigma_mps']) if 'dv_sigma_mps' in d.files else None\n"
        "  o['seed_finite']=bool(np.isfinite(np.asarray(d['dr_lvlh_m'],float)).all() and np.isfinite(np.asarray(d['dv_lvlh_mps'],float)).all()) if ('dr_lvlh_m' in d.files and 'dv_lvlh_mps' in d.files) else None\n"
        "  o['seed_dv_max']=float(np.max(np.linalg.norm(np.asarray(d['dv_lvlh_mps'],float),axis=1))) if 'dv_lvlh_mps' in d.files else None\n"
        "except Exception as e: o['seed_err']=str(e)\n"
        "try:\n"
        " if os.path.exists(ho):\n"
        "  j=json.load(open(ho)); pd=j.get('per_deputy',[]); o['ho_n']=len(pd)\n"
        "  o['ho_settle_all']=all(p.get('settle_ok') for p in pd)\n"
        "  o['ho_resid_bore_max']=max((float(p.get('resid_boresight_deg',0) or 0) for p in pd),default=0.0)\n"
        "  o['ho_resid_w_max']=max((float(p.get('resid_omega_degps',0) or 0) for p in pd),default=0.0)\n"
        "  o['ho_finite']=all(np.isfinite([float(p.get('resid_boresight_deg',0) or 0),float(p.get('resid_omega_degps',0) or 0),float(p.get('t_ho_after_release_s',0) or 0)]).all() for p in pd)\n"
        "except Exception as e: o['ho_err']=str(e)\n"
        "print(json.dumps(o))\n")
    try:
        r = subprocess.run([PY, "-c", probe], capture_output=True, text=True)
        o = json.loads((r.stdout or "{}").strip().splitlines()[-1]) if r.stdout.strip() else {}
    except Exception as e:
        print(f"[interlink] self-check could not run ({e}); proceeding (env hooks are identity-safe).")
        return
    warns = []
    if o.get("seed_err"): warns.append(f"seed unreadable: {o['seed_err']}")
    if o.get("ho_err"):   warns.append(f"handoff unreadable: {o['ho_err']}")
    if o.get("seed_finite") is False: warns.append("seed has non-finite dr/dv")
    if o.get("ho_finite") is False:   warns.append("handoff has non-finite residuals")
    sn, hn = o.get("seed_n"), o.get("ho_n")
    if sn is not None and hn is not None and sn != hn:
        warns.append(f"deputy-count MISMATCH: seed={sn} vs handoff={hn} (re-run run_separation.py to re-sync)")
    if o.get("ho_settle_all") is False:
        warns.append("not all deputies reported settle_ok=true at handoff (MRP_FB still seeded with end-state residual)")
    print(f"[interlink] seed: n={sn} dv_sigma={o.get('seed_dv_sigma')} dv_max={o.get('seed_dv_max')} m/s | "
          f"handoff: n={hn} max_resid(point={o.get('ho_resid_bore_max')} deg, rate={o.get('ho_resid_w_max')} deg/s)")
    if warns:
        for w in warns:
            print(f"[interlink][WARN] {w}")
        print("[interlink] continuing (hooks degrade safely); re-run run_separation.py if the dV must be in sync.")
    else:
        print("[interlink] OK -> deployment dV + SE(3)->MRP_FB handoff present, consistent, finite.")

_verify_interlink()

# ====== run settings (defaults) ==========================================================
M = "32"          # orientation sub-draws (phase marginalisation)
NFRAME = "7"      # frames scored per deputy near closest approach

chk = subprocess.run([PY, "-c", "import Basilisk"], capture_output=True, text=True)
print(f"[env] worker interpreter : {PY}")
print(f"[env] Basilisk import    : {'OK' if chk.returncode == 0 else 'FAILED'}")
if chk.returncode != 0:
    print("  -> The high-fidelity mission needs Basilisk. Edit PY311 / DIST3 at the top of this file")
    print("     to your Basilisk Python 3.11 and its dist3 path. (Without it, only the report figures")
    print("     can regenerate, from whatever products are already on disk.)")
    print("     detail:", (chk.stderr or "").strip().splitlines()[-1] if chk.stderr else "")

def _preflight():
    """Pre-flight self-check: compile every script this run depends on, so a half-synced or
    truncated edit (e.g. mid OneDrive sync) is caught HERE with a clear message instead of
    failing deep inside the ~1.5 h Basilisk run.  Covers all of items #1-#5 + the figure maker."""
    crit = [MC, FIGS,
            os.path.join(HERE, "guidance_v2.py"),
            os.path.join(HERE, "modules", "photogram.py"),     # item #2 knowledge + item #3 MTF
            os.path.join(HERE, "modules", "detector.py"),      # item #3 MTF + stray light
            os.path.join(HERE, "modules", "strobe.py"),        # item #5 cap-bank
            NAVSCR,    # item #2 nav export (per-deputy EKF)
            GUID,      # guidance: deploys from the separation seed (deployment dV)
            SIXDOF]    # 6-DOF MRP_FB coast: seeded at the SE(3)->MRP_FB handoff residual
    bad = []
    for f in crit:
        if not os.path.exists(f):
            bad.append((f, "MISSING")); continue
        r = subprocess.run([PY, "-m", "py_compile", f], capture_output=True, text=True)
        if r.returncode != 0:
            bad.append((f, (r.stderr or "").strip().splitlines()[-1] if r.stderr else "compile error"))
    if bad:
        print("\n[PREFLIGHT] ABORT - these scripts did not compile (likely a half-synced/truncated edit):")
        for f, e in bad:
            print(f"    {os.path.relpath(f, DESIGN)}  ->  {e}")
        print("  Re-save / force a OneDrive sync of those files, then re-run this script.")
        sys.exit(1)
    print(f"[preflight] {len(crit)} core scripts compiled clean -> items #1-#5 + figures all wired in & synced")

def run(title, prog, args):
    print("\n" + "=" * 84 + f"\n{title}\n" + " ".join([os.path.basename(prog)] + args) + "\n" + "=" * 84, flush=True)
    t0 = time.time()
    rc = subprocess.run([PY, prog] + args, cwd=HERE, env=os.environ).returncode
    print(f"[done] rc={rc}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return rc

_preflight()   # <-- abort early if any dependency is truncated / not synced

# ====== STAGE B — FULL high-fidelity mission, FROM DEPLOYMENT (guidance+nav+6dof+score) =====
# (The old low-fidelity CASE-1 baseline scoring pass has been REMOVED: this is now a single hi-fi
#  pipeline.)  Re-run the single nominal draw FRESH (clear its checkpoint) so the whole pipeline
#  reflects the latest separation seed + handoff.  Guidance deploys from the separation seed
#  (deployment dV), the ~10.8 h 6-DOF coast is seeded at the SE(3)->MRP_FB handoff, and the full
#  V2 imaging chain scores it.  ~1.5 h.
# ====== HPC FULL-MC OVERRIDE (env-gated; identity when unset) ==============================
# ARISE_MC_FULL_ARGS lets the HPC launcher (HPC/run_hifi_mc.py) reuse this driver's ENTIRE
# validated setup (hi-fi knowledge budgets, CMA stations, 500 Hz export, strobe cap, separation
# seed + handoff, preflight) but run the FULL Monte Carlo instead of the single nominal draw,
# e.g. ARISE_MC_FULL_ARGS="--n 200 --m 32 --occ 1 --nframe 7 --workers 46 --stages
# guidance,nav,6dof,score".  Unset => the original nominal single-draw behaviour, unchanged.
MC_FULL_ARGS = os.environ.get("ARISE_MC_FULL_ARGS", "").split()

if chk.returncode == 0 and MC_FULL_ARGS:
    # FULL MC mode: checkpoints are NEVER cleared here (resume-safe across Slurm resubmissions;
    # delete data/mc_v2_draws.jsonl + data/mc/ yourself for a fresh campaign).
    rc = run("STAGE B  FULL HI-FI MONTE CARLO from deployment "
             "(deploy seed -> guidance -> nav -> 6-DOF -> V2 score; all hi-fi gates honored)",
             MC, MC_FULL_ARGS)
elif chk.returncode == 0:
    # clear the nominal checkpoint so the single draw re-runs FRESH with the latest seed + handoff
    # (only when we are actually going to run STAGE B; if Basilisk is missing we keep prior results
    #  so STAGE C can still regenerate figures from them).
    for _f in ("data/mc_v2_draws_nominal.jsonl", "data/mc_v2_results_nominal.json"):
        _p = os.path.join(HERE, _f)
        if os.path.exists(_p):
            try: os.remove(_p)
            except Exception: pass
    rc = run("STAGE B (1/2)  FULL HI-FI MISSION from deployment "
             "(deploy seed -> guidance -> nav -> 6-DOF MRP_FB coast @ handoff -> V2 score; occ ON, Hapke, 24deg Sun)",
             MC, ["--nominal", "--n", "1", "--m", M, "--occ", "1", "--nframe", NFRAME,
                  "--stages", "guidance,nav,6dof,score"])
else:
    rc = 1
    print("\n[skip] STAGE B skipped (Basilisk not importable) -> only report figures will regenerate (prior results kept).")

# ====== HPC FULL-MC mode: manifest + aggregate echo, then exit (nominal record/figures N/A) ==
if MC_FULL_ARGS:
    try:
        _mf = dict(when=time.strftime("%Y-%m-%d %H:%M:%S"), rc=int(rc), python=PY,
                   args=" ".join(MC_FULL_ARGS),
                   gates={k: os.environ.get(k) for k in (
                       "ARISE_STARTRACKER_HIFI", "ARISE_FSM_HIFI", "ARISE_RW_HIFI", "ARISE_ISL_HIFI",
                       "ARISE_SUNSENSOR_HIFI", "ARISE_IMU_HIFI", "ARISE_THRUSTER_HIFI",
                       "ARISE_TIMING_HIFI", "ARISE_DETECTOR_HIFI", "ARISE_OCCLUSION_HIFI",
                       "ARISE_PHOTOGRAM_HIFI", "ARISE_NAV_HIFI", "ARISE_RATEAID_EVENTCAM",
                       "ARISE_ST_PRESET")},
                   env={k: os.environ.get(k) for k in (
                       "ARISE_SEPARATION_SEED_NPZ", "ARISE_HANDOFF_SUMMARY_JSON",
                       "ARISE_DEPUTY_TARGETS_NPZ", "ARISE_EXPORT_FINE_DT", "ARISE_EXPORT_TAIL_S",
                       "ARISE_RELNAV_SIGMA_M", "ARISE_BEARING_SIGMA_ARCSEC", "ARISE_GYRO_SIGMA_RADPS",
                       "ARISE_CLOCK_SIGMA_FLEET_S", "ARISE_CLOCK_SIGMA_INTRA_S",
                       "ARISE_STROBE_MAX_PULSES", "ARISE_APOPHIS_BSP")})
        json.dump(_mf, open(os.path.join(HERE, "data", "mc_v2_run_manifest.json"), "w"), indent=2)
        print("[record] saved data/mc_v2_run_manifest.json (full-MC mode)")
    except Exception as _e:
        print(f"[record] MC manifest write skipped: {_e}")
    _res = os.path.join(HERE, "data", "mc_v2_results.json")
    if os.path.exists(_res):
        try:
            _s = json.load(open(_res)).get("summary", {})
            _wb = _s.get("whole_body", {})
            print("\nMC V2 aggregate so far: whole-body mean "
                  f"{_wb.get('mean', float('nan')):.1f}%  (full bands: data/mc_v2_results.json)")
        except Exception as _e:
            print(f"(aggregate read skipped: {_e})")
    print("\nDone (full-MC mode).")
    sys.exit(0 if rc == 0 else 1)

# ====== Save ALL run inputs alongside the per-draw tree (for plot recreation) ===============
# MC_v2_run.py already writes the full per-draw product tree to data/mc/draw_0000/ (trajectories /
# attitude / control / nav / camera / coverage / apophis / anim).  Here we additionally snapshot the
# deployment interlink (seed + handoff + station targets) and write a run_manifest.json capturing
# every setting, so the run is fully self-describing and any figure can be recreated from
# data/mc/draw_0000/ alone.
def _save_run_record(stage_b_rc):
    import shutil
    draw = os.path.join(HERE, "data", "mc", "draw_0000")
    if not os.path.isdir(draw):
        print("[record] draw_0000 not found -> nothing to snapshot (STAGE B did not produce a draw).")
        return
    snap = os.path.join(draw, "_run_inputs"); os.makedirs(snap, exist_ok=True)
    for src in (SEP_SEED, SEP_HANDOFF, os.path.join(HERE, "refly_targets_cma.npz")):
        try:
            if os.path.exists(src): shutil.copy2(src, os.path.join(snap, os.path.basename(src)))
        except Exception as e:
            print(f"[record] copy {os.path.basename(src)} skipped: {e}")
    manifest = dict(
        when=time.strftime("%Y-%m-%d %H:%M:%S"), stage_b_rc=int(stage_b_rc), python=PY,
        M=M, NFRAME=NFRAME, stages="guidance,nav,6dof,score",
        env={k: os.environ.get(k) for k in (
            "ARISE_SEPARATION_SEED_NPZ", "ARISE_HANDOFF_SUMMARY_JSON", "ARISE_DEPUTY_TARGETS_NPZ",
            "ARISE_EXPORT_FINE_DT", "ARISE_EXPORT_TAIL_S", "ARISE_RELNAV_SIGMA_M",
            "ARISE_BEARING_SIGMA_ARCSEC", "ARISE_CLOCK_SIGMA_FLEET_S", "ARISE_CLOCK_SIGMA_INTRA_S",
            "ARISE_STROBE_MAX_PULSES")},
        products="data/mc/draw_0000/{trajectories,attitude,control,nav,camera,coverage,apophis,anim}/")
    try:
        json.dump(manifest, open(os.path.join(draw, "run_manifest.json"), "w"), indent=2)
        print(f"[record] saved run_manifest.json + _run_inputs/ snapshot under data/mc/draw_0000/")
    except Exception as e:
        print(f"[record] manifest write skipped: {e}")
_save_run_record(rc)

# ====== summary (pure stdlib; no numpy needed in this parent) ==============================
print("\n" + "#" * 84 + "\nDESIGN D NOMINAL - high-fidelity pipeline summary\n" + "#" * 84)
hf = os.path.join(HERE, "data", "mc_v2_results_nominal.json")
if os.path.exists(hf):
    s = json.load(open(hf))["summary"]
    print(f"  HIGH fidelity (Basilisk; deploy seed, occ ON, real Sun + strobe + earthshine):")
    print(f"     <=1 m   : whole {s['whole_body']['mean']:5.1f}%  sunlit {s['sunlit']['mean']:5.1f}%  dark {s['dark']['mean']:5.1f}%   dV {s['fleet_dv']['mean']:.3f} m/s")
    t = s.get("stereo_0p15m")
    if t:
        print(f"     0.15 m  : whole {t['whole']['mean']:5.1f}%  sunlit {t['sunlit_mean']:5.1f}%  dark {t['dark_mean']:5.1f}%   (NAC)")
    sh = s.get("shape_visual_hull")
    if sh:
        print(f"     +shape  : {sh['mean']:5.1f}%  (visual hull)   [targets: sunlit-side>=90, <=1m>=85]")
    print("  Per-draw tree: data/mc/draw_0000/  (facet_stereo.npz -> per-facet 'category': 0=0.15m 1=<=1m 2=shape 3=none)")
else:
    print("  (no mc_v2_results_nominal.json yet -> STAGE B was skipped or did not complete)")

# SE(3) pH -> MRP_FB handoff (the deployment->coast handoff) — echo what fed the 6-DOF stage.
if os.path.exists(SEP_HANDOFF):
    try:
        hj = json.load(open(SEP_HANDOFF)); pdp = hj.get("per_deputy", [])
        print(f"\n  SE(3) pH -> MRP_FB handoff (fed to the 6-DOF coast; thr |w|<{hj.get('threshold_omega_degps')} deg/s, "
              f"boresight<{hj.get('threshold_boresight_deg')} deg):")
        print(f"    {'dep':>3} {'t_ho after release[s]':>21} {'resid pt[deg]':>13} {'resid w[deg/s]':>15} {'MRP_FB coast[h]':>16}")
        for p in pdp:
            print(f"    {p.get('dep'):>3} {p.get('t_ho_after_release_s', float('nan')):>21.1f} "
                  f"{p.get('resid_boresight_deg', float('nan')):>13.3f} {p.get('resid_omega_degps', float('nan')):>15.3f} "
                  f"{p.get('coast_remaining_h', float('nan')):>16.2f}")
    except Exception as e:
        print(f"  [handoff] summary read skipped: {e}")

# ====== STAGE C — REPORT FIGURES + STAT TABLES (one shot) ================================
# Pure numpy+matplotlib; reads the STAGE-B products written above and writes all report figures +
# report_tables.txt / report_values.json / report_tables.tex into outputs/.  Runs even if STAGE B
# was skipped (it regenerates from whatever products are on disk; the baseline overlay is simply
# skipped when no _nom_nominal_baseline.jsonl exists, since the low-fidelity run is no longer part
# of this pipeline).
if os.path.exists(FIGS):
    run("STAGE C (2/2)  REPORT FIGURES + STAT TABLES (coverage / nav EKF / guidance / attitude / handoff / FOV cones)",
        FIGS, [])
    figdir = os.path.join(HERE, "outputs", "report_figs")
    print(f"  Report figures : {figdir}  (F1..F20 PNG)")
    print(f"  Report tables  : outputs/report_tables.txt  +  report_values.json  +  report_tables.tex")
else:
    print("\n[skip] make_report_figures.py not found -> report figures/tables not generated.")

print("\nDone.")
