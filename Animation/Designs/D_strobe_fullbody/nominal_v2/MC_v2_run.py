#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MC_v2_run.py  -  ARISE Monte Carlo **V2** orchestrator (mission-grade)
=====================================================================
Drives the full V2 pipeline per draw and propagates the published Apophis
uncertainties into flight-grade distributions of fleet dV, stereo coverage, and
photogrammetric reconstruction accuracy -- scoring every draw with the REAL V2
imaging chain (CIS120 detector + strobe + occlusion + Hapke BRDF + photogrammetry)
on the radar shape, under a TRUE NPA-tumbling attitude.

Per draw k (embarrassingly parallel; one draw per worker):
  1. GUIDANCE V2 (Basilisk): full translational + 6-DOF attitude/GNC, Apophis state
     perturbed by a Set-III covariance draw -> <draw>/{trajectories,attitude,control}/.
     (Offline --skip-basilisk: a self-contained synthetic fleet exercises the whole
      scoring + data-tree + aggregation path with no Basilisk.)
  2. SAMPLE the orientation (apophis_tumbler), photometry (brdf), timing (timing).
  3. SCORE on the flown tracks: per-frame camera (detector/strobe), occlusion, and a
     photogrammetric per-facet sigma_z -> whole/sunlit/dark coverage + recon fractions.
  4. NAV EKF per deputy; finite-burn realized dV (thruster).
  5. WRITE the full per-draw data tree (MC_V2_SPEC.md s3) + checkpoint; RESUME-safe.

Aggregate: coverage P5/P50/P95 + bootstrap CIs, dV bands vs cap, per-facet
reconstruction-probability map, recon-accuracy CDF, Wilks tolerance note.

HARDWARE honesty: Basilisk propagation+corrector are single-threaded C++ w/ live
SPICE -- NO GPU path.  Speedup = parallel CPU draws (ProcessPoolExecutor) or a
cluster job array.  V1 is never read or written.
"""
from __future__ import annotations
import os, sys, json, time, glob, argparse, subprocess, datetime, math
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = os.path.join(HERE, "modules")
DATA = os.path.join(HERE, "data")
MCDIR = os.path.join(DATA, "mc")
GUIDANCE_V2 = os.path.join(HERE, "guidance_v2.py")
DESIGN = os.path.dirname(HERE)                                  # .../D_strobe_fullbody
SCRIPTS = os.path.join(DESIGN, "scripts")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(DESIGN)))  # .../ARISE Arch
NAV_SCRIPT = os.path.join(SCRIPTS, "navigation", "run_nav_on_new_traj.py")   # mission-proven EKF nav
SIXDOF_SCRIPT = os.path.join(SCRIPTS, "sim", "build_6dof_anim_data.py")      # mission-proven Basilisk 6-DOF
MESH_NPZ = os.path.join(DATA, "radar_mesh.npz")
GEOM_NPZ = os.path.join(DATA, "encounter_geometry.npz")     # optional (A2); else default Sun
COV_NPZ = os.path.join(DATA, "apophis_set3_cov.npz")        # optional (A3); else diagonal
sys.path.insert(0, MODULES)

import hardware as HW
import brdf as BRDF
import detector as DET
import strobe as STR
import occlusion_bvh as OCC
import photogram as PG
import apophis_tumbler as TUM
import thruster as THR
import timing as TIM
import nav_stage as NAV

# Star-tracker + reaction-wheel hi-fi (opt-in): source the nav pointing budget for a STANDALONE MC
# run (matches run_nominal_and_hifi.py).  Skipped when the bearing is inherited from a parent
# (run_nominal already RSS'd both), so nothing is double-counted.
if not os.getenv("ARISE_BEARING_SIGMA_ARCSEC"):
    _st_on = os.getenv("ARISE_STARTRACKER_HIFI", "0") not in ("0", "", "false", "False")
    _rw_on = os.getenv("ARISE_RW_HIFI", "0") not in ("0", "", "false", "False")
    _b = 5.0
    if _st_on:
        import star_tracker as _STK
        if os.getenv("ARISE_FSM_HIFI", "0") not in ("0", "", "false", "False"):
            import fast_steering_mirror as _FSM
            _b = _STK.active().pointing_sigma_rad(_FSM.np.radians(_FSM.FSM.effective_startracker_rate_deg_s())) / _STK.ARCSEC
        else:
            _b = _STK.active().midslew_pointing_sigma_rad() / _STK.ARCSEC    # mid-slew imaging (no FSM)
    if _rw_on:
        import reaction_wheels as _RWK
        _b = (_b ** 2 + _RWK.RWA.operating_jitter_arcsec(NAV.np.radians(NAV.HW.STAR_TRACKER.max_slew_deg_s)) ** 2) ** 0.5
    if _st_on or _rw_on:
        os.environ["ARISE_BEARING_SIGMA_ARCSEC"] = f"{_b:.2f}"
        print(f"[nav-uq] SENSOR HI-FI: pointing budget = {_b:.2f} arcsec (ST mid-slew{' + RW microvib' if _rw_on else ''})")

V_FLYBY = 8760.0
MAX_RANGE_M = HW.GSD_LIMIT_M / HW.IFOV_RAD                  # 5 km (0.15 m GSD -> 1 m at 5 km)
DEP_STATIONS_KM = np.array([                                # b-plane stations (LVLH, ~Design B fleet)
    [-1.81, -2.04, -1.06], [-1.43, 0.79, 2.32], [-1.56, 1.78, 1.62], [-2.93, -0.73, -0.77],
    [1.55, -2.45, 0.68], [1.91, 1.96, -1.16], [1.09, -2.70, 0.39], [2.32, 1.85, -0.54]])

# baseline-to-CA elapsed (radar zero-epoch 2012-12-23 -> 2029-04-13 CA), hours
_Z = datetime.datetime(2012, 12, 23, 4, 14, 0)
_CA = datetime.datetime(2029, 4, 13, 20, 11, 21, 903405)
DT_HOURS = (_CA - _Z).total_seconds() / 3600.0


# ════════════════════════════════════════════════════════════════════════════
# sampling helpers
# ════════════════════════════════════════════════════════════════════════════
def sample_perturb(seed):
    """Apophis CA state perturbation [dx,dy,dz,dvx,dvy,dvz] (m, m/s) for the Tier-2 hook."""
    rng = np.random.default_rng(seed)
    if os.path.exists(COV_NPZ):
        try:
            return rng.multivariate_normal(np.zeros(6), np.load(COV_NPZ)["cov6"])
        except Exception:
            pass
    # anchored sol-220 B-plane magnitude (along-track dominant ~735 m, ~0.5 mm/s)
    return rng.normal(0, 1, 6) * np.array([500., 500., 500., 5e-4, 5e-4, 5e-4])

def default_sun_hat():
    """Apophis->Sun unit at CA: ecliptic longitude 24.2 deg (SPICE-confirmed real value).
    NOTE: was 203 deg (ANTI-solar) which inverts the sunlit/dark labels -- the documented V1
    bug.  This fallback now uses the real Sun so a missing/corrupt encounter_geometry.npz can
    no longer flip the hemispheres."""
    lon = math.radians(24.2); obl = math.radians(23.4)
    return np.array([math.cos(lon), math.sin(lon) * math.cos(obl), math.sin(lon) * math.sin(obl)])

def load_geometry():
    """Load the real CA Sun, but NEVER silently fall back into an inverted hemisphere.
    A corrupt/unreadable geometry file now RAISES (instead of swallowing the error and using
    a default Sun), and any loaded Sun more than 10 deg from the expected ~24 deg CA direction
    is rejected as anti-solar/inverted.  This is the guard that prevents the V1 203 deg label
    swap from ever recurring."""
    ref = default_sun_hat()
    if os.path.exists(GEOM_NPZ):
        try:
            sun = np.asarray(np.load(GEOM_NPZ)["sun_hat"], float)
            sun = sun / np.linalg.norm(sun)
        except Exception as e:
            raise RuntimeError(
                f"encounter_geometry.npz present but unreadable ({e}); refusing to run rather "
                "than silently fall back to a default Sun (hemisphere-inversion risk). "
                "Regenerate it (compute_encounter_geometry.py) or copy the readable "
                "uq_montecarlo_v2/data/encounter_geometry.npz.")
        ang = math.degrees(math.acos(float(np.clip(sun @ ref, -1.0, 1.0))))
        if ang > 10.0:
            raise RuntimeError(
                f"loaded Sun is {ang:.1f} deg from the expected ~24 deg CA Sun -- looks "
                "anti-solar/inverted; aborting to avoid swapped sunlit/dark labels.")
        return sun
    return ref

def tumbler_dcm(rng):
    """Sample an NPA tumbler and return (body->inertial DCM at CA, sampled-params dict)."""
    p = TUM.sample_tumbler(rng)
    _, dcms = TUM.integrate_window(p, np.array([0.0, 1.0]))
    R = dcms[0]
    return R, p, dict(pole_lon=p.pole_lon_deg, pole_lat=p.pole_lat_deg, P_rot_h=p.P_rot_h,
                   P_prec_h=p.P_prec_h, nutation_deg=p.nutation_deg,
                   phase_rot=p.phase_rot_deg, phase_prec=p.phase_prec_deg, I=p.I.tolist())


# ════════════════════════════════════════════════════════════════════════════
# fleet tracks: real (guidance NPZs) or a self-contained synthetic stub
# ════════════════════════════════════════════════════════════════════════════
def _triad(los, vel):
    z = los / (np.linalg.norm(los) + 1e-30)
    vp = vel - np.dot(vel, z) * z
    x = vp / (np.linalg.norm(vp) + 1e-30)
    if np.linalg.norm(x) < 1e-8:
        tmp = np.array([1., 0, 0]) if abs(z[0]) < .9 else np.array([0, 1., 0])
        x = np.cross(tmp, z); x /= np.linalg.norm(x)
    y = np.cross(z, x); x = np.cross(y, z)
    return np.column_stack((x / np.linalg.norm(x), y / np.linalg.norm(y), z))

def synth_fleet(traj_dir, window_s=2.0, n_time=61):
    """Write 8 synthetic flyby trajectory NPZs (Apophis at origin) so the full V2 scoring +
    emission tree can run with NO Basilisk.  Stations span the b-plane (1-3 km)."""
    os.makedirs(traj_dir, exist_ok=True)
    t = np.linspace(-window_s, window_s, n_time)
    vrel = np.array([0.0, V_FLYBY, 0.0])
    for i, st_km in enumerate(DEP_STATIONS_KM):
        st = st_km * 1.0e3
        station = np.array([st[0], 0.0, st[2]])             # b-plane offset (x,z); CA at t=0
        rel = station[None, :] + np.outer(t, vrel)          # deputy pos relative to Apophis
        rD = rel.copy(); rA = np.zeros_like(rD)
        vD = np.tile(vrel, (len(t), 1)); vA = np.zeros_like(vD)
        bRef = np.zeros((len(t), 3)); C_RN = np.zeros((len(t), 3, 3))
        for j in range(len(t)):
            los = (rA[j] - rD[j]); los /= (np.linalg.norm(los) + 1e-30); bRef[j] = los
            C_RN[j] = _triad(los, vD[j]).T
        dv_times = np.array([0.0, 1800.0, 3600.0])
        dv_list = np.array([[0.30 + 0.02 * i, -0.05, 0.03], [0.002, 0.001, 0.0], [0.001, 0.0, 0.0]])
        np.savez_compressed(os.path.join(traj_dir, f"apophis_rendezvous_full_D{i}.npz"),
                            t=t, rD_N=rD, vD_N=vD, rA_N=rA, vA_N=vA, C_RN=C_RN, bRef_N=bRef,
                            dv_times=dv_times, dv_list=dv_list,
                            tot_dv=np.array([float(np.sum(np.linalg.norm(dv_list, axis=1)))]))

def load_fleet(traj_glob):
    """Return list of per-deputy dicts: idx, t, rel[N,3] (deputy-Apophis), vrel[N,3], tot_dv, dv_list."""
    out = []
    for f in sorted(glob.glob(traj_glob)):
        d = np.load(f, allow_pickle=True)
        idx = int("".join(c for c in os.path.basename(f).split("_D")[-1] if c.isdigit()) or "0")
        rel = np.asarray(d["rD_N"], float) - np.asarray(d["rA_N"], float)
        vrel = np.asarray(d["vD_N"], float) - np.asarray(d["vA_N"], float)
        tot = float(np.asarray(d["tot_dv"]).ravel()[0]) if "tot_dv" in d.files else 0.0
        dvl = np.atleast_2d(np.asarray(d["dv_list"], float)) if "dv_list" in d.files else np.zeros((1, 3))
        out.append(dict(idx=idx, t=np.asarray(d["t"], float), rel=rel, vrel=vrel, tot_dv=tot, dv_list=dvl))
    return out


# ════════════════════════════════════════════════════════════════════════════
# SCORING — real V2 imaging chain on the tumbling radar mesh
# ════════════════════════════════════════════════════════════════════════════
def _snr_pulse_vec(Ne, bg_e=0.0):
    det = HW.CIS120; dark = det.dark_current_e_s() * 1e-5
    q = (det.full_well_e / 2 ** det.adc_bits) / math.sqrt(12.0)
    # bg_e: stray-light / veiling-glare pedestal [e-] (item #3) -> adds shot noise + PRNU on the bg
    var = Ne + dark + bg_e + det.read_noise_e**2 + det.dsnu_e**2 + q**2 + (det.prnu_frac * (Ne + bg_e))**2
    return Ne / np.sqrt(var)

BLOOM_BAND_DEG = 2.0   # item #3: dark facets within this band past the terminator are lost to limb bloom

# FIX: dark-side strobe sizing margin + smooth-detection logistic width (see score_draw).
DARK_SNR_TARGET = 45.0      # size the strobe to this SNR for the per-draw photometry (margin over the 30 floor)
PDET_WIDTH = 6.0            # logistic width [SNR] of the smooth detection weight around DARK_SNR_MIN
_erf_vec = np.vectorize(math.erf)

# CASE-2 highest-fidelity additions: NAC tier (0.15 m) + earthshine 2nd illuminator.
NAC_IFOV = 0.15 / 3000.0          # narrow camera reaches 0.15 m GSD at 3 km (focal 200 mm, 2x2 CIS120 mosaic)
NAC_RANGE_M = 3000.0              # range below which the NAC delivers <= 0.15 m
ESH_COADD = 50                    # earthshine co-add frames at <= 1 m (faint source)

# ── Capture feasibility (item #1) ───────────────────────────────────────────────────────────
# The asteroid sweeps past at V_FLYBY (~8.76 km/s); the line-of-sight rate is V/R = up to ~670
# deg/s at 750 m -- far beyond any reaction-wheel or fast-steering-mirror tracking.  The only
# realizable capture is an INERTIALLY PRE-POINTED, FOV-limited SNAPSHOT BURST at closest
# approach: aim where the body will be, and let it transit a wide field while a high-rate strobe-
# gated burst freezes each frame.  This gate restricts each deputy's scored frames to the ones
# the body is actually framed for, at the camera frame rate -- replacing the old "7 frames spread
# over the whole <=5 km window", which silently assumed perfect high-rate tracking.
CAM_FOV_FULL_DEG = 45.0     # custom wide-field sensor full FOV (contains the ~26 deg body + sweep at 750 m)
CAM_FRAME_RATE_HZ = 500.0   # burst frame rate (effectively capped by the trajectory sample rate when coarser)
CAM_BURST_MAX_FRAMES = 40   # hard cap on frames credited per deputy per pass
CAPTURE_GATE = True         # set False to reproduce the legacy idealized (perfect-tracking) frame selection

def capture_burst_indices(rel, t, fov_half_rad, frame_rate, max_frames, r_body_m, max_range_m):
    """Frame indices a non-tracking, inertially pre-pointed wide-field camera actually captures.
    Boresight is fixed at the closest-approach look direction; a frame counts only while the WHOLE
    body is inside the FOV (angular offset of the body centre from boresight + body half-angle <=
    FOV half-angle) and within range.  Frames are then thinned to the camera frame rate over the
    single contiguous in-FOV burst around CA, capped at max_frames."""
    rngc = np.linalg.norm(rel, axis=1)
    inr = rngc <= max_range_m
    if not inr.any():
        return np.array([], int)
    c = int(np.argmin(np.where(inr, rngc, np.inf)))            # CA index = pre-point time
    bore = -rel[c] / max(rngc[c], 1e-9)                        # boresight: body direction at CA (body at origin)
    los = -rel / np.clip(rngc, 1e-9, None)[:, None]
    theta = np.arccos(np.clip(los @ bore, -1.0, 1.0))         # body-centre offset from boresight
    body_half = np.arcsin(np.clip(r_body_m / np.clip(rngc, 1e-9, None), 0.0, 1.0))
    framed = inr & ((theta + body_half) <= fov_half_rad)
    if not framed.any():
        return np.array([c], int)                              # degenerate: at least the CA snapshot
    a = c
    while a - 1 >= 0 and framed[a - 1]:
        a -= 1
    b = c
    while b + 1 < len(framed) and framed[b + 1]:
        b += 1
    ta, tb = float(t[a]), float(t[b])
    if frame_rate and frame_rate > 0 and tb > ta:
        want = ta + np.arange(0.0, (tb - ta) + 1e-9, 1.0 / frame_rate)
        if len(want) > max_frames:
            want = np.linspace(ta, tb, max_frames)
        sel = np.clip(np.searchsorted(t, want), a, b)
    else:
        sel = np.array([c], int)
    return np.unique(np.concatenate([sel, [c]]))

def score_draw(fleet, R_body, sun_hat, hapke, mesh_npz=MESH_NPZ, occ=True, nframe=5, topup=True, model="hapke",
               earth_hat=None, esh_frac=0.0, nac_ifov=NAC_IFOV, nac_range_m=NAC_RANGE_M, esh_coadd=ESH_COADD,
               capture_gate=CAPTURE_GATE, fov_full_deg=CAM_FOV_FULL_DEG,
               frame_rate_hz=CAM_FRAME_RATE_HZ, burst_max=CAM_BURST_MAX_FRAMES,
               nav_pos_sigma=None, nav_point_sigma=None, clock_fleet_s=1e-5, clock_intra_s=1e-6,
               snr_floor_015=None):
    """Tiered stereo coverage on the tumbling mesh.  Returns the <=1 m product (whole/sunlit/dark,
    backward-compatible) PLUS the 0.15 m (NAC) tier, the visual-hull shape tier, and per-facet
    categories.  Illumination: Sun (science SNR) + strobe (dark side) + EARTHSHINE (Earth as a 2nd
    illuminator, dark-side fill at <=1 m).  Occlusion + self-shadow when occ=True (CASE-2)."""
    m = np.load(mesh_npz)
    V = m["vertices"] @ R_body.T; C = m["centroids"] @ R_body.T
    Nn = m["normals"] @ R_body.T; areas = np.asarray(m["areas"], float)
    F = len(C)
    r_body_m = float(np.max(np.linalg.norm(V, axis=1)))   # bounding radius for FOV-containment gate
    fov_half = math.radians(fov_full_deg) / 2.0
    mesh = OCC.MeshRT(V, m["faces"]) if occ else None
    lit_earth = (Nn @ earth_hat) > 0 if earth_hat is not None else np.zeros(F, bool)

    # illumination (sun) — self-shadowed if occ
    if occ:
        lit = OCC.lit_facets_gated(mesh, C, Nn, sun_hat)   # ARISE_OCCLUSION_HIFI: soft shadows; else binary
    else:
        lit = (Nn @ sun_hat) > 0

    # design operating points (built once)
    E_sun = DET.solar_irradiance(1.0)
    lit_des = DET.size_sunlit_aperture(model="hapke")
    ap_lit, t_exp, n_coadd = lit_des["aperture_m"], lit_des["t_exp_s"], lit_des["n_coadd"]
    dark_des = STR.dark_side_budget(target_snr=STR.DARK_SNR_MIN, range_m=HW.DESIGN_STANDOFF_M)
    # FIX(1): size the strobe to THIS draw's sampled photometry (+SNR margin) instead of the
    # default Hapke, so the dark-side gate is not a knife-edge that flips the whole body
    # 0<->100 between spin phases (root cause of the bimodal coverage).
    _e0 = math.radians(10.0)
    _L_def = BRDF.radiance_over_Esun(_e0, _e0, 0.0, model=model)
    _L_smp = BRDF.radiance_over_Esun(_e0, _e0, 0.0, model=model, params=hapke)
    # SNR is shot-limited (SNR ~ sqrt(Ne) ~ sqrt(E_pulse)), so to actually REACH the SNR target
    # the energy ratio must be SQUARED.  (The earlier linear factor only reached ~34, not 45.)
    # The L_def/L_smp factor stays LINEAR: it holds SNR fixed against this draw's albedo.
    E_pulse = dark_des["E_pulse_opt_J"] * (_L_def / max(_L_smp, 1e-9)) * (DARK_SNR_TARGET / STR.DARK_SNR_MIN) ** 2
    ap_dark = HW.APERTURE_MAX_M
    radfun = np.vectorize(lambda i, e, a: BRDF.radiance_over_Esun(i, e, a, model=model, params=hapke))
    Opix = (HW.CIS120.pixel_m / HW.FOCAL_M) ** 2
    # system MTF at Nyquist (item #3): diffraction x pixel x 1-px strobe-frozen smear x aberration.
    # Degrades sub-pixel matching ( sigma_disp ~ 1/MTF ) and gates a tier out if features are
    # unresolved (MTF < MTF_RESOLVE_MIN, e.g. smear >= 2 px).  NAC keeps 1-px smear via a shorter
    # strobe-gated exposure (t_exp <= IFOV_nac * R / V ~ 4 us at 750 m) -> same MTF as the WAC.
    mtf_nyq = DET.system_mtf_nyquist(HW.FNUM_MIN, smear_px=1.0)[0]

    facet_views = [[] for _ in range(F)]
    best_snr = np.zeros(F)        # FIX(2): per-facet best detection SNR, for the smooth coverage weight
    best_snr_sci = np.zeros(F)    # science-grade (Sun/strobe only) SNR, for the 0.15 m tier
    limb = np.zeros(F, bool)      # appears on the silhouette (visual-hull shape tier)
    sat_lit = False               # item #3: did the sunlit limb actually saturate in any frame?
    cam_summ = []
    for dep in fleet:
        rel = dep["rel"]; rng_center = np.linalg.norm(rel, axis=1)
        if capture_gate:
            # realizable inertial-stare burst: only frames the body is actually framed for, at the camera rate
            pick = capture_burst_indices(rel, dep["t"], fov_half, frame_rate_hz, burst_max, r_body_m, MAX_RANGE_M)
            if len(pick) == 0:
                cam_summ.append(dict(idx=dep["idx"], n_frames=0, n_seen_max=0)); continue
        else:
            # legacy idealized selection (assumes perfect high-rate tracking across the whole window)
            usable = np.where(rng_center <= MAX_RANGE_M)[0]
            if len(usable) == 0:
                cam_summ.append(dict(idx=dep["idx"], n_frames=0, n_seen_max=0)); continue
            spread = usable[np.linspace(0, len(usable) - 1, max(1, min(nframe, len(usable)))).astype(int)]
            closest = usable[int(np.argmin(rng_center[usable]))]        # always score best-SNR frame
            pick = np.unique(np.concatenate([[closest], spread]))
        # item #5: the strobe is cap-bank-limited (battery recharge ~0.1 Hz << burst rate) -> only
        # the STROBE_MAX_PULSES frames CLOSEST to CA get a pulse; later frames image sunlit-only.
        _pk = np.asarray(pick)
        strobe_frames = set(_pk[np.argsort(rng_center[_pk])[:STR.STROBE_MAX_PULSES]].tolist())
        per_frame = []
        for k in pick:
            vp = rel[k]
            look = vp[None, :] - C; rng = np.linalg.norm(look, axis=1)
            look_hat = look / rng[:, None]
            cos_e = np.einsum('ij,ij->i', Nn, look_hat)
            front = cos_e > 0
            inrange = rng <= MAX_RANGE_M
            if occ:
                vis = OCC.visible_from_gated(mesh, C, Nn, vp)   # ARISE_OCCLUSION_HIFI: partial visibility; else binary
            else:
                vis = front
            cand = vis & inrange & (cos_e > 0)
            if not cand.any():
                per_frame.append((float(np.linalg.norm(vp)), 0, 0.0)); continue
            ci = np.clip(Nn @ sun_hat, 0, 1)
            alpha_sun = np.arccos(np.clip(np.einsum('j,ij->i', sun_hat, look_hat), -1, 1))
            i_r = np.arccos(np.clip(ci, -1, 1)); e_r = np.arccos(np.clip(cos_e, -1, 1))
            snr_sun = np.zeros(F); snr_str = np.zeros(F)
            litc = cand & lit
            if litc.any():                                   # full-Sun science SNR on lit facets
                Lf = radfun(i_r[litc], e_r[litc], alpha_sun[litc])
                Ne = DET.collected_electrons(Lf, E_sun, ap_lit, t_exp)
                snr_sun[litc] = DET.snr(Ne, t_exp, n_coadd)
            # cap-bank limit (item #5): a strobe pulse fires only on the closest STROBE_MAX_PULSES frames
            strobe_set = (cand if topup else (cand & ~lit)) if (int(k) in strobe_frames) else np.zeros(F, bool)
            if strobe_set.any():                             # topup: strobe lights every imaged facet, ~0 phase
                bh = STR.body_half_angle(float(np.linalg.norm(vp)))
                Omega = STR.beam_solid_angle(bh)
                Fsurf = E_pulse / (Omega * rng[strobe_set] ** 2) * 0.8
                Lf = radfun(e_r[strobe_set], e_r[strobe_set], np.zeros(strobe_set.sum()))
                Ne = Lf * Fsurf * (math.pi * (ap_dark / 2) ** 2) * Opix / DET.PHOTON_ENERGY_J
                Ne = Ne * HW.CIS120.qe_at_band * HW.CIS120.optical_eff
                # HDR / stray light (item #3): the sunlit limb, imaged at the OPEN dark aperture,
                # over-saturates and scatters a veiling-glare pedestal (PST x limb signal) onto the
                # dark-side pixels, cutting their contrast.  Bounded by the optics' baffle/PST.
                B_stray = 0.0
                if litc.any():
                    Ne_lit_dark = DET.collected_electrons(radfun(i_r[litc], e_r[litc], alpha_sun[litc]),
                                                          E_sun, ap_dark, t_exp)
                    B_stray = DET.stray_light_background_e(float(np.median(Ne_lit_dark)))
                    if np.any(Ne_lit_dark >= HW.CIS120.full_well_e):
                        sat_lit = True               # bright limb actually saturates -> blooming applies
                snr_str[strobe_set] = _snr_pulse_vec(Ne, bg_e=B_stray)
            snr_sci = np.maximum(snr_sun, snr_str)           # science-grade (Sun OR strobe)
            # EARTHSHINE: Earth as a faint 2nd illuminator (dark-side fill at <= 1 m, deeper co-add)
            snr_esh = np.zeros(F)
            if earth_hat is not None and esh_frac > 0.0:
                # Earthshine only FILLS GENUINE GAPS: facets the Sun AND strobe failed to bring to
                # the match floor.  (Gating on < SNR_LIT_BAND[0]=300 wrongly fired on already
                # strobe-lit dark facets at SNR~34 and inflated the dark-side number.)
                esh_set = cand & lit_earth & (snr_sci < PG.SNR_MATCH_MIN)
                if esh_set.any():
                    inc_e = np.arccos(np.clip(np.einsum('j,ij->i', earth_hat, Nn[esh_set]), 0, 1))
                    Lf = radfun(inc_e, e_r[esh_set], np.zeros(esh_set.sum()))
                    Ne = DET.collected_electrons(Lf, esh_frac * E_sun, ap_lit, t_exp)
                    snr_esh[esh_set] = DET.snr(Ne, t_exp, esh_coadd)
            snr = np.maximum(snr_sci, snr_esh)               # best available illumination
            # silhouette appearance (grazing emission within shape range) -> visual-hull tier.
            # Must be ACTUALLY VISIBLE from this viewpoint (vis): otherwise occluded / back-of-body
            # grazing facets were being credited to the shape tier.
            limb |= vis & (rng <= 20000.0) & (np.abs(cos_e) <= math.cos(math.radians(78.0)))
            # FIX(2): include any matchable view; smooth per-facet detection decides coverage.
            det_ok = cand & (snr >= PG.SNR_MATCH_MIN)
            seen = np.where(det_ok)[0]
            for fi in seen:
                facet_views[fi].append(dict(range_m=float(rng[fi]), snr=float(snr[fi]),
                                            snr_sci=float(snr_sci[fi]), look_hat=look_hat[fi].copy(),
                                            dep=int(dep["idx"])))   # deputy id: intra- vs cross-deputy stereo
                if snr[fi] > best_snr[fi]:
                    best_snr[fi] = snr[fi]
                if snr_sci[fi] > best_snr_sci[fi]:
                    best_snr_sci[fi] = snr_sci[fi]
            per_frame.append((float(np.linalg.norm(vp)), int(det_ok.sum()),
                              float(np.median(snr[det_ok])) if det_ok.any() else 0.0))
        cam_summ.append(dict(idx=dep["idx"], n_frames=len(pick),
                             n_seen_max=int(max((p[1] for p in per_frame), default=0)),
                             per_frame=per_frame))

    # ---- knowledge-error spec (item #2): cross-deputy stereo limited by rel-nav + clock; self-
    #      stereo near-truth (errors common-mode over the burst).  None -> truth geometry. ----
    nav_spec = None
    if nav_pos_sigma is not None:
        nav_spec = dict(pos_sigma=nav_pos_sigma, point_sigma=nav_point_sigma,
                        clock_fleet_s=clock_fleet_s, clock_intra_s=clock_intra_s, V=V_FLYBY)

    # ---- <= 1 m tier (wide camera, Sun + strobe + earthshine) : backward-compatible "whole" ----
    _inc_deg = np.degrees(np.arccos(np.clip(Nn @ sun_hat, -1.0, 1.0)))   # sun incidence (photoclinometry; ignored when gate off)
    rec = PG.reconstruct(facet_views, areas, HW.IFOV_RAD, nav=nav_spec, mtf=mtf_nyq, incidence_deg=_inc_deg)
    sigz = rec["sigma_z_m"]; stereo_mask = np.isfinite(sigz)
    Atot = areas.sum()
    # blooming (item #3): ONLY when the sunlit limb actually saturates (the fast-flyby's short
    # ~17 us exposures usually keep it sub-full-well, so this is typically empty), dark facets
    # within BLOOM_BAND_DEG past the terminator are lost to charge bloom / un-correctable glare.
    bloom_lost = ((~lit) & (Nn @ sun_hat > -math.sin(math.radians(BLOOM_BAND_DEG)))) if sat_lit \
        else np.zeros(F, bool)
    pdet = 0.5 * (1.0 + _erf_vec((best_snr - STR.DARK_SNR_MIN) / (math.sqrt(2.0) * PDET_WIDTH)))
    cov_w = stereo_mask.astype(float) * pdet
    cov_w[bloom_lost] = 0.0
    whole = 100.0 * (areas * cov_w).sum() / Atot
    lit_frac = 100.0 * (areas * cov_w * lit).sum() / max(areas[lit].sum(), 1e-9)
    dark_frac = 100.0 * (areas * cov_w * (~lit)).sum() / max(areas[~lit].sum(), 1e-9)

    # ---- 0.15 m tier (NAC): science-grade (Sun/strobe) views within the NAC range ----
    # snr_floor_015 (opt-in): raise the per-view science-SNR floor for the 0.15 m tier above the
    # default feature-match floor (SNR_MATCH_MIN=10). None -> legacy behaviour (floor 10 + soft
    # detection weight), so every previously reported number is unchanged. A value (e.g. 300) gives
    # a stricter, sunlit-science-grade coverage: both stereo views must clear the floor (enforced by
    # the fv015 pre-filter -- the robust gate, independent of reconstruct's internal floor) AND the
    # facet's best science view must clear it (hard detection weight instead of the soft erf).
    _smin015 = PG.SNR_MATCH_MIN if snr_floor_015 is None else float(snr_floor_015)
    fv015 = [[dict(range_m=v["range_m"], snr=v["snr_sci"], look_hat=v["look_hat"], dep=v["dep"])
              for v in fvs if v["range_m"] <= nac_range_m and v["snr_sci"] >= _smin015]
             for fvs in facet_views]
    rec0 = PG.reconstruct(fv015, areas, nac_ifov if nac_ifov else HW.IFOV_RAD, nav=nav_spec, mtf=mtf_nyq, incidence_deg=_inc_deg)
    stereo_015 = np.isfinite(rec0["sigma_z_m"])
    if snr_floor_015 is None:                       # legacy soft detection weight (unchanged numbers)
        pdet0 = 0.5 * (1.0 + _erf_vec((best_snr_sci - STR.DARK_SNR_MIN) / (math.sqrt(2.0) * PDET_WIDTH)))
    else:                                           # hard science-grade floor
        pdet0 = (best_snr_sci >= _smin015).astype(float)
    cov0 = stereo_015.astype(float) * pdet0
    cov0[bloom_lost] = 0.0   # item #3: terminator bloom band also lost to the 0.15 m tier
    whole_015 = 100.0 * (areas * cov0).sum() / Atot
    sun_015 = 100.0 * (areas * cov0 * lit).sum() / max(areas[lit].sum(), 1e-9)
    dark_015 = 100.0 * (areas * cov0 * (~lit)).sum() / max(areas[~lit].sum(), 1e-9)

    # ---- visual-hull shape tier (silhouette OR any stereo) ----
    shape_mask = stereo_mask | limb
    shape_pct = 100.0 * areas[shape_mask].sum() / Atot

    # ---- per-facet category: 0 = 0.15 m, 1 = <=1 m, 2 = shape, 3 = none ----
    category = np.full(F, 3, np.int8)
    category[shape_mask] = 2
    category[stereo_mask] = 1
    category[stereo_015 & (cov0 > 0.5)] = 0

    return dict(whole=whole, sunlit=lit_frac, dark=dark_frac,
                whole_015=whole_015, sunlit_015=sun_015, dark_015=dark_015, shape=shape_pct,
                mono=100.0 * rec["mono_frac"], recon_within={str(k): v for k, v in rec["area_frac_within"].items()},
                median_sigma_m=rec["median_sigma_m"], stereo_mask=stereo_mask.astype(np.uint8),
                stereo_015_mask=stereo_015.astype(np.uint8), category=category.astype(np.int8),
                sigma_z_m=sigz, n_views=rec["n_views"].astype(int), lit_mask=lit.astype(np.uint8),
                cam=cam_summ, areas=areas)


# ════════════════════════════════════════════════════════════════════════════
# per-draw driver
# ════════════════════════════════════════════════════════════════════════════
def run_draw(spec):
    k = spec["id"]; occ = spec["occ"]; nframe = spec["nframe"]; do_bsk = spec["do_basilisk"]
    draw_root = os.path.join(MCDIR, f"draw_{k:04d}")
    traj = os.path.join(draw_root, "trajectories"); att = os.path.join(draw_root, "attitude")
    ctrl = os.path.join(draw_root, "control"); navd = os.path.join(draw_root, "nav")
    camd = os.path.join(draw_root, "camera"); covd = os.path.join(draw_root, "coverage")
    apd = os.path.join(draw_root, "apophis")
    for dd in (traj, att, ctrl, navd, camd, covd, apd):
        os.makedirs(dd, exist_ok=True)
    t0 = time.time()
    try:
        # 1) guidance V2 (Basilisk) or self-contained synthetic fleet + attitude/control emission
        stage_status = {}
        stages = spec.get("stages", set())
        if do_bsk:
            env = dict(os.environ, MPLBACKEND="Agg", ARISE_OUT_DIR=draw_root,
                       ARISE_V2_ATT_BACKEND="kinematic",   # tested follower; authoritative 6-DOF = build_6dof stage
                       PYTHONUTF8="1", PYTHONIOENCODING="utf-8",   # so Unicode prints don't crash when stdout->file on Windows
                       ARISE_APOPHIS_STATE_PERTURB=",".join(f"{x:.6e}" for x in spec["perturb"]))
            with open(os.path.join(draw_root, "guidance_v2.log"), "w") as lg:
                r = subprocess.run([sys.executable, GUIDANCE_V2], cwd=HERE, env=env,
                                   stdout=lg, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                return {"id": k, "ok": False, "err": f"guidance_v2 rc={r.returncode} (see guidance_v2.log)"}
            # mission-proven nav + 6-DOF stages, redirected into THIS draw via the config.py override
            stage_env = dict(env, ARISE_CFG_DATA_ROOT=draw_root,
                             ARISE_CFG_TRAJ_DIR=traj,
                             ARISE_CFG_ANIM_DATA=os.path.join(draw_root, "anim"),
                             ARISE_TRAJ_GLOB=os.path.join(traj, "apophis_rendezvous_full_D*.npz"))
            for nm, scr in (("nav", NAV_SCRIPT), ("6dof", SIXDOF_SCRIPT)):
                if nm in stages:
                    with open(os.path.join(draw_root, f"{nm}_stage.log"), "w") as lg:
                        rr = subprocess.run([sys.executable, scr], cwd=REPO, env=stage_env,
                                            stdout=lg, stderr=subprocess.STDOUT)
                    stage_status[nm] = "ok" if rr.returncode == 0 else f"rc={rr.returncode} (see {nm}_stage.log)"
        else:
            synth_fleet(traj)
            import guidance_v2 as GV2
            GV2.ATT_BACKEND = "kinematic"
            GV2.emit_attitude_and_control(traj, att, ctrl, seed=k)

        fleet = load_fleet(os.path.join(traj, "apophis_rendezvous_full_D*.npz"))
        if not fleet:
            return {"id": k, "ok": False, "err": "no trajectories"}

        # 2) fleet dV (corrector + finite-burn realized)
        rng = np.random.default_rng(spec["seed"] * 7 + k)
        fleet_dv = sum(d["tot_dv"] for d in fleet)
        real_dv = 0.0; prop = 0.0
        for d in fleet:
            fb = THR.fleet_delta_v(list(d["dv_list"]), HW.XB6.wet_mass_kg, rng)
            real_dv += fb["dv_total_mps"]; prop += fb["propellant_kg"]

        # 3) score under M cheap NPA-orientation sub-draws (N x M design, as V1): the expensive
        #    Basilisk trajectory is reused; only the tumbler orientation + photometry resample.
        sun_hat = load_geometry()
        # earthshine geometry: Earth direction + Lambert-sphere flux from the asteroid's geocentric
        # position at closest approach (CASE-2 dark-side fill at <= 1 m).
        earth_hat = None; esh_frac = 0.0
        try:
            _d = np.load(sorted(glob.glob(os.path.join(traj, "apophis_rendezvous_full_D*.npz")))[0], allow_pickle=True)
            _rA = np.asarray(_d["rA_N"], float); _kca = int(np.argmin(np.linalg.norm(_d["rD_N"] - _rA, axis=1)))
            _re = -_rA[_kca]; _enorm = float(np.linalg.norm(_re))
            # guard: stub runs write rA_N=0 -> _enorm=0 raises no exception but yields NaN/inf.
            # Require a real geocentric position (km-scale, finite) before enabling earthshine.
            if np.isfinite(_enorm) and _enorm > 1.0e6:
                _ekm = _enorm / 1e3; earth_hat = _re / _enorm
                esh_frac = 0.30 * 0.5 * (6371.0 / _ekm) ** 2
        except Exception:
            pass
        # relative-nav KNOWLEDGE (item #2): the EKF covariance product written by the nav stage
        # (run_nav_on_new_traj.py -> _nav_stage/nav_uncertainty.npz).  Drives cross-deputy stereo
        # baseline error in the reconstruction; absent -> truth geometry.
        nav_pos = nav_pt = None; clk_f = 1e-5; clk_i = 1e-6
        try:
            _nu = np.load(os.path.join(draw_root, "_nav_stage", "nav_uncertainty.npz"))
            _di = np.asarray(_nu["dep_idx"], int); _n = int(_di.max()) + 1
            nav_pos = np.zeros(_n); nav_pt = np.zeros(_n)
            nav_pos[_di] = np.asarray(_nu["pos_sigma_m"], float)
            nav_pt[_di] = np.asarray(_nu["point_sigma_rad"], float)
            clk_f = float(_nu["clock_fleet_s"]); clk_i = float(_nu["clock_intra_s"])
            print(f"[nav-uq] knowledge-limited reconstruction: baseline sigma {np.round(nav_pos,3)} m, "
                  f"clock fleet {clk_f*1e6:.0f} us", flush=True)
        except Exception as _e:
            print(f"[nav-uq] no nav_uncertainty.npz -> TRUTH-geometry reconstruction ({_e})", flush=True)
        M = int(spec.get("M", 1)); topup = bool(spec.get("topup", True))
        covs = []; covs015 = []; oparams = []; cov_count = None; cat_rep = None
        sc_rep = tparams_rep = R_rep = None
        for mi in range(M):
            R_body, tparams, opar = tumbler_dcm(rng)
            hapke = BRDF.sample_hapke(rng)
            sc = score_draw(fleet, R_body, sun_hat, hapke, occ=occ, nframe=nframe, topup=topup,
                            earth_hat=earth_hat, esh_frac=esh_frac,
                            nav_pos_sigma=nav_pos, nav_point_sigma=nav_pt,
                            clock_fleet_s=clk_f, clock_intra_s=clk_i)
            covs.append([sc["whole"], sc["sunlit"], sc["dark"]]); oparams.append(opar)
            covs015.append([sc["whole_015"], sc["sunlit_015"], sc["dark_015"], sc["shape"]])
            cov_count = sc["stereo_mask"].astype(float) if cov_count is None else cov_count + sc["stereo_mask"]
            if mi == 0:
                sc_rep, tparams_rep, R_rep = sc, tparams, R_body; cat_rep = sc["category"]
        covs = np.array(covs); cov_mean = covs.mean(axis=0)
        covs015 = np.array(covs015); cov015_mean = covs015.mean(axis=0)
        tim = TIM.sample_timing(len(fleet), rng)

        # 4) nav EKF per deputy + write the per-draw data tree (representative orientation + cov-prob map)
        for d in fleet:
            nv = NAV.run_ekf(d["rel"], d["t"]); pe = NAV.pointing_error(nv)
            np.savez_compressed(os.path.join(navd, f"ekf_D{d['idx']}.npz"),
                                t=d["t"], pos_sigma_m=nv["pos_sigma_m"], vel_sigma_mps=nv["vel_sigma_mps"],
                                range_m=nv["range_m"], crossrange_sigma_m=nv["crossrange_sigma_m"],
                                pointing_sigma_rad=pe)
        for cs in sc_rep["cam"]:
            pf = np.array(cs.get("per_frame", []), float).reshape(-1, 3)
            np.savez_compressed(os.path.join(camd, f"cam_D{cs['idx']}.npz"),
                                range_center_m=pf[:, 0] if len(pf) else np.zeros(0),
                                n_seen=pf[:, 1] if len(pf) else np.zeros(0),
                                median_snr=pf[:, 2] if len(pf) else np.zeros(0))
        np.savez_compressed(os.path.join(covd, "facet_stereo.npz"),
                            sigma_z_m=sc_rep["sigma_z_m"], n_views=sc_rep["n_views"],
                            stereo_mask=sc_rep["stereo_mask"], lit_mask=sc_rep["lit_mask"],
                            areas=sc_rep["areas"], facet_prob=cov_count / M,
                            category=cat_rep, stereo_015_mask=sc_rep["stereo_015_mask"])
        tg = np.linspace(-2.0, 2.0, 41)
        om_tr, dcm_tr = TUM.integrate_window(tparams_rep, tg)
        np.savez_compressed(os.path.join(apd, "attitude_truth.npz"),
                            t=tg, omega_BN_truth=om_tr, R_series=dcm_tr,
                            R_body2inertial=R_rep, sun_hat=sun_hat,
                            params=np.array([json.dumps(o) for o in oparams], dtype=object))
        meta = dict(id=k, perturb=spec["perturb"], m_orient=M, topup=topup,
                    fleet_dv_corrector=fleet_dv, fleet_dv_realized=real_dv, propellant_kg=prop,
                    timing=dict(tca_offset_s=float(tim["tca_offset_s"])),
                    coverage_mean=dict(whole=float(cov_mean[0]), sunlit=float(cov_mean[1]), dark=float(cov_mean[2])),
                    coverage_mean_0p15m={"whole": float(cov015_mean[0]), "sunlit": float(cov015_mean[1]), "dark": float(cov015_mean[2]), "shape": float(cov015_mean[3])},
                    earthshine_pct=float(100.0 * esh_frac),
                    recon_within=sc_rep["recon_within"], median_sigma_m=sc_rep["median_sigma_m"],
                    occ=bool(occ), nframe=nframe, secs=time.time() - t0)
        json.dump(meta, open(os.path.join(draw_root, "draw_meta.json"), "w"), indent=2, default=float)
        return {"id": k, "ok": True, "fleet_dv": fleet_dv, "real_dv": real_dv, "stages": stage_status,
                "cov": covs.tolist(), "cov_count": cov_count.tolist(),
                "whole": float(cov_mean[0]), "sunlit": float(cov_mean[1]), "dark": float(cov_mean[2]),
                "whole_015": float(cov015_mean[0]), "sunlit_015": float(cov015_mean[1]),
                "dark_015": float(cov015_mean[2]), "shape": float(cov015_mean[3]), "cov015": covs015.tolist(),
                "mono": float(sc_rep["mono"]), "recon_within": sc_rep["recon_within"],
                "secs": time.time() - t0}
    except Exception as e:
        import traceback
        return {"id": k, "ok": False, "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}


# ════════════════════════════════════════════════════════════════════════════
# aggregation + main
# ════════════════════════════════════════════════════════════════════════════
def _boot_ci(x, q=(2.5, 97.5), n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float(x.mean()) if len(x) else 0.0] * 2
    rng = np.random.default_rng(seed)
    bs = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return [float(np.percentile(bs, q[0])), float(np.percentile(bs, q[1]))]

def wilks_note(N):
    return (f"N={N}: 95/95 two-sided {'OK' if N >= 93 else 'need>=93'}; "
            f"99/90 one-sided {'OK' if N >= 230 else 'need>=230'}")

def aggregate_and_save(results, out, t_start):
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return None
    pc = lambda x, q: float(np.percentile(x, q))
    DV = np.array([r["fleet_dv"] for r in ok]); RDV = np.array([r["real_dv"] for r in ok])
    M = max(1, len(ok[0].get("cov", [])) or 1)
    COV = np.array([c for r in ok for c in r.get("cov", [[r["whole"], r["sunlit"], r["dark"]]])])
    W = COV[:, 0]; L = COV[:, 1]; D = COV[:, 2]
    nf = len(ok[0]["cov_count"]); fc = np.zeros(nf)
    for r in ok:
        fc += np.array(r["cov_count"])
    fc /= (len(ok) * M)
    tol_keys = list(ok[0]["recon_within"].keys())
    recon = {t: float(np.mean([r["recon_within"][t] for r in ok])) for t in tol_keys}
    # tiered bands: 0.15 m (NAC) and visual-hull shape
    C15 = np.array([c for r in ok for c in r.get("cov015",
                    [[r.get("whole_015", 0), r.get("sunlit_015", 0), r.get("dark_015", 0), r.get("shape", 0)]])])
    W15 = C15[:, 0]; L15 = C15[:, 1]; D15 = C15[:, 2]; SH = C15[:, 3]
    summ = dict(n_draws=len(ok), n_orient=M, n_coverage=int(COV.shape[0]),
                fleet_dv=dict(mean=float(DV.mean()), P50=pc(DV, 50), P95=pc(DV, 95), P99=pc(DV, 99),
                              max=float(DV.max()), realized_mean=float(RDV.mean()), cap=10.0),
                whole_body=dict(P5=pc(W, 5), P50=pc(W, 50), P95=pc(W, 95), mean=float(W.mean()),
                                ci95=_boot_ci(W)),
                sunlit=dict(P5=pc(L, 5), P50=pc(L, 50), P95=pc(L, 95), mean=float(L.mean())),
                dark=dict(P5=pc(D, 5), P50=pc(D, 50), P95=pc(D, 95), mean=float(D.mean())),
                stereo_0p15m=dict(whole=dict(mean=float(W15.mean()), P5=pc(W15, 5), P50=pc(W15, 50), P95=pc(W15, 95)),
                                  sunlit_mean=float(L15.mean()), dark_mean=float(D15.mean())),
                shape_visual_hull=dict(mean=float(SH.mean()), P5=pc(SH, 5), P95=pc(SH, 95)),
                recon_area_frac_within_tol_m=recon,
                facet_prob=dict(robust_ge0p9_pct=float((fc >= 0.9).mean() * 100),
                                never_le0p1_pct=float((fc <= 0.1).mean() * 100)),
                wall_clock_min=(time.time() - t_start) / 60.0, wilks=wilks_note(len(ok)))
    json.dump(dict(summary=summ, dv=DV.tolist(), whole=W.tolist()), open(out, "w"), indent=2)
    np.savez_compressed(out.replace(".json", "_facetprob.npz"), facet_prob=fc)
    return summ

def _bar(total, initial=0):
    try:
        from tqdm import tqdm
        return tqdm(total=total, initial=initial, ncols=92, desc="MC V2 draws", unit="draw")
    except Exception:
        class B:
            def __init__(s, t, i): s.t = t; s.n = i
            def update(s, k=1): s.n += k; print(f"  [progress] {s.n}/{s.t}", flush=True)
            def close(s): pass
        return B(total, initial)

def main():
    ap = argparse.ArgumentParser(description="ARISE MC V2 orchestrator (mission-grade)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--occ", type=int, default=1)
    ap.add_argument("--nframe", type=int, default=5, help="frames scored per deputy")
    ap.add_argument("--m", type=int, default=32, help="NPA-orientation sub-draws per Basilisk draw (NxM, like V1)")
    ap.add_argument("--topup", type=int, default=1, help="strobe terminator top-up on grazing-lit facets")
    ap.add_argument("--stages", default="guidance,nav,6dof,score",
                    help="comma list; nav=run_nav_on_new_traj.py, 6dof=build_6dof_anim_data.py (mission-proven; real runs only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nominal", action="store_true",
                    help="zero trajectory perturbation = central/nominal case (separate _nominal checkpoint)")
    ap.add_argument("--skip-basilisk", action="store_true", help="self-contained synthetic fleet (no Basilisk)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jsonl", default=None)
    a = ap.parse_args()
    # keep stub (--skip-basilisk) runs in a SEPARATE checkpoint so they never make a real
    # run resume-skip them (this footgun ate a real draw once).
    _tag = "_nominal" if a.nominal else ("_stub" if a.skip_basilisk else "")
    if a.out is None:   a.out   = os.path.join(DATA, f"mc_v2_results{_tag}.json")
    if a.jsonl is None: a.jsonl = os.path.join(DATA, f"mc_v2_draws{_tag}.jsonl")
    os.makedirs(MCDIR, exist_ok=True)
    results = []; done = set()
    if os.path.exists(a.jsonl):
        for ln in open(a.jsonl):
            try:
                r = json.loads(ln)
                if r.get("ok"):
                    results.append(r); done.add(r["id"])
            except Exception:
                pass
    specs = [dict(id=k, occ=bool(a.occ), nframe=a.nframe, do_basilisk=(not a.skip_basilisk),
                  seed=a.seed, stages=set(a.stages.split(",")), M=a.m, topup=bool(a.topup),
                  perturb=([0.0] * 6 if a.nominal else sample_perturb(a.seed * 100000 + k).tolist()))
             for k in range(a.n) if k not in done]
    print("=" * 88)
    print(f"ARISE MC V2 | N={a.n} x M={a.m} = {a.n*a.m} coverage samples  occ={'ON' if a.occ else 'off'}  "
          f"nframe={a.nframe}  topup={'ON' if a.topup else 'off'}  basilisk={'NO(stub)' if a.skip_basilisk else 'YES'}")
    print(f"  stages={a.stages}  (mission-proven: nav=run_nav_on_new_traj.py, 6dof=build_6dof_anim_data.py)")
    if a.nominal: print("  *** NOMINAL (central) case: trajectory perturbation = 0 ***")
    print(f"  resume: {len(done)} done, {len(specs)} to run | {wilks_note(a.n)}")
    print(f"  cov={COV_NPZ if os.path.exists(COV_NPZ) else 'diagonal-fallback'}  "
          f"geom={'SPICE' if os.path.exists(GEOM_NPZ) else 'default-Sun-24.2deg'}  "
          f"checkpoint={os.path.basename(a.jsonl)}")
    print("=" * 88, flush=True)
    t_start = time.time(); pbar = _bar(a.n, initial=len(done)); jf = open(a.jsonl, "a")
    if specs:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(run_draw, s): s["id"] for s in specs}
            for fu in as_completed(futs):
                r = fu.result(); results.append(r)
                jf.write(json.dumps(r) + "\n"); jf.flush()
                if r.get("ok"):
                    print(f"  draw {r['id']:4d} ok  dV={r['fleet_dv']:.3f}  whole={r['whole']:.1f}% "
                          f"lit={r['sunlit']:.1f}% dark={r['dark']:.1f}%  ({r['secs']:.1f}s)", flush=True)
                    aggregate_and_save(results, a.out, t_start)
                else:
                    print(f"  draw {r['id']:4d} FAIL: {r['err']}", flush=True)
                pbar.update(1)
    pbar.close(); jf.close()
    summ = aggregate_and_save(results, a.out, t_start)
    print("\n=== MC V2 SUMMARY ===\n" + json.dumps(summ, indent=2))
    print("saved", a.out, "+ mc_v2_draws.jsonl + mc_v2_results_facetprob.npz + per-draw trees")

if __name__ == "__main__":
    main()
