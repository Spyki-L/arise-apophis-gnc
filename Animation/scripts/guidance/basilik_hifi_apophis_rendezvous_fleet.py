#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_hifi_apophis_rendezvous_fleet.py
========================================
High-fidelity redesign of the fleet Apophis rendezvous guidance.

Goal
----
Drive each of 8 deputies from deployment (off the carrier / chief trajectory)
to a distinct target in the **Apophis LVLH frame [R, T, N]** at closest
approach, achieving a **terminal position error < 1 m** while **minimising the
per-deputy total Delta-v** from deployment to rendezvous.

Why this replaces basilik_rhig_apophis_rendezvous_fleet.py
----------------------------------------------------------
The legacy fleet script propagates relative motion with **linear CW/Hill**
dynamics. The chief here is on a geocentric *flyby* trajectory during the 2029
Apophis encounter (rel. velocity ~8.8 km/s at closest approach), where CW is
invalid -> the legacy terminal error is kilometres, not metres. This script:

  * Propagates each deputy with a **high-fidelity force model** in the geocentric
    inertial (ECI) frame — Earth (central, spherical-harmonics J2..EARTH_SH_DEG)
    + Sun + Moon + optional Apophis third/fourth-body gravity, all evaluated by
    **Basilisk's own LIVE SPICE** at every integrator step, plus cannonball
    **solar-radiation pressure** for the Blue Canyon XB6 (6U) bus, integrated with
    the adaptive **RKF45** scheme (zero NumPy in the dynamics).
  * Uses a **universal-variable Lambert** solution as the deploy-burn seed.
  * Runs a **differential-correction (multiple-shooting) TCM scheme**: the
    deploy burn plus a few mid-course Trajectory Correction Manoeuvres are
    adjusted by a finite-difference Newton step (least-norm -> small Delta-v)
    until the terminal position error is < TARGET_TOL_M.
  * **Sweeps the deploy time** and keeps the minimum-total-Delta-v solution.

Ephemerides
-----------
In the default `basilisk` backend the Sun/Moon/Earth gravity bodies are driven by
**Basilisk's live SPICE interface** inside the propagation itself — there is no
NumPy-interpolated ephemeris in the force model. **Apophis** (the pointing/LVLH
target, not a gravity body) comes from a **JPL Horizons** geocentric ICRF state
file (`Animation/data/Apophis_horizons_results.txt`, KM-S, position+velocity).
The chief/carrier trajectory comes from `chief_reference.npz`. The in-house
`numpy`/`hybrid` backends still use a SPICE-sampled Sun ephemeris (`build_ephemerides`)
for fast offline iteration only.

Output
------
For each deputy, writes `apophis_rendezvous_full_D{idx}.npz` (same schema as the
legacy fleet script) into OUT_DIR, so it is a drop-in source for the 6-DOF /
animation pipeline (`Animation/scripts/sim/build_6dof_anim_data.py`).

Run (on a Basilisk machine):
    cd "ARISE Arch"
    python Scenarios/rhig_rendezvous/basilik_hifi_apophis_rendezvous_fleet.py
"""

from __future__ import annotations
import os, sys, math, time, glob, re, contextlib, faulthandler
from datetime import datetime, timezone, timedelta
import numpy as np
import numpy.linalg as la

faulthandler.enable()   # dump the native stack to stderr on a hard crash (segfault/abort)

# ════════════════════════════════════════════════════════════════════════════
# 1) TARGETS (Apophis LVLH [R, T, N], metres) — from the design figure (km)
# ════════════════════════════════════════════════════════════════════════════
DEPUTY_TARGETS_KM = np.array([
    [-1.81, -2.04, -1.06],   # CubeSat 1
    [-1.43,  0.79,  2.32],   # CubeSat 2
    [-1.56,  1.78,  1.62],   # CubeSat 3
    [-2.93, -0.73, -0.77],   # CubeSat 4
    [ 1.55, -2.45,  0.68],   # CubeSat 5
    [ 1.91,  1.96, -1.16],   # CubeSat 6
    [ 1.09, -2.70,  0.39],   # CubeSat 7
    [ 2.32,  1.85, -0.54],   # CubeSat 8
])
DEPUTY_TARGETS_M = DEPUTY_TARGETS_KM * 1.0e3

# ════════════════════════════════════════════════════════════════════════════
# 2) CONFIG
# ════════════════════════════════════════════════════════════════════════════
DEPARTURE_UTC_ISO = "2029-04-13T05:41:45Z"

TARGET_TOL_M   = float(os.getenv("ARISE_TOL_M", "0.05"))  # tighter convergence (was 0.5)
N_TCM          = 4            # mid-course TCMs (deploy burn excluded)
# TCM schedule as fractions of the transfer. The late 0.93 TCM nulls residual
# close to arrival -> much tighter terminal convergence for long-TOF deputies.
TCM_FRACS      = (0.25, 0.55, 0.80, 0.93)
# Optional override of the TCM schedule for flight-grade re-timing experiments:
#   ARISE_TCM_FRACS="0.15,0.68,0.90,0.96"  (comma-separated fractions of TOF).
_env_tcm_fracs = os.environ.get("ARISE_TCM_FRACS", "").strip()
if _env_tcm_fracs:
    TCM_FRACS = tuple(float(x) for x in _env_tcm_fracs.split(","))
    N_TCM = len(TCM_FRACS)
    print(f"[guidance] TCM_FRACS overridden via ARISE_TCM_FRACS -> {TCM_FRACS}")
# RK4 step for BOTH targeting and export (same dt => consistent terminal error).
# Smaller = more accurate but slower; 3 s is a good balance for these orbits.
PROP_DT        = float(os.getenv("ARISE_PROP_DT", "3.0"))
# Dense near-CA export: in the final EXPORT_TAIL_S seconds the recorded
# trajectory is sampled at EXPORT_FINE_DT so the fast flyby LOS rate (~v/b) is
# resolved for exact FSM/gimbal sizing. (Targeting still uses the uniform PROP_DT.)
EXPORT_TAIL_S  = float(os.getenv("ARISE_EXPORT_TAIL_S", "30.0"))
EXPORT_FINE_DT = float(os.getenv("ARISE_EXPORT_FINE_DT", "0.01"))
# Propagation backend:
#   'basilisk' : (default) ALL propagation — Lambert-seeded differential corrector,
#                TCM Newton steps, and export — uses Basilisk's spacecraft with
#                Earth (central, spherical-harmonics J2..EARTH_SH_DEG) + Sun + Moon
#                gravity from *LIVE* Basilisk SPICE (evaluated by Basilisk at every
#                integrator step — ZERO NumPy in the dynamics). The sim is built
#                ONCE per deputy and re-initialised for each corrector evaluation;
#                SPICE kernels are NOT reloaded on re-init, so the hundreds of
#                propagations stay fast.
#   'hybrid'   : corrector in the fast in-house RK4 model, export in Basilisk
#                (exposes any model mismatch but does NOT null the Basilisk error).
#   'numpy'    : all in-house RK4 (fastest; no Basilisk).
PROP_BACKEND   = os.getenv("ARISE_PROP_BACKEND", "basilisk").lower()
# Earth gravity spherical-harmonics degree/order for the LIVE-SPICE Basilisk model.
# (Deputies fly far from Earth — J2 dominates — but a high degree costs little.)
EARTH_SH_DEG   = int(os.getenv("ARISE_EARTH_SH_DEG", "10"))

# ── Spacecraft bus: Blue Canyon Technologies XB6 (6U class) ───────────────────
# XB6 datasheet: 6U class, 4U typical payload, 98–118 W deployed arrays, ±0.002°
# pointing. The datasheet omits mass, so these are standard 6U values (the SRP
# acceleration scales as Cr·A/m); override any of them via the env vars.
BUS_MASS_KG    = float(os.getenv("ARISE_BUS_MASS_KG", "12.0"))   # ~6U CubeSat wet mass
SRP_AREA_M2    = float(os.getenv("ARISE_SRP_AREA_M2", "0.30"))   # deployed-array-dominated projected area
SRP_CR         = float(os.getenv("ARISE_SRP_CR", "1.3"))         # reflectivity coefficient
USE_SRP        = os.getenv("ARISE_SRP", "1") == "1"              # cannonball solar-radiation pressure
# Apophis 4th-body gravity (sub-metre near CA). Horizons reports GM=n.a., so GM is
# ESTIMATED (RAD=0.170 km, rho~3.2 g/cc -> M~5e10 kg). Continuous geocentric
# position comes from the Apophis SPICE kernel (pure Basilisk).
USE_APOPHIS_GRAV = os.getenv("ARISE_APOPHIS_GRAV", "0") == "1"   # opt-in (sub-0.1 m; verify first)
APOPHIS_GM     = float(os.getenv("ARISE_APOPHIS_GM", "4.0"))     # m^3/s^2 (RAD 170 m, rho~3.2 g/cc)
APOPHIS_BSP    = os.getenv("ARISE_APOPHIS_BSP", "20099942.bsp")
# Numerical integrator: 'rkf45' (adaptive Runge-Kutta-Fehlberg, STK-grade) or 'rk4'
INTEGRATOR     = os.getenv("ARISE_INTEGRATOR", "rkf45").lower()
FD_STEP        = 1.0e-3       # [m/s] finite-difference step for the Jacobian
MAX_NEWTON     = 25           # max corrector iterations (more headroom for tight tol)
# deploy-time sweep (seconds after departure epoch) to minimise total Delta-v
DEPLOY_T_MIN   = 1500.0
DEPLOY_T_PAD   = 1.2 * 3600.0 # stop sweeping this far before arrival
DEPLOY_COARSE  = 1800.0
DEPLOY_REFINE  = 300.0

CAM_BORESIGHT_B = np.array([0.0, 0.0, 1.0])

# ── Path resolution ──────────────────────────────────────────────────────────
# This script lives at <REPO>/Animation/scripts/guidance/. Resolve the Animation
# root by walking up to the "Animation" folder, then derive the repo root, so the
# input/output locations hold regardless of the launch directory.
def _find_anim(start):
    d = start
    while os.path.basename(d) != "Animation":
        nd = os.path.dirname(d)
        if nd == d:
            return os.path.dirname(os.path.dirname(os.path.dirname(start)))
        d = nd
    return d
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ANIM_DIR   = _find_anim(_SCRIPT_DIR)
_REPO_ROOT  = os.path.dirname(_ANIM_DIR)

# Guidance output (deputy trajectory NPZs) -> Animation/data/trajectories
OUT_DIR = os.getenv("ARISE_OUT_DIR", os.path.join(_ANIM_DIR, "data", "trajectories"))

# JPL Horizons geocentric ICRF state file for (99942) Apophis (the pointing/LVLH
# target — Apophis is NOT a gravity body, so it never enters the force model).
HORIZONS_PATH = os.getenv(
    "ARISE_APOPHIS_HORIZONS",
    os.path.join(_ANIM_DIR, "data", "ephemeris", "Apophis_horizons_results.txt"))

# Physical constants (SI)
MU_EARTH = 3.986004418e14
MU_SUN   = 1.32712440018e20
RE_EARTH = 6378137.0
J2_EARTH = 1.08262668e-3
AU       = 1.495978707e11

departure_dt = datetime.strptime(DEPARTURE_UTC_ISO, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# ════════════════════════════════════════════════════════════════════════════
# 3) SMALL MATH HELPERS  (LVLH / attitude conventions match the legacy script)
# ════════════════════════════════════════════════════════════════════════════
def dcm_lvlh(r, v):
    """Rows [R(radial), T(along-track), N(cross-track)] — same as legacy."""
    i = r / la.norm(r)
    k = np.cross(r, v); k /= la.norm(k)
    j = np.cross(k, i)
    return np.vstack((i, j, k))

def normalize(v, eps=1e-12):
    n = la.norm(v); return (v / n if n >= eps else np.zeros_like(v)), n

def triad_from_boresight_and_tangent(los, vel):
    zB, _ = normalize(los)
    v_perp = vel - np.dot(vel, zB) * zB
    xB, nx = normalize(v_perp)
    if nx < 1e-8:
        tmp = np.array([1., 0, 0]) if abs(zB[0]) < .9 else np.array([0, 1., 0])
        xB = normalize(np.cross(tmp, zB))[0]
    yB = np.cross(zB, xB); xB = normalize(np.cross(yB, zB))[0]; yB = normalize(np.cross(zB, xB))[0]
    return np.column_stack((xB, yB, zB))

def dcm_to_quat(C):
    tr = np.trace(C)
    if tr > 0:
        S = math.sqrt(tr + 1) * 2; q0 = .25 * S
        q1 = (C[2, 1] - C[1, 2]) / S; q2 = (C[0, 2] - C[2, 0]) / S; q3 = (C[1, 0] - C[0, 1]) / S
    else:
        i = int(np.argmax([C[0, 0], C[1, 1], C[2, 2]]))
        if i == 0:
            S = math.sqrt(1 + C[0, 0] - C[1, 1] - C[2, 2]) * 2
            q0 = (C[2, 1] - C[1, 2]) / S; q1 = .25 * S; q2 = (C[0, 1] + C[1, 0]) / S; q3 = (C[0, 2] + C[2, 0]) / S
        elif i == 1:
            S = math.sqrt(1 + C[1, 1] - C[0, 0] - C[2, 2]) * 2
            q0 = (C[0, 2] - C[2, 0]) / S; q1 = (C[0, 1] + C[1, 0]) / S; q2 = .25 * S; q3 = (C[1, 2] + C[2, 1]) / S
        else:
            S = math.sqrt(1 + C[2, 2] - C[0, 0] - C[1, 1]) * 2
            q0 = (C[1, 0] - C[0, 1]) / S; q1 = (C[0, 2] + C[2, 0]) / S; q2 = (C[1, 2] + C[2, 1]) / S; q3 = .25 * S
    q = np.array([q0, q1, q2, q3]); return q / la.norm(q)

def quat_to_mrp(q):
    q0, qv = q[0], q[1:]
    return -qv / (1 - q0) if 1 + q0 < 1e-12 else qv / (1 + q0)


# ════════════════════════════════════════════════════════════════════════════
# 4) UNIVERSAL-VARIABLE LAMBERT  (Bate–Mueller–White / Vallado, single rev)
# ════════════════════════════════════════════════════════════════════════════
def _stumpff(z):
    if z > 1e-6:
        sz = math.sqrt(z); C = (1 - math.cos(sz)) / z; S = (sz - math.sin(sz)) / sz**3
    elif z < -1e-6:
        sz = math.sqrt(-z); C = (1 - math.cosh(sz)) / z; S = (math.sinh(sz) - sz) / sz**3
    else:
        C = 0.5 - z / 24.0; S = 1.0 / 6.0 - z / 120.0
    return C, S

def lambert_uv(r1, r2, tof, mu=MU_EARTH, prograde=True):
    """Return (v1, v2) for a single-rev transfer r1->r2 in time tof."""
    r1n, r2n = la.norm(r1), la.norm(r2)
    cosdnu = np.dot(r1, r2) / (r1n * r2n)
    cross = np.cross(r1, r2)
    if prograde:
        dnu = math.acos(np.clip(cosdnu, -1, 1)) if cross[2] >= 0 else 2 * math.pi - math.acos(np.clip(cosdnu, -1, 1))
    else:
        dnu = math.acos(np.clip(cosdnu, -1, 1)) if cross[2] < 0 else 2 * math.pi - math.acos(np.clip(cosdnu, -1, 1))
    A = math.sin(dnu) * math.sqrt(r1n * r2n / (1 - math.cos(dnu)))
    if abs(A) < 1e-12:
        raise ValueError("Lambert: degenerate geometry (A~0)")
    z = 0.0
    def yz(z):
        C, S = _stumpff(z)
        return r1n + r2n + A * (z * S - 1) / math.sqrt(C)
    # bracket/seed
    lo, hi = -4 * math.pi, 4 * math.pi**2
    for _ in range(200):
        C, S = _stumpff(z)
        y = r1n + r2n + A * (z * S - 1) / math.sqrt(C)
        if A > 0 and y < 0:   # push z up until y>0
            lo = z; z = 0.5 * (z + hi); continue
        chi = math.sqrt(max(y / C, 0.0))
        t = (chi**3 * S + A * math.sqrt(y)) / math.sqrt(mu)
        if abs(t - tof) < 1e-6:
            break
        if t <= tof: lo = z
        else:        hi = z
        z = 0.5 * (lo + hi)
    C, S = _stumpff(z); y = r1n + r2n + A * (z * S - 1) / math.sqrt(C)
    f = 1 - y / r1n
    g = A * math.sqrt(y / mu)
    gdot = 1 - y / r2n
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    return v1, v2


# ════════════════════════════════════════════════════════════════════════════
# 5) HIGH-FIDELITY DYNAMICS  (Earth point mass + J2 + Sun third-body), ECI
# ════════════════════════════════════════════════════════════════════════════
def accel(r, r_sun, mu=MU_EARTH, j2=True, sun=True):
    rn = la.norm(r)
    a = -mu * r / rn**3
    if j2:
        zr = r[2] / rn
        k = -1.5 * J2_EARTH * mu * RE_EARTH**2 / rn**4
        a = a + k * np.array([(1 - 5 * zr**2) * r[0] / rn,
                              (1 - 5 * zr**2) * r[1] / rn,
                              (3 - 5 * zr**2) * r[2] / rn])
    if sun:
        d = r_sun - r
        a = a + MU_SUN * (d / la.norm(d)**3 - r_sun / la.norm(r_sun)**3)
    return a

def _propagate_numpy(r0, v0, t0, t1, sun_fn, dt=PROP_DT, tcms=None, record=False,
                     fine_t=None, fine_dt=None):
    """In-house RK4 propagate with impulsive TCMs (Earth point mass + J2 + Sun
    third-body). Returns (rT, vT), or if record a dict of per-step histories.
    If fine_t/fine_dt are given, the step shrinks to fine_dt once t >= fine_t
    (used to densely sample the fast close-approach flyby for exact FSM sizing)."""
    tcms = sorted(tcms or [], key=lambda x: x[0])
    r = np.array(r0, float); v = np.array(v0, float); t = t0; ti = 0
    if record:
        Th = [t]; Rh = [r.copy()]; Vh = [v.copy()]
    while t < t1 - 1e-9:
        step = fine_dt if (fine_t is not None and t >= fine_t) else dt
        h = min(step, t1 - t)
        if ti < len(tcms) and t <= tcms[ti][0] < t + h:   # step exactly to TCM, apply, continue
            hh = tcms[ti][0] - t
            if hh > 1e-9:
                r, v = _rk4(r, v, t, hh, sun_fn); t += hh
                if record: Th.append(t); Rh.append(r.copy()); Vh.append(v.copy())
            v = v + tcms[ti][1]; ti += 1
            if record: Th.append(t); Rh.append(r.copy()); Vh.append(v.copy())
            continue
        r, v = _rk4(r, v, t, h, sun_fn); t += h
        if record: Th.append(t); Rh.append(r.copy()); Vh.append(v.copy())
    if record:
        return dict(t=np.array(Th), r=np.array(Rh), v=np.array(Vh), rT=r, vT=v)
    return r, v

def _rk4(r, v, t, h, sun_fn):
    def deriv(rr, vv, tt):
        return vv, accel(rr, sun_fn(tt))
    k1r, k1v = deriv(r, v, t)
    k2r, k2v = deriv(r + 0.5 * h * k1r, v + 0.5 * h * k1v, t + 0.5 * h)
    k3r, k3v = deriv(r + 0.5 * h * k2r, v + 0.5 * h * k2v, t + 0.5 * h)
    k4r, k4v = deriv(r + h * k3r, v + h * k3v, t + h)
    r2 = r + h / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
    v2 = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r2, v2


# ── Basilisk high-fidelity propagator (LIVE SPICE, persistent sim per deputy) ──
# ZERO NumPy in the physics: Earth (central, spherical-harmonics J2..EARTH_SH_DEG)
# + Sun + Moon gravity all come from Basilisk's own SPICE interface, evaluated
# CONTINUOUSLY by Basilisk at every integrator step (no NumPy-interpolated
# ephemeris feed). Apophis is NOT a gravity body (GM n.a.) so it never enters the
# force model — it is only the pointing/LVLH target (from the JPL Horizons file).
#
# The sim is built ONCE per (deploy-epoch, record, dt) and re-initialised for each
# corrector evaluation. InitializeSimulation resets the clock + state but does NOT
# reload the SPICE kernels, so the hundreds of differential-correction propagations
# stay fast. Re-init re-registers the hub states (a benign Basilisk warning); we
# silence that at the OS file-descriptor level so the corrector output stays clean.
_BSK_CACHE = {}
_FIRST_INIT_DONE = [False]   # first InitializeSimulation runs visibly (real errors), rest are quiet

@contextlib.contextmanager
def _quiet():
    """Silence C/C++-level stdout+stderr (Basilisk re-init / SPICE furnish chatter).
    Python exceptions still propagate normally."""
    sys.stdout.flush(); sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    old1, old2 = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1); os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old1, 1); os.dup2(old2, 2)
        os.close(devnull); os.close(old1); os.close(old2)

def _bsk_build(t0, record, dt):
    """Persistent Basilisk sim, epoch = departure + t0 (sim-time 0 == t0):
      * spacecraft (XB6 6U bus mass) with the RKF45 adaptive integrator
      * Earth (central, spherical-harmonics J2..EARTH_SH_DEG) + Sun + Moon
        (+ optional Apophis 4th body) gravity from LIVE SPICE
      * optional cannonball solar-radiation pressure
    All ephemerides are evaluated by Basilisk itself each step — zero NumPy."""
    from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody
    from Basilisk.simulation import spacecraft, radiationPressure, svIntegrators
    from Basilisk.architecture import bskLogging
    from Basilisk import __path__ as bskPath
    bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)
    EPH = os.path.join(bskPath[0], "supportData", "EphemerisData") + os.sep

    sim = SimulationBaseClass.SimBaseClass(); sim.SetProgressBar(False)
    proc = sim.CreateNewProcess("dyn"); proc.addTask(sim.CreateNewTask("dyn", macros.sec2nano(dt)))

    gf = simIncludeGravBody.gravBodyFactory()
    earth = gf.createEarth(); earth.isCentralBody = True
    try:
        earth.useSphericalHarmonicsGravityModel(
            os.path.join(bskPath[0], "supportData", "LocalGravData", "GGM03S.txt"), EARTH_SH_DEG)
    except Exception:
        pass
    gf.createSun(); gf.createMoon()                 # SPICE indices: Earth 0, Sun 1, Moon 2

    # LIVE SPICE for Earth/Sun/Moon. Epoch is offset by t0 so sim-time 0 == this
    # deputy's deploy epoch. (NOT wrapped in _quiet — a SPICE error must be visible.)
    epoch = (departure_dt + timedelta(seconds=t0)).strftime("%Y %B %d, %H:%M:%S.0 (UTC)")
    spi = gf.createSpiceInterface(time=epoch, epochInMsg=True)
    spi.zeroBase = "Earth"
    sim.AddModelToTask("dyn", spi)                  # SPICE added first -> states current each step

    # optional Apophis 4th body — add its kernel + planet name to SPICE AFTER the
    # interface exists (so the body is known before it is queried), then attach a
    # custom grav object to that SPICE message (continuous geocentric position,
    # pure Basilisk). Any failure just disables this sub-0.1 m term; the core
    # Earth/Sun/Moon model is unaffected.
    ap_kernel = os.path.join(EPH, APOPHIS_BSP)
    if USE_APOPHIS_GRAV and os.path.exists(ap_kernel):
        try:
            from Basilisk.simulation import spiceInterface
            spi.loadSpiceKernel(APOPHIS_BSP, EPH)
            spi.addPlanetNames(spiceInterface.StringVector(["20099942"]))
            apophis = gf.createCustomGravObject("apophis", APOPHIS_GM)
            apophis.planetBodyInMsg.subscribeTo(spi.planetStateOutMsgs[len(spi.planetStateOutMsgs) - 1])
            print(f"[apophis-grav] enabled (GM={APOPHIS_GM} m^3/s^2 from {APOPHIS_BSP})")
        except Exception as e:
            print(f"[apophis-grav] disabled ({e})")

    sc = spacecraft.Spacecraft(); sc.ModelTag = "deputy"
    sc.hub.mHub = BUS_MASS_KG                        # needed so SRP force -> correct accel
    sc.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    sc.hub.IHubPntBc_B = [[0.10, 0., 0.], [0., 0.20, 0.], [0., 0., 0.25]]  # non-singular 6U inertia
    integ = None
    if INTEGRATOR == "rkf45":
        try:
            integ = svIntegrators.svIntegratorRKF45(sc)   # adaptive, STK-grade
            sc.setIntegrator(integ)                       # MUST keep a ref to `integ` (else GC -> segfault)
        except Exception as e:
            print(f"[integrator] RKF45 unavailable ({e}); using default RK4"); integ = None
    gf.addBodiesTo(sc)                              # subscribes all gravity bodies to SPICE
    sim.AddModelToTask("dyn", sc)

    # solar-radiation pressure (cannonball) — the largest remaining perturbation
    srp = None
    if USE_SRP:
        try:
            srp = radiationPressure.RadiationPressure(); srp.ModelTag = "SRP"
            srp.setUseCannonballModel()
            srp.area = SRP_AREA_M2
            srp.coefficientReflection = SRP_CR
            srp.sunEphmInMsg.subscribeTo(spi.planetStateOutMsgs[1])   # Sun
            try:
                srp.stateInMsg.subscribeTo(sc.scStateOutMsg)          # (older API; harmless if absent)
            except Exception:
                pass
            sc.addDynamicEffector(srp)
            sim.AddModelToTask("dyn", srp)
        except Exception as e:
            print(f"[SRP] WARNING: solar-radiation pressure disabled ({e})"); srp = None

    rec = sc.scStateOutMsg.recorder() if record else None
    if rec is not None:
        sim.AddModelToTask("dyn", rec)
    # keep references to ALL inline-created Basilisk objects so they are not GC'd
    # (a GC'd integrator/effector leaves Basilisk holding a dangling pointer -> segfault)
    return dict(sim=sim, sc=sc, rec=rec, gf=gf, srp=srp, integ=integ)

def _bsk_run(st, r0, v0, t0, t1, tcms):
    """Re-initialise the persistent sim to (r0,v0)@t0 and propagate to t1 applying
    impulsive TCMs. Returns (rec, rT, vT)."""
    from Basilisk.utilities import macros
    sim, sc, rec = st["sim"], st["sc"], st["rec"]
    sc.hub.r_CN_NInit = [[float(r0[0])], [float(r0[1])], [float(r0[2])]]
    sc.hub.v_CN_NInit = [[float(v0[0])], [float(v0[1])], [float(v0[2])]]
    # Show the FIRST initialisation (so any real Basilisk/SPICE error is visible);
    # silence only the repetitive re-init warnings after that (ARISE_QUIET=0 keeps
    # everything visible for debugging).
    if _FIRST_INIT_DONE[0] and os.getenv("ARISE_QUIET", "1") == "1":
        with _quiet():
            sim.InitializeSimulation()          # resets clock+state; SPICE kernels stay loaded
    else:
        sim.InitializeSimulation()
        _FIRST_INIT_DONE[0] = True
    velRef = sc.dynManager.getStateObject(sc.hub.nameOfHubVelocity)
    posRef = sc.dynManager.getStateObject(sc.hub.nameOfHubPosition)
    tcm_map = {round(tc[0], 3): np.asarray(tc[1], float) for tc in (tcms or []) if t0 < tc[0] < t1}
    for b in sorted(set([t1] + list(tcm_map))):
        sim.ConfigureStopTime(macros.sec2nano(b - t0)); sim.ExecuteSimulation()
        if round(b, 3) in tcm_map:
            dv = tcm_map[round(b, 3)]; vv = velRef.getState()
            velRef.setState([[vv[0][0] + dv[0]], [vv[1][0] + dv[1]], [vv[2][0] + dv[2]]])
    pf = posRef.getState(); vf = velRef.getState()
    rT = np.array([pf[0][0], pf[1][0], pf[2][0]]); vT = np.array([vf[0][0], vf[1][0], vf[2][0]])
    return rec, rT, vT

def _propagate_bsk(r0, v0, t0, t1, dt, tcms, record, sun_fn=None,
                   fine_t=None, fine_dt=None):
    """LIVE-SPICE Basilisk propagation. `sun_fn` is unused (Sun comes from SPICE);
    it is kept only for a uniform call signature with the in-house backend."""
    from Basilisk.utilities import macros
    key = (round(t0, 3), bool(record), round(dt, 6))
    st = _BSK_CACHE.get(key)
    if st is None:
        st = _bsk_build(t0, record, dt); _BSK_CACHE[key] = st
    rec, rT, vT = _bsk_run(st, r0, v0, t0, t1, tcms)
    if not record:
        return rT, vT
    T = np.asarray(rec.times()) * macros.NANO2SEC + t0
    R = np.asarray(rec.r_BN_N); V = np.asarray(rec.v_BN_N)
    # Dense fine-dt tail over the last (t1 - fine_t) s so the fast close-approach
    # LOS rate (~v/b) is resolved for exact FSM/gimbal sizing. The tail starts from
    # a Basilisk state propagated to exactly fine_t (still pure Basilisk physics).
    if fine_t is not None and fine_dt is not None and fine_dt < dt and t0 < fine_t < t1:
        _, rmid, vmid = _bsk_run(st, r0, v0, t0, fine_t, tcms)   # exact state @ fine_t
        fkey = (round(fine_t, 3), True, round(fine_dt, 6))
        fst = _BSK_CACHE.get(fkey)
        if fst is None:
            fst = _bsk_build(fine_t, True, fine_dt); _BSK_CACHE[fkey] = fst
        frec, frT, fvT = _bsk_run(fst, rmid, vmid, fine_t, t1, None)
        fT = np.asarray(frec.times()) * macros.NANO2SEC + fine_t
        fR = np.asarray(frec.r_BN_N); fV = np.asarray(frec.v_BN_N)
        keep = T < fine_t - 1e-6
        T = np.concatenate([T[keep], fT]); R = np.vstack([R[keep], fR]); V = np.vstack([V[keep], fV])
        rT, vT = frT, fvT
    if len(T) == 0 or T[-1] < t1 - 1e-6:        # end the recorded arc EXACTLY at arrival T
        T = np.append(T, t1); R = np.vstack([R, rT]); V = np.vstack([V, vT])
    return dict(t=T, r=R, v=V, rT=rT, vT=vT)


def propagate(r0, v0, t0, t1, sun_fn, dt=PROP_DT, tcms=None, record=False,
              fine_t=None, fine_dt=None):
    """Dispatch to the configured propagation backend (see PROP_BACKEND).
    'hybrid' uses the fast in-house model for the corrector (record=False) and
    Basilisk for the final exported trajectory (record=True)."""
    use_bsk = (PROP_BACKEND == "basilisk") or (PROP_BACKEND == "hybrid" and record)
    if use_bsk:
        return _propagate_bsk(r0, v0, t0, t1, dt, tcms, record, sun_fn, fine_t, fine_dt)
    return _propagate_numpy(r0, v0, t0, t1, sun_fn, dt, tcms, record, fine_t, fine_dt)


# ════════════════════════════════════════════════════════════════════════════
# 6) EPHEMERIDES  (Basilisk SPICE preferred; analytic fallback for offline test)
# ════════════════════════════════════════════════════════════════════════════
def build_ephemerides(duration_s, step_s=60.0):
    """Return sun_fn(t)->r_sun_geo[m], rA_fn(t), vA_fn(t) (Apophis geocentric)."""
    try:
        from Basilisk import __path__ as bskPath
        from Basilisk.simulation import spiceInterface, ephemerisConverter
        from Basilisk.utilities import SimulationBaseClass, macros
        SPICE_DIR = os.path.join(bskPath[0], "supportData", "EphemerisData") + os.sep
        KERNELS = ["naif0012.tls", "pck00010.tpc", "de-403-masses.tpc", "de430.bsp", "20099942.bsp"]
        sim = SimulationBaseClass.SimBaseClass()
        task = sim.CreateNewTask("eph", macros.sec2nano(step_s))
        sim.CreateNewProcess("ephp").addTask(task)
        spi = spiceInterface.SpiceInterface(); spi.SPICEDataPath = SPICE_DIR
        spi.UTCCalInit = departure_dt.strftime("%Y %B %d, %H:%M:%S.0 (UTC)")
        for k in KERNELS: spi.loadSpiceKernel(k, SPICE_DIR)
        spi.addPlanetNames(["Earth", "Sun", "Moon", "20099942"]); spi.zeroBase = "Earth"
        sim.AddModelToTask("eph", spi)
        conv = ephemerisConverter.EphemerisConverter()
        for m in spi.planetStateOutMsgs: conv.addSpiceInputMsg(m)
        sim.AddModelToTask("eph", conv)
        sRec = conv.ephemOutMsgs[1].recorder()   # Sun (geocentric, zeroBase=Earth)
        mRec = conv.ephemOutMsgs[2].recorder()   # Moon (geocentric)
        aRec = conv.ephemOutMsgs[3].recorder()   # Apophis (geocentric)
        sim.AddModelToTask("eph", sRec); sim.AddModelToTask("eph", mRec); sim.AddModelToTask("eph", aRec)
        sim.InitializeSimulation(); sim.ConfigureStopTime(macros.sec2nano(duration_s)); sim.ExecuteSimulation()
        N = int(duration_s / step_s) + 1
        tg = np.arange(N) * step_s
        rS = np.asarray(sRec.r_BdyZero_N)[:N]
        rM = np.asarray(mRec.r_BdyZero_N)[:N]
        rA = np.asarray(aRec.r_BdyZero_N)[:N]
        vA = np.asarray(aRec.v_BdyZero_N)[:N]
        print("[eph] using Basilisk SPICE ephemerides (Earth, Sun, Moon, Apophis)")
        def mk(arr):
            def f(t):
                if t <= 0: return arr[0]
                if t >= tg[-1]: return arr[-1]
                u = t / step_s; i = int(u); fr = u - i
                return (1 - fr) * arr[i] + fr * arr[i + 1]
            return f
        return mk(rS), mk(rM), mk(rA), mk(vA)
    except Exception as e:
        print(f"[eph] WARNING: Basilisk SPICE unavailable ({e}); using ANALYTIC fallback "
              f"(geometry approximate — do not trust science numbers).")
        # crude circular Sun at 1 AU in ecliptic; Apophis on a slow geocentric pass
        def sun_fn(t):
            w = 2 * math.pi / (365.25 * 86400.0)
            return AU * np.array([math.cos(w * t), math.sin(w * t) * math.cos(math.radians(23.4)),
                                  math.sin(w * t) * math.sin(math.radians(23.4))])
        def moon_fn(t):
            w = 2 * math.pi / (27.32 * 86400.0)
            return 3.84e8 * np.array([math.cos(w * t), math.sin(w * t), 0.0])
        def rA_fn(t):
            # ~38000 km closest pass, moving in +y at ~7.4 km/s through the window
            return np.array([-2.0e7 + 0.0 * t, -4.0e8 + 7400.0 * t, 3.0e6 + 0.0 * t])
        def vA_fn(t):
            return np.array([0.0, 7400.0, 0.0])
        return sun_fn, moon_fn, rA_fn, vA_fn


def _jd_tdb_at_departure():
    """JD(TDB) of the departure epoch. departure_dt is UTC; TDB ~= TT = UTC + 69.184 s
    in 2029 (TAI-UTC = 37 s leap seconds, TT-TAI = 32.184 s)."""
    d = departure_dt
    a = (14 - d.month) // 12; y = d.year + 4800 - a; m = d.month + 12 * a - 3
    jdn = d.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd_utc = (jdn - 0.5) + (d.hour * 3600 + d.minute * 60 + d.second) / 86400.0
    return jd_utc + 69.184 / 86400.0


def parse_horizons_apophis(path):
    """Parse a JPL Horizons GEOMETRIC cartesian state file (geocentric ICRF, KM-S,
    'position and velocity' output, JDTDB time tag) into geocentric ECI ephemeris
    functions rA_fn(t)->r[m], vA_fn(t)->v[m/s], where t is seconds after the
    departure epoch. Apophis is the pointing/LVLH target only (not a gravity body),
    so this never touches the force model."""
    if not os.path.exists(path):
        sys.exit(f"[ERR] Apophis Horizons file not found: {path}")
    lines = open(path, "r", errors="replace").read().splitlines()
    try:
        i = next(k for k, l in enumerate(lines) if l.strip() == "$$SOE") + 1
    except StopIteration:
        sys.exit(f"[ERR] no $$SOE marker in Horizons file: {path}")
    numre = re.compile(r"[-+]?\d+\.\d+E[+-]\d+")
    jd, pos, vel = [], [], []
    while i < len(lines):
        l = lines[i].strip()
        if l.startswith("$$EO") or not l:        # $$EOE (end of ephemeris) / $$EOF
            break
        jd.append(float(l.split()[0]))
        xyz = [float(x) for x in numre.findall(lines[i + 1])][:3]
        vxyz = [float(x) for x in numre.findall(lines[i + 2])][:3]
        pos.append(xyz); vel.append(vxyz); i += 3
    jd = np.asarray(jd, float)
    pos = np.asarray(pos, float) * 1.0e3      # km   -> m
    vel = np.asarray(vel, float) * 1.0e3      # km/s -> m/s
    order = np.argsort(jd); jd = jd[order]; pos = pos[order]; vel = vel[order]
    jd0 = _jd_tdb_at_departure()
    def rA_fn(t):
        q = jd0 + np.asarray(t, float) / 86400.0
        return np.array([np.interp(q, jd, pos[:, k]) for k in range(3)])
    def vA_fn(t):
        q = jd0 + np.asarray(t, float) / 86400.0
        return np.array([np.interp(q, jd, vel[:, k]) for k in range(3)])
    span_d = jd[-1] - jd[0]
    print(f"[apophis] JPL Horizons: {len(jd)} states, JDTDB {jd[0]:.5f}..{jd[-1]:.5f} "
          f"({span_d:.2f} d), departure JDTDB={jd0:.6f}")
    if not (jd[0] - 1e-6 <= jd0 <= jd[-1] + 1e-6):
        print(f"[apophis] WARNING: departure epoch {jd0:.5f} is OUTSIDE the file span "
              f"[{jd[0]:.5f}, {jd[-1]:.5f}] — check DEPARTURE_UTC_ISO / the Horizons range.")
    return rA_fn, vA_fn


def load_chief():
    """chief_state(t)->(r[m],v[m/s]); also returns TOF. Searches known locations."""
    cands = [
        os.path.join(_REPO_ROOT, "GNC", "Scenario", "Apophis", "chief_reference.npz"),
        "chief_reference.npz",
        os.path.join("GNC", "Scenario", "Apophis", "chief_reference.npz"),
    ]
    path = next((c for c in cands if os.path.exists(c)), None)
    if path is None:
        sys.exit("[ERR] chief_reference.npz not found (looked in CWD and GNC/Scenario/Apophis/).")
    d = np.load(path); T, R, V = d["times"], d["positions"], d["velocities"]
    tof = float(T[-1])
    def chief_state(t):
        r = [np.interp(t, T, R[:, k]) for k in range(3)]
        v = [np.interp(t, T, V[:, k]) for k in range(3)]
        return 1e3 * np.array(r), 1e3 * np.array(v)
    print(f"[chief] {path}  TOF={tof:.0f}s")
    return tof, chief_state


# ════════════════════════════════════════════════════════════════════════════
# 7) PER-DEPUTY TARGETING  (Lambert seed + differential-correction TCMs)
# ════════════════════════════════════════════════════════════════════════════
def target_inertial(t_arr, target_lvlh, rA_fn, vA_fn):
    rA, vA = rA_fn(t_arr), vA_fn(t_arr)
    C_A = dcm_lvlh(rA, vA)              # rows [R,T,N], maps ECI->LVLH
    return rA + C_A.T @ target_lvlh     # LVLH offset -> ECI

def solve_deputy(t_dep, T_arr, target_lvlh, chief_state, sun_fn, rA_fn, vA_fn,
                 tcm_fracs=None):
    """Return dict with controls (deploy dv + TCMs), terminal error, total dv.

    tcm_fracs : optional sequence of TCM epochs as fractions of TOF.  Defaults
    to the module TCM_FRACS; pass a custom set for re-timing experiments
    (the corrector still converges to TARGET_TOL_M, so terminal accuracy is
    preserved and only the Delta-v changes).
    """
    rC0, vC0 = chief_state(t_dep)
    r_tgt = target_inertial(T_arr, target_lvlh, rA_fn, vA_fn)
    tof = T_arr - t_dep
    # Lambert seed (two-body) -> deploy velocity
    try:
        v1, _ = lambert_uv(rC0, r_tgt, tof)
    except Exception:
        v1 = vC0.copy()
    dv_deploy0 = v1 - vC0
    # control vector u = [dv_deploy(3), tcm_1(3), ..., tcm_k(3)]
    fracs = tuple(TCM_FRACS[:N_TCM]) if tcm_fracs is None else tuple(tcm_fracs)
    n_tcm = len(fracs)
    tcm_times = [t_dep + f * tof for f in fracs]
    u = np.concatenate([dv_deploy0] + [np.zeros(3) for _ in range(n_tcm)])

    def terminal_error(u):
        dvd = u[:3]; tcms = [(tcm_times[i], u[3 + 3 * i:6 + 3 * i]) for i in range(n_tcm)]
        rT, _ = propagate(rC0, vC0 + dvd, t_dep, T_arr, sun_fn, tcms=tcms)
        return rT - r_tgt

    e = terminal_error(u)
    for _ in range(MAX_NEWTON):
        if la.norm(e) < TARGET_TOL_M:
            break
        # finite-difference Jacobian d(terminal pos)/d(controls)
        J = np.zeros((3, len(u)))
        for k in range(len(u)):
            du = np.zeros(len(u)); du[k] = FD_STEP
            J[:, k] = (terminal_error(u + du) - e) / FD_STEP
        # least-norm Newton step: minimise |delta_u| s.t. J du = -e  (=> small dv)
        try:
            du = J.T @ la.solve(J @ J.T + 1e-6 * np.eye(3), -e)
        except la.LinAlgError:
            du = J.T @ la.pinv(J @ J.T) @ (-e)
        u = u + du
        e = terminal_error(u)

    dvd = u[:3]; tcms = [(tcm_times[i], u[3 + 3 * i:6 + 3 * i]) for i in range(n_tcm)]
    tot_dv = la.norm(dvd) + sum(la.norm(t[1]) for t in tcms)
    return dict(t_dep=t_dep, T_arr=T_arr, rC0=rC0, vC0=vC0, r_tgt=r_tgt,
                dv_deploy=dvd, tcm_times=tcm_times, tcms=tcms,
                err=la.norm(e), tot_dv=tot_dv, u=u)

def _lambert_seed_dv(t_dep, T_arr, target_lvlh, chief_state, rA_fn, vA_fn):
    """Cheap deploy-Delta-v estimate (Lambert two-body, no propagation).
    The deploy burn dominates total Delta-v (mid-course TCMs are ~cm/s), so this
    is an accurate proxy for choosing the min-Delta-v deploy time."""
    rC0, vC0 = chief_state(t_dep)
    r_tgt = target_inertial(T_arr, target_lvlh, rA_fn, vA_fn)
    try:
        v1, _ = lambert_uv(rC0, r_tgt, T_arr - t_dep)
        return la.norm(v1 - vC0)
    except Exception:
        return np.inf

def min_dv_deploy(T_arr, target_lvlh, chief_state, sun_fn, rA_fn, vA_fn,
                  tcm_fracs=None):
    """Pick the min-Delta-v deploy time by the cheap Lambert proxy, then run the
    full high-fidelity differential corrector only at the chosen time."""
    tmax = max(DEPLOY_T_MIN + 1, T_arr - DEPLOY_T_PAD)
    grid = np.arange(DEPLOY_T_MIN, tmax + 1e-6, DEPLOY_COARSE)
    seed = [( _lambert_seed_dv(td, T_arr, target_lvlh, chief_state, rA_fn, vA_fn), td)
            for td in grid]
    best_td = min(seed, key=lambda x: x[0])[1]
    lo, hi = max(DEPLOY_T_MIN, best_td - DEPLOY_COARSE), min(tmax, best_td + DEPLOY_COARSE)
    for td in np.arange(lo, hi + 1e-6, DEPLOY_REFINE):
        dv = _lambert_seed_dv(td, T_arr, target_lvlh, chief_state, rA_fn, vA_fn)
        if dv < min(s[0] for s in seed):
            seed.append((dv, td)); best_td = td
    # full corrector at the chosen deploy time
    return solve_deputy(best_td, T_arr, target_lvlh, chief_state, sun_fn, rA_fn, vA_fn,
                        tcm_fracs=tcm_fracs)


# ════════════════════════════════════════════════════════════════════════════
# 8) EXPORT  (same NPZ schema as basilik_rhig_apophis_rendezvous_fleet.py)
# ════════════════════════════════════════════════════════════════════════════
def export_full(path, sol, target_lvlh, chief_state, sun_fn, rA_fn, vA_fn, arrival_iso):
    t_dep, T_arr = sol["t_dep"], sol["T_arr"]
    hist = propagate(sol["rC0"], sol["vC0"] + sol["dv_deploy"], t_dep, T_arr, sun_fn,
                     tcms=sol["tcms"], record=True,
                     fine_t=T_arr - EXPORT_TAIL_S, fine_dt=EXPORT_FINE_DT)
    t = hist["t"]; rD = hist["r"]; vD = hist["v"]; N = len(t)
    rA = np.vstack([rA_fn(tt) for tt in t]); vA = np.vstack([vA_fn(tt) for tt in t])
    rC = np.vstack([chief_state(tt)[0] for tt in t]); vC = np.vstack([chief_state(tt)[1] for tt in t])
    CA = np.empty((N, 3, 3)); CC = np.empty((N, 3, 3))
    bRef = np.empty((N, 3)); CRN = np.empty((N, 3, 3)); q = np.empty((N, 4)); sig = np.empty((N, 3))
    sepA = np.empty((N, 3)); err = np.empty(N)
    for j in range(N):
        CA[j] = dcm_lvlh(rA[j], vA[j]); CC[j] = dcm_lvlh(rC[j], vC[j])
        los, _ = normalize(rA[j] - rD[j]); bRef[j] = los
        CRN[j] = triad_from_boresight_and_tangent(los, vD[j]).T
        q[j] = dcm_to_quat(CRN[j]); sig[j] = quat_to_mrp(q[j])
        sepA[j] = CA[j] @ (rD[j] - rA[j]); err[j] = la.norm(sepA[j] - target_lvlh)
    dv_times = np.array([t_dep] + list(sol["tcm_times"]))
    dv_list = np.vstack([sol["dv_deploy"]] + [tc[1] for tc in sol["tcms"]])
    impulses = np.column_stack([dv_times, dv_list])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        t=t, rD_N=rD, vD_N=vD, rA_N=rA, vA_N=vA, rC_N=rC, vC_N=vC,
        C_A_N=CA, C_C_N=CC, bRef_N=bRef, C_RN=CRN, q_RN=q, sigma_RN=sig,
        sepA_hist=sepA, err_hist=err,
        dv_times=dv_times, dv_list=dv_list, impulses=impulses,
        camMount_B=CAM_BORESIGHT_B, APOPHIS_TARGET=target_lvlh,
        TARGET_MARGIN_M=np.array([TARGET_TOL_M]),
        t0=np.array([t_dep]), best_t=np.array([t_dep]),
        # terminal error of the ACTUALLY-PROPAGATED (e.g. Basilisk) trajectory
        err_norm=np.array([float(err[-1])]), tot_dv=np.array([sol["tot_dv"]]),
        meta=np.array([DEPARTURE_UTC_ISO, arrival_iso], dtype=object),
    )
    if abs(float(err[-1]) - sol["err"]) > TARGET_TOL_M:
        print(f"    [note] propagated terminal error {float(err[-1]):.3f} m differs from "
              f"corrector {sol['err']:.3f} m (backend model mismatch).")
    print(f"[export] {path}")


# ════════════════════════════════════════════════════════════════════════════
# 9) MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("\n========== HIGH-FIDELITY FLEET RENDEZVOUS (8 deputies) ==========")
    TOF, chief_state = load_chief()
    T_arr = TOF
    # Apophis target (pointing/LVLH) from the JPL Horizons geocentric state file.
    rA_fn, vA_fn = parse_horizons_apophis(HORIZONS_PATH)
    if PROP_BACKEND == "basilisk":
        # Propagation uses LIVE Basilisk SPICE (Earth SH J2..%d + Sun + Moon) — the
        # in-house Sun ephemeris is not needed (zero NumPy in the dynamics).
        sun_fn = None
        srp_txt = (f"SRP(cannonball A={SRP_AREA_M2} m^2, Cr={SRP_CR}, m={BUS_MASS_KG} kg)"
                   if USE_SRP else "SRP off")
        ap_txt = f"Apophis-grav(GM={APOPHIS_GM})" if USE_APOPHIS_GRAV else "Apophis-grav off"
        print(f"[prop] backend=basilisk : LIVE SPICE, integrator={INTEGRATOR}")
        print(f"       force model: Earth(SH J2..{EARTH_SH_DEG}) + Sun + Moon + {ap_txt} + {srp_txt}")
        print(f"       bus: Blue Canyon XB6 (6U)  (ZERO NumPy in the dynamics)")
    else:
        sun_fn, _moon_fn, _rA, _vA = build_ephemerides(int(TOF + 3600), step_s=60.0)
        print(f"[prop] backend={PROP_BACKEND} : in-house RK4 (Earth + J2 + Sun)")
    arrival_iso = (departure_dt + timedelta(seconds=TOF)).isoformat().replace("+00:00", "Z")
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    for idx, tgt in enumerate(DEPUTY_TARGETS_M):
        t0 = time.time()
        print(f"\n--- Deputy {idx+1}  target(LVLH km)={tgt/1e3} ---")
        sol = min_dv_deploy(T_arr, tgt, chief_state, sun_fn, rA_fn, vA_fn)
        print(f"  deploy@{sol['t_dep']:.0f}s  terminal |e|={sol['err']:.3f} m  "
              f"total dv={sol['tot_dv']:.4f} m/s  ({time.time()-t0:.1f}s)")
        export_full(os.path.join(OUT_DIR, f"apophis_rendezvous_full_D{idx}.npz"),
                    sol, tgt, chief_state, sun_fn, rA_fn, vA_fn, arrival_iso)
        rows.append((idx + 1, sol["t_dep"], sol["err"], sol["tot_dv"],
                     la.norm(sol["dv_deploy"]), [la.norm(tc[1]) for tc in sol["tcms"]]))
        # pasteable per-deputy report line (deploy UTC + TCM times + magnitudes)
        dep_utc = (departure_dt + timedelta(seconds=sol["t_dep"])).strftime("%Y-%m-%dT%H:%M:%SZ")
        tcm_s = [round(tc[0], 0) for tc in sol["tcms"]]
        tcm_mms = [round(la.norm(tc[1]) * 1e3, 3) for tc in sol["tcms"]]
        print(f"[RPT] D{idx+1} deploy_s={sol['t_dep']:.0f} deploy_utc={dep_utc} "
              f"tof_h={(T_arr-sol['t_dep'])/3600:.3f} deploy_dv_ms={la.norm(sol['dv_deploy']):.5f} "
              f"ntcm={len(sol['tcms'])} tcm_times_s={tcm_s} tcm_mms={tcm_mms} "
              f"total_dv_ms={sol['tot_dv']:.5f} terminal_err_m={sol['err']:.4f}")

    print("\n================= SUMMARY =================")
    print(f"{'Dep':>4}{'deploy[s]':>11}{'|e|[m]':>9}{'totDV[m/s]':>12}{'deployDV':>10}   TCMs[m/s]")
    for d, td, e, tot, dvd, tcms in rows:
        tcm_str = " ".join(f"{x:.3f}" for x in tcms)
        flag = "" if e < 1.0 else "  <-- NOT CONVERGED"
        print(f"{d:>4}{td:>11.0f}{e:>9.3f}{tot:>12.4f}{dvd:>10.4f}   {tcm_str}{flag}")
    print(f"\nfleet total dv = {sum(r[3] for r in rows):.4f} m/s   "
          f"(max terminal error = {max(r[2] for r in rows):.3f} m, target < 1 m)")
    print(f"outputs in: {OUT_DIR}")
    glob_path = os.path.join(OUT_DIR, "apophis_rendezvous_full_D*.npz")
    print("Next: build the 6-DOF animation data with")
    print("  python Animation/scripts/sim/build_6dof_anim_data.py")

    # release SPICE kernels held by the persistent Basilisk sims
    for st in _BSK_CACHE.values():
        try:
            st["gf"].unloadSpiceKernels()
        except Exception:
            pass


if __name__ == "__main__":
    main()
