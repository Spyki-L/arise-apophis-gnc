#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_rhig_apophis_rendezvous_export.py

Purpose
-------
Script A for RHIG: run the rendezvous guidance loop for an Apophis flyby
(using a chief reference NPZ and SPICE ephemerides), search over deploy time,
and then export a **comprehensive, time-aligned dataset** for 6-DOF pointing
and camera simulations. Plotting is delegated to `Visualize.py` (which expects
large fonts: titles/labels ≈ 50, ticks ≈ 45).

The exported NPZ provides everything needed to drive an attitude/camera sim:

    - t                       (N,)    seconds since chief-deploy instant
    - rD_N, vD_N              (N,3)   deputy ECI position/velocity [m], [m/s]
    - rA_N, vA_N              (N,3)   Apophis geocentric ECI pos/vel [m], [m/s]
    - rC_N, vC_N              (N,3)   chief ECI pos/vel [m], [m/s]
    - C_A_N, C_C_N            (N,3,3) LVLH_A<-N and LVLH_C<-N DCMs
                                     (rows are RTN axes expressed in ECI)
    - bRef_N                  (N,3)   unit LOS (deputy→Apophis) in ECI
    - C_RN                    (N,3,3) desired reference attitude DCM (N→R),
                                     camera boresight pointed at Apophis
    - q_RN                    (N,4)   scalar-first quaternion [q0,q1,q2,q3], N→R
    - sigma_RN                (N,3)   MRPs (N→R)
    - camMount_B              (3,)    camera boresight in body frame
                                     (default: +Z_B)
    - meta                    (2,)    [departure_utc_iso, arrival_utc_iso] strings
    - t0                      (1,)    deploy time [s] since departure epoch

Scenario setup
--------------
1. Chief trajectory (ECI)
   - Loaded from `chief_reference.npz`:
       * `times`     (s)
       * `positions` (km)
       * `velocities`(km/s)
   - Converted to SI and interpolated via `chief_state(t)` to provide:
       * rC_N(t), vC_N(t) for any t.
   - Total time of flight (TOF) and arrival epoch computed from the last sample.

2. Apophis ephemeris (geocentric)
   - Uses Basilisk’s SPICE interface (`SpiceInterface`) and:
       * Kernels: naif0012.tls, pck00010.tpc, de-403-masses.tpc, de430.bsp,
         and a custom Apophis SPK `20099942.bsp`.
       * `zeroBase = "Sun"` but the script forms **geocentric** states by
         subtracting Earth’s state from Apophis’s.
   - A sampler is built:
       * `rA_fn(t)`, `vA_fn(t)` returning geocentric ECI states for Apophis at
         arbitrary times over the chief’s TOF.

3. RHIG guidance and dynamics
   - State definitions:
       * `RHO_DEPLOY`, `RHOD_DEPLOY` define the initial LVLH (chief frame)
         deputy state at deploy.
       * Chief LVLH → Hill/CW dynamics used for propagation via `cw_blocks`.
   - Mixed-frame terminal map:
       * `cw_map_to_ApophisLVLH(rho, rhod, t_now)` maps current LVLH state
         to predicted terminal separation in **Apophis LVLH** and returns the
         sensitivity matrix S used for damped least-squares Δv.
   - Guidance logic:
       * Fires Δv (in chief LVLH RTN) when:
           - on an FSW cadence (DT_FSW),
           - outside the final `MIN_LOOK` window,
           - cooldown since last burn ≥ `COOLDOWN`,
           - |sep_error| > `E_TRIGGER`,
           - total burns < `MAX_IMP`.
       * Damped minimum-norm CW Δv via `damped_min_norm(S, eA)`.
       * Optional Lambert “nudge”:
           - `lambert_delta_v_now()` computes a direct rendezvous burn to a
             final inertial target near Apophis.
           - Mixed with CW Δv via `NUDGE_WEIGHT` and capped by `NUDGE_SAT`.
       * Simple scalar line search over scale factors {1, 0.5, 0.25, …} to
         ensure a minimum improvement `IMPROVE_MIN` in error norm while
         respecting burn limits [V_MIN, V_MAX].

Deploy-time search
------------------
- A multi-stage search over deploy time `t_dep`:

  1. Coarse scan:
     - t_dep ∈ [SEARCH_TMIN, TOF_SEC − SEARCH_TMAX_PAD] in steps of `COARSE_STEP`.
  2. Refinement:
     - Two refinement passes around the best coarse t_dep using
       `REFINE_STEP_1` and `REFINE_STEP_2` across a window of `REFINE_SPAN`
       seconds.

- Cost function:
      J = ||final_error|| + LAMBDA_DV * (total Δv).

- The best deploy time `best_t` is then used for a final high-authority run
  with `FINAL_V_MAX` and `FINAL_MAX_IMP`.

Final run and export
--------------------
- Once `best_t` is found, the script calls:

      final_res = run_rhig_from(best_t, v_max=FINAL_V_MAX,
                                max_imp=FINAL_MAX_IMP, record_hist=True)

- It prints a summary:
    * Departure, deploy, arrival UTC.
    * TOF (hours / seconds), target vector, achieved separation in Apophis LVLH.
    * Final error vector and norm, number of impulses, total Δv.
    * A list of each burn time and Δv (in RTN) in mm/s.

- `export_npz(NPZ_EXPORT_PATH, final_res, t0=final_res["t0"])` writes the
  consolidated NPZ described above, ready for 6-DOF attitude/camera sims or
  plotting by `Visualize.py`.

Reference attitude construction
-------------------------------
For every logged step in the final run, the script builds a robust **two-vector**
reference attitude:

- z_B := LOS (deputy → Apophis), normalized.
- x_B := deputy velocity projected onto the plane ⟂ z_B; if too small, a
  fallback using the angular-momentum direction (or a safe axis) is used.
- y_B := z_B × x_B, with re-orthogonalization to ensure a clean ONB.

From these:

- `C_NB_des` has columns [x_B, y_B, z_B].
- `C_RN` is defined as **N→R** by `C_RN = C_NB_desᵀ`.
- `q_RN` is the scalar-first quaternion of `C_RN`.
- `sigma_RN` are the MRPs of `C_RN`.

These are exported so downstream pointing sims can treat the “R” frame as the
desired body frame with the camera boresight aligned to Apophis.

Visualization hook
------------------
At the end of the script, plotting is delegated to:

    import Visualize as viz
    viz.plot_all(...)

with parameters:

- `best_t`, `DT_DYN`, `APOPHIS_TARGET`, `TARGET_MARGIN_M`,
- `E_TRIGGER`, `V_MIN`, `V_MAX`, `DV_BUDGET_MS`, `R_EARTH_KM`,
- `save_dir` (PNG save directory; `SAVE_PNG_DIR=None` to avoid saving),
- `font_size`, `tick_size`,
- `npz_path_for_nav` (optional nav NPZ for nav-error overlays).

Usage
-----
Prerequisites:

- Basilisk Python environment with SPICE data (EphemerisData directory and
  Apophis SPK `20099942.bsp`).
- `chief_reference.npz` with:
    * times [s],
    * positions [km],
    * velocities [km/s].

Then run:

    python basilik_rhig_apophis_rendezvous_export.py

The script will:

- Build the chief + Apophis geometry and RHIG timeline.
- Search for a deploy time that balances final error and Δv.
- Re-run RHIG with the best deploy time.
- Export a rich rendezvous NPZ for 6-DOF pointing simulations.
- Invoke `Visualize.plot_all` to generate analysis plots.
"""


from __future__ import annotations
import os, math, warnings
from datetime import datetime, timezone, timedelta
import numpy as np
import numpy.linalg as la
from scipy.linalg import svd

from Basilisk import __path__ as bskPath
from Basilisk.simulation  import spiceInterface, ephemerisConverter, simpleNav, spacecraft
from Basilisk.utilities   import SimulationBaseClass, macros, orbitalMotion as om
from Basilisk.fswAlgorithms import lambertPlanner, lambertSolver
from Basilisk.utilities.simIncludeGravBody import gravBodyFactory

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(precision=3, suppress=True)

# ───────────────────────── timeline (UTC) ─────────────────────────
DEPARTURE_UTC_ISO = "2029-04-13T05:41:45Z"
departure_dt = datetime.strptime(DEPARTURE_UTC_ISO, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

DEPLOY_EPOCH_REF = 3000.0  # for reference printout (optimizer overrides actual deploy)

# ───────────────────────── target (Apophis LVLH) ─────────────────
APOPHIS_TARGET = np.array([1.43, 2.55, -0.27]) * 1e3   # [m]

# Initial relative state at deploy (chief LVLH)
RHO_DEPLOY  = np.zeros(3)
RHOD_DEPLOY = np.zeros(3)

# ───────────────────────── RHIG params ───────────────────────────
DT_DYN   = 1.0
DT_FSW   = 60.0
E_TRIGGER= 5.0
IMPROVE_MIN = 3.0
V_MIN    = 0.05e-3
V_MAX    = 0.1
MIN_LOOK = 250.0
COOLDOWN = 250.0
MAX_IMP  = 60
ALPHA_DAMP_BASE = 1e-7
WEIGHTS = np.diag([1.0, 1.0, 1.0])

# Optional Lambert “nudge”
USE_LAMBERT_NUDGE = True
NUDGE_WEIGHT      = 0.4
NUDGE_SAT         = 0.4    # [m/s]
ALIGN_DEG_MIN     = 1.0    # skip nudge if angle(r_now, r_goal)<1°

# Final trim near arrival (hooks; currently not invoked in core loop)
FORCE_TRIM_AT     = 180.0
TRIM_MAX          = 0.5

# ── deploy-time search (error + λ·ΔV) ──
LAMBDA_DV         = 380.0
SEARCH_TMIN       = 1500
SEARCH_TMAX_PAD   = 1.2*3600.0
COARSE_STEP       = 900.0
REFINE_STEP_1     = 150.0
REFINE_STEP_2     = 60.0
REFINE_SPAN       = 1.2*3600.0
FINAL_V_MAX       = 0.5
FINAL_MAX_IMP     = 80

# ── plotting & export ──
TARGET_MARGIN_M   = 1.0
DV_BUDGET_MS      = None            # e.g., 1.5 to draw a budget line; None disables
SAVE_PNG_DIR      = None            # e.g., "plots" to save; None => don’t save
NPZ_EXPORT_PATH   = "apophis_rendezvous_export.npz"
CAM_BORESIGHT_B   = np.array([0.0, 0.0, 1.0])  # +Z_B is camera optical axis

# Optional: if you have a separate NPZ with SimpleNav estimates (rNavD_N, vNavD_N),
# set its path here; Visualize.py will use it to plot nav errors.
NAV_NPZ_PATH      = None            # e.g., "sixdof_nav_outputs.npz"

# ───────────────────────── constants ──────────────────────────────
try:
    from Basilisk.utilities import astroConstants as ac
    MU_EARTH  = ac.mu_Earth * 1e9
    REQ_EARTH = ac.req_Earth * 1e3
    J2_EARTH  = ac.J2_Earth
    R_EARTH_KM= ac.req_Earth/1e3
except Exception:
    MU_EARTH, REQ_EARTH, J2_EARTH = 3.986004418e14, 6_378_137.0, 1.08262668e-3
    R_EARTH_KM = 6378.137

SPICE_DIR = os.path.join(bskPath[0], "supportData", "EphemerisData") + os.sep
KERNELS = ["naif0012.tls","pck00010.tpc","de-403-masses.tpc","de430.bsp","20099942.bsp"]

# ───────────────────────── helpers ────────────────────────────────
def dcm_lvlh(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return LVLH<-ECI DCM with rows = [R̂; T̂; N̂] in ECI components."""
    i_hat = r / la.norm(r)
    k_hat = np.cross(r, v); k_hat /= la.norm(k_hat)
    j_hat = np.cross(k_hat, i_hat)
    return np.vstack((i_hat, j_hat, k_hat))  # rows RTN

def hill_rate(r: np.ndarray, v: np.ndarray) -> float:
    return la.norm(np.cross(r, v)) / (la.norm(r)**2)

def cw_blocks(n: float, dt: float):
    c = math.cos(n*dt); s = math.sin(n*dt); nt = n*dt
    Phi_rr = np.array([[4-3*c, 0, 0],[6*(s-nt), 1, 0],[0,0,c]])
    Phi_rv = np.array([[s/n, 2*(1-c)/n, 0],[-2*(1-c)/n, (4*s-3*nt)/n, 0],[0,0,s/n]])
    Phi_vr = np.array([[3*n*s,0,0],[6*n*(c-1),0,0],[0,0,-n*s]])
    Phi_vv = np.array([[c,2*s,0],[-2*s,4*c-3,0],[0,0,c]])
    return Phi_rr, Phi_rv, Phi_vr, Phi_vv

def angle_deg(a, b):
    na, nb = la.norm(a), la.norm(b)
    if na == 0 or nb == 0: return 0.0
    c = np.clip(np.dot(a,b)/(na*nb), -1.0, 1.0)
    return np.degrees(np.arccos(c))

def normalize(v, eps=1e-12):
    n = la.norm(v)
    if n < eps: return np.zeros_like(v), 0.0
    return v/n, n

def triad_from_boresight_and_tangent(los_N: np.ndarray, vel_N: np.ndarray) -> np.ndarray:
    """
    Build desired body axes in ECI:
      z_B := LOS (deputy→Apophis)
      x_B := vel projected onto plane ⟂ z_B (fallback to ĥ)
      y_B := z_B × x_B
    Returns C_NB_des with columns [x_B, y_B, z_B].
    """
    zB, _ = normalize(los_N)
    v_perp = vel_N - np.dot(vel_N, zB) * zB
    xB, nx = normalize(v_perp)
    if nx < 1e-8:
        h = np.cross(los_N, vel_N)
        h_hat, nh = normalize(h)
        if nh < 1e-8:
            tmp = np.array([1.0, 0.0, 0.0]) if abs(zB[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            xB = normalize(np.cross(tmp, zB))[0]
        else:
            xB = normalize(np.cross(h_hat, zB))[0]
    yB = np.cross(zB, xB)
    xB = normalize(np.cross(yB, zB))[0]
    yB = normalize(np.cross(zB, xB))[0]
    C_NB = np.column_stack((xB, yB, zB))
    return C_NB

def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    """Return quaternion [q0, q1, q2, q3] (scalar-first) from DCM (active, right-hand)."""
    tr = np.trace(C)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        q0 = 0.25 * S
        q1 = (C[2,1] - C[1,2]) / S
        q2 = (C[0,2] - C[2,0]) / S
        q3 = (C[1,0] - C[0,1]) / S
    else:
        i = int(np.argmax([C[0,0], C[1,1], C[2,2]]))
        if i == 0:
            S = math.sqrt(1.0 + C[0,0] - C[1,1] - C[2,2]) * 2.0
            q0 = (C[2,1] - C[1,2]) / S
            q1 = 0.25 * S
            q2 = (C[0,1] + C[1,0]) / S
            q3 = (C[0,2] + C[2,0]) / S
        elif i == 1:
            S = math.sqrt(1.0 + C[1,1] - C[0,0] - C[2,2]) * 2.0
            q0 = (C[0,2] - C[2,0]) / S
            q1 = (C[0,1] + C[1,0]) / S
            q2 = 0.25 * S
            q3 = (C[1,2] + C[2,1]) / S
        else:
            S = math.sqrt(1.0 + C[2,2] - C[0,0] - C[1,1]) * 2.0
            q0 = (C[1,0] - C[0,1]) / S
            q1 = (C[0,2] + C[2,0]) / S
            q2 = (C[1,2] + C[2,1]) / S
            q3 = 0.25 * S
    q = np.array([q0, q1, q2, q3])
    return q / la.norm(q)

def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [q0, q1, q2, q3] → MRPs σ (3,)."""
    q0 = q[0]
    qv = q[1:]
    denom = 1.0 + q0
    if denom < 1e-12:
        denom = 1.0 - q0
        return -qv / max(denom, 1e-12)
    return qv / denom

# ───────────────────────── chief NPZ (SI) ─────────────────────────
def load_chief_npz():
    data = np.load("chief_reference.npz")
    T, R, V = data["times"], data["positions"], data["velocities"]  # s, km, km/s
    TOF_SEC = float(T[-1])

    def chief_state(t_s: float) -> tuple[np.ndarray, np.ndarray]:
        r_km = np.array([np.interp(t_s, T, R[:,k]) for k in range(3)])
        v_km = np.array([np.interp(t_s, T, V[:,k]) for k in range(3)])
        return r_km*1e3, v_km*1e3
    return TOF_SEC, chief_state

TOF_SEC, chief_state = load_chief_npz()
arrival_dt = departure_dt + timedelta(seconds=TOF_SEC)

# ───────────────────────── SPICE (Apophis geocentric) ────────────
def make_apophis_sampler(departure_dt: datetime, *, duration_s: int, step_s: int = 60):
    sim  = SimulationBaseClass.SimBaseClass()
    task = sim.CreateNewTask("tsk", macros.sec2nano(step_s))
    sim.CreateNewProcess("dyn").addTask(task)

    spi = spiceInterface.SpiceInterface()
    spi.SPICEDataPath = SPICE_DIR
    spi.UTCCalInit    = departure_dt.strftime("%Y %B %d, %H:%M:%S.0 (UTC)")
    for k in KERNELS: spi.loadSpiceKernel(k, SPICE_DIR)
    spi.addPlanetNames(["Earth","Sun","20099942"])
    spi.zeroBase = "Sun"
    sim.AddModelToTask("tsk", spi)

    conv = ephemerisConverter.EphemerisConverter()
    for m in spi.planetStateOutMsgs: conv.addSpiceInputMsg(m)
    sim.AddModelToTask("tsk", conv)

    eRec = conv.ephemOutMsgs[0].recorder()
    aRec = conv.ephemOutMsgs[2].recorder()
    sim.AddModelToTask("tsk", eRec); sim.AddModelToTask("tsk", aRec)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(duration_s))
    sim.ExecuteSimulation()

    N      = int(np.floor(duration_s/step_s)) + 1
    t_grid = np.arange(N, dtype=float) * step_s
    rA_S   = aRec.r_BdyZero_N[:N]
    vA_S   = aRec.v_BdyZero_N[:N]
    rE_S   = eRec.r_BdyZero_N[:N]
    vE_S   = eRec.v_BdyZero_N[:N]

    r_geo = rA_S - rE_S
    v_geo = vA_S - vE_S

    def interp(arr, t):
        if t <= 0.0: return arr[0]
        if t >= t_grid[-1]: return arr[-1]
        u = t / step_s
        i = int(np.floor(u)); f = u - i
        return (1.0 - f) * arr[i] + f * arr[i+1]

    def rA(t_sec): return interp(r_geo, float(t_sec))
    def vA(t_sec): return interp(v_geo, float(t_sec))
    return rA, vA

rA_fn, vA_fn = make_apophis_sampler(departure_dt, duration_s=int(TOF_SEC + 3600), step_s=60)

# Sanity print
rC_RV, vC_RV = chief_state(TOF_SEC)
rA_RV, vA_RV = rA_fn(TOF_SEC), vA_fn(TOF_SEC)
C_A_RV = dcm_lvlh(rA_RV, vA_RV)
sep_check = C_A_RV @ (rC_RV - rA_RV)
print("[Timeline]")
print("  Departure UTC:", departure_dt.isoformat().replace("+00:00","Z"))
print("  Arrival   UTC:", arrival_dt.isoformat().replace("+00:00","Z"))
print(f"  TOF             : {TOF_SEC/3600.0:.2f} h  ({TOF_SEC:.1f} s)")
print("[Sanity] Chief−Apophis @Arrival (Apophis LVLH): "
      f"R={sep_check[0]/1e3:.3f} km, T={sep_check[1]/1e3:.3f} km, N={sep_check[2]/1e3:.3f} km, "
      f"|sep|={la.norm(sep_check)/1e3:.3f} km")

# ───────────────────────── mixed-frame terminal map ─────────
def cw_map_to_ApophisLVLH(rho: np.ndarray, rhod: np.ndarray, t_now: float):
    rC_now, vC_now = chief_state(t_now)
    n = hill_rate(rC_now, vC_now)
    dt = TOF_SEC - t_now

    Phi_rr, Phi_rv, _, _ = cw_blocks(n, dt)
    rho_f = Phi_rr @ rho + Phi_rv @ rhod

    rC_f, vC_f = chief_state(TOF_SEC)
    rA_f, vA_f = rA_fn(TOF_SEC), vA_fn(TOF_SEC)
    C_C_f = dcm_lvlh(rC_f, vC_f)
    C_A_f = dcm_lvlh(rA_f, vA_f)

    rD_f = rC_f + C_C_f.T @ rho_f
    sepA = C_A_f @ (rD_f - rA_f)
    S    = C_A_f @ C_C_f.T @ Phi_rv
    return sepA, S

def damped_min_norm(S: np.ndarray, eA: np.ndarray, alpha_base=ALPHA_DAMP_BASE) -> np.ndarray:
    U, s, Vt = svd(S, full_matrices=False)
    alpha = alpha_base + (0.1*(s[-1]**2) if s.size else alpha_base)
    A = S.T @ S + alpha*np.eye(S.shape[1])
    b = -S.T @ eA
    return la.solve(A, b)

# ───────────────────────── Lambert “nudge” (optional) ──────
def lambert_delta_v_now(r_now_m: np.ndarray, v_now_mps: np.ndarray,
                        r_goal_m: np.ndarray, tof_s: float) -> np.ndarray:
    sim = SimulationBaseClass.SimBaseClass()
    dynT = sim.CreateNewTask("dynT", macros.sec2nano(0.01))
    fswT = sim.CreateNewTask("fswT", macros.sec2nano(0.01))
    sim.CreateNewProcess("dyn").addTask(dynT)
    sim.CreateNewProcess("fsw").addTask(fswT)

    mu_m = gravBodyFactory().createEarth().mu

    sc = spacecraft.Spacecraft()
    sc.hub.r_CN_NInit = r_now_m
    sc.hub.v_CN_NInit = v_now_mps
    nav = simpleNav.SimpleNav(); nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    sim.AddModelToTask("dynT", sc); sim.AddModelToTask("dynT", nav)

    planner = lambertPlanner.LambertPlanner()
    planner.navTransInMsg.subscribeTo(nav.transOutMsg)
    planner.setR_TN_N(r_goal_m)
    planner.setMu(mu_m)
    planner.setNumRevolutions(0)
    planner.setManeuverTime(10.0)
    planner.setFinalTime(10.0 + float(tof_s))

    solver = lambertSolver.LambertSolver()
    try:
        solver.alignmentThreshold = np.deg2rad(0.1)
    except Exception:
        pass
    solver.lambertProblemInMsg.subscribeTo(planner.lambertProblemOutMsg)
    solRec = solver.lambertSolutionOutMsg.recorder()
    sim.AddModelToTask("fswT", planner); sim.AddModelToTask("fswT", solver); sim.AddModelToTask("fswT", solRec)

    sim.InitializeSimulation(); sim.ConfigureStopTime(macros.sec2nano(1.0)); sim.ExecuteSimulation()
    v_req = solRec.v1_N[-1]
    return v_req - v_now_mps

# ───────────────────────── RHIG core (single run) ───────────
def run_rhig_from(deploy_time_s: float,
                  v_max=V_MAX, max_imp=MAX_IMP,
                  record_hist=False):
    t0 = deploy_time_s
    Nsteps = int((TOF_SEC - t0)/DT_DYN) + 1

    rho  = RHO_DEPLOY.copy()
    rhod = RHOD_DEPLOY.copy()

    impulses = []
    last_fire = -1e9

    # allocate logs if requested
    if record_hist:
        t_hist     = np.empty(Nsteps)
        sepA_hist  = np.empty((Nsteps,3))
        err_hist   = np.empty(Nsteps)
        rD_hist    = np.empty((Nsteps,3))
        vD_hist    = np.empty((Nsteps,3))
        rA_hist    = np.empty((Nsteps,3))
        vA_hist    = np.empty((Nsteps,3))
        rC_hist    = np.empty((Nsteps,3))
        vC_hist    = np.empty((Nsteps,3))
        CA_hist    = np.empty((Nsteps,3,3))  # C_A_N (LVLH<-N)
        CC_hist    = np.empty((Nsteps,3,3))  # C_C_N (LVLH<-N)
        los_hist   = np.empty((Nsteps,3))    # unit LOS deputy→Apophis (ECI)
        CRN_hist   = np.empty((Nsteps,3,3))  # desired reference DCM N→R
        qRN_hist   = np.empty((Nsteps,4))    # quaternion N→R
        sig_hist   = np.empty((Nsteps,3))    # MRPs N→R
        dv_times   = []
        dv_list    = []

    for j in range(Nsteps):
        t_now = t0 + j*DT_DYN

        rC_now, vC_now = chief_state(t_now)
        rA_now, vA_now = rA_fn(t_now), vA_fn(t_now)
        C_C_now = dcm_lvlh(rC_now, vC_now)  # LVLH_C<-N
        C_A_now = dcm_lvlh(rA_now, vA_now)  # LVLH_A<-N
        n_now   = hill_rate(rC_now, vC_now)

        rD_now = rC_now + C_C_now.T @ rho
        vD_now = vC_now + C_C_now.T @ (rhod + np.cross(np.array([0.0,0.0,n_now]), rho))

        sepA_pred, S = cw_map_to_ApophisLVLH(rho, rhod, t_now)
        errA = sepA_pred - APOPHIS_TARGET

        # logs
        if record_hist:
            t_hist[j]    = t_now
            sepA_hist[j] = C_A_now @ (rD_now - rA_now)
            err_hist[j]  = la.norm(errA)
            rD_hist[j]   = rD_now
            vD_hist[j]   = vD_now
            rA_hist[j]   = rA_now
            vA_hist[j]   = vA_now
            rC_hist[j]   = rC_now
            vC_hist[j]   = vC_now
            CA_hist[j]   = C_A_now
            CC_hist[j]   = C_C_now

            # reference attitude pointing: LOS + roll from velocity
            los = rA_now - rD_now
            los_hat = normalize(los)[0]
            C_NB_des = triad_from_boresight_and_tangent(los_hat, vD_now)
            C_RN     = C_NB_des.T             # desired R (body) relative to N
            q_RN     = dcm_to_quat(C_RN)
            sigma    = quat_to_mrp(q_RN)

            los_hist[j] = los_hat
            CRN_hist[j] = C_RN
            qRN_hist[j] = q_RN
            sig_hist[j] = sigma

        # guidance cadence
        can_fire = (
            (j % int(DT_FSW/DT_DYN) == 0) and
            (TOF_SEC - t_now > MIN_LOOK) and
            ((t_now - last_fire) >= COOLDOWN) and
            (len(impulses) < max_imp)
        )

        if can_fire and la.norm(errA) > E_TRIGGER:
            # CW-based Δv (RTN)
            dv_rtn_cw = damped_min_norm(S, errA)

            # optional Lambert nudge
            dv_rtn_nudge = np.zeros(3)
            if USE_LAMBERT_NUDGE:
                rA_final, vA_final = rA_fn(TOF_SEC), vA_fn(TOF_SEC)
                C_A_final = dcm_lvlh(rA_final, vA_final)
                r_goal = rA_final + C_A_final.T @ APOPHIS_TARGET
                if angle_deg(rD_now, r_goal) >= ALIGN_DEG_MIN:
                    dv_eci = lambert_delta_v_now(rD_now, vD_now, r_goal, TOF_SEC - t_now)
                    mag = la.norm(dv_eci)
                    if mag > NUDGE_SAT: dv_eci *= (NUDGE_SAT/mag)
                    dv_rtn_nudge = C_C_now @ dv_eci

            dv_mix = (1.0 - NUDGE_WEIGHT)*dv_rtn_cw + NUDGE_WEIGHT*dv_rtn_nudge

            # line search
            base = la.norm(errA); chosen = None
            for scl in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                dv_try = dv_mix * scl
                m = la.norm(dv_try)
                if m < V_MIN: continue
                if m > v_max: dv_try *= v_max/m
                sep_new, _ = cw_map_to_ApophisLVLH(rho, rhod + dv_try, t_now)
                if la.norm(sep_new - APOPHIS_TARGET) < base - IMPROVE_MIN:
                    chosen = dv_try
                    break

            if chosen is not None:
                rhod += chosen
                impulses.append((t_now, chosen.copy()))
                last_fire = t_now
                if record_hist:
                    dv_times.append(t_now); dv_list.append(chosen.copy())

        # 1 s HCW propagation
        Phi_rr, Phi_rv, Phi_vr, Phi_vv = cw_blocks(n_now, DT_DYN)
        rho, rhod = Phi_rr @ rho + Phi_rv @ rhod, Phi_vr @ rho + Phi_vv @ rhod

    # terminal results
    rA_end, vA_end = rA_fn(TOF_SEC), vA_fn(TOF_SEC); C_A_end = dcm_lvlh(rA_end, vA_end)
    rC_end, vC_end = chief_state(TOF_SEC);           C_C_end = dcm_lvlh(rC_end, vC_end)
    rD_end = rC_end + C_C_end.T @ rho
    sepA_final = C_A_end @ (rD_end - rA_end)
    e_final = sepA_final - APOPHIS_TARGET
    tot_dv = sum(la.norm(dv) for _, dv in impulses)

    out = dict(
        err_norm=float(la.norm(e_final)),
        e_final=e_final, sep_final=sepA_final,
        tot_dv=tot_dv, impulses=impulses
    )
    if record_hist:
        out.update(dict(
            t_hist=t_hist, sepA_hist=sepA_hist, err_hist=err_hist,
            rD_hist=rD_hist, vD_hist=vD_hist,
            rA_hist=rA_hist, vA_hist=vA_hist,
            rC_hist=rC_hist, vC_hist=vC_hist,
            C_A_N_hist=CA_hist, C_C_N_hist=CC_hist,
            bRef_N_hist=los_hist,
            C_RN_hist=CRN_hist, q_RN_hist=qRN_hist, sigma_RN_hist=sig_hist,
            dv_times=np.array(dv_times, dtype=float),
            dv_list=np.array(dv_list) if dv_list else np.zeros((0,3)),
            t0=t0))
    return out

# ───────────────────────── deploy-time search ──────────────
def eval_cost(res):  # ||e|| + λ·ΔV
    return res["err_norm"] + LAMBDA_DV * res["tot_dv"]

tmin = SEARCH_TMIN
tmax = max(0.0, TOF_SEC - SEARCH_TMAX_PAD)
grid = np.arange(tmin, tmax+1e-6, COARSE_STEP)

best = None; best_t = None
print(f"\n[Search] Coarse deploy scan: {len(grid)} points {tmin/3600:.2f}h → {tmax/3600:.2f}h")
for t_dep in grid:
    res = run_rhig_from(t_dep, v_max=V_MAX, max_imp=MAX_IMP, record_hist=False)
    if (best is None) or (eval_cost(res) < eval_cost(best)):
        best, best_t = res, t_dep
print(f"[Search] Coarse best: t_dep={best_t/3600:.2f} h  |e|={best['err_norm']:.2f} m  ΔV={best['tot_dv']:.3f} m/s")

def refine_around(t_center, step):
    t_lo = max(tmin, t_center - REFINE_SPAN)
    t_hi = min(tmax, t_center + REFINE_SPAN)
    return np.arange(t_lo, t_hi+1e-6, step)

for step in (REFINE_STEP_1, REFINE_STEP_2):
    grid2 = refine_around(best_t, step)
    print(f"[Search] Refine {step:.0f}s: {len(grid2)} points around {best_t/3600:.2f}h")
    for t_dep in grid2:
        res = run_rhig_from(t_dep, v_max=V_MAX, max_imp=MAX_IMP, record_hist=False)
        if eval_cost(res) < eval_cost(best):
            best, best_t = res, t_dep
    print(f"[Search]  → best so far: t_dep={best_t/3600:.2f} h  |e|={best['err_norm']:.2f} m  ΔV={best['tot_dv']:.3f} m/s")

# ───────────────────────── final run at best deploy ─────────
print("\n[Final] Re-running at optimized deploy time with higher authority…")
final_res = run_rhig_from(best_t, v_max=FINAL_V_MAX, max_imp=FINAL_MAX_IMP, record_hist=True)

# ───────────────────────── reporting ─────────────────────────
deploy_dt = departure_dt + timedelta(seconds=float(best_t))
print("\n------ RHIG Rendezvous Summary (Apophis LVLH) ------")
print("Departure UTC:", departure_dt.isoformat().replace("+00:00","Z"))
print("Deploy    UTC:", deploy_dt.isoformat().replace("+00:00","Z"),
      f"(+{best_t:.1f} s)")
print("Arrival   UTC:", arrival_dt.isoformat().replace("+00:00","Z"))
print(f"TOF [h]: {TOF_SEC/3600.0:.3f}  ({TOF_SEC:.1f} s total; {TOF_SEC-best_t:.1f} s after deploy)")
print("Target   (km):", APOPHIS_TARGET/1e3)
print("Achieved (km):", final_res["sep_final"]/1e3)
print("Final error (m):", final_res["e_final"], "  ||e||=", final_res["err_norm"])
print(f"Impulses: {len(final_res['impulses'])}   Total ΔV: {final_res['tot_dv']:.6f} m/s")
for t, dv in final_res["impulses"]:
    print(f"  t={t:9.1f} s  Δv_RTΝ = [{dv[0]*1e3:8.3f} {dv[1]*1e3:8.3f} {dv[2]*1e3:8.3f}] mm/s")

# ───────────────────────── export to NPZ ────────────────────
def export_npz(path: str, res: dict, t0: float):
    t      = res["t_hist"]                          # (N,)
    rD_N   = res["rD_hist"];   vD_N  = res["vD_hist"]
    rA_N   = res["rA_hist"];   vA_N  = res["vA_hist"]
    rC_N   = res["rC_hist"];   vC_N  = res["vC_hist"]
    C_A_N  = res["C_A_N_hist"]                      # LVLH_A<-N
    C_C_N  = res["C_C_N_hist"]                      # LVLH_C<-N
    bRef_N = res["bRef_N_hist"]                     # unit LOS
    C_RN   = res["C_RN_hist"]                       # N→R
    q_RN   = res["q_RN_hist"]                       # [q0,q1,q2,q3]
    sigma  = res["sigma_RN_hist"]                   # MRPs
    camB   = CAM_BORESIGHT_B

    np.savez_compressed(
        path,
        t=t,
        rD_N=rD_N, vD_N=vD_N,
        rA_N=rA_N, vA_N=vA_N,
        rC_N=rC_N, vC_N=vC_N,
        C_A_N=C_A_N, C_C_N=C_C_N,
        bRef_N=bRef_N,
        C_RN=C_RN, q_RN=q_RN, sigma_RN=sigma,
        camMount_B=camB,
        meta=np.array([DEPARTURE_UTC_ISO, arrival_dt.isoformat().replace("+00:00","Z")], dtype=object),
        t0=np.array([t0], dtype=float)
    )
    print(f"\n[Export] Wrote rendezvous data to: {os.path.abspath(path)}")
    print("         Keys: t, rD_N, vD_N, rA_N, vA_N, rC_N, vC_N, C_A_N, C_C_N, bRef_N, C_RN, q_RN, sigma_RN, camMount_B, meta, t0")

export_npz(NPZ_EXPORT_PATH, final_res, t0=final_res["t0"])

# ───────────────────────── visualization (delegated to Visualize.py) ────────────────────
import Visualize as viz

viz.plot_all(
    final_res,
    best_t=best_t,
    DT_DYN=DT_DYN,
    APOPHIS_TARGET=APOPHIS_TARGET,
    TARGET_MARGIN_M=TARGET_MARGIN_M,
    E_TRIGGER=E_TRIGGER,
    V_MIN=V_MIN,
    V_MAX=V_MAX,
    DV_BUDGET_MS=DV_BUDGET_MS,
    R_EARTH_KM=R_EARTH_KM,
    save_dir=SAVE_PNG_DIR,     # set to None to avoid saving PNGs
    font_size=50,
    tick_size=45,
    npz_path_for_nav=NAV_NPZ_PATH   # provide an NPZ with rNavD_N, vNavD_N to enable nav-error plots
)
