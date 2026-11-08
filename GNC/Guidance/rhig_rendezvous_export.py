#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_rhig_rendezvous_export.py
=================================

Generic RHIG + NPZ exporter for a single deputy.

• Uses a chief reference trajectory (any orbit) from chief_reference.npz.
• Uses Basilisk SPICE to sample a target body (default: (99942) Apophis)
  geocentrically or relative to some central body.
• Runs a CW/Hill-based rendezvous guidance loop (RHIG) with optional
  Lambert "nudge" toward a final LVLH stand-off point in target-centred LVLH.
• Searches over deploy time to balance final error and ?V.
• Exports a **RendezvousBundle NPZ** ready to drive 6-DOF attitude / camera
  sims and plotting tools.

You can retarget by changing:
  - DEPARTURE_UTC_ISO
  - CENTRAL_BODY_NAME / TARGET_BODY_NAME
  - TARGET_SPK_NAME
  - TARGET_LVLH_M
"""

from __future__ import annotations
import os, math, warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

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

# ????????????????????????????????????????????????????????????????????????
# 1) CONFIGURATION (edit to retarget)
# ????????????????????????????????????????????????????????????????????????

# Chief trajectory NPZ (km, km/s)
CHIEF_REF_NPZ = Path(os.getenv("CHIEF_REF_NPZ", "chief_reference.npz"))

# Absolute epoch to anchor SPICE run (must match chief_reference epoch)
DEPARTURE_UTC_ISO = os.getenv("DEPARTURE_UTC_ISO", "2029-04-13T05:41:45Z")
departure_dt = datetime.strptime(DEPARTURE_UTC_ISO, "%Y-%m-%dT%H:%M:%SZ").replace(
    tzinfo=timezone.utc
)

# Central body and target body for SPICE
CENTRAL_BODY_NAME = os.getenv("CENTRAL_BODY_NAME", "Earth")
TARGET_BODY_NAME  = os.getenv("TARGET_BODY_NAME",  "20099942")   # NAIF name / ID string
TARGET_SPK_NAME   = os.getenv("TARGET_SPK_NAME",   "20099942.bsp")

# Desired stand-off in target-centred LVLH (R,T,N) [m]
TARGET_LVLH_M = np.array([1.43, 2.55, -0.27]) * 1e3   # default Apophis example

# Initial relative state at deploy (chief LVLH)
RHO_DEPLOY  = np.zeros(3)
RHOD_DEPLOY = np.zeros(3)

# RHIG guidance parameters
DT_DYN   = 1.0     # dynamics step [s]
DT_FSW   = 60.0    # guidance cadence [s]
E_TRIGGER= 5.0     # [m]
IMPROVE_MIN = 3.0  # [m]
V_MIN    = 0.05e-3 # [m/s]
V_MAX    = 0.1     # [m/s]
MIN_LOOK = 250.0   # [s] min time-to-go before we stop burning
COOLDOWN = 250.0   # [s] minimum time between burns
MAX_IMP  = 60      # max number of impulses
ALPHA_DAMP_BASE = 1e-7

# Optional Lambert “nudge”
USE_LAMBERT_NUDGE = True
NUDGE_WEIGHT      = 0.4
NUDGE_SAT         = 0.4    # [m/s]
ALIGN_DEG_MIN     = 1.0    # skip Lambert nudge if angle(r_now, r_goal) < 1°

# Deploy-time search (error + ?·?V)
LAMBDA_DV       = 380.0
SEARCH_TMIN     = 1500.0          # [s] earliest deploy
SEARCH_TMAX_PAD = 1.2*3600.0      # [s] keep at least this much time after deploy
COARSE_STEP     = 900.0           # [s] coarse grid
REFINE_STEP_1   = 150.0
REFINE_STEP_2   = 60.0
REFINE_SPAN     = 1.2*3600.0      # [s] refinement window
FINAL_V_MAX     = 0.5             # [m/s]
FINAL_MAX_IMP   = 80

# Plotting / export
TARGET_MARGIN_M = 1.0             # [m] LVLH margin band in Visualize
DV_BUDGET_MS    = None            # [m/s] ?V budget line; None disables
SAVE_PNG_DIR    = None            # e.g. "plots" to save PNGs; None => no save
NPZ_EXPORT_PATH = "rendezvous_export.npz"
CAM_BORESIGHT_B = np.array([0.0, 0.0, 1.0])  # +Z_B is camera axis

NAV_NPZ_PATH    = None            # optional nav NPZ for Visualize

# SPICE setup
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
KERNELS = [
    "naif0012.tls",
    "pck00010.tpc",
    "de-403-masses.tpc",
    "de430.bsp",
    TARGET_SPK_NAME,
]

# ????????????????????????????????????????????????????????????????????????
# 2) BASIC HELPERS
# ????????????????????????????????????????????????????????????????????????

def dcm_lvlh(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """LVLH<-ECI DCM; rows = [R?; T?; N?] in inertial components."""
    i_hat = r / la.norm(r)
    k_hat = np.cross(r, v); k_hat /= la.norm(k_hat)
    j_hat = np.cross(k_hat, i_hat)
    return np.vstack((i_hat, j_hat, k_hat))

def hill_rate(r: np.ndarray, v: np.ndarray) -> float:
    return la.norm(np.cross(r, v)) / (la.norm(r)**2)

def cw_blocks(n: float, dt: float):
    c = math.cos(n*dt); s = math.sin(n*dt); nt = n*dt
    Phi_rr = np.array([[4-3*c, 0, 0],[6*(s-nt), 1, 0],[0,0,c]])
    Phi_rv = np.array([[s/n, 2*(1-c)/n, 0],
                       [-2*(1-c)/n, (4*s-3*nt)/n, 0],
                       [0,0,s/n]])
    Phi_vr = np.array([[3*n*s,0,0],[6*n*(c-1),0,0],[0,0,-n*s]])
    Phi_vv = np.array([[c,2*s,0],[-2*s,4*c-3,0],[0,0,c]])
    return Phi_rr, Phi_rv, Phi_vr, Phi_vv

def angle_deg(a, b):
    na, nb = la.norm(a), la.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    c = np.clip(np.dot(a,b)/(na*nb), -1.0, 1.0)
    return np.degrees(math.acos(c))

def normalize(v, eps=1e-12):
    n = la.norm(v)
    if n < eps:
        return np.zeros_like(v), 0.0
    return v/n, n

def triad_from_boresight_and_tangent(los_N: np.ndarray, vel_N: np.ndarray) -> np.ndarray:
    """
    Two-vector TRIAD for reference body frame:

      z_B := LOS (deputy?target)
      x_B := vel projected onto plane ? z_B (fallback to h?)
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
    """Active DCM ? scalar-first quaternion [q0,q1,q2,q3]."""
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
    """Unit quaternion [q0,q1,q2,q3] ? MRPs ?."""
    q0, qv = q[0], q[1:]
    denom = 1.0 + q0
    if denom < 1e-12:
        denom = 1.0 - q0
        return -qv / max(denom, 1e-12)
    return qv / denom

# ????????????????????????????????????????????????????????????????????????
# 3) CHIEF REFERENCE & TARGET SPICE SAMPLER
# ????????????????????????????????????????????????????????????????????????

def load_chief_npz(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"chief reference NPZ not found: {path}")
    data = np.load(path)
    T, R, V = data["times"], data["positions"], data["velocities"]  # s, km, km/s
    TOF_SEC = float(T[-1])

    def chief_state(t_s: float) -> tuple[np.ndarray, np.ndarray]:
        r_km = np.array([np.interp(t_s, T, R[:,k]) for k in range(3)])
        v_km = np.array([np.interp(t_s, T, V[:,k]) for k in range(3)])
        return r_km*1e3, v_km*1e3     # [m], [m/s]
    return TOF_SEC, chief_state

TOF_SEC, chief_state = load_chief_npz(CHIEF_REF_NPZ)
arrival_dt = departure_dt + timedelta(seconds=TOF_SEC)

def make_target_sampler(departure_dt: datetime,
                        *, duration_s: int, step_s: int = 60):
    """
    Build geocentric target sampler rT(t), vT(t) using SPICE via Basilisk.
    """
    sim  = SimulationBaseClass.SimBaseClass()
    task = sim.CreateNewTask("tsk", macros.sec2nano(step_s))
    sim.CreateNewProcess("dyn").addTask(task)

    spi = spiceInterface.SpiceInterface()
    spi.SPICEDataPath = SPICE_DIR
    spi.UTCCalInit    = departure_dt.strftime("%Y %B %d, %H:%M:%S.0 (UTC)")
    for k in KERNELS:
        spi.loadSpiceKernel(k, SPICE_DIR)
    spi.addPlanetNames([CENTRAL_BODY_NAME, "Sun", TARGET_BODY_NAME])
    spi.zeroBase = "Sun"
    sim.AddModelToTask("tsk", spi)

    conv = ephemerisConverter.EphemerisConverter()
    for m in spi.planetStateOutMsgs:
        conv.addSpiceInputMsg(m)
    sim.AddModelToTask("tsk", conv)

    eRec = conv.ephemOutMsgs[0].recorder()  # central wrt Sun
    tRec = conv.ephemOutMsgs[2].recorder()  # target wrt Sun
    sim.AddModelToTask("tsk", eRec)
    sim.AddModelToTask("tsk", tRec)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(duration_s))
    sim.ExecuteSimulation()

    N      = int(np.floor(duration_s/step_s)) + 1
    t_grid = np.arange(N, dtype=float) * step_s
    rT_S   = tRec.r_BdyZero_N[:N]
    vT_S   = tRec.v_BdyZero_N[:N]
    rC_S   = eRec.r_BdyZero_N[:N]
    vC_S   = eRec.v_BdyZero_N[:N]

    r_geo = rT_S - rC_S
    v_geo = vT_S - vC_S

    def interp(arr, t):
        if t <= 0.0: return arr[0]
        if t >= t_grid[-1]: return arr[-1]
        u = t / step_s
        i = int(np.floor(u)); f = u - i
        return (1.0 - f) * arr[i] + f * arr[i+1]

    def rT(t_sec): return interp(r_geo, float(t_sec))
    def vT(t_sec): return interp(v_geo, float(t_sec))
    return rT, vT

rT_fn, vT_fn = make_target_sampler(departure_dt,
                                   duration_s=int(TOF_SEC + 3600),
                                   step_s=60)

# Sanity print at arrival
rC_end, vC_end = chief_state(TOF_SEC)
rT_end, vT_end = rT_fn(TOF_SEC), vT_fn(TOF_SEC)
C_T_end = dcm_lvlh(rT_end, vT_end)
sep_check = C_T_end @ (rC_end - rT_end)
print("[Timeline]")
print("  Departure UTC:", departure_dt.isoformat().replace("+00:00","Z"))
print("  Arrival   UTC:", arrival_dt.isoformat().replace("+00:00","Z"))
print(f"  TOF             : {TOF_SEC/3600.0:.2f} h  ({TOF_SEC:.1f} s)")
print("[Sanity] Chief–target @Arrival (target LVLH): "
      f"R={sep_check[0]/1e3:.3f} km, T={sep_check[1]/1e3:.3f} km, N={sep_check[2]/1e3:.3f} km, "
      f"|sep|={la.norm(sep_check)/1e3:.3f} km")

# ????????????????????????????????????????????????????????????????????????
# 4) RHIG MAPS & LAMBERT NUDGE
# ????????????????????????????????????????????????????????????????????????

def cw_map_to_target_lvlh(rho: np.ndarray, rhod: np.ndarray, t_now: float):
    """
    Map current chief LVLH state (rho,rhod) at t_now to predicted terminal
    separation in target-centred LVLH at TOF_SEC, plus sensitivity S.
    """
    rC_now, vC_now = chief_state(t_now)
    n = hill_rate(rC_now, vC_now)
    dt = TOF_SEC - t_now

    Phi_rr, Phi_rv, _, _ = cw_blocks(n, dt)
    rho_f = Phi_rr @ rho + Phi_rv @ rhod

    rC_f, vC_f = chief_state(TOF_SEC)
    rT_f, vT_f = rT_fn(TOF_SEC), vT_fn(TOF_SEC)
    C_C_f      = dcm_lvlh(rC_f, vC_f)
    C_T_f      = dcm_lvlh(rT_f, vT_f)

    rD_f   = rC_f + C_C_f.T @ rho_f
    sep_T  = C_T_f @ (rD_f - rT_f)
    S      = C_T_f @ C_C_f.T @ Phi_rv
    return sep_T, S

def damped_min_norm(S: np.ndarray, e: np.ndarray, alpha_base=ALPHA_DAMP_BASE) -> np.ndarray:
    U, s, Vt = svd(S, full_matrices=False)
    alpha = alpha_base + (0.1*(s[-1]**2) if s.size else alpha_base)
    A = S.T @ S + alpha*np.eye(S.shape[1])
    b = -S.T @ e
    return la.solve(A, b)

def lambert_delta_v_now(r_now_m: np.ndarray, v_now_mps: np.ndarray,
                        r_goal_m: np.ndarray, tof_s: float) -> np.ndarray:
    """
    Small Basilisk lambertPlanner+lambertSolver wrapper:
    returns ?v = v_req - v_now in inertial (SI).
    """
    sim = SimulationBaseClass.SimBaseClass()
    dynT = sim.CreateNewTask("dynT", macros.sec2nano(0.01))
    fswT = sim.CreateNewTask("fswT", macros.sec2nano(0.01))
    sim.CreateNewProcess("dyn").addTask(dynT)
    sim.CreateNewProcess("fsw").addTask(fswT)

    mu_m = gravBodyFactory().createEarth().mu

    sc = spacecraft.Spacecraft()
    sc.hub.r_CN_NInit = r_now_m
    sc.hub.v_CN_NInit = v_now_mps
    nav = simpleNav.SimpleNav()
    nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    sim.AddModelToTask("dynT", sc)
    sim.AddModelToTask("dynT", nav)

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
    sim.AddModelToTask("fswT", planner)
    sim.AddModelToTask("fswT", solver)
    sim.AddModelToTask("fswT", solRec)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(1.0))
    sim.ExecuteSimulation()

    v_req = solRec.v1_N[-1]
    return v_req - v_now_mps

# ????????????????????????????????????????????????????????????????????????
# 5) RHIG CORE
# ????????????????????????????????????????????????????????????????????????

def run_rhig_from(deploy_time_s: float,
                  v_max=V_MAX, max_imp=MAX_IMP,
                  record_hist: bool = False) -> dict:
    """
    Run RHIG from a given deploy time (in the chief epoch frame).
    Returns result dict with final error, ?V, impulses, and (optionally) histories.
    """
    t0 = deploy_time_s
    Nsteps = int((TOF_SEC - t0)/DT_DYN) + 1

    rho  = RHO_DEPLOY.copy()
    rhod = RHOD_DEPLOY.copy()

    impulses = []
    last_fire = -1e9

    if record_hist:
        t_hist     = np.empty(Nsteps)
        sep_hist   = np.empty((Nsteps,3))
        err_hist   = np.empty(Nsteps)
        rD_hist    = np.empty((Nsteps,3))
        vD_hist    = np.empty((Nsteps,3))
        rT_hist    = np.empty((Nsteps,3))
        vT_hist    = np.empty((Nsteps,3))
        rC_hist    = np.empty((Nsteps,3))
        vC_hist    = np.empty((Nsteps,3))
        CT_hist    = np.empty((Nsteps,3,3))
        CC_hist    = np.empty((Nsteps,3,3))
        los_hist   = np.empty((Nsteps,3))
        CRN_hist   = np.empty((Nsteps,3,3))
        qRN_hist   = np.empty((Nsteps,4))
        sig_hist   = np.empty((Nsteps,3))
        dv_times   = []
        dv_list    = []

    for j in range(Nsteps):
        t_now = t0 + j*DT_DYN

        rC_now, vC_now = chief_state(t_now)
        rT_now, vT_now = rT_fn(t_now), vT_fn(t_now)

        C_C_now = dcm_lvlh(rC_now, vC_now)
        C_T_now = dcm_lvlh(rT_now, vT_now)
        n_now   = hill_rate(rC_now, vC_now)

        rD_now = rC_now + C_C_now.T @ rho
        vD_now = vC_now + C_C_now.T @ (rhod + np.cross(np.array([0.0,0.0,n_now]), rho))

        sep_pred, S = cw_map_to_target_lvlh(rho, rhod, t_now)
        err_vec = sep_pred - TARGET_LVLH_M

        if record_hist:
            t_hist[j]    = t_now
            sep_hist[j]  = C_T_now @ (rD_now - rT_now)
            err_hist[j]  = la.norm(err_vec)
            rD_hist[j]   = rD_now
            vD_hist[j]   = vD_now
            rT_hist[j]   = rT_now
            vT_hist[j]   = vT_now
            rC_hist[j]   = rC_now
            vC_hist[j]   = vC_now
            CT_hist[j]   = C_T_now
            CC_hist[j]   = C_C_now

            los = rT_now - rD_now
            los_hat = normalize(los)[0]
            C_NB_des = triad_from_boresight_and_tangent(los_hat, vD_now)
            C_RN     = C_NB_des.T
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

        if can_fire and la.norm(err_vec) > E_TRIGGER:
            dv_rtn_cw = damped_min_norm(S, err_vec)
            dv_rtn_nudge = np.zeros(3)

            if USE_LAMBERT_NUDGE:
                rT_final, vT_final = rT_fn(TOF_SEC), vT_fn(TOF_SEC)
                C_T_final = dcm_lvlh(rT_final, vT_final)
                r_goal = rT_final + C_T_final.T @ TARGET_LVLH_M
                if angle_deg(rD_now, r_goal) >= ALIGN_DEG_MIN:
                    dv_eci = lambert_delta_v_now(rD_now, vD_now, r_goal, TOF_SEC - t_now)
                    mag = la.norm(dv_eci)
                    if mag > NUDGE_SAT:
                        dv_eci *= (NUDGE_SAT/mag)
                    dv_rtn_nudge = C_C_now @ dv_eci

            dv_mix = (1.0 - NUDGE_WEIGHT)*dv_rtn_cw + NUDGE_WEIGHT*dv_rtn_nudge

            base = la.norm(err_vec)
            chosen = None
            for scl in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                dv_try = dv_mix * scl
                m = la.norm(dv_try)
                if m < V_MIN:
                    continue
                if m > v_max:
                    dv_try *= v_max/m
                sep_new, _ = cw_map_to_target_lvlh(rho, rhod + dv_try, t_now)
                if la.norm(sep_new - TARGET_LVLH_M) < base - IMPROVE_MIN:
                    chosen = dv_try
                    break

            if chosen is not None:
                rhod += chosen
                impulses.append((t_now, chosen.copy()))
                last_fire = t_now
                if record_hist:
                    dv_times.append(t_now)
                    dv_list.append(chosen.copy())

        # 1 s HCW propagation
        Phi_rr, Phi_rv, Phi_vr, Phi_vv = cw_blocks(n_now, DT_DYN)
        rho, rhod = Phi_rr @ rho + Phi_rv @ rhod, Phi_vr @ rho + Phi_vv @ rhod

    # terminal state
    rT_end, vT_end = rT_fn(TOF_SEC), vT_fn(TOF_SEC)
    C_T_end        = dcm_lvlh(rT_end, vT_end)
    rC_end, vC_end = chief_state(TOF_SEC)
    C_C_end        = dcm_lvlh(rC_end, vC_end)
    rD_end         = rC_end + C_C_end.T @ rho
    sep_end_T      = C_T_end @ (rD_end - rT_end)
    e_final        = sep_end_T - TARGET_LVLH_M
    tot_dv         = sum(la.norm(dv) for _, dv in impulses)

    out = dict(
        err_norm=float(la.norm(e_final)),
        e_final=e_final,
        sep_final=sep_end_T,
        tot_dv=tot_dv,
        impulses=impulses,
    )
    if record_hist:
        out.update(dict(
            t_hist=t_hist,
            sepA_hist=sep_hist,         # for legacy Visualize; still “A”
            err_hist=err_hist,
            rD_hist=rD_hist, vD_hist=vD_hist,
            rA_hist=rT_hist, vA_hist=vT_hist,   # “A” = target
            rC_hist=rC_hist, vC_hist=vC_hist,
            C_A_N_hist=CT_hist, C_C_N_hist=CC_hist,
            bRef_N_hist=los_hist,
            C_RN_hist=CRN_hist, q_RN_hist=qRN_hist, sigma_RN_hist=sig_hist,
            dv_times=np.array(dv_times, dtype=float),
            dv_list=(np.array(dv_list) if dv_list else np.zeros((0,3))),
            t0=t0,
        ))
    return out

def eval_cost(res: dict) -> float:
    return res["err_norm"] + LAMBDA_DV * res["tot_dv"]

# ????????????????????????????????????????????????????????????????????????
# 6) DEPLOY-TIME SEARCH + FINAL RUN
# ????????????????????????????????????????????????????????????????????????

tmin = SEARCH_TMIN
tmax = max(0.0, TOF_SEC - SEARCH_TMAX_PAD)
grid = np.arange(tmin, tmax + 1e-6, COARSE_STEP)

best = None
best_t = None

print(f"\n[Search] Coarse deploy scan: {len(grid)} points "
      f"{tmin/3600:.2f}h ? {tmax/3600:.2f}h")
for t_dep in grid:
    res = run_rhig_from(t_dep, v_max=V_MAX, max_imp=MAX_IMP, record_hist=False)
    if (best is None) or (eval_cost(res) < eval_cost(best)):
        best, best_t = res, t_dep
print(f"[Search] Coarse best: t_dep={best_t/3600:.2f} h  "
      f"|e|={best['err_norm']:.2f} m  ?V={best['tot_dv']:.3f} m/s")

def refine_around(t_center: float, step: float):
    lo = max(tmin, t_center - REFINE_SPAN)
    hi = min(tmax, t_center + REFINE_SPAN)
    return np.arange(lo, hi + 1e-6, step)

for step in (REFINE_STEP_1, REFINE_STEP_2):
    grid2 = refine_around(best_t, step)
    print(f"[Search] Refine {step:.0f}s: {len(grid2)} points around {best_t/3600:.2f}h")
    for t_dep in grid2:
        res = run_rhig_from(t_dep, v_max=V_MAX, max_imp=MAX_IMP, record_hist=False)
        if eval_cost(res) < eval_cost(best):
            best, best_t = res, t_dep
    print(f"[Search]  ? best so far: t_dep={best_t/3600:.2f} h  "
          f"|e|={best['err_norm']:.2f} m  ?V={best['tot_dv']:.3f} m/s")

print("\n[Final] Re-running at optimized deploy time with higher authority…")
final_res = run_rhig_from(best_t, v_max=FINAL_V_MAX,
                          max_imp=FINAL_MAX_IMP, record_hist=True)

deploy_dt = departure_dt + timedelta(seconds=float(best_t))

print("\n------ RHIG Rendezvous Summary (target LVLH) ------")
print("Departure UTC:", departure_dt.isoformat().replace("+00:00","Z"))
print("Deploy    UTC:", deploy_dt.isoformat().replace("+00:00","Z"),
      f"(+{best_t:.1f} s)")
print("Arrival   UTC:", arrival_dt.isoformat().replace("+00:00","Z"))
print(f"TOF [h]: {TOF_SEC/3600.0:.3f}  ({TOF_SEC:.1f} s total; "
      f"{TOF_SEC-best_t:.1f} s after deploy)")
print("Target   (km):", TARGET_LVLH_M/1e3)
print("Achieved (km):", final_res["sep_final"]/1e3)
print("Final error (m):", final_res["e_final"],
      "  ||e||=", final_res["err_norm"])
print(f"Impulses: {len(final_res['impulses'])}   "
      f"Total ?V: {final_res['tot_dv']:.6f} m/s")
for t, dv in final_res["impulses"]:
    print(f"  t={t:9.1f} s  ?v_RT? = "
          f"[{dv[0]*1e3:8.3f} {dv[1]*1e3:8.3f} {dv[2]*1e3:8.3f}] mm/s")

# ????????????????????????????????????????????????????????????????????????
# 7) NPZ EXPORT (RendezvousBundle)
# ????????????????????????????????????????????????????????????????????????

def export_npz(path: str, res: dict, t0: float):
    t      = res["t_hist"]                          # (N,)
    rD_N   = res["rD_hist"];   vD_N  = res["vD_hist"]
    rT_N   = res["rA_hist"];   vT_N  = res["vA_hist"]
    rC_N   = res["rC_hist"];   vC_N  = res["vC_hist"]
    C_T_N  = res["C_A_N_hist"]                      # LVLH_T<-N
    C_C_N  = res["C_C_N_hist"]                      # LVLH_C<-N
    bRef_N = res["bRef_N_hist"]                     # unit LOS deputy?target
    C_RN   = res["C_RN_hist"]                       # N?R
    q_RN   = res["q_RN_hist"]                       # [q0,q1,q2,q3]
    sigma  = res["sigma_RN_hist"]                   # MRPs
    camB   = CAM_BORESIGHT_B

    np.savez_compressed(
        path,
        t=t,
        rD_N=rD_N, vD_N=vD_N,
        rT_N=rT_N, vT_N=vT_N,
        rC_N=rC_N, vC_N=vC_N,
        C_T_N=C_T_N, C_C_N=C_C_N,
        bRef_N=bRef_N,
        C_RN=C_RN, q_RN=q_RN, sigma_RN=sigma,
        camMount_B=camB,
        meta=np.array([DEPARTURE_UTC_ISO,
                       arrival_dt.isoformat().replace("+00:00","Z")],
                      dtype=object),
        t0=np.array([t0], dtype=float),
    )
    print(f"\n[Export] Wrote rendezvous bundle to: {os.path.abspath(path)}")
    print("         Keys: t, rD_N, vD_N, rT_N, vT_N, rC_N, vC_N, "
          "C_T_N, C_C_N, bRef_N, C_RN, q_RN, sigma_RN, camMount_B, meta, t0")

export_npz(NPZ_EXPORT_PATH, final_res, t0=final_res["t0"])

# ????????????????????????????????????????????????????????????????????????
# 8) VISUALIZATION (OPTIONAL)
# ????????????????????????????????????????????????????????????????????????

try:
    import Visualize as viz

    viz.plot_all(
        final_res,
        best_t=best_t,
        DT_DYN=DT_DYN,
        APOPHIS_TARGET=TARGET_LVLH_M,        # name kept for Visualize compat
        TARGET_MARGIN_M=TARGET_MARGIN_M,
        E_TRIGGER=E_TRIGGER,
        V_MIN=V_MIN,
        V_MAX=V_MAX,
        DV_BUDGET_MS=DV_BUDGET_MS,
        R_EARTH_KM=R_EARTH_KM,
        save_dir=SAVE_PNG_DIR,
        font_size=50,
        tick_size=45,
        npz_path_for_nav=NAV_NPZ_PATH,
    )
except ImportError:
    print("[INFO] Visualize.py not found; skipping plots.")

