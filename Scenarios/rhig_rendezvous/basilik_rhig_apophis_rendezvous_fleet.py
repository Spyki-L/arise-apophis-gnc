#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_rhig_apophis_rendezvous_fleet.py

Purpose
-------
Fleet edition of the RHIG rendezvous script: runs the same rendezvous
guidance/optimizer logic for **eight deputies** targeting different points in
Apophis LVLH, with progress bars and standardized NPZ exports suitable for
both detailed post-analysis and lightweight visualization.

For each deputy k, the script produces:

  • a rich bundle:  apophis_rendezvous_full_D{k}.npz
  • a light bundle: viz_inputs_D{k}.npz

These are designed so `Visualize.py` and other post-processing tools can
consume the data without any reconstruction or re-running of the guidance.

Fleet setup
-----------
- Eight deputies with distinct Apophis LVLH targets (in metres):

    DEPUTY_TARGETS_M = [
        [ 1.43,  2.55, -0.27],
        [-3.07, -0.48,  0.09],
        [ 2.39, -1.76,  0.65],
        [ 2.79,  1.18, -0.41],
        [-2.95, -0.72, -0.71],
        [-0.94,  1.83,  1.95],
        [ 1.55, -2.45,  0.68],
        [-0.71,  1.36, -2.30]
    ] × 1e3

  `APOPHIS_TARGET` is set per-deputy from this array.

- Common chief trajectory:
  * Loaded from `chief_reference.npz` with:
        times      (s),
        positions  (km),
        velocities (km/s).
  * Interpolated into SI via `chief_state(t)`.

- Apophis ephemeris:
  * SPICE-based sampler built from:
        naif0012.tls, pck00010.tpc, de-403-masses.tpc, de430.bsp,
        20099942.bsp.
  * Earth–Apophis geocentric states derived by subtracting Earth’s state.

Timeline
--------
- Departure epoch: `DEPARTURE_UTC_ISO` (e.g., 2029-04-13T05:41:45Z).
- Total time of flight `TOF_SEC` inferred from chief ephemeris.
- Deploy time `t_dep` searched per deputy; rendezvous is defined at Apophis
  arrival time.

Guidance / optimizer (per deputy)
---------------------------------
The **mathematics are unchanged** from the single-deputy RHIG script; they are
just applied in a loop for each target.

Key parameters:
- Dynamics cadence:    DT_DYN = 1.0  s
- Guidance cadence:    DT_FSW = 60.0 s
- Error trigger:       E_TRIGGER = 5.0 m
- Minimum improvement: IMPROVE_MIN = 3.0 m
- Burn limits:         V_MIN = 0.05 mm/s, V_MAX = 0.1 m/s
- Look-ahead / cooldown:
    MIN_LOOK = 250 s, COOLDOWN = 250 s
- Impulse limits:      MAX_IMP = 60 (search runs), FINAL_MAX_IMP = 80
- Damping:
    ALPHA_DAMP_BASE for damped least-squares
- Lambert nudge:
    USE_LAMBERT_NUDGE, NUDGE_WEIGHT, NUDGE_SAT, ALIGN_DEG_MIN
- Cost function:
    J = ‖e_final‖ + LAMBDA_DV · Σ|Δv|_

Core logic (`run_rhig_from`):
- State representation:
    ρ(t), ρ̇(t) in chief LVLH via Hill/CW dynamics (`cw_blocks`).
- Mixed-frame terminal mapping:
    `cw_map_to_ApophisLVLH(ρ, ρ̇, t_now)` → (sepA_pred, S)
    where sepA_pred is the predicted terminal separation in **Apophis LVLH**,
    and S is the sensitivity matrix used for damped least-squares.
- Firing rules:
    * On FSW cadence,
    * outside `MIN_LOOK` of arrival,
    * beyond cooldown since last burn,
    * fewer than `max_imp` burns,
    * current separation error norm > `E_TRIGGER`.
- Δv computation:
    * CW-based Δv from `damped_min_norm(S, eA)` in RTN.
    * Optional Lambert nudge (ECI) mapped back to chief LVLH via C_C.
    * Mixed Δv:
          dv_mix = (1 − NUDGE_WEIGHT) dv_cw + NUDGE_WEIGHT dv_nudge.
    * Scalar line search over [1, 0.5, 0.25, 0.125, 0.0625] to ensure
      a minimum error reduction (`IMPROVE_MIN`) while enforcing
      [V_MIN, V_MAX].

Deploy-time search per deputy
-----------------------------
For each deputy index `idx` and target:

1. Coarse search:
   - t_dep ∈ [SEARCH_TMIN, TOF_SEC − SEARCH_TMAX_PAD] in steps of `COARSE_STEP`.
   - For each t_dep, run:
         r = run_rhig_from(t_dep, record_hist=False)
     and pick the best based on:
         J = ‖e_final‖ + LAMBDA_DV · tot_dv.

2. Refinement passes:
   - Around the current best t_dep using steps:
         REFINE_STEP_1, REFINE_STEP_2.
   - Each pass explores a ±REFINE_SPAN window, updating the best cost.

3. Final run:
   - With the best deploy time `best_t`, call:
         final = run_rhig_from(best_t, v_max=FINAL_V_MAX,
                               max_imp=FINAL_MAX_IMP, record_hist=True,
                               show_prog=True, dep_idx=idx)
   - The final run has increased burn authority and a higher impulse count
     limit.

Exports
-------
For each deputy, the script writes two NPZ files:

1. **Rich bundle**: `apophis_rendezvous_full_D{idx}.npz`
   via `export_npz_full(path, final, t0, best_t)`:

   - Time histories:
        t, rD_N, vD_N,
        rA_N, vA_N,
        rC_N, vC_N,
        C_A_N_hist, C_C_N_hist,
        bRef_N_hist (LOS deputy→Apophis),
        C_RN_hist, q_RN_hist, sigma_RN_hist.
   - Maneuver history:
        dv_times, dv_list, impulses (dense [t, dv_R, dv_T, dv_N]).
   - Constants:
        camMount_B = CAM_BORESIGHT_B,
        APOPHIS_TARGET,
        TARGET_MARGIN_M, E_TRIGGER, V_MIN, V_MAX.
   - Bookkeeping:
        t0, best_t, err_norm, tot_dv,
        meta = [DEPARTURE_UTC_ISO, ARRIVAL_UTC_ISO],
        final_res (pickled dict, for convenience).

2. **Viz bundle**: `viz_inputs_D{idx}.npz`
   via `dump_for_visualize(path, final)`:

   - Includes every ndarray in `final` plus:
        impulses (dense),
        scalar wrappers for t0, best_t, err_norm, tot_dv,
        camMount_B, APOPHIS_TARGET,
        TARGET_MARGIN_M, E_TRIGGER, V_MIN, V_MAX,
        DEPARTURE_UTC_ISO, ARRIVAL_UTC_ISO.

   - Designed so `Visualize.py` can produce plots without needing the
     full rich bundle.

Console output and plotting
---------------------------
- Prints a summary per deputy:
    * Best deploy offset,
    * Final ‖e‖ [m], total Δv [m/s], number of impulses,
    * Tables of each burn with times and RTN components in mm/s.
- After processing all deputies, it calls:

      import Visualize as viz

      viz.plot_all(...)

  for each deputy, with parameters:

    - best_t, DT_DYN, target, TARGET_MARGIN_M,
    - E_TRIGGER, V_MIN, V_MAX, DV_BUDGET_MS,
    - R_EARTH_KM,
    - save_dir = SAVE_PNG_DIR_BASE.format(idx) or None,
    - font_size = 50, tick_size = 45,
    - optional nav NPZ: `npz_path_for_nav` if `NAV_NPZ_PATH` is provided,
    - title_prefix = f"Dep {idx}: ",
    - show=False (headless-friendly).

Usage
-----
Prerequisites:

- Basilisk installation with SPICE ephemeris data and a valid Apophis SPK
  (`20099942.bsp`).
- `chief_reference.npz` in the working directory.

Then run:

    python basilik_rhig_apophis_rendezvous_fleet.py

The script will:

- Loop over all 8 deputies and their distinct Apophis LVLH targets.
- For each:
    * Search for a good deploy time balancing final position error and Δv.
    * Run a final RHIG rendezvous with higher authority.
    * Export both rich and viz NPZ bundles.
- Generate optional figures via `Visualize.py` (PNG directories per deputy if
  `SAVE_PNG_DIR_BASE` is not None).
"""


from __future__ import annotations
import os, math, warnings, sys
from datetime import datetime, timezone, timedelta
import numpy as np
import numpy.linalg as la
from scipy.linalg import svd
# ───────────────── optional tqdm ─────────────────
try:                  from tqdm import tqdm
except ImportError:   tqdm = lambda it, **kw: it
# ───────────────── Basilisk boiler-plate ─────────────────
from Basilisk import __path__ as bskPath
from Basilisk.simulation import spiceInterface, ephemerisConverter, simpleNav, spacecraft
from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.fswAlgorithms import lambertPlanner, lambertSolver
from Basilisk.utilities.simIncludeGravBody import gravBodyFactory

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(precision=3, suppress=True)

# ╔════════════════════════════════════════════╗
# ║  2.  Fleet targets (Apophis LVLH, metres)  ║
# ╚════════════════════════════════════════════╝
DEPUTY_TARGETS_M = np.array([
    [ 1.43,  2.55, -0.27], [-3.07, -0.48,  0.09],
    [ 2.39, -1.76,  0.65], [ 2.79,  1.18, -0.41],
    [-2.95, -0.72, -0.71], [-0.94,  1.83,  1.95],
    [ 1.55, -2.45,  0.68], [-0.71,  1.36, -2.30]
]) * 1e3
APOPHIS_TARGET = DEPUTY_TARGETS_M[0].copy()       # overwritten per deputy

# ╔════════════════════════════════════════════╗
# ║  3.  Timeline                              ║
# ╚════════════════════════════════════════════╝
DEPARTURE_UTC_ISO = "2029-04-13T05:41:45Z"
departure_dt = datetime.strptime(DEPARTURE_UTC_ISO, "%Y-%m-%dT%H:%M:%SZ"
                                ).replace(tzinfo=timezone.utc)

# ╔════════════════════════════════════════════╗
# ║  4.  RHIG / optimiser constants            ║
# ╚════════════════════════════════════════════╝
DT_DYN, DT_FSW           = 1.0, 60.0
E_TRIGGER, IMPROVE_MIN   = 5.0, 3.0
V_MIN, V_MAX             = 0.05e-3, 0.1
MIN_LOOK, COOLDOWN       = 250.0, 250.0
MAX_IMP                  = 60
ALPHA_DAMP_BASE          = 1e-7
USE_LAMBERT_NUDGE        = True
NUDGE_WEIGHT, NUDGE_SAT  = 0.4, 0.4
ALIGN_DEG_MIN            = 1.0
LAMBDA_DV                = 380.0
SEARCH_TMIN              = 1500
SEARCH_TMAX_PAD          = 1.2*3600.0
COARSE_STEP              = 900.0
REFINE_STEP_1, REFINE_STEP_2 = 150.0, 60.0
REFINE_SPAN              = 1.2*3600.0
FINAL_V_MAX, FINAL_MAX_IMP    = 0.5, 80
TARGET_MARGIN_M          = 1.0
DV_BUDGET_MS             = None
SAVE_PNG_DIR_BASE        = "plots_D{}"     # None → no PNGs
CAM_BORESIGHT_B          = np.array([0.0, 0.0, 1.0])
NAV_NPZ_PATH             = None            # leave None unless you have nav data

# ╔════════════════════════════════════════════╗
# ║  5.  Constants & SPICE path                ║
# ╚════════════════════════════════════════════╝
try:
    from Basilisk.utilities import astroConstants as ac
    MU_EARTH  = ac.mu_Earth * 1e9
    R_EARTH_KM= ac.req_Earth/1e3
except Exception:
    MU_EARTH, R_EARTH_KM = 3.986004418e14, 6378.137

SPICE_DIR = os.path.join(bskPath[0], "supportData", "EphemerisData") + os.sep
KERNELS   = ["naif0012.tls", "pck00010.tpc", "de-403-masses.tpc",
             "de430.bsp", "20099942.bsp"]

# ╔════════════════════════════════════════════╗
# ║  6.  Helper maths (unchanged)              ║
# ╚════════════════════════════════════════════╝
def dcm_lvlh(r, v):
    i = r/la.norm(r)
    k = np.cross(r,v); k/=la.norm(k)
    j = np.cross(k,i)
    return np.vstack((i,j,k))

def hill_rate(r,v): return la.norm(np.cross(r,v))/la.norm(r)**2
def cw_blocks(n,dt):
    c,s,nt = math.cos(n*dt), math.sin(n*dt), n*dt
    Phi_rr = np.array([[4-3*c,0,0],[6*(s-nt),1,0],[0,0,c]])
    Phi_rv = np.array([[s/n,2*(1-c)/n,0],[-2*(1-c)/n,(4*s-3*nt)/n,0],[0,0,s/n]])
    Phi_vr = np.array([[3*n*s,0,0],[6*n*(c-1),0,0],[0,0,-n*s]])
    Phi_vv = np.array([[c,2*s,0],[-2*s,4*c-3,0],[0,0,c]])
    return Phi_rr,Phi_rv,Phi_vr,Phi_vv

def angle_deg(a,b):
    na,nb=la.norm(a),la.norm(b)
    return 0.0 if na*nb==0 else np.degrees(
        math.acos(np.clip(np.dot(a,b)/(na*nb),-1,1)))

def normalize(v,eps=1e-12):
    n=la.norm(v); return (v/n if n>=eps else np.zeros_like(v)),n

def triad_from_boresight_and_tangent(los,vel):
    zB,_      = normalize(los)
    v_perp    = vel-np.dot(vel,zB)*zB
    xB,nx     = normalize(v_perp)
    if nx<1e-8:
        h= np.cross(los,vel); h_hat,nh=normalize(h)
        tmp=np.array([1,0,0]) if abs(zB[0])<.9 else np.array([0,1,0])
        xB = normalize(np.cross(tmp if nh<1e-8 else h_hat,zB))[0]
    yB = np.cross(zB,xB)
    xB = normalize(np.cross(yB,zB))[0]
    yB = normalize(np.cross(zB,xB))[0]
    return np.column_stack((xB,yB,zB))

def dcm_to_quat(C):
    tr=np.trace(C)
    if tr>0:
        S=math.sqrt(tr+1)*2; q0=.25*S
        q1=(C[2,1]-C[1,2])/S; q2=(C[0,2]-C[2,0])/S; q3=(C[1,0]-C[0,1])/S
    else:
        i=int(np.argmax([C[0,0],C[1,1],C[2,2]]))
        if i==0:
            S=math.sqrt(1+C[0,0]-C[1,1]-C[2,2])*2
            q0=(C[2,1]-C[1,2])/S; q1=.25*S
            q2=(C[0,1]+C[1,0])/S; q3=(C[0,2]+C[2,0])/S
        elif i==1:
            S=math.sqrt(1+C[1,1]-C[0,0]-C[2,2])*2
            q0=(C[0,2]-C[2,0])/S; q1=(C[0,1]+C[1,0])/S
            q2=.25*S;            q3=(C[1,2]+C[2,1])/S
        else:
            S=math.sqrt(1+C[2,2]-C[0,0]-C[1,1])*2
            q0=(C[1,0]-C[0,1])/S
            q1=(C[0,2]+C[2,0])/S; q2=(C[1,2]+C[2,1])/S; q3=.25*S
    return np.array([q0,q1,q2,q3])/la.norm([q0,q1,q2,q3])

def quat_to_mrp(q):
    q0,qv=q[0],q[1:]; return -qv/(1-q0) if 1+q0<1e-12 else qv/(1+q0)

# ╔════════════════════════════════════════════╗
# ║  7.  Chief & Apophis ephemerides           ║
# ╚════════════════════════════════════════════╝
def load_chief_npz():
    d=np.load("chief_reference.npz")
    T,R,V=d["times"],d["positions"],d["velocities"]
    tof=float(T[-1])
    def chief_state(t):
        r=[np.interp(t,T,R[:,k]) for k in range(3)]
        v=[np.interp(t,T,V[:,k]) for k in range(3)]
        return 1e3*np.array(r),1e3*np.array(v)
    return tof,chief_state

TOF_SEC,chief_state=load_chief_npz()
arrival_dt=departure_dt+timedelta(seconds=TOF_SEC)

def make_apophis_sampler(start_dt,duration_s,step_s=60):
    sim=SimulationBaseClass.SimBaseClass()
    task=sim.CreateNewTask("task",macros.sec2nano(step_s))
    sim.CreateNewProcess("dyn").addTask(task)
    spi=spiceInterface.SpiceInterface()
    spi.SPICEDataPath=SPICE_DIR
    spi.UTCCalInit=start_dt.strftime("%Y %B %d, %H:%M:%S.0 (UTC)")
    for k in KERNELS: spi.loadSpiceKernel(k,SPICE_DIR)
    spi.addPlanetNames(["Earth","Sun","20099942"]); spi.zeroBase="Sun"
    sim.AddModelToTask("task",spi)
    conv=ephemerisConverter.EphemerisConverter()
    for m in spi.planetStateOutMsgs: conv.addSpiceInputMsg(m)
    sim.AddModelToTask("task",conv)
    eRec=conv.ephemOutMsgs[0].recorder()
    aRec=conv.ephemOutMsgs[2].recorder()
    sim.AddModelToTask("task",eRec); sim.AddModelToTask("task",aRec)
    sim.InitializeSimulation(); sim.ConfigureStopTime(macros.sec2nano(duration_s))
    sim.ExecuteSimulation()
    N=int(duration_s/step_s)+1
    t_grid=np.arange(N)*step_s
    rA=aRec.r_BdyZero_N[:N]-eRec.r_BdyZero_N[:N]
    vA=aRec.v_BdyZero_N[:N]-eRec.v_BdyZero_N[:N]
    def interp(arr,t):
        if t<=0: return arr[0]
        if t>=t_grid[-1]: return arr[-1]
        u=t/step_s; i=int(u); f=u-i
        return (1-f)*arr[i]+f*arr[i+1]
    return lambda t:interp(rA,t),lambda t:interp(vA,t)

rA_fn,vA_fn=make_apophis_sampler(departure_dt,int(TOF_SEC+3600),60)

# ╔════════════════════════════════════════════╗
# ║  8.  Mixed-frame helpers                   ║
# ╚════════════════════════════════════════════╝
def cw_map_to_ApophisLVLH(rho,rhod,t_now):
    rC,vC   = chief_state(t_now)
    n       = hill_rate(rC,vC)
    Phi_rr,Phi_rv,_,_=cw_blocks(n,TOF_SEC-t_now)
    rho_f   =Phi_rr@rho+Phi_rv@rhod
    rC_f,vC_f=chief_state(TOF_SEC)
    rA_f,vA_f=rA_fn(TOF_SEC),vA_fn(TOF_SEC)
    C_C_f=dcm_lvlh(rC_f,vC_f); C_A_f=dcm_lvlh(rA_f,vA_f)
    rD_f=rC_f+C_C_f.T@rho_f
    sepA =C_A_f@(rD_f-rA_f)
    S    =C_A_f@C_C_f.T@Phi_rv
    return sepA,S

def damped_min_norm(S,eA):
    _,s,_=svd(S,full_matrices=False)
    alpha=ALPHA_DAMP_BASE+(0.1*s[-1]**2 if s.size else ALPHA_DAMP_BASE)
    return la.solve(S.T@S+alpha*np.eye(S.shape[1]),-S.T@eA)

def lambert_delta_v_now(r_now,v_now,r_goal,tof):
    sim=SimulationBaseClass.SimBaseClass()
    dynT=sim.CreateNewTask("dynT",macros.sec2nano(0.01))
    fswT=sim.CreateNewTask("fswT",macros.sec2nano(0.01))
    sim.CreateNewProcess("dyn").addTask(dynT)
    sim.CreateNewProcess("fsw").addTask(fswT)
    mu=gravBodyFactory().createEarth().mu
    sc=spacecraft.Spacecraft(); sc.hub.r_CN_NInit=r_now; sc.hub.v_CN_NInit=v_now
    nav=simpleNav.SimpleNav(); nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    sim.AddModelToTask("dynT",sc); sim.AddModelToTask("dynT",nav)
    planner=lambertPlanner.LambertPlanner()
    planner.navTransInMsg.subscribeTo(nav.transOutMsg)
    planner.setR_TN_N(r_goal); planner.setMu(mu)
    planner.setNumRevolutions(0); planner.setManeuverTime(10); planner.setFinalTime(10+tof)
    solver=lambertSolver.LambertSolver()
    solver.lambertProblemInMsg.subscribeTo(planner.lambertProblemOutMsg)
    solRec=solver.lambertSolutionOutMsg.recorder()
    sim.AddModelToTask("fswT",planner); sim.AddModelToTask("fswT",solver)
    sim.AddModelToTask("fswT",solRec)
    sim.InitializeSimulation(); sim.ConfigureStopTime(macros.sec2nano(1))
    sim.ExecuteSimulation()
    return solRec.v1_N[-1]-v_now

# ╔════════════════════════════════════════════╗
# ║  9.  RHIG core                             ║
# ╚════════════════════════════════════════════╝
def run_rhig_from(deploy_time_s,*,v_max=V_MAX,max_imp=MAX_IMP,
                  record_hist=False,show_prog=False,dep_idx=None):
    t0=deploy_time_s
    Nsteps=int((TOF_SEC-t0)/DT_DYN)+1
    rho=np.zeros(3); rhod=np.zeros(3)
    impulses=[]; last_fire=-1e9

    if record_hist:
        t_hist       =np.empty(Nsteps)
        sep_hist     =np.empty((Nsteps,3)); err_hist=np.empty(Nsteps)
        rD_hist=vD_hist=rA_hist=vA_hist=rC_hist=vC_hist=None
        rD_hist=np.empty((Nsteps,3)); vD_hist=np.empty((Nsteps,3))
        rA_hist=np.empty((Nsteps,3)); vA_hist=np.empty((Nsteps,3))
        rC_hist=np.empty((Nsteps,3)); vC_hist=np.empty((Nsteps,3))
        CA_hist=np.empty((Nsteps,3,3)); CC_hist=np.empty((Nsteps,3,3))
        bRef_hist=np.empty((Nsteps,3)); CRN_hist=np.empty((Nsteps,3,3))
        q_hist=np.empty((Nsteps,4));  sig_hist=np.empty((Nsteps,3))
        dv_times,dv_list=[],[]

    it=range(Nsteps)
    if show_prog: it=tqdm(it,desc=f"Dep {dep_idx} final",leave=False)

    for j in it:
        t_now=t0+j*DT_DYN
        rC,vC=chief_state(t_now); rA,vA=rA_fn(t_now),vA_fn(t_now)
        C_C,C_A=dcm_lvlh(rC,vC),dcm_lvlh(rA,vA); n=hill_rate(rC,vC)
        rD=rC+C_C.T@rho
        vD=vC+C_C.T@(rhod+np.cross([0,0,n],rho))
        sep_pred,S=cw_map_to_ApophisLVLH(rho,rhod,t_now)
        errA=sep_pred-APOPHIS_TARGET

        if record_hist:
            t_hist[j]=t_now; sep_hist[j]=C_A@(rD-rA); err_hist[j]=la.norm(errA)
            rD_hist[j]=rD; vD_hist[j]=vD
            rA_hist[j]=rA; vA_hist[j]=vA
            rC_hist[j]=rC; vC_hist[j]=vC
            CA_hist[j]=C_A; CC_hist[j]=C_C
            los=rA-rD; los_hat,_=normalize(los)
            C_RN=triad_from_boresight_and_tangent(los_hat,vD).T
            q=dcm_to_quat(C_RN); sig=quat_to_mrp(q)
            bRef_hist[j]=los_hat; CRN_hist[j]=C_RN
            q_hist[j]=q; sig_hist[j]=sig

        can_fire=((j%int(DT_FSW/DT_DYN)==0) and (TOF_SEC-t_now>MIN_LOOK) and
                  (t_now-last_fire>=COOLDOWN) and (len(impulses)<max_imp))
        if can_fire and la.norm(errA)>E_TRIGGER:
            dv_cw=damped_min_norm(S,errA); dv_nudge=np.zeros(3)
            if USE_LAMBERT_NUDGE:
                rA_fin,vA_fin=rA_fn(TOF_SEC),vA_fn(TOF_SEC)
                C_A_fin=dcm_lvlh(rA_fin,vA_fin)
                r_goal=rA_fin+C_A_fin.T@APOPHIS_TARGET
                if angle_deg(rD,r_goal)>=ALIGN_DEG_MIN:
                    dv_eci=lambert_delta_v_now(rD,vD,r_goal,TOF_SEC-t_now)
                    m=la.norm(dv_eci)
                    if m>NUDGE_SAT and m>0: dv_eci*=NUDGE_SAT/m
                    dv_nudge=C_C@dv_eci
            dv=(1-NUDGE_WEIGHT)*dv_cw+NUDGE_WEIGHT*dv_nudge
            base=la.norm(errA); chosen=None
            for scl in (1,.5,.25,.125,.0625):
                dv_try=dv*scl; m=la.norm(dv_try)
                if m<V_MIN: continue
                if m>v_max: dv_try*=v_max/m
                sep_new,_=cw_map_to_ApophisLVLH(rho,rhod+dv_try,t_now)
                if la.norm(sep_new-APOPHIS_TARGET)<base-IMPROVE_MIN:
                    chosen=dv_try; break
            if chosen is not None:
                rhod+=chosen; impulses.append((t_now,chosen.copy())); last_fire=t_now
                if record_hist: dv_times.append(t_now); dv_list.append(chosen.copy())
        Phi_rr,Phi_rv,Phi_vr,Phi_vv=cw_blocks(n,DT_DYN)
        rho,rhod=Phi_rr@rho+Phi_rv@rhod, Phi_vr@rho+Phi_vv@rhod

    # terminal metrics
    rA_end,vA_end=rA_fn(TOF_SEC),vA_fn(TOF_SEC)
    C_A_end=dcm_lvlh(rA_end,vA_end)
    rC_end,vC_end=chief_state(TOF_SEC); C_C_end=dcm_lvlh(rC_end,vC_end)
    rD_end=rC_end+C_C_end.T@rho
    sep_final=C_A_end@(rD_end-rA_end)
    err_final=sep_final-APOPHIS_TARGET
    dv_tot=sum(la.norm(dv) for _,dv in impulses)

    res=dict(err_norm=float(la.norm(err_final)), e_final=err_final,
             sep_final=sep_final, tot_dv=dv_tot, impulses=impulses)
    if record_hist:
        res.update(
            t_hist=t_hist, sepA_hist=sep_hist, err_hist=err_hist,
            rD_hist=rD_hist, vD_hist=vD_hist,
            rA_hist=rA_hist, vA_hist=vA_hist,
            rC_hist=rC_hist, vC_hist=vC_hist,
            C_A_N_hist=CA_hist, C_C_N_hist=CC_hist,
            bRef_N_hist=bRef_hist,
            C_RN_hist=CRN_hist, q_RN_hist=q_hist, sigma_RN_hist=sig_hist,
            dv_times=np.array(dv_times),
            dv_list=np.array(dv_list) if dv_list else np.zeros((0,3)),
            t0=t0)
    return res

def eval_cost(r): return r["err_norm"]+LAMBDA_DV*r["tot_dv"]

# ╔════════════════════════════════════════════╗
# ║ 10.  Helpers for export bundles            ║
# ╚════════════════════════════════════════════╝
def _impulse_array(imps:list[tuple[float,np.ndarray]])->np.ndarray:
    """Dense (Nb,4)  [[t, dv_R, dv_T, dv_N], …]."""
    return np.asarray([[t, dv[0], dv[1], dv[2]] for t,dv in imps],
                      dtype=float) if imps else np.zeros((0,4))

def export_npz_full(path:str,res:dict,t0:float,best_t:float)->None:
    """Rich bundle: full histories + constants + pickled final_res."""
    np.savez_compressed(
        path,
        # ───── time histories ─────
        t=res["t_hist"],
        rD_N=res["rD_hist"], vD_N=res["vD_hist"],
        rA_N=res["rA_hist"], vA_N=res["vA_hist"],
        rC_N=res["rC_hist"], vC_N=res["vC_hist"],
        C_A_N=res["C_A_N_hist"], C_C_N=res["C_C_N_hist"],
        bRef_N=res["bRef_N_hist"],
        C_RN  =res["C_RN_hist"], q_RN=res["q_RN_hist"],
        sigma_RN=res["sigma_RN_hist"],
        # ───── manoeuvres ─────
        dv_times=res["dv_times"],
        dv_list =res["dv_list"],
        impulses=_impulse_array(res["impulses"]),
        # ───── constants ─────
        camMount_B=CAM_BORESIGHT_B,
        APOPHIS_TARGET=APOPHIS_TARGET,
        TARGET_MARGIN_M=np.array([TARGET_MARGIN_M]),
        E_TRIGGER=np.array([E_TRIGGER]),
        V_MIN=np.array([V_MIN]), V_MAX=np.array([V_MAX]),
        # ───── bookkeeping ─────
        t0=np.array([t0]), best_t=np.array([best_t]),
        err_norm=np.array([res["err_norm"]]),
        tot_dv=np.array([res["tot_dv"]]),
        meta=np.array([DEPARTURE_UTC_ISO,
                       arrival_dt.isoformat().replace("+00:00","Z")],dtype=object),
        # ───── pickled dict ────
        final_res=np.array([res],dtype=object)
    )
    print(f"[Export] {os.path.abspath(path)}")

def dump_for_visualize(path:str,res:dict)->None:
    """Lightweight bundle – every ndarray in `res` + plotting constants."""
    save:dict[str,object]={k:v for k,v in res.items() if isinstance(v,np.ndarray)}
    save["impulses"]=_impulse_array(res["impulses"])
    for k in ("t0","best_t","err_norm","tot_dv"):
        if k in res: save[k]=np.array([res[k]])
    save.update(
        camMount_B=CAM_BORESIGHT_B,
        APOPHIS_TARGET=APOPHIS_TARGET,
        TARGET_MARGIN_M=np.array([TARGET_MARGIN_M]),
        E_TRIGGER=np.array([E_TRIGGER]),
        V_MIN=np.array([V_MIN]), V_MAX=np.array([V_MAX]),
        DEPARTURE_UTC_ISO=np.array([DEPARTURE_UTC_ISO]),
        ARRIVAL_UTC_ISO=np.array([arrival_dt.isoformat().replace("+00:00","Z")])
    )
    np.savez_compressed(path,**save)
    print(f"[Visualize-dump] {os.path.abspath(path)} ({len(save)} arrays)")

# ╔════════════════════════════════════════════╗
# ║ 11.  Fleet driver                          ║
# ╚════════════════════════════════════════════╝
def main()->None:
    print("\n==========  FLEET RENDEZVOUS – 8 deputies  ==========")
    results=[]
    for idx,target in enumerate(DEPUTY_TARGETS_M):
        print(f"\n===== Deputy {idx} (target {target/1e3} km) =====")
        globals()["APOPHIS_TARGET"]=target
        png_dir=None if SAVE_PNG_DIR_BASE is None else SAVE_PNG_DIR_BASE.format(idx)
        viz_npz=f"viz_inputs_D{idx}.npz"
        full_npz=f"apophis_rendezvous_full_D{idx}.npz"

        # coarse search
        tmin,tmax=SEARCH_TMIN,max(0,TOF_SEC-SEARCH_TMAX_PAD)
        grid=np.arange(tmin,tmax+1e-6,COARSE_STEP)
        best,best_t=None,None
        for td in tqdm(grid,desc=f"Dep {idx} coarse",unit="deploy-t",leave=False):
            r=run_rhig_from(td)
            if best is None or eval_cost(r)<eval_cost(best): best,best_t=r,td

        # refinements
        for step,tag in ((REFINE_STEP_1,"ref-1"),(REFINE_STEP_2,"ref-2")):
            lo=max(tmin,best_t-REFINE_SPAN); hi=min(tmax,best_t+REFINE_SPAN)
            for td in tqdm(np.arange(lo,hi+1e-6,step),
                           desc=f"Dep {idx} {tag}",unit="deploy-t",leave=False):
                r=run_rhig_from(td)
                if eval_cost(r)<eval_cost(best): best,best_t=r,td

        print(f"  ↳ best deploy offset = {best_t:.1f} s  "
              f"(seed ΔV {best['tot_dv']:.3f} m/s, |e| {best['err_norm']:.2f} m)")

        # final propagation
        final=run_rhig_from(best_t,v_max=FINAL_V_MAX,max_imp=FINAL_MAX_IMP,
                            record_hist=True,show_prog=True,dep_idx=idx)
        print(f"  Final  |e|={final['err_norm']:.3f} m   "
              f"ΔV={final['tot_dv']:.6f} m/s   impulses={len(final['impulses'])}")

        export_npz_full(full_npz,final,final["t0"],best_t)
        dump_for_visualize(viz_npz,final)
        results.append((idx,target,best_t,final,png_dir))

    # impulse tables
    for idx,target,_,fin,_ in results:
        print(f"\n=== Impulses for Deputy {idx} (target {target/1e3} km) ===")
        if not fin["impulses"]:
            print("  <no burns>")
        else:
            print("  # |     t [s] |    Δv_R    Δv_T    Δv_N (mm/s)")
            for k,(t,dv) in enumerate(fin["impulses"],1):
                print(f" {k:2d} | {t:10.1f} |"
                      f"{dv[0]*1e3:9.3f}{dv[1]*1e3:9.3f}{dv[2]*1e3:9.3f}")

    # optional plotting (head-less ok)
    print("\nGenerating figures …")
    import Visualize as viz
    for idx,target,best_t,fin,png_dir in results:
        viz.plot_all(fin,best_t=best_t,DT_DYN=DT_DYN,
                     APOPHIS_TARGET=target,TARGET_MARGIN_M=TARGET_MARGIN_M,
                     E_TRIGGER=E_TRIGGER,V_MIN=V_MIN,V_MAX=V_MAX,
                     DV_BUDGET_MS=DV_BUDGET_MS,R_EARTH_KM=R_EARTH_KM,
                     save_dir=png_dir,font_size=50,tick_size=45,
                     npz_path_for_nav=NAV_NPZ_PATH,
                     title_prefix=f"Dep {idx}: ",show=False)
    print("…done.")

# ╔════════════════════════════════════════════╗
# ║  Entry-point                               ║
# ╚════════════════════════════════════════════╝
if __name__ == "__main__":
    main()
