#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_scenario_lvlh_station_keeping.py
========================================

Generic LVLH station-keeping demo around a chief spacecraft.

• Chief trajectory is taken from chief_reference.npz (any Earth-centred orbit).
• One deputy starts co-located with the chief, gets a coarse HCW burn, then
  uses a dead-band LVLH PD controller + Schmitt PWM to hold a target box in
  chief-centred LVLH (RTN).

This is a *generic* version of the Apophis script:
- No SPICE, no asteroid-specific geometry.
- Only the chief_reference.npz and a chosen LVLH target matter.
"""

from __future__ import annotations
import os, sys
from pathlib import Path

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt; plt.style.use("bmh")

from Basilisk.architecture          import messaging, alg_contain
from Basilisk.simulation            import (spacecraft, simpleNav,
                                            thrusterDynamicEffector)
from Basilisk.fswAlgorithms         import thrFiringSchmitt
from Basilisk.utilities             import macros, simIncludeGravBody
from Basilisk.utilities             import simIncludeThruster as sit
from Basilisk.utilities             import orbitalMotion, unitTestSupport
from MiniSim                        import MiniSim

# ??????????? scenario constants (generic) ??????????????????????????????????
# Chief reference file: times [s], positions [km], velocities [km/s]
CHIEF_NPZ = Path(os.getenv("CHIEF_REF_NPZ", "chief_reference.npz"))

# Absolute times [s] in the same epoch as chief_reference.npz
DEPLOY_EPOCH_S     = 10_023.0
BURN_EPOCH_S       = DEPLOY_EPOCH_S + 1.0
RV_EPOCH_S         = 31_170.0               # rendezvous / SK reference time

DT_DYN = 1.0     # dynamics task [s]
DT_FSW = 5.0     # controller + Schmitt [s]

# Target LVLH offset (RTN) wrt chief at RV epoch [m]  (editable)
TARGET_LVLH_M = np.array([1.43e3, 2.55e3, -0.27e3])

# Deputy & thruster parameters
SC_MASS   = 50.0          # kg
THR_MAX_N = 5e-2          # N
THR_MIN_ON= 0.1           # s  (minimum pulse)
SCH_ON, SCH_OFF = 0.6, 0.2

DB_POS = np.array([3., 3., 2.])       # [m]
DB_VEL = np.array([0.01, 0.01, 0.008])# [m/s]
Kp     = np.array([2e-4, 1e-4, 2e-4]) # [N/m]
Kd     = np.array([5e-3, 3e-3, 5e-3]) # [N·s/m]

# ??????????? helpers ??????????????????????????????????????????????????????

def dcm_lvlh(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """LVLH<-ECI DCM with rows = [R?; T?; N?] in ECI components."""
    ih = r / la.norm(r)
    k  = np.cross(r, v); k /= la.norm(k)
    j  = np.cross(k, ih)
    return np.vstack((ih, j, k))

def ivec(t: float, T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Interpolate a 3-vector array X(T) at time t (component-wise)."""
    return np.array([np.interp(t, T, X[:, k]) for k in range(3)], float)

# ??????????? chief loader ?????????????????????????????????????????????????

def load_chief_reference(npz_path: Path):
    if not npz_path.is_file():
        sys.exit(f"chief_reference.npz not found at: {npz_path}")
    ref = np.load(npz_path)
    T    = np.asarray(ref["times"], float)
    r_km = np.asarray(ref["positions"], float)
    v_km = np.asarray(ref["velocities"], float)
    return T, r_km, v_km

# ??????????? FSW module (LVLH PD + dead-band) ?????????????????????????????

class LVLHController(alg_contain.AlgContain):
    """
    Generic LVLH box controller:

    - Uses global chief ephemeris (CHIEF_T, CHIEF_R_KM, CHIEF_V_KM).
    - At sim time t (ns), maps to absolute time:
          t_abs = BURN_EPOCH_S + (t - t_reset)*NANO2SEC
      and computes chief LVLH frame at that epoch.
    - Applies a dead-band PD in LVLH and writes desired force vector into
      THRArrayCmdForceMsg (_FORCE_WRITER) to be picked up by Schmitt PWM.
    """
    def __init__(self):
        super().__init__()
        self.ModelTag = "LVLH_Box"
        self._t0_ns   = 0

    def Reset(self, t: int):
        # Remember reset time in nanoseconds
        self._t0_ns = t

    def UpdateState(self, t: int):
        # Convert sim time ? absolute chief time
        t_rel = (t - self._t0_ns) * macros.NANO2SEC      # [s] since start of sim
        t_abs = BURN_EPOCH_S + t_rel                     # [s] since chief epoch

        # Nav state (inertial)
        nav = _NAV()   # NavTransMsgReader
        rN  = np.array(nav.r_BN_N, float)   # [m]
        vN  = np.array(nav.v_BN_N, float)   # [m/s]

        # Chief state at this absolute time (km ? m)
        rC_km = ivec(t_abs, _CHIEF_T, _CHIEF_R)
        vC_km = ivec(t_abs, _CHIEF_T, _CHIEF_V)
        rC    = rC_km * 1e3
        vC    = vC_km * 1e3

        # Chief LVLH frame
        C = dcm_lvlh(rC, vC)   # rows = R,T,N
        rL = C @ (rN - rC)     # LVLH position [m]
        vL = C @ (vN - vC)     # LVLH velocity [m/s]

        # Dead-band PD box
        cmd = np.zeros(3)
        for k in range(3):
            if abs(rL[k] - _TARG[k]) > DB_POS[k] or abs(vL[k]) > DB_VEL[k]:
                cmd[k] = np.clip(
                    -Kp[k]*(rL[k] - _TARG[k]) - Kd[k]*vL[k],
                    -THR_MAX_N, THR_MAX_N
                )

        # Split RTN forces to 6 thrusters: ±X, ±Y, ±Z
        Fx, Fy, Fz = cmd
        fl = [
            max(Fx, 0.0), max(-Fx, 0.0),   # +X, -X
            max(Fy, 0.0), max(-Fy, 0.0),   # +Y, -Y
            max(Fz, 0.0), max(-Fz, 0.0),   # +Z, -Z
        ]

        for i, val in enumerate(fl):
            _FORCE.thrForce[i] = float(val)

        _FORCE_WRITER.write(_FORCE)


# ??????????? main scenario ????????????????????????????????????????????????

def run(show_plots: bool = True):
    # 1) Load chief ephemeris
    global _CHIEF_T, _CHIEF_R, _CHIEF_V
    print(f"[INFO] Loading chief trajectory from: {CHIEF_NPZ.resolve()}")
    _CHIEF_T, _CHIEF_R, _CHIEF_V = load_chief_reference(CHIEF_NPZ)

    if RV_EPOCH_S <= BURN_EPOCH_S:
        raise ValueError("RV_EPOCH_S must be greater than BURN_EPOCH_S.")

    # 2) Initial deputy state from chief at burn epoch
    rC_km, vC_km = ivec(BURN_EPOCH_S, _CHIEF_T, _CHIEF_R), ivec(BURN_EPOCH_S, _CHIEF_T, _CHIEF_V)
    r0 = rC_km * 1e3                      # [m]
    v0 = vC_km * 1e3                      # [m/s]

    # Local mean motion from chief orbit (near-circular assumption)
    # n = |h| / r^2
    h_vec = np.cross(r0, v0)
    n = la.norm(h_vec) / (la.norm(r0)**2 + 1e-12)

    # Coarse HCW-type guess for initial LVLH ?v toward target
    tau   = RV_EPOCH_S - BURN_EPOCH_S
    v0_LVLH = (2.0 * TARGET_LVLH_M) / (3.0 * tau)  # [m/s], very rough approximation
    print("Initial LVLH burn (coarse):")
    print(f"  v0_LVLH = {v0_LVLH} m/s")
    print(f"  |?V|    = {la.norm(v0_LVLH):.4f} m/s")

    # LVLH?ECI rotation at burn
    C0     = dcm_lvlh(r0, v0)
    C_NL_0 = C0.T
    v0_eci = v0 + C_NL_0 @ v0_LVLH

    # 3) MiniSim skeleton
    sim = MiniSim()
    dynT = sim.CreateNewTask("dyn", macros.sec2nano(DT_DYN))
    fswT = sim.CreateNewTask("fsw", macros.sec2nano(DT_FSW))
    sim.CreateNewProcess("dynP").addTask(dynT)
    sim.CreateNewProcess("fswP").addTask(fswT)

    # Gravity (Earth) + deputy SC
    grav = simIncludeGravBody.gravBodyFactory()
    earth = grav.createEarth()
    earth.isCentralBody = True

    sc = spacecraft.Spacecraft()
    sc.hub.mHub = SC_MASS
    sc.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d([1,0,0, 0,1,0, 0,0,1])
    sc.hub.r_CN_NInit  = list(r0)
    sc.hub.v_CN_NInit  = list(v0_eci)
    sc.hub.sigma_BNInit   = [[0.0],[0.0],[0.0]]
    sc.hub.omega_BN_BInit = [[0.0],[0.0],[0.0]]
    grav.addBodiesTo(sc)
    sim.AddModelToTask("dyn", sc)

    # nav
    nav = simpleNav.SimpleNav()
    nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    sim.AddModelToTask("dyn", nav)

    # thrusters: 6 thrusters ±X, ±Y, ±Z
    factory = sit.thrusterFactory()
    thr_kwargs = dict(
        MaxThrust=float(THR_MAX_N),
        MinOnTime=float(THR_MIN_ON),
        steadyIsp=1800.0,
        useMinPulseTime=True
    )
    thr_dirs = [
        [1,0,0], [-1,0,0],
        [0,1,0], [0,-1,0],
        [0,0,1], [0,0,-1],
    ]
    for i, d in enumerate(thr_dirs):
        factory.create("Blank_Thruster", [0,0,0], d, label=f"T{i}", **thr_kwargs)

    thrEff = thrusterDynamicEffector.ThrusterDynamicEffector()
    factory.addToSpacecraft("cluster", thrEff, sc)
    sim.AddModelToTask("dyn", thrEff)

    # Global message handles for controller
    global _NAV, _FORCE_WRITER, _FORCE, _TARG
    _NAV = messaging.NavTransMsgReader()
    _NAV.subscribeTo(nav.transOutMsg)

    _FORCE_WRITER = messaging.THRArrayCmdForceMsg()
    _FORCE        = messaging.THRArrayCmdForceMsgPayload()
    _FORCE_WRITER.write(_FORCE)   # seed

    _TARG = TARGET_LVLH_M.copy()

    # Controller
    sim.AddModelToTask("fsw", LVLHController())

    # Schmitt PWM
    cfg_seed = messaging.THRArrayConfigMsg().write(
        factory.getConfigMessage().read()
    )

    schCfg = thrFiringSchmitt.thrFiringSchmittConfig()
    schCfg.level_on  = SCH_ON
    schCfg.level_off = SCH_OFF
    schCfg.thrMinFireTime = THR_MIN_ON
    schCfg.thrConfInMsg.subscribeTo(cfg_seed)
    schCfg.thrForceInMsg.subscribeTo(_FORCE_WRITER)

    sch = thrFiringSchmitt.thrFiringSchmitt()
    sch.ModelTag = "PWM"
    sim.AddModelToTask("fsw", sch, schCfg)

    thrEff.cmdsInMsg.subscribeTo(schCfg.onTimeOutMsg)

    # Logging
    state = sc.scStateOutMsg.recorder()
    sim.AddModelToTask("dyn", state)
    onT = schCfg.onTimeOutMsg.recorder()
    sim.AddModelToTask("fsw", onT)

    # Run from burn to rendezvous (we start sim at burn epoch)
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(RV_EPOCH_S - BURN_EPOCH_S))
    sim.ExecuteSimulation()

    # 4) Results: final LVLH separation relative to chief
    rC_rv_km = ivec(RV_EPOCH_S, _CHIEF_T, _CHIEF_R)
    vC_rv_km = ivec(RV_EPOCH_S, _CHIEF_T, _CHIEF_V)
    rC_rv    = rC_rv_km * 1e3
    vC_rv    = vC_rv_km * 1e3

    C_rv = dcm_lvlh(rC_rv, vC_rv)
    r_dep_final = np.array(state.r_BN_N[-1], float)
    sep_lvlh = C_rv @ (r_dep_final - rC_rv) / 1e3   # [km]

    print(f"\nDeputy–chief LVLH @ RV_EPOCH [km]: {sep_lvlh}")
    print(f"Target LVLH [km]                  : {TARGET_LVLH_M/1e3}")
    print(f"Residual [km]                     : {sep_lvlh - TARGET_LVLH_M/1e3}")

    # 5) Cumulative ?V estimate
    t_short = onT.times() * macros.NANO2SEC
    dv_hist = np.full_like(t_short, la.norm(v0_LVLH))   # start with initial impulsive |?V|
    for k, tau in enumerate(onT.OnTimeRequest):
        dv_hist[k] += np.sum(THR_MAX_N * tau) / SC_MASS
    print(f"Total ?V (coarse estimate) = {dv_hist[-1]:.4f} m/s")

    # 6) Plots
    plt.figure()
    for i in range(6):
        plt.step(t_short/3600.0, onT.OnTimeRequest[:, i],
                 where="post", label=f"T{i}")
    plt.title("Thruster ON-time")
    plt.xlabel("Time [h]"); plt.ylabel("ON time [s]"); plt.legend()

    plt.figure()
    plt.plot(t_short/3600.0, dv_hist)
    plt.title("Cumulative ?V")
    plt.xlabel("Time [h]"); plt.ylabel("?V [m/s]")
    plt.axhline(0.4, ls="--", c="r", label="Example budget 0.4 m/s")
    plt.legend()

    if show_plots:
        plt.show()


# ?????????????????????????????????????????????????????????????????
if __name__ == "__main__":
    run()

