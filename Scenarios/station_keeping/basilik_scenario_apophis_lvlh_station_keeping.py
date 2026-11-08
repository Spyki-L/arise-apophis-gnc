#!/usr/bin/env python3
# ISC-License – AVS Lab (CU Boulder)
"""
basilik_scenario_apophis_lvlh_station_keeping.py
================================================

Purpose
-------
Single-deputy Apophis LVLH **station-keeping** demo using:

- A single initial CW-style burn at release to push the deputy toward a
  desired stand-off point in Apophis-centred LVLH,
- A dead-band PD controller in LVLH, feeding:
  * Schmitt-trigger PWM (thrFiringSchmitt) on a 6-thruster cluster,
- Basilisk core modules + a minimal “MiniSim” scheduler only (no private or
  deprecated APIs).

The goal is to hold the deputy near a desired box around the target:

    TARGET_LVLH_A = [1.43, 2.55, -0.27] km  (Apophis LVLH R,T,N)

while respecting small thrust levels and minimum pulse times.

Timeline and geometry
---------------------
Key epochs (seconds since chief departure epoch):

- DEPLOY_EPOCH = 10 023.0 s:
    * Reference time for initial LVLH relative state.
- BURN_EPOCH   = DEPLOY_EPOCH + 1.0 s:
    * HCW-style LVLH burn applied 1 s after deploy.
- RV_EPOCH     = 31 170.0 s:
    * Rendezvous / station-keeping reference time (≈ 5.8 h after burn).

Integration cadences:

- DT_DYN = 1.0 s  (dynamics step),
- DT_FSW = 5.0 s  (FSW / controller step).

Target and controller parameters
--------------------------------
- Target position in Apophis LVLH (km):

    TARGET_LVLH_A = [1.43, 2.55, -0.27]

- Deputy / thrusters:
    * Spacecraft mass:  SC_MASS   = 50.0 kg.
    * Thruster max thrust: THR_MAX_N = 0.05 N (tiny EP-like thrusters).
    * Minimum ON time per pulse: THR_MIN_ON = 0.1 s.
    * Six thrusters distributed along ±X, ±Y, ±Z (in body frame).

- Dead-band PD:
    * Position dead-band [m]: DB_POS = [3, 3, 2].
    * Velocity dead-band [m/s]: DB_VEL = [0.01, 0.01, 0.008].
    * Proportional gains (N/m): Kp = [2e−4, 1e−4, 2e−4].
    * Derivative gains (N·s/m): Kd = [5e−3, 3e−3, 5e−3].

- Schmitt parameters:
    * ON/OFF thresholds: SCH_ON, SCH_OFF (configured in `thrFiringSchmitt`).
    * Minimum fire time: THR_MIN_ON.

Chief and ephemerides
---------------------
The deputy trajectory is expressed relative to a “chief” that is stored in:

    chief_reference.npz

which must contain:

- `times`      : 1D array [s] since departure,
- `positions`  : (N,3) [km] chief ECI positions,
- `velocities` : (N,3) [km/s] chief ECI velocities.

The script loads this and defines:

    chief(t) → (rC_km(t), vC_km(t))

for the deputy’s initial state and for Apophis-relative geometry at RV_EPOCH.

Apophis SPICE sampling
----------------------
`pre_sample_spice()` builds a lightweight MiniSim run to sample Earth and
Apophis states for 12 h at dt = 600 s using:

- Kernels from Basilisk’s EphemerisData:
    naif0012.tls, pck00010.tpc, de-403-masses.tpc, de430.bsp, 20099942.bsp,
- `SpiceInterface` with `zeroBase="Sun"`,
- `ephemerisConverter` and recorders for Earth (index 0) and Apophis (index 2).

It returns:

    T    : time grid [s],
    rE   : Earth positions wrt Sun [km],
    vE   : Earth velocities wrt Sun [km/s],
    rA   : Apophis positions wrt Sun [km],
    vA   : Apophis velocities wrt Sun [km/s].

These are later combined into Apophis geocentric states inside the controller.

Controller (LVLH PD + dead-band + Schmitt PWM)
----------------------------------------------
The LVLH controller is implemented as `LVLHController`, an `AlgContain`–style
FSW module:

1. Uses a global NavTrans reader `_NAV` to get deputy inertial state:
       rN, vN = nav.r_BN_N, nav.v_BN_N.

2. At time t (nanoseconds), computes:

       tt  = clamp((t − t0) * NANO2SEC, T[0], T[-1])
       rA  = (Apophis − Earth)(tt) * 1e3        # [m]
       vA  = (Apophis − Earth)(tt) * 1e3        # [m/s]

   and Apophis LVLH frame:

       C_A = dcm_lvlh(rA, vA)                   # rows = [R̂, T̂, N̂]

3. Deputy LVLH state:

       rL = C_A @ (rN − rA)                     # position in LVLH [m]
       vL = C_A @ (vN − vA)                     # velocity in LVLH [m/s]

4. For each component k ∈ {R,T,N}:
   - If |rL[k] − TARGET[k]| > DB_POS[k] **or** |vL[k]| > DB_VEL[k],
     compute a PD force:

         F_k = −Kp[k] (rL[k] − TARGET[k]) − Kd[k] vL[k]

     clipped to ±THR_MAX_N.

5. Map component forces to individual thrusters:

   - F_RTN → [Fx+, Fx−, Fy+, Fy−, Fz+, Fz−] via sign splits,
   - Write into `_FORCE` payload and out via `_FORCE_WRITER`.

The `thrFiringSchmitt` configuration:

- Reads the thruster cluster config from `simIncludeThruster`,
- Takes `_FORCE_WRITER` as its continuous force input,
- Produces ON-time commands (`onTimeOutMsg`) that are subscribed to by
  `thrusterDynamicEffector`.

Simulation architecture
-----------------------
- MiniSim skeleton:
    sim = MiniSim()
    dyn task at DT_DYN (1s), fsw task at DT_FSW (5s).
- Dynamics process (“dynP”) runs:
    * Earth gravity, deputy spacecraft, SimpleNav, thruster effector.
- FSW process (“fswP”) runs:
    * `LVLHController`,
    * `thrFiringSchmitt` + its config struct.

Initial deputy state:
- At `BURN_EPOCH`:
    * Chief state (rC, vC) from `chief_reference.npz`.
    * Inertial deputy position/velocity:
         r0 = rC * 1e3,
         v0 = vC * 1e3 + C0ᵀ v0_LVLH,
       where:
         C0 = dcm_lvlh(r0, v0) and
         v0_LVLH = coarse CW guess ~ 2·TARGET / (3·τ) for τ = RV_EPOCH − BURN_EPOCH.

Logging and outputs
-------------------
- Recorder on the spacecraft state:
    * `state.r_BN_N` and `state.v_BN_N` over the whole run.
- Recorder on the Schmitt ON-time requests:
    * `onT.OnTimeRequest` (N×6) with per-thruster pulse times.

At the end of the run:

1. Final LVLH separation at RV_EPOCH:
   - Apophis geocentric rA_RV from SPICE tables at RV_EPOCH.
   - Deputy final inertial position r_dep.
   - Convert to LVLH and report:

        Deputy–Apophis LVLH @RV_EPOCH [km]
        Residual = (sep − TARGET_LVLH_A)

2. Cumulative ΔV estimate:
   - Start with initial LVLH burn magnitude `‖v0_LVLH‖`,
   - Add Σ (THR_MAX_N * onTime) / SC_MASS over all FSW samples.

3. Plots (if `show_plots=True`):
   - Thruster ON-time vs time for all six thrusters (step plots).
   - Cumulative ΔV vs time with an example budget line at 0.4 m/s.

Usage
-----
Prerequisites:

- Basilisk with Apophis SPK (`20099942.bsp`) available in
  `${BASILISK_ROOT}/supportData/EphemerisData`.
- A `chief_reference.npz` file in the working directory, containing:
    * `times`, `positions`, `velocities` for the chief (km, km/s).

Run:

    python basilik_scenario_apophis_lvlh_station_keeping.py

The script will:

- Pre-sample Earth + Apophis states over a 12 h SPICE window,
- Initialise the deputy with a coarse HCW burn toward the LVLH target,
- Run dead-band PD + Schmitt PWM station-keeping until RV_EPOCH,
- Print LVLH error at RV_EPOCH and total ΔV,
- Display thruster ON-time and cumulative ΔV plots (no files written).
"""


import os, sys
import numpy as np, numpy.linalg as la
import matplotlib.pyplot as plt; plt.style.use("bmh")

from Basilisk                       import __path__ as bskPathPkg
from Basilisk.architecture          import messaging, alg_contain
from Basilisk.simulation            import (spacecraft, simpleNav,
                                            spiceInterface, ephemerisConverter,
                                            thrusterDynamicEffector)
from Basilisk.fswAlgorithms         import thrFiringSchmitt
from Basilisk.utilities             import macros, simIncludeGravBody
from Basilisk.utilities             import simIncludeThruster as sit
from MiniSim                        import MiniSim

# ─────────── scenario constants ──────────────────────────────────────────
DEPLOY_EPOCH = 10_023.0
BURN_EPOCH   = DEPLOY_EPOCH + 1.0      # deputy burn
RV_EPOCH     = 31_170.0                # rendez-vous time (≈5.8 h later)

DT_DYN = 1.0;  DT_FSW = 5.0            # [s]

TARGET_LVLH_A = np.array([1.43, 2.55, -0.27])   # km

SC_MASS   = 50.0          # kg
THR_MAX_N = 5e-2          # N  (tiny EP thruster)
THR_MIN_ON= 0.1           # s  (minimum pulse)
SCH_ON, SCH_OFF = 0.6, 0.2

DB_POS = np.array([3.,3.,2.])          # m
DB_VEL = np.array([0.01,0.01,0.008])   # m s⁻¹
Kp = np.array([2e-4,1e-4,2e-4])        # N m⁻¹
Kd = np.array([5e-3,3e-3,5e-3])        # N s m⁻¹

bskRoot = bskPathPkg[0]

# ─────────── helpers ────────────────────────────────────────────────────
def dcm_lvlh(r, v):
    ih = r/la.norm(r); k = np.cross(r,v); k/=la.norm(k); j = np.cross(k,ih)
    return np.vstack((ih,j,k))

def ivec(t,T,X):          # interpolate 3-vector wrt time grid
    return np.array([np.interp(t,T,X[:,k]) for k in range(3)])

def kernel_dir():
    kd = os.path.join(bskRoot,"supportData","EphemerisData")
    if not os.path.isdir(kd):
        sys.exit(f"SPICE dir missing:\n  {kd}")
    return kd+os.sep

def load_kernels(spi, kdir, files):
    for f in files:
        full = kdir+f
        if not os.path.isfile(full):
            sys.exit(f"SPICE kernel missing:\n  {full}")
        spi.loadSpiceKernel(f, kdir)

def pre_sample_spice():
    """Sample Earth & Apophis for 12 h, dt=600 s."""
    sim = MiniSim()
    tsk = sim.CreateNewTask("sp", macros.sec2nano(600))
    sim.CreateNewProcess("P").addTask(tsk)

    spi = spiceInterface.SpiceInterface(); spi.SPICEDataPath = kernel_dir()
    spi.UTCCalInit = "2029 APR 13 11:38:28"
    load_kernels(spi, kernel_dir(),
                 ["naif0012.tls","pck00010.tpc","de-403-masses.tpc",
                  "de430.bsp","20099942.bsp"])
    spi.addPlanetNames(["Earth","Sun","20099942"]); spi.zeroBase="Sun"

    conv = ephemerisConverter.EphemerisConverter()
    for m in spi.planetStateOutMsgs: conv.addSpiceInputMsg(m)

    sim.AddModelToTask("sp", spi); sim.AddModelToTask("sp", conv)
    eR = conv.ephemOutMsgs[0].recorder()
    aR = conv.ephemOutMsgs[2].recorder()
    sim.AddModelToTask("sp", eR); sim.AddModelToTask("sp", aR)
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(12*3600))
    sim.ExecuteSimulation()

    T = np.arange(0., 12*3600+600, 600)
    return T, eR.r_BdyZero_N/1e3, eR.v_BdyZero_N/1e3, aR.r_BdyZero_N/1e3, aR.v_BdyZero_N/1e3

# ───────────-- FSW module (PD + dead-band) ─────────────────────────────
class LVLHController(alg_contain.AlgContain):
    def __init__(self): super().__init__(); self.ModelTag="LVLH"
    def Reset(self,t): self._t0=t
    def UpdateState(self,t):
        nav=_NAV(); rN,vN=np.array(nav.r_BN_N), np.array(nav.v_BN_N)
        tt=np.clip((t-self._t0)*macros.NANO2SEC, _T[0], _T[-1])
        rA=(ivec(tt,_T,_rA)-ivec(tt,_T,_rE))*1e3
        vA=(ivec(tt,_T,_vA)-ivec(tt,_T,_vE))*1e3
        C=dcm_lvlh(rA,vA); rL=C@(rN-rA); vL=C@(vN-vA)

        cmd=np.zeros(3)
        for k in range(3):
            if abs(rL[k]-_TARG[k])>DB_POS[k] or abs(vL[k])>DB_VEL[k]:
                cmd[k]=np.clip(-Kp[k]*(rL[k]-_TARG[k])-Kd[k]*vL[k],
                               -THR_MAX_N,THR_MAX_N)

        Fx,Fy,Fz=cmd
        fl=[max(Fx,0),max(-Fx,0),max(Fy,0),max(-Fy,0),max(Fz,0),max(-Fz,0)]
        for i,val in enumerate(fl): _FORCE.thrForce[i]=val
        _FORCE_WRITER.write(_FORCE)

# ─────────── main ─────────────────────────────────────────────────────
def run(show_plots=True):

    # chief ephemeris (for deputy release)
    ref=np.load("chief_reference.npz")
    chief=lambda t:(ivec(t,ref["times"],ref["positions"]),
                    ivec(t,ref["times"],ref["velocities"]))

    # SPICE table
    global _T,_rE,_vE,_rA,_vA
    _T,_rE,_vE,_rA,_vA=pre_sample_spice()

    # initial deputy state
    rC,vC=chief(BURN_EPOCH); r0=rC*1e3; v0=vC*1e3
    C0=dcm_lvlh(r0,v0)
    tau=RV_EPOCH-BURN_EPOCH
    v0_LVLH=(2*TARGET_LVLH_A*1e3)/(3*tau)      # coarse CW guess
    print("Initial LVLH burn:\n"
          f"  v0_LVLH={v0_LVLH/1e3} m/s |ΔV|={la.norm(v0_LVLH):.4f} m/s")

    # ─ MiniSim skeleton ─
    sim=MiniSim(); dynT=sim.CreateNewTask("dyn",macros.sec2nano(DT_DYN))
    fswT=sim.CreateNewTask("fsw",macros.sec2nano(DT_FSW))
    sim.CreateNewProcess("dynP").addTask(dynT)
    sim.CreateNewProcess("fswP").addTask(fswT)

    # gravity + deputy SC
    grav=simIncludeGravBody.gravBodyFactory(); earth=grav.createEarth(); earth.isCentralBody=True
    sc=spacecraft.Spacecraft(); sc.hub.mHub=SC_MASS
    sc.hub.r_CN_NInit=r0; sc.hub.v_CN_NInit=v0+C0.T@v0_LVLH
    grav.addBodiesTo(sc); sim.AddModelToTask("dyn",sc)

    # nav
    nav=simpleNav.SimpleNav(); nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    sim.AddModelToTask("dyn",nav)

    # thrusters
    factory=sit.thrusterFactory()
    tkw=dict(MaxThrust=float(THR_MAX_N),MinOnTime=float(THR_MIN_ON),
             steadyIsp=1800.,useMinPulseTime=True)
    for i,d in enumerate(([1,0,0],[-1,0,0],[0,1,0],[-1,0,0],[0,0,1],[0,0,-1])):
        factory.create("Blank_Thruster",[0,0,0],d,label=f"T{i}",**tkw)
    thrEff=thrusterDynamicEffector.ThrusterDynamicEffector()
    factory.addToSpacecraft("cluster",thrEff,sc); sim.AddModelToTask("dyn",thrEff)

    # global message handles
    global _NAV,_FORCE,_FORCE_WRITER,_TARG
    _NAV=messaging.NavTransMsgReader(); _NAV.subscribeTo(nav.transOutMsg)
    _FORCE_WRITER=messaging.THRArrayCmdForceMsg()
    _FORCE=messaging.THRArrayCmdForceMsgPayload()
    _FORCE_WRITER.write(_FORCE)               # seed
    _TARG=TARGET_LVLH_A*1e3

    # controller
    sim.AddModelToTask("fsw",LVLHController())

    # Schmitt
    cfg_seed = messaging.THRArrayConfigMsg().write(factory.getConfigMessage().read())
    schCfg=thrFiringSchmitt.thrFiringSchmittConfig()
    schCfg.level_on,SCH_OFF; schCfg.level_off=SCH_OFF
    schCfg.thrMinFireTime=THR_MIN_ON
    schCfg.thrConfInMsg.subscribeTo(cfg_seed)
    schCfg.thrForceInMsg.subscribeTo(_FORCE_WRITER)

    sch=thrFiringSchmitt.thrFiringSchmitt(); sch.ModelTag="PWM"
    sim.AddModelToTask("fsw",sch,schCfg)
    thrEff.cmdsInMsg.subscribeTo(schCfg.onTimeOutMsg)

    # logging
    state=sc.scStateOutMsg.recorder(); sim.AddModelToTask("dyn",state)
    onT =schCfg.onTimeOutMsg.recorder(); sim.AddModelToTask("fsw",onT)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(RV_EPOCH-BURN_EPOCH))
    sim.ExecuteSimulation()

    # results
    rA_RV=(ivec(RV_EPOCH,_T,_rA)-ivec(RV_EPOCH,_T,_rE))*1e3
    sep=(state.r_BN_N[-1]-rA_RV)/1e3
    print(f"\nDeputy–Apophis LVLH @RV_EPOCH [km]: {sep}")
    print(f"Residual [km]: {sep-TARGET_LVLH_A}")

    t_short=onT.times()*macros.NANO2SEC
    dv=np.full_like(t_short, la.norm(v0_LVLH))
    for k,tau in enumerate(onT.OnTimeRequest):
        dv[k]+=np.sum(THR_MAX_N*tau)/SC_MASS
    print(f"Total ΔV = {dv[-1]:.4f} m/s")

    plt.figure()
    for i in range(6):
        plt.step(t_short/3600., onT.OnTimeRequest[:,i],where='post',label=f"T{i}")
    plt.title("Thruster ON-time"); plt.xlabel("Time [h]"); plt.ylabel("s"); plt.legend()

    plt.figure(); plt.plot(t_short/3600., dv); plt.title("Cumulative ΔV")
    plt.xlabel("Time [h]"); plt.ylabel("m s⁻¹"); plt.axhline(0.4,ls="--",c='r')
    if show_plots: plt.show()

# ─────────────────────────────────────────────────────────────────
if __name__=="__main__":
    run()
