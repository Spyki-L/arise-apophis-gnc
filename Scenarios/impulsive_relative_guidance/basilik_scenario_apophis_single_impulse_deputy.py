#!/usr/bin/env python3
"""
basilik_scenario_apophis_single_impulse_deputy.py
=================================================

Purpose
-------
Design and execute a **single-impulse** transfer for a deputy spacecraft from
a given chief trajectory to a fixed target offset in Apophis-centred LVLH:

    Target LVLH position: (+1.43, +2.55, −0.27) km  in [R, T, N]

Key features:
- Uses a precomputed `chief_reference.npz` trajectory for the chief (which may
  be **geocentric or heliocentric**).
- Computes Apophis’s geocentric state at the rendezvous epoch via SPICE.
- Translates the LVLH target into an inertial target position at RV_EPOCH.
- Tries two Lambert solutions, one using Earth’s μ and one using the Sun’s μ:
    * μ_Earth = 3.986004418e5 km³/s²
    * μ_Sun   = MU_SUN/1e9 (from Basilisk orbitalMotion)
  and chooses the lower-ΔV option.
- Propagates the deputy under the chosen central body’s two-body gravity to
  RV_EPOCH using `scipy.integrate.solve_ivp`.
- Reports the achieved Apophis LVLH offset and residual error.
- Plots an Earth-centred 3D view and XY projection with:
    * GEO ring (for context),
    * Chief pre- and post-deploy,
    * Deputy trajectory,
    * 12 h Apophis track,
    * Markers for:
        - Chief start,
        - Burn point,
        - Deputy final,
        - Apophis at RV_EPOCH.

Inputs
------
1. Chief trajectory:

    chief_reference.npz

Must be present in the working directory and contain:

- `times`      : 1D array [s] since some epoch,
- `positions`  : (N, 3) positions [km] in an inertial frame,
- `velocities` : (N, 3) velocities [km/s] in the same inertial frame.

This script does not assume whether the frame is geocentric or heliocentric;
it simply uses those states consistently with the chosen μ when propagating
the deputy.

2. SPICE ephemerides:

Looks under `${BASILISK_ROOT}/supportData/EphemerisData` for:

- naif0012.tls
- pck00010.tpc
- de-403-masses.tpc
- de430.bsp
- 20099942.bsp       (Apophis SPK)

These are loaded into a small SPICE simulation to provide Earth and Apophis
states over 12 hours, with `zeroBase = "Sun"`.

Timeline and target
-------------------
- DEPLOY_EPOCH = 10 023.0 s
- BURN_EPOCH   = DEPLOY_EPOCH + 1.0 s
- RV_EPOCH     = 31 170.0 s
- Transfer duration for the deputy:
      t_end = RV_EPOCH − BURN_EPOCH

The target LVLH position (Apophis-centred) is:

    TARGET_LVLH_A = [1.43, 2.55, −0.27] km

Using Apophis state at RV_EPOCH:

1. Compute Apophis geocentric ECI at RV_EPOCH:

       r_apo, v_apo (km, km/s)

2. Build Apophis LVLH DCM:

       C_NA = dcm_lvlh(r_apo, v_apo)   # columns are [R̂, T̂, N̂] in ECI

3. Construct inertial target position (for the deputy at RV_EPOCH):

       r_dep_target = r_apo + C_NA @ TARGET_LVLH_A

Lambert ΔV selection
--------------------
The script evaluates two single-impulse Lambert solutions for the segment
from BURN_EPOCH to RV_EPOCH, each under a different central μ:

1. With Earth μ:
   - Call `lambert_dv(MU_EARTH)`:
       * Build a small Basilisk sim with a dummy spacecraft and SimpleNav.
       * Use `LambertPlanner` and `LambertSolver` to compute v_req at departure
         that connects `r_c0` to `r_dep_target` in `RV_EPOCH − BURN_EPOCH`.
       * Return Δv_E = (v_req − v_c0) [m/s].

2. With Sun μ:
   - Call `lambert_dv(MU_SUN)` in the same way.
   - Return Δv_S.

The script prints both candidates and then chooses the one with lower norm:

    chosen_dv, mu_use, label = (dv_E, MU_EARTH, "Earth") or (dv_S, MU_SUN, "Sun")

The chosen ΔV is applied in the inertial frame at BURN_EPOCH.

Deputy propagation
------------------
Once ΔV has been selected:

- Initial deputy state at BURN_EPOCH:

      r_c0, v_c0 = chief_state(BURN_EPOCH)
      y0 = [r_c0, v_c0 + chosen_dv/1e3]

  (positions and velocities in km and km/s).

- Propagate under the chosen central μ (`mu_use`) using:

      solve_ivp(two_body, [0, t_end], y0, t_eval=t_eval, ...)

  where `two_body(t, y, mu)` applies:

      a = −μ r / ‖r‖³

  in km, km/s, km/s² units.

- `t_eval` is a 1-second grid from 0 to t_end.

Final LVLH error
----------------
At the final time (RV_EPOCH):

- Deputy inertial position: r_dep[-1].
- Apophis geocentric state: r_apo, v_apo (from SPICE).
- Rebuild Apophis LVLH:

      C_A = dcm_lvlh(r_apo, v_apo)  # LVLH→ECI

- Map deputy relative position into Apophis LVLH:

      offset_final = C_A.T @ (r_dep[-1] − r_apo)   # [km]

The script prints:

- Achieved Apophis LVLH offset:

      offset_final (km)

- Residual error in metres:

      (offset_final − TARGET_LVLH_A) × 1e3

Plots
-----
1. 3D overview:
   - GEO ring at 42 164 km in the equatorial plane.
   - 12 h Apophis geocentric track (from SPICE).
   - Chief trajectory (split into pre-deploy and post-deploy segments).
   - Deputy trajectory (propagated with chosen μ).
   - Markers for:
       * Chief start,
       * Burn point,
       * Deputy final,
       * Apophis at RV_EPOCH.
   - Axis limits chosen to encompass all tracks (slightly padded).

2. XY projection:
   - Same elements projected into the XY plane.
   - Equal aspect ratio.
   - Legends and titles indicating the automatically selected central body.

Usage
-----
Prerequisites:

- A valid `chief_reference.npz` in the working directory.
- Basilisk installed with the SPICE kernels listed above.
- SciPy and Matplotlib.

Run:

    python basilik_scenario_apophis_single_impulse_deputy.py

The script will:

- Load chief ephemeris from `chief_reference.npz`,
- Sample Apophis’s ephemeris over 12 h,
- Compute Apophis LVLH target and trial Lambert ΔVs (Earth μ and Sun μ),
- Choose the lower-ΔV option,
- Propagate the deputy under the chosen central body,
- Print Apophis LVLH offset at RV_EPOCH and residual,
- Plot the GEO ring, chief, deputy, and Apophis track in 3D and XY views.
"""


# ────── imports ────────────────────────────────────────────────────────────
from __future__ import annotations
import os, sys, math
from pathlib import Path
import numpy as np, numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from Basilisk import __path__ as bskPath
from Basilisk.utilities   import SimulationBaseClass, macros, orbitalMotion
from Basilisk.utilities.simIncludeGravBody import gravBodyFactory
from Basilisk.simulation  import spacecraft, simpleNav, spiceInterface, ephemerisConverter
from Basilisk.fswAlgorithms import lambertPlanner, lambertSolver
from scipy.integrate import solve_ivp

# ────── scenario constants ────────────────────────────────────────────────
DEPLOY_EPOCH = 10_023.0
BURN_EPOCH   = DEPLOY_EPOCH + 1.0
RV_EPOCH     = 31_170.0
TARGET_LVLH_A = np.array([1.43, 2.55, -0.27])   # km  (Apophis-centred)
DT_STEP       = 1                          # deputy propagation [s]

MU_EARTH = 3.986004418e5                        # km³ s⁻²
MU_SUN   = orbitalMotion.MU_SUN/1e9             # km³ s⁻²

# ────── load chief ephemeris ───────────────────────────────────────────────
npz = Path("chief_reference.npz")
if not npz.is_file():
    sys.exit("❌ chief_reference.npz not found.")

d = np.load(npz)
t_ref, r_ref, v_ref = d["times"], d["positions"], d["velocities"]

def chief_state(t):
    r = np.array([np.interp(t, t_ref, r_ref[:,k]) for k in range(3)])
    v = np.array([np.interp(t, t_ref, v_ref[:,k]) for k in range(3)])
    return r, v

r_c0, v_c0 = chief_state(BURN_EPOCH)
r_cF, v_cF = chief_state(RV_EPOCH)

# ────── tiny SPICE run to get Apophis inertial state at RV ────────────────
SP = os.path.join(bskPath[0], "supportData", "EphemerisData") + os.sep
kern = ["naif0012.tls","pck00010.tpc","de-403-masses.tpc","de430.bsp","20099942.bsp"]

sp_sim = SimulationBaseClass.SimBaseClass()
tsk    = sp_sim.CreateNewTask("sp", macros.sec2nano(600))
sp_sim.CreateNewProcess("dyn").addTask(tsk)

spi = spiceInterface.SpiceInterface(); spi.SPICEDataPath=SP
spi.UTCCalInit="2029 APR 13 11:38:28"
for k in kern: spi.loadSpiceKernel(k, SP)
spi.addPlanetNames(["Earth","Sun","20099942"]); spi.zeroBase="Sun"

conv = ephemerisConverter.EphemerisConverter()
for m in spi.planetStateOutMsgs: conv.addSpiceInputMsg(m)
sp_sim.AddModelToTask("sp", spi); sp_sim.AddModelToTask("sp", conv)

eRec = conv.ephemOutMsgs[0].recorder()   # Earth
aRec = conv.ephemOutMsgs[2].recorder()   # Apophis
sp_sim.AddModelToTask("sp", eRec); sp_sim.AddModelToTask("sp", aRec)

sp_sim.InitializeSimulation()
sp_sim.ConfigureStopTime(macros.sec2nano(12*3600))
sp_sim.ExecuteSimulation()

idx_rv = int(round(RV_EPOCH/600))
r_apo  = (aRec.r_BdyZero_N[idx_rv]-eRec.r_BdyZero_N[idx_rv]) / 1e3  # km
v_apo  = (aRec.v_BdyZero_N[idx_rv]-eRec.v_BdyZero_N[idx_rv]) / 1e3  # km/s

# ────── LVLH utilities ────────────────────────────────────────────────────
def dcm_lvlh(r,v):
    ih=r/la.norm(r); kh=np.cross(r,v); kh/=la.norm(kh); jh=np.cross(kh,ih)
    return np.column_stack((ih,jh,kh))          # LVLH→ECI

C_NA = dcm_lvlh(r_apo, v_apo)
r_dep_target = r_apo + C_NA @ TARGET_LVLH_A      # km  (ECI goal)

# ────── helper: one Lambert ΔV for a given μ ──────────────────────────────
def lambert_dv(mu_km3):
    s = SimulationBaseClass.SimBaseClass()
    dyn=s.CreateNewTask("dyn",macros.sec2nano(0.1))
    fsw=s.CreateNewTask("fsw",macros.sec2nano(0.1))
    s.CreateNewProcess("dynP").addTask(dyn)
    s.CreateNewProcess("fswP").addTask(fsw)

    sc=spacecraft.Spacecraft(); sc.hub.r_CN_NInit=r_c0*1e3
    nav=simpleNav.SimpleNav(); nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    lp=lambertPlanner.LambertPlanner()
    lp.navTransInMsg.subscribeTo(nav.transOutMsg)
    lp.setR_TN_N(r_dep_target*1e3)
    lp.setMu(mu_km3*1e9)                         # km³→m³
    lp.setManeuverTime(10.0)
    lp.setFinalTime(10.0+RV_EPOCH-BURN_EPOCH)
    ls=lambertSolver.LambertSolver()
    ls.lambertProblemInMsg.subscribeTo(lp.lambertProblemOutMsg)
    rec=ls.lambertSolutionOutMsg.recorder()
    for m in (sc,nav): s.AddModelToTask("dyn",m)
    for m in (lp,ls,rec): s.AddModelToTask("fsw",m)
    s.InitializeSimulation(); s.ConfigureStopTime(macros.sec2nano(2.0)); s.ExecuteSimulation()
    v_req = rec.v1_N[-1]/1e3                     # km/s
    return (v_req - v_c0)*1e3                   # m/s

dv_E = lambert_dv(MU_EARTH)
dv_S = lambert_dv(MU_SUN )

# ────── choose the lower-ΔV solution ───────────────────────────────────────
chosen_dv, mu_use, label = (dv_E, MU_EARTH, "Earth") if la.norm(dv_E) < la.norm(dv_S) \
                           else (dv_S, MU_SUN, "Sun")

print(f"ΔV candidate (Earth μ): {dv_E}  |ΔV|={la.norm(dv_E):.3f} m/s")
print(f"ΔV candidate (Sun   μ): {dv_S}  |ΔV|={la.norm(dv_S):.3f} m/s\n")
print(f"⇒  Using {label}-centred solution")
print(f"ΔV used (ECI): {chosen_dv} m/s   |ΔV| = {la.norm(chosen_dv):.3f} m/s")

# ────── propagate deputy with chosen μ (two-body) ─────────────────────────
def two_body(t,y,mu): r=y[:3]; v=y[3:]; a=-mu*r/la.norm(r)**3; return np.hstack((v,a))

t_end = RV_EPOCH - BURN_EPOCH
t_eval = np.arange(0.0, t_end + DT_STEP*0.5, DT_STEP)  # ensure ≤ t_end
y0 = np.hstack((r_c0,  v_c0 + chosen_dv*1e-3))
sol = solve_ivp(two_body, [0.0, t_end], y0, t_eval=t_eval,
                rtol=1e-10, atol=1e-13, args=(mu_use,))

r_dep = sol.y[:3].T

# achieved Apophis LVLH offset
offset_final = dcm_lvlh(r_apo, v_apo).T @ (r_dep[-1] - r_apo)
print("\nAchieved Apophis-LVLH offset (km):", offset_final)
print("Residual (m):", (offset_final - TARGET_LVLH_A)*1e3, "\n")

# ────── GEO ring + Apophis 12 h track (for context) ───────────────────────
θ = np.linspace(0,2*math.pi,400)
geo_ring=np.column_stack((42164*np.cos(θ),42164*np.sin(θ),np.zeros_like(θ)))
apo_track=(aRec.r_BdyZero_N - eRec.r_BdyZero_N)/1e3

# chief split for plotting
idx_dep = np.searchsorted(t_ref, DEPLOY_EPOCH)
r_chief_pre  = r_ref[:idx_dep+1]
r_chief_post = r_ref[idx_dep:]

# ────── plots ─────────────────────────────────────────────────────────────
plt.style.use("bmh")

fig = plt.figure(figsize=(10,8))
ax  = fig.add_subplot(111, projection='3d')
ax.plot(*geo_ring.T, "--", color="0.6", label="GEO ring")
ax.plot(*apo_track.T, "b", lw=1.1, label="Apophis 12 h")
ax.plot(*r_chief_pre.T,  "c", lw=1.2, label="Chief pre-deploy")
ax.plot(*r_chief_post.T, "c", lw=2.0, label="Chief post-deploy")
ax.plot(*r_dep.T,        "m", lw=1.4, label="Deputy")
ax.scatter(*r_chief_pre[0],  c="green",  s=55, label="Chief start")
ax.scatter(*r_chief_pre[-1], c="red",    s=60, label="Burn")
ax.scatter(*r_dep[-1],       c="orange", s=70, label="Deputy final")
ax.scatter(*r_apo,           c="k",      s=50, label="Apophis @ RV")
rng=np.max(np.abs(np.vstack((geo_ring,r_ref,r_dep,apo_track))))*1.05
ax.set(xlim=(-rng,rng), ylim=(-rng,rng), zlim=(-rng,rng),
       xlabel="X [km]", ylabel="Y [km]", zlabel="Z [km]")
ax.set_title("Chief & Deputy – automatically chosen central body")
ax.legend(); plt.tight_layout()

# XY projection
plt.figure(figsize=(7,6))
plt.plot(geo_ring[:,0], geo_ring[:,1], "--", color="0.6", label="GEO ring")
plt.plot(apo_track[:,0], apo_track[:,1], "b", lw=1.1, label="Apophis track")
plt.plot(r_chief_pre[:,0],  r_chief_pre[:,1],  "c", lw=1.2, label="Chief pre-deploy")
plt.plot(r_chief_post[:,0], r_chief_post[:,1], "c", lw=2.0, label="Chief post-deploy")
plt.plot(r_dep[:,0],        r_dep[:,1],        "m", lw=1.4, label="Deputy")
plt.scatter(r_chief_pre[0,0],  r_chief_pre[0,1],  c="green",  label="Chief start")
plt.scatter(r_chief_pre[-1,0], r_chief_pre[-1,1], c="red",    label="Burn")
plt.scatter(r_dep[-1,0],       r_dep[-1,1],       c="orange", label="Deputy final")
plt.scatter(r_apo[0],          r_apo[1],          c="k",      label="Apophis @ RV")
plt.gca().set_aspect('equal')
plt.xlabel("X [km]"); plt.ylabel("Y [km]")
plt.title("ECI XY projection"); plt.legend(); plt.tight_layout()
plt.show()

