#!/usr/bin/env python3
"""
basilik_geo_to_apophis_global_dv_opt.py
=======================================

Purpose
-------
Global ΔV minimisation for a **single-impulse** GEO → Apophis transfer with
in-plane and out-of-plane freedom, using:

- Decision variables:
    * θ      ∈ [0°, 360°]   – true anomaly of the GEO spacecraft at departure,
    * i      ∈ [0°, 6°]     – inclination of the GEO orbit,
    * Ω      ∈ [0°, 360°]   – RAAN of the GEO orbit,
    * TOF    ∈ [2 h, 12 h]  – time of flight (Lambert arc duration).
- Objective:
    * Minimize `||ΔV_dep|| + ||ΔV_arr||`, where:
        - ΔV_dep is the departure burn to match the Lambert v₁,
        - ΔV_arr is a small “plane-trim” at arrival to match Apophis’s
          instantaneous velocity.

The script uses SciPy’s `differential_evolution` as a global optimizer, with
each fitness evaluation calling a Basilisk-based Lambert solver and SPICE-based
Apophis ephemerides (no direct spiceypy usage).

High-level flow
---------------
1. Set mission constants:
   - Earth and GEO parameters:
        MU_EARTH = 3.986004418e5 km³/s²
        R_EARTH  = 6378.137 km
        GEO_R    = 42164.0 km
        T_GEO    = orbital period at GEO radius
   - Departure epoch (UTC):
        DEPART_UTC = "2029 APR 13 11:38:28"
     converted to `t0_dt` (datetime).

2. Define geometric helpers:
   - `rot_z(angle)`, `rot_x(angle)`:
       * Standard rotation matrices about z and x.
   - `geo_state(theta, inc, RAAN)`:
       * Starts from a circular equatorial GEO orbit:
            r_eq = GEO_R [cosθ, sinθ, 0]
            v_eq = sqrt(μ/GEO_R) [−sinθ, cosθ, 0]
       * Applies `R = Rz(Ω) @ Rx(i)` to tilt the orbit.
       * Returns (r₁, v_geo) in km and km/s.

3. Propagation helper:
   - `rk4_two_body(r0, v0, tof, mu, steps=600)`:
       * Simple 4th-order Runge–Kutta integrator for two-body dynamics.
       * Used only for plotting the Lambert trajectory in Earth-centred
         inertial (ECI) coordinates.

4. Basilisk Lambert solver wrapper:
   - `lambert_solver(r1_km, r2_km, tof_s)`:
       * Builds a tiny Basilisk sim with:
            - Earth gravity (via `gravBodyFactory().createEarth()`),
            - A spacecraft + SimpleNav for current state,
            - `LambertPlanner` configured with:
                  target position r₂ (Apophis arrival),
                  μ = μ_Earth (SI),
                  maneuver time = 10 s,
                  final time = 10 + TOF,
                  0 revolutions,
            - `LambertSolver` to solve the Lambert problem.
       * Runs for a few seconds of sim time and returns:
            (v₁, v₂) in km/s, extracted from the solver’s output message.

5. SPICE-based ephemerides (Earth + Apophis)
   - Construct a separate Basilisk simulation (`sim_sp`) with:
       * `SpiceInterface` (path from `bskPath`, kernels:
             naif0012.tls, pck00010.tpc, de-403-masses.tpc,
             de430.bsp, 20099942.bsp),
         with:
             UTCCalInit = DEPART_UTC,
             zeroBase   = "Sun".
       * `EphemerisConverter` to generate planet ephemeris messages.
       * Recorders:
             earthRec = ephemOutMsgs[0],
             apoRec   = ephemOutMsgs[2].
       * Run for 12 h (43 200 s) at 60 s steps.

   - `apo_state(t_sec)`:
       * Returns Apophis geocentric state at t_sec:
             (r_A_geo, v_A_geo) [km, km/s]
         by subtracting Earth’s state from Apophis’s state at the closest
         60 s index.

6. Fitness function for global search
   - `fitness(x)` with x = (θ, inc, RAAN, TOF):
       * Compute GEO initial state:
             r₁, v_geo = geo_state(θ, inc, RAAN).
       * Query Apophis state at t = TOF:
             r₂, v_apo = apo_state(TOF).
       * Use `lambert_solver(r₁, r₂, TOF)` to obtain:
             v₁, v₂ = Lambert departure/arrival velocities (km/s).
       * Departure mismatch:
             ΔV_dep = v₁ − v_geo.
       * Arrival mismatch:
             ΔV_arr = v₂ − v_apo.
       * Objective:
             J = ||ΔV_dep|| + ||ΔV_arr||   (km/s).

   - Bounds:
       * θ   ∈ [0, 2π],
       * i   ∈ [0, 6°] (converted to radians),
       * Ω   ∈ [0, 2π],
       * TOF ∈ [2 h, 12 h] ≈ [7 200 s, 43 200 s].

7. Global optimisation with differential evolution
   - SciPy’s `differential_evolution` is configured with:
       * strategy  = "best1bin"
       * maxiter   = 800
       * popsize   = 120
       * tol       = 1e-7
       * polish    = True
       * updating  = "deferred"
       * workers   = -1 (parallel evaluation, if supported)
   - On completion:
       * `θ_opt, inc_opt, RAAN_opt, tof_opt = result.x`
       * `dv_opt = result.fun` (km/s).

8. Reconstruct the optimal trajectory
   - GEO departure state:
        r₁_opt, v_geo_opt = geo_state(θ_opt, inc_opt, RAAN_opt).
   - Apophis arrival state:
        r₂_opt, v_apo_opt = apo_state(tof_opt).
   - Lambert solution at optimum:
        v₁_opt, v₂_opt = lambert_solver(r₁_opt, r₂_opt, tof_opt).
   - Epochs:
        * Departure UTC: dep_dt =
              t0_dt + θ_opt/(2π) * T_GEO   (i.e., slide around the GEO ring).
        * Arrival UTC:   arr_dt = dep_dt + tof_opt seconds.
   - Trajectory for plotting:
        traj_opt = rk4_two_body(r₁_opt, v₁_opt, tof_opt, MU_EARTH, steps=600).

9. Plotting
-----------
The script generates a 3D Earth-centred view showing:

- Earth as a sphere of radius `R_EARTH` (km) via a simple `plot_surface`.
- GEO ring:
    * A tilted circle defined by `geo_state(th, inc_opt, RAAN_opt)` for a
      grid of θ ∈ [0, 2π].
- Apophis track:
    * A curve of Apophis geocentric positions sampled over 0–12 h.
- Lambert arc:
    * Propagated trajectory from `rk4_two_body`.
- Marker points:
    * Departure (ΔV depart) at r₁_opt.
    * Arrival at the final point, where the Lambert arc and Apophis coincide.

Axis limits are set based on the maximum extent of the ring, arc, and track
so that the main structures are all visible. The title shows the optimal total
ΔV in m/s.

Mission summary (stdout)
------------------------
After plotting, the script prints:

- Departure UTC (optimised):
      dep_dt in ISO-like format.
- Arrival UTC:
      arr_dt in ISO-like format.
- ΔV_depart (m/s):
      ||v₁_opt − v_geo_opt|| × 1e3.
- ΔV_arrival (m/s):
      ||v₂_opt − v_apo_opt|| × 1e3.
- ΔV_total (m/s):
      `dv_opt × 1e3`.

Usage
-----
Requirements:

- Basilisk installed with SPICE ephemeris data (especially `20099942.bsp`),
- SciPy (for differential_evolution),
- Matplotlib and NumPy.

To run the optimisation:

    python basilik_geo_to_apophis_global_dv_opt.py

This will:

- Build a SPICE-based Apophis + Earth ephemeris window around the reference
  epoch,
- Run a global search over [θ, i, Ω, TOF] using differential evolution,
- Reconstruct and visualise the best Lambert transfer,
- Print a mission-style summary of departure/arrival times and ΔV budget.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import differential_evolution

from Basilisk import __path__ as bskPath
from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.utilities.simIncludeGravBody import gravBodyFactory
from Basilisk.simulation import (
    spiceInterface, ephemerisConverter, spacecraft, simpleNav
)
from Basilisk.fswAlgorithms import lambertPlanner, lambertSolver

# — Constants —
MU_EARTH = 3.986004418e5     # km³/s²
R_EARTH  = 6378.137          # km
GEO_R    = 42164.0           # km
T_GEO    = 2*np.pi*np.sqrt(GEO_R**3/MU_EARTH)

DEPART_UTC = "2029 APR 13 11:38:28"
t0_dt      = datetime.strptime(DEPART_UTC, "%Y %b %d %H:%M:%S")

# — Helpers — 
def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def geo_state(theta, inc, RAAN):
    # Circular GEO in equatorial plane, then tilt by (i, Ω).
    r_eq = GEO_R * np.array([np.cos(theta), np.sin(theta), 0.0])
    v_eq = np.sqrt(MU_EARTH/GEO_R) * np.array([-np.sin(theta), np.cos(theta), 0.0])
    R = rot_z(RAAN) @ rot_x(inc)
    return R @ r_eq, R @ v_eq

def rk4_two_body(r0, v0, tof, mu, steps=600):
    h, state = tof/steps, np.hstack((r0, v0)).astype(float)
    pos = np.empty((steps+1, 3))
    def deriv(st):
        r, v = st[:3], st[3:]
        a = -mu * r/np.linalg.norm(r)**3
        return np.hstack((v, a))
    for k in range(steps+1):
        pos[k] = state[:3]
        k1 = deriv(state)
        k2 = deriv(state + 0.5*h*k1)
        k3 = deriv(state + 0.5*h*k2)
        k4 = deriv(state +    h*k3)
        state += (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return pos

def lambert_solver(r1_km, r2_km, tof_s):
    # Returns (v1,kms, v2,kms)
    sim = SimulationBaseClass.SimBaseClass()
    dynT = sim.CreateNewTask("dynT", macros.sec2nano(0.5))
    fswT = sim.CreateNewTask("fswT", macros.sec2nano(0.5))
    sim.CreateNewProcess("dyn").addTask(dynT)
    sim.CreateNewProcess("fsw").addTask(fswT)

    mu_m = gravBodyFactory().createEarth().mu
    sc = spacecraft.Spacecraft()
    sc.hub.r_CN_NInit = r1_km * 1e3
    nav = simpleNav.SimpleNav()
    nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    sim.AddModelToTask("dynT", sc)
    sim.AddModelToTask("dynT", nav)

    planner = lambertPlanner.LambertPlanner()
    planner.navTransInMsg.subscribeTo(nav.transOutMsg)
    planner.setR_TN_N(r2_km * 1e3)
    planner.setMu(mu_m)
    planner.setNumRevolutions(0)
    planner.setManeuverTime(10.0)
    planner.setFinalTime(10.0 + tof_s)

    solver = lambertSolver.LambertSolver()
    solver.lambertProblemInMsg.subscribeTo(planner.lambertProblemOutMsg)
    rec = solver.lambertSolutionOutMsg.recorder()
    sim.AddModelToTask("fswT", planner)
    sim.AddModelToTask("fswT", solver)
    sim.AddModelToTask("fswT", rec)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(5.0))
    sim.ExecuteSimulation()

    return rec.v1_N[-1]/1e3, rec.v2_N[-1]/1e3

# — SPICE setup: record Earth & Apophis for 12 h at 60 s steps —
SPICE_DIR = os.path.join(bskPath[0],"supportData","EphemerisData")+os.sep
KERNELS   = ["naif0012.tls","pck00010.tpc","de-403-masses.tpc","de430.bsp","20099942.bsp"]

sim_sp = SimulationBaseClass.SimBaseClass()
tsk    = sim_sp.CreateNewTask("tsk", macros.sec2nano(60))
sim_sp.CreateNewProcess("dyn").addTask(tsk)

spi = spiceInterface.SpiceInterface()
spi.SPICEDataPath = SPICE_DIR
spi.UTCCalInit    = t0_dt.strftime("%Y %B %d, %H:%M:%S.0 (UTC)")
for k in KERNELS: spi.loadSpiceKernel(k, SPICE_DIR)
spi.addPlanetNames(["Earth","Sun","20099942"])
spi.zeroBase = "Sun"
sim_sp.AddModelToTask("tsk", spi)

conv = ephemerisConverter.EphemerisConverter()
for msg in spi.planetStateOutMsgs:
    conv.addSpiceInputMsg(msg)
sim_sp.AddModelToTask("tsk", conv)

earthRec = conv.ephemOutMsgs[0].recorder()
apoRec   = conv.ephemOutMsgs[2].recorder()
sim_sp.AddModelToTask("tsk", earthRec)
sim_sp.AddModelToTask("tsk", apoRec)

sim_sp.InitializeSimulation()
sim_sp.ConfigureStopTime(macros.sec2nano(43_200))
sim_sp.ExecuteSimulation()

def apo_state(t_sec):
    # returns (r, v) geocentric [km, km/s]
    idx = int(round(t_sec/60))
    rA = apoRec.r_BdyZero_N[idx]/1e3
    rE = earthRec.r_BdyZero_N[idx]/1e3
    vA = apoRec.v_BdyZero_N[idx]/1e3
    vE = earthRec.v_BdyZero_N[idx]/1e3
    return rA - rE, vA - vE

# — Objective for differential_evolution —
def fitness(x):
    θ, inc, RAAN, tof = x
    r1, v_geo = geo_state(θ, inc, RAAN)
    r2, v_apo = apo_state(tof)
    v1, v2    = lambert_solver(r1, r2, tof)
    dv_dep    = np.linalg.norm(v1 - v_geo)
    dv_arr    = np.linalg.norm(v2 - v_apo)
    return dv_dep + dv_arr

# — Bounds: θ, i, Ω, TOF —
bounds = [
    (0,          2*np.pi),
    (0,          np.deg2rad(6)),
    (0,          2*np.pi),
    (7_200,      43_200)
]

# — Global search with Differential Evolution —
result = differential_evolution(
    fitness, bounds,
    strategy="best1bin",
    maxiter=800,
    popsize=120,
    tol=1e-7,
    polish=True,
    updating="deferred",
    workers=-1
)

θ_opt, inc_opt, RAAN_opt, tof_opt = result.x
dv_opt = result.fun

# — Reconstruct optimal trajectory & times —
r1_opt, vgeo_opt = geo_state(θ_opt, inc_opt, RAAN_opt)
r2_opt, vapo_opt = apo_state(tof_opt)
v1_opt, v2_opt   = lambert_solver(r1_opt, r2_opt, tof_opt)

dep_dt = t0_dt + timedelta(seconds=float(θ_opt/(2*np.pi)*T_GEO))
arr_dt = dep_dt + timedelta(seconds=float(tof_opt))
traj_opt = rk4_two_body(r1_opt, v1_opt, tof_opt, MU_EARTH, steps=600)

# — Plot GEO ring, transfer, and Apophis track —
theta_ring = np.linspace(0, 2*np.pi, 400)
geo_ring   = np.array([geo_state(th, inc_opt, RAAN_opt)[0] for th in theta_ring])
apo_track  = np.array([apo_state(t)[0] for t in np.linspace(0, 43_200, 721)])

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')
u,v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
ax.plot_surface(R_EARTH*np.cos(u)*np.sin(v),
                R_EARTH*np.sin(u)*np.sin(v),
                R_EARTH*np.cos(v),
                color='lightsteelblue', alpha=0.5, linewidth=0)

ax.plot(geo_ring[:,0], geo_ring[:,1], geo_ring[:,2],
        color='0.6', linewidth=0.8, label='GEO ring')
ax.scatter(*r1_opt, c='r', s=60, label='ΔV depart')
ax.plot(traj_opt[:,0], traj_opt[:,1], traj_opt[:,2],
        'c', linewidth=2, label='Lambert arc')
ax.plot(apo_track[:,0], apo_track[:,1], apo_track[:,2],
        'b', linewidth=2, label='Apophis track')
ax.scatter(*r2_opt, c='m', s=70, label='SC & Apophis @ arrival')

rng = np.max(np.abs(np.vstack((geo_ring, traj_opt, apo_track))))
ax.set_xlim(-rng,rng); ax.set_ylim(-rng,rng); ax.set_zlim(-rng,rng)
ax.set_xlabel('X [km]'); ax.set_ylabel('Y [km]'); ax.set_zlabel('Z [km]')
ax.set_title(f'Optimised transfer (ΔV = {dv_opt*1e3:5.1f} m/s)')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

# — Mission summary —
print("\nOptimised global ΔV (in‑plane + out‑of‑plane):")
print(f"  Departure UTC  : {dep_dt.strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"  Arrival   UTC  : {arr_dt.strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"  ΔV_depart (m/s): {np.linalg.norm(v1_opt - vgeo_opt)*1e3:6.1f}")
print(f"  ΔV_arrival (m/s): {np.linalg.norm(v2_opt - vapo_opt)*1e3:6.1f}")
print(f"  ΔV total (m/s): {dv_opt*1e3:6.1f}")
