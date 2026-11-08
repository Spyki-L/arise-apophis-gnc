#!/usr/bin/env python3
"""
basilik_scenario_lambert_flyby_apophis.py
=========================================

Purpose
-------
Basilisk-only (no spiceypy) scenario that designs and executes a single-burn
Lambert transfer from a GEO orbit to a fly-by of asteroid (99942) Apophis,
using Basilisk’s own SPICE interface and Lambert FSW modules.

Key features:
- Uses Basilisk’s SPICE modules to get Apophis and Earth ephemerides.
- Starts from a GEO orbit at a specified epoch.
- Solves a single-impulse Lambert problem to hit Apophis’s geocentric position
  at a prescribed time of flight (TOF).
- Applies the Lambert ΔV as a short finite burn.
- Propagates a one-satellite system under Earth gravity to the fly-by point.
- Outputs ΔV magnitude and generates a compact set of plots:
    * Earth-centred XY projection with transfer arc and markers,
    * 3-D Earth-centred view,
    * ΔV component bar chart,
    * Truth vs SimpleNav comparison for the X-position.

Mission parameters
------------------
- Departure epoch (UTC):

    EPOCH_STR = "2029 APR 13 11:38:28"

- Time of flight:

    TOF_SEC = 30_871.9 s   (≈ 8.58 h)

- Burn timing and run segments:
    * Lambert burn is commanded at `MANEUVER_TIME = 10 s` after the epoch.
    * Arrival time is `ARRIVAL_TIME = MANEUVER_TIME + TOF_SEC`.
    * The script:
        1. Propagates from 0 → MANEUVER_TIME,
        2. Applies ΔV as a 1 s finite burn (extForceTorque),
        3. Coasts from burn end to ARRIVAL_TIME.

- GEO initial orbit (Earth-centred, km and radians):
    * Semi-major axis: aG_km = 42 164.0 km
    * eG = 0, iG = 0
    * Right ascension of ascending node: RAAN ≈ 53.3953 deg
    * Argument of perigee: 0
    * True anomaly: 0

Implementation outline
----------------------
1. Phase 0 — Apophis target position at arrival
   - Build a mini Basilisk simulation:
       * `SpiceInterface` configured for EPOCH_STR and standard kernels:
             naif0012.tls, pck00010.tpc, de-403-masses.tpc,
             de430.bsp, 20099942.bsp.
         with `zeroBase = "Earth"`.
       * `ephemerisConverter.EphemerisConverter` to get ephemerides.
       * Recorder for Apophis (index 1) in the ephem outputs.
   - Run to `arrival_s = ARRIVAL_TIME`.
   - Extract Apophis geocentric position at arrival (in metres).

2. Main simulation
   - Processes & tasks:
       * `DynProc / DynTask` stepping at 1 s for dynamics and nav.
       * `FSWProc / FSWTask` stepping at 1 s for Lambert FSW.
   - Gravity:
       * Earth created via `simIncludeGravBody.gravBodyFactory()`.
       * Earth’s μ in SI used for GEO orbital elements.
   - SPICE and ephemerides for plotting:
       * Same kernel set as Phase 0, zeroBase = "Earth".
       * Ephemeris converter + recorder for Apophis states.
   - GEO spacecraft:
       * ClassicElements (a, e, i, Ω, ω, f) set to GEO parameters.
       * Convert to r0, v0 via `orbitalMotion.elem2rv(MU_km, oe)`.
       * Initialize spacecraft state in metres.
   - SimpleNav:
       * `SimpleNav` subscribes to the spacecraft’s state output.
       * Translations recorded for truth vs nav comparison.

3. Lambert guidance chain
   - `lambertPlanner.LambertPlanner`:
       * Target position set to Apophis arrival position (m).
       * Maneuver time: MANEUVER_TIME (10 s).
       * Final time: ARRIVAL_TIME.
       * μ set to Earth’s μ.
       * 0 revolutions.
   - `lambertSolver.LambertSolver`:
       * Subscribes to planner’s problem output and produces a ΔV solution.
   - `lambertValidator.LambertValidator`:
       * Validates the solution via constraints:
           - max distance to target, min orbit radius,
           - state uncertainty, ΔV uncertainty, convergence tolerance.
       * Recorder (`dvRec`) used to capture `dvInrtlCmd` for the burn.

4. Executing the burn and coasting
   - Initialize and run until `MANEUVER_TIME`.
   - Retrieve `dv_m_s = dvRec.dvInrtlCmd[-1]`.
   - Create an `ExtForceTorque` effector:
       * `extForce_N = dv_m_s * m / POST_BURN_SEC` (1 s “boxcar” thrust).
       * Attach to spacecraft and add to the dynamics task.
   - Disable FSW tasks (`fswProc.disableAllTasks()`) after the burn to avoid
     further Lambert cycles.
   - Propagate:
       * From `MANEUVER_TIME` to `MANEUVER_TIME + POST_BURN_SEC` (burn),
       * Then to ARRIVAL_TIME (coast).

Outputs and plots
-----------------
- Terminal ΔV report:

    *** GEO → Apophis fly-by ***
    Departure epoch : <EPOCH_STR>
    ΔV [km/s]       : <‖ΔV‖/1e3>

- Extracted histories:
    * `times`  : simulation times [s],
    * `geo_km` : spacecraft positions [km],
    * `nav_km` : SimpleNav positions [km],
    * `apo_km` : Apophis geocentric positions [km].

- Indexing:
    * Burn index `bIdx` and arrival index `aIdx` are computed via
      `np.searchsorted` and clamped to stay within the arrays.

If `show_plots` (or `SHOW_PLOTS`) is True:

1. Earth-centred XY view:
   - Apophis XY track (km).
   - GEO XY track (km).
   - Transfer arc between burn and arrival highlighted.
   - Burn point and fly-by point marked.

2. 3-D Earth-centred view:
   - Same sets as above, in 3D.

3. ΔV component bar chart:
   - Bars for [ΔVx, ΔVy, ΔVz] in km/s.

4. Truth vs SimpleNav (X):
   - Truth X vs time [h].
   - Nav X vs time (scatter) [h].

All plots use Matplotlib and are shown with `plt.show()`.

Usage
-----
- Ensure the SPICE kernel files listed are present in Basilisk’s
  `EphemerisData` directory, including `20099942.bsp`.
- Run:

    python basilik_scenario_lambert_flyby_apophis.py

- The script will:
    * Query Apophis’s location at the target arrival time.
    * Build and run the Lambert-guided fly-by scenario.
    * Print out the ΔV magnitude.
    * Show the diagnostic plots if `SHOW_PLOTS` is True.
"""


from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from Basilisk.utilities import (SimulationBaseClass, macros,
                                simIncludeGravBody, orbitalMotion)
from Basilisk.simulation  import (spacecraft, simpleNav,
                                  spiceInterface, ephemerisConverter,
                                  extForceTorque)
from Basilisk.fswAlgorithms import (lambertPlanner, lambertSolver,
                                    lambertValidator)
from Basilisk import __path__ as bskPath

SPICE_PATH = os.path.join(bskPath[0], "supportData", "EphemerisData") + os.sep

# Mission parameters
EPOCH_STR     = "2029 APR 13 11:38:28"
TOF_SEC       = 30_871.9
MANEUVER_TIME = 10.0                   # burn 10 s after epoch
ARRIVAL_TIME  = MANEUVER_TIME + TOF_SEC
POST_BURN_SEC = 1.0
SHOW_PLOTS    = True

# GEO orbit elements (km, rad)
aG_km   = 42_164.0
eG, iG, raanG, argpG, ta0G = 0.0, 0.0, np.deg2rad(53.3953), 0.0, 0.0


def get_apophis_pos_m(arrival_s):
    """Phase‑0 sim: SPICE → Apophis ECI position at t = arrival_s (metres)."""
    sim  = SimulationBaseClass.SimBaseClass()
    task = sim.CreateNewTask("EphemTask", macros.sec2nano(10.0))
    proc = sim.CreateNewProcess("EphemProc")
    proc.addTask(task)

    # SPICE interface
    spice = spiceInterface.SpiceInterface()
    spice.SPICEDataPath = SPICE_PATH
    spice.UTCCalInit = datetime.strptime(
        EPOCH_STR, "%Y %b %d %H:%M:%S"
    ).strftime("%Y %B %d, %H:%M:%S.0 (UTC)")
    for k in ["naif0012.tls", "pck00010.tpc", "de-403-masses.tpc",
              "de430.bsp", "20099942.bsp"]:
        spice.loadSpiceKernel(k, spice.SPICEDataPath)
    spice.addPlanetNames(["Earth", "20099942"])
    spice.zeroBase = "Earth"
    sim.AddModelToTask("EphemTask", spice)

    # Ephemeris converter
    conv = ephemerisConverter.EphemerisConverter()
    for msg in spice.planetStateOutMsgs:
        conv.addSpiceInputMsg(msg)
    sim.AddModelToTask("EphemTask", conv)

    # Record Apophis (index 1)
    apoRec = conv.ephemOutMsgs[1].recorder()
    sim.AddModelToTask("EphemTask", apoRec)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(arrival_s))
    sim.ExecuteSimulation()

    return apoRec.r_BdyZero_N[-1]  # metres


def run(show_plots=True):
    # 1) Compute target at arrival
    rT_m = get_apophis_pos_m(ARRIVAL_TIME)

    # 2) Build main simulation
    sim     = SimulationBaseClass.SimBaseClass()
    dynTask = "DynTask"
    fswTask = "FSWTask"
    dynProc = sim.CreateNewProcess("DynProc")
    fswProc = sim.CreateNewProcess("FSWProc")
    dynProc.addTask(sim.CreateNewTask(dynTask, macros.sec2nano(1.0)))
    fswProc.addTask(sim.CreateNewTask(fswTask, macros.sec2nano(1.0)))

    # Gravity factory
    gravF = simIncludeGravBody.gravBodyFactory()
    earth = gravF.createEarth()
    mu_m  = earth.mu

    # SPICE & ephemerides for plotting
    spice = spiceInterface.SpiceInterface()
    spice.SPICEDataPath = SPICE_PATH
    spice.UTCCalInit = datetime.strptime(
        EPOCH_STR, "%Y %b %d %H:%M:%S"
    ).strftime("%Y %B %d, %H:%M:%S.0 (UTC)")
    for k in ["naif0012.tls", "pck00010.tpc", "de-403-masses.tpc",
              "de430.bsp", "20099942.bsp"]:
        spice.loadSpiceKernel(k, spice.SPICEDataPath)
    spice.addPlanetNames(["Earth", "20099942"])
    spice.zeroBase = "Earth"
    sim.AddModelToTask(dynTask, spice)

    conv = ephemerisConverter.EphemerisConverter()
    for msg in spice.planetStateOutMsgs:
        conv.addSpiceInputMsg(msg)
    sim.AddModelToTask(dynTask, conv)

    # GEO spacecraft
    sc = spacecraft.Spacecraft()
    sc.ModelTag = "GEOsat"
    sc.gravField.gravBodies = spacecraft.GravBodyVector(
        list(gravF.gravBodies.values()))
    oe = orbitalMotion.ClassicElements()
    oe.a, oe.e, oe.i, oe.Omega, oe.omega, oe.f = (
        aG_km, eG, iG, raanG, argpG, ta0G)
    r0_km, v0_km = orbitalMotion.elem2rv(mu_m*1e-9, oe)
    sc.hub.r_CN_NInit = r0_km*1e3
    sc.hub.v_CN_NInit = v0_km*1e3
    sim.AddModelToTask(dynTask, sc)

    # SimpleNav
    nav = simpleNav.SimpleNav()
    nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    sim.AddModelToTask(dynTask, nav)

    # Recorders
    geoRec = sc.scStateOutMsg.recorder()
    navRec = nav.transOutMsg.recorder()
    apoRec = conv.ephemOutMsgs[1].recorder()
    sim.AddModelToTask(dynTask, geoRec)
    sim.AddModelToTask(dynTask, navRec)
    sim.AddModelToTask(dynTask, apoRec)

    # Lambert FSW chain
    planner = lambertPlanner.LambertPlanner()
    planner.navTransInMsg.subscribeTo(nav.transOutMsg)
    planner.setR_TN_N(rT_m)
    planner.setManeuverTime(MANEUVER_TIME)
    planner.setFinalTime(ARRIVAL_TIME)
    planner.setMu(mu_m)
    planner.setNumRevolutions(0)
    sim.AddModelToTask(fswTask, planner)

    solver = lambertSolver.LambertSolver()
    solver.lambertProblemInMsg.subscribeTo(planner.lambertProblemOutMsg)
    sim.AddModelToTask(fswTask, solver)

    validator = lambertValidator.LambertValidator()
    validator.navTransInMsg.subscribeTo(nav.transOutMsg)
    validator.lambertProblemInMsg.subscribeTo(planner.lambertProblemOutMsg)
    validator.lambertSolutionInMsg.subscribeTo(solver.lambertSolutionOutMsg)
    validator.lambertPerformanceInMsg.subscribeTo(solver.lambertPerformanceOutMsg)
    validator.setManeuverTime(MANEUVER_TIME)
    validator.setFinalTime(ARRIVAL_TIME)
    validator.setMaxDistanceTarget(1e5)
    validator.setMinOrbitRadius(6_378_000.0)
    validator.setUncertaintyStates(np.diag([1.,1.,1.,0.01,0.01,0.01]))
    validator.setUncertaintyDV(0.1)
    validator.setDvConvergenceTolerance(0.5)
    sim.AddModelToTask(fswTask, validator)

    dvRec = validator.dvBurnCmdOutMsg.recorder()
    sim.AddModelToTask(fswTask, dvRec)

    # 3) Run to burn
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(MANEUVER_TIME))
    sim.ExecuteSimulation()

    # Apply ΔV as 1 s extForceTorque
    dv_m_s = dvRec.dvInrtlCmd[-1]
    burn = extForceTorque.ExtForceTorque()
    burn.extForce_N = (dv_m_s * sc.hub.mHub / POST_BURN_SEC).tolist()
    sc.addDynamicEffector(burn)
    sim.AddModelToTask(dynTask, burn)

    # Disable FSW post‑burn to suppress warnings
    fswProc.disableAllTasks()
    sim.logThisMessage = False

    sim.ConfigureStopTime(macros.sec2nano(MANEUVER_TIME + POST_BURN_SEC))
    sim.ExecuteSimulation()

    # Coast to arrival
    sim.ConfigureStopTime(macros.sec2nano(ARRIVAL_TIME))
    sim.ExecuteSimulation()

    # Report ΔV
    print("\n*** GEO → Apophis fly‑by ***")
    print("Departure epoch :", EPOCH_STR)
    print(f"ΔV [km/s]       : {np.linalg.norm(dv_m_s)/1e3:.6f}")

    # Extract for plotting
    times  = geoRec.times() * macros.NANO2SEC
    geo_km = geoRec.r_BN_N / 1e3
    nav_km = navRec.r_BN_N / 1e3
    apo_km = apoRec.r_BdyZero_N / 1e3

    # Fix index out‐of‐bounds by clamping
    bIdx = np.searchsorted(times, MANEUVER_TIME)
    aIdx = np.searchsorted(times, ARRIVAL_TIME, side='right') - 1
    bIdx = min(bIdx, len(times)-1)
    aIdx = min(aIdx, len(times)-1)

    if show_plots:
        # XY projection
        plt.figure(figsize=(6,6))
        plt.plot(apo_km[:,0], apo_km[:,1], label="Apophis")
        plt.plot(geo_km[:,0], geo_km[:,1], label="GEO")
        plt.plot(geo_km[bIdx:aIdx,0], geo_km[bIdx:aIdx,1],
                 lw=2, label="Transfer arc")
        plt.scatter(geo_km[bIdx,0], geo_km[bIdx,1], c='g', s=60,
                    label="Burn")
        plt.scatter(apo_km[aIdx,0], apo_km[aIdx,1], c='r', marker='^',
                    s=60, label="Fly‑by")
        plt.gca().set_aspect('equal','box')
        plt.xlabel("X [km]"); plt.ylabel("Y [km]")
        plt.title("Earth‑centred XY view")
        plt.legend(); plt.grid(True)

        # 3‑D view
        fig = plt.figure(figsize=(8,6))
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot(apo_km[:,0], apo_km[:,1], apo_km[:,2], label="Apophis")
        ax.plot(geo_km[:,0], geo_km[:,1], geo_km[:,2], label="GEO")
        ax.plot(geo_km[bIdx:aIdx,0], geo_km[bIdx:aIdx,1],
                geo_km[bIdx:aIdx,2], lw=2, label="Transfer arc")
        ax.scatter(geo_km[bIdx,0], geo_km[bIdx,1], geo_km[bIdx,2], c='g', s=60)
        ax.scatter(apo_km[aIdx,0], apo_km[aIdx,1], apo_km[aIdx,2], c='r', s=60)
        ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
        ax.set_title("3‑D Earth‑centred view"); ax.legend()

        # ΔV components
        plt.figure()
        plt.bar(["ΔVx","ΔVy","ΔVz"], dv_m_s/1e3)
        plt.ylabel("km/s"); plt.title("Lambert ΔV components")

        # Truth vs. SimpleNav (X)
        plt.figure(figsize=(8,4))
        plt.plot(times/3600, geo_km[:,0], label="Truth X")
        plt.plot(times/3600, nav_km[:,0], '.', ms=2, label="Nav X")
        plt.xlabel("Time [h]"); plt.ylabel("X [km]")
        plt.title("Spacecraft X‑position (Truth vs. SimpleNav)")
        plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run(SHOW_PLOTS)
