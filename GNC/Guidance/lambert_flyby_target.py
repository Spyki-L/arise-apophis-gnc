#!/usr/bin/env python3
"""
basilik_scenario_lambert_flyby_target.py
========================================

Purpose
-------
Basilisk-only (no spiceypy) scenario that designs and executes a single-burn
Lambert transfer from a GEO orbit to a fly-by of a generic SPICE target body
(e.g., asteroid (99942) Apophis, a planet, etc.), using Basilisk’s SPICE
interface and Lambert FSW modules.

Key features:
- Uses Basilisk’s SPICE modules to get Earth + target ephemerides.
- Starts from a GEO orbit at a specified epoch.
- Solves a single-impulse Lambert problem to hit the target’s geocentric
  position at a prescribed time of flight (TOF).
- Applies the Lambert ΔV as a short finite burn.
- Propagates a one-satellite system under Earth gravity to the fly-by point.
- Outputs ΔV magnitude and generates a compact set of plots:
    * Earth-centred XY projection with transfer arc and markers,
    * 3-D Earth-centred view,
    * ΔV component bar chart,
    * Truth vs SimpleNav comparison for the X-position.

To change the target, edit TARGET_NAME and TARGET_SPK below.

Usage
-----
Ensure the SPICE kernel files listed are present in Basilisk’s
EphemerisData directory, including TARGET_SPK. Then run:

    python basilik_scenario_lambert_flyby_target.py
"""

from __future__ import annotations

from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    simIncludeGravBody,
    orbitalMotion,
)
from Basilisk.simulation import (
    spacecraft,
    simpleNav,
    spiceInterface,
    ephemerisConverter,
    extForceTorque,
)
from Basilisk.fswAlgorithms import (
    lambertPlanner,
    lambertSolver,
    lambertValidator,
)
from Basilisk import __path__ as bskPath

# ───────────────────────── USER PARAMETERS ─────────────────────────

SPICE_PATH = os.path.join(bskPath[0], "supportData", "EphemerisData") + os.sep

# Target body: SPICE name/ID and SPK file
# Change these two lines to reuse the scenario for other bodies.
TARGET_NAME = "20099942"      # e.g. "(99942) Apophis" NAIF ID string
TARGET_SPK  = "20099942.bsp"  # SPK for the target

# Mission parameters
EPOCH_STR     = "2029 APR 13 11:38:28"
TOF_SEC       = 30_871.9
MANEUVER_TIME = 10.0          # burn 10 s after epoch
ARRIVAL_TIME  = MANEUVER_TIME + TOF_SEC
POST_BURN_SEC = 1.0
SHOW_PLOTS    = True

# GEO orbit elements (km, rad)
aG_km   = 42_164.0
eG      = 0.0
iG      = 0.0
raanG   = np.deg2rad(53.3953)
argpG   = 0.0
ta0G    = 0.0

KERNEL_LIST = [
    "naif0012.tls",
    "pck00010.tpc",
    "de-403-masses.tpc",
    "de430.bsp",
    TARGET_SPK,
]


# ───────────────────────── helpers ─────────────────────────

def ensure_kernels():
    missing = [
        k for k in KERNEL_LIST
        if not os.path.isfile(os.path.join(SPICE_PATH, k))
    ]
    if missing:
        msg = "Missing SPICE kernels:\n  " + "\n  ".join(
            os.path.join(SPICE_PATH, k) for k in missing
        )
        raise FileNotFoundError(msg)


def get_target_pos_m(arrival_s: float) -> np.ndarray:
    """
    Phase-0 sim: SPICE → target ECI position at t = arrival_s (metres),
    Earth-centred.
    """
    sim  = SimulationBaseClass.SimBaseClass()
    task = sim.CreateNewTask("EphemTask", macros.sec2nano(10.0))
    proc = sim.CreateNewProcess("EphemProc")
    proc.addTask(task)

    spice = spiceInterface.SpiceInterface()
    spice.SPICEDataPath = SPICE_PATH
    spice.UTCCalInit = datetime.strptime(
        EPOCH_STR, "%Y %b %d %H:%M:%S"
    ).strftime("%Y %B %d, %H:%M:%S.0 (UTC)")

    for k in KERNEL_LIST:
        spice.loadSpiceKernel(k, spice.SPICEDataPath)

    # Earth (0) and target (1) in the ephemeris list
    spice.addPlanetNames(["Earth", TARGET_NAME])
    spice.zeroBase = "Earth"
    sim.AddModelToTask("EphemTask", spice)

    conv = ephemerisConverter.EphemerisConverter()
    for msg in spice.planetStateOutMsgs:
        conv.addSpiceInputMsg(msg)
    sim.AddModelToTask("EphemTask", conv)

    # Target is index 1
    tgtRec = conv.ephemOutMsgs[1].recorder()
    sim.AddModelToTask("EphemTask", tgtRec)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(arrival_s))
    sim.ExecuteSimulation()

    return tgtRec.r_BdyZero_N[-1]  # metres, Earth-centred


def run(show_plots: bool = True):
    ensure_kernels()

    # 1) Compute target position at arrival
    rT_m = get_target_pos_m(ARRIVAL_TIME)

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

    # SPICE & ephemerides for plotting (Earth + target)
    spice = spiceInterface.SpiceInterface()
    spice.SPICEDataPath = SPICE_PATH
    spice.UTCCalInit = datetime.strptime(
        EPOCH_STR, "%Y %b %d %H:%M:%S"
    ).strftime("%Y %B %d, %H:%M:%S.0 (UTC)")
    for k in KERNEL_LIST:
        spice.loadSpiceKernel(k, spice.SPICEDataPath)
    spice.addPlanetNames(["Earth", TARGET_NAME])
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
        list(gravF.gravBodies.values())
    )

    oe = orbitalMotion.ClassicElements()
    oe.a     = aG_km
    oe.e     = eG
    oe.i     = iG
    oe.Omega = raanG
    oe.omega = argpG
    oe.f     = ta0G

    r0_km, v0_km = orbitalMotion.elem2rv(mu_m * 1e-9, oe)
    sc.hub.r_CN_NInit = r0_km * 1e3
    sc.hub.v_CN_NInit = v0_km * 1e3
    sim.AddModelToTask(dynTask, sc)

    # SimpleNav
    nav = simpleNav.SimpleNav()
    nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    sim.AddModelToTask(dynTask, nav)

    # Recorders
    geoRec = sc.scStateOutMsg.recorder()
    navRec = nav.transOutMsg.recorder()
    tgtRec = conv.ephemOutMsgs[1].recorder()
    sim.AddModelToTask(dynTask, geoRec)
    sim.AddModelToTask(dynTask, navRec)
    sim.AddModelToTask(dynTask, tgtRec)

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

    # 3) Run to burn epoch
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(MANEUVER_TIME))
    sim.ExecuteSimulation()

    # Apply ΔV as 1 s extForceTorque
    dv_m_s = dvRec.dvInrtlCmd[-1]
    burn = extForceTorque.ExtForceTorque()
    burn.extForce_N = (dv_m_s * sc.hub.mHub / POST_BURN_SEC).tolist()
    sc.addDynamicEffector(burn)
    sim.AddModelToTask(dynTask, burn)

    # Disable FSW post-burn to suppress extra Lambert cycles
    fswProc.disableAllTasks()
    sim.logThisMessage = False

    sim.ConfigureStopTime(macros.sec2nano(MANEUVER_TIME + POST_BURN_SEC))
    sim.ExecuteSimulation()

    # Coast to arrival
    sim.ConfigureStopTime(macros.sec2nano(ARRIVAL_TIME))
    sim.ExecuteSimulation()

    # ΔV report
    print(f"\n*** GEO → {TARGET_NAME} fly-by ***")
    print("Departure epoch :", EPOCH_STR)
    print(f"ΔV [km/s]       : {np.linalg.norm(dv_m_s)/1e3:.6f}")

    # Extract histories for plotting
    times  = geoRec.times() * macros.NANO2SEC
    geo_km = geoRec.r_BN_N / 1e3
    nav_km = navRec.r_BN_N / 1e3
    tgt_km = tgtRec.r_BdyZero_N / 1e3

    # Clamp indices
    bIdx = np.searchsorted(times, MANEUVER_TIME)
    aIdx = np.searchsorted(times, ARRIVAL_TIME, side="right") - 1
    bIdx = min(max(bIdx, 0), len(times) - 1)
    aIdx = min(max(aIdx, 0), len(times) - 1)

    if show_plots:
        # XY projection
        plt.figure(figsize=(6, 6))
        plt.plot(tgt_km[:, 0], tgt_km[:, 1], label=TARGET_NAME)
        plt.plot(geo_km[:, 0], geo_km[:, 1], label="GEO")
        plt.plot(
            geo_km[bIdx:aIdx, 0],
            geo_km[bIdx:aIdx, 1],
            lw=2,
            label="Transfer arc",
        )
        plt.scatter(geo_km[bIdx, 0], geo_km[bIdx, 1], c="g", s=60, label="Burn")
        plt.scatter(
            tgt_km[aIdx, 0],
            tgt_km[aIdx, 1],
            c="r",
            marker="^",
            s=60,
            label="Fly-by",
        )
        plt.gca().set_aspect("equal", "box")
        plt.xlabel("X [km]")
        plt.ylabel("Y [km]")
        plt.title("Earth-centred XY view")
        plt.legend()
        plt.grid(True)

        # 3-D view
        fig = plt.figure(figsize=(8, 6))
        ax3 = fig.add_subplot(111, projection="3d")
        ax3.plot(tgt_km[:, 0], tgt_km[:, 1], tgt_km[:, 2], label=TARGET_NAME)
        ax3.plot(geo_km[:, 0], geo_km[:, 1], geo_km[:, 2], label="GEO")
        ax3.plot(
            geo_km[bIdx:aIdx, 0],
            geo_km[bIdx:aIdx, 1],
            geo_km[bIdx:aIdx, 2],
            lw=2,
            label="Transfer arc",
        )
        ax3.scatter(
            geo_km[bIdx, 0], geo_km[bIdx, 1], geo_km[bIdx, 2], c="g", s=60
        )
        ax3.scatter(
            tgt_km[aIdx, 0], tgt_km[aIdx, 1], tgt_km[aIdx, 2], c="r", s=60
        )
        ax3.set_xlabel("X [km]")
        ax3.set_ylabel("Y [km]")
        ax3.set_zlabel("Z [km]")
        ax3.set_title("3-D Earth-centred view")
        ax3.legend()

        # ΔV components
        plt.figure()
        plt.bar(["ΔVx", "ΔVy", "ΔVz"], dv_m_s / 1e3)
        plt.ylabel("km/s")
        plt.title("Lambert ΔV components")

        # Truth vs. SimpleNav (X)
        plt.figure(figsize=(8, 4))
        plt.plot(times/3600.0, geo_km[:, 0], label="Truth X")
        plt.plot(times/3600.0, nav_km[:, 0], ".", ms=2, label="Nav X")
        plt.xlabel("Time [h]")
        plt.ylabel("X [km]")
        plt.title("Spacecraft X-position (Truth vs. SimpleNav)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run(SHOW_PLOTS)

