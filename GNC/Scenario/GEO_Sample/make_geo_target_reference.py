#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_geo_chief_and_target_basilisk.py
=====================================

Use Basilisk to generate TWO GEO trajectories:

  • chief_reference.npz
  • target_geo_reference.npz

Both files contain:
    times      : (N,)   [s]
    positions  : (N,3)  [km]
    velocities : (N,3)  [km/s]

Chief and target both orbit Earth in GEO:
  - Same circular equatorial orbit (a ≈ 42164 km)
  - Target has a configurable along-track phase offset.

The NPZs are saved in the SAME FOLDER as this .py file.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from Basilisk import __path__ as bskPath
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simIncludeGravBody,
    unitTestSupport,
)
from Basilisk.simulation import spacecraft


# ───────────── USER PARAMETERS ─────────────

# GEO orbit radius [km]
A_GEO_KM = 42164.0

# Simulation length and step
T_FINAL_SEC = 6.0 * 3600.0        # 6 hours of motion
DT_SEC      = 60.0                # 60 s dynamics step

# Target phase offset relative to chief [deg]
TARGET_PHASE_OFFSET_DEG = 30.0


def build_geo_spacecraft(model_tag: str, a_km: float, phase_deg: float, mu: float) -> spacecraft.Spacecraft:
    """
    Create a Basilisk spacecraft on a circular equatorial GEO orbit with a given phase.
    a_km      : semi-major axis [km]
    phase_deg : true anomaly [deg] at t=0
    mu        : central-body µ [m^3/s^2]
    """
    sc = spacecraft.Spacecraft()
    sc.ModelTag = model_tag

    # Simple inertia / mass
    I = [100.0, 0.0,   0.0,
         0.0,   100.0, 0.0,
         0.0,   0.0,   100.0]
    sc.hub.mHub = 500.0  # kg
    sc.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    sc.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

    # Classic GEO elements (circular, equatorial)
    oe = orbitalMotion.ClassicElements()
    oe.a     = a_km * 1e3         # [m]
    oe.e     = 0.0
    oe.i     = 0.0
    oe.Omega = 0.0
    oe.omega = 0.0
    oe.f     = np.deg2rad(phase_deg)

    rN, vN = orbitalMotion.elem2rv(mu, oe)

    sc.hub.r_CN_NInit   = rN
    sc.hub.v_CN_NInit   = vN
    sc.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    sc.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]

    return sc


def main():
    # Folder where THIS script lives
    script_dir = Path(__file__).resolve().parent

    # Output NPZ names in the same folder as the script
    chief_npz  = script_dir / "chief_reference.npz"
    target_npz = script_dir / "target_geo_reference.npz"

    # 1) Build Basilisk sim skeleton
    sim = SimulationBaseClass.SimBaseClass()

    dyn_process_name = "dynProcess"
    dyn_task_name    = "dynTask"
    dyn_process = sim.CreateNewProcess(dyn_process_name)
    dyn_process.addTask(
        sim.CreateNewTask(dyn_task_name, macros.sec2nano(DT_SEC))
    )

    # 2) Gravity: Earth as central body
    grav_factory = simIncludeGravBody.gravBodyFactory()
    earth = grav_factory.createEarth()
    earth.isCentralBody = True
    mu_earth = earth.mu  # [m^3/s^2]

    # 3) Create chief and target spacecraft in GEO
    chief = build_geo_spacecraft("chief",  A_GEO_KM, 0.0,                  mu_earth)
    target = build_geo_spacecraft("target", A_GEO_KM, TARGET_PHASE_OFFSET_DEG, mu_earth)

    # Attach gravity
    grav_factory.addBodiesTo(chief)
    grav_factory.addBodiesTo(target)

    # 4) Add spacecraft to task
    sim.AddModelToTask(dyn_task_name, chief)
    sim.AddModelToTask(dyn_task_name, target)

    # 5) Recorders for both chief and target
    chief_rec  = chief.scStateOutMsg.recorder()
    target_rec = target.scStateOutMsg.recorder()
    sim.AddModelToTask(dyn_task_name, chief_rec)
    sim.AddModelToTask(dyn_task_name, target_rec)

    # 6) Initialize and run simulation
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(T_FINAL_SEC))
    sim.ExecuteSimulation()

    # 7) Extract logs and convert to NPZ format
    t_ns = np.array(chief_rec.times())               # [ns]
    times_s = t_ns * macros.NANO2SEC                # [s]

    r_chief_m   = np.array(chief_rec.r_BN_N)        # [m]
    v_chief_m_s = np.array(chief_rec.v_BN_N)        # [m/s]

    r_targ_m    = np.array(target_rec.r_BN_N)       # [m]
    v_targ_m_s  = np.array(target_rec.v_BN_N)       # [m/s]

    # Convert to km / km/s
    r_chief_km   = r_chief_m   / 1e3
    v_chief_km_s = v_chief_m_s / 1e3

    r_targ_km    = r_targ_m    / 1e3
    v_targ_km_s  = v_targ_m_s  / 1e3

    # 8) Save NPZ bundles in the script folder
    np.savez_compressed(
        chief_npz,
        times     = times_s,
        positions = r_chief_km,
        velocities= v_chief_km_s,
    )
    print(f"[GEO] chief_reference written to {chief_npz}")

    np.savez_compressed(
        target_npz,
        times     = times_s,
        positions = r_targ_km,
        velocities= v_targ_km_s,
    )
    print(f"[GEO] target_geo_reference written to {target_npz}")
    print(f"[GEO] duration = {T_FINAL_SEC/3600.0:.2f} h, step = {DT_SEC:.0f} s")


if __name__ == "__main__":
    main()
