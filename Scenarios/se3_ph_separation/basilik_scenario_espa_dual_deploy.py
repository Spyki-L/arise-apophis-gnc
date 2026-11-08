"""
basilik_scenario_espa_dual_deploy.py
====================================

Purpose
-------
Template Basilisk scenario that models two small satellites (“sat1” and “sat2”)
mounted on an ESPA ring, then separates them via instantaneous ΔV impulses.
The script records the motion and generates a set of plots showing:

- Each satellite’s scalar separation from the ESPA carrier vs time.
- Component-wise relative position (Δx, Δy, Δz) vs time.
- In-plane (XY) relative trajectories.
- Full 3D relative trajectories.

This is intended as a starting point for more detailed ESPA separation studies
and for sanity-checking relative motion.

Scenario setup
--------------
- ESPA carrier:
    * Mass: `espaMass = 500 kg`.
    * Inertia: diagonal approximation from `I_flat`.
    * Orbit: 700 km circular LEO at 45° inclination:
         a = REQ_EARTH + 700 km,
         e = 0,
         i = 45 deg,
         Ω = 0,
         ω = 0,
         f = 0.
      Initial position and velocity are computed via:
         r0, v0 = orbitalMotion.elem2rv(earth.mu, oe).

- Satellites:
    * sat1 and sat2 masses: 50 kg each.
    * Initially co-located with the ESPA carrier (r_BcB_B = 0).
    * Each given a simple diagonal inertia proportional to mass.

Simulation architecture
-----------------------
- A single process `simProcess` and a single task `simTask`, with 1 s step:
      dt = macros.sec2nano(1.0)
      process.addTask(sim.CreateNewTask(taskName, dt))

- Gravity:
    * Earth created via `gravBodyFactory.createBodies(['earth'])`.
    * Earth set as the central body:
          earth.isCentralBody = True.

- Models added to the task:
    * ESPA spacecraft.
    * sat1 and sat2 spacecraft.
    * `SimpleNav` for ESPA (currently used only for convenience / extensibility).
    * Three state recorders:
          rec_espa, rec_sat1, rec_sat2.

Deployment sequence
-------------------
The script executes three segments:

1. Propagate from t = 0 to `deploy1TimeSec` (default 600 s).
2. Apply an instantaneous ΔV to sat1.
3. Propagate to `deploy2TimeSec` (default 1200 s).
4. Apply an instantaneous ΔV to sat2.
5. Propagate to t = 1800 s.

Impulse model:
- Sat1:
    * Δv₁ = [ +0.1, 0.0, 0.0 ] m/s applied at `deploy1TimeSec`.
- Sat2:
    * Δv₂ = [ −0.1, 0.0, 0.0 ] m/s applied at `deploy2TimeSec`.

Implementation detail:
- Velocities are modified directly via the dynamic state manager:

      vref = satX.dynManager.getStateObject(satX.hub.nameOfHubVelocity)
      vref.setState(vref.getState() + dv)

Outputs and plots
-----------------
After the final propagation to 1800 s, the script:

1. Pulls the recorded inertial positions:

    times = rec_espa.times() * macros.NANO2SEC
    posE  = rec_espa.r_BN_N
    pos1  = rec_sat1.r_BN_N
    pos2  = rec_sat2.r_BN_N

2. Computes relative vectors:

    rel1 = pos1 − posE
    rel2 = pos2 − posE

3. Plots:

   - Scalar separation vs time:
        * ‖pos1 − posE‖ and ‖pos2 − posE‖ vs time [s].

   - Component-wise relative separation vs time:
        * Δx, Δy, Δz for sat1 and sat2 in separate figures.

   - In-plane XY relative trajectories:
        * rel1[:,0] vs rel1[:,1] and rel2[:,0] vs rel2[:,1], with equal aspect.

   - Full 3D relative trajectories:
        * rel1 and rel2 plotted in 3D (X, Y, Z) with labels and legend.

Usage
-----
Call `run()` directly or use the module as a script:

    python basilik_scenario_espa_dual_deploy.py

Optional parameters:

- `show_plots` (bool)       : toggle interactive plotting (default True).
- `deploy1TimeSec` (float)  : first deploy time [s] (default 600).
- `deploy2TimeSec` (float)  : second deploy time [s] (default 1200).

The script will:

- Build the ESPA + two-satellite system in a 700 km 45° LEO.
- Execute the sequential deployments via instantaneous ΔV.
- Record all relevant states.
- Generate the separation and relative-motion plots if `show_plots` is True.
"""


import numpy as np
import matplotlib.pyplot as plt
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    simIncludeGravBody,
    orbitalMotion,
    unitTestSupport
)
from Basilisk.simulation import spacecraft
from Basilisk.simulation import simpleNav


def run(show_plots=True, deploy1TimeSec=600.0, deploy2TimeSec=1200.0):
    # User parameters
    espaMass = 500.0    # kg
    sat1Mass = 50.0     # kg
    sat2Mass = 50.0     # kg
    # Inertia for all bodies (diagonal approximation)
    I_flat = [100.0, 0.0, 0.0,
              0.0, 150.0, 0.0,
              0.0, 0.0, 200.0]

    # Initial orbit: circular 700 km LEO at 45° inclination
    oe = orbitalMotion.ClassicElements()
    oe.a = orbitalMotion.REQ_EARTH + 700e3
    oe.e = 0.0
    oe.i = 45.0 * macros.D2R
    oe.Omega = 0.0
    oe.omega = 0.0
    oe.f = 0.0

    # --- Simulation setup ---
    sim = SimulationBaseClass.SimBaseClass()
    process = sim.CreateNewProcess('simProcess')
    taskName = 'simTask'
    dt = macros.sec2nano(1.0)
    process.addTask(sim.CreateNewTask(taskName, dt))

    # Create Earth as central body
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravFactory.createBodies(['earth'])
    earth = gravFactory.gravBodies['earth']
    earth.isCentralBody = True

    # Create the ESPA ring (carrier)
    espa = spacecraft.Spacecraft()
    espa.ModelTag = 'ESPA'
    espa.hub.mHub = espaMass
    espa.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I_flat)
    # Initialize on the reference orbit
    r0, v0 = orbitalMotion.elem2rv(earth.mu, oe)
    espa.hub.r_CN_NInit = r0
    espa.hub.v_CN_NInit = v0
    sim.AddModelToTask(taskName, espa)

    # Attach a simple navigation module to ESPA
    nav_espa = simpleNav.SimpleNav()
    nav_espa.scStateInMsg.subscribeTo(espa.scStateOutMsg)
    sim.AddModelToTask(taskName, nav_espa)

    # Helper to create co-located satellite
    def create_sat(name, mass):
        sat = spacecraft.Spacecraft()
        sat.ModelTag = name
        sat.hub.mHub = mass
        sat.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
        I_flat_local = [mass, 0.0, 0.0,
                        0.0, mass, 0.0,
                        0.0, 0.0, mass]
        sat.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I_flat_local)
        return sat

    sat1 = create_sat('sat1', sat1Mass)
    sat2 = create_sat('sat2', sat2Mass)
    sim.AddModelToTask(taskName, sat1)
    sim.AddModelToTask(taskName, sat2)

    # Record states for separation calculation
    rec_espa = espa.scStateOutMsg.recorder()
    rec_sat1 = sat1.scStateOutMsg.recorder()
    rec_sat2 = sat2.scStateOutMsg.recorder()
    sim.AddModelToTask(taskName, rec_espa)
    sim.AddModelToTask(taskName, rec_sat1)
    sim.AddModelToTask(taskName, rec_sat2)

    # Initialize once
    sim.InitializeSimulation()

    # --- Simulation sequence ---
    # 1) propagate to first deployment
    sim.ConfigureStopTime(macros.sec2nano(deploy1TimeSec))
    sim.ExecuteSimulation()
    # instantaneous DV for sat1
    dv1 = np.array([0.1, 0.0, 0.0])
    vref1 = sat1.dynManager.getStateObject(sat1.hub.nameOfHubVelocity)
    vref1.setState(vref1.getState() + dv1)

    # 2) propagate to second deployment
    sim.ConfigureStopTime(macros.sec2nano(deploy2TimeSec))
    sim.ExecuteSimulation()
    # instantaneous DV for sat2
    dv2 = np.array([-0.1, 0.0, 0.0])
    vref2 = sat2.dynManager.getStateObject(sat2.hub.nameOfHubVelocity)
    vref2.setState(vref2.getState() + dv2)

    # 3) final propagate to 1800 s
    sim.ConfigureStopTime(macros.sec2nano(1800.0))
    sim.ExecuteSimulation()

    # --- Post-process separation ---
    times = rec_espa.times() * macros.NANO2SEC
    posE = np.array(rec_espa.r_BN_N)
    pos1 = np.array(rec_sat1.r_BN_N)
    pos2 = np.array(rec_sat2.r_BN_N)

    sep1 = np.linalg.norm(pos1 - posE, axis=1)
    sep2 = np.linalg.norm(pos2 - posE, axis=1)

    # --- Plot results ---
    plt.figure()
    plt.plot(times, sep1, label='sat1 separation')
    plt.plot(times, sep2, label='sat2 separation')
    plt.xlabel('Time [s]')
    plt.ylabel('Separation [m]')
    plt.legend()

            # --- Additional verification plots ---

    # Pull recorded data arrays
    times = rec_espa.times() * macros.NANO2SEC
    posE  = np.array(rec_espa.r_BN_N)
    pos1  = np.array(rec_sat1.r_BN_N)
    pos2  = np.array(rec_sat2.r_BN_N)

    # Compute relative position vectors
    rel1 = pos1 - posE
    rel2 = pos2 - posE

    # 1) Component‐wise separation vs time
    # Δx
    plt.figure()
    plt.plot(times, rel1[:,0], label='sat1 Δx')
    plt.plot(times, rel2[:,0], label='sat2 Δx')
    plt.xlabel('Time [s]')
    plt.ylabel('Δx [m]')
    plt.legend()
    # Δy
    plt.figure()
    plt.plot(times, rel1[:,1], label='sat1 Δy')
    plt.plot(times, rel2[:,1], label='sat2 Δy')
    plt.xlabel('Time [s]')
    plt.ylabel('Δy [m]')
    plt.legend()
    # Δz
    plt.figure()
    plt.plot(times, rel1[:,2], label='sat1 Δz')
    plt.plot(times, rel2[:,2], label='sat2 Δz')
    plt.xlabel('Time [s]')
    plt.ylabel('Δz [m]')
    plt.legend()

    # 2) In‐plane (XY) relative trajectory
    plt.figure()
    plt.plot(rel1[:,0], rel1[:,1], label='sat1 rel. XY')
    plt.plot(rel2[:,0], rel2[:,1], label='sat2 rel. XY')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.gca().set_aspect('equal', 'box')

    # 3) Full 3D relative trajectory
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(rel1[:,0], rel1[:,1], rel1[:,2], label='sat1 rel.')
    ax.plot(rel2[:,0], rel2[:,1], rel2[:,2], label='sat2 rel.')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()

    if show_plots:
        plt.show()



if __name__ == '__main__':
    run()
