#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_scenario_single_impulse_hcw_plot.py
===========================================

Purpose
-------
Minimal Basilisk scenario that demonstrates a single-impulse
Hill–Clohessy–Wiltshire (HCW) maneuver:

- One deputy is released at a specified deploy time,
- Receives a single HCW-designed LVLH ΔV 1 second later,
- Then coasts for a fixed duration to reach a commanded RTN offset
  relative to a chief,
- Finally, the script plots the inertial XY trajectories of chief and deputy.

This is written against the current Basilisk API (no deprecated addEvent /
setActive calls) and uses a SysModel-based impulse scheduler.

Guidance helper
---------------
The function

    hcw_single_impulse_to_target(rf_lvlh_m, n, dt)

implements the standard CW/HCW closed-form solution for the special case

    ρ(0) = 0,  ρ̇(0) = v0  →  ρ(dt) = rf_lvlh_m,

under linear HCW dynamics with chief mean motion n over a time-of-flight dt.
It returns the LVLH ΔV vector v0 (in m/s) that should be applied at t = 0.

You can pull that helper into a library module, e.g.

    gnc/guidance/hcw_guidance.py

and reuse it across scenarios.

Usage
-----
- Tune the user parameters at the top (deploy, burn, rendezvous times and
  desired RTN offset).
- Optionally replace the toy heliocentric orbit with a more realistic chief
  ephemeris.
- Run:

    python basilik_scenario_single_impulse_hcw_plot.py

The script will:
- Print the LVLH impulse magnitude,
- Run the Basilisk simulation to the rendezvous epoch,
- Plot the chief and deputy inertial XY trajectories showing the effect of
  the single HCW ΔV.
"""

from __future__ import annotations

import numpy as np
import numpy.linalg as la

from Basilisk.utilities   import (
    macros,
    SimulationBaseClass,
    orbitalMotion,
    unitTestSupport,
)
from Basilisk.simulation  import spacecraft, extForceTorque
from Basilisk.architecture import sysModel

# ───────────────────── USER PARAMETERS ─────────────────────

# Times [s] since simulation epoch
DEPLOY_EPOCH_S      = 10_023.0
BURN_EPOCH_S        = DEPLOY_EPOCH_S + 1.0
RENDEZVOUS_EPOCH_S  = 31_170.0        # ≈ 5.80 h after start

# Desired final relative offset in chief LVLH [km] (R,T,N)
R_F_LVLH_KM         = np.array([2.65, 3.77, -1.49])

# Deputy mass [kg]
M_DEP               = 12.0

# Dynamics step (1 Hz)
DYN_RATE_NS         = macros.sec2nano(1.0)

# Chief orbit model (toy heliocentric circular orbit)
MU_CENTRAL = orbitalMotion.MU_SUN       # m³/s² (change to MU_EARTH for LEO/GEO)
A_CHIEF    = 4.5e11                      # m  (semi-major axis, here ~3 AU)

# ───────────────────── HCW GUIDANCE HELPER ─────────────────────

def hcw_single_impulse_to_target(
    rf_lvlh_m: np.ndarray,
    n: float,
    dt: float,
) -> np.ndarray:
    """
    Closed-form HCW/Lin. CW guidance: initial LVLH velocity that drives

        ρ(0) = 0      →     ρ(dt) = rf_lvlh_m

    for a deputy relative to a chief with mean motion `n` and time-of-flight `dt`.

    Parameters
    ----------
    rf_lvlh_m : (3,) ndarray
        Desired final RTN/LVLH position at time dt [m].
    n : float
        Mean motion of the chief [rad/s].
    dt : float
        Time-of-flight [s].

    Returns
    -------
    v0_lvlh_mps : (3,) ndarray
        Required initial LVLH velocity [m/s] at t=0.
    """
    rf_lvlh_m = np.asarray(rf_lvlh_m, dtype=float).reshape(3)
    sn = np.sin(n * dt)
    cs = np.cos(n * dt)

    # Guard against singular cases (sn ≈ 0)
    if abs(sn) < 1e-10:
        raise ValueError("HCW mapping ill-conditioned: sin(n*dt) ≈ 0.")

    # Standard CW mapping for R/T plane with ρ(0) = 0
    A = np.array([
        [ sn / n,           2 * (1 - cs) / n ],
        [ -2 * (1 - cs) / n, (4 * sn - 3 * n * dt) / n ],
    ])
    vx0, vy0 = la.solve(A, rf_lvlh_m[:2])
    vz0 = n * rf_lvlh_m[2] / sn

    return np.array([vx0, vy0, vz0], dtype=float)


# Backwards-compatible alias (matches original name)
def hcw_initial_vel(rf_L_m: np.ndarray, n: float, dt: float) -> np.ndarray:
    """Alias for hcw_single_impulse_to_target for legacy calls."""
    return hcw_single_impulse_to_target(rf_L_m, n, dt)


# ───────────────────── BUILD BASILISK SIMULATION ─────────────────────

sim  = SimulationBaseClass.SimBaseClass()
proc = sim.CreateNewProcess("proc")
sim.CreateNewTask("dyn", DYN_RATE_NS)          # dynamics task “dyn”

col = lambda v: [[v[0]], [v[1]], [v[2]]]       # 3×1 column helper

# Chief + deputy on same simple circular orbit
n_chief = np.sqrt(MU_CENTRAL / A_CHIEF**3)     # rad/s (mean motion)
r0_N    = np.array([A_CHIEF, 0.0, 0.0])        # inertial position [m]
v0_N    = np.array([0.0, A_CHIEF * n_chief, 0.0])  # inertial velocity [m/s]

def make_spacecraft(tag: str, mass: float) -> spacecraft.Spacecraft:
    sc = spacecraft.Spacecraft()
    sc.ModelTag = tag
    sc.hub.mHub = mass
    sc.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(
        [1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0]
    )
    sc.hub.r_CN_NInit = col(r0_N)
    sc.hub.v_CN_NInit = col(v0_N)
    sc.hub.sigma_BNInit   = [[0.0], [0.0], [0.0]]
    sc.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]
    sim.AddModelToTask("dyn", sc)
    return sc

chief = make_spacecraft("chief", 200.0)
dep   = make_spacecraft("deputy", M_DEP)

# ───────────────────── COMPUTE ΔV (HCW) ─────────────────────

dt_coast = RENDEZVOUS_EPOCH_S - BURN_EPOCH_S
v_imp_L  = hcw_single_impulse_to_target(R_F_LVLH_KM * 1e3, n_chief, dt_coast)  # [m/s] in LVLH

# LVLH → inertial (simple RTN frame at t=0)
r_hat = r0_N / la.norm(r0_N)
h_hat = np.cross(r0_N, v0_N); h_hat /= la.norm(h_hat)
t_hat = np.cross(h_hat, r_hat)
C_NL  = np.column_stack((r_hat, t_hat, h_hat))      # columns are LVLH axes in inertial
dv_N  = C_NL @ v_imp_L                              # inertial ΔV [m/s]

print(f"|ΔV| (LVLH) = {la.norm(v_imp_L):.3f} m/s", flush=True)

# ───────────────────── IMPULSE EFFECTOR (1-s FORCE) ─────────────────────

F_imp_N = dv_N * M_DEP                             # N (for Δt = 1 s)
FT = extForceTorque.ExtForceTorque(); FT.ModelTag = "oneImpulse"
FT.extForce_N = [0.0, 0.0, 0.0]
dep.addDynamicEffector(FT)
sim.AddModelToTask("dyn", FT)

class ImpulseScheduler(sysModel.SysModel):
    """Turn the force on for exactly one step at BURN_EPOCH_S."""
    def __init__(self, eff: extForceTorque.ExtForceTorque,
                 burn_sec: float,
                 force_vec: np.ndarray):
        super().__init__()
        self.eff      = eff
        self.burn_ns  = macros.sec2nano(burn_sec)
        self.force    = list(force_vec)
        self.fired    = False

    def UpdateState(self, now_ns: int):
        if not self.fired and now_ns >= self.burn_ns:
            self.eff.extForce_N = self.force
            self.fired = True
        else:
            self.eff.extForce_N = [0.0, 0.0, 0.0]

sim.AddModelToTask("dyn", ImpulseScheduler(FT, BURN_EPOCH_S, F_imp_N))

# ───────────────────── HEARTBEAT PRINTER ─────────────────────

class ProgressPrinter(sysModel.SysModel):
    """Print simulation time every step_hrs hours."""
    def __init__(self, step_hrs: float = 0.25):
        super().__init__()
        self.step_ns = macros.sec2nano(step_hrs * 3600.0)
        self.next_ns = self.step_ns

    def UpdateState(self, now_ns: int):
        if now_ns >= self.next_ns:
            t_hr = now_ns * macros.NANO2SEC / 3600.0
            print(f"  … simulation t = {t_hr:.2f} h", flush=True)
            self.next_ns += self.step_ns

sim.AddModelToTask("dyn", ProgressPrinter())

# ───────────────────── RECORDERS ─────────────────────

chief_rec = chief.scStateOutMsg.recorder()
dep_rec   = dep.scStateOutMsg.recorder()
sim.AddModelToTask("dyn", chief_rec)
sim.AddModelToTask("dyn", dep_rec)

# ───────────────────── RUN SIMULATION ─────────────────────

sim.InitializeSimulation()
print("Simulation started — please wait …", flush=True)
sim.ConfigureStopTime(macros.sec2nano(RENDEZVOUS_EPOCH_S))
sim.ExecuteSimulation()
print("Simulation complete.", flush=True)

# ───────────────────── PLOT RESULTS ─────────────────────

import matplotlib.pyplot as plt  # import here to avoid backend overhead

rC_km = np.vstack([chief_rec.r_BN_N[:, j] for j in range(3)]).T / 1e3
rD_km = np.vstack([ dep_rec.r_BN_N[:, j] for j in range(3)]).T / 1e3

plt.figure(figsize=(6, 6))
plt.plot(rC_km[:, 0], rC_km[:, 1], label="Chief")
plt.plot(rD_km[:, 0], rD_km[:, 1], label="Deputy")
plt.scatter(*rC_km[0, :2], c="k", marker="o")
plt.scatter(*rD_km[0, :2], c="k", marker="x")
plt.xlabel("$x_N$  [km]")
plt.ylabel("$y_N$  [km]")
plt.gca().set_aspect("equal", "box")
plt.title("ECI XY Trajectories")
plt.legend()
plt.tight_layout()
plt.show()

