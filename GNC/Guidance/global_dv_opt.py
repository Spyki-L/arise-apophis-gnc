#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_geo_to_target_global_dv_opt.py
======================================

Purpose
-------
Global ΔV minimisation for a single-impulse GEO → target-body transfer with
in-plane and out-of-plane freedom, using:

  • Decision variables:
      θ   ∈ [0°, 360°]   – true anomaly of the GEO spacecraft at departure,
      i   ∈ [0°,  6°]    – inclination of the GEO orbit,
      Ω   ∈ [0°, 360°]   – RAAN of the GEO orbit,
      TOF ∈ [2 h, 12 h]  – Lambert arc time of flight.

  • Objective:
      Minimise ||ΔV_dep|| + ||ΔV_arr||, where:
        – ΔV_dep is the departure burn to match the Lambert v₁,
        – ΔV_arr is a “plane-trim” at arrival to match the target’s
          instantaneous velocity.

This is the same architecture you had for Apophis, but parameterised by
a generic SPICE target (name/ID + SPK file).

Usage
-----
  • Make sure the SPICE SPK for your target is in Basilisk’s EphemerisData
    folder (see TARGET_SPK below).
  • Adjust:
      TARGET_NAME, TARGET_SPK
      REF_EPOCH_UTC (reference epoch for SPICE)
      bounds if you want different TOF ranges, etc.
  • Run:

      python basilik_geo_to_target_global_dv_opt.py

  • The script will:
      – Build a SPICE-based target + Earth ephemeris window,
      – Run a global search over [θ, i, Ω, TOF] via differential evolution,
      – Reconstruct and visualise the best Lambert transfer,
      – Print a mission-style summary of departure/arrival times and ΔV budget.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from Basilisk import __path__ as bskPath
from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.utilities.simIncludeGravBody import gravBodyFactory
from Basilisk.simulation import (
    spiceInterface,
    ephemerisConverter,
    spacecraft,
    simpleNav,
)
from Basilisk.fswAlgorithms import lambertPlanner, lambertSolver

# ───────────────────────── USER CONSTANTS ─────────────────────────

# Central body (Earth)
MU_EARTH = 3.986004418e5  # km^3/s^2
R_EARTH  = 6378.137       # km
GEO_R    = 42164.0        # km
T_GEO    = 2 * np.pi * np.sqrt(GEO_R**3 / MU_EARTH)

# Target body (SPICE name/ID string + SPK)
# Change these two lines to re-use for other bodies.
TARGET_NAME = "20099942"      # e.g. "(99942) Apophis" NAIF ID as string
TARGET_SPK  = "20099942.bsp"  # its SPK file

# Reference epoch (used as t=0 for SPICE sampling)
REF_EPOCH_UTC = "2029 APR 13 11:38:28"
t0_dt         = datetime.strptime(REF_EPOCH_UTC, "%Y %b %d %H:%M:%S")

# SPICE kernels (Earth, planets, + target)
SPICE_DIR = os.path.join(bskPath[0], "supportData", "EphemerisData") + os.sep
KERNELS   = [
    "naif0012.tls",
    "pck00010.tpc",
    "de-403-masses.tpc",
    "de430.bsp",
    TARGET_SPK,
]

# ───────────────────────── Helpers (geometry) ─────────────────────────

def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  c, -s],
                     [0.0,  s,  c]])

def geo_state(theta: float, inc: float, RAAN: float) -> tuple[np.ndarray, np.ndarray]:
    """
    GEO orbit state at true anomaly θ [rad], inclination inc [rad],
    and RAAN [rad]. Returns (r, v) in km, km/s.
    """
    r_eq = GEO_R * np.array([np.cos(theta), np.sin(theta), 0.0])
    v_eq = np.sqrt(MU_EARTH / GEO_R) * np.array([-np.sin(theta), np.cos(theta), 0.0])
    R = rot_z(RAAN) @ rot_x(inc)
    return R @ r_eq, R @ v_eq

def rk4_two_body(r0: np.ndarray, v0: np.ndarray, tof: float,
                 mu: float, steps: int = 600) -> np.ndarray:
    """Basic RK4 propagator under two-body gravity; returns positions (steps+1,3)."""
    h = tof / steps
    state = np.hstack((r0, v0)).astype(float)
    pos = np.empty((steps + 1, 3))

    def deriv(st):
        r, v = st[:3], st[3:]
        a = -mu * r / np.linalg.norm(r)**3
        return np.hstack((v, a))

    for k in range(steps + 1):
        pos[k] = state[:3]
        k1 = deriv(state)
        k2 = deriv(state + 0.5*h*k1)
        k3 = deriv(state + 0.5*h*k2)
        k4 = deriv(state +      h*k3)
        state += (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return pos


# ───────────────────────── Lambert solver wrapper ─────────────────────────

def lambert_solver(r1_km: np.ndarray,
                   r2_km: np.ndarray,
                   tof_s: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Use Basilisk LambertPlanner + LambertSolver to solve a single-rev Lambert
    arc from r1_km to r2_km in tof_s seconds. Returns (v1, v2) in km/s.
    """
    sim = SimulationBaseClass.SimBaseClass()
    dynT = sim.CreateNewTask("dynT", macros.sec2nano(0.5))
    fswT = sim.CreateNewTask("fswT", macros.sec2nano(0.5))
    sim.CreateNewProcess("dyn").addTask(dynT)
    sim.CreateNewProcess("fsw").addTask(fswT)

    mu_m = gravBodyFactory().createEarth().mu  # in m^3/s^2

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
    planner.setFinalTime(10.0 + float(tof_s))

    solver = lambertSolver.LambertSolver()
    solver.lambertProblemInMsg.subscribeTo(planner.lambertProblemOutMsg)
    rec = solver.lambertSolutionOutMsg.recorder()

    sim.AddModelToTask("fswT", planner)
    sim.AddModelToTask("fswT", solver)
    sim.AddModelToTask("fswT", rec)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(5.0))
    sim.ExecuteSimulation()

    v1 = rec.v1_N[-1] / 1e3  # km/s
    v2 = rec.v2_N[-1] / 1e3
    return v1, v2


# ───────────────────────── SPICE: Earth + target ephemerides ─────────────────────────

sim_sp = SimulationBaseClass.SimBaseClass()
tsk    = sim_sp.CreateNewTask("tsk", macros.sec2nano(60.0))
sim_sp.CreateNewProcess("dyn").addTask(tsk)

spi = spiceInterface.SpiceInterface()
spi.SPICEDataPath = SPICE_DIR
spi.UTCCalInit    = t0_dt.strftime("%Y %B %d, %H:%M:%S.0 (UTC)")

for k in KERNELS:
    path = os.path.join(SPICE_DIR, k)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing SPICE kernel: {path}")
    spi.loadSpiceKernel(k, SPICE_DIR)

# Earth, Sun, Target
spi.addPlanetNames(["Earth", "Sun", TARGET_NAME])
spi.zeroBase = "Sun"
sim_sp.AddModelToTask("tsk", spi)

conv = ephemerisConverter.EphemerisConverter()
for msg in spi.planetStateOutMsgs:
    conv.addSpiceInputMsg(msg)
sim_sp.AddModelToTask("tsk", conv)

earthRec = conv.ephemOutMsgs[0].recorder()
tgtRec   = conv.ephemOutMsgs[2].recorder()
sim_sp.AddModelToTask("tsk", earthRec)
sim_sp.AddModelToTask("tsk", tgtRec)

sim_sp.InitializeSimulation()
sim_sp.ConfigureStopTime(macros.sec2nano(43_200.0))  # 12h window
sim_sp.ExecuteSimulation()

def target_state(t_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Geocentric target state at t_sec seconds after REF_EPOCH_UTC.
    Returns (r, v) in km, km/s.
    """
    idx = int(round(t_sec / 60.0))
    rT = tgtRec.r_BdyZero_N[idx] / 1e3
    rE = earthRec.r_BdyZero_N[idx] / 1e3
    vT = tgtRec.v_BdyZero_N[idx] / 1e3
    vE = earthRec.v_BdyZero_N[idx] / 1e3
    return rT - rE, vT - vE


# ───────────────────────── Objective for global optimisation ─────────────────────────

def fitness(x: np.ndarray) -> float:
    """
    x = (θ, inc, RAAN, tof)

    θ   : true anomaly at departure [rad],
    inc : inclination [rad],
    RAAN: RAAN [rad],
    tof : Lambert time of flight [s].
    """
    theta, inc, RAAN, tof = x
    r1, v_geo = geo_state(theta, inc, RAAN)
    r2, v_tgt = target_state(tof)
    v1, v2    = lambert_solver(r1, r2, tof)

    dv_dep = np.linalg.norm(v1 - v_geo)
    dv_arr = np.linalg.norm(v2 - v_tgt)
    return float(dv_dep + dv_arr)


# ───────────────────────── Global search (differential evolution) ─────────────────────────

bounds = [
    (0.0,        2.0*np.pi),          # θ
    (0.0,        np.deg2rad(6.0)),    # inc (0–6 deg)
    (0.0,        2.0*np.pi),          # RAAN
    (7_200.0,    43_200.0),           # TOF: 2–12 hours
]

result = differential_evolution(
    fitness,
    bounds,
    strategy="best1bin",
    maxiter=800,
    popsize=120,
    tol=1e-7,
    polish=True,
    updating="deferred",
    workers=-1,
)

theta_opt, inc_opt, RAAN_opt, tof_opt = result.x
dv_opt = float(result.fun)

# ───────────────────────── Reconstruct optimal trajectory & epochs ─────────────────────────

r1_opt, v_geo_opt = geo_state(theta_opt, inc_opt, RAAN_opt)
r2_opt, v_tgt_opt = target_state(tof_opt)
v1_opt, v2_opt    = lambert_solver(r1_opt, r2_opt, tof_opt)

# Approximate departure epoch by sliding along GEO ring
dep_dt = t0_dt + timedelta(seconds=float(theta_opt / (2.0*np.pi) * T_GEO))
arr_dt = dep_dt + timedelta(seconds=float(tof_opt))

traj_opt = rk4_two_body(r1_opt, v1_opt, tof_opt, MU_EARTH, steps=600)

# ───────────────────────── Plot GEO ring, transfer, target track ─────────────────────────

theta_ring = np.linspace(0.0, 2.0*np.pi, 400)
geo_ring   = np.array([geo_state(th, inc_opt, RAAN_opt)[0] for th in theta_ring])
tgt_track  = np.array([target_state(t)[0] for t in np.linspace(0.0, 43_200.0, 721)])

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')

u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
ax.plot_surface(
    R_EARTH * np.cos(u) * np.sin(v),
    R_EARTH * np.sin(u) * np.sin(v),
    R_EARTH * np.cos(v),
    color='lightsteelblue', alpha=0.5, linewidth=0,
)

ax.plot(
    geo_ring[:, 0], geo_ring[:, 1], geo_ring[:, 2],
    color='0.6', linewidth=0.8, label='GEO ring',
)
ax.scatter(*r1_opt, c='r', s=60, label='ΔV depart')

ax.plot(
    traj_opt[:, 0], traj_opt[:, 1], traj_opt[:, 2],
    'c', linewidth=2, label='Lambert arc',
)
ax.plot(
    tgt_track[:, 0], tgt_track[:, 1], tgt_track[:, 2],
    'b', linewidth=2, label=f'{TARGET_NAME} track',
)
ax.scatter(*r2_opt, c='m', s=70, label='SC & target @ arrival')

rng = np.max(np.abs(np.vstack((geo_ring, traj_opt, tgt_track))))
ax.set_xlim(-rng, rng)
ax.set_ylim(-rng, rng)
ax.set_zlim(-rng, rng)

ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')
ax.set_title(f'Optimised transfer to {TARGET_NAME} (ΔV = {dv_opt*1e3:5.1f} m/s)')
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# ───────────────────────── Mission summary ─────────────────────────

print("\nOptimised global ΔV (in-plane + out-of-plane):")
print(f"  Target body    : {TARGET_NAME}")
print(f"  Departure UTC  : {dep_dt.strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"  Arrival   UTC  : {arr_dt.strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"  ΔV_depart (m/s): {np.linalg.norm(v1_opt - v_geo_opt)*1e3:6.1f}")
print(f"  ΔV_arrival(m/s): {np.linalg.norm(v2_opt - v_tgt_opt)*1e3:6.1f}")
print(f"  ΔV total  (m/s): {dv_opt*1e3:6.1f}")

