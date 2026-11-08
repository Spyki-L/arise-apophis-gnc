#!/usr/bin/env python3
"""
basilik_scenario_apophis_single_impulse_mroe.py
===============================================

Purpose
-------
Design an **optimised single-impulse rendezvous** with (99942) Apophis using a
Mean Relative Orbital Elements (M-ROE) framework, so the chief’s elliptic
orbit is handled correctly under J2, rather than with simple HCW.

Key ideas:
- Use M-ROE to size the initial LVLH ΔV so that at RV_EPOCH the deputy reaches
  a desired target in the chief-centred LVLH frame.
- Use Hill-frame dead-band PD with small thruster pulses to trim residual
  error while keeping ΔV small.
- Work robustly with **all Basilisk releases** by:
  * Having a version-agnostic `rv2classic()` wrapper for `orbitalMotion.rv2elem`,
  * Falling back to standard MU/REQ/J2 constants if MROE or astroConstants
    are not available.

Scenario timeline
-----------------
All times are in seconds since a reference epoch (consistent with
`chief_reference.npz` and SPICE):

- DEPLOY_EPOCH = 10 023.0 s   (t₀)
- BURN_EPOCH   = 10 024.0 s   (initial impulse)
- RV_EPOCH     = 31 170.0 s   (rendezvous time)

Integration cadences:

- DT_DYN = 1.0 s   (LVLH propagation “physics” step)
- DT_FSW = 5.0 s   (PD controller update cadence)

Target and spacecraft parameters
--------------------------------
- Chief-centred LVLH rendezvous target (meters):

    TARGET = [1.43, 2.55, −0.27] km → *1e3 → [m]*

- Deputy / thruster:
    * `SC_MASS = 30.0 kg`
    * `THR_MAX = 5e-4 N` (0.5 mN, micro-thrusters)

- Dead-band PD (Hill/LVLH):
    * Position dead-band [m]:
        DB_POS = [3, 3, 2]
    * Velocity dead-band [m/s]:
        DB_VEL = [0.01, 0.01, 0.008]
    * PD gains:
        Kp = [2e−4, 1e−4, 2e−4] N·m⁻¹
        Kd = [5e−3, 3e−3, 5e−3] N·s·m⁻¹

Chief ephemeris (ECI)
---------------------
The script uses a chief trajectory stored in:

    chief_reference.npz

which must contain:

- `times`      : time stamps [s],
- `positions`  : (N,3) ECI positions [km],
- `velocities` : (N,3) ECI velocities [km/s].

Helper `chief_state(t_s)` returns (r, v) for the chief in metres at an
arbitrary UTC-seconds epoch t_s.

Apophis ephemeris (geocentric)
------------------------------
Via SPICE (Basilisk `SpiceInterface`):

- Loaded kernels:
    naif0012.tls, pck00010.tpc, de-403-masses.tpc,
    de430.bsp, 20099942.bsp.
- `zeroBase = "Sun"`; Earth and Apophis are sampled for a fixed span, then
  geocentric states are formed:

      rA(t) = (Apophis − Earth)(t)
      vA(t) = (Apophis − Earth)(t)

A pair of interpolator lambdas `(rA_fn, vA_fn)` is returned by
`make_apophis_sampler(...)`.

Frame helpers
-------------
- `dcm_lvlh(r, v)`:
    * Builds LVLH DCM with rows [R̂, T̂, N̂] expressed in ECI.

- `rv2classic(mu, r, v)`:
    * A robust wrapper around `orbitalMotion.rv2elem` that:
        - Handles the “new” signature:
              rv2elem(mu, r, v, outElem)
        - Or the “old” one:
              rv2elem(mu, r, v) → ClassicElements or tuple,
          and always returns an `om.ClassicElements` object.

M-ROE based ΔV sizing
---------------------
The main M-ROE flow for sizing the initial impulse:

1. At BURN_EPOCH, get chief state in metres:
       rB_burn, vB_burn = chief_state(BURN_EPOCH).

2. Convert to osculating ClassicElements:
       ce_c_b = rv2classic(MU_EARTH, rB_burn, vB_burn).

3. For a trial LVLH impulse dv_rtn (RTN / LVLH axes at burn):
   - Build ECI rotation:
         C_burn = dcm_lvlh(rB_burn, vB_burn).
   - Convert LVLH ΔV to ECI:
         dv_eci = C_burn.T @ dv_rtn.
   - Deputy osculating COEs at burn:
         dep_r_b = rB_burn,
         dep_v_b = vB_burn + dv_eci,
         ce_d_b  = rv2classic(MU_EARTH, dep_r_b, dep_v_b).

   - Compute initial mean ROE:

         roe0 = mroe(ce_c_b, ce_d_b)

4. Map chief from osculating → mean elements (J2, first order) at burn:

       chief_b_mean = om.ClassicElements()
       om.clMeanOscMap(REQ_EARTH, J2_EARTH, ce_c_b, chief_b_mean, sgn=-1)

   and compute chief mean motion:

       n_c = sqrt(MU_EARTH / a_c³).

5. Propagate mean ROE to RV_EPOCH:

       roef = propagate_mroe(roe0, n_c, Δt = RV_EPOCH − BURN_EPOCH).

6. At RV_EPOCH, build chief mean elements again and derive LVLH mapping, then:

       rho_f, _ = mroe_to_lvlh(roef, chief_f_mean)

   where `rho_f` is the predicted LVLH relative state [ρ_R, ρ_T, ρ_N]
   in metres.

The residual for the ΔV solver stacks:

- Position error:  `rho_f − TARGET`,
- ΔV regularization: `γ * dv` (with γ = 10.0 to keep ΔV small and stable).

A nonlinear least-squares solve (`scipy.optimize.least_squares`) determines
dv0_rtn (LVLH ΔV at BURN_EPOCH) that drives the deputy towards the target.

Hill-frame trimming (dead-band PD)
----------------------------------
After the initial optimised impulse:

- The script propagates a simple HCW-based LVLH state:

      state = [ρ, ρ̇] ∈ ℝ⁶

for k = 0..steps (steps = (RV_EPOCH−DEPLOY_EPOCH)/DT_DYN):

1. At k == burn_k (BURN_EPOCH), add the impulse:

      state[3:] += dv0_rtn

2. Every DT_FSW steps:
   - Compute errors:
         err_r = ρ − TARGET,
         err_v = ρ̇.
   - Form PD control:

         u = −Kp * err_r − Kd * err_v

   - Apply dead-bands and saturation:
         Favg = clip(u, −THR_MAX, THR_MAX).

   - Update cumulative ΔV:

         cum_dv += ||Favg|| * DT_FSW / SC_MASS.

3. Between FSW steps, propagate with HCW equations using current Favg:

      n = ||rB_burn × vB_burn|| / ||rB_burn||²,

      a_R = 3n² ρ_R + 2n ρ̇_T + F_R/SC_MASS,
      a_T = −2n ρ̇_R + F_T/SC_MASS,
      a_N = −n² ρ_N + F_N/SC_MASS,

      state += DT_DYN * [ρ̇, a].

Final report
------------
At RV_EPOCH:

- Apophis geocentric ECI:

      rA_end, vA_end = rA_fn(RV_EPOCH), vA_fn(RV_EPOCH).

- Deputy ECI:

      chief_r_end, _ = chief_state(RV_EPOCH)
      C_C_end = dcm_lvlh(chief_r_end, vB_burn)  # assumed nearly constant
      ρ_f = state[:3]
      deputy_r_eci = chief_r_end + C_C_end.T @ ρ_f

- Apophis LVLH:

      C_A = dcm_lvlh(rA_end, vA_end)
      ap_offset = C_A @ (deputy_r_eci − rA_end) / 1e3  # [km]

The script prints:

- Final deputy LVLH (chief frame) in km,
- Position error vs target (in m),
- Deputy–Apophis offset in Apophis LVLH [km],
- Total ΔV (optimised impulse + PD “pulses”) in m/s.

Plot
----
A small “quick look” plot shows the final position error norm:

- Single point showing ‖ρ_f − TARGET‖ [m].

Usage
-----
Prerequisites:

- Basilisk installation with Apophis SPK (20099942.bsp) in the standard
  EphemerisData directory.
- `chief_reference.npz` in the working directory.
- MROE tools:
    * Either `MROE.py` providing MU, REQ, J2, mroe, propagate_mroe,
      mroe_to_lvlh, or
    * Basilisk’s astroConstants (for MU/REQ/J2) plus a compatible MROE module.

Run:

    python basilik_scenario_apophis_single_impulse_mroe.py

The script will:

- Build the chief + Apophis geometry,
- Solve for an M-ROE-consistent single-impulse LVLH ΔV,
- Apply simple Hill-frame PD trimming with dead-bands,
- Report final LVLH and Apophis LVLH offsets and total ΔV,
- Show a small diagnostic plot of the final position error magnitude.
"""

from __future__ import annotations
import os
import sys
import math
import warnings
from pathlib import Path

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from Basilisk import __path__ as bskPath
from Basilisk.simulation  import spacecraft, simpleNav, spiceInterface, ephemerisConverter
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion as om,
)

# ────────────── constants (try mroe_tools first, fall back otherwise) ──────
try:
    from MROE import MU as MU_EARTH, REQ as REQ_EARTH, J2 as J2_EARTH
except ImportError:
    try:
        from Basilisk.utilities import astroConstants as ac

        MU_EARTH  = ac.mu_Earth  * 1.0e9   # km³/s² → m³/s²
        REQ_EARTH = ac.req_Earth * 1.0e3   # km → m
        J2_EARTH  = ac.J2_Earth
    except ImportError:
        MU_EARTH  = 3.986_004_418e14       # m³ s⁻²
        REQ_EARTH = 6_378_137.0            # m
        J2_EARTH  = 1.082_626_68e-3

from MROE import mroe, propagate_mroe, mroe_to_lvlh

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────── scenario parameters ──────────────────────────────────
DEPLOY_EPOCH = 10_023.0               # s  (t₀)
BURN_EPOCH   = 10_024.0               # s  (ΔV)
RV_EPOCH     = 31_170.0               # s  (rendez-vous)

DT_DYN = 1.0                          # s  (sim integration)
DT_FSW = 5.0                          # s  (PD update)

SC_MASS = 30.0                        # kg
THR_MAX = 5e-4                        # N (0.5 mN)

TARGET  = np.array([1.43, 2.55, -0.27]) * 1e3   # m (chief-centred LVLH)

DB_POS  = np.array([3., 3., 2.])       # m
DB_VEL  = np.array([0.01, 0.01, 0.008])# m s⁻¹
Kp      = np.array([2e-4, 1e-4, 2e-4]) # N m⁻¹
Kd      = np.array([5e-3, 3e-3, 5e-3]) # N s m⁻¹

# ─────────────────── chief ephemeris helper ───────────────────────────────
def chief_state(t_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Chief ECI (r, v) in metres at UTC-seconds t_s."""
    data = np.load("chief_reference.npz")     # km, km s⁻¹
    T, R, V = data["times"], data["positions"], data["velocities"]
    r_km = np.array([np.interp(t_s, T, R[:, k]) for k in range(3)])
    v_km = np.array([np.interp(t_s, T, V[:, k]) for k in range(3)])
    return r_km * 1e3, v_km * 1e3            # → m, m s⁻¹

rB_burn, vB_burn = chief_state(BURN_EPOCH)

# ─────────────────── SPICE sampler (Apophis w.r.t Earth) ──────────────────
KERNELS = [
    "naif0012.tls",
    "pck00010.tpc",
    "de-403-masses.tpc",
    "de430.bsp",
    "20099942.bsp",
]
SPICE_DIR = (Path(bskPath[0]) / "supportData" / "EphemerisData").resolve()

def ensure_kernels() -> None:
    missing = [k for k in KERNELS if not (SPICE_DIR / k).is_file()]
    if missing:
        print("Missing SPICE kernels:", *missing, sep="\n  ")
        sys.exit(1)

def make_apophis_sampler(start_iso: str,
                         *, span_s: int = 12*3600, step_s: int = 60):
    sim  = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("spP")
    task = sim.CreateNewTask("spT", macros.sec2nano(step_s))
    proc.addTask(task)

    spi = spiceInterface.SpiceInterface()
    spi.SPICEDataPath = str(SPICE_DIR) + os.sep
    spi.UTCCalInit    = start_iso
    for k in KERNELS:
        spi.loadSpiceKernel(k, str(SPICE_DIR) + os.sep)
    spi.addPlanetNames(["Earth", "Sun", "20099942"])
    spi.zeroBase = "Sun"
    sim.AddModelToTask("spT", spi)

    conv = ephemerisConverter.EphemerisConverter()
    for m in spi.planetStateOutMsgs:
        conv.addSpiceInputMsg(m)
    sim.AddModelToTask("spT", conv)

    eR = conv.ephemOutMsgs[0].recorder()   # Earth
    aR = conv.ephemOutMsgs[2].recorder()   # Apophis
    sim.AddModelToTask("spT", eR)
    sim.AddModelToTask("spT", aR)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(span_s))
    sim.ExecuteSimulation()

    def rA(t: float) -> np.ndarray:
        idx = int(round((t - DEPLOY_EPOCH) / step_s))
        return (aR.r_BdyZero_N[idx] - eR.r_BdyZero_N[idx]) * 1e3  # m

    def vA(t: float) -> np.ndarray:
        idx = int(round((t - DEPLOY_EPOCH) / step_s))
        return (aR.v_BdyZero_N[idx] - eR.v_BdyZero_N[idx]) * 1e3  # m s⁻¹

    return rA, vA

ensure_kernels()
rA_fn, vA_fn = make_apophis_sampler("2029 APR 13 11:38:28")

# ─────────────────── frame helpers ────────────────────────────────────────
def dcm_lvlh(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    i_hat = r / la.norm(r)
    k_hat = np.cross(r, v); k_hat /= la.norm(k_hat)
    j_hat = np.cross(k_hat, i_hat)
    return np.vstack((i_hat, j_hat, k_hat))      # rows = RTN axes (ECI comps)

# robust rv→ClassicElements (works on every Basilisk build) -----------------
def rv2classic(mu: float,
               r: np.ndarray,
               v: np.ndarray) -> om.ClassicElements:
    """
    Return a fully-filled ClassicElements object from position r and velocity v.
    Handles both calling conventions of orbitalMotion.rv2elem.
    """
    try:
        # Newer API: rv2elem(mu, r, v, outElem)
        elem = om.ClassicElements()
        om.rv2elem(mu, r, v, elem)
        return elem
    except TypeError:
        # Older API: rv2elem(mu, r, v) → ClassicElements / tuple
        out = om.rv2elem(mu, r, v)
        if isinstance(out, om.ClassicElements):
            return out
        a, e, inc, Om, argp, f = out           # tuple form
        elem = om.ClassicElements()
        elem.a, elem.e, elem.i = a, e, inc
        elem.Omega, elem.omega, elem.f = Om, argp, f
        return elem

# ─────────────────── M-ROE propagation for ΔV search ──────────────────────
def propagate_roe(dv_rtn: np.ndarray) -> np.ndarray:
    ce_c_b = rv2classic(MU_EARTH, rB_burn, vB_burn)

    C_burn = dcm_lvlh(rB_burn, vB_burn)
    dv_eci = C_burn.T @ dv_rtn
    dep_r_b, dep_v_b = rB_burn, vB_burn + dv_eci
    ce_d_b = rv2classic(MU_EARTH, dep_r_b, dep_v_b)

    roe0 = mroe(ce_c_b, ce_d_b)

    chief_b_mean = om.ClassicElements()
    om.clMeanOscMap(REQ_EARTH, J2_EARTH, ce_c_b, chief_b_mean, -1)  # osc→mean
    n_c = math.sqrt(MU_EARTH / chief_b_mean.a**3)

    roef = propagate_mroe(roe0, n_c, RV_EPOCH - BURN_EPOCH)

    rC_f, vC_f = chief_state(RV_EPOCH)
    ce_c_f = rv2classic(MU_EARTH, rC_f, vC_f)
    chief_f_mean = om.ClassicElements()
    om.clMeanOscMap(REQ_EARTH, J2_EARTH, ce_c_f, chief_f_mean, -1)

    rho_f, _ = mroe_to_lvlh(roef, chief_f_mean)
    return rho_f

γ = 10.0
residual = lambda dv: np.hstack((propagate_roe(dv) - TARGET, γ * dv))

opt = least_squares(residual, np.zeros(3), xtol=1e-12, ftol=1e-12, gtol=1e-12)
dv0_rtn = opt.x
print(
    f"Optimised ΔV (LVLH-RTN) @{BURN_EPOCH}s: "
    f"|ΔV|={la.norm(dv0_rtn):.3f} m/s   vector={dv0_rtn*1e3} mm/s"
)

# ─────────────────── simple PD trimming (Hill frame) ──────────────────────
steps  = int((RV_EPOCH - DEPLOY_EPOCH) / DT_DYN)
state  = np.zeros(6)           # ρ, ρ̇  (chief-centred LVLH)
Favg   = np.zeros(3)
cum_dv = la.norm(dv0_rtn)
burn_k = int((BURN_EPOCH - DEPLOY_EPOCH) / DT_DYN)

for k in range(steps):
    if k == burn_k:
        state[3:] += dv0_rtn

    if k % int(DT_FSW / DT_DYN) == 0:
        err_r, err_v = state[:3] - TARGET, state[3:]
        u = -Kp * err_r - Kd * err_v
        Favg = np.clip(u, -THR_MAX, THR_MAX)
        cum_dv += la.norm(Favg) * DT_FSW / SC_MASS

    rx, ry, rz, vx, vy, vz = state
    n = la.norm(np.cross(rB_burn, vB_burn)) / la.norm(rB_burn) ** 2
    ax = 3 * n * n * rx + 2 * n * vy + Favg[0] / SC_MASS
    ay = -2 * n * vx      + Favg[1] / SC_MASS
    az = -n * n * rz      + Favg[2] / SC_MASS
    state += DT_DYN * np.array([vx, vy, vz, ax, ay, az])

# ─────────────────── final report ─────────────────────────────────────────
rA_end, vA_end = rA_fn(RV_EPOCH), vA_fn(RV_EPOCH)
C_A = dcm_lvlh(rA_end, vA_end)

chief_r_end, _ = chief_state(RV_EPOCH)
C_C_end = dcm_lvlh(chief_r_end, vB_burn)   # orientation ≈ const. over coast

rho_f = state[:3]
deputy_r_eci = chief_r_end + C_C_end.T @ rho_f
ap_offset = C_A @ (deputy_r_eci - rA_end) / 1e3          # km

print("\n------ Rendez-vous summary ------")
print("Final deputy LVLH (chief):", rho_f / 1e3, "km")
print("Position error vs target :", rho_f - TARGET, "m")
print("Deputy-Apophis offset    :", ap_offset, "km (Apophis-LVLH)")
print(f"Total ΔV (impulse + pulses): {cum_dv:.3f} m/s")

# ─────────────────── quick plot (optional) ────────────────────────────────
plt.figure(figsize=(6, 3))
plt.title("Final position error magnitude")
plt.plot([0], [la.norm(rho_f - TARGET)], "ro")
plt.ylabel("‖ρ − ρ_target‖ [m]")
plt.xlabel("end of simulation")
plt.grid(True)
plt.show(block=True)
