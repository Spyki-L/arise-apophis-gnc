#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_scenario_single_impulse_mroe.py
=======================================

Purpose
-------
Single-impulse rendezvous design using **Mean Relative Orbital Elements** (M-ROE).

Given:
  • A chief trajectory in ECI (from a generic `chief_reference.npz`),
  • A burn epoch t_burn and a rendezvous epoch t_rv,
  • A desired relative LVLH (RTN) offset at rendezvous,

this script:

  1) Uses M-ROE to size a single Δv at t_burn in the chief LVLH frame, such that
     the deputy ends up near the desired LVLH offset at t_rv.
  2) Prints the optimised Δv (RTN components and magnitude), the predicted final
     LVLH offset, and the residual error.
  3) Can be used as a **donor** for a library helper like:

        mroe_single_impulse_to_target(...)

in something like `gnc/guidance/mroe_guidance.py`.

Assumptions
-----------
  • Earth-centred dynamics (uses Earth μ and J2).
  • Chief trajectory is stored in `chief_reference.npz` with fields:

        times      : (N,)  [s] since some epoch (e.g., departure),
        positions  : (N,3) [km] ECI,
        velocities : (N,3) [km/s] ECI.

  • We treat those as “osculating” states and let the M-ROE utilities map
    them to mean elements via a first-order J2 model.

Dependencies
------------
  • Basilisk (for orbitalMotion / ClassicElements),
  • `mroe_tools.py` (preferred) or `MROE.py` providing:

        MU_EARTH, REQ_EARTH, J2_EARTH
        mroe(...)
        propagate_mroe(...)
        mroe_to_lvlh(...)

  • SciPy (for least_squares),
  • NumPy.

Usage
-----
  1. Generate a chief trajectory (any Earth-centred one) and save as:

         chief_reference.npz

     with keys "times", "positions", "velocities" (units: km, km/s).

  2. Adjust:
         BURN_EPOCH
         RV_EPOCH
         TARGET_LVLH_M

  3. Run:

         python basilik_scenario_single_impulse_mroe.py

  The script will print:
    • Optimised Δv in RTN [m/s] and [mm/s],
    • Final LVLH offset [m] and position error [m],
    • Total |Δv|.
"""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path

import numpy as np
import numpy.linalg as la
from scipy.optimize import least_squares

from Basilisk.utilities import orbitalMotion as om

# ─────────────────── import M-ROE utilities (mroe_tools or MROE) ───────────────────

for _modname in ("mroe_tools", "MROE"):
    try:
        mroemod = __import__(_modname)
        break
    except ModuleNotFoundError:
        mroemod = None

if mroemod is None:
    raise ImportError("Neither mroe_tools.py nor MROE.py is importable; "
                      "place one of them on PYTHONPATH.")

mroe            = mroemod.mroe
propagate_mroe  = mroemod.propagate_mroe
mroe_to_lvlh    = mroemod.mroe_to_lvlh
MU_EARTH        = mroemod.MU_EARTH   # [m³/s²]
REQ_EARTH       = mroemod.REQ_EARTH  # [m]
J2_EARTH        = mroemod.J2_EARTH

try:
    from Basilisk.utilities.orbitalMotion import ClassicElements
except ImportError:
    ClassicElements = None


# ─────────────────── USER “KNOBS” ───────────────────

# Chief trajectory NPZ (Earth-centred)
CHIEF_NPZ_PATH = Path(os.getenv("CHIEF_REF_NPZ", "chief_reference.npz"))

# Times [s] since chief_reference.npz epoch
BURN_EPOCH = 10_024.0      # t_burn
RV_EPOCH   = 31_170.0      # rendezvous time

# Desired final LVLH (RTN) position of deputy relative to chief [m]
# (Can be any RTN offset; Apophis-esque example commented out.)
TARGET_LVLH_M = np.array([1.43e3, 2.55e3, -0.27e3])   # [m]
# TARGET_LVLH_M = np.array([0.0, 500.0, 0.0])         # example: along-track slot

# Regularisation weight on Δv in the least-squares residual
DV_REG_WEIGHT = 10.0   # γ; larger = penalise large Δv more strongly


# ─────────────────── small helpers ───────────────────

def dcm_lvlh(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Build LVLH DCM with rows = [R_hat; T_hat; N_hat] in ECI components.

    r, v : chief ECI position/velocity [m]/[m/s].
    """
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    i_hat = r / la.norm(r)
    k_hat = np.cross(r, v)
    k_hat /= la.norm(k_hat)
    j_hat = np.cross(k_hat, i_hat)
    return np.vstack((i_hat, j_hat, k_hat))


def rv2classic(mu: float,
               r: np.ndarray,
               v: np.ndarray) -> om.ClassicElements:
    """
    Robust wrapper around Basilisk orbitalMotion.rv2elem / rv2coe that always
    returns a ClassicElements-like object, for use with M-ROE routines.
    """
    r = np.asarray(r, float)
    v = np.asarray(v, float)

    # Newer signature: rv2elem(mu, r, v, outElem)
    try:
        elem = om.ClassicElements()
        om.rv2elem(mu, r, v, elem)
        return elem
    except TypeError:
        pass

    # Older style: rv2elem(mu, r, v) → ClassicElements or tuple
    out = om.rv2elem(mu, r, v)
    if isinstance(out, om.ClassicElements):
        return out

    # Tuple form: (a, e, i, Ω, ω, f)
    a, e, inc, Om, argp, f = out
    elem = om.ClassicElements()
    elem.a      = float(a)
    elem.e      = float(e)
    elem.i      = float(inc)
    elem.Omega  = float(Om)
    elem.omega  = float(argp)
    elem.f      = float(f)
    return elem


def load_chief_reference(npz_path: Path):
    """
    Load chief_reference.npz and return time grid and ECI states in SI units.

    Expected keys:
      times      : (N,) [s]
      positions  : (N,3) [km]
      velocities : (N,3) [km/s]

    Returns
    -------
    t_ref_s : (N,) ndarray
    r_ref_m : (N,3) ndarray
    v_ref_mps : (N,3) ndarray
    """
    if not npz_path.is_file():
        raise FileNotFoundError(f"chief_reference.npz not found at: {npz_path}")

    data = np.load(npz_path)
    t_ref   = np.asarray(data["times"], float)
    r_km    = np.asarray(data["positions"], float)
    v_kmps  = np.asarray(data["velocities"], float)

    r_m   = r_km * 1e3
    v_mps = v_kmps * 1e3
    return t_ref, r_m, v_mps


def interp_state(t_ref: np.ndarray,
                 r_ref: np.ndarray,
                 v_ref: np.ndarray,
                 t_query: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple component-wise interpolation of (r,v) at time t_query.
    """
    r = np.array([np.interp(t_query, t_ref, r_ref[:, k]) for k in range(3)])
    v = np.array([np.interp(t_query, t_ref, v_ref[:, k]) for k in range(3)])
    return r, v


# ─────────────────── M-ROE propagation for a candidate Δv ───────────────────

def propagate_roe_to_rendezvous(
    dv_rtn_mps: np.ndarray,
    r_burn_m: np.ndarray,
    v_burn_mps: np.ndarray,
    r_rv_m: np.ndarray,
    v_rv_mps: np.ndarray,
    t_burn_s: float,
    t_rv_s: float,
) -> np.ndarray:
    """
    For a candidate Δv in chief LVLH-RTN at t_burn, compute the resulting final
    LVLH position at t_rv using M-ROE propagation.

    Parameters
    ----------
    dv_rtn_mps : (3,) ndarray
        LVLH Δv at the burn epoch [m/s] in RTN.
    r_burn_m, v_burn_mps : (3,) ndarray
        Chief ECI state at t_burn [m], [m/s].
    r_rv_m,   v_rv_mps   : (3,) ndarray
        Chief ECI state at rendezvous time t_rv [m], [m/s].
    t_burn_s, t_rv_s     : float
        Burn and rendezvous epochs [s].

    Returns
    -------
    rho_f_m : (3,) ndarray
        Predicted deputy LVLH position [m] at t_rv in the chief LVLH frame.
    """
    dt = float(t_rv_s - t_burn_s)
    if dt <= 0.0:
        raise ValueError("t_rv_s must be greater than t_burn_s.")

    # Chief classical elements at burn (osculating)
    ce_c_b = rv2classic(MU_EARTH, r_burn_m, v_burn_mps)

    # LVLH rotation at burn
    C_LN_burn = dcm_lvlh(r_burn_m, v_burn_mps)  # rows: R,T,N in inertial
    C_NL_burn = C_LN_burn.T

    # Map LVLH Δv into ECI
    dv_eci_mps = C_NL_burn @ dv_rtn_mps

    # Deputy osculating state right after the burn
    r_dep_b = r_burn_m.copy()
    v_dep_b = v_burn_mps + dv_eci_mps

    ce_d_b = rv2classic(MU_EARTH, r_dep_b, v_dep_b)

    # M-ROE at burn
    roe0 = mroe(ce_c_b, ce_d_b, MU_EARTH, 0.0)

    # Chief mean elements at burn (J2 mapping, osculating → mean)
    chief_b_mean = om.ClassicElements()
    om.clMeanOscMap(REQ_EARTH, J2_EARTH, ce_c_b, chief_b_mean, -1)  # sgn = −1: osc→mean

    n_c = math.sqrt(MU_EARTH / chief_b_mean.a**3)  # chief mean motion [rad/s]

    # Propagate M-ROE linearly
    roe_f = propagate_mroe(roe0, n_c, dt)

    # Chief mean elements at rendezvous
    ce_c_rv = rv2classic(MU_EARTH, r_rv_m, v_rv_mps)
    chief_rv_mean = om.ClassicElements()
    om.clMeanOscMap(REQ_EARTH, J2_EARTH, ce_c_rv, chief_rv_mean, -1)

    # Map M-ROE → LVLH position/velocity
    rho_f_m, _ = mroe_to_lvlh(roe_f, chief_rv_mean, MU_EARTH)
    return rho_f_m


# ─────────────────── main optimisation (single-impulse) ───────────────────

def solve_single_impulse_mroe(
    t_ref_s: np.ndarray,
    r_ref_m: np.ndarray,
    v_ref_mps: np.ndarray,
    t_burn_s: float,
    t_rv_s: float,
    target_lvlh_m: np.ndarray,
    gamma_reg: float = DV_REG_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve for a single LVLH Δv (RTN) at t_burn that drives the deputy
    towards `target_lvlh_m` at t_rv, using linear M-ROE.

    Parameters
    ----------
    t_ref_s, r_ref_m, v_ref_mps :
        Chief time and ECI states in SI.
    t_burn_s : float
        Burn epoch [s].
    t_rv_s   : float
        Rendezvous epoch [s] (> t_burn_s).
    target_lvlh_m : (3,) ndarray
        Desired final LVLH position [m] at t_rv.
    gamma_reg : float
        Regularisation weight on Δv in the least-squares residual.

    Returns
    -------
    dv_rtn_mps : (3,) ndarray
        Optimised LVLH-RTN Δv [m/s].
    rho_f_m    : (3,) ndarray
        Predicted final LVLH position at t_rv [m] using that Δv.
    """
    target_lvlh_m = np.asarray(target_lvlh_m, float).reshape(3)

    # Chief states at burn and rendezvous
    r_burn_m, v_burn_mps = interp_state(t_ref_s, r_ref_m, v_ref_mps, t_burn_s)
    r_rv_m,   v_rv_mps   = interp_state(t_ref_s, r_ref_m, v_ref_mps, t_rv_s)

    def residual(dv_rtn_flat: np.ndarray) -> np.ndarray:
        dv_rtn = dv_rtn_flat.reshape(3)
        rho_f = propagate_roe_to_rendezvous(
            dv_rtn, r_burn_m, v_burn_mps, r_rv_m, v_rv_mps, t_burn_s, t_rv_s
        )
        # Stack position error and regularisation on Δv
        return np.hstack((rho_f - target_lvlh_m, gamma_reg * dv_rtn))

    x0 = np.zeros(3)  # initial guess = no Δv
    sol = least_squares(residual, x0, xtol=1e-12, ftol=1e-12, gtol=1e-12)

    dv_opt = sol.x.reshape(3)
    rho_f_opt = propagate_roe_to_rendezvous(
        dv_opt, r_burn_m, v_burn_mps, r_rv_m, v_rv_mps, t_burn_s, t_rv_s
    )
    return dv_opt, rho_f_opt


# ─────────────────── script entrypoint ───────────────────

def main():
    print(f"[INFO] Loading chief trajectory from: {CHIEF_NPZ_PATH.resolve()}")
    t_ref_s, r_ref_m, v_ref_mps = load_chief_reference(CHIEF_NPZ_PATH)

    print(f"[INFO] Burn epoch  t_burn = {BURN_EPOCH:.1f} s")
    print(f"[INFO] Rendezvous  t_rv   = {RV_EPOCH:.1f} s")
    if RV_EPOCH <= BURN_EPOCH:
        raise ValueError("RV_EPOCH must be greater than BURN_EPOCH.")

    dv_rtn, rho_f = solve_single_impulse_mroe(
        t_ref_s, r_ref_m, v_ref_mps,
        t_burn_s=BURN_EPOCH,
        t_rv_s=RV_EPOCH,
        target_lvlh_m=TARGET_LVLH_M,
        gamma_reg=DV_REG_WEIGHT,
    )

    dv_mag = la.norm(dv_rtn)
    err_vec = rho_f - TARGET_LVLH_M
    err_mag = la.norm(err_vec)

    print("\n====== M-ROE Single-Impulse Rendezvous (generic) ======")
    print(f"Δv (LVLH-RTN) [m/s]   : {dv_rtn}")
    print(f"Δv (LVLH-RTN) [mm/s]  : {dv_rtn*1e3}")
    print(f"|Δv| [m/s]            : {dv_mag:.3f}")
    print("")
    print(f"Final LVLH position [m]: {rho_f}")
    print(f"Target LVLH position [m]: {TARGET_LVLH_M}")
    print(f"Position error [m]     : {err_vec}")
    print(f"‖error‖ [m]            : {err_mag:.3f}")
    print("=======================================================")


if __name__ == "__main__":
    main()

