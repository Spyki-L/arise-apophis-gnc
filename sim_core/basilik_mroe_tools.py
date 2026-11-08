#!/usr/bin/env python3
"""
basilik_mroe_tools.py

Purpose
-------
Utility module for working with **Mean Relative Orbital Elements (M-ROE)** on
top of Basilisk’s `orbitalMotion` utilities. It provides a small, focused API
for:

- Converting osculating Classical Orbital Elements (COE) to **mean** COE via a
  first-order J2 mean–osculating mapping.
- Computing 6-element **Mean ROE** vectors between a chief and deputy at a
  given epoch (with optional time offset).
- Linearly propagating M-ROE in time under two-body dynamics.
- Mapping M-ROE back to instantaneous LVLH relative states (ρ, ρ̇) using the
  Schaub–Alfriend formulation.

The functions are set up to be **Basilisk-version agnostic**: if
`astroConstants` is not available, the module falls back to hard-wired Earth
constants.

Earth constants (version-agnostic)
----------------------------------
On import, the module tries:

    from Basilisk.utilities import astroConstants as ac

If available (modern Basilisk):

- MU_EARTH  = ac.mu_Earth  (converted from km³/s² to m³/s²)
- REQ_EARTH = ac.req_Earth (converted from km to m)
- J2_EARTH  = ac.J2_Earth

If not available (older Basilisk or custom builds), it falls back to:

- MU_EARTH  = 3.986004418e14  m³/s²
- REQ_EARTH = 6378.137e3      m
- J2_EARTH  = 1.08262668e-3

Public API
----------
1) `mroe(chief_osc, dep_osc, mu=MU, dt=0.0) -> np.ndarray`
   - Inputs:
       chief_osc, dep_osc : `orbitalMotion.ClassicElements` (osculating) at a
                            base epoch t₀.
       mu                 : gravitational parameter [m³/s²], default Earth MU.
       dt                 : seconds since t₀ (optional phase offset).
   - Behavior:
       * Maps both osculating COE to mean COE via `clMeanOscMap` (sgn = −1).
       * Computes mean anomalies M_c, M_d at t₀ + dt using their respective
         mean motions n_c, n_d.
       * Evaluates the mean ROE components:
             δa  = (a_d − a_c)/a_c
             δλ  = (u_d − u_c) + (Ω_d − Ω_c) cos(i_c)
             δex = e_d cos(ω_d) − e_c cos(ω_c)
             δey = e_d sin(ω_d) − e_c sin(ω_c)
             δix = i_d − i_c
             δiy = (Ω_d − Ω_c) sin(i_c)
         and wraps δλ into (−π, π].
   - Output:
       * 6×1 numpy array [δa, δλ, δex, δey, δix, δiy].

2) `propagate_mroe(roe0, n_c, t) -> np.ndarray`
   - Inputs:
       roe0 : initial mean ROE [δa, δλ, δex, δey, δix, δiy].
       n_c  : chief mean motion [rad/s].
       t    : propagation time [s].
   - Behavior:
       * Uses the standard linearized two-body ROE propagation:
             δλ(t) = δλ₀ + 1.5 n_c t δa
             [δex, δey] and [δix, δiy] rotate by angle φ = n_c t
               via planar rotation matrices.
       * Wraps δλ(t) into (−π, π].
   - Output:
       * Propagated ROE at time t, as [δa, δλ, δex, δey, δix, δiy].

3) `mroe_to_lvlh(roe, chief_mean, mu=MU) -> (ρ, ρ̇)`
   - Inputs:
       roe         : mean ROE [δa, δλ, δex, δey, δix, δiy] at some epoch.
       chief_mean  : `ClassicElements` (mean) for the chief at that same epoch.
       mu          : gravitational parameter [m³/s²], default Earth MU.
   - Behavior:
       * Converts the chief’s mean elements into:
             a, e, i, Ω, ω, M.
       * Solves Kepler’s equation M → E via a local Newton–Raphson
         (`kepler_M2E`), then obtains true anomaly ν and argument of latitude u.
       * Uses Schaub–Alfriend style formulas to map ROE to LVLH:
             ρ_R, ρ_T, ρ_N
             ρ̇_R, ρ̇_T, ρ̇_N
         based on a, e, u, and n = √(μ/a³).
   - Output:
       * Tuple `(ρ, ρ̇)` where each is a 3-element numpy array in the chief’s
         LVLH frame [R, T, N] with units [m] and [m/s].

4) Internal helpers
   - `_osc2mean(elem_osc)`:
       * Wraps Basilisk `clMeanOscMap` (J2 1st-order) for osc→mean conversion.
   - `_mean_anomaly(el)`:
       * Computes mean anomaly M from true anomaly f and eccentricity e.
   - `_ce_to_np(el)`:
       * Packs a ClassicElements into numpy `[a, e, i, Ω, ω, M]`.
   - `kepler_M2E(M, e, tol=1e-12)`:
       * Simple scalar Newton–Raphson solver for E given M, e.

Usage
-----
Typical pattern inside a Basilisk script:

    from sim_core.basilik_mroe_tools import mroe, propagate_mroe, mroe_to_lvlh
    from Basilisk.utilities import orbitalMotion as om

    chief_osc = om.ClassicElements()
    dep_osc   = om.ClassicElements()
    # ... fill Classical elements in meters/radians ...

    # Compute ROE at epoch
    δα = mroe(chief_osc, dep_osc)

    # Propagate ROE 600 s forward
    a_c = chief_osc.a
    n_c = np.sqrt(MU / a_c**3)
    δα_600 = propagate_mroe(δα, n_c, 600.0)

    # Convert back to instantaneous LVLH relative state at that epoch
    chief_mean = _osc2mean(chief_osc)  # or use an existing mean element set
    ρ, ρ̇ = mroe_to_lvlh(δα_600, chief_mean)

This module assumes SI units (meters, seconds, radians) internally and uses
Earth J2 and radius by default, but `mu` can be overridden for other bodies.
"""


import numpy as np
from Basilisk.utilities import orbitalMotion as om

# ─────────────── Earth constants with version-agnostic fallback ──────────
try:
    from Basilisk.utilities import astroConstants as ac            # ≥ v2.1
    MU_EARTH  = ac.mu_Earth   * 1.0e9    # km³/s² → m³/s²
    REQ_EARTH = ac.req_Earth  * 1.0e3    # km     → m
    J2_EARTH  = ac.J2_Earth
except ImportError:                                               # older
    MU_EARTH  = 3.986004418e14          # m³ s⁻²
    REQ_EARTH = 6378.137e3              # m
    J2_EARTH  = 1.08262668e-3
# ------------------------------------------------------------------------

MU  = MU_EARTH
REQ = REQ_EARTH
J2  = J2_EARTH

# ========== helpers ======================================================
def _osc2mean(elem_osc):
    """
    Osculating → mean ClassicElements (first-order J2).  Returns a COPY.
    """
    out = om.ClassicElements()
    om.clMeanOscMap(REQ, J2, elem_osc, out, -1)    # sgn = −1  (osc→mean)
    return out


def _mean_anomaly(el):
    """Return mean anomaly for a **mean** element set."""
    E = om.f2E(el.f, el.e)
    return om.E2M(E, el.e)


def _ce_to_np(el):
    """ClassicElements ➜ numpy array [a,e,i,Ω,ω,M] (m, rad)."""
    return np.array([el.a, el.e, el.i, el.Omega, el.omega, _mean_anomaly(el)])


def kepler_M2E(M, e, tol=1e-12):
    """Scalar Kepler solver (Newton-Raphson)."""
    E = M
    for _ in range(30):
        dE = (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E


# ========== main API =====================================================
def mroe(chief_osc, dep_osc, mu=MU, dt=0.0):
    """
    Compute the **Mean ROE** vector [δa, δλ, δex, δey, δix, δiy] at t₀+dt.

    chief_osc, dep_osc : ClassicElements (osculating) at epoch t₀
    mu                 : gravitational parameter  [m³ s⁻²]
    dt                 : seconds since t₀
    """
    c_m = _osc2mean(chief_osc)
    d_m = _osc2mean(dep_osc)

    a_c, e_c, i_c, Om_c, argp_c, M0_c = _ce_to_np(c_m)
    a_d, e_d, i_d, Om_d, argp_d, M0_d = _ce_to_np(d_m)

    n_c = np.sqrt(mu / a_c**3)
    n_d = np.sqrt(mu / a_d**3)
    M_c = M0_c + n_c*dt
    M_d = M0_d + n_d*dt

    u_c = argp_c + M_c
    u_d = argp_d + M_d

    δa  = (a_d - a_c)/a_c
    δλ  = (u_d - u_c) + (Om_d - Om_c)*np.cos(i_c)
    δex = e_d*np.cos(argp_d) - e_c*np.cos(argp_c)
    δey = e_d*np.sin(argp_d) - e_c*np.sin(argp_c)
    δix = i_d - i_c
    δiy = (Om_d - Om_c)*np.sin(i_c)

    δλ  = (δλ + np.pi) % (2*np.pi) - np.pi   # wrap to (-π, π]

    return np.array([δa, δλ, δex, δey, δix, δiy])


def propagate_mroe(roe0, n_c, t):
    """
    Linear time-propagation of mean ROE under two-body dynamics.

    n_c : mean motion of the chief (rad s⁻¹)
    t   : propagation time (s)
    """
    da, dl0, ex0, ey0, ix0, iy0 = roe0
    φ = n_c * t
    c, s = np.cos(φ), np.sin(φ)

    dl   = dl0 + 1.5*φ*da
    dl   = (dl + np.pi) % (2*np.pi) - np.pi
    ex, ey = c*ex0 + s*ey0, -s*ex0 + c*ey0
    ix, iy = c*ix0 + s*iy0, -s*ix0 + c*iy0

    return np.array([da, dl, ex, ey, ix, iy])


def mroe_to_lvlh(roe, chief_mean, mu=MU):
    """
    Map a Mean-ROE vector ➜ instantaneous LVLH (ρ, ρ̇)  [m], [m/s].

    chief_mean : ClassicElements (mean) at the *same* epoch as `roe`.
    """
    a, e, inc, Om, argp, M = _ce_to_np(chief_mean)
    n = np.sqrt(mu / a**3)
    E = kepler_M2E(M, e)
    ν = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    u = argp + ν

    da, dl, ex, ey, ix, iy = roe
    ρR = a*(da -  ex*np.cos(u) - ey*np.sin(u))
    ρT = a*(2*dl + 2*ex*np.sin(u) - 2*ey*np.cos(u))
    ρN = a*(ix*np.sin(u) - iy*np.cos(u))
    ρdR = a*n*( ey*np.cos(u) -  ex*np.sin(u))
    ρdT = a*n*( 2 + 2*ex*np.cos(u) + 2*ey*np.sin(u))
    ρdN = a*n*( ix*np.cos(u) +  iy*np.sin(u))

    return (np.array([ρR, ρT, ρN]),
            np.array([ρdR, ρdT, ρdN]))

