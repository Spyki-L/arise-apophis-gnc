#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rafti_separation.py
===================

Generic RAFTI-style separation driver for Basilisk scenarios.

This module provides a small utility that:

- Models each RAFTI push as a half-sine force with peak F0 over [t_start,t_end],
- Applies equal-and-opposite forces and torques at the carrier and deputy mounts,
- Emits warnings if commanded Δv / peak force / equivalent mass exceed
  user-guide limits.

Intended usage
--------------

1) Build your carrier + deputies with `spacecraft.Spacecraft` and
   `extForceTorque.ExtForceTorque` effectors:

    sc_car   : carrier Spacecraft
    eft_car  : carrier ExtForceTorque (for separation)
    sc_deps  : list of deputy Spacecraft
    eft_deps : list of deputy ExtForceTorque (for separation on each deputy)

2) Construct a list of SepParams — one per deputy:

    from gnc.utils.rafti_separation import SepParams, RAFTISeparation, RAFTI

    sched = []
    for k in range(N):
        sched.append(
            SepParams(
                t_start = 10.0 + 5.0*k,
                t_end   = 10.0 + 5.0*k + 0.30,
                F0      = F0_k,
                s_car_B = s_dir_k,         # (3,) port normal in carrier body
                r_car_B = r_port_k,        # (3,) port position in carrier body
                r_dep_B = r_mount_k,       # (3,) mount point in deputy body
                dv_cmd  = dv_cmd_k,        # [m/s], for logging / checks
            )
        )

3) Instantiate the driver:

    sep = RAFTISeparation(
        dt      = sim_dt,
        sc_car  = sc_car,
        eff_car = eft_car,
        sc_dep  = sc_deps,
        eff_dep = eft_deps,
        sched   = sched,
        m_car   = mass_carrier,
        m_dep   = mass_deputy,
    )

4) In your main loop, call:

    sep.step(t_now)

on each time step t_now [s].

This module depends on Basilisk's messaging and extForceTorque, but is
otherwise agnostic to the rest of the simulation architecture.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numpy.linalg as la

from Basilisk.architecture import messaging
from Basilisk.simulation   import extForceTorque
from Basilisk.simulation   import spacecraft


# ───────────────────────── RAFTI limits / constants ─────────────────────────

class RAFTI:
    """
    RAFTI user-guide limits and tolerances (informational).

    These are *not* enforced; RAFTISeparation only prints warnings
    when commands exceed these values.
    """
    # Docking / utility misalignments
    DOCK_XY_TOL_M    = 0.010     # ±10 mm
    DOCK_ANG_TOL_DEG = 5.0       # ±5° about X/Y/Z

    UTIL_XY_TOL_M    = 0.001     # ±1 mm
    UTIL_Z_ANG_DEG   = 2.0       # ±2° about Z

    # Utility preload
    UTIL_PRELOAD_N   = 2000.0    # 2 kN preload

    # Rated approach conditions
    DOCK_VEL_MAX_MPS = 0.025     # 25 mm/s
    EQMASS_MAX_KG    = 1000.0    # ≤ 1000 kg equivalent mass

    # Mount design-load limit
    CONTACT_LIMIT_N  = 11600.0   # mounting-interface limit [N]


# ───────────────────────── geometry helpers ─────────────────────────

def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z,  y],
                     [z,   0.0, -x],
                     [-y,  x,   0.0]], float)


def _mrp_to_dcm(sigma: np.ndarray) -> np.ndarray:
    """
    Minimal MRP→DCM map, copied here so we don't depend on other modules.
    """
    s = np.asarray(sigma, float).reshape(3)
    s2 = float(np.dot(s, s))
    I3 = np.eye(3)
    if s2 < 1e-16:
        return I3.copy()
    S = _skew(s)
    return I3 + (8.0 * S @ S - 4.0 * (1.0 - s2) * S) / (1.0 + s2) ** 2


def equivalent_mass(m1: float, m2: float) -> float:
    """Reduced mass for two-body contact: m_eq = (m1 m2) / (m1 + m2)."""
    return (m1 * m2) / (m1 + m2)


# ───────────────────────── schedule dataclass ─────────────────────────

@dataclass
class SepParams:
    """
    RAFTI separation parameters for a single deputy.

    Parameters
    ----------
    t_start : float
        Burn start time [s].
    t_end : float
        Burn end time [s].
    F0 : float
        Peak half-sine force [N].
    s_car_B : (3,) ndarray
        Unit vector in carrier body frame giving port normal (carrier→deputy).
    r_car_B : (3,) ndarray
        Position of RAFTI port in carrier body frame [m].
    r_dep_B : (3,) ndarray
        Mount point of RAFTI interface in deputy body frame [m].
    dv_cmd : float
        Commanded Δv for the deputy [m/s] (for logging/warnings only).
    """
    t_start: float
    t_end: float
    F0: float
    s_car_B: np.ndarray
    r_car_B: np.ndarray
    r_dep_B: np.ndarray
    dv_cmd: float


# ───────────────────────── RAFTI driver ─────────────────────────

class RAFTISeparation:
    """
    RAFTISeparation(dt, sc_car, eff_car, sc_dep, eff_dep, sched, m_car, m_dep)

    A simple equal-and-opposite separation driver that:

    - Reads the current carrier and deputy attitudes from their spacecraft
      state outputs,
    - Applies half-sine forces at each RAFTI interface,
    - Generates body forces and torques for each body via extForceTorque.

    Notes
    -----
    • The constructor subscribes each extForceTorque effector to internally
      owned CmdTorqueBodyMsg / CmdForceBodyMsg instances.
    • To "disable" separation at some time, simply stop calling step(t),
      or give an empty schedule.
    """

    def __init__(
        self,
        dt: float,
        sc_car: spacecraft.Spacecraft,
        eff_car: extForceTorque.ExtForceTorque,
        sc_dep: List[spacecraft.Spacecraft],
        eff_dep: List[extForceTorque.ExtForceTorque],
        sched: List[SepParams],
        m_car: float,
        m_dep: float,
    ):
        self.dt = float(dt)
        self.sc_car = sc_car
        self.eff_car = eff_car
        self.sc_dep = list(sc_dep)
        self.eff_dep = list(eff_dep)
        self.sched = list(sched)
        self.m_car = float(m_car)
        self.m_dep = float(m_dep)

        if len(self.sc_dep) != len(self.eff_dep) or len(self.sc_dep) != len(self.sched):
            raise ValueError("Length mismatch: sc_dep, eff_dep, and sched must have same length.")

        # Command messages for carrier
        self.car_tauMsg = messaging.CmdTorqueBodyMsg()
        self.car_forMsg = messaging.CmdForceBodyMsg()
        eff_car.cmdTorqueInMsg.subscribeTo(self.car_tauMsg)
        eff_car.cmdForceBodyInMsg.subscribeTo(self.car_forMsg)

        # Command messages for each deputy
        self.dep_tauMsg = [messaging.CmdTorqueBodyMsg() for _ in self.sc_dep]
        self.dep_forMsg = [messaging.CmdForceBodyMsg() for _ in self.sc_dep]
        for ef, tm, fm in zip(self.eff_dep, self.dep_tauMsg, self.dep_forMsg):
            ef.cmdTorqueInMsg.subscribeTo(tm)
            ef.cmdForceBodyInMsg.subscribeTo(fm)

    # ---- internal half-sine profile ---------------------------------------
    def _half_sine(self, t: float, t0: float, T: float, F0: float) -> float:
        if t < t0 or t > t0 + T:
            return 0.0
        return F0 * math.sin(math.pi * (t - t0) / T)

    # ---- main call ---------------------------------------------------------
    def step(self, t: float) -> None:
        """
        Apply RAFTI separation forces / torques at time t [s].

        This must be called once per simulation step *after* the spacecraft
        attitudes have been propagated.
        """
        # Carrier attitude
        car_state = self.sc_car.scStateOutMsg.read()
        sigma_car = np.asarray(car_state.sigma_BN, float).reshape(3)
        C_BN_car = _mrp_to_dcm(sigma_car)
        C_NB_car = C_BN_car.T

        tau_car = np.zeros(3)
        f_car = np.zeros(3)

        for i, sp in enumerate(self.sched):
            dep_state = self.sc_dep[i].scStateOutMsg.read()
            sigma_dep = np.asarray(dep_state.sigma_BN, float).reshape(3)
            C_BN_dep = _mrp_to_dcm(sigma_dep)
            C_NB_dep = C_BN_dep.T

            # RAFTI rating checks near the start of each push
            if abs(t - sp.t_start) < 0.5 * self.dt:
                m_eq = equivalent_mass(self.m_car, self.m_dep)
                if m_eq > RAFTI.EQMASS_MAX_KG:
                    print(f"[RAFTI WARN dep{i+1}] m_eq={m_eq:.1f} kg exceeds {RAFTI.EQMASS_MAX_KG:.0f} kg rating")
                if sp.dv_cmd > RAFTI.DOCK_VEL_MAX_MPS:
                    print(f"[RAFTI WARN dep{i+1}] Δv_cmd={sp.dv_cmd:.3f} m/s exceeds RAFTI {RAFTI.DOCK_VEL_MAX_MPS*1e3:.0f} mm/s rating")
                if sp.F0 > RAFTI.CONTACT_LIMIT_N:
                    print(f"[RAFTI WARN dep{i+1}] peak push {sp.F0:.0f} N > contact limit {RAFTI.CONTACT_LIMIT_N:.0f} N")

            # Instantaneous force magnitude for this push
            Fmag = self._half_sine(t, sp.t_start, sp.t_end - sp.t_start, sp.F0)
            if Fmag <= 0.0:
                # zero commands to this deputy
                tm = messaging.CmdTorqueBodyMsgPayload()
                fm = messaging.CmdForceBodyMsgPayload()
                tm.torqueRequestBody = [0.0, 0.0, 0.0]
                fm.forceRequestBody = [0.0, 0.0, 0.0]
                self.dep_tauMsg[i].write(tm)
                self.dep_forMsg[i].write(fm)
                continue

            # Contact normal in inertial (carrier frame → inertial)
            s_car_B = np.asarray(sp.s_car_B, float).reshape(3)
            r_car_B = np.asarray(sp.r_car_B, float).reshape(3)
            r_dep_B = np.asarray(sp.r_dep_B, float).reshape(3)

            s_N = C_NB_car @ s_car_B
            F_N = Fmag * s_N

            # Equal and opposite forces in each body frame
            f_car_i_B = C_BN_car @ (+F_N)
            f_dep_i_B = C_BN_dep @ (-F_N)

            # Torques at the respective lever arms
            tau_car_i = np.cross(r_car_B, f_car_i_B)
            tau_dep_i = np.cross(r_dep_B, f_dep_i_B)

            # Write deputy commands
            tm_dep = messaging.CmdTorqueBodyMsgPayload()
            fm_dep = messaging.CmdForceBodyMsgPayload()
            tm_dep.torqueRequestBody = list(tau_dep_i)
            fm_dep.forceRequestBody = list(f_dep_i_B)
            self.dep_tauMsg[i].write(tm_dep)
            self.dep_forMsg[i].write(fm_dep)

            # Accumulate carrier wrench
            tau_car += tau_car_i
            f_car += f_car_i_B

        # Publish net carrier commands
        tm_car = messaging.CmdTorqueBodyMsgPayload()
        fm_car = messaging.CmdForceBodyMsgPayload()
        tm_car.torqueRequestBody = list(tau_car)
        fm_car.forceRequestBody = list(f_car)
        self.car_tauMsg.write(tm_car)
        self.car_forMsg.write(fm_car)

