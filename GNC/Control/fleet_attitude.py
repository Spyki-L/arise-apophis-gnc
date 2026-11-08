#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fleet_attitude.py
=================

Generic fleet-level 6-DOF attitude simulation for a deputy swarm.

This is a direct adaptation of your working Apophis fleet mini-sim, but
with a mission-agnostic rendezvous bundle:

    RendezvousBundle(t, rD_N, vD_N, bRef_N, optional q_RN)

The stack is:

    FleetAttitudeConfig  – global knobs (dt, gains, noise, logging, etc.)
    RendezvousBundle     – per-deputy rendezvous trajectory
    Deputy               – one spacecraft worth of Basilisk modules
    FleetAttitudeSim     – orchestrates deputies, tasks, and logging

Each deputy gets:

- spacecraft + extForceTorque
- simpleNav (optionally noisy)
- attTrackingError + mrpFeedback (+ optional torque LPF)
- its own AttRefMsg (driven by the rendezvous bundle)
- recorders for error, nav, truth, and torque

The scenario (e.g. geo_fleet_attitude_demo.py) is responsible for:

- Loading rendezvous_full_D*.npz into RendezvousBundle objects
- Instantiating FleetAttitudeConfig (dt, gains, noise)
- Calling FleetAttitudeSim.run() and FleetAttitudeSim.save_logs()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import math

import numpy as np
import numpy.linalg as la

from Basilisk.architecture import messaging, sys_model_task
from Basilisk.simulation import spacecraft, simpleNav, extForceTorque
from Basilisk.fswAlgorithms import (
    attTrackingError,
    mrpFeedback,
    lowPassFilterTorqueCommand,
)
from Basilisk.utilities import macros


# ╔═══════════════════════════════════════════════════════════════════════╗
# Helpers: rotations, TRIAD, recorder-safe extraction
# ╚═══════════════════════════════════════════════════════════════════════╝

def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    """DCM → quaternion (scalar-first)."""
    tr = float(np.trace(C))
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        q0 = 0.25 * S
        q1 = (C[2, 1] - C[1, 2]) / S
        q2 = (C[0, 2] - C[2, 0]) / S
        q3 = (C[1, 0] - C[0, 1]) / S
    else:
        i = int(np.argmax([C[0, 0], C[1, 1], C[2, 2]]))
        if i == 0:
            S = math.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2]) * 2.0
            q0 = (C[2, 1] - C[1, 2]) / S
            q1 = 0.25 * S
            q2 = (C[0, 1] + C[1, 0]) / S
            q3 = (C[0, 2] + C[2, 0]) / S
        elif i == 1:
            S = math.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2]) * 2.0
            q0 = (C[0, 2] - C[2, 0]) / S
            q1 = (C[0, 1] + C[1, 0]) / S
            q2 = 0.25 * S
            q3 = (C[1, 2] + C[2, 1]) / S
        else:
            S = math.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1]) * 2.0
            q0 = (C[1, 0] - C[0, 1]) / S
            q1 = (C[0, 2] + C[2, 0]) / S
            q2 = (C[1, 2] + C[2, 1]) / S
            q3 = 0.25 * S
    q = np.array([q0, q1, q2, q3])
    return q / la.norm(q)


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Quaternion (scalar-first) → DCM."""
    q0, q1, q2, q3 = q
    qv = np.array([q1, q2, q3])

    return (q0 * q0 - np.dot(qv, qv)) * np.eye(3) \
        + 2.0 * np.outer(qv, qv) \
        + 2.0 * q0 * np.array([[0.0, -q3,  q2],
                               [q3,  0.0, -q1],
                               [-q2, q1,  0.0]])


def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    """Quaternion (scalar-first) → MRP."""
    q0 = float(q[0])
    qv = np.array(q[1:], dtype=float)
    if 1.0 + q0 < 1e-12:
        # shadow set
        denom = max(1.0 - q0, 1e-12)
        return -qv / denom
    return qv / (1.0 + q0)


def normalize(v: np.ndarray, eps: float = 1e-15) -> tuple[np.ndarray, float]:
    """Return (unit_vector, norm) with graceful handling of tiny norms."""
    n = float(la.norm(v))
    if n < eps:
        return np.zeros_like(v), 0.0
    return v / n, n


def triad_from_boresight_and_tangent(los: np.ndarray,
                                     vel: np.ndarray) -> np.ndarray:
    """
    Build C_NB from LOS (→ z_B) and tangential velocity (→ x_B).

    Matches the TRIAD logic used in your Apophis mini-sim.
    """
    zB, _ = normalize(los)
    v_perp = vel - np.dot(vel, zB) * zB
    xB, nx = normalize(v_perp)
    if nx < 1e-10:
        tmp = np.array([1.0, 0.0, 0.0]) if abs(zB[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        xB = normalize(np.cross(tmp, zB))[0]
    yB = np.cross(zB, xB)
    xB = normalize(np.cross(yB, zB))[0]
    yB = normalize(np.cross(zB, xB))[0]
    return np.column_stack((xB, yB, zB))


def safe_field(rec, name: str, cols: int = 3) -> np.ndarray:
    """
    Robustly pull a field off a Recorder object and coerce it to (N,cols).
    If the field is missing, returns a (0,cols) array.
    """
    if not hasattr(rec, name):
        return np.zeros((0, cols))
    arr = np.array(getattr(rec, name))
    if arr.ndim == 1:
        if arr.size == 0:
            return np.zeros((0, cols))
        if arr.size % cols == 0:
            return arr.reshape(-1, cols)
        return arr.reshape(-1, 1)
    return arr


# ╔═══════════════════════════════════════════════════════════════════════╗
# Data classes: configuration and rendezvous bundles
# ╚═══════════════════════════════════════════════════════════════════════╝

@dataclass
class FleetAttitudeConfig:
    # task periods [s]
    dt_dyn: float = 0.1
    dt_sens: float = 0.1

    # controller gains
    k_sigma: float = 0.08
    p_rate: float = 0.9
    ki: float = -1.0
    i_lim: float = 0.01

    # torque LPF
    use_lpf: bool = True
    lpf_cut: float = 0.8  # Hz

    # spacecraft properties
    mass_kg: float = 250.0
    inertia: tuple[float, float, float] = (18.0, 15.0, 12.0)
    init_y_deg: float = 5.0

    # SimpleNav noise (1σ); if None → defaults configured in __post_init__
    nav_sig: Optional[dict] = None

    # logging flags
    log_nav: bool = True
    log_truth: bool = True

    # runtime
    t_stop: Optional[float] = None
    prefer_qrn: bool = True  # if bundle has q_RN, prefer it over TRIAD

    def __post_init__(self):
        if self.nav_sig is None:
            self.nav_sig = dict(
                pos=50.0,
                vel=0.05,
                att=5e-4,
                rate=2e-5,
                sun=1e-3,
                dv=0.0,
            )


@dataclass
class RendezvousBundle:
    """
    Mission-agnostic GEO rendezvous bundle for one deputy.

    Fields are exactly what your GEO rendezvous .npz files contain,
    but internally we mirror the Apophis NPZBundle naming so the rest
    of the code is identical.
    """
    t: np.ndarray          # (N,)
    rD_N: np.ndarray       # (N,3)
    vD_N: np.ndarray       # (N,3)
    bRef_N: np.ndarray     # (N,3)
    q_RN: Optional[np.ndarray] = None  # (N,4) or None

    def __post_init__(self):
        self.t = np.asarray(self.t, float)
        self.rD = np.asarray(self.rD_N, float)
        self.vD = np.asarray(self.vD_N, float)
        self.bRef = np.asarray(self.bRef_N, float)

        if self.q_RN is not None:
            self.qRN = np.asarray(self.q_RN, float)
            self.use_q = True
        else:
            self.qRN = None
            self.use_q = False

    def interp(self, arr: np.ndarray, t_abs: float) -> np.ndarray:
        """Linear interpolation for array samples over self.t."""
        t = self.t
        if t_abs <= t[0]:
            return arr[0]
        if t_abs >= t[-1]:
            return arr[-1]
        j = int(np.searchsorted(t, t_abs) - 1)
        f = (t_abs - t[j]) / (t[j + 1] - t[j])
        return (1.0 - f) * arr[j] + f * arr[j + 1]


# ╔═══════════════════════════════════════════════════════════════════════╗
# Deputy: one spacecraft worth of modules
# ╚═══════════════════════════════════════════════════════════════════════╝

class Deputy:
    def __init__(self, idx: int, bundle: RendezvousBundle, cfg: FleetAttitudeConfig):
        self.i = idx
        self.data = bundle
        self.cfg = cfg

        self.ref = messaging.AttRefMsg()   # to be written each step
        self.recs: Dict[str, Any] = {}     # recorders registered in FleetAttitudeSim

        self._build_modules()

    # ------------------------------------------------------------------ #
    # Module construction (essentially identical to Apophis mini-sim)    #
    # ------------------------------------------------------------------ #
    def _build_modules(self):
        cfg = self.cfg
        Ixx, Iyy, Izz = cfg.inertia
        theta = math.radians(cfg.init_y_deg)

        # small offset about body +Y to induce a transient
        R_off = np.array([[math.cos(theta), 0.0, math.sin(theta)],
                          [0.0,            1.0, 0.0],
                          [-math.sin(theta), 0.0, math.cos(theta)]])

        # spacecraft hub
        self.sc = spacecraft.Spacecraft()
        self.sc.ModelTag = f"Dep{self.i}"
        self.sc.hub.mHub = cfg.mass_kg
        self.sc.hub.IHubPntBc_B = [[Ixx, 0.0, 0.0],
                                   [0.0, Iyy, 0.0],
                                   [0.0, 0.0, Izz]]

        # initial attitude from q_RN if available, else TRIAD
        if self.data.use_q and cfg.prefer_qrn:
            C_NB0 = quat_to_dcm(self.data.qRN[0]).T
        else:
            C_NB0 = triad_from_boresight_and_tangent(
                self.data.bRef[0], self.data.vD[0]
            )
        C_NB0 = C_NB0 @ R_off
        sigma0 = quat_to_mrp(dcm_to_quat(C_NB0))
        self.sc.hub.sigma_BNInit = [[float(s)] for s in sigma0]
        self.sc.hub.r_CN_NInit = list(self.data.rD[0])
        self.sc.hub.v_CN_NInit = list(self.data.vD[0])

        # dynamics effector (torque input)
        self.torque = extForceTorque.ExtForceTorque()
        self.torque.ModelTag = f"torque_{self.i}"

        # SimpleNav
        self.nav = simpleNav.SimpleNav()
        self.nav.ModelTag = f"nav_{self.i}"
        # scState subscription wired in FleetAttitudeSim._build_tasks

        if cfg.nav_sig:
            sig = cfg.nav_sig
            # 6 groups: pos, vel, att, rate, sun, dv
            sig_vec = np.hstack([
                sig["pos"] * np.ones(3),
                sig["vel"] * np.ones(3),
                sig["att"] * np.ones(3),
                sig["rate"] * np.ones(3),
                sig["sun"] * np.ones(3),
                sig["dv"] * np.ones(3),
            ])
            walk = 3.0 * sig_vec
            P = np.diag(sig_vec ** 2) ** 0.5   # std dev on diagonal

            self.nav.walkBounds = walk
            self.nav.PMatrix = P

        # FSW
        self.err = attTrackingError.attTrackingError()
        self.err.ModelTag = f"err_{self.i}"

        self.ctl = mrpFeedback.mrpFeedback()
        self.ctl.ModelTag = f"ctl_{self.i}"
        self.ctl.K = cfg.k_sigma
        self.ctl.P = cfg.p_rate
        self.ctl.Ki = cfg.ki
        self.ctl.integralLimit = cfg.i_lim

        self.lpf = None
        if cfg.use_lpf:
            self.lpf = lowPassFilterTorqueCommand.lowPassFilterTorqueCommand()
            self.lpf.ModelTag = f"lpf_{self.i}"
            self.lpf.h = cfg.dt_dyn
            self.lpf.wc = 2.0 * math.pi * cfg.lpf_cut

    # ------------------------------------------------------------------ #
    # Reference builder: σ_RN(t), ω_RN_N(t)                              #
    # ------------------------------------------------------------------ #
    def ref_sigma_omega(self, t_abs: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Build reference attitude MRPs and rate at absolute time t_abs.

        If bundle has q_RN and cfg.prefer_qrn: finite-difference ω_RN_N.
        Otherwise, use TRIAD with zero reference rate.
        """
        if self.data.use_q and self.cfg.prefer_qrn:
            t_arr = self.data.t
            q_arr = self.data.qRN

            t0 = float(t_arr[0])
            t1 = float(t_arr[-1])

            q_prev = self.data.interp(q_arr, max(t0, t_abs - dt))
            q_curr = self.data.interp(q_arr, t_abs)
            q_next = self.data.interp(q_arr, min(t1, t_abs + dt))

            C_prev = quat_to_dcm(q_prev)
            C_curr = quat_to_dcm(q_curr)
            C_next = quat_to_dcm(q_next)

            Cdot = (C_next - C_prev) / max(2.0 * dt, 1e-6)
            S = Cdot @ C_curr.T
            omega = -0.5 * np.array([
                S[2, 1] - S[1, 2],
                S[0, 2] - S[2, 0],
                S[1, 0] - S[0, 1],
            ])
            sigma = quat_to_mrp(q_curr)
            return sigma, omega

        # TRIAD fallback
        rD = self.data.interp(self.data.rD, t_abs)
        vD = self.data.interp(self.data.vD, t_abs)
        bN = self.data.interp(self.data.bRef, t_abs)
        C = triad_from_boresight_and_tangent(bN, vD)
        sigma = quat_to_mrp(dcm_to_quat(C.T))
        omega = np.zeros(3)
        return sigma, omega


# ╔═══════════════════════════════════════════════════════════════════════╗
# FleetAttitudeSim: orchestrates deputies, tasks, and logging
# ╚═══════════════════════════════════════════════════════════════════════╝

class FleetAttitudeSim:
    """
    Fleet-level attitude simulator.

    Usage pattern (from scenario):

        bundles = [...]  # list of RendezvousBundle
        cfg = FleetAttitudeConfig(...)
        sim = FleetAttitudeSim(cfg, bundles)
        sim.run()
        sim.save_logs(log_dir)

    """

    def __init__(self, cfg: FleetAttitudeConfig, bundles: List[RendezvousBundle]):
        self.cfg = cfg
        self.bundles = bundles
        self.deps: List[Deputy] = [Deputy(i, b, cfg) for i, b in enumerate(bundles)]

        print(f"[FleetAttitudeSim] Found {len(self.bundles)} rendezvous bundles.")
        for b_i, b in enumerate(self.bundles):
            print(f"    deputy {b_i:02d}: N={len(b.t)} samples, "
                  f"t0={b.t[0]:.1f} s, t1={b.t[-1]:.1f} s")

        self._build_tasks()

    # ------------------------------------------------------------------ #
    # Task wiring (straight from Apophis mini-sim, minus NPZ discovery)  #
    # ------------------------------------------------------------------ #
    def _build_tasks(self):
        cfg = self.cfg

        self.dyn = sys_model_task.SysModelTask(int(macros.sec2nano(cfg.dt_dyn)))
        self.fsw = sys_model_task.SysModelTask(int(macros.sec2nano(cfg.dt_dyn)))
        self.sns = sys_model_task.SysModelTask(int(macros.sec2nano(cfg.dt_sens)))

        # shared vehicle-config message (inertia)
        Ixx, Iyy, Izz = cfg.inertia
        vehCfgPayload = messaging.VehicleConfigMsgPayload()
        vehCfgPayload.ISCPntB_B = [
            Ixx, 0.0, 0.0,
            0.0, Iyy, 0.0,
            0.0, 0.0, Izz,
        ]
        vehCfg = messaging.VehicleConfigMsg().write(vehCfgPayload)

        for d in self.deps:
            p = 100 * d.i  # priority offset

            # hook up nav and spacecraft
            d.sc.addDynamicEffector(d.torque)
            d.nav.scStateInMsg.subscribeTo(d.sc.scStateOutMsg)

            # FSW wiring
            d.err.attNavInMsg.subscribeTo(d.nav.attOutMsg)
            d.err.attRefInMsg.subscribeTo(d.ref)
            d.ctl.guidInMsg.subscribeTo(d.err.attGuidOutMsg)
            d.ctl.vehConfigInMsg.subscribeTo(vehCfg)

            if d.lpf:
                d.lpf.cmdTorqueInMsg.subscribeTo(d.ctl.cmdTorqueOutMsg)
                d.torque.cmdTorqueInMsg.subscribeTo(d.lpf.cmdTorqueOutMsg)
            else:
                d.torque.cmdTorqueInMsg.subscribeTo(d.ctl.cmdTorqueOutMsg)

            # add to tasks
            self.sns.AddNewObject(d.nav, 10 + p)

            self.fsw.AddNewObject(d.err, 20 + p)
            self.fsw.AddNewObject(d.ctl, 10 + p)
            if d.lpf:
                self.fsw.AddNewObject(d.lpf, 5 + p)

            self.dyn.AddNewObject(d.torque, 10 + p)
            self.dyn.AddNewObject(d.sc, 5 + p)

            # recorders
            rec = d.err.attGuidOutMsg.recorder()
            self.fsw.AddNewObject(rec, -1)
            d.recs["err"] = rec

            rec = d.ctl.cmdTorqueOutMsg.recorder()
            self.fsw.AddNewObject(rec, -1)
            d.recs["torq"] = rec

            if cfg.log_nav:
                rec = d.nav.attOutMsg.recorder()
                self.sns.AddNewObject(rec, -1)
                d.recs["navA"] = rec

                rec = d.nav.transOutMsg.recorder()
                self.sns.AddNewObject(rec, -1)
                d.recs["navT"] = rec

            if cfg.log_truth:
                rec = d.sc.scStateOutMsg.recorder()
                self.dyn.AddNewObject(rec, -1)
                d.recs["truth"] = rec

        # initialize tasks
        for task in (self.sns, self.fsw, self.dyn):
            task.SelfInitTaskList()
            task.ResetTaskList(0)

        # time span
        t0 = min(b.t[0] for b in self.bundles)
        t1 = max(b.t[-1] for b in self.bundles)
        self.t0_abs = float(t0)
        self.t_end = float(min(cfg.t_stop or t1, t1))

    # ------------------------------------------------------------------ #
    # Run loop                                                           #
    # ------------------------------------------------------------------ #
    def run(self):
        dt = self.cfg.dt_dyn
        N = int((self.t_end - self.t0_abs) / dt) + 1

        try:
            from tqdm import tqdm
            bar = tqdm(range(N), ncols=80, desc="FleetAtt")
        except Exception:
            bar = range(N)

        for k in bar:
            t_rel = k * dt
            t_abs = self.t0_abs + t_rel
            now_ns = int(macros.sec2nano(t_rel))

            # update reference messages for each deputy
            for d in self.deps:
                sigma_RN, omega_RN_N = d.ref_sigma_omega(t_abs, dt)

                ref_pl = messaging.AttRefMsgPayload()
                ref_pl.sigma_RN = list(sigma_RN)
                ref_pl.omega_RN_N = list(omega_RN_N)
                ref_pl.domega_RN_N = [0.0, 0.0, 0.0]
                d.ref.write(ref_pl)

            # run tasks at this time
            self.sns.ExecuteTaskList(now_ns)
            self.fsw.ExecuteTaskList(now_ns)
            self.dyn.ExecuteTaskList(now_ns)

        print("[FleetAttitudeSim] simulation complete.")

    # ------------------------------------------------------------------ #
    # Logging                                                            #
    # ------------------------------------------------------------------ #
    def save_logs(self, log_dir: Path):
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)

        for d in self.deps:
            fname = log_dir / f"dep{d.i:02d}_logs.npz"

            fields: Dict[str, Any] = dict(
                t_err=d.recs["err"].times() * macros.NANO2SEC,
                sigma_BR=safe_field(d.recs["err"], "sigma_BR", 3),
                omega_BR=safe_field(d.recs["err"], "omega_BR_B", 3),
                t_torque=d.recs["torq"].times() * macros.NANO2SEC,
                torque_B=safe_field(d.recs["torq"], "torqueRequestBody", 3),
            )

            if self.cfg.log_nav:
                fields.update(
                    t_navA=d.recs["navA"].times() * macros.NANO2SEC,
                    sigma_nav=safe_field(d.recs["navA"], "sigma_BN_B", 3),
                    omega_nav=safe_field(d.recs["navA"], "omega_BN_B", 3),
                    t_navT=d.recs["navT"].times() * macros.NANO2SEC,
                    r_nav=safe_field(d.recs["navT"], "r_BN_N", 3),
                    v_nav=safe_field(d.recs["navT"], "v_BN_N", 3),
                )

            if self.cfg.log_truth:
                fields.update(
                    t_true=d.recs["truth"].times() * macros.NANO2SEC,
                    r_true=safe_field(d.recs["truth"], "r_BN_N", 3),
                    v_true=safe_field(d.recs["truth"], "v_BN_N", 3),
                    sigma_true=safe_field(d.recs["truth"], "sigma_BN", 3),
                    omega_true=safe_field(d.recs["truth"], "omega_BN_B", 3),
                )

            np.savez_compressed(fname, **fields)
            print(f"[FleetAttitudeSim] saved {fname}")
