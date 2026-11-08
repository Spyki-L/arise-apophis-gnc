# Apophis/gnc/control/attitude_stack.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attitude_stack.py
=================

Generic 6-DOF attitude + SimpleNav + MRP-feedback stack.

This module does **no scenario-specific work**:
- It does not read NPZ files,
- It does not construct reference attitudes,
- It does not propagate orbital dynamics.

Instead it:
- Builds a spacecraft hub with given mass/inertia,
- Attaches an ExtForceTorque effector,
- Builds SimpleNav (optionally noisy) on top of the spacecraft,
- Wires attTrackingError + mrpFeedback (+ optional torque LPF),
- Creates three SysModelTasks (sensors, FSW, dynamics),
- Sets up recorders for error, nav, truth, and control torque.

A scenario script is expected to:
- Set sc.hub initial conditions (r_CN_NInit, v_CN_NInit, sigma_BNInit, ...),
- Provide an AttRefMsgPayload each step (σ_RN, ω_RN_N, etc.),
- Step the tasks and post-process the logs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import numpy.linalg as la

from Basilisk.architecture import messaging, sys_model_task
from Basilisk.simulation   import spacecraft, simpleNav, extForceTorque
from Basilisk.fswAlgorithms import (
    attTrackingError,
    mrpFeedback,
    lowPassFilterTorqueCommand,
)
from Basilisk.utilities import macros


# ────────────────────────── config & return types ────────────────────────────

@dataclass
class AttitudeStackConfig:
    # timing
    dt_dyn: float = 0.1
    dt_sens: float = 0.1

    # spacecraft properties
    mass_kg: float = 250.0
    inertia_diag: tuple[float, float, float] = (18.0, 15.0, 12.0)

    # SimpleNav noise (1-σ)
    use_noisy_nav: bool = True
    sigma_pos: float = 50.0      # [m]
    sigma_vel: float = 0.05      # [m/s]
    sigma_att: float = 5e-4      # [MRP]
    sigma_rate: float = 2e-5     # [rad/s]
    sigma_sun: float = 1e-3      # sun-line error
    sigma_dv: float = 0.0        # ΔV error (unused by many builds)

    # MRP feedback gains
    k_sigma: float = 0.08        # K gain in mrpFeedback
    p_rate:  float = 0.9         # P gain in mrpFeedback
    ki:      float = -1.0        # Ki < 0 disables integral action
    i_lim:   float = 0.01        # integral clamp [N·m]

    # torque LPF
    use_torque_lpf: bool = True
    lpf_cutoff_hz: float = 0.8   # [Hz]


@dataclass
class AttitudeStack:
    """Bundle of everything the scenario needs to drive the attitude loop."""
    # core modules
    sc: spacecraft.Spacecraft
    nav: simpleNav.SimpleNav
    extFT: extForceTorque.ExtForceTorque
    attErr: attTrackingError.attTrackingError
    ctl: mrpFeedback.mrpFeedback
    lpf: Optional[lowPassFilterTorqueCommand.lowPassFilterTorqueCommand]
    attRefMsg: messaging.AttRefMsg

    # tasks
    sensTask: sys_model_task.SysModelTask
    fswTask:  sys_model_task.SysModelTask
    dynTask:  sys_model_task.SysModelTask

    # recorders (for plotting / logging)
    logs: Dict[str, Any]


# ─────────────────────────── builder function ────────────────────────────────

def build_attitude_stack(cfg: AttitudeStackConfig) -> AttitudeStack:
    """
    Construct a fully wired attitude-control stack.

    The returned AttitudeStack has:
      - Spacecraft hub (mass + inertia, but **no** initial conditions set),
      - SimpleNav attached,
      - attTrackingError + mrpFeedback (+ optional LPF) wired into an
        ExtForceTorque effector,
      - Three SysModelTasks (sensors, FSW, dynamics),
      - Recorder objects already added to the tasks.

    The caller should:
      - Set sc.hub.*Init fields **after** this call,
      - Drive AttRefMsg each step,
      - Call task.SelfInitTaskList() and ResetTaskList(0) once before running,
      - Step the tasks in their main loop.
    """
    ns = lambda s: int(macros.sec2nano(s))

    # ── Spacecraft hub ───────────────────────────────────────────────────────
    sc = spacecraft.Spacecraft()
    sc.ModelTag = "Deputy"

    sc.hub.mHub = cfg.mass_kg
    Ixx, Iyy, Izz = cfg.inertia_diag
    sc.hub.IHubPntBc_B = [[Ixx, 0.0, 0.0],
                          [0.0, Iyy, 0.0],
                          [0.0, 0.0, Izz]]
    sc.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # body frame origin at hub

    # No initial attitude / position set here — scenario owns that.

    # ── Torque effector ─────────────────────────────────────────────────────
    extFT = extForceTorque.ExtForceTorque()
    extFT.ModelTag = "extFT"
    sc.addDynamicEffector(extFT)

    # ── SimpleNav with optional drift ───────────────────────────────────────
    nav = simpleNav.SimpleNav()
    nav.ModelTag = "simpleNav"
    nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)

    if cfg.use_noisy_nav:
        sig = cfg
        # 3σ walkBounds: [pos, vel, att, rate, sun, dv]
        walk_bounds = np.hstack([
            3 * sig.sigma_pos * np.ones(3),
            3 * sig.sigma_vel * np.ones(3),
            3 * sig.sigma_att * np.ones(3),
            3 * sig.sigma_rate * np.ones(3),
            3 * sig.sigma_sun * np.ones(3),
            3 * sig.sigma_dv  * np.ones(3),
        ])
        sigma_vec = np.hstack([
            sig.sigma_pos * np.ones(3),
            sig.sigma_vel * np.ones(3),
            sig.sigma_att * np.ones(3),
            sig.sigma_rate * np.ones(3),
            sig.sigma_sun * np.ones(3),
            sig.sigma_dv  * np.ones(3),
        ])
        PMatrix = np.diag(sigma_vec**2)**0.5  # diagonal √P
        nav.walkBounds = walk_bounds
        nav.PMatrix    = PMatrix

        # Some Basilisk builds expose these toggles:
        for attr in ("crossTrans", "crossAtt"):
            if hasattr(nav, attr):
                setattr(nav, attr, True)

    # ── FSW: tracking error + MRP feedback (+ LPF) ──────────────────────────
    attErr = attTrackingError.attTrackingError()
    attErr.ModelTag = "attError"

    ctl = mrpFeedback.mrpFeedback()
    ctl.ModelTag = "mrpFB"
    ctl.K  = cfg.k_sigma
    ctl.P  = cfg.p_rate
    ctl.Ki = cfg.ki
    ctl.integralLimit = cfg.i_lim

    lpf: Optional[lowPassFilterTorqueCommand.lowPassFilterTorqueCommand] = None
    if cfg.use_torque_lpf:
        lpf = lowPassFilterTorqueCommand.lowPassFilterTorqueCommand()
        lpf.ModelTag = "lpfTorque"
        lpf.h  = float(cfg.dt_dyn)
        lpf.wc = float(2 * np.pi * cfg.lpf_cutoff_hz)

    # Static vehicle config message
    vehCfgPayload = messaging.VehicleConfigMsgPayload()
    vehCfgPayload.ISCPntB_B = [Ixx, 0.0, 0.0,
                               0.0, Iyy, 0.0,
                               0.0, 0.0, Izz]
    vehCfgMsg = messaging.VehicleConfigMsg().write(vehCfgPayload)

    # Attitude reference message (scenario writes Payloads to this)
    attRefMsg = messaging.AttRefMsg()

    # Wiring
    attErr.attNavInMsg.subscribeTo(nav.attOutMsg)
    attErr.attRefInMsg.subscribeTo(attRefMsg)

    ctl.guidInMsg.subscribeTo(attErr.attGuidOutMsg)
    ctl.vehConfigInMsg.subscribeTo(vehCfgMsg)

    if lpf is not None:
        lpf.cmdTorqueInMsg.subscribeTo(ctl.cmdTorqueOutMsg)
        extFT.cmdTorqueInMsg.subscribeTo(lpf.cmdTorqueOutMsg)
    else:
        extFT.cmdTorqueInMsg.subscribeTo(ctl.cmdTorqueOutMsg)

    # ── Tasks ───────────────────────────────────────────────────────────────
    sensTask = sys_model_task.SysModelTask(ns(cfg.dt_sens))
    fswTask  = sys_model_task.SysModelTask(ns(cfg.dt_dyn))
    dynTask  = sys_model_task.SysModelTask(ns(cfg.dt_dyn))

    sensTask.AddNewObject(nav,  10)
    fswTask.AddNewObject(attErr, 20)
    fswTask.AddNewObject(ctl,   10)
    if lpf is not None:
        fswTask.AddNewObject(lpf, 5)
    dynTask.AddNewObject(extFT, 10)
    dynTask.AddNewObject(sc,    5)

    # ── Recorders ───────────────────────────────────────────────────────────
    errLog  = attErr.attGuidOutMsg.recorder()
    navALog = nav.attOutMsg      .recorder()
    navTLog = nav.transOutMsg    .recorder()
    scLog   = sc.scStateOutMsg   .recorder()
    ctlLog  = ctl.cmdTorqueOutMsg.recorder()

    fswTask.AddNewObject(errLog,  -1)
    sensTask.AddNewObject(navALog,-1)
    sensTask.AddNewObject(navTLog,-1)
    dynTask.AddNewObject(scLog,   -1)
    fswTask.AddNewObject(ctlLog,  -1)

    logs = dict(err=errLog, navA=navALog, navT=navTLog, truth=scLog, torque=ctlLog)

    # Initialize tasks (caller could also do this, but it’s convenient here)
    for task in (sensTask, fswTask, dynTask):
        task.SelfInitTaskList()
        task.ResetTaskList(0)

    return AttitudeStack(
        sc=sc,
        nav=nav,
        extFT=extFT,
        attErr=attErr,
        ctl=ctl,
        lpf=lpf,
        attRefMsg=attRefMsg,
        sensTask=sensTask,
        fswTask=fswTask,
        dynTask=dynTask,
        logs=logs,
    )

