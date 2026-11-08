#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
se3_ph_layer.py
===============

Generic SE(3) port–Hamiltonian control layer for Basilisk.

Provides
--------
- Thruster      : simple thruster geometry + performance.
- VehicleConfig : mass / inertia / gains / thruster layout.
- SE3Layer      : port–Hamiltonian SE(3) tracker with:

    * attitude + translation damping and stiffness,
    * integral action with anti-windup,
    * optional passivity observer / controller (PO/PC),
    * one-step latency prediction,
    * latency buffer on wrench application,
    * bounded non-linear least-squares (BNLS) allocator from
      (τ_residual, f_des) → thruster duty cycles,
    * rich internal logging for analysis and plotting.

The layer is **mission-agnostic**: it does not know about Apophis, ESPA,
RAFTI, etc. You provide:

    R_d(t)  : desired attitude DCM (N→R/B),
    p_d_N(t): desired position (or “anchor” point) in inertial N,
    los_N   : optional LOS vector (for boresight metrics only).

Typical usage
-------------
1. Build a VehicleConfig with mass, inertia, and thruster layout.
2. Construct SE3Layer with dt, engage_time, latency_s, and gain knobs.
3. Each dynamics step, call:

       ctrl.step(t,
                 attNav = nav.attOutMsg.read(),
                 transNav = nav.transOutMsg.read(),
                 R_d = desired_dcm,
                 p_d_N = desired_pos_N,
                 los_N = optional_los_vector,
                 cam_axis = "z")

   The controller will write CmdTorqueBodyMsg and CmdForceBodyMsg into
   `ctrl.ctrlTorqueMsg` and `ctrl.ctrlForceMsg`, which you subscribe to
   from `ExtForceTorque`.

4. After the run, inspect logs such as:

       ctrl.t_list
       ctrl.Hd_list
       ctrl.tau_d_hist, ctrl.torque_apply_hist
       ctrl.boresight_err_hist, ctrl.roll_about_los_err_hist
       ctrl.dv_cum, ctrl.mp_cum
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import math
import numpy as np
import numpy.linalg as la
from Basilisk.architecture import messaging


# ───────────────────────── basic math helpers ──────────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z,   y],
                     [  z,  0.0, -x],
                     [ -y,  x,  0.0]], float)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = la.norm(v)
    return v/n if n > eps else np.array([1.0, 0.0, 0.0], float)

def mrp_to_dcm(sigma: np.ndarray) -> np.ndarray:
    """Modified Rodrigues Parameters → DCM (N→B)."""
    s2 = float(np.dot(sigma, sigma))
    I3 = np.eye(3)
    if s2 < 1e-16:
        return I3.copy()
    S = skew(sigma)
    c = 1.0 + s2
    return I3 + (8*S@S - 4*(1.0 - s2)*S) / (c*c)

def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    """DCM → unit quaternion [q0,q1,q2,q3] (scalar-first)."""
    tr = float(np.trace(C))
    if tr > 0.0:
        S  = math.sqrt(tr + 1.0) * 2.0
        q0 = 0.25 * S
        q1 = (C[2,1] - C[1,2]) / S
        q2 = (C[0,2] - C[2,0]) / S
        q3 = (C[1,0] - C[0,1]) / S
    else:
        d = np.array([C[0,0], C[1,1], C[2,2]])
        i = int(np.argmax(d))
        if i == 0:
            S  = math.sqrt(1.0 + C[0,0] - C[1,1] - C[2,2]) * 2.0
            q0 = (C[2,1] - C[1,2]) / S
            q1 = 0.25 * S
            q2 = (C[0,1] + C[1,0]) / S
            q3 = (C[0,2] + C[2,0]) / S
        elif i == 1:
            S  = math.sqrt(1.0 + C[1,1] - C[0,0] - C[2,2]) * 2.0
            q0 = (C[0,2] - C[2,0]) / S
            q1 = (C[0,1] + C[1,0]) / S
            q2 = 0.25 * S
            q3 = (C[1,2] + C[2,1]) / S
        else:
            S  = math.sqrt(1.0 + C[2,2] - C[0,0] - C[1,1]) * 2.0
            q0 = (C[1,0] - C[0,1]) / S
            q1 = (C[0,2] + C[2,0]) / S
            q2 = (C[1,2] + C[2,1]) / S
            q3 = 0.25 * S
    q = np.array([q0, q1, q2, q3], float)
    return q / la.norm(q)

def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], float)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], float)

def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    """Unit quaternion → MRPs."""
    q0, qv = q[0], q[1:]
    d = 1.0 + q0
    if d < 1e-12:
        d = 1.0 - q0
        return -qv / max(d, 1e-12)
    return qv / d

def dcm_to_euler_zyx(C: np.ndarray) -> np.ndarray:
    """Return [roll, pitch, yaw] (XYZ / 3-2-1) from DCM."""
    psi   = math.atan2(C[1,0], C[0,0])  # yaw (Z)
    theta = math.asin(-C[2,0])          # pitch (Y)
    phi   = math.atan2(C[2,1], C[2,2])  # roll (X)
    return np.array([phi, theta, psi])

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + math.pi) % (2.0*math.pi) - math.pi

def shortest_arc_error(R: np.ndarray, R_d: np.ndarray) -> np.ndarray:
    """Quaternion vector-part error giving the shortest rotation from R→R_d."""
    q    = dcm_to_quat(R)
    q_d  = dcm_to_quat(R_d)
    q_e  = quat_mul(q_d, quat_conj(q))
    if q_e[0] < 0.0:
        q_e = -q_e
    return q_e[1:]

def so3_log(R: np.ndarray) -> np.ndarray:
    """Matrix log on SO(3) → axis-angle vector."""
    c  = clamp((np.trace(R) - 1.0)/2.0, -1.0, 1.0)
    th = math.acos(c)
    if th < 1e-10:
        return np.zeros(3)
    return (th/(2.0*math.sin(th))) * np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ])

def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Axis-angle vector → SO(3) rotation matrix."""
    th = la.norm(phi)
    if th < 1e-10:
        return np.eye(3) + skew(phi)
    k = phi/th; K = skew(k)
    return np.eye(3) + math.sin(th)*K + (1.0 - math.cos(th))*(K@K)


# ───────────────────────── config dataclasses ───────────────────────────────

@dataclass
class Thruster:
    r_B: np.ndarray
    a_B: np.ndarray
    T: float
    Isp: float
    name: str = "T"

@dataclass
class VehicleConfig:
    name: str
    mass_kg: float
    inertia_B: np.ndarray
    sigma0_BN: np.ndarray
    omega0_B:  np.ndarray
    r0_N:      np.ndarray
    v0_N:      np.ndarray
    # SE(3) tracker gains (ζ, ωn)
    zeta_att:   float
    wn_att:     float
    zeta_vel:   float
    wn_vel:     float
    # allocation & limits
    rw_tau_max: float
    w_force:    float
    w_torque:   float
    reg_lambda: float
    thrusters:  List[Thruster]


# ───────────────────────── SE(3) pH layer ───────────────────────────────────

class SE3Layer:
    """
    Port–Hamiltonian SE(3) controller with wheels + jets and PO/PC.

    The main method is:

        step(t, attNav, transNav, R_d, p_d_N, los_N=None, cam_axis="z")

    which reads nav states, computes a desired wrench, allocates it, and
    writes torque/force messages.
    """

    def __init__(self, veh: VehicleConfig, dt: float, engage_time: float,
                 latency_s: float,
                 enable_po_pc: bool = True,
                 use_latency_predict: bool = True,
                 cmd_smooth_tau_s: float = 0.06,
                 cmd_smooth_for_s: float = 0.08,
                 shape_ramp_s: float = 6.0,
                 tau_deadband_nm: float = 0.0,
                 f_deadband_n: float = 0.0):

        self.veh = veh
        self.dt  = dt
        self.engage_time = engage_time

        J = veh.inertia_B
        m = veh.mass_kg

        # base gains
        self.K_R_base = J * (veh.wn_att**2)
        self.D_R      = 2.0 * veh.zeta_att * veh.wn_att * J
        self.K_p      = (m * (veh.wn_vel**2)) * np.eye(3)
        self.D_v      = (2.0 * veh.zeta_vel * veh.wn_vel * m) * np.eye(3)

        # integral gains & limits (anti-windup)
        self.KI_R   = 0.015 * np.diag(np.diag(self.K_R_base))
        self.KI_p   = 0.06  * self.K_p
        self.int_eR = np.zeros(3)
        self.int_ep = np.zeros(3)
        self.intR_max = 4.0*np.ones(3)
        self.intp_max = 0.8*np.ones(3)

        # shaping (ramp)
        self.shape_ramp_s = float(shape_ramp_s)

        # port messages
        self.ctrlTorqueMsg = messaging.CmdTorqueBodyMsg()
        self.ctrlForceMsg  = messaging.CmdForceBodyMsg()

        # latency
        self.L = int(round(max(0.0, latency_s)/dt))
        self.use_latency_predict = use_latency_predict
        self.enable_po_pc        = enable_po_pc

        # command smoothing
        self.alpha_tau = dt / (cmd_smooth_tau_s + dt)
        self.alpha_for = dt / (cmd_smooth_for_s + dt)
        self.tau_smooth = np.zeros(3)
        self.f_smooth   = np.zeros(3)

        # deadbands
        self.tau_dead = float(tau_deadband_nm)
        self.f_dead   = float(f_deadband_n)

        # thruster mapping
        self.B, self.Ts, self.Isps = self._build_wrench_map(veh.thrusters)
        self.delta_prev: Optional[np.ndarray] = None

        # latency buffers
        self.buf_tau: List[np.ndarray] = []
        self.buf_for: List[np.ndarray] = []
        self.omega_prev = np.zeros(3)
        self.v_prev     = np.zeros(3)

        # PO tank
        self.E_po = 0.0

        # logs
        self.t_list: List[float]   = []
        self.Hd_list: List[float]  = []
        self.tau_d_hist: List[np.ndarray] = []
        self.f_d_hist:   List[np.ndarray] = []
        self.torque_cmd_hist:  List[np.ndarray] = []
        self.force_cmd_hist:   List[np.ndarray] = []
        self.torque_apply_hist:List[np.ndarray] = []
        self.force_apply_hist: List[np.ndarray] = []
        self.omega_hist:  List[np.ndarray] = []
        self.vB_hist:     List[np.ndarray] = []
        self.sigma_hist:  List[np.ndarray] = []
        self.euler_hist:  List[np.ndarray] = []
        self.euler_des_hist: List[np.ndarray] = []
        self.thrust_total_hist: List[float] = []
        self.po_scale_hist:  List[float] = []
        self.E_po_hist:     List[float] = []
        self.boresight_err_hist: List[float] = []
        self.roll_about_los_err_hist: List[float] = []

        # monotonicity counters
        self._Hd_prev: Optional[float] = None
        self.energy_total   = 0
        self.energy_noninc  = 0

        # Δv / prop accounting
        self.dv_cum = 0.0
        self.mp_cum = 0.0

        # previous desired attitude
        self.Rd_prev: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------

    def _build_wrench_map(self, thr: List[Thruster]):
        if not thr:
            return np.zeros((6,0)), np.zeros(0), np.zeros(0)
        cols = []
        Ts   = []
        Isps = []
        for th in thr:
            r = np.asarray(th.r_B).reshape(3)
            a = unit(np.asarray(th.a_B).reshape(3))
            cols.append(np.hstack((np.cross(r, th.T*a), th.T*a)))
            Ts.append(th.T)
            Isps.append(th.Isp)
        return np.column_stack(cols), np.array(Ts, float), np.array(Isps, float)

    def _bnls(self, B, u, delta_prev, w_tau, w_for, lam, lb, ub):
        n = B.shape[1]
        if delta_prev is None or delta_prev.shape[0] != n:
            delta_prev = np.zeros(n)
        W    = np.diag([w_tau, w_tau, w_tau, w_for, w_for, w_for])
        BtWB = B.T @ W @ B + lam * np.eye(n)
        BtWu = B.T @ W @ u + lam * delta_prev
        delta = np.clip(la.solve(BtWB, BtWu), lb, ub)
        # simple active-set refinement
        for _ in range(6):
            low  = (delta <= lb + 1e-9)
            high = (delta >= ub - 1e-9)
            free = ~(low | high)
            if not np.any(free):
                break
            Bf = B[:, free]
            K  = la.pinv(Bf.T @ W @ Bf + lam * np.eye(np.count_nonzero(free)))
            rhs = Bf.T @ W @ (u
                              - B[:, low]  @ np.full(np.count_nonzero(low),  lb)
                              - B[:, high] @ np.full(np.count_nonzero(high), ub)
                             ) + lam * delta_prev[free]
            delta_f = K @ rhs
            delta   = delta.copy()
            delta[free] = delta_f
            delta = np.clip(delta, lb, ub)
        return delta

    def _shape_gain(self, t: float) -> float:
        if t <= self.engage_time:
            return 0.0
        return clamp((t - self.engage_time)/max(1e-6, self.shape_ramp_s), 0.0, 1.0)

    def _K_R_scheduled(self, e_R: np.ndarray) -> np.ndarray:
        """Gain scheduling in attitude stiffness based on ‖e_R‖."""
        k = la.norm(e_R)
        scale = 1.0 + 1.8 * math.tanh(2.5 * k)   # up to ~2.8×
        return scale * self.K_R_base

    def _desired_rate_body(self, R: np.ndarray, R_d: np.ndarray) -> np.ndarray:
        """SO(3) log/exp rate generation in the body frame."""
        if self.Rd_prev is None:
            self.Rd_prev = R_d.copy()
            return np.zeros(3)
        Delta      = self.Rd_prev.T @ R_d
        phi        = so3_log(Delta)
        omega_d_Bd = phi / self.dt
        C_BBd      = R.T @ R_d
        self.Rd_prev = R_d.copy()
        return C_BBd @ omega_d_Bd

    # ---------------------------------------------------------------------
    # main step
    # ---------------------------------------------------------------------

    def step(self,
             t: float,
             attNav: messaging.NavAttMsgPayload,
             transNav: messaging.NavTransMsgPayload,
             R_d: np.ndarray,
             p_d_N: np.ndarray,
             los_N: Optional[np.ndarray] = None,
             cam_axis: str = "z") -> None:
        """
        One SE(3) control step.

        Parameters
        ----------
        t       : float
            Current simulation time [s].
        attNav  : NavAttMsgPayload
            Attitude nav message (σ_BN, ω_BN_B).
        transNav: NavTransMsgPayload
            Translation nav message (r_BN_N, v_BN_N).
        R_d     : (3,3) ndarray
            Desired attitude DCM (N→R/B).
        p_d_N   : (3,) ndarray
            Desired position anchor (e.g., separation point) in N.
        los_N   : optional (3,)
            LOS vector in N, used only for boresight/roll metrics.
        cam_axis: {"x","y","z"}
            Which body axis is treated as the camera boresight for metrics.
        """
        dt = self.dt
        J  = self.veh.inertia_B
        m  = self.veh.mass_kg

        sigma = np.array(attNav.sigma_BN)
        R     = mrp_to_dcm(sigma)
        omega = np.array(attNav.omega_BN_B)
        r_N   = np.array(getattr(transNav, 'r_BN_N', [0.0,0.0,0.0]), float)
        v_N   = np.array(getattr(transNav, 'v_BN_N', [0.0,0.0,0.0]), float)
        v_B   = R @ v_N

        omega_d_B = self._desired_rate_body(R, R_d)

        # latency prediction
        if self.use_latency_predict:
            omega_p = omega + (omega - self.omega_prev)
            vB_p    = v_B  + (v_B  - self.v_prev)
            vN_p    = v_N  + (v_N  - R.T @ self.v_prev)
        else:
            omega_p = omega.copy()
            vB_p    = v_B.copy()
            vN_p    = v_N.copy()

        # SE(3) errors
        e_R = shortest_arc_error(R, R_d)
        e_w = omega_p - omega_d_B
        e_p = r_N - p_d_N
        e_v = vN_p

        # gains and ramp
        g   = self._shape_gain(t)
        K_R = self._K_R_scheduled(e_R)

        # preliminary wrench
        tau_pre = - (self.D_R @ e_w) - g*(K_R @ e_R) - (self.KI_R @ self.int_eR)
        F_N_pre = - (self.D_v @ e_v) - g*(self.K_p @ e_p) - (self.KI_p @ self.int_ep)
        f_pre   = R @ F_N_pre

        # smoothing
        self.tau_smooth = (1.0 - self.alpha_tau)*self.tau_smooth + self.alpha_tau * tau_pre
        self.f_smooth   = (1.0 - self.alpha_for)*self.f_smooth   + self.alpha_for  * f_pre
        tau_des = self.tau_smooth
        f_des   = self.f_smooth

        # PO/PC
        s_scale  = 1.0
        allow_int = True
        if self.enable_po_pc:
            P   = float(tau_des @ omega + f_des @ v_B)
            dE  = P * dt
            if dE > 0.0 and self.E_po + dE > 0.0:
                s_scale = max(0.0, min(1.0, -self.E_po/(dE + 1e-12)))
            if s_scale < 0.999:
                allow_int = False
            tau_des *= s_scale
            f_des   *= s_scale
            P_eff   = float(tau_des @ omega + f_des @ v_B)
            self.E_po += P_eff * dt
        self.po_scale_hist.append(s_scale)
        self.E_po_hist.append(self.E_po)

        # wheel + jets split
        tau_rw = tau_des.copy()
        tn     = la.norm(tau_rw)
        if tn > self.veh.rw_tau_max > 0.0:
            tau_rw *= (self.veh.rw_tau_max / tn)
        tau_res = tau_des - tau_rw

        # allocation
        if la.norm(tau_res) < self.tau_dead and la.norm(f_des) < self.f_dead:
            delta   = np.zeros(0)
            tau_thr = np.zeros(3)
            f_thr   = np.zeros(3)
        else:
            if self.B.size > 0:
                u     = np.hstack((tau_res, f_des))
                delta = self._bnls(self.B, u, self.delta_prev,
                                   self.veh.w_torque, self.veh.w_force,
                                   self.veh.reg_lambda, 0.0, 1.0)
                self.delta_prev = delta.copy()
                uf     = self.B @ delta
                tau_thr = uf[:3]
                f_thr   = uf[3:]
            else:
                delta   = np.zeros(0)
                tau_thr = np.zeros(3)
                f_thr   = np.zeros(3)

        tau_real = tau_rw + tau_thr
        f_real   = f_thr

        # latency buffer
        self.buf_tau.append(tau_real)
        self.buf_for.append(f_real)
        if len(self.buf_tau) > self.L:
            tau_apply = self.buf_tau.pop(0)
            f_apply   = self.buf_for.pop(0)
        else:
            tau_apply = np.zeros(3)
            f_apply   = np.zeros(3)

        # publish
        t_pl = messaging.CmdTorqueBodyMsgPayload()
        t_pl.torqueRequestBody = list(tau_apply)
        f_pl = messaging.CmdForceBodyMsgPayload()
        f_pl.forceRequestBody  = list(f_apply)
        self.ctrlTorqueMsg.write(t_pl)
        self.ctrlForceMsg.write(f_pl)

        # integrals
        if allow_int:
            self.int_eR = np.clip(self.int_eR + e_R*dt, -self.intR_max, self.intR_max)
            self.int_ep = np.clip(self.int_ep + e_p*dt, -self.intp_max, self.intp_max)

        # energies
        V_R = 0.5 * g * float(e_R.T @ (K_R @ e_R))
        V_p = 0.5 * g * float(e_p.T @ (self.K_p @ e_p))
        H   = 0.5*float(omega.T@(J@omega)) + 0.5*float(m*(v_B@v_B)) + V_R + V_p

        if t >= self.engage_time:
            self.energy_total += 1
            if self._Hd_prev is not None and H <= self._Hd_prev + 1e-12:
                self.energy_noninc += 1
            self._Hd_prev = H

        # pointing metrics (optional)
        if los_N is not None:
            u = unit(los_N)
            if cam_axis.lower() == "z":
                zB_N = (R.T)[:,2]
                xB_N = (R.T)[:,0]
                xBd_N= (R_d.T)[:,0]
            elif cam_axis.lower() == "x":
                zB_N = (R.T)[:,0]
                xB_N = (R.T)[:,1]
                xBd_N= (R_d.T)[:,1]
            else:
                zB_N = (R.T)[:,1]
                xB_N = (R.T)[:,2]
                xBd_N= (R_d.T)[:,2]

            boresight_deg = math.degrees(
                math.acos(clamp(float(np.dot(zB_N, u)), -1.0, 1.0))
            )

            P = np.eye(3) - np.outer(u,u)
            a = unit(P @ xB_N)
            b = unit(P @ xBd_N)
            roll_err_deg = math.degrees(
                math.atan2(la.norm(np.cross(a,b)), float(np.dot(a,b)))
            )

            self.boresight_err_hist.append(boresight_deg)
            self.roll_about_los_err_hist.append(roll_err_deg)
        else:
            self.boresight_err_hist.append(float("nan"))
            self.roll_about_los_err_hist.append(float("nan"))

        # logs
        self.t_list.append(t)
        self.Hd_list.append(H)
        self.tau_d_hist.append(tau_des.copy())
        self.f_d_hist.append(f_des.copy())
        self.torque_cmd_hist.append(tau_real.copy())
        self.force_cmd_hist.append(f_real.copy())
        self.torque_apply_hist.append(tau_apply.copy())
        self.force_apply_hist.append(f_apply.copy())
        self.omega_hist.append(omega.copy())
        self.vB_hist.append(v_B.copy())
        self.sigma_hist.append(sigma.copy())
        self.thrust_total_hist.append(
            float(np.sum(self.Ts * delta)) if (self.B.size and delta.size) else 0.0
        )

        # Euler logs (wrapped)
        rpy   = dcm_to_euler_zyx(R)
        rpy_d = dcm_to_euler_zyx(R_d)
        diff  = wrap_to_pi(rpy - rpy_d)
        rpy_wrapped = rpy_d + diff
        self.euler_hist.append(rpy_wrapped)
        self.euler_des_hist.append(rpy_d)

        # Δv / prop accounting
        if self.B.size and delta.size > 0:
            impulse = float(np.sum(self.Ts * delta) * dt)
            self.dv_cum += impulse / m
            if impulse > 0.0:
                self.mp_cum += impulse / (
                    np.average(self.Isps, weights=self.Ts*delta) * 9.80665
                )

        self.omega_prev = omega.copy()
        self.v_prev     = v_B.copy()
