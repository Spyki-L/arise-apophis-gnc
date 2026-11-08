#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_scenario_espa42_rafti_xb6_se3_lookat.py
===============================================

Purpose
-------
Basilisk scenario for a **single** Blue Canyon XB6 6U CubeSat mounted on an
ESPA 42-inch ring and separated via a RAFTI-style half-sine push. After
separation, the CubeSat’s camera boresight (default +Z_B) continuously
“looks back” at the ESPA port while a geometric SE(3) port–Hamiltonian (pH)
controller:

- Tracks a continuous look-at attitude reference derived from the carrier’s
  current port pose,
- Uses ω_d from SO(3) log/exp (feed-forward rate generation),
- Applies integral action with anti-windup,
- Includes a passivity observer/controller (PO/PC),
- Handles command latency via prediction and buffering,
- Allocates torque/force between reaction wheels and a small 12-jet thruster
  array via BNLS.

The script **only shows plots** using Matplotlib and writes no files.

Scenario setup
--------------
- Geometry:
  * ESPA ring: 42" diameter ⇒ radius ≈ 0.5334 m.
  * Single port at angle 0 on the ring, with:
      - `espa_port_r_B` : port position in carrier body frame,
      - `espa_port_dir_B` : outward port normal (used as RAFTI push direction).
  * XB6 deputy initially aligned with that port:
      - +Y_B along the port normal,
      - +Z_B roughly along +Z_carrier,
      - +X_B completing a right-handed triad.

- Carrier:
  * Mass ≈ 650 kg, inertia diag([1200, 1000, 800]) kg·m².
  * 4 MR-22N-like thrusters (22 N, Isp ≈ 235 s) pointing −Y_B.
  * Controlled by a “mild” SE(3) pH layer providing damping in attitude and
    translation so the carrier behaves as a stable base during separation.

- Deputy (XB6 6U):
  * Mass ≈ 12 kg, inertia diag([0.025, 0.020, 0.018]) kg·m².
  * 12-jet layout:
      - ±X, ±Y, ±Z jets at small offsets (L ≈ 0.065 m),
      - Provides both torque and small net force authority.
  * Controlled by a higher-bandwidth SE(3) pH controller with a small
    look-at shaping term.

RAFTI separation
----------------
- RAFTI push is modeled as a half-sine force:

      F(t) = F0 sin(π (t − t0) / Tpush),  t ∈ [t0, t0 + Tpush]

  with:

  * Base time: t0 ≈ 10 s,
  * Duration:  Tpush = 0.30 s,
  * Δv_cmd sampled from N(0.018, 0.005²) m/s, clipped to [0.004, 0.025] m/s,
  * F0 = m_dep · Δv_cmd · π / (2·Tpush).

- RAFTISeparation:
  * Computes contact normals and lever arms for both carrier and deputy.
  * Applies equal-and-opposite forces:
      - Carrier: +F_N expressed in carrier body,
      - Deputy: −F_N expressed in deputy body.
  * Applies corresponding torques τ = r × F for each body:
      - Carrier: τ_car,
      - Deputy: τ_dep.

SE(3) pH controller (SE3Layer)
------------------------------
For both carrier and deputy, SE3Layer implements:

- Gains:
    * Rotational:
          K_R_base = J ω_n_att²,
          D_R      = 2 ζ_att ω_n_att J.
    * Translational:
          K_p = m ω_n_vel² I,
          D_v = 2 ζ_vel ω_n_vel m I.
    * Integral:
          KI_R, KI_p with component-wise clamping (anti-windup).

- Look-at shaping (deputy only):
    * For t ≥ engage_time:
        - Build a desired attitude R_d via `look_at_continuous`:
            + Align the camera axis (default +Z_B) with LOS from deputy to
              the current port position in the inertial frame.
            + Choose roll about LOS to stay close to the previous attitude.
        - Attitude error e_R computed via shortest-arc quaternion difference.
        - Position error e_p = r − p_ref:
            + p_ref freezes at the deputy’s position at the end of the RAFTI
              push, so translation is damped around the separation point.

- Latency prediction:
    * One-step prediction of ω and v_B to counteract actuation delay.

- PO/PC:
    * Monitors power P = τ · ω + f · v_B and integrates a passivity tank
      E_po.
    * Scales the commanded wrench if E_po would become positive, and disables
      integral updates while scaling is active.

- Allocation:
    * Splits torque into wheel + jet parts with a per-axis wheel limit:
          ‖τ_rw‖ ≤ rw_tau_max.
    * Uses a BNLS allocator (`_bnls`) with bounds 0 ≤ δ ≤ 1 to map residual
      torque + desired force into thruster duty cycles.
    * Buffers the resulting wrench and applies it after L steps of latency.

Logging and metrics
-------------------
SE3Layer logs:

- Time-series:
    * `t_list`              : time stamps [s],
    * `Hd_list`             : shaped SE(3) energy,
    * `tau_d_hist`, `f_d_hist`   : desired torques/forces,
    * `torque_cmd_hist`, `force_cmd_hist`,
    * `torque_apply_hist`, `force_apply_hist`,
    * `omega_hist`, `vB_hist`, `sigma_hist`,
    * `euler_hist` (wrapped) and `euler_des_hist` (desired),
    * `thrust_total_hist`,
    * `boresight_err_hist`, `roll_about_los_err_hist`,
    * `po_scale_hist`, `E_po_hist`.

- Scalar metrics:
    * Energy non-increasing fraction after engage_time,
    * Cumulative control Δv (`dv_cum`),
    * Cumulative propellant mass (`mp_cum`).

At the end of the run, `_report` prints:

- For carrier and XB6:
    * H_d non-increasing fraction,
    * Δv [m/s],
    * Propellant mass [g].

Plots (display only)
--------------------
The `_plots` method constructs and shows:

1. Δv comparison:
   - Bar chart comparing RAFTI Δv_cmd vs control Δv for the XB6 deputy.

2. Energy & history stacks:
   - H_SE3,d vs time with engage_time marker.
   - Norms of τ_d, τ_cmd, τ_applied vs time.
   - Norms of f_d, f_cmd, f_applied vs time.
   - Total thruster sum ∑T·δ.
   - ω components (ω_x, ω_y, ω_z) vs time.
   - MRP components (σ₁, σ₂, σ₃) vs time.

3. Euler tracking (wrapped ZYX):
   - roll, pitch, yaw vs time:
       - Desired (dashed) vs achieved (wrapped, solid),
       - Shows how the camera tracks the ESPA port reference frame.

4. Pointing quality:
   - Boresight miss angle [deg] vs time, with a 1° reference line.
   - Roll-about-LOS error [deg] vs time, with a 2° reference line.

5. Relative torque/force tracking error:
   - ‖τ_d − τ_cmd‖/‖τ_d‖ and ‖τ_d − τ_appl‖/‖τ_d‖ vs time.
   - ‖f_d − f_cmd‖/‖f_d‖ and ‖f_d − f_appl‖/‖f_d‖ vs time.

All plots are shown with `plt.show()` and **no files are written**.

Usage
-----
Run in a Basilisk-enabled Python environment:

    python basilik_scenario_espa42_rafti_xb6_se3_lookat.py

The script will:

- Build the ESPA 42" carrier + XB6 deputy with a RAFTI-style separation.
- Run SE(3) pH control until the end of the configured horizon.
- Print carrier and XB6 energy/Δv metrics.
- Display the Δv bars, energy histories, attitude tracking, pointing
  quality, and tracking-error plots.
"""


from __future__ import annotations
import math, dataclasses as dc, traceback
from typing import List, Optional
import numpy as np
import numpy.linalg as la

# Basilisk
from Basilisk.architecture import messaging, sys_model_task
from Basilisk.simulation   import spacecraft, simpleNav, extForceTorque
from Basilisk.utilities    import macros

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

import matplotlib.pyplot as plt

# ───────────────────────── math helpers ─────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    x,y,z=v; return np.array([[0,-z,y],[z,0,-x],[-y,x,0]], float)

def unit(v: np.ndarray, eps=1e-12) -> np.ndarray:
    n=la.norm(v); return (v/n if n>eps else np.array([1.,0.,0.],float))

def clamp(x, lo, hi): return max(lo, min(hi, x))

def mrp_to_dcm(s: np.ndarray) -> np.ndarray:
    s2=float(np.dot(s,s)); I3=np.eye(3)
    if s2<1e-16: return I3.copy()
    S=skew(s); c=1+s2
    return I3 + (8*S@S - 4*(1-s2)*S) / (c*c)

def dcm_to_euler_zyx(C: np.ndarray) -> np.ndarray:
    psi   = math.atan2(C[1,0], C[0,0])  # yaw (Z)
    theta = math.asin(-C[2,0])          # pitch (Y)
    phi   = math.atan2(C[2,1], C[2,2])  # roll (X)
    return np.array([phi, theta, psi])

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2*np.pi) - np.pi

def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    tr = float(np.trace(C))
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        q0 = 0.25 * S
        q1 = (C[2,1] - C[1,2]) / S
        q2 = (C[0,2] - C[2,0]) / S
        q3 = (C[1,0] - C[0,1]) / S
    else:
        d = np.array([C[0,0], C[1,1], C[2,2]])
        i = int(np.argmax(d))
        if i == 0:
            S = math.sqrt(1.0 + C[0,0] - C[1,1] - C[2,2]) * 2.0
            q0 = (C[2,1] - C[1,2]) / S
            q1 = 0.25 * S
            q2 = (C[0,1] + C[1,0]) / S
            q3 = (C[0,2] + C[2,0]) / S
        elif i == 1:
            S = math.sqrt(1.0 + C[1,1] - C[0,0] - C[2,2]) * 2.0
            q0 = (C[0,2] - C[2,0]) / S
            q1 = (C[0,1] + C[1,0]) / S
            q2 = 0.25 * S
            q3 = (C[1,2] + C[2,1]) / S
        else:
            S = math.sqrt(1.0 + C[2,2] - C[0,0] - C[1,1]) * 2.0
            q0 = (C[1,0] - C[0,1]) / S
            q1 = (C[0,2] + C[2,0]) / S
            q2 = (C[1,2] + C[2,1]) / S
            q3 = 0.25 * S
    q = np.array([q0,q1,q2,q3], float)
    return q / la.norm(q)

def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], float)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], float)

def shortest_arc_error(R: np.ndarray, R_d: np.ndarray) -> np.ndarray:
    """Quaternion vector-part error with shortest-arc sign."""
    q   = dcm_to_quat(R)
    q_d = dcm_to_quat(R_d)
    q_e = quat_mul(q_d, quat_conj(q))
    if q_e[0] < 0.0:
        q_e = -q_e
    return q_e[1:]

def so3_exp(phi: np.ndarray) -> np.ndarray:
    th = la.norm(phi)
    if th < 1e-10:
        return np.eye(3) + skew(phi)
    k = phi/th; K = skew(k)
    return np.eye(3) + math.sin(th)*K + (1-math.cos(th))*(K@K)

def so3_log(R: np.ndarray) -> np.ndarray:
    c = clamp((np.trace(R)-1.0)/2.0, -1.0, 1.0)
    th = math.acos(c)
    if th < 1e-10:
        return np.zeros(3)
    return (th/(2*math.sin(th))) * np.array([R[2,1]-R[1,2],
                                             R[0,2]-R[2,0],
                                             R[1,0]-R[0,1]])

# ───────────────────────── “look‑at” with roll continuity ─────────────────────────

def camera_pointing_C_BN(u_N: np.ndarray, up_N: np.ndarray, cam_axis: str="z") -> np.ndarray:
    """Instantaneous look-at; no temporal continuity."""
    u = unit(u_N); zN = unit(up_N)
    if cam_axis.lower()=="z":
        zB = u
        xB = unit(np.cross(zN,zB)) if la.norm(np.cross(zN,zB))>1e-8 else unit(np.cross(np.array([1,0,0]), zB))
        yB = np.cross(zB,xB)
    elif cam_axis.lower()=="x":
        xB = u
        yB = unit(np.cross(xB,zN)) if la.norm(np.cross(xB,zN))>1e-8 else unit(np.cross(xB, np.array([0,1,0])))
        zB = np.cross(xB,yB)
    else:
        yB = u
        zB = unit(np.cross(np.array([1,0,0]),yB)) if la.norm(np.cross(np.array([1,0,0]),yB))>1e-8 else unit(np.cross(np.array([0,1,0]),yB))
        xB = np.cross(yB,zB)
    C_NB = np.column_stack((xB,yB,zB))
    return C_NB.T

def look_at_continuous(u_N: np.ndarray, up_N: np.ndarray,
                       R_prev: Optional[np.ndarray], cam_axis: str="z") -> np.ndarray:
    """Continuous look-at: align boresight with u, then rotate about u to stay close to R_prev."""
    R0 = camera_pointing_C_BN(u_N, up_N, cam_axis)
    if R_prev is None:
        return R0
    # choose the in-plane axis to preserve based on cam_axis
    if cam_axis.lower()=="z":
        b_prev = (R_prev.T)[:,0]    # x_B in N (prev)
        b_curr = (R0.T)[:,0]        # x_B from raw look
    elif cam_axis.lower()=="x":
        b_prev = (R_prev.T)[:,1]
        b_curr = (R0.T)[:,1]
    else:
        b_prev = (R_prev.T)[:,2]
        b_curr = (R0.T)[:,2]
    u = unit(u_N)
    P = np.eye(3) - np.outer(u,u)
    a = unit(P@b_curr); b = unit(P@b_prev)
    # rotation γ about u that best aligns a→b
    sin_g = np.dot(u, np.cross(a,b))
    cos_g = float(np.dot(a,b))
    gamma = math.atan2(sin_g, cos_g)  # small, continuous
    R_align = so3_exp(gamma * u)
    return R_align @ R0

# ───────────────────────── configs ─────────────────────────

@dc.dataclass
class Thruster:
    r_B: np.ndarray
    a_B: np.ndarray
    T: float
    Isp: float
    name: str="T"

@dc.dataclass
class VehicleConfig:
    name: str
    mass_kg: float
    inertia_B: np.ndarray
    sigma0_BN: np.ndarray
    omega0_B:  np.ndarray
    r0_N:      np.ndarray
    v0_N:      np.ndarray
    # SE(3) tracker (ζ, ωn)
    zeta_att:   float
    wn_att:     float
    zeta_vel:   float
    wn_vel:     float
    # allocation
    rw_tau_max: float
    w_force:    float
    w_torque:   float
    reg_lambda: float
    thrusters: List[Thruster]

@dc.dataclass
class SimConfig:
    dt: float = 0.02
    T_end: float = 240.0
    latency_s: float = 0.050
    shape_ramp_s: float = 6.0
    cmd_smooth_tau_s: float = 0.05
    cmd_smooth_for_s: float = 0.06
    tau_deadband_nm: float = 0.0005
    f_deadband_n:    float = 0.001
    enable_po_pc: bool = True
    use_latency_predict: bool = True

# ───────────────────────── SE(3) pH tracker ─────────────────────────

class SE3Layer:
    def __init__(self, veh: VehicleConfig, dt: float, engage_time: float, latency_s: float,
                 enable_po_pc=True, use_latency_predict=True,
                 cmd_smooth_tau_s=0.06, cmd_smooth_for_s=0.08,
                 shape_ramp_s=6.0, tau_deadband_nm=0.0, f_deadband_n=0.0):
        self.veh=veh; self.dt=dt; self.engage_time=engage_time
        J=veh.inertia_B; m=veh.mass_kg
        # base gains
        self.K_R_base = J * (veh.wn_att**2)
        self.D_R      = 2.0*veh.zeta_att*veh.wn_att*J
        self.K_p      = (m*(veh.wn_vel**2)) * np.eye(3)
        self.D_v      = (2.0*veh.zeta_vel*veh.wn_vel*m) * np.eye(3)
        # integrals (AW)
        self.KI_R = 0.015 * np.diag(np.diag(self.K_R_base))
        self.KI_p = 0.06  * self.K_p
        self.int_eR = np.zeros(3); self.int_ep = np.zeros(3)
        self.intR_max = 4.0*np.ones(3); self.intp_max = 0.8*np.ones(3)
        # ramp
        self.shape_ramp_s = float(shape_ramp_s)
        # ports
        self.ctrlTorqueMsg = messaging.CmdTorqueBodyMsg()
        self.ctrlForceMsg  = messaging.CmdForceBodyMsg()
        # latency
        self.L=int(round(max(0.0,latency_s)/dt))
        self.use_latency_predict=use_latency_predict
        self.enable_po_pc=enable_po_pc
        # smoothing
        self.alpha_tau = dt/(cmd_smooth_tau_s + dt)
        self.alpha_for = dt/(cmd_smooth_for_s + dt)
        self.tau_smooth = np.zeros(3); self.f_smooth = np.zeros(3)
        # deadbands
        self.tau_dead = float(tau_deadband_nm)
        self.f_dead   = float(f_deadband_n)
        # allocation map
        self.B, self.Ts, self.Isps = self._build_wrench_map(veh.thrusters)
        self.delta_prev: Optional[np.ndarray]=None
        # buffers
        self.buf_tau=[]; self.buf_for=[]
        self.omega_prev=np.zeros(3); self.v_prev=np.zeros(3)
        self.E_po=0.0
        # logs
        self.t_list=[]; self.Hd_list=[]
        self.tau_d_hist=[]; self.f_d_hist=[]
        self.torque_cmd_hist=[]; self.force_cmd_hist=[]
        self.torque_apply_hist=[]; self.force_apply_hist=[]
        self.omega_hist=[]; self.vB_hist=[]; self.sigma_hist=[]
        self.euler_hist=[]; self.euler_des_hist=[]
        self.thrust_total_hist=[]
        self.po_scale_hist=[]; self.E_po_hist=[]
        self.boresight_err_hist=[]; self.roll_about_los_err_hist=[]
        # desired memory
        self.Rd_prev=None
        # energy monotonicity
        self._Hd_prev=None; self.energy_total=0; self.energy_noninc=0
        # Δv/prop
        self.dv_cum=0.0; self.mp_cum=0.0

    def _build_wrench_map(self, thr: List[Thruster]):
        if not thr: return np.zeros((6,0)), np.zeros(0), np.zeros(0)
        cols=[]; Ts=[]; Isps=[]
        for th in thr:
            r=np.asarray(th.r_B).reshape(3); a=unit(np.asarray(th.a_B).reshape(3))
            cols.append(np.hstack((np.cross(r, th.T*a), th.T*a)))
            Ts.append(th.T); Isps.append(th.Isp)
        return np.column_stack(cols), np.array(Ts), np.array(Isps)

    def _bnls(self,B,u,delta_prev,w_tau,w_for,lam,lb,ub):
        n=B.shape[1]
        if delta_prev is None or delta_prev.shape[0]!=n: delta_prev=np.zeros(n)
        W=np.diag([w_tau,w_tau,w_tau,w_for,w_for,w_for])
        BtWB=B.T@W@B + lam*np.eye(n); BtWu=B.T@W@u + lam*delta_prev
        delta=np.clip(la.solve(BtWB,BtWu),lb,ub)
        for _ in range(6):
            low=(delta<=lb+1e-9); high=(delta>=ub-1e-9); free=~(low|high)
            if not np.any(free): break
            Bf=B[:,free]
            K=la.pinv(Bf.T@W@Bf + lam*np.eye(np.count_nonzero(free)))
            rhs=Bf.T@W@(u - B[:,low]@np.full(np.count_nonzero(low), lb)
                          - B[:,high]@np.full(np.count_nonzero(high), ub)) + lam*delta_prev[free]
            delta_f=K@rhs; delta=delta.copy(); delta[free]=delta_f; delta=np.clip(delta,lb,ub)
        return delta

    def _shape_gain(self, t: float) -> float:
        if t <= self.engage_time: return 0.0
        return clamp((t - self.engage_time)/max(1e-6,self.shape_ramp_s), 0.0, 1.0)

    def _K_R_scheduled(self, e_R: np.ndarray) -> np.ndarray:
        # amplify stiffness for large attitude errors
        k = la.norm(e_R)
        scale = 1.0 + 1.8*math.tanh(2.5*k)      # up to ~2.8×
        return scale * self.K_R_base

    def _desired_rate_body(self, R: np.ndarray, R_d: np.ndarray) -> np.ndarray:
        if self.Rd_prev is None:
            self.Rd_prev = R_d.copy()
            return np.zeros(3)
        Delta = self.Rd_prev.T @ R_d
        phi   = so3_log(Delta)
        omega_d_Bd = phi / self.dt
        C_BBd = R.T @ R_d
        self.Rd_prev = R_d.copy()
        return C_BBd @ omega_d_Bd

    def step(self, t: float, attNav: messaging.NavAttMsgPayload, transNav: messaging.NavTransMsgPayload,
             R_d: np.ndarray, p_d_N: np.ndarray, los_N: Optional[np.ndarray]=None, cam_axis: str="z"):
        dt=self.dt; J=self.veh.inertia_B; m=self.veh.mass_kg

        # states (Basilisk)
        sigma=np.array(attNav.sigma_BN); R=mrp_to_dcm(sigma)
        omega=np.array(attNav.omega_BN_B)
        r_N = np.array(getattr(transNav, 'r_BN_N', [0.0,0.0,0.0]), float)
        v_N = np.array(getattr(transNav, 'v_BN_N', [0.0,0.0,0.0]), float)
        v_B=R@v_N

        # desired rate
        omega_d_B = self._desired_rate_body(R, R_d)

        # latency prediction
        if self.use_latency_predict:
            omega_p = omega + (omega - self.omega_prev)
            vB_p    = v_B  + (v_B  - self.v_prev)
            vN_p    = v_N  + (v_N  - R.T@self.v_prev)
        else:
            omega_p = omega.copy(); vB_p=v_B.copy(); vN_p=v_N.copy()

        # SE(3) errors
        e_R = shortest_arc_error(R, R_d)
        e_ω = omega_p - omega_d_B
        e_p = r_N - p_d_N
        e_v = vN_p

        # gains & ramp
        g   = self._shape_gain(t)
        K_R = self._K_R_scheduled(e_R)

        # preliminary (unsmoothed) wrench
        tau_pre = - (self.D_R @ e_ω) - g*(K_R @ e_R) - (self.KI_R @ self.int_eR)
        F_N_pre = - (self.D_v @ e_v) - g*(self.K_p @ e_p) - (self.KI_p @ self.int_ep)
        f_pre   = R @ F_N_pre

        # smoothing
        self.tau_smooth = (1-self.alpha_tau)*self.tau_smooth + self.alpha_tau*tau_pre
        self.f_smooth   = (1-self.alpha_for)*self.f_smooth   + self.alpha_for  *f_pre
        tau_des = self.tau_smooth; f_des = self.f_smooth

        # PO/PC
        s=1.0; allow_int=True
        if self.enable_po_pc:
            P   = float(tau_des @ omega + f_des @ v_B)
            dE  = P * dt
            if dE>0.0 and self.E_po + dE > 0.0:
                s = max(0.0, min(1.0, -self.E_po/(dE+1e-12)))
            if s < 0.999: allow_int=False
            tau_des *= s; f_des *= s
            P_eff = float(tau_des @ omega + f_des @ v_B)
            self.E_po += P_eff * dt
        self.po_scale_hist.append(s); self.E_po_hist.append(self.E_po)

        # wheels + jets (jets track force and some torque)
        tau_rw=tau_des.copy(); tn=la.norm(tau_rw)
        if tn>self.veh.rw_tau_max>0: tau_rw *= (self.veh.rw_tau_max/tn)
        tau_res=tau_des - tau_rw

        # deadband & allocation
        if la.norm(tau_res) < self.tau_dead and la.norm(f_des) < self.f_dead:
            delta=np.zeros(0); tau_thr=np.zeros(3); f_thr=np.zeros(3)
        else:
            if self.B.size>0:
                u=np.hstack((tau_res,f_des))
                delta=self._bnls(self.B,u,self.delta_prev,self.veh.w_torque,self.veh.w_force,self.veh.reg_lambda,0.0,1.0)
                self.delta_prev=delta.copy()
                uf=self.B@delta; tau_thr=uf[:3]; f_thr=uf[3:]
            else:
                delta=np.zeros(0); tau_thr=np.zeros(3); f_thr=np.zeros(3)

        tau_real=tau_rw + tau_thr; f_real=f_thr

        # latency buffer
        self.buf_tau.append(tau_real); self.buf_for.append(f_real)
        if len(self.buf_tau)>self.L:
            tau_apply=self.buf_tau.pop(0); f_apply=self.buf_for.pop(0)
        else:
            tau_apply=np.zeros(3); f_apply=np.zeros(3)

        # publish
        t_pl=messaging.CmdTorqueBodyMsgPayload(); t_pl.torqueRequestBody=list(tau_apply)
        f_pl=messaging.CmdForceBodyMsgPayload();  f_pl.forceRequestBody =list(f_apply)
        self.ctrlTorqueMsg.write(t_pl); self.ctrlForceMsg.write(f_pl)

        # integrals (anti-windup)
        if allow_int:
            self.int_eR = np.clip(self.int_eR + e_R*dt, -self.intR_max, self.intR_max)
            self.int_ep = np.clip(self.int_ep + e_p*dt, -self.intp_max, self.intp_max)

        # shaped energy
        V_R = 0.5 * g * float(e_R.T @ (K_R @ e_R))
        V_p = 0.5 * g * float(e_p.T @ (self.K_p @ e_p))
        H   = 0.5*float(omega.T@(J@omega)) + 0.5*float(self.veh.mass_kg*(v_B@v_B)) + V_R + V_p

        if t>=self.engage_time:
            self.energy_total+=1
            if self._Hd_prev is not None and H <= self._Hd_prev+1e-12:
                self.energy_noninc+=1
            self._Hd_prev = H

        # diagnostics: pointing & roll-about-LOS
        if los_N is not None:
            zB_N = (R.T)[:,2] if cam_axis.lower()=="z" else ((R.T)[:,0] if cam_axis.lower()=="x" else (R.T)[:,1])
            u = unit(los_N)
            boresight_deg = math.degrees(math.acos(clamp(float(np.dot(zB_N,u)),-1.0,1.0)))
            if cam_axis.lower()=="z":
                xB_N = (R.T)[:,0]; xBd_N = (R_d.T)[:,0]
            elif cam_axis.lower()=="x":
                yB_N = (R.T)[:,1]; yBd_N = (R_d.T)[:,1]; xB_N=yB_N; xBd_N=yBd_N
            else:
                zB_Np = (R.T)[:,2]; zBd_Np = (R_d.T)[:,2]; xB_N=zB_Np; xBd_N=zBd_Np
            P = np.eye(3) - np.outer(u,u)
            a = unit(P@xB_N); b = unit(P@xBd_N)
            roll_err_deg = math.degrees(math.atan2(la.norm(np.cross(a,b)), float(np.dot(a,b))))
            self.boresight_err_hist.append(boresight_deg)
            self.roll_about_los_err_hist.append(roll_err_deg)
        else:
            self.boresight_err_hist.append(np.nan); self.roll_about_los_err_hist.append(np.nan)

        # logs
        self.t_list.append(t); self.Hd_list.append(H)
        self.tau_d_hist.append(tau_des.copy()); self.f_d_hist.append(f_des.copy())
        self.torque_cmd_hist.append(tau_real.copy()); self.force_cmd_hist.append(f_real.copy())
        self.torque_apply_hist.append(tau_apply.copy()); self.force_apply_hist.append(f_apply.copy())
        self.omega_hist.append(omega.copy()); self.vB_hist.append(v_B.copy()); self.sigma_hist.append(sigma.copy())
        self.thrust_total_hist.append(float(np.sum(self.Ts*delta)) if self.B.size and delta.size else 0.0)

        # store Euler (wrapped difference for plotting clarity)
        rpy   = dcm_to_euler_zyx(R)
        rpy_d = dcm_to_euler_zyx(R_d)
        diff  = wrap_to_pi(rpy - rpy_d)
        rpy_wrapped = rpy_d + diff         # achieved wrapped near desired
        self.euler_hist.append(rpy_wrapped)
        self.euler_des_hist.append(rpy_d)

        # Δv/prop accounting
        if self.B.size and delta.size>0:
            impulse=float(np.sum(self.Ts*delta)*dt)
            self.dv_cum += impulse/self.veh.mass_kg
            if impulse>0:
                self.mp_cum += impulse / (np.average(self.Isps, weights=self.Ts*delta)*9.80665)

        self.omega_prev=omega.copy(); self.v_prev=v_B.copy()

# ───────────────────────── RAFTI separation (single deputy) ─────────────────────────

@dc.dataclass
class SepParams:
    t_start: float
    t_end:   float
    F0:      float
    s_car_B: np.ndarray
    r_car_B: np.ndarray
    r_dep_B: np.ndarray
    dv_cmd:  float

class RAFTISeparation:
    def __init__(self, dt: float,
                 sc_car: spacecraft.Spacecraft, eff_car: extForceTorque.ExtForceTorque,
                 sc_dep: List[spacecraft.Spacecraft], eff_dep: List[extForceTorque.ExtForceTorque],
                 sched: List[SepParams], m_car: float, m_dep: float):
        self.dt=dt; self.sched=sched
        self.sc_car=sc_car; self.eff_car=eff_car
        self.sc_dep=sc_dep; self.eff_dep=eff_dep
        self.m_car=m_car; self.m_dep=m_dep
        self.car_tauMsg=messaging.CmdTorqueBodyMsg()
        self.car_forMsg=messaging.CmdForceBodyMsg()
        self.dep_tauMsg=[messaging.CmdTorqueBodyMsg() for _ in sc_dep]
        self.dep_forMsg=[messaging.CmdForceBodyMsg() for _ in sc_dep]
        eff_car.cmdTorqueInMsg   .subscribeTo(self.car_tauMsg)
        eff_car.cmdForceBodyInMsg.subscribeTo(self.car_forMsg)
        for ef,tm,fm in zip(eff_dep,self.dep_tauMsg,self.dep_forMsg):
            ef.cmdTorqueInMsg   .subscribeTo(tm)
            ef.cmdForceBodyInMsg.subscribeTo(fm)

    def _half_sine(self, t, t0, T, F0):
        if t<t0 or t>t0+T: return 0.0
        return F0*math.sin(math.pi*(t-t0)/T)

    def step(self, t: float):
        C_BN_car = mrp_to_dcm(np.array(self.sc_car.scStateOutMsg.read().sigma_BN))
        C_NB_car = C_BN_car.T
        tau_car = np.zeros(3); f_car = np.zeros(3)
        for i,sp in enumerate(self.sched):
            C_BN_dep = mrp_to_dcm(np.array(self.sc_dep[i].scStateOutMsg.read().sigma_BN))
            C_NB_dep = C_BN_dep.T
            Fmag = self._half_sine(t, sp.t_start, sp.t_end-sp.t_start, sp.F0)
            if Fmag<=0:
                tm=messaging.CmdTorqueBodyMsgPayload(); tm.torqueRequestBody=[0.0,0.0,0.0]
                fm=messaging.CmdForceBodyMsgPayload();  fm.forceRequestBody =[0.0,0.0,0.0]
                self.dep_tauMsg[i].write(tm); self.dep_forMsg[i].write(fm)
                continue
            s_N = C_NB_car @ sp.s_car_B
            F_N = Fmag * s_N
            f_car_i_B =  C_BN_car @ ( +F_N )
            f_dep_i_B =  C_BN_dep @ ( -F_N )
            tau_car_i = np.cross(sp.r_car_B, f_car_i_B)
            tau_dep_i = np.cross(sp.r_dep_B, f_dep_i_B)
            tm=messaging.CmdTorqueBodyMsgPayload(); tm.torqueRequestBody=list(tau_dep_i)
            fm=messaging.CmdForceBodyMsgPayload();  fm.forceRequestBody =list(f_dep_i_B)
            self.dep_tauMsg[i].write(tm); self.dep_forMsg[i].write(fm)
            tau_car += tau_car_i; f_car += f_car_i_B
        tm=messaging.CmdTorqueBodyMsgPayload(); tm.torqueRequestBody=list(tau_car)
        fm=messaging.CmdForceBodyMsgPayload();  fm.forceRequestBody =list(f_car)
        self.car_tauMsg.write(tm); self.car_forMsg.write(fm)

# ───────────────────────── build/run/plot ─────────────────────────

class Fleet:
    def __init__(self):
        self.dt=0.02
        self.sim = SimConfig()

        # ESPA circle: 42" (Ø=1.0668 m)
        self.espa_radius = 42.0*0.0254/2.0  # 0.5334 m

        # Carrier
        I_car=np.diag([1200.,1000.,800.]); m_car=650.0
        mr22_T, mr22_Isp = 22.0, 235.0
        thr_c = [
            Thruster(np.array([ 0.8, 0.0,  0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp),
            Thruster(np.array([-0.8, 0.0,  0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp),
            Thruster(np.array([ 0.8, 0.0, -0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp),
            Thruster(np.array([-0.8, 0.0, -0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp),
        ]
        self.car = VehicleConfig(
            name="carrier", mass_kg=m_car, inertia_B=I_car,
            sigma0_BN=np.zeros(3), omega0_B=np.zeros(3), r0_N=np.zeros(3), v0_N=np.zeros(3),
            zeta_att=0.9, wn_att=0.12, zeta_vel=1.0, wn_vel=0.10,
            rw_tau_max=0.25, w_force=40.0, w_torque=10.0, reg_lambda=3e-4, thrusters=thr_c
        )

        # Deputy (XB6 6U) with torque-capable 12-jet layout
        I_dep=np.diag([0.025,0.020,0.018]); m_dep=12.0
        T_small, Isp_small = 0.27, 217.0
        L = 0.065
        thr_d=[]
        for s in (+1,-1):
            thr_d += [Thruster(np.array([0, +L, +L]), np.array([s,0,0]), T_small, Isp_small),
                      Thruster(np.array([0, -L, -L]), np.array([s,0,0]), T_small, Isp_small)]
        for s in (+1,-1):
            thr_d += [Thruster(np.array([+L,0,+L]), np.array([0,s,0]), T_small, Isp_small),
                      Thruster(np.array([-L,0,-L]), np.array([0,s,0]), T_small, Isp_small)]
        for s in (+1,-1):
            thr_d += [Thruster(np.array([+L,+L,0]), np.array([0,0,s]), T_small, Isp_small),
                      Thruster(np.array([-L,-L,0]), np.array([0,0,s]), T_small, Isp_small)]

        ring_R = self.espa_radius; z_mount = 0.0; ang = 0.0
        self.espa_port_r_B   = np.array([ ring_R*math.cos(ang), ring_R*math.sin(ang), z_mount ], float)
        self.espa_port_dir_B = np.array([ math.cos(ang),        math.sin(ang),        0.0    ], float)

        # initial deputy pose aligned with port local frame
        bY = self.espa_port_dir_B; bZ=np.array([0,0,1.0])
        if abs(np.dot(bY/np.linalg.norm(bY),bZ))>0.99: bZ=np.array([1,0,0])
        bX = unit(np.cross(bY,bZ)); bZ = unit(np.cross(bX,bY))
        R_BD_BC = np.column_stack((bX,bY,bZ))
        sigma0 = self.dcm_to_mrp_safe(R_BD_BC)

        self.dep = VehicleConfig(
            name="XB6_6U", mass_kg=m_dep, inertia_B=I_dep,
            sigma0_BN=sigma0, omega0_B=np.zeros(3),
            r0_N=self.espa_port_r_B.copy(), v0_N=np.zeros(3),
            zeta_att=1.25, wn_att=1.00,     # faster attitude with ωd feed-forward
            zeta_vel=1.10, wn_vel=0.30,     # translation damping
            rw_tau_max=0.08,
            w_force=220.0, w_torque=60.0, reg_lambda=5e-4,
            thrusters=thr_d
        )

        # tasks
        dt=self.dt
        self.dyn = sys_model_task.SysModelTask(int(macros.sec2nano(dt)))
        self.sns = sys_model_task.SysModelTask(int(macros.sec2nano(dt)))

        # carrier objects
        self.sc_car = spacecraft.Spacecraft(); self.sc_car.ModelTag="SC_carrier"
        self.sc_car.hub.mHub=self.car.mass_kg; self.sc_car.hub.IHubPntBc_B=self.car.inertia_B.tolist()
        self.sc_car.hub.sigma_BNInit=list(self.car.sigma0_BN)
        self.sc_car.hub.omega_BN_BInit=list(self.car.omega0_B)
        self.sc_car.hub.r_CN_NInit=list(self.car.r0_N); self.sc_car.hub.v_CN_NInit=list(self.car.v0_N)
        self.nav_car = simpleNav.SimpleNav(); self.nav_car.ModelTag="NAV_carrier"
        self.nav_car.scStateInMsg.subscribeTo(self.sc_car.scStateOutMsg)
        self.eft_car_ctrl = extForceTorque.ExtForceTorque(); self.eft_car_ctrl.ModelTag="EFT_car_ctrl"
        self.eft_car_sep  = extForceTorque.ExtForceTorque(); self.eft_car_sep .ModelTag="EFT_car_sep"
        self.sc_car.addDynamicEffector(self.eft_car_ctrl); self.sc_car.addDynamicEffector(self.eft_car_sep)

        # deputy objects
        self.sc_dep = spacecraft.Spacecraft(); self.sc_dep.ModelTag="SC_XB6"
        self.sc_dep.hub.mHub=self.dep.mass_kg; self.sc_dep.hub.IHubPntBc_B=self.dep.inertia_B.tolist()
        self.sc_dep.hub.sigma_BNInit=list(self.dep.sigma0_BN); self.sc_dep.hub.omega_BN_BInit=list(self.dep.omega0_B)
        self.sc_dep.hub.r_CN_NInit=list(self.dep.r0_N);       self.sc_dep.hub.v_CN_NInit=list(self.dep.v0_N)
        self.nav_dep = simpleNav.SimpleNav(); self.nav_dep.ModelTag="NAV_XB6"
        self.nav_dep.scStateInMsg.subscribeTo(self.sc_dep.scStateOutMsg)
        self.eft_dep_ctrl = extForceTorque.ExtForceTorque(); self.eft_dep_ctrl.ModelTag="EFT_XB6_ctrl"
        self.eft_dep_sep  = extForceTorque.ExtForceTorque(); self.eft_dep_sep .ModelTag="EFT_XB6_sep"
        self.sc_dep.addDynamicEffector(self.eft_dep_ctrl); self.sc_dep.addDynamicEffector(self.eft_dep_sep)

        # add to tasks
        p=0
        self.sns.AddNewObject(self.nav_car,10+p); self.dyn.AddNewObject(self.sc_car,5+p)
        self.dyn.AddNewObject(self.eft_car_ctrl,10+p); self.dyn.AddNewObject(self.eft_car_sep,11+p)
        self.sns.AddNewObject(self.nav_dep,  20+p)
        self.dyn.AddNewObject(self.sc_dep,    15+p)
        self.dyn.AddNewObject(self.eft_dep_ctrl,   20+p)
        self.dyn.AddNewObject(self.eft_dep_sep,    21+p)

        # controllers
        last_sep_end = 10.0 + 0.30
        self.ctrl_car = SE3Layer(self.car, dt, last_sep_end, self.sim.latency_s,
                                 enable_po_pc=self.sim.enable_po_pc, use_latency_predict=self.sim.use_latency_predict,
                                 cmd_smooth_tau_s=self.sim.cmd_smooth_tau_s, cmd_smooth_for_s=self.sim.cmd_smooth_for_s,
                                 shape_ramp_s=0.0, tau_deadband_nm=0.0015, f_deadband_n=0.015)
        self.ctrl_dep = SE3Layer(self.dep, dt, last_sep_end, self.sim.latency_s,
                                 enable_po_pc=self.sim.enable_po_pc, use_latency_predict=self.sim.use_latency_predict,
                                 cmd_smooth_tau_s=self.sim.cmd_smooth_tau_s, cmd_smooth_for_s=self.sim.cmd_smooth_for_s,
                                 shape_ramp_s=self.sim.shape_ramp_s,
                                 tau_deadband_nm=self.sim.tau_deadband_nm, f_deadband_n=self.sim.f_deadband_n)

        # wiring
        self.eft_car_ctrl.cmdTorqueInMsg   .subscribeTo(self.ctrl_car.ctrlTorqueMsg)
        self.eft_car_ctrl.cmdForceBodyInMsg.subscribeTo(self.ctrl_car.ctrlForceMsg)
        self.eft_dep_ctrl.cmdTorqueInMsg   .subscribeTo(self.ctrl_dep.ctrlTorqueMsg)
        self.eft_dep_ctrl.cmdForceBodyInMsg.subscribeTo(self.ctrl_dep.ctrlForceMsg)

        # RAFTI schedule
        self.sched=[]
        self.t0_base=10.0; Tpush=0.30
        rng=np.random.default_rng(42)
        dv = float(np.clip(rng.normal(0.018, 0.005), 0.004, 0.025))
        F0 = self.dep.mass_kg * dv * math.pi / (2.0*Tpush)
        r_dep_B = rng.normal(scale=0.02, size=3)
        self.sched.append(SepParams(t_start=self.t0_base, t_end=self.t0_base+Tpush, F0=F0,
                                    s_car_B=self.espa_port_dir_B.copy(),
                                    r_car_B=self.espa_port_r_B.copy(),
                                    r_dep_B=r_dep_B, dv_cmd=dv))
        self.sep = RAFTISeparation(self.dt, self.sc_car, self.eft_car_sep,
                                   [self.sc_dep], [self.eft_dep_sep], self.sched,
                                   m_car=self.car.mass_kg, m_dep=self.dep.mass_kg)

        # Camera pointing parameters
        self.CAM_AXIS = "z"               # change to 'x' or 'y' if needed
        self.UP_REF_N = np.array([0.0,0.0,1.0])

        # Position reference freezes at end of push
        self.p_ref_frozen=False; self.p_d_N_frozen=np.zeros(3)

        # store continuous desired attitude between steps
        self.Rd_cont_prev=None

        # init tasks
        for tsk in (self.sns, self.dyn):
            tsk.SelfInitTaskList(); tsk.ResetTaskList(0)

        need = self.t0_base + 120.0
        if need > self.sim.T_end: self.sim.T_end = need

    @staticmethod
    def dcm_to_mrp_safe(C: np.ndarray) -> np.ndarray:
        q = dcm_to_quat(C); q0,qv=q[0],q[1:]
        return (-qv/(1-q0) if 1+q0<1e-12 else qv/(1+q0))

    def _port_position_in_N(self):
        C_BN_car = mrp_to_dcm(np.array(self.sc_car.scStateOutMsg.read().sigma_BN))
        C_NB_car = C_BN_car.T
        r_car_N  = np.array(self.nav_car.transOutMsg.read().r_BN_N, float)
        return r_car_N + C_NB_car @ self.espa_port_r_B

    def _camera_Rd_cont(self, r_dep_N: np.ndarray):
        r_port_N = self._port_position_in_N()
        los_N = r_port_N - r_dep_N
        if la.norm(los_N) < 1e-8:
            los_N = np.array([1.0,0.0,0.0])
        Rd = look_at_continuous(los_N, self.UP_REF_N, self.Rd_cont_prev, cam_axis=self.CAM_AXIS)
        self.Rd_cont_prev = Rd.copy()
        return Rd, los_N

    def run(self):
        now_ns=0; N=int(self.sim.T_end/self.dt)
        last_sep_end = self.t0_base + 0.30

        for k in tqdm(range(N), desc="ESPA(42in)+XB6 — SE(3) tracker", ncols=80):
            t=k*self.dt
            self.sns.ExecuteTaskList(now_ns)
            self.sep.step(t)

            # carrier damping
            self.ctrl_car.step(t, self.nav_car.attOutMsg.read(), self.nav_car.transOutMsg.read(),
                               R_d=np.eye(3), p_d_N=np.zeros(3))

            # deputy
            r_dep_N = np.array(self.nav_dep.transOutMsg.read().r_BN_N, float)

            # freeze translation ref
            if (not self.p_ref_frozen) and t >= last_sep_end:
                self.p_d_N_frozen = r_dep_N.copy()
                self.p_ref_frozen = True
            p_ref = self.p_d_N_frozen if self.p_ref_frozen else r_dep_N.copy()

            # pointing
            if t >= last_sep_end:
                R_d, los_N = self._camera_Rd_cont(r_dep_N)
            else:
                R_d = mrp_to_dcm(self.dep.sigma0_BN); los_N=None

            self.ctrl_dep.step(t, self.nav_dep.attOutMsg.read(), self.nav_dep.transOutMsg.read(),
                               R_d=R_d, p_d_N=p_ref, los_N=los_N, cam_axis=self.CAM_AXIS)

            self.dyn.ExecuteTaskList(now_ns)
            now_ns+=int(macros.sec2nano(self.dt))

        self._report(last_sep_end)
        self._plots(last_sep_end)

    def _report(self, t_eng: float):
        c=self.ctrl_car; d=self.ctrl_dep
        print(f"[carrier] H_d non-increasing fraction after t≥{t_eng:.1f}s: {c.energy_noninc/max(1,c.energy_total):.3f}, "
              f"Δv={c.dv_cum:.4f} m/s, prop={c.mp_cum*1e3:.3f} g")
        print(f"[XB6]     H_d non-increasing fraction after t≥{t_eng:.1f}s: {d.energy_noninc/max(1,d.energy_total):.3f}, "
              f"Δv={d.dv_cum:.4f} m/s, prop={d.mp_cum*1e3:.3f} g")

    def _plots(self, engage_t: float):
        ctl = self.ctrl_dep; sp = self.sched[0]

        # Δv
        plt.figure(figsize=(7,4))
        plt.bar([0.8], [sp.dv_cmd], width=0.35, label="RAFTI Δv_cmd")
        plt.bar([1.2], [ctl.dv_cum], width=0.35, label="Control Δv")
        plt.xticks([0.8,1.2], ["XB6","XB6"]); plt.ylabel("Δv [m/s]")
        plt.title("Separation Δv vs Control Δv"); plt.legend(); plt.tight_layout(); plt.show()

        # energy & histories
        t = np.array(ctl.t_list); Hd=np.array(ctl.Hd_list)
        tau_d=np.array(ctl.tau_d_hist); tau_c=np.array(ctl.torque_cmd_hist); tau_a=np.array(ctl.torque_apply_hist)
        f_d=np.array(ctl.f_d_hist); f_c=np.array(ctl.force_cmd_hist); f_a=np.array(ctl.force_apply_hist)
        omg=np.array(ctl.omega_hist); sig=np.array(ctl.sigma_hist); thrust=np.array(ctl.thrust_total_hist)

        def nrm(X): return la.norm(X,axis=1) if (isinstance(X,np.ndarray) and X.ndim==2 and X.shape[0]==t.size) else np.zeros_like(t)
        fig,axs=plt.subplots(6,1,figsize=(10,14),sharex=True)
        axs[0].plot(t,Hd); axs[0].axvline(engage_t,ls="--",alpha=0.6); axs[0].set_ylabel("H_SE3,d")
        axs[0].set_title("XB6 — energy & histories")
        axs[1].plot(t,nrm(tau_d),label="|τ_d|"); axs[1].plot(t,nrm(tau_c),label="|τ_cmd|"); axs[1].plot(t,nrm(tau_a),label="|τ_appl|"); axs[1].legend()
        axs[1].set_ylabel("Torque norm")
        axs[2].plot(t,nrm(f_d),label="|f_d|"); axs[2].plot(t,nrm(f_c),label="|f_cmd|"); axs[2].plot(t,nrm(f_a),label="|f_appl|"); axs[2].legend()
        axs[2].set_ylabel("Force norm")
        axs[3].plot(t, thrust if thrust.size else np.zeros_like(t)); axs[3].set_ylabel("∑T·δ [N]")
        if omg.ndim==2 and omg.shape[1]==3:
            axs[4].plot(t,omg[:,0],label="ω_x"); axs[4].plot(t,omg[:,1],label="ω_y"); axs[4].plot(t,omg[:,2],label="ω_z"); axs[4].legend(); axs[4].set_ylabel("ω [rad/s]")
        if sig.ndim==2 and sig.shape[1]==3:
            axs[5].plot(t,sig[:,0],label="σ1"); axs[5].plot(t,sig[:,1],label="σ2"); axs[5].plot(t,sig[:,2],label="σ3"); axs[5].legend(); axs[5].set_ylabel("MRP")
        axs[-1].set_xlabel("time [s]"); fig.tight_layout(); plt.show()

        # Euler tracking (wrapped, no 2π jumps)
        rpy   = np.array(ctl.euler_hist)
        rpy_d = np.array(ctl.euler_des_hist)
        if rpy.shape[0]==t.size:
            labels=["roll φ","pitch θ","yaw ψ"]
            fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
            for k in range(3):
                axs[k].plot(t,rpy_d[:,k],'--',label=f"{labels[k]} (desired)")
                axs[k].plot(t,rpy[:,k],label=f"{labels[k]} (achieved)")
                axs[k].set_ylabel("[rad]"); axs[k].legend(loc="upper right")
            axs[-1].set_xlabel("time [s]"); fig.suptitle("Euler ZYX tracking to ESPA port (wrapped)"); fig.tight_layout(); plt.show()

        # boresight & roll-about-LOS
        bores=np.array(ctl.boresight_err_hist); rollE=np.array(ctl.roll_about_los_err_hist)
        fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
        ax[0].plot(t,bores); ax[0].axhline(1.0,ls="--",alpha=0.3); ax[0].set_ylabel("boresight [deg]")
        ax[1].plot(t,rollE); ax[1].axhline(2.0,ls="--",alpha=0.3); ax[1].set_ylabel("roll about LOS [deg]"); ax[1].set_xlabel("time [s]")
        fig.suptitle("Pointing quality"); fig.tight_layout(); plt.show()

        # relative errors (gated near zero)
        def rel_err(a,b):
            if a.size==0 or b.size==0: return np.zeros_like(t)
            den = la.norm(a,axis=1); ref=max(np.max(den),1.0)
            num = la.norm(a-b,axis=1); mask=den>1e-3*ref
            return np.where(mask, num/np.maximum(den,1e-12), 0.0)
        e_tau_cmd=rel_err(tau_d,tau_c); e_tau_app=rel_err(tau_d,tau_a)
        e_f_cmd  =rel_err(f_d,f_c);     e_f_app  =rel_err(f_d,f_a)
        fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
        ax[0].plot(t,e_tau_cmd,label="τ desired→cmd"); ax[0].plot(t,e_tau_app,label="τ desired→appl"); ax[0].legend(); ax[0].set_ylabel("rel τ err")
        ax[1].plot(t,e_f_cmd,label="f desired→cmd"); ax[1].plot(t,e_f_app,label="f desired→appl"); ax[1].legend(); ax[1].set_ylabel("rel f err"); ax[1].set_xlabel("time [s]")
        fig.tight_layout(); plt.show()

def main():
    try:
        f=Fleet(); f.run()
        print("[DONE] ESPA(42\") + RAFTI‑GRIP + XB6 6U — camera points to ESPA port. Plots shown.")
    except Exception:
        print("\n[ERROR] Unhandled exception:\n")
        print(traceback.format_exc())

if __name__=="__main__":
    main()
