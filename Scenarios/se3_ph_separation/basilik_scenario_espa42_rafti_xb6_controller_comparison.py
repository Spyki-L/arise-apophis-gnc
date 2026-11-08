#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_scenario_espa42_rafti_xb6_controller_comparison.py

Purpose
-------
Basilisk scenario that compares three attitude-control architectures for an
ESPA-mounted XB6-class 6U CubeSat during a RAFTI-style separation from a
42-inch ESPA ring. The carrier bus remains mildly damped, while the deputy
tracks a continuous “look-at” attitude command that keeps the camera axis
pointed at the ESPA port with smooth roll about the line of sight.

Controllers under test
----------------------
1. PH_SE3
   - Full geometric port–Hamiltonian SE(3) tracker that commands both forces
     and torques through a thruster allocator.
   - Uses natural frequency / damping-shape gains (ζ, ω_n) for attitude
     and translation, with integral terms on rotation and position.
   - Includes wheel torque saturation, thruster BNLS allocation, and a
     shaping ramp after separation completes.

2. MRP_FB
   - Standard Basilisk `mrpFeedback` attitude controller.
   - Tracks an `AttRef` stream generated from the continuous look-at frame.
   - Optional low-pass filter on commanded torque for cleaner actuation.

3. PID
   - Basilisk `mrpSteering` + `rateServoFullNonlinear` chain (nonlinear
     steering law feeding a rate servo).
   - Also tracks the same `AttRef` stream and can use an optional torque
     low-pass filter.

Scenario setup
--------------
- Carrier:
  * 650 kg bus with diagonal inertia and four MR-22N-class thrusters.
  * Attitude and translation mildly damped with a PH-SE3 controller to
    keep the carrier well behaved during and after separation.
- Deputy (XB6 6U):
  * 12 kg, diagonal inertia, and a 12-jet thruster layout sized for small
    force authority.
  * Mounted at a single ESPA port on a 42" ring (radius ≈ 0.5334 m).
  * Initial attitude aligned to the local ESPA-port frame so the body
    frame is naturally “docked” to the port.

- RAFTI separation:
  * Single half-sine push (t ≈ 10.0–10.3 s) with peak force chosen to
    realize a stochastic Δv within RAFTI-like limits.
  * Equal-and-opposite force applied to carrier and deputy via
    `extForceTorque` effectors.
  * Same RAFTI schedule and random seed used for all three controller runs
    so the comparison is apples-to-apples.

Look-at guidance & reference generation
---------------------------------------
- The “camera” axis (default +Z_B) is commanded to point from the deputy to
  the moving ESPA port location in the inertial frame.
- A continuous-roll look-at function builds a DCM `R_d` that:
  * Aligns the camera axis with the line-of-sight to the port.
  * Uses a fixed inertial “up” direction to define roll about the LOS.
  * Enforces temporal continuity in roll to avoid jumps.
- For MRP_FB and PID:
  * `R_d` is converted into `AttRef` sigmas, angular rates, and angular
    accelerations by differentiating the desired frame.
- For PH_SE3:
  * `R_d` and a frozen post-separation reference position `p_d_N` are fed
    directly into the SE(3) controller as attitude and translation targets.

Outputs and metrics
-------------------
For each controller kind ∈ {PH_SE3, MRP_FB, PID}:
- Full time histories of:
  * Euler ZYX angles (roll, pitch, yaw) for the deputy, wrapped around the
    desired trajectory.
  * Boresight miss angle (camera axis vs. LOS to port) [deg].
  * Roll error about the LOS [deg].
  * Commanded torque magnitude |τ| [N·m].
  * Momentum-tracking error ‖J(ω − ω_ref)‖ [N·m·s].
- RMS performance metrics printed to the console:
  * RMS torque magnitude.
  * RMS momentum error.

Plotting
--------
At the end of the runs, the script overlays all three controllers on:
- Euler ZYX vs. time (with desired trajectories).
- Boresight miss and roll-about-LOS vs. time.
- Torque magnitude and momentum error vs. time.

Usage
-----
Run this script directly with Basilisk on the Python path:

    python basilik_scenario_espa42_rafti_xb6_controller_comparison.py

The simulation will execute the ESPA+RAFTI separation three times (one per
controller) and display comparison plots plus RMS metrics in the console.
"""


from __future__ import annotations
import math, dataclasses as dc
from dataclasses import field
from typing import List, Optional, Dict
import numpy as np, numpy.linalg as la
import matplotlib.pyplot as plt

# Basilisk
from Basilisk.architecture import messaging, sys_model_task
from Basilisk.simulation   import spacecraft, simpleNav, extForceTorque
from Basilisk.fswAlgorithms import (
    attTrackingError, mrpFeedback, mrpSteering, rateServoFullNonlinear,
    lowPassFilterTorqueCommand
)
from Basilisk.utilities    import macros

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it


# ───────────────────────── math helpers ─────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    x,y,z=v; return np.array([[0,-z,y],[z,0,-x],[-y,x,0]], float)

def normalize(v: np.ndarray, eps=1e-12):
    n=la.norm(v); return (v/n if n>eps else np.zeros_like(v), n)

def unit(v: np.ndarray, eps=1e-12) -> np.ndarray:
    return normalize(v, eps)[0]

def proj_SO3(C: np.ndarray) -> np.ndarray:
    """Project 3x3 matrix to SO(3) robustly; fall back to Gram–Schmidt if SVD fails."""
    C = np.asarray(C, float)
    if C.shape != (3,3) or not np.all(np.isfinite(C)):
        return np.eye(3)
    try:
        U,S,Vt = la.svd(C)
        R = U @ Vt
        if la.det(R) < 0.0:
            U[:,-1] *= -1.0
            R = U @ Vt
        return R
    except la.LinAlgError:
        # Fallback: Gram–Schmidt orthonormalization of the first two columns
        x = C[:,0]
        if not np.all(np.isfinite(x)) or la.norm(x) < 1e-12:
            x = np.array([1.0,0.0,0.0])
        else:
            x = x/la.norm(x)
        y = C[:,1]
        if not np.all(np.isfinite(y)):
            y = np.array([0.0,1.0,0.0])
        y = y - x*np.dot(x,y)
        if la.norm(y) < 1e-12:
            y = np.array([0.0,1.0,0.0]) - x*np.dot(x, np.array([0.0,1.0,0.0]))
            y = y/la.norm(y) if la.norm(y)>1e-12 else np.array([0.0,1.0,0.0])
        else:
            y = y/la.norm(y)
        z = np.cross(x,y)
        if la.norm(z) < 1e-12:
            z = np.array([0.0,0.0,1.0])
        else:
            z = z/la.norm(z)
        R = np.column_stack((x,y,z))
        if la.det(R) < 0.0:
            R[:,2] *= -1.0
        return R

def clamp(x, lo, hi):
    return lo if x<lo else (hi if x>hi else x)

def dcm_to_euler_zyx(C_BN: np.ndarray):
    """(roll φ, pitch θ, yaw ψ) robust ZYX from DCM (B from N)."""
    C = proj_SO3(np.asarray(C_BN, float))
    c20 = clamp(C[2,0], -1.0, 1.0)
    θ   = -math.asin(c20)
    ϕ   = math.atan2(clamp(C[2,1],-1.0,1.0), clamp(C[2,2],-1.0,1.0))
    ψ   = math.atan2(clamp(C[1,0],-1.0,1.0), clamp(C[0,0],-1.0,1.0))
    return np.array([ϕ, θ, ψ])

def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    tr = float(np.trace(C))
    if tr>0:
        S = math.sqrt(tr+1.0)*2.0
        q0=0.25*S; q1=(C[2,1]-C[1,2])/S; q2=(C[0,2]-C[2,0])/S; q3=(C[1,0]-C[0,1])/S
    else:
        d = np.array([C[0,0],C[1,1],C[2,2]]); i=int(np.argmax(d))
        if i==0:
            S=math.sqrt(1+C[0,0]-C[1,1]-C[2,2])*2
            q0=(C[2,1]-C[1,2])/S; q1=.25*S; q2=(C[0,1]+C[1,0])/S; q3=(C[0,2]+C[2,0])/S
        elif i==1:
            S=math.sqrt(1+C[1,1]-C[0,0]-C[2,2])*2
            q0=(C[0,2]-C[2,0])/S; q1=(C[0,1]+C[1,0])/S; q2=.25*S; q3=(C[1,2]+C[2,1])/S
        else:
            S=math.sqrt(1+C[2,2]-C[0,0]-C[1,1])*2
            q0=(C[1,0]-C[0,1])/S; q1=(C[0,2]+C[2,0])/S; q2=(C[1,2]+C[2,1])/S; q3=.25*S
    q = np.array([q0,q1,q2,q3]); return q/la.norm(q)

def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    q0,qv=q[0],q[1:]; return -qv/(1-q0) if 1+q0<1e-12 else qv/(1+q0)

def mrp_to_dcm(s: np.ndarray) -> np.ndarray:
    s2=float(np.dot(s,s)); I3=np.eye(3)
    if s2<1e-16: return I3.copy()
    S=skew(s); c=1+s2
    return I3 + (8*S@S - 4*(1-s2)*S) / (c*c)

def so3_log(R: np.ndarray) -> np.ndarray:
    c = clamp((np.trace(R)-1.0)/2.0, -1.0, 1.0)
    th = math.acos(c)
    if th < 1e-10: return np.zeros(3)
    return (th/(2.0*math.sin(th))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def so3_exp(phi: np.ndarray) -> np.ndarray:
    th = la.norm(phi)
    if th < 1e-10: return np.eye(3) + skew(phi)
    k = phi/th; K=skew(k)
    return np.eye(3) + math.sin(th)*K + (1-math.cos(th))*(K@K)

def quat_conj(q): return np.array([q[0],-q[1],-q[2],-q[3]])
def quat_mul(q1,q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2])

# ───────────────────────── “look‑at” with roll continuity ─────────────────────────

def camera_pointing_C_BN(u_N: np.ndarray, up_N: np.ndarray, cam_axis: str="z") -> np.ndarray:
    """Instantaneous look-at; +Z_B (or axis) aligns to u_N; up_N sets roll."""
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
    """Continuous roll: choose the roll that stays closest to R_prev."""
    R0 = camera_pointing_C_BN(u_N, up_N, cam_axis)
    if R_prev is None: return R0
    if cam_axis.lower()=="z":
        b_prev = (R_prev.T)[:,0]; b_curr=(R0.T)[:,0]
    elif cam_axis.lower()=="x":
        b_prev = (R_prev.T)[:,1]; b_curr=(R0.T)[:,1]
    else:
        b_prev = (R_prev.T)[:,2]; b_curr=(R0.T)[:,2]
    u = unit(u_N); P = np.eye(3)-np.outer(u,u)
    a = unit(P@b_curr); b = unit(P@b_prev)
    sin_g = np.dot(u, np.cross(a,b)); cos_g=float(np.dot(a,b))
    gamma = math.atan2(sin_g, cos_g)
    return so3_exp(gamma*u) @ R0


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
    thrusters:  List[Thruster]

@dc.dataclass
class SimConfig:
    # timing
    dt        : float = 0.02
    T_end     : float = 240.0
    latency_s : float = 0.050
    seed      : int   = 42

    # PH-SE3 shaping
    shape_ramp_s : float = 6.0
    cmd_smooth_tau_s: float = 0.05
    cmd_smooth_for_s: float = 0.06
    tau_deadband_nm : float = 0.0005
    f_deadband_n    : float = 0.001
    enable_po_pc    : bool  = True
    use_latency_predict: bool = True

    # Attitude-only controllers (torque) — retuned to track look-at
    K_fb  : float = 2.2
    P_fb  : float = 45.0
    Ki_fb : float = 0.03
    I_lim_fb : float = 0.08

    K1_str: float = 1.2
    K3_str: float = 0.0
    wmax  : float = 1.6
    P_rate: float = 65.0
    Ki_rate: float = 0.04
    I_lim_rate: float = 0.06
    use_lpf : bool = True
    lpf_cut_hz: float = 0.8

    # pointing / camera
    CAM_AXIS : str = "z"
    UP_REF_N : np.ndarray = field(default_factory=lambda: np.array([0.0,0.0,1.0]))

CFG = SimConfig()


# ───────────────────────── SE(3) pH tracker (force + torque) ─────────────────────────

class PHSE3:
    def __init__(self, veh: VehicleConfig, dt: float, engage_time: float, cfg: SimConfig):
        self.veh=veh; self.dt=dt; self.engage_time=engage_time; self.cfg=cfg
        J=veh.inertia_B; m=veh.mass_kg
        self.K_R_base = J * (veh.wn_att**2)
        self.D_R      = 2.0*veh.zeta_att*veh.wn_att*J
        self.K_p      = (m*(veh.wn_vel**2)) * np.eye(3)
        self.D_v      = (2.0*veh.zeta_vel*veh.wn_vel*m) * np.eye(3)
        self.KI_R = 0.015*np.diag(np.diag(self.K_R_base)); self.KI_p = 0.06*self.K_p
        self.int_eR=np.zeros(3); self.int_ep=np.zeros(3); self.intR_max=4*np.ones(3); self.intp_max=0.8*np.ones(3)

        # messages
        self.ctrlTorqueMsg = messaging.CmdTorqueBodyMsg()
        self.ctrlForceMsg  = messaging.CmdForceBodyMsg()

        # latency & smoothing
        self.L=int(round(max(0.0,cfg.latency_s)/dt))
        self.use_latency_predict=cfg.use_latency_predict
        self.enable_po_pc=cfg.enable_po_pc
        self.alpha_tau=dt/(cfg.cmd_smooth_tau_s+dt); self.alpha_for=dt/(cfg.cmd_smooth_for_s+dt)
        self.tau_smooth=np.zeros(3); self.f_smooth=np.zeros(3)
        self.tau_dead=cfg.tau_deadband_nm; self.f_dead=cfg.f_deadband_n

        # allocator
        self.B, self.Ts, self.Isps = self._build_wrench_map(veh.thrusters)
        self.delta_prev=None

        # buffers
        self.buf_tau=[]; self.buf_for=[]
        self.omega_prev=np.zeros(3); self.v_prev=np.zeros(3)

        self.torque_apply_hist=[]

    def _build_wrench_map(self, thr: List[Thruster]):
        if not thr: return np.zeros((6,0)), np.zeros(0), np.zeros(0)
        cols=[]; Ts=[]; Isps=[]
        for th in thr:
            r=np.asarray(th.r_B); a=unit(np.asarray(th.a_B))
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
            low=(delta<=lb+1e-9); high=(delta>=ub+1e-9); free=~(low|high)
            if not np.any(free): break
            Bf=B[:,free]; K=la.pinv(Bf.T@W@Bf + lam*np.eye(np.count_nonzero(free)))
            rhs=Bf.T@W@(u - B[:,low]@np.full(np.count_nonzero(low), lb)
                          - B[:,high]@np.full(np.count_nonzero(high), ub)) + lam*delta_prev[free]
            delta_f=K@rhs; delta=delta.copy(); delta[free]=delta_f; delta=np.clip(delta,lb,ub)
        return delta

    def _shape_gain(self, t: float) -> float:
        if self.cfg.shape_ramp_s<=0: return 1.0 if t>=self.engage_time else 0.0
        if t <= self.engage_time: return 0.0
        return clamp((t-self.engage_time)/self.cfg.shape_ramp_s, 0.0, 1.0)

    def _desired_rate_body(self, R: np.ndarray, R_d: np.ndarray) -> np.ndarray:
        if not hasattr(self, "Rd_prev"):
            self.Rd_prev = R_d.copy(); return np.zeros(3)
        Delta = self.Rd_prev.T @ R_d
        phi   = so3_log(Delta)
        omega_d_Bd = phi / self.dt
        C_BBd = R.T @ R_d
        self.Rd_prev = R_d.copy()
        return C_BBd @ omega_d_Bd

    def step(self, t: float,
             attNav: messaging.NavAttMsgPayload,
             transNav: messaging.NavTransMsgPayload,
             R_d: np.ndarray, p_d_N: np.ndarray):
        dt=self.dt; J=self.veh.inertia_B; m=self.veh.mass_kg
        sigma=np.array(attNav.sigma_BN); R=mrp_to_dcm(sigma)
        omega=np.array(attNav.omega_BN_B)
        r_N=np.array(getattr(transNav,'r_BN_N',[0,0,0]),float)
        v_N=np.array(getattr(transNav,'v_BN_N',[0,0,0]),float)
        v_B=R@v_N

        omega_d_B = self._desired_rate_body(R, R_d)
        if self.use_latency_predict:
            omega_p = omega + (omega - self.omega_prev); vB_p = v_B + (v_B - self.v_prev)
            vN_p    = v_N  + (v_N  - R.T@self.v_prev)
        else:
            omega_p=omega; vB_p=v_B; vN_p=v_N

        # attitude & translation errors
        q   = dcm_to_quat(R); qd=dcm_to_quat(R_d)
        qe  = quat_mul(qd, quat_conj(q))
        if qe[0] < 0.0: qe=-qe
        e_R = qe[1:]
        e_ω = omega_p - omega_d_B
        e_p = r_N - p_d_N
        e_v = vN_p

        g   = self._shape_gain(t)
        K_R = self.K_R_base

        tau_pre = - (self.D_R @ e_ω) - g*(K_R @ e_R)
        F_N_pre = - (self.D_v @ e_v) - g*(self.K_p @ e_p)
        f_pre   = R @ F_N_pre

        # wheels + jets (allocator)
        tau_des=tau_pre; f_des=f_pre
        tau_rw=tau_des.copy(); tn=la.norm(tau_rw)
        if tn>self.veh.rw_tau_max>0: tau_rw *= (self.veh.rw_tau_max/tn)
        tau_res=tau_des - tau_rw

        if la.norm(tau_res) < 1e-9 and la.norm(f_des) < 1e-9:
            tau_thr=np.zeros(3); f_thr=np.zeros(3)
        else:
            if self.B.size>0:
                u=np.hstack((tau_res,f_des))
                delta=self._bnls(self.B,u,getattr(self,"delta_prev",None),self.veh.w_torque,self.veh.w_force,self.veh.reg_lambda,0.0,1.0)
                self.delta_prev=delta.copy()
                uf=self.B@delta; tau_thr=uf[:3]; f_thr=uf[3:]
            else:
                tau_thr=np.zeros(3); f_thr=np.zeros(3)

        tau_apply=tau_rw + tau_thr
        # publish
        t_pl=messaging.CmdTorqueBodyMsgPayload(); t_pl.torqueRequestBody=list(tau_apply)
        f_pl=messaging.CmdForceBodyMsgPayload();  f_pl.forceRequestBody =list(f_thr)
        self.ctrlTorqueMsg.write(t_pl); self.ctrlForceMsg.write(f_pl)

        self.torque_apply_hist.append(tau_apply.copy())
        self.omega_prev=omega.copy(); self.v_prev=v_B.copy()


# ───────────────────────── RAFTI separation (unchanged) ─────────────────────────

@dc.dataclass
class SepParams:
    t_start: float
    t_end  : float
    F0     : float
    s_car_B: np.ndarray
    r_car_B: np.ndarray
    r_dep_B: np.ndarray
    dv_cmd : float

class RAFTISeparation:
    def __init__(self, dt: float,
                 sc_car: spacecraft.Spacecraft, eff_car: extForceTorque.ExtForceTorque,
                 sc_dep: List[spacecraft.Spacecraft], eff_dep: List[extForceTorque.ExtForceTorque],
                 sched: List[SepParams], m_car: float, m_dep: float):
        self.dt=dt; self.sched=sched
        self.sc_car=sc_car; self.eff_car=eff_car
        self.sc_dep=sc_dep; self.eff_dep=eff_dep
        self.car_tauMsg=messaging.CmdTorqueBodyMsg()
        self.car_forMsg=messaging.CmdForceBodyMsg()
        self.dep_tauMsg=[messaging.CmdTorqueBodyMsg() for _ in sc_dep]
        self.dep_forMsg=[messaging.CmdForceBodyMsg() for _ in sc_dep]
        eff_car.cmdTorqueInMsg   .subscribeTo(self.car_tauMsg)
        eff_car.cmdForceBodyInMsg.subscribeTo(self.car_forMsg)
        for ef,tm,fm in zip(eff_dep,self.dep_tauMsg,self.dep_forMsg):
            ef.cmdTorqueInMsg   .subscribeTo(tm)
            ef.cmdForceBodyInMsg.subscribeTo(fm)

    @staticmethod
    def _half_sine(t, t0, T, F0):
        if t<t0 or t>t0+T: return 0.0
        return F0*math.sin(math.pi*(t-t0)/T)

    def step(self, t: float):
        C_BN_car = mrp_to_dcm(np.array(self.sc_car.scStateOutMsg.read().sigma_BN))
        C_NB_car = C_BN_car.T
        f_car=np.zeros(3)
        for i,sp in enumerate(self.sched):
            C_BN_dep = mrp_to_dcm(np.array(self.sc_dep[i].scStateOutMsg.read().sigma_BN))
            C_NB_dep = C_BN_dep.T
            Fmag = self._half_sine(t, sp.t_start, sp.t_end-sp.t_start, sp.F0)
            if Fmag<=0:
                self.dep_tauMsg[i].write(messaging.CmdTorqueBodyMsgPayload())
                self.dep_forMsg[i].write(messaging.CmdForceBodyMsgPayload()); continue
            s_N = C_NB_car @ sp.s_car_B
            F_N = Fmag*s_N
            f_car_i_B =  C_BN_car @ (+F_N)
            f_dep_i_B =  C_BN_dep @ (-F_N)
            tau_dep   = np.cross(sp.r_dep_B, f_dep_i_B)
            tm=messaging.CmdTorqueBodyMsgPayload(); tm.torqueRequestBody=list(tau_dep)
            fm=messaging.CmdForceBodyMsgPayload();  fm.forceRequestBody =list(f_dep_i_B)
            self.dep_tauMsg[i].write(tm); self.dep_forMsg[i].write(fm)
            f_car   += f_car_i_B
        self.car_tauMsg.write(messaging.CmdTorqueBodyMsgPayload()) # zero torque
        fm=messaging.CmdForceBodyMsgPayload(); fm.forceRequestBody=list(f_car)
        self.car_forMsg.write(fm)


# ───────────────────────── Case builder (one controller per run) ─────────────────────────

class Case:
    def __init__(self, kind: str, cfg: SimConfig):
        """kind ∈ {'PH_SE3','MRP_FB','PID'}"""
        self.kind = kind; self.cfg=cfg; self.dt=cfg.dt
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

        # Deputy (XB6 6U) + 12-jet layout (for PH_SE3 force authority)
        I_dep=np.diag([0.025,0.020,0.018]); m_dep=12.0
        T_small, Isp_small = 0.27, 217.0; L = 0.065
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

        # ESPA port geometry (port at angle 0 on ring)
        ring_R = self.espa_radius; z_mount=0.0; ang=0.0
        self.espa_port_r_B   = np.array([ ring_R*math.cos(ang), ring_R*math.sin(ang), z_mount ], float)
        self.espa_port_dir_B = np.array([ math.cos(ang),        math.sin(ang),        0.0    ], float)

        # initial deputy pose aligned to port-local frame
        bY = self.espa_port_dir_B; bZ=np.array([0,0,1.0])
        if abs(np.dot(bY/la.norm(bY),bZ))>0.99: bZ=np.array([1,0,0])
        bX = unit(np.cross(bY,bZ)); bZ = unit(np.cross(bX,bY))
        R_BD_BC = np.column_stack((bX,bY,bZ))
        sigma0 = quat_to_mrp(dcm_to_quat(R_BD_BC))

        self.dep = VehicleConfig(
            name="XB6_6U", mass_kg=m_dep, inertia_B=I_dep,
            sigma0_BN=sigma0, omega0_B=np.zeros(3),
            r0_N=self.espa_port_r_B.copy(), v0_N=np.zeros(3),
            zeta_att=1.25, wn_att=1.00, zeta_vel=1.10, wn_vel=0.30,
            rw_tau_max=0.08, w_force=220.0, w_torque=60.0, reg_lambda=5e-4, thrusters=thr_d
        )

        # tasks
        self.dyn = sys_model_task.SysModelTask(int(macros.sec2nano(self.dt)))
        self.sns = sys_model_task.SysModelTask(int(macros.sec2nano(self.dt)))

        # Carrier objects
        self.sc_car = spacecraft.Spacecraft(); self.sc_car.ModelTag=f"SC_car_{kind}"
        self.sc_car.hub.mHub=self.car.mass_kg; self.sc_car.hub.IHubPntBc_B=self.car.inertia_B.tolist()
        self.sc_car.hub.sigma_BNInit=list(self.car.sigma0_BN)
        self.sc_car.hub.omega_BN_BInit=list(self.car.omega0_B)
        self.sc_car.hub.r_CN_NInit=list(self.car.r0_N); self.sc_car.hub.v_CN_NInit=list(self.car.v0_N)
        self.nav_car = simpleNav.SimpleNav(); self.nav_car.scStateInMsg.subscribeTo(self.sc_car.scStateOutMsg)
        self.eft_car_ctrl = extForceTorque.ExtForceTorque(); self.eft_car_sep=extForceTorque.ExtForceTorque()
        self.sc_car.addDynamicEffector(self.eft_car_ctrl); self.sc_car.addDynamicEffector(self.eft_car_sep)

        # Deputy objects
        self.sc_dep = spacecraft.Spacecraft(); self.sc_dep.ModelTag=f"SC_dep_{kind}"
        self.sc_dep.hub.mHub=self.dep.mass_kg; self.sc_dep.hub.IHubPntBc_B=self.dep.inertia_B.tolist()
        self.sc_dep.hub.sigma_BNInit=list(self.dep.sigma0_BN); self.sc_dep.hub.omega_BN_BInit=list(self.dep.omega0_B)
        self.sc_dep.hub.r_CN_NInit=list(self.dep.r0_N); self.sc_dep.hub.v_CN_NInit=list(self.dep.v0_N)
        self.nav_dep = simpleNav.SimpleNav(); self.nav_dep.scStateInMsg.subscribeTo(self.sc_dep.scStateOutMsg)
        self.eft_dep_ctrl=extForceTorque.ExtForceTorque(); self.eft_dep_sep=extForceTorque.ExtForceTorque()
        self.sc_dep.addDynamicEffector(self.eft_dep_ctrl); self.sc_dep.addDynamicEffector(self.eft_dep_sep)

        # add to tasks
        p=0
        self.sns.AddNewObject(self.nav_car,10+p); self.dyn.AddNewObject(self.sc_car,5+p)
        self.dyn.AddNewObject(self.eft_car_ctrl,10+p); self.dyn.AddNewObject(self.eft_car_sep,11+p)
        self.sns.AddNewObject(self.nav_dep,20+p); self.dyn.AddNewObject(self.sc_dep,15+p)
        self.dyn.AddNewObject(self.eft_dep_ctrl,20+p); self.dyn.AddNewObject(self.eft_dep_sep,21+p)

        # Controllers
        last_sep_end = 10.0 + 0.30
        # Carrier: mild damping
        self.ctrl_car = PHSE3(self.car, self.dt, last_sep_end, CFG)
        self.eft_car_ctrl.cmdTorqueInMsg   .subscribeTo(self.ctrl_car.ctrlTorqueMsg)
        self.eft_car_ctrl.cmdForceBodyInMsg.subscribeTo(self.ctrl_car.ctrlForceMsg)

        if kind == "PH_SE3":
            self.ctrl_dep = PHSE3(self.dep, self.dt, last_sep_end, CFG)
            self.eft_dep_ctrl.cmdTorqueInMsg   .subscribeTo(self.ctrl_dep.ctrlTorqueMsg)
            self.eft_dep_ctrl.cmdForceBodyInMsg.subscribeTo(self.ctrl_dep.ctrlForceMsg)
            self.dep_mode = "ph"

        elif kind == "MRP_FB":
            self.err = attTrackingError.attTrackingError()
            self.err.attNavInMsg.subscribeTo(self.nav_dep.attOutMsg)
            self.ref = messaging.AttRefMsg()
            self.err.attRefInMsg.subscribeTo(self.ref)

            ctl = mrpFeedback.mrpFeedback()
            ctl.K  = CFG.K_fb
            ctl.P  = CFG.P_fb
            ctl.Ki = CFG.Ki_fb
            ctl.integralLimit = CFG.I_lim_fb
            vehCfg = messaging.VehicleConfigMsgPayload()
            vehCfg.ISCPntB_B = [self.dep.inertia_B[0,0],0,0, 0,self.dep.inertia_B[1,1],0, 0,0,self.dep.inertia_B[2,2]]
            self._vehCfgMsg = messaging.VehicleConfigMsg().write(vehCfg)
            ctl.vehConfigInMsg.subscribeTo(self._vehCfgMsg)
            ctl.guidInMsg.subscribeTo(self.err.attGuidOutMsg)

            if cfg.use_lpf:
                lpf = lowPassFilterTorqueCommand.lowPassFilterTorqueCommand()
                lpf.h = cfg.dt; lpf.wc = 2*np.pi*cfg.lpf_cut_hz
                lpf.cmdTorqueInMsg.subscribeTo(ctl.cmdTorqueOutMsg)
                self.eft_dep_ctrl.cmdTorqueInMsg.subscribeTo(lpf.cmdTorqueOutMsg)
                self._lpf = lpf
                self._srcTorqueMsg_fb = lpf.cmdTorqueOutMsg
            else:
                self.eft_dep_ctrl.cmdTorqueInMsg.subscribeTo(ctl.cmdTorqueOutMsg)
                self._lpf = None
                self._srcTorqueMsg_fb = ctl.cmdTorqueOutMsg

            self._ctl_fb = ctl
            self.dep_mode = "fb"

            self.fswtask = sys_model_task.SysModelTask(int(macros.sec2nano(self.dt)))
            self.fswtask.AddNewObject(self.err,30)
            self.fswtask.AddNewObject(self._ctl_fb,20)
            if self._lpf: self.fswtask.AddNewObject(self._lpf,15)

        elif kind == "PID":
            self.err = attTrackingError.attTrackingError()
            self.err.attNavInMsg.subscribeTo(self.nav_dep.attOutMsg)
            self.ref = messaging.AttRefMsg()
            self.err.attRefInMsg.subscribeTo(self.ref)

            steer = mrpSteering.mrpSteering()
            steer.K1 = CFG.K1_str
            steer.K3 = CFG.K3_str
            steer.omega_max = CFG.wmax
            steer.guidInMsg.subscribeTo(self.err.attGuidOutMsg)

            servo = rateServoFullNonlinear.rateServoFullNonlinear()
            servo.P = CFG.P_rate
            servo.Ki = CFG.Ki_rate
            servo.integralLimit = CFG.I_lim_rate
            vehCfg = messaging.VehicleConfigMsgPayload()
            vehCfg.ISCPntB_B = [self.dep.inertia_B[0,0],0,0, 0,self.dep.inertia_B[1,1],0, 0,0,self.dep.inertia_B[2,2]]
            self._vehCfgMsg = messaging.VehicleConfigMsg().write(vehCfg)
            servo.vehConfigInMsg.subscribeTo(self._vehCfgMsg)
            servo.rateSteeringInMsg.subscribeTo(steer.rateCmdOutMsg)
            servo.guidInMsg.subscribeTo(self.err.attGuidOutMsg)

            if cfg.use_lpf:
                lpf = lowPassFilterTorqueCommand.lowPassFilterTorqueCommand()
                lpf.h = cfg.dt; lpf.wc = 2*np.pi*cfg.lpf_cut_hz
                lpf.cmdTorqueInMsg.subscribeTo(servo.cmdTorqueOutMsg)
                self.eft_dep_ctrl.cmdTorqueInMsg.subscribeTo(lpf.cmdTorqueOutMsg)
                self._lpf = lpf
                self._srcTorqueMsg_pid = lpf.cmdTorqueOutMsg
            else:
                self.eft_dep_ctrl.cmdTorqueInMsg.subscribeTo(servo.cmdTorqueOutMsg)
                self._lpf=None
                self._srcTorqueMsg_pid = servo.cmdTorqueOutMsg

            self._steer=steer; self._servo=servo
            self.dep_mode = "pid"

            self.fswtask = sys_model_task.SysModelTask(int(macros.sec2nano(self.dt)))
            self.fswtask.AddNewObject(self.err,30)
            self.fswtask.AddNewObject(self._steer,25)
            self.fswtask.AddNewObject(self._servo,20)
            if self._lpf: self.fswtask.AddNewObject(self._lpf,15)

        else:
            raise ValueError("Unknown controller kind")

        # RAFTI schedule (seeded; identical per run)
        rng = np.random.default_rng(cfg.seed)
        self.t0_base=10.0; Tpush=0.30
        dv = float(np.clip(rng.normal(0.018, 0.005), 0.004, 0.025))
        F0 = self.dep.mass_kg * dv * math.pi / (2.0*Tpush)
        r_dep_B = rng.normal(scale=0.02, size=3)
        self.sched=[ SepParams(t_start=self.t0_base, t_end=self.t0_base+Tpush, F0=F0,
                               s_car_B=self.espa_port_dir_B.copy(),
                               r_car_B=self.espa_port_r_B.copy(),
                               r_dep_B=r_dep_B, dv_cmd=dv) ]
        self.sep = RAFTISeparation(self.dt, self.sc_car, self.eft_car_sep,
                                   [self.sc_dep], [self.eft_dep_sep],
                                   self.sched, m_car=self.car.mass_kg, m_dep=self.dep.mass_kg)

        # initialize tasks
        for t in (self.sns, self.dyn):
            t.SelfInitTaskList(); t.ResetTaskList(0)
        if hasattr(self, "fswtask"):
            self.fswtask.SelfInitTaskList(); self.fswtask.ResetTaskList(0)

        # book‑keeping for reference‑rate generation and metrics
        self.Rd_cont_prev=None
        self.prev_C_RN = mrp_to_dcm(self.dep.sigma0_BN)
        self.prev_omega_N = np.zeros(3)
        self.last_omega_RN_N = np.zeros(3)
        self.p_ref_frozen=False; self.p_d_N_frozen=np.zeros(3)

        # prime reference once to avoid "unwritten" errors (FB/PID only)
        if self.dep_mode in ("fb","pid"):
            pl = messaging.AttRefMsgPayload()
            pl.sigma_RN = list(quat_to_mrp(dcm_to_quat(self.prev_C_RN)))
            pl.omega_RN_N = [0.0,0.0,0.0]; pl.domega_RN_N=[0.0,0.0,0.0]
            self.ref.write(pl)

    def _port_position_in_N(self):
        C_BN_car = mrp_to_dcm(np.array(self.sc_car.scStateOutMsg.read().sigma_BN))
        C_NB_car = C_BN_car.T
        r_car_N  = np.array(self.nav_car.transOutMsg.read().r_BN_N, float)
        return r_car_N + C_NB_car @ self.espa_port_r_B

    def _camera_Rd_cont(self, r_dep_N: np.ndarray):
        r_port_N = self._port_position_in_N()
        los_N = r_port_N - r_dep_N
        if la.norm(los_N) < 1e-8: los_N=np.array([1.0,0.0,0.0])
        Rd = look_at_continuous(los_N, CFG.UP_REF_N, self.Rd_cont_prev, cam_axis=CFG.CAM_AXIS)
        self.Rd_cont_prev = Rd.copy()
        return Rd, los_N

    @staticmethod
    def _safe_tau_from(msg) -> np.ndarray:
        try:
            pl = msg.read()
            arr = np.array(pl.torqueRequestBody, dtype=float)
            if not np.all(np.isfinite(arr)): return np.zeros(3)
            return arr
        except Exception:
            return np.zeros(3)

    def run(self) -> Dict[str,np.ndarray]:
        now_ns=0; N=int(self.cfg.T_end/self.dt)
        last_sep_end = self.t0_base + 0.30
        bar = tqdm(range(N), desc=f"ESPA(42in)+XB6 — {self.kind}", ncols=80)

        # logs
        t_arr=[]; rpy=[]; rpy_d=[]; miss=[]; roll_los=[]
        tau_mag=[]; H_err=[]

        for k in bar:
            t=k*self.dt
            # sense + separation
            self.sns.ExecuteTaskList(now_ns)
            self.sep.step(t)

            # carrier (mild damping)
            self.ctrl_car.step(t, self.nav_car.attOutMsg.read(), self.nav_car.transOutMsg.read(),
                               R_d=np.eye(3), p_d_N=np.zeros(3))

            # deputy reference
            r_dep_N = np.array(self.nav_dep.transOutMsg.read().r_BN_N, float)
            if (not self.p_ref_frozen) and t >= last_sep_end:
                self.p_d_N_frozen = r_dep_N.copy(); self.p_ref_frozen=True
            p_ref = self.p_d_N_frozen if self.p_ref_frozen else r_dep_N.copy()

            if t >= last_sep_end:
                R_d, los_N = self._camera_Rd_cont(r_dep_N)
            else:
                R_d = mrp_to_dcm(self.dep.sigma0_BN); los_N=None

            # correct reference rates for FB/PID
            Delta = R_d @ self.prev_C_RN.T
            omega_RN_R  = so3_log(Delta) / self.dt
            omega_RN_N  = R_d.T @ omega_RN_R
            domega_RN_N = (omega_RN_N - self.prev_omega_N) / self.dt
            self.prev_C_RN   = R_d
            self.prev_omega_N = omega_RN_N
            self.last_omega_RN_N = omega_RN_N

            if self.dep_mode == "ph":
                self.ctrl_dep.step(t, self.nav_dep.attOutMsg.read(), self.nav_dep.transOutMsg.read(),
                                   R_d=R_d, p_d_N=p_ref)
            else:
                pl = messaging.AttRefMsgPayload()
                pl.sigma_RN    = list(quat_to_mrp(dcm_to_quat(R_d)))
                pl.omega_RN_N  = list(omega_RN_N)
                pl.domega_RN_N = list(domega_RN_N)
                self.ref.write(pl)
                self.fswtask.ExecuteTaskList(now_ns)

            # dynamics step
            self.dyn.ExecuteTaskList(now_ns)
            now_ns += int(macros.sec2nano(self.dt))

            # logs for overlays (robust Euler)
            C_BN = mrp_to_dcm(np.array(self.sc_dep.scStateOutMsg.read().sigma_BN))
            rpy_i  = dcm_to_euler_zyx(C_BN)
            rpy_di = dcm_to_euler_zyx(R_d)
            diff = (rpy_i - rpy_di + np.pi)%(2*np.pi) - np.pi
            t_arr.append(t); rpy.append(rpy_di + diff); rpy_d.append(rpy_di)

            if los_N is not None:
                zB_N = C_BN.T[:,2] if CFG.CAM_AXIS.lower()=="z" else (C_BN.T[:,0] if CFG.CAM_AXIS.lower()=="x" else C_BN.T[:,1])
                u = unit(los_N)
                miss.append(math.degrees(math.acos(clamp(float(np.dot(zB_N,u)),-1.0,1.0))))
                if CFG.CAM_AXIS.lower()=="z":
                    xB_N = C_BN.T[:,0]; xBd_N = R_d.T[:,0]
                elif CFG.CAM_AXIS.lower()=="x":
                    xB_N = C_BN.T[:,1]; xBd_N = R_d.T[:,1]
                else:
                    xB_N = C_BN.T[:,2]; xBd_N = R_d.T[:,2]
                P = np.eye(3) - np.outer(u,u)
                a=unit(P@xB_N); b=unit(P@xBd_N)
                roll_los.append(math.degrees(math.atan2(la.norm(np.cross(a,b)), float(np.dot(a,b)))))
            else:
                miss.append(np.nan); roll_los.append(np.nan)

            # torque magnitude & momentum-tracking error (safe)
            if self.dep_mode == "ph":
                tau = self.ctrl_dep.torque_apply_hist[-1] if self.ctrl_dep.torque_apply_hist else np.zeros(3)
            elif self.dep_mode == "fb":
                tau = self._safe_tau_from(self._srcTorqueMsg_fb)
            else:
                tau = self._safe_tau_from(self._srcTorqueMsg_pid)
            tau_mag.append(la.norm(tau))

            w_B = np.array(self.nav_dep.attOutMsg.read().omega_BN_B, float)
            H_err.append(la.norm(self.dep.inertia_B @ (w_B - (C_BN @ self.last_omega_RN_N))))

        # package logs
        L: Dict[str,np.ndarray] = {}
        L["t"]     = np.array(t_arr)
        rpy = np.array(rpy); rpy_d = np.array(rpy_d)
        L["roll"], L["pitch"], L["yaw"]   = rpy[:,0], rpy[:,1], rpy[:,2]
        L["roll_d"], L["pitch_d"], L["yaw_d"] = rpy_d[:,0], rpy_d[:,1], rpy_d[:,2]
        L["miss"]  = np.array(miss); L["roll_los"]=np.array(roll_los)
        L["tau_mag"]=np.array(tau_mag); L["H_err"]=np.array(H_err)

        # RMS summaries
        tau_rms = float(np.sqrt(np.mean(L["tau_mag"]**2))) if L["tau_mag"].size else 0.0
        H_rms   = float(np.sqrt(np.mean(L["H_err"]**2)))   if L["H_err"].size   else 0.0
        print(f"[{self.kind}] torque_rms={tau_rms:.4f} N·m, momentum_err_rms={H_rms:.4f} N·m·s")

        return L


# ───────────────────────── plots ─────────────────────────

def plot_overlays(logs: Dict[str,Dict[str,np.ndarray]]):
    colors = dict(PH_SE3="#d62728", MRP_FB="#1f77b4", PID="#2ca02c")
    order  = ["PH_SE3","MRP_FB","PID"]

    # Euler
    fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
    axs[0].set_title("Euler ZYX (wrapped to desired)")
    for name in order: axs[0].plot(logs[name]["t"], logs[name]["roll"], color=colors[name], label=name)
    axs[0].plot(logs[order[0]]["t"], logs[order[0]]["roll_d"], 'k--', lw=1, label="desired")
    axs[0].set_ylabel("roll ϕ [rad]"); axs[0].grid(True); axs[0].legend()

    for name in order: axs[1].plot(logs[name]["t"], logs[name]["pitch"], color=colors[name])
    axs[1].plot(logs[order[0]]["t"], logs[order[0]]["pitch_d"], 'k--', lw=1)
    axs[1].set_ylabel("pitch θ [rad]"); axs[1].grid(True)

    for name in order: axs[2].plot(logs[name]["t"], logs[name]["yaw"], color=colors[name])
    axs[2].plot(logs[order[0]]["t"], logs[order[0]]["yaw_d"], 'k--', lw=1)
    axs[2].set_ylabel("yaw ψ [rad]"); axs[2].set_xlabel("time [s]"); axs[2].grid(True)
    plt.tight_layout(); plt.show()

    # Pointing
    fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True)
    ax[0].set_title("Pointing to ESPA port")
    for name in order: ax[0].plot(logs[name]["t"], logs[name]["miss"], color=colors[name], label=name)
    ax[0].axhline(1.0,ls="--",c="k",lw=0.8); ax[0].set_ylabel("boresight miss [deg]"); ax[0].legend(); ax[0].grid(True)
    for name in order: ax[1].plot(logs[name]["t"], logs[name]["roll_los"], color=colors[name])
    ax[1].axhline(2.0,ls="--",c="k",lw=0.8); ax[1].set_ylabel("roll about LOS [deg]"); ax[1].set_xlabel("time [s]"); ax[1].grid(True)
    plt.tight_layout(); plt.show()

    # Actuation & momentum error
    fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True)
    ax[0].set_title("Torque magnitude & Momentum error")
    for name in order: ax[0].plot(logs[name]["t"], logs[name]["tau_mag"], color=colors[name], label=name)
    ax[0].set_ylabel("|τ| [N·m]"); ax[0].legend(); ax[0].grid(True)
    for name in order: ax[1].plot(logs[name]["t"], logs[name]["H_err"], color=colors[name])
    ax[1].set_ylabel("||J(ω−ω_ref)|| [N·m·s]"); ax[1].set_xlabel("time [s]"); ax[1].grid(True)
    plt.tight_layout(); plt.show()


# ───────────────────────── main ─────────────────────────

def main():
    kinds = ["PH_SE3", "MRP_FB", "PID"]
    all_logs={}
    for k in kinds:
        case = Case(k, CFG)
        logs = case.run()
        all_logs[k]=logs
    plot_overlays(all_logs)
    print("[DONE] ESPA(42\") + RAFTI + XB6 — controller comparison complete (plots shown).")

if __name__=="__main__":
    main()
