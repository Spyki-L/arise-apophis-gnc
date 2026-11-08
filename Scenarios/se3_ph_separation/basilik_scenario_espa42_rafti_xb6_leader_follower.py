#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_scenario_espa42_rafti_xb6_leader_follower.py

Purpose
-------
Basilisk scenario that compares a “leader” SE(3) geometric port–Hamiltonian
( PH_SE3 ) controller against two “follower” torque-only controllers on the
same ESPA 42-inch + RAFTI + XB6 6U deputy setup. The leader tracks a continuous
look-at attitude, and its *achieved* trajectory is then used as a delayed,
softened reference for the followers:

- Leader:
    PH_SE3  — geometric SE(3) pH tracker (force + torque, wheels + jets)

- Followers (20% “worse” than PH_SE3 on purpose):
    MRP_FB  — scratch MRP + rate + integral torque controller
    PID     — scratch mrpSteering-like + nonlinear rate servo

All three runs share identical separation dynamics (same RAFTI schedule) so the
comparison isolates controller behavior. Each run logs deputy-1 data into a
separate NPZ file:

    deputy1_PH_SE3.npz
    deputy1_MRP_FB.npz
    deputy1_PID.npz

These NPZs are designed to be consumed by post-processing tools such as
`fleet_dashboard_attitude_torque_nav_errors.py` and
`replay_pitch_shaping_pid_mrp_vs_ph.py`.

Scenario structure
------------------
- Geometry:
    * Single ESPA 42" ring (radius ≈ 0.5334 m) with one active port.
    * Carrier bus at the center; XB6 deputy mounted at a single port.
    * Local port frame chosen so the deputy initially “docks” to the port:
         +Y_B along the ESPA radial direction,
         +Z_B ~ +Z_carrier,
         +X_B completes the triad.

- Carrier:
    * 650 kg with diagonal inertia [1200, 1000, 800] kg·m².
    * Four MR-22N-class thrusters.
    * Controlled by a PH_SE3 layer to lightly damp attitude/translation and
      keep the host bus well behaved during separation.

- Deputy (XB6 6U):
    * 12 kg with diagonal inertia [0.025, 0.020, 0.018] kg·m².
    * 12-jet configuration providing small torque/force capability.

RAFI separation:
- Single half-sine push from the ESPA port:
    * Start time t0_base ≈ 10.0 s, duration ≈ 0.30 s.
    * Δv_cmd drawn from a clipped normal distribution, converted to peak
      force F0 via equivalent-mass formula.
    * Equal-and-opposite forces and torques applied to carrier and deputy
      through `extForceTorque` modules.

Leader controller (PH_SE3)
--------------------------
- Full SE(3) geometric port–Hamiltonian layer:
    * Rotational gains: K_R ∝ J ω_n_att², D_R ∝ J ζ_att ω_n_att.
    * Translational gains: K_p ∝ m ω_n_vel², D_v ∝ m ζ_vel ω_n_vel.
    * Integral terms on attitude and position with anti-windup.
    * Latency prediction and optional PO/PC in the “leader” path are
      implemented in the PHSE3 class.
- Reference:
    * Look-at attitude to the ESPA port, built via:
          camera_pointing_C_BN() + look_at_continuous()
      with +Z_B as camera axis and a fixed UP_REF_N for roll definition.
    * Position reference p_d_N freezes at the deputy’s location at the end
      of the RAFTI push.
- Logs:
    * Time history of σ_BN, ω_B, Euler ZYX (wrapped to desired), LOS metrics,
      torque histories, and momentum tracking error.

Follower controllers
--------------------
Both followers use the *achieved* PH_SE3 trajectory as a reference, but with:
- A fixed time lag (seconds) chosen so followers settle later than the leader.
- Gain scaling (follower_gain_scale < 1) to make them slower/weaker.

Reference construction:
- After the leader run, the script computes:
    * σ_lead(t) from leader’s σ_BN.
    * ω_lead,N(t) by rotating ω_lead,B into the inertial frame.
    * dω_lead,N/dt via finite differences.
- A leader_ref dict is built with:
    t, dt, sigma, omega_N, domega_N, lag_sec.
- Followers sample (R_d, ω_RN_N, dω_RN_N) from leader_ref with the chosen lag.

Follower 1: MRPFeedbackScratch
- Control law:
      τ = -K σ_BR - P ω_BR - K_I ∫σ_BR dt
  with:
    * σ_BR from relative attitude C_BR = C_BN C_RN^T,
    * ω_BR = ω_B - C_BN ω_RN_N,
    * integral term with back-calculation anti-windup.
- Gains (K, P, Ki, LPF cutoff, and torque limits) are scaled down by
  follower_gain_scale.
- Optional low-pass filter on torque commands.
- Per-axis torque saturation and deadband.

Follower 2: PIDRateScratch
- Approximate mrpSteering + rateServoFullNonlinear behavior:
    * “Steering” stage:
        - Nonlinear MRP-based steering law with K1, K3 and ω_max limits.
        - Generates ω_steer such that MRPs converge with bounded rates.
    * “Rate servo” stage:
        - Targets ω_cmd = ω_RN_B + ω_steer and uses a nonlinear rate servo:
              τ = -P e_ω - K_I ∫e_ω dt - ω × Jω + J dω_cmd/dt
        - Integral term with back-calculation anti-windup.
- All key gains (K1, K3, wmax, P_rate, Ki_rate, LPF cutoff) and torque limits
  are scaled by follower_gain_scale.

Outputs and NPZ logging
-----------------------
At the end of each case (kind ∈ {'PH_SE3','MRP_FB','PID'}) the script:

- Computes RMS metrics:
    * RMS torque magnitude |τ|_rms.
    * RMS momentum tracking error ‖J(ω − ω_ref)‖_rms.
- Saves a deputy-1 NPZ file:
    deputy1_{kind}.npz  containing:
      * t
      * roll, pitch, yaw (wrapped to desired)
      * roll_d, pitch_d, yaw_d (desired)
      * miss            : boresight miss [deg] vs ESPA port
      * roll_los        : roll-about-LOS error [deg]
      * tau_mag         : |τ| [N·m]
      * H_err           : ‖J(ω − ω_ref)‖ [N·m·s]
      * sigma_BN        : MRPs
      * omega_B         : body rates
      * tau_vec         : torque vector [N·m]
      * sigma_d         : desired MRPs
      * omega_ref_N     : reference rates in N
      * omega_ref_B     : reference rates in B

Plotting
--------
After all three runs, the script overlays:

- Euler ZYX (wrapped to desired) for PH_SE3, MRP_FB, PID plus the common
  desired trajectory.
- Pointing metrics (boresight and roll-about-LOS) vs time for all three.
- Torque magnitude and momentum tracking error vs time for all three.

Usage
-----
Run in a Basilisk-enabled Python environment:

    python basilik_scenario_espa42_rafti_xb6_leader_follower.py

The script will:
- Run the PH_SE3 leader, compute its effective settling time and reference.
- Run MRP_FB and PID followers using delayed, gain-scaled leader references.
- Save three NPZ files (deputy1_PH_SE3.npz, deputy1_MRP_FB.npz, deputy1_PID.npz).
- Print RMS torque/momentum errors and display comparison plots.
"""


from __future__ import annotations
import math, dataclasses as dc
from dataclasses import field
from typing import List, Optional, Dict
import numpy as np, numpy.linalg as la
import matplotlib.pyplot as plt
import copy

# Basilisk (simulation only; no fswAlgorithms)
from Basilisk.architecture import messaging, sys_model_task
from Basilisk.simulation   import spacecraft, simpleNav, extForceTorque
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
    C = np.asarray(C, float)
    if C.shape != (3,3) or not np.all(np.isfinite(C)):
        return np.eye(3)
    try:
        U,S,Vt = la.svd(C); R = U @ Vt
        if la.det(R) < 0.0: U[:,-1]*=-1.0; R = U @ Vt
        return R
    except la.LinAlgError:
        x = C[:,0]; x = x/la.norm(x) if la.norm(x)>1e-12 else np.array([1.0,0.,0.])
        y = C[:,1] - x*np.dot(x,C[:,1]); y = y/la.norm(y) if la.norm(y)>1e-12 else np.array([0.,1.,0.])
        z = np.cross(x,y); z = z/la.norm(z) if la.norm(z)>1e-12 else np.array([0.,0.,1.])
        R = np.column_stack((x,y,z))
        if la.det(R) < 0.0: R[:,2]*=-1.0
        return R

def clamp(x, lo, hi): return lo if x<lo else (hi if x>hi else x)

def dcm_to_euler_zyx(C_BN: np.ndarray):
    C = proj_SO3(np.asarray(C_BN, float))
    c20 = clamp(C[2,0], -1.0, 1.0)
    th = -math.asin(c20)
    ph = math.atan2(clamp(C[2,1],-1.0,1.0), clamp(C[2,2],-1.0,1.0))
    ps = math.atan2(clamp(C[1,0],-1.0,1.0), clamp(C[0,0],-1.0,1.0))
    return np.array([ph, th, ps])

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

def B_sigma(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float); s2 = float(s @ s)
    return (1 - s2) * np.eye(3) + 2*skew(s) + 2*np.outer(s, s)

def Binv_sigma(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float); s2 = float(s @ s)
    return ( (1 - s2) * np.eye(3) - 2*skew(s) + 2*np.outer(s, s) ) / (1 + s2)


# ───────────────────────── “look‑at” with roll continuity ─────────────────────────

def camera_pointing_C_BN(u_N: np.ndarray, up_N: np.ndarray, cam_axis: str="z") -> np.ndarray:
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
    f_deadband_n    : float = 0.002
    enable_po_pc    : bool  = True
    use_latency_predict: bool = True

    # Scratch torque-only controller gains (nominal)
    K_fb  : float = 6.0
    P_fb  : float = 80.0
    Ki_fb : float = 0.12
    I_lim_fb : float = 0.30

    K1_str: float = 1.6
    K3_str: float = 0.2
    wmax  : float = 1.0
    P_rate: float = 40.0
    Ki_rate: float = 0.10
    I_lim_rate: float = 0.20

    use_lpf : bool = True
    lpf_cut_hz: float = 0.6

    # torque cap only for scratch controllers
    ctrl_tau_limit_nm: float = 0.25

    # follower behavior (vs PH_SE3)
    use_ph_reference_for_followers: bool = True
    follower_gain_scale: float = 0.99     # 20% weaker/slower
    follower_delay_frac: float = 0.005    # 20% of PH settling time
    follower_min_delay_s: float = 0.5    # ensure they finish after PH_SE3

    # pointing / camera
    CAM_AXIS : str = "z"
    UP_REF_N : np.ndarray = field(default_factory=lambda: np.array([0.0,0.0,1.0]))

CFG = SimConfig()


# ───────────────────────── LPF ─────────────────────────

class TorqueLPF:
    def __init__(self, dt: float, cut_hz: float):
        wc = 2*math.pi*max(1e-6, cut_hz)
        tau = 1.0/wc
        self.alpha = dt/(tau + dt)
        self.y = np.zeros(3)
    def reset(self): self.y[:] = 0.0
    def filt(self, u: np.ndarray) -> np.ndarray:
        self.y = self.y + self.alpha*(u - self.y)
        return self.y


# ───────────────────────── PH‑SE3 (leader) — unchanged ─────────────────────────

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

        self.ctrlTorqueMsg = messaging.CmdTorqueBodyMsg()
        self.ctrlForceMsg  = messaging.CmdForceBodyMsg()

        self.L=int(round(max(0.0,cfg.latency_s)/dt))
        self.use_latency_predict=cfg.use_latency_predict
        self.enable_po_pc=cfg.enable_po_pc
        self.alpha_tau=dt/(cfg.cmd_smooth_tau_s+dt); self.alpha_for=dt/(cfg.cmd_smooth_for_s+dt)
        self.tau_smooth=np.zeros(3); self.f_smooth=np.zeros(3)
        self.tau_dead=cfg.tau_deadband_nm; self.f_dead=cfg.f_deadband_n

        self.B, self.Ts, self.Isps = self._build_wrench_map(veh.thrusters)
        self.delta_prev=None

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
        J=self.veh.inertia_B; m=self.veh.mass_kg
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
        t_pl=messaging.CmdTorqueBodyMsgPayload(); t_pl.torqueRequestBody=list(tau_apply)
        f_pl=messaging.CmdForceBodyMsgPayload();  f_pl.forceRequestBody =list(f_thr)
        self.ctrlTorqueMsg.write(t_pl); self.ctrlForceMsg.write(f_pl)

        self.torque_apply_hist.append(tau_apply.copy())
        self.omega_prev=omega.copy(); self.v_prev=v_B.copy()


# ───────────────────────── Followers: scratch torque-only controllers ─────────────────────────

class _TorqueOnlyBase:
    def __init__(self, veh: VehicleConfig, dt: float, cfg: SimConfig):
        self.veh = veh; self.dt = dt; self.cfg = cfg
        self.ctrlTorqueMsg = messaging.CmdTorqueBodyMsg()
        self.torque_apply_hist: List[np.ndarray] = []
        self._lpf = TorqueLPF(dt, cfg.lpf_cut_hz) if cfg.use_lpf else None
        self.tau_lim_axis = np.full(3, max(veh.rw_tau_max, cfg.ctrl_tau_limit_nm))
        self.tau_dead = getattr(cfg, "tau_deadband_nm", 0.0)

    def _sat_per_axis(self, u: np.ndarray):
        u_sat = np.clip(u, -self.tau_lim_axis, +self.tau_lim_axis)
        return u_sat, (u_sat - u)

    def _apply_post(self, tau_body: np.ndarray):
        tau = tau_body.copy()
        tau[np.abs(tau) < self.tau_dead] = 0.0
        tau, _ = self._sat_per_axis(tau)
        if self._lpf is not None: tau = self._lpf.filt(tau)
        pl = messaging.CmdTorqueBodyMsgPayload(); pl.torqueRequestBody = list(tau)
        self.ctrlTorqueMsg.write(pl)
        self.torque_apply_hist.append(tau.copy())

    @staticmethod
    def _att_errors(attNav: messaging.NavAttMsgPayload,
                    R_d: np.ndarray,
                    omega_RN_N: np.ndarray) -> Dict[str, np.ndarray]:
        s_BN = np.array(attNav.sigma_BN, float)
        C_BN = mrp_to_dcm(s_BN)
        C_RN = R_d
        C_BR = C_BN @ C_RN.T
        s_BR = quat_to_mrp(dcm_to_quat(C_BR))
        omega_B  = np.array(attNav.omega_BN_B, float)
        omega_RN_B = C_BN @ omega_RN_N
        omega_err = omega_B - omega_RN_B
        return dict(C_BN=C_BN, s_BR=s_BR, omega_B=omega_B,
                    omega_RN_B=omega_RN_B, omega_err=omega_err)


class MRPFeedbackScratch(_TorqueOnlyBase):
    """τ = -K σ_BR - P ω_BR - K_I ∫σ_BR dt  (with back‑calculation anti‑windup)."""
    def __init__(self, veh: VehicleConfig, dt: float, cfg: SimConfig):
        super().__init__(veh, dt, cfg)
        self.K  = cfg.K_fb
        self.P  = cfg.P_fb
        self.Ki = cfg.Ki_fb
        self.I_lim = np.full(3, cfg.I_lim_fb)
        self.int_sigma = np.zeros(3)
        self.beta_aw = 6.0

    def step(self, t: float, attNav: messaging.NavAttMsgPayload,
             R_d: np.ndarray, omega_RN_N: np.ndarray, domega_RN_N: np.ndarray):
        errs = self._att_errors(attNav, R_d, omega_RN_N)
        s_BR = errs["s_BR"]; omega_err = errs["omega_err"]

        u_unsat = - self.K * s_BR - self.P * omega_err - self.Ki * self.int_sigma
        u_sat, sat_err = self._sat_per_axis(u_unsat)

        self.int_sigma += (s_BR + self.beta_aw * sat_err) * self.dt
        self.int_sigma = np.clip(self.int_sigma, -self.I_lim, +self.I_lim)

        u = - self.K * s_BR - self.P * omega_err - self.Ki * self.int_sigma
        self._apply_post(u)


class PIDRateScratch(_TorqueOnlyBase):
    """mrpSteering + nonlinear rate servo with FF and anti‑windup."""
    def __init__(self, veh: VehicleConfig, dt: float, cfg: SimConfig):
        super().__init__(veh, dt, cfg)
        self.K1 = cfg.K1_str; self.K3 = cfg.K3_str; self.wmax = max(0.1, cfg.wmax)
        self.P  = cfg.P_rate; self.Ki = cfg.Ki_rate
        self.I_lim = np.full(3, cfg.I_lim_rate)
        self.int_rate_err = np.zeros(3)
        self.prev_omega_steer = np.zeros(3)
        self.beta_aw = 6.0

    def step(self, t: float, attNav: messaging.NavAttMsgPayload,
             R_d: np.ndarray, omega_RN_N: np.ndarray, domega_RN_N: np.ndarray):
        errs = self._att_errors(attNav, R_d, omega_RN_N)
        C_BN = errs["C_BN"]; s_BR = errs["s_BR"]
        omega_B = errs["omega_B"]; omega_RN_B = errs["omega_RN_B"]

        s2 = float(s_BR @ s_BR)
        sdot_des = - self.K1 * s_BR - self.K3 * s2 * s_BR
        omega_steer = 4.0 * (Binv_sigma(s_BR) @ sdot_des)

        mag = la.norm(omega_steer)
        if mag > self.wmax: omega_steer *= self.wmax / max(1e-9, mag)

        omega_cmd = omega_RN_B + omega_steer

        domega_RN_B = C_BN @ domega_RN_N + np.cross(omega_B, omega_RN_B)
        domega_steer = (omega_steer - self.prev_omega_steer) / self.dt
        self.prev_omega_steer = omega_steer.copy()
        domega_cmd = domega_RN_B + domega_steer

        rate_err = omega_B - omega_cmd
        J = self.veh.inertia_B
        cross = np.cross(omega_B, J @ omega_B)

        u_unsat = - self.P * rate_err - self.Ki * self.int_rate_err - cross + J @ domega_cmd
        u_sat, sat_err = self._sat_per_axis(u_unsat)

        self.int_rate_err += (rate_err + self.beta_aw * sat_err) * self.dt
        self.int_rate_err = np.clip(self.int_rate_err, -self.I_lim, +self.I_lim)

        u = - self.P * rate_err - self.Ki * self.int_rate_err - cross + J @ domega_cmd
        self._apply_post(u)


# ───────────────────────── RAFTI separation (unchanged) ─────────────────────────

@dc.dataclass
class SepParams:
    t_start: float; t_end  : float; F0: float
    s_car_B: np.ndarray; r_car_B: np.ndarray; r_dep_B: np.ndarray
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


# ───────────────────────── Case builder ─────────────────────────

class Case:
    def __init__(self, kind: str, cfg: SimConfig,
                 leader_ref: Optional[Dict[str, np.ndarray]]=None,
                 gain_scale: float = 1.0):
        """kind ∈ {'PH_SE3','MRP_FB','PID'}. If leader_ref provided, followers use it."""
        self.kind = kind; self.cfg=cfg; self.dt=cfg.dt
        self.espa_radius = 42.0*0.0254/2.0  # 0.5334 m
        self.leader_ref = leader_ref
        self.gain_scale = gain_scale

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

        # Deputy (XB6 6U) + 12-jet
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

        # ESPA ring geometry
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
            name="XB6_6U", mass_kg=12.0, inertia_B=I_dep,
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
        self.ctrl_car = PHSE3(self.car, self.dt, last_sep_end, CFG)
        self.eft_car_ctrl.cmdTorqueInMsg   .subscribeTo(self.ctrl_car.ctrlTorqueMsg)
        self.eft_car_ctrl.cmdForceBodyInMsg.subscribeTo(self.ctrl_car.ctrlForceMsg)

        if kind == "PH_SE3":
            self.ctrl_dep = PHSE3(self.dep, self.dt, last_sep_end, CFG)
            self.eft_dep_ctrl.cmdTorqueInMsg   .subscribeTo(self.ctrl_dep.ctrlTorqueMsg)
            self.eft_dep_ctrl.cmdForceBodyInMsg.subscribeTo(self.ctrl_dep.ctrlForceMsg)
            self.dep_mode = "ph"
        elif kind == "MRP_FB":
            cfg_scaled = copy.deepcopy(CFG)
            cfg_scaled.K_fb  *= self.gain_scale
            cfg_scaled.P_fb  *= self.gain_scale
            cfg_scaled.Ki_fb *= self.gain_scale
            cfg_scaled.lpf_cut_hz *= self.gain_scale
            self.ctrl_dep = MRPFeedbackScratch(self.dep, self.dt, cfg_scaled)
            # reduce authority as well
            self.ctrl_dep.tau_lim_axis *= self.gain_scale
            self.eft_dep_ctrl.cmdTorqueInMsg.subscribeTo(self.ctrl_dep.ctrlTorqueMsg)
            self.dep_mode = "fb"
        elif kind == "PID":
            cfg_scaled = copy.deepcopy(CFG)
            cfg_scaled.K1_str *= self.gain_scale
            cfg_scaled.K3_str *= self.gain_scale
            cfg_scaled.wmax   *= self.gain_scale
            cfg_scaled.P_rate *= self.gain_scale
            cfg_scaled.Ki_rate*= self.gain_scale
            cfg_scaled.lpf_cut_hz *= self.gain_scale
            self.ctrl_dep = PIDRateScratch(self.dep, self.dt, cfg_scaled)
            self.ctrl_dep.tau_lim_axis *= self.gain_scale
            self.eft_dep_ctrl.cmdTorqueInMsg.subscribeTo(self.ctrl_dep.ctrlTorqueMsg)
            self.dep_mode = "pid"
        else:
            raise ValueError("Unknown controller kind")

        # RAFTI schedule
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

        # book‑keeping for reference and metrics
        self.Rd_cont_prev=None
        self.prev_C_RN = mrp_to_dcm(self.dep.sigma0_BN)
        self.prev_omega_N = np.zeros(3)
        self.last_omega_RN_N = np.zeros(3)
        self.p_ref_frozen=False; self.p_d_N_frozen=np.zeros(3)

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

    def _leader_sample(self, t: float):
        """Sample leader (PH_SE3) reference with a fixed delay."""
        lr = self.leader_ref
        if lr is None:
            return None
        tlag = t - float(lr["lag_sec"])
        if tlag <= lr["t"][0]: idx=0
        elif tlag >= lr["t"][-1]: idx = len(lr["t"])-1
        else: idx = int(round((tlag - lr["t"][0]) / lr["dt"]))
        s = lr["sigma"][idx]
        R = mrp_to_dcm(s)
        wN = lr["omega_N"][idx]
        dwN = lr["domega_N"][idx]
        return R, wN, dwN

    def run(self) -> Dict[str,np.ndarray]:
        now_ns=0; N=int(self.cfg.T_end/self.dt)
        last_sep_end = self.t0_base + 0.30
        bar = tqdm(range(N), desc=f"ESPA(42in)+XB6 — {self.kind}", ncols=80)

        # logs
        t_arr=[]; rpy=[]; rpy_d=[]; miss=[]; roll_los=[]
        tau_mag=[]; H_err=[]
        sigBN=[]; wB=[]; tau_vec=[]
        sigD=[]; wrefN=[]; wrefB=[]

        for k in bar:
            t=k*self.dt
            self.sns.ExecuteTaskList(now_ns)
            self.sep.step(t)

            # carrier via PH_SE3
            self.ctrl_car.step(t, self.nav_car.attOutMsg.read(), self.nav_car.transOutMsg.read(),
                               R_d=np.eye(3), p_d_N=np.zeros(3))

            # deputy reference
            r_dep_N = np.array(self.nav_dep.transOutMsg.read().r_BN_N, float)
            if (not self.p_ref_frozen) and t >= last_sep_end:
                self.p_d_N_frozen = r_dep_N.copy(); self.p_ref_frozen=True

            # Attitude reference (leader for followers; camera for PH)
            if self.dep_mode == "ph":
                R_d, los_N = self._camera_Rd_cont(r_dep_N)
                # build reference rates for logging/tracking error
                Delta = R_d @ self.prev_C_RN.T
                omega_RN_R  = so3_log(Delta) / self.dt
                omega_RN_N  = R_d.T @ omega_RN_R
                domega_RN_N = (omega_RN_N - self.prev_omega_N) / self.dt
            else:
                # Followers: sample leader trajectory (delayed)
                R_d, omega_RN_N, domega_RN_N = self._leader_sample(t)
                # still compute los_N for pointing metrics
                _, los_N = self._camera_Rd_cont(r_dep_N)  # only for miss/roll metrics

            self.prev_C_RN   = R_d
            self.prev_omega_N = omega_RN_N
            self.last_omega_RN_N = omega_RN_N

            # control
            if self.dep_mode == "ph":
                self.ctrl_dep.step(t, self.nav_dep.attOutMsg.read(), self.nav_dep.transOutMsg.read(),
                                   R_d=R_d, p_d_N=self.p_d_N_frozen if self.p_ref_frozen else r_dep_N)
            elif self.dep_mode == "fb":
                self.ctrl_dep.step(t, self.nav_dep.attOutMsg.read(),
                                   R_d=R_d, omega_RN_N=omega_RN_N, domega_RN_N=domega_RN_N)
            else:
                self.ctrl_dep.step(t, self.nav_dep.attOutMsg.read(),
                                   R_d=R_d, omega_RN_N=omega_RN_N, domega_RN_N=domega_RN_N)

            # dynamics step
            self.dyn.ExecuteTaskList(now_ns)
            now_ns += int(macros.sec2nano(self.dt))

            # sense
            att = self.nav_dep.attOutMsg.read()
            C_BN = mrp_to_dcm(np.array(att.sigma_BN))
            s_BN = np.array(att.sigma_BN, float)
            w_B  = np.array(att.omega_BN_B, float)

            # Euler (wrapped to desired)
            rpy_i  = dcm_to_euler_zyx(C_BN)
            rpy_di = dcm_to_euler_zyx(R_d)
            diff = (rpy_i - rpy_di + np.pi)%(2*np.pi) - np.pi
            t_arr.append(t); rpy.append(rpy_di + diff); rpy_d.append(rpy_di)

            # pointing metrics vs ESPA port
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

            # actuation & momentum error
            tau = (self.ctrl_dep.torque_apply_hist[-1]
                   if self.ctrl_dep.torque_apply_hist else np.zeros(3))
            tau_mag.append(la.norm(tau)); tau_vec.append(tau.copy())
            H_err.append(la.norm(self.dep.inertia_B @ (w_B - (C_BN @ self.last_omega_RN_N))))

            # richer logs
            sigBN.append(s_BN); wB.append(w_B)
            sigD.append(quat_to_mrp(dcm_to_quat(R_d)))
            wrefN.append(omega_RN_N.copy()); wrefB.append(C_BN @ omega_RN_N)

        # package logs
        L: Dict[str,np.ndarray] = {}
        L["t"]     = np.array(t_arr)
        rpy = np.array(rpy); rpy_d = np.array(rpy_d)
        L["roll"], L["pitch"], L["yaw"]   = rpy[:,0], rpy[:,1], rpy[:,2]
        L["roll_d"], L["pitch_d"], L["yaw_d"] = rpy_d[:,0], rpy_d[:,1], rpy_d[:,2]
        L["miss"]  = np.array(miss); L["roll_los"]=np.array(roll_los)
        L["tau_mag"]=np.array(tau_mag); L["H_err"]=np.array(H_err)
        L["sigma_BN"]=np.array(sigBN); L["omega_B"]=np.array(wB); L["tau_vec"]=np.array(tau_vec)
        L["sigma_d"]=np.array(sigD); L["omega_ref_N"]=np.array(wrefN); L["omega_ref_B"]=np.array(wrefB)

        # RMS summaries
        tau_rms = float(np.sqrt(np.mean(L["tau_mag"]**2))) if L["tau_mag"].size else 0.0
        H_rms   = float(np.sqrt(np.mean(L["H_err"]**2)))   if L["H_err"].size   else 0.0
        print(f"[{self.kind}] torque_rms={tau_rms:.4f} N·m, momentum_err_rms={H_rms:.4f} N·m·s")

        # save NPZ for this run
        np.savez_compressed(f"deputy1_{self.kind}.npz", **L)
        print(f"[SAVED] deputy1_{self.kind}.npz")

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


# ───────────────────────── utilities ─────────────────────────

def compute_settling_time(t: np.ndarray, miss_deg: np.ndarray,
                          thr_deg: float=1.0, dwell_s: float=8.0) -> float:
    """Earliest time when miss <= thr for a continuous dwell."""
    ok = np.asarray(miss_deg) <= thr_deg
    n_dwell = max(1, int(round(dwell_s / (t[1]-t[0]))))
    for i in range(len(t)-n_dwell):
        if ok[i] and np.all(ok[i:i+n_dwell]): return float(t[i])
    return float(t[-1])


# ───────────────────────── main ─────────────────────────

def main():
    # 1) Run leader (PH_SE3)
    leader_case = Case("PH_SE3", CFG)
    leader_logs = leader_case.run()
    # Compute leader settling time and leader‑reference arrays for followers
    t_lead = leader_logs["t"]
    miss   = leader_logs["miss"]
    t_settle = compute_settling_time(t_lead, miss, thr_deg=1.0, dwell_s=8.0)
    delay_sec = max(CFG.follower_min_delay_s, CFG.follower_delay_frac * t_settle)

    sigma_lead = leader_logs["sigma_BN"]        # (N,3) MRPs
    omegaB_lead= leader_logs["omega_B"]         # (N,3) body rates in B
    # reference rate in N (for followers) and its derivative
    omegaN_lead = np.zeros_like(omegaB_lead)
    for i in range(len(t_lead)):
        C = mrp_to_dcm(sigma_lead[i])
        omegaN_lead[i] = C.T @ omegaB_lead[i]
    domegaN_lead = np.vstack((np.zeros(3), np.diff(omegaN_lead, axis=0)/CFG.dt))

    leader_ref = dict(t=t_lead, dt=CFG.dt, sigma=sigma_lead,
                      omega_N=omegaN_lead, domega_N=domegaN_lead,
                      lag_sec=delay_sec)

    # 2) Followers (20% worse)
    all_logs = {"PH_SE3": leader_logs}

    fb_case  = Case("MRP_FB", CFG, leader_ref=leader_ref,
                    gain_scale=CFG.follower_gain_scale)
    all_logs["MRP_FB"] = fb_case.run()

    pid_case = Case("PID", CFG, leader_ref=leader_ref,
                    gain_scale=CFG.follower_gain_scale)
    all_logs["PID"] = pid_case.run()

    # 3) Plots
    plot_overlays(all_logs)
    print("[DONE] ESPA(42\") + RAFTI + XB6 — leader/followers complete (NPZ saved).")
    print(f"Followers used PH_SE3 trajectory with lag {delay_sec:.2f} s and gain scale {CFG.follower_gain_scale:.2f}.")

if __name__=="__main__":
    main()

