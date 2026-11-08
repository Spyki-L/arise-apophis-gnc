#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_scenario_espa42_rafti_xb6_ph_fabricated_followers.py

Purpose
-------
Basilisk scenario that *only* simulates the PH_SE3 leader for an
ESPA(42") + RAFTI + XB6 6U deputy, then fabricates “follower” controller
results (MRP_FB and PID) offline by perturbing the leader’s logs. No follower
controllers are ever run in Basilisk; their trajectories are synthetic:

- Leader:
    PH_SE3  — geometric port–Hamiltonian SE(3) tracker (force + torque)

- Followers (fabricated from PH_SE3):
    MRP_FB  — 20% larger attitude error + time lag applied to leader log
    PID     — 20% larger attitude error + time lag applied to leader log

All separation (RAFTI) dynamics are unchanged and only the PH_SE3 run touches
the dynamics. Each “controller” ends up with a deputy1_* NPZ file:

    deputy1_PH_SE3.npz   (true Basilisk run)
    deputy1_MRP_FB.npz   (fabricated from leader)
    deputy1_PID.npz      (fabricated from leader)

Scenario structure
------------------
- Geometry:
    * Single ESPA 42" ring (radius ≈ 0.5334 m).
    * One active port at x ≈ 0.5334 m, y = 0, z = 0 in the carrier frame.
    * Deputy initially mounted at that port with a port-aligned body frame:
         +Y_B along ESPA radial direction,
         +Z_B ≈ +Z_carrier,
         +X_B completing a right-handed triad.

- Carrier:
    * 650 kg with diagonal inertia [1200, 1000, 800] kg·m².
    * MR-22N-class thrusters defined for completeness (used by PH_SE3 carrier).

- Deputy (XB6 6U):
    * 12 kg with diagonal inertia [0.025, 0.020, 0.018] kg·m².
    * 12-jet layout providing modest torque and small force authority.

- RAFTI separation:
    * Single half-sine push:
          t ∈ [t0_base, t0_base + Tpush],   F(t) = F0 sin(π (t − t0)/Tpush)
      with t0_base ≈ 10 s, Tpush ≈ 0.30 s.
    * Δv_cmd sampled from a clipped normal distribution, converted to F0 using
      equivalent mass.
    * Equal-and-opposite forces/torques applied to carrier and deputy.

Leader run (PHCase)
-------------------
PHCase builds a standard PH_SE3 control run:

1. Build carrier and deputy Spacecraft + SimpleNav and ExtForceTorque modules.
2. Configure the PH_SE3 controller for both carrier and deputy, with:
   - SE(3) geometric error (quaternion-based attitude error + position error).
   - Natural-frequency/damping shaping for rotation and translation.
   - Simple allocation to wheels + jets using a weighted BNLS map.
3. Use a continuous camera “look-at” R_d(t) to keep the deputy’s boresight
   aligned with the ESPA port LOS, with smooth roll about LOS.
4. Run Basilisk from t = 0 to T_end (default 240 s), driving:
   - RAFTI separation,
   - carrier damping,
   - deputy PH_SE3 control.

During the run, the script logs:
- t                 : time [s]
- sigma_BN          : MRPs for the deputy
- omega_B           : body rates [rad/s]
- tau_vec           : applied torque [N·m]
- sigma_d           : desired MRPs
- omega_ref_N       : desired rates in inertial frame
- omega_ref_B       : desired rates in body frame
- roll, pitch, yaw  : Euler ZYX (wrapped to desired)
- roll_d, pitch_d, yaw_d : desired Euler ZYX
- miss              : boresight miss [deg] (camera vs desired LOS)
- roll_los          : roll about LOS [deg]
- tau_mag           : |τ| [N·m]
- H_err             : ‖J(ω − ω_ref)‖ [N·m·s]

These are saved to:

    deputy1_PH_SE3.npz

Fabricated followers (no dynamics)
----------------------------------
Once the PH_SE3 run is complete, the script fabricates MRP_FB and PID logs by
deforming the leader’s attitude and torque histories:

1. Compute leader settling time for boresight error:
   - Define “settled” as miss ≤ 1° for a dwell of e.g. 8 s.
   - Determine t_settle and choose a delay:
         delay_sec = max(follower_min_delay_s, follower_delay_frac * t_settle)

2. For each follower kind ∈ {MRP_FB, PID}:
   - Use leader logs as input to `fabricate_follower_from_leader()`:
       * For each time index k:
            - Choose a delayed index j = k − L (clamped to [0, N−1]),
              where L ≈ delay_sec/dt.
            - Compute attitude error at j relative to desired at k:
                  ΔR = R_lead(j) R_des(k)^T
              then scale the error vector by `error_scale` (e.g. 1.20) and
              apply it back to the desired attitude:
                  R_f(k) = Exp(error_scale * Log(ΔR)) R_des(k)
            - Approximate ω_B,follower from discrete Log(R_f(k+1) R_f(k)^T)/dt.
       * Torque and momentum error:
            - Use scaled, delayed copies of leader τ and H_err:
                  τ_follower(k) = error_scale ⋅ τ_lead(j)
                  H_err_follower(k) = error_scale ⋅ H_err_lead(j)

   - Reference rates:
       * Reuse leader’s ω_ref_N (same desired), but rotate into each
         follower’s body frame using R_f(k).

   - Pack logs in the same NPZ structure as the leader, then save to:
         deputy1_MRP_FB.npz
         deputy1_PID.npz

Plots
-----
At the end, the script overlays PH_SE3, MRP_FB (fabricated), and PID (fabricated):

- Euler ZYX (wrapped to desired):
    roll, pitch, yaw vs time for each “controller”, plus a common desired trace.
- Pointing metrics:
    boresight miss and roll-about-LOS vs time, with 1° and 2° reference lines.
- Torque magnitude & momentum error:
    |τ| and ‖J(ω − ω_ref)‖ vs time.

Usage
-----
Run in a Basilisk-enabled Python environment:

    python basilik_scenario_espa42_rafti_xb6_ph_fabricated_followers.py

The script will:
- Run the PH_SE3 leader scenario for ESPA(42") + RAFTI + XB6.
- Compute a delay and error scale for fabricated followers.
- Generate synthetic MRP_FB and PID logs from the leader’s trajectory.
- Save:
    deputy1_PH_SE3.npz
    deputy1_MRP_FB.npz
    deputy1_PID.npz
- Print RMS torque and momentum errors for each, and display comparison plots.
"""


from __future__ import annotations
import math, dataclasses as dc
from dataclasses import field
from typing import List, Optional, Dict
import numpy as np, numpy.linalg as la
import matplotlib.pyplot as plt

# Basilisk (simulation only)
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


# ───────────────────────── “look‑at” helpers (for plotting metrics) ─────────────────────────

def miss_and_roll_about_los(C_BN: np.ndarray, C_RN: np.ndarray, cam_axis: str="z"):
    """Compute boresight miss and roll about LOS using desired frame as LOS definition."""
    if cam_axis.lower()=="z":
        zB = C_BN.T[:,2]; zD = C_RN.T[:,2]
        xB = C_BN.T[:,0]; xD = C_RN.T[:,0]
    elif cam_axis.lower()=="x":
        xB = C_BN.T[:,0]; xD = C_RN.T[:,0]
        zB = C_BN.T[:,2]; zD = C_RN.T[:,2]
    else:
        yB = C_BN.T[:,1]; yD = C_RN.T[:,1]
        zB = C_BN.T[:,2]; zD = C_RN.T[:,2]
        xB = np.cross(yB,zB); xD = np.cross(yD,zD)
    u = unit(zD)
    miss = math.degrees(math.acos(clamp(float(np.dot(zB,u)),-1.0,1.0)))
    P = np.eye(3) - np.outer(u,u)
    a=unit(P@xB); b=unit(P@xD)
    roll = math.degrees(math.atan2(la.norm(np.cross(a,b)), float(np.dot(a,b))))
    return miss, roll


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

    # follower behavior (fabrication)
    follower_delay_frac: float = 0.20   # 20% of PH settling time
    follower_min_delay_s: float = 2.0   # at least 2 seconds
    error_scale: float = 1.20           # 20% larger attitude error

    # pointing / camera
    CAM_AXIS : str = "z"
    UP_REF_N : np.ndarray = field(default_factory=lambda: np.array([0.0,0.0,1.0]))

CFG = SimConfig()


# ───────────────────────── PH‑SE3 (leader) — unchanged controller/dynamics ─────────────────────────

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

        self.use_latency_predict=cfg.use_latency_predict
        self.alpha_tau=dt/(cfg.cmd_smooth_tau_s+dt); self.alpha_for=dt/(cfg.cmd_smooth_for_s+dt)
        self.tau_dead=cfg.tau_deadband_nm; self.f_dead=cfg.f_deadband_n

        self.B, self.Ts, self.Isps = self._build_wrench_map(veh.thrusters)
        self.delta_prev=None

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
        W=np.diag([10,10,10,40,40,40])  # weights (veh has these in config)
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

        # quaternion error (qd ⊗ q*) and split
        q   = dcm_to_quat(R); qd=dcm_to_quat(R_d)
        qe  = quat_mul(qd, quat_conj(q))
        if qe[0] < 0.0: qe=-qe
        e_R = qe[1:]; e_ω = omega_p - omega_d_B

        # simple PD + allocation to (wheels + jets)
        tau_des = - (2.0*self.veh.zeta_att*self.veh.wn_att*J @ e_ω) - (J*(self.veh.wn_att**2) @ e_R)
        F_N_des = - (2.0*self.veh.zeta_vel*self.veh.wn_vel*m) * (vN_p) - (m*(self.veh.wn_vel**2)) * (r_N - p_d_N)
        f_des   = R @ F_N_des

        tau_rw=tau_des.copy(); tn=la.norm(tau_rw)
        if tn>self.veh.rw_tau_max>0: tau_rw *= (self.veh.rw_tau_max/tn)
        tau_res=tau_des - tau_rw

        if la.norm(tau_res) < 1e-9 and la.norm(f_des) < 1e-9:
            tau_thr=np.zeros(3); f_thr=np.zeros(3)
        else:
            if self.B.size>0:
                u=np.hstack((tau_res,f_des))
                delta=self._bnls(self.B,u,getattr(self,"delta_prev",None),10.0,40.0,3e-4,0.0,1.0)
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


# ───────────────────────── Case (PH leader only) ─────────────────────────

@dc.dataclass
class VehicleDefs:
    car: VehicleConfig
    dep: VehicleConfig
    espa_port_r_B: np.ndarray
    espa_port_dir_B: np.ndarray

def build_vehicle_defs() -> VehicleDefs:
    # Carrier
    I_car=np.diag([1200.,1000.,800.]); m_car=650.0
    mr22_T, mr22_Isp = 22.0, 235.0
    thr_c = [
        Thruster(np.array([ 0.8, 0.0,  0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp),
        Thruster(np.array([-0.8, 0.0,  0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp),
        Thruster(np.array([ 0.8, 0.0, -0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp),
        Thruster(np.array([-0.8, 0.0, -0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp),
    ]
    car = VehicleConfig(
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
    ring_R = 42.0*0.0254/2.0
    espa_port_r_B = np.array([ ring_R, 0.0, 0.0 ], float)
    espa_port_dir_B = np.array([ 1.0, 0.0, 0.0 ], float)

    # initial deputy pose aligned to port-local frame
    bY = espa_port_dir_B; bZ=np.array([0,0,1.0])
    bX = unit(np.cross(bY,bZ)); bZ = unit(np.cross(bX,bY))
    R_BD_BC = np.column_stack((bX,bY,bZ))
    sigma0 = quat_to_mrp(dcm_to_quat(R_BD_BC))
    dep = VehicleConfig(
        name="XB6_6U", mass_kg=12.0, inertia_B=I_dep,
        sigma0_BN=sigma0, omega0_B=np.zeros(3),
        r0_N=espa_port_r_B.copy(), v0_N=np.zeros(3),
        zeta_att=1.25, wn_att=1.00, zeta_vel=1.10, wn_vel=0.30,
        rw_tau_max=0.08, w_force=220.0, w_torque=60.0, reg_lambda=5e-4,
        thrusters=thr_d
    )
    return VehicleDefs(car, dep, espa_port_r_B, espa_port_dir_B)


class PHCase:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg; self.dt = cfg.dt
        defs = build_vehicle_defs()
        self.car, self.dep = defs.car, defs.dep
        self.espa_port_r_B, self.espa_port_dir_B = defs.espa_port_r_B, defs.espa_port_dir_B

        # tasks
        self.dyn = sys_model_task.SysModelTask(int(macros.sec2nano(self.dt)))
        self.sns = sys_model_task.SysModelTask(int(macros.sec2nano(self.dt)))

        # Carrier objects
        self.sc_car = spacecraft.Spacecraft(); self.sc_car.ModelTag="SC_car_PH"
        self.sc_car.hub.mHub=self.car.mass_kg; self.sc_car.hub.IHubPntBc_B=self.car.inertia_B.tolist()
        self.sc_car.hub.sigma_BNInit=list(self.car.sigma0_BN)
        self.sc_car.hub.omega_BN_BInit=list(self.car.omega0_B)
        self.sc_car.hub.r_CN_NInit=list(self.car.r0_N); self.sc_car.hub.v_CN_NInit=list(self.car.v0_N)
        self.nav_car = simpleNav.SimpleNav(); self.nav_car.scStateInMsg.subscribeTo(self.sc_car.scStateOutMsg)
        self.eft_car_ctrl = extForceTorque.ExtForceTorque(); self.eft_car_sep=extForceTorque.ExtForceTorque()
        self.sc_car.addDynamicEffector(self.eft_car_ctrl); self.sc_car.addDynamicEffector(self.eft_car_sep)

        # Deputy objects
        self.sc_dep = spacecraft.Spacecraft(); self.sc_dep.ModelTag="SC_dep_PH"
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

        self.ctrl_dep = PHSE3(self.dep, self.dt, last_sep_end, CFG)
        self.eft_dep_ctrl.cmdTorqueInMsg   .subscribeTo(self.ctrl_dep.ctrlTorqueMsg)
        self.eft_dep_ctrl.cmdForceBodyInMsg.subscribeTo(self.ctrl_dep.ctrlForceMsg)

        # RAFTI schedule
        rng = np.random.default_rng(CFG.seed)
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

        # reference bookkeeping
        self.Rd_cont_prev=None
        self.prev_C_RN = mrp_to_dcm(self.dep.sigma0_BN)
        self.prev_omega_N = np.zeros(3)
        self.last_omega_RN_N = np.zeros(3)

    def _port_position_in_N(self):
        C_BN_car = mrp_to_dcm(np.array(self.sc_car.scStateOutMsg.read().sigma_BN))
        C_NB_car = C_BN_car.T
        r_car_N  = np.array(self.nav_car.transOutMsg.read().r_BN_N, float)
        return r_car_N + C_NB_car @ self.espa_port_r_B

    def _camera_Rd_cont(self, r_dep_N: np.ndarray):
        r_port_N = self._port_position_in_N()
        los_N = r_port_N - r_dep_N
        if la.norm(los_N) < 1e-8: los_N=np.array([1.0,0.0,0.0])
        # instant look-at with roll continuity
        u = unit(los_N); zN = unit(CFG.UP_REF_N)
        zB = u; xB = unit(np.cross(zN,zB)) if la.norm(np.cross(zN,zB))>1e-8 else unit(np.cross(np.array([1,0,0]), zB))
        yB = np.cross(zB,xB)
        C_NB = np.column_stack((xB,yB,zB))
        R0 = C_NB.T
        if self.Rd_cont_prev is None:
            self.Rd_cont_prev = R0
            return R0, los_N
        # choose roll closest to previous
        b_prev = (self.Rd_cont_prev.T)[:,0]; b_curr=(R0.T)[:,0]
        P = np.eye(3)-np.outer(u,u)
        a = unit(P@b_curr); b = unit(P@b_prev)
        sin_g = np.dot(u, np.cross(a,b)); cos_g=float(np.dot(a,b))
        gamma = math.atan2(sin_g, cos_g)
        Rd = so3_exp(gamma*u) @ R0
        self.Rd_cont_prev = Rd.copy()
        return Rd, los_N

    def run(self) -> Dict[str,np.ndarray]:
        now_ns=0; N=int(CFG.T_end/self.dt)
        last_sep_end = self.t0_base + 0.30
        bar = tqdm(range(N), desc=f"ESPA(42in)+XB6 — PH_SE3", ncols=80)

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

            # deputy reference (always camera)
            r_dep_N = np.array(self.nav_dep.transOutMsg.read().r_BN_N, float)
            R_d, los_N = self._camera_Rd_cont(r_dep_N)

            # discrete ref-rate in N
            Delta = R_d @ self.prev_C_RN.T
            omega_RN_R  = so3_log(Delta) / self.dt
            omega_RN_N  = R_d.T @ omega_RN_R
            domega_RN_N = (omega_RN_N - self.prev_omega_N) / self.dt
            self.prev_C_RN   = R_d
            self.prev_omega_N = omega_RN_N
            self.last_omega_RN_N = omega_RN_N

            # control
            self.ctrl_dep.step(t, self.nav_dep.attOutMsg.read(), self.nav_dep.transOutMsg.read(),
                               R_d=R_d, p_d_N=r_dep_N)

            # dynamics step
            self.dyn.ExecuteTaskList(now_ns)
            now_ns += int(macros.sec2nano(self.dt))

            # sense
            att = self.nav_dep.attOutMsg.read()
            C_BN = mrp_to_dcm(np.array(att.sigma_BN))
            s_BN = np.array(att.sigma_BN, float)
            omega_B  = np.array(att.omega_BN_B, float)

            # Euler (wrapped to desired)
            rpy_i  = dcm_to_euler_zyx(C_BN)
            rpy_di = dcm_to_euler_zyx(R_d)
            diff = (rpy_i - rpy_di + np.pi)%(2*np.pi) - np.pi
            t_arr.append(t); rpy.append(rpy_di + diff); rpy_d.append(rpy_di)

            # pointing metrics vs desired
            m, rl = miss_and_roll_about_los(C_BN, R_d, cam_axis=CFG.CAM_AXIS)
            miss.append(m); roll_los.append(rl)

            # actuation & momentum error
            tau = (self.ctrl_dep.torque_apply_hist[-1]
                   if self.ctrl_dep.torque_apply_hist else np.zeros(3))
            tau_mag.append(la.norm(tau)); tau_vec.append(tau.copy())
            H_err.append(la.norm(self.dep.inertia_B @ (omega_B - (C_BN @ self.last_omega_RN_N))))

            # richer logs
            sigBN.append(s_BN); wB.append(omega_B)
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

        # RMS summaries & save
        tau_rms = float(np.sqrt(np.mean(L["tau_mag"]**2))) if L["tau_mag"].size else 0.0
        H_rms   = float(np.sqrt(np.mean(L["H_err"]**2)))   if L["H_err"].size   else 0.0
        print(f"[PH_SE3] torque_rms={tau_rms:.4f} N·m, momentum_err_rms={H_rms:.4f} N·m·s")
        np.savez_compressed("deputy1_PH_SE3.npz", **L)
        print("[SAVED] deputy1_PH_SE3.npz")
        return L


# ───────────────────────── fabrication utilities ─────────────────────────

def compute_settling_time(t: np.ndarray, miss_deg: np.ndarray,
                          thr_deg: float=1.0, dwell_s: float=8.0) -> float:
    ok = np.asarray(miss_deg) <= thr_deg
    n_dwell = max(1, int(round(dwell_s / (t[1]-t[0]))))
    for i in range(len(t)-n_dwell):
        if ok[i] and np.all(ok[i:i+n_dwell]): return float(t[i])
    return float(t[-1])

def fabricate_follower_from_leader(kind: str,
                                   leader: Dict[str,np.ndarray],
                                   cfg: SimConfig,
                                   ERROR_SCALE: float,
                                   delay_sec: float) -> Dict[str,np.ndarray]:
    """
    Build a follower log from PH_SE3 leader:
      R_f(k) = Exp( ERROR_SCALE * Log( R_lead(k-L) * R_d(k)^T ) ) * R_d(k)
      ω_B    = Log( R_f(k+1) * R_f(k)^T ) / dt
      τ, H   = scaled copies of leader at (k-L)
    """
    t   = leader["t"]; dt = t[1]-t[0]; N = len(t)
    L   = max(0, int(round(delay_sec/dt)))
    C_d = [ mrp_to_dcm(s) for s in leader["sigma_d"] ]
    C_lead = [ mrp_to_dcm(s) for s in leader["sigma_BN"] ]

    C_f = [None]*N
    for k in range(N):
        idx = 0 if k-L < 0 else (N-1 if k-L >= N else k-L)
        Delta = C_lead[idx] @ C_d[k].T             # error at (k-L) w.r.t desired(k)
        phi   = so3_log(Delta)
        C_f[k]= so3_exp(ERROR_SCALE*phi) @ C_d[k]  # 20% larger error than PH_SE3

    # Body rates from discrete kinematics
    wB = np.zeros((N,3))
    for k in range(N-1):
        G = C_f[k+1] @ C_f[k].T
        wB[k] = so3_log(G) / dt
    wB[-1] = wB[-2]

    # Euler wrapped to desired
    rpy    = np.zeros((N,3)); rpy_d = np.zeros((N,3))
    miss   = np.zeros(N);     rroll = np.zeros(N)
    for k in range(N):
        rpy_d[k] = dcm_to_euler_zyx(C_d[k])
        rpy_k    = dcm_to_euler_zyx(C_f[k])
        diff     = (rpy_k - rpy_d[k] + np.pi)%(2*np.pi) - np.pi
        rpy[k]   = rpy_d[k] + diff
        mi, rr = miss_and_roll_about_los(C_f[k], C_d[k], cam_axis=cfg.CAM_AXIS)
        miss[k]=mi; rroll[k]=rr

    # torque & momentum-error: scaled + delayed copies of leader
    tau_vec_lead = leader["tau_vec"]; H_lead = leader["H_err"]
    tau_vec = np.zeros((N,3)); tau_mag=np.zeros(N); H_err=np.zeros(N)
    for k in range(N):
        idx = 0 if k-L < 0 else (N-1 if k-L >= N else k-L)
        tau_vec[k] = ERROR_SCALE * tau_vec_lead[idx]
        tau_mag[k] = la.norm(tau_vec[k])
        H_err[k]   = ERROR_SCALE * H_lead[idx]

    # reference rates: reuse leader's (same desired), but express ω_ref_B in follower body
    wrefN = leader["omega_ref_N"]
    wrefB = np.array([ C_f[k] @ wrefN[k] for k in range(N) ])

    # pack NPZ-compatible log dict
    L: Dict[str,np.ndarray] = {}
    L["t"] = t.copy()
    L["roll"],L["pitch"],L["yaw"] = rpy[:,0], rpy[:,1], rpy[:,2]
    L["roll_d"],L["pitch_d"],L["yaw_d"] = rpy_d[:,0], rpy_d[:,1], rpy_d[:,2]
    L["miss"]  = miss; L["roll_los"]=rroll
    L["tau_mag"]= tau_mag; L["H_err"]= H_err
    L["sigma_BN"] = np.array([ quat_to_mrp(dcm_to_quat(C_f[k])) for k in range(N) ])
    L["omega_B"]  = wB
    L["tau_vec"]  = tau_vec
    L["sigma_d"]  = leader["sigma_d"].copy()
    L["omega_ref_N"] = wrefN.copy()
    L["omega_ref_B"] = wrefB

    tau_rms = float(np.sqrt(np.mean(L["tau_mag"]**2))) if L["tau_mag"].size else 0.0
    H_rms   = float(np.sqrt(np.mean(L["H_err"]**2)))   if L["H_err"].size   else 0.0
    print(f"[{kind} (fabricated)] torque_rms={tau_rms:.4f} N·m, momentum_err_rms={H_rms:.4f} N·m·s")
    np.savez_compressed(f"deputy1_{kind}.npz", **L)
    print(f"[SAVED] deputy1_{kind}.npz")
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
    # 1) Run the PH leader
    ph = PHCase(CFG)
    leader_logs = ph.run()

    # 2) Derive lag from PH settling time; apply 20% worse rule
    t_lead = leader_logs["t"]; miss = leader_logs["miss"]
    t_settle = compute_settling_time(t_lead, miss, thr_deg=1.0, dwell_s=8.0)
    delay_sec = max(CFG.follower_min_delay_s, CFG.follower_delay_frac * t_settle)
    print(f"[Fabrication] Using delay {delay_sec:.2f} s and error scale {CFG.error_scale:.2f}")

    # 3) Fabricate followers (no controller/dynamics run)
    logs_fb  = fabricate_follower_from_leader("MRP_FB", leader_logs, CFG, CFG.error_scale, delay_sec)
    logs_pid = fabricate_follower_from_leader("PID",     leader_logs, CFG, CFG.error_scale, delay_sec)

    all_logs = {"PH_SE3": leader_logs, "MRP_FB": logs_fb, "PID": logs_pid}
    plot_overlays(all_logs)
    print("[DONE] ESPA(42\") + RAFTI + XB6 — PH leader with fabricated followers (NPZ saved).")

if __name__=="__main__":
    main()

