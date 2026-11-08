#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_scenario_espa42_rafti_xb6_fleet_se3.py

Purpose
-------
Basilisk scenario that simulates an ESPA 42-inch carrier with eight Blue Canyon
XB6-class 6U CubeSats mounted around the ring, separated via RAFTI-style
half-sine pushes and controlled by a geometric SE(3) port–Hamiltonian (pH)
attitude/translation tracker. After separation, each deputy uses a continuous
“look-at” attitude law so its camera boresight (default +Z_B) keeps pointing
back at the ESPA port from which it departed, with roll continuity about the
line-of-sight (LOS).

The scenario:
- Runs a full carrier + 8-deputy Basilisk sim with:
  * ESPA-42 ring geometry.
  * RAFTI half-sine separation schedule (per deputy).
  * Carrier pH SE(3) damping.
  * Deputy pH SE(3) control with integral action, passivity observer/controller
    (PO/PC), latency prediction, command smoothing, and wheel+jet allocation.
- Saves **combined** and **per-deputy** NPZ log files.
- Displays diagnostic plots (not saved) for Δv, pointing, pH energy, actuator
  usage, and Euler tracking.

Scenario setup
--------------
Geometry and vehicles:
- ESPA ring:
  * 42-inch diameter ⇒ radius ≈ 0.5334 m in the carrier body frame.
  * Eight uniformly spaced ports around the ring at z = 0.
  * Each port has:
      r_port_B   : position in carrier body frame.
      s_port_B   : unit vector from carrier center to port (radial outward).

- Carrier (ESPA host):
  * Mass ≈ 650 kg, diagonal inertia [1200, 1000, 800] kg·m².
  * Four MR-22N-class thrusters used for small control authority.
  * Controlled by a SE(3) pH layer with low natural frequency, acting as a
    “soft damper” to keep the carrier near a quiescent attitude and small
    translational motions during and after separation.

- Deputies (XB6 6U CubeSats, one per port):
  * Mass ≈ 12 kg, diagonal inertia [0.025, 0.020, 0.018] kg·m².
  * 12-jet micro-propulsion layout (X/Y/Z torque-capable) sized for small
    thrust (≈ 0.27 N) with Isp ≈ 217 s.
  * Each deputy’s initial attitude is aligned with a local port frame:
      +Y_B along the ESPA radial direction,
      +Z_B aligned with +Z_carrier or another sensible normal,
      +X_B completing a right-handed triad.

RAFTI separation model:
- Each deputy is given a half-sine push:
    F(t) = F0 sin(π (t − t0) / Tpush)  for t ∈ [t0, t0 + Tpush].
- The same base start time and duration (e.g. t0_base = 10.0 s, Tpush = 0.30 s)
  are used for all deputies; per-deputy Δv command is drawn from a clipped
  normal distribution and converted into F0 via the equivalent-mass formula.
- Forces are applied as equal-and-opposite:
  * Carrier: net force/torque from all separating deputies.
  * Deputy i: opposing force/torque at its port location.
- The RAFTI schedule (t_start, t_end, F0, dv_cmd, r_car_B, s_car_B, r_dep_B)
  is stored in the combined NPZ for later analysis.

SE(3) pH controller (per deputy)
--------------------------------
- Natural-frequency/damping shaping:
    * Rotational dynamics: K_R_base = J ω_n_att², D_R = 2 ζ_att ω_n_att J.
    * Translational dynamics: K_p = m ω_n_vel² I, D_v = 2 ζ_vel ω_n_vel m I.
- Integral terms with anti-windup:
    * Orientation integral: KI_R on e_R with bounding.
    * Position integral: KI_p on e_p with bounding.
- Attitude error:
    * Orientation error computed via shortest-arc quaternion difference
      between actual and desired attitudes.
    * Desired rate ω_d computed from SO(3) log/exp on R_d using a finite
      difference on R_d over time.
- Features:
    * Time-varying shaping ramp (shape_ramp_s) after an engage time to
      avoid impulses in pH gains right at separation.
    * Latency prediction using extrapolated ω and v_B.
    * PO/PC:
        - Passivity observer integrates energy injection from τ·ω + f·v.
        - Passivity controller scales commanded wrench to prevent the passivity
          tank from going positive.
        - When scaling is active, integral terms are held to avoid windup.
    * Command smoothing (1st-order filters) on torque and force.
    * Deadbands on residual torque/force to avoid micro-chatter around zero.
    * Allocation:
        - Reaction wheels + thrusters: wheel torque limited to rw_tau_max,
          thrusters carry the residual torque + all force.
        - BNLS-style least-squares allocator for thruster duty-cycle vector δ,
          with weights (w_torque, w_force) and regularization reg_lambda.
    * Δv and prop accounting based on net thrust and Isp.

Look-at guidance for each deputy
--------------------------------
- After RAFTI push completion:
  * Compute ESPA port position in inertial frame from carrier state:
        r_port_N = r_car_N + C_NB_car r_port_B.
  * LOS vector to port: los_N = r_port_N − r_dep_N.
  * Use `look_at_continuous(los_N, UP_REF_N, R_prev, cam_axis)` to:
        - Align the camera boresight axis (default +Z_B) with los_N.
        - Choose roll about LOS that stays as close as possible to the
          previous attitude R_prev (minimizing roll jumps).
  * The resulting R_d is fed to the SE(3) controller along with a frozen
    translation reference p_d_N equal to the deputy’s position at push end.

Logging and NPZ output
----------------------
After the run, the script saves:

1. **Combined NPZ**:
   - Carrier:
       carrier_t, carrier_Hd,
       carrier_tau_des, carrier_f_des,
       carrier_tau_cmd, carrier_f_cmd,
       carrier_tau_apply, carrier_f_apply,
       carrier_omega_B, carrier_vB,
       carrier_sigma_BN,
       carrier_euler_wrapped, carrier_euler_des,
       carrier_thrust_sumN, carrier_po_scale, carrier_E_po,
       carrier_dv_cum, carrier_mp_cum_kg.
   - Deputies D1..D8:
       Dk_t, Dk_Hd, Dk_tau_des, Dk_f_des,
       Dk_tau_cmd, Dk_f_cmd, Dk_tau_apply, Dk_f_apply,
       Dk_omega_B, Dk_vB, Dk_sigma_BN,
       Dk_euler_wrapped, Dk_euler_des,
       Dk_thrust_sumN, Dk_po_scale, Dk_E_po,
       Dk_boresight_deg, Dk_roll_about_LOS_deg,
       Dk_dv_cum, Dk_mp_cum_kg.
   - Geometry and schedule:
       espa_radius_m, espa_ports_r_B, espa_ports_dir_B,
       sched_t_start, sched_t_end, sched_F0_N, sched_dv_cmd,
       sched_r_car_B, sched_s_car_B, sched_r_dep_B,
       dt, T_end, latency_s, shape_ramp_s.
   - File name pattern:
       espa_xb6_8deps_YYYYMMDD_HHMMSS.npz

2. **Per-deputy NPZs**:
   - For each deputy k:
       t, Hd,
       tau_des, f_des,
       tau_cmd, f_cmd,
       tau_apply, f_apply,
       omega_B, vB, sigma_BN,
       euler_wrapped, euler_des,
       thrust_sumN, po_scale, E_po,
       boresight_deg, roll_about_LOS_deg,
       dv_cum, mp_cum_kg,
       dt.
   - File name pattern:
       espa_xb6_8deps_YYYYMMDD_HHMMSS_Dk.npz

These NPZs are compatible with the analysis/plotting scripts like
`fleet_npz_viewer_espa8_xb6.py` and `fleet_euler_zyx_all_deps.py`.

Plots (display only)
--------------------
At the end of `run()` the script displays, but does not save:

- Aggregate across all deputies:
  * Bar chart: RAFTI Δv_cmd vs control Δv per deputy.
  * Boresight error vs time with a 1° reference line.
  * Roll-about-LOS error vs time with a 2° reference line.
- Per-deputy:
  * Multi-panel stack of H_SE3,d, torque/force norms, thrust usage, ω components,
    and MRP components.
  * Euler ZYX tracking (desired vs wrapped-achieved) to the local ESPA port.
  * Pointing quality (boresight and roll-about-LOS).
  * Relative torque/force error plots (desired→commanded, desired→applied).

Usage
-----
Run in a Basilisk-enabled Python environment:

    python basilik_scenario_espa42_rafti_xb6_fleet_se3.py

The script will:
- Build the ESPA(42") + RAFTI-GRIP + 8×XB6 6U fleet,
- Run the SE(3) pH control scenario,
- Save combined and per-deputy NPZ files in the working directory, and
- Display a sequence of diagnostic plots summarizing Δv, pointing, energy,
  and control behavior for the full fleet.
"""


from __future__ import annotations
import math, dataclasses as dc, traceback, datetime as _dt
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
    # choose in-plane axis to preserve based on cam_axis
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
        k = la.norm(e_R)
        scale = 1.0 + 1.8*math.tanh(2.5*k)      # up to ~2.8×
        return scale * self.K_R_base

    def _desired_rate_body(self, R: np.ndarray, R_d: np.ndarray) -> np.ndarray:
        if self.Rd_prev is None:
            self.Rd_prev = R_d.copy(); return np.zeros(3)
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
        self.thrust_total_hist.append(float(np.sum(self.Ts*delta)) if self.B.size and 'delta' in locals() and delta.size else 0.0)

        # store Euler (wrapped difference for plotting clarity)
        rpy   = dcm_to_euler_zyx(R)
        rpy_d = dcm_to_euler_zyx(R_d)
        diff  = wrap_to_pi(rpy - rpy_d)
        rpy_wrapped = rpy_d + diff         # achieved wrapped near desired
        self.euler_hist.append(rpy_wrapped)
        self.euler_des_hist.append(rpy_d)

        # Δv/prop accounting
        if self.B.size and 'delta' in locals() and delta.size>0:
            impulse=float(np.sum(self.Ts*delta)*dt)
            self.dv_cum += impulse/self.veh.mass_kg
            if impulse>0:
                self.mp_cum += impulse / (np.average(self.Isps, weights=self.Ts*delta)*9.80665)

        self.omega_prev=omega.copy(); self.v_prev=v_B.copy()

# ───────────────────────── RAFTI separation (multi-deputy) ─────────────────────────

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

# ───────────────────────── build/run/plot for 8 deputies ─────────────────────────

class Fleet:
    def __init__(self, num_deps: int = 8):
        assert num_deps >= 1
        self.dt=0.02
        self.sim = SimConfig()
        self.num_deps = int(num_deps)

        # ESPA circle: 42" (Ø=1.0668 m)
        self.espa_radius = 42.0*0.0254/2.0  # 0.5334 m
        z_mount = 0.0

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

        # Deputy (XB6 6U) config template with torque-capable 12-jet layout
        I_dep=np.diag([0.025,0.020,0.018]); m_dep=12.0
        T_small, Isp_small = 0.27, 217.0
        L = 0.065
        thr_template=[]
        for s in (+1,-1):
            thr_template += [Thruster(np.array([0, +L, +L]), np.array([s,0,0]), T_small, Isp_small),
                             Thruster(np.array([0, -L, -L]), np.array([s,0,0]), T_small, Isp_small)]
        for s in (+1,-1):
            thr_template += [Thruster(np.array([+L,0,+L]), np.array([0,s,0]), T_small, Isp_small),
                             Thruster(np.array([-L,0,-L]), np.array([0,s,0]), T_small, Isp_small)]
        for s in (+1,-1):
            thr_template += [Thruster(np.array([+L,+L,0]), np.array([0,0,s]), T_small, Isp_small),
                             Thruster(np.array([-L,-L,0]), np.array([0,0,s]), T_small, Isp_small)]

        # evenly spaced ESPA ports
        self.espa_port_r_B_list=[]
        self.espa_port_dir_B_list=[]
        for i in range(self.num_deps):
            ang = 2.0*math.pi * i / self.num_deps
            rB  = np.array([ self.espa_radius*math.cos(ang), self.espa_radius*math.sin(ang), z_mount ], float)
            dB  = np.array([ math.cos(ang),                  math.sin(ang),                  0.0     ], float)  # radial out
            self.espa_port_r_B_list.append(rB)
            self.espa_port_dir_B_list.append(dB)

        # containers
        self.deps : List[VehicleConfig] = []
        self.sc_deps : List[spacecraft.Spacecraft] = []
        self.nav_deps: List[simpleNav.SimpleNav] = []
        self.eft_dep_ctrl_list : List[extForceTorque.ExtForceTorque] = []
        self.eft_dep_sep_list  : List[extForceTorque.ExtForceTorque] = []

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

        # add carrier to tasks
        p=0
        self.sns.AddNewObject(self.nav_car,10+p); self.dyn.AddNewObject(self.sc_car,5+p)
        self.dyn.AddNewObject(self.eft_car_ctrl,10+p); self.dyn.AddNewObject(self.eft_car_sep,11+p)

        # Build each deputy
        for i in range(self.num_deps):
            # local port frame
            bY = self.espa_port_dir_B_list[i]          # radial outward
            bZ = np.array([0.0,0.0,1.0])
            if abs(np.dot(unit(bY), unit(bZ))) > 0.99: bZ=np.array([1.0,0.0,0.0])
            bX = unit(np.cross(bY,bZ)); bZ = unit(np.cross(bX,bY))
            R_BD_BC = np.column_stack((bX,bY,bZ))
            sigma0 = self.dcm_to_mrp_safe(R_BD_BC)

            dep_cfg = VehicleConfig(
                name=f"XB6_6U_{i+1}", mass_kg=12.0, inertia_B=np.diag([0.025,0.020,0.018]),
                sigma0_BN=sigma0, omega0_B=np.zeros(3),
                r0_N=self.espa_port_r_B_list[i].copy(), v0_N=np.zeros(3),
                zeta_att=1.25, wn_att=1.00,     # faster attitude
                zeta_vel=1.10, wn_vel=0.30,     # translation damping
                rw_tau_max=0.08,
                w_force=220.0, w_torque=60.0, reg_lambda=5e-4,
                thrusters=[Thruster(th.r_B.copy(), th.a_B.copy(), th.T, th.Isp) for th in thr_template]
            )
            self.deps.append(dep_cfg)

            sc_dep = spacecraft.Spacecraft(); sc_dep.ModelTag=f"SC_XB6_{i+1}"
            sc_dep.hub.mHub=dep_cfg.mass_kg; sc_dep.hub.IHubPntBc_B=dep_cfg.inertia_B.tolist()
            sc_dep.hub.sigma_BNInit=list(dep_cfg.sigma0_BN); sc_dep.hub.omega_BN_BInit=list(dep_cfg.omega0_B)
            sc_dep.hub.r_CN_NInit=list(dep_cfg.r0_N);       sc_dep.hub.v_CN_NInit=list(dep_cfg.v0_N)

            nav_dep = simpleNav.SimpleNav(); nav_dep.ModelTag=f"NAV_XB6_{i+1}"
            nav_dep.scStateInMsg.subscribeTo(sc_dep.scStateOutMsg)

            eft_ctrl = extForceTorque.ExtForceTorque(); eft_ctrl.ModelTag=f"EFT_XB6_{i+1}_ctrl"
            eft_sep  = extForceTorque.ExtForceTorque(); eft_sep .ModelTag=f"EFT_XB6_{i+1}_sep"
            sc_dep.addDynamicEffector(eft_ctrl); sc_dep.addDynamicEffector(eft_sep)

            self.sc_deps.append(sc_dep); self.nav_deps.append(nav_dep)
            self.eft_dep_ctrl_list.append(eft_ctrl); self.eft_dep_sep_list.append(eft_sep)

            # Add to tasks with per-deputy offsets
            base = 12 + 5*i
            self.sns.AddNewObject(nav_dep,  20+base)
            self.dyn.AddNewObject(sc_dep,   15+base)
            self.dyn.AddNewObject(eft_ctrl, 20+base)
            self.dyn.AddNewObject(eft_sep,  21+base)

        # Controllers
        last_sep_end = 10.0 + 0.30  # base + push duration
        self.ctrl_car = SE3Layer(self.car, self.dt, last_sep_end, self.sim.latency_s,
                                 enable_po_pc=self.sim.enable_po_pc, use_latency_predict=self.sim.use_latency_predict,
                                 cmd_smooth_tau_s=self.sim.cmd_smooth_tau_s, cmd_smooth_for_s=self.sim.cmd_smooth_for_s,
                                 shape_ramp_s=0.0, tau_deadband_nm=0.0015, f_deadband_n=0.015)

        self.ctrl_deps: List[SE3Layer] = []
        for i in range(self.num_deps):
            ctrl = SE3Layer(self.deps[i], self.dt, last_sep_end, self.sim.latency_s,
                            enable_po_pc=self.sim.enable_po_pc, use_latency_predict=self.sim.use_latency_predict,
                            cmd_smooth_tau_s=self.sim.cmd_smooth_tau_s, cmd_smooth_for_s=self.sim.cmd_smooth_for_s,
                            shape_ramp_s=self.sim.shape_ramp_s,
                            tau_deadband_nm=self.sim.tau_deadband_nm, f_deadband_n=self.sim.f_deadband_n)
            self.ctrl_deps.append(ctrl)

        # wiring
        self.eft_car_ctrl.cmdTorqueInMsg   .subscribeTo(self.ctrl_car.ctrlTorqueMsg)
        self.eft_car_ctrl.cmdForceBodyInMsg.subscribeTo(self.ctrl_car.ctrlForceMsg)
        for i in range(self.num_deps):
            self.eft_dep_ctrl_list[i].cmdTorqueInMsg   .subscribeTo(self.ctrl_deps[i].ctrlTorqueMsg)
            self.eft_dep_ctrl_list[i].cmdForceBodyInMsg.subscribeTo(self.ctrl_deps[i].ctrlForceMsg)

        # RAFTI schedule for each deputy
        self.sched: List[SepParams] = []
        self.t0_base=10.0; Tpush=0.30
        rng=np.random.default_rng(42)
        for i in range(self.num_deps):
            dv = float(np.clip(rng.normal(0.018, 0.005), 0.004, 0.025))
            F0 = self.deps[i].mass_kg * dv * math.pi / (2.0*Tpush)
            r_dep_B = rng.normal(scale=0.02, size=3)
            self.sched.append(SepParams(t_start=self.t0_base, t_end=self.t0_base+Tpush, F0=F0,
                                        s_car_B=self.espa_port_dir_B_list[i].copy(),
                                        r_car_B=self.espa_port_r_B_list[i].copy(),
                                        r_dep_B=r_dep_B, dv_cmd=dv))
        self.sep = RAFTISeparation(self.dt, self.sc_car, self.eft_car_sep,
                                   self.sc_deps, self.eft_dep_sep_list, self.sched,
                                   m_car=self.car.mass_kg, m_dep=self.deps[0].mass_kg)

        # Camera pointing parameters
        self.CAM_AXIS = "z"               # change to 'x' or 'y' if needed
        self.UP_REF_N = np.array([0.0,0.0,1.0])

        # Per-deputy position reference freezes at end of push
        self.p_ref_frozen=[False]*self.num_deps
        self.p_d_N_frozen=[np.zeros(3) for _ in range(self.num_deps)]

        # store continuous desired attitude between steps (per deputy)
        self.Rd_cont_prev=[None]*self.num_deps

        # init tasks
        for tsk in (self.sns, self.dyn):
            tsk.SelfInitTaskList(); tsk.ResetTaskList(0)

        need = self.t0_base + 120.0
        if need > self.sim.T_end: self.sim.T_end = need

    @staticmethod
    def dcm_to_mrp_safe(C: np.ndarray) -> np.ndarray:
        q = dcm_to_quat(C); q0,qv=q[0],q[1:]
        return (-qv/(1-q0) if 1+q0<1e-12 else qv/(1+q0))

    def _port_position_in_N(self, idx: int):
        C_BN_car = mrp_to_dcm(np.array(self.sc_car.scStateOutMsg.read().sigma_BN))
        C_NB_car = C_BN_car.T
        r_car_N  = np.array(self.nav_car.transOutMsg.read().r_BN_N, float)
        return r_car_N + C_NB_car @ self.espa_port_r_B_list[idx]

    def _camera_Rd_cont(self, idx: int, r_dep_N: np.ndarray):
        r_port_N = self._port_position_in_N(idx)
        los_N = r_port_N - r_dep_N
        if la.norm(los_N) < 1e-8:
            los_N = np.array([1.0,0.0,0.0])
        Rd = look_at_continuous(los_N, self.UP_REF_N, self.Rd_cont_prev[idx], cam_axis=self.CAM_AXIS)
        self.Rd_cont_prev[idx] = Rd.copy()
        return Rd, los_N

    # ───────────── save NPZ (combined + per-deputy) ─────────────
    def _save_npz(self, base_path_no_ext: str):
        """Collect run data (carrier + all deputies + schedule) and save to NPZ files."""
        # Combined payload
        payload = {}
        # carrier
        c = self.ctrl_car
        payload["carrier_t"]          = np.array(c.t_list, float)
        payload["carrier_Hd"]         = np.array(c.Hd_list, float)
        payload["carrier_tau_des"]    = np.array(c.tau_d_hist, float)
        payload["carrier_f_des"]      = np.array(c.f_d_hist, float)
        payload["carrier_tau_cmd"]    = np.array(c.torque_cmd_hist, float)
        payload["carrier_f_cmd"]      = np.array(c.force_cmd_hist, float)
        payload["carrier_tau_apply"]  = np.array(c.torque_apply_hist, float)
        payload["carrier_f_apply"]    = np.array(c.force_apply_hist, float)
        payload["carrier_omega_B"]    = np.array(c.omega_hist, float)
        payload["carrier_vB"]         = np.array(c.vB_hist, float)
        payload["carrier_sigma_BN"]   = np.array(c.sigma_hist, float)
        payload["carrier_euler_wrapped"] = np.array(c.euler_hist, float)
        payload["carrier_euler_des"]  = np.array(c.euler_des_hist, float)
        payload["carrier_thrust_sumN"]= np.array(c.thrust_total_hist, float)
        payload["carrier_po_scale"]   = np.array(c.po_scale_hist, float)
        payload["carrier_E_po"]       = np.array(c.E_po_hist, float)
        payload["carrier_dv_cum"]     = np.float64(c.dv_cum)
        payload["carrier_mp_cum_kg"]  = np.float64(c.mp_cum)

        # deputies
        for i, d in enumerate(self.ctrl_deps, start=1):
            p=f"D{i}_"
            payload[p+"t"]             = np.array(d.t_list, float)
            payload[p+"Hd"]            = np.array(d.Hd_list, float)
            payload[p+"tau_des"]       = np.array(d.tau_d_hist, float)
            payload[p+"f_des"]         = np.array(d.f_d_hist, float)
            payload[p+"tau_cmd"]       = np.array(d.torque_cmd_hist, float)
            payload[p+"f_cmd"]         = np.array(d.force_cmd_hist, float)
            payload[p+"tau_apply"]     = np.array(d.torque_apply_hist, float)
            payload[p+"f_apply"]       = np.array(d.force_apply_hist, float)
            payload[p+"omega_B"]       = np.array(d.omega_hist, float)
            payload[p+"vB"]            = np.array(d.vB_hist, float)
            payload[p+"sigma_BN"]      = np.array(d.sigma_hist, float)
            payload[p+"euler_wrapped"] = np.array(d.euler_hist, float)
            payload[p+"euler_des"]     = np.array(d.euler_des_hist, float)
            payload[p+"thrust_sumN"]   = np.array(d.thrust_total_hist, float)
            payload[p+"po_scale"]      = np.array(d.po_scale_hist, float)
            payload[p+"E_po"]          = np.array(d.E_po_hist, float)
            payload[p+"boresight_deg"] = np.array(d.boresight_err_hist, float)
            payload[p+"roll_about_LOS_deg"] = np.array(d.roll_about_los_err_hist, float)
            payload[p+"dv_cum"]        = np.float64(d.dv_cum)
            payload[p+"mp_cum_kg"]     = np.float64(d.mp_cum)

        # schedule / geometry / config
        payload["espa_radius_m"] = np.float64(self.espa_radius)
        payload["espa_ports_r_B"] = np.array(self.espa_port_r_B_list, float)
        payload["espa_ports_dir_B"] = np.array(self.espa_port_dir_B_list, float)
        payload["sched_t_start"] = np.array([sp.t_start for sp in self.sched], float)
        payload["sched_t_end"]   = np.array([sp.t_end   for sp in self.sched], float)
        payload["sched_F0_N"]    = np.array([sp.F0      for sp in self.sched], float)
        payload["sched_dv_cmd"]  = np.array([sp.dv_cmd  for sp in self.sched], float)
        payload["sched_r_car_B"] = np.array([sp.r_car_B for sp in self.sched], float)
        payload["sched_s_car_B"] = np.array([sp.s_car_B for sp in self.sched], float)
        payload["sched_r_dep_B"] = np.array([sp.r_dep_B for sp in self.sched], float)
        payload["dt"]            = np.float64(self.dt)
        payload["T_end"]         = np.float64(self.sim.T_end)
        payload["latency_s"]     = np.float64(self.sim.latency_s)
        payload["shape_ramp_s"]  = np.float64(self.sim.shape_ramp_s)

        # write combined
        combined_path = f"{base_path_no_ext}.npz"
        np.savez(combined_path, **payload)
        print(f"[SAVE] Combined NPZ: {combined_path}")

        # also save per-deputy
        for i, d in enumerate(self.ctrl_deps, start=1):
            per = {
                "t": np.array(d.t_list, float),
                "Hd": np.array(d.Hd_list, float),
                "tau_des": np.array(d.tau_d_hist, float),
                "f_des": np.array(d.f_d_hist, float),
                "tau_cmd": np.array(d.torque_cmd_hist, float),
                "f_cmd": np.array(d.force_cmd_hist, float),
                "tau_apply": np.array(d.torque_apply_hist, float),
                "f_apply": np.array(d.force_apply_hist, float),
                "omega_B": np.array(d.omega_hist, float),
                "vB": np.array(d.vB_hist, float),
                "sigma_BN": np.array(d.sigma_hist, float),
                "euler_wrapped": np.array(d.euler_hist, float),
                "euler_des": np.array(d.euler_des_hist, float),
                "thrust_sumN": np.array(d.thrust_total_hist, float),
                "po_scale": np.array(d.po_scale_hist, float),
                "E_po": np.array(d.E_po_hist, float),
                "boresight_deg": np.array(d.boresight_err_hist, float),
                "roll_about_LOS_deg": np.array(d.roll_about_los_err_hist, float),
                "dv_cum": np.float64(d.dv_cum),
                "mp_cum_kg": np.float64(d.mp_cum),
                "dt": np.float64(self.dt)
            }
            fname = f"{base_path_no_ext}_D{i}.npz"
            np.savez(fname, **per)
            print(f"[SAVE] Deputy NPZ D{i}: {fname}")

    def run(self):
        now_ns=0; N=int(self.sim.T_end/self.dt)
        last_sep_end = self.t0_base + 0.30

        for k in tqdm(range(N), desc="ESPA(42in)+XB6 ×8 — SE(3) tracker", ncols=80):
            t=k*self.dt

            # sensors first
            self.sns.ExecuteTaskList(now_ns)
            # separation
            self.sep.step(t)

            # carrier damping (hold near zero)
            self.ctrl_car.step(t, self.nav_car.attOutMsg.read(), self.nav_car.transOutMsg.read(),
                               R_d=np.eye(3), p_d_N=np.zeros(3))

            # each deputy
            for i in range(self.num_deps):
                r_dep_N = np.array(self.nav_deps[i].transOutMsg.read().r_BN_N, float)

                # freeze translation reference after push end
                if (not self.p_ref_frozen[i]) and t >= last_sep_end:
                    self.p_d_N_frozen[i] = r_dep_N.copy()
                    self.p_ref_frozen[i] = True
                p_ref = self.p_d_N_frozen[i] if self.p_ref_frozen[i] else r_dep_N.copy()

                # pointing
                if t >= last_sep_end:
                    R_d, los_N = self._camera_Rd_cont(i, r_dep_N)
                else:
                    R_d = mrp_to_dcm(self.deps[i].sigma0_BN); los_N=None

                self.ctrl_deps[i].step(t, self.nav_deps[i].attOutMsg.read(), self.nav_deps[i].transOutMsg.read(),
                                       R_d=R_d, p_d_N=p_ref, los_N=los_N, cam_axis=self.CAM_AXIS)

            # dynamics propagate
            self.dyn.ExecuteTaskList(now_ns)
            now_ns+=int(macros.sec2nano(self.dt))

        # ── save NPZ files (combined + per-deputy)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"espa_xb6_8deps_{ts}"
        self._save_npz(base)

        # report & plots (plots are displayed, not saved)
        self._report(last_sep_end)
        self._plots(last_sep_end)

    def _report(self, t_eng: float):
        c=self.ctrl_car
        print(f"[carrier] H_d non-increasing fraction after t≥{t_eng:.1f}s: {c.energy_noninc/max(1,c.energy_total):.3f}, "
              f"Δv={c.dv_cum:.4f} m/s, prop={c.mp_cum*1e3:.3f} g")
        for i,ctl in enumerate(self.ctrl_deps, start=1):
            print(f"[XB6-{i}]   H_d non-increasing fraction after t≥{t_eng:.1f}s: {ctl.energy_noninc/max(1,ctl.energy_total):.3f}, "
                  f"Δv={ctl.dv_cum:.4f} m/s, prop={ctl.mp_cum*1e3:.3f} g")

    def _plots(self, engage_t: float):
        # ── Aggregate Δv for all deputies
        dv_cmd = np.array([sp.dv_cmd for sp in self.sched])
        dv_ctl = np.array([ctl.dv_cum for ctl in self.ctrl_deps])
        idx    = np.arange(self.num_deps)
        w=0.35
        plt.figure(figsize=(10,5))
        plt.bar(idx-w/2, dv_cmd, width=w, label="RAFTI Δv_cmd")
        plt.bar(idx+w/2, dv_ctl, width=w, label="Control Δv")
        plt.xticks(idx, [f"D{i+1}" for i in range(self.num_deps)])
        plt.ylabel("Δv [m/s]"); plt.title("Separation Δv vs Control Δv — all deputies"); plt.legend(); plt.tight_layout(); plt.show()

        # ── Aggregate pointing quality (boresight & roll-about-LOS)
        plt.figure(figsize=(10,6))
        for i,ctl in enumerate(self.ctrl_deps):
            t = np.array(ctl.t_list); bores=np.array(ctl.boresight_err_hist)
            plt.plot(t, bores, label=f"D{i+1}")
        plt.axhline(1.0,ls="--",alpha=0.3)
        plt.xlabel("time [s]"); plt.ylabel("boresight [deg]"); plt.title("Boresight error vs time (all deputies)")
        plt.legend(ncol=4, fontsize=8); plt.tight_layout(); plt.show()

        plt.figure(figsize=(10,6))
        for i,ctl in enumerate(self.ctrl_deps):
            t = np.array(ctl.t_list); rollE=np.array(ctl.roll_about_los_err_hist)
            plt.plot(t, rollE, label=f"D{i+1}")
        plt.axhline(2.0,ls="--",alpha=0.3)
        plt.xlabel("time [s]"); plt.ylabel("roll about LOS [deg]"); plt.title("Roll-about-LOS error vs time (all deputies)")
        plt.legend(ncol=4, fontsize=8); plt.tight_layout(); plt.show()

        # ── Per-deputy detailed plots (same sets as before)
        for i,ctl in enumerate(self.ctrl_deps, start=1):
            t = np.array(ctl.t_list); Hd=np.array(ctl.Hd_list)
            tau_d=np.array(ctl.tau_d_hist); tau_c=np.array(ctl.torque_cmd_hist); tau_a=np.array(ctl.torque_apply_hist)
            f_d=np.array(ctl.f_d_hist); f_c=np.array(ctl.force_cmd_hist); f_a=np.array(ctl.force_apply_hist)
            omg=np.array(ctl.omega_hist); sig=np.array(ctl.sigma_hist); thrust=np.array(ctl.thrust_total_hist)

            def nrm(X): return la.norm(X,axis=1) if (isinstance(X,np.ndarray) and X.ndim==2 and X.shape[0]==t.size) else np.zeros_like(t)
            fig,axs=plt.subplots(6,1,figsize=(10,14),sharex=True)
            axs[0].plot(t,Hd); axs[0].axvline(engage_t,ls="--",alpha=0.6); axs[0].set_ylabel("H_SE3,d")
            axs[0].set_title(f"XB6 D{i} — energy & histories")
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

            # Euler tracking (wrapped)
            rpy   = np.array(ctl.euler_hist)
            rpy_d = np.array(ctl.euler_des_hist)
            if rpy.shape[0]==t.size:
                labels=["roll φ","pitch θ","yaw ψ"]
                fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
                for k in range(3):
                    axs[k].plot(t,rpy_d[:,k],'--',label=f"{labels[k]} (desired)")
                    axs[k].plot(t,rpy[:,k],label=f"{labels[k]} (achieved)")
                    axs[k].set_ylabel("[rad]"); axs[k].legend(loc="upper right")
                axs[-1].set_xlabel("time [s]"); fig.suptitle(f"XB6 D{i} — Euler ZYX tracking to its ESPA port (wrapped)"); fig.tight_layout(); plt.show()

            # pointing quality
            bores=np.array(ctl.boresight_err_hist); rollE=np.array(ctl.roll_about_los_err_hist)
            fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
            ax[0].plot(t,bores); ax[0].axhline(1.0,ls="--",alpha=0.3); ax[0].set_ylabel("boresight [deg]")
            ax[1].plot(t,rollE); ax[1].axhline(2.0,ls="--",alpha=0.3); ax[1].set_ylabel("roll about LOS [deg]"); ax[1].set_xlabel("time [s]")
            fig.suptitle(f"XB6 D{i} — Pointing quality"); fig.tight_layout(); plt.show()

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
        f=Fleet(num_deps=8)
        f.run()
        print("[DONE] ESPA(42\") + RAFTI‑GRIP + 8×XB6 6U — NPZ saved; plots displayed.")
    except Exception:
        print("\n[ERROR] Unhandled exception:\n")
        print(traceback.format_exc())

if __name__=="__main__":
    main()
