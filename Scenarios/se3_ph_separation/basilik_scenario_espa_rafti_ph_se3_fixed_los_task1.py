#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_scenario_espa_rafti_ph_se3_fixed_los_task1.py

Purpose
-------
Task 1 sandbox for ARISE: simulate RAFTI-based ESPA separation with a
port–Hamiltonian SE(3) Layer-1 damping controller and *fixed* LOS shaping for
each deputy. Every deputy is pushed off an ESPA-like ring by an equal-and-
opposite half-sine RAFTI force, then gently steered so its body boresight axis
tracks a prescribed inertial LOS direction over a 240 s window.

Key features
------------
1. RAFTI half-sine separation
   - Equal-and-opposite forces/torques applied via `ExtForceTorque`:
       * At each RAFTI mount on the carrier body frame.
       * At the corresponding deputy mount point in that deputy’s body frame.
   - For deputy i:
       * Start time: t0 = 10 s + (i−1)·5 s
       * Duration:   Tpush = 0.30 s
       * Δv_cmd ~ N(0.018, 0.005²) m/s, clipped to [0.004, 0.025] m/s
         (consistent with a RAFTI-like 25 mm/s separation bound).
       * Peak force F0 computed from Δv_cmd, deputy mass, and half-sine profile.
   - At push start, RAFTI checks print warnings if:
       * Equivalent mass meq(m_car, m_dep) > 1000 kg.
       * Δv_cmd > 0.025 m/s.
       * F0 > 11.6 kN (contact-load rating).

2. Fixed LOS shaping in N
   - `LOS_OVERRIDE` (8×3) stores a unit LOS_N per deputy, in the inertial frame.
   - `BORE_IDX` and `BORE_SGN` specify which body axis is the boresight:
       * 0/1/2 for x/y/z; sign decides forward (+) or backward (−).
   - For each deputy, a *constant* reference attitude C_RN (N→B) is built such that:
       * `C_RN^T · e_bore = LOS_N`, i.e., the selected body axis expressed in N
         equals the desired LOS.
       * `frame_from_los()` constructs a full orthonormal body triad from LOS_N
         and a tangent, then returns C_RN with rows equal to the body axes in N.

3. PH(SE3) Layer-1 damping with LOS “spring”
   - For each vehicle (carrier + deputies), `PHLayer1` implements a simple
     Layer-1 SE(3) energy tank:
       * Rotational damping:
             τ_damp = −K_ω ω̄,   K_ω ∝ J / τ_ω
       * Translational damping:
             f_damp = −K_v v̄_B, K_v ∝ m / τ_v
         using mid-point (Tustin) states:
             ω̄ = ½(ω_k + ω_{k−1}),  v̄_B = ½(v_B,k + v_B,k−1).
   - LOS shaping term:
       * Compute C_BR = C_BN C_RNᵀ and its MRP σ_BR.
       * Add a small SO(3) spring torque:
             τ_shape = −K_sigma σ_BR
         with scalar K_sigma = K_SIGMA_ALIGN, contributing a potential
             V_shape = ½ K_sigma ‖σ_BR‖²
         so H_d = H_mech + V_shape.
   - Latency:
       * Commanded (τ_cmd, f_cmd) are pushed through a latency buffer of
         length L = round(latency_s / dt) before being applied to the
         ExtForceTorque module.

4. Vehicles and tasks
   - Carrier:
       * Mass ≈ 650 kg, inertia diag([1200,1000,800]) kg·m².
       * 4 MR-22N-like thrusters are defined, but deputies here use pure
         Layer-1 damping without thruster BNLS for simplicity.
   - Deputies (8 total):
       * Mass ≈ 12 kg, inertia diag([0.025,0.020,0.018]) kg·m².
       * Mounted around a 1 m ring; local body frames aligned with port normals.
   - Basilisk tasks:
       * `sns` task runs SimpleNav for carrier + all deputies.
       * `dyn` task runs spacecraft dynamics and RAFTI + control effectors.

5. Optional RHIG reference (for r/v error plots)
   - If RHIG NPZs (apophis_rendezvous_full_D*.npz) are available:
       * Each deputy is associated with a RHIG file via simple name matching
         (D0/D1/…).
       * For each t, desired r_des and v_des are interpolated from RHIG
         using a local per-deputy time-offset window (t_dep + ALIGN_S).
   - If no RHIG is found or valid, the script falls back to ‖r‖, ‖v‖ as proxies.

Outputs and diagnostics (display only)
--------------------------------------
1. Text report
   - For each deputy:
       * Engage time t_eng = t_sep_end + 0 s (end of its push).
       * H_d monotonic fraction: (# steps with H_d(k) ≤ H_d(k−1)) / total.
       * H_mech monotonic fraction: same for H_mech.
       * Max |τ_appl| [N·m].
       * Mean relative torque tracking error:
             ⟨‖τ_des − τ_appl‖ / ‖τ_des‖⟩.
       * Δv estimate (from control effort) [m/s].

   - Final LOS/SO(3) errors at each deputy’s “handoff” time:
       * t_handoff = t_dep + ALIGN_S.
       * Desired LOS_N from C_RN^T e_bore.
       * Achieved LOS_N from C_BN^T e_bore.
       * LOS error angle [deg].
       * SO(3) attitude error angle between C_BN and C_RN [deg].

2. Plots (per deputy)
   - (A) R entries:
       * 3×3 grid of C_BN(t) entries vs constant C_RN^des entries.
       * Time traces for each R_ij with dashed constant reference.
   - (B) Momenta and torques:
       * p = J ω_B components vs time.
       * u = applied torque components vs time.
   - (C) Position/velocity errors:
       * If RHIG present: ‖r − r_des‖ [m] and ‖v − v_des‖ [m/s].
       * Else: ‖r‖ and ‖v‖ plotted as proxies.

All figures are displayed using `plt.show()`; nothing is saved to disk.

Usage
-----
Run in a Basilisk-enabled Python environment, optionally with RHIG NPZs
available in `APOPHIS_FLEET_DIR`:

    python basilik_scenario_espa_rafti_ph_se3_fixed_los_task1.py

The script will:
- Build an ESPA-ring carrier with eight deputies.
- Execute RAFTI half-sine separations.
- Run pH(SE3) Layer-1 damping plus fixed LOS shaping per deputy.
- Print controller performance summaries and display per-deputy diagnostics for
  attitude/LOS tracking and r/v error behavior.
"""


from __future__ import annotations
import os, math, dataclasses as dc
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.max_open_warning'] = 100

# Basilisk
from Basilisk.architecture import messaging, sys_model_task
from Basilisk.simulation   import spacecraft, simpleNav, extForceTorque
from Basilisk.utilities    import macros

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x


# ─────────────────────────────────────────────────────────────────────────────
# 0) User LOS targets (N frame). Each row is one deputy’s target LOS (unit).
#    Boresight axis in body to align with LOS: choose x(0)/y(1)/z(2) and sign.
# ─────────────────────────────────────────────────────────────────────────────
LOS_OVERRIDE = np.array([
  [-0.4517,-0.7381,-0.5011],
  [-0.8587,-0.4330,-0.2743],
  [-0.5078,-0.8231,-0.2544],
  [-0.3483,-0.7868,-0.5095],
  [-0.8622,-0.3805,-0.3343],
  [-0.6578,-0.7075,-0.2584],
  [-0.6534,-0.7362,-0.1760],
  [-0.6359,-0.4818,-0.6029]
], dtype=float)

BORE_IDX, BORE_SGN = 1, +1   # align body +Y axis to LOS (common for ESPA mounts)


# ─────────────────────────────────────────────────────────────────────────────
# 1) Small utilities + attitude algebra
# ─────────────────────────────────────────────────────────────────────────────
def unit(v: np.ndarray) -> np.ndarray:
    n = la.norm(v); return (v/n if n>1e-12 else np.array([1.0,0.0,0.0]))

def normalize(v: np.ndarray, eps=1e-12) -> Tuple[np.ndarray,float]:
    n = la.norm(v); return ((v/n if n>=eps else np.zeros_like(v)), n)

def skew(v: np.ndarray) -> np.ndarray:
    x,y,z=v; return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]], float)

def mrp_to_dcm(sigma: np.ndarray) -> np.ndarray:
    s2=float(np.dot(sigma,sigma)); I3=np.eye(3)
    if s2<1e-16: return I3
    S=skew(sigma)
    return I3 + (8*S@S - 4*(1-s2)*S) / (1+s2)**2

def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    tr=np.trace(C)
    if tr>0:
        S=math.sqrt(tr+1.0)*2.0; q0=0.25*S
        q1=(C[2,1]-C[1,2])/S; q2=(C[0,2]-C[2,0])/S; q3=(C[1,0]-C[0,1])/S
    else:
        i=int(np.argmax([C[0,0],C[1,1],C[2,2]]))
        if i==0:
            S=math.sqrt(1.0+C[0,0]-C[1,1]-C[2,2])*2.0
            q0=(C[2,1]-C[1,2])/S; q1=0.25*S; q2=(C[0,1]+C[1,0])/S; q3=(C[0,2]+C[2,0])/S
        elif i==1:
            S=math.sqrt(1.0+C[1,1]-C[0,0]-C[2,2])*2.0
            q0=(C[0,2]-C[2,0])/S; q1=(C[0,1]+C[1,0])/S; q2=0.25*S; q3=(C[1,2]+C[2,1])/S
        else:
            S=math.sqrt(1.0+C[2,2]-C[0,0]-C[1,1])*2.0
            q0=(C[1,0]-C[0,1])/S; q1=(C[0,2]+C[2,0])/S; q2=(C[1,2]+C[2,1])/S; q3=0.25*S
    q=np.array([q0,q1,q2,q3], float); return q/la.norm(q)

def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    q0,qv=q[0],q[1:]; return -qv/(1-q0) if 1+q0<1e-12 else qv/(1+q0)

def dcm_to_mrp(C: np.ndarray) -> np.ndarray:
    return quat_to_mrp(dcm_to_quat(C))

def frame_from_los(C_N: np.ndarray, bore_idx:int, bore_sgn:int) -> np.ndarray:
    """
    Build C_RN (N→B) such that  C_RN^T · e_bore = LOS_N.
    i.e., the body boresight axis (in N) equals the desired LOS.
    Implementation: set the corresponding **row** to LOS_N, and complete an ONB.
    """
    los = unit(C_N)
    # build a tangent to LOS
    t = np.array([1,0,0]) if abs(los[0])<0.9 else np.array([0,1,0])
    if bore_idx == 2:           # ẑ is boresight
        zB = los if bore_sgn>0 else -los
        xB = unit(np.cross(t, zB)); yB = unit(np.cross(zB, xB))
    elif bore_idx == 1:         # ŷ is boresight
        yB = los if bore_sgn>0 else -los
        zB = unit(np.cross(np.cross(t, yB), yB)); xB = unit(np.cross(yB, zB))
    else:                        # x̂ is boresight
        xB = los if bore_sgn>0 else -los
        yB = unit(np.cross(np.cross(t, xB), xB)); zB = unit(np.cross(xB, yB))
    # C_RN (N→B) has **rows** equal to B axes expressed in N
    return np.vstack((xB, yB, zB))


# ─────────────────────────────────────────────────────────────────────────────
# 2) RHIG loader (optional, for r/v desired)
# ─────────────────────────────────────────────────────────────────────────────
class RHIG:
    def __init__(self, path: Path):
        n = np.load(path, allow_pickle=True)
        self.t = np.array(n["t"], float)
        self.rD = np.array(n["rD_N"], float) if "rD_N" in n else None
        self.vD = np.array(n["vD_N"], float) if "vD_N" in n else None
        self.t0 = float(n["t0"][0]) if "t0" in n else float(self.t[0])

    def interp(self, arr: np.ndarray, t_abs: float) -> np.ndarray:
        t = self.t
        if t_abs<=t[0]:  return arr[0]
        if t_abs>=t[-1]: return arr[-1]
        j = np.searchsorted(t, t_abs) - 1
        f = (t_abs - t[j])/(t[j+1]-t[j])
        return (1-f)*arr[j] + f*arr[j+1]

    def r_at(self, t_abs: float) -> Optional[np.ndarray]:
        return None if self.rD is None else self.interp(self.rD, t_abs)

    def v_at(self, t_abs: float) -> Optional[np.ndarray]:
        return None if self.vD is None else self.interp(self.vD, t_abs)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Vehicle & sim config
# ─────────────────────────────────────────────────────────────────────────────
@dc.dataclass
class Thruster:
    r_B: np.ndarray; a_B: np.ndarray; T: float; Isp: float; name: str="T"

@dc.dataclass
class VehicleConfig:
    name: str; mass_kg: float; inertia_B: np.ndarray
    sigma0_BN: np.ndarray; omega0_B: np.ndarray
    r0_N: np.ndarray; v0_N: np.ndarray
    tau_omega_s: float; tau_v_s: float; rw_tau_max: float
    w_force: float; w_torque: float; reg_lambda: float
    thrusters: List[Thruster]=dc.field(default_factory=list)

@dc.dataclass
class SimConfig:
    dt: float = 0.02
    latency_s: float = 0.05
    ALIGN_S: float = 240.0
    K_SIGMA_ALIGN: float = 0.02          # small SO(3) spring
    # RHIG NPZs (optional, for r/v desired)
    rhig_dir: Path = Path(os.getenv("APOPHIS_FLEET_DIR", "."))
    rhig_glob: str = "apophis_rendezvous_full_D*.npz"


# ─────────────────────────────────────────────────────────────────────────────
# 4) pH(SE3) layer with energy/track logs
# ─────────────────────────────────────────────────────────────────────────────
class PHLayer1:
    def __init__(self, veh: VehicleConfig, dt: float, engage_time: float,
                 latency_s: float, C_RN_des: np.ndarray, K_sigma: float):
        self.veh=veh; self.dt=dt; self.engage_time=engage_time
        J=veh.inertia_B; m=veh.mass_kg
        self.K_omega=(1.0/max(veh.tau_omega_s,1e-6))*J
        self.K_v    =(m   /max(veh.tau_v_s,1e-6))*np.eye(3)
        self.K_sigma=float(K_sigma)
        self.C_RN_des=np.array(C_RN_des, float)

        self.ctrlTorqueMsg = messaging.CmdTorqueBodyMsg()
        self.ctrlForceMsg  = messaging.CmdForceBodyMsg()
        self.L=int(round(max(0.0,latency_s)/dt))
        self.buf_tau: List[np.ndarray]=[]; self.buf_for: List[np.ndarray]=[]

        self.B=np.zeros((6,0)); self.Ts=np.zeros(0); self.Isps=np.zeros(0)
        if veh.thrusters:
            cols=[]; Ts=[]; Isps=[]
            for th in veh.thrusters:
                r=th.r_B.reshape(3); a=unit(th.a_B.reshape(3))
                cols.append(np.hstack((np.cross(r, th.T*a), th.T*a)))
                Ts.append(th.T); Isps.append(th.Isp)
            self.B=np.column_stack(cols); self.Ts=np.array(Ts); self.Isps=np.array(Isps)
        self.delta_prev=None

        # logs
        self.t_list=[]; self.Hd_list=[]; self.Hm_list=[]
        self.sigma_hist=[]; self.omega_hist=[]; self.rN_hist=[]; self.vN_hist=[]
        self.tau_d_hist=[]; self.tau_cmd_hist=[]; self.tau_appl_hist=[]
        self.energy_total=0; self.energy_noninc=0; self.me_total=0; self.me_noninc=0
        self.dv_cum=0.0; self.mp_cum=0.0
        self.omega_prev=np.zeros(3); self.v_prev=np.zeros(3)
        self._Hd_prev=None; self._Hm_prev=None

    def _bnls(self,B,u,delta_prev,w_tau,w_for,lam,lb,ub):
        n=B.shape[1]
        if delta_prev is None or delta_prev.shape[0]!=n: delta_prev=np.zeros(n)
        W=np.diag([w_tau,w_tau,w_tau,w_for,w_for,w_for])
        BtWB=B.T@W@B + lam*np.eye(n); BtWu=B.T@W@u + lam*delta_prev
        delta=np.clip(la.solve(BtWB,BtWu),lb,ub)
        for _ in range(5):
            low=(delta<=lb+1e-9); high=(delta>=ub-1e-9); free=~(low|high)
            if not np.any(free): break
            Bf=B[:,free]
            K=la.pinv(Bf.T@W@Bf + lam*np.eye(np.count_nonzero(free)))
            rhs=Bf.T@W@(u - B[:,low]@np.full(np.count_nonzero(low), lb)
                          - B[:,high]@np.full(np.count_nonzero(high), ub)) + lam*delta_prev[free]
            delta_f=K@rhs; delta=delta.copy(); delta[free]=delta_f; delta=np.clip(delta,lb,ub)
        return delta

    def step(self, t: float, attNav: messaging.NavAttMsgPayload, transNav: messaging.NavTransMsgPayload):
        J=self.veh.inertia_B; m=self.veh.mass_kg
        sigma=np.array(attNav.sigma_BN); C_BN=mrp_to_dcm(sigma)
        omega=np.array(attNav.omega_BN_B)
        rN=np.array(transNav.r_BN_N); vN=np.array(transNav.v_BN_N); vB=C_BN@vN

        # LOS spring
        C_BR = C_BN @ self.C_RN_des.T
        sigma_BR = dcm_to_mrp(C_BR)
        tau_shape = - self.K_sigma * sigma_BR
        V_shape   = 0.5 * self.K_sigma * float(sigma_BR @ sigma_BR)

        # pH damping (mid-point)
        omega_bar=0.5*(omega+self.omega_prev); v_bar=0.5*(vB+self.v_prev)
        tau_des = - self.K_omega @ omega_bar + tau_shape
        f_des   = - self.K_v     @ v_bar

        # simple torque application (can be extended with BNLS)
        tau_cmd = tau_des.copy()
        tn = la.norm(tau_cmd)
        if tn>self.veh.rw_tau_max>0: tau_cmd *= (self.veh.rw_tau_max/tn)
        f_cmd = f_des.copy()

        # latency buffer
        self.buf_tau.append(tau_cmd); self.buf_for.append(f_cmd)
        if len(self.buf_tau) > self.L:
            tau_appl=self.buf_tau.pop(0); f_appl=self.buf_for.pop(0)
        else:
            tau_appl=np.zeros(3); f_appl=np.zeros(3)

        # publish
        t_pl=messaging.CmdTorqueBodyMsgPayload(); t_pl.torqueRequestBody=list(tau_appl)
        f_pl=messaging.CmdForceBodyMsgPayload();  f_pl.forceRequestBody =list(f_appl)
        self.ctrlTorqueMsg.write(t_pl); self.ctrlForceMsg.write(f_pl)

        # energies, monotonicity
        Hm = 0.5*float(omega.T@(J@omega)) + 0.5*float(m*(vB@vB))
        Hd = Hm + V_shape
        if t >= self.engage_time:
            self.energy_total += 1; self.me_total += 1
            if self._Hd_prev is not None and Hd <= self._Hd_prev + 1e-12:
                self.energy_noninc += 1
            if self._Hm_prev is not None and Hm <= self._Hm_prev + 1e-12:
                self.me_noninc += 1
        self._Hd_prev = Hd; self._Hm_prev = Hm

        # logs
        self.t_list.append(t); self.Hd_list.append(Hd); self.Hm_list.append(Hm)
        self.sigma_hist.append(sigma.copy()); self.omega_hist.append(omega.copy())
        self.rN_hist.append(rN.copy()); self.vN_hist.append(vN.copy())
        self.tau_d_hist.append(tau_des.copy()); self.tau_cmd_hist.append(tau_cmd.copy()); self.tau_appl_hist.append(tau_appl.copy())

        self.omega_prev=omega.copy(); self.v_prev=vB.copy()


# ─────────────────────────────────────────────────────────────────────────────
# 5) RAFTI equal-and-opposite half-sine separation
# ─────────────────────────────────────────────────────────────────────────────
@dc.dataclass
class SepParams:
    t_start: float; t_end: float; F0: float
    s_car_B: np.ndarray; r_car_B: np.ndarray; r_dep_B: np.ndarray
    dv_cmd: float

def meq(m1,m2): return (m1*m2)/(m1+m2)

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

    @staticmethod
    def _hs(F0,t,t0,T):
        return 0.0 if (t<t0 or t>t0+T) else F0*math.sin(math.pi*(t-t0)/T)

    def step(self, t: float):
        C_BN_car = mrp_to_dcm(np.array(self.sc_car.scStateOutMsg.read().sigma_BN))
        C_NB_car = C_BN_car.T
        tau_car = np.zeros(3); f_car = np.zeros(3)
        for i,sp in enumerate(self.sched):
            C_BN_dep = mrp_to_dcm(np.array(self.sc_dep[i].scStateOutMsg.read().sigma_BN))
            C_NB_dep = C_BN_dep.T

            # rating warnings at push start
            if abs(t - sp.t_start) < 0.5*self.dt:
                m_eq = meq(self.m_car, self.m_dep)
                if m_eq > 1000.0:
                    print(f"[RAFTI warn dep{i+1}] m_eq={m_eq:.1f} kg > 1000 kg")
                if sp.dv_cmd > 0.025:
                    print(f"[RAFTI warn dep{i+1}] Δv_cmd={sp.dv_cmd:.3f} m/s > 0.025")
                if sp.F0 > 11600.0:
                    print(f"[RAFTI warn dep{i+1}] peak F0={sp.F0:.0f} N > 11600")

            Fmag = self._hs(sp.F0, t, sp.t_start, sp.t_end-sp.t_start)
            if Fmag <= 0.0:
                tm=messaging.CmdTorqueBodyMsgPayload(); fm=messaging.CmdForceBodyMsgPayload()
                tm.torqueRequestBody=[0,0,0]; fm.forceRequestBody=[0,0,0]
                self.dep_tauMsg[i].write(tm); self.dep_forMsg[i].write(fm); continue

            s_N = C_NB_car @ sp.s_car_B
            F_N = Fmag * s_N
            f_car_i_B =  C_BN_car @ (+F_N)
            f_dep_i_B =  C_BN_dep @ (-F_N)
            tau_car_i = np.cross(sp.r_car_B, f_car_i_B)
            tau_dep_i = np.cross(sp.r_dep_B, f_dep_i_B)

            tm=messaging.CmdTorqueBodyMsgPayload(); fm=messaging.CmdForceBodyMsgPayload()
            tm.torqueRequestBody=list(tau_dep_i); fm.forceRequestBody=list(f_dep_i_B)
            self.dep_tauMsg[i].write(tm); self.dep_forMsg[i].write(fm)

            tau_car += tau_car_i; f_car += f_car_i_B

        tm=messaging.CmdTorqueBodyMsgPayload(); fm=messaging.CmdForceBodyMsgPayload()
        tm.torqueRequestBody=list(tau_car); fm.forceRequestBody=list(f_car)
        self.car_tauMsg.write(tm); self.car_forMsg.write(fm)


# ─────────────────────────────────────────────────────────────────────────────
# 6) Fleet assembly, run, reporting, display
# ─────────────────────────────────────────────────────────────────────────────
class Fleet:
    def __init__(self):
        self.cfg = SimConfig()

        # Carrier (illustrative jets)
        I_car=np.diag([1200.,1000.,800.]); m_car=650.0
        mr22_T, mr22_Isp = 22.0, 235.0
        thr_c = [
            Thruster(np.array([ 0.8, 0.0,  0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp, "MR22_1"),
            Thruster(np.array([-0.8, 0.0,  0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp, "MR22_2"),
            Thruster(np.array([ 0.8, 0.0, -0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp, "MR22_3"),
            Thruster(np.array([-0.8, 0.0, -0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp, "MR22_4"),
        ]
        self.car = VehicleConfig("carrier", m_car, I_car, np.zeros(3), np.zeros(3),
                                 np.zeros(3), np.zeros(3),
                                 60.0, 40.0, 0.02, 100.0, 1.0, 1e-4, thr_c)

        # Deputies (MPS-like)
        I_dep=np.diag([0.025,0.020,0.018]); m_dep=12.0
        self.deps: List[VehicleConfig]=[]
        ring_R=1.0
        for i in range(8):
            ang=i*(2*math.pi/8)
            s_car_B=np.array([math.cos(ang), math.sin(ang), 0.0])
            r_car_B=np.array([ring_R*math.cos(ang), ring_R*math.sin(ang), 0.0])
            # deputy mount frame: +Y along outward normal
            bY=unit(s_car_B); bZ=np.array([0,0,1.0])
            if abs(np.dot(bY,bZ))>0.99: bZ=np.array([1,0,0])
            bX=unit(np.cross(bY,bZ)); bZ=unit(np.cross(bX,bY))
            C_BD_BC=np.vstack((bX, bY, bZ))  # N→B rows
            sigma0=dcm_to_mrp(C_BD_BC)
            self.deps.append(
                VehicleConfig(f"dep{i+1}", m_dep, I_dep, sigma0, np.zeros(3),
                              r_car_B.copy(), np.zeros(3),
                              22.0, 28.0, 0.02,    # slightly stronger rot damping
                              200.0, 0.5, 2e-4, []))

        # Tasks
        self.dt=self.cfg.dt
        self.dyn=sys_model_task.SysModelTask(int(macros.sec2nano(self.dt)))
        self.sns=sys_model_task.SysModelTask(int(macros.sec2nano(self.dt)))

        # Carrier sc/nav/eff
        self.sc_car=spacecraft.Spacecraft(); self.sc_car.ModelTag="SC_car"
        self.sc_car.hub.mHub=self.car.mass_kg; self.sc_car.hub.IHubPntBc_B=self.car.inertia_B.tolist()
        self.sc_car.hub.sigma_BNInit=self.car.sigma0_BN.tolist()
        self.sc_car.hub.omega_BN_BInit=self.car.omega0_B.tolist()
        self.sc_car.hub.r_CN_NInit=self.car.r0_N.tolist(); self.sc_car.hub.v_CN_NInit=self.car.v0_N.tolist()
        self.nav_car=simpleNav.SimpleNav(); self.nav_car.scStateInMsg.subscribeTo(self.sc_car.scStateOutMsg)
        self.eft_car_ctrl=extForceTorque.ExtForceTorque()
        self.eft_car_sep =extForceTorque.ExtForceTorque()
        self.sc_car.addDynamicEffector(self.eft_car_ctrl); self.sc_car.addDynamicEffector(self.eft_car_sep)

        # Deputies sc/nav/eff
        self.sc_dep=[]; self.nav_dep=[]; self.eft_dep_ctrl=[]; self.eft_dep_sep=[]
        for dep in self.deps:
            sc=spacecraft.Spacecraft(); sc.ModelTag=f"SC_{dep.name}"
            sc.hub.mHub=dep.mass_kg; sc.hub.IHubPntBc_B=dep.inertia_B.tolist()
            sc.hub.sigma_BNInit=dep.sigma0_BN.tolist(); sc.hub.omega_BN_BInit=dep.omega0_B.tolist()
            sc.hub.r_CN_NInit=dep.r0_N.tolist(); sc.hub.v_CN_NInit=dep.v0_N.tolist()
            nav=simpleNav.SimpleNav(); nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
            ec=extForceTorque.ExtForceTorque(); es=extForceTorque.ExtForceTorque()
            sc.addDynamicEffector(ec); sc.addDynamicEffector(es)
            self.sc_dep.append(sc); self.nav_dep.append(nav)
            self.eft_dep_ctrl.append(ec); self.eft_dep_sep.append(es)

        # add to tasks
        self.sns.AddNewObject(self.nav_car,10); self.dyn.AddNewObject(self.sc_car,5)
        self.dyn.AddNewObject(self.eft_car_ctrl,10); self.dyn.AddNewObject(self.eft_car_sep,11)
        for k,(sc,nav,ec,es) in enumerate(zip(self.sc_dep,self.nav_dep,self.eft_dep_ctrl,self.eft_dep_sep),1):
            self.sns.AddNewObject(nav,10+10*k)
            self.dyn.AddNewObject(sc,5+10*k)
            self.dyn.AddNewObject(ec,10+10*k)
            self.dyn.AddNewObject(es,11+10*k)
        for t in (self.sns,self.dyn): t.SelfInitTaskList(); t.ResetTaskList(0)

        # constant desired C_RN from LOS
        self.C_RN_des = [frame_from_los(LOS_OVERRIDE[i], BORE_IDX, BORE_SGN) for i in range(8)]

        # controllers
        last_sep_end = 10.0 + (len(self.deps)-1)*5.0 + 0.30
        self.ctrl_car=PHLayer1(self.car, self.dt, last_sep_end, self.cfg.latency_s,
                               C_RN_des=np.eye(3), K_sigma=0.0)
        self.ctrl_dep=[]
        for i,dep in enumerate(self.deps):
            t_end = 10.0 + i*5.0 + 0.30
            self.ctrl_dep.append(PHLayer1(dep, self.dt, t_end, self.cfg.latency_s,
                                          C_RN_des=self.C_RN_des[i],
                                          K_sigma=self.cfg.K_SIGMA_ALIGN))
        self.eft_car_ctrl.cmdTorqueInMsg   .subscribeTo(self.ctrl_car.ctrlTorqueMsg)
        self.eft_car_ctrl.cmdForceBodyInMsg.subscribeTo(self.ctrl_car.ctrlForceMsg)
        for ctl,ec in zip(self.ctrl_dep,self.eft_dep_ctrl):
            ec.cmdTorqueInMsg   .subscribeTo(ctl.ctrlTorqueMsg)
            ec.cmdForceBodyInMsg.subscribeTo(ctl.ctrlForceMsg)

        # separation schedule (half-sine, 0.30 s, from 10 s, 5 s apart)
        self.t0_base=10.0; self.spacing=5.0; Tpush=0.30
        self.sched=[]
        rng = np.random.default_rng(42)
        for i in range(8):
            t0 = self.t0_base + i*self.spacing
            dv = float(np.clip(np.random.default_rng(42+i).normal(0.018,0.005), 0.004, 0.025))
            F0 = self.deps[i].mass_kg * dv * math.pi / (2.0*Tpush)
            ang=i*(2*math.pi/8)
            s_car_B=unit(np.array([math.cos(ang), math.sin(ang), 0.0]))
            r_car_B=np.array([ring_R*math.cos(ang), ring_R*math.sin(ang), 0.0])
            r_dep_B=rng.normal(scale=0.02, size=3)
            self.sched.append(SepParams(t0,t0+Tpush,F0,s_car_B,r_car_B,r_dep_B,dv))
        self.sep=RAFTISeparation(self.dt, self.sc_car, self.eft_car_sep,
                                 self.sc_dep, self.eft_dep_sep, self.sched,
                                 m_car=self.car.mass_kg, m_dep=self.deps[0].mass_kg)

        # overall horizon so last deputy gets 240 s window
        self.Tsim = self.t0_base + (len(self.deps)-1)*self.spacing + self.cfg.ALIGN_S

        # optional RHIG bundles for r/v desired (if available)
        rhig_files = sorted(self.cfg.rhig_dir.glob(self.cfg.rhig_glob))
        self.rhig = [None]*8
        if rhig_files:
            for i in range(8):
                cand=[p for p in rhig_files if f"D{i}" in p.stem]
                path=cand[0] if cand else rhig_files[min(i, len(rhig_files)-1)]
                try:
                    self.rhig[i]=RHIG(path)
                    print(f"[RHIG] dep{i+1} ↔ {path.name}")
                except Exception as e:
                    print(f"[RHIG] failed to load for dep{i+1}: {e}")

    def run(self):
        now_ns=0; N=int(self.Tsim/self.dt)+1
        for k in tqdm(range(N), desc="RAFTI + pH(SE3) with LOS shaping", ncols=80):
            t=k*self.dt
            self.sns.ExecuteTaskList(now_ns)
            self.sep.step(t)
            self.ctrl_car.step(t, self.nav_car.attOutMsg.read(), self.nav_car.transOutMsg.read())
            for ctl,nav in zip(self.ctrl_dep,self.nav_dep):
                ctl.step(t, nav.attOutMsg.read(), nav.transOutMsg.read())
            self.dyn.ExecuteTaskList(now_ns)
            now_ns += int(macros.sec2nano(self.dt))

        self._print_controller_report()
        self._display_plots()

    # ── Controller report (monotonicity, stats, final LOS/SO(3) errors)
    def _print_controller_report(self):
        e_bore = np.zeros(3); e_bore[BORE_IDX] = float(BORE_SGN)
        print("\n=== Controller report (per deputy) ===")
        print("veh   | engage[s] | Hd_noninc | Hm_noninc | max|τ_appl| [N·m] | mean rel τ_err | Δv [m/s] |")
        for i,ctl in enumerate(self.ctrl_dep,1):
            t_eng = 10.0 + (i-1)*5.0 + 0.30
            Hd_frac = ctl.energy_noninc / max(1, ctl.energy_total)
            Hm_frac = ctl.me_noninc     / max(1, ctl.me_total)
            tau_appl = np.array(ctl.tau_appl_hist)
            tau_des  = np.array(ctl.tau_d_hist)
            if tau_des.size and tau_appl.size:
                num = la.norm(tau_des - tau_appl, axis=1)
                den = np.maximum(la.norm(tau_des, axis=1), 1e-12)
                rel_tau = np.mean(num/den)
            else:
                rel_tau = 0.0
            max_tau = float(np.max(la.norm(tau_appl, axis=1))) if tau_appl.size else 0.0
            print(f"dep{i} | {t_eng:9.2f} | {Hd_frac:9.3f} | {Hm_frac:9.3f} | {max_tau:15.4f} | {rel_tau:14.4f} | {ctl.dv_cum:7.4f} |")

        # Final LOS & SO(3) errors at handoff (t_dep+240 s)
        print("\nFinal LOS (desired vs achieved) at each deputy’s handoff (t_dep+240 s)")
        print("dep | desired_LOS_N (x y z)          | achieved_LOS_N (x y z)         |  LOS err [deg] | SO(3) err [deg]")
        for i,ctl in enumerate(self.ctrl_dep,1):
            t_target = self.t0_base + (i-1)*self.spacing + self.cfg.ALIGN_S
            tt = np.array(ctl.t_list); j = int(np.argmin(np.abs(tt - t_target)))
            C_BN = mrp_to_dcm(np.array(ctl.sigma_hist[j])); C_RN = self.C_RN_des[i-1]
            C_NB = C_BN.T; C_NR = C_RN.T
            los_des = unit(C_NR @ e_bore)      # desired LOS in N
            los_ach = unit(C_NB @ e_bore)      # achieved LOS in N
            ang = math.degrees(math.acos(np.clip(float(los_des @ los_ach), -1.0, 1.0)))
            so3 = math.degrees(math.acos(np.clip(0.5*(np.trace(C_BN @ C_RN.T)-1.0), -1.0, 1.0)))
            print(f"{i:3d} | [{los_des[0]: .4f} {los_des[1]: .4f} {los_des[2]: .4f}] |"
                  f" [{los_ach[0]: .4f} {los_ach[1]: .4f} {los_ach[2]: .4f}] |"
                  f" {ang:12.3f} | {so3:12.3f}")

    # ── Plots
    def _display_plots(self):
        for i,(dep,ctl) in enumerate(zip(self.deps,self.ctrl_dep),1):
            t = np.array(ctl.t_list)
            C_des = self.C_RN_des[i-1]
            sig_hist = np.array(ctl.sigma_hist)
            C_hist = np.array([mrp_to_dcm(s) for s in sig_hist])  # [T,3,3]

            # (A) R entries grid
            figA, axs = plt.subplots(3,3, figsize=(10,7), sharex=True)
            figA.suptitle(f"Deputy {i} — R entries (C_BN vs C_RN^des)")
            for r in range(3):
                for c in range(3):
                    axs[r,c].plot(t, C_hist[:,r,c], lw=1.1, label="R" if (r,c)==(0,0) else None)
                    axs[r,c].plot(t, np.full_like(t, C_des[r,c]), 'k--', lw=1.0, label="R_des" if (r,c)==(0,0) else None)
                    axs[r,c].set_ylim([-1.1,1.1]); axs[r,c].grid(True, alpha=0.3)
            axs[0,0].legend(fontsize=8); axs[2,1].set_xlabel("t [s]")

            # (B) momenta & torques
            figB, (axP, axU) = plt.subplots(2,1, figsize=(10,6), sharex=True)
            figB.suptitle(f"Deputy {i} — momenta and torques")
            J = dep.inertia_B; omg = np.array(ctl.omega_hist); p = omg @ J.T
            axP.plot(t,p[:,0],label="p_x"); axP.plot(t,p[:,1],label="p_y"); axP.plot(t,p[:,2],label="p_z")
            axP.set_ylabel("momenta [N·m·s]"); axP.legend(ncol=3,fontsize=8); axP.grid(True, alpha=0.3)
            u = np.array(ctl.tau_appl_hist) if len(ctl.tau_appl_hist) else np.zeros((t.size,3))
            axU.plot(t,u[:,0],label="u_x"); axU.plot(t,u[:,1],label="u_y"); axU.plot(t,u[:,2],label="u_z")
            axU.set_ylabel("torques [N·m]"); axU.set_xlabel("t [s]"); axU.legend(ncol=3,fontsize=8); axU.grid(True, alpha=0.3)

            # (C) r/v errors vs RHIG (if available)
            figC, (axR, axV) = plt.subplots(2,1, figsize=(10,6), sharex=True)
            figC.suptitle(f"Deputy {i} — ‖r − r_des‖, ‖v − v_des‖ (RHIG)")
            rN = np.array(ctl.rN_hist); vN = np.array(ctl.vN_hist)
            ref = self.rhig[i-1]
            if ref is not None and rN.size and vN.size:
                t_dep0 = self.t0_base + (i-1)*self.spacing
                t_abs = ref.t0 + np.clip(t - t_dep0, 0.0, self.cfg.ALIGN_S)
                rD = np.array([ref.r_at(tt) for tt in t_abs]) if ref.rD is not None else np.zeros_like(rN)
                vD = np.array([ref.v_at(tt) for tt in t_abs]) if ref.vD is not None else np.zeros_like(vN)
                axR.plot(t, la.norm(rN - rD, axis=1)); axR.set_ylabel("‖r − r_des‖ [m]")
                axV.plot(t, la.norm(vN - vD, axis=1)); axV.set_ylabel("‖v − v_des‖ [m/s]")
            else:
                axR.plot(t, la.norm(rN, axis=1)); axR.set_ylabel("‖r‖ [m] (proxy)")
                axV.plot(t, la.norm(vN, axis=1)); axV.set_ylabel("‖v‖ [m/s] (proxy)")
            axR.grid(True, alpha=0.3); axV.grid(True, alpha=0.3); axV.set_xlabel("t [s]")

        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7) Entry
# ─────────────────────────────────────────────────────────────────────────────
def main():
    fleet = Fleet()
    fleet.run()
    print("[DONE] Task 1 — pH(SE3) with LOS shaping. Start your MRP holding next.")

if __name__ == "__main__":
    main()
