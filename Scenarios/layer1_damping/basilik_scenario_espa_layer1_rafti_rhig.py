#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_scenario_espa_layer1_rafti_rhig.py

Purpose
-------
End-to-end Basilisk scenario that stress-tests the ARISE Layer-1 control
architecture during ESPA ring separations. A carrier spacecraft ejects a ring of
eight CubeSat deputies via RAFTI-style half-sine pushers, while a port–Hamiltonian
(SE(3)) damping layer, LOS shaping, and a passivity observer/controller (PO/PC)
keep the system energetically well behaved under actuation latency and thruster
limits.

High-level flow
---------------
1. Build a Basilisk fleet:
   - One carrier bus with MR-22N-class thrusters and two extForceTorque effectors
     (control + separation).
   - Eight deputies with MPS-120-class thrusters arranged on an ESPA-like ring,
     each with its own simpleNav and extForceTorque pairs.
2. Configure Layer-1 controllers:
   - Port–Hamiltonian damping in rotation and translation.
   - Small LOS shaping term that aligns each deputy to a rendezvous frame.
   - PO/PC term that clamps discrete-time energy injection and tracks a passivity
     tank state.
   - One-step latency predictor for both state and (optionally) LOS reference.
3. Ingest RHIG guidance NPZ files for each deputy:
   - Load `apophis_rendezvous_full_D*.npz` or `viz_inputs_D*.npz` from
     `APOPHIS_FLEET_DIR`.
   - Use `q_RN` when available; otherwise reconstruct RHIG attitude via a
     TRIAD solution on the boresight and velocity vectors.
   - Either lock to a final alignment attitude or follow a time-parameterized
     “segment” of the RHIG attitude history.
4. Drive RAFTI separations:
   - For each deputy, schedule a half-sine push with randomized misalignments
     in mount direction/position.
   - Compute and warn on equivalent mass, peak force, and commanded Δv
     against RAFTI limits.
   - Apply equal-and-opposite forces and torques to carrier and deputy via
     extForceTorque messages.
5. Run the simulation loop:
   - Manually advance simpleNav and dynamics SysModelTasks at fixed `dt`.
   - At each step, feed nav states and LOS references into the Layer-1
     controllers and update RAFTI forces.
   - Track Hamiltonian, mechanical energy, PO energy, thruster usage,
     Δv, and propellant estimates for the carrier and all deputies.

Outputs
-------
- Console summary of:
  * RAFTI schedule, peak forces, and impulses per deputy.
  * pH/BNLS controller gains, latency settings, and LOS shaping gain.
  * Energy monotonicity fractions (H_d, H_mech, PO_E) after engagement.
  * Cumulative control Δv and propellant mass for carrier and deputies.
- `layer1_report/` directory containing:
  * `layer1_summary.csv` with RAFTI and control metrics per vehicle.
  * PNG plots of energy non-increasing fractions, Δv budgets, RAFTI peak
    forces, fleet ω overlays, and detailed per-deputy histories and
    relative torque/force errors.

Environment
-----------
Requires a working Basilisk install and RHIG NPZ files. Set

    APOPHIS_FLEET_DIR

to the directory that contains the RHIG `.npz` files used for LOS shaping.
Run as a stand-alone script to generate Layer-1 ESPA separation reports.
"""


from __future__ import annotations
import os, math, dataclasses as dc
from pathlib import Path
from typing import List, Tuple, Optional
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

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")   # headless-safe
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ──────────────────────────────────────────────────────────────────────────
# RAFTI constants
# ──────────────────────────────────────────────────────────────────────────

class RAFTI:
    DOCK_XY_TOL_M    = 0.010
    DOCK_ANG_TOL_DEG = 5.0
    UTIL_XY_TOL_M    = 0.001
    UTIL_Z_ANG_DEG   = 2.0
    UTIL_PRELOAD_N   = 2000.0
    DOCK_VEL_MAX_MPS = 0.025
    EQMASS_MAX_KG    = 1000.0
    CONTACT_LIMIT_N  = 11600.0


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    x,y,z=v; return np.array([[0,-z,y],[z,0,-x],[-y,x,0]], float)

def normalize(v: np.ndarray, eps=1e-12) -> Tuple[np.ndarray,float]:
    n=la.norm(v); return ((v/n if n>=eps else np.zeros_like(v)), n)

def unit(v: np.ndarray) -> np.ndarray:
    n=la.norm(v); return (v/n if n>1e-12 else np.array([1.,0.,0.],float))

def mrp_to_dcm(sigma: np.ndarray) -> np.ndarray:
    s2=float(np.dot(sigma,sigma)); I3=np.eye(3)
    if s2<1e-16: return I3.copy()
    S=skew(sigma)
    return I3 + (8*S@S - 4*(1-s2)*S) / (1+s2)**2

def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    tr = np.trace(C)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        q0 = 0.25 * S
        q1 = (C[2,1] - C[1,2]) / S
        q2 = (C[0,2] - C[2,0]) / S
        q3 = (C[1,0] - C[0,1]) / S
    else:
        i = int(np.argmax([C[0,0], C[1,1], C[2,2]]))
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
    q = np.array([q0, q1, q2, q3])
    return q / la.norm(q)

def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    q0,qv=q[0],q[1:]; return -qv/(1-q0) if 1+q0<1e-12 else qv/(1+q0)

def dcm_to_mrp(C: np.ndarray) -> np.ndarray:
    return quat_to_mrp(dcm_to_quat(C))

def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    q0,q1,q2,q3 = q; qv = np.array([q1,q2,q3])
    return (q0*q0 - np.dot(qv,qv))*np.eye(3) \
         + 2*np.outer(qv,qv) \
         + 2*q0*np.array([[ 0, -q3,  q2],
                          [ q3,  0, -q1],
                          [-q2,  q1,  0]])

def triad_from_boresight_and_tangent(los_N: np.ndarray, vel_N: np.ndarray) -> np.ndarray:
    zB,_ = normalize(los_N); v_perp = vel_N - np.dot(vel_N, zB)*zB
    xB,nx= normalize(v_perp)
    if nx<1e-10:
        tmp = np.array([1,0,0]) if abs(zB[0])<.9 else np.array([0,1,0])
        xB = normalize(np.cross(tmp, zB))[0]
    yB = np.cross(zB,xB)
    xB = normalize(np.cross(yB,zB))[0]
    yB = normalize(np.cross(zB,xB))[0]
    return np.column_stack((xB,yB,zB))

def tilt_vec(v: np.ndarray, sigma_deg: float, rng: np.random.Generator) -> np.ndarray:
    v=unit(v)
    if sigma_deg<=0: return v
    a = rng.normal(scale=math.radians(sigma_deg), size=3)
    a -= np.dot(a,v)*v
    ang = la.norm(a)
    if ang<1e-10: return v
    k = a/ang; K=skew(k)
    R = np.eye(3) + math.sin(ang)*K + (1-math.cos(ang))*(K@K)
    return unit(R@v)

def meq(m1, m2):
    return (m1*m2)/(m1+m2)


# ──────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ──────────────────────────────────────────────────────────────────────────

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
    # pH parameters
    tau_omega_s: float
    tau_v_s:     float
    rw_tau_max:  float
    # BNLS
    w_force: float
    w_torque: float
    reg_lambda: float
    thrusters: List[Thruster]=dc.field(default_factory=list)

@dc.dataclass
class SimConfig:
    dt: float = 0.02
    T_end: float = 240.0
    latency_s: float = 0.050
    log_dir: Path = Path("layer1_logs")
    # RHIG NPZ intake
    rhig_dir: Path = Path(os.getenv("APOPHIS_FLEET_DIR", "."))
    rhig_glob_full: str = "apophis_rendezvous_full_D*.npz"
    rhig_glob_viz:  str = "viz_inputs_D*.npz"
    prefer_qrn: bool = True
    # reference policy
    ref_policy: str = "final_at_align"   # or "segment"
    align_s: float = 240.0
    extend_to_full_window: bool = True
    # shaping gain (small)
    K_sigma_align: float = 0.03
    # passivity & prediction
    enable_po_pc: bool = True
    use_latency_predict: bool = True
    predict_ref: bool = True


# ──────────────────────────────────────────────────────────────────────────
# RHIG NPZ wrapper
# ──────────────────────────────────────────────────────────────────────────

class RHIGBundle:
    def __init__(self, path: Path, prefer_qrn: bool = True):
        n = np.load(path, allow_pickle=True)
        self.path = path
        self.t    = n["t"].astype(float)
        self.rD   = n["rD_N"] if "rD_N" in n else None
        self.vD   = n["vD_N"] if "vD_N" in n else None
        self.rA   = n["rA_N"] if "rA_N" in n else None
        self.vA   = n["vA_N"] if "vA_N" in n else None
        self.bRef = n["bRef_N"] if "bRef_N" in n else None
        self.has_q= ("q_RN" in n) and prefer_qrn
        self.qRN  = n["q_RN"] if self.has_q else None
        self.t0   = float(n["t0"][0]) if "t0" in n else float(self.t[0])

    def _interp(self, arr: np.ndarray, t_abs: float) -> np.ndarray:
        t = self.t
        if t_abs <= t[0]:  return arr[0]
        if t_abs >= t[-1]: return arr[-1]
        j = np.searchsorted(t, t_abs) - 1
        f = (t_abs - t[j])/(t[j+1]-t[j] + 1e-15)
        return (1-f)*arr[j] + f*arr[j+1]

    def sigma_RN_at(self, t_abs: float) -> np.ndarray:
        if self.has_q:
            q = self._interp(self.qRN, t_abs)
            return quat_to_mrp(q)
        # TRIAD fallback
        if self.bRef is not None and self.vD is not None:
            bN = self._interp(self.bRef, t_abs)
            vN = self._interp(self.vD,   t_abs)
        else:
            if self.rA is None or self.rD is None or self.vD is None:
                return np.zeros(3)
            bN = unit(self._interp(self.rA, t_abs) - self._interp(self.rD, t_abs))
            vN = self._interp(self.vD, t_abs)
        C = triad_from_boresight_and_tangent(bN, vN)
        return quat_to_mrp(dcm_to_quat(C.T))


# ──────────────────────────────────────────────────────────────────────────
# Port–Hamiltonian Layer-1 with LOS shaping + PO/PC + latency predict
# ──────────────────────────────────────────────────────────────────────────

class PHLayer1:
    def __init__(self, veh: VehicleConfig, dt: float, engage_time: float, latency_s: float,
                 K_sigma: float|np.ndarray = 0.0, enable_po_pc=True, use_latency_predict=True):
        self.veh=veh; self.dt=dt; self.engage_time=engage_time
        J=veh.inertia_B; m=veh.mass_kg
        self.K_omega=(1.0/max(veh.tau_omega_s,1e-6))*J
        self.K_v    =(m   /max(veh.tau_v_s,1e-6))*np.eye(3)
        self.K_eta  =1e-2*np.eye(6)
        self.K_sigma = (np.eye(3)*float(K_sigma)) if np.ndim(K_sigma)==0 else np.array(K_sigma,float)

        self.ctrlTorqueMsg = messaging.CmdTorqueBodyMsg()
        self.ctrlForceMsg  = messaging.CmdForceBodyMsg()
        self.L=int(round(max(0.0,latency_s)/dt))
        self.use_latency_predict=use_latency_predict
        self.enable_po_pc=enable_po_pc

        self.buf_tau: List[np.ndarray]=[]; self.buf_for: List[np.ndarray]=[]
        self.B, self.Ts, self.Isps = self._build_wrench_map(veh.thrusters)
        self.delta_prev: Optional[np.ndarray]=None
        self.omega_prev=np.zeros(3); self.v_prev=np.zeros(3)
        self._Hd_prev=None; self.energy_total=0; self.energy_noninc=0
        self.dv_cum=0.0; self.mp_cum=0.0
        self.t_list=[]; self.Hd_list=[]
        self.tau_d_hist=[]; self.f_d_hist=[]
        self.tau_rw_hist=[]; self.tau_thr_hist=[]
        self.torque_cmd_hist=[]; self.force_cmd_hist=[]
        self.torque_apply_hist=[]; self.force_apply_hist=[]
        self.delta_hist=[]; self.thrust_total_hist=[]
        self.omega_hist=[]; self.vB_hist=[]; self.sigma_hist=[]
        self.sigma_ref_hist=[]; self.sigma_BR_hist=[]
        self.E_po=0.0; self.po_scale_hist=[]; self.E_po_hist=[]

        # near-1 fractions counters
        self.me_prev = None
        self.me_total = 0
        self.me_noninc = 0
        self.po_prev = None
        self.po_total = 0
        self.po_noninc = 0

    def _build_wrench_map(self, thr: List[Thruster]):
        if not thr: return np.zeros((6,0)), np.zeros(0), np.zeros(0)
        cols=[]; Ts=[]; Isps=[]
        for th in thr:
            r=th.r_B.reshape(3); a=unit(th.a_B.reshape(3))
            cols.append(np.hstack((np.cross(r, th.T*a), th.T*a)))
            Ts.append(th.T); Isps.append(th.Isp)
        return np.column_stack(cols), np.array(Ts), np.array(Isps)

    def _bnls(self,B,u,delta_prev,w_tau,w_for,lam,lb,ub):
        n=B.shape[1]
        if delta_prev is None or delta_prev.shape[0]!=n: delta_prev=np.zeros(n)
        W=np.diag([w_tau,w_tau,w_tau,w_for,w_for,w_for])
        BtWB=B.T@W@B + lam*np.eye(n); BtWu=B.T@W@u + lam*delta_prev
        delta=np.clip(la.solve(BtWB,BtWu),lb,ub)
        for _ in range(8):
            low=(delta<=lb+1e-9); high=(delta>=ub-1e-9); free=~(low|high)
            if not np.any(free): break
            Bf=B[:,free]
            K=la.pinv(Bf.T@W@Bf + lam*np.eye(np.count_nonzero(free)))
            rhs=Bf.T@W@(u - B[:,low]@np.full(np.count_nonzero(low), lb)
                          - B[:,high]@np.full(np.count_nonzero(high), ub)) + lam*delta_prev[free]
            delta_f=K@rhs; delta=delta.copy(); delta[free]=delta_f; delta=np.clip(delta,lb,ub)
        return delta

    def step(self, t: float, attNav: messaging.NavAttMsgPayload, transNav: messaging.NavTransMsgPayload,
             sigma_RN_ref: Optional[np.ndarray] = None):
        dt=self.dt; J=self.veh.inertia_B; m=self.veh.mass_kg
        sigma=np.array(attNav.sigma_BN); C_BN=mrp_to_dcm(sigma)
        omega=np.array(attNav.omega_BN_B)
        v_N=np.array(transNav.v_BN_N); v_B=C_BN@v_N

        # latency prediction (1 step)
        if self.use_latency_predict:
            omega_p = omega + (omega - self.omega_prev)
            vB_p    = v_B  + (v_B  - self.v_prev)
        else:
            omega_p = omega.copy(); vB_p=v_B.copy()

        # shaping toward LOS
        tau_shape=np.zeros(3); V_shape=0.0
        if sigma_RN_ref is not None:
            C_RN=mrp_to_dcm(sigma_RN_ref)
            C_BR=C_BN @ C_RN.T
            sigma_BR=dcm_to_mrp(C_BR)
            tau_shape = -(self.K_sigma @ sigma_BR)
            V_shape   = 0.5 * float(sigma_BR.T @ (self.K_sigma @ sigma_BR))
            self.sigma_ref_hist.append(sigma_RN_ref.copy()); self.sigma_BR_hist.append(sigma_BR.copy())

        # desired wrench (predicted states)
        tau_des = - self.K_omega @ omega_p + tau_shape
        f_des   = - self.K_v     @ vB_p

        # PO/PC scaling
        s=1.0
        if self.enable_po_pc:
            P = float(tau_des @ omega + f_des @ v_B)
            dE = P * dt
            if dE>0.0 and self.E_po + dE > 0.0:
                s = max(0.0, min(1.0, -self.E_po/(dE+1e-12)))
            tau_des *= s; f_des *= s
            P_eff = float(tau_des @ omega + f_des @ v_B)
            self.E_po += P_eff * dt
        self.po_scale_hist.append(s); self.E_po_hist.append(self.E_po)

        # RW split + BNLS allocation
        tau_rw=tau_des.copy(); tn=la.norm(tau_rw)
        if tn>self.veh.rw_tau_max>0: tau_rw *= (self.veh.rw_tau_max/tn)
        tau_res=tau_des - tau_rw

        tau_thr=np.zeros(3); f_thr=np.zeros(3); delta=np.zeros(0)
        if self.B.size>0:
            u=np.hstack((tau_res,f_des))
            delta=self._bnls(self.B,u,self.delta_prev,self.veh.w_torque,self.veh.w_force,self.veh.reg_lambda,0.0,1.0)
            self.delta_prev=delta.copy()
            uf=self.B@delta; tau_thr=uf[:3]; f_thr=uf[3:]

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

        # shaped & mechanical energies
        Hd = 0.5*float(omega.T@(J@omega)) + 0.5*float(m*(v_B@v_B)) + V_shape
        H_mech = 0.5*float(omega.T@(J@omega)) + 0.5*float(m*(v_B@v_B))

        if t>=self.engage_time:
            self.energy_total+=1
            if self._Hd_prev is not None and Hd<=self._Hd_prev+1e-12:
                self.energy_noninc+=1
            self._Hd_prev = Hd

            self.me_total += 1
            if self.me_prev is not None and H_mech <= self.me_prev + 1e-12:
                self.me_noninc += 1
            self.me_prev = H_mech

            self.po_total += 1
            if self.po_prev is not None and self.E_po <= self.po_prev + 1e-12:
                self.po_noninc += 1
            self.po_prev = self.E_po

        # logs
        self.t_list.append(t); self.Hd_list.append(Hd)
        self.tau_d_hist.append(tau_des.copy()); self.f_d_hist.append(f_des.copy())
        self.tau_rw_hist.append(tau_rw.copy());  self.tau_thr_hist.append(tau_thr.copy())
        self.torque_cmd_hist.append(tau_real.copy()); self.force_cmd_hist.append(f_real.copy())
        self.torque_apply_hist.append(tau_apply.copy()); self.force_apply_hist.append(f_apply.copy())
        self.delta_hist.append(delta.copy() if delta.size else np.zeros(0))
        self.omega_hist.append(omega.copy()); self.vB_hist.append(v_B.copy()); self.sigma_hist.append(sigma.copy())
        self.thrust_total_hist.append(float(np.sum(self.Ts*delta)) if self.B.size else 0.0)

        # Δv/prop
        if delta.size>0:
            impulse=float(np.sum(self.Ts*delta)*dt)
            self.dv_cum += impulse/m
            if impulse>0:
                self.mp_cum += impulse / (np.average(self.Isps, weights=self.Ts*delta)*9.80665)

        self.omega_prev=omega.copy(); self.v_prev=v_B.copy()


# ──────────────────────────────────────────────────────────────────────────
# RAFTI separation driver
# ──────────────────────────────────────────────────────────────────────────

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
            if abs(t - sp.t_start) < 0.5*self.dt:
                m_eq = meq(self.m_car, self.m_dep)
                if m_eq > RAFTI.EQMASS_MAX_KG:
                    print(f"[RAFTI WARN dep{i+1}] m_eq={m_eq:.1f} kg exceeds {RAFTI.EQMASS_MAX_KG:.0f} kg rating")
                if sp.dv_cmd > RAFTI.DOCK_VEL_MAX_MPS:
                    print(f"[RAFTI WARN dep{i+1}] Δv_cmd={sp.dv_cmd:.3f} m/s exceeds RAFTI 25 mm/s rating")
                if sp.F0 > RAFTI.CONTACT_LIMIT_N:
                    print(f"[RAFTI WARN dep{i+1}] peak push {sp.F0:.0f} N > contact load {RAFTI.CONTACT_LIMIT_N:.0f} N")
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


# ──────────────────────────────────────────────────────────────────────────
# Fleet assembly & run + report & plots
# ──────────────────────────────────────────────────────────────────────────

class Fleet:
    def __init__(self):
        self.dt=0.02
        self.sim = SimConfig(dt=self.dt, T_end=240.0, latency_s=0.05, log_dir=Path("layer1_logs"))

        rng=np.random.default_rng(42)

        # Carrier config (MR-22N-like)
        mr22_T, mr22_Isp = 22.0, 235.0
        thr_c = [
            Thruster(np.array([ 0.8, 0.0,  0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp, "MR22_1"),
            Thruster(np.array([-0.8, 0.0,  0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp, "MR22_2"),
            Thruster(np.array([ 0.8, 0.0, -0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp, "MR22_3"),
            Thruster(np.array([-0.8, 0.0, -0.5]), np.array([0.0,-1.0,0.0]), mr22_T, mr22_Isp, "MR22_4"),
        ]
        I_car=np.diag([1200.,1000.,800.]); m_car=650.0
        self.car = VehicleConfig(name="carrier", mass_kg=m_car, inertia_B=I_car,
                                 sigma0_BN=np.zeros(3), omega0_B=np.zeros(3),
                                 r0_N=np.zeros(3), v0_N=np.zeros(3),
                                 tau_omega_s=60.0, tau_v_s=40.0, rw_tau_max=0.020,
                                 w_force=100.0, w_torque=1.0, reg_lambda=1e-4,
                                 thrusters=thr_c)

        # Deputies (MPS-120-like)
        mps_T, mps_Isp = 0.27, 217.0
        thr_d = [
            Thruster(np.array([ 0.05, 0.0,  0.05]), np.array([ 0.05,-0.998,  0.05]), mps_T, mps_Isp, "MPS_1"),
            Thruster(np.array([-0.05, 0.0,  0.05]), np.array([-0.05,-0.998,  0.05]), mps_T, mps_Isp, "MPS_2"),
            Thruster(np.array([ 0.05, 0.0, -0.05]), np.array([ 0.05,-0.998, -0.05]), mps_T, mps_Isp, "MPS_3"),
            Thruster(np.array([-0.05, 0.0, -0.05]), np.array([-0.05,-0.998, -0.05]), mps_T, mps_Isp, "MPS_4"),
        ]
        I_dep=np.diag([0.025,0.020,0.018]); m_dep=12.0

        self.deps: List[VehicleConfig]=[]
        self.mount_dirs_car_B=[]; self.mount_pos_car_B=[]; self.mount_R_BD_BC=[]

        ring_R = 1.00; z_mount = 0.0
        for i in range(8):
            ang= i*(2*math.pi/8.0)
            s_car_B = np.array([math.cos(ang), math.sin(ang), 0.0], float)
            r_car_B = np.array([ ring_R*math.cos(ang), ring_R*math.sin(ang), z_mount ], float)
            bY = s_car_B; bZ=np.array([0,0,1.0])
            if abs(np.dot(unit(bY),bZ))>0.99: bZ=np.array([1,0,0])
            bX = unit(np.cross(bY,bZ)); bZ = unit(np.cross(bX,bY))
            R_BD_BC = np.column_stack((bX,bY,bZ))
            sigma0 = dcm_to_mrp(R_BD_BC)
            r0     = r_car_B.copy(); v0=np.zeros(3)
            dep=VehicleConfig(name=f"dep{i+1}", mass_kg=m_dep, inertia_B=I_dep,
                              sigma0_BN=sigma0, omega0_B=np.zeros(3), r0_N=r0, v0_N=v0,
                              tau_omega_s=25.0, tau_v_s=30.0, rw_tau_max=0.020,
                              w_force=200.0, w_torque=0.5, reg_lambda=2e-4,
                              thrusters=thr_d)
            self.deps.append(dep); self.mount_dirs_car_B.append(s_car_B)
            self.mount_pos_car_B.append(r_car_B); self.mount_R_BD_BC.append(R_BD_BC)

        # Tasks
        dt=self.dt
        self.dyn = sys_model_task.SysModelTask(int(macros.sec2nano(dt)))
        self.sns = sys_model_task.SysModelTask(int(macros.sec2nano(dt)))

        # Carrier spacecraft + nav + two effectors
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

        # Deputies spacecraft
        self.sc_dep=[]; self.nav_dep=[]; self.eft_dep_ctrl=[]; self.eft_dep_sep=[]
        for dep in self.deps:
            sc=spacecraft.Spacecraft(); sc.ModelTag=f"SC_{dep.name}"
            sc.hub.mHub=dep.mass_kg; sc.hub.IHubPntBc_B=dep.inertia_B.tolist()
            sc.hub.sigma_BNInit=list(dep.sigma0_BN); sc.hub.omega_BN_BInit=list(dep.omega0_B)
            sc.hub.r_CN_NInit=list(dep.r0_N);       sc.hub.v_CN_NInit=list(dep.v0_N)
            nav=simpleNav.SimpleNav(); nav.ModelTag=f"NAV_{dep.name}"
            nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
            ec=extForceTorque.ExtForceTorque(); ec.ModelTag=f"EFT_{dep.name}_ctrl"
            es=extForceTorque.ExtForceTorque(); es.ModelTag=f"EFT_{dep.name}_sep"
            sc.addDynamicEffector(ec); sc.addDynamicEffector(es)
            self.sc_dep.append(sc); self.nav_dep.append(nav)
            self.eft_dep_ctrl.append(ec); self.eft_dep_sep.append(es)

        # Add to tasks
        p=0
        self.sns.AddNewObject(self.nav_car,10+p); self.dyn.AddNewObject(self.sc_car,5+p)
        self.dyn.AddNewObject(self.eft_car_ctrl,10+p); self.dyn.AddNewObject(self.eft_car_sep,11+p)
        for i,(sc,nav,ec,es) in enumerate(zip(self.sc_dep,self.nav_dep,self.eft_dep_ctrl,self.eft_dep_sep),1):
            self.sns.AddNewObject(nav,  10+p+10*i)
            self.dyn.AddNewObject(sc,    5+p+10*i)
            self.dyn.AddNewObject(ec,   10+p+10*i)
            self.dyn.AddNewObject(es,   11+p+10*i)

        # Controllers (PO/PC + predictor)
        last_sep_end = 10.0 + (len(self.deps)-1)*5.0 + 0.30
        self.ctrl_car = PHLayer1(self.car, dt, engage_time=last_sep_end, latency_s=self.sim.latency_s,
                                 K_sigma=0.0, enable_po_pc=self.sim.enable_po_pc,
                                 use_latency_predict=self.sim.use_latency_predict)
        self.ctrl_dep=[]
        for k,dep in enumerate(self.deps):
            t_end = 10.0 + k*5.0 + 0.30
            self.ctrl_dep.append(PHLayer1(dep, dt, engage_time=t_end, latency_s=self.sim.latency_s,
                                          K_sigma=self.sim.K_sigma_align,
                                          enable_po_pc=self.sim.enable_po_pc,
                                          use_latency_predict=self.sim.use_latency_predict))

        # Wire control ports
        self.eft_car_ctrl.cmdTorqueInMsg   .subscribeTo(self.ctrl_car.ctrlTorqueMsg)
        self.eft_car_ctrl.cmdForceBodyInMsg.subscribeTo(self.ctrl_car.ctrlForceMsg)
        for ctl,ec in zip(self.ctrl_dep, self.eft_dep_ctrl):
            ec.cmdTorqueInMsg   .subscribeTo(ctl.ctrlTorqueMsg)
            ec.cmdForceBodyInMsg.subscribeTo(ctl.ctrlForceMsg)

        # RAFTI schedule
        self.sched=[]
        self.t0_base=10.0; self.spacing=5.0; Tpush=0.30
        for k in range(len(self.deps)):
            t0 = self.t0_base + k*self.spacing
            dv = float(np.clip(np.random.default_rng(42+k).normal(0.018, 0.005), 0.004, RAFTI.DOCK_VEL_MAX_MPS))
            F0 = self.deps[k].mass_kg * dv * math.pi / (2.0*Tpush)
            s_car_B = tilt_vec(self.mount_dirs_car_B[k], 2.0, rng)
            r_car_B = self.mount_pos_car_B[k] + rng.normal(scale=0.02, size=3)
            r_dep_B = rng.normal(scale=0.02, size=3)
            self.sched.append(SepParams(t_start=t0, t_end=t0+Tpush, F0=F0,
                                        s_car_B=s_car_B, r_car_B=r_car_B, r_dep_B=r_dep_B,
                                        dv_cmd=dv))
        self.sep = RAFTISeparation(self.dt, self.sc_car, self.eft_car_sep,
                                   self.sc_dep, self.eft_dep_sep, self.sched,
                                   m_car=self.car.mass_kg, m_dep=self.deps[0].mass_kg)

        # Finalize tasks
        for tsk in (self.sns,self.dyn): tsk.SelfInitTaskList(); tsk.ResetTaskList(0)

        # RHIG references
        self.rhig_refs: List[Optional[RHIGBundle]] = self._load_rhig_refs()
        self.sigma_ref_const: List[Optional[np.ndarray]] = [None]*len(self.deps)
        if self.sim.ref_policy == "final_at_align":
            for k,ref in enumerate(self.rhig_refs):
                if ref is None: self.sigma_ref_const[k]=np.zeros(3); continue
                self.sigma_ref_const[k] = ref.sigma_RN_at(ref.t0 + self.sim.align_s)

        # Extend horizon so last deputy gets align window
        if self.sim.extend_to_full_window:
            last_sep_start = self.t0_base + (len(self.deps)-1)*self.spacing
            need = last_sep_start + self.sim.align_s
            if need > self.sim.T_end: self.sim.T_end = need

    def _load_rhig_refs(self) -> List[Optional[RHIGBundle]]:
        dirp = self.sim.rhig_dir
        full = sorted(dirp.glob(self.sim.rhig_glob_full))
        if not full:
            full = sorted(dirp.glob(self.sim.rhig_glob_viz))
            if not full:
                print("[RHIG] No NPZ found; proceeding without external references.")
                return [None]*len(self.deps)
        bundles=[]
        for k in range(len(self.deps)):
            cand = [p for p in full if f"D{k}" in p.stem]
            p = cand[0] if cand else (full[k] if k < len(full) else full[-1])
            try:
                bundles.append(RHIGBundle(p, prefer_qrn=self.sim.prefer_qrn))
                print(f"[RHIG] dep{k+1} ↔ {p.name}")
            except Exception as e:
                print(f"[RHIG] Failed to load {p}: {e}"); bundles.append(None)
        return bundles

    def _engage_time_of_dep(self, i:int)->float:  # end of push
        return 10.0 + i*5.0 + 0.30
    def _last_engage(self)->float:
        return self._engage_time_of_dep(len(self.deps)-1)

    def run(self):
        now_ns=0; N=int(self.sim.T_end/self.dt)
        for k in tqdm(range(N), desc="Layer-1 ESPA+RAFTI+PH+PO", ncols=80):
            t=k*self.dt
            self.sns.ExecuteTaskList(now_ns)
            self.sep.step(t)

            # Carrier (no LOS shaping)
            self.ctrl_car.step(t, self.nav_car.attOutMsg.read(), self.nav_car.transOutMsg.read(), None)

            # Deputies (LOS from RHIG)
            for i,(ctl,nav,ref) in enumerate(zip(self.ctrl_dep, self.nav_dep, self.rhig_refs)):
                sigma_ref=None
                if ref is not None:
                    if self.sim.ref_policy=="final_at_align":
                        sigma_ref = self.sigma_ref_const[i]
                    else: # "segment"
                        t_dep0 = self.t0_base + i*self.spacing
                        tau = max(0.0, t - t_dep0)
                        t_abs = ref.t0 + min(self.sim.align_s, tau)
                        if self.sim.predict_ref: t_abs += self.sim.latency_s
                        sigma_ref = ref.sigma_RN_at(t_abs)
                ctl.step(t, nav.attOutMsg.read(), nav.transOutMsg.read(), sigma_RN_ref=sigma_ref)

            self.dyn.ExecuteTaskList(now_ns)
            now_ns+=int(macros.sec2nano(self.dt))

        # summary + full report + artifacts
        self._print_run_summary()
        self._print_full_report()
        self._save_plots_and_csvs()

    def _print_run_summary(self):
        frac = self.ctrl_car.energy_noninc/max(1,self.ctrl_car.energy_total)
        print(f"[carrier] H_d monotonic after t≥{self._last_engage():.1f}s: {frac:.3f}, "
              f"Δv≃{self.ctrl_car.dv_cum:.4f} m/s, m_p={self.ctrl_car.mp_cum*1e3:.3f} g")
        for i,ctl in enumerate(self.ctrl_dep,1):
            frac = ctl.energy_noninc/max(1,ctl.energy_total)
            print(f"[dep{i}]    H_d monotonic after t≥{self._engage_time_of_dep(i-1):.1f}s: {frac:.3f}, "
                  f"Δv≃{ctl.dv_cum:.4f} m/s, m_p={ctl.mp_cum*1e3:.3f} g")

    def _print_full_report(self):
        print("\n=== RAFTI Separation Schedule & Loads ===")
        print("dep | t_start–t_end [s] | dv_cmd [m/s] | F0_peak [N] | Impulse [N·s]")
        for k,sp in enumerate(self.sched,1):
            J = 2.0*sp.F0*(sp.t_end-sp.t_start)/math.pi
            print(f"{k:3d} | {sp.t_start:5.2f}–{sp.t_end:5.2f}    | {sp.dv_cmd:8.5f}    | {sp.F0:9.3f}   | {J:8.5f}")

        print("\n=== Controller & pH Parameters ===")
        def diag(M): M=np.asarray(M); return np.diag(M) if M.ndim==2 else M
        I_car=self.car.inertia_B; m_car=self.car.mass_kg
        Kω_c=diag(I_car)/self.car.tau_omega_s; Kv_c=np.full(3,m_car/self.car.tau_v_s)
        print(f"Carrier: Kω={Kω_c}, Kv={Kv_c}, RW_max={self.car.rw_tau_max:.3f} N·m, "
              f"BNLS (wτ,wf,λ)=(1,100,1e-4), latency steps={self.ctrl_car.L}")
        I_d=self.deps[0].inertia_B; m_d=self.deps[0].mass_kg
        Kω_d=diag(I_d)/self.deps[0].tau_omega_s; Kv_d=np.full(3,m_d/self.deps[0].tau_v_s)
        print(f"Deputies: Kω={Kω_d}, Kv={Kv_d}, RW_max={self.deps[0].rw_tau_max:.3f} N·m, "
              f"BNLS (wτ,wf,λ)=(0.5,200,2e-4), latency steps={self.ctrl_dep[0].L}")
        print(f"LOS shaping gain K_sigma_align={self.sim.K_sigma_align:.3f} N·m/rad; "
              f"PO/PC={'ON' if self.sim.enable_po_pc else 'OFF'}; predictor={'ON' if self.sim.use_latency_predict else 'OFF'}; ref_policy={self.sim.ref_policy}")

        print("\n=== Energy Monotonicity & Control Usage ===")
        print("veh   | engage[s] | H_d frac  | H_mech frac | PO_E frac | steps_noninc/total (H_d) | Δv_ctrl [m/s] | m_p [g]")
        c=self.ctrl_car; totHd=max(1,c.energy_total); totMe=max(1,c.me_total); totPo=max(1,c.po_total)
        print(f"carrier| {self._last_engage():9.1f} | {c.energy_noninc/totHd:8.3f} | {c.me_noninc/totMe:10.3f} | {c.po_noninc/totPo:8.3f} | "
              f"{c.energy_noninc:5d}/{totHd:5d}           | {c.dv_cum:10.4f} | {c.mp_cum*1e3:7.3f}")
        for i,ctl in enumerate(self.ctrl_dep,1):
            totHd=max(1,ctl.energy_total); totMe=max(1,ctl.me_total); totPo=max(1,ctl.po_total)
            print(f"dep{i:1d}  | {self._engage_time_of_dep(i-1):9.1f} | {ctl.energy_noninc/totHd:8.3f} | "
                  f"{ctl.me_noninc/totMe:10.3f} | {ctl.po_noninc/totPo:8.3f} | {ctl.energy_noninc:5d}/{totHd:5d}           | "
                  f"{ctl.dv_cum:10.4f} | {ctl.mp_cum*1e3:7.3f}")

    def _save_plots_and_csvs(self):
        outdir = Path("layer1_report"); outdir.mkdir(parents=True, exist_ok=True)
        import csv
        rows=[]
        rows.append(["carrier",np.nan,np.nan,self._last_engage(),
                    np.nan,np.nan,np.nan,
                    self.ctrl_car.energy_noninc/max(1,self.ctrl_car.energy_total),
                    self.ctrl_car.me_noninc/max(1,self.ctrl_car.me_total),
                    self.ctrl_car.po_noninc/max(1,self.ctrl_car.po_total),
                    self.ctrl_car.dv_cum,self.ctrl_car.mp_cum*1e3])
        for i,(ctl,sp) in enumerate(zip(self.ctrl_dep,self.sched),1):
            rows.append([f"dep{i}",sp.t_start,sp.t_end,self._engage_time_of_dep(i-1),
                         sp.dv_cmd,sp.F0, 2.0*sp.F0*(sp.t_end-sp.t_start)/math.pi,
                         ctl.energy_noninc/max(1,ctl.energy_total),
                         ctl.me_noninc/max(1,ctl.me_total),
                         ctl.po_noninc/max(1,ctl.po_total),
                         ctl.dv_cum,ctl.mp_cum*1e3])
        with open(outdir/"layer1_summary.csv","w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["entity","t_push_start","t_push_end","engage",
                        "rafti_dv_cmd","rafti_F0","rafti_impulse",
                        "Hd_noninc_frac","Hmech_noninc_frac","POE_noninc_frac",
                        "dv_ctrl","prop_g"])
            w.writerows(rows)

        if not HAS_MPL:
            print("[plots] matplotlib not available; skipping figures.")
            return

        # Energy non-increasing fractions per deputy
        deps=list(range(1,len(self.ctrl_dep)+1))
        y_hd=[self.ctrl_dep[i-1].energy_noninc/max(1,self.ctrl_dep[i-1].energy_total) for i in deps]
        y_me=[self.ctrl_dep[i-1].me_noninc/max(1,self.ctrl_dep[i-1].me_total) for i in deps]
        y_po=[self.ctrl_dep[i-1].po_noninc/max(1,self.ctrl_dep[i-1].po_total) for i in deps]
        x=np.arange(1,len(deps)+1)

        plt.figure(figsize=(10,4))
        plt.bar(x-0.25,y_me,width=0.25,label="H_mech")
        plt.bar(x,      y_po,width=0.25,label="PO_E")
        plt.bar(x+0.25,y_hd,width=0.25,label="H_d")
        plt.ylim(0,1.05)
        plt.title("Energy non-increasing fraction per deputy")
        plt.xlabel("Deputy #"); plt.ylabel("fraction (0–1)")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir/"energy_fractions_per_deputy.png",dpi=150); plt.close()

        # Δv comparison: RAFTI vs control
        dv_cmd=[sp.dv_cmd for sp in self.sched]
        dv_thr=[self.ctrl_dep[i-1].dv_cum for i in deps]
        plt.figure(figsize=(10,4))
        plt.bar(x-0.2,dv_cmd,width=0.4,label="RAFTI Δv_cmd")
        plt.bar(x+0.2,dv_thr,width=0.4,label="Control Δv")
        plt.title("Separation Δv budget vs control usage")
        plt.xlabel("Deputy #"); plt.ylabel("Δv [m/s]"); plt.legend()
        plt.tight_layout(); plt.savefig(outdir/"dv_cmd_vs_control_per_deputy.png",dpi=150); plt.close()

        # Peak RAFTI force
        F0s=[sp.F0 for sp in self.sched]
        plt.figure(figsize=(10,4))
        plt.bar(x,F0s)
        plt.title("Peak RAFTI half-sine force per deputy")
        plt.xlabel("Deputy #"); plt.ylabel("F0 [N]")
        plt.tight_layout(); plt.savefig(outdir/"peak_rafti_force_per_deputy.png",dpi=150); plt.close()

        # ================================
        # EXTRA PLOTS (like the screenshots)
        # ================================
        def _np(a):
            return np.array(a) if not isinstance(a, np.ndarray) else a
        eps = 1e-12

        # (A) Rel. error plots per deputy
        for i, ctl in enumerate(self.ctrl_dep, 1):
            t = _np(ctl.t_list)
            tau_d = _np(ctl.tau_d_hist); tau_c = _np(ctl.torque_cmd_hist); tau_a = _np(ctl.torque_apply_hist)
            f_d   = _np(ctl.f_d_hist);   f_c   = _np(ctl.force_cmd_hist);   f_a   = _np(ctl.force_apply_hist)
            if t.size == 0: continue
            def rel_err(a,b):
                if a.size == 0 or b.size == 0: return np.zeros_like(t)
                num = la.norm(a-b,axis=1); den = np.maximum(la.norm(a,axis=1), eps)
                return num/den
            e_tau_cmd = rel_err(tau_d, tau_c); e_tau_app = rel_err(tau_d, tau_a)
            e_f_cmd   = rel_err(f_d,   f_c);   e_f_app   = rel_err(f_d,   f_a)
            fig, ax = plt.subplots(2,1,figsize=(10,6),sharex=True)
            ax[0].plot(t, e_tau_cmd, label="τ: desired→cmd")
            ax[0].plot(t, e_tau_app, label="τ: desired→appl")
            ax[0].set_ylabel("Rel. torque error"); ax[0].legend(loc="upper right", fontsize=9)
            ax[1].plot(t, e_f_cmd, label="f: desired→cmd")
            ax[1].plot(t, e_f_app, label="f: desired→appl")
            ax[1].set_ylabel("Rel. force error"); ax[1].set_xlabel("time [s]")
            ax[1].legend(loc="upper right", fontsize=9)
            fig.tight_layout(); fig.savefig(outdir/f"rel_errors_dep{i:02d}.png",dpi=150); plt.close(fig)

        # (B) Multi-panel energy & histories per deputy
        for i, ctl in enumerate(self.ctrl_dep, 1):
            t = _np(ctl.t_list); Hd=_np(ctl.Hd_list)
            tau_d=_np(ctl.tau_d_hist); tau_c=_np(ctl.torque_cmd_hist); tau_a=_np(ctl.torque_apply_hist)
            f_d=_np(ctl.f_d_hist); f_c=_np(ctl.force_cmd_hist); f_a=_np(ctl.force_apply_hist)
            omg=_np(ctl.omega_hist); sig=_np(ctl.sigma_hist); thrust=_np(ctl.thrust_total_hist)
            if t.size == 0: continue
            def nrm(X):
                return la.norm(X,axis=1) if (isinstance(X,np.ndarray) and X.ndim==2 and X.shape[0]==t.size) else np.zeros_like(t)
            fig,axs=plt.subplots(6,1,figsize=(10,14),sharex=True)
            axs[0].plot(t,Hd); axs[0].axvline(self._engage_time_of_dep(i-1),ls="--",alpha=0.6)
            axs[0].set_ylabel("H_d"); axs[0].set_title(f"Deputy {i} — pH energy & histories")
            axs[1].plot(t,nrm(tau_d),label="|τ_d|"); axs[1].plot(t,nrm(tau_c),label="|τ_cmd|"); axs[1].plot(t,nrm(tau_a),label="|τ_appl|")
            axs[1].set_ylabel("Torque norm"); axs[1].legend(loc="upper right",fontsize=9)
            axs[2].plot(t,nrm(f_d),label="|f_d|"); axs[2].plot(t,nrm(f_c),label="|f_cmd|"); axs[2].plot(t,nrm(f_a),label="|f_appl|")
            axs[2].set_ylabel("Force norm"); axs[2].legend(loc="upper right",fontsize=9)
            axs[3].plot(t, thrust if thrust.size else np.zeros_like(t)); axs[3].set_ylabel("∑T·δ [N]")
            if isinstance(omg,np.ndarray) and omg.ndim==2 and omg.shape[1]==3:
                axs[4].plot(t,omg[:,0],label="ω_x"); axs[4].plot(t,omg[:,1],label="ω_y"); axs[4].plot(t,omg[:,2],label="ω_z")
                axs[4].set_ylabel("ω [rad/s]"); axs[4].legend(loc="upper right",fontsize=9)
            else: axs[4].set_ylabel("ω [rad/s]")
            if isinstance(sig,np.ndarray) and sig.ndim==2 and sig.shape[1]==3:
                axs[5].plot(t,sig[:,0],label="σ_1"); axs[5].plot(t,sig[:,1],label="σ_2"); axs[5].plot(t,sig[:,2],label="σ_3")
                axs[5].set_ylabel("MRP σ"); axs[5].legend(loc="upper right",fontsize=9)
            else: axs[5].set_ylabel("MRP σ")
            axs[-1].set_xlabel("time [s]"); fig.tight_layout()
            fig.savefig(outdir/f"energy_histories_dep{i:02d}.png",dpi=150); plt.close(fig)

        # (C) Fleet overlay of |ω| norms
        fig,ax=plt.subplots(1,1,figsize=(10,5))
        for i,ctl in enumerate(self.ctrl_dep,1):
            t=_np(ctl.t_list); omg=_np(ctl.omega_hist)
            if isinstance(omg,np.ndarray) and omg.ndim==2 and omg.shape[0]==t.size:
                ax.plot(t, la.norm(omg,axis=1), label=f"dep{i}")
        ax.set_title("Deputy ω norms"); ax.set_xlabel("time [s]"); ax.set_ylabel("|ω| [rad/s]")
        ax.legend(ncol=4,fontsize=9); fig.tight_layout()
        fig.savefig(outdir/"fleet_omega_overlay.png",dpi=150); plt.close(fig)

        print("[plots] saved to:", outdir.resolve())
        print("  -", (outdir/"energy_fractions_per_deputy.png").resolve())
        print("  -", (outdir/"dv_cmd_vs_control_per_deputy.png").resolve())
        print("  -", (outdir/"peak_rafti_force_per_deputy.png").resolve())
        print("  -", (outdir/"fleet_omega_overlay.png").resolve())
        print("  -", (outdir/"layer1_summary.csv").resolve())


def main():
    f=Fleet(); f.run()
    print("[DONE] ESPA separation via RAFTI + Layer-1 damping + PO/PC. See layer1_report/ for plots & CSVs.")


if __name__=="__main__":
    main()
