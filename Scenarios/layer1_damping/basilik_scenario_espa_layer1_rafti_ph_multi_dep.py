#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_scenario_espa_layer1_rafti_ph_multi_dep.py

Purpose
-------
Basilisk scenario that exercises the ARISE Layer-1 damping architecture for an
ESPA-style carrier with multiple CubeSat deputies, focusing on:

- RAFTI-aware separation forces (half-sine pushes) applied at each mount with
  equal-and-opposite forces/torques on carrier and deputies.
- A 50 Hz midpoint (Tustin) port–Hamiltonian (pH) Layer-1 damping controller
  acting on each vehicle (carrier + all deputies).
- BNLS thruster allocation, reaction-wheel torque limits, latency buffers, and
  Δv / propellant accounting.
- On-screen diagnostics and plots only — **no files are written**.

Scenario overview
-----------------
Geometry and vehicles:
- Carrier:
  * “MR-22N-like” bus with 4 illustrative thrusters (22 N, Isp ≈ 235 s).
  * Mass m_car ≈ 650 kg, diagonal inertia diag([1200, 1000, 800]) kg·m².
  * Initial attitude and translational state at rest in inertial frame.

- Deputies:
  * Eight 12-kg CubeSats with inertia diag([0.025, 0.020, 0.018]) kg·m².
  * MPS-120-like 4-thruster layout per deputy (small thrust, off-axis nozzles).
  * Mounted uniformly around a 1 m radius ESPA-like ring (z_mount = 0).
  * Each deputy body frame is aligned with the local port frame:
      +Y_B along the outward ESPA radial direction,
      +Z_B ≈ +Z_car,
      +X_B = +Y_B × +Z_B.

RAFTI separation model:
- Each deputy receives a half-sine push:
    F(t) = F0 sin(π (t − t0) / Tpush),   t ∈ [t0, t0+Tpush],
  with:
    * t0_base = 10 s and per-deputy spacing of 5 s between pushes.
    * Tpush = 0.30 s for all deputies.
    * Δv_cmd ~ N(0.018, (0.005)²) m/s, clipped to [0.004, 0.025] m/s and
      consistent with RAFTI’s 25 mm/s relative-velocity rating.

- RAFTI-aware checks at push start:
    * Equivalent mass m_eq = m_car m_dep / (m_car + m_dep) ≤ 1000 kg.
    * Δv_cmd ≤ 0.025 m/s (25 mm/s RAFTI approach/separation limit).
    * Peak F0 ≤ CONTACT_LIMIT_N (≈ 11.6 kN) — a warning is printed otherwise.

- Mount variability:
    * Contact normals s_car_B randomly tilted with σ ≈ 2 deg.
    * Lever arms r_car_B and r_dep_B perturbed with σ ≈ 2 cm.
    * These perturbations emulate 1-σ RAFTI stack-up in mounting / tilt.

Layer-1 pH damping controller
-----------------------------
- Implemented by `PHLayer1` for each vehicle (carrier + 8 deputies).
- Dynamics:
  * Uses 6-DOF kinetic energy:
        H_d = ½ ωᵀ J ω + ½ m v_Bᵀ v_B + ½ ηᵀ K_eta η,
    where η = [φ; ρ] is a progress variable (here kept at zero for pure damping).
  * Midpoint (Tustin) evaluation:
        ω̄ = ½ (ω_k + ω_{k−1}), v̄_B = ½ (v_B,k + v_B,k−1).

- Control law:
  * Desired body-frame wrench:
        τ_d = −K_ω ω̄
        f_d = −K_v v̄_B
    where K_ω ∝ J / τ_ω and K_v ∝ m / τ_v, with τ_ω, τ_v set per vehicle.
  * Wheel torque limit:
        ‖τ_rw‖ ≤ rw_tau_max, jets handle residual torque and all force.
  * Thruster BNLS allocation:
        - 6×N “wrench” matrix B constructed from thruster location and direction.
        - Weighted least-squares solve with (w_torque, w_force, λ) and box
          constraints 0 ≤ δ_i ≤ 1 for duty cycles.
  * Latency:
        - Commanded torque/force stored in a ring buffer of length L and applied
          after L steps to emulate actuation delay.

- Energy monotonicity:
  * H_d is tracked once each deputy has finished its own push (engage_time).
  * For t ≥ engage_time, the script counts how often H_d(k) ≤ H_d(k−1); it
    prints:
        non-increasing fraction, Δv, propellant mass, and any violations
    per vehicle.

Logging (in memory only)
------------------------
For each `PHLayer1` controller, the scenario tracks:

- t_list, Hd_list
- tau_d_hist, f_d_hist
- tau_rw_hist, tau_thr_hist
- torque_cmd_hist, force_cmd_hist
- torque_apply_hist, force_apply_hist
- delta_hist, thrust_total_hist
- omega_hist, vB_hist, sigma_hist
- energy_violations (time & ΔH_d for > 0 jumps)
- dv_cum, mp_cum (Δv and propellant usage estimates)

These are not written to disk; they are only used for printed reports and
interactive plots.

Plots (display only)
--------------------
At the end of the run, the script:

1. Prints:
   - Carrier and each deputy’s:
        * H_d monotonic fraction after engage_time,
        * Δv ≈ … m/s, and m_p ≈ … g.

2. For the carrier and each deputy (if matplotlib is available):
   - Multi-panel figure with:
        * H_d vs time (with engage_time marker),
        * Norms of τ_d, τ_cmd, τ_applied,
        * Norms of f_d, f_cmd, f_applied,
        * Total thrust ∑ Tδ,
        * ω components (if available),
        * σ (MRP) components.
   - Relative torque/force error plots:
        * ‖τ_d − τ_cmd‖ / ‖τ_d‖ and ‖τ_d − τ_applied‖ / ‖τ_d‖,
        * ‖f_d − f_cmd‖ / ‖f_d‖ and ‖f_d − f_applied‖ / ‖f_d‖.

3. Fleet overlay:
   - Single plot of ω norms (‖ω‖) per deputy vs time.

All plots are generated with `plt.show()`; nothing is saved.

Usage
-----
Run in a Basilisk-enabled Python environment:

    python basilik_scenario_espa_layer1_rafti_ph_multi_dep.py

The script will:
- Build an ESPA-ring carrier with eight deputies and RAFTI-style separation.
- Run Layer-1 pH damping at 50 Hz with BNLS allocation and latency.
- Print per-vehicle energy/Δv/prop summaries.
- Display interactive diagnostic plots, without writing any log files.
"""


from __future__ import annotations
import math, dataclasses as dc
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

# ──────────────────────────────────────────────────────────────────────────
# Optional plotting (headless-safe). If unavailable, plots are skipped.
# ──────────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    # NOTE: do NOT force Agg backend so plt.show() works interactively
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ──────────────────────────────────────────────────────────────────────────
# RAFTI constraints we enforce (see notes below)
# ──────────────────────────────────────────────────────────────────────────

class RAFTI:
    # Docking/utility allowances and ratings (RAFTI User Guide)
    # – Docking/utility misalignment to GRIP (±10 mm, ±5° about X/Y/Z)
    DOCK_XY_TOL_M   = 0.010
    DOCK_ANG_TOL_DEG= 5.0
    # – Utility (valve/open) misalignment limits (±1 mm, ±2° about Z)
    UTIL_XY_TOL_M   = 0.001
    UTIL_Z_ANG_DEG  = 2.0
    # – Preload for utility connection (clamp)
    UTIL_PRELOAD_N  = 2000.0
    # – Rated approach relative speed & equivalent mass
    DOCK_VEL_MAX_MPS= 0.025       # 25 mm/s
    EQMASS_MAX_KG   = 1000.0      # ≤1000 kg equivalent mass
    # – Mounting/Contact loads that inform checks (not directly used to push)
    CONTACT_LIMIT_N = 11600.0     # design load case at mounting interface

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    x,y,z=v; return np.array([[0,-z,y],[z,0,-x],[-y,x,0]], float)

def mrp_to_dcm(sigma: np.ndarray) -> np.ndarray:
    s2=float(np.dot(sigma,sigma)); I3=np.eye(3)
    if s2<1e-16: return I3.copy()
    S=skew(sigma)
    return I3 + (8*S@S - 4*(1-s2)*S) / (1+s2)**2

def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    tr=np.trace(C)
    if tr>0:
        S=math.sqrt(tr+1.0)*2.0; q0=0.25*S
        q1=(C[2,1]-C[1,2])/S; q2=(C[0,2]-C[2,0])/S; q3=(C[1,0]-C[0,1])/S
    else:
        i=int(np.argmax([C[0,0],C[1,1],C[2,2]]))
        if   i==0:
            S=math.sqrt(1.0+C[0,0]-C[1,1]-C[2,2])*2.0; q0=(C[2,1]-C[1,2])/S; q1=0.25*S
            q2=(C[0,1]+C[1,0])/S; q3=(C[0,2]+C[2,0])/S
        elif i==1:
            S=math.sqrt(1.0+C[1,1]-C[0,0]-C[2,2])*2.0; q0=(C[0,2]-C[2,0])/S
            q1=(C[0,1]+C[1,0])/S; q2=0.25*S; q3=(C[1,2]+C[2,1])/S
        else:
            S=math.sqrt(1.0+C[2,2]-C[0,0]-C[1,1])*2.0; q0=(C[1,0]-C[0,1])/S
            q1=(C[0,2]+C[2,0])/S; q2=(C[1,2]+C[2,1])/S; q3=0.25*S
    q=np.array([q0,q1,q2,q3]); return q/la.norm(q)

def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    q0,qv=q[0],q[1:]; return -qv/(1-q0) if 1+q0<1e-12 else qv/(1+q0)

def dcm_to_mrp(C: np.ndarray)->np.ndarray:
    return quat_to_mrp(dcm_to_quat(C))

def unit(v):
    n=la.norm(v)
    return (v/n if n>1e-12 else np.array([1,0,0],float))

def tilt_vec(v: np.ndarray, sigma_deg: float, rng: np.random.Generator) -> np.ndarray:
    v = unit(v)
    if sigma_deg<=0: return v
    a = rng.normal(scale=math.radians(sigma_deg), size=3)
    a -= np.dot(a,v)*v
    ang = la.norm(a)
    if ang<1e-10: return v
    k = a/ang; K=skew(k)
    R = np.eye(3) + math.sin(ang)*K + (1-math.cos(ang))*(K@K)
    return unit(R@v)

def meq(m1, m2):   # equivalent (reduced) mass for contact
    return (m1*m2)/(m1+m2)

# ──────────────────────────────────────────────────────────────────────────
# Configuration dataclasses
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
    # Note: no log_dir; we do not save any files

# ──────────────────────────────────────────────────────────────────────────
# Port–Hamiltonian Layer-1 damping
# ──────────────────────────────────────────────────────────────────────────

class PHLayer1:
    def __init__(self, veh: VehicleConfig, dt: float, engage_time: float, latency_s: float):
        self.veh=veh; self.dt=dt; self.engage_time=engage_time
        J=veh.inertia_B; m=veh.mass_kg
        self.K_omega=(1.0/max(veh.tau_omega_s,1e-6))*J
        self.K_v    =(m   /max(veh.tau_v_s,1e-6))*np.eye(3)
        self.K_eta  =1e-2*np.eye(6)
        self.ctrlTorqueMsg = messaging.CmdTorqueBodyMsg()
        self.ctrlForceMsg  = messaging.CmdForceBodyMsg()
        self.L=int(round(max(0.0,latency_s)/dt))
        self.buf_tau: List[np.ndarray]=[]; self.buf_for: List[np.ndarray]=[]
        self.B, self.Ts, self.Isps = self._build_wrench_map(veh.thrusters)
        self.delta_prev: Optional[np.ndarray]=None
        self.omega_prev=np.zeros(3); self.v_prev=np.zeros(3)
        self._Hd_prev=None; self.energy_total=0; self.energy_noninc=0
        self.dv_cum=0.0; self.mp_cum=0.0
        self.t_list=[]; self.Hd_list=[]

        # High-rate logs for diagnostics and plots
        self.tau_d_hist=[]; self.f_d_hist=[]
        self.tau_rw_hist=[]; self.tau_thr_hist=[]
        self.torque_cmd_hist=[]; self.force_cmd_hist=[]
        self.torque_apply_hist=[]; self.force_apply_hist=[]
        self.delta_hist=[]; self.thrust_total_hist=[]
        self.omega_hist=[]; self.vB_hist=[]; self.sigma_hist=[]
        self.energy_violations=[]  # list of (t, ΔH_d>0)

    def _build_wrench_map(self, thr: List[Thruster]) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
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
            Bf=B[:,free]; K=la.pinv(Bf.T@W@Bf + lam*np.eye(np.count_nonzero(free)))
            rhs=Bf.T@W@(u - B[:,low]@np.full(np.count_nonzero(low), lb)
                          - B[:,high]@np.full(np.count_nonzero(high), ub)) + lam*delta_prev[free]
            delta_f=K@rhs; delta=delta.copy(); delta[free]=delta_f; delta=np.clip(delta,lb,ub)
        return delta

    def step(self, t: float, attNav: messaging.NavAttMsgPayload, transNav: messaging.NavTransMsgPayload):
        dt=self.dt; J=self.veh.inertia_B; m=self.veh.mass_kg
        sigma=np.array(attNav.sigma_BN); C_BN=mrp_to_dcm(sigma)
        omega=np.array(attNav.omega_BN_B)
        v_N=np.array(transNav.v_BN_N); v_B=C_BN@v_N

        phi=np.zeros(3); rho=np.zeros(3); eta=np.hstack((phi,rho))  # Layer-1: pure damping

        omega_bar=0.5*(omega+self.omega_prev); v_bar=0.5*(v_B+self.v_prev)
        tau_d= - self.K_omega @ omega_bar
        f_d  = - self.K_v     @ v_bar

        tau_rw=tau_d.copy(); tn=la.norm(tau_rw)
        if tn>self.veh.rw_tau_max>0: tau_rw*= (self.veh.rw_tau_max/tn)
        tau_res=tau_d - tau_rw

        tau_thr=np.zeros(3); f_thr=np.zeros(3); delta=np.zeros(0)
        if self.B.size>0:
            delta=self._bnls(self.B, np.hstack((tau_res,f_d)), self.delta_prev,
                             self.veh.w_torque, self.veh.w_force, self.veh.reg_lambda, 0.0, 1.0)
            self.delta_prev=delta.copy()
            uf=self.B@delta; tau_thr=uf[:3]; f_thr=uf[3:]

        tau_real=tau_rw + tau_thr; f_real=f_thr

        # log desired/commanded
        self.tau_d_hist.append(tau_d.copy()); self.f_d_hist.append(f_d.copy())
        self.tau_rw_hist.append(tau_rw.copy()); self.tau_thr_hist.append(tau_thr.copy())
        self.torque_cmd_hist.append(tau_real.copy()); self.force_cmd_hist.append(f_real.copy())
        self.delta_hist.append(delta.copy() if delta.size else np.zeros(0))
        self.omega_hist.append(omega.copy()); self.vB_hist.append(v_B.copy()); self.sigma_hist.append(sigma.copy())
        self.thrust_total_hist.append(float(np.sum(self.Ts*delta)) if self.Ts.size else 0.0)

        self.buf_tau.append(tau_real); self.buf_for.append(f_real)
        if len(self.buf_tau)>self.L:
            tau_apply=self.buf_tau.pop(0); f_apply=self.buf_for.pop(0)
        else:
            tau_apply=np.zeros(3); f_apply=np.zeros(3)
        # log applied (after latency)
        self.torque_apply_hist.append(tau_apply.copy()); self.force_apply_hist.append(f_apply.copy())

        t_pl=messaging.CmdTorqueBodyMsgPayload(); t_pl.torqueRequestBody=list(tau_apply)
        f_pl=messaging.CmdForceBodyMsgPayload();  f_pl.forceRequestBody =list(f_apply)
        self.ctrlTorqueMsg.write(t_pl); self.ctrlForceMsg.write(f_pl)

        Hd=0.5*omega.T@(J@omega) + 0.5*m*(v_B@v_B) + 0.5*eta.T@(self.K_eta@eta)
        if t>=self.engage_time:
            self.energy_total+=1
            if self._Hd_prev is not None:
                if Hd<=self._Hd_prev+1e-12:
                    self.energy_noninc+=1
                else:
                    self.energy_violations.append((float(t), float(Hd-self._Hd_prev)))
            self._Hd_prev=Hd
        self.t_list.append(t); self.Hd_list.append(Hd)

        if delta.size>0:
            impulse=float(np.sum(self.Ts*delta)*dt)
            self.dv_cum += impulse/m
            if impulse>0:
                self.mp_cum += impulse / (np.average(self.Isps, weights=self.Ts*delta)*9.80665)

        self.omega_prev=omega.copy(); self.v_prev=(v_B.copy())

# ──────────────────────────────────────────────────────────────────────────
# RAFTI separation driver (equal-and-opposite)
# ──────────────────────────────────────────────────────────────────────────

@dc.dataclass
class SepParams:
    t_start: float
    t_end:   float
    F0:      float
    s_car_B: np.ndarray
    r_car_B: np.ndarray
    r_dep_B: np.ndarray
    # bookkeeping for enforcement/telemetry
    dv_cmd:  float

class RAFTISeparation:
    """Half-sine push forces at each RAFTI mount (equal-and-opposite)."""
    def __init__(self, dt: float,
                 sc_car: spacecraft.Spacecraft, eff_car: extForceTorque.ExtForceTorque,
                 sc_dep: List[spacecraft.Spacecraft], eff_dep: List[extForceTorque.ExtForceTorque],
                 sched: List[SepParams],
                 m_car: float, m_dep: float):
        self.dt=dt; self.sched=sched
        self.sc_car=sc_car; self.eff_car=eff_car
        self.sc_dep=sc_dep; self.eff_dep=eff_dep
        self.m_car=m_car; self.m_dep=m_dep
        # messages
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
        # Read transforms once
        C_BN_car = mrp_to_dcm(np.array(self.sc_car.scStateOutMsg.read().sigma_BN))
        C_NB_car = C_BN_car.T

        tau_car = np.zeros(3); f_car = np.zeros(3)
        for i,sp in enumerate(self.sched):
            # deputy i transforms
            C_BN_dep = mrp_to_dcm(np.array(self.sc_dep[i].scStateOutMsg.read().sigma_BN))
            C_NB_dep = C_BN_dep.T

            # peak enforcement check at t=t0 (once)
            if abs(t - sp.t_start) < 0.5*self.dt:
                # RAFTI ratings gate: approach/separation Δv and equivalent mass
                m_eq = meq(self.m_car, self.m_dep)
                if m_eq > RAFTI.EQMASS_MAX_KG:
                    print(f"[RAFTI WARN dep{i+1}] m_eq={m_eq:.1f} kg exceeds {RAFTI.EQMASS_MAX_KG:.0f} kg rating")
                if sp.dv_cmd > RAFTI.DOCK_VEL_MAX_MPS:
                    print(f"[RAFTI WARN dep{i+1}] Δv_cmd={sp.dv_cmd:.3f} m/s exceeds RAFTI 25 mm/s rating")
                if sp.F0 > RAFTI.CONTACT_LIMIT_N:
                    print(f"[RAFTI WARN dep{i+1}] peak push {sp.F0:.0f} N > contact load {RAFTI.CONTACT_LIMIT_N:.0f} N")

            # force magnitude (half-sine)
            Fmag = self._half_sine(t, sp.t_start, sp.t_end-sp.t_start, sp.F0)
            if Fmag<=0:
                # write zeros to deputy i
                tm=messaging.CmdTorqueBodyMsgPayload(); tm.torqueRequestBody=[0.0,0.0,0.0]
                fm=messaging.CmdForceBodyMsgPayload();  fm.forceRequestBody =[0.0,0.0,0.0]
                self.dep_tauMsg[i].write(tm); self.dep_forMsg[i].write(fm)
                continue

            # contact normal in inertial (via carrier body)
            s_N = C_NB_car @ sp.s_car_B
            F_N = Fmag * s_N

            # equal-and-opposite in body frames
            f_car_i_B =  C_BN_car @ ( +F_N )
            f_dep_i_B =  C_BN_dep @ ( -F_N )
            tau_car_i = np.cross(sp.r_car_B, f_car_i_B)
            tau_dep_i = np.cross(sp.r_dep_B, f_dep_i_B)

            # deputy i publish
            tm=messaging.CmdTorqueBodyMsgPayload(); tm.torqueRequestBody=list(tau_dep_i)
            fm=messaging.CmdForceBodyMsgPayload();  fm.forceRequestBody =list(f_dep_i_B)
            self.dep_tauMsg[i].write(tm); self.dep_forMsg[i].write(fm)

            # accumulate for carrier
            tau_car += tau_car_i; f_car += f_car_i_B

        # carrier publish (superposed)
        tm=messaging.CmdTorqueBodyMsgPayload(); tm.torqueRequestBody=list(tau_car)
        fm=messaging.CmdForceBodyMsgPayload();  fm.forceRequestBody =list(f_car)
        self.car_tauMsg.write(tm); self.car_forMsg.write(fm)

# ──────────────────────────────────────────────────────────────────────────
# Fleet assembly & run
# ──────────────────────────────────────────────────────────────────────────

class Fleet:
    def __init__(self):
        self.dt=0.02
        self.sim = SimConfig(dt=self.dt, T_end=240.0, latency_s=0.05)

        rng=np.random.default_rng(42)

        # Carrier config (MR-22N illustrative)
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

        # Deputies (MPS-120 illustrative)
        mps_T, mps_Isp = 0.27, 217.0
        thr_d = [
            Thruster(np.array([ 0.05, 0.0,  0.05]), np.array([ 0.05,-0.998,  0.05]), mps_T, mps_Isp, "MPS_1"),
            Thruster(np.array([-0.05, 0.0,  0.05]), np.array([-0.05,-0.998,  0.05]), mps_T, mps_Isp, "MPS_2"),
            Thruster(np.array([ 0.05, 0.0, -0.05]), np.array([ 0.05,-0.998, -0.05]), mps_T, mps_Isp, "MPS_3"),
            Thruster(np.array([-0.05, 0.0, -0.05]), np.array([-0.05,-0.998, -0.05]), mps_T, mps_Isp, "MPS_4"),
        ]
        I_dep=np.diag([0.025,0.020,0.018]); m_dep=12.0

        self.deps: List[VehicleConfig]=[]
        self.mount_dirs_car_B=[]   # outward normals at each mount (carrier body)
        self.mount_pos_car_B=[]    # lever arms on carrier body
        self.mount_R_BD_BC=[]      # deputy body DCSY wrt carrier body CSY at mount

        ring_R = 1.00; z_mount = 0.0
        for i in range(8):
            ang= i*(2*math.pi/8.0)
            s_car_B = np.array([math.cos(ang), math.sin(ang), 0.0], float) # outward
            r_car_B = np.array([ ring_R*math.cos(ang), ring_R*math.sin(ang), z_mount ], float)
            # Deputy body axes: bY along s_car_B (points outward), bZ ~ +Z_car, bX = bY×bZ
            bY = s_car_B; bZ=np.array([0,0,1.0])
            if abs(np.dot(unit(bY),bZ))>0.99: bZ=np.array([1,0,0])
            bX = unit(np.cross(bY,bZ)); bZ = unit(np.cross(bX,bY))
            R_BD_BC = np.column_stack((bX,bY,bZ))  # carrier->deputy body

            sigma0 = dcm_to_mrp(R_BD_BC)
            r0     = r_car_B.copy()
            v0     = np.zeros(3)

            dep=VehicleConfig(name=f"dep{i+1}", mass_kg=m_dep, inertia_B=I_dep,
                              sigma0_BN=sigma0, omega0_B=np.zeros(3), r0_N=r0, v0_N=v0,
                              tau_omega_s=25.0, tau_v_s=30.0, rw_tau_max=0.020,
                              w_force=200.0, w_torque=0.5, reg_lambda=2e-4,
                              thrusters=thr_d)
            self.deps.append(dep); self.mount_dirs_car_B.append(s_car_B)
            self.mount_pos_car_B.append(r_car_B); self.mount_R_BD_BC.append(R_BD_BC)

        # Build tasks
        dt=self.dt
        self.dyn = sys_model_task.SysModelTask(int(macros.sec2nano(dt)))
        self.sns = sys_model_task.SysModelTask(int(macros.sec2nano(dt)))

        # Carrier spacecraft + nav + two effectors
        self.sc_car = spacecraft.Spacecraft(); self.sc_car.ModelTag="SC_carrier"
        self.sc_car.hub.mHub=self.car.mass_kg; self.sc_car.hub.IHubPntBc_B=self.car.inertia_B.tolist()
        self.sc_car.hub.sigma_BNInit=list(self.car.sigma0_BN)
        self.sc_car.hub.omega_BN_BInit=list(self.car.omega0_B)
        self.sc_car.hub.r_CN_NInit=list(self.car.r0_N)
        self.sc_car.hub.v_CN_NInit=list(self.car.v0_N)
        self.nav_car = simpleNav.SimpleNav(); self.nav_car.ModelTag="NAV_carrier"
        self.nav_car.scStateInMsg.subscribeTo(self.sc_car.scStateOutMsg)
        self.eft_car_ctrl = extForceTorque.ExtForceTorque(); self.eft_car_ctrl.ModelTag="EFT_car_ctrl"
        self.eft_car_sep  = extForceTorque.ExtForceTorque(); self.eft_car_sep .ModelTag="EFT_car_sep"
        self.sc_car.addDynamicEffector(self.eft_car_ctrl); self.sc_car.addDynamicEffector(self.eft_car_sep)

        # Deputies
        self.sc_dep: List[spacecraft.Spacecraft]=[]
        self.nav_dep: List[simpleNav.SimpleNav]=[]
        self.eft_dep_ctrl: List[extForceTorque.ExtForceTorque]=[]
        self.eft_dep_sep:  List[extForceTorque.ExtForceTorque]=[]
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

        # Add objects to tasks
        p=0
        self.sns.AddNewObject(self.nav_car,10+p); self.dyn.AddNewObject(self.sc_car,5+p)
        self.dyn.AddNewObject(self.eft_car_ctrl,10+p); self.dyn.AddNewObject(self.eft_car_sep,11+p)
        for i,(sc,nav,ec,es) in enumerate(zip(self.sc_dep,self.nav_dep,self.eft_dep_ctrl,self.eft_dep_sep),1):
            self.sns.AddNewObject(nav,  10+p+10*i)
            self.dyn.AddNewObject(sc,    5+p+10*i)
            self.dyn.AddNewObject(ec,   10+p+10*i)
            self.dyn.AddNewObject(es,   11+p+10*i)

        # Controllers (start energy checks only after the deputy finishes its separation)
        last_sep_end = 10.0 + (len(self.deps)-1)*5.0 + 0.30
        self.ctrl_car = PHLayer1(self.car, dt, engage_time=last_sep_end, latency_s=self.sim.latency_s)
        self.ctrl_dep: List[PHLayer1]=[]
        for k,dep in enumerate(self.deps):
            t_end = 10.0 + k*5.0 + 0.30
            self.ctrl_dep.append(PHLayer1(dep, dt, engage_time=t_end, latency_s=self.sim.latency_s))

        # Wire control ports
        self.eft_car_ctrl.cmdTorqueInMsg   .subscribeTo(self.ctrl_car.ctrlTorqueMsg)
        self.eft_car_ctrl.cmdForceBodyInMsg.subscribeTo(self.ctrl_car.ctrlForceMsg)
        for ctl,ec in zip(self.ctrl_dep, self.eft_dep_ctrl):
            ec.cmdTorqueInMsg   .subscribeTo(ctl.ctrlTorqueMsg)
            ec.cmdForceBodyInMsg.subscribeTo(ctl.ctrlForceMsg)

        # Separation schedule with per-deputy randomness consistent with RAFTI rating:
        # Δv ~ N(0.018, 0.005^2) m/s (cap at 0.025 m/s); tilt σ≈2°, lever σ≈2 cm
        sched=[]
        t0_base=10.0; spacing=5.0; Tpush=0.30
        for k in range(len(self.deps)):
            t0 = t0_base + k*spacing
            dv = float(np.clip(np.random.default_rng(42+k).normal(0.018, 0.005), 0.004, RAFTI.DOCK_VEL_MAX_MPS))
            F0 = self.deps[k].mass_kg * dv * math.pi / (2.0*Tpush)
            s_car_B = tilt_vec(self.mount_dirs_car_B[k], 2.0, rng)
            r_car_B = self.mount_pos_car_B[k] + rng.normal(scale=0.02, size=3)
            r_dep_B = rng.normal(scale=0.02, size=3)
            sched.append(SepParams(t_start=t0, t_end=t0+Tpush, F0=F0,
                                   s_car_B=s_car_B, r_car_B=r_car_B, r_dep_B=r_dep_B,
                                   dv_cmd=dv))
        self.sep = RAFTISeparation(self.dt, self.sc_car, self.eft_car_sep,
                                   self.sc_dep, self.eft_dep_sep, sched,
                                   m_car=self.car.mass_kg, m_dep=self.deps[0].mass_kg)

        # Finalize tasks
        for tsk in (self.sns, self.dyn): tsk.SelfInitTaskList(); tsk.ResetTaskList(0)

    def run(self):
        now_ns=0; N=int(self.sim.T_end/self.dt)
        for k in tqdm(range(N), desc="Layer-1 ESPA+RAFTI+PH", ncols=80):
            t=k*self.dt
            self.sns.ExecuteTaskList(now_ns)
            self.sep.step(t)
            self.ctrl_car.step(t, self.nav_car.attOutMsg.read(), self.nav_car.transOutMsg.read())
            for ctl,nav in zip(self.ctrl_dep, self.nav_dep):
                ctl.step(t, nav.attOutMsg.read(), nav.transOutMsg.read())
            self.dyn.ExecuteTaskList(now_ns)
            now_ns+=int(macros.sec2nano(self.dt))

        # Summary prints (no file I/O)
        frac = self.ctrl_car.energy_noninc/max(1,self.ctrl_car.energy_total)
        print(f"[carrier] H_d monotonic after t≥{self.ctrl_car.engage_time:.1f}s: {frac:.3f}, "
              f"Δv≃{self.ctrl_car.dv_cum:.4f} m/s, m_p={self.ctrl_car.mp_cum*1e3:.3f} g")
        for i,ctl in enumerate(self.ctrl_dep,1):
            frac = ctl.energy_noninc/max(1,ctl.energy_total)
            print(f"[dep{i}]    H_d monotonic after t≥{ctl.engage_time:.1f}s: {frac:.3f}, "
                  f"Δv≃{ctl.dv_cum:.4f} m/s, m_p={ctl.mp_cum*1e3:.3f} g")

        # ───────────────────────────────────────────────────────────────
        # Controller diagnostics and plots (display only)
        # ───────────────────────────────────────────────────────────────
        def _np(x): return np.array(x) if not isinstance(x, np.ndarray) else x

        def controller_report(prefix: str, veh: VehicleConfig, ctl: PHLayer1):
            def diag(M):
                M=np.asarray(M)
                return np.diag(M) if M.ndim==2 else M
            Kω=diag(ctl.K_omega); Kv=diag(ctl.K_v)
            tau_d = _np(ctl.tau_d_hist); f_d=_np(ctl.f_d_hist)
            tau_c = _np(ctl.torque_cmd_hist); f_c=_np(ctl.force_cmd_hist)
            tau_a = _np(ctl.torque_apply_hist); f_a=_np(ctl.force_apply_hist)
            eps = 1e-12
            def rel_err(a,b):
                if a.size==0: return np.array([0.0])
                num=la.norm(a-b,axis=1)
                den=np.maximum(la.norm(a,axis=1), eps)
                return num/den
            re_t_cmd = rel_err(tau_d, tau_c)
            re_t_app = rel_err(tau_d, tau_a)
            re_f_cmd = rel_err(f_d, f_c)
            re_f_app = rel_err(f_d, f_a)
            def stats(x):
                return np.nanmean(x), np.nanpercentile(x,95), np.nanmax(x)
            mtc,p95tc,maxtc=stats(re_t_cmd); mta,p95ta,maxta=stats(re_t_app)
            mfc,p95fc,maxfc=stats(re_f_cmd); mfa,p95fa,maxfa=stats(re_f_app)
            # monotonicity
            vio_count=len(ctl.energy_violations)
            first_vio=f"{ctl.energy_violations[0][0]:.3f}s, ΔH={ctl.energy_violations[0][1]:.3e}" if vio_count else "None"
            # maxima
            tnorm_cmd=np.max(la.norm(tau_c,axis=1)) if tau_c.size else 0.0
            tnorm_app=np.max(la.norm(tau_a,axis=1)) if tau_a.size else 0.0
            fnorm_cmd=np.max(la.norm(f_c,axis=1)) if f_c.size else 0.0
            fnorm_app=np.max(la.norm(f_a,axis=1)) if f_a.size else 0.0
            thrust_pk = float(np.max(_np(ctl.thrust_total_hist))) if len(ctl.thrust_total_hist) else 0.0

            print(f"\n[{prefix}] Controller info")
            print(f"  τ_ω={veh.tau_omega_s:.2f}s → diag(K_ω)={Kω}")
            print(f"  τ_v={veh.tau_v_s:.2f}s → diag(K_v)={Kv}")
            print(f"  K_eta=1e-2 I6, RW τ_max={veh.rw_tau_max:.3f} N·m, latency={ctl.L} steps ≃{ctl.L*ctl.dt:.3f}s")
            print(f"  BNLS: wτ={veh.w_torque:.3g}, wf={veh.w_force:.3g}, λ={veh.reg_lambda:.2e}, B.shape={ctl.B.shape}")
            if len(veh.thrusters):
                thr_info = ", ".join([f"{th.name}:{th.T:.2f}N@Isp{th.Isp:.0f}" for th in veh.thrusters])
                print(f"  Thrusters ({len(veh.thrusters)}): {thr_info}")
            print(f"  Δv_accum={ctl.dv_cum:.4f} m/s, m_p={ctl.mp_cum*1e3:.3f} g")
            print(f"  Max |τ| cmd/appl = {tnorm_cmd:.3f}/{tnorm_app:.3f} N·m; Max |f| cmd/appl = {fnorm_cmd:.3f}/{fnorm_app:.3f} N; Peak ∑Tδ={thrust_pk:.3f} N")
            print(f"  Monotonicity after t≥{ctl.engage_time:.2f}s: non-increasing fraction "
                  f"{ctl.energy_noninc/max(1,ctl.energy_total):.3f}; violations={vio_count}; first={first_vio}")
            print(f"  Rel. torque error (desired→cmd/appl) mean/95%/max = "
                  f"{mtc:.3f}/{p95tc:.3f}/{maxtc:.3f}  |  {mta:.3f}/{p95ta:.3f}/{maxta:.3f}")
            print(f"  Rel. force  error (desired→cmd/appl) mean/95%/max = "
                  f"{mfc:.3f}/{p95fc:.3f}/{maxfc:.3f}  |  {mfa:.3f}/{p95fa:.3f}/{maxfa:.3f}")

        def make_plots(prefix: str, title: str, ctl: PHLayer1):
            if not HAS_MPL:
                print(f"[{prefix}] matplotlib not available; skipping plots.")
                return
            t=_np(ctl.t_list)
            Hd=_np(ctl.Hd_list)
            tau_d=_np(ctl.tau_d_hist); tau_c=_np(ctl.torque_cmd_hist); tau_a=_np(ctl.torque_apply_hist)
            f_d=_np(ctl.f_d_hist); f_c=_np(ctl.force_cmd_hist); f_a=_np(ctl.force_apply_hist)
            omg=_np(ctl.omega_hist); sig=_np(ctl.sigma_hist)
            thrust=_np(ctl.thrust_total_hist)
            eps=1e-12

            # Overview figure
            fig,axs=plt.subplots(6,1,figsize=(10,14),sharex=True)
            axs[0].plot(t,Hd); axs[0].axvline(ctl.engage_time,linestyle='--',alpha=0.6)
            axs[0].set_ylabel("H_d"); axs[0].set_title(f"{title} — pH energy & histories")
            def nrm(x): return la.norm(x,axis=1) if x.size else np.zeros_like(t)
            axs[1].plot(t,nrm(tau_d),label="|τ_d|")
            axs[1].plot(t,nrm(tau_c),label="|τ_cmd|")
            axs[1].plot(t,nrm(tau_a),label="|τ_appl|"); axs[1].set_ylabel("Torque norm")
            axs[1].legend(loc="upper right",fontsize=8)
            axs[2].plot(t,nrm(f_d),label="|f_d|")
            axs[2].plot(t,nrm(f_c),label="|f_cmd|")
            axs[2].plot(t,nrm(f_a),label="|f_appl|"); axs[2].set_ylabel("Force norm")
            axs[2].legend(loc="upper right",fontsize=8)
            axs[3].plot(t,thrust); axs[3].set_ylabel("∑T·δ  [N]")
            if omg.size:
                axs[4].plot(t,omg[:,0],label="ω_x"); axs[4].plot(t,omg[:,1],label="ω_y"); axs[4].plot(t,omg[:,2],label="ω_z")
                axs[4].set_ylabel("ω [rad/s]"); axs[4].legend(loc="upper right",fontsize=8)
            if sig.size:
                axs[5].plot(t,sig[:,0],label="σ_1"); axs[5].plot(t,sig[:,1],label="σ_2"); axs[5].plot(t,sig[:,2],label="σ_3")
                axs[5].set_ylabel("MRP σ"); axs[5].legend(loc="upper right",fontsize=8)
            axs[-1].set_xlabel("time [s]"); fig.tight_layout()
            plt.show()
            plt.close(fig)

            # Error ratios
            def rel(a,b):
                if not a.size: return np.zeros_like(t)
                num=la.norm(a-b,axis=1); den=np.maximum(la.norm(a,axis=1),eps); return num/den
            eτ_cmd=rel(tau_d,tau_c); eτ_app=rel(tau_d,tau_a)
            ef_cmd=rel(f_d,f_c);     ef_app=rel(f_d,f_a)
            fig2,ax2=plt.subplots(2,1,figsize=(10,6),sharex=True)
            ax2[0].plot(t,eτ_cmd,label="τ: desired→cmd"); ax2[0].plot(t,eτ_app,label="τ: desired→appl")
            ax2[0].set_ylabel("Rel. torque error"); ax2[0].legend(loc="upper right",fontsize=8)
            ax2[1].plot(t,ef_cmd,label="f: desired→cmd"); ax2[1].plot(t,ef_app,label="f: desired→appl")
            ax2[1].set_ylabel("Rel. force error"); ax2[1].set_xlabel("time [s]"); ax2[1].legend(loc="upper right",fontsize=8)
            fig2.tight_layout()
            plt.show()
            plt.close(fig2)

        def fleet_overlay_omega():
            if not HAS_MPL:
                return
            fig,ax=plt.subplots(1,1,figsize=(10,5))
            for i,ctl in enumerate(self.ctrl_dep,1):
                t=_np(ctl.t_list); omg=_np(ctl.omega_hist)
                if omg.size: ax.plot(t, la.norm(omg,axis=1), label=f"dep{i}")
            ax.set_title("Deputy ω norms"); ax.set_xlabel("time [s]"); ax.set_ylabel("|ω| [rad/s]")
            ax.legend(ncol=4,fontsize=8); fig.tight_layout()
            plt.show()
            plt.close(fig)

        # Per-vehicle reports and plots
        controller_report("carrier", self.car, self.ctrl_car)
        make_plots("carrier", "Carrier", self.ctrl_car)
        for i,(dep,ctl) in enumerate(zip(self.deps,self.ctrl_dep),1):
            tag=f"dep{i:02d}"
            controller_report(tag, dep, ctl)
            make_plots(tag, f"Deputy {i}", ctl)
        fleet_overlay_omega()

# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def main():
    fleet=Fleet()
    fleet.run()
    print("[DONE] ESPA separation via RAFTI + Layer-1 damping (plots shown; no files saved).")

if __name__=="__main__":
    main()
