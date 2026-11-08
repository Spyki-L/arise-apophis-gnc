#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_scenario_espa_layer1_rafti_ph_multi_dep_logging.py

Purpose
-------
Basilisk scenario for ARISE Layer-1 that couples an ESPA-style multi-deputy
separation via RAFTI adaptors with a midpoint (Tustin) port–Hamiltonian
damping layer on the carrier and all deputies. This version both **displays**
diagnostic plots with `plt.show()` and **saves numeric logs** (`*.npz`) for
offline analysis.

High-level behavior
-------------------
1. Build a carrier + 8-deputy ESPA ring:
   - Carrier:
     * Mass ≈ 650 kg, inertia diag([1200, 1000, 800]) kg·m².
     * Four MR-22N-like thrusters (22 N, Isp ≈ 235 s) arranged around the bus.
   - Deputies:
     * Eight 12 kg CubeSats (MPS-120-like) with inertia diag([0.025, 0.020, 0.018]) kg·m².
     * Mounted uniformly on a ring of radius 1.0 m at z = 0.
     * Local body frames aligned with the ESPA ports:
         +Y_B along the outward ring normal,
         +Z_B ≈ +Z_car,
         +X_B = +Y_B × +Z_B.
     * Each deputy has a 4-thruster layout with small off-axis nozzles.

2. RAFTI-aware separation driver:
   - Per-deputy schedule:
     * Start times: t₀ = 10.0 s + k·5.0 s (k = 0..7).
     * Duration: Tpush = 0.30 s for each push.
     * Commanded Δv:
         Δv_cmd ~ N(0.018, 0.005²) m/s, clipped to [0.004, 0.025] m/s to stay
         within a RAFTI-like 25 mm/s separation rating.
     * Peak force F₀ = m_dep · Δv_cmd · π / (2·Tpush).
   - Per-deputy variability:
     * Contact normal s_car_B tilted with σ ≈ 2° around the nominal outward
       normal.
     * Lever arm r_car_B perturbed with σ ≈ 2 cm.
     * Deputy mount offset r_dep_B also perturbed with σ ≈ 2 cm.
   - RAFTI checks at push start:
     * Equivalent mass m_eq(m_car, m_dep) ≤ 1000 kg; warn if exceeded.
     * Δv_cmd ≤ 0.025 m/s; warn if exceeded.
     * Peak F₀ ≤ 11.6 kN (contact load rating); warn if exceeded.
   - Forces applied as equal-and-opposite:
     * Carrier: +F_N at the RAFTI mount, mapped into body as f_car_B and a
       torque τ_car = r_car_B × f_car_B.
     * Deputy i: −F_N at its own mount, with corresponding τ_dep = r_dep_B × f_dep_B.

3. Layer-1 pH damping (PHLayer1):
   - Implements pure SE(3) damping (no potential shaping) at 50 Hz using a
     midpoint (Tustin) scheme:
       * Rotational damping:
             τ_d = −K_ω ω̄,  with K_ω ≈ J / τ_ω.
       * Translational damping:
             f_d = −K_v v̄_B, with K_v ≈ m / τ_v.
       * Midpoint states:
             ω̄   = ½(ω_k + ω_{k−1}),
             v̄_B = ½(v_B,k + v_B,k−1).
   - Allocation and latency:
       * Reaction-wheel authority: ‖τ_rw‖ ≤ rw_tau_max; jets carry residual
         torque and all force.
       * BNLS allocator:
             min‖W^{1/2}(B δ − [τ_res; f_d])‖² + λ‖δ − δ_prev‖²,
         subject to 0 ≤ δ_i ≤ 1, where B is 6×N wrench map, W uses (w_torque,
         w_force), and λ = reg_lambda.
       * Command latency:
         - (τ_real, f_real) are added to a buffer; only after L steps
           (L ≈ latency_s / dt) are they applied to the ExtForceTorque module.
   - Energy monitoring:
       * Hamiltonian:
             H_d = ½ ωᵀJω + ½ m v_Bᵀv_B + ½ ηᵀK_eta η,
           with η ≡ 0 here (pure damping).
       * Once t ≥ engage_time (end of each vehicle’s last push), the script
         tracks how often H_d(k) ≤ H_d(k−1) and records any violations
         (time stamp and ΔH_d > 0).
   - Δv and propellant accounting:
       * Uses summed thrust ∑(T_i δ_i) over dt for each step.
       * Assumes a representative Isp per thruster and integrates approximate
         Δv and mass usage m_p.

4. Simulation layout
   - Two SysModelTasks:
       * `sns` — SimpleNav updates for carrier and all deputies.
       * `dyn` — spacecraft dynamics, separation forces, and control forces.
   - Time step:
       * dt = 0.02 s (50 Hz).
   - Duration:
       * T_end = 240 s (after the last deputy finishes separation).

Outputs
-------
1. Numeric logs (NPZ, saved under `layer1_logs/`):
   - Per-vehicle summary files:
       * `carrier_layer1_logs.npz`
       * `dep##_layer1_logs.npz` (one per deputy)
     each containing:
       - t                    : time array [s]
       - Hd                   : Hamiltonian history H_d(t)
       - dv_cum               : cumulative Δv estimate [m/s]
       - mp_cum               : cumulative propellant mass estimate [kg]
       - energy_noninc_frac   : fraction of steps with non-increasing H_d.

   - Rich timeseries files:
       * `carrier_layer1_timeseries.npz`
       * `dep##_layer1_timeseries.npz`
     each including:
       - t, Hd
       - tau_d, f_d
       - tau_cmd, f_cmd
       - tau_apply, f_apply
       - delta, thrust_total
       - omega, vB, sigma
       - rN (inertial position history)
       - engage_time
       - energy_violations (t, ΔH_d) for any increases.

2. Console report:
   - Carrier and each deputy:
       * H_d monotonic fraction after its engage_time.
       * Δv accumulated and equivalent propellant mass (m_p).
       * BNLS and controller settings (K_ω, K_v, wτ, wf, λ, RW limit, latency).
       * Rel. torque/force tracking errors (desired→commanded, desired→applied)
         with mean, 95th percentile, and max.

3. Plots (displayed only; not saved):
   - For carrier and each deputy:
       * Multi-panel figure:
           + H_d vs time with engage_time marker.
           + Norms of τ_d, τ_cmd, τ_appl.
           + Norms of f_d, f_cmd, f_appl.
           + ∑T δ (effective thrust) vs time.
           + ω components (ω_x, ω_y, ω_z) vs time.
           + MRP components (σ₁, σ₂, σ₃) vs time.
       * Relative torque/force error figure:
           + ‖τ_d − τ_cmd‖ / ‖τ_d‖ and ‖τ_d − τ_appl‖ / ‖τ_d‖ vs time.
           + ‖f_d − f_cmd‖ / ‖f_d‖ and ‖f_d − f_appl‖ / ‖f_d‖ vs time.
   - Fleet-wide ω overlay:
       * Single plot of ‖ω‖ vs time for all deputies.
   - r(t) grid:
       * M×3 grid (M = number of deputies) plotting r_x, r_y, r_z vs time for
         each deputy, showing inertial position behavior during and after
         separation.

Usage
-----
Run in a Basilisk-enabled Python environment:

    python basilik_scenario_espa_layer1_rafti_ph_multi_dep_logging.py

The script will:
- Build an ESPA-like carrier with 8 deputies and RAFTI-style separation.
- Run Layer-1 pH damping with BNLS allocation, RW limits, and latency.
- Print per-vehicle controller/energy statistics.
- Save summary and rich timeseries NPZ logs under `layer1_logs/`.
- Display interactive plots for energy, actuation, kinematics, and position
  histories without writing any images to disk.
"""


from __future__ import annotations

import math
import dataclasses as dc
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import numpy.linalg as la

# Basilisk
from Basilisk.architecture import messaging, sys_model_task
from Basilisk.simulation import spacecraft, simpleNav, extForceTorque
from Basilisk.utilities import macros

# Progress bar (optional)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(it, **kw): return it

# ──────────────────────────────────────────────────────────────────────────
# Plotting (display; do NOT force headless backend). If unavailable, skip.
# ──────────────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ──────────────────────────────────────────────────────────────────────────
# RAFTI constraints we enforce/check (reference: RAFTI User Guide)
# ──────────────────────────────────────────────────────────────────────────
class RAFTI:
    # Docking/utility allowances and ratings
    DOCK_XY_TOL_M = 0.010          # ±10 mm
    DOCK_ANG_TOL_DEG = 5.0         # ±5° about X/Y/Z
    UTIL_XY_TOL_M = 0.001          # ±1 mm
    UTIL_Z_ANG_DEG = 2.0           # ±2° about Z
    UTIL_PRELOAD_N = 2000.0        # 2 kN preload
    DOCK_VEL_MAX_MPS = 0.025       # 25 mm/s
    EQMASS_MAX_KG = 1000.0         # ≤ 1000 kg equivalent mass
    CONTACT_LIMIT_N = 11600.0      # design load case at mounting interface


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z,  y],
                     [z,   0.0, -x],
                     [-y,  x,   0.0]], float)

def mrp_to_dcm(sigma: np.ndarray) -> np.ndarray:
    s2 = float(np.dot(sigma, sigma))
    I3 = np.eye(3)
    if s2 < 1e-16:
        return I3.copy()
    S = skew(sigma)
    return I3 + (8*S@S - 4*(1 - s2)*S) / (1 + s2)**2

def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    tr = np.trace(C)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        q0 = 0.25 * S
        q1 = (C[2, 1] - C[1, 2]) / S
        q2 = (C[0, 2] - C[2, 0]) / S
        q3 = (C[1, 0] - C[0, 1]) / S
    else:
        i = int(np.argmax([C[0, 0], C[1, 1], C[2, 2]]))
        if i == 0:
            S = math.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2]) * 2.0
            q0 = (C[2, 1] - C[1, 2]) / S
            q1 = 0.25 * S
            q2 = (C[0, 1] + C[1, 0]) / S
            q3 = (C[0, 2] + C[2, 0]) / S
        elif i == 1:
            S = math.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2]) * 2.0
            q0 = (C[0, 2] - C[2, 0]) / S
            q1 = (C[0, 1] + C[1, 0]) / S
            q2 = 0.25 * S
            q3 = (C[1, 2] + C[2, 1]) / S
        else:
            S = math.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1]) * 2.0
            q0 = (C[1, 0] - C[0, 1]) / S
            q1 = (C[0, 2] + C[2, 0]) / S
            q2 = (C[1, 2] + C[2, 1]) / S
            q3 = 0.25 * S
    q = np.array([q0, q1, q2, q3], float)
    return q / la.norm(q)

def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    q0, qv = q[0], q[1:]
    return -qv/(1 - q0) if 1 + q0 < 1e-12 else qv/(1 + q0)

def dcm_to_mrp(C: np.ndarray) -> np.ndarray:
    return quat_to_mrp(dcm_to_quat(C))

def unit(v: np.ndarray) -> np.ndarray:
    n = la.norm(v)
    return (v/n if n > 1e-12 else np.array([1.0, 0.0, 0.0], float))

def tilt_vec(v: np.ndarray, sigma_deg: float, rng: np.random.Generator) -> np.ndarray:
    v = unit(v)
    if sigma_deg <= 0:
        return v
    a = rng.normal(scale=math.radians(sigma_deg), size=3)
    a -= np.dot(a, v) * v
    ang = la.norm(a)
    if ang < 1e-10:
        return v
    k = a / ang
    K = skew(k)
    R = np.eye(3) + math.sin(ang)*K + (1 - math.cos(ang))*(K@K)
    return unit(R @ v)

def meq(m1: float, m2: float) -> float:
    return (m1 * m2) / (m1 + m2)


# ──────────────────────────────────────────────────────────────────────────
# Configuration dataclasses
# ──────────────────────────────────────────────────────────────────────────
@dc.dataclass
class Thruster:
    r_B: np.ndarray
    a_B: np.ndarray
    T: float
    Isp: float
    name: str = "T"

@dc.dataclass
class VehicleConfig:
    name: str
    mass_kg: float
    inertia_B: np.ndarray
    sigma0_BN: np.ndarray
    omega0_B: np.ndarray
    r0_N: np.ndarray
    v0_N: np.ndarray
    # pH parameters
    tau_omega_s: float
    tau_v_s: float
    rw_tau_max: float
    # BNLS
    w_force: float
    w_torque: float
    reg_lambda: float
    thrusters: List[Thruster] = dc.field(default_factory=list)

@dc.dataclass
class SimConfig:
    dt: float = 0.02
    T_end: float = 240.0
    latency_s: float = 0.050
    log_dir: Path = Path("layer1_logs")


# ──────────────────────────────────────────────────────────────────────────
# Port–Hamiltonian Layer-1 damping
# ──────────────────────────────────────────────────────────────────────────
class PHLayer1:
    def __init__(self, veh: VehicleConfig, dt: float, engage_time: float, latency_s: float):
        self.veh = veh
        self.dt = dt
        self.engage_time = engage_time

        J = veh.inertia_B
        m = veh.mass_kg
        self.K_omega = (1.0 / max(veh.tau_omega_s, 1e-6)) * J
        self.K_v = (m / max(veh.tau_v_s, 1e-6)) * np.eye(3)
        self.K_eta = 1e-2 * np.eye(6)

        self.ctrlTorqueMsg = messaging.CmdTorqueBodyMsg()
        self.ctrlForceMsg = messaging.CmdForceBodyMsg()

        # Command latency buffer (steps)
        self.L = int(round(max(0.0, latency_s) / dt))
        self.buf_tau: List[np.ndarray] = []
        self.buf_for: List[np.ndarray] = []

        self.B, self.Ts, self.Isps = self._build_wrench_map(veh.thrusters)
        self.delta_prev: Optional[np.ndarray] = None

        # Previous kinematics for midpoint rule
        self.omega_prev = np.zeros(3)
        self.v_prev = np.zeros(3)

        # Energy monitors
        self._Hd_prev: Optional[float] = None
        self.energy_total = 0
        self.energy_noninc = 0
        self.energy_violations: List[Tuple[float, float]] = []

        # Accounting
        self.dv_cum = 0.0
        self.mp_cum = 0.0

        # High-rate logs
        self.t_list: List[float] = []
        self.Hd_list: List[float] = []
        self.tau_d_hist: List[np.ndarray] = []
        self.f_d_hist: List[np.ndarray] = []
        self.tau_rw_hist: List[np.ndarray] = []
        self.tau_thr_hist: List[np.ndarray] = []
        self.torque_cmd_hist: List[np.ndarray] = []
        self.force_cmd_hist: List[np.ndarray] = []
        self.torque_apply_hist: List[np.ndarray] = []
        self.force_apply_hist: List[np.ndarray] = []
        self.delta_hist: List[np.ndarray] = []
        self.thrust_total_hist: List[float] = []
        self.omega_hist: List[np.ndarray] = []
        self.vB_hist: List[np.ndarray] = []
        self.sigma_hist: List[np.ndarray] = []
        self.rN_hist: List[np.ndarray] = []  # << inertial position history

    def _build_wrench_map(self, thr: List[Thruster]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not thr:
            return np.zeros((6, 0)), np.zeros(0), np.zeros(0)
        cols = []
        Ts = []
        Isps = []
        for th in thr:
            r = th.r_B.reshape(3)
            a = unit(th.a_B.reshape(3))
            cols.append(np.hstack((np.cross(r, th.T * a), th.T * a)))
            Ts.append(th.T)
            Isps.append(th.Isp)
        return np.column_stack(cols), np.array(Ts, float), np.array(Isps, float)

    def _bnls(self, B, u, delta_prev, w_tau, w_for, lam, lb, ub):
        """Bounded, regularized weighted least squares for thruster allocations."""
        n = B.shape[1]
        if delta_prev is None or delta_prev.shape[0] != n:
            delta_prev = np.zeros(n)
        W = np.diag([w_tau, w_tau, w_tau, w_for, w_for, w_for])
        BtWB = B.T @ W @ B + lam * np.eye(n)
        BtWu = B.T @ W @ u + lam * delta_prev
        delta = np.clip(la.solve(BtWB, BtWu), lb, ub)

        # Simple clipping-aware refinement
        for _ in range(8):
            low = (delta <= lb + 1e-9)
            high = (delta >= ub - 1e-9)
            free = ~(low | high)
            if not np.any(free):
                break
            Bf = B[:, free]
            K = la.pinv(Bf.T @ W @ Bf + lam * np.eye(np.count_nonzero(free)))
            rhs = Bf.T @ W @ (u
                              - B[:, low] @ np.full(np.count_nonzero(low), lb)
                              - B[:, high] @ np.full(np.count_nonzero(high), ub)) + lam * delta_prev[free]
            delta_f = K @ rhs
            delta = delta.copy()
            delta[free] = delta_f
            delta = np.clip(delta, lb, ub)
        return delta

    def step(self, t: float,
             attNav: messaging.NavAttMsgPayload,
             transNav: messaging.NavTransMsgPayload):
        dt = self.dt
        J = self.veh.inertia_B
        m = self.veh.mass_kg

        sigma = np.array(attNav.sigma_BN)
        C_BN = mrp_to_dcm(sigma)
        omega = np.array(attNav.omega_BN_B)

        r_N = np.array(transNav.r_BN_N)     # inertial position
        v_N = np.array(transNav.v_BN_N)     # inertial velocity
        v_B = C_BN @ v_N                    # body velocity

        # Layer-1: pure damping (no potential shaping)
        phi = np.zeros(3)
        rho = np.zeros(3)
        eta = np.hstack((phi, rho))

        # Midpoint (Tustin) damping torques/forces
        omega_bar = 0.5 * (omega + self.omega_prev)
        v_bar = 0.5 * (v_B + self.v_prev)

        tau_d = - self.K_omega @ omega_bar
        f_d = - self.K_v @ v_bar

        # RW authority
        tau_rw = tau_d.copy()
        tn = la.norm(tau_rw)
        if tn > self.veh.rw_tau_max > 0:
            tau_rw *= (self.veh.rw_tau_max / tn)

        tau_res = tau_d - tau_rw

        tau_thr = np.zeros(3)
        f_thr = np.zeros(3)
        delta = np.zeros(0)
        if self.B.size > 0:
            delta = self._bnls(self.B, np.hstack((tau_res, f_d)), self.delta_prev,
                               self.veh.w_torque, self.veh.w_force, self.veh.reg_lambda, 0.0, 1.0)
            self.delta_prev = delta.copy()
            uf = self.B @ delta
            tau_thr = uf[:3]
            f_thr = uf[3:]

        tau_real = tau_rw + tau_thr
        f_real = f_thr

        # Logs (desired/commanded)
        self.tau_d_hist.append(tau_d.copy())
        self.f_d_hist.append(f_d.copy())
        self.tau_rw_hist.append(tau_rw.copy())
        self.tau_thr_hist.append(tau_thr.copy())
        self.torque_cmd_hist.append(tau_real.copy())
        self.force_cmd_hist.append(f_real.copy())
        self.delta_hist.append(delta.copy() if delta.size else np.zeros(0))
        self.omega_hist.append(omega.copy())
        self.vB_hist.append(v_B.copy())
        self.sigma_hist.append(sigma.copy())
        self.rN_hist.append(r_N.copy())
        self.thrust_total_hist.append(float(np.sum(self.Ts * delta)) if self.Ts.size else 0.0)

        # Apply after latency buffer
        self.buf_tau.append(tau_real)
        self.buf_for.append(f_real)
        if len(self.buf_tau) > self.L:
            tau_apply = self.buf_tau.pop(0)
            f_apply = self.buf_for.pop(0)
        else:
            tau_apply = np.zeros(3)
            f_apply = np.zeros(3)

        self.torque_apply_hist.append(tau_apply.copy())
        self.force_apply_hist.append(f_apply.copy())

        # Publish
        t_pl = messaging.CmdTorqueBodyMsgPayload()
        t_pl.torqueRequestBody = list(tau_apply)
        f_pl = messaging.CmdForceBodyMsgPayload()
        f_pl.forceRequestBody = list(f_apply)
        self.ctrlTorqueMsg.write(t_pl)
        self.ctrlForceMsg.write(f_pl)

        # Energy bookkeeping (start checking after engage_time)
        Hd = 0.5 * omega.T @ (J @ omega) + 0.5 * m * (v_B @ v_B) + 0.5 * eta.T @ (self.K_eta @ eta)
        if t >= self.engage_time:
            self.energy_total += 1
            if self._Hd_prev is not None:
                if Hd <= self._Hd_prev + 1e-12:
                    self.energy_noninc += 1
                else:
                    self.energy_violations.append((float(t), float(Hd - self._Hd_prev)))
            self._Hd_prev = Hd

        self.t_list.append(t)
        self.Hd_list.append(Hd)

        # Δv / prop accounting
        if delta.size > 0:
            impulse = float(np.sum(self.Ts * delta) * dt)
            self.dv_cum += impulse / m
            if impulse > 0 and np.any(self.Ts * delta > 0):
                self.mp_cum += impulse / (np.average(self.Isps, weights=self.Ts * delta) * 9.80665)

        # Update previous states
        self.omega_prev = omega.copy()
        self.v_prev = v_B.copy()


# ──────────────────────────────────────────────────────────────────────────
# RAFTI separation driver (equal-and-opposite)
# ──────────────────────────────────────────────────────────────────────────
@dc.dataclass
class SepParams:
    t_start: float
    t_end: float
    F0: float
    s_car_B: np.ndarray
    r_car_B: np.ndarray
    r_dep_B: np.ndarray
    dv_cmd: float  # for checks/telemetry

class RAFTISeparation:
    """Half-sine push forces at each RAFTI mount (equal-and-opposite)."""

    def __init__(self, dt: float,
                 sc_car: spacecraft.Spacecraft, eff_car: extForceTorque.ExtForceTorque,
                 sc_dep: List[spacecraft.Spacecraft], eff_dep: List[extForceTorque.ExtForceTorque],
                 sched: List[SepParams],
                 m_car: float, m_dep: float):
        self.dt = dt
        self.sched = sched
        self.sc_car = sc_car
        self.eff_car = eff_car
        self.sc_dep = sc_dep
        self.eff_dep = eff_dep
        self.m_car = m_car
        self.m_dep = m_dep

        # Messages
        self.car_tauMsg = messaging.CmdTorqueBodyMsg()
        self.car_forMsg = messaging.CmdForceBodyMsg()
        self.dep_tauMsg = [messaging.CmdTorqueBodyMsg() for _ in sc_dep]
        self.dep_forMsg = [messaging.CmdForceBodyMsg() for _ in sc_dep]

        eff_car.cmdTorqueInMsg.subscribeTo(self.car_tauMsg)
        eff_car.cmdForceBodyInMsg.subscribeTo(self.car_forMsg)
        for ef, tm, fm in zip(eff_dep, self.dep_tauMsg, self.dep_forMsg):
            ef.cmdTorqueInMsg.subscribeTo(tm)
            ef.cmdForceBodyInMsg.subscribeTo(fm)

    @staticmethod
    def _half_sine(t, t0, T, F0):
        if t < t0 or t > t0 + T:
            return 0.0
        return F0 * math.sin(math.pi * (t - t0) / T)

    def step(self, t: float):
        # Carrier transform (once per step)
        C_BN_car = mrp_to_dcm(np.array(self.sc_car.scStateOutMsg.read().sigma_BN))
        C_NB_car = C_BN_car.T

        tau_car = np.zeros(3)
        f_car = np.zeros(3)

        for i, sp in enumerate(self.sched):
            # deputy i transforms
            C_BN_dep = mrp_to_dcm(np.array(self.sc_dep[i].scStateOutMsg.read().sigma_BN))
            C_NB_dep = C_BN_dep.T

            # Peak enforcement check near t_start
            if abs(t - sp.t_start) < 0.5 * self.dt:
                m_eq = meq(self.m_car, self.m_dep)
                if m_eq > RAFTI.EQMASS_MAX_KG:
                    print(f"[RAFTI WARN dep{i+1}] m_eq={m_eq:.1f} kg exceeds {RAFTI.EQMASS_MAX_KG:.0f} kg rating")
                if sp.dv_cmd > RAFTI.DOCK_VEL_MAX_MPS:
                    print(f"[RAFTI WARN dep{i+1}] Δv_cmd={sp.dv_cmd:.3f} m/s exceeds RAFTI 25 mm/s rating")
                if sp.F0 > RAFTI.CONTACT_LIMIT_N:
                    print(f"[RAFTI WARN dep{i+1}] peak push {sp.F0:.0f} N > contact load {RAFTI.CONTACT_LIMIT_N:.0f} N")

            # Force magnitude (half-sine profile)
            Fmag = self._half_sine(t, sp.t_start, sp.t_end - sp.t_start, sp.F0)
            if Fmag <= 0:
                # publish zeros for deputy i
                tm = messaging.CmdTorqueBodyMsgPayload(); tm.torqueRequestBody = [0.0, 0.0, 0.0]
                fm = messaging.CmdForceBodyMsgPayload();  fm.forceRequestBody  = [0.0, 0.0, 0.0]
                self.dep_tauMsg[i].write(tm); self.dep_forMsg[i].write(fm)
                continue

            # Contact normal in inertial (via carrier body)
            s_N = C_NB_car @ sp.s_car_B
            F_N = Fmag * s_N

            # Equal-and-opposite in body frames
            f_car_i_B = C_BN_car @ (+F_N)
            f_dep_i_B = C_BN_dep @ (-F_N)
            tau_car_i = np.cross(sp.r_car_B, f_car_i_B)
            tau_dep_i = np.cross(sp.r_dep_B, f_dep_i_B)

            # Deputy publish
            tm = messaging.CmdTorqueBodyMsgPayload(); tm.torqueRequestBody = list(tau_dep_i)
            fm = messaging.CmdForceBodyMsgPayload();  fm.forceRequestBody  = list(f_dep_i_B)
            self.dep_tauMsg[i].write(tm); self.dep_forMsg[i].write(fm)

            # Accumulate carrier wrench
            tau_car += tau_car_i
            f_car += f_car_i_B

        # Carrier publish (superposed)
        tm = messaging.CmdTorqueBodyMsgPayload(); tm.torqueRequestBody = list(tau_car)
        fm = messaging.CmdForceBodyMsgPayload();  fm.forceRequestBody  = list(f_car)
        self.car_tauMsg.write(tm); self.car_forMsg.write(fm)


# ──────────────────────────────────────────────────────────────────────────
# Fleet assembly & run
# ──────────────────────────────────────────────────────────────────────────
class Fleet:
    def __init__(self):
        self.dt = 0.02
        self.sim = SimConfig(dt=self.dt, T_end=240.0, latency_s=0.05, log_dir=Path("layer1_logs"))

        rng = np.random.default_rng(42)

        # Carrier config (MR-22N illustrative)
        mr22_T, mr22_Isp = 22.0, 235.0
        thr_c = [
            Thruster(np.array([+0.8, 0.0, +0.5]), np.array([0.0, -1.0, 0.0]), mr22_T, mr22_Isp, "MR22_1"),
            Thruster(np.array([-0.8, 0.0, +0.5]), np.array([0.0, -1.0, 0.0]), mr22_T, mr22_Isp, "MR22_2"),
            Thruster(np.array([+0.8, 0.0, -0.5]), np.array([0.0, -1.0, 0.0]), mr22_T, mr22_Isp, "MR22_3"),
            Thruster(np.array([-0.8, 0.0, -0.5]), np.array([0.0, -1.0, 0.0]), mr22_T, mr22_Isp, "MR22_4"),
        ]
        I_car = np.diag([1200., 1000., 800.])
        m_car = 650.0
        self.car = VehicleConfig(
            name="carrier", mass_kg=m_car, inertia_B=I_car,
            sigma0_BN=np.zeros(3), omega0_B=np.zeros(3),
            r0_N=np.zeros(3), v0_N=np.zeros(3),
            tau_omega_s=60.0, tau_v_s=40.0, rw_tau_max=0.020,
            w_force=100.0, w_torque=1.0, reg_lambda=1e-4,
            thrusters=thr_c
        )

        # Deputies (MPS-120 illustrative)
        mps_T, mps_Isp = 0.27, 217.0
        thr_d = [
            Thruster(np.array([+0.05, 0.0, +0.05]), np.array([+0.05, -0.998, +0.05]), mps_T, mps_Isp, "MPS_1"),
            Thruster(np.array([-0.05, 0.0, +0.05]), np.array([-0.05, -0.998, +0.05]), mps_T, mps_Isp, "MPS_2"),
            Thruster(np.array([+0.05, 0.0, -0.05]), np.array([+0.05, -0.998, -0.05]), mps_T, mps_Isp, "MPS_3"),
            Thruster(np.array([-0.05, 0.0, -0.05]), np.array([-0.05, -0.998, -0.05]), mps_T, mps_Isp, "MPS_4"),
        ]
        I_dep = np.diag([0.025, 0.020, 0.018])
        m_dep = 12.0

        self.deps: List[VehicleConfig] = []
        self.mount_dirs_car_B: List[np.ndarray] = []  # outward normals at each mount (carrier body)
        self.mount_pos_car_B: List[np.ndarray] = []   # lever arms on carrier body
        self.mount_R_BD_BC: List[np.ndarray] = []     # deputy body DCSY wrt carrier body at mount

        ring_R = 1.00
        z_mount = 0.0
        for i in range(8):
            ang = i * (2 * math.pi / 8.0)
            s_car_B = np.array([math.cos(ang), math.sin(ang), 0.0], float)  # outward normal
            r_car_B = np.array([ring_R * math.cos(ang), ring_R * math.sin(ang), z_mount], float)
            # Deputy body axes: bY along s_car_B (points outward), bZ ~ +Z_car, bX = bY×bZ
            bY = s_car_B
            bZ = np.array([0, 0, 1.0], float)
            if abs(np.dot(unit(bY), bZ)) > 0.99:
                bZ = np.array([1, 0, 0], float)
            bX = unit(np.cross(bY, bZ))
            bZ = unit(np.cross(bX, bY))
            R_BD_BC = np.column_stack((bX, bY, bZ))  # carrier->deputy body DCM

            sigma0 = dcm_to_mrp(R_BD_BC)
            r0 = r_car_B.copy()
            v0 = np.zeros(3)

            dep = VehicleConfig(
                name=f"dep{i+1}", mass_kg=m_dep, inertia_B=I_dep,
                sigma0_BN=sigma0, omega0_B=np.zeros(3), r0_N=r0, v0_N=v0,
                tau_omega_s=25.0, tau_v_s=30.0, rw_tau_max=0.020,
                w_force=200.0, w_torque=0.5, reg_lambda=2e-4,
                thrusters=thr_d
            )
            self.deps.append(dep)
            self.mount_dirs_car_B.append(s_car_B)
            self.mount_pos_car_B.append(r_car_B)
            self.mount_R_BD_BC.append(R_BD_BC)

        # Build tasks (50 Hz)
        dt = self.dt
        self.dyn = sys_model_task.SysModelTask(int(macros.sec2nano(dt)))
        self.sns = sys_model_task.SysModelTask(int(macros.sec2nano(dt)))

        # Carrier spacecraft + nav + two effectors (control + separation)
        self.sc_car = spacecraft.Spacecraft(); self.sc_car.ModelTag = "SC_carrier"
        self.sc_car.hub.mHub = self.car.mass_kg
        self.sc_car.hub.IHubPntBc_B = self.car.inertia_B.tolist()
        self.sc_car.hub.sigma_BNInit = list(self.car.sigma0_BN)
        self.sc_car.hub.omega_BN_BInit = list(self.car.omega0_B)
        self.sc_car.hub.r_CN_NInit = list(self.car.r0_N)
        self.sc_car.hub.v_CN_NInit = list(self.car.v0_N)
        self.nav_car = simpleNav.SimpleNav(); self.nav_car.ModelTag = "NAV_carrier"
        self.nav_car.scStateInMsg.subscribeTo(self.sc_car.scStateOutMsg)
        self.eft_car_ctrl = extForceTorque.ExtForceTorque(); self.eft_car_ctrl.ModelTag = "EFT_car_ctrl"
        self.eft_car_sep  = extForceTorque.ExtForceTorque(); self.eft_car_sep .ModelTag = "EFT_car_sep"
        self.sc_car.addDynamicEffector(self.eft_car_ctrl)
        self.sc_car.addDynamicEffector(self.eft_car_sep)

        # Deputies
        self.sc_dep: List[spacecraft.Spacecraft] = []
        self.nav_dep: List[simpleNav.SimpleNav] = []
        self.eft_dep_ctrl: List[extForceTorque.ExtForceTorque] = []
        self.eft_dep_sep:  List[extForceTorque.ExtForceTorque] = []
        for dep in self.deps:
            sc = spacecraft.Spacecraft(); sc.ModelTag = f"SC_{dep.name}"
            sc.hub.mHub = dep.mass_kg; sc.hub.IHubPntBc_B = dep.inertia_B.tolist()
            sc.hub.sigma_BNInit = list(dep.sigma0_BN); sc.hub.omega_BN_BInit = list(dep.omega0_B)
            sc.hub.r_CN_NInit   = list(dep.r0_N);      sc.hub.v_CN_NInit   = list(dep.v0_N)
            nav = simpleNav.SimpleNav(); nav.ModelTag = f"NAV_{dep.name}"
            nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
            ec = extForceTorque.ExtForceTorque(); ec.ModelTag = f"EFT_{dep.name}_ctrl"
            es = extForceTorque.ExtForceTorque(); es.ModelTag = f"EFT_{dep.name}_sep"
            sc.addDynamicEffector(ec); sc.addDynamicEffector(es)
            self.sc_dep.append(sc); self.nav_dep.append(nav)
            self.eft_dep_ctrl.append(ec); self.eft_dep_sep.append(es)

        # Add to tasks
        p = 0
        self.sns.AddNewObject(self.nav_car, 10 + p)
        self.dyn.AddNewObject(self.sc_car,   5 + p)
        self.dyn.AddNewObject(self.eft_car_ctrl, 10 + p)
        self.dyn.AddNewObject(self.eft_car_sep,  11 + p)
        for i, (sc, nav, ec, es) in enumerate(zip(self.sc_dep, self.nav_dep, self.eft_dep_ctrl, self.eft_dep_sep), 1):
            self.sns.AddNewObject(nav, 10 + p + 10 * i)
            self.dyn.AddNewObject(sc,  5 + p + 10 * i)
            self.dyn.AddNewObject(ec, 10 + p + 10 * i)
            self.dyn.AddNewObject(es, 11 + p + 10 * i)

        # Controllers (start energy checks only after the deputy finishes separation)
        last_sep_end = 10.0 + (len(self.deps) - 1) * 5.0 + 0.30
        self.ctrl_car = PHLayer1(self.car, dt, engage_time=last_sep_end, latency_s=self.sim.latency_s)
        self.ctrl_dep: List[PHLayer1] = []
        for k, dep in enumerate(self.deps):
            t_end = 10.0 + k * 5.0 + 0.30
            self.ctrl_dep.append(PHLayer1(dep, dt, engage_time=t_end, latency_s=self.sim.latency_s))

        # Wire control ports
        self.eft_car_ctrl.cmdTorqueInMsg.subscribeTo(self.ctrl_car.ctrlTorqueMsg)
        self.eft_car_ctrl.cmdForceBodyInMsg.subscribeTo(self.ctrl_car.ctrlForceMsg)
        for ctl, ec in zip(self.ctrl_dep, self.eft_dep_ctrl):
            ec.cmdTorqueInMsg.subscribeTo(ctl.ctrlTorqueMsg)
            ec.cmdForceBodyInMsg.subscribeTo(ctl.ctrlForceMsg)

        # Separation schedule with per-deputy randomness consistent with RAFTI rating
        # Δv ~ N(0.018, 0.005^2) m/s (cap at 0.025 m/s); tilt σ≈2°, lever σ≈2 cm
        sched: List[SepParams] = []
        t0_base = 10.0; spacing = 5.0; Tpush = 0.30
        for k in range(len(self.deps)):
            t0 = t0_base + k * spacing
            dv = float(np.clip(np.random.default_rng(42 + k).normal(0.018, 0.005),
                               0.004, RAFTI.DOCK_VEL_MAX_MPS))
            F0 = self.deps[k].mass_kg * dv * math.pi / (2.0 * Tpush)
            s_car_B = tilt_vec(self.mount_dirs_car_B[k], 2.0, rng)
            r_car_B = self.mount_pos_car_B[k] + rng.normal(scale=0.02, size=3)
            r_dep_B = rng.normal(scale=0.02, size=3)
            sched.append(SepParams(t_start=t0, t_end=t0 + Tpush, F0=F0,
                                   s_car_B=s_car_B, r_car_B=r_car_B, r_dep_B=r_dep_B,
                                   dv_cmd=dv))

        self.sep = RAFTISeparation(self.dt, self.sc_car, self.eft_car_sep,
                                   self.sc_dep, self.eft_dep_sep, sched,
                                   m_car=self.car.mass_kg, m_dep=self.deps[0].mass_kg)

        # Finalize tasks
        for tsk in (self.sns, self.dyn):
            tsk.SelfInitTaskList()
            tsk.ResetTaskList(0)

    # Utilities for logging/plotting
    @staticmethod
    def _np(x):
        return np.array(x) if not isinstance(x, np.ndarray) else x

    def _dump_timeseries(self, prefix: str, ctl: PHLayer1):
        """Save rich timeseries without altering existing log schema (npz only)."""
        out = {
            "t": self._np(ctl.t_list),
            "Hd": self._np(ctl.Hd_list),
            "tau_d": self._np(ctl.tau_d_hist),
            "f_d": self._np(ctl.f_d_hist),
            "tau_cmd": self._np(ctl.torque_cmd_hist),
            "f_cmd": self._np(ctl.force_cmd_hist),
            "tau_apply": self._np(ctl.torque_apply_hist),
            "f_apply": self._np(ctl.force_apply_hist),
            "delta": self._np(ctl.delta_hist),
            "thrust_total": self._np(ctl.thrust_total_hist),
            "omega": self._np(ctl.omega_hist),
            "vB": self._np(ctl.vB_hist),
            "sigma": self._np(ctl.sigma_hist),
            "rN": self._np(ctl.rN_hist),
            "engage_time": np.array([ctl.engage_time]),
            "energy_violations": self._np(ctl.energy_violations) if len(ctl.energy_violations) else np.zeros((0, 2))
        }
        np.savez_compressed(self.sim.log_dir / f"{prefix}_layer1_timeseries.npz", **out)

    def _controller_report(self, prefix: str, veh: VehicleConfig, ctl: PHLayer1):
        def diag(M):
            M = np.asarray(M)
            return np.diag(M) if M.ndim == 2 else M

        Kω = diag(ctl.K_omega); Kv = diag(ctl.K_v)
        tau_d = self._np(ctl.tau_d_hist); f_d = self._np(ctl.f_d_hist)
        tau_c = self._np(ctl.torque_cmd_hist); f_c = self._np(ctl.force_cmd_hist)
        tau_a = self._np(ctl.torque_apply_hist); f_a = self._np(ctl.force_apply_hist)

        eps = 1e-12
        def rel_err(a, b):
            if a.size == 0:
                return np.array([0.0])
            num = la.norm(a - b, axis=1)
            den = np.maximum(la.norm(a, axis=1), eps)
            return num / den

        re_t_cmd = rel_err(tau_d, tau_c)
        re_t_app = rel_err(tau_d, tau_a)
        re_f_cmd = rel_err(f_d, f_c)
        re_f_app = rel_err(f_d, f_a)

        def stats(x):
            return np.nanmean(x), np.nanpercentile(x, 95), np.nanmax(x)

        mtc, p95tc, maxtc = stats(re_t_cmd)
        mta, p95ta, maxta = stats(re_t_app)
        mfc, p95fc, maxfc = stats(re_f_cmd)
        mfa, p95fa, maxfa = stats(re_f_app)

        # monotonicity
        vio_count = len(ctl.energy_violations)
        first_vio = f"{ctl.energy_violations[0][0]:.3f}s, ΔH={ctl.energy_violations[0][1]:.3e}" if vio_count else "None"
        # maxima
        tnorm_cmd = float(np.max(la.norm(tau_c, axis=1))) if tau_c.size else 0.0
        tnorm_app = float(np.max(la.norm(tau_a, axis=1))) if tau_a.size else 0.0
        fnorm_cmd = float(np.max(la.norm(f_c, axis=1))) if f_c.size else 0.0
        fnorm_app = float(np.max(la.norm(f_a, axis=1))) if f_a.size else 0.0
        thrust_pk = float(np.max(self._np(ctl.thrust_total_hist))) if len(ctl.thrust_total_hist) else 0.0

        print(f"\n[{prefix}] Controller info")
        print(f"  τ_ω={veh.tau_omega_s:.2f}s → diag(K_ω)={Kω}")
        print(f"  τ_v={veh.tau_v_s:.2f}s → diag(K_v)={Kv}")
        print(f"  K_eta=1e-2 I6, RW τ_max={veh.rw_tau_max:.3f} N·m, latency={ctl.L} steps ≃{ctl.L * ctl.dt:.3f}s")
        print(f"  BNLS: wτ={veh.w_torque:.3g}, wf={veh.w_force:.3g}, λ={veh.reg_lambda:.2e}, B.shape={ctl.B.shape}")
        if len(veh.thrusters):
            thr_info = ", ".join([f"{th.name}:{th.T:.2f}N@Isp{th.Isp:.0f}" for th in veh.thrusters])
            print(f"  Thrusters ({len(veh.thrusters)}): {thr_info}")
        print(f"  Δv_accum={ctl.dv_cum:.4f} m/s, m_p={ctl.mp_cum*1e3:.3f} g")
        print(f"  Max |τ| cmd/appl = {tnorm_cmd:.3f}/{tnorm_app:.3f} N·m; Max |f| cmd/appl = {fnorm_cmd:.3f}/{fnorm_app:.3f} N; Peak ∑Tδ={thrust_pk:.3f} N")
        print(f"  Monotonicity after t≥{ctl.engage_time:.2f}s: non-increasing fraction "
              f"{ctl.energy_noninc / max(1, ctl.energy_total):.3f}; violations={vio_count}; first={first_vio}")
        print(f"  Rel. torque error (desired→cmd/appl) mean/95%/max = "
              f"{mtc:.3f}/{p95tc:.3f}/{maxtc:.3f}  |  {mta:.3f}/{p95ta:.3f}/{maxta:.3f}")
        print(f"  Rel. force  error (desired→cmd/appl) mean/95%/max = "
              f"{mfc:.3f}/{p95fc:.3f}/{maxfc:.3f}  |  {mfa:.3f}/{p95fa:.3f}/{maxfa:.3f}")

    def _make_plots(self, prefix: str, title: str, ctl: PHLayer1):
        if not HAS_MPL:
            print(f"[{prefix}] matplotlib not available; skipping plots.")
            return
        t = self._np(ctl.t_list)
        Hd = self._np(ctl.Hd_list)
        tau_d = self._np(ctl.tau_d_hist)
        tau_c = self._np(ctl.torque_cmd_hist)
        tau_a = self._np(ctl.torque_apply_hist)
        f_d = self._np(ctl.f_d_hist)
        f_c = self._np(ctl.force_cmd_hist)
        f_a = self._np(ctl.force_apply_hist)
        omg = self._np(ctl.omega_hist)
        sig = self._np(ctl.sigma_hist)
        thrust = self._np(ctl.thrust_total_hist)
        eps = 1e-12

        # Overview figure
        fig, axs = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
        axs[0].plot(t, Hd)
        axs[0].axvline(ctl.engage_time, linestyle='--', alpha=0.6)
        axs[0].set_ylabel("H_d")
        axs[0].set_title(f"{title} — pH energy & histories")

        def nrm(x): return la.norm(x, axis=1) if x.size else np.zeros_like(t)

        axs[1].plot(t, nrm(tau_d), label="|τ_d|")
        axs[1].plot(t, nrm(tau_c), label="|τ_cmd|")
        axs[1].plot(t, nrm(tau_a), label="|τ_appl|")
        axs[1].set_ylabel("Torque norm")
        axs[1].legend(loc="upper right", fontsize=8)

        axs[2].plot(t, nrm(f_d), label="|f_d|")
        axs[2].plot(t, nrm(f_c), label="|f_cmd|")
        axs[2].plot(t, nrm(f_a), label="|f_appl|")
        axs[2].set_ylabel("Force norm")
        axs[2].legend(loc="upper right", fontsize=8)

        axs[3].plot(t, thrust)
        axs[3].set_ylabel("∑T·δ  [N]")

        if omg.size:
            axs[4].plot(t, omg[:, 0], label="ω_x")
            axs[4].plot(t, omg[:, 1], label="ω_y")
            axs[4].plot(t, omg[:, 2], label="ω_z")
            axs[4].set_ylabel("ω [rad/s]")
            axs[4].legend(loc="upper right", fontsize=8)
        if sig.size:
            axs[5].plot(t, sig[:, 0], label="σ_1")
            axs[5].plot(t, sig[:, 1], label="σ_2")
            axs[5].plot(t, sig[:, 2], label="σ_3")
            axs[5].set_ylabel("MRP σ")
            axs[5].legend(loc="upper right", fontsize=8)

        axs[-1].set_xlabel("time [s]")
        fig.tight_layout()
        plt.show()

        # Error ratios
        def rel(a, b):
            if not a.size:
                return np.zeros_like(t)
            num = la.norm(a - b, axis=1)
            den = np.maximum(la.norm(a, axis=1), eps)
            return num / den

        eτ_cmd = rel(tau_d, tau_c); eτ_app = rel(tau_d, tau_a)
        ef_cmd = rel(f_d, f_c);     ef_app = rel(f_d, f_a)

        fig2, ax2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax2[0].plot(t, eτ_cmd, label="τ: desired→cmd")
        ax2[0].plot(t, eτ_app, label="τ: desired→appl")
        ax2[0].set_ylabel("Rel. torque error")
        ax2[0].legend(loc="upper right", fontsize=8)

        ax2[1].plot(t, ef_cmd, label="f: desired→cmd")
        ax2[1].plot(t, ef_app, label="f: desired→appl")
        ax2[1].set_ylabel("Rel. force error")
        ax2[1].set_xlabel("time [s]")
        ax2[1].legend(loc="upper right", fontsize=8)

        fig2.tight_layout()
        plt.show()

    def _fleet_overlay_omega(self):
        if not HAS_MPL:
            return
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for i, ctl in enumerate(self.ctrl_dep, 1):
            t = self._np(ctl.t_list)
            omg = self._np(ctl.omega_hist)
            if omg.size:
                ax.plot(t, la.norm(omg, axis=1), label=f"dep{i}")
        ax.set_title("Deputy ω norms")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("|ω| [rad/s]")
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        plt.show()

    def _plot_deputy_r_grid(self):
        """Display a grid of r entries (r_x, r_y, r_z vs time) for each deputy."""
        if not HAS_MPL:
            print("[r-grid] matplotlib not available; skipping plots.")
            return
        M = len(self.ctrl_dep)
        fig, axes = plt.subplots(M, 3, figsize=(12, 2.4 * M), sharex=True)
        if M == 1:
            axes = np.array([axes])  # shape (1,3)

        labels = [r"$r_x$ [m]", r"$r_y$ [m]", r"$r_z$ [m]"]
        for row, (ctl, dep_idx) in enumerate(zip(self.ctrl_dep, range(1, M + 1))):
            t = self._np(ctl.t_list)
            r = self._np(ctl.rN_hist)
            if r.size == 0:
                r = np.zeros((len(t), 3))
            for col in range(3):
                ax = axes[row, col]
                ax.plot(t, r[:, col])
                if row == 0:
                    ax.set_title(labels[col])
                if col == 0:
                    ax.set_ylabel(f"dep{dep_idx}")
            if row == M - 1:
                for col in range(3):
                    axes[row, col].set_xlabel("time [s]")

        fig.suptitle("Deputy inertial position r(t) entries", y=0.995)
        fig.tight_layout()
        plt.show()

    # Main run loop
    def run(self):
        now_ns = 0
        N = int(self.sim.T_end / self.dt)
        for k in tqdm(range(N), desc="Layer-1 ESPA+RAFTI+PH", ncols=80):
            t = k * self.dt
            # Sensors (nav)
            self.sns.ExecuteTaskList(now_ns)
            # Evaluate separation forces
            self.sep.step(t)
            # Controllers
            self.ctrl_car.step(t, self.nav_car.attOutMsg.read(), self.nav_car.transOutMsg.read())
            for ctl, nav in zip(self.ctrl_dep, self.nav_dep):
                ctl.step(t, nav.attOutMsg.read(), nav.transOutMsg.read())
            # Dynamics
            self.dyn.ExecuteTaskList(now_ns)
            now_ns += int(macros.sec2nano(self.dt))

        # Save summary logs
        self.sim.log_dir.mkdir(exist_ok=True)
        def dump(path: Path, ctl: PHLayer1):
            np.savez_compressed(path,
                t=np.array(ctl.t_list),
                Hd=np.array(ctl.Hd_list),
                dv_cum=np.array([ctl.dv_cum]),
                mp_cum=np.array([ctl.mp_cum]),
                energy_noninc_frac=np.array([ctl.energy_noninc / max(1, ctl.energy_total)]))
        dump(self.sim.log_dir / "carrier_layer1_logs.npz", self.ctrl_car)
        for i, ctl in enumerate(self.ctrl_dep, 1):
            dump(self.sim.log_dir / f"dep{i:02d}_layer1_logs.npz", ctl)

        # Console report
        frac = self.ctrl_car.energy_noninc / max(1, self.ctrl_car.energy_total)
        print(f"[carrier] H_d monotonic after t≥{self.ctrl_car.engage_time:.1f}s: {frac:.3f}, "
              f"Δv≃{self.ctrl_car.dv_cum:.4f} m/s, m_p={self.ctrl_car.mp_cum*1e3:.3f} g")
        for i, ctl in enumerate(self.ctrl_dep, 1):
            frac = ctl.energy_noninc / max(1, ctl.energy_total)
            print(f"[dep{i}]    H_d monotonic after t≥{ctl.engage_time:.1f}s: {frac:.3f}, "
                  f"Δv≃{ctl.dv_cum:.4f} m/s, m_p={ctl.mp_cum*1e3:.3f} g")

        # Per-vehicle reports, dumps, and plots (display only)
        self._controller_report("carrier", self.car, self.ctrl_car)
        self._dump_timeseries("carrier", self.ctrl_car)
        self._make_plots("carrier", "Carrier", self.ctrl_car)

        for i, (dep, ctl) in enumerate(zip(self.deps, self.ctrl_dep), 1):
            tag = f"dep{i:02d}"
            self._controller_report(tag, dep, ctl)
            self._dump_timeseries(tag, ctl)
            self._make_plots(tag, f"Deputy {i}", ctl)

        self._fleet_overlay_omega()
        self._plot_deputy_r_grid()

        print("[DONE] ESPA separation via RAFTI + Layer-1 damping. Logs:", self.sim.log_dir)


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────
def main():
    fleet = Fleet()
    fleet.run()

if __name__ == "__main__":
    main()

