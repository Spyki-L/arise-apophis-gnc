#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_scenario_fleet_attitude_los_snapshots.py

Purpose
-------
Run a Basilisk-based 6-DOF attitude simulation for a fleet of rendezvous
CubeSats driven by RHIG reference bundles, then print line-of-sight (LOS)
alignment metrics and display a small set of “snap-style” diagnostic plots for
each deputy. The intent is rapid visual inspection rather than log archival:
figures are shown interactively and not saved by default.

High-level behavior
-------------------
1. RHIG NPZ ingestion
   - Search `APOPHIS_FLEET_DIR` (or `.` if unset) for files matching:
         apophis_rendezvous_full_D*.npz
   - For each deputy NPZ:
       * Load time history of deputy and asteroid states (rD_N, vD_N, rA_N, vA_N).
       * Use `q_RN` as the rendezvous-frame attitude when available (preferred).
       * Otherwise, reconstruct the reference attitude via a TRIAD solution that
         uses the boresight `bRef_N` (or rA−rD) and the deputy velocity vD_N.
       * Provide time interpolation utilities so the FSW can query reference
         attitude and boresight at arbitrary absolute times.

2. Fleet setup in Basilisk
   - For each deputy:
       * Instantiate a `Spacecraft` hub with diagonal inertia and a small initial
         Y-axis attitude offset (configurable).
       * Initialize the truth attitude to the first RHIG frame, optionally
         multiplied by a fixed offset rotation.
       * Attach an `ExtForceTorque` module (torque only) and a `SimpleNav` filter
         with user-tunable covariance / random-walk settings.
       * Set up an attitude FSW chain:
           + `attTrackingError` to compute σ_BR and ω_BR.
           + `mrpFeedback` with configurable gains (K, P, Ki, integral limit).
           + Optional `lowPassFilterTorqueCommand` low-pass filter for torque.
       * Wire FSW outputs into the torque effector.

   - Three SysModelTask layers are used:
       * `sns`  — sensor / SimpleNav updates.
       * `fsw`  — attitude guidance + control.
       * `dyn`  — spacecraft dynamics + torque effectors.

3. Simulation loop
   - Determine the common absolute time window from all NPZ bundles.
   - Step from `t0_abs` to `t_end` with fixed `dt_dyn`:
       * For each deputy:
           + Query the RHIG bundle at current absolute time to get σ_RN(t) and
             desired angular rate ω_RN_N(t) (if quaternions are provided) or
             σ_RN(t) with zero rate (if TRIAD-based).
           + Write an `AttRefMsg` (σ_RN, ω_RN_N, dω_RN_N=0) into that deputy’s
             reference message.
       * Execute sensor, FSW, and dynamics task lists at the current simulation
         time.
   - Optional log saving to compressed NPZ (disabled by default; see `save_logs()`).

4. Post-run analysis and visualization
   For each deputy, using the recorded truth, torque, and (optionally) nav logs:

   a) LOS alignment and boresight detection
      - Build the relative rotation
            R(t) = C_RN(t)^T C_BN(t)
        so R ≈ I when the body tracks the rendezvous frame.
      - At the final time step:
          * Determine the desired LOS in N from the RHIG boresight history.
          * Automatically pick the body axis (±x_B, ±y_B, ±z_B) whose inertial
            direction best aligns with the desired LOS, unless `BORESIGHT_HINT`
            is set to explicitly choose +x/+y/+z or −x/−y/−z.
          * Compute the achieved LOS in N as C_NB(t_end) e_axis.
          * Print a summary table row containing:
                dep index, desired LOS_N, achieved LOS_N, final angle error [deg], chosen axis.

   b) Figure set (display-only, per deputy)
      - Figure 1: R = C_RN^T C_BN entries (3×3 grid)
          * Each subplot shows one element R_ij(t) vs time with dashed
            reference (1 on diagonal, 0 off-diagonal).
          * Visual check that R → I as tracking error goes to zero.

      - Figure 2: angular momenta and torques
          * Top: body-frame momentum p_B = J ω_BN_B components vs time.
          * Bottom: commanded torque components u_B vs time (from the controller).

      - Figure 3: relative position and velocity errors
          * Position error e_r = r_true − r_des components and norm ‖e_r‖.
          * Velocity error e_v = v_true − v_des components and norm ‖e_v‖.
          * Gives a quick sense of how well the deputy follows the RHIG
            translational guidance in this attitude-only control test.

   - After each deputy’s three figures are shown, they are explicitly closed to
     avoid Matplotlib’s “more than 20 figures open” warning.

Configuration knobs
-------------------
- `BORESIGHT_HINT`   : optional integer to force which body axis is treated as
                       the camera LOS (+0/+1/+2 for +x/+y/+z, −0/−1/−2 for
                       −x/−y/−z); `None` = auto-detect.
- `SimulationConfig` : dataclass with:
     * fleet_dir, npz_glob: where to find RHIG NPZ bundles.
     * dt_dyn, dt_sens    : task periods.
     * k_sigma, p_rate, ki, i_lim: mrpFeedback gains and integral limits.
     * use_lpf, lpf_cut   : torque low-pass filter toggle and cutoff frequency.
     * mass_kg, inertia   : spacecraft mass and diagonal inertia.
     * init_y_deg         : initial body Y-tilt angle [deg].
     * nav_sig            : SimpleNav noise levels for pos/vel/att/rate/sun/Δv.
     * log_nav, log_truth : whether to attach and record nav/truth messages.
     * log_dir            : optional NPZ output directory for `save_logs`.
     * t_stop             : optional manual stop time; defaults to RHIG coverage.
     * prefer_qrn         : whether to prefer RHIG quaternions over TRIAD.

Usage
-----
Ensure Basilisk is importable, set `APOPHIS_FLEET_DIR` to the directory
containing `apophis_rendezvous_full_D*.npz` (if not using the current working
directory), then run:

    python basilik_scenario_fleet_attitude_los_snapshots.py

The script will:
- Build and run the Basilisk fleet simulation.
- Print a compact LOS alignment summary per deputy.
- Display three interactive diagnostic figures per deputy, closing them before
  moving on to the next vehicle.
"""


from __future__ import annotations
import os, sys, math, time, dataclasses as dc
from pathlib import Path
from typing import Tuple, List
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# ── Basilisk imports ────────────────────────────────────────────────────────
from Basilisk.architecture import messaging, sys_model_task
from Basilisk.simulation   import spacecraft, simpleNav, extForceTorque
from Basilisk.fswAlgorithms import attTrackingError, mrpFeedback, lowPassFilterTorqueCommand
from Basilisk.utilities import macros
# ────────────────────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════════════════════╗
# 0)  USER HINT (optional)
# ╚══════════════════════════════════════════════════════════════════════════╝
# None  → auto-detect the body boresight axis that best aligns with desired LOS at the end
# 0/1/2 → +x/+y/+z   ;  -0/-1/-2 → -x/-y/-z
BORESIGHT_HINT = None


# ╔══════════════════════════════════════════════════════════════════════════╗
# 1)  CONFIGURATION
# ╚══════════════════════════════════════════════════════════════════════════╝
@dc.dataclass
class SimulationConfig:
    fleet_dir : Path = Path(os.getenv("APOPHIS_FLEET_DIR", "."))
    npz_glob  : str  = "apophis_rendezvous_full_D*.npz"

    # task periods [s]
    dt_dyn  : float = 0.1
    dt_sens : float = 0.1

    # controller
    k_sigma : float = 0.08
    p_rate  : float = 0.9
    ki      : float = -1.0
    i_lim   : float = 0.01

    # torque LPF
    use_lpf : bool  = True
    lpf_cut : float = 0.8        # [Hz]

    # spacecraft
    mass_kg : float = 250.0
    inertia : tuple[float,float,float] = (18., 15., 12.)
    init_y_deg : float = 5.0

    # SimpleNav noise (1-σ)
    nav_sig = dict(pos=50., vel=.05, att=5e-4, rate=2e-5, sun=1e-3, dv=0.)

    # logging flags
    log_nav   : bool = True
    log_truth : bool = True
    log_dir   : Path = Path("fleet_logs")

    # runtime
    t_stop    : float|None = None
    prefer_qrn: bool = True

CFG = SimulationConfig()


# ╔══════════════════════════════════════════════════════════════════════════╗
# 2)  NPZ  wrapper (rendezvous bundle)
# ╚══════════════════════════════════════════════════════════════════════════╝
class NPZBundle:
    def __init__(self, path: Path, prefer_qrn: bool):
        n = np.load(path)
        self.path = path
        self.t    = n["t"].astype(float)
        self.rD   = n["rD_N"];  self.vD = n["vD_N"]
        self.rA   = n["rA_N"];  self.vA = n["vA_N"]
        self.bRef = n["bRef_N"] if "bRef_N" in n else (
            (self.rA-self.rD) / la.norm(self.rA-self.rD, axis=1)[:,None]
        )
        self.use_q = ("q_RN" in n) and prefer_qrn
        self.qRN   = n["q_RN"] if self.use_q else None

    def interp(self, arr: np.ndarray, t_abs: float) -> np.ndarray:
        t = self.t
        if t_abs <= t[0]:  return arr[0]
        if t_abs >= t[-1]: return arr[-1]
        j = np.searchsorted(t, t_abs) - 1
        f = (t_abs - t[j])/(t[j+1] - t[j])
        return (1-f)*arr[j] + f*arr[j+1]


# ╔══════════════════════════════════════════════════════════════════════════╗
# 3)  Deputy (Basilisk modules for one spacecraft)
# ╚══════════════════════════════════════════════════════════════════════════╝
class Deputy:
    def __init__(self, idx: int, bundle: NPZBundle):
        self.i     = idx
        self.data  = bundle
        self.ref   = messaging.AttRefMsg()          # populated each step
        self.recs  = {}                             # recorder handles
        self._build_modules()

    def _build_modules(self):
        Ixx,Iyy,Izz = CFG.inertia
        theta = math.radians(CFG.init_y_deg)

        # small initial Y tilt
        R_off = np.array([[ math.cos(theta),0, math.sin(theta)],
                          [ 0,1,0 ],
                          [-math.sin(theta),0, math.cos(theta)]])

        # spacecraft hub
        self.sc = spacecraft.Spacecraft()
        self.sc.ModelTag = f"Dep{self.i}"
        self.sc.hub.mHub = CFG.mass_kg
        self.sc.hub.IHubPntBc_B = [[Ixx,0,0],[0,Iyy,0],[0,0,Izz]]

        # attitude init from desired
        if self.data.use_q:
            C_NB0 = quat_to_dcm(self.data.qRN[0]).T
        else:
            C_NB0 = triad_from_boresight_and_tangent(self.data.bRef[0], self.data.vD[0])
        C_NB0 = C_NB0 @ R_off
        sigma0 = quat_to_mrp(dcm_to_quat(C_NB0))
        self.sc.hub.sigma_BNInit = [[float(s)] for s in sigma0]
        self.sc.hub.r_CN_NInit   = list(self.data.rD[0])
        self.sc.hub.v_CN_NInit   = list(self.data.vD[0])

        # effector + nav
        self.torque = extForceTorque.ExtForceTorque(); self.torque.ModelTag=f"torque_{self.i}"
        self.nav = simpleNav.SimpleNav(); self.nav.ModelTag=f"nav_{self.i}"

        if CFG.nav_sig:
            sig = CFG.nav_sig
            walk = np.hstack([3*sig['pos']*np.ones(3),
                              3*sig['vel']*np.ones(3),
                              3*sig['att']*np.ones(3),
                              3*sig['rate']*np.ones(3),
                              3*sig['sun']*np.ones(3),
                              3*sig['dv'] *np.ones(3)])
            P = np.diag(np.hstack([sig['pos']*np.ones(3),
                                   sig['vel']*np.ones(3),
                                   sig['att']*np.ones(3),
                                   sig['rate']*np.ones(3),
                                   sig['sun']*np.ones(3),
                                   sig['dv'] *np.ones(3)])**2)**0.5
            self.nav.walkBounds = walk; self.nav.PMatrix = P

        # FSW
        self.err = attTrackingError.attTrackingError(); self.err.ModelTag=f"err_{self.i}"
        self.ctl = mrpFeedback.mrpFeedback();          self.ctl.ModelTag=f"ctl_{self.i}"
        self.ctl.K  = CFG.k_sigma; self.ctl.P = CFG.p_rate
        self.ctl.Ki = CFG.ki      ; self.ctl.integralLimit = CFG.i_lim

        self.lpf = None
        if CFG.use_lpf:
            self.lpf = lowPassFilterTorqueCommand.lowPassFilterTorqueCommand()
            self.lpf.ModelTag = f"lpf_{self.i}"
            self.lpf.h  = CFG.dt_dyn
            self.lpf.wc = 2*math.pi*CFG.lpf_cut

        # wiring
        self.sc.addDynamicEffector(self.torque)
        self.nav.scStateInMsg.subscribeTo(self.sc.scStateOutMsg)

    def ref_sigma_omega(self, t_abs: float, dt: float) -> Tuple[np.ndarray,np.ndarray]:
        """Desired attitude at absolute time t_abs as MRP + desired body rate."""
        if self.data.use_q:
            q_prev = self.data.interp(self.data.qRN, max(self.data.t[0], t_abs-dt))
            q_curr = self.data.interp(self.data.qRN, t_abs)
            q_next = self.data.interp(self.data.qRN, min(self.data.t[-1], t_abs+dt))
            Cdot   = (quat_to_dcm(q_next)-quat_to_dcm(q_prev))/max(2*dt,1e-6)
            S      = Cdot @ quat_to_dcm(q_curr).T
            omega  = -.5*np.array([S[2,1]-S[1,2], S[0,2]-S[2,0], S[1,0]-S[0,1]])
            sigma  = quat_to_mrp(q_curr); return sigma, omega
        rD = self.data.interp(self.data.rD, t_abs)
        vD = self.data.interp(self.data.vD, t_abs)
        bN = self.data.interp(self.data.bRef, t_abs)
        C  = triad_from_boresight_and_tangent(bN, vD)
        return quat_to_mrp(dcm_to_quat(C.T)), np.zeros(3)


# ╔══════════════════════════════════════════════════════════════════════════╗
# 4)  FleetSim (orchestrates tasks + run)
# ╚══════════════════════════════════════════════════════════════════════════╝
class FleetSim:
    def __init__(self, cfg: SimulationConfig):
        self.cfg    = cfg
        self.bundles= self._discover_npz()
        self.deps   = [Deputy(i,b) for i,b in enumerate(self.bundles)]
        self._build_tasks()

    def _discover_npz(self) -> List[NPZBundle]:
        paths = sorted(self.cfg.fleet_dir.glob(self.cfg.npz_glob))
        if not paths:
            print("[ERR] no NPZ matched"); sys.exit(1)
        return [NPZBundle(p, self.cfg.prefer_qrn) for p in paths]

    def _build_tasks(self):
        cfg = self.cfg
        self.dyn = sys_model_task.SysModelTask(int(macros.sec2nano(cfg.dt_dyn)))
        self.fsw = sys_model_task.SysModelTask(int(macros.sec2nano(cfg.dt_dyn)))
        self.sns = sys_model_task.SysModelTask(int(macros.sec2nano(cfg.dt_sens)))

        # static inertia message to controller
        Ixx, Iyy, Izz = self.cfg.inertia
        vehCfgPayload = messaging.VehicleConfigMsgPayload()
        vehCfgPayload.ISCPntB_B = [Ixx,0,0, 0,Iyy,0, 0,0,Izz]
        vehCfg = messaging.VehicleConfigMsg().write(vehCfgPayload)

        for d in self.deps:
            p = 100*d.i
            d.err.attNavInMsg .subscribeTo(d.nav.attOutMsg)
            d.err.attRefInMsg .subscribeTo(d.ref)
            d.ctl.guidInMsg   .subscribeTo(d.err.attGuidOutMsg)
            d.ctl.vehConfigInMsg.subscribeTo(vehCfg)

            if d.lpf:
                d.lpf.cmdTorqueInMsg.subscribeTo(d.ctl.cmdTorqueOutMsg)
                d.torque.cmdTorqueInMsg.subscribeTo(d.lpf.cmdTorqueOutMsg)
            else:
                d.torque.cmdTorqueInMsg.subscribeTo(d.ctl.cmdTorqueOutMsg)

            self.sns.AddNewObject(d.nav,  10+p)
            self.fsw.AddNewObject(d.err,  20+p); self.fsw.AddNewObject(d.ctl,10+p)
            if d.lpf: self.fsw.AddNewObject(d.lpf, 5+p)
            self.dyn.AddNewObject(d.torque,10+p); self.dyn.AddNewObject(d.sc,5+p)

            # recorders
            rec = d.err.attGuidOutMsg.recorder(); self.fsw.AddNewObject(rec,-1); d.recs["err"]=rec
            rec = d.ctl.cmdTorqueOutMsg.recorder(); self.fsw.AddNewObject(rec,-1); d.recs["torq"]=rec

            if cfg.log_nav:
                rec = d.nav.attOutMsg.recorder();  self.sns.AddNewObject(rec,-1); d.recs["navA"]=rec
                rec = d.nav.transOutMsg.recorder();self.sns.AddNewObject(rec,-1); d.recs["navT"]=rec
            if cfg.log_truth:
                rec = d.sc.scStateOutMsg.recorder();self.dyn.AddNewObject(rec,-1); d.recs["truth"]=rec

        for t in (self.sns,self.fsw,self.dyn):
            t.SelfInitTaskList(); t.ResetTaskList(0)

        # timing
        t0 = min(b.t[0] for b in self.bundles)
        t1 = max(b.t[-1] for b in self.bundles)
        self.t0_abs = t0
        self.t_end  = min(self.cfg.t_stop or t1, t1)

    def run(self):
        N  = int((self.t_end - self.t0_abs)/self.cfg.dt_dyn) + 1
        dt = self.cfg.dt_dyn
        try:
            from tqdm import tqdm
            bar = tqdm(range(N), ncols=80, desc="Sim")
        except Exception:
            bar = range(N)

        for k in bar:
            t_rel = k*dt; t_abs = self.t0_abs + t_rel
            now_ns = int(macros.sec2nano(t_rel))

            for d in self.deps:
                sigma_ref, omega_ref = d.ref_sigma_omega(t_abs, dt)
                pl = messaging.AttRefMsgPayload()
                pl.sigma_RN    = list(sigma_ref)
                pl.omega_RN_N  = list(omega_ref)
                pl.domega_RN_N = [0.0, 0.0, 0.0]
                d.ref.write(pl)

            self.sns.ExecuteTaskList(now_ns)
            self.fsw.ExecuteTaskList(now_ns)
            self.dyn.ExecuteTaskList(now_ns)

        print("[DONE] simulation")


# ╔══════════════════════════════════════════════════════════════════════════╗
# 5)  Analysis + Plots (display only) + LOS printout
# ╚══════════════════════════════════════════════════════════════════════════╝
def analyze_and_plot(sim: FleetSim):
    Ixx, Iyy, Izz = sim.cfg.inertia
    J = np.diag([Ixx, Iyy, Izz])

    def unit_vec(v, eps=1e-15):
        v = np.asarray(v, float).reshape(3)
        n = la.norm(v)
        return v/n if n >= eps else np.array([1.0,0.0,0.0])

    def pick_boresight_axis(C_RN_final: np.ndarray, bRef_final: np.ndarray, hint=None):
        """Choose body axis e_axis so that (C_NR * e_axis) ≈ desired LOS."""
        if hint is not None:
            k = abs(int(hint)); sgn = -1 if (isinstance(hint,int) and hint<0) else 1
            e = np.zeros(3); e[k] = sgn; return e, k, sgn
        C_NR = C_RN_final.T
        best = None
        for k in (0,1,2):
            for sgn in (+1,-1):
                e = np.zeros(3); e[k]=sgn
                v = C_NR @ e
                score = float(v @ unit_vec(bRef_final))
                if (best is None) or (abs(score) > abs(best[0])):
                    best = (score, k, sgn, e)
        _, k, sgn, e = best
        return e, k, sgn

    print("\nFinal LOS (desired vs achieved, each deputy)")
    print("dep |  desired_LOS_N (x y z)           |  achieved_LOS_N (x y z)          |  angle_err [deg]  | axis")

    for d in sim.deps:
        # truth
        t_true = d.recs["truth"].times()*macros.NANO2SEC
        r_true = np.asarray(getattr(d.recs["truth"], "r_BN_N")).reshape(-1,3)
        v_true = np.asarray(getattr(d.recs["truth"], "v_BN_N")).reshape(-1,3)
        sig_tr = np.asarray(getattr(d.recs["truth"], "sigma_BN")).reshape(-1,3)
        omg_tr = np.asarray(getattr(d.recs["truth"], "omega_BN_B")).reshape(-1,3)

        # torques
        t_tq   = d.recs["torq"].times()*macros.NANO2SEC
        u_B    = np.asarray(getattr(d.recs["torq"], "torqueRequestBody")).reshape(-1,3)

        # desired streams on t_true grid
        t_abs = sim.t0_abs + t_true
        r_des = np.vstack([d.data.interp(d.data.rD, t) for t in t_abs])
        v_des = np.vstack([d.data.interp(d.data.vD, t) for t in t_abs])

        if d.data.use_q:
            C_RN = np.stack([quat_to_dcm(d.data.interp(d.data.qRN, t)) for t in t_abs], axis=0)
            bRef = np.stack([unit_vec(d.data.interp(d.data.bRef, t)) for t in t_abs], axis=0)
        else:
            bRef = np.vstack([d.data.interp(d.data.bRef, t) for t in t_abs])
            C_RN = np.stack([triad_from_boresight_and_tangent(
                                bRef[k], d.data.interp(d.data.vD, t_abs[k]))
                             for k in range(len(t_abs))], axis=0)
            bRef = np.stack([unit_vec(bRef[k]) for k in range(len(t_abs))], axis=0)

        # achieved orientation, R, LOS
        C_BN = np.stack([mrp_to_dcm(sig_tr[k]) for k in range(len(t_true))], axis=0)
        R    = np.einsum('tij,tjk->tik', C_RN.transpose(0,2,1), C_BN)

        # choose boresight axis
        e_bore, axis_idx, sgn = pick_boresight_axis(C_RN[-1], bRef[-1], hint=BORESIGHT_HINT)

        # achieved LOS in N: C_NB * e_bore
        C_NB = C_BN.transpose(0,2,1)
        los_ach = np.einsum('tij,j->ti', C_NB, e_bore)
        los_des = bRef

        # final LOS print
        ldes_f, lach_f = unit_vec(los_des[-1]), unit_vec(los_ach[-1])
        ang = math.degrees(math.acos(np.clip(float(ldes_f @ lach_f), -1.0, 1.0)))
        axis_name = ('+x','+y','+z')[axis_idx] if sgn>0 else ('-x','-y','-z')[axis_idx]
        print(f"{d.i:3d} | [{ldes_f[0]: .4f} {ldes_f[1]: .4f} {ldes_f[2]: .4f}] |"
              f" [{lach_f[0]: .4f} {lach_f[1]: .4f} {lach_f[2]: .4f}] | {ang:8.3f}       | {axis_name}")

        # errors, momenta
        e_r  = r_true - r_des; e_v = v_true - v_des
        e_rn = la.norm(e_r, axis=1); e_vn = la.norm(e_v, axis=1)
        p_B  = (J @ omg_tr.T).T

        # ---- (1) R entries (3×3) ----
        figR, axs = plt.subplots(3,3, figsize=(11,8), sharex=True, sharey=True)
        figR.suptitle(f"Deputy {d.i} — R entries (R = C_RN^T C_BN)")
        for i in range(3):
            for j in range(3):
                axs[i,j].plot(t_true, R[:,i,j], lw=1.6)
                axs[i,j].axhline(1.0 if i==j else 0.0, ls='--', lw=0.9, alpha=0.7)
                axs[i,j].grid(True, ls=':', alpha=0.4)
        axs[2,1].set_xlabel("t [s]"); axs[1,0].set_ylabel("R entries")

        # ---- (2) Momenta & torques ----
        figM, (axP, axU) = plt.subplots(2,1, figsize=(10,6))
        figM.suptitle(f"Deputy {d.i} — Momenta and Torques")
        axP.plot(t_true, p_B[:,0], label='p_x'); axP.plot(t_true, p_B[:,1], label='p_y'); axP.plot(t_true, p_B[:,2], label='p_z')
        axP.set_ylabel("momenta [N m s]"); axP.grid(True, ls=':', alpha=0.4); axP.legend(ncol=3, fontsize=9)
        axU.plot(t_tq,   u_B[:,0], label='u_x'); axU.plot(t_tq,   u_B[:,1], label='u_y'); axU.plot(t_tq,   u_B[:,2], label='u_z')
        axU.set_ylabel("torques [N m]"); axU.set_xlabel("t [s]"); axU.grid(True, ls=':', alpha=0.4); axU.legend(ncol=3, fontsize=9)

        # ---- (3) Relative pos/vel errors ----
        figE, (axR, axV) = plt.subplots(2,1, figsize=(10,6), sharex=True)
        figE.suptitle(f"Deputy {d.i} — Relative errors to desired")
        axR.plot(t_true, e_r[:,0], label='e_rx'); axR.plot(t_true, e_r[:,1], label='e_ry'); axR.plot(t_true, e_r[:,2], label='e_rz')
        axR.plot(t_true, e_rn,     label='||e_r||', lw=2.0)
        axR.set_ylabel("pos error [m]"); axR.grid(True, ls=':', alpha=0.4); axR.legend(ncol=4, fontsize=9)
        axV.plot(t_true, e_v[:,0], label='e_vx'); axV.plot(t_true, e_v[:,1], label='e_vy'); axV.plot(t_true, e_v[:,2], label='e_vz')
        axV.plot(t_true, e_vn,     label='||e_v||', lw=2.0)
        axV.set_ylabel("vel error [m/s]"); axV.set_xlabel("t [s]"); axV.grid(True, ls=':', alpha=0.4); axV.legend(ncol=4, fontsize=9)

        # Show then close to avoid “>20 figures open”
        plt.show()
        plt.close(figR); plt.close(figM); plt.close(figE)


# ╔══════════════════════════════════════════════════════════════════════════╗
# 6)  MAIN
# ╚══════════════════════════════════════════════════════════════════════════╝
def main():
    sim = FleetSim(CFG)
    t0 = time.time(); sim.run(); print(f"[TIMING] {time.time()-t0:.1f} s")
    # You can enable saving logs if you want raw .npz traces:
    # if CFG.log_dir: save_logs(sim)
    analyze_and_plot(sim)


# ╔══════════════════════════════════════════════════════════════════════════╗
# 7)  Math & attitude helpers
# ╚══════════════════════════════════════════════════════════════════════════╝
def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    tr=np.trace(C)
    if tr>0:
        S=math.sqrt(tr+1)*2; q0=.25*S
        q1=(C[2,1]-C[1,2])/S; q2=(C[0,2]-C[2,0])/S; q3=(C[1,0]-C[0,1])/S
    else:
        i=int(np.argmax([C[0,0],C[1,1],C[2,2]]))
        if i==0:
            S=math.sqrt(1+C[0,0]-C[1,1]-C[2,2])*2
            q0=(C[2,1]-C[1,2])/S; q1=.25*S; q2=(C[0,1]+C[1,0])/S; q3=(C[0,2]+C[2,0])/S
        elif i==1:
            S=math.sqrt(1+C[1,1]-C[0,0]-C[2,2])*2
            q0=(C[0,2]-C[2,0])/S; q1=(C[0,1]+C[1,0])/S; q2=.25*S; q3=(C[1,2]+C[2,1])/S
        else:
            S=math.sqrt(1+C[2,2]-C[0,0]-C[1,1])*2
            q0=(C[1,0]-C[0,1])/S; q1=(C[0,2]+C[2,0])/S; q2=(C[1,2]+C[2,1])/S; q3=.25*S
    q=np.array([q0,q1,q2,q3]); return q/la.norm(q)

def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    q0,q1,q2,q3 = q; qv = np.array([q1,q2,q3], float)
    return (q0*q0 - np.dot(qv,qv))*np.eye(3) \
         + 2*np.outer(qv,qv) \
         + 2*q0*np.array([[ 0, -q3,  q2],
                          [ q3,  0, -q1],
                          [-q2,  q1,  0]])

def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    q0,qv=q[0],q[1:]
    return -qv/(1-q0) if 1+q0<1e-12 else qv/(1+q0)

def normalize(v,eps=1e-15):
    n=la.norm(v); 
    return (v/n,n) if n>=eps else (np.zeros_like(v),0.0)

def triad_from_boresight_and_tangent(los: np.ndarray, vel: np.ndarray) -> np.ndarray:
    z,_=normalize(los); v_perp=vel-np.dot(vel,z)*z
    x,nx=normalize(v_perp)
    if nx<1e-10:
        tmp=np.array([1,0,0]) if abs(z[0])<.9 else np.array([0,1,0])
        x=normalize(np.cross(tmp,z))[0]
    y=np.cross(z,x); x=normalize(np.cross(y,z))[0]; y=normalize(np.cross(z,x))[0]
    return np.column_stack((x,y,z))

def skew(v):
    x,y,z = v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]], float)

def mrp_to_dcm(sigma):
    s2=float(np.dot(sigma,sigma)); I3=np.eye(3)
    if s2<1e-16: return I3.copy()
    S=skew(sigma)
    return I3 + (8*S@S - 4*(1-s2)*S) / (1+s2)**2

# (optional) raw log saver
def save_logs(sim: FleetSim):
    cfg = sim.cfg; cfg.log_dir.mkdir(exist_ok=True)
    for d in sim.deps:
        fname = cfg.log_dir / f"dep{d.i:02d}_logs.npz"
        fields = dict(
            t_err = d.recs["err"].times()*macros.NANO2SEC,
            sigma_BR = np.array(getattr(d.recs["err"],"sigma_BR")),
            omega_BR = np.array(getattr(d.recs["err"],"omega_BR_B")),
            t_torque = d.recs["torq"].times()*macros.NANO2SEC,
            torque_B = np.array(getattr(d.recs["torq"],"torqueRequestBody"))
        )
        if cfg.log_nav:
            fields.update(
                t_navA = d.recs["navA"].times()*macros.NANO2SEC,
                sigma_nav = np.array(getattr(d.recs["navA"],"sigma_BN_B")),
                omega_nav = np.array(getattr(d.recs["navA"],"omega_BN_B")),
                t_navT = d.recs["navT"].times()*macros.NANO2SEC,
                r_nav = np.array(getattr(d.recs["navT"],"r_BN_N")),
                v_nav = np.array(getattr(d.recs["navT"],"v_BN_N")))
        if cfg.log_truth:
            fields.update(
                t_true = d.recs["truth"].times()*macros.NANO2SEC,
                r_true = np.array(getattr(d.recs["truth"],"r_BN_N")),
                v_true = np.array(getattr(d.recs["truth"],"v_BN_N")),
                sigma_true = np.array(getattr(d.recs["truth"],"sigma_BN")),
                omega_true = np.array(getattr(d.recs["truth"],"omega_BN_B")))
        np.savez_compressed(fname, **fields)
        print(f"[SAVE] {fname}")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
