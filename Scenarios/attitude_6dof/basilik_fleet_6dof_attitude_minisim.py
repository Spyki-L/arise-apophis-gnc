#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_fleet_6dof_attitude_minisim.py
======================================

Purpose
-------
Fleet-level **6-DOF attitude** simulation for multiple Apophis deputies, using
a clean, object-oriented architecture without `SimulationBaseClass`. It is a
systemic refactor of the original “fleet 6-DOF attitude” script, preserving
functionality while making the structure explicit:

    ┌── SimulationConfig   # global knobs (dt, gains, noise, logging, etc.)
    ├── NPZBundle          # wraps each rendezvous .npz file
    ├── Deputy             # one spacecraft worth of Basilisk modules
    └── FleetSim           # orchestrates deputies, tasks, and logging

Each deputy:

- Reads its reference trajectory and attitude from an Apophis rendezvous NPZ,
- Builds its own `spacecraft`, `simpleNav`, `attTrackingError`, `mrpFeedback`,
  and optional torque LPF chain,
- Tracks a time-varying attitude reference R(t) with rate ω_RN_N(t),
- Exports “fleet_logs/dep??_logs.npz” files for downstream plotting.

Inputs and NPZ discovery
------------------------
`SimulationConfig` controls where to find the fleet NPZs:

- `fleet_dir` : base directory (defaults to `$APOPHIS_FLEET_DIR` or `.`),
- `npz_glob`  : glob pattern (default `"apophis_rendezvous_full_D*.npz"`).

`FleetSim._discover_npz()` collects all matching files and wraps each in an
`NPZBundle`, which expects per-deputy NPZs containing at least:

- `t`       : (N,) time stamps [s] since a common epoch,
- `rD_N`    : (N,3) deputy ECI positions [m],
- `vD_N`    : (N,3) deputy ECI velocities [m/s],
- `rA_N`    : (N,3) Apophis geocentric positions [m],
- `vA_N`    : (N,3) Apophis geocentric velocities [m/s],
- `bRef_N`  : (N,3) boresight/reference LOS (optional; rebuilt from rA−rD),
- `q_RN`    : (N,4) scalar-first quaternions N→R (optional; used if present).

`NPZBundle` provides a robust `interp(arr, t_abs)` method for time-aligned
interpolation of any array at absolute time `t_abs`.

Per-deputy modules (Deputy)
---------------------------
For each NPZ bundle, `Deputy` builds:

- Hub and initial state:
    * Mass and inertia from `SimulationConfig.inertia` and `mass_kg`.
    * Initial attitude:
        - Either from q_RN[0] (N→R) converted to C_NB,
        - Or via TRIAD from bRef_N[0] and vD_N[0].
        - A small offset about body +Y (CFG.init_y_deg) is applied to enforce
          a visible transient.
    * Initial position/velocity from the NPZ’s first rD_N/vD_N sample.

- Dynamics:
    * `spacecraft.Spacecraft` with an `extForceTorque` effector to apply
      body torques.

- Navigation:
    * `simpleNav.SimpleNav` attaching to `sc.scStateOutMsg`.
    * If `CFG.nav_sig` is set:
        - `walkBounds` and `PMatrix` are configured so nav errors follow a
          Gauss–Markov process (pos/vel/att/rate/sun/ΔV).

- FSW:
    * `attTrackingError.attTrackingError()` per deputy:
        - Inputs: nav attitude (`nav.attOutMsg`) and a per-deputy `AttRefMsg`.
    * `mrpFeedback.mrpFeedback()` per deputy with:
        - K = CFG.k_sigma, P = CFG.p_rate, Ki = CFG.ki, integralLimit = CFG.i_lim.
    * Optional `lowPassFilterTorqueCommand`:
        - Enabled when `CFG.use_lpf` is True; cutoff `CFG.lpf_cut`.

- Reference builder:
    * `Deputy.ref_sigma_omega(t_abs, dt)`:
        - If q_RN exists and `CFG.prefer_qrn`:
            + Interpolates q_RN(t_abs±dt) and q_RN(t_abs),
            + Computes Cdot and back out ω_RN_N by finite differencing,
            + Returns σ_RN (MRPs) and ω_RN_N.
        - Else:
            + Rebuilds attitude via TRIAD from bRef_N(t_abs) and vD_N(t_abs),
            + Returns σ_RN and zero ω_RN_N.

FleetSim orchestration
----------------------
`FleetSim` wires all deputies into three `sys_model_task.SysModelTask` objects:

- `sns` : sensor/nav task at `cfg.dt_sens`,
- `fsw` : FSW task at `cfg.dt_dyn`,
- `dyn` : dynamics task at `cfg.dt_dyn`.

Key wiring details:

- Each deputy’s spacecraft adds `extForceTorque` as a dynamic effector.
- `simpleNav` subscribes to each spacecraft state.
- `attTrackingError` consumes nav attitude and per-deputy AttRef message.
- `mrpFeedback` consumes `attGuidOutMsg` and a **shared** VehicleConfigMsg:
    * VehicleConfigMsgPayload.ISCPntB_B is constructed from `CFG.inertia`
      once in `_build_tasks`.
- Optional LPF sits between mrpFeedback and extForceTorque if enabled.

All tasks call:

    task.SelfInitTaskList()
    task.ResetTaskList(0)

and then run synchronously in the main loop.

Simulation time base
--------------------
`FleetSim` derives:

- `t0_abs` : minimum t[0] across all bundles,
- `t_end`  : maximum t[-1] across all bundles (or `CFG.t_stop` if specified).

The main loop runs:

    N = int((t_end − t0_abs)/dt_dyn) + 1

At each step:

1. Compute `t_rel` and `t_abs = t0_abs + t_rel`.
2. For each deputy:
    * Compute (σ_RN, ω_RN_N) = ref_sigma_omega(t_abs, dt).
    * Fill an `AttRefMsgPayload` **without kwargs**, then set:
          sigma_RN, omega_RN_N, domega_RN_N = 0.
    * Write into the deputy’s AttRef message.
3. Execute tasks at `now_ns = sec2nano(t_rel)`:
    * `sns.ExecuteTaskList(now_ns)`
    * `fsw.ExecuteTaskList(now_ns)`
    * `dyn.ExecuteTaskList(now_ns)`

A `tqdm` progress bar is used to track simulation progress.

Logging
-------
If `CFG.log_dir` is set, `FleetSim.save_logs()` writes one NPZ per deputy:

- File pattern:
      fleet_logs/dep{ii}_logs.npz

- Core fields:
    * `t_err`, `sigma_BR`, `omega_BR`      – from attTrackingError recorder,
    * `t_torque`, `torque_B`               – from mrpFeedback recorder.

- Optional nav fields (if `CFG.log_nav`):
    * `t_navA`, `sigma_nav`, `omega_nav`   – nav attitude logs,
    * `t_navT`, `r_nav`, `v_nav`           – nav translation logs.

- Optional truth fields (if `CFG.log_truth`):
    * `t_true`, `r_true`, `v_true`,
      `sigma_true`, `omega_true`.

The helper `safe_field(rec, name, cols)` converts recorder fields into
well-shaped arrays, robust against variations in array shapes.

Usage
-----
1. Place one or more rendezvous NPZ bundles in the fleet directory:

       export APOPHIS_FLEET_DIR=/path/to/npz_dir

   or leave it as the current directory and use the default glob:
       apophis_rendezvous_full_D*.npz

2. Adjust `SimulationConfig` (gains, noise, init_y_deg, logging flags) at the
   top of the script as needed.

3. Run:

    python basilik_fleet_6dof_attitude_minisim.py

The script will:

- Discover all NPZ rendezvous files and wrap them in NPZBundle,
- Build a Deputy (SC + nav + FSW) for each,
- Run a time-aligned 6-DOF attitude tracking simulation across the fleet,
- Print run-time, and
- Save per-deputy logs into `fleet_logs/dep??_logs.npz` for later plotting
  (e.g. using `fleet_quicklook_plots.py` or other analysis tools).
"""

from __future__ import annotations
import os, sys, math, glob, time, dataclasses as dc
from pathlib import Path
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
# 1)  CONFIGURATION  (all knobs in one dataclass)
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

    # logging
    log_nav   : bool = True
    log_truth : bool = True
    log_dir   : Path = Path("fleet_logs")

    # runtime
    t_stop    : float|None = None
    prefer_qrn: bool = True


CFG = SimulationConfig()


# ╔══════════════════════════════════════════════════════════════════════════╗
# 2)  NPZ  wrapper  (trajectory + helpers)
# ╚══════════════════════════════════════════════════════════════════════════╝
class NPZBundle:
    def __init__(self, path: Path, prefer_qrn: bool):
        n = np.load(path)
        self.t    = n["t"].astype(float)
        self.rD   = n["rD_N"];  self.vD = n["vD_N"]
        self.rA   = n["rA_N"];  self.vA = n["vA_N"]
        self.bRef = n["bRef_N"] if "bRef_N" in n else (
                    (self.rA-self.rD) /
                    la.norm(self.rA-self.rD, axis=1)[:,None])
        self.use_q = ("q_RN" in n) and prefer_qrn
        self.qRN   = n["q_RN"] if self.use_q else None

    # linear interpolation ---------------------------------------------------
    def interp(self, arr: np.ndarray, t_abs: float) -> np.ndarray:
        t = self.t
        if t_abs<=t[0]  : return arr[0]
        if t_abs>=t[-1] : return arr[-1]
        j = np.searchsorted(t, t_abs) - 1
        f = (t_abs-t[j])/(t[j+1]-t[j])
        return (1-f)*arr[j] + f*arr[j+1]


# ╔══════════════════════════════════════════════════════════════════════════╗
# 3)  Deputy  (all Basilisk modules for one spacecraft)
# ╚══════════════════════════════════════════════════════════════════════════╝
class Deputy:
    def __init__(self, idx: int, bundle: NPZBundle):
        self.i     = idx
        self.data  = bundle
        self.ref   = messaging.AttRefMsg()          # populated each step
        self.recs  = {}                             # recorder handles
        self._build_modules()

    # ─── module construction ────────────────────────────────────────────────
    def _build_modules(self):
        Ixx,Iyy,Izz = CFG.inertia
        theta = math.radians(CFG.init_y_deg)

        # helper rotation offset
        R_off = np.array([[ math.cos(theta),0, math.sin(theta)],
                          [ 0,1,0 ],
                          [-math.sin(theta),0, math.cos(theta)]])

        # hub ---------------------------------------------------------------
        self.sc = spacecraft.Spacecraft()
        self.sc.ModelTag = f"Dep{self.i}"
        self.sc.hub.mHub = CFG.mass_kg
        self.sc.hub.IHubPntBc_B = [[Ixx,0,0],[0,Iyy,0],[0,0,Izz]]

        # attitude init
        if self.data.use_q:
            C_NB0 = quat_to_dcm(self.data.qRN[0]).T
        else:
            C_NB0 = triad_from_boresight_and_tangent(
                        self.data.bRef[0], self.data.vD[0])
        C_NB0 = C_NB0 @ R_off
        sigma0 = quat_to_mrp(dcm_to_quat(C_NB0))
        self.sc.hub.sigma_BNInit = [[float(s)] for s in sigma0]
        self.sc.hub.r_CN_NInit   = list(self.data.rD[0])
        self.sc.hub.v_CN_NInit   = list(self.data.vD[0])

        # effector & nav ----------------------------------------------------
        self.torque = extForceTorque.ExtForceTorque()
        self.torque.ModelTag = f"torque_{self.i}"

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

        # FSW ---------------------------------------------------------------
        self.err = attTrackingError.attTrackingError()
        self.err.ModelTag = f"err_{self.i}"

        self.ctl = mrpFeedback.mrpFeedback()
        self.ctl.ModelTag = f"ctl_{self.i}"
        self.ctl.K  = CFG.k_sigma; self.ctl.P = CFG.p_rate
        self.ctl.Ki = CFG.ki      ; self.ctl.integralLimit = CFG.i_lim

        self.lpf = None
        if CFG.use_lpf:
            self.lpf = lowPassFilterTorqueCommand.lowPassFilterTorqueCommand()
            self.lpf.ModelTag = f"lpf_{self.i}"
            self.lpf.h  = CFG.dt_dyn
            self.lpf.wc = 2*math.pi*CFG.lpf_cut

    # ─── reference builder ─────────────────────────────────────────────────
    def ref_sigma_omega(self, t_abs: float, dt: float) -> tuple[np.ndarray,np.ndarray]:
        if self.data.use_q:
            q_prev = self.data.interp(self.data.qRN, max(self.data.t[0], t_abs-dt))
            q_curr = self.data.interp(self.data.qRN, t_abs)
            q_next = self.data.interp(self.data.qRN, min(self.data.t[-1], t_abs+dt))
            Cdot   = (quat_to_dcm(q_next)-quat_to_dcm(q_prev))/max(2*dt,1e-6)
            S      = Cdot @ quat_to_dcm(q_curr).T
            omega  = -.5*np.array([S[2,1]-S[1,2], S[0,2]-S[2,0], S[1,0]-S[0,1]])
            sigma  = quat_to_mrp(q_curr); return sigma, omega
        # fallback TRIAD
        rD = self.data.interp(self.data.rD, t_abs)
        vD = self.data.interp(self.data.vD, t_abs)
        bN = self.data.interp(self.data.bRef, t_abs)
        C  = triad_from_boresight_and_tangent(bN,vD)
        return quat_to_mrp(dcm_to_quat(C.T)), np.zeros(3)


# ╔══════════════════════════════════════════════════════════════════════════╗
# 4)  FleetSim  (orchestrates the whole show)
# ╚══════════════════════════════════════════════════════════════════════════╝
class FleetSim:
    def __init__(self, cfg: SimulationConfig):
        self.cfg    = cfg
        self.bundles= self._discover_npz()
        self.deps   = [Deputy(i,b) for i,b in enumerate(self.bundles)]
        self._build_tasks()

    # NPZ discovery ----------------------------------------------------------
    def _discover_npz(self) -> list[NPZBundle]:
        paths = sorted(self.cfg.fleet_dir.glob(self.cfg.npz_glob))
        if not paths:
            print("[ERR] no NPZ matched"); sys.exit(1)
        return [NPZBundle(p, self.cfg.prefer_qrn) for p in paths]

    # task & wiring ----------------------------------------------------------
    def _build_tasks(self):
        cfg = self.cfg
        self.dyn = sys_model_task.SysModelTask(int(macros.sec2nano(cfg.dt_dyn)))
        self.fsw = sys_model_task.SysModelTask(int(macros.sec2nano(cfg.dt_dyn)))
        self.sns = sys_model_task.SysModelTask(int(macros.sec2nano(cfg.dt_sens)))

        # static vehicle config msg
        # ── static vehicle-inertia message (FIX) ──────────────────────────
        Ixx, Iyy, Izz = self.cfg.inertia
        vehCfgPayload = messaging.VehicleConfigMsgPayload()      # ← blank payload
        vehCfgPayload.ISCPntB_B = [Ixx, 0, 0,
                                   0, Iyy, 0,
                                   0, 0, Izz]                    # ← set after init
        vehCfg          = messaging.VehicleConfigMsg().write(vehCfgPayload)


        for d in self.deps:
            p = 100*d.i  # priority offset

            # wiring -------------------------------------------------------
            d.sc.addDynamicEffector(d.torque)
            d.nav.scStateInMsg.subscribeTo(d.sc.scStateOutMsg)

            d.err.attNavInMsg .subscribeTo(d.nav.attOutMsg)
            d.err.attRefInMsg .subscribeTo(d.ref)
            d.ctl.guidInMsg   .subscribeTo(d.err.attGuidOutMsg)
            d.ctl.vehConfigInMsg.subscribeTo(vehCfg)

            if d.lpf:
                d.lpf.cmdTorqueInMsg.subscribeTo(d.ctl.cmdTorqueOutMsg)
                d.torque.cmdTorqueInMsg.subscribeTo(d.lpf.cmdTorqueOutMsg)
            else:
                d.torque.cmdTorqueInMsg.subscribeTo(d.ctl.cmdTorqueOutMsg)

            # add to tasks -----------------------------------------------
            self.sns.AddNewObject(d.nav,  10+p)
            self.fsw.AddNewObject(d.err,  20+p); self.fsw.AddNewObject(d.ctl,10+p)
            if d.lpf: self.fsw.AddNewObject(d.lpf, 5+p)
            self.dyn.AddNewObject(d.torque,10+p); self.dyn.AddNewObject(d.sc,5+p)

            # recorders ---------------------------------------------------
            rec = d.err.attGuidOutMsg.recorder(); self.fsw.AddNewObject(rec,-1)
            d.recs["err"] = rec
            rec = d.ctl.cmdTorqueOutMsg.recorder(); self.fsw.AddNewObject(rec,-1)
            d.recs["torq"]= rec

            if cfg.log_nav:
                rec = d.nav.attOutMsg.recorder();  self.sns.AddNewObject(rec,-1)
                d.recs["navA"]=rec
                rec = d.nav.transOutMsg.recorder();self.sns.AddNewObject(rec,-1)
                d.recs["navT"]=rec
            if cfg.log_truth:
                rec = d.sc.scStateOutMsg.recorder();self.dyn.AddNewObject(rec,-1)
                d.recs["truth"]=rec

        for t in (self.sns,self.fsw,self.dyn):
            t.SelfInitTaskList(); t.ResetTaskList(0)

        # timing ---------------------------------------------------------
        t0 = min(b.t[0] for b in self.bundles)
        t1 = max(b.t[-1] for b in self.bundles)
        self.t0_abs = t0
        self.t_end  = min(self.cfg.t_stop or t1, t1)

    # main loop --------------------------------------------------------------
    def run(self):
        N = int((self.t_end - self.t0_abs)/self.cfg.dt_dyn) + 1
        from tqdm import tqdm; bar=tqdm(range(N),ncols=80,desc="Sim")
        dt = self.cfg.dt_dyn
        for k in bar:
            t_rel = k*dt; t_abs = self.t0_abs + t_rel
            now_ns = int(macros.sec2nano(t_rel))

            # update references
            for d in self.deps:
                σ, ω = d.ref_sigma_omega(t_abs, dt)

                # ✱✱ FIX ✱✱  (no kwargs – set fields after construction)
                ref_pl = messaging.AttRefMsgPayload()
                ref_pl.sigma_RN    = list(σ)
                ref_pl.omega_RN_N  = list(ω)
                ref_pl.domega_RN_N = [0.0, 0.0, 0.0]

                d.ref.write(ref_pl)
            # step
            self.sns.ExecuteTaskList(now_ns)
            self.fsw.ExecuteTaskList(now_ns)
            self.dyn.ExecuteTaskList(now_ns)
        print("[DONE] simulation")

    # save logs --------------------------------------------------------------
    def save_logs(self):
        cfg = self.cfg; cfg.log_dir.mkdir(exist_ok=True)
        for d in self.deps:
            fname = cfg.log_dir / f"dep{d.i:02d}_logs.npz"
            fields = dict(
                t_err = d.recs["err"].times()*macros.NANO2SEC,
                sigma_BR = safe_field(d.recs["err"],"sigma_BR",3),
                omega_BR = safe_field(d.recs["err"],"omega_BR_B",3),
                t_torque = d.recs["torq"].times()*macros.NANO2SEC,
                torque_B = safe_field(d.recs["torq"],"torqueRequestBody",3)
            )
            if cfg.log_nav:
                fields.update(
                    t_navA = d.recs["navA"].times()*macros.NANO2SEC,
                    sigma_nav = safe_field(d.recs["navA"],"sigma_BN_B",3),
                    omega_nav = safe_field(d.recs["navA"],"omega_BN_B",3),
                    t_navT = d.recs["navT"].times()*macros.NANO2SEC,
                    r_nav = safe_field(d.recs["navT"],"r_BN_N",3),
                    v_nav = safe_field(d.recs["navT"],"v_BN_N",3))
            if cfg.log_truth:
                fields.update(
                    t_true = d.recs["truth"].times()*macros.NANO2SEC,
                    r_true = safe_field(d.recs["truth"],"r_BN_N",3),
                    v_true = safe_field(d.recs["truth"],"v_BN_N",3),
                    sigma_true = safe_field(d.recs["truth"],"sigma_BN",3),
                    omega_true = safe_field(d.recs["truth"],"omega_BN_B",3))
            np.savez_compressed(fname, **fields)
            print(f"[SAVE] {fname}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# 5)  MAIN  entry
# ╚══════════════════════════════════════════════════════════════════════════╝
def main():
    sim = FleetSim(CFG)
    t0 = time.time(); sim.run(); print(f"[TIMING] {time.time()-t0:.1f} s")
    if CFG.log_dir: sim.save_logs()


# ── helpers copied unchanged from original ─────────────────────────────────
def dcm_to_quat(C):  # (same as before) ...
    tr=np.trace(C)
    if tr>0: S=math.sqrt(tr+1)*2; q0=.25*S; q1=(C[2,1]-C[1,2])/S; q2=(C[0,2]-C[2,0])/S; q3=(C[1,0]-C[0,1])/S
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

# --- replace the helper block at the bottom of the script -------------------
def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """
    Quaternion (scalar-first) -> DCM  — corrected
    """
    q0, q1, q2, q3 = q
    qv = np.array([q1, q2, q3])

    # ‼️ change la.dot(...)  ->  np.dot(...)
    return (q0*q0 - np.dot(qv, qv)) * np.eye(3) \
         + 2 * np.outer(qv, qv) \
         + 2 * q0 * np.array([[ 0, -q3,  q2],
                              [ q3,  0, -q1],
                              [-q2,  q1,  0]])

    return (q0*q0-la.dot(qv,qv))*np.eye(3)+2*np.outer(qv,qv)+2*q0*np.array([[0,-q3,q2],[q3,0,-q1],[-q2,q1,0]])
def quat_to_mrp(q): q0,qv=q[0],q[1:]; return -qv/(1-q0) if 1+q0<1e-12 else qv/(1+q0)
def normalize(v,eps=1e-15): n=la.norm(v); return (v/n,n) if n>=eps else (np.zeros_like(v),0)
def triad_from_boresight_and_tangent(los,vel):
    z,_=normalize(los); v_perp=vel-np.dot(vel,z)*z; x,nx=normalize(v_perp)
    if nx<1e-10: tmp=np.array([1,0,0]) if abs(z[0])<.9 else np.array([0,1,0]); x=normalize(np.cross(tmp,z))[0]
    y=np.cross(z,x); x=normalize(np.cross(y,z))[0]; y=normalize(np.cross(z,x))[0]
    return np.column_stack((x,y,z))
def safe_field(rec,name,cols=3):
    if not hasattr(rec,name): return np.zeros((0,cols))
    arr=np.array(getattr(rec,name))
    if arr.ndim==1: return arr.reshape(-1,cols) if arr.size%cols==0 else arr.reshape(-1,1)
    return arr
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
