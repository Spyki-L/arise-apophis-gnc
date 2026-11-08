#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rhig_rendezvous_fleet.py
========================

Generic RHIG fleet rendezvous module for multiple deputies.

This wraps the original basilik_rhig_rendezvous_fleet.py logic
into a reusable RHIGFleet class, without changing the underlying
Basilisk SPICE and Lambert modules.

Additions in this version
-------------------------
1. Support for generic targets:
   - SPICE body (central_body_name, target_body_name, target_spk_name), or
   - NPZ trajectory bundle (target_npz) with times [s], positions [km], velocities [km/s].

2. GEO / “tight final error” usage:
   - Configurable final error tolerance:
       FINAL_ERR_REL_TOL  (fraction of ‖target LVLH‖, default 5%)
       FINAL_ERR_ABS_MIN  (floor in metres)
   - After a standard RHIG run, you can *optionally* compress impulses and
     add a terminal CW burn that drives the final error below the tolerance.

3. Impulse compression:
   - Early impulses are merged into a single “shaping” burn.
   - A terminal burn near arrival is computed via a CW terminal map and
     damped least-squares to hit the LVLH target.
   - The trajectory with this compressed schedule is re-propagated
     (pure CW/Hill) and exported.

The NPZ field `APOPHIS_TARGET` is kept for compatibility with existing
plotting code and should be interpreted generically as “target LVLH stand-off”.
"""

from __future__ import annotations
import os, math, warnings, sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Tuple, List, Dict

import numpy as np
import numpy.linalg as la
from scipy.linalg import svd

# ───────────────── optional tqdm ─────────────────
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda it, **kw: it  # type: ignore

# ───────────────── Basilisk imports ─────────────────
from Basilisk import __path__ as bskPath
from Basilisk.simulation import spiceInterface, ephemerisConverter, simpleNav, spacecraft
from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.fswAlgorithms import lambertPlanner, lambertSolver
from Basilisk.utilities.simIncludeGravBody import gravBodyFactory

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(precision=3, suppress=True)


# ╔════════════════════════════════════════════╗
# 1. Config dataclass                          ║
# ╚════════════════════════════════════════════╝

@dataclass
class RHIGFleetConfig:
    # Paths / ephemerides
    chief_ref_npz: Path = Path(os.getenv("CHIEF_REF_NPZ", "chief_reference.npz"))
    spice_dir: Path = Path(os.path.join(bskPath[0], "supportData", "EphemerisData"))

    # Optional target trajectory NPZ (times, positions, velocities in km/km/s)
    # If not None, this is used instead of SPICE for rA(t), vA(t).
    target_npz: Path | None = None

    # SPICE epoch for chief departure / target sampling
    departure_utc_iso: str = os.getenv("DEPARTURE_UTC_ISO", "2029-04-13T05:41:45Z")

    # Central and target bodies (SPICE) — used if target_npz is None
    central_body_name: str = os.getenv("CENTRAL_BODY_NAME", "Earth")
    target_body_name: str  = os.getenv("TARGET_BODY_NAME",  "20099942")
    target_spk_name: str   = os.getenv("TARGET_SPK_NAME",   "20099942.bsp")

    # Fleet LVLH targets per deputy (R,T,N) [m] – set externally
    fleet_targets_m: np.ndarray | None = None

    # RHIG / optimiser constants (core behaviour)
    DT_DYN: float         = 1.0
    DT_FSW: float         = 60.0
    E_TRIGGER: float      = 5.0
    IMPROVE_MIN: float    = 3.0
    V_MIN: float          = 0.05e-3
    V_MAX: float          = 0.1
    MIN_LOOK: float       = 250.0
    COOLDOWN: float       = 250.0
    MAX_IMP: int          = 60
    ALPHA_DAMP_BASE: float= 1e-7
    USE_LAMBERT_NUDGE: bool = True
    NUDGE_WEIGHT: float   = 0.4
    NUDGE_SAT: float      = 0.4
    ALIGN_DEG_MIN: float  = 1.0
    LAMBDA_DV: float      = 380.0

    SEARCH_TMIN: float    = 1500.0
    SEARCH_TMAX_PAD: float= 1.2*3600.0
    COARSE_STEP: float    = 900.0
    REFINE_STEP_1: float  = 150.0
    REFINE_STEP_2: float  = 60.0
    REFINE_SPAN: float    = 1.2*3600.0

    FINAL_V_MAX: float    = 0.5
    FINAL_MAX_IMP: int    = 80

    TARGET_MARGIN_M: float= 1.0
    DV_BUDGET_MS: float | None = None

    # Plotting / exports
    save_png_dir_base: str | None = "plots_D{}"
    cam_boresight_B: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    nav_npz_path: str | None = None

    # ───── new knobs for GEO / tight-final-error / compression ─────
    FINAL_ERR_REL_TOL: float = 0.05      # 5% of ‖target LVLH‖
    FINAL_ERR_ABS_MIN: float = 5.0       # floor in metres

    ENABLE_COMPRESSION: bool = True      # turn on impulse compression
    COMPRESS_KEEP_LAST: int = 4          # keep last N burns as-is
    TERMINAL_MARGIN: float = 600.0       # terminal burn at TOF - margin [s]
    TERMINAL_V_MAX: float | None = None  # optional cap; None = unlimited


# ╔════════════════════════════════════════════╗
# 2. Helper maths                              ║
# ╚════════════════════════════════════════════╝

def dcm_lvlh(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    i = r / la.norm(r)
    k = np.cross(r, v); k /= la.norm(k)
    j = np.cross(k, i)
    return np.vstack((i, j, k))

def hill_rate(r: np.ndarray, v: np.ndarray) -> float:
    return la.norm(np.cross(r, v)) / (la.norm(r)**2 + 1e-12)

def cw_blocks(n: float, dt: float):
    c, s, nt = math.cos(n*dt), math.sin(n*dt), n*dt
    Phi_rr = np.array([[4-3*c, 0,      0],
                       [6*(s-nt), 1,    0],
                       [0,        0,    c]])
    Phi_rv = np.array([[s/n,          2*(1-c)/n,      0],
                       [-2*(1-c)/n,   (4*s-3*nt)/n,   0],
                       [0,            0,              s/n]])
    Phi_vr = np.array([[3*n*s,       0,          0],
                       [6*n*(c-1),   0,          0],
                       [0,           0,         -n*s]])
    Phi_vv = np.array([[c,   2*s,        0],
                       [-2*s, 4*c-3,     0],
                       [0,    0,          c]])
    return Phi_rr, Phi_rv, Phi_vr, Phi_vv

def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = la.norm(a), la.norm(b)
    if na * nb == 0:
        return 0.0
    return float(np.degrees(
        math.acos(np.clip(np.dot(a, b)/(na*nb), -1.0, 1.0))
    ))

def normalize(v: np.ndarray, eps: float = 1e-12):
    n = la.norm(v)
    return (v/n if n >= eps else np.zeros_like(v)), n

def triad_from_boresight_and_tangent(los: np.ndarray, vel: np.ndarray) -> np.ndarray:
    zB, _ = normalize(los)
    v_perp = vel - np.dot(vel, zB) * zB
    xB, nx = normalize(v_perp)
    if nx < 1e-8:
        h = np.cross(los, vel); h_hat, nh = normalize(h)
        tmp = np.array([1,0,0]) if abs(zB[0]) < .9 else np.array([0,1,0])
        xB = normalize(np.cross(tmp if nh < 1e-8 else h_hat, zB))[0]
    yB = np.cross(zB, xB)
    xB = normalize(np.cross(yB, zB))[0]
    yB = normalize(np.cross(zB, xB))[0]
    return np.column_stack((xB, yB, zB))

def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    tr = float(np.trace(C))
    if tr > 0:
        S  = math.sqrt(tr+1)*2
        q0 = 0.25 * S
        q1 = (C[2,1] - C[1,2]) / S
        q2 = (C[0,2] - C[2,0]) / S
        q3 = (C[1,0] - C[0,1]) / S
    else:
        i = int(np.argmax([C[0,0], C[1,1], C[2,2]]))
        if i == 0:
            S  = math.sqrt(1 + C[0,0] - C[1,1] - C[2,2]) * 2
            q0 = (C[2,1] - C[1,2]) / S
            q1 = 0.25 * S
            q2 = (C[0,1] + C[1,0]) / S
            q3 = (C[0,2] + C[2,0]) / S
        elif i == 1:
            S  = math.sqrt(1 + C[1,1] - C[0,0] - C[2,2]) * 2
            q0 = (C[0,2] - C[2,0]) / S
            q1 = (C[0,1] + C[1,0]) / S
            q2 = 0.25 * S
            q3 = (C[1,2] + C[2,1]) / S
        else:
            S  = math.sqrt(1 + C[2,2] - C[0,0] - C[1,1]) * 2
            q0 = (C[1,0] - C[0,1]) / S
            q1 = (C[0,2] + C[2,0]) / S
            q2 = (C[1,2] + C[2,1]) / S
            q3 = 0.25 * S
    q = np.array([q0, q1, q2, q3], float)
    return q / la.norm(q)

def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    q0, qv = q[0], q[1:]
    denom = 1.0 + q0
    if denom < 1e-12:
        denom = 1.0 - q0
        return -qv / max(denom, 1e-12)
    return qv / denom


# ╔════════════════════════════════════════════╗
# 3. RHIGFleet class                           ║
# ╚════════════════════════════════════════════╝

class RHIGFleet:
    """
    RHIGFleet(cfg)

    - Loads chief_reference.npz (km, km/s) and builds chief_state(t).
    - Builds target_state(t) either from:
         * SPICE (central + target body)  OR
         * a target_npz trajectory bundle (times,pos,vel in km/km/s).
    - Runs RHIG (run_rhig_from) for each deputy LVLH target in cfg.fleet_targets_m.
    - Optionally compresses impulses and adds a terminal CW burn to
      force final error below a configurable tolerance.
    - Exports rendezvous_full_D{k}.npz + viz_inputs_D{k}.npz via methods.
    """

    def __init__(self, cfg: RHIGFleetConfig):
        self.cfg = cfg

        # SPICE path and kernels (used if target_npz is None)
        self.SPICE_DIR = str(cfg.spice_dir) + os.sep
        self.KERNELS = [
            "naif0012.tls",
            "pck00010.tpc",
            "de-403-masses.tpc",
            "de430.bsp",
            cfg.target_spk_name,
        ]

        # Gravity / radius (for plotting)
        try:
            from Basilisk.utilities import astroConstants as ac
            self.MU_EARTH  = ac.mu_Earth * 1e9
            self.R_EARTH_KM= ac.req_Earth/1e3
        except Exception:
            self.MU_EARTH, self.R_EARTH_KM = 3.986004418e14, 6378.137

        # Chief reference
        self.TOF_SEC, self.chief_state = self._load_chief_npz()

        # Departure/arrival
        self.departure_dt = datetime.strptime(
            cfg.departure_utc_iso, "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        self.arrival_dt = self.departure_dt + timedelta(seconds=self.TOF_SEC)

        # Target sampler: SPICE or NPZ
        if cfg.target_npz is not None:
            self.rA_fn, self.vA_fn = self._load_target_npz(cfg.target_npz)
        else:
            self.rA_fn, self.vA_fn = self._make_target_sampler(
                self.departure_dt, int(self.TOF_SEC + 3600), 60
            )

        # Generic “target LVLH” placeholder for NPZ exports
        self.APOPHIS_TARGET = np.array([0.0, 0.0, 0.0])

    # ---------- ephemeris loaders (chief / target) ----------

    def _load_chief_npz(self) -> Tuple[float, Callable[[float], Tuple[np.ndarray, np.ndarray]]]:
        cfg = self.cfg
        if not cfg.chief_ref_npz.is_file():
            sys.exit(f"chief_reference not found: {cfg.chief_ref_npz}")
        d = np.load(cfg.chief_ref_npz)
        T, R, V = d["times"], d["positions"], d["velocities"]
        tof = float(T[-1])

        def chief_state(t: float) -> Tuple[np.ndarray, np.ndarray]:
            r = [np.interp(t, T, R[:, k]) for k in range(3)]
            v = [np.interp(t, T, V[:, k]) for k in range(3)]
            return 1e3 * np.array(r), 1e3 * np.array(v)

        return tof, chief_state

    def _load_target_npz(
        self,
        npz_path: Path,
    ) -> Tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
        """
        Load a target trajectory from a NPZ with:
            times      : (N,)   [s]
            positions  : (N,3)  [km]
            velocities : (N,3)  [km/s]
        """
        npz_path = Path(npz_path)
        if not npz_path.is_file():
            sys.exit(f"target_npz not found: {npz_path}")
        d = np.load(npz_path)
        T    = np.asarray(d["times"], float)
        R_km = np.asarray(d["positions"], float)
        V_km = np.asarray(d["velocities"], float)

        def rA(t: float) -> np.ndarray:
            r = [np.interp(t, T, R_km[:, k]) for k in range(3)]
            return 1e3 * np.array(r)

        def vA(t: float) -> np.ndarray:
            v = [np.interp(t, T, V_km[:, k]) for k in range(3)]
            return 1e3 * np.array(v)

        print(f"[RHIGFleet] using target NPZ: {npz_path.resolve()}")
        return rA, vA

    def _make_target_sampler(
        self,
        start_dt: datetime,
        duration_s: int,
        step_s: int = 60,
    ) -> Tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
        """
        Original SPICE-based target sampler (central + target body).
        """
        cfg = self.cfg

        sim = SimulationBaseClass.SimBaseClass()
        task = sim.CreateNewTask("task", macros.sec2nano(step_s))
        sim.CreateNewProcess("dyn").addTask(task)

        spi = spiceInterface.SpiceInterface()
        spi.SPICEDataPath = self.SPICE_DIR
        spi.UTCCalInit = start_dt.strftime("%Y %B %d, %H:%M:%S.0 (UTC)")
        for k in self.KERNELS:
            spi.loadSpiceKernel(k, self.SPICE_DIR)
        spi.addPlanetNames([cfg.central_body_name, "Sun", cfg.target_body_name])
        spi.zeroBase = "Sun"
        sim.AddModelToTask("task", spi)

        conv = ephemerisConverter.EphemerisConverter()
        for m in spi.planetStateOutMsgs:
            conv.addSpiceInputMsg(m)
        sim.AddModelToTask("task", conv)

        cRec = conv.ephemOutMsgs[0].recorder()
        tRec = conv.ephemOutMsgs[2].recorder()
        sim.AddModelToTask("task", cRec)
        sim.AddModelToTask("task", tRec)

        sim.InitializeSimulation()
        sim.ConfigureStopTime(macros.sec2nano(duration_s))
        sim.ExecuteSimulation()

        N = int(duration_s/step_s) + 1
        t_grid = np.arange(N)*step_s
        rT = tRec.r_BdyZero_N[:N] - cRec.r_BdyZero_N[:N]
        vT = tRec.v_BdyZero_N[:N] - cRec.v_BdyZero_N[:N]

        def interp(arr: np.ndarray, t: float) -> np.ndarray:
            if t <= 0:
                return arr[0]
            if t >= t_grid[-1]:
                return arr[-1]
            u = t/step_s
            i = int(u)
            f = u - i
            return (1-f)*arr[i] + f*arr[i+1]

        print(f"[RHIGFleet] using SPICE target: {cfg.target_body_name} ({cfg.target_spk_name})")
        return lambda t: interp(rT, t), lambda t: interp(vT, t)

    # ---------- RHIG core (as in original) ----------

    def _cw_map_to_target_lvlh(self, rho: np.ndarray, rhod: np.ndarray, t_now: float):
        """
        Mixed-frame terminal map using CW/Hill and current APOPHIS_TARGET.
        """
        rC, vC   = self.chief_state(t_now)
        n        = hill_rate(rC, vC)
        Phi_rr, Phi_rv, _, _ = cw_blocks(n, self.TOF_SEC - t_now)
        rho_f   = Phi_rr @ rho + Phi_rv @ rhod
        rC_f, vC_f = self.chief_state(self.TOF_SEC)
        rA_f, vA_f = self.rA_fn(self.TOF_SEC), self.vA_fn(self.TOF_SEC)
        C_C_f = dcm_lvlh(rC_f, vC_f)
        C_A_f = dcm_lvlh(rA_f, vA_f)
        rD_f  = rC_f + C_C_f.T @ rho_f
        sepA  = C_A_f @ (rD_f - rA_f)
        S     = C_A_f @ C_C_f.T @ Phi_rv
        return sepA, S

    def _damped_min_norm(self, S: np.ndarray, eA: np.ndarray) -> np.ndarray:
        _, s, _ = svd(S, full_matrices=False)
        alpha = self.cfg.ALPHA_DAMP_BASE + (0.1*s[-1]**2 if s.size else self.cfg.ALPHA_DAMP_BASE)
        return la.solve(S.T @ S + alpha*np.eye(S.shape[1]), -S.T @ eA)

    def _lambert_delta_v_now(self, r_now: np.ndarray, v_now: np.ndarray, r_goal: np.ndarray, tof: float) -> np.ndarray:
        """
        Original lambert_delta_v_now logic, preserved.
        """
        sim = SimulationBaseClass.SimBaseClass()
        dynT = sim.CreateNewTask("dynT", macros.sec2nano(0.01))
        fswT = sim.CreateNewTask("fswT", macros.sec2nano(0.01))
        sim.CreateNewProcess("dyn").addTask(dynT)
        sim.CreateNewProcess("fsw").addTask(fswT)
        mu = gravBodyFactory().createEarth().mu

        sc = spacecraft.Spacecraft()
        sc.hub.r_CN_NInit = r_now
        sc.hub.v_CN_NInit = v_now

        nav = simpleNav.SimpleNav()
        nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)

        sim.AddModelToTask("dynT", sc)
        sim.AddModelToTask("dynT", nav)

        planner = lambertPlanner.LambertPlanner()
        planner.navTransInMsg.subscribeTo(nav.transOutMsg)
        planner.setR_TN_N(r_goal)
        planner.setMu(mu)
        planner.setNumRevolutions(0)
        planner.setManeuverTime(10)
        planner.setFinalTime(10+tof)

        solver = lambertSolver.LambertSolver()
        solver.lambertProblemInMsg.subscribeTo(planner.lambertProblemOutMsg)

        solRec = solver.lambertSolutionOutMsg.recorder()
        sim.AddModelToTask("fswT", planner)
        sim.AddModelToTask("fswT", solver)
        sim.AddModelToTask("fswT", solRec)

        sim.InitializeSimulation()
        sim.ConfigureStopTime(macros.sec2nano(1))
        sim.ExecuteSimulation()

        return solRec.v1_N[-1] - v_now

    def run_rhig_from(
        self,
        deploy_time_s: float,
        *,
        v_max: float | None = None,
        max_imp: int | None = None,
        record_hist: bool = False,
        show_prog: bool = False,
        dep_idx: int | None = None,
    ) -> Dict:
        """
        Run RHIG from one deploy time, for CURRENT self.APOPHIS_TARGET.
        Logic preserved from original run_rhig_from.
        """
        cfg = self.cfg
        if v_max is None:
            v_max = cfg.V_MAX
        if max_imp is None:
            max_imp = cfg.MAX_IMP

        t0 = deploy_time_s
        Nsteps = int((self.TOF_SEC - t0) / cfg.DT_DYN) + 1
        rho  = np.zeros(3)
        rhod = np.zeros(3)
        impulses: List[Tuple[float, np.ndarray]] = []
        last_fire = -1e9

        if record_hist:
            t_hist   = np.empty(Nsteps)
            sep_hist = np.empty((Nsteps, 3))
            err_hist = np.empty(Nsteps)
            rD_hist  = np.empty((Nsteps, 3))
            vD_hist  = np.empty((Nsteps, 3))
            rA_hist  = np.empty((Nsteps, 3))
            vA_hist  = np.empty((Nsteps, 3))
            rC_hist  = np.empty((Nsteps, 3))
            vC_hist  = np.empty((Nsteps, 3))
            CA_hist  = np.empty((Nsteps, 3, 3))
            CC_hist  = np.empty((Nsteps, 3, 3))
            bRef_hist= np.empty((Nsteps, 3))
            CRN_hist = np.empty((Nsteps, 3, 3))
            q_hist   = np.empty((Nsteps, 4))
            sig_hist = np.empty((Nsteps, 3))
            dv_times: List[float] = []
            dv_list:  List[np.ndarray] = []

        it = range(Nsteps)
        if show_prog:
            it = tqdm(it, desc=f"Dep {dep_idx} final", leave=False)

        for j in it:
            t_now = t0 + j*cfg.DT_DYN
            rC, vC = self.chief_state(t_now)
            rA, vA = self.rA_fn(t_now), self.vA_fn(t_now)
            C_C, C_A = dcm_lvlh(rC, vC), dcm_lvlh(rA, vA)
            n = hill_rate(rC, vC)

            rD = rC + C_C.T @ rho
            vD = vC + C_C.T @ (rhod + np.cross([0, 0, n], rho))

            sep_pred, S = self._cw_map_to_target_lvlh(rho, rhod, t_now)
            errA = sep_pred - self.APOPHIS_TARGET

            if record_hist:
                t_hist[j]   = t_now
                sep_hist[j] = C_A @ (rD - rA)
                err_hist[j] = la.norm(errA)
                rD_hist[j]  = rD
                vD_hist[j]  = vD
                rA_hist[j]  = rA
                vA_hist[j]  = vA
                rC_hist[j]  = rC
                vC_hist[j]  = vC
                CA_hist[j]  = C_A
                CC_hist[j]  = C_C

                los = rA - rD
                los_hat, _ = normalize(los)
                C_RN = triad_from_boresight_and_tangent(los_hat, vD).T
                q    = dcm_to_quat(C_RN)
                sig  = quat_to_mrp(q)
                bRef_hist[j] = los_hat
                CRN_hist[j]  = C_RN
                q_hist[j]    = q
                sig_hist[j]  = sig

            can_fire = (
                (j % int(cfg.DT_FSW / cfg.DT_DYN) == 0)
                and (self.TOF_SEC - t_now > cfg.MIN_LOOK)
                and (t_now - last_fire >= cfg.COOLDOWN)
                and (len(impulses) < max_imp)
            )
            if can_fire and la.norm(errA) > cfg.E_TRIGGER:
                dv_cw = self._damped_min_norm(S, errA)
                dv_nudge = np.zeros(3)
                if cfg.USE_LAMBERT_NUDGE:
                    rA_fin, vA_fin = self.rA_fn(self.TOF_SEC), self.vA_fn(self.TOF_SEC)
                    C_A_fin = dcm_lvlh(rA_fin, vA_fin)
                    r_goal = rA_fin + C_A_fin.T @ self.APOPHIS_TARGET
                    if angle_deg(rD, r_goal) >= cfg.ALIGN_DEG_MIN:
                        dv_eci = self._lambert_delta_v_now(rD, vD, r_goal, self.TOF_SEC - t_now)
                        m = la.norm(dv_eci)
                        if m > cfg.NUDGE_SAT and m > 0:
                            dv_eci *= cfg.NUDGE_SAT / m
                        dv_nudge = C_C @ dv_eci

                dv = (1 - cfg.NUDGE_WEIGHT)*dv_cw + cfg.NUDGE_WEIGHT*dv_nudge
                base = la.norm(errA)
                chosen = None
                for scl in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    dv_try = dv * scl
                    m = la.norm(dv_try)
                    if m < cfg.V_MIN:
                        continue
                    if m > v_max:
                        dv_try *= v_max / m
                    sep_new, _ = self._cw_map_to_target_lvlh(rho, rhod + dv_try, t_now)
                    if la.norm(sep_new - self.APOPHIS_TARGET) < base - cfg.IMPROVE_MIN:
                        chosen = dv_try
                        break
                if chosen is not None:
                    rhod += chosen
                    impulses.append((t_now, chosen.copy()))
                    last_fire = t_now
                    if record_hist:
                        dv_times.append(t_now)
                        dv_list.append(chosen.copy())

            Phi_rr, Phi_rv, Phi_vr, Phi_vv = cw_blocks(n, cfg.DT_DYN)
            rho, rhod = Phi_rr @ rho + Phi_rv @ rhod, Phi_vr @ rho + Phi_vv @ rhod

        # terminal metrics at TOF_SEC
        rA_end, vA_end = self.rA_fn(self.TOF_SEC), self.vA_fn(self.TOF_SEC)
        C_A_end = dcm_lvlh(rA_end, vA_end)
        rC_end, vC_end = self.chief_state(self.TOF_SEC)
        C_C_end = dcm_lvlh(rC_end, vC_end)
        rD_end = rC_end + C_C_end.T @ rho

        sep_final = C_A_end @ (rD_end - rA_end)
        err_final = sep_final - self.APOPHIS_TARGET
        dv_tot = sum(la.norm(dv) for _, dv in impulses)

        res: Dict[str, object] = dict(
            err_norm=float(la.norm(err_final)),
            e_final=err_final,
            sep_final=sep_final,
            tot_dv=dv_tot,
            impulses=impulses,
        )
        if record_hist:
            res.update(
                t_hist=t_hist,
                sepA_hist=sep_hist,
                err_hist=err_hist,
                rD_hist=rD_hist,
                vD_hist=vD_hist,
                rA_hist=rA_hist,
                vA_hist=vA_hist,
                rC_hist=rC_hist,
                vC_hist=vC_hist,
                C_A_N_hist=CA_hist,
                C_C_N_hist=CC_hist,
                bRef_N_hist=bRef_hist,
                C_RN_hist=CRN_hist,
                q_RN_hist=q_hist,
                sigma_RN_hist=sig_hist,
                dv_times=np.array(dv_times),
                dv_list=np.array(dv_list) if dv_list else np.zeros((0, 3)),
                t0=t0,
            )
        return res

    def eval_cost(self, r: Dict[str, object]) -> float:
        return r["err_norm"] + self.cfg.LAMBDA_DV * r["tot_dv"]  # type: ignore

    # ---------- pure propagation of a fixed impulse profile ----------

    def _propagate_profile(
        self,
        t0: float,
        impulses: List[Tuple[float, np.ndarray]],
        *,
        t_end: float | None = None,
        record_hist: bool = False,
    ) -> Dict[str, object]:
        """
        Propagate CW/Hill dynamics from t0 to t_end (default TOF_SEC)
        with a *given list of impulses* (t, dv in LVLH-RTN).

        This is used for:
          - re-simulating compressed impulse schedules
          - computing the state at a chosen terminal-burn time.
        """
        cfg = self.cfg
        if t_end is None:
            t_end = self.TOF_SEC

        impulses = sorted(list(impulses), key=lambda x: x[0])
        imp_idx = 0
        n_imp = len(impulses)

        Nsteps = max(1, int((t_end - t0)/cfg.DT_DYN) + 1)
        rho  = np.zeros(3)
        rhod = np.zeros(3)

        if record_hist:
            t_hist   = np.empty(Nsteps)
            sep_hist = np.empty((Nsteps, 3))
            err_hist = np.empty(Nsteps)
            rD_hist  = np.empty((Nsteps, 3))
            vD_hist  = np.empty((Nsteps, 3))
            rA_hist  = np.empty((Nsteps, 3))
            vA_hist  = np.empty((Nsteps, 3))
            rC_hist  = np.empty((Nsteps, 3))
            vC_hist  = np.empty((Nsteps, 3))
            CA_hist  = np.empty((Nsteps, 3, 3))
            CC_hist  = np.empty((Nsteps, 3, 3))
            bRef_hist= np.empty((Nsteps, 3))
            CRN_hist = np.empty((Nsteps, 3, 3))
            q_hist   = np.empty((Nsteps, 4))
            sig_hist = np.empty((Nsteps, 3))

        for j in range(Nsteps):
            t_now = t0 + j*cfg.DT_DYN
            if t_now > t_end + 1e-9:
                break

            # apply any impulses scheduled ~ at this time
            while imp_idx < n_imp and abs(impulses[imp_idx][0] - t_now) <= 0.5*cfg.DT_DYN:
                _, dv = impulses[imp_idx]
                rhod += dv
                imp_idx += 1

            rC, vC = self.chief_state(t_now)
            rA, vA = self.rA_fn(t_now), self.vA_fn(t_now)
            C_C, C_A = dcm_lvlh(rC, vC), dcm_lvlh(rA, vA)
            n = hill_rate(rC, vC)

            rD = rC + C_C.T @ rho
            vD = vC + C_C.T @ (rhod + np.cross([0, 0, n], rho))

            if record_hist:
                t_hist[j]   = t_now
                sep_hist[j] = C_A @ (rD - rA)
                err_hist[j] = la.norm(sep_hist[j] - self.APOPHIS_TARGET)
                rD_hist[j]  = rD
                vD_hist[j]  = vD
                rA_hist[j]  = rA
                vA_hist[j]  = vA
                rC_hist[j]  = rC
                vC_hist[j]  = vC
                CA_hist[j]  = C_A
                CC_hist[j]  = C_C

                los = rA - rD
                los_hat, _ = normalize(los)
                C_RN = triad_from_boresight_and_tangent(los_hat, vD).T
                q    = dcm_to_quat(C_RN)
                sig  = quat_to_mrp(q)
                bRef_hist[j] = los_hat
                CRN_hist[j]  = C_RN
                q_hist[j]    = q
                sig_hist[j]  = sig

            Phi_rr, Phi_rv, Phi_vr, Phi_vv = cw_blocks(n, cfg.DT_DYN)
            rho, rhod = Phi_rr @ rho + Phi_rv @ rhod, Phi_vr @ rho + Phi_vv @ rhod

        # final state at t_end (≈ last step)
        rA_end, vA_end = self.rA_fn(t_end), self.vA_fn(t_end)
        C_A_end = dcm_lvlh(rA_end, vA_end)
        rC_end, vC_end = self.chief_state(t_end)
        C_C_end = dcm_lvlh(rC_end, vC_end)
        rD_end = rC_end + C_C_end.T @ rho
        sep_final = C_A_end @ (rD_end - rA_end)

        res: Dict[str, object] = dict(
            sep_final=sep_final,
            rho_final=rho,
            rhod_final=rhod,
        )
        if record_hist:
            res.update(
                t_hist=t_hist,
                sepA_hist=sep_hist,
                err_hist=err_hist,
                rD_hist=rD_hist,
                vD_hist=vD_hist,
                rA_hist=rA_hist,
                vA_hist=vA_hist,
                rC_hist=rC_hist,
                vC_hist=vC_hist,
                C_A_N_hist=CA_hist,
                C_C_N_hist=CC_hist,
                bRef_N_hist=bRef_hist,
                C_RN_hist=CRN_hist,
                q_RN_hist=q_hist,
                sigma_RN_hist=sig_hist,
            )
        return res

    # ---------- compression + terminal CW burn ----------

    def compress_impulses_with_terminal(
        self,
        target_LVLH_m: np.ndarray,
        t0: float,
        impulses: List[Tuple[float, np.ndarray]],
    ) -> Dict[str, object]:
        """
        Take a RHIG result's impulses, compress them into:
          - a single early “shaping” burn (merged),
          - the last few burns unchanged,
          - a single terminal CW burn near arrival.

        Then re-propagate with this compressed schedule and return
        a full result bundle (histories, error, tot_dv, impulses,...).

        This is the main new trick to:
          - keep number of impulses small, and
          - enforce a tight final error tolerance.
        """
        cfg = self.cfg
        if not impulses:
            # no shaping, only terminal burn
            base_impulses: List[Tuple[float, np.ndarray]] = []
        else:
            impulses = sorted(list(impulses), key=lambda x: x[0])
            # keep last N burns, merge the earlier ones
            keep_last = max(0, min(cfg.COMPRESS_KEEP_LAST, len(impulses)))
            split_idx = len(impulses) - keep_last
            early = impulses[:split_idx]
            late  = impulses[split_idx:]

            base_impulses = []
            if early:
                t_early = early[0][0]
                dv_early = np.zeros(3)
                for _, dv in early:
                    dv_early += dv
                base_impulses.append((t_early, dv_early))
            base_impulses.extend(late)

        # choose terminal burn time
        t_term_nom = self.TOF_SEC - cfg.TERMINAL_MARGIN
        t_term = max(t0, t_term_nom)
        if base_impulses:
            last_pre = max(t for t, _ in base_impulses)
            t_term = max(t_term, last_pre + cfg.DT_DYN)

        # ensure we don't burn after arrival
        if t_term >= self.TOF_SEC - cfg.DT_DYN:
            t_term = self.TOF_SEC - cfg.DT_DYN

        # Propagate shaping burns up to the terminal time
        self.APOPHIS_TARGET = np.asarray(target_LVLH_m, float).copy()
        pre_res = self._propagate_profile(
            t0, base_impulses, t_end=t_term, record_hist=False
        )
        rho_t  = pre_res["rho_final"]
        rhod_t = pre_res["rhod_final"]

        # CW terminal map to TOF_SEC from t_term
        sep_pred, S = self._cw_map_to_target_lvlh(rho_t, rhod_t, t_term)
        eA = sep_pred - self.APOPHIS_TARGET
        dv_term = self._damped_min_norm(S, eA)

        # optional cap on terminal dv
        if cfg.TERMINAL_V_MAX is not None:
            m = la.norm(dv_term)
            if m > cfg.TERMINAL_V_MAX and m > 0:
                dv_term *= cfg.TERMINAL_V_MAX / m

        full_impulses = list(base_impulses) + [(t_term, dv_term)]

        # Re-propagate full compressed schedule all the way to TOF_SEC
        full_res = self._propagate_profile(
            t0, full_impulses, t_end=self.TOF_SEC, record_hist=True
        )
        # terminal metrics
        sep_final = full_res["sep_final"]
        err_final = sep_final - self.APOPHIS_TARGET
        dv_tot = sum(la.norm(dv) for _, dv in full_impulses)

        # attach RHIG-style fields
        full_res.update(
            impulses=full_impulses,
            sep_final=sep_final,
            e_final=err_final,
            err_norm=float(la.norm(err_final)),
            tot_dv=dv_tot,
            dv_times=np.array([t for t, _ in full_impulses]),
            dv_list=np.array([dv for _, dv in full_impulses]) if full_impulses else np.zeros((0, 3)),
            t0=t0,
        )
        return full_res

    # ---------- NPZ export helpers ----------

    @staticmethod
    def _impulse_array(imps: List[Tuple[float, np.ndarray]]) -> np.ndarray:
        return (
            np.asarray([[t, dv[0], dv[1], dv[2]] for t, dv in imps], dtype=float)
            if imps
            else np.zeros((0, 4))
        )

    def export_npz_full(self, path: str, res: Dict[str, object], t0: float, best_t: float):
        np.savez_compressed(
            path,
            t=res["t_hist"],
            rD_N=res["rD_hist"], vD_N=res["vD_hist"],
            rA_N=res["rA_hist"], vA_N=res["vA_hist"],
            rC_N=res["rC_hist"], vC_N=res["vC_hist"],
            C_A_N=res["C_A_N_hist"], C_C_N=res["C_C_N_hist"],
            bRef_N=res["bRef_N_hist"],
            C_RN=res["C_RN_hist"], q_RN=res["q_RN_hist"],
            sigma_RN=res["sigma_RN_hist"],
            dv_times=res["dv_times"],
            dv_list=res["dv_list"],
            impulses=self._impulse_array(res["impulses"]),  # type: ignore
            camMount_B=self.cfg.cam_boresight_B,
            APOPHIS_TARGET=self.APOPHIS_TARGET,
            TARGET_MARGIN_M=np.array([self.cfg.TARGET_MARGIN_M]),
            E_TRIGGER=np.array([self.cfg.E_TRIGGER]),
            V_MIN=np.array([self.cfg.V_MIN]),
            V_MAX=np.array([self.cfg.V_MAX]),
            t0=np.array([t0]),
            best_t=np.array([best_t]),
            err_norm=np.array([res["err_norm"]]),
            tot_dv=np.array([res["tot_dv"]]),
            meta=np.array(
                [
                    self.cfg.departure_utc_iso,
                    self.arrival_dt.isoformat().replace("+00:00", "Z"),
                ],
                dtype=object,
            ),
            final_res=np.array([res], dtype=object),
        )
        print(f"[Export] {Path(path).resolve()}")

    def dump_for_visualize(self, path: str, res: Dict[str, object]):
        save: Dict[str, object] = {
            k: v for k, v in res.items() if isinstance(v, np.ndarray)
        }
        save["impulses"] = self._impulse_array(res["impulses"])  # type: ignore
        for k in ("t0", "best_t", "err_norm", "tot_dv"):
            if k in res:
                save[k] = np.array([res[k]])
        save.update(
            camMount_B=self.cfg.cam_boresight_B,
            APOPHIS_TARGET=self.APOPHIS_TARGET,
            TARGET_MARGIN_M=np.array([self.cfg.TARGET_MARGIN_M]),
            E_TRIGGER=np.array([self.cfg.E_TRIGGER]),
            V_MIN=np.array([self.cfg.V_MIN]),
            V_MAX=np.array([self.cfg.V_MAX]),
            DEPARTURE_UTC_ISO=np.array([self.cfg.departure_utc_iso]),
            ARRIVAL_UTC_ISO=np.array([self.arrival_dt.isoformat().replace("+00:00","Z")]),
        )
        np.savez_compressed(path, **save)
        print(f"[Visualize-dump] {Path(path).resolve()} ({len(save)} arrays)")

    # ---------- fleet driver ----------

    def run_fleet(self) -> List[Tuple[int, np.ndarray, float, Dict[str, object], str | None]]:
        """
        Run RHIG for each target in cfg.fleet_targets_m.

        For each deputy:
          1. Standard RHIG deploy-time search (coarse + 2 refinements).
          2. Final RHIG run with higher authority (FINAL_V_MAX, FINAL_MAX_IMP).
          3. Optional compression + terminal CW burn to reduce the number
             of impulses and enforce a tight final error tolerance.

        Returns a list of (idx, target, best_t, final_result, png_dir).
        """
        cfg = self.cfg
        if cfg.fleet_targets_m is None:
            raise ValueError("cfg.fleet_targets_m must be set to an array of LVLH targets [m].")

        results = []
        print("\n==========  RHIG FLEET RENDEZVOUS  ==========")

        for idx, target in enumerate(cfg.fleet_targets_m):
            print(f"\n===== Deputy {idx} (target {target/1e3} km in LVLH) =====")
            self.APOPHIS_TARGET = np.asarray(target, float).copy()

            png_dir = None if cfg.save_png_dir_base is None else cfg.save_png_dir_base.format(idx)
            viz_npz = f"viz_inputs_D{idx}.npz"
            full_npz = f"rendezvous_full_D{idx}.npz"

            # coarse search
            tmin, tmax = cfg.SEARCH_TMIN, max(0.0, self.TOF_SEC - cfg.SEARCH_TMAX_PAD)
            grid = np.arange(tmin, tmax + 1e-6, cfg.COARSE_STEP)
            best, best_t = None, None
            for td in tqdm(grid, desc=f"Dep {idx} coarse", unit="deploy-t", leave=False):
                r = self.run_rhig_from(td)
                if best is None or self.eval_cost(r) < self.eval_cost(best):
                    best, best_t = r, td

            # refinements
            for step, tag in ((cfg.REFINE_STEP_1, "ref-1"), (cfg.REFINE_STEP_2, "ref-2")):
                lo = max(tmin, best_t - cfg.REFINE_SPAN)
                hi = min(tmax, best_t + cfg.REFINE_SPAN)
                for td in tqdm(np.arange(lo, hi + 1e-6, step),
                               desc=f"Dep {idx} {tag}", unit="deploy-t", leave=False):
                    r = self.run_rhig_from(td)
                    if self.eval_cost(r) < self.eval_cost(best):
                        best, best_t = r, td

            print(f"  ↳ best deploy offset = {best_t:.1f} s  "
                  f"(seed ΔV {best['tot_dv']:.3f} m/s, |e| {best['err_norm']:.2f} m)")

            # final RHIG propagation
            final_raw = self.run_rhig_from(
                best_t, v_max=cfg.FINAL_V_MAX, max_imp=cfg.FINAL_MAX_IMP,
                record_hist=True, show_prog=True, dep_idx=idx
            )
            print(f"  Raw final  |e|={final_raw['err_norm']:.3f} m   "
                  f"ΔV={final_raw['tot_dv']:.6f} m/s   impulses={len(final_raw['impulses'])}")

            final = final_raw

            # optional compression + terminal burn
            if cfg.ENABLE_COMPRESSION:
                compressed = self.compress_impulses_with_terminal(target, final_raw["t0"], final_raw["impulses"])  # type: ignore

                # error tolerance: 5% of ‖target‖, but at least FINAL_ERR_ABS_MIN
                target_norm = la.norm(target)
                err_tol = max(cfg.FINAL_ERR_ABS_MIN, cfg.FINAL_ERR_REL_TOL * target_norm)

                print(f"  Compressed |e|={compressed['err_norm']:.3f} m   "
                      f"ΔV={compressed['tot_dv']:.6f} m/s   "
                      f"impulses={len(compressed['impulses'])}   (tol={err_tol:.2f} m)")

                # if compressed solution still outside tolerance, keep raw;
                # otherwise use compressed as final export.
                if compressed["err_norm"] <= err_tol:  # type: ignore
                    final = compressed
                else:
                    print("  [WARN] compressed profile above tolerance; keeping raw RHIG result.")

            # export NPZs using whichever final profile we selected
            self.export_npz_full(full_npz, final, final["t0"], best_t)  # type: ignore
            self.dump_for_visualize(viz_npz, final)
            results.append((idx, target, best_t, final, png_dir))

        return results
