#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilik_6dof_attitude_minisim_apophis.py
========================================

Purpose
-------
Standalone **6-DOF attitude + camera pointing** demo for a deputy spacecraft
flying an Apophis rendezvous trajectory exported by “Script A”:

- Uses only Basilisk **core modules** and `sys_model_task` — no
  SimulationBaseClass.
- Aligns its internal time base exactly to the NPZ produced by Script A
  (e.g. `apophis_rendezvous_export.npz`).
- Reconstructs a time-varying reference attitude and reference rates
  ω_RN_N from `q_RN(t)` (or TRIAD if `q_RN` isn’t present).
- Adds a small initial misalignment so you see a transient in tracking.
- Includes SimpleNav with Gauss–Markov drift (walkBounds + PMatrix) so nav
  errors evolve over time instead of staying static.
- Logs everything with recorders and builds multiple robust plots at the end.
- Provides a progress bar + timing prints during the run.

Inputs and time base
--------------------
The script reads a Script A-style NPZ, by default:

    NPZ_PATH = os.getenv("APOPHIS_NPZ", "apophis_rendezvous_export.npz")

Expected keys include:

- `t`         : (N,) absolute seconds from departure epoch,
- `rD_N`      : (N,3) deputy ECI positions [m],
- `vD_N`      : (N,3) deputy ECI velocities [m/s],
- `rA_N`, `vA_N` : (N,3) Apophis geocentric ECI states [m], [m/s],
- `bRef_N`    : (N,3) unit LOS from deputy to Apophis in ECI (optional;
                 rebuilt from rA_N − rD_N if missing),
- `q_RN`      : (N,4) scalar-first quaternions N→R (reference frame), optional,
- `camMount_B`: (3,) camera boresight in body frame (default +Z_B if missing).

The script:

- Aligns its timeline by taking:
      t0_abs = t_grid[0]  (first NPZ time),
      t_end  = t_grid[-1] or user-provided T_STOP.
- Uses `t_abs = t0_abs + t_rel` (where t_rel is the sim’s internal time)
  to interpolate NPZ reference data.

Reference attitude and rates
----------------------------
The reference frame “R” is interpreted as the desired body frame (e.g. camera
frame) relative to inertial (N). At each time step:

- If `USE_QRN_FROM_NPZ` and `q_RN` exists:
    * Interpolate q_RN at t_abs, and also at (t_abs − dt) and (t_abs + dt).
    * Convert to DCMs:
          C_prev, C_curr, C_next = quat_to_dcm(q_prev), ...
    * Approximate Ċ_RN:
          Cdot = (C_next − C_prev) / (2 dt).
    * Compute skew-symmetric S_N = Cdot Cᵀ and back out ω_RN_N.

- Else (no q_RN):
    * Rebuild the desired attitude via TRIAD using:
          los_N = bRef_N(t_abs),
          vel_N = vD_N(t_abs),
      to define +Z_B and tangent +X_B, then treat that as “R”.
    * Use zero reference rate (ω_RN_N = 0).

The resulting (σ_RN, ω_RN_N) are written into an `AttRefMsg` each step and
fed into Basilisk’s FSW blocks.

Dynamics and FSW stack
----------------------
- Spacecraft hub:
    * Mass: MASS_KG (default 250 kg),
    * Inertia: diag(IXX, IYY, IZZ) [kg·m²],
    * Initial position/velocity: r_D(t0), v_D(t0) from NPZ,
    * Initial attitude: reference attitude at t0 plus a small yaw offset
      about +Y_B (INIT_Y_DEG).

- Dynamics:
    * `spacecraft.Spacecraft` with an `extForceTorque` effector for torques.
    * No orbit forces are applied; the translational state is initialized from
      NPZ and left unforced (this script is purely about attitude and camera).

- Sensors / Nav:
    * `simpleNav.SimpleNav` fed by `sc.scStateOutMsg`.
    * If `USE_NOISY_NAV` is True:
        - `walkBounds` and `PMatrix` are set so the internal Gauss–Markov
          model produces *drifting* errors for:
              pos, vel, att (MRP), rate, sun-line, ΔV.
        - If supported by the Basilisk build, `crossTrans` and `crossAtt`
          are enabled.

- FSW:
    * `attTrackingError.attTrackingError()`:
         - Receives nav attitude and AttRef,
         - Outputs attitude and rate errors (`sigma_BR`, `omega_BR_B`).
    * `mrpFeedback.mrpFeedback()`:
         - Gains:
               K = K_SIGMA,
               P = P_RATE,
               Ki = KI (set negative to disable integral).
         - integralLimit = I_LIM.
    * Optional `lowPassFilterTorqueCommand`:
         - Enabled when `USE_TORQUE_LPF` is True,
         - h  = LPF_DT,
         - wc = 2π · LPF_CUTOFF_HZ.
    * Torque commands are fed into `extForceTorque`.

Tasks and scheduling
--------------------
No SimulationBaseClass is used. Instead:

- `sys_model_task.SysModelTask` constructs three tasks:

    * sensTask (DT_SENS): SimpleNav and its recorder(s),
    * fswTask  (DT_DYN) : attTrackingError, mrpFeedback, and optional LPF,
    * dynTask  (DT_DYN) : extForceTorque and spacecraft dynamics.

- The script manually calls:
      task.SelfInitTaskList()
      task.ResetTaskList(0)
  for each task before starting the loop.

Main loop
---------
The main loop runs:

    Nstep = floor((T_STOP − t0_abs)/DT_DYN) + 1

At each step:

1. Compute relative and absolute times (t_rel, t_abs).
2. Build reference (σ_RN, ω_RN_N) via `ref_sigma_and_rate(t_abs, DT_DYN)`.
3. Write AttRef message.
4. Execute:
      sensTask.ExecuteTaskList(now_ns)
      fswTask.ExecuteTaskList(now_ns)
      dynTask.ExecuteTaskList(now_ns)

with `now_ns = sec2nano(t_rel)`.

A progress bar (`tqdm` if available, otherwise a simple % logger) tracks
progress and prints total wall-time when finished.

Plots
-----
At the end, the script crops all plots to `T_PLOT_MAX` seconds and generates:

1. Attitude-tracking error (MRP components):
    - σ_BR components vs time, if present.

2. Body-rate error:
    - ω_BR components vs time.

3. Commanded torque:
    - Lx, Ly, Lz from `ctl.cmdTorqueOutMsg`.

4. SimpleNav pointing error norm:
    - Computes nav-based σ_BN_nav and reference σ_RN_nav, then MRPs σ_BR_nav
      and their norm vs time.

5. SimpleNav translation errors:
    - Position/velocity error norms ‖Δr_nav‖, ‖Δv_nav‖ vs time either:
         - Against NPZ truth (`COMPARE_NAV_TO_NPZ = True`), or
         - Against `scLog` (spacecraft “truth” state).

All plots use `matplotlib`, call `plt.tight_layout()`, and finally `plt.show()`.

Usage
-----
1. Generate an Apophis rendezvous NPZ from Script A (e.g.
   `apophis_rendezvous_export.npz`) and place it in the working directory.
2. Optionally set:

       export APOPHIS_NPZ=/path/to/your.npz

3. Adjust user “knobs” at the top (gains, noise levels, INIT_Y_DEG, etc.)
   as desired.

4. Run:

    python basilik_6dof_attitude_minisim_apophis.py

The script will:

- Load the NPZ and align its sim time to that absolute time base,
- Build a one-spacecraft attitude sim with SimpleNav and MRP feedback,
- Generate reference attitudes/rates from q_RN (or TRIAD),
- Run a 6-DOF attitude simulation with noisy nav (if enabled),
- Print progress and wall-time,
- Show the full set of pointing and nav error plots.
"""


from __future__ import annotations
import os, sys, math, time
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# Basilisk core (no SimulationBaseClass)
from Basilisk.architecture import messaging
from Basilisk.architecture import sys_model_task

# FSW
from Basilisk.fswAlgorithms import attTrackingError
from Basilisk.fswAlgorithms import mrpFeedback
from Basilisk.fswAlgorithms import lowPassFilterTorqueCommand

# Dynamics / sim
from Basilisk.simulation import spacecraft
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav

# Utils
from Basilisk.utilities import macros

# ════════════════════════════════════════════════════════════════════════════
# 1) USER “KNOBS”
# ════════════════════════════════════════════════════════════════════════════
NPZ_PATH            = os.getenv("APOPHIS_NPZ", "apophis_rendezvous_export.npz")

# Task periods
DT_DYN   = 0.1     # [s] dynamics + FSW
DT_SENS  = 0.1     # [s] sensors/nav

# Runtime
T_STOP   = None    # [s] None → run to end of NPZ

# Controller (MRP feedback)
K_SIGMA  = 0.08    # [rad/s]
P_RATE   = 0.9     # [N·m·s]
KI       = -1.0    # <0 disables integral term
I_LIM    = 0.01    # [N·m]

# Low‑pass filter on torque (optional)
USE_TORQUE_LPF = True
LPF_CUTOFF_HZ  = 0.8
LPF_DT         = DT_DYN

# Spacecraft inertia / mass
MASS_KG = 250.0
IXX, IYY, IZZ = 18.0, 15.0, 12.0

# ── SimpleNav noise  (NOW WITH DRIFT!) ──────────────────────────────────────
USE_NOISY_NAV = True                  # toggle noise model

# 1‑σ process standard deviations for each nav state
SIGMA_POS   = 50.0      # [m]
SIGMA_VEL   = 0.05      # [m/s]
SIGMA_ATT   = 5e-4      # [MRP]
SIGMA_RATE  = 2e-5      # [rad/s]
SIGMA_SUN   = 1e-3      # sun‑line err (optional)
SIGMA_DV    = 0.0       # ∆V err (unused)

# Initial attitude offset to create a transient (deg about body‑Y)
INIT_Y_DEG = 5.0

# Prefer using reference attitude from NPZ (q_RN) if present
USE_QRN_FROM_NPZ = True

# Compare SimpleNav translation to NPZ deputy truth (else to SC truth)
COMPARE_NAV_TO_NPZ = False

# ════════════════════════════════════════════════════════════════════════════
# 2) HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════
def normalize(v: np.ndarray, eps: float = 1e-15):
    n = la.norm(v)
    return (v / n, n) if n >= eps else (np.zeros_like(v), 0.0)

def triad_from_boresight_and_tangent(los_N: np.ndarray, vel_N: np.ndarray) -> np.ndarray:
    zB, _ = normalize(los_N)
    v_perp = vel_N - np.dot(vel_N, zB) * zB
    xB, nx = normalize(v_perp)
    if nx < 1e-10:  # LOS ≈ VEL → pick arbitrary X‑axis
        tmp = np.array([1., 0., 0.]) if abs(zB[0]) < 0.9 else np.array([0., 1., 0.])
        xB = normalize(np.cross(tmp, zB))[0]
    yB = np.cross(zB, xB)
    xB = normalize(np.cross(yB, zB))[0]
    yB = normalize(np.cross(zB, xB))[0]
    return np.column_stack((xB, yB, zB))  # C_NB

def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    tr = np.trace(C)
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        q0 = 0.25 * S
        q1 = (C[2,1] - C[1,2]) / S
        q2 = (C[0,2] - C[2,0]) / S
        q3 = (C[1,0] - C[0,1]) / S
    else:
        i = int(np.argmax([C[0,0], C[1,1], C[2,2]]))
        if   i == 0:
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

def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = q
    qv = np.array([q1, q2, q3])
    C = (q0*q0 - np.dot(qv, qv)) * np.eye(3) + 2*np.outer(qv, qv) + 2*q0 * np.array([
        [ 0, -q3,  q2],
        [ q3,  0, -q1],
        [-q2, q1,  0]
    ])
    return C

def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    q0 = q[0]; qv = q[1:]
    denom = 1.0 + q0
    if denom < 1e-12:
        denom = 1.0 - q0
        return -qv / max(denom, 1e-12)
    return qv / denom

def mrp_subtract(sigma_BN: np.ndarray, sigma_RN: np.ndarray) -> np.ndarray:
    num = (sigma_BN - sigma_RN) + np.cross(sigma_BN, sigma_RN)
    den = 1.0 + np.dot(sigma_RN, sigma_RN) - np.dot(sigma_BN, sigma_RN)
    return num / den

def safe_field_to_array(rec, name: str, cols: int = 3) -> np.ndarray:
    if not hasattr(rec, name):
        return np.zeros((0, cols))
    data = getattr(rec, name)
    try:
        arr = np.array(data)
    except Exception:
        return np.zeros((0, cols))
    if arr.ndim == 1:
        if arr.size == 0:
            return np.zeros((0, cols))
        if arr.size == cols:
            return arr.reshape(1, cols)
        return arr.reshape(-1, 1)
    return arr

def get_progress_bar(total: int):
    try:
        from tqdm import tqdm
        return tqdm(total=total, unit="step", ncols=80)
    except Exception:
        class _BareBar:                     # minimal fallback
            def __init__(self, N): self.N = N; self.k = 0; self.prev = -1
            def update(self, n=1):
                self.k += n
                pct = int(100*self.k/self.N)
                if pct//5 != self.prev//5:
                    self.prev = pct
                    print(f"[Progress] {pct:3d}% ({self.k}/{self.N})")
            def close(self): print("[Progress] 100% (done)")
        return _BareBar(total)

# ════════════════════════════════════════════════════════════════════════════
# 3) LOAD NPZ TRAJECTORY
# ════════════════════════════════════════════════════════════════════════════
print(f"[INFO] Loading NPZ: {os.path.abspath(NPZ_PATH)}")
if not os.path.exists(NPZ_PATH):
    print("[ERROR] NPZ not found. Run Script A first."); sys.exit(1)

npz = np.load(NPZ_PATH)
t_grid   = npz["t"].astype(float)             # absolute seconds from departure
rD_N_all = npz["rD_N"];  vD_N_all = npz["vD_N"]
rA_N_all = npz["rA_N"];  vA_N_all = npz["vA_N"]
bRef_N_all = (npz["bRef_N"] if "bRef_N" in npz else
              (rA_N_all - rD_N_all) / la.norm(rA_N_all - rD_N_all, axis=1)[:, None])
cam_b_B   = npz["camMount_B"] if "camMount_B" in npz else np.array([0., 0., 1.])

use_qRN = ("q_RN" in npz) and USE_QRN_FROM_NPZ
if use_qRN:
    qRN_all = npz["q_RN"]  # (N,4) scalar‑first
    print("[INFO] Using q_RN (N→R) from NPZ for AttRef.")
else:
    print("[INFO] Rebuilding AttRef from bRef_N + vD_N using TRIAD.")

t0_abs   = float(t_grid[0])
t1_abs   = float(t_grid[-1])
T_STOP   = min(T_STOP or t1_abs, t1_abs)
print(f"[INFO] Time base (absolute): t0={t0_abs:.1f}s → tEnd={T_STOP:.1f}s  (Δ={T_STOP - t0_abs:.1f}s)")
print(f"[INFO] Boresight axis (body‑frame): {cam_b_B}")

# ════════════════════════════════════════════════════════════════════════════
# 4) BUILD MODULES
# ════════════════════════════════════════════════════════════════════════════
print("[INFO] Building modules …")

# ── Spacecraft hub ──────────────────────────────────────────────────────────
sc = spacecraft.Spacecraft();  sc.ModelTag = "Deputy"
sc.hub.mHub        = MASS_KG
sc.hub.IHubPntBc_B = [[IXX, 0, 0], [0, IYY, 0], [0, 0, IZZ]]
sc.hub.r_BcB_B     = [[0.0], [0.0], [0.0]]

# Initial attitude = desired at t0 with small misalignment about +Y_B
if use_qRN:
    C_RN0  = quat_to_dcm(qRN_all[0])
    C_NB0  = C_RN0.T
else:
    C_NB0  = triad_from_boresight_and_tangent(bRef_N_all[0], vD_N_all[0])

theta   = math.radians(INIT_Y_DEG)
R_off   = np.array([[ math.cos(theta), 0.0,  math.sin(theta)],
                    [ 0.0           , 1.0,  0.0           ],
                    [-math.sin(theta), 0.0,  math.cos(theta)]])
C_NB0   = C_NB0 @ R_off
sigma0  = quat_to_mrp(dcm_to_quat(C_NB0))

sc.hub.sigma_BNInit   = [[float(sigma0[0])], [float(sigma0[1])], [float(sigma0[2])]]
sc.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]

# Initialize translation to NPZ at t0 (we don’t propagate orbit forces here)
sc.hub.r_CN_NInit = list(rD_N_all[0])
sc.hub.v_CN_NInit = list(vD_N_all[0])

# ── External torque effector ────────────────────────────────────────────────
extFT = extForceTorque.ExtForceTorque();  extFT.ModelTag = "extFT"

# ── SimpleNav (with new drift) ──────────────────────────────────────────────
nav = simpleNav.SimpleNav();             nav.ModelTag = "simpleNav"
if USE_NOISY_NAV:
    # Build 18‑×‑1 walkBounds (3‑σ)                                   (pos vel att rate sun ΔV)
    walkBounds = np.hstack([3*SIGMA_POS*np.ones(3),
                            3*SIGMA_VEL*np.ones(3),
                            3*SIGMA_ATT*np.ones(3),
                            3*SIGMA_RATE*np.ones(3),
                            3*SIGMA_SUN*np.ones(3),
                            3*SIGMA_DV*np.ones(3)])
    # Build √P (Cholesky of diagonal covariance)
    sigma_vec = np.hstack([SIGMA_POS*np.ones(3),
                           SIGMA_VEL*np.ones(3),
                           SIGMA_ATT*np.ones(3),
                           SIGMA_RATE*np.ones(3),
                           SIGMA_SUN*np.ones(3),
                           SIGMA_DV*np.ones(3)])
    PMatrix = np.diag(sigma_vec**2)**0.5      # diagonal ⇒ √P
    nav.walkBounds = walkBounds
    nav.PMatrix    = PMatrix
    # Optional cross‑correlation switches (guard for older Basilisk builds)
    for attr in ("crossTrans", "crossAtt"):
        if hasattr(nav, attr):
            setattr(nav, attr, True)

# ── FSW blocks ──────────────────────────────────────────────────────────────
attErr = attTrackingError.attTrackingError();  attErr.ModelTag = "attError"

ctl     = mrpFeedback.mrpFeedback();           ctl.ModelTag = "mrpFB"
ctl.K   = K_SIGMA
ctl.P   = P_RATE
ctl.Ki  = KI
ctl.integralLimit = I_LIM

if USE_TORQUE_LPF:
    lpf     = lowPassFilterTorqueCommand.lowPassFilterTorqueCommand()
    lpf.ModelTag = "lpfTorque"
    lpf.h  = float(LPF_DT)
    lpf.wc = float(2*math.pi*LPF_CUTOFF_HZ)

# ── Static messages ─────────────────────────────────────────────────────────
vehCfgPayload           = messaging.VehicleConfigMsgPayload()
vehCfgPayload.ISCPntB_B = [IXX,0,0, 0,IYY,0, 0,0,IZZ]
vehCfgMsg               = messaging.VehicleConfigMsg().write(vehCfgPayload)

attRefMsg = messaging.AttRefMsg()   # written each step

# ════════════════════════════════════════════════════════════════════════════
# 5) WIRING MESSAGES
# ════════════════════════════════════════════════════════════════════════════
print("[INFO] Wiring message connections …")

sc.addDynamicEffector(extFT)

nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)

attErr.attNavInMsg.subscribeTo(nav.attOutMsg)    # nav attitude → error
attErr.attRefInMsg.subscribeTo(attRefMsg)        # reference → error

ctl.guidInMsg    .subscribeTo(attErr.attGuidOutMsg)
ctl.vehConfigInMsg.subscribeTo(vehCfgMsg)

if USE_TORQUE_LPF:
    lpf.cmdTorqueInMsg  .subscribeTo(ctl.cmdTorqueOutMsg)
    extFT.cmdTorqueInMsg.subscribeTo(lpf.cmdTorqueOutMsg)
else:
    extFT.cmdTorqueInMsg.subscribeTo(ctl.cmdTorqueOutMsg)

# ════════════════════════════════════════════════════════════════════════════
# 6) TASKS & RECORDERS
# ════════════════════════════════════════════════════════════════════════════
print("[INFO] Creating SysModelTasks & recorders …")
ns = lambda s: int(macros.sec2nano(s))   # helper

dynTask  = sys_model_task.SysModelTask(ns(DT_DYN))
fswTask  = sys_model_task.SysModelTask(ns(DT_DYN))
sensTask = sys_model_task.SysModelTask(ns(DT_SENS))

sensTask.AddNewObject(nav,  10)
fswTask .AddNewObject(attErr, 20); fswTask.AddNewObject(ctl, 10)
if USE_TORQUE_LPF: fswTask.AddNewObject(lpf, 5)
dynTask .AddNewObject(extFT, 10); dynTask.AddNewObject(sc, 5)

# ── Recorders (create ONCE; reuse after run) ────────────────────────────────
errLog  = attErr.attGuidOutMsg.recorder()
navALog = nav.attOutMsg       .recorder()
navTLog = nav.transOutMsg     .recorder()
scLog   = sc.scStateOutMsg    .recorder()
ctlLog  = ctl.cmdTorqueOutMsg .recorder()

fswTask .AddNewObject(errLog,  -1)
sensTask.AddNewObject(navALog, -1)
sensTask.AddNewObject(navTLog, -1)
dynTask .AddNewObject(scLog,   -1)
fswTask .AddNewObject(ctlLog,  -1)

# Initialize tasks
for task in (sensTask, fswTask, dynTask):
    task.SelfInitTaskList()
    task.ResetTaskList(0)

# ════════════════════════════════════════════════════════════════════════════
# 7) HELPER — REFERENCE ATTITUDE SERIES
# ════════════════════════════════════════════════════════════════════════════
def interp_series_abs(t_ref: np.ndarray, arr: np.ndarray, t_abs: float) -> np.ndarray:
    if t_abs <= t_ref[0] : return arr[0]
    if t_abs >= t_ref[-1]: return arr[-1]
    j = np.searchsorted(t_ref, t_abs) - 1
    f = (t_abs - t_ref[j]) / (t_ref[j+1] - t_ref[j])
    return (1.0 - f) * arr[j] + f * arr[j+1]

def ref_sigma_and_rate(t_abs: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build σ_RN and ω_RN_N at absolute time t_abs.
    • If q_RN exists → central‑difference for rate
    • Else rebuilt via TRIAD with zero rate
    """
    if use_qRN:
        t_prev = max(t_grid[0], t_abs - dt)
        t_next = min(t_grid[-1], t_abs + dt)
        q_prev = interp_series_abs(t_grid, qRN_all, t_prev)
        q_curr = interp_series_abs(t_grid, qRN_all, t_abs)
        q_next = interp_series_abs(t_grid, qRN_all, t_next)
        C_prev = quat_to_dcm(q_prev); C_curr = quat_to_dcm(q_curr)
        C_next = quat_to_dcm(q_next)
        Cdot   = (C_next - C_prev) / max(2*dt, 1e-6)
        S_N    = Cdot @ C_curr.T
        omega_RN_N = -np.array([S_N[2,1]-S_N[1,2], S_N[0,2]-S_N[2,0], S_N[1,0]-S_N[0,1]]) * 0.5
        sigma_RN   = quat_to_mrp(q_curr)
        return sigma_RN, omega_RN_N
    else:
        rD   = interp_series_abs(t_grid, rD_N_all, t_abs)
        vD   = interp_series_abs(t_grid, vD_N_all, t_abs)
        bN   = interp_series_abs(t_grid, bRef_N_all, t_abs)
        C_NB = triad_from_boresight_and_tangent(bN, vD)
        q_RN = dcm_to_quat(C_NB.T)
        return quat_to_mrp(q_RN), np.zeros(3)

# ════════════════════════════════════════════════════════════════════════════
# 8) MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════
Nstep   = int(math.floor((T_STOP - t0_abs)/DT_DYN)) + 1
print(f"[RUN ] Main loop aligned to NPZ epoch: steps={Nstep}, dt={DT_DYN}s")

bar  = get_progress_bar(Nstep)
wall0 = time.time()

for k in range(Nstep):
    t_rel = k * DT_DYN
    t_abs = t0_abs + t_rel
    now_ns = int(macros.sec2nano(t_rel))

    # ── NEW: get reference attitude and rate ──
    σ_RN, ω_RN_N = ref_sigma_and_rate(t_abs, DT_DYN)   # ← add this line

    # update AttRef message
    ref = messaging.AttRefMsgPayload()       # blank constructor
    ref.sigma_RN     = list(σ_RN)
    ref.omega_RN_N   = list(ω_RN_N)
    ref.domega_RN_N  = [0.0, 0.0, 0.0]
    attRefMsg.write(ref)

    # execute tasks …
    sensTask.ExecuteTaskList(now_ns)
    fswTask .ExecuteTaskList(now_ns)
    dynTask .ExecuteTaskList(now_ns)

    bar.update(1)

bar.close()
print(f"[DONE] Sim wall‑time: {time.time()-wall0:.2f}s")

# ════════════════════════════════════════════════════════════════════════════
# 9) PLOTS  –– ​truncate all figures at T_PLOT_MAX seconds
# ════════════════════════════════════════════════════════════════════════════
print("[INFO] Creating plots …")

T_PLOT_MAX = 7530.0                     # [s] hard cut-off for every plot

def crop(t: np.ndarray, *arrs: np.ndarray):
    """Keep only samples with t ≤ T_PLOT_MAX and return (t, *cropped arrays)."""
    keep = t <= T_PLOT_MAX
    return (t[keep],) + tuple(a[keep] for a in arrs)

# ── Attitude-tracking error ────────────────────────────────────────────────
t_err = errLog.times() * macros.NANO2SEC
σ_BR   = safe_field_to_array(errLog, "sigma_BR",   3)
ω_BR   = safe_field_to_array(errLog, "omega_BR_B", 3)
t_err, σ_BR, ω_BR = crop(t_err, σ_BR, ω_BR)

plt.figure(figsize=(10,5))
if σ_BR.size: plt.plot(t_err, σ_BR); plt.legend(['σ₁','σ₂','σ₃'])
plt.xlim(0, T_PLOT_MAX)
plt.xlabel('Time [s]'); plt.ylabel('MRP Error [–]')
plt.title('Attitude Tracking Error'); plt.grid(True)

plt.figure(figsize=(10,5))
if ω_BR.size: plt.plot(t_err, ω_BR); plt.legend(['ω₁','ω₂','ω₃'])
plt.xlim(0, T_PLOT_MAX)
plt.xlabel('Time [s]'); plt.ylabel('Rate Error [rad/s]')
plt.title('Body-Rate Error'); plt.grid(True)

# ── Commanded torque ───────────────────────────────────────────────────────
u_t = ctlLog.times() * macros.NANO2SEC
u_b = safe_field_to_array(ctlLog, "torqueRequestBody", 3)
u_t, u_b = crop(u_t, u_b)

plt.figure(figsize=(10,5))
if u_b.size: plt.plot(u_t, u_b); plt.legend(['Lx','Ly','Lz'])
plt.xlim(0, T_PLOT_MAX)
plt.xlabel('Time [s]'); plt.ylabel('Torque [N·m]')
plt.title('Commanded Torque'); plt.grid(True)

# ── SimpleNav-based pointing error norm ────────────────────────────────────
t_nav = navALog.times() * macros.NANO2SEC
σ_BN_nav = safe_field_to_array(navALog, "sigma_BN_B", 3)
if not σ_BN_nav.size:
    σ_BN_nav = safe_field_to_array(navALog, "sigma_BN", 3)

t_nav, σ_BN_nav = crop(t_nav, σ_BN_nav)

if t_nav.size:
    σ_RN_nav = np.vstack([ref_sigma_and_rate(t0_abs + τ, DT_DYN)[0] for τ in t_nav])
    σ_BR_nav = np.array([mrp_subtract(σ_BN_nav[i], σ_RN_nav[i])
                         for i in range(len(t_nav))])
    plt.figure(figsize=(10,5))
    plt.plot(t_nav, la.norm(σ_BR_nav, axis=1))
    plt.xlim(0, T_PLOT_MAX)
    plt.xlabel('Time [s]'); plt.ylabel('‖σ_BR(nav)‖ [–]')
    plt.title('SimpleNav Pointing Error Norm'); plt.grid(True)

# ── SimpleNav translational errors ─────────────────────────────────────────
t_tr = navTLog.times() * macros.NANO2SEC
r_nav = safe_field_to_array(navTLog, "r_BN_N", 3)
v_nav = safe_field_to_array(navTLog, "v_BN_N", 3)
t_tr, r_nav, v_nav = crop(t_tr, r_nav, v_nav)

if t_tr.size:
    if COMPARE_NAV_TO_NPZ:
        r_ref = np.vstack([interp_series_abs(t_grid, rD_N_all, t0_abs + τ) for τ in t_tr])
        v_ref = np.vstack([interp_series_abs(t_grid, vD_N_all, t0_abs + τ) for τ in t_tr])
        suffix = " vs NPZ"
    else:
        t_true = scLog.times() * macros.NANO2SEC
        r_true = safe_field_to_array(scLog, "r_BN_N", 3)
        v_true = safe_field_to_array(scLog, "v_BN_N", 3)
        if t_true.size and r_true.shape[0] == t_true.size:
            r_ref = np.column_stack([np.interp(t_tr, t_true, r_true[:,j]) for j in range(3)])
            v_ref = np.column_stack([np.interp(t_tr, t_true, v_true[:,j]) for j in range(3)])
        else:
            r_ref = np.zeros_like(r_nav); v_ref = np.zeros_like(v_nav)
        suffix = " vs SC truth"

    e_r = r_nav - r_ref
    e_v = v_nav - v_ref

    plt.figure(figsize=(10,5))
    plt.plot(t_tr, la.norm(e_r, axis=1))
    plt.xlim(0, T_PLOT_MAX)
    plt.xlabel('Time [s]'); plt.ylabel('‖Δr_nav‖ [m]')
    plt.title(f'SimpleNav Position Error Norm{suffix}'); plt.grid(True)

    plt.figure(figsize=(10,5))
    plt.plot(t_tr, la.norm(e_v, axis=1))
    plt.xlim(0, T_PLOT_MAX)
    plt.xlabel('Time [s]'); plt.ylabel('‖Δv_nav‖ [m/s]')
    plt.title(f'SimpleNav Velocity Error Norm{suffix}'); plt.grid(True)

plt.tight_layout()
print("[INFO] Showing plots …")
plt.show()
print("[INFO] Finished.")

