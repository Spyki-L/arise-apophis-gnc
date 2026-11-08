#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
guidance_v2.py  -  ARISE Monte Carlo **V2** fleet guidance
==========================================================
High-fidelity TRANSLATIONAL guidance (reused, proven) + **full 6-DOF attitude/GNC**
+ **per-draw emission** of the attitude / control time series.

Design
------
* The translational guidance is NOT re-derived.  V2 imports the proven Design-D
  script `basilik_hifi_apophis_rendezvous_fleet.py` as a library and runs its
  `main()` (LIVE-SPICE Basilisk force model: Earth SH J2..10 + Sun + Moon + SRP,
  RKF45, Lambert-seeded differential-correction TCMs).  That writes the 8 deputy
  trajectory NPZs (same schema as V1) into `<draw>/trajectories/`.
* V2 ADDS, per deputy, a real **6-DOF attitude-control** pass that tracks the
  camera->Apophis pointing reference with **XB6 XACT reaction wheels**:
    - flight path  : a Basilisk reaction-wheel MRP-feedback stack (simpleNav ->
                     attTrackingError -> mrpFeedback -> rwMotorTorque ->
                     reactionWheelStateEffector), driven by the precomputed q_RN(t).
    - verified path: a numpy wheel-momentum/rate-LIMITED follower (unit-tested
                     here).  Used automatically if Basilisk attitude modules are
                     absent (or ARISE_V2_ATT_BACKEND=kinematic).  Because the bus
                     cannot slew at the ~v/b (>10 rad/s) line-of-sight rate near
                     closest approach, the body LAGS and the fast-steering mirror
                     (+/-2 deg) covers the residual; this is modeled explicitly so
                     the emitted attitude/pointing numbers stay physical.
* V2 ADDS a finite-burn **control** emission using the MPS-130 model in
  `modules/thruster.py` (misalignment, MIB quantization, finite-burn loss, propellant).

Per-draw output tree (honors ARISE_OUT_DIR = the draw root):
    <draw>/trajectories/  apophis_rendezvous_full_D*.npz   (base guidance)
    <draw>/attitude/      att_D*.npz   q_BN(t), omega(t), wheel speeds/torque, pointing-err(t), fsm_residual(t)
    <draw>/control/       ctrl_D*.npz  per-burn cmd/realized dV, MIB, burn duration, propellant
    <draw>/guidance_v2_meta.json

Env: ARISE_OUT_DIR, ARISE_APOPHIS_STATE_PERTURB (base guidance), ARISE_V2_ATT_BACKEND, ARISE_V2_CTRL_BW_HZ
Run (Basilisk env): python guidance_v2.py     Offline test: python guidance_v2.py --self-test
"""
from __future__ import annotations
import os, sys, glob, json, math, time, argparse
import numpy as np
import numpy.linalg as la

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = os.path.join(HERE, "modules")
DESIGN = os.path.dirname(HERE)
GUID_DIR = os.path.join(DESIGN, "scripts", "guidance")
for _p in (MODULES, GUID_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hardware as HW
try:
    import thruster as THR
except Exception:
    THR = None

ATT_BACKEND = os.getenv("ARISE_V2_ATT_BACKEND", "kinematic").lower()  # robust default; basilisk RW path is opt-in
STARTRACKER_HIFI = os.getenv("ARISE_STARTRACKER_HIFI", "0") not in ("0", "", "false", "False")


def _startracker_extra(omega_BN_B):
    """Rate-dependent XACT-50 star-tracker knowledge sigma [rad] + availability along the attitude
    profile.  Returns {} (identity-when-absent) unless ARISE_STARTRACKER_HIFI is enabled."""
    if not STARTRACKER_HIFI:
        return {}
    try:
        import star_tracker as _ST
    except Exception:
        return {}
    wmag = la.norm(np.atleast_2d(np.asarray(omega_BN_B, float)), axis=1)
    sig = np.array([_ST.active().pointing_sigma_rad(float(w)) for w in wmag])
    avail = np.array([float(_ST.active().sigma_axes(float(w))["available"]) for w in wmag])
    return dict(st_pointing_sigma_rad=sig, st_available=avail)
CTRL_BW_HZ  = float(os.getenv("ARISE_V2_CTRL_BW_HZ", "0.05"))
CAM_BORE_B  = np.array([0.0, 0.0, 1.0])
I_WHEEL     = 4.0e-5                         # [kg m^2] XACT-50 wheel (H~25 mNms @ ~625 rad/s)
WHEEL_H_CAP = HW.STAR_TRACKER.momentum_mNms_per_axis * 1e-3 / 2.0   # 25 mNms/wheel (XACT-50: 50 mNms/axis, 2 wheels/axis)
FSM_THROW_RAD = math.radians(2.0)            # fast-steering-mirror half-throw
ARCSEC      = math.radians(1.0 / 3600.0)
RW_HIFI     = os.getenv("ARISE_RW_HIFI", "0") not in ("0", "", "false", "False")


def _rw_hifi(wheel_speed_rads, wheel_torque_Nm, point_err_rad):
    """Hi-fi reaction-wheel model along the profile: motor-limited (realized) wheel torque +
    microvibration jitter folded into the pointing error.  Returns (wheel_torque, point_err,
    extra).  Identity-when-absent unless ARISE_RW_HIFI is enabled."""
    if not RW_HIFI:
        return wheel_torque_Nm, point_err_rad, {}
    try:
        import reaction_wheels as _RW
    except Exception:
        return wheel_torque_Nm, point_err_rad, {}
    wt = _RW.RWA.realized_torque(wheel_torque_Nm, wheel_speed_rads)          # motor saturation + friction
    ws = np.atleast_2d(np.asarray(wheel_speed_rads, float))
    jit = np.array([_RW.RWA.fleet_jitter_arcsec(row) * ARCSEC for row in ws])  # rad, per frame (fleet RSS)
    pe = np.sqrt(np.asarray(point_err_rad, float) ** 2 + jit ** 2)           # microvibration -> pointing error
    return wt, pe, dict(rw_jitter_rad=jit)


# ════════════════════════════════════════════════════════════════════════════
# quaternion / DCM helpers (scalar-first; self-contained)
# ════════════════════════════════════════════════════════════════════════════
def _normalize(v, eps=1e-12):
    n = la.norm(v)
    return v / n if n >= eps else v * 0.0

def _dcm_to_quat(C):
    tr = np.trace(C)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0; q0 = 0.25 * S
        q1 = (C[2, 1] - C[1, 2]) / S; q2 = (C[0, 2] - C[2, 0]) / S; q3 = (C[1, 0] - C[0, 1]) / S
    else:
        i = int(np.argmax([C[0, 0], C[1, 1], C[2, 2]]))
        if i == 0:
            S = math.sqrt(1 + C[0, 0] - C[1, 1] - C[2, 2]) * 2
            q0 = (C[2, 1] - C[1, 2]) / S; q1 = 0.25 * S; q2 = (C[0, 1] + C[1, 0]) / S; q3 = (C[0, 2] + C[2, 0]) / S
        elif i == 1:
            S = math.sqrt(1 + C[1, 1] - C[0, 0] - C[2, 2]) * 2
            q0 = (C[0, 2] - C[2, 0]) / S; q1 = (C[0, 1] + C[1, 0]) / S; q2 = 0.25 * S; q3 = (C[1, 2] + C[2, 1]) / S
        else:
            S = math.sqrt(1 + C[2, 2] - C[0, 0] - C[1, 1]) * 2
            q0 = (C[1, 0] - C[0, 1]) / S; q1 = (C[0, 2] + C[2, 0]) / S; q2 = (C[1, 2] + C[2, 1]) / S; q3 = 0.25 * S
    q = np.array([q0, q1, q2, q3]); return q / la.norm(q)

def _quat_to_dcm(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2*q2 + q3*q3),     2*(q1*q2 + q0*q3),     2*(q1*q3 - q0*q2)],
        [    2*(q1*q2 - q0*q3), 1 - 2*(q1*q1 + q3*q3),     2*(q2*q3 + q0*q1)],
        [    2*(q1*q3 + q0*q2),     2*(q2*q3 - q0*q1), 1 - 2*(q1*q1 + q2*q2)]])

def _quat_mul(a, b):
    a0, a1, a2, a3 = a; b0, b1, b2, b3 = b
    return np.array([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0])

def _quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def _quat_propagate(q, omega_body, dt):
    """Advance body attitude by a body-frame rate over dt (right-multiply increment)."""
    th = la.norm(omega_body) * dt
    if th < 1e-12:
        dq = np.array([1.0, 0.5*omega_body[0]*dt, 0.5*omega_body[1]*dt, 0.5*omega_body[2]*dt])
    else:
        ax = omega_body / la.norm(omega_body)
        dq = np.array([math.cos(th/2), *(math.sin(th/2) * ax)])
    out = _quat_mul(q, dq); return out / la.norm(out)

def _triad(los, vel):
    zB = _normalize(los)
    v_perp = vel - np.dot(vel, zB) * zB
    xB = _normalize(v_perp)
    if la.norm(xB) < 1e-8:
        tmp = np.array([1., 0, 0]) if abs(zB[0]) < .9 else np.array([0, 1., 0])
        xB = _normalize(np.cross(tmp, zB))
    yB = _normalize(np.cross(zB, xB)); xB = _normalize(np.cross(yB, zB))
    return np.column_stack((xB, yB, zB))


# ════════════════════════════════════════════════════════════════════════════
# 6-DOF attitude — VERIFIED numpy wheel-momentum/rate-limited follower
# ════════════════════════════════════════════════════════════════════════════
def attitude_kinematic(t, C_RN, bRef_N, bus: HW.XB6Bus = None, ctrl_bw_hz=CTRL_BW_HZ):
    """Closed-loop attitude TRUTH: a reaction-wheel-momentum/rate-LIMITED follower of the
    precomputed reference C_RN(t) (maps N->body).  Away from closest approach the LOS rate is
    below the wheel-limited body rate (~0.16 rad/s) so the body tracks to the XACT accuracy
    floor (~10 arcsec).  Near CA the LOS rate reaches ~v/b (>10 rad/s); the body saturates and
    a slew LAG accumulates (and decays as the wheels catch up).  The fast-steering mirror
    (+/-2 deg) covers the small residual; `fsm_residual_rad` flags whatever is beyond the FSM
    throw (the sub-second near-CA frame loss).  All numbers stay physical.
    """
    bus = bus or HW.XB6
    t = np.asarray(t, float); N = len(t)
    C_RN = np.asarray(C_RN, float); bRef_N = np.asarray(bRef_N, float)
    Idiag = bus.inertia_diag(); I = np.diag(Idiag)

    # ideal reference body rate (finite diff of C_RN; spikes toward ~v/b at CA)
    omega_ref = np.zeros((N, 3))
    for j in range(N):
        jm, jp = max(0, j - 1), min(N - 1, j + 1); dt = t[jp] - t[jm]
        if dt <= 0:
            continue
        Cd = (C_RN[jp] - C_RN[jm]) / dt
        Wsk = -Cd @ C_RN[j].T
        omega_ref[j] = 0.5 * np.array([Wsk[2, 1] - Wsk[1, 2], Wsk[0, 2] - Wsk[2, 0], Wsk[1, 0] - Wsk[0, 1]])
    omega_ref_mag = la.norm(omega_ref, axis=1)

    # wheel-momentum-limited maximum body rate (XACT-class 4-wheel pyramid)
    H_axis = 2.0 * WHEEL_H_CAP                        # XACT-50: 2 wheels project per axis -> 0.050 N m s
    omega_max = min(H_axis / float(Idiag.max()),      # raw wheel-saturation cap (~12.6 deg/s at this inertia)...
                    np.radians(HW.STAR_TRACKER.max_slew_deg_s))   # ...held to the rated 10 deg/s slew (tracker stays locked, no dropout)

    # actual body rate = reference direction, capped at omega_max
    omega_act = omega_ref.copy()
    for j in range(N):
        if omega_ref_mag[j] > omega_max and omega_ref_mag[j] > 0:
            omega_act[j] = omega_ref[j] * (omega_max / omega_ref_mag[j])

    # integrate body attitude from the rate-limited rate (start aligned to the reference)
    q = _dcm_to_quat(C_RN[0]); q_BN = np.zeros((N, 4))
    for j in range(N):
        q_BN[j] = q
        if j < N - 1:
            q = _quat_propagate(q, omega_act[j], t[j + 1] - t[j])

    # pointing error: XACT accuracy floor + accumulated slew lag during wheel saturation
    adcs_acc = bus.pointing_accuracy_arcsec * ARCSEC
    point_err = np.zeros(N); fsm_res = np.zeros(N)
    lag = 0.0
    for j in range(N):
        dt = t[j] - t[j - 1] if j > 0 else 0.0
        if omega_ref_mag[j] > omega_max:
            lag += (omega_ref_mag[j] - omega_max) * dt          # body falls behind the LOS
        else:
            lag = max(0.0, lag - (omega_max - omega_ref_mag[j]) * dt)   # wheels catch up
        point_err[j] = min(math.pi, math.sqrt(adcs_acc**2 + lag**2))
        fsm_res[j] = max(0.0, point_err[j] - FSM_THROW_RAD)

    # reaction-wheel momentum (gyrostat holds -I*omega) and required wheel torque
    hw_body = -(I @ omega_act.T).T
    beta = math.radians(54.7356)
    W4 = np.array([[math.cos(beta), 0., -math.cos(beta), 0.],
                   [0., math.cos(beta), 0., -math.cos(beta)],
                   [math.sin(beta), math.sin(beta), math.sin(beta), math.sin(beta)]])
    Wp = W4.T @ la.inv(W4 @ W4.T)
    wheel_speed = (Wp @ hw_body.T).T / I_WHEEL
    Lc = np.gradient(hw_body, t, axis=0) + np.cross(omega_act, (I @ omega_act.T).T)
    wheel_torque = (Wp @ Lc.T).T
    wheel_torque, point_err, rw_extra = _rw_hifi(wheel_speed, wheel_torque, point_err)
    st_extra = _startracker_extra(omega_act)
    return dict(t=t, q_BN=q_BN, omega_BN_B=omega_act, wheel_speed_rads=wheel_speed,
                wheel_torque_Nm=wheel_torque, ctrl_torque_Nm=Lc,
                pointing_err_rad=point_err, fsm_residual_rad=fsm_res,
                los_rate_rps=omega_ref_mag, omega_ref_rps=omega_ref, backend="kinematic", **st_extra, **rw_extra)


# ════════════════════════════════════════════════════════════════════════════
# 6-DOF attitude — Basilisk reaction-wheel flight path (runs in the Basilisk env)
# ════════════════════════════════════════════════════════════════════════════
def attitude_basilisk(d):
    """Full Basilisk RW attitude-control sim tracking q_RN(t).  Stack: spacecraft(hub) + 4 RW
    pyramid + simpleNav -> attTrackingError -> mrpFeedback -> rwMotorTorque ->
    reactionWheelStateEffector.  Same return schema as attitude_kinematic.  Raises on any
    missing piece so the caller falls back to the verified kinematic follower."""
    from Basilisk.utilities import SimulationBaseClass, macros, simIncludeRW, RigidBodyKinematics as rbk
    from Basilisk.simulation import spacecraft, reactionWheelStateEffector, simpleNav
    from Basilisk.fswAlgorithms import attTrackingError, mrpFeedback, rwMotorTorque
    from Basilisk.architecture import sysModel, messaging, bskLogging
    bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)

    t = np.asarray(d["t"], float); C_RN = np.asarray(d["C_RN"], float)
    bRef_N = np.asarray(d["bRef_N"], float)
    N = len(t); t0 = float(t[0]); dur = float(t[-1] - t[0])
    dt = float(np.clip(np.min(np.diff(t)) if N > 1 else 0.05, 1e-3, 0.05))
    sigRN = np.array([rbk.C2MRP(C_RN[j]) for j in range(N)])
    Idiag = HW.XB6.inertia_diag()

    sim = SimulationBaseClass.SimBaseClass(); sim.SetProgressBar(False)
    proc = sim.CreateNewProcess("att"); proc.addTask(sim.CreateNewTask("att", macros.sec2nano(dt)))

    sc = spacecraft.Spacecraft(); sc.ModelTag = "deputy6dof"
    sc.hub.mHub = HW.XB6.wet_mass_kg
    sc.hub.IHubPntBc_B = [[float(Idiag[0]), 0., 0.], [0., float(Idiag[1]), 0.], [0., 0., float(Idiag[2])]]
    sc.hub.sigma_BNInit = [[float(sigRN[0][0])], [float(sigRN[0][1])], [float(sigRN[0][2])]]
    sim.AddModelToTask("att", sc)

    rwFactory = simIncludeRW.rwFactory(); beta = 54.75
    axes = [[math.cos(math.radians(beta)), 0.0, math.sin(math.radians(beta))],
            [0.0, math.cos(math.radians(beta)), math.sin(math.radians(beta))],
            [-math.cos(math.radians(beta)), 0.0, math.sin(math.radians(beta))],
            [0.0, -math.cos(math.radians(beta)), math.sin(math.radians(beta))]]
    for a in axes:
        # XB6/XACT-class small wheel (~15 mNms, ~4 mNm) — NOT the big Honeywell HR12 preset (12-50 Nms).
        try:
            rwFactory.create('BCT_RWP015', a, Omega=0.0)                 # Blue Canyon RWP015 (15 mNms) if present
        except Exception:
            rwFactory.create('custom', a, Js=I_WHEEL, Omega=0.0, maxMomentum=2.0*WHEEL_H_CAP, u_max=0.004)
    numRW = rwFactory.getNumOfDevices()
    rwSE = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwFactory.addToSpacecraft("rw", rwSE, sc); sim.AddModelToTask("att", rwSE, 6)
    rwConfigMsg = rwFactory.getConfigMessage()

    sNav = simpleNav.SimpleNav(); sNav.ModelTag = "nav"
    sNav.scStateInMsg.subscribeTo(sc.scStateOutMsg); sim.AddModelToTask("att", sNav)

    class _Ref(sysModel.SysModel):
        def __init__(s):
            super().__init__(); s.attRefOutMsg = messaging.AttRefMsg()
        def UpdateState(s, nanos):
            tt = nanos * 1.0e-9 + t0
            sig = np.array([np.interp(tt, t, sigRN[:, k]) for k in range(3)])
            P = messaging.AttRefMsgPayload()
            P.sigma_RN = sig.tolist(); P.omega_RN_N = [0., 0., 0.]; P.domega_RN_N = [0., 0., 0.]
            s.attRefOutMsg.write(P, nanos, s.moduleID)
    ref = _Ref(); sim.AddModelToTask("att", ref)

    attErr = attTrackingError.attTrackingError()
    attErr.attRefInMsg.subscribeTo(ref.attRefOutMsg)
    attErr.attNavInMsg.subscribeTo(sNav.attOutMsg); sim.AddModelToTask("att", attErr)

    vcData = messaging.VehicleConfigMsgPayload()
    vcData.ISCPntB_B = [Idiag[0], 0, 0, 0, Idiag[1], 0, 0, 0, Idiag[2]]
    vcMsg = messaging.VehicleConfigMsg().write(vcData)
    fbk = mrpFeedback.mrpFeedback(); fbk.K = 0.05; fbk.Ki = -1.0; fbk.P = 0.5; fbk.integralLimit = 0.0
    fbk.guidInMsg.subscribeTo(attErr.attGuidOutMsg); fbk.vehConfigInMsg.subscribeTo(vcMsg)
    fbk.rwParamsInMsg.subscribeTo(rwConfigMsg); fbk.rwSpeedsInMsg.subscribeTo(rwSE.rwSpeedOutMsg)
    sim.AddModelToTask("att", fbk)

    rwMT = rwMotorTorque.rwMotorTorque(); rwMT.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    rwMT.vehControlInMsg.subscribeTo(fbk.cmdTorqueOutMsg); rwMT.rwParamsInMsg.subscribeTo(rwConfigMsg)
    rwSE.rwMotorCmdInMsg.subscribeTo(rwMT.rwMotorTorqueOutMsg); sim.AddModelToTask("att", rwMT)

    scRec = sc.scStateOutMsg.recorder(); rwRec = rwSE.rwSpeedOutMsg.recorder()
    sim.AddModelToTask("att", scRec); sim.AddModelToTask("att", rwRec)
    sim.InitializeSimulation(); sim.ConfigureStopTime(macros.sec2nano(dur)); sim.ExecuteSimulation()

    tt = np.asarray(scRec.times()) * macros.NANO2SEC + t0
    sigBN = np.asarray(scRec.sigma_BN); omega = np.asarray(scRec.omega_BN_B)
    Om = np.asarray(rwRec.wheelSpeeds)[:, :numRW]
    q_BN = np.array([_dcm_to_quat(rbk.MRP2C(s)) for s in sigBN])
    perr = np.zeros(len(tt)); fsm = np.zeros(len(tt))
    for j in range(len(tt)):
        bore_N = rbk.MRP2C(sigBN[j]).T @ CAM_BORE_B
        los = _normalize(np.array([np.interp(tt[j], t, bRef_N[:, k]) for k in range(3)]))
        perr[j] = math.acos(float(np.clip(np.dot(_normalize(bore_N), los), -1, 1)))
        fsm[j] = max(0.0, perr[j] - FSM_THROW_RAD)
    _wt, perr, rw_extra = _rw_hifi(Om, np.zeros((len(tt), numRW)), perr)
    st_extra = _startracker_extra(omega)
    return dict(t=tt, q_BN=q_BN, omega_BN_B=omega, wheel_speed_rads=Om,
                wheel_torque_Nm=np.zeros((len(tt), numRW)), ctrl_torque_Nm=np.zeros((len(tt), 3)),
                pointing_err_rad=perr, fsm_residual_rad=fsm, los_rate_rps=np.zeros(len(tt)),
                omega_ref_rps=np.zeros((len(tt), 3)), backend="basilisk", **st_extra, **rw_extra)


def attitude_6dof(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if ATT_BACKEND == "basilisk":
        try:
            return attitude_basilisk(d)
        except (Exception, SystemExit) as e:
            print(f"  [att] Basilisk RW stack unavailable ({type(e).__name__}: {e}); kinematic follower.")
    return attitude_kinematic(d["t"], d["C_RN"], d["bRef_N"], HW.XB6)


# ════════════════════════════════════════════════════════════════════════════
# finite-burn CONTROL emission (MPS-130 model, modules/thruster.py)
# ════════════════════════════════════════════════════════════════════════════
def control_finite_burn(npz_path, rng=None):
    rng = rng or np.random.default_rng()
    d = np.load(npz_path, allow_pickle=True)
    dv_times = np.asarray(d["dv_times"], float); dv_list = np.atleast_2d(np.asarray(d["dv_list"], float))
    mass = HW.XB6.wet_mass_kg
    cmd, real, perr, nmib, dur = [], [], [], [], []
    prop = 0.0; m = mass
    for dv in dv_list:
        if THR is not None:
            dvr, mp, e = THR.apply_burn(dv, m, rng)
            cmd.append(e["mag_cmd"]); real.append(float(np.linalg.norm(dvr))); perr.append(e["point_err_deg"])
            nmib.append(e["n_mib"]); prop += mp; m -= mp
        else:
            mag = float(np.linalg.norm(dv)); cmd.append(mag); real.append(mag); perr.append(0.0); nmib.append(0)
        dur.append(float(np.linalg.norm(dv)) * m / HW.MPS130.thrust_N)
    return dict(burn_times_s=dv_times, dv_cmd_mps=np.asarray(cmd), dv_real_mps=np.asarray(real),
                point_err_deg=np.asarray(perr), n_mib=np.asarray(nmib, int), burn_dur_s=np.asarray(dur),
                propellant_kg=float(prop), dv_total_cmd_mps=float(np.sum(cmd)),
                dv_total_real_mps=float(np.sum(real)))


# ════════════════════════════════════════════════════════════════════════════
# per-draw emission
# ════════════════════════════════════════════════════════════════════════════
def _draw_root():
    out = os.path.normpath(os.getenv("ARISE_OUT_DIR", os.path.join(HERE, "data", "mc", "draw_0000")))
    return os.path.dirname(out) if os.path.basename(out) == "trajectories" else out

def emit_attitude_and_control(traj_dir, att_dir, ctrl_dir, seed=0):
    os.makedirs(att_dir, exist_ok=True); os.makedirs(ctrl_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(traj_dir, "apophis_rendezvous_full_D*.npz")))
    if not files:
        print(f"  [emit] no trajectory NPZs in {traj_dir}"); return []
    summary = []
    for f in files:
        idx = "".join(ch for ch in os.path.basename(f).split("_D")[-1] if ch.isdigit()) or "0"
        rng = np.random.default_rng(seed * 1000 + int(idx))
        att = attitude_6dof(f)
        np.savez_compressed(os.path.join(att_dir, f"att_D{idx}.npz"),
                            t=att["t"], q_BN=att["q_BN"], omega_BN_B=att["omega_BN_B"],
                            wheel_speed_rads=att["wheel_speed_rads"], wheel_torque_Nm=att["wheel_torque_Nm"],
                            ctrl_torque_Nm=att["ctrl_torque_Nm"], pointing_err_rad=att["pointing_err_rad"],
                            fsm_residual_rad=att["fsm_residual_rad"], los_rate_rps=att["los_rate_rps"],
                            backend=np.array([att["backend"]]),
                            **{k: att[k] for k in ("st_pointing_sigma_rad", "st_available", "rw_jitter_rad") if k in att})
        ctl = control_finite_burn(f, rng)
        np.savez_compressed(os.path.join(ctrl_dir, f"ctrl_D{idx}.npz"),
                            burn_times_s=ctl["burn_times_s"], dv_cmd_mps=ctl["dv_cmd_mps"],
                            dv_real_mps=ctl["dv_real_mps"], point_err_deg=ctl["point_err_deg"],
                            n_mib=ctl["n_mib"], burn_dur_s=ctl["burn_dur_s"],
                            propellant_kg=np.array([ctl["propellant_kg"]]),
                            dv_total_real_mps=np.array([ctl["dv_total_real_mps"]]))
        pe = float(np.degrees(np.max(att["pointing_err_rad"])) * 3600.0)
        fsm_lost = float((att["fsm_residual_rad"] > 0).mean() * 100.0)
        summary.append(dict(deputy=int(idx), att_backend=att["backend"], max_point_err_arcsec=pe,
                            frac_beyond_fsm_pct=fsm_lost, dv_real_mps=ctl["dv_total_real_mps"],
                            propellant_g=ctl["propellant_kg"] * 1e3))
        print(f"  [emit] D{idx}: att={att['backend']} maxPErr={pe:.0f}\" beyondFSM={fsm_lost:.1f}% "
              f"dV_real={ctl['dv_total_real_mps']:.4f} m/s prop={ctl['propellant_kg']*1e3:.1f} g")
    return summary


def main():
    draw_root = _draw_root()
    traj_dir = os.path.join(draw_root, "trajectories")
    att_dir = os.path.join(draw_root, "attitude"); ctrl_dir = os.path.join(draw_root, "control")
    for dd in (traj_dir, att_dir, ctrl_dir):
        os.makedirs(dd, exist_ok=True)
    print("=" * 78)
    print(f"ARISE guidance V2  draw_root={draw_root}  att_backend={ATT_BACKEND}  ctrl_bw={CTRL_BW_HZ} Hz")
    print("=" * 78)
    os.environ["ARISE_OUT_DIR"] = traj_dir
    import importlib
    G = importlib.import_module("basilik_hifi_apophis_rendezvous_fleet")
    G.OUT_DIR = traj_dir
    G.main()
    summ = emit_attitude_and_control(traj_dir, att_dir, ctrl_dir)
    json.dump(dict(draw_root=draw_root, att_backend=ATT_BACKEND, deputies=summ),
              open(os.path.join(draw_root, "guidance_v2_meta.json"), "w"), indent=2)
    print(f"[guidance_v2] done -> {draw_root}")


def _self_test():
    import tempfile
    print("guidance_v2 self-test (synthetic flyby track, numpy only)")
    v = 8760.0; t = np.linspace(-4.0, 4.0, 401); standoff = 750.0
    rA = np.zeros((len(t), 3))
    rD = np.column_stack([np.full_like(t, standoff), v * t, np.zeros_like(t)])
    vD = np.column_stack([np.zeros_like(t), np.full_like(t, v), np.zeros_like(t)])
    bRef = np.zeros((len(t), 3)); C_RN = np.zeros((len(t), 3, 3))
    for j in range(len(t)):
        los = _normalize(rA[j] - rD[j]); bRef[j] = los
        C_RN[j] = _triad(los, vD[j]).T
    dv_times = np.array([0.0, 1800.0, 3600.0])
    dv_list = np.array([[0.4, -0.1, 0.05], [0.002, 0.001, -0.001], [0.001, 0.0, 0.0]])
    tmp = tempfile.mkdtemp(prefix="gv2_selftest_")
    tdir = os.path.join(tmp, "trajectories"); os.makedirs(tdir, exist_ok=True)
    np.savez_compressed(os.path.join(tdir, "apophis_rendezvous_full_D0.npz"),
                        t=t, rD_N=rD, vD_N=vD, rA_N=rA, vA_N=np.zeros_like(rD),
                        C_RN=C_RN, bRef_N=bRef, dv_times=dv_times, dv_list=dv_list, tot_dv=np.array([0.41]))
    global ATT_BACKEND; ATT_BACKEND = "kinematic"
    emit_attitude_and_control(tdir, os.path.join(tmp, "attitude"), os.path.join(tmp, "control"))
    a = np.load(os.path.join(tmp, "attitude", "att_D0.npz"))
    c = np.load(os.path.join(tmp, "control", "ctrl_D0.npz"))
    print("  attitude keys :", list(a.files))
    print("  q_BN[0]=%s |q|=%.6f  (round-trip ref at window start)" % (np.round(a["q_BN"][0], 4), la.norm(a["q_BN"][0])))
    print("  max|omega| %.3f rad/s  peak wheel %.0f rad/s (%.0f rpm)  peak|torque| %.2e Nm"
          % (np.abs(a["omega_BN_B"]).max(), np.abs(a["wheel_speed_rads"]).max(),
             np.abs(a["wheel_speed_rads"]).max() * 60 / (2 * math.pi), np.abs(a["wheel_torque_Nm"]).max()))
    print("  pointing err  : floor %.1f\"  peak %.2f deg   beyond-FSM frames %.1f%%"
          % (np.degrees(a["pointing_err_rad"].min()) * 3600, np.degrees(a["pointing_err_rad"].max()),
             (a["fsm_residual_rad"] > 0).mean() * 100))
    print("  control dV    : cmd %.4f -> real %.4f m/s  prop %.2f g  MIB %s"
          % (c["dv_cmd_mps"].sum(), float(c["dv_total_real_mps"][0]), float(c["propellant_kg"][0]) * 1e3, c["n_mib"].tolist()))
    assert a["q_BN"].shape == (len(t), 4) and a["wheel_speed_rads"].shape[1] == 4
    assert abs(la.norm(a["q_BN"][0]) - 1.0) < 1e-6
    assert np.degrees(a["pointing_err_rad"].min()) * 3600 < 60.0     # arcsec-class away from CA
    assert np.abs(a["omega_BN_B"]).max() < 1.0                       # wheel-rate-limited (sane)
    print("  SELF-TEST PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    _self_test() if a.self_test else main()
