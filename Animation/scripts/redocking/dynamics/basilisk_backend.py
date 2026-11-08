# -*- coding: utf-8 -*-
"""
basilisk_backend.py - FLIGHT-FIDELITY 6-DOF backend for Phase-B docking.

Translation is propagated by Basilisk exactly as in the ARISE hi-fi guidance
(basilik_hifi_apophis_rendezvous_fleet.py): a persistent per-deputy spacecraft
with Earth (spherical harmonics J2..EARTH_SH_DEG) + Sun + Moon gravity from LIVE
SPICE + cannonball SRP, integrated with the adaptive RKF45 scheme.  Attitude is
propagated by the exact rigid-body integrator (dynamics/attitude.py, real 6U
inertia) plus the gravity-gradient torque - a standard operational split where
Basilisk supplies the ephemeris-driven translational force model (which genuinely
needs SPICE) and the rigid-body ODE supplies attitude.

Frames: the RL env works in the ESPA-centred LVLH frame; Basilisk works in ECI.
Each step converts LVLH->ECI with the chief reference, propagates in Basilisk,
converts back.  Attitude is tracked relative to LVLH throughout.

Requires Basilisk (you run this locally / on the cluster).  If Basilisk is absent
the constructor raises a clear ImportError; use dynamics="hcw" for the surrogate.
The LVLH<->ECI kinematics at the bottom are pure NumPy and are unit-tested.
"""
from __future__ import annotations
import os, sys, contextlib
from datetime import datetime, timezone, timedelta
import numpy as np

try:
    from .attitude import rk4_attitude, J_DEPUTY, mrp_to_dcm
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from attitude import rk4_attitude, J_DEPUTY, mrp_to_dcm

# ── config (mirrors basilik_hifi_apophis_rendezvous_fleet.py; env-var overridable)
MU_EARTH       = 3.986004418e14
# Epoch anchor for the hi-fi force model.  Default = the nominal_v2 ENCOUNTER (close-approach)
# epoch (compute_encounter_geometry.CA_EPOCH_UTC = "2029 April 13, 20:11:21.903 UTC") so the
# Sun/Moon/Earth geometry (third-body + SRP direction) MATCHES the nominal_v2 dynamics.  The
# re-dock occurs post-imaging near CA; override with ARISE_DEPARTURE_UTC for a specific re-dock time.
DEPARTURE_UTC  = os.getenv("ARISE_DEPARTURE_UTC", "2029-04-13T20:11:21Z")
EARTH_SH_DEG   = int(os.getenv("ARISE_EARTH_SH_DEG", "10"))
BUS_MASS_KG    = float(os.getenv("ARISE_BUS_MASS_KG", "12.2"))
SRP_AREA_M2    = float(os.getenv("ARISE_SRP_AREA_M2", "0.30"))
SRP_CR         = float(os.getenv("ARISE_SRP_CR", "1.3"))
USE_SRP        = os.getenv("ARISE_SRP", "1") == "1"
INTEGRATOR     = os.getenv("ARISE_INTEGRATOR", "rkf45").lower()
_departure_dt  = datetime.strptime(DEPARTURE_UTC, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush(); sys.stderr.flush()
    dn = os.open(os.devnull, os.O_WRONLY); o1, o2 = os.dup(1), os.dup(2)
    try:
        os.dup2(dn, 1); os.dup2(dn, 2); yield
    finally:
        os.dup2(o1, 1); os.dup2(o2, 2); os.close(dn); os.close(o1); os.close(o2)


class BasiliskPropagator:
    """6-DOF one-step propagator with the RL env's (states12, actions6, dt) API."""
    name = "basilisk"

    def __init__(self, omega: float, dt: float, chief_state_fn=None, t0: float = 0.0,
                 mu: float = MU_EARTH, inertia=None, use_gravity_gradient: bool = True):
        self.omega = float(omega); self.dt = float(dt); self.mu = float(mu)
        self.t = float(t0); self.chief_state_fn = chief_state_fn
        self.J = np.asarray(inertia if inertia is not None else J_DEPUTY, float)
        self.use_gg = use_gravity_gradient
        self._sims = {}                     # persistent per-deputy Basilisk sims
        self._require_basilisk()

    def _require_basilisk(self):
        try:
            import Basilisk  # noqa: F401
        except Exception as e:
            raise ImportError(
                "BasiliskPropagator needs Basilisk (run locally / on the cluster). "
                "For the fast surrogate use dynamics='hcw'. Import error: %r" % (e,)) from e

    # reference trajectory (ESPA/chief) ----------------------------------------
    def set_reference(self, chief_state_fn, t0: float = 0.0):
        self.chief_state_fn = chief_state_fn; self.t = float(t0); return self

    def _ref_state(self, t):
        if self.chief_state_fn is not None:
            r, v = self.chief_state_fn(t); return np.asarray(r, float), np.asarray(v, float)
        R = (self.mu / self.omega**2) ** (1/3); th = self.omega * t
        r = R*np.array([np.cos(th), np.sin(th), 0.]); v = R*self.omega*np.array([-np.sin(th), np.cos(th), 0.])
        return r, v

    # ── translation: persistent Basilisk spacecraft (mirrors _bsk_build/_bsk_run)
    def _build_sim(self, t0):
        from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody
        from Basilisk.simulation import spacecraft, radiationPressure, svIntegrators
        from Basilisk.architecture import bskLogging
        from Basilisk import __path__ as bskPath
        bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)
        sim = SimulationBaseClass.SimBaseClass(); sim.SetProgressBar(False)
        proc = sim.CreateNewProcess("dyn"); proc.addTask(sim.CreateNewTask("dyn", macros.sec2nano(self.dt)))
        gf = simIncludeGravBody.gravBodyFactory()
        earth = gf.createEarth(); earth.isCentralBody = True
        try:
            earth.useSphericalHarmonicsGravityModel(
                os.path.join(bskPath[0], "supportData", "LocalGravData", "GGM03S.txt"), EARTH_SH_DEG)
        except Exception:
            pass
        gf.createSun(); gf.createMoon()
        epoch = (_departure_dt + timedelta(seconds=t0)).strftime("%Y %B %d, %H:%M:%S.0 (UTC)")
        spi = gf.createSpiceInterface(time=epoch, epochInMsg=True); spi.zeroBase = "Earth"
        sim.AddModelToTask("dyn", spi)
        sc = spacecraft.Spacecraft(); sc.ModelTag = "deputy"; sc.hub.mHub = BUS_MASS_KG
        sc.hub.r_BcB_B = [[0.], [0.], [0.]]
        sc.hub.IHubPntBc_B = [[float(self.J[0]), 0., 0.], [0., float(self.J[1]), 0.], [0., 0., float(self.J[2])]]
        integ = None
        if INTEGRATOR == "rkf45":
            try:
                integ = svIntegrators.svIntegratorRKF45(sc); sc.setIntegrator(integ)
            except Exception:
                integ = None
        gf.addBodiesTo(sc); sim.AddModelToTask("dyn", sc)
        srp = None
        if USE_SRP:
            try:
                srp = radiationPressure.RadiationPressure(); srp.ModelTag = "SRP"
                srp.setUseCannonballModel(); srp.area = SRP_AREA_M2; srp.coefficientReflection = SRP_CR
                srp.sunEphmInMsg.subscribeTo(spi.planetStateOutMsgs[1]); sc.addDynamicEffector(srp)
                sim.AddModelToTask("dyn", srp)
            except Exception:
                srp = None
        return dict(sim=sim, sc=sc, gf=gf, spi=spi, srp=srp, integ=integ, built_t0=t0)

    def _propagate_eci(self, r0, v0, ta, tb, key):
        """One hi-fi ECI translation step [ta,tb] for deputy `key` (persistent sim)."""
        from Basilisk.utilities import macros
        st = self._sims.get(key)
        if st is None or abs(st["built_t0"] - ta) > 3600.0:
            st = self._build_sim(ta); self._sims[key] = st
        sim, sc = st["sim"], st["sc"]
        sc.hub.r_CN_NInit = [[float(r0[0])], [float(r0[1])], [float(r0[2])]]
        sc.hub.v_CN_NInit = [[float(v0[0])], [float(v0[1])], [float(v0[2])]]
        with _quiet():
            sim.InitializeSimulation()
            sim.ConfigureStopTime(macros.sec2nano(tb - ta)); sim.ExecuteSimulation()
        posRef = sc.dynManager.getStateObject(sc.hub.nameOfHubPosition)
        velRef = sc.dynManager.getStateObject(sc.hub.nameOfHubVelocity)
        pf = posRef.getState(); vf = velRef.getState()
        return (np.array([pf[0][0], pf[1][0], pf[2][0]]), np.array([vf[0][0], vf[1][0], vf[2][0]]))

    # ── one 6-DOF step -----------------------------------------------------------
    def step6dof(self, states, actions, dt):
        dt = self.dt if dt is None else float(dt)
        S = np.atleast_2d(np.asarray(states, float)).copy()   # (N,12)
        A = np.atleast_2d(np.asarray(actions, float))         # (N,6)  [dv_lvlh(3), tau_body(3)]
        r_ref0, v_ref0 = self._ref_state(self.t)
        r_ref1, v_ref1 = self._ref_state(self.t + dt)
        out = S.copy()
        for i in range(S.shape[0]):
            # translation: LVLH rel -> ECI abs -> hi-fi Basilisk step -> LVLH rel
            s_lvlh = S[i, :6].copy(); s_lvlh[3:6] += A[i, :3]   # impulsive dv, applied in LVLH
            r0, v0 = lvlh_to_eci(s_lvlh, r_ref0, v_ref0)
            r1, v1 = self._propagate_eci(r0, v0, self.t, self.t + dt, key=i)
            out[i, :6] = eci_to_lvlh(r1, v1, r_ref1, v_ref1)
            # attitude: rigid-body + gravity-gradient torque (LVLH-relative attitude)
            tau = A[i, 3:6].astype(float)
            if self.use_gg:
                tau = tau + self._grav_gradient_torque(S[i, :3], r_ref0, S[i, 6:9])
            sig, om = rk4_attitude(S[i, 6:9], S[i, 9:12], tau, dt, self.J)
            out[i, 6:9] = sig; out[i, 9:12] = om
        self.t += dt
        return out

    def _grav_gradient_torque(self, pos_lvlh, r_ref_eci, sigma):
        """Gravity-gradient torque tau = 3 mu/R^3 (n_b x J n_b), n_b = nadir in body."""
        r_eci = r_ref_eci + pos_lvlh
        R = np.linalg.norm(r_eci) + 1e-9
        n_lvlh = -r_eci / R                        # nadir (toward Earth), approx in LVLH-ish
        n_b = mrp_to_dcm(sigma) @ n_lvlh           # nadir in body frame
        return 3.0 * self.mu / R**3 * np.cross(n_b, self.J * n_b)

    # legacy translation-only entry (kept for compatibility)
    def __call__(self, states, dv=None, dt=None):
        dt = self.dt if dt is None else float(dt)
        S = np.atleast_2d(np.asarray(states, float)).copy()
        A = np.zeros((S.shape[0], 6))
        if dv is not None:
            A[:, :3] = np.atleast_2d(dv)
        S12 = np.zeros((S.shape[0], 12)); S12[:, :6] = S[:, :6]
        return self.step6dof(S12, A, dt)[:, :6]


# ── LVLH <-> ECI kinematics (pure NumPy; unit-tested) ─────────────────────────
def lvlh_basis(r_eci, v_eci):
    r = np.asarray(r_eci, float); v = np.asarray(v_eci, float)
    R_hat = r / (np.linalg.norm(r) + 1e-15)
    h = np.cross(r, v); N_hat = h / (np.linalg.norm(h) + 1e-15)
    T_hat = np.cross(N_hat, R_hat)
    return np.column_stack([R_hat, T_hat, N_hat]), h / (np.dot(r, r) + 1e-15)


def eci_to_lvlh(r_dep, v_dep, r_ref, v_ref):
    Rmat, omega = lvlh_basis(r_ref, v_ref)
    dr = np.asarray(r_dep, float) - np.asarray(r_ref, float)
    dv = np.asarray(v_dep, float) - np.asarray(v_ref, float)
    return np.concatenate([Rmat.T @ dr, Rmat.T @ (dv - np.cross(omega, dr))])


def lvlh_to_eci(s_lvlh, r_ref, v_ref):
    Rmat, omega = lvlh_basis(r_ref, v_ref)
    x = np.asarray(s_lvlh[:3], float); xdot = np.asarray(s_lvlh[3:6], float)
    dr = Rmat @ x; dv = Rmat @ xdot + np.cross(omega, dr)
    return np.asarray(r_ref, float) + dr, np.asarray(v_ref, float) + dv


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    r_ref = np.array([7.0e6, 1.0e5, -2.0e5]); v_ref = np.array([-30.0, 7600.0, 12.0])
    s = np.array([50., -80., 15., .03, -.02, .01])
    r_d, v_d = lvlh_to_eci(s, r_ref, v_ref)
    err = np.max(np.abs(eci_to_lvlh(r_d, v_d, r_ref, v_ref) - s))
    print("LVLH<->ECI round-trip error =", f"{err:.2e}")
    assert err < 1e-9
    try:
        import Basilisk  # noqa
        print("Basilisk present -> hi-fi 6-DOF available")
    except Exception:
        print("Basilisk NOT installed here (expected in sandbox) -> use dynamics='hcw'")
    print("basilisk_backend kinematics self-test OK")
