# -*- coding: utf-8 -*-
"""
hardware_bridge.py - wire the REAL nominal_v2 flight-hardware models into the
re-docking flight package for maximum fidelity.

Imports the ARISE V2 single-source-of-truth hardware (Designs/D_strobe_fullbody/
nominal_v2/modules): the Blue Canyon XB6 bus, Aerojet MPS-130 (AF-M315E) thruster,
XACT-50-class reaction-wheel assembly, XACT-50 Gen-3 star tracker, and the nav-grade
IMU gyro.  Exposes them to flight/actuators.py and flight/sensors.py.

If the nominal_v2 modules are not reachable, AVAILABLE=False and the flight package
falls back to its self-contained analytic hardware models (so it still runs anywhere).
Presets honour the same env vars as nominal_v2 (ARISE_ST_PRESET, ARISE_IMU_PRESET,
ARISE_THRUSTER_HIFI, ...).
"""
from __future__ import annotations
import os, sys
import numpy as np
import config_redock as C

AVAILABLE = False
IMPORT_ERR = None
_CANDIDATES = [
    os.path.join(C.ANIM_DIR, "Designs", "D_strobe_fullbody", "nominal_v2", "modules"),
    os.path.join(C.REPO_ROOT, "Designs", "D_strobe_fullbody", "nominal_v2", "modules"),
    os.environ.get("ARISE_NOMINAL_V2_MODULES", ""),
]

HW = thruster = None
BUS = MPS130 = RWA = ST = GYRO = SUN = None

for _p in _CANDIDATES:
    if _p and os.path.isdir(_p):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        try:
            import hardware as HW            # noqa
            import thruster                  # noqa
            import reaction_wheels as _rw     # noqa
            import star_tracker as _st        # noqa
            import imu_gyro as _imu           # noqa
            BUS = HW.XB6                      # Blue Canyon XB6 (6U)
            MPS130 = HW.MPS130               # Aerojet MPS-130 (AF-M315E)
            RWA = _rw.ReactionWheelParams()  # XACT-50-class reaction wheels
            ST = _st.active()                # XACT-50 Gen-3 star tracker (default gen3x2)
            GYRO = _imu.active()             # nav-grade IMU gyro (default)
            SUN = HW.SUN_SENSOR
            AVAILABLE = True
            break
        except Exception as e:
            IMPORT_ERR = e


def deputy_inertia():
    """Real XB6 principal inertia [kg m^2] (falls back to the config value)."""
    if AVAILABLE:
        return np.asarray(BUS.inertia_diag(), float)
    return np.asarray(C.ESPA["deputy_inertia"], float)


def deputy_mass_kg():
    return float(BUS.wet_mass_kg) if AVAILABLE else float(C.ESPA["deputy_mass_kg"])


def summary():
    if AVAILABLE:
        J = deputy_inertia()
        return (f"nominal_v2 HARDWARE wired: {BUS.name}; {MPS130.name} "
                f"(Isp {MPS130.isp_s}s, MIB {MPS130.min_impulse_bit_Ns*1e3:.0f} mN.s); "
                f"{RWA.name} (tau_max {RWA.max_torque_Nm*1e3:.1f} mN.m, h {RWA.max_momentum_Nms*1e3:.0f} mN.m.s); "
                f"ST {ST.name}; gyro {GYRO.name}; J={np.round(J,4)} kg.m^2")
    return f"nominal_v2 hardware NOT reachable ({IMPORT_ERR}); using analytic fallback models"


if __name__ == "__main__":
    print(summary())
    if AVAILABLE:
        rng = np.random.default_rng(0)
        dv_real, m_prop, err = thruster.apply_burn(np.array([0.03, 0, 0]), deputy_mass_kg(), rng, MPS130)
        print(f"MPS-130 burn: cmd 0.03 -> real {np.linalg.norm(dv_real):.4f} m/s, "
              f"{err['n_mib']} MIB, point_err {err['point_err_deg']:.2f} deg, prop {m_prop*1e3:.2f} g")
        tau = RWA.realized_torque(np.array([1e-3, 0, 0]), np.array([100.0, 0, 0]))
        print(f"RWA realized torque @100 rad/s: {np.round(tau,6)} N.m")
        print(f"ST error @0.01 rad/s: {np.round(ST.measure_error(0.01, rng)*1e6,1)} urad")
        st = None
        wm, st = GYRO.measure(np.array([0.01, 0, -0.02]), rng, st, 1.0)
        print(f"gyro measure: {np.round(wm,5)} rad/s")
