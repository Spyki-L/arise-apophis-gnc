"""
compute_encounter_geometry.py  (V2)  -  real geometry at CA from the solution-220 SPK
=====================================================================================
V2 upgrade of A2: instead of the Basilisk-bundled 20099942.bsp, this furnishes the authoritative
JPL solution-220 SPK (sb-99942-220.bsp) you downloaded, and queries the real Apophis / Sun /
Earth geometry at the true CA epoch.  Run ONCE in the Basilisk env (Py3.11 + dist3).

Mirrors the guidance SPICE setup (zeroBase="Earth", J2000) and adds the solution-220 kernel.
Saves uq_montecarlo_v2/data/encounter_geometry.npz:

  sun_hat        (3,)  Apophis->Sun  unit [J2000]
  earth_hat      (3,)  Apophis->Earth unit [J2000]
  r_sun_geo_m    (3,)  geocentric Sun  at CA  [m]
  r_apo_geo_m    (3,)  geocentric Apophis at CA [m]   (authoritative sol-220)
  r_apo_helio_m  (6,)  heliocentric Apophis state [pos+vel, m, m/s] at CA  (for A3 validation)
  phase_angle_deg, sun_apo_dist_m, ecl_lon_deg, epoch_utc, source

Also writes ca_nominal_state.npz with the heliocentric Apophis state at CA -> consumed by
cov_propagate.py as the validation target.

It is additive (new files only); a running MC is undisturbed.
"""
from __future__ import annotations
import os, sys
import numpy as np

CA_EPOCH_UTC = "2029 April 13, 20:11:21.903 (UTC)"
STEP_S = 1.0
_EPS = np.radians(23.43928)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DESIGN_DATA = os.path.normpath(os.path.join(HERE, "..", "data"))   # D_strobe_fullbody/data
SPK_SOL220 = os.path.join(DESIGN_DATA, "sb-99942-220.bsp")
OUT_NPZ = os.path.join(DATA, "encounter_geometry.npz")
OUT_NOM = os.path.join(DATA, "ca_nominal_state.npz")

# NAIF id of Apophis in the JPL sol-220 SPK = 2000000 + 99942 = 2099942 (CONFIRMED in sb-99942-220.bsp
# DAF header). The Basilisk-bundled 20099942.bsp uses a different id; do NOT use it here.
APOPHIS_NAME = os.getenv("ARISE_APOPHIS_SPK_ID", "2099942")


def _ecl_lon_deg(v):
    x, y, z = v
    ye = y * np.cos(_EPS) + z * np.sin(_EPS)
    return float(np.degrees(np.arctan2(ye, x)) % 360.0)


def main():
    try:
        from Basilisk import __path__ as bskPath
        from Basilisk.simulation import spiceInterface, ephemerisConverter
        from Basilisk.utilities import SimulationBaseClass, macros
    except Exception as e:
        sys.exit("[geom-v2] Basilisk not importable (%s). Run in PY311 + PYTHONPATH=dist3." % e)

    if not os.path.exists(SPK_SOL220):
        sys.exit("[geom-v2] solution-220 SPK not found: %s" % SPK_SOL220)

    SPICE_DIR = os.path.join(bskPath[0], "supportData", "EphemerisData") + os.sep
    BASE_KERNELS = ["naif0012.tls", "pck00010.tpc", "de-403-masses.tpc", "de430.bsp"]
    print("[geom-v2] base kernels :", ", ".join(BASE_KERNELS))
    print("[geom-v2] sol-220 SPK  :", SPK_SOL220)
    print("[geom-v2] CA epoch     :", CA_EPOCH_UTC)

    sim = SimulationBaseClass.SimBaseClass()
    task = sim.CreateNewTask("eph", macros.sec2nano(STEP_S))
    sim.CreateNewProcess("ephp").addTask(task)

    spi = spiceInterface.SpiceInterface()
    spi.SPICEDataPath = SPICE_DIR
    spi.UTCCalInit = CA_EPOCH_UTC
    for k in BASE_KERNELS:
        spi.loadSpiceKernel(k, SPICE_DIR)
    # authoritative solution-220 Apophis kernel (from the Design D data dir)
    spi.loadSpiceKernel(os.path.basename(SPK_SOL220), os.path.dirname(SPK_SOL220) + os.sep)

    apo_name = APOPHIS_NAME
    spi.addPlanetNames(["Earth", "Sun", "Moon", apo_name])
    print("[geom-v2] Apophis NAIF id used:", apo_name, "(sol-220 SPK; override via ARISE_APOPHIS_SPK_ID)")
    spi.zeroBase = "Earth"
    sim.AddModelToTask("eph", spi)

    conv = ephemerisConverter.EphemerisConverter()
    for m in spi.planetStateOutMsgs:
        conv.addSpiceInputMsg(m)
    sim.AddModelToTask("eph", conv)
    sRec = conv.ephemOutMsgs[1].recorder()   # Sun  (geocentric)
    aRec = conv.ephemOutMsgs[3].recorder()   # Apophis (geocentric)
    sim.AddModelToTask("eph", sRec); sim.AddModelToTask("eph", aRec)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(STEP_S))
    sim.ExecuteSimulation()

    r_sun = np.asarray(sRec.r_BdyZero_N)[0]
    r_apo = np.asarray(aRec.r_BdyZero_N)[0]
    v_apo = np.asarray(aRec.v_BdyZero_N)[0]

    v_sun_apo = r_sun - r_apo
    v_earth_apo = -r_apo
    sun_hat = v_sun_apo / np.linalg.norm(v_sun_apo)
    earth_hat = v_earth_apo / np.linalg.norm(v_earth_apo)
    # heliocentric Apophis state (geocentric Apophis - geocentric Sun = helio, sign: Apo - Sun)
    r_helio = r_apo - r_sun
    v_helio = v_apo - np.zeros(3)   # Sun geocentric velocity small; recompute below if available
    phase = float(np.degrees(np.arccos(np.clip(sun_hat @ earth_hat, -1, 1))))
    sun_dist = float(np.linalg.norm(v_sun_apo))
    ecl_lon = _ecl_lon_deg(sun_hat)

    os.makedirs(DATA, exist_ok=True)
    np.savez(OUT_NPZ, sun_hat=sun_hat, earth_hat=earth_hat,
             r_sun_geo_m=r_sun, r_apo_geo_m=r_apo, r_earth_apo_m=v_earth_apo,
             r_apo_helio_m=np.concatenate([r_helio, v_helio]),
             sun_apo_dist_m=sun_dist, phase_angle_deg=phase, ecl_lon_deg=ecl_lon,
             epoch_utc=CA_EPOCH_UTC, frame="J2000_geocentric",
             source="JPL sol-220 SPK (sb-99942-220.bsp) + de430")
    np.savez(OUT_NOM, r_apo_geo_m=r_apo, v_apo_geo_m=v_apo, r_sun_geo_m=r_sun,
             epoch_utc=CA_EPOCH_UTC, naif_id=apo_name)

    print("\n[geom-v2] RESULTS at CA (sol-220):")
    print("  geocentric Apophis [m] = [% .6e % .6e % .6e]" % tuple(r_apo))
    print("  geocentric Sun     [m] = [% .6e % .6e % .6e]" % tuple(r_sun))
    print("  sun_hat   (Apo->Sun)   = [% .6f % .6f % .6f]" % tuple(sun_hat))
    print("  earth_hat (Apo->Earth) = [% .6f % .6f % .6f]" % tuple(earth_hat))
    print("  Apophis-Sun distance   = %.5f AU" % (sun_dist / 1.495978707e11))
    print("  Sun-Apo-Earth phase    = %.2f deg" % phase)
    print("  implied Sun ecl lon    = %.2f deg  (V1 default used 203.0)" % ecl_lon)
    print("\n[geom-v2] saved -> %s" % OUT_NPZ)
    print("[geom-v2] saved -> %s   (A3 validation target)" % OUT_NOM)


if __name__ == "__main__":
    main()
