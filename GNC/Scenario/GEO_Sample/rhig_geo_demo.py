#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rhig_geo_demo.py
================

Example use of RHIGFleet for a simple GEO target:

- Target: a GEO object with trajectory from target_geo_reference.npz
          (times, positions, velocities in km/km/s).
- Chief: a GEO trajectory from chief_reference.npz (Hill frame built around this).
- Deputies: 3, each with a different LVLH target in [R, T, N] [km]
           around the GEO target.

This script:
- Ensures chief_reference.npz and target_geo_reference.npz exist,
- Configures RHIGFleetConfig with GEO-tuned settings (few impulses),
- Instantiates RHIGFleet,
- Runs RHIG rendezvous for each deputy (coarse/refined deploy-time search),
- Compresses impulses + adds a terminal CW burn (if enabled),
- Exports rendezvous_full_Dk.npz and viz_inputs_Dk.npz in this folder,
- Prints impulse tables.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np

# ───────────────── add ARISE Arch root to sys.path ──────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

# Find the parent directory that contains the "GNC" folder
ROOT = None
for parent in SCRIPT_DIR.parents:
    if (parent / "GNC").is_dir():
        ROOT = parent
        break

if ROOT is None:
    # Fallback: go up three levels (GEO_Sample -> Scenario -> GNC -> ARISE Arch)
    ROOT = SCRIPT_DIR.parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNC.Guidance.rhig_rendezvous_fleet import RHIGFleetConfig, RHIGFleet


# ───────────────── paths & config ───────────────────────────────────────────

CHIEF_REF_NPZ  = (SCRIPT_DIR / "chief_reference.npz").resolve()
TARGET_REF_NPZ = (SCRIPT_DIR / "target_geo_reference.npz").resolve()

DEPARTURE_UTC_ISO = os.getenv("DEPARTURE_UTC_ISO", "2030-01-01T00:00:00Z")
departure_dt = datetime.strptime(DEPARTURE_UTC_ISO, "%Y-%m-%dT%H:%M:%SZ").replace(
    tzinfo=timezone.utc
)

# 3 deputies with LVLH targets [km] around the GEO target
FLEET_TARGETS_KM = np.array([
    [  5.0,  6.0,  3.5],
    [ -8.0,  5.5, -7.3],
    [  5.5, -3.5,  9.8],
], dtype=float)
FLEET_TARGETS_M = FLEET_TARGETS_KM * 1e3  # [m]


def main():
    if not CHIEF_REF_NPZ.is_file():
        raise FileNotFoundError(f"chief_reference.npz not found at {CHIEF_REF_NPZ}")
    if not TARGET_REF_NPZ.is_file():
        raise FileNotFoundError(f"target_geo_reference.npz not found at {TARGET_REF_NPZ}")

    # Load chief_reference to get TOF, mainly to report arrival time
    d = np.load(CHIEF_REF_NPZ)
    T = np.asarray(d["times"], float)
    TOF_SEC = float(T[-1])

    arrival_dt = departure_dt + timedelta(seconds=TOF_SEC)

    print(f"[GEO DEMO] chief reference   : {CHIEF_REF_NPZ}")
    print(f"[GEO DEMO] target reference  : {TARGET_REF_NPZ}")
    print(f"[GEO DEMO] TOF_SEC           : {TOF_SEC:.1f} s")
    print(f"[GEO DEMO] departure UTC ISO : {DEPARTURE_UTC_ISO}")
    print(f"[GEO DEMO] arrival UTC ISO   : {arrival_dt.isoformat().replace('+00:00','Z')}")

    # GEO-tuned config:
    cfg = RHIGFleetConfig(
        # ephemerides
        chief_ref_npz      = CHIEF_REF_NPZ,
        target_npz         = TARGET_REF_NPZ,   # use NPZ target instead of SPICE
        departure_utc_iso  = DEPARTURE_UTC_ISO,
        central_body_name  = os.getenv("CENTRAL_BODY_NAME", "Earth"),
        target_body_name   = os.getenv("TARGET_BODY_NAME", "GEO_TARGET"),
        target_spk_name    = os.getenv("TARGET_SPK_NAME", "dummy.bsp"),

        # fleet LVLH targets (metres)
        fleet_targets_m    = FLEET_TARGETS_M,

        # RHIG / optimiser constants (GEO-friendly)
        DT_DYN        = 1.0,       # 1 s dynamics
        DT_FSW        = 120.0,     # guidance every 2 min
        E_TRIGGER     = 300.0,     # only fire if >300 m error
        IMPROVE_MIN   = 30.0,      # require ≥30 m improvement per burn
        V_MIN         = 1e-3,      # 1 mm/s minimum useful Δv
        V_MAX         = 50.0,      # up to 50 m/s per burn in search
        MIN_LOOK      = 600.0,     # don't fire too close to arrival
        COOLDOWN      = 600.0,     # 10 min between burns
        MAX_IMP       = 6,         # search phase: at most 6 impulses

        ALPHA_DAMP_BASE   = 1e-7,
        USE_LAMBERT_NUDGE = False, # keep GEO demo simple / fast
        NUDGE_WEIGHT      = 0.4,
        NUDGE_SAT         = 0.4,
        ALIGN_DEG_MIN     = 1.0,

        # cost = |e_final| + LAMBDA_DV * Σ|Δv|
        LAMBDA_DV         = 1.0,

        SEARCH_TMIN       = 1500.0,
        SEARCH_TMAX_PAD   = 1.2 * 3600.0,
        COARSE_STEP       = 900.0,
        REFINE_STEP_1     = 300.0,
        REFINE_STEP_2     = 120.0,
        REFINE_SPAN       = 1.2 * 3600.0,

        FINAL_V_MAX       = 200.0, # final RHIG run: up to 200 m/s per burn
        FINAL_MAX_IMP     = 20,    # final RHIG run: at most 20 impulses

        TARGET_MARGIN_M   = 1.0,
        DV_BUDGET_MS      = None,
        save_png_dir_base = None,  # keep RHIGFleet itself headless
        nav_npz_path      = None,

        # tight final error and compression
        FINAL_ERR_REL_TOL = 0.05,   # 5% of ‖target‖
        FINAL_ERR_ABS_MIN = 50.0,   # at least 50 m
        ENABLE_COMPRESSION= True,
        COMPRESS_KEEP_LAST= 3,      # keep last 3 burns as-is
        TERMINAL_MARGIN   = 600.0,  # terminal burn 10 min before arrival
        TERMINAL_V_MAX    = None,   # no explicit cap on terminal burn
    )

    rhig = RHIGFleet(cfg)
    results = rhig.run_fleet()  # produces rendezvous_full_Dk & viz_inputs_Dk

    # Print impulse tables
    for dep_idx, target_m, _, fin, _ in results:
        print(f"\n=== Impulses for Deputy {dep_idx} (target {target_m/1e3} km in LVLH around GEO target) ===")
        imps = fin["impulses"]
        if not imps:
            print("  <no burns>")
        else:
            print("  # |     t [s] |    Δv_R    Δv_T    Δv_N (mm/s)")
            for k, (t, dv) in enumerate(imps, 1):
                print(
                    f" {k:2d} | {t:10.1f} |"
                    f"{dv[0]*1e3:9.3f}{dv[1]*1e3:9.3f}{dv[2]*1e3:9.3f}"
                )
            print(f"  Total ΔV = {fin['tot_dv']:.3f} m/s")
            print(f"  Final |e| = {fin['err_norm']:.3f} m")

    print("\n[GEO DEMO] Done. Rendezvous NPZs (chief + target GEO + deputies) ready for GNC/FDI tests.")


if __name__ == "__main__":
    main()
