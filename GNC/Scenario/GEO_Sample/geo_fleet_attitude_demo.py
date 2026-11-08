#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geo_fleet_attitude_demo.py
==========================

GEO rendezvous swarm – fleet-level 6-DOF attitude simulation.

This script is the GEO analogue of your Apophis fleet mini-sim:

1. Discover rendezvous_full_D*.npz in this folder.
2. Load them as RendezvousBundle objects.
3. Configure FleetAttitudeConfig using the NPZ sample time.
4. Build and run FleetAttitudeSim.
5. Save logs to `fleet_logs/dep??_logs.npz`.
6. Print a small sanity check for dep00_logs.npz.

Use your existing fleet_quicklook_plots.py with:

    LOG_DIR = Path("fleet_logs")

to visualize the results.
"""

from pathlib import Path
import sys
import time

import numpy as np

# ── locate ARISE root so we can import GNC.Control --------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

ROOT = None
for parent in SCRIPT_DIR.parents:
    if (parent / "GNC").is_dir():
        ROOT = parent
        break
if ROOT is None:
    ROOT = SCRIPT_DIR.parents[2]   # fallback: .../ARISE Arch

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNC.Control.fleet_attitude import (
    FleetAttitudeConfig,
    RendezvousBundle,
    FleetAttitudeSim,
)


# ╔══════════════════════════════════════════════════════════════════════╗
# NPZ loading
# ╚══════════════════════════════════════════════════════════════════════╝

def discover_geo_bundles(directory: Path) -> list[Path]:
    """Find rendezvous_full_D*.npz in the given directory."""
    paths = sorted(directory.glob("rendezvous_full_D*.npz"))
    if not paths:
        raise FileNotFoundError(f"No rendezvous_full_D*.npz found in {directory}")
    print(f"[GEO ATT] Found {len(paths)} rendezvous bundles:")
    for p in paths:
        print(f"  - {p.name}")
    return paths


def load_rendezvous_bundle(npz_path: Path) -> RendezvousBundle:
    """
    Load one rendezvous_full_Dk.npz → RendezvousBundle.

    Expected fields in each NPZ:
      t       : (N,)   [s]
      rD_N    : (N,3)  deputy ECI position [m]
      vD_N    : (N,3)  deputy ECI velocity [m/s]
      bRef_N  : (N,3)  LOS / boresight unit vector in ECI
      q_RN    : (N,4)  optional reference attitude N→R [scalar-first]
    """
    with np.load(npz_path, allow_pickle=True) as n:
        t = np.asarray(n["t"], float)
        rD_N = np.asarray(n["rD_N"], float)
        vD_N = np.asarray(n["vD_N"], float)
        bRef_N = np.asarray(n["bRef_N"], float)
        q_RN = np.asarray(n["q_RN"], float) if "q_RN" in n else None

    return RendezvousBundle(
        t=t,
        rD_N=rD_N,
        vD_N=vD_N,
        bRef_N=bRef_N,
        q_RN=q_RN,
    )


# ╔══════════════════════════════════════════════════════════════════════╗
# Sanity check helper
# ╚══════════════════════════════════════════════════════════════════════╝

def describe_array(name, arr):
    """Pretty one‑line description of an array from the NPZ logs."""
    arr = np.asarray(arr)
    shape = arr.shape
    if arr.size == 0:
        return f"{name}: shape={shape}, min=nan, max=nan"
    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))
    # NOTE: no space after the '+'
    return (f"{name}: shape={shape}, "
            f"min={min_val:+.3e}, max={max_val:+.3e}")



def sanity_check_first_dep(log_dir: Path):
    """Print quick stats for dep00_logs.npz if it exists."""
    fpath = log_dir / "dep00_logs.npz"
    if not fpath.is_file():
        print(f"[GEO ATT] No {fpath.name} found for sanity check.")
        return

    logs = np.load(fpath)
    print(f"[GEO ATT] Sanity check {fpath.name}:")
    for key in ("t_err", "sigma_BR", "omega_BR", "t_torque", "torque_B",
                "t_navT", "r_nav", "v_nav"):
        if key in logs:
            print("   ", describe_array(key, logs[key]))
    print()


# ╔══════════════════════════════════════════════════════════════════════╗
# Main entry
# ╚══════════════════════════════════════════════════════════════════════╝

def main():
    print(f"[GEO ATT] Scenario directory : {SCRIPT_DIR}")

    npz_paths = discover_geo_bundles(SCRIPT_DIR)
    bundles = [load_rendezvous_bundle(p) for p in npz_paths]

    # Use first bundle to set timing
    t0 = float(bundles[0].t[0])
    dt_hist = float(np.median(np.diff(bundles[0].t)))
    t1 = float(bundles[0].t[-1])

    print(f"[GEO ATT] t0       = {t0:.1f} s")
    print(f"[GEO ATT] dt_hist  = {dt_hist:.3f} s")
    print(f"[GEO ATT] t_end    = {t1:.1f} s")
    print(f"[GEO ATT] Building FleetAttitudeSim...")

    # Fleet-level config (same knobs as Apophis mini-sim)
    cfg = FleetAttitudeConfig(
        dt_dyn=dt_hist,
        dt_sens=dt_hist,
        k_sigma=0.08,
        p_rate=0.9,
        ki=-1.0,
        i_lim=0.01,
        use_lpf=True,
        lpf_cut=0.8,
        mass_kg=250.0,
        inertia=(18.0, 15.0, 12.0),
        init_y_deg=5.0,
        nav_sig=dict(pos=50., vel=0.05, att=5e-4, rate=2e-5, sun=1e-3, dv=0.0),
        log_nav=True,
        log_truth=True,
        t_stop=None,
        prefer_qrn=True,
    )

    sim = FleetAttitudeSim(cfg, bundles)

    print("[GEO ATT] Running FleetAttitudeSim...")
    t_wall0 = time.time()
    sim.run()
    t_wall1 = time.time()
    print(f"[GEO ATT] Simulation wall-clock: {t_wall1 - t_wall0:.1f} s")

    log_dir = SCRIPT_DIR / "fleet_logs"
    print(f"[GEO ATT] Saving logs to: {log_dir}")
    sim.save_logs(log_dir)

    # small quick sanity check on dep00
    sanity_check_first_dep(log_dir)

    print("[GEO ATT] Done.")


if __name__ == "__main__":
    main()
