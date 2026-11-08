#!/usr/bin/env python3

"""
rhig_boresight_quicklook.py

Purpose
-------
Quick-look tool for inspecting RHIG rendezvous NPZ files. For each deputy file
matching a glob pattern, the script reconstructs and visualizes the reference
boresight vector and an implied body attitude at the initial epoch.

High-level behavior
-------------------
1. Scan a fleet directory for NPZ files (default:
       apophis_rendezvous_full_D*.npz
   under the current working directory).
2. For each file:
   - Load deputy and asteroid position/velocity histories (rD_N, vD_N, rA_N).
   - Use `bRef_N` as the boresight history if present; otherwise construct a
     deputy→asteroid line-of-sight vector from (rA_N - rD_N).
   - Print the initial boresight vector and the initial geometric LOS.
   - Build a body-frame TRIAD at t0 using the boresight and velocity to form a
     right-handed attitude DCM (N←B) and print that matrix.
3. Plot the three components of the boresight vector vs. time for each deputy.

Usage
-----
Adjust `fleet_dir` and `glob_pat` at the top of the script to point at the
desired RHIG NPZ directory and filename pattern, then run:

    python rhig_boresight_quicklook.py

A separate plot window is produced for each deputy NPZ file.
"""

import numpy as np
import numpy.linalg as la
from pathlib import Path
import matplotlib.pyplot as plt

# ── SETTINGS ───────────────────────────────────────────────────────
fleet_dir = Path(".")   # directory with NPZ files
glob_pat  = "apophis_rendezvous_full_D*.npz"

# ── UTILITIES ─────────────────────────────────────────────────────
def normalize(v):
    n = la.norm(v)
    return v/n if n > 1e-12 else np.zeros_like(v)

def triad_from_boresight_and_tangent(los, vel):
    z = normalize(los)
    v_perp = vel - np.dot(vel, z)*z
    x = normalize(v_perp)
    if la.norm(x) < 1e-10:
        tmp = np.array([1,0,0]) if abs(z[0])<.9 else np.array([0,1,0])
        x = normalize(np.cross(tmp,z))
    y = normalize(np.cross(z,x))
    x = normalize(np.cross(y,z))
    y = normalize(np.cross(z,x))
    return np.column_stack((x,y,z))   # DCM [NB]

# ── MAIN LOOP ─────────────────────────────────────────────────────
paths = sorted(fleet_dir.glob(glob_pat))
if not paths:
    raise FileNotFoundError(f"No NPZ files found at {fleet_dir}/{glob_pat}")

for i,p in enumerate(paths):
    n = np.load(p)
    rD, vD, rA = n["rD_N"], n["vD_N"], n["rA_N"]

    if "bRef_N" in n:
        bRef = n["bRef_N"]
    else:
        bRef = (rA - rD)/la.norm(rA - rD, axis=1)[:,None]

    print(f"\nDeputy {i}: {p.name}")
    print(f"  Initial boresight (N-frame): {bRef[0]}")
    print(f"  Initial deputy→asteroid LOS: {(rA[0]-rD[0]) / la.norm(rA[0]-rD[0])}")

    # also compute orientation triad at t0
    C_NB0 = triad_from_boresight_and_tangent(bRef[0], vD[0])
    print("  Target orientation DCM (N<-B):")
    print(C_NB0)

    # plot boresight vs time
    t = n["t"].astype(float)
    fig,ax = plt.subplots(3,1,figsize=(6,6),sharex=True)
    for k in range(3):
        ax[k].plot(t, bRef[:,k], label=f"bRef[{k}]")
        ax[k].set_ylabel(["X","Y","Z"][k])
    ax[-1].set_xlabel("time [s]")
    fig.suptitle(f"Deputy {i} boresight vector in N-frame")
    plt.tight_layout()
    plt.show()

