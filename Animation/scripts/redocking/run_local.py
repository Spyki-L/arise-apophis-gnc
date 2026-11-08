# -*- coding: utf-8 -*-
"""
run_local.py - ONE-CLICK local verification (VS Code: just press Run).

Runs the whole re-docking chain on your machine and writes the figures, so you
can check (1) the per-deputy RETURN Delta-v and (2) the 6-DOF DDPG final-approach
docking logic, before committing to the full HPC training.

Edit DYNAMICS below: "hcw" (fast surrogate, runs anywhere) or "basilisk" (hi-fi,
uses your Basilisk + real chief_reference.npz).  No command-line arguments.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==================== RUN CONFIG (edit here) ====================
DYNAMICS = "basilisk"      # "hcw" | "basilisk"
SEED     = 2
# ===============================================================

import config_redock as C
from guidance import phase_a_gnc_return as pa
from eval import docking_demo

def main():
    art = C.ensure_art_dir()
    print("\n########## 1) PHASE-A: return Delta-v per deputy ##########")
    pa.USE_BASILISK = (DYNAMICS == "basilisk")
    pa.main()

    print("\n########## 2) PHASE-B: 6-DOF DDPG final-approach docking ##########")
    res = docking_demo.main(out=art, seed=SEED, dynamics=DYNAMICS)

    print("\n########## SUMMARY ##########")
    print(f"  dynamics backend : {DYNAMICS}")
    print(f"  deputies docked  : {res.n_captured}/8")
    print(f"  final miss [m]   : max {res.final_dist_m.max():.3f}")
    print(f"  figures          : {os.path.join(art, 'phaseA_return_dv.png')}")
    print(f"                     {os.path.join(art, 'dock_trajectories.png')}")
    print("  (open the two PNGs to verify your trajectories)\n")

if __name__ == "__main__":
    main()
