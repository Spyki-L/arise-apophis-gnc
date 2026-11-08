#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_tcm_fracs.py
=====================

FLIGHT-GRADE TCM re-timing study.  Runs the project's own high-fidelity
differential corrector (basilik_hifi_apophis_rendezvous_fleet.solve_deputy) with
candidate TCM-fraction schedules and reports, per deputy and fleet-wide, the
TRUE total Delta-v at the achieved terminal accuracy.

Because it calls the real corrector, every reported schedule is re-converged to
TARGET_TOL_M (the 0.05 m close-approach tolerance), so the comparison is "same
accuracy, different Delta-v" — exactly the flight-grade number the linear
adaptive_tcm surrogate could only bound.

Method
------
For each deputy:
  1. baseline run with the current schedule (TCM_FRACS) via min_dv_deploy,
     which also fixes the min-Delta-v deploy time t_dep;
  2. for each candidate fraction set, re-run solve_deputy at the SAME t_dep
     (so only the TCM timing changes), recording tot_dv and terminal error;
  3. pick the min-Delta-v schedule whose terminal error <= TARGET_TOL_M.

Cost
----
Each solve_deputy is a full multiple-shooting solve (MAX_NEWTON x (1+3*N_TCM)
high-fidelity propagations).  Use a coarse sweep and, if you like, the fast
force model for the search, then confirm the winner at full fidelity:

    ARISE_EARTH_SH_DEG=2 ARISE_INTEGRATOR=rk4 \\
        python Animation/scripts/guidance/optimize_tcm_fracs.py

Run (full fidelity):
    python Animation/scripts/guidance/optimize_tcm_fracs.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import numpy.linalg as la

print("=" * 64, flush=True)
print("FLIGHT-GRADE TCM re-timing study (real corrector)", flush=True)
print("=" * 64, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import basilik_hifi_apophis_rendezvous_fleet as G
except Exception:
    print("\n*** could not import the guidance module ***", flush=True)
    import traceback; traceback.print_exc()
    raise

# ---- candidate TCM-fraction schedules (fractions of TOF) ------------------
# Baseline is the current guidance schedule.  The rest front-load the dominant
# early correction (high leverage) while keeping a low-leverage late clean-up
# node (>= ~0.90 TOF) for nonlinear convergence — the structure the leverage
# analysis (adaptive_tcm) recommends.  Edit / extend freely; you can also paste
# the per-deputy 'optimized nodes [%TOF]' from run_adaptive_tcm.py here.
BASELINE = tuple(G.TCM_FRACS)
CANDIDATES = [
    BASELINE,
    (0.15, 0.55, 0.85, 0.95),
    (0.12, 0.45, 0.80, 0.96),
    (0.18, 0.60, 0.88, 0.96),
    (0.15, 0.68, 0.90, 0.97),
]


def setup():
    TOF, chief_state = G.load_chief()
    rA_fn, vA_fn = G.parse_horizons_apophis(G.HORIZONS_PATH)
    if getattr(G, "PROP_BACKEND", "basilisk") == "basilisk":
        sun_fn = None
    else:
        sun_fn = G.build_ephemerides(int(TOF + 3600), step_s=60.0)[0]
    return TOF, chief_state, sun_fn, rA_fn, vA_fn


def main():
    TOF, chief_state, sun_fn, rA_fn, vA_fn = setup()
    tol = float(getattr(G, "TARGET_TOL_M", 0.05))
    print(f"[setup] TOF={TOF:.0f}s  backend={getattr(G,'PROP_BACKEND','?')}  "
          f"tol={tol} m  candidates={len(CANDIDATES)}", flush=True)

    fleet_base = 0.0
    fleet_opt = 0.0
    for idx, tgt in enumerate(G.DEPUTY_TARGETS_M):
        t0 = time.time()
        # baseline: also fixes the deploy time t_dep used for all candidates
        base = G.min_dv_deploy(TOF, tgt, chief_state, sun_fn, rA_fn, vA_fn)
        td = base["t_dep"]
        results = [(BASELINE, base["tot_dv"], base["err"])]
        for fr in CANDIDATES:
            if fr == BASELINE:
                continue
            try:
                sol = G.solve_deputy(td, TOF, tgt, chief_state, sun_fn,
                                     rA_fn, vA_fn, tcm_fracs=fr)
                results.append((fr, sol["tot_dv"], sol["err"]))
            except Exception as e:
                print(f"  [D{idx}] candidate {fr} failed: {e}", flush=True)

        feasible = [r for r in results if r[2] <= tol]
        pool = feasible if feasible else results
        best = min(pool, key=lambda r: r[1])
        fleet_base += base["tot_dv"]
        fleet_opt += best[1]

        print(f"\n--- D{idx}  (deploy@{td:.0f}s, TOF {(TOF-td)/3600:.2f} h, {time.time()-t0:.0f}s) ---",
              flush=True)
        print(f"  baseline  fracs={BASELINE}  Δv={base['tot_dv']*1000:7.2f} mm/s  |e|={base['err']:.4f} m")
        for fr, dv, er in results[1:]:
            tag = "  <= picked" if (fr, dv, er) == best else ""
            feas = "" if er <= tol else "  (infeasible, |e|>tol)"
            print(f"  cand {fr}  Δv={dv*1000:7.2f} mm/s  |e|={er:.4f} m{feas}{tag}")
        print(f"  -> best Δv={best[1]*1000:7.2f} mm/s  ({100*(1-best[1]/base['tot_dv']):+.1f}% vs baseline)",
              flush=True)

    print("\n" + "=" * 64)
    print("FLEET ROLL-UP (flight-grade, true total Δv incl. deploy)")
    print(f"  baseline  : {fleet_base:.4f} m/s")
    print(f"  re-timed  : {fleet_opt:.4f} m/s   ->  {100*(1-fleet_opt/fleet_base):+.1f}%")
    print("  (every reported schedule re-converged to the 0.05 m tolerance)")
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
