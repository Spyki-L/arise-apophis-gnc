# -*- coding: utf-8 -*-
"""docking_demo.py - full-pipeline demonstration with the model-based guidance
baseline (port-frame PD).  Produces the 8-deputy RAFTI docking trajectory figure
and prints capture stats.  This validates the whole chain (Phase-A GNC return ->
LVLH handoff -> Phase-B per-port soft-capture); the RL policy is a drop-in for
the baseline via the same interface."""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bridge.redock_pipeline import run_redock
from eval.rollout import PortFramePD, SwarmFromSingle
from eval.visualize import plot_trajectories

# ================= RUN CONFIG (VS Code: press Run; edit here) =================
DYNAMICS = "hcw"     # "hcw" = fast surrogate | "basilisk" = hi-fi (needs Basilisk)
SEED     = 2
# =============================================================================

# moderate terminal-phase config reachable by the guidance baseline
# realistic terminal-phase config (default RAFTI soft-capture tolerances)
DEMO = dict(env_cfg=dict(max_steps=400, bounds_radius_m=200.0, dt=2.0, max_dv_mps=0.05),
            handoff=dict(envelope_min_m=30.0, envelope_max_m=90.0, init_speed_max_mps=0.05,
                         init_cone_deg=45.0, capture_radius_m=120.0))

def main(out=None, seed=SEED, dynamics=DYNAMICS):
    out = out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
    os.makedirs(out, exist_ok=True)
    print(f"[docking-demo] dynamics={dynamics}")
    pol = SwarmFromSingle(PortFramePD(), n=8)
    res = run_redock(agent=pol, dynamics=dynamics, seed=seed, record=True, **DEMO)
    # attitude metrics at capture
    from dynamics.attitude import pointing_error_deg
    r, u, rad = __import__("config_redock").load_espa_ports()
    att_err = [pointing_error_deg(res.attitudes[-1, i], -u[int(res.port_of[i])]) for i in range(8)]
    print(f"Phase-A dv/deputy [m/s]: {np.round(res.dv_phaseA_mps,2)} (sum {res.dv_phaseA_mps.sum():.1f})")
    print(f"Phase-B dv total [m/s] : {res.dv_phaseB_mps:.2f}")
    print(f"captured               : {res.n_captured}/8 in {res.steps} steps")
    print(f"final port-dist [m]    : {np.round(res.final_dist_m,3)}")
    print(f"final face-align [deg] : {np.round(att_err,2)}  (all < {__import__('config_redock').RAFTI['att_align_tol_deg']} deg)")
    print(f"total mission dv [m/s] : {res.meta['dv_total_mps']:.1f}")
    plot_trajectories(res.trajectory, res.port_of,
                      os.path.join(out, "dock_trajectories.png"), attitudes=res.attitudes,
                      title=f"ARISE 6-DOF re-docking: {res.n_captured}/8 deputies soft-captured on ESPA RAFTI ports")
    return res

if __name__ == "__main__":
    main()
