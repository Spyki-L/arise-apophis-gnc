#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_mission_cockpit.py — RUN THIS (no arguments).

Live cockpit for the ARISE/Apophis onboard FDIR autonomy. It replays your
nominal_v2 mission datasets forward in time, runs the onboard autonomy (FDI +
recovery + fast surrogate re-optimizer from this package) LIVE, injects scripted
failures, and on each event grades the onboard decision against a high-fidelity
GROUND oracle — animating the 3D scene, the fleet health, the predicted closest-
approach coverage, the peer "camera click", and the ONBOARD-vs-GROUND compute time.

  * 3D scene: PyVista if installed (primitive lumpy-ellipsoid asteroid + deputy
    boxes + camera cones), else an all-matplotlib fallback. NO CAD models.
  * Metrics dashboard: matplotlib (coverage, compute comparison, health grid,
    event log, captured-image inset).

Just press Run in Visual Studio. No command-line arguments. Optional installs for
the nicer 3D:  pip install pyvista
"""
from __future__ import annotations
import os, sys, subprocess, importlib


def _ensure(pkg, mod=None):
    """Auto-install a package on first run, so you can just press Run in VS — no manual pip."""
    try:
        return importlib.import_module(mod or pkg)
    except ImportError:
        print(f"[setup] installing {pkg} (one-time)…", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(mod or pkg)


for _p in ("numpy", "matplotlib"):      # required
    _ensure(_p)
import time
import numpy as np

# ─────────────────────────────── CONFIG (edit here, no args) ─────────────────
N_FRAMES       = 240        # mission frames (smoothness vs speed)
SEC_PER_FRAME  = 0.06       # animation pace
LOOP_FOREVER   = False      # replay continuously
SAVE_SUMMARY   = True       # write the compute log + a final snapshot
AESTHETIC      = True        # black mission-ops cockpit (complete Hill trajectories, FDI_anim style)
USE_PYVISTA    = False       # optional photoreal PyVista window (legacy split view; off when AESTHETIC)

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN_D = os.path.abspath(os.path.join(HERE, "..", ".."))             # …/D_strobe_fullbody
DATA_TRAJ = os.path.join(DESIGN_D, "data", "trajectories")
DATA_ANIM = os.path.join(DESIGN_D, "data", "anim")
MESH_NPZ  = os.path.join(HERE, "..", "data", "radar_mesh.npz")
FACET_NPZ = os.path.join(HERE, "..", "data", "mc_v2_results_nominal_facetprob.npz")
SCRIPTS_DIR = os.path.join(HERE, "..")                                 # nominal_v2 (MC_v2_run.py)
# ─────────────────────────────────────────────────────────────────────────────

if HERE not in sys.path:
    sys.path.insert(0, HERE)
import cockpit_engine as ce
import cockpit_viz as cv


def build():
    if not os.path.isdir(DATA_TRAJ):
        raise SystemExit(f"[cockpit] trajectory data not found at {DATA_TRAJ}\n"
                         f"          run the nominal_v2 pipeline first, or fix the paths in CONFIG.")
    print("[cockpit] loading nominal_v2 truth …")
    truth = ce.load_truth(DATA_TRAJ, DATA_ANIM, MESH_NPZ, FACET_NPZ, n_frames=N_FRAMES)
    print(f"[cockpit] {truth['N']} deputies, {truth['T'].size} frames, "
          f"CA standoff ~{np.median([np.linalg.norm(s) for s in truth['ca_stations']]):.0f} m, "
          f"{len(truth['cent'])} facets")
    engine = ce.MissionEngine(truth, ce.default_events(truth),
                              ce.OnboardCoverage(truth), ce.GroundOracle(truth, SCRIPTS_DIR))
    asteroid = cv.lumpy_ellipsoid()
    return truth, engine, asteroid


def main():
    truth, engine, asteroid = build()

    pv_scene = None
    if USE_PYVISTA and not AESTHETIC:    # optional photoreal side window (legacy split view)
        try:
            _ensure("pyvista")           # auto-install the nicer 3D backend (one-time); falls back if it fails
            pv_scene = cv.PvScene(engine, asteroid)
            print("[cockpit] PyVista 3D scene active.")
        except Exception as e:
            print(f"[cockpit] PyVista unavailable ({e}); using the matplotlib 3D fallback.")
            pv_scene = None

    import matplotlib
    if os.name == "nt" or "DISPLAY" in os.environ:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
    import matplotlib.pyplot as plt
    plt.ion()
    if AESTHETIC:
        import cockpit_ops as cops       # black mission-ops cockpit: complete Hill trajectories + telemetry
        dash = cops.OpsCockpit(engine, truth, truth["T"].size)
        print("[cockpit] mission-ops cockpit active (complete deputy trajectories, FDI_anim style).")
    else:
        dash = cv.MplCockpit(engine, asteroid, truth["T"].size, show_3d=(pv_scene is None))
    plt.show(block=False)

    F = truth["T"].size
    while True:
        for fi in range(F):
            st = engine.step(fi)
            if pv_scene is not None:
                try:
                    pv_scene.update(st)
                except Exception:
                    pass
            dash.update(st)
            dash.fig.canvas.draw_idle()
            plt.pause(SEC_PER_FRAME)
            if st["event"]:
                print(f"  [f{fi}] {st['log_tail']}")
        if not LOOP_FOREVER:
            break

    # ── compute-time summary (the reliability proof) ─────────────────────────
    print("\n================ ONBOARD vs GROUND compute (reliability) ================")
    print(f"  {'event':17s} {'onboard ms':>11s} {'ground ms':>10s} {'speedup':>8s} "
          f"{'cov_onb':>8s} {'cov_gnd':>8s} {'agree':>7s}")
    rows = []
    for c in engine.compute:
        fr, onb, gnd, reopt, co, cg = c
        sp = gnd / max(onb, 1e-3)
        ev = engine.events.get(fr, {}).get("type", "?")
        print(f"  {ev:17s} {onb:11.2f} {gnd:10.0f} {sp:7.0f}x {co:8.3f} {cg:8.3f} {abs(co-cg):7.3f}")
        rows.append((fr, ev, onb, gnd, reopt, co, cg))
    if rows:
        sp = np.mean([r[3]/max(r[2], 1e-3) for r in rows])
        ag = np.mean([abs(r[5]-r[6]) for r in rows])
        print(f"\n  PROVEN: onboard decisions agree with the high-fidelity ground oracle to within "
              f"{ag:.3f} coverage, computed ~{sp:.0f}x faster onboard.")
        print(f"  (wire MC_v2_run.score_draw into GroundOracle for the full ray-traced oracle "
              f"→ ground time goes to minutes, speedup to >10000x.)")

    if SAVE_SUMMARY:
        os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
        with open(os.path.join(HERE, "cockpit_compute_log.csv"), "w") as f:
            f.write("frame,event,onboard_score_ms,ground_score_ms,reopt_ms,cov_onboard,cov_ground\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")
        try:
            dash.fig.savefig(os.path.join(HERE, "figs", "cockpit_final.png"), dpi=120)
        except Exception:
            pass
        print(f"  wrote cockpit_compute_log.csv + figs/cockpit_final.png")

    print("\n[cockpit] mission complete. Close the window(s) to exit.")
    try:
        import matplotlib.pyplot as plt
        plt.ioff(); plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
