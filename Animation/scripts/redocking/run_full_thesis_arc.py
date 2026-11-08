# -*- coding: utf-8 -*-
r"""
run_full_thesis_arc.py — ONE-CLICK full mission arc, end to end (VS Code: press Run).

Covers the WHOLE thesis in one run:
    deploy → cruise → close approach → IMAGING (camera pictures)   [reuse existing npz]
      → PHASE-A return to the chief                                [hi-fi relative STM]
      → RL TERMINAL 6-DOF RE-DOCK on Basilisk                      [open-loop policy on Basilisk truth]
then SAVES every phase's npz and (optionally) renders the individual + combined videos.

No command-line arguments — edit CONFIG below.  The imaging leg REUSES your existing
data/trajectories/apophis_rendezvous_full_D*.npz (set RUN_IMAGING=True to re-run the
~1.5 h mission via run_nominal_and_hifi.py first).  The return + dock run in Basilisk;
this script auto-adds <basilisk-develop>/dist3 so `import Basilisk` works with no env vars.
"""
from __future__ import annotations
import os, sys, glob, json, time, subprocess, numpy as np

# ================= CONFIG (edit; no CLI args) =================
CONFIG = dict(
    SEED           = 7,
    DOCK_DYNAMICS  = "basilisk",   # "basilisk" (hi-fi 6-DOF) | "hcw" | "auto"
    POLICY         = "bc",         # trained policy (falls back to verified PD)
    OUT_DIR        = None,         # default: ./thesis_arc_out next to this script
    RUN_IMAGING    = False,        # False = reuse existing npz; True = re-run the ~1.5 h mission first
    RUN_RETURN     = True,
    RUN_DOCK       = True,
    RETURN_ARCS    = 140,
    MAKE_VIDEOS    = True,         # render the per-phase + basic combined mp4s (make_arc_videos)
    MAKE_CINEMATIC_FILM = True,    # render the premium cinematic full-mission-arc film (make_mission_film)
    FILM_SECONDS   = 62,          # cinematic film length
    FILM_FPS       = 30,
    FILM_DPI       = 100,
)
# Only used if RUN_IMAGING=True (mission re-run) — your Basilisk Python 3.11:
PY311 = r"C:\Program Files\Python311\python.exe"
# =============================================================

def _redock_dir(start):
    d=os.environ.get("REDOCK_DIR","")
    if d and os.path.isfile(os.path.join(d,"config_redock.py")): return d
    d=os.path.abspath(start)
    for _ in range(6):
        if os.path.isfile(os.path.join(d,"config_redock.py")): return d
        s=os.path.join(d,"redocking")
        if os.path.isfile(os.path.join(s,"config_redock.py")): return s
        d=os.path.dirname(d)
    for p in sys.path:
        if p and os.path.isfile(os.path.join(p,"config_redock.py")): return p
    raise RuntimeError("could not find redocking package; set env REDOCK_DIR")

RD=_redock_dir(os.path.dirname(os.path.abspath(__file__))); 
if RD not in sys.path: sys.path.insert(0, RD)

def _add_basilisk():
    try:
        import Basilisk; return True
    except Exception: pass
    d=RD
    for _ in range(8):
        c=os.path.join(d,"dist3")
        if os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0,c)
            try:
                import Basilisk; print(f"[setup] Basilisk via {c}"); return True
            except Exception: pass
        d=os.path.dirname(d)
    return False

def _load_cockpit():
    import importlib.util
    p=os.environ.get("COCKPIT_PY", os.path.join(RD,"run_redock_cockpit.py"))
    spec=importlib.util.spec_from_file_location("rrc",p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    cfg=dict(CONFIG); import config_redock as C
    out=cfg["OUT_DIR"] or os.path.join(os.path.dirname(os.path.abspath(__file__)),"thesis_arc_out")
    npz_dir=os.path.join(out,"arc_npz"); os.makedirs(npz_dir,exist_ok=True)
    t_all=time.time()

    # ---- STAGE 0: imaging npz present? ----
    files=sorted(glob.glob(os.path.join(C.ANIM_DIR,"data","trajectories","apophis_rendezvous_full_D*.npz")))
    print(f"[stage0] imaging npz: {len(files)} deputy files "
          + ("FOUND" if len(files)>=C.N_DEPUTIES else "MISSING (run the mission first)"))
    if cfg["RUN_IMAGING"]:
        mission=os.path.join(C.ANIM_DIR,"Designs","D_strobe_fullbody","nominal_v2","run_nominal_and_hifi.py")
        py=PY311 if os.path.exists(PY311) else sys.executable
        print("[stage1] re-running the full imaging mission (~1.5 h) ...")
        subprocess.run([py, mission], check=False)
        files=sorted(glob.glob(os.path.join(C.ANIM_DIR,"data","trajectories","apophis_rendezvous_full_D*.npz")))
    if len(files)<C.N_DEPUTIES:
        print("[abort] imaging npz missing; set RUN_IMAGING=True or run run_nominal_and_hifi.py first."); return

    dyn="hcw"
    if cfg["DOCK_DYNAMICS"] in ("basilisk","auto"):
        dyn="basilisk" if _add_basilisk() else "hcw"
        if dyn=="hcw": print("[setup] Basilisk not importable -> HCW surrogate for the dock.")
    rrc=_load_cockpit()

    # ---- STAGE 2: RL terminal dock (open-loop policy on Basilisk 6-DOF) ----
    dk=None
    if cfg["RUN_DOCK"]:
        print(f"[stage2] RL terminal dock — open-loop policy on {dyn} 6-DOF (all 8 deputies) ...")
        rcfg=dict(rrc.CONFIG); rcfg.update(DYNAMICS=dyn,POLICY=cfg["POLICY"],SEED=cfg["SEED"])
        dk,actor=rrc.rollout_dock(rcfg,dyn)
        np.savez_compressed(os.path.join(npz_dir,"dock_rollout.npz"),**{k:v for k,v in dk.items()})
        print(f"[stage2] saved dock_rollout.npz  (docked {dk['n_captured']}/8)")

    # ---- STAGE 3: Phase-A return (hi-fi relative STM) ----
    if cfg["RUN_RETURN"] and dk is not None:
        print("[stage3] Phase-A return (relative-motion STM; hi-fi if Basilisk present) ...")
        from guidance.phase_a_gnc_return import plan_return
        pa=plan_return(files=files)
        formation=np.asarray(pa.meta["X0"])[:,:3]           # real post-imaging LVLH positions
        pb0=dk["traj"][0]
        arcs=rrc.synth_return(formation,pb0,cfg["RETURN_ARCS"])
        np.savez_compressed(os.path.join(npz_dir,"return_arc.npz"),arcs=arcs,port_of=dk["port_of"],
                            dv_total=np.float64(float(pa.dv_total_mps.sum())),formation=formation,pb0=pb0,
                            dv_per_deputy=pa.dv_total_mps,backend=str(pa.backend))
        print(f"[stage3] saved return_arc.npz  (fleet return Δv {pa.dv_total_mps.sum():.1f} m/s, {pa.backend})")

    # ---- STAGE 4: manifest ----
    man=dict(seed=cfg["SEED"],dock_dynamics=dyn,policy=cfg["POLICY"],
             imaging_npz=len(files),generated=time.strftime("%Y-%m-%d %H:%M"),
             docked=(int(dk["n_captured"]) if dk is not None else None))
    json.dump(man,open(os.path.join(out,"run_manifest.json"),"w"),indent=2)
    print(f"[stage4] wrote run_manifest.json")

    # ---- STAGE 5: videos ----
    if cfg["MAKE_VIDEOS"]:
        print("[stage5] rendering videos (imaging + return + dock + combined) ...")
        import importlib.util
        mp=os.path.join(RD,"make_arc_videos.py")
        spec=importlib.util.spec_from_file_location("mav",mp); mav=importlib.util.module_from_spec(spec); spec.loader.exec_module(mav)
        mav.make_all(os.path.join(out,"videos"),imaging_frames=90,
                     return_np=os.path.join(npz_dir,"return_arc.npz"),
                     dock_np=os.path.join(npz_dir,"dock_rollout.npz"),which=("00","01","02","03"))
    # ---- STAGE 6: cinematic full-mission-arc film (Apophis CAD, camera feeds, FDIR + RL brain) ----
    if cfg.get("MAKE_CINEMATIC_FILM", True):
        print("[stage6] rendering the CINEMATIC full-mission-arc film "
              "(separation -> CA imaging w/ Apophis CAD + camera feeds -> return -> RL dock) ...")
        try:
            import importlib.util
            fp=os.environ.get("MMF_PY", os.path.join(RD,"make_mission_film.py"))
            spec=importlib.util.spec_from_file_location("mmf",fp); mmf=importlib.util.module_from_spec(spec); spec.loader.exec_module(mmf)
            mmf.CONFIG.update(ARC_DIR=npz_dir, OUT=os.path.join(out,"videos"),
                              SECONDS=cfg.get("FILM_SECONDS",62), FPS=cfg.get("FILM_FPS",30),
                              DPI=cfg.get("FILM_DPI",100), PREVIEW=False, SEED=cfg["SEED"])
            mmf.main()
        except Exception as e:
            import traceback; print(f"[stage6] cinematic film failed ({type(e).__name__}: {e})"); traceback.print_exc()

    print(f"\n[done] full thesis arc -> {out}  ({time.time()-t_all:.0f}s)")

if __name__=="__main__":
    main()
