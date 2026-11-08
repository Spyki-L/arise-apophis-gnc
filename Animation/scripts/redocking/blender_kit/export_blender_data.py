# -*- coding: utf-8 -*-
"""
export_blender_data.py — packs the REAL ARISE mission datasets into a Blender-
friendly bundle (blender_kit/data/): mission_bundle.json + apophis.obj.

Run with your normal python (numpy only), no arguments:
    python export_blender_data.py
Then open/run build_mission_film.py inside Blender.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.dirname(HERE)                       # scripts/redocking
sys.path.insert(0, RD)
os.environ.setdefault("REDOCK_DIR", RD)
import config_redock as C
import apophis_cad as ACAD
import make_mission_film as MF                   # reuse the tested loaders

FPS = 30
SHOTS = dict(                                    # frame ranges of the 150 s film
    s1_departure=(0, 599), s2_orbits=(600, 1199), s3_separation=(1200, 1949),
    s4_imaging=(1950, 3149), s5_return=(3150, 3599), s6_dock=(3600, 4199),
    s7_finale=(4200, 4499),
)

def mrp_to_quat(s):
    s = np.asarray(s, float); n2 = float(s @ s)
    q0 = (1.0 - n2) / (1.0 + n2); qv = 2.0 * s / (1.0 + n2)
    q = np.array([q0, qv[0], qv[1], qv[2]])
    return q / np.linalg.norm(q)

def mat_to_quat(R):
    R = np.asarray(R, float)
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2; w = S/4
        x = (R[2,1]-R[1,2])/S; y = (R[0,2]-R[2,0])/S; z = (R[1,0]-R[0,1])/S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/S; x=S/4
        y=(R[0,1]+R[1,0])/S; z=(R[0,2]+R[2,0])/S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/S
        x=(R[0,1]+R[1,0])/S; y=S/4; z=(R[1,2]+R[2,1])/S
    else:
        S = np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/S
        x=(R[0,2]+R[2,0])/S; y=(R[1,2]+R[2,1])/S; z=S/4
    q = np.array([w,x,y,z]); return q/np.linalg.norm(q)

def aim_quat(fwd, up=(0,0,1)):
    """Quaternion rotating body +X onto fwd (world), body +Z near world up."""
    f = np.asarray(fwd,float); f = f/ (np.linalg.norm(f)+1e-12)
    u = np.asarray(up,float)
    if abs(f @ u) > 0.95: u = np.array([0,1,0.0])
    y = np.cross(u,f); y/=np.linalg.norm(y)+1e-12; z = np.cross(f,y)
    return mat_to_quat(np.column_stack([f,y,z]))

def resample(arr, n):
    arr = np.asarray(arr,float); T = arr.shape[0]
    idx = np.linspace(0, T-1, n)
    lo = np.floor(idx).astype(int); hi = np.minimum(lo+1, T-1); w = (idx-lo)
    out = arr[lo]*(1-np.expand_dims(w,tuple(range(1,arr.ndim)))) \
        + arr[hi]*np.expand_dims(w,tuple(range(1,arr.ndim)))
    return out

def main():
    print("[export] loading mission datasets (reusing film loaders)…")
    MF.load_epoch_map()
    D = MF.load_all(os.path.join(RD, "thesis_arc_out", "arc_npz"))
    out = {}
    out["fps"] = FPS; out["shots"] = SHOTS

    # ---- Apophis mesh -> OBJ (scene units: metres) ------------------------
    ap = D["ap"]; V, F = ap["V"], ap["F"]
    with open(os.path.join(HERE,"data","apophis.obj"),"w") as f:
        f.write("o Apophis\n")
        for v in V: f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for t in F: f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
    out["apophis_scale_m"] = float(ap["scale"])
    out["sun_dir"] = list(map(float, ap["sun"]))

    # ---- S2 orbit overview: REAL ECI tracks, Earth-radius units -----------
    import glob as _g
    d0 = np.load(sorted(_g.glob(os.path.join(C.ANIM_DIR,"data","trajectories",
                 "apophis_rendezvous_full_D*.npz")))[0], allow_pickle=True)
    Re = 6371e3
    rC = resample(d0["rC_N"], 400) / Re
    rA = resample(d0["rA_N"], 400) / Re
    out["orbit"] = dict(chief=rC.round(4).tolist(), apophis=rA.round(4).tolist())

    # ---- S3 separation: real 6-DOF (metres) -------------------------------
    sep = D["sep"]; n = 240
    tr = resample(sep["tr"], n)                        # (n,8,3)
    qt = np.zeros((n,8,4))
    att = resample(sep["att"], n)
    for i in range(n):
        for k in range(8):
            qt[i,k] = mrp_to_quat(att[i,k])
    out["separation"] = dict(pos=tr.round(4).tolist(), quat=qt.round(5).tolist())

    # ---- S4 imaging: formation + tumble + REAL strobe events --------------
    mis = D["mission"]; apo = np.asarray(mis["apo"][-1]); form = np.asarray(mis["tr"][-1])
    off = (form - apo)                                  # deputy offsets from asteroid, m
    qaim = [aim_quat(-off[k]).round(5).tolist() for k in range(8)]
    Rser = ap["Rser"] if ap["Rser"] is not None else np.eye(3)[None]
    qser = [mat_to_quat(R).round(5).tolist() for R in Rser]
    f0,f1 = SHOTS["s4_imaging"]; nfr = f1-f0+1
    ev = []
    for e in D["img_events"]:
        u = (e["t"]+2.2)/4.4
        ev.append(dict(frame=int(f0+u*(nfr-1)), k=int(e["k"]),
                       rng=float(e["rng"]), snr=float(e["snr"])))
    out["imaging"] = dict(offsets=off.round(2).tolist(), aim_quat=qaim,
                          tumble_quat=qser, events=ev)

    # ---- S5 return + S6 dock: real trajectories (metres) ------------------
    ret = D["ret"]; out["ret"] = dict(pos=resample(ret["arcs"],150).round(3).tolist())
    dk = D["dock"]; Tb = dk["traj"].shape[0]
    knots_u = [0,0.30,0.52,0.95,1.0]; knots_k = [0,220,278,350,Tb-1]   # slow-mo warp
    nfrd = SHOTS["s6_dock"][1]-SHOTS["s6_dock"][0]+1
    ks = np.interp(np.linspace(0,1,nfrd), knots_u, knots_k)
    pos = np.zeros((nfrd,8,3)); qt = np.zeros((nfrd,8,4))
    att = np.asarray(dk["att"],float); trj = np.asarray(dk["traj"],float)
    for i,kf in enumerate(ks):
        k0 = int(kf)
        pos[i] = trj[k0]
        for k in range(8):
            qt[i,k] = mrp_to_quat(att[k0,k])
    caps_f = [int(np.interp(float(D["ft"][k]),knots_k,[0,0.30,0.52,0.95,1.0])*(nfrd-1))
              + SHOTS["s6_dock"][0] for k in range(8)]
    out["dock"] = dict(pos=pos.round(4).tolist(), quat=qt.round(5).tolist(),
                       capture_frames=caps_f)

    # ---- ring geometry (real config) --------------------------------------
    r,u,rad = C.load_espa_ports()
    out["ring"] = dict(ports=np.asarray(r).round(4).tolist(),
                       dirs=np.asarray(u).round(4).tolist(),
                       radius=float(rad), height=float(C.ESPA["height_m"]),
                       deputy_dims=list(C.ESPA["deputy_dims"]))

    # ---- title/finale stats (real) ----------------------------------------
    out["stats"] = dict(images=int(D["nimg"].sum()), coverage=float(D["cov"]["whole"]),
                        dark=float(D["cov"]["dark"]),
                        dv_return=float(np.asarray(ret["dv_total"])),
                        captures=int(np.asarray(dk["n_captured"])))
    p = os.path.join(HERE,"data","mission_bundle.json")
    json.dump(out, open(p,"w"))
    print(f"[export] wrote {p} ({os.path.getsize(p)/1e6:.1f} MB) + apophis.obj")

if __name__ == "__main__":
    main()
