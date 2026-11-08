#!/usr/bin/env python3
"""Score the CENTRAL Design D trajectories with explicit fidelity knobs (sandbox, numpy-only,
no Basilisk, no MC disturbance). Resumable per-orientation (fits the per-call limit)."""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "modules"))
import MC_v2_run as MC, brdf as BRDF

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--occ", type=int, default=0); ap.add_argument("--model", default="hapke")
ap.add_argument("--sun", choices=["default", "real"], default="real"); ap.add_argument("--topup", type=int, default=1)
ap.add_argument("--m", type=int, default=8); ap.add_argument("--nframe", type=int, default=7); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--fresh", action="store_true", help="truncate this tag's jsonl before scoring")
a = ap.parse_args()
TRAJ = os.path.join(HERE, "..", "data", "trajectories", "apophis_rendezvous_full_D*.npz")
MESH = os.path.join(HERE, "data", "radar_mesh.npz")
JL = os.path.join(HERE, f"_nom_{a.tag}.jsonl")
sun = MC.default_sun_hat() if a.sun == "default" else MC.load_geometry()
fleet = None
for _ in range(6):
    try: fleet = MC.load_fleet(TRAJ); assert fleet; break
    except Exception: time.sleep(2)
if a.fresh and os.path.exists(JL):
    open(JL, "w").close()
done = set()
if os.path.exists(JL):
    for ln in open(JL):
        try: done.add(json.loads(ln)["mi"])
        except Exception: pass
todo = [mi for mi in range(a.m) if mi not in done][:a.batch]
jf = open(JL, "a")
for mi in todo:
    rng = np.random.default_rng(1000 + mi)        # shared per-orientation seed across configs
    R, _, _ = MC.tumbler_dcm(rng); hp = BRDF.sample_hapke(rng)
    t0 = time.time()
    sc = MC.score_draw(fleet, R, sun, hp, mesh_npz=MESH, occ=bool(a.occ), nframe=a.nframe, topup=bool(a.topup), model=a.model)
    rec = dict(mi=mi, whole=sc["whole"], sunlit=sc["sunlit"], dark=sc["dark"], secs=round(time.time()-t0, 1))
    jf.write(json.dumps(rec) + "\n"); jf.flush()
    print(f"[{a.tag}] mi={mi} whole={sc['whole']:.1f} sunlit={sc['sunlit']:.1f} dark={sc['dark']:.1f} ({rec['secs']}s)", flush=True)
jf.close()
print(f"[{a.tag}] {len(done)+len(todo)}/{a.m} done", flush=True)
rows=[json.loads(l) for l in open(JL)] if os.path.exists(JL) else []
if len(rows) >= a.m:
    arr=np.array([[r["whole"],r["sunlit"],r["dark"]] for r in rows])
    print(f"[{a.tag}] SUMMARY N={len(arr)} whole {arr[:,0].mean():.1f}% sunlit {arr[:,1].mean():.1f}% dark {arr[:,2].mean():.1f}%", flush=True)
