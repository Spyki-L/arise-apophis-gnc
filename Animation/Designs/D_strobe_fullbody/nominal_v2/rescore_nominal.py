#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rescore_nominal.py - re-score the SAVED nominal fleet with the FIXED score_draw.

No Basilisk re-run: this only re-runs the cheap imaging/coverage scoring on the
already-flown draw_0000 trajectories, over M spin-phase sub-draws. It exists to show
the corrected (non-bimodal) coverage band after the dark-side-gate fix in MC_v2_run.py,
and is the same path you'd use to re-score the 256 saved MC draws later.

  --occ 1   faithful occlusion-on band (run on the Windows/Basilisk box; ~slow)
  --occ 0   fast sandbox check (occlusion off; absolute level ~15% high, shape correct)
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "modules"))
import MC_v2_run as MC
import brdf as BRDF

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--occ", type=int, default=0)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--nframe", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20290413)
    ap.add_argument("--tag", default="fixed")
    a = ap.parse_args()

    draw = os.path.join(HERE, "data", "mc", "draw_0000")
    fleet = MC.load_fleet(os.path.join(draw, "trajectories", "apophis_rendezvous_full_D*.npz"))
    sun = np.load(os.path.join(draw, "apophis", "attitude_truth.npz"), allow_pickle=True)["sun_hat"]
    rng = np.random.default_rng(a.seed)

    W = []; L = []; D = []; covc = None
    for mi in range(a.m):
        R, tp, op = MC.tumbler_dcm(rng)
        h = BRDF.sample_hapke(rng)
        sc = MC.score_draw(fleet, R, sun, h, occ=bool(a.occ), nframe=a.nframe, topup=True)
        W.append(sc["whole"]); L.append(sc["sunlit"]); D.append(sc["dark"])
        covc = sc["stereo_mask"].astype(float) if covc is None else covc + sc["stereo_mask"]
        print("ph%2d whole=%5.1f sun=%5.1f dark=%5.1f" % (mi, sc["whole"], sc["sunlit"], sc["dark"]), flush=True)

    W = np.array(W); L = np.array(L); D = np.array(D)
    pc = lambda x, q: float(np.percentile(x, q))
    out = dict(occ=bool(a.occ), m=a.m, nframe=a.nframe,
               whole=dict(mean=float(W.mean()), P5=pc(W, 5), P50=pc(W, 50), P95=pc(W, 95), std=float(W.std())),
               sunlit=dict(mean=float(L.mean()), P5=pc(L, 5), P50=pc(L, 50), P95=pc(L, 95)),
               dark=dict(mean=float(D.mean()), P5=pc(D, 5), P50=pc(D, 50), P95=pc(D, 95)),
               whole_list=W.tolist(), sunlit_list=L.tolist(), dark_list=D.tolist(),
               facet_prob=(covc / a.m).tolist())
    op = os.path.join(HERE, "data", f"mc_v2_results_nominal_{a.tag}.json")
    json.dump(out, open(op, "w"), indent=1)
    print("\nBAND whole: mean %.1f  P5 %.1f P50 %.1f P95 %.1f  std %.1f  | #<5=%d #>95=%d #mid(20-80)=%d"
          % (W.mean(), pc(W, 5), pc(W, 50), pc(W, 95), W.std(), (W < 5).sum(), (W > 95).sum(), ((W > 20) & (W < 80)).sum()))
    print("wrote", op)

if __name__ == "__main__":
    main()
