#!/usr/bin/env python3
"""Chunked runner for exp_formpeer scenarios. Usage: exp_fp_part.py <SCEN> <n> [mad_floor]"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_formpeer as F
import run_fdi_nominal as RFN

CTX = {}
def _init():
    if "fleet" not in CTX:
        fleet, tca = RFN.load_fleet(F.DATA)
        events, miss = F.precompute(fleet, tca)
        CTX.update(fleet=fleet, tca=tca, events=events, miss=miss)

def one(args):
    scen, seed, floor = args
    rng = np.random.default_rng(seed)
    return F.run_trial(CTX["fleet"], CTX["events"], CTX["miss"], CTX["tca"], scen, rng, mad_floor=floor)

def main():
    scen = sys.argv[1]; n = int(sys.argv[2])
    floor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    t0 = time.time(); _init()
    args = [(scen, 40000 + hash(scen) % 1000 + i, floor) for i in range(n)]
    with Pool(2, initializer=_init) as p:
        rs = list(p.imap_unordered(one, args, chunksize=4))
    ok = wrong = missed = 0; fleetwide = 0; early_peer = 0; delays = []
    healthy_trials = 0; fa_trials = 0; peaks = []
    for r in rs:
        foc = r["focus"]; lab = r["settled"][foc]
        exp = {"FORM_LOCAL": "FORM", "NAV_GLOBAL_STARVED": "NAV"}.get(scen)
        if exp:
            if lab == exp:
                ok += 1
                if r["t_first"][foc] is not None: delays.append(r["t_first"][foc])
            elif lab == "NONE": missed += 1
            else: wrong += 1
            if r["first"][foc] == "PEER": early_peer += 1
            if r["n_form_trip"] >= 2: fleetwide += 1
        else:
            allh = all(v == "HEALTHY" for v in r["states"].values())
            healthy_trials += int(allh); fa_trials += int(not allh)
            peaks.append(r["peakS_peer"])
    lo, hi = F.wilson_ci(ok if exp else healthy_trials, n)
    out = dict(scen=scen, n=n, mad_floor=floor, correct=ok, wrong=wrong, missed=missed,
               ci=[lo, hi], early_peer_first=early_peer, n_form_fleetwide=fleetwide,
               healthy_trials=healthy_trials, fa_trials=fa_trials,
               peakS_peer_median=float(np.median(peaks)) if peaks else None,
               delay_median=float(np.median(delays)) if delays else None,
               wall_s=time.time() - t0)
    tag = scen + ("" if floor == 0.25 else f"_f{floor:g}")
    Path(f"../results/part_fp_{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("scen","n","correct","wrong","missed","ci","early_peer_first","n_form_fleetwide","healthy_trials","fa_trials","peakS_peer_median","delay_median")}), flush=True)
    print(f"wrote part_fp_{tag}.json in {out['wall_s']:.0f}s", flush=True)

if __name__ == "__main__":
    main()
