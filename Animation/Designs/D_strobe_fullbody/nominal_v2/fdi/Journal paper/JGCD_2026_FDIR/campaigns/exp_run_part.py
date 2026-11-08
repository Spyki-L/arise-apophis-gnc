#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_run_part.py -- chunked, parallel runner for the Apophis MC campaigns.

Usage:  python3 exp_run_part.py <part> <n_trials>
Parts:
  det:NAV det:ATT det:PROP det:COMM det:PFA
  iso:legacy iso:fixed
  mag:NAV mag:ATT mag:PROP
Each part writes results/part_<name>.json (+ delays npz for det parts).
Merge later with exp_merge.py.
"""
from __future__ import annotations
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import exp_apophis_mc as M          # noqa: E402
import run_fdi_nominal as RFN       # noqa: E402

RESULTS = CODE.parent / "results"
FLEET = None


def _init():
    global FLEET
    if FLEET is None:
        FLEET, _ = RFN.load_fleet(M.DATA)
    return FLEET


def trial_det(args):
    sc, seed, legacy, mag_kw = args
    rng = np.random.default_rng(seed)
    g = FLEET[int(rng.integers(0, len(FLEET)))]
    if sc == "NONE":
        tf = 1e18
    else:
        ph = float(rng.uniform(0.1, 0.9))
        tf = float(g["t"][0]) + ph * (float(g["t"][-1]) - float(g["t"][0]))
    lab, td = M.run_dep(g, sc, tf, rng, legacy=legacy, **(mag_kw or {}))
    return lab, td


def run_block(sc, n, seed0, legacy=False, mag_kw=None, workers=2):
    args = [(sc, seed0 + i, legacy, mag_kw) for i in range(n)]
    with Pool(workers, initializer=_init) as p:
        out = list(p.imap_unordered(trial_det, args, chunksize=8))
    return out


def main():
    part = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    t0 = time.time()
    _init()
    kind, name = part.split(":")
    res = {}

    if kind == "det":
        sc = "NONE" if name == "PFA" else name
        seed0 = dict(NAV=1000, ATT=2000, PROP=3000, COMM=4000, PFA=5000)[name]
        out = run_block(sc, n, seed0)
        labs = [o[0] for o in out]
        tds = [o[1] for o in out if o[1] is not None and o[0] == sc]
        if name == "PFA":
            fa = sum(1 for l in labs if l != "NONE")
            lo, hi = M.wilson_ci(fa, n)
            res = dict(part=part, n=n, fa=fa, Pfa=fa / n, ci=[lo, hi])
        else:
            nc = sum(1 for l in labs if l == sc)
            lo, hi = M.wilson_ci(nc, n)
            lat = np.array(tds) if tds else np.array([np.nan])
            conf = {}
            for l in labs:
                conf[l] = conf.get(l, 0) + 1
            res = dict(part=part, n=n, nc=nc, Pd=nc / n, ci=[lo, hi],
                       lat_median=float(np.nanmedian(lat)),
                       lat_p95=float(np.nanpercentile(lat, 95)), confusion=conf)
            np.savez_compressed(RESULTS / f"delays_{name}.npz", delays=lat)

    elif kind == "iso":
        legacy = (name == "legacy")
        LAB = ["NAV", "ATT", "PROP", "COMM"]
        conf = {}
        for sc in LAB:
            seed0 = 10000 + LAB.index(sc) * 1000     # same seeds for both modes
            out = run_block(sc, n, seed0, legacy=legacy)
            row = {}
            for l, _ in out:
                row[l] = row.get(l, 0) + 1
            conf[sc] = row
            print(f"  iso:{name} {sc} diag={100*row.get(sc,0)/n:.1f}%", flush=True)
        res = dict(part=part, n=n, confusion=conf)

    elif kind == "mag":
        grids = {"NAV": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0],
                 "ATT": [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0],
                 "PROP": [1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0, 8.0]}
        rows = []
        for gi, m in enumerate(grids[name]):
            kw = {name: m} if name != "ATT" else {"ATT": m, "RATE": 0.05 * m}
            out = run_block(name, n, 20000 + gi * 500, mag_kw=kw)
            nc = sum(1 for l, _ in out if l == name)
            tds = [td for l, td in out if td is not None and l == name]
            lo, hi = M.wilson_ci(nc, n)
            med = float(np.median(tds)) if tds else float("nan")
            rows.append(dict(mag=m, Pd=nc / n, ci=[lo, hi], lat_median=med, n=n))
            print(f"  mag:{name} m={m}: Pd={nc/n:.2f} med={med:.0f}s", flush=True)
        res = dict(part=part, n=n, rows=rows)

    res["wall_s"] = time.time() - t0
    fn = RESULTS / f"part_{kind}_{name}.json"
    fn.write_text(json.dumps(res, indent=1))
    print(f"wrote {fn.name} in {res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
