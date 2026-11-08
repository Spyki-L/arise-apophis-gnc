#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_apophis_mc.py -- Journal MC campaigns on the REAL 8-deputy Apophis fleet.

Campaigns (all on the fixed causal FDI core, numpy only):
  [A] detection      : Pd / Pfa / latency per fault class (NAV/ATT/PROP/COMM), Wilson CIs
  [B] isolation      : confusion matrix, fixed core AND legacy core (before/after)
  [C] magnitude      : Pd + latency vs injected fault magnitude (detectability curves)

Writes results/apophis_mc.json + raw delay arrays results/apophis_mc_delays.npz
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import numpy.linalg as la

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from fdi_core import FDICore, FDIConfig, FDIState, FaultType  # noqa: E402
import run_fdi_nominal as RFN                                  # noqa: E402

RESULTS = CODE.parent / "results"
RESULTS.mkdir(exist_ok=True)

DATA = Path("/sessions/cool-hopeful-franklin/mnt/Animation/Designs/D_strobe_fullbody/nominal_v2/data/mc/draw_0000/_nav_stage")

HB = 60.0


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))


def fdi_cfg(legacy=False):
    c = FDIConfig(use_nav_channel=True, use_att_channel=True, use_comm_channel=True,
                  use_form_channel=False, use_peer_channel=False)
    c.nu_nav = 1.5; c.leak_nav = 0.05; c.h_nav = 15.0
    if legacy:
        c.legacy_isolation = True
    return c


def run_dep(g, scn, tf, rng, legacy=False,
            NAV=5.0, ATT=1.0, RATE=0.05, PROP=5.0):
    """Stream one deputy timeline through the core with a fault of class `scn`
    injected from time tf onward. Returns (first_label, detect_delay_s)."""
    t = g["t"]
    fdi = FDICore(fdi_cfg(legacy))
    fdi.reset(float(t[0]))
    hb = HB
    first_label = FaultType.NONE
    t_detect = None
    for k in range(t.size):
        tk = float(t[k]); on = tk >= tf
        innov = g["ypre"][k]
        if (not np.all(np.isfinite(innov))) or bool(g["rejected"][k]):
            ik = Sk = None
        else:
            innov = innov.copy()
            if scn == "NAV" and on:
                innov = innov + NAV * g["sig"][k]
            ik = innov; sg = np.maximum(g["sig"][k], 1e-6); Sk = np.diag(sg ** 2)
        dvc = g["dv"][k]; burn = bool(la.norm(dvc) > 0.0); de = Pvv = None
        if burn and k > 0:
            Pvv = g["P"][k, 3:6, 3:6]
            de = (g["x"][k, 3:6] - g["x"][k - 1, 3:6]) - dvc
            if scn == "PROP" and on:
                sv = np.sqrt(np.maximum(np.diag(Pvv), 1e-6))
                de = de + PROP * sv * rng.normal(size=3)
        sB = g["sBR"][k]; oB = g["oBR"][k]
        if scn == "ATT" and on:
            u = rng.normal(size=3); u /= la.norm(u)
            sB = sB + np.tan(0.25 * np.deg2rad(ATT)) * u
            u = rng.normal(size=3); u /= la.norm(u)
            oB = oB + np.deg2rad(RATE) * u
        dt = 0.0 if k == 0 else float(t[k] - t[k - 1])
        hb = hb + dt if (scn == "COMM" and on) else HB
        s = fdi.step(t=tk, innov=ik, S_innov=Sk, dv_est_H=de, P_vv=Pvv, burn_flag=burn,
                     sigma_BR=sB, omega_BR_B=oB, hb_age=hb, hb_nominal=HB)
        if s.state.state in (FDIState.SUSPECT, FDIState.FAULTED, FDIState.DEAD):
            if first_label is FaultType.NONE and s.state.fault_type is not FaultType.NONE:
                first_label = s.state.fault_type
                if on and t_detect is None and s.state.fault_type.name == scn:
                    t_detect = tk - tf
    return first_label.name, t_detect


def detection_campaign(fleet, n_trials, seed=0):
    rng = np.random.default_rng(seed)
    SC = ["NAV", "ATT", "PROP", "COMM"]
    LAB = ["NONE", "NAV", "ATT", "PROP", "COMM", "FORM", "PEER"]
    res = {}; conf = {s: {l: 0 for l in LAB} for s in SC}
    delays = {}
    for sc in SC:
        nc = 0; lat = []
        for _ in range(n_trials):
            g = fleet[int(rng.integers(0, len(fleet)))]
            ph = float(rng.uniform(0.1, 0.9))
            tf = float(g["t"][0]) + ph * (float(g["t"][-1]) - float(g["t"][0]))
            lab, td = run_dep(g, sc, tf, rng)
            conf[sc][lab] += 1
            if lab == sc:
                nc += 1
                if td is not None:
                    lat.append(td)
        lo, hi = wilson_ci(nc, n_trials)
        lat_a = np.array(lat) if lat else np.array([np.nan])
        delays[sc] = lat_a
        res[sc] = dict(Pd=nc / n_trials, Pd_ci=[lo, hi], n=n_trials,
                       lat_median=float(np.nanmedian(lat_a)),
                       lat_p95=float(np.nanpercentile(lat_a, 95)))
        print(f"  [A] {sc}: Pd={nc/n_trials:.3f} [{lo:.3f},{hi:.3f}] "
              f"med={np.nanmedian(lat_a):.0f}s p95={np.nanpercentile(lat_a,95):.0f}s", flush=True)
    fa = 0
    for _ in range(n_trials):
        g = fleet[int(rng.integers(0, len(fleet)))]
        lab, _ = run_dep(g, "NONE", 1e18, rng)
        fa += int(lab != "NONE")
    plo, phi = wilson_ci(fa, n_trials)
    print(f"  [A] Pfa={fa/n_trials:.4f} [{plo:.4f},{phi:.4f}] (n={n_trials})", flush=True)
    return dict(per_scenario=res, confusion=conf,
                Pfa=fa / n_trials, Pfa_ci=[plo, phi], Pfa_n=n_trials), delays


def isolation_before_after(fleet, n_trials, seed=1):
    LAB = ["NONE", "NAV", "ATT", "PROP", "COMM"]
    out = {}
    for mode, legacy in (("legacy", True), ("fixed", False)):
        rng = np.random.default_rng(seed)
        conf = {s: {l: 0 for l in LAB} for s in LAB[1:]}
        for sc in LAB[1:]:
            for _ in range(n_trials):
                g = fleet[int(rng.integers(0, len(fleet)))]
                ph = float(rng.uniform(0.1, 0.9))
                tf = float(g["t"][0]) + ph * (float(g["t"][-1]) - float(g["t"][0]))
                lab, _ = run_dep(g, sc, tf, rng, legacy=legacy)
                if lab not in conf[sc]:
                    conf[sc][lab] = 0
                conf[sc][lab] += 1
            diag = conf[sc][sc] / n_trials
            print(f"  [B:{mode}] {sc}: diag={100*diag:.1f}%", flush=True)
        out[mode] = conf
    return out


def magnitude_sweep(fleet, n_trials, seed=2):
    grids = {
        "NAV": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0],       # innovation bias [sigma]
        "ATT": [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0],       # attitude bias [deg]
        "PROP": [1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0, 8.0],     # dv mismatch [sigma]
    }
    out = {}
    for sc, grid in grids.items():
        rows = []
        for m in grid:
            rng = np.random.default_rng(seed)
            nc = 0; lat = []
            for _ in range(n_trials):
                g = fleet[int(rng.integers(0, len(fleet)))]
                ph = float(rng.uniform(0.1, 0.9))
                tf = float(g["t"][0]) + ph * (float(g["t"][-1]) - float(g["t"][0]))
                kw = {sc: m} if sc != "ATT" else {"ATT": m, "RATE": 0.05 * m}
                lab, td = run_dep(g, sc, tf, rng, **kw)
                if lab == sc:
                    nc += 1
                    if td is not None:
                        lat.append(td)
            lo, hi = wilson_ci(nc, n_trials)
            med = float(np.nanmedian(lat)) if lat else float("nan")
            rows.append(dict(mag=m, Pd=nc / n_trials, ci=[lo, hi], lat_median=med, n=n_trials))
            print(f"  [C] {sc} m={m}: Pd={nc/n_trials:.2f} med={med:.0f}s", flush=True)
        out[sc] = rows
    return out


def main():
    t0 = time.time()
    n_det = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_iso = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    n_mag = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    print(f"loading fleet from {DATA}", flush=True)
    fleet, _tca = RFN.load_fleet(DATA)
    print(f"fleet loaded ({time.time()-t0:.0f}s)", flush=True)

    det, delays = detection_campaign(fleet, n_det)
    iso = isolation_before_after(fleet, n_iso)
    mag = magnitude_sweep(fleet, n_mag)

    out = dict(detection=det, isolation=iso, magnitude=mag,
               meta=dict(n_det=n_det, n_iso=n_iso, n_mag=n_mag,
                         data=str(DATA), wall_s=time.time() - t0))
    (RESULTS / "apophis_mc.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(RESULTS / "apophis_mc_delays.npz",
                        **{f"delay_{k}": v for k, v in delays.items()})
    print(f"DONE in {time.time()-t0:.0f}s -> results/apophis_mc.json", flush=True)


if __name__ == "__main__":
    main()
