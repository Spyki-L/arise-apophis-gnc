#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
isolation_before_after.py — real-fleet PROP/NAV/ATT/COMM isolation confusion
matrix, run with the LEGACY isolation rule and the P1 FIXED core, side by side.

Proves the P1 PROP fix on real data: legacy labels injected PROP faults NONE
(~0% isolation, reproducing fdi_mc_isolation.tex); the fixed core isolates PROP.

Usage:
    python isolation_before_after.py --data <dir with deputy bundles> [--n 300]

Validated result (n=120): PROP isolation 0.0% (legacy) -> 99.2% (fixed),
NAV/ATT/COMM 100% in both.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import numpy.linalg as la

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # import fdi_core from fdi/
from fdi_core import FDICore, FDIConfig, FDIState, FaultType

N_DEP = 8
NAV_BIAS = 5.0; ATT_DEG = 1.0; ATT_RATE = 0.05; PROP_SIG = 5.0; HB = 60.0
LABELS = ["NONE", "NAV", "ATT", "PROP", "COMM"]
SCEN = ["NAV", "ATT", "PROP", "COMM"]


def _interp_vec(tg, arr, tq):
    out = np.empty((np.size(tq), arr.shape[1]))
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(tq, tg, arr[:, j])
    return out


def load_fleet(root: Path):
    t0 = min(float(np.load(root / f"apophis_rendezvous_full_D{d}.npz")["t"][0]) for d in range(N_DEP))
    F = []
    for d in range(N_DEP):
        nav = np.load(root / f"apophis_swarm_FDI_rgazel_D{d}.npz")
        att = np.load(root / f"dep{d:02d}_logs.npz")
        t = nav["t"].astype(float); te = att["t_err"].astype(float)
        tq = np.clip(t - t0, te[0], te[-1] - 60.0)
        F.append(dict(t=t, x=nav["x_est"].astype(float), P=nav["P"].astype(float),
                      yp=nav["y_pre"].astype(float), sg=nav["sigma_meas"].astype(float),
                      dv=nav["dv_seq_H"].astype(float),
                      sB=_interp_vec(te, att["sigma_BR"].astype(float), tq),
                      oB=_interp_vec(te, att["omega_BR"].astype(float), tq)))
    return F


def cfg(legacy):
    c = FDIConfig(use_nav_channel=True, use_att_channel=True, use_comm_channel=True,
                  use_form_channel=False, use_peer_channel=False, legacy_isolation=legacy)
    c.nu_nav = 1.5; c.leak_nav = 0.05; c.h_nav = 15.0
    return c


def run_dep(g, scn, tf, legacy, rng):
    t = g["t"]; fdi = FDICore(cfg(legacy)); fdi.reset(float(t[0])); hb = HB; final = FaultType.NONE
    for k in range(t.size):
        tk = float(t[k]); on = tk >= tf
        innov = g["yp"][k].copy()
        if not np.all(np.isfinite(innov)):
            ik = Sk = None
        else:
            if scn == "NAV" and on:
                innov = innov + NAV_BIAS * g["sg"][k]
            ik = innov; sg = np.maximum(g["sg"][k], 1e-3); Sk = np.diag(sg ** 2)
        dvc = g["dv"][k]; burn = la.norm(dvc) > 0.0; de = Pvv = None
        if burn and k > 0:
            Pvv = g["P"][k, 3:6, 3:6]; de = (g["x"][k, 3:6] - g["x"][k - 1, 3:6]) - dvc
            if scn == "PROP" and on:
                sv = np.sqrt(np.maximum(np.diag(Pvv), 1e-6)); de = de + PROP_SIG * sv * rng.normal(size=3)
        sB = g["sB"][k]; oB = g["oB"][k]
        if scn == "ATT" and on:
            u = rng.normal(size=3); u /= la.norm(u); sB = sB + np.deg2rad(ATT_DEG) * u
            u = rng.normal(size=3); u /= la.norm(u); oB = oB + np.deg2rad(ATT_RATE) * u
        dt = 0.0 if k == 0 else float(t[k] - t[k - 1])
        hb = hb + dt if (scn == "COMM" and on) else HB
        s = fdi.step(t=tk, innov=ik, S_innov=Sk, dv_est_H=de, P_vv=Pvv, burn_flag=burn,
                     sigma_BR=sB, omega_BR_B=oB, hb_age=hb, hb_nominal=HB)
        if s.state.state in (FDIState.SUSPECT, FDIState.FAULTED, FDIState.DEAD):
            final = s.state.fault_type
    return final


def confusion(F, legacy, ntr, rng):
    C = {s: {l: 0 for l in LABELS} for s in SCEN}
    for sc in SCEN:
        for _ in range(ntr):
            g = F[int(rng.integers(0, N_DEP))]
            ph = float(rng.uniform(0.1, 0.9))
            tf = float(g["t"][0]) + ph * (float(g["t"][-1]) - float(g["t"][0]))
            C[sc][run_dep(g, sc, tf, legacy, rng).name] += 1
    return C


def show(name, C, ntr):
    print(f"\n{name}  (rows=injected truth, cols=identified label; N={ntr}/scenario)")
    print("  truth |" + "".join(f"{l:>7}" for l in LABELS) + "   | diag%")
    for sc in SCEN:
        r = C[sc]; tot = sum(r.values())
        print(f"  {sc:>5} |" + "".join(f"{r[l]:>7}" for l in LABELS) + f"   | {100*r[sc]/tot:5.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with apophis_swarm_FDI_rgazel_D*.npz etc.")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    F = load_fleet(Path(a.data)); rng = np.random.default_rng(a.seed); t1 = time.time()
    Cl = confusion(F, True, a.n, rng); show("BEFORE (legacy isolation rule)", Cl, a.n)
    Cf = confusion(F, False, a.n, rng); show("AFTER  (P1 fixed core)", Cf, a.n)
    print(f"\n  PROP isolation: legacy {100*Cl['PROP']['PROP']/a.n:.1f}%  ->  fixed {100*Cf['PROP']['PROP']/a.n:.1f}%")
    print(f"  elapsed {time.time()-t1:.1f}s")
