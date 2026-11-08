#!/usr/bin/env python3
"""
exp_regimes.py -- Cross-regime transfer study for the six-channel FDI core.

Three additional mission regimes (LEO formation keeping, GEO inspection,
cislunar NRHO-class station keep) are emulated at the residual interface:
each regime sets the estimation cadence, arc length, burn sparsity, and
sensor noise scales; the SAME FDI core with the SAME thresholds processes
whitened EKF-style innovations, attitude errors, per-burn dv errors, and
heartbeats. Faults are injected at the same normalized magnitudes as the
Apophis campaign (5-sigma NAV, 1-deg ATT, 5-sigma PROP, COMM dropout).

Also: sensor-quality sweep (noise x1..x10 with FIXED absolute fault size).

Usage: exp_regimes.py <REGIME|SENSOR> <n_trials>
"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import numpy.linalg as la
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fdi_core import FDICore, FDIConfig, FDIState, FaultType
from exp_apophis_mc import wilson_ci

REGIMES = {
    #             dt[s] T[s]      burn dt[s]  sig_r[m] sig_ang[deg]
    "LEO":  dict(dt=10.0, T=3*3600.0,  burn_every=1800.0, sig_r=1.0, sig_ang=0.01),
    "GEO":  dict(dt=60.0, T=24*3600.0, burn_every=21600.0, sig_r=5.0, sig_ang=0.02),
    "NRHO": dict(dt=60.0, T=48*3600.0, burn_every=43200.0, sig_r=10.0, sig_ang=0.02),
}
HB = 60.0

def cfg_default():
    c = FDIConfig(use_nav_channel=True, use_att_channel=True, use_comm_channel=True,
                  use_form_channel=False, use_peer_channel=False)
    c.nu_nav = 1.5; c.leak_nav = 0.05; c.h_nav = 15.0
    return c

def run_trial(args):
    regime, scn, seed, noise_scale, abs_fault = args
    R = REGIMES[regime]
    rng = np.random.default_rng(seed)
    dt = R["dt"]; n_steps = int(R["T"] / dt)
    tf = float(rng.uniform(0.1, 0.9)) * R["T"] if scn != "NONE" else np.inf
    sig = np.array([R["sig_r"], np.deg2rad(R["sig_ang"]), np.deg2rad(R["sig_ang"])]) * noise_scale
    burn_period = R["burn_every"]; next_burn = burn_period
    sigP = 0.02 * noise_scale                     # dv execution 1-sigma [m/s]
    fdi = FDICore(cfg_default()); fdi.reset(0.0)
    hb = HB; first = FaultType.NONE; t_det = None
    NAVm, ATTd, RATEd, PROPm = 5.0, 1.0, 0.05, 5.0
    for k in range(1, n_steps + 1):
        t = k * dt; on = t >= tf
        innov = rng.normal(0.0, 1.0, 3) * sig
        if scn == "NAV" and on:
            bias = (NAVm * sig) if not abs_fault else np.array([NAVm * R["sig_r"],
                     NAVm * np.deg2rad(R["sig_ang"]), NAVm * np.deg2rad(R["sig_ang"])])
            innov = innov + bias
        Sk = np.diag(np.maximum(sig, 1e-9) ** 2)
        burn = t >= next_burn; de = Pvv = None
        if burn:
            next_burn += burn_period
            Pvv = np.eye(3) * sigP ** 2
            de = rng.normal(0.0, sigP, 3)
            if scn == "PROP" and on:
                de = de + PROPm * sigP * rng.normal(size=3)
        sB = np.abs(rng.normal(0, np.tan(0.25 * np.deg2rad(0.01)), 3))
        oB = np.abs(rng.normal(0, np.deg2rad(0.001), 3))
        if scn == "ATT" and on:
            u = rng.normal(size=3); u /= la.norm(u)
            sB = sB + np.tan(0.25 * np.deg2rad(ATTd)) * u
            u = rng.normal(size=3); u /= la.norm(u)
            oB = oB + np.deg2rad(RATEd) * u
        hb = hb + dt if (scn == "COMM" and on) else HB
        s = fdi.step(t=t, innov=innov, S_innov=Sk, dv_est_H=de, P_vv=Pvv, burn_flag=bool(burn),
                     sigma_BR=sB, omega_BR_B=oB, hb_age=hb, hb_nominal=HB)
        if s.state.state in (FDIState.SUSPECT, FDIState.FAULTED, FDIState.DEAD):
            if first is FaultType.NONE and s.state.fault_type is not FaultType.NONE:
                first = s.state.fault_type
                if on and t_det is None and s.state.fault_type.name == scn:
                    t_det = t - tf
    return first.name, t_det

def campaign(regime, n, noise_scale=1.0, abs_fault=False, seed0=0):
    SC = ["NAV", "ATT", "PROP", "COMM"]
    res = {}
    for sc in SC + ["NONE"]:
        args = [(regime, sc, seed0 + 100000 + i * 7 + SC.index(sc) * 10000 if sc in SC else seed0 + 900000 + i, noise_scale, abs_fault) for i in range(n)]
        with Pool(2) as p:
            out = list(p.imap_unordered(run_trial, args, chunksize=8))
        if sc == "NONE":
            fa = sum(1 for l, _ in out if l != "NONE")
            res["Pfa"] = fa / n; res["Pfa_ci"] = list(wilson_ci(fa, n))
        else:
            nc = sum(1 for l, _ in out if l == sc)
            tds = [td for l, td in out if td is not None and l == sc]
            res[sc] = dict(Pd=nc / n, ci=list(wilson_ci(nc, n)),
                           lat_median=float(np.median(tds)) if tds else None,
                           lat_p95=float(np.percentile(tds, 95)) if tds else None, n=n)
        print(f"  {regime} x{noise_scale:g} {sc}: {json.dumps(res.get(sc, res.get('Pfa')))}", flush=True)
    return res

def main():
    what = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    t0 = time.time()
    if what == "SENSOR":
        out = {}
        for ns in (1.0, 2.0, 5.0, 10.0):
            out[f"x{ns:g}"] = campaign("LEO", n, noise_scale=ns, abs_fault=True, seed0=int(ns * 31))
        fn = Path("../results/regime_SENSOR.json")
    else:
        out = campaign(what, n)
        fn = Path(f"../results/regime_{what}.json")
    fn.write_text(json.dumps(dict(res=out, n=n, wall_s=time.time() - t0), indent=1))
    print(f"wrote {fn.name} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
