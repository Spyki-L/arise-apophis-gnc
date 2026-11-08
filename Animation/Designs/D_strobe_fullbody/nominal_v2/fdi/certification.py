#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
certification.py — F10 journal-grade certification campaigns. Pure numpy.

  wilson_ci                 binomial confidence intervals
  byzantine_campaign        f-of-N Byzantine tolerance via the PEER-CBF consistency residual
  recovery_campaign         recovery efficacy (coverage recovered + beat-tau*) via reoptimizer + F3
  recoverability_grid       F3 recoverability frontier over (authority x onset)
  load_fleet/detection_campaign  Pd / Pfa / detection-latency / isolation confusion on the REAL fleet

These produce the evidence tables for the certification dossier and the journal paper.
"""
from __future__ import annotations
import numpy as np, numpy.linalg as la


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    hw = (z/d) * np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (max(0.0, c-hw), min(1.0, c+hw))


# ── Byzantine f-of-N tolerance (PEER-CBF consistency) ────────────────────────

def byzantine_campaign(N=8, f_grid=(0, 1, 2, 3, 4), n_trials=500, d_safe=50.,
                       sigma_range=0.1, lie_m=60., thr=8., seed=0):
    """Sweep the number of Byzantine liars f. A liar reports a wrong position; honest deputies
    report truth. The robust (median) consistency residual flags liars and spares honest nodes
    while f < N/2. Returns per-f detection rate (liars caught) and false-alarm rate (honest flagged)."""
    from peer_cbf import consistency_residual
    rng = np.random.default_rng(seed); out = {}
    for f in f_grid:
        det = fa = n_liar = n_honest = 0
        for _ in range(n_trials):
            ang = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.normal(0, 0.05, N)
            pos = np.stack([1500*np.cos(ang), 1500*np.sin(ang), 200*np.sin(2*ang)], 1)
            rep = pos + rng.normal(0, 0.05, pos.shape)
            liars = rng.choice(N, size=f, replace=False)
            for L in liars:
                u = rng.normal(0, 1, 3); rep[L] = pos[L] + u/(la.norm(u)+1e-9)*lie_m
            mr = np.full((N, N), np.nan)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        mr[i, j] = la.norm(pos[i]-pos[j]) + rng.normal(0, sigma_range)
            res = consistency_residual(rep, mr, sigma_range)
            for d in range(N):
                if d in liars:
                    n_liar += 1; det += int(res[d] > thr)
                else:
                    n_honest += 1; fa += int(res[d] > thr)
        out[f] = dict(det=det, n_liar=n_liar, fa=fa, n_honest=n_honest,
                      Pd=(det/n_liar if n_liar else None), Pfa=fa/max(n_honest, 1),
                      Pd_ci=wilson_ci(det, n_liar) if n_liar else None)
    return out


# ── recovery efficacy (reoptimizer + F3 deadline) ────────────────────────────

def recovery_campaign(n_trials=120, n_iter=120, seed=0):
    """Random single/double deputy losses at random onset; measure coverage recovered and the
    beat-tau* (reachable-in-time) fraction."""
    from reoptimizer import fibonacci_sphere, coverage_surrogate, optimize_stations
    from reachability import cw_blocks_full
    rng = np.random.default_rng(seed)
    pts, norms = fibonacci_sphere(160, 200.)
    M = 8; ang = np.linspace(0, 2*np.pi, M, endpoint=False); shell = 1400.
    geom = np.array([[shell*np.cos(a), shell*np.sin(a), 500*np.sin(2*a)] for a in ang])
    reach = np.full(M, 700.); kw = dict(R_max=4000., th_min=8., th_max=45.)
    st0, _, _ = optimize_stations(geom, list(range(M)), pts, norms, reach, n_iter=300, seed=1, **kw)
    nominal = geom.copy()
    for d in range(M): nominal[d] = st0[d]
    cov_nom = coverage_surrogate(nominal, pts, norms, **kw)
    recs = []; beat = 0
    def reach_rad(t_go, dv=1.0, n=1e-4):
        _, Prv, _, _ = cw_blocks_full(n, max(t_go, 1.0)); return la.norm(Prv)*dv
    for _ in range(n_trials):
        k = rng.integers(1, 3)
        lost = list(rng.choice(M, size=k, replace=False))
        active = [d for d in range(M) if d not in lost]
        cov_lost = coverage_surrogate(nominal[active], pts, norms, **kw)
        st, cov_opt, _ = optimize_stations(nominal, active, pts, norms, reach, n_iter=n_iter,
                                           seed=int(rng.integers(1e6)), **kw)
        gap = cov_nom - cov_lost
        recs.append(np.clip((cov_opt - cov_lost)/gap if gap > 1e-6 else 1.0, 0, 1.5))
        t_go = rng.uniform(300, 6000)
        max_move = max(la.norm(st[i]-nominal[active[i]]) for i in range(len(active)))
        beat += int(max_move <= reach_rad(t_go))
    recs = np.array(recs)
    return dict(cov_nom=cov_nom, mean_recovered=float(recs.mean()), p10=float(np.percentile(recs, 10)),
                beat_tau_frac=beat/n_trials, n=n_trials)


# ── recoverability grid (F3 magnitude × onset) ───────────────────────────────

def recoverability_grid(g_grid=None, tgo_grid=None):
    from reachability import FormTube, propagate_pos
    g_grid = np.linspace(0.05, 1.0, 20) if g_grid is None else g_grid
    tgo_grid = np.linspace(300., 9000., 24) if tgo_grid is None else tgo_grid
    n = 1e-4; rho = np.array([1000., 2000., -300.]); rhod = np.zeros(3)
    P0 = np.diag([1, 1, 1, 1e-6, 1e-6, 1e-6]); dv = np.array([0.1, 0.1, 0.1])
    base = FormTube(n=n, rho=rho, rhod=rhod, P0=P0, dv_axis=dv, t_go=3600., slot=None)
    c = propagate_pos(rho, rhod, n, 3600.); base.slot = c + np.array([200., 0, 0])
    Z = np.zeros((len(g_grid), len(tgo_grid)))
    for ig, g in enumerate(g_grid):
        for it, tgo in enumerate(tgo_grid):
            Z[ig, it] = base.with_authority(dv*g).with_t_go(tgo).is_reachable()
    return g_grid, tgo_grid, Z


# ── detection / isolation / latency on the real fleet ────────────────────────

def load_fleet(data_dir, N=8):
    """Load the 8-deputy rel-nav + attitude bundles, attitude pre-interpolated onto the nav grid."""
    import os
    t0 = min(float(np.load(os.path.join(data_dir, f"apophis_rendezvous_full_D{d}.npz"))["t"][0]) for d in range(N))
    F = []
    for d in range(N):
        nav = np.load(os.path.join(data_dir, f"apophis_swarm_FDI_rgazel_D{d}.npz"))
        att = np.load(os.path.join(data_dir, f"dep{d:02d}_logs.npz"))
        t = nav["t"].astype(float); te = att["t_err"].astype(float)
        tq = np.clip(t - t0, te[0], te[-1] - 60.)
        def iv(arr):
            o = np.empty((t.size, 3))
            for j in range(3):
                o[:, j] = np.interp(tq, te, arr[:, j])
            return o
        F.append(dict(t=t, x=nav["x_est"].astype(float), P=nav["P"].astype(float),
                      yp=nav["y_pre"].astype(float), sg=nav["sigma_meas"].astype(float),
                      dv=nav["dv_seq_H"].astype(float),
                      sB=iv(att["sigma_BR"].astype(float)), oB=iv(att["omega_BR"].astype(float))))
    return F


def _fdi_cfg():
    from fdi_core import FDIConfig
    c = FDIConfig(use_nav_channel=True, use_att_channel=True, use_comm_channel=True,
                  use_form_channel=False, use_peer_channel=False)
    c.nu_nav = 1.5; c.leak_nav = 0.05; c.h_nav = 15.0
    return c


def _run_dep(g, scn, tf, rng, NAV=5., ATT=1., RATE=0.05, PROP=5., HB=60.):
    from fdi_core import FDICore, FDIState, FaultType
    t = g["t"]; fdi = FDICore(_fdi_cfg()); fdi.reset(float(t[0])); hb = HB
    first_label = FaultType.NONE; t_detect = None
    for k in range(t.size):
        tk = float(t[k]); on = tk >= tf
        innov = g["yp"][k].copy()
        if not np.all(np.isfinite(innov)):
            ik = Sk = None
        else:
            if scn == "NAV" and on:
                innov = innov + NAV*g["sg"][k]
            ik = innov; sg = np.maximum(g["sg"][k], 1e-3); Sk = np.diag(sg**2)
        dvc = g["dv"][k]; burn = la.norm(dvc) > 0.; de = Pvv = None
        if burn and k > 0:
            Pvv = g["P"][k, 3:6, 3:6]; de = (g["x"][k, 3:6]-g["x"][k-1, 3:6]) - dvc
            if scn == "PROP" and on:
                sv = np.sqrt(np.maximum(np.diag(Pvv), 1e-6)); de = de + PROP*sv*rng.normal(size=3)
        sB = g["sB"][k]; oB = g["oB"][k]
        if scn == "ATT" and on:
            u = rng.normal(size=3); u /= la.norm(u); sB = sB + np.deg2rad(ATT)*u
            u = rng.normal(size=3); u /= la.norm(u); oB = oB + np.deg2rad(RATE)*u
        dt = 0. if k == 0 else float(t[k]-t[k-1])
        hb = hb + dt if (scn == "COMM" and on) else HB
        s = fdi.step(t=tk, innov=ik, S_innov=Sk, dv_est_H=de, P_vv=Pvv, burn_flag=burn,
                     sigma_BR=sB, omega_BR_B=oB, hb_age=hb, hb_nominal=HB)
        if s.state.state in (FDIState.SUSPECT, FDIState.FAULTED, FDIState.DEAD):
            if first_label is FaultType.NONE and s.state.fault_type is not FaultType.NONE:
                first_label = s.state.fault_type
                if on and t_detect is None and s.state.fault_type.name == scn:
                    t_detect = tk - tf
    return first_label.name, t_detect


def detection_campaign(fleet, n_trials=120, seed=0, N=8):
    rng = np.random.default_rng(seed)
    SC = ["NAV", "ATT", "PROP", "COMM"]; LAB = ["NONE", "NAV", "ATT", "PROP", "COMM"]
    res = {}; conf = {s: {l: 0 for l in LAB} for s in SC}
    for sc in SC:
        nc = 0; lat = []
        for _ in range(n_trials):
            g = fleet[int(rng.integers(0, N))]
            ph = float(rng.uniform(0.1, 0.9)); tf = float(g["t"][0]) + ph*(float(g["t"][-1])-float(g["t"][0]))
            lab, td = _run_dep(g, sc, tf, rng)
            conf[sc][lab] += 1
            if lab == sc:
                nc += 1
                if td is not None:
                    lat.append(td)
        lo, hi = wilson_ci(nc, n_trials); lat = np.array(lat) if lat else np.array([np.nan])
        res[sc] = dict(Pd=nc/n_trials, Pd_ci=(lo, hi), n=n_trials,
                       lat_median=float(np.nanmedian(lat)), lat_p95=float(np.nanpercentile(lat, 95)))
    fa = 0; ntot = 0
    for _ in range(n_trials):
        g = fleet[int(rng.integers(0, N))]
        lab, _ = _run_dep(g, "NONE", 1e18, rng)
        ntot += 1; fa += int(lab != "NONE")
    return dict(per_scenario=res, confusion=conf, Pfa=fa/ntot, Pfa_ci=wilson_ci(fa, ntot), Pfa_n=ntot)
