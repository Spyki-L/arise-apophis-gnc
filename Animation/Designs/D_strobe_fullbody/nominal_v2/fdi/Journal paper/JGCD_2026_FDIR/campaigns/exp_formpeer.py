#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_formpeer.py -- Causal FORM/PEER campaigns on the real fleet.

  [D] FORM_LOCAL  : one deputy's predicted terminal miss ramps away from the
                    fleet (guidance-side fault). Expect FORM & PEER both trip
                    on that deputy -> settled label FORM.
  [E] NAV_GLOBAL  : fleet near station (consensus scale floored), a common-mode
                    bias ramps every deputy's miss together. Expect FORM trips
                    fleet-wide with PEER quiet -> settled label NAV.
  [F] PEER stress : homogeneous-fleet false-alarm test, MAD floor on vs off,
                    with small random report jitter across trials.

All campaigns process events in GLOBAL TIME ORDER (strictly causal consensus),
restricted to the final 1.5 h before TCA (FORM window + margin).

Writes results/formpeer.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import numpy.linalg as la

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from fdi_core import FDICore, FDIConfig, FDIState, FaultType   # noqa: E402
from channels import causal_form_residual, PeerConsensusBuffer  # noqa: E402
import run_fdi_nominal as RFN                                   # noqa: E402

RESULTS = CODE.parent / "results"
DATA = Path("/sessions/cool-hopeful-franklin/mnt/Animation/Designs/D_strobe_fullbody/nominal_v2/data/mc/draw_0000/_nav_stage")

FORM_WINDOW_S = 3600.0
FORM_SCALE_M = 250.0
HB = 60.0
MIN_T_GO_S = 900.0


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))


def cfg_formpeer():
    c = FDIConfig(use_nav_channel=True, use_att_channel=True, use_comm_channel=False,
                  use_form_channel=True, use_peer_channel=True,
                  nu_form=3.0, leak_form=0.02, h_form=30.0,
                  nu_peer=3.0, leak_peer=0.05, h_peer=20.0)
    c.nu_nav = 1.5; c.leak_nav = 0.05; c.h_nav = 15.0
    return c


def precompute(fleet, t_tca):
    """Precompute each deputy's raw predicted terminal miss on its nav grid
    (expensive part), so trials only re-scale them."""
    t_lo = t_tca - FORM_WINDOW_S - 1800.0
    events = []
    miss = {}
    for d, g in enumerate(fleet):
        t = g["t"]
        sel = np.where((t >= t_lo) & (t < t_tca))[0]
        m = np.full(t.size, np.nan)
        for k in sel:
            tk = float(t[k])
            if tk < t_tca - FORM_WINDOW_S:
                continue
            m[k] = causal_form_residual(
                g["x"][k, 0:3], g["x"][k, 3:6], tk, t_tca,
                RFN._at(g["tf"], g["rC"], tk), RFN._at(g["tf"], g["vC"], tk),
                RFN._at(g["tf"], g["rC"], t_tca), RFN._at(g["tf"], g["vC"], t_tca),
                RFN._at(g["tf"], g["rA"], t_tca), RFN._at(g["tf"], g["vA"], t_tca),
                station_cmd=g["target"], scale_m=1.0, w_timetogo=1.0)
        miss[d] = m
        for k in sel:
            events.append((float(t[k]), d, int(k)))
    events.sort(key=lambda e: e[0])
    return events, miss


RAMP_M = 1200.0          # additive predicted-miss ramp magnitude at TCA [m]


def run_trial(fleet, events, miss, t_tca, scenario, rng, mad_floor=0.25):
    """One causal fleet run. Scenarios:
      FORM_LOCAL           additive miss ramp on ONE deputy (live consensus)
      NAV_GLOBAL_ABSORBED  same ramp on ALL deputies, live consensus
      NAV_GLOBAL_STARVED   same ramp on ALL deputies, ISL outage (no peer
                           reports arrive; scale falls back to the floor)
      PEER_STRESS          synthetic near-identical fleet reports (homogeneous
                           pathology); no fault injected
    """
    cfg = cfg_formpeer()
    n_dep = len(fleet)
    cores = {d: FDICore(cfg) for d in range(n_dep)}
    t_lo = events[0][0]
    for d in range(n_dep):
        cores[d].reset(t0=t_lo)
    latency = 1e9 if scenario == "NAV_GLOBAL_STARVED" else 0.0
    peer = PeerConsensusBuffer(n_dep=n_dep, latency_s=latency, mad_floor=mad_floor)

    focus = int(rng.integers(0, n_dep))
    if scenario in ("FORM_LOCAL", "NAV_GLOBAL_ABSORBED", "NAV_GLOBAL_STARVED"):
        t_fault = float(rng.uniform(t_tca - FORM_WINDOW_S, t_tca - MIN_T_GO_S))
    else:
        t_fault = np.inf
    eps = 1e-3 * rng.normal(size=n_dep)
    eps[focus] = 5e-3            # PEER_STRESS: one slightly-different healthy deputy

    first = {d: None for d in range(n_dep)}
    t_first = {d: None for d in range(n_dep)}
    form_trip = set(); peer_trip = set()
    peakS_peer = 0.0

    for t_k, d, k in events:
        g = fleet[d]
        innov = g["ypre"][k]
        if (not np.all(np.isfinite(innov))) or bool(g["rejected"][k]):
            ik = Sk = None
        else:
            ik = innov; sg = np.maximum(g["sig"][k], 1e-3); Sk = np.diag(sg ** 2)
        dvc = g["dv"][k]; burn = bool(la.norm(dvc) > 0.0); de = Pvv = None
        if burn and k > 0:
            Pvv = g["P"][k, 3:6, 3:6]
            de = (g["x"][k, 3:6] - g["x"][k - 1, 3:6]) - dvc
        rf = rp = None
        m = miss[d][k]
        if np.isfinite(m):
            if scenario == "PEER_STRESS":
                m = 1.0 * (1.0 + eps[d])          # near-identical fleet reports [m]
            frac = max(0.0, (t_k - t_fault)) / max(t_tca - t_fault, 1.0)
            if scenario == "FORM_LOCAL" and d == focus and t_k >= t_fault:
                m = m + RAMP_M * frac
            elif scenario in ("NAV_GLOBAL_ABSORBED", "NAV_GLOBAL_STARVED") and t_k >= t_fault:
                m = m + RAMP_M * frac
            ttg = max(t_tca - t_k, 0.0)
            w = 1.0 + 1.5 * np.clip(1.0 - ttg / FORM_WINDOW_S, 0.0, 1.0)
            peer.push(d, t_k, m)
            scale = peer.fleet_scale(t_k, floor=FORM_SCALE_M)
            rf = (m / scale) * w
            rp = peer.r_peer(d, t_k, own_value=m, scale=scale)
        snap = cores[d].step(t=t_k, innov=ik, S_innov=Sk, dv_est_H=de, P_vv=Pvv,
                             burn_flag=burn, sigma_BR=g["sBR"][k], omega_BR_B=g["oBR"][k],
                             hb_age=HB, hb_nominal=HB, r_form=rf, r_peer=rp)
        st = snap.state
        if cores[d].S_form > cfg.h_form:
            form_trip.add(d)
        if cores[d].S_peer > cfg.h_peer:
            peer_trip.add(d)
        peakS_peer = max(peakS_peer, cores[d].S_peer)
        if st.state in (FDIState.SUSPECT, FDIState.FAULTED, FDIState.DEAD):
            if first[d] is None and st.fault_type is not FaultType.NONE:
                first[d] = st.fault_type.name
                t_first[d] = t_k - t_fault if np.isfinite(t_fault) else None

    settled = {d: cores[d].state.fault_type.name for d in range(n_dep)}
    states = {d: cores[d].state.state.name for d in range(n_dep)}
    return dict(focus=focus, first=first, settled=settled, states=states,
                t_first=t_first, n_form_trip=len(form_trip),
                n_peer_trip=len(peer_trip), peakS_peer=peakS_peer)


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    t0 = time.time()
    fleet, t_tca = RFN.load_fleet(DATA)
    events, miss = precompute(fleet, t_tca)
    print(f"fleet + misses precomputed: {len(events)} events in window ({time.time()-t0:.0f}s)", flush=True)

    out = {}

    # [D] FORM_LOCAL
    rng = np.random.default_rng(7)
    ok = wrong = missed = 0; early_peer = 0; delays = []
    for i in range(n_trials):
        r = run_trial(fleet, events, miss, t_tca, "FORM_LOCAL", rng)
        lab = r["settled"][r["focus"]]
        if lab == "FORM":
            ok += 1
            if r["t_first"][r["focus"]] is not None:
                delays.append(r["t_first"][r["focus"]])
        elif lab == "NONE":
            missed += 1
        else:
            wrong += 1
        if r["first"][r["focus"]] == "PEER":
            early_peer += 1
    lo, hi = wilson_ci(ok, n_trials)
    out["FORM_LOCAL"] = dict(n=n_trials, correct=ok, wrong=wrong, missed=missed,
                             ci=[lo, hi], early_peer_first=early_peer,
                             delay_median=float(np.median(delays)) if delays else None)
    print(f"[D] FORM_LOCAL: correct={ok}/{n_trials} [{lo:.3f},{hi:.3f}] "
          f"wrong={wrong} miss={missed} early-PEER-first={early_peer}", flush=True)

    # [E1] NAV_GLOBAL absorbed (live consensus cancels the common mode)
    rng = np.random.default_rng(8)
    n_quiet = 0; n_any_alarm = 0
    for i in range(n_trials):
        r = run_trial(fleet, events, miss, t_tca, "NAV_GLOBAL_ABSORBED", rng)
        healthy = all(r["states"][d] == "HEALTHY" for d in r["states"])
        n_quiet += int(healthy)
        n_any_alarm += int(not healthy)
    lo, hi = wilson_ci(n_quiet, n_trials)
    out["NAV_GLOBAL_ABSORBED"] = dict(n=n_trials, fleet_stays_healthy=n_quiet,
                                      false_alarms=n_any_alarm, ci=[lo, hi])
    print(f"[E1] NAV_GLOBAL absorbed: healthy={n_quiet}/{n_trials} [{lo:.3f},{hi:.3f}]", flush=True)

    # [E2] NAV_GLOBAL starved (ISL outage -> floor scale; FORM fleet-wide, PEER quiet -> NAV)
    rng = np.random.default_rng(88)
    ok = wrong = missed = 0; fleetwide = 0
    for i in range(n_trials):
        r = run_trial(fleet, events, miss, t_tca, "NAV_GLOBAL_STARVED", rng)
        lab = r["settled"][r["focus"]]
        if lab == "NAV":
            ok += 1
        elif lab == "NONE":
            missed += 1
        else:
            wrong += 1
        if r["n_form_trip"] >= 2:
            fleetwide += 1
    lo, hi = wilson_ci(ok, n_trials)
    out["NAV_GLOBAL_STARVED"] = dict(n=n_trials, correct=ok, wrong=wrong, missed=missed,
                                     ci=[lo, hi], n_form_fleetwide=fleetwide)
    print(f"[E2] NAV_GLOBAL starved: correct={ok}/{n_trials} [{lo:.3f},{hi:.3f}] "
          f"wrong={wrong} miss={missed} FORM-fleetwide={fleetwide}", flush=True)

    # [F] PEER stress: near-ident