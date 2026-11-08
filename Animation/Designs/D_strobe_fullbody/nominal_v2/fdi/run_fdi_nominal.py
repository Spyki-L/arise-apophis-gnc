#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_fdi_nominal.py — causal, onboard-style FDI driver (P0).

The offline ``apophis_swarm_fdi_main.py`` post-processes saved bundles using
GLOBAL, future-aware information (truth min-range TCA, a whole-timeline fleet
median for PEER, truth FORM geometry).  This driver runs the SAME FDI core but
**causally**, the way it would run onboard:

  * TCA is taken from the CHIEF↔Apophis closest approach (onboard ephemeris),
    not the deputy truth min-range (``channels.predicted_tca``).
  * FORM is the deputy's predicted terminal miss from its OWN estimate to its
    COMMANDED station (``channels.causal_form_residual``).
  * PEER is a streaming, latency-aware, MAD-floored consensus over peer reports
    received so far (``channels.PeerConsensusBuffer``) — events are processed in
    GLOBAL TIME ORDER so no deputy ever sees a peer's future.

On nominal (fault-free) data every deputy should end HEALTHY — the causal
reproduction of the offline healthy-fleet result.  Inject a fault (or feed a
faulted bundle) and the same core detects/isolates it in real time.

Usage:  python run_fdi_nominal.py [--data DIR] [--no-formpeer]
Writes: fdi/health_history.npz  (+ console summary)
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import numpy.linalg as la

from fdi_core import FDICore, FDIConfig, FDIState, FaultType
from channels import (predicted_tca, causal_form_residual, PeerConsensusBuffer,
                      dcm_lvlh)

N_DEP = 8
NAV_PREFIX = "apophis_swarm_FDI_rgazel_D"
FORM_WINDOW_S = 3600.0     # FORM active only in the last hour before TCA
FORM_SCALE_M = 250.0       # causal requirement scale (floor), NOT a future fleet median
HB_NOMINAL = 60.0
ATT_TERMINAL_GUARD_S = 60.0

# ── MC_v2 attitude adapter constants ─────────────────────────────────────────
# guidance_v2's follower saturates at the wheel-limited slew rate near CA, so the raw
# pointing error legitimately grows toward pi there (slew lag).  The FDI therefore
# monitors the DEVIATION from that expected, onboard-computable envelope, never the raw
# lag.  OMEGA_MAX: XACT-50 rated 10 deg/s slew binds below the 12.6 deg/s wheel-momentum
# cap (see attitude_kinematic).  Floor: healthy pointing envelope (ADCS accuracy+jitter).
OMEGA_MAX_RADPS = float(np.deg2rad(float(os.getenv("ARISE_FDI_OMEGA_MAX_DEGS", "10.0"))))
OMEGA_MARGIN = 1.03
ATT_ENV_FLOOR_RAD = float(np.deg2rad(float(os.getenv("ARISE_FDI_ATT_FLOOR_ARCSEC", "30.0")) / 3600.0))


def _interp_vec(tg, arr, tq):
    out = np.empty((np.size(tq), arr.shape[1]))
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(tq, tg, arr[:, j])
    return out


def _expected_att_envelope(tf, C_RN):
    """Onboard-computable healthy pointing envelope theta_env(t) [rad].

    Reproduces guidance_v2.attitude_kinematic's slew-lag recursion from the SAME commanded
    reference C_RN(t) the deputy carries: the body rate is wheel-limited at OMEGA_MAX_RADPS,
    so while |w_ref| exceeds the cap a lag = integral(w_ref - w_max) dt accumulates (and
    decays as the wheels catch up).  Healthy pointing error stays inside
    sqrt(floor^2 + lag^2); the ATT channel scores only the EXCESS beyond this envelope."""
    tf = np.asarray(tf, float); C = np.asarray(C_RN, float); N = tf.size
    w_ref = np.zeros(N)
    if N >= 3:
        dt2 = (tf[2:] - tf[:-2])
        ok = dt2 > 0
        Cd = np.zeros((N - 2, 3, 3))
        Cd[ok] = (C[2:][ok] - C[:-2][ok]) / dt2[ok, None, None]
        Wsk = -np.einsum("nij,nkj->nik", Cd, C[1:-1])
        w_ref[1:-1] = 0.5 * np.sqrt((Wsk[:, 2, 1] - Wsk[:, 1, 2]) ** 2 +
                                    (Wsk[:, 0, 2] - Wsk[:, 2, 0]) ** 2 +
                                    (Wsk[:, 1, 0] - Wsk[:, 0, 1]) ** 2)
        w_ref[0] = w_ref[1]; w_ref[-1] = w_ref[-2]
    lag = np.zeros(N); acc = 0.0
    for j in range(1, N):
        dt = tf[j] - tf[j - 1]
        if w_ref[j] > OMEGA_MAX_RADPS:
            acc += (w_ref[j] - OMEGA_MAX_RADPS) * dt
        else:
            acc = max(0.0, acc - (OMEGA_MAX_RADPS - w_ref[j]) * dt)
        lag[j] = acc
    return np.minimum(np.pi, np.sqrt(ATT_ENV_FLOOR_RAD ** 2 + lag ** 2))


def load_att_mc(att_path, tf, C_RN, t_nav):
    """MC_v2 attitude bundle (att_D*.npz) -> (sigma_BR, omega_BR) channel inputs, as
    ENVELOPE DEVIATIONS: zero for a healthy deputy (including through the CA slew lag);
    a tumble / wheel fault shows up as pointing beyond the lag envelope and/or body rate
    beyond wheel capability.  |sigma_BR| = tan(dtheta/4) maps excess angle to MRP norm."""
    att = np.load(att_path)
    ta = att["t"].astype(float)
    th = att["pointing_err_rad"].astype(float)
    wb = la.norm(att["omega_BN_B"].astype(float), axis=1)
    th_env = np.interp(ta, np.asarray(tf, float), _expected_att_envelope(tf, C_RN))
    d_th = np.maximum(0.0, th - th_env)
    d_w = np.maximum(0.0, wb - OMEGA_MAX_RADPS * OMEGA_MARGIN)
    tq = np.clip(t_nav, ta[0], ta[-1] - ATT_TERMINAL_GUARD_S)
    sBR = np.zeros((t_nav.size, 3)); sBR[:, 0] = np.tan(0.25 * np.interp(tq, ta, d_th))
    oBR = np.zeros((t_nav.size, 3)); oBR[:, 0] = np.interp(tq, ta, d_w)
    return sBR, oBR


def load_fleet(root: Path):
    fleet = []
    t0_abs = min(float(np.load(root / f"apophis_rendezvous_full_D{d}.npz")["t"][0]) for d in range(N_DEP))
    for d in range(N_DEP):
        nav = np.load(root / f"{NAV_PREFIX}{d}.npz")
        full = np.load(root / f"apophis_rendezvous_full_D{d}.npz", allow_pickle=True)
        t = nav["t"].astype(float)
        att_path = root / f"dep{d:02d}_logs.npz"          # legacy GNC-run logs
        mc_att_path = root.parent / "attitude" / f"att_D{d}.npz"   # MC_v2 draw layout
        has_att = att_path.exists() or mc_att_path.exists()
        if att_path.exists():
            att = np.load(att_path)
            te = att["t_err"].astype(float)
            tq = np.clip(t - t0_abs, te[0], te[-1] - ATT_TERMINAL_GUARD_S)
            sBR = _interp_vec(te, att["sigma_BR"].astype(float), tq)
            oBR = _interp_vec(te, att["omega_BR"].astype(float), tq)
        elif mc_att_path.exists():               # MC_v2 attitude bundle (envelope deviation)
            sBR, oBR = load_att_mc(mc_att_path, full["t"].astype(float), full["C_RN"].astype(float), t)
        else:                                    # attitude logs absent -> ATT channel skipped
            sBR = np.zeros((t.size, 3)); oBR = np.zeros((t.size, 3))
        y_pre = nav["y_pre"].astype(float); y_post = nav["y_post"].astype(float)
        finite = np.isfinite(y_pre).all(axis=1)
        rejected = finite & np.all(y_pre == y_post, axis=1)     # EKF-gated outlier (skip in NAV)
        fleet.append(dict(
            d=d, t=t, x=nav["x_est"].astype(float), P=nav["P"].astype(float),
            ypre=y_pre, sig=nav["sigma_meas"].astype(float), dv=nav["dv_seq_H"].astype(float),
            rejected=rejected,
            sBR=sBR, oBR=oBR, has_att=has_att,
            tf=full["t"].astype(float), rC=full["rC_N"].astype(float), vC=full["vC_N"].astype(float),
            rA=full["rA_N"].astype(float), vA=full["vA_N"].astype(float),
            target=np.asarray(full["APOPHIS_TARGET"], float).reshape(3),
        ))
    # causal TCA from CHIEF↔Apophis closest approach (onboard ephemeris), per deputy's chief arc
    sep = la.norm(fleet[0]["rC"] - fleet[0]["rA"], axis=1)
    tca = float(fleet[0]["tf"][int(np.argmin(sep))])
    return fleet, predicted_tca(tca, t_now=float(fleet[0]["t"][0]))


def _at(tf, arr, tq):
    return _interp_vec(tf, arr, np.array([tq]))[0]


def run(root: Path, use_formpeer=True):
    fleet, t_tca = load_fleet(root)
    cfg = FDIConfig(use_nav_channel=True, use_att_channel=True, use_comm_channel=False,
                    use_form_channel=use_formpeer, use_peer_channel=use_formpeer,
                    nu_form=3.0, leak_form=0.02, h_form=30.0,
                    nu_peer=3.0, leak_peer=0.05, h_peer=20.0)
    cfg.nu_nav = 1.5; cfg.leak_nav = 0.05; cfg.h_nav = 15.0
    if not all(g.get("has_att", True) for g in fleet):
        cfg.use_att_channel = False
        print("  [note] attitude logs (dep*_logs.npz / attitude/att_D*.npz) absent -> ATT channel disabled; running NAV/PROP/FORM/PEER")

    cores = {d: FDICore(cfg) for d in range(N_DEP)}
    for d in range(N_DEP):
        cores[d].reset(t0=float(fleet[d]["t"][0]))
    peer = PeerConsensusBuffer(n_dep=N_DEP, latency_s=0.0, mad_floor=0.25)
    hist = {d: [] for d in range(N_DEP)}

    # GLOBAL TIME ORDER event list → strictly causal peer consensus
    events = sorted(((float(fleet[d]["t"][k]), d, k)
                     for d in range(N_DEP) for k in range(fleet[d]["t"].size)),
                    key=lambda e: e[0])

    for t_k, d, k in events:
        g = fleet[d]
        # NAV (outlier-aware skip)
        innov = g["ypre"][k]
        if (not np.all(np.isfinite(innov))) or bool(g["rejected"][k]):
            ik = Sk = None
        else:
            ik = innov; sg = np.maximum(g["sig"][k], 1e-3); Sk = np.diag(sg ** 2)
        # PROP
        dvc = g["dv"][k]; burn = bool(la.norm(dvc) > 0.0); de = Pvv = None
        if burn and k > 0:
            Pvv = g["P"][k, 3:6, 3:6]; de = (g["x"][k, 3:6] - g["x"][k - 1, 3:6]) - dvc
        # FORM/PEER (causal)
        rf = rp = None
        if use_formpeer and (t_tca - FORM_WINDOW_S) <= t_k < t_tca:
            ttg = max(t_tca - t_k, 0.0)
            w = 1.0 + 1.5 * np.clip(1.0 - ttg / FORM_WINDOW_S, 0.0, 1.0)
            miss = causal_form_residual(
                g["x"][k, 0:3], g["x"][k, 3:6], t_k, t_tca,
                _at(g["tf"], g["rC"], t_k), _at(g["tf"], g["vC"], t_k),
                _at(g["tf"], g["rC"], t_tca), _at(g["tf"], g["vC"], t_tca),
                _at(g["tf"], g["rA"], t_tca), _at(g["tf"], g["vA"], t_tca),
                station_cmd=g["target"], scale_m=1.0, w_timetogo=1.0)   # RAW predicted miss [m]
            peer.push(d, t_k, miss)
            scale = peer.fleet_scale(t_k, floor=FORM_SCALE_M)           # causal consensus scale
            rf = (miss / scale) * w
            rp = peer.r_peer(d, t_k, own_value=miss, scale=scale)
        # ATT
        sB = g["sBR"][k]; oB = g["oBR"][k]
        snap = cores[d].step(t=t_k, innov=ik, S_innov=Sk, dv_est_H=de, P_vv=Pvv,
                             burn_flag=burn, sigma_BR=sB, omega_BR_B=oB,
                             hb_age=HB_NOMINAL, hb_nominal=HB_NOMINAL, r_form=rf, r_peer=rp)
        hist[d].append((t_k, snap.state.state.value, snap.state.fault_type.value, snap.state.confidence))

    print(f"causal FDI run · TCA(onboard)={t_tca:.1f}s · FORM/PEER={'on' if use_formpeer else 'off'}")
    n_healthy = 0
    for d in range(N_DEP):
        st = cores[d].state
        peakS = max(cores[d].S_nav, cores[d].S_att, cores[d].S_form, cores[d].S_peer)
        flag = "OK " if st.state is FDIState.HEALTHY else ">>>"
        print(f"  {flag} D{d}: state={st.state.name:8s} fault={st.fault_type.name:5s} peakS={peakS:6.2f}")
        n_healthy += int(st.state is FDIState.HEALTHY)
    print(f"  {n_healthy}/{N_DEP} deputies HEALTHY")
    out = Path(__file__).resolve().parent / "health_history.npz"
    np.savez_compressed(out, **{f"D{d}": np.asarray(hist[d], float) for d in range(N_DEP)})
    print(f"  wrote {out.name}")
    return cores, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="causal onboard-style FDI driver (P0)")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[4] / "data" / "_nav_stage"),
                    help="dir with apophis_swarm_FDI_rgazel_D*.npz (+ apophis_rendezvous_full_D*.npz)")
    ap.add_argument("--no-formpeer", action="store_true")
    a = ap.parse_args()
    run(Path(a.data), use_formpeer=not a.no_formpeer)
