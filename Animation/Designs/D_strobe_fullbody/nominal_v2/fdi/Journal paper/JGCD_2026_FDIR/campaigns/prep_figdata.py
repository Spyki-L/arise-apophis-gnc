#!/usr/bin/env python3
"""Prepare figure data: nav errors/RMS + nominal FDI histories (residuals & CUSUMs)."""
import json, sys
from pathlib import Path
import numpy as np
import numpy.linalg as la
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fdi_core import FDICore, FDIConfig, FDIState, FaultType
from channels import causal_form_residual, PeerConsensusBuffer
import run_fdi_nominal as RFN

DATA = Path("/sessions/cool-hopeful-franklin/mnt/Animation/Designs/D_strobe_fullbody/nominal_v2/data/mc/draw_0000/_nav_stage")
OUT = Path(__file__).resolve().parent.parent / "results"

fleet, t_tca = RFN.load_fleet(DATA)

# ── nav errors + RMS table ────────────────────────────────────────────────────
nav = {}
rms_rows = []
for d, g in enumerate(fleet):
    raw = np.load(DATA / f"apophis_swarm_FDI_rgazel_D{d}.npz")
    err = raw["x_est"][:, 0:3] - raw["x_true"][:, 0:3]
    sig3 = 3.0 * np.sqrt(np.maximum(raw["P"][:, [0, 1, 2], [0, 1, 2]], 0.0))
    t = raw["t"].astype(float)
    nav[f"t_D{d}"] = t; nav[f"err_D{d}"] = err; nav[f"sig3_D{d}"] = sig3
    arc_h = (t[-1] - t[0]) / 3600.0
    rms = np.sqrt(np.nanmean(err ** 2, axis=0))
    rms3d = float(np.sqrt(np.nansum(rms ** 2)))
    inside = np.mean(np.all(np.abs(err) <= sig3, axis=1))
    rms_rows.append(dict(dep=d, arc_h=float(arc_h), R=float(rms[0]), T=float(rms[1]),
                         N=float(rms[2]), rms3d=rms3d, frac3sig=float(inside)))
    print(f"D{d}: arc {arc_h:.2f} h  RMS {rms.round(3)}  3D {rms3d:.2f}  in3sig {100*inside:.1f}%")
np.savez_compressed(OUT / "nav_errors.npz", **nav)
json.dump(rms_rows, open(OUT / "nav_rms.json", "w"), indent=1)

# ── nominal causal FDI histories with residual/CUSUM logging ─────────────────
cfg = FDIConfig(use_nav_channel=True, use_att_channel=True, use_comm_channel=False,
                use_form_channel=True, use_peer_channel=True,
                nu_form=3.0, leak_form=0.02, h_form=30.0,
                nu_peer=3.0, leak_peer=0.05, h_peer=20.0)
cfg.nu_nav = 1.5; cfg.leak_nav = 0.05; cfg.h_nav = 15.0
N = len(fleet)
cores = {d: FDICore(cfg) for d in range(N)}
for d in range(N):
    cores[d].reset(t0=float(fleet[d]["t"][0]))
peer = PeerConsensusBuffer(n_dep=N, latency_s=0.0, mad_floor=0.25)
events = sorted(((float(fleet[d]["t"][k]), d, k) for d in range(N) for k in range(fleet[d]["t"].size)),
                key=lambda e: e[0])
H = {d: dict(t=[], r_nav=[], S_nav=[], r_att=[], S_att=[], r_prop=[], S_prop=[],
             r_form=[], S_form=[], r_peer=[], S_peer=[], state=[], burn=[]) for d in range(N)}
FORM_WINDOW_S, FORM_SCALE_M, HB = 3600.0, 250.0, 60.0
for t_k, d, k in events:
    g = fleet[d]
    innov = g["ypre"][k]
    if (not np.all(np.isfinite(innov))) or bool(g["rejected"][k]):
        ik = Sk = None
    else:
        ik = innov; sg = np.maximum(g["sig"][k], 1e-6); Sk = np.diag(sg ** 2)
    dvc = g["dv"][k]; burn = bool(la.norm(dvc) > 0.0); de = Pvv = None
    if burn and k > 0:
        Pvv = g["P"][k, 3:6, 3:6]; de = (g["x"][k, 3:6] - g["x"][k - 1, 3:6]) - dvc
    rf = rp = None
    if (t_tca - FORM_WINDOW_S) <= t_k < t_tca:
        ttg = max(t_tca - t_k, 0.0)
        w = 1.0 + 1.5 * np.clip(1.0 - ttg / FORM_WINDOW_S, 0.0, 1.0)
        miss = causal_form_residual(
            g["x"][k, 0:3], g["x"][k, 3:6], t_k, t_tca,
            RFN._at(g["tf"], g["rC"], t_k), RFN._at(g["tf"], g["vC"], t_k),
            RFN._at(g["tf"], g["rC"], t_tca), RFN._at(g["tf"], g["vC"], t_tca),
            RFN._at(g["tf"], g["rA"], t_tca), RFN._at(g["tf"], g["vA"], t_tca),
            station_cmd=g["target"], scale_m=1.0, w_timetogo=1.0)
        peer.push(d, t_k, miss)
        scale = peer.fleet_scale(t_k, floor=FORM_SCALE_M)
        rf = (miss / scale) * w
        rp = peer.r_peer(d, t_k, own_value=miss, scale=scale)
    snap = cores[d].step(t=t_k, innov=ik, S_innov=Sk, dv_est_H=de, P_vv=Pvv,
                         burn_flag=burn, sigma_BR=g["sBR"][k], omega_BR_B=g["oBR"][k],
                         hb_age=HB, hb_nominal=HB, r_form=rf, r_peer=rp)
    c = cores[d]; h = H[d]
    h["t"].append(t_k); h["burn"].append(burn)
    h["r_nav"].append(c.r_nav); h["S_nav"].append(c.S_nav)
    h["r_att"].append(c.r_att); h["S_att"].append(c.S_att)
    h["r_prop"].append(c.r_prop); h["S_prop"].append(c.S_prop)
    h["r_form"].append(rf if rf is not None else np.nan); h["S_form"].append(c.S_form)
    h["r_peer"].append(rp if rp is not None else np.nan); h["S_peer"].append(c.S_peer)
    h["state"].append(snap.state.state.value)
pk = {}
for d in range(N):
    for kk, v in H[d].items():
        pk[f"{kk}_D{d}"] = np.asarray(v, float)
pk["t_tca"] = np.array([t_tca])
np.savez_compressed(OUT / "fdi_nominal_hist.npz", **pk)
peaks = {d: dict(NAV=max(H[d]["S_nav"]) / cfg.h_nav, ATT=max(H[d]["S_att"]) / cfg.h_att,
                 PROP=max(H[d]["S_prop"]) / cfg.h_prop, FORM=max(H[d]["S_form"]) / cfg.h_form,
                 PEER=max(H[d]["S_peer"]) / cfg.h_peer) for d in range(N)}
json.dump(peaks, open(OUT / "fleet_peaks.json", "w"), indent=1)
print("states final:", {d: FDIState(int(H[d]['state'][-1])).name for d in range(N)})
print("wrote nav_errors.npz, nav_rms.json, fdi_nominal_hist.npz, fleet_peaks.json")
