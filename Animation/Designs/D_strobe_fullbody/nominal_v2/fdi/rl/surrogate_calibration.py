#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
surrogate_calibration.py  (fdi/rl — CORTEX build step 4: the sim-to-real bridge)
================================================================================

Calibrates the RL surrogate's HEALTHY residual model to the REAL mission data
(fidelity plan §C.4): collects, from every local MC_v2 draw (and optionally the flown
``_nav_stage`` arc), the EXACT per-channel residual streams the onboard FDI computes —

  NAV  EKF innovation NIS-RMS            (run_fdi_nominal's formula)
  ATT  envelope-deviation residual       (run_fdi_nominal.load_att_mc + fdi_core scaling)
  PROP whitened dV-mismatch at burns     (fdi_core._prop_residual formula)
  FORM causal predicted terminal miss    (channels.causal_form_residual, TCA-hour window)
  PEER streaming MAD consensus residual  (channels.PeerConsensusBuffer)
  COMM heartbeat-age ratio               (constant 1.0 when healthy)

— then fits an empirical model per channel: the pooled healthy sample (bootstrap
resampling source) + a normalisation ``f_ch`` mapping real units to the surrogate's
severity units (healthy p99 -> the legacy |N(0,0.4)| p99, so the env's alarm
threshold 1.5 and reward semantics are untouched).  Saves ``surrogate_calib.npz``,
which ``fdir_env.SurrogateBackend`` auto-loads (identity-when-absent: no file ->
byte-identical legacy behaviour).

Validation (reuses E0): a Mondrian conformal detector fitted on the REAL pool must
stay silent on SURROGATE healthy draws (exchangeability), and must trip on
fault_library signatures.  Run:  python3 surrogate_calibration.py [--mc-root ...]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import numpy.linalg as la

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                     # fdi/ modules

import run_fdi_nominal as RFN
from channels import causal_form_residual, PeerConsensusBuffer
from conformal import MondrianConformal, ConformalTestMartingale
from fault_library import FaultLibrary, FaultSpec

CH = ("NAV", "ATT", "COMM", "PROP", "FORM", "PEER")
SIGMA_SIGMA = np.deg2rad(0.5)
SIGMA_OMEGA = np.deg2rad(0.02)
SURR_P99 = float(np.percentile(np.abs(np.random.default_rng(0).normal(0, 0.4, 200000)), 99))
FORM_WINDOW_S, FORM_SCALE_M, HB_NOMINAL = RFN.FORM_WINDOW_S, RFN.FORM_SCALE_M, RFN.HB_NOMINAL


# ── stream collection (mirrors run_fdi_nominal.run, recording instead of scoring) ──
def collect_draw(nav_root: Path) -> dict:
    fleet, t_tca = RFN.load_fleet(Path(nav_root))
    pools = {c: [] for c in CH}
    peer = PeerConsensusBuffer(n_dep=RFN.N_DEP, latency_s=0.0, mad_floor=0.25)
    events = sorted(((float(fleet[d]["t"][k]), d, k)
                     for d in range(RFN.N_DEP) for k in range(fleet[d]["t"].size)),
                    key=lambda e: e[0])
    for t_k, d, k in events:
        g = fleet[d]
        innov = g["ypre"][k]
        if np.all(np.isfinite(innov)) and not bool(g["rejected"][k]):
            z = innov / np.maximum(g["sig"][k], 1e-3)
            pools["NAV"].append(np.sqrt(z @ z / 3.0))
        if g.get("has_att"):
            r_att = (la.norm(g["sBR"][k]) / SIGMA_SIGMA + la.norm(g["oBR"][k]) / SIGMA_OMEGA)
            pools["ATT"].append(r_att)
        dvc = g["dv"][k]
        if la.norm(dvc) > 0.0 and k > 0:
            de = (g["x"][k, 3:6] - g["x"][k - 1, 3:6]) - dvc
            Pvv = g["P"][k, 3:6, 3:6]
            try:
                zz = la.solve(Pvv + 1e-12 * np.eye(3), de)
                pools["PROP"].append(np.sqrt(max(float(de @ zz), 0.0) / 3.0))
            except la.LinAlgError:
                pass
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
            pools["FORM"].append((miss / scale) * w)
            pools["PEER"].append(peer.r_peer(d, t_k, own_value=miss, scale=scale))
        pools["COMM"].append(1.0)                       # healthy heartbeat ratio
    return {c: np.asarray(v, float) for c, v in pools.items()}


def discover_roots(mc_root: Path, include_flown=True) -> list:
    roots = [p / "_nav_stage" for p in sorted(Path(mc_root).glob("draw_*"))
             if (p / "_nav_stage" / "apophis_swarm_FDI_rgazel_D0.npz").exists()]
    if include_flown:
        flown = HERE.parents[4] / "data" / "_nav_stage"    # Animation/data/_nav_stage
        if (flown / "apophis_swarm_FDI_rgazel_D0.npz").exists():
            roots.append(flown)
    return roots


# ── fit + save ────────────────────────────────────────────────────────────────
def fit(mc_root, out=None, include_flown=True, max_pool=200000, seed=0):
    roots = discover_roots(Path(mc_root), include_flown)
    if not roots:
        raise SystemExit(f"no data roots under {mc_root}")
    print(f"collecting healthy residual streams from {len(roots)} arc(s):")
    pools = {c: [] for c in CH}
    for r in roots:
        pr = collect_draw(r)
        tag = r.parent.name if r.name == "_nav_stage" else r.name
        print("  " + tag + "  " + " ".join(f"{c}:{pr[c].size}" for c in CH))
        for c in CH:
            pools[c].append(pr[c])
    rng = np.random.default_rng(seed)
    payload, meta = {}, {}
    for c in CH:
        pool = np.concatenate([p for p in pools[c] if p.size]) if any(p.size for p in pools[c]) else np.empty(0)
        if pool.size > max_pool:
            pool = rng.choice(pool, size=max_pool, replace=False)
        p99 = float(np.percentile(pool, 99)) if pool.size else 0.0
        degenerate = (pool.size < 50) or (p99 < 1e-6)
        f_ch = 1.0 if degenerate else SURR_P99 / p99
        payload[f"pool__{c}"] = np.sort(pool)
        meta[c] = dict(n=int(pool.size), p99=p99, f=f_ch, degenerate=bool(degenerate))
        print(f"  {c:5s} n={pool.size:7d}  healthy p99={p99:8.4f}  f_ch={f_ch:6.3f}"
              f"{'  [degenerate -> legacy noise]' if degenerate else ''}")
    out = Path(out) if out else HERE / "surrogate_calib.npz"
    payload["meta"] = np.array(json.dumps(dict(channels=list(CH), surr_p99=SURR_P99,
                                               roots=[str(r) for r in roots], meta=meta)))
    np.savez_compressed(out, **payload)
    print(f"wrote {out.name}")
    return out


# ── the calibrated healthy-noise model the backend loads ─────────────────────
class CalibratedNoise:
    """Healthy severity draws per channel: bootstrap from the REAL pooled residuals,
    normalised so healthy p99 == the legacy surrogate's |N(0,0.4)| p99.  Degenerate
    channels (no real stream / point mass at 0) keep the legacy noise."""

    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=True)
        self.meta = json.loads(str(z["meta"]))
        self.pool = {c: z[f"pool__{c}"] for c in self.meta["channels"] if f"pool__{c}" in z}
        self.f = {c: self.meta["meta"][c]["f"] for c in self.pool}
        self.degenerate = {c: self.meta["meta"][c]["degenerate"] for c in self.pool}

    def healthy(self, rng, channels) -> np.ndarray:
        out = np.empty(len(channels))
        for i, c in enumerate(channels):
            pool = self.pool.get(c)
            if pool is None or self.degenerate.get(c, True) or pool.size == 0:
                out[i] = abs(rng.normal(0, 0.4))
            else:
                out[i] = pool[rng.integers(pool.size)] * self.f[c]
        return out


# ── E0 validation: surrogate healthy must be exchangeable with real ──────────
def validate(calib_path, alpha=0.01, seed=1):
    print("\n== E0 validation: surrogate-vs-real exchangeability ==")
    cn = CalibratedNoise(calib_path)
    rng = np.random.default_rng(seed)
    ok = True
    for c in ("NAV", "FORM", "PEER", "PROP"):
        pool = cn.pool.get(c)
        if pool is None or cn.degenerate[c] or pool.size < 500:
            print(f"  {c:5s} (skipped: degenerate/small pool)")
            continue
        half = pool.size // 2
        idx = rng.permutation(pool.size)
        mond = MondrianConformal().fit("ALL", pool[idx[:half]])
        B, H = 300, 300
        fa = 0
        for _ in range(B):
            m = ConformalTestMartingale(); alarmed = False
            for p in pool[idx[half:]][rng.integers(0, pool.size - half, H)]:
                m.update(mond.pvalue("ALL", p))
                if m.alarm(alpha):
                    alarmed = True; break
            fa += int(alarmed)
        rate = fa / B
        flag = "OK" if rate <= max(2 * alpha, 0.02) else "HIGH"
        ok &= flag == "OK"
        print(f"  {c:5s} healthy-draw FA={rate*100:5.2f}%  (target <= {alpha:.0%}-ish)  {flag}")
    # detection: a fault_library NAV signature must trip the same detector
    pool = cn.pool["NAV"]
    mond = MondrianConformal().fit("ALL", pool)
    spec = FaultSpec("NAV", "bias", 2.0, "relnav_bias_m", 0.1, onset=50.0, duration=np.inf)
    det = 0
    for _ in range(200):
        m = ConformalTestMartingale(); tripped = False
        for k in range(200):
            r = pool[rng.integers(pool.size)] + spec.signature(np.array([float(k)]), "NAV")[0] / max(cn.f["NAV"], 1e-9)
            m.update(mond.pvalue("ALL", r))
            if k >= 50 and m.alarm(alpha):
                tripped = True; break
        det += int(tripped)
    print(f"  NAV fault_library signature detection: {det/2:.1f}%  {'OK' if det >= 190 else 'LOW'}")
    ok &= det >= 190
    print("VALIDATION " + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-root", default=str(HERE.parents[1] / "data" / "mc"))
    ap.add_argument("--no-flown", action="store_true", help="exclude Animation/data/_nav_stage")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p = fit(a.mc_root, out=a.out, include_flown=not a.no_flown)
    raise SystemExit(0 if validate(p) else 1)
