#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conformal_mc_calibrate.py  (Design D / nominal_v2 / fdi — CORTEX-E phase E0, MC upgrade)
========================================================================================

Upgrade of ``conformal_calibrate.py`` from the single flown nominal arc to the **MC_v2
healthy ensemble**: every local ``data/mc/draw_*/`` (200 hi-fi dispersed missions on the
cluster; pull them down and this script ingests however many are present).

Why this matters: the paper claim upgrades from "0 false alarms on ONE 10.3 h arc" to
"FA measured across N dispersed healthy missions" — the calibration now covers the
dispersion the fleet will actually fly, not one geometry.

Channels calibrated
-------------------
* NAV — real EKF innovation NIS-RMS (identical formula to ``fdi_core._nav_residual``),
  reusing ``conformal_calibrate.nav_residual`` / phase labelling.
* ATT — NEW: the envelope-deviation residual from ``run_fdi_nominal.load_att_mc``
  (pointing beyond the wheel-limited slew-lag envelope + rate beyond wheel capability),
  scaled exactly as ``fdi_core._att_residual`` (sigma_sigma=0.5 deg, sigma_omega=0.02 deg/s).
  Healthy ATT mass sits at ~0 by construction; the conformal p-value handles the point
  mass (p=1 at 0, p=1/(n+1) for any exceedance) so the martingale stays silent when healthy.

Split policy (``--split auto``)
-------------------------------
* >=2 draws → **cross-mission**: calibrate on even-indexed draws (all 8 deputies),
  validate on odd-indexed draws — the honest "calibrate on missions you flew in MC,
  monitor a mission you never saw" test.
* 1 draw → falls back to the original cross-deputy split (calib 0,2,4,6 / val 1,3,5,7).

Outputs
-------
* ``conformal_mc_calib.npz`` (in fdi/) — pooled per-phase calibration sets NAV__PH /
  ATT__PH + provenance (draw ids). ``build_detector_mc()`` loads it into a fitted
  ``PhaseScheduledConformalDetector`` for ``FDICore.attach_conformal``.
* console report: (a) end-to-end FA over every held-out real arc, (b) block-bootstrap
  FA/phase, (c) injected-drift detection power.

Pure numpy.  Run:  python3 conformal_mc_calibrate.py [--mc-root ../data/mc] [--alpha 0.01]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import numpy.linalg as la

from conformal import (MondrianConformal, ConformalTestMartingale, PhaseScheduledConformalDetector,
                       conformal_pvalues, conformal_threshold, martingale_logM_path)
from conformal_calibrate import nav_residual, PH, NAVP, FULLP, N
import run_fdi_nominal as RFN

SIGMA_SIGMA = np.deg2rad(0.5)     # fdi_core.FDIConfig defaults (keep in lockstep)
SIGMA_OMEGA = np.deg2rad(0.02)
HERE = Path(__file__).resolve().parent


# ── per-draw loading ─────────────────────────────────────────────────────────
def discover_draws(mc_root: Path) -> list:
    out = []
    for p in sorted(Path(mc_root).glob("draw_*")):
        ns = p / "_nav_stage"
        if all((ns / f"{NAVP}{d}.npz").exists() for d in range(N)):
            out.append(p)
    return out


def load_draw(draw: Path) -> list:
    """8 deputies of one draw -> per-deputy dict(t, r_nav, r_att, use, phase)."""
    ns = draw / "_nav_stage"
    t0 = min(float(np.load(ns / f"{FULLP}{d}.npz", allow_pickle=True)["t"][0]) for d in range(N))
    fleet = []
    for d in range(N):
        nav = np.load(ns / f"{NAVP}{d}.npz")
        t_abs = nav["t"].astype(float)
        r_nav, use = nav_residual(nav)
        r_att = None
        att_p = draw / "attitude" / f"att_D{d}.npz"
        if att_p.exists():
            full = np.load(ns / f"{FULLP}{d}.npz", allow_pickle=True)
            sBR, oBR = RFN.load_att_mc(att_p, full["t"].astype(float),
                                       full["C_RN"].astype(float), t_abs)
            r_att = la.norm(sBR, axis=1) / SIGMA_SIGMA + la.norm(oBR, axis=1) / SIGMA_OMEGA
        fleet.append(dict(d=d, t=t_abs - t0, r=r_nav, r_att=r_att, use=use))
    allt = np.concatenate([o["t"] for o in fleet])
    edges = np.quantile(allt, [0.0, 0.25, 0.5, 0.8, 1.0])          # timeline-fraction phases
    for o in fleet:
        idx = np.clip(np.searchsorted(edges, o["t"], side="right") - 1, 0, 3)
        o["phase"] = np.array(PH)[idx]
    return fleet


def pool_arcs(arcs, key="r") -> dict:
    out = {ph: [] for ph in PH}
    for o in arcs:
        rr = o[key] if key != "r_att" else o.get("r_att")
        if rr is None:
            continue
        for ph in PH:
            out[ph].append(rr[o["use"] & (o["phase"] == ph)])
    return {ph: (np.concatenate(v) if v else np.empty(0)) for ph, v in out.items()}


# ── calibration entry points ─────────────────────────────────────────────────
def calibrate_mc(mc_root, split="auto"):
    draws = discover_draws(Path(mc_root))
    if not draws:
        raise SystemExit(f"no draws with complete _nav_stage under {mc_root}")
    per_draw = {p.name: load_draw(p) for p in draws}
    names = list(per_draw)
    if split == "auto":
        split = "draws" if len(names) >= 2 else "deputies"
    if split == "draws":
        cal_arcs = [o for n in names[0::2] for o in per_draw[n]]
        val_arcs = [o for n in names[1::2] for o in per_draw[n]]
        cal_ids = names[0::2]; val_ids = names[1::2]
    else:
        cal_arcs = [o for n in names for o in per_draw[n] if o["d"] in (0, 2, 4, 6)]
        val_arcs = [o for n in names for o in per_draw[n] if o["d"] in (1, 3, 5, 7)]
        cal_ids = [f"{n}:deps0246" for n in names]; val_ids = [f"{n}:deps1357" for n in names]
    return dict(draws=names, split=split, cal_arcs=cal_arcs, val_arcs=val_arcs,
                cal_ids=cal_ids, val_ids=val_ids,
                calib_nav=pool_arcs(cal_arcs, "r"), val_nav=pool_arcs(val_arcs, "r"),
                calib_att=pool_arcs(cal_arcs, "r_att"), val_att=pool_arcs(val_arcs, "r_att"))


def build_detector_mc(npz_path=None, alpha=0.01) -> PhaseScheduledConformalDetector:
    """Load conformal_mc_calib.npz -> fitted detector (NAV + ATT), for attach_conformal."""
    p = Path(npz_path) if npz_path else HERE / "conformal_mc_calib.npz"
    z = np.load(p, allow_pickle=True)
    healthy = {}
    for k in z.files:
        if "__" not in k:
            continue
        ch, ph = k.split("__", 1)
        healthy.setdefault(ch, {})[ph] = z[k]
    return PhaseScheduledConformalDetector(alpha=alpha).fit(healthy)


# ── measurement ──────────────────────────────────────────────────────────────
def _arc_fa(arcs, mond_by_ch, alpha):
    """End-to-end martingale false alarms over each real held-out arc, per channel."""
    fa = {ch: 0 for ch in mond_by_ch}; n_arc = {ch: 0 for ch in mond_by_ch}
    for o in arcs:
        for ch, mond in mond_by_ch.items():
            rr = o["r"] if ch == "NAV" else o.get("r_att")
            if rr is None:
                continue
            n_arc[ch] += 1
            m = ConformalTestMartingale()
            for k in np.where(o["use"])[0]:
                m.update(mond.pvalue(o["phase"][k], rr[k]))
                if m.alarm(alpha):
                    fa[ch] += 1
                    break
    return fa, n_arc


def measure(mc_root, alpha=0.01, seed=0, B=1000, split="auto"):
    cc = calibrate_mc(mc_root, split=split)
    thr = -np.log(alpha)
    e = np.linspace(0.05, 0.95, 19)
    print(f"== MC-ensemble conformal calibration ==")
    print(f"   draws: {len(cc['draws'])} {cc['draws'][:6]}{'...' if len(cc['draws'])>6 else ''}")
    print(f"   split: {cc['split']}  calib arcs={len(cc['cal_arcs'])}  val arcs={len(cc['val_arcs'])}")

    mond = {"NAV": MondrianConformal(), "ATT": MondrianConformal()}
    have_att = any(cc["calib_att"][ph].size for ph in PH)
    for ph in PH:
        if cc["calib_nav"][ph].size:
            mond["NAV"].fit(ph, cc["calib_nav"][ph])
        if cc["calib_att"][ph].size:
            mond["ATT"].fit(ph, cc["calib_att"][ph])
        c = cc["calib_nav"][ph]
        h = conformal_threshold(c, alpha) if c.size else float("inf")
        att_n = cc["calib_att"][ph].size
        print(f"   {ph:11s} NAV calib n={c.size:6d} h(a={alpha:.0%})={h:6.3f} p99={np.percentile(c,99):6.3f}"
              + (f" | ATT n={att_n:6d} p99.9={np.percentile(cc['calib_att'][ph],99.9):.2e}" if att_n else ""))
    if not have_att:
        mond.pop("ATT")

    fa, n_arc = _arc_fa(cc["val_arcs"], mond, alpha)
    print(f"\n(a) END-TO-END FA over held-out REAL arcs ({cc['split']} split):")
    for ch in mond:
        hrs = sum((o['t'][-1]-o['t'][0]) for o in cc['val_arcs'] if (o['r'] if ch=='NAV' else o.get('r_att')) is not None)/3600.0
        print(f"    {ch}: {fa[ch]}/{n_arc[ch]} arcs alarmed  ({hrs:.1f} healthy flight-hours)")

    rng = np.random.default_rng(seed)
    H = 400
    print(f"\n(b) MEASURED NAV FA rate (block-bootstrap, H={H}, B={B}, target ARL0>={int(1/alpha)}):")
    for ph in PH:
        c, v = np.sort(cc["calib_nav"][ph]), cc["val_nav"][ph]
        if v.size < H + 5:
            print(f"    {ph:11s} (insufficient val data)")
            continue
        fam = 0
        for _ in range(B):
            s = rng.integers(0, v.size - H)
            if (martingale_logM_path(conformal_pvalues(c, v[s:s+H], presorted=True), e) >= thr).any():
                fam += 1
        print(f"    {ph:11s} Mondrian FA={fam/B*100:5.2f}%  {'OK' if fam/B <= alpha*1.8 else 'HIGH'}")

    print(f"\n(c) DETECTION on held-out residuals + injected NAV drift:")
    o = cc["val_arcs"][0]
    base = o["r"][o["use"]]; phs = o["phase"][o["use"]]
    for bump in (0.8, 1.5, 2.5):
        det = 0; delays = []
        for _ in range(300):
            s = rng.integers(0, base.size - 300)
            seg = base[s:s+300].copy(); pseg = phs[s:s+300]
            seg[120:] += bump
            m = ConformalTestMartingale()
            for k in range(300):
                m.update(mond["NAV"].pvalue(pseg[k], seg[k]))
                if k >= 120 and m.alarm(alpha):
                    det += 1; delays.append(k - 120); break
        md = int(np.median(delays)) if delays else -1
        print(f"    drift +{bump:.1f}: power={det/3:5.1f}%  ARL1(med)={md} samples")

    out = HERE / "conformal_mc_calib.npz"
    payload = {f"NAV__{ph}": cc["calib_nav"][ph] for ph in PH}
    if have_att:
        payload.update({f"ATT__{ph}": cc["calib_att"][ph] for ph in PH})
    payload["meta"] = np.array(json.dumps(dict(draws=cc["draws"], split=cc["split"],
                                               alpha=alpha, cal=cc["cal_ids"], val=cc["val_ids"])))
    np.savez_compressed(out, **payload)
    print(f"\nwrote {out.name}  (channels: {sorted({k.split('__')[0] for k in payload if '__' in k})}, "
          f"{len(cc['draws'])} draw(s), split={cc['split']})")
    return fa, n_arc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-root", default=str(HERE.parent / "data" / "mc"))
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--split", default="auto", choices=("auto", "draws", "deputies"))
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    measure(a.mc_root, alpha=a.alpha, B=a.B, split=a.split)
