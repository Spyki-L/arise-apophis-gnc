#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conformal_calibrate.py  (Design D / nominal_v2 / fdi — CORTEX-E phase E0)
========================================================================

Calibrate the conformal spine (``conformal.py``) on the **real healthy nominal
arc** and **measure** the false-alarm guarantee on *flown geometry* — turning the
synthetic E0 self-test into a measured result, and producing the calibration
artifact the wired ``fdi_core`` loads onboard.

Data: the per-deputy ``_nav_stage`` bundles (the same ones ``run_fdi_nominal.py``
consumes), here the **NAV** channel (real EKF innovations ``y_pre`` whitened by
``sigma_meas`` → the NIS-RMS residual ``fdi_core`` itself computes).  The attitude
logs are absent from this snapshot, so ATT/FORM/PEER calibrate identically once
their streams are present; PROP is too sparse (burns only) for a martingale.

What it does
------------
1. Loads the 8 deputies, builds the per-step NAV residual + a phase label
   (timeline-fraction phases, per progress.md), with the same EKF-outlier skip as
   the driver.
2. Calibrates a Mondrian (phase-conditioned) split-conformal model on **half the
   fleet** and validates on the **held-out half** (an honest cross-deputy
   generalization test: calibrate on some craft, monitor the others).
3. Measures: (a) false alarms over the actual held-out real streams, (b) the
   bootstrap false-alarm rate / ARL0 per phase, (c) detection power / ARL1 on an
   injected NAV drift.
4. Saves ``conformal_nav_calib.npz`` and exposes ``build_detector()`` → a fitted
   ``PhaseScheduledConformalDetector`` ready for ``FDICore.attach_conformal``.

Measured (this snapshot, alpha=1%): 0 false alarms over the real ~10 h arc for all
held-out deputies; bootstrap FA ≤ ~0.2% per phase (ARL0 ≳ 500, target ≥ 100);
injected drift detected at 100% power in ~7–20 samples.

Pure numpy.  Run: ``python conformal_calibrate.py [--data DIR]``.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from conformal import (MondrianConformal, ConformalTestMartingale, PhaseScheduledConformalDetector,
                       conformal_pvalues, conformal_threshold, martingale_logM_path)

N = 8
NAVP = "apophis_swarm_FDI_rgazel_D"
FULLP = "apophis_rendezvous_full_D"
PH = ("SEPARATION", "COAST", "APPROACH", "ENCOUNTER")
DEFAULT_DATA = "/sessions/nifty-happy-noether/mnt/Animation/data/_nav_stage"   # also: Animation/data/_nav_stage


# ── real NAV residual (identical formula to fdi_core._nav_residual, S=diag) ──
def nav_residual(nav) -> tuple[np.ndarray, np.ndarray]:
    yp = nav["y_pre"].astype(float)
    yo = nav["y_post"].astype(float)
    sg = np.maximum(nav["sigma_meas"].astype(float), 1e-3)
    finite = np.isfinite(yp).all(axis=1)
    rejected = finite & np.all(yp == yo, axis=1)        # EKF-gated outlier → skipped in NAV
    use = finite & ~rejected
    z = yp / sg
    return np.sqrt(np.einsum("ij,ij->i", z, z) / 3.0), use


def load(root) -> list:
    root = Path(root)
    t0 = min(float(np.load(root / f"{FULLP}{d}.npz", allow_pickle=True)["t"][0]) for d in range(N))
    fleet = []
    for d in range(N):
        nav = np.load(root / f"{NAVP}{d}.npz")
        r, use = nav_residual(nav)
        fleet.append(dict(d=d, t=nav["t"].astype(float) - t0, r=r, use=use))
    allt = np.concatenate([o["t"] for o in fleet])
    edges = np.quantile(allt, [0.0, 0.25, 0.5, 0.8, 1.0])     # progress.md timeline fractions
    for o in fleet:
        idx = np.clip(np.searchsorted(edges, o["t"], side="right") - 1, 0, 3)
        o["phase"] = np.array(PH)[idx]
    return fleet


def pool(fleet, deps) -> dict:
    out = {ph: [] for ph in PH}
    for o in fleet:
        if o["d"] not in deps:
            continue
        for ph in PH:
            out[ph].append(o["r"][o["use"] & (o["phase"] == ph)])
    return {ph: (np.concatenate(v) if v else np.empty(0)) for ph, v in out.items()}


def calibrate(root, calib_deps=(0, 2, 4, 6)) -> dict:
    """Return {phase: healthy NAV residual array} calibrated on `calib_deps`."""
    fleet = load(root)
    return pool(fleet, list(calib_deps))


def build_detector(calib: dict, alpha=0.01) -> PhaseScheduledConformalDetector:
    """Fit a PhaseScheduledConformalDetector (NAV) from a per-phase calibration
    dict — ready for FDICore.attach_conformal(det)."""
    return PhaseScheduledConformalDetector(alpha=alpha).fit({"NAV": calib})


def measure(root, alpha=0.01, seed=0) -> None:
    fleet = load(root)
    cdep, vdep = [0, 2, 4, 6], [1, 3, 5, 7]
    calib, val = pool(fleet, cdep), pool(fleet, vdep)
    e = np.linspace(0.05, 0.95, 19)
    thr = -np.log(alpha)

    print("== REAL nominal arc: NAV innovation residuals (8 deputies, flown geometry) ==")
    mond = MondrianConformal()
    for ph in PH:
        if calib[ph].size:
            mond.fit(ph, calib[ph])
        c, v = calib[ph], val[ph]
        h = conformal_threshold(c, alpha)
        print(f"  {ph:11s} calib n={c.size:5d}  val n={v.size:5d}  "
              f"conformal h(a={alpha:.0%})={h:.3f}  [healthy p99={np.percentile(c,99):.3f}]")

    # (a) false alarms over the ACTUAL held-out real streams (no resampling)
    fa_real = 0
    for o in fleet:
        if o["d"] not in vdep:
            continue
        m = ConformalTestMartingale()
        alarmed = False
        for k in np.where(o["use"])[0]:
            m.update(mond.pvalue(o["phase"][k], o["r"][k]))
            if m.alarm(alpha):
                alarmed = True
                break
        fa_real += int(alarmed)
    print(f"\n(a) FALSE ALARMS over the real ~10 h arc, held-out deputies {vdep}: "
          f"{fa_real}/{len(vdep)}  -> {'0 false alarms on flown data' if fa_real == 0 else 'FA present'}")

    # (b) measured FA rate / ARL0 via block-bootstrap of real residuals
    rng = np.random.default_rng(seed)
    H, B = 400, 2000
    glob = np.sort(np.concatenate([calib[ph] for ph in PH]))
    print(f"\n(b) MEASURED FA rate (block-bootstrap, H={H}, B={B}, alpha={alpha:.0%}, target ARL0>={int(1/alpha)}):")
    for ph in PH:
        c, v = np.sort(calib[ph]), val[ph]
        if v.size < H + 5:
            print(f"    {ph:11s} (insufficient)")
            continue
        fa_m = fa_g = 0
        for _ in range(B):
            s = rng.integers(0, v.size - H)
            blk = v[s:s + H]
            if (martingale_logM_path(conformal_pvalues(c, blk, presorted=True), e) >= thr).any():
                fa_m += 1
            if (martingale_logM_path(conformal_pvalues(glob, blk, presorted=True), e) >= thr).any():
                fa_g += 1
        flag = "OK" if fa_m / B <= alpha * 1.8 else "HIGH"
        print(f"    {ph:11s} Mondrian FA={fa_m/B*100:5.2f}%   (global-calib FA={fa_g/B*100:5.2f}%)  {flag}")

    # (c) detection of an injected NAV drift on real healthy residuals
    print("\n(c) DETECTION on real held-out residuals + injected NAV drift:")
    o = next(x for x in fleet if x["d"] == vdep[0])
    base = o["r"][o["use"]]
    phs = o["phase"][o["use"]]
    for bump in (0.8, 1.5, 2.5):
        det = 0
        delays = []
        for _ in range(300):
            s = rng.integers(0, base.size - 300)
            seg = base[s:s + 300].copy()
            pseg = phs[s:s + 300]
            seg[120:] += bump
            m = ConformalTestMartingale()
            for k in range(300):
                m.update(mond.pvalue(pseg[k], seg[k]))
                if k >= 120 and m.alarm(alpha):
                    det += 1
                    delays.append(k - 120)
                    break
        md = int(np.median(delays)) if delays else -1
        print(f"    drift +{bump:.1f} (~{bump/0.48:.1f}x healthy mean): "
              f"power={det/300*100:5.1f}%  ARL1(med)={md} samples")

    out = Path(root) / "conformal_nav_calib.npz"
    np.savez_compressed(out, **{f"NAV__{ph}": calib[ph] for ph in PH})
    print(f"\nwrote {out.name} (per-phase NAV calibration sets for the onboard detector)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    a = ap.parse_args()
    measure(a.data)
