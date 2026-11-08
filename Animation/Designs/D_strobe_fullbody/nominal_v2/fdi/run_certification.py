#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_certification.py — F10 certification campaign runner (prints the dossier tables).

Usage:
    python run_certification.py --data <dir with the 8 deputy bundles> [--n 200]

The --data dir holds apophis_swarm_FDI_rgazel_D*.npz, apophis_rendezvous_full_D*.npz,
dep0*_logs.npz (e.g. GNC/Scenario/Apophis/). Byzantine + recovery campaigns need no data.
"""
from __future__ import annotations
import argparse
import certification as cert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dir with the deputy bundles (detection campaign)")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    print("=" * 74)
    print("ARISE/Apophis swarm FDIR — certification campaign (F10)")
    print("=" * 74)

    # 1) detection / isolation / latency on the real fleet
    if a.data:
        print(f"\n[1] Detection / isolation / latency  (N={a.n}/scenario, real fleet)")
        F = cert.load_fleet(a.data)
        d = cert.detection_campaign(F, n_trials=a.n)
        print(f"  {'fault':5s} {'Pd':>6s}  {'95% CI':>16s}  {'lat median':>11s}  {'lat p95':>9s}")
        for sc, v in d["per_scenario"].items():
            lo, hi = v["Pd_ci"]
            print(f"  {sc:5s} {v['Pd']:6.3f}  [{lo:.3f}, {hi:.3f}]  {v['lat_median']:8.0f} s   {v['lat_p95']:7.0f} s")
        plo, phi = d["Pfa_ci"]
        print(f"  Pfa (per healthy deputy/encounter) = {d['Pfa']:.4f}  [{plo:.4f}, {phi:.4f}]  (n={d['Pfa_n']})")
        print("\n  Isolation confusion (rows=truth, cols=identified):")
        LAB = ["NONE", "NAV", "ATT", "PROP", "COMM"]
        print("    truth |" + "".join(f"{l:>7}" for l in LAB))
        for sc, row in d["confusion"].items():
            print(f"    {sc:>5} |" + "".join(f"{row[l]:>7}" for l in LAB))
    else:
        print("\n[1] Detection campaign skipped (pass --data <bundle dir>).")

    # 2) Byzantine f-of-N tolerance
    print(f"\n[2] Byzantine f-of-N tolerance  (PEER-CBF consistency, N=8)")
    b = cert.byzantine_campaign(n_trials=max(a.n*2, 300))
    print(f"  {'f liars':8s} {'Pd(liar)':>10s} {'Pfa(honest)':>12s}")
    for f, v in b.items():
        pd = f"{v['Pd']:.3f}" if v["Pd"] is not None else "  n/a"
        print(f"  {f:<8d} {pd:>10s} {v['Pfa']:>12.4f}")
    print("  -> robust while f < N/2 = 4; the median breaks at f = 4 (theoretical bound).")

    # 3) recovery efficacy
    print(f"\n[3] Recovery efficacy  (random single/double deputy loss)")
    r = cert.recovery_campaign(n_trials=max(a.n // 2, 80))
    print(f"  nominal coverage         = {r['cov_nom']:.3f}")
    print(f"  mean gap recovered       = {100*r['mean_recovered']:.1f}%  (p10 = {100*r['p10']:.1f}%)")
    print(f"  beat-tau* (reachable)    = {100*r['beat_tau_frac']:.1f}%   over n={r['n']} losses")

    print("\n" + "=" * 74)
    print("Campaign complete. See CERTIFICATION_DOSSIER.md and figs/certification_summary.png")


if __name__ == "__main__":
    main()
