#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
causal_discovery.py  (Design D / nominal_v2 / fdi — CORTEX-E phase E5, MOONSHOT)
==============================================================================

Online causal-graph discovery.  Refines the fixed E2 fault->channel graph from
flight data: from labeled single-fault episodes it estimates each edge's strength
P(channel | fault) and ACCEPTS an edge only when GROUND-VALIDATED — enough samples
(n >= n_min) and a Wilson lower-confidence bound above threshold — so spurious edges
are rejected before a refined graph is uploaded.

It recovers the hand-specified edges (F1=1), discovers a 'hidden' edge the fixed
graph missed (e.g. ATT->NAV), and the discovered graph IMPROVES abduction where the
fixed graph is wrong (an ATT fault that also perturbs NAV is correctly isolated as a
single cause instead of over-diagnosed as ATT+NAV).

Builds on the E2 SCM (causal_model).  Pure numpy.  Run: python causal_discovery.py
(self-test, 5/5).
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
import causal_model as cm

FAULTS = cm.FAULTS
CHANNELS = cm.CHANNELS


def wilson_lower(k, n, z=1.96):
    """Wilson lower confidence bound for a binomial proportion k/n."""
    if n == 0:
        return 0.0
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return center - half


def discover(episodes, n_min=30, conf_thresh=0.5, z=1.96):
    """episodes = list of (true_faults:set, observed_channels:set).  Estimate each
    fault->channel edge from SINGLE-fault episodes and ACCEPT it only when
    ground-validated: n >= n_min and Wilson lower bound of P(c|f) >= conf_thresh."""
    counts = {f: {c: [0, 0] for c in CHANNELS} for f in FAULTS}     # [k_anom, n]
    for faults, obs in episodes:
        if len(faults) != 1:
            continue
        f = next(iter(faults))
        for c in CHANNELS:
            counts[f][c][1] += 1
            counts[f][c][0] += int(c in obs)
    edges = {}
    for f in FAULTS:
        acc = {c for c in CHANNELS
               if counts[f][c][1] >= n_min and wilson_lower(*counts[f][c], z) >= conf_thresh}
        edges[f] = frozenset(acc)
    return edges


def edge_set(graph):
    return {(f, c) for f, cs in graph.items() for c in cs}


def prf(disc, true):
    D, T = edge_set(disc), edge_set(true)
    tp = len(D & T)
    prec = tp / len(D) if D else 1.0
    rec = tp / len(T) if T else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def abduce_with(effects, observed, max_card=3):
    """Minimal-cost abduction using an ARBITRARY fault->channel graph (the
    discovered one) — the E2 abduction generalized to a learned graph."""
    O = frozenset(observed)
    best, bestc = None, 1e9
    for k in range(0, max_card + 1):
        for S in combinations(FAULTS, k):
            pred = frozenset().union(*(effects[f] for f in S)) if S else frozenset()
            c = cm.W_CARD * len(S) + cm.W_UNCOV * len(O - pred) + cm.W_OVER * len(pred - O)
            if c < bestc:
                bestc, best = c, S
    return best


def _gen(strengths, faults, n_per, noise=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    eps = []
    for f in faults:
        for _ in range(n_per):
            obs = {c for c in CHANNELS if rng.random() < strengths.get((f, c), noise)}
            eps.append(({f}, obs))
    return eps


def _selftest():
    passed = total = 0

    def check(name, ok, d=""):
        nonlocal passed, total
        total += 1
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"   {d}" if d else ""))

    rng = np.random.default_rng(3)
    base = {(f, c): 0.95 for f in FAULTS for c in cm.EFFECTS[f]}        # the E2 graph as strengths
    true_E2 = {f: frozenset(cm.EFFECTS[f]) for f in FAULTS}

    disc = discover(_gen(base, FAULTS, 300, rng=rng))
    p, r, f1 = prf(disc, true_E2)
    check("T1 recover E2 fault->channel edges from data", f1 >= 0.98,
          f"precision={p:.2f} recall={r:.2f} F1={f1:.2f}")

    hidden = dict(base); hidden[("ATT", "NAV")] = 0.75
    true_h = {f: frozenset(set(cm.EFFECTS[f]) | ({"NAV"} if f == "ATT" else set())) for f in FAULTS}
    disc_h = discover(_gen(hidden, FAULTS, 400, rng=rng))
    check("T2 discover a hidden edge (ATT->NAV) absent from the fixed graph",
          "NAV" in disc_h["ATT"] and "NAV" not in cm.EFFECTS["ATT"],
          f"discovered ATT->{set(disc_h['ATT'])}  (fixed ATT->{set(cm.EFFECTS['ATT'])})")

    p3, r3, f3 = prf(disc_h, true_h)
    n_min_reject = discover(_gen(base, FAULTS, 12, rng=rng))            # n < n_min -> nothing accepted
    check("T3 gate: no spurious edges (precision=1) & under-sampled -> rejected",
          p3 >= 0.99 and len(edge_set(n_min_reject)) == 0,
          f"precision(hidden run)={p3:.2f}  edges@n=12={len(edge_set(n_min_reject))}")

    f_small = prf(discover(_gen(hidden, FAULTS, 15, rng=rng)), true_h)[2]
    f_big = prf(discover(_gen(hidden, FAULTS, 400, rng=rng)), true_h)[2]
    check("T4 convergence: F1 increases with samples", f_big >= 0.95 and f_big > f_small,
          f"F1(n=15)={f_small:.2f} -> F1(n=400)={f_big:.2f}")

    O = {"ATT", "NAV"}                                                  # a pure ATT fault (with hidden edge)
    fixed_dx = abduce_with({f: frozenset(cm.EFFECTS[f]) for f in FAULTS}, O)
    disc_dx = abduce_with(disc_h, O)
    check("T5 discovered graph -> correct single-cause diagnosis (fixed graph over-diagnoses)",
          set(disc_dx) == {"ATT"} and set(fixed_dx) != {"ATT"},
          f"fixed graph dx={fixed_dx}  discovered graph dx={disc_dx}")

    print("-" * 66)
    print(f"E5 causal-discovery self-test: {passed}/{total} PASS")
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
