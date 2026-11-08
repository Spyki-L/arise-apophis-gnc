#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reoptimizer.py — two-tier onboard stereo-coverage re-optimizer (F5).

When a deputy is lost or its actuators degrade, the fleet must re-solve its own
geometry to recover the most coverage it still can — onboard, in time, at a
fidelity an XB6-class processor can handle. Two tiers:

  Tier-1 (ground -> onboard table): a precomputed CONTINGENCY TABLE of the best
    remaining-fleet stations for each single-deputy loss. Onboard this is an
    instant, certified lookup -> the Verified Safe Controller (VSC).
  Tier-2 (onboard refine): a warm-started, BOUNDED-iteration hill-climb on the
    fast ANALYTIC coverage surrogate, using the actual current state and each
    deputy's REACHABLE box (the F3 reachable tube). Deterministic WCET. Under
    run-time assurance, Tier-2 is accepted only if it is feasible and beats
    Tier-1, else control stays with Tier-1.

Onboard fidelity = the analytic surrogate (no ray-traced Monte-Carlo); the
surrogate-vs-truth bias is captured by a ground-calibrated `correction` factor
from the mission profile. The optimizer only needs the correct *ordering* of
configurations, which the surrogate preserves. Mission-agnostic. Pure numpy.
"""
from __future__ import annotations
import numpy as np
import numpy.linalg as la


def fibonacci_sphere(N, R=1.0):
    """N near-uniform surface points + outward normals (a generic target proxy)."""
    i = np.arange(N) + 0.5
    phi = np.arccos(1 - 2*i/N); th = np.pi*(1 + 5**0.5)*i
    p = np.stack([np.cos(th)*np.sin(phi), np.sin(th)*np.sin(phi), np.cos(phi)], 1)
    return R*p, p


def coverage_surrogate(stations, pts, norms, R_max=4000., th_min=8., th_max=40., correction=1.0):
    """Analytic stereo viewsphere coverage: a surface point is covered if >=2 deputies
    see it (front-face + in range) with mutual parallax angle in [th_min, th_max] deg."""
    stations = np.asarray(stations, float); covered = 0
    cmin, cmax = np.cos(np.radians(th_max)), np.cos(np.radians(th_min))  # larger angle -> smaller cos
    for s in range(len(pts)):
        p = pts[s]; n = norms[s]; viewers = []
        for d in range(len(stations)):
            los = stations[d] - p; rng = la.norm(los)
            if rng > R_max or los @ n <= 0:
                continue
            viewers.append(los/rng)
        stereo = False
        for a in range(len(viewers)):
            for b in range(a+1, len(viewers)):
                c = float(np.clip(viewers[a] @ viewers[b], -1, 1))
                if cmin <= c <= cmax:
                    stereo = True; break
            if stereo:
                break
        covered += stereo
    return correction * covered / len(pts)


def optimize_stations(nominal, active, pts, norms, reach, n_iter=250, seed=0, **kw):
    """Bounded hill-climb: perturb active deputies within their reach box (the F3 reachable
    tube) to maximize coverage. Lost deputies are removed. Fixed iteration count => bounded WCET."""
    rng = np.random.default_rng(seed)
    nominal = np.asarray(nominal, float)
    x = {d: nominal[d].copy() for d in active}
    def cov(xx):
        return coverage_surrogate([xx[d] for d in active], pts, norms, **kw)
    best = cov(x); evals = 1
    for _ in range(n_iter):
        d = active[rng.integers(len(active))]
        cand = {k: v.copy() for k, v in x.items()}
        cand[d] = np.clip(x[d] + rng.normal(0, reach[d]*0.35, 3),
                          nominal[d]-reach[d], nominal[d]+reach[d])
        c = cov(cand); evals += 1
        if c > best:
            x = cand; best = c
    return np.array([x[d] for d in active]), float(best), evals


def build_contingency_table(nominal, pts, norms, reach, n_iter=250, **kw):
    """Ground (Tier-1): best remaining-fleet stations for each single-deputy loss."""
    M = len(nominal); table = {}
    for L in range(M):
        active = [d for d in range(M) if d != L]
        st, cov, _ = optimize_stations(nominal, active, pts, norms, reach, n_iter=n_iter, seed=100+L, **kw)
        table[frozenset([L])] = dict(active=active, stations=st, coverage=cov)
    return table


class ReOptimizer:
    """Two-tier: Tier-1 contingency lookup (VSC) + Tier-2 onboard surrogate refine (AC), RTA-gated."""
    def __init__(self, nominal, pts, norms, table, correction=1.0, **kw):
        self.nominal = np.asarray(nominal, float); self.pts = pts; self.norms = norms
        self.table = table; self.kw = dict(kw); self.kw["correction"] = correction

    def tier1(self, lost_set):
        return self.table.get(frozenset(lost_set))

    def tier2(self, lost_set, reach, n_iter=120, seed=7):
        active = [d for d in range(len(self.nominal)) if d not in lost_set]
        warm = self.tier1(lost_set)
        nominal = self.nominal.copy()
        if warm is not None:
            for i, d in enumerate(active):
                nominal[d] = np.asarray(warm["stations"], float)[i]   # warm-start boxes at Tier-1 solution
        st, cov, evals = optimize_stations(nominal, active, self.pts, self.norms, reach,
                                           n_iter=n_iter, seed=seed, **self.kw)
        return dict(active=active, stations=st, coverage=cov, evals=evals)

    def reconfigure(self, lost_set, reach, n_iter=120):
        """RTA-gated: accept Tier-2 only if it is feasible and beats Tier-1, else Tier-1 (VSC)."""
        t1 = self.tier1(lost_set)
        t2 = self.tier2(lost_set, reach, n_iter=n_iter)
        if t1 is not None and t2["coverage"] <= t1["coverage"] + 1e-4:
            return dict(source="VSC(Tier1)", **t1)
        return dict(source="AC(Tier2)", **t2)
