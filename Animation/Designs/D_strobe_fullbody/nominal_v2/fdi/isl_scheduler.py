#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
isl_scheduler.py — joint cross-link (ISL) scheduler (F4).

Allocates each deputy's limited simultaneous ISL links over the time-varying
crosslink graph to JOINTLY serve three objectives, phase-prioritized:

  * navigation observability  (good ranging geometry / baseline diversity)
  * diagnosis observability   (every deputy heard by >= k peers, so the f-of-N /
                               structural-diagnosability condition holds)
  * safety coordination       (collision-critical neighbors always linked)

Phase weight w_phase in [0,1]: 0 = far from CA (favor long baselines / health
spread for consensus), 1 = near CA (favor close pairs: ranging precision +
collision coordination). Deterministic greedy under a per-deputy link budget B:
  (A) reserve collision-critical links, (B) guarantee k-coverage, (C) fill by score.

Mission-agnostic. Pure numpy.
"""
from __future__ import annotations
import numpy as np
import numpy.linalg as la


def feasible_links(positions, R_link):
    M = len(positions); out = []
    for i in range(M):
        for j in range(i + 1, M):
            r = la.norm(np.asarray(positions[i], float) - positions[j])
            if r <= R_link:
                out.append((i, j, r))
    return out


def _score(r, R_link, w_phase):
    near = R_link / max(r, 1.0)           # close pairs: ranging precision + collision
    far = r / R_link                      # long baselines: observability / coverage spread
    return w_phase * near + (1.0 - w_phase) * far


def schedule_isl(positions, B=4, k_diag=3, w_phase=0.5, R_link=4000.0, r_coll=300.0):
    M = len(positions)
    deg = np.zeros(M, int); links = set()
    feas = feasible_links(positions, R_link)

    def linked(a, b):
        return (min(a, b), max(a, b)) in {(min(x, y), max(x, y)) for x, y in links}

    def add(i, j):
        if linked(i, j) or deg[i] >= B or deg[j] >= B:
            return False
        links.add((min(i, j), max(i, j))); deg[i] += 1; deg[j] += 1; return True

    # (A) collision-critical links first
    for i, j, r in sorted([f for f in feas if f[2] <= r_coll], key=lambda f: f[2]):
        add(i, j)

    # (B) guarantee diagnosis k-coverage
    for _ in range(M * B):
        progressed = False
        for d in np.argsort(deg):                 # lowest-degree deputy first
            if deg[d] >= k_diag:
                continue
            cands = sorted([((j if i == d else i), _score(r, R_link, w_phase))
                            for i, j, r in feas if (i == d or j == d) and not linked(d, j if i == d else i)],
                           key=lambda c: -c[1])
            for peer, _s in cands:
                if deg[peer] < B and add(d, peer):
                    progressed = True; break
            if progressed:
                break
        if not progressed:
            break
    shortfall = [int(d) for d in range(M) if deg[d] < k_diag]

    # (C) fill remaining budget by score
    for i, j, r in sorted(feas, key=lambda f: -_score(f[2], R_link, w_phase)):
        add(i, j)

    # metrics
    A = np.zeros((M, M))
    for i, j in links:
        A[i, j] = A[j, i] = 1.0
    lam2 = float(np.sort(np.linalg.eigvalsh(np.diag(A.sum(1)) - A))[1])
    obs = []
    for d in range(M):
        dirs = [(np.asarray(positions[p], float) - positions[d]) /
                max(la.norm(np.asarray(positions[p], float) - positions[d]), 1e-9)
                for p in range(M) if A[d, p] > 0]
        if len(dirs) >= 2:
            Sd = np.mean([np.outer(u, u) for u in dirs], axis=0)
            obs.append(float(np.linalg.eigvalsh(Sd)[0]))       # higher min-eig = better spread
        else:
            obs.append(0.0)
    coll_pairs = {(min(i, j), max(i, j)) for i, j, r in feas if r <= r_coll}
    coll_reserved = coll_pairs.issubset(links)
    return dict(links=links, adjacency=A, degree=deg, lambda2=lam2,
                min_indegree=int(deg.min()), k_coverage=bool(deg.min() >= k_diag),
                shortfall=shortfall, nav_observability=float(np.mean(obs)),
                collision_reserved=bool(coll_reserved))
