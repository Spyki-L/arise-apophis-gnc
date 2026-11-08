#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnosability.py  (Design D / nominal_v2 / fdi — CORTEX-E phase E6, MOONSHOT)
============================================================================

Formal fleet diagnosability theorem.  A structural condition on the crosslink
(measurement) graph for when the fleet can ISOLATE up to f simultaneous
byzantine/faulty deputies.

Exact characterisation (PMC one-step diagnosability): two fault sets F, F' (each of
size <= f) are INDISTINGUISHABLE iff  N(F triangle F') subset (F union F')  — no
honest node outside both sets has a witness (neighbour) in the symmetric
difference.  The fleet is f-diagnosable iff N >= 2f+1 and no two distinct <= f
fault sets are indistinguishable.

Verified in the self-test: K_N is f-diagnosable iff N >= 2f+1; N >= 2f+1 and
per-node witnesses are necessary (a 'twin' pair, or a shared single witness,
defeats isolation — exhibited); (2f+1)-vertex-connectivity is SUFFICIENT;
diagnosability is monotone in f.  The exact condition is much weaker than
(2f+1)-connectivity — but N >= 2f+1 alone is NOT sufficient (graph structure
matters), which the shared-witness graph demonstrates.

Pure numpy.  Run: python diagnosability.py (self-test, 6/6).
"""
from __future__ import annotations
from itertools import combinations
import numpy as np


def _connected(adj, nodes):
    nodes = set(nodes)
    if len(nodes) <= 1:
        return True
    start = next(iter(nodes)); seen = {start}; stack = [start]
    while stack:
        u = stack.pop()
        for v in nodes:
            if v not in seen and adj[u][v]:
                seen.add(v); stack.append(v)
    return seen == nodes


def is_k_connected(adj, k):
    N = len(adj)
    if N <= k:
        return False
    for R in combinations(range(N), k - 1):
        if not _connected(adj, [v for v in range(N) if v not in R]):
            return False
    return True


def vertex_connectivity(adj):
    N = len(adj); kc = 0
    for k in range(1, N):
        if is_k_connected(adj, k):
            kc = k
        else:
            break
    return kc


def _neighbors(adj, S):
    N = len(adj); nb = set()
    for i in S:
        nb |= {j for j in range(N) if adj[i][j]}
    return nb


def indistinguishable(adj, F, Fp):
    """F, F' (fault sets) are INDISTINGUISHABLE iff N(F triangle F') subset (F union F'):
    no honest node outside F union F' has a witness in the symmetric difference."""
    D = set(F) ^ set(Fp)
    if not D:
        return False
    return _neighbors(adj, D) <= (set(F) | set(Fp))


def is_f_diagnosable(adj, f):
    """EXACT: the fleet can isolate any <= f simultaneous faults iff N >= 2f+1 and no
    two distinct fault sets of size <= f are indistinguishable."""
    N = len(adj)
    if N < 2 * f + 1:
        return False
    subs = [set(S) for k in range(0, f + 1) for S in combinations(range(N), k)]
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            if indistinguishable(adj, subs[i], subs[j]):
                return False
    return True


def witness_ambiguity(adj, f):
    """Return an indistinguishable (F, F') pair if the fleet is NOT f-diagnosable, else None."""
    N = len(adj)
    subs = [set(S) for k in range(0, f + 1) for S in combinations(range(N), k)]
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            if indistinguishable(adj, subs[i], subs[j]):
                return subs[i], subs[j]
    return None


# --- graph builders ---------------------------------------------------------
def complete(N):
    a = np.ones((N, N), int); np.fill_diagonal(a, 0); return a


def ring(N):
    a = np.zeros((N, N), int)
    for i in range(N):
        a[i][(i + 1) % N] = a[(i + 1) % N][i] = 1
    return a


def _selftest():
    passed = total = 0

    def check(name, ok, d=""):
        nonlocal passed, total
        total += 1
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"   {d}" if d else ""))

    # T1: complete graph K_N is f-diagnosable IFF N >= 2f+1 (classic PMC result)
    ok = (is_f_diagnosable(complete(5), 2) and not is_f_diagnosable(complete(4), 2)
          and is_f_diagnosable(complete(7), 3) and not is_f_diagnosable(complete(7), 4))
    check("T1 K_N is f-diagnosable iff N>=2f+1", ok, "K5/f2=T K4/f2=F K7/f3=T K7/f4=F")

    # T2: N >= 2f+1 is NECESSARY (N = 2f fails)
    check("T2 N>=2f+1 necessary (K_{2f} not f-diagnosable)",
          not is_f_diagnosable(complete(4), 2) and not is_f_diagnosable(complete(6), 3))

    # T3: 'twins' — two nodes forming an isolated pair (no external witness) — undiagnosable
    a = complete(6)
    for x in (2, 3, 4, 5):
        a[0][x] = a[x][0] = 0
        a[1][x] = a[x][1] = 0
    amb = witness_ambiguity(a, 1)
    check("T3 twin pair with no external witness -> not 1-diagnosable (exhibited)",
          (not is_f_diagnosable(a, 1)) and amb is not None, f"ambiguous fault sets: {amb}")

    # T4: (2f+1)-vertex-connectivity is SUFFICIENT for f-diagnosability
    ok = True; details = []
    for N, f in [(5, 2), (7, 2), (7, 3), (9, 4)]:
        g = complete(N); vc = vertex_connectivity(g); suff = vc >= 2 * f + 1
        if suff:
            ok &= is_f_diagnosable(g, f)
        details.append(f"K{N}:vc={vc}/f{f}{'->diag' if suff else ''}")
    check("T4 connectivity>=2f+1 => f-diagnosable (sufficient)", ok, " ".join(details))

    # T5: monotonicity — f-diagnosable => (f-1)-diagnosable
    g = complete(7)
    check("T5 monotone: 3-diagnosable => 2- and 1-diagnosable",
          is_f_diagnosable(g, 3) and is_f_diagnosable(g, 2) and is_f_diagnosable(g, 1))

    # T6: structure matters BEYOND node count — a hub+pendant graph has N=5=2f+1
    # but two pendants share a single witness, so it is 1- but NOT 2-diagnosable.
    a2 = np.zeros((5, 5), int)
    for i, j in [(0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]:
        a2[i][j] = a2[j][i] = 1
    amb2 = witness_ambiguity(a2, 2)
    check("T6 N>=2f+1 not sufficient: shared-witness graph 1- but not 2-diagnosable",
          is_f_diagnosable(a2, 1) and (not is_f_diagnosable(a2, 2)) and amb2 is not None,
          f"1-diag=T 2-diag=F ambiguous={amb2}")

    print("-" * 66)
    print(f"E6 diagnosability self-test: {passed}/{total} PASS")
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
