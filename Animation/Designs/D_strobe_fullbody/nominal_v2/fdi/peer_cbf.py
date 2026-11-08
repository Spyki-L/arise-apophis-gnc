#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
peer_cbf.py — PEER channel as a robust CBF layer (F4).

The PEER channel stops being a MAD outlier score (which blew up on a homogeneous
fleet) and becomes the fleet's safety + consistency supervisor, expressed as
Control Barrier Functions. Three barrier families, each a residual source:

  collision    : pairwise keep-out CBF   h_ij = |r_ij|^2 - d_safe^2,
                 margin = hdot + gamma*h  (>=0 safe); residual = violation rate,
                 with a heavy penalty for already being inside the keep-out.
  connectivity : algebraic connectivity (Fiedler value lambda2) of the range
                 graph; >0 means the fleet can still carry nav + consensus.
  consistency  : each deputy's REPORTED position vs peers' MEASURED ranges.
                 Byzantine-robust by construction: a liar disagrees with MANY
                 independent rangers (its median is high) while an honest deputy
                 has only its single link to the liar bad (median outvotes it).

PEER fault = persistent CBF violation or range-inconsistency. A homogeneous,
healthy fleet trips none of these — the documented MAD-collapse false alarm is
gone structurally. Mission-agnostic. Pure numpy.
"""
from __future__ import annotations
import numpy as np
import numpy.linalg as la


def collision_cbf(r_i, r_j, v_i, v_j, d_safe, gamma):
    rij = np.asarray(r_i, float) - np.asarray(r_j, float)
    vij = np.asarray(v_i, float) - np.asarray(v_j, float)
    h = float(rij @ rij) - d_safe**2
    hdot = 2.0 * float(rij @ vij)
    return h, hdot, hdot + gamma * h          # margin >= 0 is safe


def fleet_collision(positions, velocities, d_safe, gamma):
    M = len(positions); viol = np.zeros(M); pairs = []
    for i in range(M):
        for j in range(i + 1, M):
            h, hd, margin = collision_cbf(positions[i], positions[j],
                                          velocities[i], velocities[j], d_safe, gamma)
            v = max(0.0, -margin) / (gamma * d_safe**2)        # CBF-violation rate (normalized)
            if h < 0.0:
                v += 5.0 * (-h) / d_safe**2                    # already inside keep-out (hard, weighted)
            pairs.append((i, j, h, margin, v))
            viol[i] = max(viol[i], v); viol[j] = max(viol[j], v)
    return viol, pairs


def connectivity_lambda2(positions, R_link):
    M = len(positions); A = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            if la.norm(np.asarray(positions[i], float) - positions[j]) <= R_link:
                A[i, j] = A[j, i] = 1.0
    L = np.diag(A.sum(1)) - A
    w = np.sort(np.linalg.eigvalsh(L))
    return float(w[1]), A                                      # Fiedler value; >0 => connected


def consistency_residual(reported, measured_ranges, sigma_range):
    """Per-deputy median |measured_ij - ||rep_i-rep_j|| | / sigma over its links (robust)."""
    reported = np.asarray(reported, float); M = len(reported); res = np.zeros(M)
    for i in range(M):
        diffs = []
        for j in range(M):
            if i == j:
                continue
            mij = measured_ranges[i, j]
            if not np.isfinite(mij):
                continue
            diffs.append(abs(mij - la.norm(reported[i] - reported[j])) / max(sigma_range, 1e-6))
        res[i] = float(np.median(diffs)) if diffs else 0.0
    return res


class PeerCBF:
    def __init__(self, d_safe=50.0, R_link=5000.0, gamma=0.1, sigma_range=0.1, lambda_min=0.2):
        self.d_safe = d_safe; self.R_link = R_link; self.gamma = gamma
        self.sigma_range = sigma_range; self.lambda_min = lambda_min

    def evaluate(self, positions, velocities, reported, measured_ranges):
        coll, pairs = fleet_collision(positions, velocities, self.d_safe, self.gamma)
        lam2, A = connectivity_lambda2(positions, self.R_link)
        cons = consistency_residual(reported, measured_ranges, self.sigma_range)
        conn_pen = max(0.0, self.lambda_min - lam2)            # fleet-level; added to all if disconnected
        r_peer = np.maximum(coll, cons) + conn_pen             # per-deputy PEER residual for the CUSUM
        return dict(r_peer=r_peer, collision=coll, consistency=cons,
                    lambda2=lam2, conn_penalty=conn_pen, pairs=pairs, adjacency=A)
