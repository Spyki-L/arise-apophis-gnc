#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crosslink_scheduler.py
======================

Round-robin cross-link pairing for a deputy swarm.

Given N deputies, builds N-1 "rounds" of disjoint pairs such that:

- each deputy appears in exactly one pair per round
- every unordered pair {i,j} appears exactly once across all rounds

This is a 1-factorization of K_N (classic circle method). For odd N,
we add a dummy node; any pair containing dummy is ignored.

Usage
-----
    sched = CrosslinkScheduler(N_dep)
    pairs_k = sched.pairs_for_step(k)  # list of (i, j) at time index k
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


def _round_robin_pairs(N: int) -> List[List[Tuple[int, int]]]:
    """
    Internal: generate rounds of disjoint pairs for N nodes (0..N-1).
    Assumes N is even.
    """
    nodes = list(range(N))
    rounds: List[List[Tuple[int, int]]] = []

    for _ in range(N - 1):
        pairs: List[Tuple[int, int]] = []
        for i in range(N // 2):
            a = nodes[i]
            b = nodes[N - 1 - i]
            pairs.append((a, b))
        rounds.append(pairs)
        # rotate all but first element
        nodes = [nodes[0]] + nodes[-1:] + nodes[1:-1]

    return rounds


@dataclass
class CrosslinkScheduler:
    N_dep: int

    def __post_init__(self):
        N = self.N_dep
        self._dummy = (N % 2 == 1)
        if self._dummy:
            N_eff = N + 1
        else:
            N_eff = N

        base_rounds = _round_robin_pairs(N_eff)

        self.rounds: List[List[Tuple[int, int]]] = []
        for r in base_rounds:
            pairs: List[Tuple[int, int]] = []
            for i, j in r:
                # drop pairs involving dummy node
                if self._dummy and (i == N_eff - 1 or j == N_eff - 1):
                    continue
                if i < self.N_dep and j < self.N_dep:
                    pairs.append((i, j))
            self.rounds.append(pairs)

        self.n_rounds = len(self.rounds)

    def pairs_for_step(self, k: int) -> List[Tuple[int, int]]:
        """
        Return list of (i,j) pairs to be measured at discrete time index k.
        Rounds repeat cyclically.
        """
        if self.n_rounds == 0:
            return []
        r = k % self.n_rounds
        return self.rounds[r]

