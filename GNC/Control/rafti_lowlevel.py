#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rafti_lowlevel.py
=================

Low-level utilities for thruster-based wrench allocation, intended to be
reused by SE(3) controllers and separation drivers.

This module does **not** depend on Basilisk; it just works with any
"thruster-like" objects that expose:

    • r_B : (3,) position of thruster in body frame [m]
    • a_B : (3,) nominal thrust direction unit-vector in body frame
    • T   : scalar max thrust [N]
    • Isp : specific impulse [s] (used only for ΔV / prop estimates)

Main utilities
--------------

1) build_wrench_map(thrusters)
   ----------------------------
Given an iterable of thrusters, builds the 6×N wrench matrix B such that

    u = B δ   with   u = [ τ_B ; f_B ]   (6×1),  δ ∈ [0,1]^N

where τ_B is body torque, f_B is body force, and δ are throttle fractions
or duty cycles.

Returns:

    B     : (6, N) wrench map
    Ts    : (N,)   max thrust per thruster [N]
    Isps  : (N,)   specific impulses [s]

2) bnls_solve(...)
   ----------------
Bounded, regularised, weighted least-squares allocator:

    min_δ ‖ W (B δ − u) ‖² + λ ‖ δ − δ_prev ‖²
    s.t.  lb ≤ δ ≤ ub

with W = diag(wτ,wτ,wτ,wf,wf,wf). This is meant as a "general-purpose"
backend for controllers that want a soft, thruster-level allocation
without coding the linear algebra inline.

Typical usage from a controller:

    u_des   = np.hstack((tau_residual, f_desired))    # 6-vector
    delta   = bnls_solve(B, u_des, delta_prev,
                         w_tau=w_torque, w_for=w_force,
                         lam=reg_lambda, lb=0.0, ub=1.0)
    u_alloc = B @ delta
"""

from __future__ import annotations
from typing import Iterable, Tuple, Optional

import numpy as np
import numpy.linalg as la


def build_wrench_map(thrusters: Iterable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 6×N wrench map for an iterable of thruster-like objects.

    Parameters
    ----------
    thrusters : iterable
        Any sequence of objects with attributes:
            r_B : (3,) position in body frame [m]
            a_B : (3,) thrust-direction unit vector in body frame
            T   : scalar max thrust [N]
            Isp : scalar specific impulse [s]

    Returns
    -------
    B : (6, N) ndarray
        Wrench map, columns are [ τ_i ; f_i ] for each thruster i.
    Ts : (N,) ndarray
        Max thrust per thruster [N].
    Isps : (N,) ndarray
        Specific impulses for each thruster [s].
    """
    thrusters = list(thrusters)
    if not thrusters:
        return np.zeros((6, 0)), np.zeros(0), np.zeros(0)

    cols = []
    Ts = []
    Isps = []

    for th in thrusters:
        r = np.asarray(th.r_B, dtype=float).reshape(3)
        a = np.asarray(th.a_B, dtype=float).reshape(3)
        n = la.norm(a)
        a = a / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
        T = float(th.T)
        f = T * a
        tau = np.cross(r, f)
        cols.append(np.hstack((tau, f)))
        Ts.append(T)
        Isps.append(float(getattr(th, "Isp", 0.0)))

    B = np.column_stack(cols)
    Ts = np.array(Ts, dtype=float)
    Isps = np.array(Isps, dtype=float)
    return B, Ts, Isps


def bnls_solve(
    B: np.ndarray,
    u: np.ndarray,
    delta_prev: Optional[np.ndarray] = None,
    w_tau: float = 10.0,
    w_for: float = 40.0,
    lam: float = 3e-4,
    lb: float = 0.0,
    ub: float = 1.0,
    max_iter: int = 8,
) -> np.ndarray:
    """
    Bounded, weighted, regularised least-squares solver for thruster allocation.

    Solve

        min_δ  ‖ W (B δ − u) ‖² + λ ‖ δ − δ_prev ‖²
        s.t.   lb ≤ δ ≤ ub

    where W = diag(wτ, wτ, wτ, wf, wf, wf) and u = [τ_des; f_des].

    This is a modestly robust allocator for small-ish N; it uses a
    simple clip + active-set refinement loop.

    Parameters
    ----------
    B : (6, N) ndarray
        Wrench map (from build_wrench_map).
    u : (6,) ndarray
        Desired wrench [τ_des; f_des].
    delta_prev : (N,) ndarray, optional
        Previous solution (for temporal smoothing). If None, zero vector
        is used.
    w_tau : float, default 10.0
        Rotational weighting in W (for first three rows).
    w_for : float, default 40.0
        Translational weighting in W (for last three rows).
    lam : float, default 3e-4
        Regularisation weight on (δ − δ_prev).
    lb, ub : float, defaults (0.0, 1.0)
        Lower and upper bounds for each δ_i.
    max_iter : int, default 8
        Number of corrective refinement sweeps.

    Returns
    -------
    delta : (N,) ndarray
        Bounded allocation vector.
    """
    B = np.asarray(B, float)
    u = np.asarray(u, float).reshape(6)
    m, n = B.shape
    if m != 6:
        raise ValueError(f"B must be 6×N, got {B.shape}")

    if delta_prev is None or len(delta_prev) != n:
        delta_prev = np.zeros(n, dtype=float)
    else:
        delta_prev = np.asarray(delta_prev, float).reshape(n)

    # Weight matrix W (diagonal)
    W = np.diag([w_tau, w_tau, w_tau, w_for, w_for, w_for])
    BtW = B.T @ W
    BtWB = BtW @ B + lam * np.eye(n)
    BtWu = BtW @ u + lam * delta_prev

    # Initial unconstrained solution
    delta = la.solve(BtWB, BtWu)
    delta = np.clip(delta, lb, ub)

    # Active-set style refinement
    for _ in range(max_iter):
        low = delta <= lb + 1e-9
        high = delta >= ub - 1e-9
        free = ~(low | high)
        if not np.any(free):
            break

        Bf = B[:, free]
        nf = np.count_nonzero(free)
        # Regularised normal equations for free subset
        K = la.pinv(Bf.T @ W @ Bf + lam * np.eye(nf))
        u_eff = u - B[:, low] @ np.full(np.count_nonzero(low), lb) \
                  - B[:, high] @ np.full(np.count_nonzero(high), ub)
        rhs = Bf.T @ W @ u_eff + lam * delta_prev[free]
        delta_f = K @ rhs

        delta = delta.copy()
        delta[free] = delta_f
        delta = np.clip(delta, lb, ub)

    return delta

