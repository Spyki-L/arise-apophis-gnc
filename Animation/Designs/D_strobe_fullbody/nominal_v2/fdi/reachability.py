#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reachability.py — FORM as a 4D reachable tube (F3).

Replaces the scalar predicted-miss FORM residual (and the P0 consensus-scale
placeholder) with a **4D reachable tube**: the time-parameterised envelope, in
(t, x, y, z), of where a deputy can be along its trajectory to its slot in the
optimised formation at closest approach. The tube cross-section at look-ahead
time dt is the Minkowski sum

    R(dt) = c(dt)  ⊕  E_know(dt)  ⊕  E_ctrl(dt)
      c(dt)      = CW-propagated nominal relative position (no control)
      E_know(dt) = k·σ knowledge ellipsoid  = CW-propagated nav covariance  (inflates if nav degrades)
      E_ctrl(dt) = control-reachable set     = image of the remaining-ΔV box  (shrinks if actuators degrade)

A **FORM fault** is, by definition, "my CA slot has left the tube": the slot's
Mahalanobis distance to the terminal cross-section exceeds 1. The same object
yields the **recoverability frontier** (the latest moment a reconfiguration can
still place the deputy on a useful slot) — i.e. the deadline τ* the recovery
value function needs. Causal, onboard-cheap (CW matrices + 3×3 ellipsoid math).

Pure numpy. Mission-agnostic: mean motion, authority, covariance, and the slot
all come from the caller / mission profile.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
import numpy.linalg as la


# ── Clohessy–Wiltshire state-transition blocks ──────────────────────────────

def cw_blocks_full(n, dt):
    c = np.cos(n*dt); s = np.sin(n*dt); nt = n*dt
    Prr = np.array([[4-3*c, 0, 0], [6*(s-nt), 1, 0], [0, 0, c]], float)
    Prv = np.array([[s/n, 2*(1-c)/n, 0], [-2*(1-c)/n, (4*s-3*nt)/n, 0], [0, 0, s/n]], float)
    Pvr = np.array([[3*n*s, 0, 0], [6*n*(c-1), 0, 0], [0, 0, -n*s]], float)
    Pvv = np.array([[c, 2*s, 0], [-2*s, 4*c-3, 0], [0, 0, c]], float)
    return Prr, Prv, Pvr, Pvv


def propagate_pos(rho, rhod, n, dt):
    Prr, Prv, _, _ = cw_blocks_full(n, dt)
    return Prr @ np.asarray(rho, float) + Prv @ np.asarray(rhod, float)


def knowledge_shape(P0, n, dt, k_sigma=3.0):
    """k·σ terminal-position knowledge ellipsoid (shape matrix, E = {x: xᵀS⁻¹x ≤ 1})."""
    Prr, Prv, Pvr, Pvv = cw_blocks_full(n, dt)
    Phi = np.block([[Prr, Prv], [Pvr, Pvv]])
    Pt = Phi @ np.asarray(P0, float) @ Phi.T
    return (k_sigma**2) * Pt[0:3, 0:3]


def control_shape(dv_axis, n, dt):
    """Control-reachable position set for a single impulsive burn of per-axis
    authority dv_axis applied now over the remaining dt (conservative onboard proxy)."""
    _, Prv, _, _ = cw_blocks_full(n, dt)
    return Prv @ np.diag(np.asarray(dv_axis, float)**2) @ Prv.T


def minkowski_outer(A, B):
    """Trace-optimal outer ellipsoid of the Minkowski sum E(0,A) ⊕ E(0,B)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    tA, tB = np.trace(A), np.trace(B)
    if tA <= 1e-12: return B.copy()
    if tB <= 1e-12: return A.copy()
    p = np.sqrt(tA / tB)                 # trace-optimal parameter (minimises tr of the result)
    return (1.0 + 1.0/p)*A + (1.0 + p)*B


# ── the 4D FORM tube ─────────────────────────────────────────────────────────

@dataclass
class FormTube:
    n: float                 # chief mean motion [rad/s]
    rho: np.ndarray          # current relative position estimate [m]
    rhod: np.ndarray         # current relative velocity estimate [m/s]
    P0: np.ndarray           # current 6×6 covariance
    dv_axis: np.ndarray      # per-axis remaining control authority [m/s]
    t_go: float              # time-to-go to CA [s]
    slot: np.ndarray         # required formation slot position at CA [m]
    k_sigma: float = 3.0
    n_grid: int = 25

    def cross_section(self, dt):
        c = propagate_pos(self.rho, self.rhod, self.n, dt)
        Sk = knowledge_shape(self.P0, self.n, dt, self.k_sigma)
        Sc = control_shape(self.dv_axis, self.n, dt)
        return c, minkowski_outer(Sk, Sc)

    def shape_4d(self):
        """The propagating 4D shape: list of (look-ahead dt, center, shape) over [0, t_go]."""
        return [(dt, *self.cross_section(dt)) for dt in np.linspace(0.0, self.t_go, self.n_grid)]

    def slot_mahalanobis(self, dt=None):
        if dt is None: dt = self.t_go
        c, S = self.cross_section(dt)
        d = np.asarray(self.slot, float) - c
        return float(np.sqrt(d @ la.solve(S + 1e-9*np.eye(3), d)))

    def is_reachable(self, dt=None):
        return self.slot_mahalanobis(dt) <= 1.0

    def r_form(self):
        """FORM residual = slot Mahalanobis to the terminal tube (1 = boundary).
        Reachable → <1 (below nu, no CUSUM accumulation); unreachable → >1 (FORM fault)."""
        return self.slot_mahalanobis(self.t_go)

    # ── deformations (return a modified copy) ────────────────────────────────
    def with_authority(self, dv_axis):
        t = copy.copy(self); t.dv_axis = np.asarray(dv_axis, float); return t       # actuation change
    def with_nav_cov(self, P0):
        t = copy.copy(self); t.P0 = np.asarray(P0, float); return t                 # nav change
    def with_slot(self, slot):
        t = copy.copy(self); t.slot = np.asarray(slot, float); return t             # spin / re-task change
    def with_t_go(self, t_go):
        t = copy.copy(self); t.t_go = float(t_go); return t                         # CA-geometry change


def recoverability_frontier(base, g_grid, tgo_grid):
    """For each surviving-authority fraction g, the latest onset (smallest time-to-go) at which
    the slot is still reachable. Returns {g: latest_reachable_t_go or None}."""
    out = {}
    for g in g_grid:
        latest = None
        for tgo in sorted(tgo_grid):              # ascending: smallest reachable t_go = latest you can act
            if base.with_authority(np.asarray(base.dv_axis, float)*g).with_t_go(tgo).is_reachable():
                latest = float(tgo); break
        out[float(g)] = latest
    return out
