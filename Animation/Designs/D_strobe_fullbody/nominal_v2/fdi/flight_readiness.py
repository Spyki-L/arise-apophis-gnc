#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flight_readiness.py — F9 flight-readiness verification toolkit. Pure numpy.

Provides the quantitative evidence behind the safety case (SAFETY_CASE.md):
  * determinism + bounded memory of the flight path
  * worst-case execution time (WCET) timing + a compute-budget table
  * CBF forward-invariance (a safety filter that keeps h>=0, swept over ICs)
  * RTA soundness (no unsafe AC proposal is ever executed)
  * fixed-point / float32 consistency of the VSC-critical math
  * graceful degradation when the AC fails
"""
from __future__ import annotations
import time
import numpy as np
import numpy.linalg as la


# ── timing / determinism / memory ───────────────────────────────────────────

def time_call(fn, *a, n=40, **kw):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(*a, **kw); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.max(ts))         # median, worst observed


def arrays_equal(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return a.shape == b.shape and np.array_equal(a, b)


def fdi_state_footprint(core):
    """Count the mutable state of an FDI core; must be O(1) (bounded) regardless of run length."""
    scalars = sum(isinstance(getattr(core, n, None), float)
                  for n in ("S_nav", "S_att", "S_comm", "S_prop", "S_form", "S_peer",
                            "r_nav", "r_att", "r_comm", "r_prop", "r_form", "r_peer"))
    latch = len(getattr(core, "episode_channels", set()))
    return dict(scalars=scalars, latch=latch)


# ── CBF forward-invariance: safety filter + sweep ────────────────────────────

def cbf_velocity_filter(r_ij, v_des, d_safe, gamma):
    """Minimal-norm correction to the desired relative velocity so the discrete CBF condition
    2 r.v + gamma(|r|^2 - d^2) >= 0 holds (keeps the pair outside the keep-out)."""
    h = float(r_ij @ r_ij) - d_safe**2
    lhs = 2.0 * float(r_ij @ v_des) + gamma * h
    if lhs >= 0.0:
        return v_des
    lam = -lhs / (2.0 * float(r_ij @ r_ij) + 1e-9)
    return v_des + lam * r_ij


def verify_cbf_invariance(n_trials=400, d_safe=50., gamma=0.2, dt=1.0, steps=400, seed=0):
    """Sweep random initial separations with a nominal control that pulls the pair together;
    with the CBF filter the keep-out is NEVER breached. Returns (min_h_filtered, min_h_unfiltered)."""
    rng = np.random.default_rng(seed); min_h_f = np.inf; min_h_u = np.inf
    for _ in range(n_trials):
        r = rng.normal(0, 1, 3); r = r / la.norm(r) * rng.uniform(d_safe*1.5, d_safe*5)
        ru = r.copy()
        for _ in range(steps):
            v = cbf_velocity_filter(r, -0.08 * r, d_safe, gamma)   # nominal pulls them together
            r = r + v * dt
            min_h_f = min(min_h_f, float(r @ r) - d_safe**2)
            ru = ru + (-0.08 * ru) * dt                            # unfiltered: collapses
            min_h_u = min(min_h_u, float(ru @ ru) - d_safe**2)
    return min_h_f, min_h_u


# ── RTA soundness: no unsafe proposal ever executed ──────────────────────────

def rta_soundness(n_trials=2000, d_safe=50., seed=0):
    """Throw random AC proposals (a mix of safe and unsafe) at the simplex; assert no unsafe
    action is executed and every accepted action satisfies the safety predicate."""
    from rta import RTAExecutive, Controller
    rng = np.random.default_rng(seed)
    nominal = np.array([[100., 0, 0], [-100., 0, 0], [0, 100., 0], [0, -100., 0]])
    reach = np.full(4, 60.)

    def safe(stations, ctx):
        for d in range(4):
            if la.norm(stations[d] - nominal[d]) > reach[d] + 1e-6:
                return False
        for a in range(4):
            for b in range(a+1, 4):
                if la.norm(stations[a] - stations[b]) < d_safe:
                    return False
        return True

    rta = RTAExecutive(safety_check=safe, vsc_fallback=lambda c: nominal.copy())
    executed_unsafe = 0; accepted_unsafe_prop = 0
    for _ in range(n_trials):
        if rng.random() < 0.5:
            prop = nominal + rng.normal(0, 12, nominal.shape)      # small -> safe
        else:
            prop = nominal + rng.normal(0, 90, nominal.shape)      # large -> often unsafe
        d = rta.decide({}, prop)
        if not safe(d.action, {}):
            executed_unsafe += 1
        if d.source is Controller.AC and not safe(prop, {}):
            accepted_unsafe_prop += 1
    return dict(executed_unsafe=executed_unsafe, accepted_unsafe_prop=accepted_unsafe_prop,
                ac=rta.n_ac, vsc=rta.n_vsc, trials=n_trials)


# ── fixed-point / float32 consistency of VSC-critical math ───────────────────

def fixed_point_consistency(n=2000, d_safe=50., seed=0):
    """The VSC-critical decisions (CBF safe/unsafe verdict) must be identical in float32 and
    float64 -> the critical path does not require double precision (fixed-point feasible)."""
    rng = np.random.default_rng(seed); mism = 0
    for _ in range(n):
        r = rng.normal(0, 100, 3); v = rng.normal(0, 2, 3)
        h64 = np.float64(r @ r) - np.float64(d_safe)**2
        m64 = 2*np.float64(r @ v) + 0.1*h64
        r32 = r.astype(np.float32); v32 = v.astype(np.float32)
        h32 = np.float32(r32 @ r32) - np.float32(d_safe)**2
        m32 = np.float32(2)*np.float32(r32 @ v32) + np.float32(0.1)*h32
        if (m64 >= 0) != (m32 >= 0):
            mism += 1
    return mism


# ── compute-budget table ─────────────────────────────────────────────────────

def compute_budget(measurements, cadence_s, target_slowdown):
    """measurements: {layer: (median_s, worst_s)}. Returns per-layer onboard WCET estimate
    (worst_s * target_slowdown) vs the cadence budget, with pass/fail margin."""
    rows = []
    for layer, (med, worst) in measurements.items():
        onboard = worst * target_slowdown.get(layer, 1.0)
        budget = cadence_s.get(layer, 0.1)
        rows.append(dict(layer=layer, sandbox_us=med*1e6, onboard_ms=onboard*1e3,
                         budget_ms=budget*1e3, ok=onboard <= budget))
    return rows
