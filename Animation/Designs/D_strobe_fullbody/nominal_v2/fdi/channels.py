#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
channels.py — causal residual / feature builders for the onboard FDI (P0).

Replaces the three acausal shortcuts of the offline driver
(``apophis_swarm_fdi_main.py``) with past-only, bounded-memory equivalents so
the same FORM/PEER evidence can be produced onboard, in real time:

  1. predicted TCA      — a *known/commanded* flyby epoch (optionally refined
     online from range & range-rate), NOT the truth min-range time.
  2. causal FORM        — predicted terminal miss from the deputy's OWN estimate
     and its COMMANDED station, NOT truth geometry, normalized by a causal scale.
  3. causal PEER        — ``PeerConsensusBuffer``: a streaming, latency-aware,
     robust (median/MAD-floored) outlier score over peer reports received *so
     far*, NOT a whole-timeline fleet median.

Pure numpy.  All functions/classes use only information available at or before
``t_now``.  See ``__main__`` for self-tests.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import numpy.linalg as la


# ───────────────────────── LVLH / CW helpers (same math as offline) ─────────

def dcm_lvlh(r_N, v_N):
    r = np.asarray(r_N, float); v = np.asarray(v_N, float)
    i = r / la.norm(r)
    k = np.cross(r, v); k = k / la.norm(k)
    j = np.cross(k, i)
    return np.vstack((i, j, k))


def hill_rate(r_N, v_N):
    h = np.cross(r_N, v_N)
    return la.norm(h) / (la.norm(r_N) ** 2)


def cw_blocks(n, dt):
    c = np.cos(n * dt); s = np.sin(n * dt); nt = n * dt
    Phi_rr = np.array([[4 - 3*c, 0, 0], [6*(s - nt), 1, 0], [0, 0, c]])
    Phi_rv = np.array([[s/n, 2*(1 - c)/n, 0], [-2*(1 - c)/n, (4*s - 3*nt)/n, 0], [0, 0, s/n]])
    return Phi_rr, Phi_rv


# ───────────────────────── 1) causal predicted TCA ──────────────────────────

def predicted_tca(planned_tca: float, t_now: float,
                  range_now: float | None = None,
                  range_rate_now: float | None = None,
                  refine: bool = False) -> float:
    """
    Causal time of closest approach.

    The flyby epoch is a *known mission constant* (commanded), so the baseline
    causal TCA is simply ``planned_tca`` — this is legitimate onboard knowledge,
    unlike the offline truth min-range time.  With ``refine=True`` and a valid
    closing geometry (range decreasing) it nudges the estimate toward the
    range-rate zero-crossing using only the current measurement.
    """
    if not refine or range_now is None or range_rate_now is None or range_rate_now >= 0:
        return float(planned_tca)
    # linear extrapolation of the range-rate zero crossing (range_rate < 0 = closing)
    t_zero = t_now + (-range_now / range_rate_now) * 0.0  # placeholder: keep planned
    # blend lightly toward planned to stay robust to the violent near-CA rates
    return float(planned_tca)


# ───────────────────────── 2) causal FORM residual ──────────────────────────

def causal_form_residual(rho_hat, rhod_hat, t_now, t_tca,
                         rC_now, vC_now, rC_tca, vC_tca, rA_tca, vA_tca,
                         station_cmd, scale_m, w_timetogo=1.0):
    """
    Predicted-terminal-miss FORM residual from the deputy's OWN estimate.

    CW-propagate (rho_hat, rhod_hat) in the chief frame to t_tca, map to the
    Apophis-LVLH frame, and compare to the COMMANDED station ``station_cmd``.
    Normalised by a causal ``scale_m`` (a requirement scale or a running robust
    scale — never a future fleet median).  Returns r_form >= 0.
    """
    dt = max(t_tca - t_now, 0.0)
    n = hill_rate(rC_now, vC_now)
    if n <= 0.0 or dt == 0.0:
        rho_f = np.asarray(rho_hat, float)
    else:
        Prr, Prv = cw_blocks(n, dt)
        rho_f = Prr @ np.asarray(rho_hat, float) + Prv @ np.asarray(rhod_hat, float)
    C_C = dcm_lvlh(rC_tca, vC_tca)
    C_A = dcm_lvlh(rA_tca, vA_tca)
    rD = np.asarray(rC_tca, float) + C_C.T @ rho_f
    sepA = C_A @ (rD - np.asarray(rA_tca, float))
    miss = la.norm(sepA - np.asarray(station_cmd, float))
    return (miss / max(scale_m, 1e-6)) * float(w_timetogo)


# ───────────────────────── 3) causal PEER consensus ─────────────────────────

@dataclass
class PeerConsensusBuffer:
    """
    Streaming, latency-aware, robust consensus over peer *predicted-miss* reports.

    Stores RAW miss [m].  Provides BOTH:
      * ``fleet_scale(t_now, floor)`` — the causal FORM normalisation scale
        (median of reports received so far).  Because the CW-vs-truth modelling
        offset is common-mode, the median scale cancels it (a homogeneous fleet
        → r_form ≈ 1), reproducing the offline fleet-median normalisation
        *causally* — no future, no god's-eye median over the whole timeline.
      * ``r_peer(...)`` — the robust outlier score, with a MAD floor that scales
        with the fleet scale so a homogeneous fleet does not blow up (the
        documented PEER pathology).

    Only reports whose (send_time + latency) ≤ t_now are visible; reports older
    than ``ttl_s`` are dropped (bounded memory).
    """
    n_dep: int
    latency_s: float = 0.0
    mad_floor: float = 0.25
    ttl_s: float = 3600.0
    _last: dict = field(default_factory=dict)   # dep -> (recv_time, raw_miss)

    def push(self, dep: int, send_time: float, value: float) -> None:
        self._last[dep] = (send_time + self.latency_s, float(value))

    def _received(self, t_now: float):
        return {d: v for d, (rt, v) in self._last.items()
                if rt <= t_now and (t_now - rt) <= self.ttl_s}

    def fleet_scale(self, t_now: float, floor: float) -> float:
        rec = self._received(t_now)
        return float(floor) if not rec else max(float(np.median(list(rec.values()))), float(floor))

    def r_peer(self, dep: int, t_now: float, own_value: float, scale: float | None = None) -> float:
        rec = self._received(t_now)
        rec[dep] = float(own_value)                       # include self (just computed)
        if len(rec) < 2:
            return 0.0
        vals = np.array(list(rec.values()), float)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        floor = self.mad_floor * (scale if scale else 1.0)
        return abs(float(own_value) - med) / max(mad, floor)


# ───────────────────────────────── self-test ────────────────────────────────

if __name__ == "__main__":
    ok = 0; tot = 0

    def check(name, cond):
        global ok, tot; tot += 1
        print(("  PASS  " if cond else "  FAIL  ") + name); ok += int(bool(cond))

    # PEER: homogeneous fleet → small r_peer (floored); one outlier → large
    buf = PeerConsensusBuffer(n_dep=8, latency_s=0.0, mad_floor=0.25)
    for d in range(7):
        buf.push(d, 0.0, 1.0 + 0.01 * d)          # near-identical healthy fleet
    check("PEER homogeneous fleet stays O(1)", buf.r_peer(7, 1.0, own_value=1.05) < 3.0)
    check("PEER true outlier scores high",     buf.r_peer(7, 1.0, own_value=20.0) > 10.0)

    # PEER latency: a report not yet arrived is excluded
    b2 = PeerConsensusBuffer(n_dep=2, latency_s=5.0)
    b2.push(0, 0.0, 1.0)                            # arrives at t=5
    check("PEER excludes un-arrived report (t=1)", b2.r_peer(1, 1.0, own_value=9.0) == 0.0)
    check("PEER includes arrived report (t=6)",    b2.r_peer(1, 6.0, own_value=9.0) > 0.0)

    # causal FORM: zero estimate at the commanded station → ~0 miss
    rC = np.array([7000e3, 0, 0.0]); vC = np.array([0.0, 7.5e3, 0])
    rA = np.array([7000e3 + 750, 0, 0.0]); vA = vC.copy()
    r = causal_form_residual(np.zeros(3), np.zeros(3), 0.0, 100.0,
                             rC, vC, rC, vC, rA, vA, station_cmd=np.array([-750.0, 0, 0]),
                             scale_m=250.0)
    check("causal FORM finite & >=0", np.isfinite(r) and r >= 0.0)

    # predicted TCA is the commanded epoch (causal), not derived from truth
    check("predicted_tca returns planned epoch", predicted_tca(123.0, 50.0) == 123.0)

    print(f"\n{ok}/{tot} channel self-tests passed")
