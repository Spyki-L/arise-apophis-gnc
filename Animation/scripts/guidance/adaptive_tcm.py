#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adaptive_tcm.py
===============

Integrated **adaptive trajectory-correction-maneuver (TCM)** scheduler for the
ARISE deputy fleet.  The goal is to hit each deputy's terminal station to the
0.05 m close-approach tolerance while spending less Δv than the fixed-fraction
(0.25/0.55/0.80/0.93 TOF) schedule that the current guidance uses.

It implements the two layers we designed:

Layer 1 — leverage-weighted node re-timing
------------------------------------------
The Δv cost of fixing a given terminal-position miss scales as ``1 / L(t)``
where ``L(t) = ||Φ_rv(t_f, t)||`` is the terminal-miss leverage (metres of
terminal motion per m/s of correction).  ``L`` collapses by ~200× across a
deputy arc, so *when* a correction is applied dominates *how many* are applied.
``optimize_node_times`` places the correction epochs to minimise expected total
Δv (front-loading toward high leverage), subject to a knowledge gate (no burn
before the navigation filter has converged) and a minimum spacing.

Layer 2 — covariance-aware deadband execution
---------------------------------------------
At each node the scheduler uses the navigation estimate ``x̂`` and covariance
``P`` (from the Hill EKF) to predict the terminal miss ``m̂(t_f)`` and its
dispersion ``σ_term``.  It then

  * **skips** the burn when ``|m̂| + k·σ_term ≤ tol`` (deadband at the
    requirement — no Δv spent polishing inside tolerance), else
  * **sizes** ``Δv = −Φ_rv(t_f, t)^{-1} · m̂`` to null the *estimated* miss
    (not a worst-case allocation), and carries the maneuver-execution
    covariance forward so the next node accounts for it.

Frames & conventions (match the rest of the pipeline)
-----------------------------------------------------
* Relative state is deputy − chief.  The linearised state-transition matrix is
  integrated in the **chief-centred ECI** frame (two-body gravity gradient
  about the chief reference); this avoids the constant-mean-motion assumption
  of HCW, which is invalid over the eccentric GEO→flyby transfer.
* The Hill EKF state ``x = [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N]`` (chief LVLH) is
  converted to/from the ECI relative state with :func:`hill_to_eci_rel` /
  :func:`hill_cov_to_eci_rel`, using the same ``dcm_lvlh`` / ``hill_rate`` as
  ``GNC.Navigation.hill_ekf``.
* Absolute terminal miss is **propagator-agnostic**: pass ``terminal_miss_fn``
  backed by your Basilisk high-fidelity propagator for the authoritative miss;
  a first-order linear fallback (``LinearTerminalModel``) is built in so the
  module runs and self-tests standalone.

This module is plain NumPy/SciPy (SciPy optional) — no Basilisk import — so it
can be unit-tested off the flight machine.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, List

import numpy as np
import numpy.linalg as la

MU_EARTH = 3.986004418e14  # [m^3/s^2]

# --------------------------------------------------------------------------
# Optional: pull dcm_lvlh / hill_rate from the project's hill_ekf so the frame
# definitions are guaranteed identical.  Fall back to local copies if the GNC
# package is not importable (e.g. running this file in isolation).
# --------------------------------------------------------------------------
try:  # pragma: no cover - exercised on the flight machine
    _here = os.path.dirname(os.path.abspath(__file__))
    for _p in (os.path.join(_here, "..", ".."), os.path.join(_here, "..", "..", "..")):
        _p = os.path.abspath(_p)
        if os.path.isdir(os.path.join(_p, "GNC")) and _p not in sys.path:
            sys.path.insert(0, _p)
    from GNC.Navigation.hill_ekf import dcm_lvlh, hill_rate  # type: ignore
except Exception:  # local fallbacks (identical math)
    def dcm_lvlh(r: np.ndarray, v: np.ndarray) -> np.ndarray:
        r = np.asarray(r, float).reshape(3); v = np.asarray(v, float).reshape(3)
        ih = r / la.norm(r)
        k = np.cross(r, v); k /= la.norm(k)
        j = np.cross(k, ih)
        return np.vstack((ih, j, k))

    def hill_rate(r: np.ndarray, v: np.ndarray) -> float:
        h = np.cross(r, v)
        return float(la.norm(h) / (la.norm(r) ** 2 + 1e-12))


# ══════════════════════════════════════════════════════════════════════════
# Linearised relative-motion state-transition matrix  Φ(t_f, t)
# ══════════════════════════════════════════════════════════════════════════

def gravity_gradient(r_c_N: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    """Two-body gravity-gradient tensor G = ∂a/∂r about the chief position."""
    r_c_N = np.asarray(r_c_N, float).reshape(3)
    rn = la.norm(r_c_N)
    rh = r_c_N / rn
    return -mu / rn ** 3 * (np.eye(3) - 3.0 * np.outer(rh, rh))


class RelativeSTM:
    """
    State-transition matrix Φ(t_f, t) for the linearised deputy–chief relative
    motion, integrated **backward** from the terminal epoch along the chief
    reference trajectory.

    Dynamics (chief-centred ECI):  d/dt [δr; δv] = A(t) [δr; δv],
        A(t) = [[0, I], [G(t), 0]],   G(t) = gravity_gradient(r_c(t)).

    The backward integration uses  dΦ/dt = -Φ A(t),  Φ(t_f) = I,  so a single
    sweep yields Φ(t_f, t) for every t on the grid.
    """

    def __init__(
        self,
        t_ref: np.ndarray,
        rC_N: np.ndarray,
        vC_N: Optional[np.ndarray] = None,
        mu: float = MU_EARTH,
        n_grid: int = 8000,
        t_f: Optional[float] = None,
    ):
        self.t_ref = np.asarray(t_ref, float).reshape(-1)
        self.rC_N = np.asarray(rC_N, float).reshape(-1, 3)
        self.vC_N = None if vC_N is None else np.asarray(vC_N, float).reshape(-1, 3)
        self.mu = float(mu)
        self.t_f = float(self.t_ref[-1] if t_f is None else t_f)
        self.t0 = float(self.t_ref[0])
        self._build(n_grid)

    # -- chief position interpolation --------------------------------------
    def _r_c(self, t: float) -> np.ndarray:
        t = min(max(t, self.t_ref[0]), self.t_ref[-1])
        j = int(np.searchsorted(self.t_ref, t)) - 1
        j = min(max(j, 0), len(self.t_ref) - 2)
        f = (t - self.t_ref[j]) / (self.t_ref[j + 1] - self.t_ref[j] + 1e-15)
        return (1 - f) * self.rC_N[j] + f * self.rC_N[j + 1]

    def _A(self, t: float) -> np.ndarray:
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 0:3] = gravity_gradient(self._r_c(t), self.mu)
        return A

    def _build(self, n_grid: int) -> None:
        tg = np.linspace(self.t_f, self.t0, int(n_grid))
        dt = tg[1] - tg[0]  # negative
        Phi = np.eye(6)
        store = np.empty((len(tg), 6, 6))

        def deriv(P, t):
            return -P @ self._A(t)

        for i, t in enumerate(tg):
            store[i] = Phi
            if i < len(tg) - 1:
                k1 = deriv(Phi, t)
                k2 = deriv(Phi + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = deriv(Phi + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = deriv(Phi + dt * k3, t + dt)
                Phi = Phi + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        # store ascending in time for interpolation
        self._tg = tg[::-1].copy()
        self._Phi = store[::-1].copy()

    # -- public API --------------------------------------------------------
    def Phi(self, t: float) -> np.ndarray:
        """Φ(t_f, t) by linear interpolation of the stored flow."""
        t = min(max(t, self.t0), self.t_f)
        j = int(np.searchsorted(self._tg, t)) - 1
        j = min(max(j, 0), len(self._tg) - 2)
        f = (t - self._tg[j]) / (self._tg[j + 1] - self._tg[j] + 1e-15)
        return (1 - f) * self._Phi[j] + f * self._Phi[j + 1]

    def Phi_rv(self, t: float) -> np.ndarray:
        """Top-right 3×3 block: ∂r(t_f)/∂Δv(t)."""
        return self.Phi(t)[0:3, 3:6]

    def leverage(self, t: float) -> float:
        """L(t) = ||Φ_rv(t_f, t)||₂  [metres terminal per m/s correction]."""
        return float(la.norm(self.Phi_rv(t), 2))


# ══════════════════════════════════════════════════════════════════════════
# Frame conversions:  Hill (LVLH) EKF state  ->  chief-centred ECI relative
# ══════════════════════════════════════════════════════════════════════════

def hill_to_eci_rel(
    rho: np.ndarray, rho_dot: np.ndarray, r_c_N: np.ndarray, v_c_N: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse of the eci->Hill conversion used by the EKF:
        v_rel_N = C_HNᵀ ρ̇ + ω_N × δr_N,   δr_N = C_HNᵀ ρ,  ω_N = C_HNᵀ[0,0,n].
    """
    C_HN = dcm_lvlh(r_c_N, v_c_N)
    n = hill_rate(r_c_N, v_c_N)
    dr_N = C_HN.T @ np.asarray(rho, float).reshape(3)
    omega_N = C_HN.T @ np.array([0.0, 0.0, n])
    dv_N = C_HN.T @ np.asarray(rho_dot, float).reshape(3) + np.cross(omega_N, dr_N)
    return dr_N, dv_N


def _hill_to_eci_transform(r_c_N: np.ndarray, v_c_N: np.ndarray) -> np.ndarray:
    """6×6 Jacobian T mapping Hill state [ρ, ρ̇] to ECI relative [δr, δv]."""
    C_HN = dcm_lvlh(r_c_N, v_c_N)
    n = hill_rate(r_c_N, v_c_N)
    omega_N = C_HN.T @ np.array([0.0, 0.0, n])
    Wx = np.array([[0, -omega_N[2], omega_N[1]],
                   [omega_N[2], 0, -omega_N[0]],
                   [-omega_N[1], omega_N[0], 0]])
    T = np.zeros((6, 6))
    T[0:3, 0:3] = C_HN.T
    T[3:6, 0:3] = Wx @ C_HN.T
    T[3:6, 3:6] = C_HN.T
    return T


def hill_cov_to_eci_rel(P_hill: np.ndarray, r_c_N: np.ndarray, v_c_N: np.ndarray) -> np.ndarray:
    """Transform a 6×6 Hill covariance to the chief-centred ECI relative frame."""
    T = _hill_to_eci_transform(r_c_N, v_c_N)
    return T @ np.asarray(P_hill, float).reshape(6, 6) @ T.T


# ══════════════════════════════════════════════════════════════════════════
# Terminal-miss model (propagator-agnostic, with a linear fallback)
# ══════════════════════════════════════════════════════════════════════════

class LinearTerminalModel:
    """
    First-order terminal-miss model using the relative STM.  Predicts the
    terminal relative *position* of a deputy from its current relative state,
    and the miss against a fixed terminal target r*_N(t_f).

    Swap this out for a high-fidelity propagator on the flight machine by
    passing your own ``terminal_miss_fn(y_rel_N, t) -> (miss_N, J)`` to the
    scheduler, where ``J = ∂miss/∂Δv`` (use ``stm.Phi_rv(t)`` if unsure).
    """

    def __init__(self, stm: RelativeSTM, target_rel_pos_N: np.ndarray):
        self.stm = stm
        self.target = np.asarray(target_rel_pos_N, float).reshape(3)

    def predict_state(self, y_rel_N: np.ndarray, t: float) -> np.ndarray:
        """Propagate a 6-state relative deviation to t_f via Φ(t_f, t)."""
        return self.stm.Phi(t) @ np.asarray(y_rel_N, float).reshape(6)

    def miss(self, y_rel_N: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (terminal position miss [3], sensitivity ∂miss/∂Δv = Φ_rv)."""
        yf = self.predict_state(y_rel_N, t)
        miss_N = yf[0:3] - self.target
        return miss_N, self.stm.Phi_rv(t)


# ══════════════════════════════════════════════════════════════════════════
# Correction sizing, covariance propagation, deadband decision
# ══════════════════════════════════════════════════════════════════════════

def size_correction(Phi_rv: np.ndarray, miss_N: np.ndarray) -> np.ndarray:
    """Δv that nulls the predicted terminal miss: solve Φ_rv Δv = -miss."""
    Phi_rv = np.asarray(Phi_rv, float).reshape(3, 3)
    return -la.solve(Phi_rv, np.asarray(miss_N, float).reshape(3))


def terminal_pos_cov(Phi: np.ndarray, P_rel_N: np.ndarray) -> np.ndarray:
    """Terminal position covariance: top-left 3×3 of Φ P Φᵀ."""
    M = np.asarray(Phi, float) @ np.asarray(P_rel_N, float) @ np.asarray(Phi, float).T
    return M[0:3, 0:3]


def terminal_sigma(Phi: np.ndarray, P_rel_N: np.ndarray) -> float:
    """Scalar 1σ terminal-miss magnitude = sqrt of largest eigenvalue of the
    terminal position covariance (worst-case direction)."""
    C = terminal_pos_cov(Phi, P_rel_N)
    w = np.linalg.eigvalsh(0.5 * (C + C.T))
    return float(np.sqrt(max(w[-1], 0.0)))


def deadband_skip(miss_N: np.ndarray, sigma_term: float, tol: float, k: float) -> bool:
    """Skip the correction if |miss| + k·σ_term is within tolerance."""
    return bool(la.norm(miss_N) + k * sigma_term <= tol)


# ══════════════════════════════════════════════════════════════════════════
# Configuration & the scheduler
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AdaptiveTCMConfig:
    tol_m: float = 0.05            # terminal close-approach tolerance [m]
    k_sigma: float = 3.0           # deadband confidence multiplier (3σ)
    max_nodes: int = 4             # max number of correction opportunities
    earliest_frac: float = 0.08    # no TCM before this fraction of TOF (nav gate)
    latest_frac: float = 0.97      # no TCM after this fraction of TOF
    late_node_min_frac: float = 0.90  # REQUIRE the last node at/after this frac
    min_dt_s: float = 600.0        # minimum spacing between corrections [s]
    # maneuver-execution error model (applied to truth in MC; carried in cov)
    exec_err_frac: float = 0.01    # 1σ proportional magnitude error (1%)
    exec_err_floor_mps: float = 1.0e-4   # 1σ fixed pointing/quantization floor


@dataclass
class TCMEvent:
    t: float
    frac: float
    dv_N: np.ndarray
    dv_mag: float
    predicted_miss_m: float
    sigma_term_m: float
    skipped: bool


class AdaptiveTCMScheduler:
    """
    Closed-loop scheduler combining leverage-weighted node timing (Layer 1) and
    covariance-aware deadband execution (Layer 2).

    Parameters
    ----------
    stm : RelativeSTM
    cfg : AdaptiveTCMConfig
    terminal_miss_fn : callable (y_rel_N[6], t) -> (miss_N[3], J[3x3])
        Returns the predicted terminal position miss and its Δv-sensitivity.
        Defaults to the built-in LinearTerminalModel when ``target_rel_pos_N``
        is supplied instead.
    target_rel_pos_N : (3,) optional
        Terminal relative-position target (used only by the linear fallback).
    """

    def __init__(
        self,
        stm: RelativeSTM,
        cfg: AdaptiveTCMConfig,
        terminal_miss_fn: Optional[Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]]] = None,
        target_rel_pos_N: Optional[np.ndarray] = None,
    ):
        self.stm = stm
        self.cfg = cfg
        if terminal_miss_fn is None:
            if target_rel_pos_N is None:
                raise ValueError("Provide terminal_miss_fn or target_rel_pos_N.")
            self._model = LinearTerminalModel(stm, target_rel_pos_N)
            self.terminal_miss_fn = self._model.miss
        else:
            self.terminal_miss_fn = terminal_miss_fn

    # ---- Layer 1: candidate node times -----------------------------------
    def candidate_times(self, n_nodes: Optional[int] = None) -> np.ndarray:
        """
        Front-loaded node placement: spread nodes uniformly in log-leverage
        between the earliest (nav-gated) and latest allowed epochs.  Because
        L(t) falls quasi-exponentially, equal log-L spacing front-loads the
        corrections toward high leverage (cheap Δv) while still keeping a late
        clean-up node for the residual.
        """
        n = self.cfg.max_nodes if n_nodes is None else n_nodes
        t0, tf = self.stm.t0, self.stm.t_f
        ta = t0 + self.cfg.earliest_frac * (tf - t0)
        tb = t0 + self.cfg.latest_frac * (tf - t0)
        La, Lb = self.stm.leverage(ta), self.stm.leverage(tb)
        # target leverages equally spaced in log between La and Lb
        targetsL = np.exp(np.linspace(np.log(La), np.log(Lb), n))
        # invert L(t) -> t on a fine grid (L is monotone decreasing in t)
        tg = np.linspace(ta, tb, 600)
        Lg = np.array([self.stm.leverage(t) for t in tg])
        order = np.argsort(Lg)
        times = np.interp(targetsL, Lg[order], tg[order])
        times = np.sort(times)
        # enforce minimum spacing
        out = [times[0]]
        for t in times[1:]:
            if t - out[-1] >= self.cfg.min_dt_s:
                out.append(t)
        return np.array(out)

    def optimize_node_times(
        self,
        eval_total_dv: Callable[[Sequence[float]], float],
        n_nodes: Optional[int] = None,
    ) -> np.ndarray:
        """
        Refine node epochs to minimise a caller-supplied expected-total-Δv
        objective (typically a Monte-Carlo mean from ``simulate_policy``).
        Uses SciPy if available, else returns the front-loaded candidates.
        """
        x0 = self.candidate_times(n_nodes)
        try:
            from scipy.optimize import minimize
        except Exception:
            return x0
        t0, tf = self.stm.t0, self.stm.t_f
        lo = t0 + self.cfg.earliest_frac * (tf - t0)
        hi = t0 + self.cfg.latest_frac * (tf - t0)

        late_min = t0 + self.cfg.late_node_min_frac * (tf - t0)

        def obj(ts):
            ts = np.sort(np.clip(ts, lo, hi))           # bound by clipping
            if np.any(np.diff(ts) < self.cfg.min_dt_s):
                return 1e9                              # spacing penalty
            if ts[-1] < late_min:                       # require a late clean-up node
                return 1e9                              # (else dispersion blows up)
            return float(eval_total_dv(ts))

        res = minimize(obj, np.asarray(x0, float), method="Nelder-Mead",
                       options={"xatol": 30.0, "fatol": 1e-6, "maxiter": 600})
        x_best = np.sort(np.clip(res.x, lo, hi))
        # keep whichever (seed vs optimized) is actually better
        return x_best if obj(res.x) <= obj(x0) else np.sort(x0)

    # ---- Layer 2: deadband execution at a single node --------------------
    def evaluate_node(
        self, y_rel_N_est: np.ndarray, P_rel_N: np.ndarray, t: float
    ) -> TCMEvent:
        """Decide & size the correction at one node from the nav estimate."""
        miss_N, J = self.terminal_miss_fn(np.asarray(y_rel_N_est, float).reshape(6), t)
        sig = terminal_sigma(self.stm.Phi(t), P_rel_N)
        frac = (t - self.stm.t0) / (self.stm.t_f - self.stm.t0)
        if deadband_skip(miss_N, sig, self.cfg.tol_m, self.cfg.k_sigma):
            return TCMEvent(t, frac, np.zeros(3), 0.0, float(la.norm(miss_N)), sig, True)
        dv = size_correction(J, miss_N)
        return TCMEvent(t, frac, dv, float(la.norm(dv)), float(la.norm(miss_N)), sig, False)


# ══════════════════════════════════════════════════════════════════════════
# Monte-Carlo policy evaluation (linear-model surrogate; swap in Basilisk)
# ══════════════════════════════════════════════════════════════════════════

def simulate_policy(
    stm: RelativeSTM,
    cfg: AdaptiveTCMConfig,
    node_times: Sequence[float],
    y0_rel_N: np.ndarray,
    target_rel_pos_N: np.ndarray,
    nav_pos_sigma: float = 0.6,
    nav_vel_sigma: float = 0.01,
    n_mc: int = 400,
    process_accel_psd: float = 0.0,
    seed: int = 0,
) -> dict:
    """
    Linear-model Monte-Carlo of the closed-loop deadband policy.

    Truth relative deviation is propagated with the STM between nodes (a
    first-order surrogate — replace with the Basilisk propagator for the
    flight-grade number).  At each node the estimate = truth + nav error;
    the policy sizes Δv from the estimate; the executed Δv carries a 1%+floor
    error; the residual is propagated to t_f.

    Returns a dict with per-sample total Δv, terminal miss, and summary stats.
    """
    rng = np.random.default_rng(seed)
    node_times = np.sort(np.asarray(node_times, float))
    model = LinearTerminalModel(stm, target_rel_pos_N)
    Pnav = np.diag([nav_pos_sigma ** 2] * 3 + [nav_vel_sigma ** 2] * 3)

    totals = np.zeros(n_mc)
    miss_end = np.zeros(n_mc)
    n_burns = np.zeros(n_mc, int)

    target = np.asarray(target_rel_pos_N, float).reshape(3)
    for s in range(n_mc):
        y_true = np.asarray(y0_rel_N, float).reshape(6).copy()
        t_prev = stm.t0
        total_dv = 0.0
        nb = 0
        for t in node_times:
            # propagate truth deviation t_prev -> t :
            #   Φ(t, t_prev) = Φ(t_f, t)^{-1} · Φ(t_f, t_prev)
            Phi_t_tprev = la.solve(stm.Phi(t), stm.Phi(t_prev))
            y_true = Phi_t_tprev @ y_true
            if process_accel_psd > 0.0:
                dtp = t - t_prev
                y_true[3:6] += rng.normal(
                    scale=process_accel_psd * np.sqrt(max(dtp, 0.0)), size=3)
            # nav estimate at this node = truth + nav error
            y_est = y_true + np.concatenate([
                rng.normal(scale=nav_pos_sigma, size=3),
                rng.normal(scale=nav_vel_sigma, size=3)])
            # policy decision / sizing from the estimate + covariance
            miss_N, J = model.miss(y_est, t)
            sig = terminal_sigma(stm.Phi(t), Pnav)
            if not deadband_skip(miss_N, sig, cfg.tol_m, cfg.k_sigma):
                dv = size_correction(J, miss_N)
                # execution error: proportional (1σ frac) + fixed floor
                err = (rng.normal(scale=cfg.exec_err_frac, size=3) * dv
                       + rng.normal(scale=cfg.exec_err_floor_mps, size=3))
                dv_exec = dv + err
                y_true[3:6] += dv_exec
                total_dv += la.norm(dv_exec)
                nb += 1
            t_prev = t
        # propagate the residual deviation from the last node to t_f
        yf = _prop_to_tf(stm, y_true, t_prev)        # Φ(t_f, t_prev) y(t_prev)
        miss_end[s] = la.norm(yf[0:3] - target)
        totals[s] = total_dv
        n_burns[s] = nb

    pct = lambda a, q: float(np.percentile(a, q))
    return {
        "node_times": node_times,
        "total_dv": totals,
        "terminal_miss": miss_end,
        "n_burns": n_burns,
        "mean_total_dv": float(np.mean(totals)),
        "p95_total_dv": pct(totals, 95),
        "mean_burns": float(np.mean(n_burns)),
        "frac_within_tol": float(np.mean(miss_end <= cfg.tol_m)),
        "p95_terminal_miss": pct(miss_end, 95),
    }


def _prop_to_tf(stm: RelativeSTM, y_at_t: np.ndarray, t: float) -> np.ndarray:
    """Propagate a relative deviation from time t to t_f: Φ(t_f, t) y(t)."""
    return stm.Phi(t) @ np.asarray(y_at_t, float).reshape(6)


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("adaptive_tcm.py is a library (no standalone action).")
    print("Run the driver instead:")
    print("    python Animation/scripts/guidance/run_adaptive_tcm.py")


__all__ = [
    "MU_EARTH", "gravity_gradient", "RelativeSTM",
    "hill_to_eci_rel", "hill_cov_to_eci_rel",
    "LinearTerminalModel", "size_correction", "terminal_pos_cov",
    "terminal_sigma", "deadband_skip",
    "AdaptiveTCMConfig", "TCMEvent", "AdaptiveTCMScheduler",
    "simulate_policy",
]
