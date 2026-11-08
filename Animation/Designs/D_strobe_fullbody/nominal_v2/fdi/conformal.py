#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conformal.py  (Design D / nominal_v2 / fdi — CORTEX-E phase E0)
==============================================================

The **conformal (epistemic) spine**.  Replaces the hand-tuned leaky-CUSUM
thresholds (``nu``/``h`` in ``fdi_core.FDIConfig``) with **phase-scheduled,
distribution-free** calibration that gives a *guaranteed* false-alarm rate per
mission phase on the violently non-stationary encounter.

Three pieces, all pure-numpy, causal/streaming, O(1)-O(N_calib) memory:

  1. ``MondrianConformal`` - per-(channel, phase) **split-conformal** calibration
     of HEALTHY nonconformity scores -> conformal p-values and thresholds.
     Conditioning on the phase (Mondrian taxonomy) restores exchangeability under
     non-stationarity - the "phase-scheduled" mechanism.  Under exchangeability a
     conformal p-value is super-uniform: ``P(p <= a) <= a`` (finite-sample,
     distribution-free - no Gaussian / residual-model assumption).

  2. ``ConformalTestMartingale`` - a non-negative **test martingale** on the
     p-value stream (uniform mixture of power martingales ``g_e(p)=e*p^(e-1)``).
     Because ``E_null[g_e(p)] = 1`` each ``M^e`` is a martingale with ``M_0 = 1``,
     and so is their mixture.  By **Ville's inequality**
     ``P( sup_t M_t >= 1/a ) <= a``, so alarming at ``M_t >= 1/a`` bounds the
     whole-stream false-alarm probability by ``a`` => ``ARL0 >= ~1/a``.  O(1) state.

  3. ``PhaseScheduledConformalDetector`` - wires (1)+(2) per channel into a
     drop-in causal detector that consumes the SAME per-step residual scores
     ``fdi_core`` already produces (``r_nav, r_att, r_comm, r_prop, r_form,
     r_peer``), conditioned on the current phase, and alarms with a guaranteed
     false-alarm rate ``alpha``.

``conformal_threshold(calib, alpha)`` returns the per-phase split-conformal upper
threshold - the minimal drop-in for a static ``h`` if you keep the CUSUM
accumulator instead of the martingale.

This is phase **E0** of ``CORTEX_ADVANCED_CAPABILITIES.md`` and the realisation of
the master plan's "phase-scheduled conformal sequential detection".  Run
``python conformal.py`` for the self-test (the E0 exit artifact: measured ARL0 /
false-alarm rate meets target across phases).  Self-test = 6/6.

Design notes
------------
* **Split vs online conformal.**  We use *split* conformal with a fixed HEALTHY
  calibration set per (channel, phase) - the flight-practical design (calibrate on
  the ground / an early healthy window, then monitor).  Given the calibration set,
  the test p-values are independent and (super-)uniform, so the mixture martingale
  is a genuine (super-)martingale and Ville's bound holds.  Growing/online conformal
  is the [MOONSHOT] refinement (E8).
* **Why no leak is needed.**  Under the null the mixture martingale stays bounded
  (the e->1 component floors log M near 0), so it does not drift into false alarms
  the way a raw integrator would, yet a true shift drives it up fast.  A
  ``reset()`` re-arms it when an episode ends (mirrors fdi_core's per-episode latch).
* **Coverage is marginal.**  Conformal guarantees coverage *averaged over the
  calibration draw*; the self-test verifies the marginal guarantee (T5) and the
  Ville/ARL0 guarantee (T2/T4), which hold given the calibration set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Optional

import numpy as np


# the ARISE encounter phases (default taxonomy; ANY hashable key works) -------
ARISE_PHASES = ("SEPARATION", "COAST", "APPROACH", "ENCOUNTER")
# the fdi_core channels this calibrates (same names / order) ------------------
CHANNELS = ("NAV", "ATT", "COMM", "PROP", "FORM", "PEER")


# --------------------- core split-conformal primitives ----------------------

def conformal_pvalue(calib: np.ndarray, score: float) -> float:
    """Right-tailed split-conformal p-value (scalar).

        p = (1 + #{c in calib : c >= score}) / (len(calib) + 1)

    Under exchangeability of (calib + {score}) the p-value is super-uniform:
    ``P(p <= a) <= a`` for every a, with finite-sample, distribution-free validity.
    Larger residual => smaller p => more anomalous.  Never 0 (so log p is finite)."""
    calib = np.asarray(calib, dtype=float)
    n = calib.size
    if n == 0:
        return 1.0
    ge = int(np.count_nonzero(calib >= score))
    return (1.0 + ge) / (n + 1.0)


def conformal_pvalues(calib: np.ndarray, scores, presorted: bool = False) -> np.ndarray:
    """Vectorized right-tailed conformal p-values for an array of ``scores``
    against ``calib`` (same definition as ``conformal_pvalue``) - O((n+m)*log n)
    via ``searchsorted``.  Used for batch calibration and the self-test.  Returns
    an array shaped like ``scores``."""
    calib = np.asarray(calib, dtype=float)
    if not presorted:
        calib = np.sort(calib)
    n = calib.size
    scores = np.asarray(scores, dtype=float)
    if n == 0:
        return np.ones_like(scores)
    ge = n - np.searchsorted(calib, scores, side="left")   # #{c >= s}
    return (1.0 + ge) / (n + 1.0)


def conformal_threshold(calib: np.ndarray, alpha: float) -> float:
    """Split-conformal upper threshold: the smallest ``h`` with, under the null,
    ``P(score > h) <= alpha`` (finite-sample, marginal).  It is the
    ceil((1-a)(n+1))-th order statistic of the calibration scores - a drop-in for a
    hand-tuned static ``h`` (per channel, per phase).  Returns ``+inf`` if the
    calibration set is too small to certify the level (conservative)."""
    calib = np.sort(np.asarray(calib, dtype=float))
    n = calib.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((1.0 - alpha) * (n + 1)))   # rank giving guaranteed coverage
    if k > n:
        return float("inf")
    return float(calib[k - 1])


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(x - m))))


def martingale_logM_path(pvalues, eps_grid: Optional[np.ndarray] = None) -> np.ndarray:
    """Vectorized log-martingale trajectory for a 1-D p-value stream - the batch
    equivalent of stepping ``ConformalTestMartingale`` (used for analysis and
    cross-checked against the streaming class in the self-test)."""
    e = (np.linspace(0.05, 0.95, 19) if eps_grid is None else np.asarray(eps_grid, float))
    p = np.clip(np.asarray(pvalues, float), 1e-12, 1.0)
    A = np.cumsum(np.log(p))
    n = np.arange(1, p.size + 1)
    L = n[:, None] * np.log(e)[None, :] + (e[None, :] - 1.0) * A[:, None]   # (H,K)
    m = L.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(L - m).sum(axis=1))) - float(np.log(e.size))


# --------------------- Mondrian (phase-conditioned) calibrator --------------

_EMPTY = np.empty(0, dtype=float)


@dataclass
class MondrianConformal:
    """Per-phase split-conformal calibration of one channel's HEALTHY scores.

    ``fit`` populates a (sorted) calibration set keyed by phase; ``pvalue`` and
    ``threshold`` are conditioned on the phase, which is what makes the guarantee
    hold across the non-stationary encounter (Mondrian / class-conditional
    conformal).  A static single-window calibrator is shown to FALSE-ALARM under
    non-stationarity in the self-test (T4); Mondrian holds the rate."""
    max_calib: int = 4000
    _calib: dict = field(default_factory=dict)   # phase -> 1-D np.ndarray (sorted)

    def fit(self, phase: Hashable, scores) -> "MondrianConformal":
        self._calib[phase] = np.sort(np.asarray(scores, dtype=float).ravel())[-self.max_calib:]
        return self

    def n(self, phase: Hashable) -> int:
        a = self._calib.get(phase)
        return 0 if a is None else int(a.size)

    def pvalue(self, phase: Hashable, score: float) -> float:
        a = self._calib.get(phase, _EMPTY)
        return float(conformal_pvalues(a, np.array([score]), presorted=True)[0])

    def threshold(self, phase: Hashable, alpha: float) -> float:
        return conformal_threshold(self._calib.get(phase, _EMPTY), alpha)


# --------------------- conformal test martingale ----------------------------

@dataclass
class ConformalTestMartingale:
    """Mixture-of-power conformal test martingale on a stream of p-values.

    Betting functions ``g_e(p) = e * p^(e-1)`` for ``e`` in ``eps_grid`` satisfy
    ``E_null[g_e(p)] = 1`` (p ~ U(0,1)), so each ``M^e_t = prod_i g_e(p_i)`` is a
    non-negative martingale with ``M_0 = 1``; their uniform mixture is too.

    Closed-form O(1) update: with ``A = sum_i log p_i`` and count ``n``,
    ``log M^e_t = n*log e + (e-1)*A``, and ``log M_t = logsumexp_e(...) - log|grid|``.

    ``alarm(alpha)`` is True once ``M_t >= 1/alpha``; by Ville's inequality the
    probability of EVER alarming under the null is <= alpha (=> ARL0 >= ~1/alpha)."""
    eps_grid: np.ndarray = field(default_factory=lambda: np.linspace(0.05, 0.95, 19))
    _A: float = 0.0      # running sum log p
    _n: int = 0          # samples seen
    logM: float = 0.0    # current log-martingale (M_0 = 1 => logM_0 = 0)

    def reset(self) -> None:
        self._A = 0.0
        self._n = 0
        self.logM = 0.0

    def update(self, p: float) -> float:
        p = float(min(max(p, 1e-12), 1.0))
        self._A += float(np.log(p))
        self._n += 1
        e = np.asarray(self.eps_grid, dtype=float)
        L = self._n * np.log(e) + (e - 1.0) * self._A      # log of each power mart.
        self.logM = _logsumexp(L) - float(np.log(e.size))
        return self.logM

    def alarm(self, alpha: float) -> bool:
        return self.logM >= -float(np.log(alpha))

    @property
    def M(self) -> float:
        return float(np.exp(self.logM))


# --------------------- drop-in phase-scheduled detector ---------------------

class PhaseScheduledConformalDetector:
    """Causal, phase-scheduled conformal detector - a drop-in alternative to the
    leaky-CUSUM ``(nu, h)`` per channel.  Consumes the same per-step residual
    scores ``fdi_core`` computes, conditioned on the current phase, and alarms
    with a guaranteed false-alarm rate ``alpha`` per channel.

    Usage
    -----
        det = PhaseScheduledConformalDetector(alpha=0.01)
        det.fit({"NAV": {"COAST": healthy_nav_coast, ...}, ...})   # ground calib
        out = det.step("APPROACH", {"NAV": r_nav, "ATT": r_att, ...})
        # out["NAV"] -> {"p": ..., "logM": ..., "alarm": bool}
    """

    def __init__(self, alpha: float = 0.01, channels=CHANNELS,
                 eps_grid: Optional[np.ndarray] = None, max_calib: int = 4000):
        self.alpha = float(alpha)
        self.channels = tuple(channels)
        self._grid = (np.linspace(0.05, 0.95, 19) if eps_grid is None
                      else np.asarray(eps_grid, float))
        self.calib = {ch: MondrianConformal(max_calib=max_calib) for ch in self.channels}
        self.mart = {ch: ConformalTestMartingale(eps_grid=self._grid) for ch in self.channels}

    def fit(self, healthy_scores: dict) -> "PhaseScheduledConformalDetector":
        """healthy_scores: {channel: {phase: array_of_healthy_scores}}."""
        for ch, by_phase in healthy_scores.items():
            if ch not in self.calib:
                continue
            for phase, scores in by_phase.items():
                self.calib[ch].fit(phase, scores)
        return self

    def step(self, phase: Hashable, scores: dict) -> dict:
        out = {}
        for ch, s in scores.items():
            if ch not in self.calib:
                continue
            p = self.calib[ch].pvalue(phase, float(s))
            logM = self.mart[ch].update(p)
            out[ch] = {"p": p, "logM": logM, "alarm": self.mart[ch].alarm(self.alpha)}
        return out

    def reset(self, channel: Optional[str] = None) -> None:
        """Re-arm the martingale(s) - call when an episode ends / deputy returns
        HEALTHY (mirrors fdi_core's per-episode latch reset)."""
        if channel is None:
            for m in self.mart.values():
                m.reset()
        else:
            self.mart[channel].reset()


# ============================== self-test ===================================
# The E0 exit artifact: distribution-free coverage, ARL0 / false-alarm control
# ACROSS PHASES, detection power, and the necessity of phase-conditioning.
# Vectorized (searchsorted p-values + martingale scanned across trials).

def _gen_healthy(rng, phase, shape):
    """Synthetic HEALTHY nonconformity scores (positive, chi-like) whose SCALE
    grows toward the encounter - the non-stationarity the spine must absorb."""
    scale = {"SEPARATION": 1.0, "COAST": 1.2, "APPROACH": 1.8, "ENCOUNTER": 2.6}[phase]
    shape = (shape, 2) if np.isscalar(shape) else tuple(shape) + (2,)
    return np.abs(rng.normal(0.0, scale, size=shape)).sum(axis=-1)


def _logsumexp_axis1(L):
    m = L.max(axis=1, keepdims=True)
    return m[:, 0] + np.log(np.exp(L - m).sum(axis=1))


def _mart_scan(logp_MH, alpha, e):
    """(M,H) log p-values -> (alarmed_mask, first_cross_step) for the mixture
    martingale.  Vectorized over trials; loops over time (O(1) state per trial)."""
    M, H = logp_MH.shape
    loge = np.log(e)
    logK = float(np.log(e.size))
    thr = -float(np.log(alpha))
    A = np.zeros(M)
    crossed = np.zeros(M, bool)
    at = np.full(M, -1)
    for t in range(H):
        A += logp_MH[:, t]
        L = (t + 1) * loge[None, :] + (e[None, :] - 1.0) * A[:, None]   # (M,K)
        logM = _logsumexp_axis1(L) - logK
        newly = (logM >= thr) & (~crossed)
        at[newly] = t
        crossed |= newly
    return crossed, at


def _selftest() -> int:
    rng = np.random.default_rng(7)
    e = np.linspace(0.05, 0.95, 19)
    passed = total = 0
    NCAL = 3000
    alpha = 0.01

    def check(name, ok, detail=""):
        nonlocal passed, total
        total += 1
        passed += int(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))

    # T1: conformal coverage (super-uniformity), per phase ---------------------
    ok_all = True
    det = []
    for phase in ARISE_PHASES:
        cal = _gen_healthy(rng, phase, NCAL)
        test = _gen_healthy(rng, phase, 40000)
        ps = conformal_pvalues(cal, test)
        for a in (0.05, 0.10, 0.20):
            ok_all &= (float(np.mean(ps <= a)) <= a + 0.012)
        det.append(f"{phase[:4]}:{float(np.mean(ps<=0.10)):.3f}")
    check("T1 conformal coverage P(p<=a)<=a per phase  (emp@0.10)", ok_all, " ".join(det))

    # T2: ARL0 / false-alarm control ACROSS PHASES (phase-conditioned) ---------
    H, M = 600, 1000
    fa = {}
    ok_all = True
    for phase in ARISE_PHASES:
        cal = np.sort(_gen_healthy(rng, phase, NCAL))
        streams = _gen_healthy(rng, phase, (M, H))
        logp = np.log(conformal_pvalues(cal, streams.ravel(), presorted=True).reshape(M, H))
        crossed, _ = _mart_scan(logp, alpha, e)
        fa[phase] = float(crossed.mean())
        ok_all &= (fa[phase] <= alpha * 1.8)
    det = " ".join(f"{p[:4]}:{fa[p]*100:.1f}%" for p in ARISE_PHASES)
    check(f"T2 ARL0 across phases: FA<= {alpha:.0%} (Ville) [target ARL0>={int(1/alpha)}]", ok_all, det)

    # T3: detection power / ARL1 (fault -> fast alarm) -------------------------
    phase = "APPROACH"
    cal = np.sort(_gen_healthy(rng, phase, NCAL))
    H3, M3 = 200, 800
    streams = _gen_healthy(rng, phase, (M3, H3)) + 3.0          # ~persistent shift
    logp = np.log(conformal_pvalues(cal, streams.ravel(), presorted=True).reshape(M3, H3))
    crossed, at = _mart_scan(logp, alpha, e)
    power = float(crossed.mean())
    arl1 = float(np.median(at[crossed]) + 1) if crossed.any() else float("inf")
    check("T3 detection power>=0.95 & ARL1 small", (power >= 0.95 and arl1 <= 40),
          f"power={power:.3f}  ARL1(med)={arl1:.0f}  (vs ARL0>={int(1/alpha)})")

    # T4: phase-conditioning is NECESSARY -------------------------------------
    # Honest failure mode of NOT phase-conditioning: a STATIC single-window
    # calibration (once, on quiet COAST data) false-alarms when the healthy
    # residual scale rises into the encounter.  Mondrian (per-phase) holds.
    seg, M4 = 400, 1000
    static_cal = np.sort(_gen_healthy(rng, "COAST", NCAL))     # calibrate ONCE on quiet data
    mond = MondrianConformal()
    for p in ARISE_PHASES:
        mond.fit(p, _gen_healthy(rng, p, NCAL))
    lg = np.empty((M4, seg * len(ARISE_PHASES)))
    lm = np.empty((M4, seg * len(ARISE_PHASES)))
    for i, phase in enumerate(ARISE_PHASES):
        block = _gen_healthy(rng, phase, (M4, seg))            # non-stationary walk
        sl = slice(i * seg, (i + 1) * seg)
        lg[:, sl] = np.log(conformal_pvalues(static_cal, block.ravel(), presorted=True).reshape(M4, seg))
        lm[:, sl] = np.log(conformal_pvalues(mond._calib[phase], block.ravel(), presorted=True).reshape(M4, seg))
    fg = float(_mart_scan(lg, alpha, e)[0].mean())
    fm = float(_mart_scan(lm, alpha, e)[0].mean())
    check("T4 phase-conditioning necessary (static-calib FA >> a; Mondrian FA <= a)",
          (fg > 0.10 and fm <= alpha * 1.8), f"static={fg*100:.1f}%  mondrian={fm*100:.1f}%")

    # T5: drop-in threshold validity - MARGINAL P(score>h)<=alpha -------------
    ok_all = True
    det = []
    R = 300
    for phase in ARISE_PHASES:
        exc = []
        for _ in range(R):
            cal = _gen_healthy(rng, phase, 2000)
            h = conformal_threshold(cal, alpha=0.05)
            exc.append(float(np.mean(_gen_healthy(rng, phase, 2000) > h)))
        mexc = float(np.mean(exc))                  # marginal over calibration draws
        ok_all &= (mexc <= 0.05 + 0.006)
        det.append(f"{phase[:4]}:{mexc:.3f}")
    check("T5 conformal_threshold drop-in: marginal P(score>h)<=0.05", ok_all, " ".join(det))

    # T6: causality + determinism + class/vectorized cross-check --------------
    cal = np.sort(_gen_healthy(rng, "ENCOUNTER", NCAL))
    s1 = _gen_healthy(np.random.default_rng(3), "ENCOUNTER", 120) + 2.0
    p1 = conformal_pvalues(cal, s1, presorted=True)
    m = ConformalTestMartingale()
    traj_stream = np.array([m.update(p) for p in p1])
    m2 = ConformalTestMartingale()
    traj_again = np.array([m2.update(p) for p in p1[:60]])
    traj_vec = martingale_logM_path(p1)
    causal = np.allclose(traj_stream[:60], traj_again)     # @60 unaffected by 60..120
    classmatch = np.allclose(traj_stream, traj_vec)        # streaming == vectorized
    check("T6 causal & deterministic & streaming==vectorized",
          (causal and classmatch), f"causal={causal} class_xcheck={classmatch}")

    print("-" * 66)
    print(f"E0 conformal spine self-test: {passed}/{total} PASS")
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
