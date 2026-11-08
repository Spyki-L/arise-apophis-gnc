#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dual_control.py  (Design D / nominal_v2 / fdi — CORTEX-E phase E6, MOONSHOT)
==========================================================================

Dual-control / optimal-input active diagnosis.  E3 selects PASSIVE probes; some
hypothesis pairs are attribute-identical and stay undiagnosable (E3 flags them).
E6 adds ACTIVE EXCITATION: it designs an optimal, safety-bounded input u* (a small
relative maneuver / commanded test burn) that maximises the MINIMUM pairwise output
separation between the competing hypotheses, so faults that are passively
indistinguishable become distinguishable.

Model: hypothesis h responds y_h(u) = G_h u + b_h  (G = fault-dependent response
gain, b = passive bias).  If two hypotheses share b (passively identical) but
differ in G, an input u != 0 separates them.  The headline: a NAV/estimation fault
and a PROP/thruster fault look identical at rest, but under a commanded test burn
the thruster fault shows a large dV-execution error while the nav fault barely
responds -> 0 -> 4.5 sigma separation within the safe input bound.  A value tradeoff
skips the maneuver when the hypotheses are already passively separable.

Pure numpy (the safe input set is a norm ball = the CBF-safe maneuver envelope).
Run: python dual_control.py (self-test, 5/5).
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np


@dataclass
class ResponseModel:
    name: str
    G: np.ndarray      # response gain (output_dim x input_dim): how the fault reacts to input u
    b: np.ndarray      # passive bias (output_dim): what is seen at u=0


def output(m: ResponseModel, u):
    return m.G @ np.asarray(u, float) + m.b


def separation(models, u, sigma=1.0):
    """Minimum pairwise output separation (in noise-sigma units)."""
    ys = [output(m, u) for m in models]
    return min(np.linalg.norm(ys[i] - ys[j]) / sigma
               for i, j in combinations(range(len(ys)), 2))


def design_input(models, u_max, sigma=1.0, n_samp=6000, seed=0):
    """Optimal safety-bounded input u* = argmax_{||u||<=u_max} min-pairwise separation.
    Random search on the safe ball + local refinement (the input set is the CBF-safe
    maneuver envelope; here a norm ball)."""
    rng = np.random.default_rng(seed)
    din = models[0].G.shape[1]
    best_u = np.zeros(din)
    best = separation(models, best_u, sigma)
    for _ in range(n_samp):
        v = rng.normal(size=din)
        v = v / (np.linalg.norm(v) + 1e-12) * u_max * rng.uniform(0.3, 1.0)
        s = separation(models, v, sigma)
        if s > best:
            best, best_u = s, v
    for _ in range(3000):                                    # local refine
        v = best_u + rng.normal(scale=0.05 * u_max, size=din)
        if np.linalg.norm(v) <= u_max:
            s = separation(models, v, sigma)
            if s > best:
                best, best_u = s, v
    return best_u, best


def recommend(models, u_max, sigma=1.0, sep_target=3.0):
    """Value tradeoff: if the hypotheses are already passively separable (u=0 gives
    >= sep_target), recommend NO maneuver; else design the optimal excitation."""
    passive = separation(models, np.zeros(models[0].G.shape[1]), sigma)
    if passive >= sep_target:
        return np.zeros(models[0].G.shape[1]), passive, False
    u, s = design_input(models, u_max, sigma)
    return u, s, True


# passively-identical NAV vs PROP: same off-station+thrust bias, different response
# to a COMMANDED test burn (a thruster/PROP fault shows a large dV-execution error;
# a NAV/estimation fault barely responds) -> the E3-undiagnosable pair, now separable.
def _nav_prop_models():
    b = np.array([1.0, 0.0])                                 # identical passive signature
    G_nav = np.array([[0.1, 0.0], [0.0, 0.1]])              # weak response to commanded burn
    G_prop = np.array([[1.0, 0.0], [0.0, 1.0]])            # strong dV-execution error
    return [ResponseModel("NAV+PEER", G_nav, b), ResponseModel("PROP+PEER", G_prop, b)]


def _selftest():
    passed = total = 0

    def check(name, ok, d=""):
        nonlocal passed, total
        total += 1
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"   {d}" if d else ""))

    sigma = 0.2
    models = _nav_prop_models()

    # T1: passively indistinguishable -> active input SEPARATES (closes E3 gap)
    sep0 = separation(models, np.zeros(2), sigma)
    u_star, sep_star = design_input(models, u_max=1.0, sigma=sigma)
    check("T1 passive-indistinguishable pair separated by optimal input (closes E3 gap)",
          sep0 < 0.05 and sep_star > 3.0,
          f"sep(u=0)={sep0:.2f}sigma -> sep(u*)={sep_star:.2f}sigma  |u*|={np.linalg.norm(u_star):.2f}")

    # T2: the designed input respects the safety bound
    check("T2 designed input within safety bound |u*|<=u_max", np.linalg.norm(u_star) <= 1.0 + 1e-6,
          f"|u*|={np.linalg.norm(u_star):.3f} <= 1.0")

    # T3: optimal >= best random input drawn from the SAME safe ball
    rng = np.random.default_rng(1)
    def _rand_u(d=2, umax=1.0):
        v = rng.normal(size=d); v = v / (np.linalg.norm(v) + 1e-12) * umax * rng.uniform(0.0, 1.0); return v
    rand = max(separation(models, _rand_u(), sigma) for _ in range(3000))
    check("T3 optimal input >= best random input (same safe ball)", sep_star >= rand - 1e-6,
          f"optimal={sep_star:.2f} best_random={rand:.2f}")

    # T4: value tradeoff — passively separable pair -> NO maneuver recommended
    sep_ok = [ResponseModel("A", np.eye(2), np.array([0.0, 0.0])),
              ResponseModel("B", np.eye(2), np.array([2.0, 0.0]))]     # differ passively
    u_r, s_r, maneuver = recommend(sep_ok, u_max=1.0, sigma=sigma, sep_target=3.0)
    check("T4 value tradeoff: no maneuver when passively separable",
          (not maneuver) and np.linalg.norm(u_r) == 0.0, f"passive_sep={s_r:.1f}sigma maneuver={maneuver}")

    # T5: multi-hypothesis — 3 hyps, all pairwise separated after design
    m3 = [ResponseModel("h1", np.array([[0.1, 0], [0, 0.1]]), np.array([1.0, 0.0])),
          ResponseModel("h2", np.array([[1.0, 0], [0, 0.1]]), np.array([1.0, 0.0])),
          ResponseModel("h3", np.array([[0.1, 0], [0, 1.0]]), np.array([1.0, 0.0]))]
    _, s3 = design_input(m3, u_max=1.0, sigma=sigma)
    check("T5 multi-hypothesis: min pairwise separation resolved", s3 > 3.0,
          f"min-pairwise sep(u*)={s3:.2f}sigma over 3 hypotheses")

    print("-" * 66)
    print(f"E6 dual-control self-test: {passed}/{total} PASS")
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
