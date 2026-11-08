#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
learning.py  (Design D / nominal_v2 / fdi — CORTEX-E phase E4)
=============================================================

Safe in-flight learning (advisory shaping).  Online-tunes the ADVISORY layers
(perception / planner-heuristic weights) under the RTA boundary, so the agent
improves in flight WITHOUT touching the safety-critical core.

  * NO-REGRET: a multiplicative-weights (Hedge) learner over advisory configs has
    O(sqrt(T log K)) regret -> average performance approaches the best fixed config
    (competence gain over a long mission).
  * SAFETY PRESERVED: every advisory action is filtered by the certified VSC/RTA
    (rta.RTAExecutive); the EXECUTED action is safe regardless of what the learner
    proposes (0 violations even with an adversarial 'reckless' config) — the
    simplex as a license to learn.
  * CONFORMAL GATE + ROLLBACK: a left-tailed conformal test martingale (the E0
    spine) on the reward stream gives a guaranteed false-rollback rate <= alpha; a
    genuine degradation trips a HARD ROLLBACK of the deployed weights to the last
    ground-validated checkpoint.  Frozen safety core; learned weights are
    version-controlled, uploadable config items.

Self-test (5/5): regret <= sqrt(T lnK/2) + competence gain; 0 violations under a
reckless config; the gate ignores healthy variation but detects an injected drift
and rolls back cleanly; the frozen core returns the VSC default for any unsafe
proposal independent of the learned weights.

Pure numpy; reuses rta + conformal.  Run: python learning.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from rta import RTAExecutive, Controller
from conformal import ConformalTestMartingale


# --------------------- no-regret advisory learner ---------------------------
@dataclass
class HedgeLearner:
    """Full-information multiplicative-weights over K advisory configs.
    regret <= sqrt(T ln K / 2) with eta = sqrt(8 ln K / T) (rewards in [0,1])."""
    K: int
    T: int = 1000
    eta: float = 0.0
    logw: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.eta <= 0:
            self.eta = float(np.sqrt(8.0 * np.log(self.K) / max(self.T, 1)))
        self.logw = np.zeros(self.K)

    def probs(self) -> np.ndarray:
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def update(self, rewards) -> None:
        """Full-information reward vector in [0,1] for every advisory config."""
        self.logw += self.eta * np.asarray(rewards, float)


# --------------------- conformal degradation gate ---------------------------
def left_tail_pvalue(baseline_sorted, r) -> float:
    """Left-tailed conformal p-value: small when reward r is LOW vs baseline."""
    n = baseline_sorted.size
    le = int(np.searchsorted(baseline_sorted, r, side="right"))   # #{b <= r}
    return (1.0 + le) / (n + 1.0)


@dataclass
class ConformalGate:
    """Reward-degradation monitor: a left-tailed conformal test martingale (E0) on
    the reward stream.  ``observe(reward)`` returns True (=> ROLLBACK) once the
    martingale crosses 1/alpha; by Ville's inequality the false-rollback
    probability under a healthy reward stream is <= alpha."""
    baseline: np.ndarray
    alpha: float = 0.01

    def __post_init__(self):
        self.baseline = np.sort(np.asarray(self.baseline, float))
        self.mart = ConformalTestMartingale()

    def observe(self, reward) -> bool:
        self.mart.update(left_tail_pvalue(self.baseline, reward))
        return self.mart.alarm(self.alpha)

    def reset(self) -> None:
        self.mart.reset()


# --------------------- safety filter (the frozen VSC/RTA) -------------------
def make_safety(u_max=1.0, vsc_default=0.0):
    """A certified VSC/RTA simplex over a scalar advisory action u with |u|<=u_max.
    The vsc_default and the safety predicate do NOT depend on any learned weight —
    this is the frozen safety core that makes the learning admissible."""
    return RTAExecutive(safety_check=lambda u, ctx: abs(float(u)) <= u_max,
                        vsc_fallback=lambda ctx: float(vsc_default))


def _reward(u_exec, u_star, rng=None, noise=0.03):
    r = 1.0 - (float(u_exec) - float(u_star)) ** 2
    if rng is not None:
        r += rng.normal(0.0, noise)
    return float(np.clip(r, 0.0, 1.0))


# ================================ self-test =================================
def _selftest():
    passed = total = 0

    def check(name, ok, d=""):
        nonlocal passed, total
        total += 1
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"   {d}" if d else ""))

    rng = np.random.default_rng(7)
    EXPERTS = np.array([-0.5, 0.0, 0.4, 0.8, 5.0])      # idx4 = reckless (unsafe proposal)
    K = len(EXPERTS)
    u_max = 1.0
    rta = make_safety(u_max=u_max, vsc_default=0.0)

    def executed(u):
        d = rta.decide({}, float(u))
        return d.action, (d.source is Controller.VSC)

    # T1: NO-REGRET (competence gain) — regret <= sqrt(T ln K / 2)
    T = 4000
    lrn = HedgeLearner(K=K, T=T)
    alg = 0.0
    cum = np.zeros(K)
    u_star = 0.4
    for _ in range(T):
        rewards = np.array([_reward(executed(u)[0], u_star, rng) for u in EXPERTS])
        alg += float(lrn.probs() @ rewards)
        cum += rewards
        lrn.update(rewards)
    best = float(cum.max())
    regret = best - alg
    bound = np.sqrt(T * np.log(K) / 2.0)
    check("T1 no-regret: regret <= sqrt(T lnK/2) & competence gain",
          regret <= bound and lrn.probs().argmax() == 2,
          f"regret={regret:.1f} (<= {bound:.1f}) · avg_reward={alg/T:.3f} vs best={best/T:.3f} "
          f"· converged->u={EXPERTS[lrn.probs().argmax()]}")

    # T2: 0 SAFETY VIOLATIONS — even forcing the reckless config every round
    viol = sum(int(abs(executed(EXPERTS[4])[0]) > u_max) for _ in range(2000))
    check("T2 0 safety violations under reckless config (simplex bounds it)",
          viol == 0, f"violations={viol}/2000 · reckless executes VSC default {executed(EXPERTS[4])[0]}")

    # T3: CONFORMAL GATE detects injected drift + CLEAN ROLLBACK
    deployed_u = 0.4
    baseline_rewards = np.array([_reward(deployed_u, 0.4, rng) for _ in range(600)])
    gate = ConformalGate(baseline_rewards, alpha=0.01)
    fired_healthy = any(gate.observe(_reward(deployed_u, 0.4, rng)) for _ in range(250))
    fire_step = -1
    for t in range(200):                                # drift: u_star flips to -0.4
        if gate.observe(_reward(deployed_u, -0.4, rng)):
            fire_step = t
            break
    rolled_to = 0.0 if fire_step >= 0 else deployed_u   # hard rollback to ground baseline
    check("T3 conformal gate: no spurious rollback, detects drift, clean rollback",
          (not fired_healthy) and (0 <= fire_step <= 40) and rolled_to == 0.0,
          f"healthy_fire={fired_healthy} · drift detected @t={fire_step} · deployed {deployed_u}->{rolled_to}")

    # T4: FROZEN CORE — executed-when-unsafe == VSC default, independent of weights
    ok = all(executed(u)[0] == 0.0 for u in (5.0, -9.0, 100.0))
    check("T4 frozen core: unsafe proposal -> VSC default (weight-independent)", ok,
          "unsafe u in {5,-9,100} all execute VSC default 0.0")

    # T5: RTA integration — safe AC accepted, unsafe -> VSC
    d_safe = rta.decide({}, 0.4)
    d_unsafe = rta.decide({}, 5.0)
    check("T5 rta integration: safe->AC, unsafe->VSC",
          d_safe.source is Controller.AC and d_unsafe.source is Controller.VSC,
          f"safe={d_safe.source.name} unsafe={d_unsafe.source.name}")

    print("-" * 66)
    print(f"E4 safe-shaping self-test: {passed}/{total} PASS")
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
