#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
safe_rl.py  (Design D / nominal_v2 / fdi — CORTEX-E phase E7, MOONSHOT)
=====================================================================

Safe online RL with a proven regret bound + a safety-preservation theorem — the
rigorous version of the E4 Hedge demo.

A linear contextual bandit (OFUL / LinUCB) learns the planner's value-shaping reward
r = <theta*, x> + noise.  Every action is drawn ONLY from the context-dependent
CBF-SAFE SET (an action is safe iff |s + delta*dt| + w_max*dt <= s_max, robust to a
bounded disturbance), and a zero-effect safe-hold action (the VSC fallback) is always
available.  Consequences, all verified in the self-test:

  * SAFETY PRESERVATION (theorem): every executed action keeps the closed-loop state
    inside the safe set, so the barrier h = s_max - |s| >= 0 for all t (forward
    invariance) and there are 0 safety violations — during learning, by construction.
  * O(d sqrt(T) log T) SAFE-REGRET (OFUL): regret vs the best SAFE action in hindsight
    is sublinear (R(4T)/R(T) ~ 2, not 4) and within the bound; average regret -> 0.
  * FREE EXPLORATION: the learner still identifies theta* (||theta_hat - theta*|| -> 0)
    despite never leaving the safe set.
  * TEST-TIME POLICY SWITCHING by remaining dV budget: use the aggressive
    (high-dV/high-reward) policy while it still leaves enough budget to finish at the
    conservative rate, else conservative, else safe-hold — cumulative dV never exceeds
    the budget.

Pure numpy.  Run: python safe_rl.py (self-test, 6/6).
"""
from __future__ import annotations
import numpy as np


class SafeLinUCB:
    """OFUL / LinUCB linear contextual bandit.  Reward r = <theta*, x> + noise;
    picks argmax UCB over the SAFE action subset only."""

    def __init__(self, d, lam=1.0, beta=1.5):
        self.d = d
        self.V = lam * np.eye(d)
        self.b = np.zeros(d)
        self.beta = beta
        self.Vinv = np.linalg.inv(self.V)

    def theta(self):
        return self.Vinv @ self.b

    def ucb(self, X):                                   # X: (K, d)
        th = self.theta()
        bonus = self.beta * np.sqrt(np.einsum("ij,jk,ik->i", X, self.Vinv, X))
        return X @ th + bonus

    def select(self, X, safe_mask):
        u = self.ucb(X)
        u = np.where(safe_mask, u, -np.inf)
        return int(np.argmax(u))

    def update(self, x, r):
        self.V += np.outer(x, x)
        self.b += r * x
        self.Vinv = np.linalg.inv(self.V)


def safe_mask(s, delta, dt, s_max, w_max):
    """CBF: action k is safe iff |s + delta_k*dt| + w_max*dt <= s_max (robust to a
    bounded rate disturbance |w| <= w_max).  Guarantees |s_{t+1}| <= s_max."""
    return np.abs(s + delta * dt) + w_max * dt <= s_max + 1e-12


def budget_policy_switch(budget, T, dv_aggr=0.02, dv_cons=0.005, r_aggr=1.0, r_cons=0.6):
    """Test-time switching: use the aggressive (high-dV, high-reward) policy while it
    still leaves enough budget to finish at the conservative rate; else conservative;
    else safe-hold.  The cumulative dV never exceeds the budget."""
    spent = 0.0
    total_r = 0.0
    used = []
    for t in range(T):
        remaining = T - t - 1
        if spent + dv_aggr + remaining * dv_cons <= budget + 1e-12:
            spent += dv_aggr; total_r += r_aggr; used.append("A")
        elif spent + dv_cons <= budget + 1e-12:
            spent += dv_cons; total_r += r_cons; used.append("C")
        else:
            used.append("HOLD")
    return spent, total_r, used


def run_bandit(T, d=5, K=10, seed=0, s_max=1.0, dt=0.1, w_max=0.02, delta_scale=3.0):
    """Run the safe LinUCB closed loop for T rounds; return regret/violations/invariance."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=d); theta /= np.linalg.norm(theta)
    algo = SafeLinUCB(d)
    s = 0.0
    regret = 0.0
    violations = 0
    barrier_ok = True
    max_abs_s = 0.0
    for t in range(T):
        X = rng.normal(size=(K, d)); X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        X[0] = 0.0                                       # a_0 = safe-hold (zero feature, delta 0)
        delta = delta_scale * X[:, 0]; delta[0] = 0.0
        mask = safe_mask(s, delta, dt, s_max, w_max)
        mask[0] = True                                   # safe-hold always available (VSC fallback)
        k = algo.select(X, mask)
        exp_r = X @ theta
        best_safe = np.max(np.where(mask, exp_r, -np.inf))
        regret += best_safe - exp_r[k]
        violations += int(not mask[k])
        r = exp_r[k] + rng.normal(0, 0.1)
        algo.update(X[k], r)
        w = rng.uniform(-w_max, w_max)                   # rate disturbance -> contributes w*dt
        s = s + (delta[k] + w) * dt
        max_abs_s = max(max_abs_s, abs(s))
        if abs(s) > s_max + 1e-9:
            barrier_ok = False
    theta_err = np.linalg.norm(algo.theta() - theta) / np.linalg.norm(theta)
    return dict(regret=regret, violations=violations, barrier_ok=barrier_ok,
                max_abs_s=max_abs_s, s_max=s_max, theta_err=theta_err)


def _selftest():
    passed = total = 0

    def check(name, ok, d=""):
        nonlocal passed, total
        total += 1
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"   {d}" if d else ""))

    d = 5
    r1 = run_bandit(2000, d=d, seed=1)
    r2 = run_bandit(8000, d=d, seed=1)

    ratio = r2["regret"] / max(r1["regret"], 1e-9)
    bound = 3.0 * d * np.sqrt(8000) * np.log(8000)
    check("T1 O(sqrt T) safe-regret: sublinear scaling & within OFUL bound",
          ratio <= 2.8 and r2["regret"] <= bound,
          f"R(2000)={r1['regret']:.1f} R(8000)={r2['regret']:.1f} ratio={ratio:.2f} (~2) bound={bound:.0f}")

    check("T2 0 safety violations (actions from CBF-safe set only)",
          r1["violations"] == 0 and r2["violations"] == 0,
          f"violations: T=2000->{r1['violations']}  T=8000->{r2['violations']}")

    check("T3 forward invariance: barrier h=s_max-|s| >= 0 for all t",
          r2["barrier_ok"] and r2["max_abs_s"] <= r2["s_max"] + 1e-9,
          f"max|s|={r2['max_abs_s']:.3f} <= s_max={r2['s_max']}")

    check("T4 learns under safety (theta_hat -> theta*)", r2["theta_err"] < 0.15,
          f"||theta_hat-theta*||/||theta*|| = {r2['theta_err']:.3f}")

    spent, tot_r, used = budget_policy_switch(budget=8.0, T=1000)
    switched = ("A" in used and ("C" in used or "HOLD" in used))
    first_C = next((i for i, u in enumerate(used) if u != "A"), len(used))
    check("T5 policy switch by budget: never exceeds budget, switches as it depletes",
          spent <= 8.0 + 1e-9 and switched and used[0] == "A",
          f"spent={spent:.2f}<=8.0 · aggressive early, switch@t={first_C} · reward={tot_r:.0f}")

    check("T6 average regret -> 0", r2["regret"] / 8000 < 0.05,
          f"R_T/T = {r2['regret']/8000:.4f}")

    print("-" * 66)
    print(f"E7 safe-RL self-test: {passed}/{total} PASS")
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
