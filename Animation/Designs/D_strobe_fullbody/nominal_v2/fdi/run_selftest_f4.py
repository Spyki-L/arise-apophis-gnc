#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_selftest_f4.py — F4 proofs: PEER-as-CBF + joint ISL scheduler.

  T1 homogeneous healthy fleet -> NO PEER false alarm (the MAD pathology is gone)
  T2 collision course -> PEER fault on the offending pair
  T3 connectivity loss -> lambda2 ~ 0 + penalty
  T4 Byzantine liar isolated robustly; honest deputies unaffected
  T5 ISL: adequate budget -> k-coverage met + connected
  T6 ISL: near-CA reserves the collision-critical link
  T7 ISL: infeasible budget -> graceful shortfall (no crash)
"""
import numpy as np, numpy.linalg as la
from peer_cbf import PeerCBF
from isl_scheduler import schedule_isl

rng = np.random.default_rng(0)
M = 8
ang = np.linspace(0, 2*np.pi, M, endpoint=False)
POS = np.array([[1500*np.cos(a), 1500*np.sin(a), 200*np.sin(2*a)] for a in ang])
VEL = np.zeros((M, 3))
REP = POS + rng.normal(0, 0.05, POS.shape)

def ranges(P, sig=0.1):
    R = np.full((M, M), np.nan)
    for i in range(M):
        for j in range(M):
            if i != j:
                R[i, j] = la.norm(P[i]-P[j]) + rng.normal(0, sig)
    return R

cbf = PeerCBF(d_safe=50., R_link=4000., gamma=0.1, sigma_range=0.1, lambda_min=0.2)
NU = 3.0
ok = 0; tot = 0
def check(name, cond, extra=""):
    global ok, tot; tot += 1
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else "")); ok += int(bool(cond))

r1 = cbf.evaluate(POS, VEL, REP, ranges(POS))
check("T1 homogeneous healthy fleet -> NO PEER false alarm", r1["r_peer"].max() < NU,
      f"max r_peer={r1['r_peer'].max():.2f}  lambda2={r1['lambda2']:.1f}")

p2 = POS.copy(); p2[1] = POS[0] + np.array([40., 0, 0]); v2 = VEL.copy(); v2[1] = np.array([-5., 0, 0])
r2 = cbf.evaluate(p2, v2, p2, ranges(p2))
check("T2 collision course -> PEER fault on the pair", r2["r_peer"][0] > NU and r2["r_peer"][1] > NU,
      f"r_peer[D0,D1]={r2['r_peer'][0]:.2f},{r2['r_peer'][1]:.2f}")

p3 = POS.copy(); p3[7] = np.array([50000., 0, 0])
r3 = cbf.evaluate(p3, VEL, p3, ranges(p3))
check("T3 connectivity loss -> lambda2~0 + penalty", r3["lambda2"] < 0.2 and r3["conn_penalty"] > 0,
      f"lambda2={r3['lambda2']:.3f}")

rep4 = REP.copy(); rep4[3] = POS[3] + np.array([60., 0, 0])
r4 = cbf.evaluate(POS, VEL, rep4, ranges(POS))
honest = np.delete(r4["consistency"], 3)
check("T4 Byzantine liar isolated (robust), honest unaffected",
      r4["consistency"][3] > 10*honest.max() and honest.max() < NU,
      f"liar={r4['consistency'][3]:.1f} maxhonest={honest.max():.2f}")

s = schedule_isl(POS, B=4, k_diag=3, w_phase=0.5, R_link=4000., r_coll=300.)
check("T5 adequate budget -> k-coverage met + connected",
      s["k_coverage"] and s["lambda2"] > 0.1, f"min_indeg={s['min_indegree']} lambda2={s['lambda2']:.2f}")

p6 = POS.copy(); p6[1] = POS[0] + np.array([200., 0, 0])
s6 = schedule_isl(p6, B=4, k_diag=3, w_phase=1.0, R_link=4000., r_coll=300.)
check("T6 near-CA reserves the collision-critical link", s6["collision_reserved"] and ((0, 1) in s6["links"]),
      f"reserved={s6['collision_reserved']}")

s7 = schedule_isl(POS, B=2, k_diag=3, w_phase=0.5, R_link=4000., r_coll=300.)
check("T7 infeasible budget -> graceful shortfall (no crash)",
      (not s7["k_coverage"]) and isinstance(s7["shortfall"], list), f"shortfall n={len(s7['shortfall'])}")

print(f"\n{ok}/{tot} passed")
