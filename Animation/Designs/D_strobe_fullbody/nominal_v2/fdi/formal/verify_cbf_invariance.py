#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FR2 · Property 2 — CBF forward-invariance of the collision keep-out.

THEOREM.  Let h(r) = |r|^2 - d_safe^2 (r = relative position), single-integrator
step r_{k+1} = r_k + v*dt.  If the discrete CBF condition
        C := 2 r_k·v + gamma*h(r_k)  >=  0
holds and 0 <= gamma*dt <= 1 and h(r_k) >= 0, then h(r_{k+1}) >= 0.
=> the keep-out set { h >= 0 } is forward-invariant (never breached).

Proof strategy (machine-checked, replaces the sampled F9 T4 300x300 sweep):
  (1) sympy: EXACT algebraic identity
        h_{k+1} = h_k*(1 - gamma*dt) + dt*C + dt^2*|v|^2
      (this rewrites the geometry as a sum whose non-negativity is manifest).
  (2) Z3 : over the REALS, the hypotheses + NOT(h_{k+1} >= 0) is UNSAT
      (a for-all proof over the whole continuous state space).
  (3) numeric bridge: the ACTUAL flight_readiness.cbf_velocity_filter keeps the
      real trajectory outside the keep-out; and the float32==float64 verdict check
      (fixed-point feasibility of the safety math).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sympy_identity():
    import sympy as sp
    # scalars: R2=|r|^2, RV=r.v, V2=|v|^2  (exact, coordinate-free)
    R2, RV, V2, dt, g, d = sp.symbols("R2 RV V2 dt gamma d_safe", real=True)
    hk = R2 - d**2
    h_next_geom = (R2 + 2*dt*RV + dt**2*V2) - d**2      # |r + v*dt|^2 - d^2
    C = 2*RV + g*hk                                     # the CBF margin
    h_next_form = hk*(1 - g*dt) + dt*C + dt**2*V2       # claimed non-negative decomposition
    return sp.simplify(h_next_geom - h_next_form)        # must be exactly 0


def z3_forall_proof():
    from z3 import Reals, Solver, Not, unsat
    R2, RV, V2, dt, g, d = Reals("R2 RV V2 dt gamma d_safe")
    hk = R2 - d*d
    C = 2*RV + g*hk
    h_next = R2 + 2*dt*RV + dt*dt*V2 - d*d
    s = Solver()
    s.add(R2 >= 0, V2 >= 0, RV*RV <= R2*V2)   # r, v are real 3-vectors (Cauchy-Schwarz)
    s.add(dt > 0, g >= 0, g*dt <= 1)          # positive step; class-K gain within the stable range
    s.add(hk >= 0)                            # currently inside the safe set
    s.add(C >= 0)                             # the CBF condition is enforced this step
    s.add(Not(h_next >= 0))                   # ADVERSARY: try to breach the keep-out next step
    return s.check() == unsat                 # UNSAT  ==>  theorem holds for ALL states


def numeric_bridge():
    import flight_readiness as fr
    mf, mu = fr.verify_cbf_invariance(n_trials=400, steps=400)   # runs the ACTUAL filter
    mism = fr.fixed_point_consistency(n=5000)                    # float32 vs float64 verdicts
    return mf, mu, mism


def main():
    diff = sympy_identity()
    z_ok = z3_forall_proof()
    mf, mu, mism = numeric_bridge()
    print("=" * 74)
    print("FR2 · Property 2 — CBF FORWARD-INVARIANCE (collision keep-out)")
    print("=" * 74)
    print(f"[sympy] exact identity  h_next == h_k(1-gamma*dt) + dt*C + dt^2|v|^2 :  "
          f"residual = {diff}  ({'IDENTITY' if diff == 0 else 'MISMATCH'})")
    print(f"[Z3]    for-all reals: hypotheses & NOT(h_next>=0) is UNSAT ........ {z_ok}")
    print(f"[bridge] real cbf_velocity_filter: min h filtered={mf:.3f} (>=0)  "
          f"unfiltered={mu:.0f} (collides)")
    print(f"[bridge] fixed-point: float32 vs float64 safe/unsafe mismatches = {mism}")
    proved = (diff == 0 and z_ok and mf > -1e-6 and mu < -100.0 and mism == 0)
    print("-" * 74)
    print("RESULT:  " + ("PROVED — { h >= 0 } is forward-invariant under the CBF condition."
                         if proved else "*** NOT PROVED ***"))
    return 0 if proved else 1


if __name__ == "__main__":
    sys.exit(main())
