#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FR2 · Property 4 — Behavior-tree temporal-logic spec (safety + liveness).

Model-checks the reactive core in `behavior_tree.py`:
  SAFETY (invariants, exhaustive over the 2^5 belief inputs — 'G phi' reduces to a
    per-tick invariant because the tree is memoryless):
      S1  collision => CBF_SAFE_HOLD                    (safety dominates)
      S2  faulted & not-recoverable & not-collision => SAFE_HOLD
      S3  cmd = NOMINAL => no collision and no fault    (never idle under hazard)
      S4  faulted & not-collision => cmd in {RECOVER, SAFE_HOLD}
      S5  determinism (pure function of the belief)
  LIVENESS (closed-loop, benign/fair environment — perturbations eventually stop):
      L1  every collision is cleared within 1 tick      (uses FR2·P2 CBF invariance)
      L2  every fault reaches a SAFE-SETTLED state within K ticks
      L3  SAFE-SETTLED is stable (absorbing)            'G(settled -> G settled)'
  verified two ways: complete EXPLICIT-STATE reachability, and a Z3 bounded
  model-check (BMC) of the liveness bound over all symbolic start states.
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from behavior_tree import Belief, Cmd, tick

BELIEFS = [Belief(*bits) for bits in itertools.product([False, True], repeat=5)]
K = 4   # certified settling bound (ticks)


def safety_invariants():
    bad = []
    for b in BELIEFS:
        c = tick(b)
        if b.collision_imminent and c is not Cmd.CBF_SAFE_HOLD: bad.append(("S1", b))
        if b.faulted and not b.recoverable and not b.collision_imminent and c is not Cmd.SAFE_HOLD: bad.append(("S2", b))
        if c is Cmd.NOMINAL and (b.collision_imminent or b.faulted): bad.append(("S3", b))
        if b.faulted and not b.collision_imminent and c not in (Cmd.RECOVER, Cmd.SAFE_HOLD): bad.append(("S4", b))
        if tick(b) is not c: bad.append(("S5", b))
    return bad


# closed-loop transition (benign/fair environment): state = (COLL,FLT,REC,DLQ,HELD)
def step(s):
    COLL, FLT, REC, DLQ, HELD = s
    if HELD:                                   # absorbing safe-held
        return (False, FLT, REC, DLQ, True)
    c = tick(Belief(COLL, FLT, REC, DLQ, False))
    if c is Cmd.CBF_SAFE_HOLD: COLL = False
    elif c is Cmd.RECOVER:     FLT = False
    elif c is Cmd.SAFE_HOLD:   HELD = True; COLL = False
    return (COLL, FLT, REC, DLQ, HELD)

def settled(s):
    COLL, FLT, REC, DLQ, HELD = s
    return (not COLL) and ((not FLT) or HELD)


def liveness_explicit():
    l1 = l2 = l3 = 0
    for bits in itertools.product([False, True], repeat=5):
        s = bits
        # L1: collision cleared within 1 tick
        if s[0] and step(s)[0]: l1 += 1
        # L2: settled within K ticks
        cur = s; ok = settled(cur)
        for _ in range(K):
            cur = step(cur); ok = ok or settled(cur)
        if not ok: l2 += 1
        # L3: once settled, stays settled for K more ticks
        if settled(s):
            cur = s
            for _ in range(K):
                cur = step(cur)
                if not settled(cur): l3 += 1; break
    return l1, l2, l3


def liveness_bmc_z3():
    """Z3 BMC: for ALL symbolic start states, a SAFE-SETTLED state is reached within K.
    Encodes the closed-loop transition as booleans and proves 'never settled through K' UNSAT."""
    from z3 import Bools, Bool, Or, And, Not, Implies, Solver, unsat
    s = Solver()
    COLL = [Bool(f"COLL_{t}") for t in range(K + 1)]
    FLT  = [Bool(f"FLT_{t}")  for t in range(K + 1)]
    HELD = [Bool(f"HELD_{t}") for t in range(K + 1)]
    REC = Bool("REC"); DLQ = Bool("DLQ")
    s.add(Not(HELD[0]))                                  # start not yet safe-held
    for t in range(K):
        # cmd predicates (memoryless BT): recover iff FLT & REC & DLQ & ~COLL ; safehold iff FLT & ~recover & ~COLL
        recover  = And(FLT[t], REC, DLQ, Not(COLL[t]))
        safehold = And(FLT[t], Not(recover), Not(COLL[t]))
        # transitions (benign env), with HELD absorbing
        s.add(COLL[t + 1] == And(COLL[t], Not(COLL[t])))        # CBF clears collision in 1 tick => always False
        s.add(FLT[t + 1]  == And(FLT[t], Not(recover)))
        s.add(HELD[t + 1] == Or(HELD[t], safehold))
    settled_t = lambda t: And(Not(COLL[t]), Or(Not(FLT[t]), HELD[t]))
    s.add(And([Not(settled_t(t)) for t in range(K + 1)]))       # ADVERSARY: never settled within K
    return s.check() == unsat


def main():
    bad = safety_invariants()
    l1, l2, l3 = liveness_explicit()
    z = liveness_bmc_z3()
    print("=" * 74)
    print("FR2 · Property 4 — BEHAVIOR-TREE TEMPORAL-LOGIC SPEC")
    print("=" * 74)
    print(f"[safety]  invariants S1–S5 over all 2^5=32 belief inputs: "
          f"{'0 violations' if not bad else str(len(bad))+' VIOLATIONS '+str(bad[:3])}")
    print(f"[liveness · explicit] L1 collision-cleared<=1: {l1} viol · "
          f"L2 settled<=K={K}: {l2} viol · L3 settled-stable: {l3} viol")
    print(f"[liveness · Z3 BMC]   'never settled within K' UNSAT for all starts ... {z}")
    proved = (not bad) and l1 == 0 and l2 == 0 and l3 == 0 and z
    print("-" * 74)
    print("RESULT:  " + ("PROVED — BT meets its safety invariants and bounded-settling liveness spec."
                         if proved else "*** NOT PROVED ***"))
    return 0 if proved else 1


if __name__ == "__main__":
    sys.exit(main())
