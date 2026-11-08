#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
behavior_tree.py — CORTEX verifiable reactive core (the reflex tier).

A small, DETERMINISTIC, BOUNDED-DEPTH behavior tree that maps the typed belief
(the OBC's FDI/CBF verdicts + deadline) to a single safe command each 10 Hz tick.
Priority (Selector, highest first):

  1. collision imminent      -> CBF_SAFE_HOLD     (safety dominates everything)
  2. core FAULTED            -> RECOVER  if recoverable AND within deadline
                                else SAFE_HOLD    (uncorrectable -> safe)
  3. re-opt opportunity      -> DELIBERATE (call the bounded planner) if in time
  4. otherwise               -> NOMINAL

It is memoryless per tick (a pure function of the belief) -> amenable to
temporal-logic verification (BehaVerify-style). The committed command is a
PROPOSAL/derived reflex; rta.py remains the actuation gate. Pure stdlib.

Run `python3 behavior_tree.py` for the exhaustive command table + safety self-test.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Cmd(Enum):
    NOMINAL = 0
    DELIBERATE = 1
    RECOVER = 2
    SAFE_HOLD = 3
    CBF_SAFE_HOLD = 4


class Status(Enum):
    SUCCESS = 1
    FAILURE = 0


@dataclass(frozen=True)
class Belief:
    collision_imminent: bool     # CBF margin < 0 / predicted keep-out breach
    faulted: bool                # OBC FDI core FAULTED on >=1 deputy
    recoverable: bool            # the isolated fault has a soft recovery
    within_deadline: bool        # reconfiguration reachable in time-to-go
    reopt_req: bool              # a coverage re-optimisation opportunity exists


# ── BT node primitives: tick(b) -> (Status, Cmd|None) ; bounded, deterministic ──
class Node:
    def tick(self, b: Belief): raise NotImplementedError

class Condition(Node):
    def __init__(self, fn, name=""): self.fn = fn; self.name = name
    def tick(self, b): return (Status.SUCCESS, None) if self.fn(b) else (Status.FAILURE, None)

class Action(Node):
    def __init__(self, cmd: Cmd): self.cmd = cmd
    def tick(self, b): return (Status.SUCCESS, self.cmd)

class Sequence(Node):
    """AND: run children in order; fail fast; carry the last emitted command."""
    def __init__(self, children): self.children = children
    def tick(self, b):
        cmd = None
        for c in self.children:
            st, cc = c.tick(b)
            if st is Status.FAILURE:
                return (Status.FAILURE, None)
            if cc is not None:
                cmd = cc
        return (Status.SUCCESS, cmd)

class Selector(Node):
    """OR (priority): return the first child that SUCCEEDS."""
    def __init__(self, children): self.children = children
    def tick(self, b):
        for c in self.children:
            st, cc = c.tick(b)
            if st is Status.SUCCESS:
                return (Status.SUCCESS, cc)
        return (Status.FAILURE, None)


# ── the reactive FDIR core (fixed structure, depth <= 4) ──
_ROOT = Selector([
    Sequence([Condition(lambda b: b.collision_imminent, "collision"),
              Action(Cmd.CBF_SAFE_HOLD)]),
    Sequence([Condition(lambda b: b.faulted, "faulted"),
              Selector([
                  Sequence([Condition(lambda b: b.recoverable, "recoverable"),
                            Condition(lambda b: b.within_deadline, "deadline"),
                            Action(Cmd.RECOVER)]),
                  Action(Cmd.SAFE_HOLD)])]),
    Sequence([Condition(lambda b: b.reopt_req, "reopt"),
              Condition(lambda b: b.within_deadline, "deadline"),
              Action(Cmd.DELIBERATE)]),
    Action(Cmd.NOMINAL),
])

MAX_DEPTH = 4   # static structural bound on the tick recursion


def tick(b: Belief) -> Cmd:
    """One reflex tick: belief -> safe command. Total and deterministic."""
    st, cmd = _ROOT.tick(b)
    return cmd if cmd is not None else Cmd.SAFE_HOLD   # default-safe (never None)


if __name__ == "__main__":
    import itertools
    bad = 0
    print(f"{'COLL FLT REC DLQ ROP':>22}  ->  cmd")
    for coll, flt, rec, dlq, rop in itertools.product([False, True], repeat=5):
        b = Belief(coll, flt, rec, dlq, rop); c = tick(b)
        # inline safety self-test (the complete proof is in formal/verify_behavior_tree.py)
        if coll and c is not Cmd.CBF_SAFE_HOLD: bad += 1
        if flt and (not rec or not dlq) and not coll and c is not Cmd.SAFE_HOLD: bad += 1
        if c is Cmd.NOMINAL and (coll or flt): bad += 1
        print(f"{int(coll)}    {int(flt)}   {int(rec)}   {int(dlq)}   {int(rop)}   ".rjust(22) + f"  ->  {c.name}")
    print("-" * 40)
    print(f"safety self-test: {'PASS (0 violations / 32)' if bad == 0 else str(bad)+' VIOLATIONS'}")
