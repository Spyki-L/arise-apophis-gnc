#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FR2 · Property 3 — the RTA decision policy is deterministic, total, and single-branch."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rta import RTAExecutive, Controller, RTAReason
_SENT = ("p",)


def _mk(safe):
    return RTAExecutive(safety_check=lambda a, c: safe, vsc_fallback=lambda c: "VSC", validity_check=None)


def main():
    total = det_bad = total_bad = reason_bad = 0
    reasons = set()
    for competent, present, valid, deadline, safe in itertools.product([False, True], repeat=5):
        prop = _SENT if present else None
        d1 = _mk(safe).decide({}, prop, ac_valid=valid, ac_within_deadline=deadline, ac_competent=competent)
        d2 = _mk(safe).decide({}, prop, ac_valid=valid, ac_within_deadline=deadline, ac_competent=competent)
        total += 1
        if (d1.source, d1.reason) != (d2.source, d2.reason): det_bad += 1     # determinism
        if d1.action is None: total_bad += 1                                   # totality (always an action)
        if not isinstance(d1.reason, RTAReason): reason_bad += 1               # exactly one enumerated reason
        reasons.add(d1.reason)
    print("=" * 74); print("FR2 · Property 3 — DECISION-POLICY DETERMINISM / TOTALITY"); print("=" * 74)
    print(f"[exhaustive] {total} cases:  non-deterministic={det_bad}  no-action(total)={total_bad}  "
          f"bad-reason={reason_bad}  distinct-reasons-reached={len(reasons)}")
    proved = det_bad == 0 and total_bad == 0 and reason_bad == 0
    print("-" * 74)
    print("RESULT:  " + ("PROVED — deterministic, total, finite-branch decision policy."
                         if proved else "*** NOT PROVED ***"))
    return 0 if proved else 1


if __name__ == "__main__":
    sys.exit(main())
