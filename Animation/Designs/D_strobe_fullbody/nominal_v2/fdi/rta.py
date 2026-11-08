#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rta.py — Run-Time Assurance (simplex) skeleton (F2).

The flight-safety spine for the onboard "AI". A high-performance, possibly
learned **Advanced Controller (AC)** proposes actions (reconfigurations,
re-points, recovery commands). A **Verified Safe Controller (VSC)** — the
certified baseline (contingency-table lookup + CBF-safe defaults) — is always
available. A **Decision Module** validates every AC proposal against:

    1. validity   (the AC actually produced a well-formed proposal),
    2. deadline   (it produced it in time — bounded WCET), and
    3. safety     (a hard predicate: CBF feasibility, dV/MIB limits, ...).

If any check fails, control reverts to the VSC. Therefore the AC can only ever
*improve* on the certified-safe baseline; it can never make the system unsafe.
That property is the entire flight-safety argument for flying learning-based
autonomy. Every decision is logged with the reason for ground audit.

This module is generic: `safety_check` and `vsc_fallback` are injected by each
later phase (F5 plugs in the contingency table as the VSC; F4 plugs in CBF
feasibility as the safety predicate).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Any, List


class Controller(Enum):
    VSC = auto()   # verified safe controller (certified baseline)
    AC = auto()    # advanced controller (the performance / AI layer)


class RTAReason(Enum):
    AC_ACCEPTED = auto()
    AC_ABSENT = auto()
    AC_INVALID = auto()
    AC_LATE = auto()
    AC_UNSAFE = auto()
    AC_INCOMPETENT = auto()   # E1: AC self-assessed low competence (metacognition)


@dataclass
class RTADecision:
    action: Any
    source: Controller
    reason: RTAReason
    note: str = ""


@dataclass
class RTAExecutive:
    """
    Simplex executive. Provide:
      * safety_check(action, context) -> bool   (HARD safety predicate)
      * vsc_fallback(context)         -> action  (always-safe baseline action)
    Optionally a deadline budget; the caller reports whether the AC met it.
    """
    safety_check: Callable[[Any, Any], bool]
    vsc_fallback: Callable[[Any], Any]
    validity_check: Optional[Callable[[Any], bool]] = None   # well-formedness of the proposal
    log: List[RTADecision] = field(default_factory=list)
    n_ac: int = 0
    n_vsc: int = 0

    def decide(self, context: Any, ac_proposal: Optional[Any],
               ac_valid: bool = True, ac_within_deadline: bool = True,
               ac_competent: bool = True) -> RTADecision:
        if not ac_competent:
            d = RTADecision(self.vsc_fallback(context), Controller.VSC, RTAReason.AC_INCOMPETENT,
                            "AC self-assessed low competence (metacognition E1) -> VSC")
        elif ac_proposal is None:
            d = RTADecision(self.vsc_fallback(context), Controller.VSC, RTAReason.AC_ABSENT,
                            "AC produced no proposal -> VSC")
        elif (not ac_valid) or (self.validity_check is not None and not self.validity_check(ac_proposal)):
            d = RTADecision(self.vsc_fallback(context), Controller.VSC, RTAReason.AC_INVALID,
                            "AC proposal malformed -> VSC")
        elif not ac_within_deadline:
            d = RTADecision(self.vsc_fallback(context), Controller.VSC, RTAReason.AC_LATE,
                            "AC missed WCET deadline -> VSC")
        elif not self.safety_check(ac_proposal, context):
            d = RTADecision(self.vsc_fallback(context), Controller.VSC, RTAReason.AC_UNSAFE,
                            "AC proposal failed safety predicate -> VSC")
        else:
            d = RTADecision(ac_proposal, Controller.AC, RTAReason.AC_ACCEPTED, "AC accepted")
        self.log.append(d)
        if d.source is Controller.AC:
            self.n_ac += 1
        else:
            self.n_vsc += 1
        return d

    def summary(self) -> str:
        return (f"RTA decisions: {len(self.log)} total · AC accepted {self.n_ac} · "
                f"VSC fallbacks {self.n_vsc}")
