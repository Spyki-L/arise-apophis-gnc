# -*- coding: utf-8 -*-
"""
executive.py - autonomy mission state machine for one deputy's re-dock.

States:
  RETURN        far/mid-field GNC return (Phase-A) to the capture envelope
  HANDOFF       arrived in the envelope; hand to terminal control
  APPROACH      glideslope approach toward the RAFTI port
  TERMINAL      inside the terminal ball; soft-capture control
  CAPTURE       capture criteria met; confirm over N steps
  DOCKED        latched (terminal, success)
  ABORT         safety/FDIR abort -> command retreat
  RETREAT       back off to a safe standoff
  REACQUIRE     re-enter APPROACH (bounded retries)
  SAFE_HOLD     station-keep and wait for ground (terminal, safe)
  FAILED        retries exhausted / unrecoverable (terminal)

Transitions consume the nav estimate (range, capture), the RTA status and the FDIR
recommendation, with timeouts and a bounded reacquire count.  Returns (phase,
guidance_mode) where guidance_mode tells the loop what to command.
"""
from __future__ import annotations
from flight import config_flight as F

TERMINAL_BALL = 3.0
CAPTURE_CONFIRM_STEPS = 3
STANDOFF_M = 15.0


class Executive:
    def __init__(self):
        self.phase = "HANDOFF"; self.t_in = 0.0; self.reacq = 0; self._cap = 0
        self.max_reacq = F.FDIR["max_reacquire"]; self.max_steps = F.GNC["max_steps"]
        self.steps = 0

    def step(self, dist, captured, rta_status, fdir_action, dt):
        self.steps += 1; self.t_in += dt
        p = self.phase
        # global safety pre-emption (except in terminal/safe states)
        if p in ("APPROACH", "TERMINAL", "HANDOFF"):
            if rta_status == "ABORT" or fdir_action == "ABORT":
                self._to("ABORT"); return self.phase, "retreat"
            if fdir_action == "SAFE_HOLD":
                self._to("SAFE_HOLD"); return self.phase, "hold"

        if p == "HANDOFF":
            self._to("APPROACH")
        elif p == "APPROACH":
            if captured:
                self._to("CAPTURE")
            elif dist < TERMINAL_BALL:
                self._to("TERMINAL")
            elif dist > F.SAFETY["abort_range_m"] or self.steps > self.max_steps:
                self._to("ABORT")
        elif p == "TERMINAL":
            if captured:
                self._to("CAPTURE")
            elif dist > TERMINAL_BALL * 3:
                self._to("APPROACH")
            elif self.steps > self.max_steps:
                self._to("ABORT")
        elif p == "CAPTURE":
            # a soft-capture LATCH accumulates solid captures; it does not require
            # them consecutive (residual body-rate limit cycle flickers the flag).
            self._cap += int(captured)
            if self._cap >= CAPTURE_CONFIRM_STEPS:
                self._to("DOCKED")
            elif dist > TERMINAL_BALL:                       # only drop out if it really backs off
                self._to("TERMINAL")
        elif p == "ABORT":
            self._to("RETREAT")
        elif p == "RETREAT":
            if dist > STANDOFF_M:
                if self.reacq < self.max_reacq:
                    self.reacq += 1; self._to("APPROACH")
                else:
                    self._to("SAFE_HOLD")
        # DOCKED / SAFE_HOLD / FAILED are terminal

        mode = {"HANDOFF": "approach", "APPROACH": "approach", "TERMINAL": "approach",
                "CAPTURE": "approach", "DOCKED": "idle", "ABORT": "retreat",
                "RETREAT": "retreat", "REACQUIRE": "approach", "SAFE_HOLD": "hold",
                "FAILED": "hold"}[self.phase]
        return self.phase, mode

    def _to(self, s):
        if s != self.phase:
            self.phase = s; self.t_in = 0.0; self._cap = 0

    @property
    def done(self):
        return self.phase in ("DOCKED", "SAFE_HOLD", "FAILED")


if __name__ == "__main__":
    ex = Executive()
    seq = []
    # nominal approach -> terminal -> capture -> docked
    for dist, cap in [(80, 0), (50, 0), (10, 0), (2, 0), (0.04, 1), (0.04, 1), (0.04, 1), (0.04, 1)]:
        ph, mode = ex.step(dist, bool(cap), "NOMINAL", "NOMINAL", 1.0); seq.append(ph)
    print("nominal sequence:", seq); assert ex.phase == "DOCKED"
    ex2 = Executive()
    ph, _ = ex2.step(50, 0, "ABORT", "NOMINAL", 1.0); print("RTA abort ->", ex2.phase)
    assert ex2.phase == "ABORT"
    print("executive OK")
