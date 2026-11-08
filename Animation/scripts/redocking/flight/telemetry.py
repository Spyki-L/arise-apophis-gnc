# -*- coding: utf-8 -*-
"""telemetry.py - flight event + health logging for one deputy's re-dock."""
from __future__ import annotations
import numpy as np


class Telemetry:
    def __init__(self, deputy_id=0):
        self.id = deputy_id; self.rows = []; self.events = []

    def log(self, step, t, phase, ctrl_mode, dist, rta_status, faults, prop_frac, nav_sigma):
        self.rows.append(dict(step=step, t=t, phase=phase, ctrl=ctrl_mode, dist=dist,
                              rta=rta_status, faults=list(faults), prop=prop_frac, nav_sigma=nav_sigma))

    def event(self, t, msg):
        self.events.append((t, msg))

    def summary(self):
        if not self.rows:
            return {}
        phases = [r["phase"] for r in self.rows]
        return dict(deputy=self.id, steps=len(self.rows), final_phase=phases[-1],
                    min_dist=min(r["dist"] for r in self.rows),
                    n_rta_limited=sum(r["rta"] == "LIMITED" for r in self.rows),
                    n_rta_abort=sum(r["rta"] == "ABORT" for r in self.rows),
                    faults=sorted(set(f for r in self.rows for f in r["faults"])),
                    prop_used_frac=self.rows[-1]["prop"], events=self.events)


if __name__ == "__main__":
    tm = Telemetry(1)
    tm.event(0.0, "handoff"); tm.log(1, 1.0, "APPROACH", "RL", 80.0, "NOMINAL", [], 0.0, 0.4)
    tm.log(2, 2.0, "TERMINAL", "RL", 2.0, "LIMITED", [], 0.05, 0.3)
    print("telemetry summary:", tm.summary()); print("telemetry OK")
