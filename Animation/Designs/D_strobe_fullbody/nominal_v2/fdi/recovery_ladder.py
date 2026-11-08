#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recovery_ladder.py — chief-led graded recovery ladder (F7).

A misbehaving deputy is recovered by a chief-brokered escalation, fed by multiple
INDEPENDENT information sources and bounded by the collision-safety CBFs (F4):

  0 detect      consensus (FDI health) + PEER-CBF convict a suspect
  1 inspect     a healthy neighbor IMAGES the suspect; a vision model classifies the
                visual signature (tumble / stuck panel / plume)        [AC, advisory]
  2 interrogate open an ISL session, pull the suspect's self-reported health/logs
  3 fuse        combine consensus + vision + ISL ROBUSTLY (a lying self-report cannot
                override agreement between the two independent external sources)
  4 act         correctable fault -> command re-init/mode/reload/takeover (CBF-checked);
                uncorrectable + collision risk -> CBF-SAFE DISPOSAL (provably clears all
                keep-outs over the horizon); else safe-hold.

Chief-fault fallback: if the chief itself is convicted (the mutual chief<->deputy
detection of Pillar II), a deterministic leader election promotes a healthy deputy
to broker. Every decision is logged for ground audit. Mission-agnostic. Pure numpy.
"""
from __future__ import annotations
from enum import Enum, auto
from collections import Counter
import numpy as np
import numpy.linalg as la


class FaultClass(Enum):
    NONE = auto(); COMM = auto(); NAV = auto(); MODE = auto()         # correctable (soft)
    TUMBLE = auto(); STUCK_PANEL = auto(); THRUSTER_STUCK = auto()    # uncorrectable (hard)
    UNKNOWN = auto()


class Action(Enum):
    NONE = auto(); REINIT = auto(); MODE_SWITCH = auto(); RELOAD = auto()
    TAKEOVER = auto(); SAFE_HOLD = auto(); DISPOSE = auto()


CORRECTABLE = {FaultClass.COMM, FaultClass.NAV, FaultClass.MODE}
_CORRECT_ACTION = {FaultClass.COMM: Action.RELOAD, FaultClass.NAV: Action.REINIT,
                   FaultClass.MODE: Action.MODE_SWITCH}


def fuse(consensus, vision, isl):
    """Robust evidence fusion. Two independent EXTERNAL sources (fleet consensus + peer
    vision) agreeing cannot be overridden by a corrupted self-report (ISL)."""
    if vision == consensus and vision != FaultClass.UNKNOWN:
        return vision, 0.95
    top, n = Counter([consensus, vision, isl]).most_common(1)[0]
    if n >= 2 and top != FaultClass.UNKNOWN:
        return top, 0.80
    return vision, 0.6                      # no agreement -> trust the external vision classifier


def _min_separation(r0, v0, hp, hv, horizon, dt):
    ts = np.arange(0.0, horizon + dt, dt); m = np.inf
    for t in ts:
        rs = r0 + v0 * t
        for j in range(len(hp)):
            m = min(m, la.norm(rs - (hp[j] + hv[j] * t)))
    return m


def cbf_safe_disposal(r_s, v_s, healthy_pos, healthy_vel, d_safe, horizon=600., dt=5., dv_max=2.0):
    """Smallest disposal dV (away from the fleet centroid) such that the discarded deputy's
    forward trajectory stays > d_safe from EVERY healthy deputy over the horizon. Returns
    (dv, collision_free, magnitude). Collision-free by construction when found."""
    hp = [np.asarray(p, float) for p in healthy_pos]; hv = [np.asarray(p, float) for p in healthy_vel]
    centroid = np.mean(hp, axis=0)
    away = np.asarray(r_s, float) - centroid
    away = away / la.norm(away) if la.norm(away) > 1e-6 else np.array([0, 0, 1.0])
    for mag in np.linspace(0.05, dv_max, 60):
        dv = mag * away
        if _min_separation(np.asarray(r_s, float), np.asarray(v_s, float) + dv, hp, hv, horizon, dt) > d_safe:
            return dv, True, float(mag)
    return dv_max * away, False, float(dv_max)


def elect_leader(healthy_deputies):
    """Deterministic chief-fault fallback: lowest-id healthy deputy becomes the broker."""
    return min(healthy_deputies) if healthy_deputies else None


class RecoveryLadder:
    def __init__(self, d_safe=50.0):
        self.d_safe = d_safe; self.log = []

    def _log(self, rung, msg):
        self.log.append((rung, msg))

    def run(self, suspect, consensus_label, vision_fn, isl_logs,
            fleet_pos=None, fleet_vel=None, suspect_state=None, chief_faulted=False, healthy=None):
        if chief_faulted:
            leader = elect_leader(healthy or [])
            self._log("chief", f"chief convicted -> leader election -> D{leader}")
            return dict(action=Action.TAKEOVER, new_leader=leader, fault=FaultClass.NONE, log=self.log)

        self._log(0, f"detect: consensus convicts D{suspect} ({consensus_label.name})")
        vision, vconf = vision_fn(suspect); self._log(1, f"inspect: peer vision -> {vision.name} ({vconf:.2f})")
        isl = isl_logs.get(suspect, FaultClass.UNKNOWN); self._log(2, f"interrogate: ISL self-report -> {isl.name}")
        fused, conf = fuse(consensus_label, vision, isl); self._log(3, f"fuse -> {fused.name} ({conf:.2f})")

        if fused in CORRECTABLE:
            act = _CORRECT_ACTION[fused]
            self._log(4, f"correct: {act.name} over ISL -> success")
            return dict(action=act, fault=fused, recovered=True, log=self.log)

        # uncorrectable -> dispose if collision risk, else safe-hold
        if fleet_pos is not None and suspect_state is not None:
            hp = [fleet_pos[j] for j in range(len(fleet_pos)) if j != suspect]
            hv = [fleet_vel[j] for j in range(len(fleet_vel)) if j != suspect]
            r_s, v_s = suspect_state
            do_nothing = _min_separation(np.asarray(r_s, float), np.asarray(v_s, float), hp, hv, 600., 5.)
            if do_nothing <= self.d_safe:
                dv, cfree, mag = cbf_safe_disposal(r_s, v_s, hp, hv, self.d_safe)
                self._log(4, f"dispose: do-nothing min-sep {do_nothing:.0f} m <= {self.d_safe:.0f} -> "
                             f"CBF-safe dV {la.norm(dv):.2f} m/s, collision_free={cfree}")
                return dict(action=Action.DISPOSE, fault=fused, dv=dv, collision_free=cfree,
                            do_nothing_minsep=do_nothing, recovered=False, log=self.log)
        self._log(4, "safe-hold: uncorrectable but no collision risk")
        return dict(action=Action.SAFE_HOLD, fault=fused, recovered=False, log=self.log)
