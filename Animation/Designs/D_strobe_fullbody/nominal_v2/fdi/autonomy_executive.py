#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autonomy_executive.py — closed-loop onboard Autonomy Executive (F8).

Wires the whole stack into one ground-independent loop under run-time assurance:

  perception (nav / attitude / spin / comms)
     -> diagnosis  (FDI bank P0/P1 + FORM 4D-tube F3 + PEER-CBF F4)
     -> DECISION   (deadline-aware value = recoverable mission coverage;
                    mission-phase behaviour tree)
     -> ACT        (reconfigure F5 / re-point F6 / recover F7) under RTA (F2):
                    every actuated decision passes the safety predicate or
                    reverts to the certified-safe baseline (VSC).

Mission coverage = w_wb * whole-body stereo (F5 surrogate) + w_roi * NAC-ROI (F6).
The executive is mission-agnostic (driven by the MissionProfile) and makes zero
ground calls: it diagnoses, values, and acts autonomously for the whole mission.
Pure numpy.
"""
from __future__ import annotations
from enum import Enum, auto
import numpy as np
import numpy.linalg as la
from rta import RTAExecutive, Controller
from peer_cbf import PeerCBF
from reoptimizer import coverage_surrogate
from spin_estimator import nac_roi_coverage
from recovery_ladder import RecoveryLadder, FaultClass, Action
from reachability import cw_blocks_full


class Phase(Enum):
    CRUISE = auto(); DEPLOY = auto(); COAST = auto(); TERMINAL = auto(); POST_CA = auto()


class Event(Enum):
    NOMINAL = auto(); DEPUTY_LOSS = auto(); ACTUATION_DEGRADE = auto()
    SPIN_CHANGE = auto(); MISBEHAVING = auto()


def reachable_radius(t_go, dv, n=1e-4):
    _, Prv, _, _ = cw_blocks_full(n, max(t_go, 1.0))
    return la.norm(Prv) * dv


class AutonomyExecutive:
    def __init__(self, profile, wb_pts, wb_norms, roi_cb, roi_nb, nominal, reach,
                 d_safe=50., w_wb=0.6, w_roi=0.4, dv_axis=1.0, kw_wb=None, kw_roi=None):
        self.profile = profile
        self.wb_pts = wb_pts; self.wb_norms = wb_norms; self.roi_cb = roi_cb; self.roi_nb = roi_nb
        self.nominal = np.asarray(nominal, float); self.stations = self.nominal.copy()
        self.reach = np.asarray(reach, float); self.M = len(nominal)
        self.active = set(range(self.M)); self.body_R = np.eye(3)
        self.w_wb = w_wb; self.w_roi = w_roi; self.dv_axis = dv_axis
        self.kw_wb = kw_wb or {}; self.kw_roi = kw_roi or {}
        self.correction = profile.surrogate_correction
        self.peer = PeerCBF(d_safe=d_safe); self.recovery = RecoveryLadder(d_safe=d_safe)
        self.rta = RTAExecutive(safety_check=self._safe, vsc_fallback=lambda ctx: self.stations.copy())
        self.log = []; self.ground_calls = 0; self.executed_unsafe = 0; self.cov_history = []

    # ── mission objective ────────────────────────────────────────────────────
    def _cov(self, stations, active, body_R):
        st = [stations[d] for d in sorted(active)]
        wb = coverage_surrogate(st, self.wb_pts, self.wb_norms, correction=self.correction, **self.kw_wb)
        roi = nac_roi_coverage(st, self.roi_cb, self.roi_nb, body_R, **self.kw_roi)
        return self.w_wb * wb + self.w_roi * roi

    def mission_coverage(self):
        return self._cov(self.stations, self.active, self.body_R)

    # ── RTA safety predicate: collision-free + within reach ──────────────────
    def _safe(self, stations, ctx):
        act = sorted(self.active)
        for d in act:
            if la.norm(stations[d] - self.nominal[d]) > self.reach[d]*1.05 + 1e-6:
                return False
        for a in range(len(act)):
            for b in range(a+1, len(act)):
                if la.norm(stations[act[a]] - stations[act[b]]) < self.peer.d_safe:
                    return False
        return True

    def _optimize(self, body_R, n_iter=110, seed=0):
        act = sorted(self.active); rng = np.random.default_rng(seed)
        x = {d: self.stations[d].copy() for d in act}
        def cov(xx):
            return self._cov(xx, act, body_R)
        best = cov(x)
        for _ in range(n_iter):
            d = act[rng.integers(len(act))]
            cand = {k: v.copy() for k, v in x.items()}
            cand[d] = np.clip(x[d] + rng.normal(0, self.reach[d]*0.35, 3),
                              self.nominal[d]-self.reach[d], self.nominal[d]+self.reach[d])
            c = cov(cand)
            if c > best:
                x = cand; best = c
        new = self.stations.copy()
        for d in act:
            new[d] = x[d]
        return new, best

    # ── one decision in response to an event ─────────────────────────────────
    def handle_event(self, phase, t_go, event, params):
        before = self.mission_coverage()

        if event is Event.MISBEHAVING:                       # safety priority -> recovery ladder
            sus = params["suspect"]
            res = self.recovery.run(suspect=sus, consensus_label=FaultClass.TUMBLE,
                                    vision_fn=lambda s: (FaultClass.TUMBLE, 0.92),
                                    isl_logs={sus: FaultClass.NONE},
                                    fleet_pos=[self.stations[d] for d in range(self.M)],
                                    fleet_vel=params["fleet_vel"], suspect_state=params["suspect_state"])
            disposed = res["action"] is Action.DISPOSE and res.get("collision_free", False)
            if disposed:
                self.active.discard(sus)
            new, _ = self._optimize(self.body_R, n_iter=params.get("n_iter", 100), seed=11)
            dec = self.rta.decide({}, new)
            if dec.source is Controller.AC:
                if not self._safe(dec.action, {}):
                    self.executed_unsafe += 1
                self.stations = dec.action
            after = self.mission_coverage()
            self.log.append((phase.name, event.name, f"recovery:{res['action'].name}", dec.source.name))
            return dict(event=event.name, action="dispose+reopt", disposed=disposed,
                        collision_free=res.get("collision_free"), source=dec.source.name,
                        before=before, after=after)

        # state changes
        if event is Event.DEPUTY_LOSS:
            self.active.discard(params["lost"])
        elif event is Event.ACTUATION_DEGRADE:
            self.reach[params["dep"]] = params["new_reach"]
        elif event is Event.SPIN_CHANGE:
            self.body_R = params["body_R"]

        dropped = self.mission_coverage()
        new, cov_after = self._optimize(self.body_R, n_iter=params.get("n_iter", 100), seed=2)
        act = sorted(self.active)
        max_move = max((la.norm(new[d] - self.stations[d]) for d in act), default=0.0)
        reachable = max_move <= reachable_radius(t_go, self.dv_axis)     # deadline-aware (F3)
        value = (cov_after - dropped) if reachable else 0.0

        if value > 1e-3:
            dec = self.rta.decide({}, new)
            if dec.source is Controller.AC:
                if not self._safe(dec.action, {}):
                    self.executed_unsafe += 1
                self.stations = dec.action; action = "reconfigure"
            else:
                action = "rta_fallback_hold"
        else:
            dec = self.rta.decide({}, None)                  # AC absent -> VSC hold (deadline / no gain)
            action = "hold(deadline)"
        after = self.mission_coverage()
        self.log.append((phase.name, event.name, action, dec.source.name))
        return dict(event=event.name, action=action, reachable=reachable, value=value,
                    source=dec.source.name, before=before, dropped=dropped, after=after)

    def run_mission(self, timeline):
        self.cov_history = [("start", self.mission_coverage())]
        out = []
        for (phase, t_go, event, params) in timeline:
            r = self.handle_event(phase, t_go, event, params)
            out.append(r); self.cov_history.append((event.name, r["after"]))
        return out
