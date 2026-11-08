#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fdi_core.py  (Design D / nominal_v2 — causal rehome)
====================================================

Causal, self-contained Fault Detection & Isolation engine for the ARISE/Apophis
deputy swarm.  This is the **P0 + P1** evolution of the original
``GNC/Navigation/fdi_core.py`` (kept verbatim in ``_source_snapshot/``):

  P0  Rehome + causal:
      * No ``GNC`` import dependency — numpy only, drop-in importable from
        ``nominal_v2/fdi/``.
      * Same public API (``FDICore``, ``FDIConfig``, ``FDIState``, ``FaultType``,
        ``ResidualSnapshot``) and the same ``step(...)`` signature, so the
        existing MC harnesses can import this core unchanged.
      * The engine was already a streaming/past-only ``step()``; the acausal
        pieces lived in the *driver* (global truth TCA, whole-timeline fleet
        median, truth FORM target).  Those are replaced by the causal builders
        in ``channels.py``; this core consumes only per-step inputs.

  P1  Defect fixes (both exposed by the project's own Monte-Carlo data):
      * PROP isolation was ~0 % (injected PROP labelled NONE 100 %): the old
        rule returned PROP only under ``prop_bad AND (nav|att|form bad)``, so a
        *pure* propulsion fault was un-isolable.  FIX: a leaky **PROP CUSUM**
        (``S_prop``) makes PROP a first-class, self-sufficient label whose
        evidence persists between sparse burns.
      * FORM_LOCAL settled on the wrong label ~99.5 % (PEER fires first and
        sticks).  FIX: decouple detection *timing* from the settled *label*.
        A per-episode **latch** records every channel that crossed threshold
        since the deputy last left HEALTHY; the settled label (declared at the
        FAULTED transition) is the precedence over the *latched set*, so an
        early PEER-only reading no longer overrides a later FORM trip.

A ``legacy_isolation`` flag reproduces the original classifier exactly so the
before/after improvement can be measured on identical data.

Channels
--------
NAV  innovation NIS RMS · ATT attitude+rate · COMM heartbeat age ·
PROP per-burn Δv mismatch (now CUSUM-accumulated) · FORM task/terminal-miss ·
PEER peer-consistency outlier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Hashable

import numpy as np
import numpy.linalg as la


# ───────────────────── enums & dataclasses ──────────────────────────────────

class FDIState(Enum):
    HEALTHY = auto()
    SUSPECT = auto()
    FAULTED = auto()
    DEAD    = auto()


class FaultType(Enum):
    NONE  = auto()
    PROP  = auto()
    ATT   = auto()
    NAV   = auto()
    COMM  = auto()
    FORM  = auto()   # local formation / task failure
    PEER  = auto()   # inconsistent with peers


@dataclass
class FDIConfig:
    # ── enable / disable channels ────────────────────────────────────────────
    use_nav_channel:  bool = True
    use_att_channel:  bool = True
    use_comm_channel: bool = True
    use_form_channel: bool = False
    use_peer_channel: bool = False

    # ── CUSUM nominal levels (ν) and leak factors (0 = pure integrator) ───────
    nu_nav:   float = 1.0
    leak_nav: float = 0.05
    nu_att:   float = 1.0
    leak_att: float = 0.05
    nu_comm:  float = 1.0
    leak_comm: float = 0.02
    nu_form:  float = 1.0
    leak_form: float = 0.02
    nu_peer:  float = 1.0
    leak_peer: float = 0.05

    # ── CUSUM thresholds ─────────────────────────────────────────────────────
    h_nav:  float = 12.0
    h_att:  float = 8.0
    h_comm: float = 10.0
    h_form: float = 25.0
    h_peer: float = 8.0

    # ── PROP channel (P1 fix): now a leaky CUSUM, first-class label ───────────
    # nu_prop/leak_prop/h_prop tune the accumulator so a single ~5σ faulty burn
    # trips and the evidence persists (small leak) long enough for the
    # persistence machine to confirm, while a consistent (~1) nominal burn does
    # not accumulate.  prop_threshold is retained for the legacy per-burn gate.
    nu_prop:        float = 1.0
    leak_prop:      float = 0.005
    h_prop:         float = 3.0
    prop_threshold: float = 4.0   # legacy per-burn ≈4σ gate (used iff legacy_isolation)

    # ── persistence requirements [samples] ───────────────────────────────────
    n_suspect: int = 8     # HEALTHY→SUSPECT
    n_fault:   int = 20    # SUSPECT→FAULTED

    # ── attitude scaling for residual ────────────────────────────────────────
    sigma_sigma: float = np.deg2rad(0.5)   # [rad]
    sigma_omega: float = np.deg2rad(0.02)  # [rad/s]

    # ── COMM timeout (multiples of nominal heartbeat) ────────────────────────
    comm_trip_multiple: float = 5.0

    # ── P1 behaviour switches ────────────────────────────────────────────────
    prop_first_class: bool = True   # PROP isolable without nav/att/form also bad
    form_peer_latch:  bool = True   # settled label from per-episode latched set
    use_conformal:    bool = False  # conformal-spine (E0) alarms drive the bad flags
    legacy_isolation: bool = False  # True → reproduce the original classifier exactly


@dataclass
class FDIRState:
    state: FDIState
    fault_type: FaultType
    confidence: float
    t_last_change: float


@dataclass
class ResidualSnapshot:
    t: float
    r_nav: float
    r_prop: float
    r_att: float
    r_comm: float
    r_form: float
    r_peer: float
    state: FDIRState


# ───────────────────── core class ───────────────────────────────────────────

class FDICore:
    """
    Filter-agnostic, causal FDI core.  One ``step()`` per mid-rate tick.
    All internal state is O(1) (scalars + small counters) — bounded memory,
    flight-software friendly.
    """

    _CH_NAMES = ("NAV", "ATT", "COMM", "PROP", "FORM", "PEER")

    def __init__(self, cfg: FDIConfig):
        self.cfg = cfg
        self.conformal = None        # optional PhaseScheduledConformalDetector (E0)
        self.reset(t0=0.0)

    def attach_conformal(self, detector):
        """Attach a fitted conformal.PhaseScheduledConformalDetector (E0). When
        cfg.use_conformal and a phase is passed to step(), its per-channel alarms
        drive the bad flags in place of the static CUSUM thresholds."""
        self.conformal = detector
        self._conf_alarm = {}
        return self

    # ───────────── reset ────────────────────────────────────────────────────

    def reset(self, t0: float = 0.0) -> None:
        self.state = FDIRState(
            state=FDIState.HEALTHY,
            fault_type=FaultType.NONE,
            confidence=0.0,
            t_last_change=t0,
        )
        # CUSUM scalars (PROP now included)
        self.S_nav = 0.0
        self.S_att = 0.0
        self.S_comm = 0.0
        self.S_prop = 0.0
        self.S_form = 0.0
        self.S_peer = 0.0

        # persistence counters
        self.k_suspect = 0
        self.k_fault = 0

        # last residuals
        self.r_nav = 0.0
        self.r_prop = 0.0
        self.r_att = 0.0
        self.r_comm = 0.0
        self.r_form = 0.0
        self.r_peer = 0.0

        # P1: per-episode latch — channels that crossed threshold since HEALTHY
        self.episode_channels: set[str] = set()

        self.last_snapshot: Optional[ResidualSnapshot] = None
        self._conf_alarm: dict = {}
        if getattr(self, "conformal", None) is not None:
            self.conformal.reset()

    # ───────────── residuals ────────────────────────────────────────────────

    def _nav_residual(self, innov: np.ndarray, S_innov: np.ndarray) -> float:
        y = np.asarray(innov, float).reshape(-1)
        S = np.asarray(S_innov, float).reshape(y.size, y.size)
        try:
            r2 = float(y.T @ la.solve(S + 1e-12 * np.eye(S.shape[0]), y))
        except la.LinAlgError:
            r2 = float(y.T @ y)
        return np.sqrt(r2 / max(float(y.size), 1.0))

    def _prop_residual(self, dv_est, P_vv) -> float:
        if dv_est is None or P_vv is None:
            return 0.0
        dv = np.asarray(dv_est, float).reshape(3)
        P_vv = np.asarray(P_vv, float).reshape(3, 3)
        try:
            r2 = float(dv.T @ la.solve(P_vv + 1e-12 * np.eye(3), dv))
        except la.LinAlgError:
            r2 = float(dv.T @ dv)
        return np.sqrt(r2 / 3.0)

    def _att_residual(self, sigma_BR, omega_BR_B) -> float:
        s = np.asarray(sigma_BR, float).reshape(3)
        w = np.asarray(omega_BR_B, float).reshape(3)
        e_sigma = la.norm(s) / max(self.cfg.sigma_sigma, 1e-8)
        e_omega = la.norm(w) / max(self.cfg.sigma_omega, 1e-8)
        return e_sigma + e_omega

    def _comm_residual(self, hb_age: float, hb_nominal: float) -> float:
        if hb_nominal <= 0.0:
            return 0.0
        return hb_age / hb_nominal

    # ───────────── CUSUM update ──────────────────────────────────────────────

    @staticmethod
    def _update_cusum(value: float, S_prev: float, nu: float, leak: float) -> float:
        """One-sided leaky CUSUM: S_k = max(0, (1-leak) S_{k-1} + (value - nu))."""
        return max(0.0, (1.0 - leak) * S_prev + (value - nu))

    # ───────────── channel "bad" flags ──────────────────────────────────────

    def _bad_flags(self) -> dict:
        cfg = self.cfg
        if cfg.use_conformal and self.conformal is not None:
            ca = self._conf_alarm
            return dict(
                NAV=cfg.use_nav_channel and bool(ca.get("NAV", False)),
                ATT=cfg.use_att_channel and bool(ca.get("ATT", False)),
                COMM=cfg.use_comm_channel and bool(ca.get("COMM", False)),
                FORM=cfg.use_form_channel and bool(ca.get("FORM", False)),
                PEER=cfg.use_peer_channel and bool(ca.get("PEER", False)),
                PROP=(self.S_prop > cfg.h_prop),   # PROP stays on its P1 CUSUM
            )
        prop_bad = (
            (self.S_prop > cfg.h_prop) if not cfg.legacy_isolation
            else (self.r_prop >= cfg.prop_threshold)
        )
        return dict(
            NAV=cfg.use_nav_channel and (self.S_nav > cfg.h_nav),
            ATT=cfg.use_att_channel and (self.S_att > cfg.h_att),
            COMM=cfg.use_comm_channel and (self.S_comm > cfg.h_comm),
            FORM=cfg.use_form_channel and (self.S_form > cfg.h_form),
            PEER=cfg.use_peer_channel and (self.S_peer > cfg.h_peer),
            PROP=prop_bad,
        )

    # ───────────── instantaneous classification (timing / SUSPECT) ───────────

    def _classify_instantaneous(self, bad: dict) -> FaultType:
        """
        Instantaneous fault candidate — drives detection *timing* and the
        SUSPECT label.  Used every step.  In legacy mode this reproduces the
        original gated logic exactly.
        """
        cfg = self.cfg

        if cfg.legacy_isolation:
            # ---- ORIGINAL logic (PROP gated behind nav/att/form) ----
            if bad["COMM"]:
                return FaultType.COMM
            if bad["FORM"] and bad["PEER"]:
                return FaultType.FORM
            if bad["PEER"] and not bad["FORM"]:
                return FaultType.PEER
            if bad["FORM"] and not bad["PEER"]:
                return FaultType.NAV
            if bad["PROP"] and (bad["NAV"] or bad["ATT"] or bad["FORM"]):
                return FaultType.PROP
            if bad["ATT"] and not bad["PROP"]:
                return FaultType.ATT
            if bad["NAV"] and not (bad["PROP"] or bad["ATT"]):
                return FaultType.NAV
            return FaultType.NONE

        # ---- P1 logic: PROP promoted to a first-class, self-sufficient label ----
        if bad["COMM"]:
            return FaultType.COMM
        if cfg.prop_first_class and bad["PROP"]:
            return FaultType.PROP
        if bad["FORM"] and bad["PEER"]:
            return FaultType.FORM
        if bad["PEER"] and not bad["FORM"]:
            return FaultType.PEER
        if bad["FORM"] and not bad["PEER"]:
            return FaultType.NAV          # common-mode (FORM high fleet-wide, PEER quiet)
        if bad["ATT"]:
            return FaultType.ATT
        if bad["NAV"]:
            return FaultType.NAV
        return FaultType.NONE

    # ───────────── settled classification (latched set, FAULTED) ─────────────

    def _classify_settled(self) -> FaultType:
        """
        Settled isolation label from the per-episode latched channel set.
        This is what downstream guidance/recovery consumes once a deputy is
        FAULTED.  Resolves FORM_LOCAL vs PEER-only vs NAV_GLOBAL correctly even
        when PEER tripped first.
        """
        L = self.episode_channels
        if "COMM" in L:
            return FaultType.COMM
        if "PROP" in L:
            return FaultType.PROP
        if "FORM" in L and "PEER" in L:
            return FaultType.FORM          # local formation fault (both fire)
        if "PEER" in L and "FORM" not in L:
            return FaultType.PEER          # peer-inconsistent only
        if "FORM" in L and "PEER" not in L:
            return FaultType.NAV           # common-mode nav (FORM fleet-wide, PEER quiet)
        if "ATT" in L:
            return FaultType.ATT
        if "NAV" in L:
            return FaultType.NAV
        return FaultType.NONE

    # ───────────── main step ─────────────────────────────────────────────────

    def step(
        self,
        t: float,
        *,
        innov: Optional[np.ndarray],
        S_innov: Optional[np.ndarray],
        dv_est_H: Optional[np.ndarray],
        P_vv: Optional[np.ndarray],
        burn_flag: bool,
        sigma_BR: np.ndarray,
        omega_BR_B: np.ndarray,
        hb_age: float,
        hb_nominal: float,
        r_form: Optional[float] = None,
        r_peer: Optional[float] = None,
        phase: Optional[Hashable] = None,
    ) -> ResidualSnapshot:
        cfg = self.cfg

        # ── residuals ────────────────────────────────────────────────────────
        nav_has_meas = (innov is not None) and (S_innov is not None)
        self.r_nav = self._nav_residual(innov, S_innov) if (nav_has_meas and cfg.use_nav_channel) else 0.0
        self.r_prop = self._prop_residual(dv_est_H, P_vv) if burn_flag else 0.0
        self.r_att = self._att_residual(sigma_BR, omega_BR_B)
        self.r_comm = self._comm_residual(hb_age, hb_nominal)
        self.r_form = float(r_form) if (r_form is not None) else 0.0
        self.r_peer = float(r_peer) if (r_peer is not None) else 0.0

        # ── CUSUM updates ─────────────────────────────────────────────────────
        if cfg.use_nav_channel:
            if nav_has_meas:
                self.S_nav = self._update_cusum(self.r_nav, self.S_nav, cfg.nu_nav, cfg.leak_nav)
            else:
                self.S_nav = (1.0 - cfg.leak_nav) * self.S_nav   # no meas → leak down
        else:
            self.S_nav = 0.0

        self.S_att = self._update_cusum(self.r_att, self.S_att, cfg.nu_att, cfg.leak_att) if cfg.use_att_channel else 0.0
        self.S_comm = self._update_cusum(self.r_comm, self.S_comm, cfg.nu_comm, cfg.leak_comm) if cfg.use_comm_channel else 0.0

        # PROP CUSUM (P1): accumulate evidence at burns; between sparse burns just
        # hold/leak (do NOT subtract nu when there is no burn, else a few burn-free
        # steps would erase the evidence — mirrors the NAV no-measurement branch).
        if burn_flag:
            self.S_prop = self._update_cusum(self.r_prop, self.S_prop, cfg.nu_prop, cfg.leak_prop)
        else:
            self.S_prop = (1.0 - cfg.leak_prop) * self.S_prop

        if cfg.use_form_channel:
            if r_form is not None and r_form > 0.0:
                self.S_form = self._update_cusum(self.r_form, self.S_form, cfg.nu_form, cfg.leak_form)
            else:
                self.S_form = (1.0 - cfg.leak_form) * self.S_form
        else:
            self.S_form = 0.0

        if cfg.use_peer_channel:
            if r_peer is not None and r_peer > 0.0:
                self.S_peer = self._update_cusum(self.r_peer, self.S_peer, cfg.nu_peer, cfg.leak_peer)
            else:
                self.S_peer = (1.0 - cfg.leak_peer) * self.S_peer
        else:
            self.S_peer = 0.0

        # ── bad flags + latch update (P1) ─────────────────────────────────────
        # conformal spine (E0): feed per-step residuals -> per-channel alarms
        if cfg.use_conformal and self.conformal is not None and phase is not None:
            scores = {}
            if cfg.use_nav_channel and nav_has_meas: scores["NAV"] = self.r_nav
            if cfg.use_att_channel:                  scores["ATT"] = self.r_att
            if cfg.use_comm_channel:                 scores["COMM"] = self.r_comm
            if burn_flag:                            scores["PROP"] = self.r_prop
            if cfg.use_form_channel and (r_form is not None and r_form > 0.0): scores["FORM"] = self.r_form
            if cfg.use_peer_channel and (r_peer is not None and r_peer > 0.0): scores["PEER"] = self.r_peer
            if scores: self.conformal.step(phase, scores)
            self._conf_alarm = {ch: self.conformal.mart[ch].alarm(self.conformal.alpha)
                                for ch in self.conformal.channels}

        bad = self._bad_flags()
        for ch in self._CH_NAMES:
            if bad[ch]:
                self.episode_channels.add(ch)

        # ── instantaneous candidate (timing) ─────────────────────────────────
        fault_candidate = self._classify_instantaneous(bad)

        # ── persistence + state machine ───────────────────────────────────────
        if fault_candidate is FaultType.NONE:
            self.k_suspect = max(self.k_suspect - 1, 0)
            self.k_fault = max(self.k_fault - 1, 0)
            if self.k_suspect == 0 and self.k_fault == 0:
                # Evidence fully cleared. De-escalate a SUSPECT that never convicted back to
                # HEALTHY (FAULTED/DEAD are latched terminal states, resolved by the recovery
                # engine's reset() — they are NOT demoted here).
                if self.state.state is FDIState.SUSPECT:
                    self.state.state = FDIState.HEALTHY
                    self.state.t_last_change = t
                if self.state.state is FDIState.HEALTHY:
                    self.state.fault_type = FaultType.NONE
                    self.episode_channels.clear()   # episode over → reset latch
        else:
            self.k_suspect += 1
            if self.k_suspect >= cfg.n_suspect:
                if self.state.state is FDIState.HEALTHY:
                    self.state.state = FDIState.SUSPECT
                    self.state.t_last_change = t
                # SUSPECT label = instantaneous candidate (preserves early warning / timing)
                self.state.fault_type = fault_candidate

            if self.state.state is FDIState.SUSPECT:
                self.k_fault += 1
                if self.k_fault >= cfg.n_fault:
                    self.state.state = FDIState.FAULTED
                    self.state.t_last_change = t

        # ── settled label once FAULTED (P1 latch) ─────────────────────────────
        if self.state.state in (FDIState.FAULTED, FDIState.DEAD):
            if cfg.form_peer_latch and not cfg.legacy_isolation:
                self.state.fault_type = self._classify_settled()
            else:
                self.state.fault_type = fault_candidate

        # ── COMM → DEAD ────────────────────────────────────────────────────────
        if (self.state.state in (FDIState.SUSPECT, FDIState.FAULTED)
                and fault_candidate is FaultType.COMM
                and self.r_comm >= cfg.comm_trip_multiple):
            self.state.state = FDIState.DEAD
            self.state.fault_type = FaultType.COMM
            self.state.t_last_change = t

        # ── confidence ─────────────────────────────────────────────────────────
        max_severity = max(self.S_nav, self.S_att, self.S_comm, self.S_prop, self.S_form, self.S_peer)
        self.state.confidence = 1.0 - np.exp(-max_severity / max(1.0, cfg.h_nav))

        snap = ResidualSnapshot(
            t=t, r_nav=self.r_nav, r_prop=self.r_prop, r_att=self.r_att,
            r_comm=self.r_comm, r_form=self.r_form, r_peer=self.r_peer, state=self.state,
        )
        self.last_snapshot = snap
        return snap
