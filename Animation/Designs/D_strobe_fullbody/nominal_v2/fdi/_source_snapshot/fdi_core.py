#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fdi_core.py
===========

Fault Detection & Isolation core for a Hill-frame deputy, built to sit
on top of the HillEKF in hill_ekf.py and your existing attitude stack.

Channels
--------
- NAV   : innovation-based residual with leaky CUSUM
- ATT   : attitude + rate residual with leaky CUSUM
- COMM  : heartbeat-age residual with leaky CUSUM
- PROP  : per-burn Δv mismatch (no CUSUM)
- FORM  : task / formation performance (e.g. predicted terminal miss)
- PEER  : peer-consistency residual (how "off" you are vs the fleet)

FORM is intended for flyby-style tasks: "are you going to be where you have
to be, when you have to be there?"  The main script provides a normalised
scalar r_form >= 0 (≈1 nominal, >>1 bad) based on predicted terminal miss.
PEER is a robust outlier score based on how r_form compares to other deputies.

Classification logic:
- FORM+PEER high → local formation fault (you are the outlier).
- FORM high, PEER low → everyone is struggling → treat as NAV / global issue,
  not a single-deputy FORM fault.
- PEER high alone → you are inconsistent with peers → PEER fault.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
import numpy.linalg as la


# ───────────────────── enums & basic dataclasses ────────────────────────────

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
    # Enable / disable channels
    use_nav_channel:   bool = True
    use_att_channel:   bool = True
    use_comm_channel:  bool = True
    use_form_channel:  bool = False
    use_peer_channel:  bool = False

    # CUSUM nominal levels (ν) and leak factors (0 = no leak, ~0.01–0.1 common)
    nu_nav:    float = 1.0
    leak_nav:  float = 0.05

    nu_att:    float = 1.0
    leak_att:  float = 0.05

    nu_comm:   float = 1.0
    leak_comm: float = 0.02

    nu_form:   float = 1.0
    leak_form: float = 0.02

    nu_peer:   float = 1.0
    leak_peer: float = 0.05

    # CUSUM thresholds
    # (start conservative; tune by Monte Carlo so healthy runs stay below)
    h_nav:   float = 12.0
    h_att:   float = 8.0
    h_comm:  float = 10.0
    h_form:  float = 25.0
    h_peer:  float = 8.0

    # Per-burn propulsion threshold (no CUSUM for PROP)
    prop_threshold: float = 4.0   # ≈ 4σ; tune from MC

    # Persistence requirements [samples]
    n_suspect: int = 8     # HEALTHY→SUSPECT
    n_fault:   int = 20    # SUSPECT→FAULTED

    # Attitude scaling for residual
    sigma_sigma: float = np.deg2rad(0.5)   # [rad]
    sigma_omega: float = np.deg2rad(0.02)  # [rad/s]

    # COMM timeout (in multiples of nominal heartbeat)
    comm_trip_multiple: float = 5.0


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
    Filter-agnostic FDI core.

    Typical usage at e.g. 10 Hz mid-rate:

        cfg = FDIConfig()
        fdi = FDICore(cfg)

        snap = fdi.step(
            t          = t_abs,
            innov      = ekf.innov,   # or None if no NAV meas this step
            S_innov    = ekf.S,       # or None if no NAV meas
            dv_est_H   = dv_est_H,    # None if not a burn step
            P_vv       = P_vv,
            burn_flag  = burn_flag,
            sigma_BR   = sigma_BR,
            omega_BR_B = omega_BR_B,
            hb_age     = hb_age,
            hb_nominal = heartbeat_period,
            r_form     = r_form,      # scalar, >=0 or None if inactive
            r_peer     = r_peer,      # scalar, >=0 or None if inactive
        )
    """

    def __init__(self, cfg: FDIConfig):
        self.cfg = cfg
        self.reset(t0=0.0)

    # ───────────── reset ────────────────────────────────────────────────────

    def reset(self, t0: float = 0.0) -> None:
        """Reset all FDI internal state for a fresh run starting at t0."""
        self.state = FDIRState(
            state=FDIState.HEALTHY,
            fault_type=FaultType.NONE,
            confidence=0.0,
            t_last_change=t0,
        )

        # CUSUM scalars
        self.S_nav   = 0.0
        self.S_att   = 0.0
        self.S_comm  = 0.0
        self.S_form  = 0.0
        self.S_peer  = 0.0

        # persistence counters
        self.k_suspect = 0
        self.k_fault   = 0

        # last residuals
        self.r_nav   = 0.0
        self.r_prop  = 0.0
        self.r_att   = 0.0
        self.r_comm  = 0.0
        self.r_form  = 0.0
        self.r_peer  = 0.0

        self.last_snapshot: Optional[ResidualSnapshot] = None

    # ───────────── residuals ────────────────────────────────────────────────

    def _nav_residual(self, innov: np.ndarray, S_innov: np.ndarray) -> float:
        """
        Normalised NAV residual:

            r_nav = sqrt( y^T S^{-1} y / dof )

        If you pass S = diag(sigma^2), this reduces to the RMS of
        (y / sigma) across axes.
        """
        y = np.asarray(innov, float).reshape(-1)
        S = np.asarray(S_innov, float).reshape(y.size, y.size)
        try:
            r2 = float(y.T @ la.solve(S + 1e-12 * np.eye(S.shape[0]), y))
        except la.LinAlgError:
            r2 = float(y.T @ y)
        dof = float(y.size)
        return np.sqrt(r2 / max(dof, 1.0))

    def _prop_residual(
        self,
        dv_est: Optional[np.ndarray],
        P_vv: Optional[np.ndarray],
    ) -> float:
        """
        r_prop = sqrt(Δv^T P_vv^{-1} Δv / 3), or 0 if no burn this step.
        """
        if dv_est is None or P_vv is None:
            return 0.0
        dv = np.asarray(dv_est, float).reshape(3)
        P_vv = np.asarray(P_vv, float).reshape(3, 3)
        try:
            r2 = float(dv.T @ la.solve(P_vv + 1e-12 * np.eye(3), dv))
        except la.LinAlgError:
            r2 = float(dv.T @ dv)
        return np.sqrt(r2 / 3.0)

    def _att_residual(self, sigma_BR: np.ndarray, omega_BR_B: np.ndarray) -> float:
        """
        Combine attitude and rate errors into a single scalar residual.
        """
        s = np.asarray(sigma_BR, float).reshape(3)
        w = np.asarray(omega_BR_B, float).reshape(3)

        e_sigma = la.norm(s) / max(self.cfg.sigma_sigma, 1e-8)
        e_omega = la.norm(w) / max(self.cfg.sigma_omega, 1e-8)
        return e_sigma + e_omega

    def _comm_residual(self, hb_age: float, hb_nominal: float) -> float:
        """
        Normalised heartbeat age: ≈1 when on time, grows as comm is lost.
        """
        if hb_nominal <= 0.0:
            return 0.0
        return hb_age / hb_nominal

    # ───────────── CUSUM updates (with leak) ────────────────────────────────

    @staticmethod
    def _update_cusum(value: float, S_prev: float, nu: float, leak: float) -> float:
        """
        One-sided *leaky* CUSUM:

            S_k = max(0, (1 - leak) S_{k-1} + (value - nu))

        leak in [0,1). 0 = no leak (pure integrator).
        """
        S = (1.0 - leak) * S_prev + (value - nu)
        return max(0.0, S)

    # ───────────── classification helper ────────────────────────────────────

    def _classify_fault(self) -> FaultType:
        """
        Pattern classification based on which channels are bad.

        - NAV / ATT / COMM / FORM / PEER use their CUSUMs.
        - PROP uses per-burn threshold on r_prop (no CUSUM).

        Priorities:
        1) COMM (you might be dead to the network)
        2) FORM/PEER (mission-level geometry)
        3) PROP (if burns are obviously wrong and geometry is bad)
        4) ATT / NAV
        """
        cfg = self.cfg

        nav_bad   = cfg.use_nav_channel   and (self.S_nav   > cfg.h_nav)
        att_bad   = cfg.use_att_channel   and (self.S_att   > cfg.h_att)
        comm_bad  = cfg.use_comm_channel  and (self.S_comm  > cfg.h_comm)
        form_bad  = cfg.use_form_channel  and (self.S_form  > cfg.h_form)
        peer_bad  = cfg.use_peer_channel  and (self.S_peer  > cfg.h_peer)
        prop_bad  = self.r_prop >= cfg.prop_threshold

        # 1) comm dominates – you might as well be dead
        if comm_bad:
            return FaultType.COMM

        # 2) formation / peer logic
        #    - FORM & PEER high → you are the formation outlier
        if form_bad and peer_bad:
            return FaultType.FORM

        #    - PEER high alone → inconsistent with fleet
        if peer_bad and not form_bad:
            return FaultType.PEER

        #    - FORM high, PEER low → everyone is off → treat as NAV / global
        if form_bad and not peer_bad:
            return FaultType.NAV

        # 3) propulsion: large Δv mismatch plus geometry/nav/att trouble
        if prop_bad and (nav_bad or att_bad or form_bad):
            return FaultType.PROP

        # 4) attitude vs nav
        if att_bad and not prop_bad:
            return FaultType.ATT
        if nav_bad and not (prop_bad or att_bad):
            return FaultType.NAV

        return FaultType.NONE

    # ───────────── main step ────────────────────────────────────────────────

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
    ) -> ResidualSnapshot:
        """
        Update all residuals + CUSUM + state machine.

        Parameters
        ----------
        t : float
            Current absolute time [s].
        innov, S_innov : from HillEKF (set both to None if no NAV meas)
        dv_est_H       : (3,) Δv error in Hill (x_post[3:6] - x_pred[3:6])
        P_vv           : (3,3) covariance of velocity block (post-update)
        burn_flag      : True if a burn was commanded during this step
        sigma_BR       : (3,) attitude error MRPs from attTrackingError
        omega_BR_B     : (3,) rate error in body frame [rad/s]
        hb_age         : time since last heartbeat [s]
        hb_nominal     : nominal heartbeat period [s]
        r_form         : scalar formation/task residual (>=0) or None if inactive
        r_peer         : scalar peer-consistency residual (>=0) or None if inactive
        """
        cfg = self.cfg

        # --- residuals ---
        nav_has_meas = (innov is not None) and (S_innov is not None)

        if nav_has_meas and cfg.use_nav_channel:
            self.r_nav = self._nav_residual(innov, S_innov)
        else:
            self.r_nav = 0.0

        self.r_prop = self._prop_residual(dv_est_H, P_vv) if burn_flag else 0.0
        self.r_att  = self._att_residual(sigma_BR, omega_BR_B)
        self.r_comm = self._comm_residual(hb_age, hb_nominal)

        self.r_form = float(r_form) if (r_form is not None) else 0.0
        self.r_peer = float(r_peer) if (r_peer is not None) else 0.0

        # --- CUSUMs (NAV / ATT / COMM / FORM / PEER) ---

        if cfg.use_nav_channel:
            if nav_has_meas:
                self.S_nav = self._update_cusum(
                    self.r_nav, self.S_nav, cfg.nu_nav, cfg.leak_nav
                )
            else:
                # no NAV update this step → just leak existing S_nav down
                self.S_nav = (1.0 - cfg.leak_nav) * self.S_nav
        else:
            self.S_nav = 0.0

        if cfg.use_att_channel:
            self.S_att = self._update_cusum(
                self.r_att, self.S_att, cfg.nu_att, cfg.leak_att
            )
        else:
            self.S_att = 0.0

        if cfg.use_comm_channel:
            self.S_comm = self._update_cusum(
                self.r_comm, self.S_comm, cfg.nu_comm, cfg.leak_comm
            )
        else:
            self.S_comm = 0.0

        if cfg.use_form_channel:
            if r_form is not None and r_form > 0.0:
                self.S_form = self._update_cusum(
                    self.r_form, self.S_form, cfg.nu_form, cfg.leak_form
                )
            else:
                self.S_form = (1.0 - cfg.leak_form) * self.S_form
        else:
            self.S_form = 0.0

        if cfg.use_peer_channel:
            if r_peer is not None and r_peer > 0.0:
                self.S_peer = self._update_cusum(
                    self.r_peer, self.S_peer, cfg.nu_peer, cfg.leak_peer
                )
            else:
                self.S_peer = (1.0 - cfg.leak_peer) * self.S_peer
        else:
            self.S_peer = 0.0

        # --- classify current pattern ---
        fault_candidate = self._classify_fault()

        # --- persistence and state machine ---
        if fault_candidate is FaultType.NONE:
            # No strong evidence – decay towards HEALTHY
            self.k_suspect = max(self.k_suspect - 1, 0)
            self.k_fault   = max(self.k_fault   - 1, 0)
            if self.k_suspect == 0 and self.k_fault == 0:
                self.state.fault_type = FaultType.NONE
        else:
            self.k_suspect += 1
            if self.k_suspect >= cfg.n_suspect:
                if self.state.state is FDIState.HEALTHY:
                    self.state.state = FDIState.SUSPECT
                    self.state.t_last_change = t
                self.state.fault_type = fault_candidate

            if self.state.state is FDIState.SUSPECT:
                self.k_fault += 1
                if self.k_fault >= cfg.n_fault:
                    self.state.state = FDIState.FAULTED
                    self.state.fault_type = fault_candidate
                    self.state.t_last_change = t

        # COMM → DEAD transition
        if (
            self.state.state in (FDIState.SUSPECT, FDIState.FAULTED)
            and fault_candidate is FaultType.COMM
            and self.r_comm >= cfg.comm_trip_multiple
        ):
            self.state.state = FDIState.DEAD
            self.state.fault_type = FaultType.COMM
            self.state.t_last_change = t

        # Confidence: saturating function of max channel severity
        max_severity = max(
            self.S_nav,
            self.S_att,
            self.S_comm,
            self.S_form,
            self.S_peer,
            self.r_prop,
        )
        self.state.confidence = 1.0 - np.exp(
            -max_severity / max(1.0, cfg.h_nav)
        )

        snap = ResidualSnapshot(
            t=t,
            r_nav=self.r_nav,
            r_prop=self.r_prop,
            r_att=self.r_att,
            r_comm=self.r_comm,
            r_form=self.r_form,
            r_peer=self.r_peer,
            state=self.state,
        )
        self.last_snapshot = snap
        return snap
