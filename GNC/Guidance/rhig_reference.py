#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rhig_reference.py
=================

Lightweight wrapper around "RHIG-style" rendezvous NPZ files used in the
Apophis high-speed flyby / rendezvous architecture.

The goal is to provide a clean interface for **guidance / attitude reference**
tasks, decoupled from Basilisk itself:

- Time-aligned interpolation of deputy / asteroid states
- Access to line-of-sight (boresight) vectors
- Access to reference attitudes σ_RN(t) and q_RN(t) if present
- TRIAD-based fallback when quaternion histories are absent

Expected NPZ contents (Script A / fleet NPZ conventions)
--------------------------------------------------------

At minimum, a RHIG bundle is expected to contain:

- t          : (N,) float, absolute seconds from some epoch
- rD_N       : (N,3) deputy ECI positions [m] (optional but recommended)
- vD_N       : (N,3) deputy ECI velocities [m/s]
- rA_N       : (N,3) target (e.g. Apophis) ECI positions [m]
- vA_N       : (N,3) target ECI velocities [m/s]
- bRef_N     : (N,3) LOS / boresight vectors (unit) in ECI (optional)
- q_RN       : (N,4) scalar-first quaternions N→R (reference attitude) (optional)
- camMount_B : (3,) camera boresight in body frame (optional)
- t0         : (1,) deployment offset [s] (optional; used as convenience)

This module does **not** depend on Basilisk. It is safe to import
from pure-GNC code, and can be used to feed any simulation framework.

Usage
-----

    from gnc.guidance.rhig_reference import RHIGBundle

    ref = RHIGBundle("apophis_rendezvous_full_D0.npz")

    # reference attitude (MRPs) at some absolute time
    sigma_RN = ref.sigma_RN_at(t_abs)

    # boresight in ECI
    b_hat = ref.boresight_N_at(t_abs)

    # deputy / asteroid state in ECI
    rD, vD = ref.deputy_state_N_at(t_abs)
    rA, vA = ref.target_state_N_at(t_abs)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Iterable

import numpy as np
import numpy.linalg as la


# ───────────────────────── basic attitude helpers ─────────────────────────

def _normalize(v: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    v = np.asarray(v, float).reshape(-1)
    n = la.norm(v)
    return (v / n if n >= eps else np.zeros_like(v)), n


def triad_from_boresight_and_tangent(los_N: np.ndarray, vel_N: np.ndarray) -> np.ndarray:
    """
    Construct a body frame from a line-of-sight and a velocity-like tangent.

    Given LOS (deputy→target) and an inertial velocity, define:

        z_B := unit(LOS)
        x_B := V projected onto plane ⟂ z_B  (fallback to arbitrary orthogonal)
        y_B := z_B × x_B

    Returns
    -------
    C_NB : (3,3) ndarray
        DCM mapping from inertial N to body B: v_B = C_NB v_N.
    """
    zB, _ = _normalize(los_N)
    v_perp = vel_N - np.dot(vel_N, zB) * zB
    xB, nx = _normalize(v_perp)
    if nx < 1e-10:
        # LOS and velocity nearly aligned -> pick any orthogonal axis
        tmp = np.array([1.0, 0.0, 0.0]) if abs(zB[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        xB = _normalize(np.cross(tmp, zB))[0]
    yB = np.cross(zB, xB)
    xB = _normalize(np.cross(yB, zB))[0]
    yB = _normalize(np.cross(zB, xB))[0]
    return np.column_stack((xB, yB, zB))


def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    """DCM → scalar-first quaternion [q0, q1, q2, q3]."""
    C = np.asarray(C, float).reshape(3, 3)
    tr = np.trace(C)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        q0 = 0.25 * S
        q1 = (C[2, 1] - C[1, 2]) / S
        q2 = (C[0, 2] - C[2, 0]) / S
        q3 = (C[1, 0] - C[0, 1]) / S
    else:
        d = np.array([C[0, 0], C[1, 1], C[2, 2]])
        i = int(np.argmax(d))
        if i == 0:
            S = np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2]) * 2.0
            q0 = (C[2, 1] - C[1, 2]) / S
            q1 = 0.25 * S
            q2 = (C[0, 1] + C[1, 0]) / S
            q3 = (C[0, 2] + C[2, 0]) / S
        elif i == 1:
            S = np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2]) * 2.0
            q0 = (C[0, 2] - C[2, 0]) / S
            q1 = (C[0, 1] + C[1, 0]) / S
            q2 = 0.25 * S
            q3 = (C[1, 2] + C[2, 1]) / S
        else:
            S = np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1]) * 2.0
            q0 = (C[1, 0] - C[0, 1]) / S
            q1 = (C[0, 2] + C[2, 0]) / S
            q2 = (C[1, 2] + C[2, 1]) / S
            q3 = 0.25 * S
    q = np.array([q0, q1, q2, q3], float)
    return q / la.norm(q)


def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [q0,q1,q2,q3] → MRPs σ."""
    q = np.asarray(q, float).reshape(4)
    q0 = q[0]
    qv = q[1:]
    denom = 1.0 + q0
    if denom < 1e-12:
        # shadow set
        denom = 1.0 - q0
        return -qv / max(denom, 1e-12)
    return qv / denom


# ───────────────────────── data wrapper ─────────────────────────

@dataclass
class RHIGBundle:
    """
    Wrapper for a single RHIG rendezvous NPZ file.

    Attributes
    ----------
    path : Path
        Underlying NPZ path.
    t : (N,) ndarray
        Time stamps [s] since some reference epoch.
    rD, vD : (N,3) or None
        Deputy ECI state history [m], [m/s].
    rA, vA : (N,3) or None
        Target (e.g. Apophis) ECI state history [m], [m/s].
    bRef : (N,3) or None
        LOS vectors in ECI; if absent, rebuilt as unit(rA − rD).
    has_q : bool
        True if usable q_RN is present.
    qRN : (N,4) or None
        Scalar-first quaternions N→R (reference attitude).
    cam_b_B : (3,) ndarray
        Camera boresight in body frame (defaults to +Z_B).
    t0 : float
        Deployment epoch offset (if present; else t[0]).
    """
    path: Path
    t: np.ndarray
    rD: Optional[np.ndarray]
    vD: Optional[np.ndarray]
    rA: Optional[np.ndarray]
    vA: Optional[np.ndarray]
    bRef: Optional[np.ndarray]
    has_q: bool
    qRN: Optional[np.ndarray]
    cam_b_B: np.ndarray
    t0: float

    # ---- construction helper ------------------------------------------------
    def __init__(self, path: str | Path, prefer_qrn: bool = True):
        self.path = Path(path)
        n = np.load(self.path, allow_pickle=True)

        self.t = np.asarray(n["t"], float)
        self.rD = np.asarray(n["rD_N"], float) if "rD_N" in n else None
        self.vD = np.asarray(n["vD_N"], float) if "vD_N" in n else None
        self.rA = np.asarray(n["rA_N"], float) if "rA_N" in n else None
        self.vA = np.asarray(n["vA_N"], float) if "vA_N" in n else None

        if "bRef_N" in n:
            self.bRef = np.asarray(n["bRef_N"], float)
        elif self.rA is not None and self.rD is not None:
            diff = self.rA - self.rD
            norms = la.norm(diff, axis=1)[:, None]
            self.bRef = diff / np.clip(norms, 1e-12, None)
        else:
            self.bRef = None

        self.has_q = ("q_RN" in n) and prefer_qrn
        self.qRN = np.asarray(n["q_RN"], float) if self.has_q else None

        self.cam_b_B = np.asarray(
            n["camMount_B"], float
        ) if "camMount_B" in n else np.array([0.0, 0.0, 1.0])

        if "t0" in n:
            self.t0 = float(np.asarray(n["t0"], float).ravel()[0])
        else:
            self.t0 = float(self.t[0])

    # ---- interpolation helpers ----------------------------------------------
    def _interp(self, arr: np.ndarray, t_abs: float) -> np.ndarray:
        """Linear interpolation helper over self.t."""
        t = self.t
        if t_abs <= t[0]:
            return arr[0]
        if t_abs >= t[-1]:
            return arr[-1]
        j = np.searchsorted(t, t_abs) - 1
        f = (t_abs - t[j]) / (t[j + 1] - t[j] + 1e-15)
        return (1.0 - f) * arr[j] + f * arr[j + 1]

    # ---- ECI states ---------------------------------------------------------
    def deputy_state_N_at(self, t_abs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (rD, vD) in ECI [m], [m/s] at t_abs."""
        if self.rD is None or self.vD is None:
            raise RuntimeError("Deputy state (rD_N, vD_N) not available in NPZ.")
        return self._interp(self.rD, t_abs), self._interp(self.vD, t_abs)

    def target_state_N_at(self, t_abs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (rA, vA) in ECI [m], [m/s] at t_abs."""
        if self.rA is None or self.vA is None:
            raise RuntimeError("Target state (rA_N, vA_N) not available in NPZ.")
        return self._interp(self.rA, t_abs), self._interp(self.vA, t_abs)

    # ---- LOS / boresight ----------------------------------------------------
    def boresight_N_at(self, t_abs: float) -> np.ndarray:
        """
        LOS / boresight direction in ECI at t_abs.

        If bRef_N is present, that is used; otherwise uses unit(rA − rD).
        """
        if self.bRef is not None:
            return self._interp(self.bRef, t_abs)
        if self.rA is None or self.rD is None:
            raise RuntimeError("Cannot derive LOS: missing rA_N/rD_N in NPZ.")
        rA, rD = self.target_state_N_at(t_abs)[0], self.deputy_state_N_at(t_abs)[0]
        return _normalize(rA - rD)[0]

    # ---- attitude references -----------------------------------------------
    def q_RN_at(self, t_abs: float) -> np.ndarray:
        """
        Quaternion N→R (scalar-first) at t_abs.

        If q_RN is not available, this is reconstructed from TRIAD using
        boresight_N_at and vD_N.
        """
        if self.has_q and self.qRN is not None:
            return self._interp(self.qRN, t_abs)

        # TRIAD fallback: R is a "body" aligned to LOS & tangent
        bN = self.boresight_N_at(t_abs)
        if self.vD is None:
            raise RuntimeError("vD_N not available; cannot TRIAD-construct attitude.")
        vN = self._interp(self.vD, t_abs)
        C_NB = triad_from_boresight_and_tangent(bN, vN)
        return dcm_to_quat(C_NB.T)  # N→B/R

    def sigma_RN_at(self, t_abs: float) -> np.ndarray:
        """
        MRPs σ_RN (N→R) at t_abs.

        Uses stored q_RN if present; otherwise TRIAD-based reconstruction.
        """
        q = self.q_RN_at(t_abs)
        return quat_to_mrp(q)


# ───────────────────────── fleet helper (optional) ─────────────────────────

def discover_rhig_npz(
    base_dir: str | Path,
    full_pattern: str = "apophis_rendezvous_full_D*.npz",
    viz_pattern: str = "viz_inputs_D*.npz",
) -> list[Path]:
    """
    Discover RHIG-style NPZ files under base_dir.

    Looks first for "full" rendezvous bundles matching `full_pattern`, and
    falls back to "viz_inputs" style files if none are found.

    Returns a (possibly empty) list of Paths.
    """
    base = Path(base_dir)
    full = sorted(base.glob(full_pattern))
    if full:
        return full
    viz = sorted(base.glob(viz_pattern))
    return viz

