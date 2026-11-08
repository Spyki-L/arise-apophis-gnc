#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spin_estimator.py — onboard spin-state estimation + spin-aware NAC re-point (F6).

If the target's spin/tumble state at final approach differs from the a-priori
prediction, the surface regions that face the fleet change, so the deputies must
re-point / re-station to keep the high-priority region resolved at 0.15 m NAC.

  SpinEstimator  — fits the body spin rate + phase from a short arc of
                   imagery-derived phase observations (apparent rotation of a
                   tracked feature/limb), predicts the body orientation at CA,
                   detects an off-nominal rate, and gates the estimate against
                   angular-momentum conservation. Advisory under RTA: it never
                   actuates directly; it feeds the re-optimizer.
  nac_roi_coverage / optimize_for_spin — recompute the required viewing geometry
                   for the (re-)estimated CA orientation and re-point/re-station
                   the fleet (the F5 bounded hill-climb, spin-parameterised).

Principal-axis spin model; full NPA (Euler) precession is the higher-fidelity
refinement. Mission-agnostic. Pure numpy.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.linalg as la


def _unit(v):
    v = np.asarray(v, float); n = la.norm(v); return v/n if n > 0 else v


def rot_about_axis(axis, angle):
    """Rodrigues rotation matrix about unit axis by angle [rad]."""
    a = _unit(axis); c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s*K + (1-c)*(K@K)


@dataclass
class SpinEstimator:
    axis: np.ndarray
    omega_prior: float
    t0: float = 0.0

    def __post_init__(self):
        self.axis = _unit(self.axis)
        self.omega = float(self.omega_prior); self.theta0 = 0.0; self.resid = 0.0

    def update(self, t_obs, phi_obs):
        """Linear least-squares fit of phi(t) = theta0 + omega*(t - t0) to the observed phase arc."""
        t = np.asarray(t_obs, float); phi = np.unwrap(np.asarray(phi_obs, float))
        A = np.column_stack([np.ones_like(t), t - self.t0])
        sol, *_ = la.lstsq(A, phi, rcond=None)
        self.theta0, self.omega = float(sol[0]), float(sol[1])
        self.resid = float(np.sqrt(np.mean((A @ sol - phi)**2)))
        return self.omega, self.theta0

    def predict_phase(self, t):
        return self.theta0 + self.omega * (np.asarray(t, float) - self.t0)

    def predict_R(self, t):
        return rot_about_axis(self.axis, float(self.predict_phase(t)))

    def changed(self, tol_radps):
        return abs(self.omega - self.omega_prior) > tol_radps

    def am_consistent(self, I_axis, L_expected, rel_tol=0.1):
        """|L| = I_axis * omega must match the conserved expectation within rel_tol (sanity gate)."""
        return abs(I_axis * self.omega - L_expected) <= rel_tol * abs(L_expected)


# ── spin-aware NAC coverage of a region of interest (ROI) ────────────────────

def make_roi_facets(Rt, u_roi_body, half_angle_deg, n=400):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2*i/n); th = np.pi*(1+5**0.5)*i
    pts = np.stack([np.cos(th)*np.sin(phi), np.sin(th)*np.sin(phi), np.cos(phi)], 1)
    u = _unit(u_roi_body); sel = pts @ u >= np.cos(np.radians(half_angle_deg))
    return Rt*pts[sel], pts[sel]                      # ROI facet centers (body), outward normals (body)


def nac_roi_coverage(stations, centers_body, normals_body, body_R, target_center=np.zeros(3),
                     R_nac=3000., incidence_max_deg=25.):
    """Fraction of ROI facets resolved at 0.15 m NAC: a deputy within R_nac viewing the facet
    within the incidence cone (so it resolves the surface, not a grazing glance)."""
    stations = np.asarray(stations, float); cmax = np.cos(np.radians(incidence_max_deg)); cov = 0
    for cb, nb in zip(centers_body, normals_body):
        c = target_center + body_R @ cb; nrm = body_R @ nb
        seen = False
        for st in stations:
            los = st - c; rng = la.norm(los)
            if rng > R_nac:
                continue
            if (los/rng) @ nrm >= cmax:
                seen = True; break
        cov += seen
    return cov / max(len(centers_body), 1)


def optimize_for_spin(nominal, active, centers_body, normals_body, body_R, reach,
                      n_iter=250, seed=0, **kw):
    """Bounded hill-climb of active deputy stations within reach boxes (F3 tube) to maximize
    NAC ROI coverage for the given body orientation. Deterministic WCET."""
    rng = np.random.default_rng(seed); nominal = np.asarray(nominal, float)
    x = {d: nominal[d].copy() for d in active}
    def cov(xx):
        return nac_roi_coverage([xx[d] for d in active], centers_body, normals_body, body_R, **kw)
    best = cov(x)
    for _ in range(n_iter):
        d = active[rng.integers(len(active))]
        cand = {k: v.copy() for k, v in x.items()}
        cand[d] = np.clip(x[d] + rng.normal(0, reach[d]*0.35, 3), nominal[d]-reach[d], nominal[d]+reach[d])
        c = cov(cand)
        if c > best:
            x = cand; best = c
    return np.array([x[d] for d in active]), float(best)
