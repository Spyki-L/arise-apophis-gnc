#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_selftest_f6.py — F6 proofs: onboard spin estimation + spin-aware NAC re-point.

  T1 spin estimator recovers the true rate from noisy imagery
  T2 estimator detects the spin change vs the a-priori prediction
  T3 angular-momentum consistency gate (accepts the true |L|, rejects a spurious one)
  T4 a spin change drops NAC 0.15 m coverage with fixed stations
  T5 spin-aware re-point recovers the NAC coverage
  T6 deadline gate (F3): re-point only if reachable in the remaining time-to-go, else hold
"""
import numpy as np
from spin_estimator import (SpinEstimator, rot_about_axis, make_roi_facets,
                            nac_roi_coverage, optimize_for_spin)
from reachability import cw_blocks_full

rng = np.random.default_rng(0)
axis = np.array([0, 0, 1.]); t0 = 0.0; t_ca = 600.
ok = 0; tot = 0
def check(name, cond, extra=""):
    global ok, tot; tot += 1
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else "")); ok += int(bool(cond))

w_true = np.radians(2.0)
est = SpinEstimator(axis=axis, omega_prior=np.radians(1.0), t0=t0)
t_obs = np.linspace(0, 120, 40)
est.update(t_obs, w_true*(t_obs-t0) + rng.normal(0, np.radians(0.3), t_obs.size))
check("T1 spin estimator recovers true rate from noisy imagery",
      abs(est.omega - w_true) < np.radians(0.1), f"est={np.degrees(est.omega):.3f} true=2.000 deg/s")
check("T2 estimator detects the spin change vs a-priori",
      est.changed(np.radians(0.3)) and abs(est.predict_phase(t_ca) - est.omega_prior*t_ca) > np.radians(60),
      f"CA phase a-priori={np.degrees(est.omega_prior*t_ca):.0f} est={np.degrees(est.predict_phase(t_ca)):.0f} deg")
check("T3 angular-momentum consistency gate (accept true, reject spurious)",
      est.am_consistent(100., 100.*w_true) and not est.am_consistent(100., 100.*np.radians(10.)),
      "true |L| accepted, 5x-off rejected")

Rt = 200.; cb, nb = make_roi_facets(Rt, [1, 0, 0], 22., n=300)
M = 8; ang = np.linspace(0, 2*np.pi, M, endpoint=False); shell = 1200.
geom = np.array([[shell*np.cos(a), shell*np.sin(a), 300*np.sin(2*a)] for a in ang])
reach = np.full(M, 1500.); kw = dict(R_nac=3000., incidence_max_deg=25.)
R_prior = rot_about_axis(axis, est.omega_prior*(t_ca-t0))
R_actual = est.predict_R(t_ca)
st0, c0 = optimize_for_spin(geom, list(range(M)), cb, nb, R_prior, reach, n_iter=300, seed=2, **kw)
nominal = geom.copy()
for d in range(M): nominal[d] = st0[d]
c_drop = nac_roi_coverage(nominal, cb, nb, R_actual, **kw)
st1, c_rec = optimize_for_spin(nominal, list(range(M)), cb, nb, R_actual, reach, n_iter=300, seed=4, **kw)
check("T4 spin change drops NAC coverage (fixed stations)", c_drop < c0 - 0.1, f"{c0:.2f} -> {c_drop:.2f}")
check("T5 spin-aware re-point recovers NAC coverage", c_rec > c_drop + 0.1 and c_rec > 0.9*c0,
      f"re-pointed={c_rec:.2f}")

n_mm = 1e-4
def reachable_radius(t_go, dv):
    _, Prv, _, _ = cw_blocks_full(n_mm, t_go)
    return np.linalg.norm(Prv) * dv
def gated_repoint(nominal, new, t_go, dv):
    rad = reachable_radius(t_go, dv)
    feasible = all(np.linalg.norm(new[i] - nominal[i]) <= rad for i in range(len(new)))
    return "re-point" if feasible else "hold pre-point"
dec_far = gated_repoint(nominal, st1, t_go=3600., dv=0.5)
dec_near = gated_repoint(nominal, st1, t_go=20., dv=0.5)
check("T6 deadline gate: re-point when reachable, hold when too late",
      dec_far == "re-point" and dec_near == "hold pre-point", f"far={dec_far} near={dec_near}")

print(f"\n{ok}/{tot} passed")
