#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_selftest_f3.py — F3 proofs for the FORM 4D reachable tube.

Run:  python3 run_selftest_f3.py
  T1 nominal slot is inside the tube (reachable, d<1)
  T2 thruster degradation shrinks E_ctrl → slot exits the tube (d>1) → FORM fault
  T3 nav degradation inflates E_know → tube grows → slot Mahalanobis decreases
  T4 spin change moves the required slot → slot exits the tube (d>1)
  T5 shorter time-to-go (CA-geometry change) → less leverage → harder (d up)
  T6 recoverability frontier is monotone (more authority → can act later)
  T7 the 4D shape is a propagating set of cross-sections
"""
import numpy as np
from reachability import FormTube, recoverability_frontier, propagate_pos

n = 1.0e-4
t_go = 3600.0
rho = np.array([1000., 2000., -300.])
rhod = np.array([0., 0., 0.])
P0 = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])          # pos sigma 1 m, vel sigma 1e-3 m/s
dv = np.array([0.1, 0.1, 0.1])                         # 0.1 m/s per axis remaining
c = propagate_pos(rho, rhod, n, t_go)
slot = c + np.array([200., 0., 0.])                    # 200 m offset from uncontrolled terminal

base = FormTube(n=n, rho=rho, rhod=rhod, P0=P0, dv_axis=dv, t_go=t_go, slot=slot)
ok = 0; tot = 0
def check(name, cond, extra=""):
    global ok, tot; tot += 1
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else "")); ok += int(bool(cond))

d0 = base.r_form()
check("T1 nominal slot reachable (d<1)", d0 < 1.0, f"d={d0:.3f}")
check("T2 thruster degradation -> slot exits (d>1)", base.with_authority(dv*0.1).r_form() > 1.0,
      f"d={base.with_authority(dv*0.1).r_form():.3f}")
d_nav = base.with_nav_cov(P0*100.).r_form()
check("T3 nav degradation grows tube (d decreases)", d_nav < d0, f"d_nav={d_nav:.3f} < d0={d0:.3f}")
d_spin = base.with_slot(c + np.array([4000., 0., 0.])).r_form()
check("T4 spin change moves slot out (d>1)", d_spin > 1.0, f"d={d_spin:.3f}")
d_short = base.with_t_go(1800.).slot_mahalanobis(1800.)
check("T5 shorter time-to-go, harder (d up)", d_short > d0, f"d_short={d_short:.3f} > d0={d0:.3f}")

g_grid = [0.1, 0.3, 0.6, 1.0]
tgo_grid = np.linspace(600., 10000., 30)
fr = recoverability_frontier(base, g_grid, tgo_grid)
vals = [fr[g] for g in g_grid]                          # g ascending (more authority)
finite = [v for v in vals if v is not None]
first = next((i for i, v in enumerate(vals) if v is not None), len(vals))
no_none_after = all(v is not None for v in vals[first:])     # once reachable, stays reachable
decreasing = all(a >= b for a, b in zip(finite, finite[1:])) # more authority -> act later (smaller t_go)
check("T6 frontier monotone (more authority -> act later)", no_none_after and decreasing and len(finite) >= 2,
      f"latest t_go by g: {[None if v is None else round(v) for v in vals]}")

shp = base.shape_4d()
check("T7 4D shape = propagating cross-sections", len(shp) == base.n_grid and shp[-1][1].shape == (3,))

print(f"\n{ok}/{tot} passed")
