#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_selftest_f8.py — F8 proofs: closed-loop Autonomy Executive over a whole mission.

  T1 deputy loss        -> autonomous reconfigure recovers coverage
  T2 actuation degrade  -> re-task within the reduced reach (F3 box)
  T3 spin change (reachable) -> re-point recovers
  T4 misbehaving deputy -> CBF-safe disposal (collision-free), removed from formation
  T5 too-late spin      -> deadline hold (RTA -> VSC)
  T6 zero ground intervention
  T7 RTA: no unsafe action executed; both AC accepts and VSC fallbacks exercised
  T8 mission preserved despite 2 lost + degrade + spin
"""
import os
import numpy as np, numpy.linalg as la
from mission_profile import load_profile
from autonomy_executive import AutonomyExecutive, Phase, Event
from reoptimizer import fibonacci_sphere
from spin_estimator import make_roi_facets, rot_about_axis

HERE = os.path.dirname(os.path.abspath(__file__))
profile = load_profile(os.path.join(HERE, "profiles", "apophis.json"))
wb_pts, wb_norms = fibonacci_sphere(120, 200.)
roi_cb, roi_nb = make_roi_facets(200., [1, 0, 0], 22., n=120)
M = 8; ang = np.linspace(0, 2*np.pi, M, endpoint=False); shell = 1300.
geom = np.array([[shell*np.cos(a), shell*np.sin(a), 350*np.sin(2*a)] for a in ang])
reach = np.full(M, 1500.)
kw_wb = dict(R_max=4000., th_min=8., th_max=45.); kw_roi = dict(R_nac=3000., incidence_max_deg=25.)

exe = AutonomyExecutive(profile, wb_pts, wb_norms, roi_cb, roi_nb, geom, reach,
                        d_safe=50., w_wb=0.6, w_roi=0.4, dv_axis=1.0, kw_wb=kw_wb, kw_roi=kw_roi)
nom, _ = exe._optimize(np.eye(3), n_iter=350, seed=3)
exe.nominal = nom.copy(); exe.stations = nom.copy()
cov0 = exe.mission_coverage()

d6 = nom[6]; others = [nom[j] for j in range(M) if j != 6]
nearest = min(others, key=lambda p: la.norm(p - d6))
v6 = 2.0 * (nearest - d6) / la.norm(nearest - d6)
fleet_vel = [np.zeros(3)]*M

timeline = [
    (Phase.COAST, 3000., Event.DEPUTY_LOSS, dict(lost=2, n_iter=120)),
    (Phase.COAST, 2500., Event.ACTUATION_DEGRADE, dict(dep=5, new_reach=60., n_iter=120)),
    (Phase.TERMINAL, 1500., Event.SPIN_CHANGE, dict(body_R=rot_about_axis([0, 0, 1], np.radians(60.)), n_iter=120)),
    (Phase.TERMINAL, 1200., Event.MISBEHAVING, dict(suspect=6, fleet_vel=fleet_vel, suspect_state=(d6, v6), n_iter=100)),
    (Phase.TERMINAL, 12., Event.SPIN_CHANGE, dict(body_R=rot_about_axis([0, 0, 1], np.radians(95.)), n_iter=120)),
]
res = exe.run_mission(timeline)
cov_final = exe.mission_coverage()

ok = 0; tot = 0
def check(name, cond, extra=""):
    global ok, tot; tot += 1
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else "")); ok += int(bool(cond))

check("T1 deputy loss -> autonomous reconfigure recovers",
      res[0]["action"] == "reconfigure" and res[0]["after"] > res[0]["dropped"],
      f"{res[0]['dropped']:.3f} -> {res[0]['after']:.3f}")
check("T2 actuation degrade -> re-task within the reduced reach",
      la.norm(exe.stations[5] - exe.nominal[5]) <= 60.*1.05 + 1e-6,
      f"D5 moved {la.norm(exe.stations[5]-exe.nominal[5]):.1f} m (<=60)")
check("T3 spin change (reachable) -> re-point recovers",
      res[2]["action"] == "reconfigure" and res[2]["after"] > res[2]["dropped"],
      f"{res[2]['dropped']:.3f} -> {res[2]['after']:.3f}")
check("T4 misbehaving deputy -> CBF-safe disposal (collision-free)",
      res[3]["disposed"] and res[3]["collision_free"] and (6 not in exe.active),
      f"disposed={res[3]['disposed']} active={sorted(exe.active)}")
check("T5 too-late spin change -> deadline hold (VSC)",
      "hold" in res[4]["action"] and res[4]["source"] == "VSC",
      f"action={res[4]['action']} source={res[4]['source']}")
check("T6 zero ground intervention", exe.ground_calls == 0, f"ground_calls={exe.ground_calls}")
check("T7 RTA: no unsafe executed, both AC + VSC exercised",
      exe.executed_unsafe == 0 and exe.rta.n_ac > 0 and exe.rta.n_vsc > 0,
      f"unsafe={exe.executed_unsafe} AC={exe.rta.n_ac} VSC={exe.rta.n_vsc}")
check("T8 mission preserved despite 2 lost + degrade + spin", cov_final > 0.55,
      f"nominal={cov0:.3f} final={cov_final:.3f}")

print(f"\n{ok}/{tot} passed")
