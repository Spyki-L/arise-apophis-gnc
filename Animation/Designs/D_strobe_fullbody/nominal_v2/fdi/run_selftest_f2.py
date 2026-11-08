#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_selftest_f2.py — F2 proofs: mission-agnostic core + RTA simplex.

Run:  python3 run_selftest_f2.py
  T1 both profiles load and expose the generic schema
  T2 profiles differ in mission-specifics (target/fleet/objective)
  T3 the SAME FDI core runs unmodified on both profiles (healthy on nominal)
  T4 RTA accepts a safe AC proposal
  T5 RTA falls back to VSC on: unsafe / absent / invalid / late AC
  T6 the core contains no hard-coded mission strings (grep-style introspection)
"""
from __future__ import annotations
import os
import numpy as np
from mission_profile import load_profile, fdi_config_from_profile
from rta import RTAExecutive, Controller, RTAReason
from fdi_core import FDICore, FDIState, FaultType

HERE = os.path.dirname(os.path.abspath(__file__))
I3 = np.eye(3); Z3 = np.zeros(3)


def _run_healthy(cfg, n_steps=200):
    """Drive one deputy's core with nominal inputs; expect HEALTHY."""
    fdi = FDICore(cfg); fdi.reset(0.0); snap = None
    for k in range(n_steps):
        snap = fdi.step(t=float(k), innov=Z3, S_innov=I3, dv_est_H=None, P_vv=None,
                        burn_flag=False, sigma_BR=Z3, omega_BR_B=Z3,
                        hb_age=60.0, hb_nominal=60.0, r_form=None, r_peer=None)
    return snap


def T1_load():
    ap = load_profile(os.path.join(HERE, "profiles", "apophis.json"))
    toy = load_profile(os.path.join(HERE, "profiles", "toy_inspect.json"))
    assert ap.fleet.n_deputies == 8 and toy.fleet.n_deputies == 4
    assert ap.target.kind == "shape_spin" and toy.target.kind == "point"
    return f"T1 profiles load: apophis(n=8,shape_spin)  toy(n=4,point)"


def T2_differ():
    ap = load_profile(os.path.join(HERE, "profiles", "apophis.json"))
    toy = load_profile(os.path.join(HERE, "profiles", "toy_inspect.json"))
    assert ap.objective.name != toy.objective.name
    assert ap.fleet.bus != toy.fleet.bus
    assert ap.encounter.v_rel_kms != toy.encounter.v_rel_kms
    return "T2 mission-specifics differ (objective/bus/encounter)"


def T3_same_core():
    results = {}
    for name in ("apophis", "toy_inspect"):
        p = load_profile(os.path.join(HERE, "profiles", f"{name}.json"))
        cfg = fdi_config_from_profile(p)         # only numbers come from the profile
        snap = _run_healthy(cfg)                  # identical core code path
        results[name] = snap.state.state
        assert snap.state.state is FDIState.HEALTHY, (name, snap.state.state)
    return f"T3 same core healthy on both: apophis={results['apophis'].name} toy={results['toy_inspect'].name}"


def _make_rta():
    # safety predicate: action (a dV vector magnitude) must be <= budget and finite
    def safety(action, ctx):
        try:
            return np.all(np.isfinite(action)) and float(np.linalg.norm(action)) <= ctx["dv_budget"]
        except Exception:
            return False
    def vsc(ctx):
        return np.zeros(3)   # safe baseline: do nothing (always feasible)
    def valid(action):
        return np.all(np.isfinite(np.asarray(action, float)))   # well-formedness
    return RTAExecutive(safety_check=safety, vsc_fallback=vsc, validity_check=valid)


def T4_rta_accept():
    rta = _make_rta(); ctx = {"dv_budget": 1.0}
    d = rta.decide(ctx, ac_proposal=np.array([0.1, 0.0, 0.0]))
    assert d.source is Controller.AC and d.reason is RTAReason.AC_ACCEPTED
    return "T4 RTA accepts safe AC proposal"


def T5_rta_fallbacks():
    ctx = {"dv_budget": 1.0}
    r1 = _make_rta(); d1 = r1.decide(ctx, ac_proposal=np.array([5.0, 0, 0]))                 # unsafe (>budget)
    r2 = _make_rta(); d2 = r2.decide(ctx, ac_proposal=None)                                   # absent
    r3 = _make_rta(); d3 = r3.decide(ctx, ac_proposal=np.array([np.nan, 0, 0]))               # invalid
    r4 = _make_rta(); d4 = r4.decide(ctx, ac_proposal=np.array([0.1, 0, 0]), ac_within_deadline=False)  # late
    assert d1.reason is RTAReason.AC_UNSAFE and d1.source is Controller.VSC
    assert d2.reason is RTAReason.AC_ABSENT and d2.source is Controller.VSC
    assert d3.reason is RTAReason.AC_INVALID and d3.source is Controller.VSC
    assert d4.reason is RTAReason.AC_LATE and d4.source is Controller.VSC
    assert np.allclose(d1.action, 0.0)   # VSC safe baseline used
    return "T5 RTA -> VSC on unsafe / absent / invalid / late"


def T6_no_mission_strings():
    # the CORE modules must not hard-code mission identifiers
    core = ["fdi_core.py", "rta.py", "reachability.py"]
    banned = ["apophis", "xb6", "espa", "strobe"]
    hits = []
    for fn in core:
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            continue
        low = open(path, "r", encoding="utf-8", errors="ignore").read().lower()
        for b in banned:
            if b in low:
                hits.append(f"{fn}:{b}")
    assert not hits, f"core references mission specifics: {hits}"
    return "T6 core modules contain no hard-coded mission strings"


if __name__ == "__main__":
    tests = [T1_load, T2_differ, T3_same_core, T4_rta_accept, T5_rta_fallbacks, T6_no_mission_strings]
    ok = 0
    for t in tests:
        try:
            print("  PASS  " + t()); ok += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(tests)} passed")
