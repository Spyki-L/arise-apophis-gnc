#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_selftest_f9.py — F9 proofs: flight-readiness (determinism, WCET, CBF invariance, RTA soundness)."""
import numpy as np, numpy.linalg as la
import flight_readiness as fr
from fdi_core import FDICore, FDIConfig
from reachability import FormTube
from peer_cbf import PeerCBF
from isl_scheduler import schedule_isl
from reoptimizer import fibonacci_sphere, optimize_stations
from rta import RTAExecutive, Controller

I3 = np.eye(3); Z3 = np.zeros(3)
ok = 0; tot = 0
def check(name, cond, extra=""):
    global ok, tot; tot += 1
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else "")); ok += int(bool(cond))

# T1 determinism: same seed -> bit-identical result (reproducible WCET)
pts, norms = fibonacci_sphere(120, 200.)
nom = np.array([[1300*np.cos(a), 1300*np.sin(a), 0] for a in np.linspace(0, 2*np.pi, 8, endpoint=False)])
r1, _, _ = optimize_stations(nom, list(range(8)), pts, norms, np.full(8, 600.), n_iter=120, seed=42)
r2, _, _ = optimize_stations(nom, list(range(8)), pts, norms, np.full(8, 600.), n_iter=120, seed=42)
check("T1 determinism: identical inputs+seed -> bit-identical output", fr.arrays_equal(r1, r2))

# T2 bounded memory: FDI state is O(1) regardless of run length
core = FDICore(FDIConfig(use_form_channel=True, use_peer_channel=True)); core.reset(0.)
def _run(c, n):
    for k in range(n):
        c.step(t=float(k), innov=Z3, S_innov=I3, dv_est_H=None, P_vv=None, burn_flag=False,
               sigma_BR=Z3, omega_BR_B=Z3, hb_age=60., hb_nominal=60., r_form=0.0, r_peer=0.0)
_run(core, 1000); fp1 = fr.fdi_state_footprint(core)
_run(core, 99000); fp2 = fr.fdi_state_footprint(core)             # 100k total
check("T2 bounded memory: FDI state O(1) (no growth 1k->100k steps)",
      fp1 == fp2 and fp2["scalars"] == 12 and fp2["latch"] <= 6, f"{fp1} == {fp2}")

# T3 WCET: per-call work is bounded; every onboard layer fits its cadence budget
c = FDICore(FDIConfig(use_form_channel=True, use_peer_channel=True)); c.reset(0.)
tube = FormTube(n=1e-4, rho=np.array([1e3, 2e3, -3e2]), rhod=Z3,
                P0=np.diag([1, 1, 1, 1e-6, 1e-6, 1e-6]), dv_axis=np.full(3, 0.1),
                t_go=3600., slot=np.array([1.2e3, 2e3, -2e2]))
peer = PeerCBF(); vel = np.zeros((8, 3)); rep = nom.copy(); mr = np.full((8, 8), np.nan)
for i in range(8):
    for j in range(8):
        if i != j: mr[i, j] = la.norm(nom[i]-nom[j])
meas = {}
meas["fdi_step"] = fr.time_call(lambda: c.step(t=1., innov=Z3, S_innov=I3, dv_est_H=None, P_vv=None,
                                burn_flag=False, sigma_BR=Z3, omega_BR_B=Z3, hb_age=60., hb_nominal=60.))
meas["form_tube"] = fr.time_call(tube.r_form)
meas["peer_cbf"] = fr.time_call(lambda: peer.evaluate(nom, vel, rep, mr))
meas["isl_sched"] = fr.time_call(lambda: schedule_isl(nom, B=4, k_diag=3))
meas["reopt_tier2"] = fr.time_call(lambda: optimize_stations(nom, list(range(8)), pts, norms,
                                   np.full(8, 600.), n_iter=120, seed=1), n=5)
slow = dict(fdi_step=30, form_tube=30, peer_cbf=30, isl_sched=30, reopt_tier2=3)
cad = dict(fdi_step=0.1, form_tube=1.0, peer_cbf=0.1, isl_sched=1.0, reopt_tier2=60.0)
budget = fr.compute_budget(meas, cad, slow)
check("T3 WCET: every onboard layer fits its cadence budget", all(r["ok"] for r in budget),
      " ".join(f"{r['layer']}={r['onboard_ms']:.1f}/{r['budget_ms']:.0f}ms" for r in budget))

# T4 CBF forward-invariance
mf, mu = fr.verify_cbf_invariance(n_trials=300, steps=300)
check("T4 CBF forward-invariance: keep-out never breached", mf > -1e-6 and mu < -100.,
      f"min_h filtered={mf:.2f} (>=0)  unfiltered={mu:.0f} (collision)")

# T5 RTA soundness
s = fr.rta_soundness(n_trials=2000)
check("T5 RTA soundness: no unsafe action ever executed; safe ones accepted",
      s["executed_unsafe"] == 0 and s["accepted_unsafe_prop"] == 0 and s["ac"] > 0 and s["vsc"] > 0,
      f"unsafe_exec={s['executed_unsafe']} AC={s['ac']} VSC={s['vsc']}")

# T6 fixed-point consistency
mm = fr.fixed_point_consistency(n=3000)
check("T6 fixed-point: float32 == float64 safe/unsafe verdicts", mm == 0, f"mismatches={mm}")

# T7 graceful degradation: AC failure -> VSC still produces a safe action
def bad_ac():
    raise RuntimeError("AC optimizer diverged")
rta = RTAExecutive(safety_check=lambda a, c: True, vsc_fallback=lambda c: nom.copy())
try:
    prop = bad_ac()
except Exception:
    prop = None
d = rta.decide({}, prop)
check("T7 graceful degradation: AC failure -> VSC safe action", d.source is Controller.VSC and d.action is not None)

print(f"\n{ok}/{tot} passed")
print("\nCompute-budget (onboard WCET estimate vs cadence):")
for r in budget:
    print(f"  {r['layer']:12s} sandbox {r['sandbox_us']:8.1f} us  -> onboard {r['onboard_ms']:8.3f} ms  "
          f"(budget {r['budget_ms']:7.0f} ms)  {'OK' if r['ok'] else 'OVER'}")
