#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_selftest_p01.py — deterministic, pure-numpy proofs for the causal core (P0/P1).
(Fresh-filename copy of _test_fdi_core.py to dodge OneDrive edit-sync lag on the mount.)

Run:  python3 run_selftest_p01.py
  T1 healthy nominal stays HEALTHY
  T2 PROP fix: pure prop fault → PROP (FIXED) vs NONE (LEGACY)  [the 0% defect]
  T3 NAV sanity: both → NAV
  T4 FORM/PEER latch: PEER first, FORM transient earlier → FIXED FORM vs LEGACY PEER [99.5% mislabel]
  T5 NAV_GLOBAL: FORM high / PEER quiet → NAV
  T6 PEER_ONLY:  PEER high / FORM quiet → PEER
  T7 causality: state@160 independent of steps>160
"""
from __future__ import annotations
import numpy as np
from fdi_core import FDICore, FDIConfig, FDIState, FaultType

I3 = np.eye(3); Z3 = np.zeros(3)


def _cfg(legacy: bool, form=False, peer=False) -> FDIConfig:
    return FDIConfig(
        use_nav_channel=True, use_att_channel=True, use_comm_channel=True,
        use_form_channel=form, use_peer_channel=peer,
        nu_form=3.0, leak_form=0.02, h_form=30.0,
        nu_peer=3.0, leak_peer=0.05, h_peer=20.0,
        legacy_isolation=legacy,
    )


def _step(fdi, t, *, innov=Z3, dv=None, P=None, burn=False,
          sBR=Z3, wBR=Z3, hb=60.0, hbn=60.0, rf=None, rp=None):
    return fdi.step(t=t, innov=innov, S_innov=I3, dv_est_H=dv, P_vv=P, burn_flag=burn,
                    sigma_BR=sBR, omega_BR_B=wBR, hb_age=hb, hb_nominal=hbn, r_form=rf, r_peer=rp)


def run_stream(cfg, steps):
    fdi = FDICore(cfg); fdi.reset(t0=0.0); snap = None
    for k, kw in enumerate(steps):
        snap = _step(fdi, float(k), **kw)
    return fdi, snap


def _prop_stream(n=300, t_fault=100):
    steps = []
    for k in range(n):
        if k % 10 == 0:
            dv = np.array([8.0, 0, 0]) if k >= t_fault else np.array([1.0, 0, 0])
            steps.append(dict(dv=dv, P=I3, burn=True))
        else:
            steps.append(dict())
    return steps


def T1_healthy():
    _, s = run_stream(_cfg(False, True, True), [dict() for _ in range(300)])
    assert s.state.state is FDIState.HEALTHY and s.state.fault_type is FaultType.NONE, s.state
    return "T1 healthy -> HEALTHY/NONE"


def T2_prop_fix():
    steps = _prop_stream()
    _, fx = run_stream(_cfg(False), steps)
    _, lg = run_stream(_cfg(True), steps)
    assert fx.state.fault_type is FaultType.PROP, f"FIXED got {fx.state.fault_type}"
    assert lg.state.fault_type is FaultType.NONE, f"LEGACY got {lg.state.fault_type}"
    return f"T2 PROP: FIXED={fx.state.fault_type.name}/{fx.state.state.name}  LEGACY={lg.state.fault_type.name} (was 0% defect)"


def T3_nav():
    steps = [dict(innov=(np.array([5., 5., 5.]) if k >= 100 else Z3)) for k in range(300)]
    _, fx = run_stream(_cfg(False), steps); _, lg = run_stream(_cfg(True), steps)
    assert fx.state.fault_type is FaultType.NAV and lg.state.fault_type is FaultType.NAV
    return "T3 NAV: both -> NAV"


def T4_formpeer():
    steps = [dict(rf=(8.0 if 60 <= k < 75 else 0.0), rp=(12.0 if k >= 50 else 0.0)) for k in range(260)]
    _, fx = run_stream(_cfg(False, True, True), steps)
    _, lg = run_stream(_cfg(True, True, True), steps)
    assert fx.state.fault_type is FaultType.FORM, f"FIXED got {fx.state.fault_type}"
    assert lg.state.fault_type is FaultType.PEER, f"LEGACY got {lg.state.fault_type}"
    return f"T4 FORM_LOCAL latch: FIXED={fx.state.fault_type.name}  LEGACY={lg.state.fault_type.name} (the 99.5% mislabel)"


def T5_nav_global():
    steps = [dict(rf=(40.0 if k >= 50 else 0.0), rp=0.0) for k in range(220)]
    _, fx = run_stream(_cfg(False, True, True), steps)
    assert fx.state.fault_type is FaultType.NAV, fx.state.fault_type
    return "T5 NAV_GLOBAL (FORM high, PEER quiet) -> NAV"


def T6_peer_only():
    steps = [dict(rf=0.0, rp=(12.0 if k >= 50 else 0.0)) for k in range(220)]
    _, fx = run_stream(_cfg(False, True, True), steps)
    assert fx.state.fault_type is FaultType.PEER, fx.state.fault_type
    return "T6 PEER_ONLY (PEER high, FORM quiet) -> PEER"


def T7_causality():
    full = _prop_stream(300, 100)
    fa, _ = run_stream(_cfg(False), full[:160])
    fc = FDICore(_cfg(False)); fc.reset(0.0); snap160 = None
    for k, kw in enumerate(full):
        s = _step(fc, float(k), **kw)
        if k == 159:
            snap160 = (fc.S_prop, s.state.state, s.state.fault_type)
    a = (fa.S_prop, fa.state.state, fa.state.fault_type)
    assert abs(a[0] - snap160[0]) < 1e-9 and a[1] is snap160[1] and a[2] is snap160[2], (a, snap160)
    return "T7 causality: state@160 independent of steps>160"


if __name__ == "__main__":
    tests = [T1_healthy, T2_prop_fix, T3_nav, T4_formpeer, T5_nav_global, T6_peer_only, T7_causality]
    ok = 0
    for t in tests:
        try:
            print("  PASS  " + t()); ok += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{ok}/{len(tests)} passed")
