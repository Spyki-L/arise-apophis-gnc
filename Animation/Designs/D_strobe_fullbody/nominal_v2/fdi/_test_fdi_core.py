#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_test_fdi_core.py — deterministic, pure-numpy proofs for the causal core (P0/P1).

Run:  python3 _test_fdi_core.py
Proves, with controlled synthetic residual streams:
  T1 healthy nominal stays HEALTHY (no false alarm)
  T2 PROP fix: a pure propulsion fault isolates to PROP (FIXED) but is missed/NONE (LEGACY)
  T3 NAV sanity: both legacy and fixed isolate a NAV fault to NAV
  T4 FORM/PEER latch: PEER fires first; FORM evidence earlier in the episode is not
     forgotten → FIXED settles FORM while LEGACY (instantaneous) settles PEER
  T5 NAV_GLOBAL: FORM high / PEER quiet → settled NAV (both)
  T6 PEER_ONLY:  PEER high / FORM quiet → settled PEER
  T7 causality: state after k steps is identical whether or not future steps exist
"""
from __future__ import annotations
import numpy as np
from fdi_core import FDICore, FDIConfig, FDIState, FaultType

I3 = np.eye(3)
Z3 = np.zeros(3)


def _cfg(legacy: bool, form=False, peer=False) -> FDIConfig:
    return FDIConfig(
        use_nav_channel=True, use_att_channel=True, use_comm_channel=True,
        use_form_channel=form, use_peer_channel=peer,
        # form/peer detection thresholds (match the main script's intent)
        nu_form=3.0, leak_form=0.02, h_form=30.0,
        nu_peer=3.0, leak_peer=0.05, h_peer=20.0,
        legacy_isolation=legacy,
    )


def _quiet_step(fdi, t, *, innov=Z3, dv=None, P=None, burn=False,
                sBR=Z3, wBR=Z3, hb=60.0, hbn=60.0, rf=None, rp=None):
    return fdi.step(t=t, innov=innov, S_innov=I3, dv_est_H=dv, P_vv=P, burn_flag=burn,
                    sigma_BR=sBR, omega_BR_B=wBR, hb_age=hb, hb_nominal=hbn, r_form=rf, r_peer=rp)


def run_stream(cfg, steps):
    """steps: list of kwargs dicts for _quiet_step; returns final snapshot."""
    fdi = FDICore(cfg); fdi.reset(t0=0.0)
    snap = None
    for k, kw in enumerate(steps):
        snap = _quiet_step(fdi, float(k), **kw)
    return fdi, snap


def T1_healthy():
    steps = [dict() for _ in range(300)]
    _, snap = run_stream(_cfg(legacy=False, form=True, peer=True), steps)
    assert snap.state.state is FDIState.HEALTHY, snap.state.state
    assert snap.state.fault_type is FaultType.NONE
    return "T1 healthy → HEALTHY/NONE"


def _prop_stream(n=300, t_fault=100):
    steps = []
    for k in range(n):
        burn = (k % 10 == 0)
        if burn:
            dv = np.array([8.0, 0, 0]) if k >= t_fault else np.array([1.0, 0, 0])
            steps.append(dict(dv=dv, P=I3, burn=True))
        else:
            steps.append(dict())
    return steps


def T2_prop_fix():
    steps = _prop_stream()
    _, fixed = run_stream(_cfg(legacy=False), steps)
    _, legacy = run_stream(_cfg(legacy=True), steps)
    assert fixed.state.fault_type is FaultType.PROP, f"FIXED got {fixed.state.fault_type}"
    assert fixed.state.state in (FDIState.SUSPECT, FDIState.FAULTED)
    assert legacy.state.fault_type is FaultType.NONE, f"LEGACY got {legacy.state.fault_type}"
    return f"T2 PROP: FIXED={fixed.state.fault_type.name}/{fixed.state.state.name}  LEGACY={legacy.state.fault_type.name} (was the 0% defect)"


def T3_nav_sanity():
    steps = []
    for k in range(300):
        innov = np.array([5.0, 5.0, 5.0]) if k >= 100 else Z3   # 5σ bias (S_innov=I)
        steps.append(dict(innov=innov))
    _, fixed = run_stream(_cfg(legacy=False), steps)
    _, legacy = run_stream(_cfg(legacy=True), steps)
    assert fixed.state.fault_type is FaultType.NAV, fixed.state.fault_type
    assert legacy.state.fault_type is FaultType.NAV, legacy.state.fault_type
    return "T3 NAV: both → NAV"


def T4_formpeer_latch():
    # PEER high & sustained from k=50; FORM is a modest, TRANSIENT bump 60..75 that
    # crosses h_form then leaks back below it well before the end.  At the final step
    # only PEER is instantaneously bad → LEGACY settles PEER; the latch remembers FORM
    # ever fired during the episode → FIXED settles FORM.
    steps = []
    for k in range(260):
        rp = 12.0 if k >= 50 else 0.0
        rf = 8.0 if 60 <= k < 75 else 0.0
        steps.append(dict(rf=rf, rp=rp))
    _, fixed = run_stream(_cfg(legacy=False, form=True, peer=True), steps)
    _, legacy = run_stream(_cfg(legacy=True, form=True, peer=True), steps)
    assert fixed.state.fault_type is FaultType.FORM, f"FIXED got {fixed.state.fault_type}"
    assert legacy.state.fault_type is FaultType.PEER, f"LEGACY got {legacy.state.fault_type}"
    return f"T4 FORM_LOCAL latch: FIXED={fixed.state.fault_type.name}  LEGACY={legacy.state.fault_type.name} (the 99.5% mislabel)"


def T5_nav_global():
    steps = [dict(rf=(40.0 if k >= 50 else 0.0), rp=0.0) for k in range(220)]
    _, fixed = run_stream(_cfg(legacy=False, form=True, peer=True), steps)
    assert fixed.state.fault_type is FaultType.NAV, fixed.state.fault_type
    return "T5 NAV_GLOBAL (FORM high, PEER quiet) → NAV"


def T6_peer_only():
    steps = [dict(rf=0.0, rp=(12.0 if k >= 50 else 0.0)) for k in range(220)]
    _, fixed = run_stream(_cfg(legacy=False, form=True, peer=True), steps)
    assert fixed.state.fault_type is FaultType.PEER, fixed.state.fault_type
    return "T6 PEER_ONLY (PEER high, FORM quiet) → PEER"


def T7_causality():
    # The state reached after k steps must not depend on any step > k.
    full = _prop_stream(n=300, t_fault=100)
    fdi_a, _ = run_stream(_cfg(legacy=False), full[:160])
    fdi_b, _ = run_stream(_cfg(legacy=False), full)          # then read back at 160
    fdi_c = FDICore(_cfg(legacy=False)); fdi_c.reset(0.0)
    snap160 = None
    for k, kw in enumerate(full):
        s = _quiet_step(fdi_c, float(k), **kw)
        if k == 159:
            snap160 = (fdi_c.S_prop, s.state.state, s.state.fault_type)
    a = (fdi_a.S_prop, fdi_a.state.state, fdi_a.state.fault_type)
    assert abs(a[0] - snap160[0]) < 1e-9 and a[1] is snap160[1] and a[2] is snap160[2], (a, snap160)
    return "T7 causality: state@160 independent of steps>160"


if __name__ == "__main__":
    tests = [T1_healthy, T2_prop_fix, T3_nav_sanity, T4_formpeer_latch,
             T5_nav_global, T6_peer_only, T7_causality]
    ok = 0
    for t in tests:
        try:
            print("  PASS  " + t())
            ok += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{ok}/{len(tests)} passed")
