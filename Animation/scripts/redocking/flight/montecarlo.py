# -*- coding: utf-8 -*-
"""
montecarlo.py - flight V&V dispersion + fault-injection campaign.

Runs the closed-loop FlightGNC over N dispersed cases (handoff state, attitude,
sensor/actuator seeds) with a configurable fraction injecting a stuck-thruster
fault, and verifies the flight requirements:
  * outcome is DOCKED or a SAFE terminal state (never collision/failure),
  * captured contact within the RAFTI soft-capture envelope,
  * bounded propellant, no keep-out violations.
Returns aggregate statistics for the V&V report.

--seed / --fault-frac / --csv expose the dispersion seed, fault rate, and a
per-case CSV dump; the faulted x outcome cross-tab confirms whether the non-docks
are exactly the injected-fault cases (i.e. the dock rate is fault-limited).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config_redock as C
from envs.rafti import RaftiPorts
from flight import config_flight as F
from flight.gnc_loop import FlightGNC


def _ports():
    r, u, rad = C.load_espa_ports(); rcfg = dict(C.RAFTI); rcfg.update(F.CAPTURE_ENVELOPE)
    return RaftiPorts(r, u, rad, rcfg, deputy_dims=C.ESPA["deputy_dims"], height=C.ESPA["height_m"])


def run_campaign(n=200, fault_frac=0.15, dynamics="hcw", seed0=0, verbose=True, agent=None):
    ports = _ports()
    out = dict(docked=0, safe=0, failed=0, contacts=[], props=[], steps=[], phases=[], faults=0,
               n=n, cases=[])
    for k in range(n):
        rs = seed0 * 1_000_000 + k        # disjoint per-seed blocks -> INDEPENDENT campaigns (was seed0+k -> overlapping)
        r = np.random.default_rng(rs)
        port = int(r.integers(0, ports.P))
        rr = 40 + 60 * r.random()
        d = r.normal(size=3); d /= np.linalg.norm(d)
        # start within a wide cone on the port side + dispersion
        axis = ports.dir_B[port]
        dirn = 0.85 * axis + 0.15 * d; dirn /= np.linalg.norm(dirn)
        r0 = ports.capture_pt[port] + rr * dirn + r.normal(scale=3, size=3)
        v0 = -F.HANDOFF["capture_speed_mps"] * r.random() * axis + 0.02 * r.normal(size=3)
        sig0 = np.tan(np.radians(F.HANDOFF["init_att_err_deg"]) / 4) * r.random() * \
            (lambda x: x / np.linalg.norm(x))(r.normal(size=3))
        s0 = np.concatenate([r0, v0, sig0, np.zeros(3)])
        fault_axis = int(r.integers(0, 3)) if r.random() < fault_frac else None
        faulted = fault_axis is not None
        if faulted:
            out["faults"] += 1
        gnc = FlightGNC(ports, port, agent=agent, seed=rs, fault_axis=fault_axis, dynamics=dynamics)
        res = gnc.run(s0); S = res["summary"]
        out["phases"].append(S["final_phase"]); out["steps"].append(S["steps"])
        out["props"].append(S["prop_used_frac"])
        if S["captured"]:
            out["docked"] += 1; out["contacts"].append(S["contact"]["dist"]); outcome = "DOCKED"
        elif S["final_phase"] in ("SAFE_HOLD",):
            out["safe"] += 1; outcome = "SAFE"
        else:
            out["failed"] += 1; outcome = "FAILED"
        out["cases"].append(dict(k=k, seed=rs, faulted=int(faulted),
                                 fault_axis=(fault_axis if faulted else -1), outcome=outcome,
                                 phase=S["final_phase"],
                                 contact=(float(S["contact"]["dist"]) if S["captured"] else float("nan")),
                                 prop=float(S["prop_used_frac"]), steps=int(S["steps"]),
                                 nav_err=float(S.get("nav_final_err", float("nan")))))
        if verbose and (k + 1) % 50 == 0:
            print(f"  {k+1}/{n}: docked {out['docked']} safe {out['safe']} failed {out['failed']}", flush=True)
    out["dock_rate"] = 100 * out["docked"] / n
    out["safe_rate"] = 100 * out["safe"] / n
    out["fail_rate"] = 100 * out["failed"] / n
    return out


def verify(stats):
    """Requirements verification (pass/fail)."""
    reqs = []
    reqs.append(("No collision/failure outcomes", stats["failed"] == 0))
    reqs.append(("Dock OR safe-hold on every case", stats["docked"] + stats["safe"] == stats["n"]))
    if stats["contacts"]:
        reqs.append(("Captured contact within envelope",
                     max(stats["contacts"]) <= F.CAPTURE_ENVELOPE["capture_pos_tol_m"] + 1e-6))
    reqs.append(("Propellant bounded < 100%", max(stats["props"]) < 1.0))
    return reqs


def crosstab(stats):
    """faulted x outcome counts -> shows whether the non-docks ARE the injected faults."""
    ct = {(f, o): 0 for f in (0, 1) for o in ("DOCKED", "SAFE", "FAILED")}
    for c in stats["cases"]:
        ct[(c["faulted"], c["outcome"])] += 1
    return ct


if __name__ == "__main__":
    import argparse, csv
    from flight import hardware_bridge as HWB
    from flight.controller import load_flight_agent
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="trained policy (default: auto-detect RL_POLICY_CKPT)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dynamics", default="hcw", choices=["hcw", "basilisk"])
    ap.add_argument("--seed", type=int, default=0, help="seed0 for the dispersion campaign")
    ap.add_argument("--fault-frac", type=float, default=0.15, dest="fault_frac")
    ap.add_argument("--csv", default=None, help="optional per-case CSV output path")
    args, _ = ap.parse_known_args()
    ckpt = args.ckpt or (F.RL_POLICY_CKPT if F.rl_policy_available() else None)
    agent = None
    if ckpt:
        try:
            agent, ind, na = load_flight_agent(ckpt); print(f"  policy   : RL {ind}->{na}  ({ckpt})")
        except Exception as e:
            print(f"  policy   : FALLBACK (could not load {ckpt}: {e})")
    else:
        print("  policy   : FALLBACK (verified glideslope; no trained checkpoint found)")
    print(f"ARISE flight re-docking - Monte-Carlo V&V ({args.n} cases, {args.fault_frac*100:.0f}% fault-injected, seed={args.seed})")
    print(f"  {HWB.summary()}")
    st = run_campaign(n=args.n, fault_frac=args.fault_frac, dynamics=args.dynamics, seed0=args.seed, agent=agent)
    print(f"\n  DOCKED  : {st['dock_rate']:.1f}%   SAFE-HOLD: {st['safe_rate']:.1f}%   "
          f"FAILED: {st['fail_rate']:.1f}%   (faults injected: {st['faults']})")
    if st["contacts"]:
        print(f"  contact dist: mean {np.mean(st['contacts'])*100:.1f} cm  max {np.max(st['contacts'])*100:.1f} cm")
    print(f"  propellant: mean {100*np.mean(st['props']):.1f}%  max {100*np.max(st['props']):.1f}%")
    ct = crosstab(st)
    print("\n  faulted x outcome (are the non-docks exactly the faults?):")
    print(f"    nominal : DOCKED {ct[(0,'DOCKED')]:4d}  SAFE {ct[(0,'SAFE')]:4d}  FAILED {ct[(0,'FAILED')]:4d}")
    print(f"    faulted : DOCKED {ct[(1,'DOCKED')]:4d}  SAFE {ct[(1,'SAFE')]:4d}  FAILED {ct[(1,'FAILED')]:4d}")
    if st["failed"] == 0:
        print(f"  [rule-of-three] 0 collisions in {args.n} -> 95% upper bound on collision rate <= {300.0/args.n:.3f}%")
    print("\n  REQUIREMENTS VERIFICATION:")
    for name, ok in verify(st):
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["k","seed","faulted","fault_axis","outcome","phase","contact","prop","steps","nav_err"])
            w.writeheader(); w.writerows(st["cases"])
        print(f"  wrote per-case CSV -> {args.csv}")
