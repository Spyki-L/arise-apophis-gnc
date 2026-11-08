# -*- coding: utf-8 -*-
"""
run_flight.py - ONE-CLICK flight GNC demonstration (VS Code: press Run).

Runs the closed-loop flight re-dock for one deputy (sensors -> EKF nav -> executive
-> control -> RTA -> FDIR -> reaction-wheel/RCS -> dynamics), writes a telemetry
timeline figure, and prints the outcome.  No command-line arguments.

Edit CONFIG below.  Use flight/montecarlo.py for the full dispersion V&V campaign.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config_redock as C
from envs.rafti import RaftiPorts
from flight import config_flight as F
from flight.gnc_loop import FlightGNC
from flight import hardware_bridge as HWB

# ==================== RUN CONFIG (edit here) ====================
PORT      = 0
SEED      = 0
DYNAMICS  = "hcw"      # "hcw" | "basilisk"
FAULT_AXIS = None      # None, or 0/1/2 to inject a stuck thruster on that axis
# ===============================================================


def main():
    art = C.ensure_art_dir()
    rr, uu, rad = C.load_espa_ports(); rcfg = dict(C.RAFTI); rcfg.update(F.CAPTURE_ENVELOPE)
    ports = RaftiPorts(rr, uu, rad, rcfg, deputy_dims=C.ESPA["deputy_dims"], height=C.ESPA["height_m"])
    r0 = ports.capture_pt[PORT] + 60 * ports.dir_B[PORT]
    s0 = np.concatenate([r0, -0.05 * ports.dir_B[PORT],
                         0.2 * np.random.default_rng(SEED).normal(size=3), np.zeros(3)])
    print(f"[flight] {F.summary()}")
    print(f"[hardware] {HWB.summary()}")
    # auto-load the trained terminal-docking policy if present (else verified fallback)
    agent = None
    if F.rl_policy_available():
        from flight.controller import load_flight_agent
        try:
            agent, ind, na = load_flight_agent(F.RL_POLICY_CKPT)
            print(f"[control] RL policy {ind}->{na}  ({F.RL_POLICY_CKPT})")
        except Exception as e:
            print(f"[control] FALLBACK (could not load policy: {e})")
    else:
        print("[control] FALLBACK (verified glideslope; drop a checkpoint at "
              "artifacts/flight_policy.pt to use RL)")
    gnc = FlightGNC(ports, PORT, agent=agent, seed=SEED, fault_axis=FAULT_AXIS, dynamics=DYNAMICS)
    res = gnc.run(s0); S = res["summary"]; rows = gnc.tm.rows

    print(f"\n  outcome        : {S['final_phase']}  (captured={S['captured']})")
    print(f"  contact dist   : {S['contact']['dist']*100:.1f} cm   face-align {S['contact']['att_err']:.2f} deg")
    print(f"  contact speed  : axial {S['contact']['v_axial']*1000:+.1f} mm/s  rate {S['contact']['rate']*1e3:.1f} mrad/s")
    print(f"  nav error      : {S['nav_final_err']*100:.1f} cm   propellant {100*S['prop_used_frac']:.1f}%")
    print(f"  RTA limited    : {S['n_rta_limited']} steps   faults: {S['faults'] or 'none'}")

    # telemetry timeline
    t = [r["t"] for r in rows]; dist = [r["dist"] for r in rows]
    prop = [100 * r["prop"] for r in rows]; nsig = [r["nav_sigma"] for r in rows]
    phase_idx = {p: i for i, p in enumerate(
        ["HANDOFF", "APPROACH", "TERMINAL", "CAPTURE", "DOCKED", "ABORT", "RETREAT", "SAFE_HOLD", "FAILED"])}
    ph = [phase_idx.get(r["phase"], 0) for r in rows]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].semilogy(t, dist, lw=1.4, color="#4C72B0"); ax[0].axhline(F.CAPTURE_ENVELOPE["capture_pos_tol_m"],
                   color="r", ls="--", lw=0.8, label="capture envelope")
    ax[0].set_ylabel("range to port [m]"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title(f"ARISE flight re-dock — deputy {PORT} — outcome: {S['final_phase']}")
    ax[1].plot(t, ph, drawstyle="steps-post", lw=1.6, color="#55A868")
    ax[1].set