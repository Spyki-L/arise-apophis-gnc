#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_fdi_mc.py
=======================

Monte Carlo campaign for the Apophis swarm FDI (NAV / ATT / PROP / COMM).

Per trial we:
  • load the same fleet data as apophis_swarm_fdi_main.py
  • choose ONE deputy at random to fault (or none for NOMINAL)
  • choose a random fault injection time within that deputy's nav arc
  • inject a fault into one channel (NAV / ATT / PROP / COMM)
  • run FDICore for ALL 8 deputies (FORM/PEER disabled)
  • record:
        - whether the faulty deputy was detected with the right label
        - detection delay [s]
        - false alarms on healthy deputies
        - fault injection phase in the nav arc (0–1)

Outputs
-------
Console summary + plots:

  - fdi_mc_summary.png       : Pd & Pfa vs scenario
  - fdi_mc_delay_<SC>.png    : detection delay histogram per scenario
  - fdi_mc_phase_<SC>.png    : fault phase vs detection delay scatter
                               (slowest 10% circled, misses as crosses)

And a data file:

  - fdi_mc_results.npz       : all stats needed to replot (per scenario)

Run from the Basilisk root with:

    python "ARISE Arch/GNC/Scenario/Apophis/apophis_swarm_fdi_mc.py"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import sys

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# ───────────────── locate ARISE root / import FDI core ─────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

ROOT = None
for parent in SCRIPT_DIR.parents:
    if (parent / "GNC").is_dir():
        ROOT = parent
        break
if ROOT is None:
    ROOT = SCRIPT_DIR.parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNC.Navigation.fdi_core import (  # noqa: E402
    FDICore,
    FDIConfig,
    FDIState,
    FaultType,
)

# Re-use helpers + fleet loader from the main FDI script
from apophis_swarm_fdi_main import (   # noqa: E402
    N_DEP,
    _find_global_t0_abs,
    _interp_vec_series,
    _load_fleet_data,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Monte-Carlo parameters
# ╚══════════════════════════════════════════════════════════════════════════╝

N_TRIALS_PER_SCENARIO = 20000
RNG_SEED = 12345

# NAV fault: innovation bias in [σ] units
NAV_BIAS_SIGMA = 5.0

# ATT fault: attitude + rate biases
ATT_BIAS_SIGMA_DEG = 1.0       # [deg] MRP-norm approx
ATT_RATE_BIAS_DEG = 0.05       # [deg/s]

# PROP fault: Δv mismatch in multiples of 1σ of P_vv
PROP_FAULT_SIGMA = 5.0

# COMM fault: heartbeat loss beyond this multiple trips the COMM channel
HB_NOMINAL = 60.0  # [s]


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Scenarios & stats (no FORM)
# ╚══════════════════════════════════════════════════════════════════════════╝

class Scenario(Enum):
    NOMINAL = auto()
    NAV     = auto()
    ATT     = auto()
    PROP    = auto()
    COMM    = auto()


EXPECTED_FAULT = {
    Scenario.NOMINAL: FaultType.NONE,
    Scenario.NAV:     FaultType.NAV,
    Scenario.ATT:     FaultType.ATT,
    Scenario.PROP:    FaultType.PROP,
    Scenario.COMM:    FaultType.COMM,
}


@dataclass
class FaultPlan:
    scenario: Scenario
    dep_fault: int | None
    t_fault: float | None
    phase: float | None       # fault phase in nav arc [0,1]


@dataclass
class ScenarioStats:
    n_trials: int = 0
    n_faulty_deps: int = 0
    n_detected_correct: int = 0
    n_missed: int = 0
    n_false_alarm: int = 0
    det_delays: list[float] = field(default_factory=list)

    # for “phase vs delay” plots
    inj_phase_detected: list[float] = field(default_factory=list)
    det_delay_detected: list[float] = field(default_factory=list)
    inj_phase_missed:   list[float] = field(default_factory=list)


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Helpers
# ╚══════════════════════════════════════════════════════════════════════════╝

def _rand_unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = la.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0])


def _sample_fault_plan(fleet: list[dict],
                       scenario: Scenario,
                       rng: np.random.Generator) -> FaultPlan:
    """Pick a random faulty deputy + fault injection time and phase."""
    if scenario is Scenario.NOMINAL:
        return FaultPlan(scenario=scenario, dep_fault=None, t_fault=None, phase=None)

    dep_fault = int(rng.integers(0, N_DEP))
    t_nav = fleet[dep_fault]["t_nav"]
    t_start = float(t_nav[0])
    t_end   = float(t_nav[-1])

    # choose phase in [0.1, 0.9] of this deputy's nav arc
    phase = float(rng.uniform(0.1, 0.9))
    t_fault = t_start + phase * (t_end - t_start)
    return FaultPlan(scenario=scenario, dep_fault=dep_fault, t_fault=t_fault, phase=phase)


# ╔══════════════════════════════════════════════════════════════════════════╗
#  One full-fleet trial (NAV / ATT / PROP / COMM only)
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_trial(
    fleet: list[dict],
    cfg: FDIConfig,
    t0_abs: float,
    plan: FaultPlan,
    rng: np.random.Generator,
) -> dict[int, dict]:
    """
    Run one Monte-Carlo trial over the *entire* fleet.

    Returns per-deputy dict:

        results[d] = {
            "faulty": bool,
            "expected_fault": FaultType,
            "detected_correct": bool,
            "detected_wrong": bool,
            "false_alarm": bool,
            "det_delay": float | None,
        }
    """
    results: dict[int, dict] = {}

    for d in range(N_DEP):
        data = fleet[d]
        t_nav      = data["t_nav"]
        x_est      = data["x_est"]
        P_hist     = data["P_hist"]
        y_pre      = data["y_pre"]
        sigma_meas = data["sigma_meas"]
        dv_seq_H   = data["dv_seq_H"]
        t_err      = data["t_err"]
        sigma_BR   = data["sigma_BR"]
        omega_BR_B = data["omega_BR_B"]

        # attitude sim ran with t_rel = t_abs - t0_abs
        t_att_rel = t_nav - t0_abs

        fdi = FDICore(cfg)
        fdi.reset(t0=float(t_nav[0]))

        faulty = (
            plan.scenario is not Scenario.NOMINAL and
            plan.dep_fault is not None and
            d == plan.dep_fault
        )
        expected_fault = EXPECTED_FAULT[plan.scenario] if faulty else FaultType.NONE

        detected_correct = False
        detected_wrong   = False
        false_alarm      = False
        det_delay: float | None = None

        hb_age = HB_NOMINAL

        for k in range(t_nav.size):
            t_k    = float(t_nav[k])
            t_relk = float(t_att_rel[k])

            # --- NAV channel: innovation (with optional fault) ---
            innov = y_pre[k].copy()
            if np.any(~np.isfinite(innov)):
                innov_k = None
                S_innov_k = None
            else:
                if faulty and plan.scenario is Scenario.NAV and t_k >= (plan.t_fault or t_k):
                    # bias innovation in the direction of the local σ
                    bias = NAV_BIAS_SIGMA * sigma_meas[k]
                    innov += bias

                innov_k = innov
                sigma_k = np.maximum(sigma_meas[k], 1e-3)
                R_k     = np.diag(sigma_k**2)
                S_innov_k = R_k

            # --- PROP channel: dv mismatch (fault only at burns) ---
            dv_cmd    = dv_seq_H[k]
            burn_flag = bool(la.norm(dv_cmd) > 0.0)

            dv_est_H = None
            P_vv     = None
            if burn_flag and k > 0:
                v_prev = x_est[k-1, 3:6]
                v_now  = x_est[k,   3:6]
                dv_est = (v_now - v_prev) - dv_cmd
                P_vv   = P_hist[k, 3:6, 3:6]

                if faulty and plan.scenario is Scenario.PROP and t_k >= (plan.t_fault or t_k):
                    if P_vv is not None:
                        sigma_v = np.sqrt(np.maximum(np.diag(P_vv), 1e-6))
                    else:
                        sigma_v = np.ones(3)
                    dv_est += PROP_FAULT_SIGMA * sigma_v * rng.normal(size=3)

                dv_est_H = dv_est

            # --- ATT channel: interpolate and optionally bias ---
            sig_BR_k = _interp_vec_series(t_err, sigma_BR,   t_relk)
            omg_BR_k = _interp_vec_series(t_err, omega_BR_B, t_relk)

            if faulty and plan.scenario is Scenario.ATT and t_k >= (plan.t_fault or t_k):
                sig_BR_k = sig_BR_k + np.deg2rad(ATT_BIAS_SIGMA_DEG) * _rand_unit_vec(rng)
                omg_BR_k = omg_BR_k + np.deg2rad(ATT_RATE_BIAS_DEG) * _rand_unit_vec(rng)

            # --- COMM channel: simple heartbeat age model ---
            if k == 0:
                dt = 0.0
            else:
                dt = float(t_nav[k] - t_nav[k-1])

            if faulty and plan.scenario is Scenario.COMM and t_k >= (plan.t_fault or t_k):
                # stop receiving heartbeats; hb_age grows without bound
                hb_age += dt
            else:
                # healthy: reset age to nominal each cycle
                hb_age = HB_NOMINAL

            hb_age_k = hb_age

            snap = fdi.step(
                t=t_k,
                innov=innov_k,
                S_innov=S_innov_k,
                dv_est_H=dv_est_H,
                P_vv=P_vv,
                burn_flag=burn_flag,
                sigma_BR=sig_BR_k,
                omega_BR_B=omg_BR_k,
                hb_age=hb_age_k,
                hb_nominal=HB_NOMINAL,
                r_form=None,      # FORM/PEER disabled in MC
                r_peer=None,
            )

            cur_state = snap.state.state
            cur_ft    = snap.state.fault_type

            # --- bookkeeping for stats ---
            if faulty:
                # ignore “detections” before we actually inject the fault
                if t_k >= (plan.t_fault or t_k):
                    if (not detected_correct and
                        cur_ft == expected_fault and
                        cur_state in (FDIState.SUSPECT, FDIState.FAULTED, FDIState.DEAD)):
                        detected_correct = True
                        det_delay = max(0.0, t_k - (plan.t_fault or t_k))
                    elif (not detected_wrong and
                          cur_ft not in (FaultType.NONE, expected_fault) and
                          cur_state in (FDIState.SUSPECT, FDIState.FAULTED, FDIState.DEAD)):
                        detected_wrong = True
            else:
                if (cur_ft is not FaultType.NONE and
                    cur_state in (FDIState.SUSPECT, FDIState.FAULTED, FDIState.DEAD)):
                    false_alarm = True

        results[d] = dict(
            faulty=faulty,
            expected_fault=expected_fault,
            detected_correct=detected_correct,
            detected_wrong=detected_wrong,
            false_alarm=false_alarm,
            det_delay=det_delay,
        )

    return results


def update_stats(stats: ScenarioStats,
                 trial_results: dict[int, dict],
                 plan: FaultPlan) -> None:
    stats.n_trials += 1

    phase = plan.phase  # None for NOMINAL

    for _, res in trial_results.items():
        faulty = res["faulty"]
        if faulty:
            stats.n_faulty_deps += 1
            if res["detected_correct"]:
                stats.n_detected_correct += 1
                if res["det_delay"] is not None:
                    stats.det_delays.append(res["det_delay"])
                    if phase is not None:
                        stats.inj_phase_detected.append(phase)
                        stats.det_delay_detected.append(res["det_delay"])
            else:
                stats.n_missed += 1
                if phase is not None:
                    stats.inj_phase_missed.append(phase)
        else:
            if res["false_alarm"]:
                stats.n_false_alarm += 1


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Plotting
# ╚══════════════════════════════════════════════════════════════════════════╝

def _plot_summary(stats: dict[Scenario, ScenarioStats]) -> None:
    scenarios = [Scenario.NAV, Scenario.ATT, Scenario.PROP, Scenario.COMM]
    labels = [sc.name for sc in scenarios]

    Pd_vals = []
    Pfa_vals = []

    for sc in scenarios:
        st = stats[sc]
        if st.n_faulty_deps > 0:
            Pd = st.n_detected_correct / st.n_faulty_deps
        else:
            Pd = 0.0

        # each non-NOMINAL trial has exactly 1 faulty dep → healthy = N_DEP-1
        n_healthy_deps = st.n_trials * (N_DEP - 1)
        Pfa = st.n_false_alarm / max(n_healthy_deps, 1)

        Pd_vals.append(Pd)
        Pfa_vals.append(Pfa)

    x = np.arange(len(scenarios))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.bar(x, Pd_vals)
    ax1.set_ylabel("Detection probability Pd")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis="y", alpha=0.4)

    ax2.bar(x, Pfa_vals)
    ax2.set_ylabel("False-alarm rate Pfa\n(per healthy deputy)")
    ax2.set_xlabel("Fault scenario")
    ax2.grid(True, axis="y", alpha=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    fig.suptitle("FDI Monte-Carlo summary (NAV / ATT / PROP / COMM)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = SCRIPT_DIR / "fdi_mc_summary.png"
    fig.savefig(out, dpi=200)
    print(f"[plot] {out.name}")


def _plot_delay_histograms(stats: dict[Scenario, ScenarioStats]) -> None:
    for sc in [Scenario.NAV, Scenario.ATT, Scenario.PROP, Scenario.COMM]:
        st = stats[sc]
        if not st.det_delays:
            continue

        delays = np.asarray(st.det_delays)
        delays_min = delays / 60.0

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(delays_min, bins=20, edgecolor="k", alpha=0.7)
        ax.set_xlabel("Detection delay [min]")
        ax.set_ylabel("Count")
        ax.set_title(f"Detection delay histogram – {sc.name} faults")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = SCRIPT_DIR / f"fdi_mc_delay_{sc.name}.png"
        fig.savefig(out, dpi=200)
        print(f"[plot] {out.name}")


def _plot_phase_vs_delay(stats: dict[Scenario, ScenarioStats]) -> None:
    """
    "Phase space" plot: fault injection phase (0–1) vs detection delay,
    color-coded by delay. Missed detections are shown as black crosses
    at the top. The slowest ~10% detections are additionally highlighted
    with open circles as "critical" cases.
    """
    for sc in [Scenario.NAV, Scenario.ATT, Scenario.PROP, Scenario.COMM]:
        st = stats[sc]
        if not st.inj_phase_detected and not st.inj_phase_missed:
            continue

        phases_det = np.asarray(st.inj_phase_detected)
        delays_det_min = np.asarray(st.det_delay_detected) / 60.0  # [min]

        phases_miss = np.asarray(st.inj_phase_missed)

        fig, ax = plt.subplots(figsize=(7, 5))

        y_max = 1.0
        if phases_det.size > 0:
            # base scatter, colored by delay
            sca = ax.scatter(
                phases_det * 100.0,      # [%] along nav arc
                delays_det_min,
                c=delays_det_min,
                s=10,
                cmap="viridis",
                alpha=0.7,
            )
            cbar = fig.colorbar(sca, ax=ax)
            cbar.set_label("Detection delay [min]")
            y_max = max(delays_det_min.max(), 1.0)

            # highlight "critical" detections: slowest 10%
            crit_threshold = np.percentile(delays_det_min, 90.0)
            crit_mask = delays_det_min >= crit_threshold
            if np.any(crit_mask):
                ax.scatter(
                    phases_det[crit_mask] * 100.0,
                    delays_det_min[crit_mask],
                    facecolors="none",
                    edgecolors="red",
                    s=80,
                    linewidths=1.5,
                    label="critical (slowest 10%)",
                )

        # missed detections: put crosses slightly above max delay
        if phases_miss.size > 0:
            y_miss = np.full_like(phases_miss, y_max * 1.05)
            ax.scatter(
                phases_miss * 100.0,
                y_miss,
                marker="x",
                s=25,
                color="k",
                label="missed detection",
            )
            ax.set_ylim(0, y_max * 1.2)
        else:
            ax.set_ylim(0, y_max * 1.1)

        ax.set_xlabel("Fault injection phase in nav arc [%]")
        ax.set_ylabel("Detection delay [min]")
        ax.set_title(f"Fault phase vs detection delay – {sc.name} faults")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        out = SCRIPT_DIR / f"fdi_mc_phase_{sc.name}.png"
        fig.savefig(out, dpi=200)
        print(f"[plot] {out.name}")


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Saving MC results as NPZ
# ╚══════════════════════════════════════════════════════════════════════════╝

def _save_results_npz(stats: dict[Scenario, ScenarioStats]) -> None:
    """
    Save all Monte-Carlo results for later re-plotting.

    For each scenario SC we store:
      SC_n_trials, SC_n_faulty_deps, SC_n_detected_correct,
      SC_n_missed, SC_n_false_alarm,
      SC_det_delays          (1D, seconds),
      SC_inj_phase_detected  (1D, phase in [0,1]),
      SC_det_delay_detected  (1D, seconds),
      SC_inj_phase_missed    (1D, phase in [0,1])
    """
    npz_data: dict[str, np.ndarray] = {}

    for sc in Scenario:
        st = stats[sc]
        prefix = sc.name

        npz_data[f"{prefix}_n_trials"]           = np.array(st.n_trials, dtype=int)
        npz_data[f"{prefix}_n_faulty_deps"]      = np.array(st.n_faulty_deps, dtype=int)
        npz_data[f"{prefix}_n_detected_correct"] = np.array(st.n_detected_correct, dtype=int)
        npz_data[f"{prefix}_n_missed"]           = np.array(st.n_missed, dtype=int)
        npz_data[f"{prefix}_n_false_alarm"]      = np.array(st.n_false_alarm, dtype=int)

        npz_data[f"{prefix}_det_delays"]         = np.asarray(st.det_delays, dtype=float)
        npz_data[f"{prefix}_inj_phase_detected"] = np.asarray(st.inj_phase_detected, dtype=float)
        npz_data[f"{prefix}_det_delay_detected"] = np.asarray(st.det_delay_detected, dtype=float)
        npz_data[f"{prefix}_inj_phase_missed"]   = np.asarray(st.inj_phase_missed, dtype=float)

    out = SCRIPT_DIR / "fdi_mc_results.npz"
    np.savez_compressed(out, **npz_data)
    print(f"[npz] Saved MC results -> {out.name}")


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Entry point
# ╚══════════════════════════════════════════════════════════════════════════╝

def main_mc(data_dir: str | Path | None = None) -> None:
    if data_dir is None:
        root = SCRIPT_DIR
    else:
        root = Path(data_dir).resolve()

    print(f"\nFDI Monte Carlo over Apophis fleet in {root}")

    t0_abs = _find_global_t0_abs(root)
    print(f"Global t0_abs (earliest RHIG sample) = {t0_abs:.1f} s")

    # fleet data (no FORM / PEER needed here)
    fleet = _load_fleet_data(root)

    # FDI configuration: NAV / ATT / PROP / COMM only
    cfg = FDIConfig(
        use_nav_channel=True,
        use_att_channel=True,
        use_comm_channel=True,
        use_form_channel=False,
        use_peer_channel=False,
    )

    rng = np.random.default_rng(RNG_SEED)

    scenarios = [
        Scenario.NOMINAL,
        Scenario.NAV,
        Scenario.ATT,
        Scenario.PROP,
        Scenario.COMM,
    ]

    stats: dict[Scenario, ScenarioStats] = {sc: ScenarioStats() for sc in scenarios}

    for sc in scenarios:
        print(f"\n=== Scenario {sc.name} ===")
        for trial in range(N_TRIALS_PER_SCENARIO):
            plan = _sample_fault_plan(fleet, sc, rng)
            trial_res = run_trial(fleet, cfg, t0_abs, plan, rng)
            update_stats(stats[sc], trial_res, plan)

            if (trial + 1) % 20 == 0 or trial == 0:
                print(f"  trial {trial+1}/{N_TRIALS_PER_SCENARIO}", end="\r")

        print(f"  completed {N_TRIALS_PER_SCENARIO} trials")

    # ── summary printout ────────────────────────────────────────────────────
    print("\n========== Monte-Carlo summary ==========")
    for sc in scenarios:
        st = stats[sc]
        exp_ft = EXPECTED_FAULT[sc]
        print(f"\nScenario {sc.name} (truth fault = {exp_ft.name})")
        if st.n_faulty_deps > 0:
            Pd = st.n_detected_correct / st.n_faulty_deps
        else:
            Pd = 0.0

        n_healthy_deps = st.n_trials * (N_DEP - (1 if sc is not Scenario.NOMINAL else 0))
        Pfa = st.n_false_alarm / max(n_healthy_deps, 1)

        print(f"  trials            : {st.n_trials}")
        print(f"  faulty deputies   : {st.n_faulty_deps}")
        print(f"  correct detections: {st.n_detected_correct} "
              f"({Pd*100:.1f} %)")
        print(f"  missed detections : {st.n_missed}")
        print(f"  false alarms      : {st.n_false_alarm} "
              f"({Pfa*100:.2f} % per healthy dep)")

        if st.det_delays:
            arr = np.asarray(st.det_delays)
            print(f"  detection delay [s]: "
                  f"mean={arr.mean():.1f}, "
                  f"p50={np.percentile(arr,50):.1f}, "
                  f"p90={np.percentile(arr,90):.1f}")
        else:
            print("  detection delay   : n/a (no correct detections)")

    # ── plots ───────────────────────────────────────────────────────────────
    _plot_summary(stats)
    _plot_delay_histograms(stats)
    _plot_phase_vs_delay(stats)

    # ── save raw MC results for later re-plotting ──────────────────────────
    _save_results_npz(stats)

    print("\nMonte-Carlo FDI campaign complete.")
    plt.show()


if __name__ == "__main__":
    main_mc()
