#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_fdi_mc_formpeer.py
================================

Local-vs-global isolation experiment for the Apophis FDI layer.

Two scenarios exercise the FORM and PEER channels and the FORM-vs-PEER
precedence rule:

  * FORM_LOCAL  — a single random deputy suffers a slow task-miss drift.
    Expected FDI signature: FORM high AND PEER high on that deputy;
    per the precedence rule the SETTLED label is FORM.

  * NAV_GLOBAL  — the same drift is applied to all eight deputies.  Every
    deputy's predicted miss grows together, so FORM trips fleet-wide but
    the median-absolute-deviation stays tight and PEER does NOT trip.
    Per the precedence rule the SETTLED label is NAV.

Why "first" vs "settled" label
-------------------------------
The PEER channel reacts to the *relative* outlier signature, which
appears the moment the faulty deputy departs from the fleet median.
The FORM channel reacts to the *absolute* miss, which is small early in
the ramp and only crosses nu_form after the ramp has grown enough.
PEER therefore commonly fires FIRST in a FORM_LOCAL trial, even though
the precedence rule selects FORM once both channels are above
threshold.

Isolation accuracy is judged on the SETTLED label (the FaultType
declared at the end of the encounter, which is what downstream guidance
sees).  Detection latency is judged on the FIRST label (when the FDI
core first declares any non-NONE fault — usually PEER for FORM_LOCAL,
giving early warning, and NAV for NAV_GLOBAL).

Fault model
-----------
Additive linear ramp on r_form, clipped to zero outside [t_fault, t_arr].

Outputs
-------
  fdi_mc_formpeer_results.npz
  fdi_mc_formpeer_labels.png   (2 x 2: first-label vs settled-label)
  fdi_mc_formpeer.tex

Usage
-----
    python apophis_swarm_fdi_mc_formpeer.py [--n-trials 2000] \
        [--target-peak 12] [--seed 12345]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

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
    FDICore, FDIConfig, FDIState, FaultType,
)
from apophis_swarm_fdi_main import (  # noqa: E402
    N_DEP,
    FORM_WINDOW_DURATION_S,
    FORM_SIGMA_M,
    FORM_USE_CW_PREDICTOR,
    _find_global_t0_abs,
    _interp_vec_series,
    _load_fleet_data,
    _estimate_global_arrival_time,
    _compute_formation_residuals,
    _compute_peer_residuals,
)
from fdi_mc_tex_table import (  # noqa: E402
    build_metadata,
    emit_formpeer_table_tex,
    wilson_ci,
)


DEFAULT_N_TRIALS    = 2000
DEFAULT_SEED        = 12345
DEFAULT_TARGET_PEAK = 12.0
HB_NOMINAL          = 60.0
MIN_T_GO_S          = 900.0  # >= 15 min between t_fault and t_arr

SCENARIOS = ["FORM_LOCAL", "NAV_GLOBAL"]


def _apply_form_ramp(fleet_mod, affected, t_fault, t_arr, target_peak):
    window_dur = max(t_arr - t_fault, 1.0)
    for d in affected:
        t_nav = fleet_mod[d]["t_nav"]
        r_f_mod = fleet_mod[d]["r_form"].copy()
        active = (t_nav >= t_fault) & (t_nav <= t_arr)
        if np.any(active):
            r_f_mod[active] += target_peak * (t_nav[active] - t_fault) / window_dur
        fleet_mod[d]["r_form"] = r_f_mod
    _compute_peer_residuals(fleet_mod)


def _run_trial(fleet_nominal, cfg, t0_abs, t_arr_global, scenario,
               target_peak, rng):
    fleet_mod = [dict(f) for f in fleet_nominal]
    for d in range(N_DEP):
        fleet_mod[d]["r_form"] = fleet_nominal[d]["r_form"].copy()
        fleet_mod[d]["r_peer"] = fleet_nominal[d]["r_peer"].copy()

    if scenario == "FORM_LOCAL":
        dep_focus = int(rng.integers(0, N_DEP))
        affected = [dep_focus]
    else:
        affected = list(range(N_DEP))
        dep_focus = int(rng.integers(0, N_DEP))

    t_window_start = t_arr_global - FORM_WINDOW_DURATION_S
    t_fault = float(rng.uniform(t_window_start, t_arr_global - MIN_T_GO_S))

    _apply_form_ramp(fleet_mod, affected, t_fault, t_arr_global, target_peak)

    expected = "FORM" if scenario == "FORM_LOCAL" else "NAV"
    per_dep_first = {}
    per_dep_settled = {}
    n_form_trips = 0
    n_peer_trips = 0
    focus_first_label = "NONE"
    focus_settled_label = "NONE"
    focus_delay = None

    for d in range(N_DEP):
        data = fleet_mod[d]
        t_nav      = data["t_nav"]
        x_est      = data["x_est"]
        P_hist     = data["P_hist"]
        y_pre      = data["y_pre"]
        sigma_meas = data["sigma_meas"]
        dv_seq_H   = data["dv_seq_H"]
        t_err      = data["t_err"]
        sigma_BR   = data["sigma_BR"]
        omega_BR_B = data["omega_BR_B"]
        r_form     = data["r_form"]
        r_peer     = data["r_peer"]

        t_att_rel = t_nav - t0_abs
        fdi = FDICore(cfg)
        fdi.reset(t0=float(t_nav[0]))

        form_tripped_here = False
        peer_tripped_here = False
        first_label = "NONE"
        first_label_delay = None
        last_label = "NONE"
        hb_age = HB_NOMINAL

        for k in range(t_nav.size):
            t_k    = float(t_nav[k])
            t_relk = float(t_att_rel[k])

            innov = y_pre[k]
            if np.any(~np.isfinite(innov)):
                innov_k = None
                S_innov_k = None
            else:
                innov_k = innov
                sigma_k = np.maximum(sigma_meas[k], 1e-3)
                S_innov_k = np.diag(sigma_k ** 2)

            dv_cmd = dv_seq_H[k]
            burn_flag = bool(la.norm(dv_cmd) > 0.0)
            dv_est_H = None
            P_vv = None
            if burn_flag and k > 0:
                v_prev = x_est[k-1, 3:6]
                v_now  = x_est[k,   3:6]
                dv_est_H = (v_now - v_prev) - dv_cmd
                P_vv = P_hist[k, 3:6, 3:6]

            sig_BR_k = _interp_vec_series(t_err, sigma_BR,   t_relk)
            omg_BR_k = _interp_vec_series(t_err, omega_BR_B, t_relk)

            snap = fdi.step(
                t=t_k,
                innov=innov_k,
                S_innov=S_innov_k,
                dv_est_H=dv_est_H,
                P_vv=P_vv,
                burn_flag=burn_flag,
                sigma_BR=sig_BR_k,
                omega_BR_B=omg_BR_k,
                hb_age=hb_age,
                hb_nominal=HB_NOMINAL,
                r_form=float(r_form[k]) if r_form[k] > 0.0 else None,
                r_peer=float(r_peer[k]) if r_peer[k] > 0.0 else None,
            )

            if fdi.S_form > cfg.h_form:
                form_tripped_here = True
            if fdi.S_peer > cfg.h_peer:
                peer_tripped_here = True

            cur_state = snap.state.state
            cur_ft    = snap.state.fault_type
            tripped = cur_state in (FDIState.SUSPECT, FDIState.FAULTED, FDIState.DEAD)

            if tripped and cur_ft is not FaultType.NONE and first_label == "NONE":
                if t_k >= t_fault:
                    first_label = cur_ft.name
                    first_label_delay = max(0.0, t_k - t_fault)

            if tripped and cur_ft is not FaultType.NONE:
                last_label = cur_ft.name

        per_dep_first[d]   = first_label
        per_dep_settled[d] = last_label
        if form_tripped_here:
            n_form_trips += 1
        if peer_tripped_here:
            n_peer_trips += 1
        if d == dep_focus:
            focus_first_label   = first_label
            focus_settled_label = last_label
            focus_delay = first_label_delay

    return dict(
        scenario=scenario,
        dep_focus=dep_focus,
        affected=affected,
        t_fault=t_fault,
        expected=expected,
        first_label=focus_first_label,
        settled_label=focus_settled_label,
        faulty_label=focus_settled_label,
        det_delay=focus_delay,
        per_dep_first=per_dep_first,
        per_dep_settled=per_dep_settled,
        per_dep_labels=per_dep_settled,
        n_form_trips=n_form_trips,
        n_peer_trips=n_peer_trips,
    )


LABEL_ORDER = ["NONE", "NAV", "ATT", "PROP", "COMM", "FORM", "PEER"]
LABEL_COLOR = {
    "NONE":  "#7F7F7F",
    "NAV":   "#0072B2",
    "ATT":   "#009E73",
    "PROP":  "#D55E00",
    "COMM":  "#56B4E9",
    "FORM":  "#CC79A7",
    "PEER":  "#E69F00",
}


def _plot_label_histograms_dual(first_hist, settled_hist, out_path):
    n_sc = len(SCENARIOS)
    fig, axes = plt.subplots(2, n_sc, figsize=(5.5 * n_sc, 7.5), sharey="row")
    if n_sc == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    rows = [
        ("first",   first_hist,   "First declared label  (detection)"),
        ("settled", settled_hist, "Settled label at encounter end  (isolation)"),
    ]
    for r_idx, (_, hist, row_title) in enumerate(rows):
        for c_idx, sc in enumerate(SCENARIOS):
            ax = axes[r_idx, c_idx]
            cnt = hist[sc]
            vals = [cnt.get(lab, 0) for lab in LABEL_ORDER]
            colors = [LABEL_COLOR[lab] for lab in LABEL_ORDER]
            ax.bar(LABEL_ORDER, vals, color=colors, edgecolor="0.2")
            total = sum(vals)
            for i, v in enumerate(vals):
                if v == 0:
                    continue
                ax.text(i, v + 0.012 * max(vals + [1]),
                        f"{100*v/max(total,1):.1f}%",
                        ha="center", va="bottom", fontsize=8)
            ax.set_title(f"{sc} - {row_title}")
            if r_idx == 1:
                ax.set_xlabel("Identified label")
            ax.grid(True, axis="y", alpha=0.35)
        axes[r_idx, 0].set_ylabel("Trial count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[plot] {out_path.name}")
    plt.close(fig)


def main(data_dir=None, n_trials=DEFAULT_N_TRIALS,
         target_peak=DEFAULT_TARGET_PEAK, seed=DEFAULT_SEED):
    root = Path(data_dir).resolve() if data_dir is not None else SCRIPT_DIR

    print(f"\nFDI FORM/PEER local-vs-global MC in {root}")
    print(f"  trials/scenario = {n_trials}, target_peak = {target_peak}, seed = {seed}")

    t0_abs = _find_global_t0_abs(root)
    fleet = _load_fleet_data(root)
    t_arr_global, _ = _estimate_global_arrival_time(fleet)

    _compute_formation_residuals(
        fleet,
        t_arr_global=t_arr_global,
        window_duration_s=FORM_WINDOW_DURATION_S,
        sigma_form=FORM_SIGMA_M,
        use_cw_predictor=FORM_USE_CW_PREDICTOR,
    )
    _compute_peer_residuals(fleet)

    cfg = FDIConfig(
        use_nav_channel=True,
        use_att_channel=True,
        use_comm_channel=False,
        use_form_channel=True,
        use_peer_channel=True,
        nu_form=3.0, leak_form=0.02, h_form=30.0,
        nu_peer=3.0, leak_peer=0.05, h_peer=20.0,
    )
    cfg.nu_nav   = 1.5
    cfg.leak_nav = 0.05
    cfg.h_nav    = 15.0

    rng = np.random.default_rng(seed)

    counts = {sc: dict(n_trials=0, n_faulty=0, n_correct=0,
                       n_wrong_label=0, n_missed=0, n_false_alarm=0,
                       n_form_trips_fleet=0,
                       n_first_peer_settled_form=0,
                       det_delays=[])
              for sc in SCENARIOS}
    settled_hist = {sc: {lab: 0 for lab in LABEL_ORDER} for sc in SCENARIOS}
    first_hist   = {sc: {lab: 0 for lab in LABEL_ORDER} for sc in SCENARIOS}

    for sc in SCENARIOS:
        print(f"\n=== Scenario {sc} ===")
        for trial in range(n_trials):
            res = _run_trial(fleet, cfg, t0_abs, t_arr_global, sc, target_peak, rng)
            counts[sc]["n_trials"] += 1
            counts[sc]["n_faulty"] += 1
            settled_hist[sc][res["settled_label"]] = (
                settled_hist[sc].get(res["settled_label"], 0) + 1
            )
            first_hist[sc][res["first_label"]] = (
                first_hist[sc].get(res["first_label"], 0) + 1
            )
            if res["first_label"] == "PEER" and res["settled_label"] == "FORM":
                counts[sc]["n_first_peer_settled_form"] += 1

            if res["settled_label"] == res["expected"]:
                counts[sc]["n_correct"] += 1
                if res["det_delay"] is not None:
                    counts[sc]["det_delays"].append(res["det_delay"])
            elif res["settled_label"] == "NONE":
                counts[sc]["n_missed"] += 1
            else:
                counts[sc]["n_wrong_label"] += 1

            if res["n_form_trips"] >= 2:
                counts[sc]["n_form_trips_fleet"] += 1

            for d, lab in res["per_dep_settled"].items():
                if d in res["affected"]:
                    continue
                if lab != "NONE":
                    counts[sc]["n_false_alarm"] += 1

            if (trial + 1) % 25 == 0 or trial == 0:
                print(f"  trial {trial+1}/{n_trials}", end="\r")
        print(f"  completed {counts[sc]['n_trials']} trials")

    print("\n========== FORM/PEER summary ==========")
    for sc in SCENARIOS:
        c = counts[sc]
        nt = c["n_trials"]
        Pc = c["n_correct"] / max(c["n_faulty"], 1)
        Pc_lo, Pc_hi = wilson_ci(c["n_correct"], max(c["n_faulty"], 1))
        Pm = c["n_missed"] / max(c["n_faulty"], 1)
        Pw = c["n_wrong_label"] / max(c["n_faulty"], 1)
        print(f"  {sc:11s}  trials={nt:5d}  "
              f"settled-correct={100*Pc:5.2f}% [{100*Pc_lo:.2f},{100*Pc_hi:.2f}]  "
              f"wrong-label={100*Pw:5.2f}%  missed={100*Pm:5.2f}%  "
              f"FORM trips on >=2 deps in {c['n_form_trips_fleet']} trials")
        print(f"               first-label hist  : " +
              ", ".join(f"{lab}={first_hist[sc].get(lab, 0)}"
                        for lab in LABEL_ORDER if first_hist[sc].get(lab, 0) > 0))
        print(f"               settled-label hist: " +
              ", ".join(f"{lab}={settled_hist[sc].get(lab, 0)}"
                        for lab in LABEL_ORDER if settled_hist[sc].get(lab, 0) > 0))
        print(f"               PEER first -> FORM settled in "
              f"{c['n_first_peer_settled_form']} trials  (early-warning behaviour)")

    meta = build_metadata(
        rng_seed=seed,
        n_trials=n_trials,
        fdi_cfg=cfg,
        fault_params=dict(
            target_peak=target_peak,
            FORM_WINDOW_DURATION_S=FORM_WINDOW_DURATION_S,
            FORM_SIGMA_M=FORM_SIGMA_M,
            FORM_USE_CW_PREDICTOR=FORM_USE_CW_PREDICTOR,
            MIN_T_GO_S=MIN_T_GO_S,
            n_dep=N_DEP,
        ),
        script_name="apophis_swarm_fdi_mc_formpeer.py",
    )

    out_npz = root / "fdi_mc_formpeer_results.npz"
    np.savez_compressed(
        out_npz,
        scenarios=np.array(SCENARIOS),
        n_trials=np.array([counts[sc]["n_trials"] for sc in SCENARIOS]),
        n_faulty=np.array([counts[sc]["n_faulty"] for sc in SCENARIOS]),
        n_correct=np.array([counts[sc]["n_correct"] for sc in SCENARIOS]),
        n_wrong_label=np.array([counts[sc]["n_wrong_label"] for sc in SCENARIOS]),
        n_missed=np.array([counts[sc]["n_missed"] for sc in SCENARIOS]),
        n_false_alarm=np.array([counts[sc]["n_false_alarm"] for sc in SCENARIOS]),
        n_form_trips_fleet=np.array(
            [counts[sc]["n_form_trips_fleet"] for sc in SCENARIOS]),
        n_first_peer_settled_form=np.array(
            [counts[sc]["n_first_peer_settled_form"] for sc in SCENARIOS]),
        det_delays=np.array(
            [np.asarray(counts[sc]["det_delays"], float) for sc in SCENARIOS],
            dtype=object,
        ),
        settled_label_histograms=np.array(
            [[settled_hist[sc].get(lab, 0) for lab in LABEL_ORDER]
             for sc in SCENARIOS], dtype=int),
        first_label_histograms=np.array(
            [[first_hist[sc].get(lab, 0) for lab in LABEL_ORDER]
             for sc in SCENARIOS], dtype=int),
        label_histograms=np.array(
            [[settled_hist[sc].get(lab, 0) for lab in LABEL_ORDER]
             for sc in SCENARIOS], dtype=int),
        label_order=np.array(LABEL_ORDER),
        metadata=np.array(meta, dtype=object),
    )
    print(f"\n[npz] wrote {out_npz.name}")

    emit_formpeer_table_tex(out_npz, out_path=root / "fdi_mc_formpeer.tex")
    _plot_label_histograms_dual(first_hist, settled_hist,
                                root / "fdi_mc_formpeer_labels.png")

    print("\nFORM/PEER MC complete.")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="FDI FORM/PEER local-vs-global MC.")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    p.add_argument("--target-peak", type=float, default=DEFAULT_TARGET_PEAK)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    main(
        data_dir=args.data_dir,
        n_trials=args.n_trials,
        target_peak=args.target_peak,
        seed=args.seed,
    )
