#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_fdi_mc_isolation.py
=================================

Monte-Carlo isolation campaign for the Apophis FDI layer.

This script extends ``apophis_swarm_fdi_mc.py`` to capture the *full*
confusion matrix between injected fault type and identified fault label,
not just the binary correct/incorrect flag the baseline MC tracks.  It
directly answers the journal reviewer's question

    "Detection is not isolation.  Show that when channel X fires it is
    actually labelling the right fault."

For each trial we:

  * pick a scenario from {NOMINAL, NAV, ATT, PROP, COMM};
  * draw one random deputy and a random fault phase in [0.1, 0.9] of
    that deputy's nav arc (NOMINAL trials inject nothing);
  * inject the fault into one channel;
  * run the FDI core for ALL 8 deputies; and
  * record the *first* fault label the FDI declares on the faulty deputy
    after t_fault, and any false labels declared on the healthy deputies.

The fault types are the same five injection scenarios + ``NONE`` for the
identified-label axis (a deputy that stays HEALTHY).  FORM and PEER faults
are NOT injected here — those live in
``apophis_swarm_fdi_mc_formpeer.py`` — but the FDI core's FORM/PEER
channels are disabled in this script so we are evaluating the
NAV/ATT/PROP/COMM detector bank in isolation.

Outputs
-------
  fdi_mc_isolation_results.npz   :
      Full counts plus the confusion matrix, with reproducibility metadata.
  fdi_mc_isolation_confusion.png :
      Heatmap of truth vs identified label.

Usage
-----
    python apophis_swarm_fdi_mc_isolation.py [--n-trials 2000] [--seed 12345]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent

# Locate ARISE root and import the FDI core / shared helpers from the
# existing MC script.
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
from apophis_swarm_fdi_main import (  # noqa: E402
    N_DEP,
    _find_global_t0_abs,
    _interp_vec_series,
    _load_fleet_data,
)
from fdi_mc_tex_table import (  # noqa: E402
    build_metadata,
    emit_confusion_matrix_tex,
    wilson_ci,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Fault model defaults (kept consistent with apophis_swarm_fdi_mc.py)
# ╚══════════════════════════════════════════════════════════════════════════╝

DEFAULT_N_TRIALS = 2000
DEFAULT_SEED     = 12345

NAV_BIAS_SIGMA       = 5.0
ATT_BIAS_SIGMA_DEG   = 1.0
ATT_RATE_BIAS_DEG    = 0.05
PROP_FAULT_SIGMA     = 5.0
HB_NOMINAL           = 60.0

SCENARIOS_INJECTED = ["NOMINAL", "NAV", "ATT", "PROP", "COMM"]


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Helpers
# ╚══════════════════════════════════════════════════════════════════════════╝

def _rand_unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = la.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0])


def _sample_fault(
    fleet: list[dict],
    scenario: str,
    rng: np.random.Generator,
) -> tuple[int | None, float | None, float | None]:
    """Pick (dep_fault, t_fault, phase) for one trial."""
    if scenario == "NOMINAL":
        return None, None, None
    dep_fault = int(rng.integers(0, N_DEP))
    t_nav = fleet[dep_fault]["t_nav"]
    phase = float(rng.uniform(0.1, 0.9))
    t_fault = float(t_nav[0]) + phase * (float(t_nav[-1]) - float(t_nav[0]))
    return dep_fault, t_fault, phase


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Single trial
# ╚══════════════════════════════════════════════════════════════════════════╝

def _run_trial(
    fleet: list[dict],
    cfg: FDIConfig,
    t0_abs: float,
    scenario: str,
    dep_fault: int | None,
    t_fault: float | None,
    rng: np.random.Generator,
) -> dict:
    """
    Run one FDI Monte-Carlo trial.

    Returns
    -------
    dict with keys:
        scenario          : str
        dep_fault         : int or None
        t_fault           : float or None
        faulty_label      : str   identified label on the faulty deputy
                                  (the *first* label declared after t_fault;
                                  ``NONE`` if the deputy stayed HEALTHY)
        det_delay         : float or None   detection delay in seconds
        per_dep_labels    : dict[int, str]  final identified label on each
                                            healthy deputy (None on the
                                            faulty one)
    """
    expected = {
        "NOMINAL": "NONE",
        "NAV":     "NAV",
        "ATT":     "ATT",
        "PROP":    "PROP",
        "COMM":    "COMM",
    }[scenario]

    per_dep_labels: dict[int, str] = {}
    faulty_label = "NONE"
    det_delay: float | None = None

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

        t_att_rel = t_nav - t0_abs

        fdi = FDICore(cfg)
        fdi.reset(t0=float(t_nav[0]))

        faulty = (dep_fault is not None) and (d == dep_fault)

        # track the first declared non-NONE label on this deputy
        first_label = "NONE"
        first_label_delay: float | None = None

        hb_age = HB_NOMINAL

        for k in range(t_nav.size):
            t_k    = float(t_nav[k])
            t_relk = float(t_att_rel[k])
            inject_now = (faulty and t_fault is not None and t_k >= t_fault)

            # NAV channel
            innov = y_pre[k].copy()
            if np.any(~np.isfinite(innov)):
                innov_k = None
                S_innov_k = None
            else:
                if inject_now and scenario == "NAV":
                    bias = NAV_BIAS_SIGMA * sigma_meas[k]
                    innov += bias
                innov_k = innov
                sigma_k = np.maximum(sigma_meas[k], 1e-3)
                S_innov_k = np.diag(sigma_k ** 2)

            # PROP channel
            dv_cmd = dv_seq_H[k]
            burn_flag = bool(la.norm(dv_cmd) > 0.0)
            dv_est_H = None
            P_vv = None
            if burn_flag and k > 0:
                v_prev = x_est[k-1, 3:6]
                v_now  = x_est[k,   3:6]
                dv_est = (v_now - v_prev) - dv_cmd
                P_vv   = P_hist[k, 3:6, 3:6]
                if inject_now and scenario == "PROP":
                    sigma_v = np.sqrt(np.maximum(np.diag(P_vv), 1e-6))
                    dv_est = dv_est + PROP_FAULT_SIGMA * sigma_v * rng.normal(size=3)
                dv_est_H = dv_est

            # ATT channel
            sig_BR_k = _interp_vec_series(t_err, sigma_BR,   t_relk)
            omg_BR_k = _interp_vec_series(t_err, omega_BR_B, t_relk)
            if inject_now and scenario == "ATT":
                sig_BR_k = sig_BR_k + np.deg2rad(ATT_BIAS_SIGMA_DEG) * _rand_unit_vec(rng)
                omg_BR_k = omg_BR_k + np.deg2rad(ATT_RATE_BIAS_DEG) * _rand_unit_vec(rng)

            # COMM channel
            dt = 0.0 if k == 0 else float(t_nav[k] - t_nav[k-1])
            if inject_now and scenario == "COMM":
                hb_age += dt
            else:
                hb_age = HB_NOMINAL

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
                r_form=None,
                r_peer=None,
            )

            cur_state = snap.state.state
            cur_ft    = snap.state.fault_type

            if (cur_state in (FDIState.SUSPECT, FDIState.FAULTED, FDIState.DEAD)
                    and cur_ft is not FaultType.NONE):
                # only count detections that occur after fault injection on the
                # faulty deputy; on healthy deputies any non-NONE label is
                # a false alarm
                if faulty:
                    if (t_fault is not None) and (t_k < t_fault):
                        continue
                    if first_label == "NONE":
                        first_label = cur_ft.name
                        first_label_delay = max(0.0, t_k - (t_fault or t_k))
                else:
                    if first_label == "NONE":
                        first_label = cur_ft.name

        if faulty:
            faulty_label = first_label
            det_delay = first_label_delay
        else:
            per_dep_labels[d] = first_label

    return dict(
        scenario=scenario,
        dep_fault=dep_fault,
        t_fault=t_fault,
        expected=expected,
        faulty_label=faulty_label,
        det_delay=det_delay,
        per_dep_labels=per_dep_labels,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Confusion-matrix accounting
# ╚══════════════════════════════════════════════════════════════════════════╝

# Rows are truth labels; columns are identified labels.  ``NONE`` represents
# "deputy stayed HEALTHY" — for a faulty deputy this is a miss, for a healthy
# deputy this is a correct non-detection.
TRUTH_LABELS = ["NONE", "NAV", "ATT", "PROP", "COMM"]
PRED_LABELS  = ["NONE", "NAV", "ATT", "PROP", "COMM", "FORM", "PEER"]


def _label_index(label: str, labels: list[str]) -> int:
    try:
        return labels.index(label)
    except ValueError:
        return -1


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Plotting
# ╚══════════════════════════════════════════════════════════════════════════╝

def _plot_confusion_heatmap(M: np.ndarray, out_path: Path) -> None:
    """Render confusion matrix as a heatmap with row-normalised colors."""
    M = np.asarray(M, int)
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    M_norm = M / row_sums

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    im = ax.imshow(M_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-normalised fraction")

    ax.set_xticks(range(len(PRED_LABELS)))
    ax.set_xticklabels(PRED_LABELS, rotation=20, ha="right")
    ax.set_yticks(range(len(TRUTH_LABELS)))
    ax.set_yticklabels(TRUTH_LABELS)
    ax.set_xlabel("Identified label on faulty deputy")
    ax.set_ylabel("Truth (injected fault type)")
    ax.set_title("Isolation confusion matrix")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            n = M[i, j]
            if n == 0:
                continue
            txt_color = "white" if M_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{n}\n({100.0*M_norm[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=8, color=txt_color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[plot] {out_path.name}")
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Entry point
# ╚══════════════════════════════════════════════════════════════════════════╝

def main(
    data_dir: str | Path | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = DEFAULT_SEED,
) -> None:
    root = Path(data_dir).resolve() if data_dir is not None else SCRIPT_DIR

    print(f"\nFDI isolation MC over Apophis fleet in {root}")
    print(f"  trials/scenario = {n_trials}, seed = {seed}")

    t0_abs = _find_global_t0_abs(root)
    fleet = _load_fleet_data(root)

    # NAV/ATT/PROP/COMM FDI bank (FORM/PEER deliberately disabled here).
    cfg = FDIConfig(
        use_nav_channel=True,
        use_att_channel=True,
        use_comm_channel=True,
        use_form_channel=False,
        use_peer_channel=False,
    )
    cfg.nu_nav   = 1.5
    cfg.leak_nav = 0.05
    cfg.h_nav    = 15.0

    rng = np.random.default_rng(seed)

    # confusion matrix accumulator: (truth, predicted_on_faulty)
    M = np.zeros((len(TRUTH_LABELS), len(PRED_LABELS)), dtype=int)
    # per-scenario counters for the headline table
    counts: dict[str, dict] = {
        sc: dict(n_trials=0, n_faulty=0, n_correct=0, n_missed=0,
                 n_false_alarm=0, det_delays=[])
        for sc in SCENARIOS_INJECTED
    }

    for sc in SCENARIOS_INJECTED:
        print(f"\n=== Scenario {sc} ===")
        for trial in range(n_trials):
            dep_fault, t_fault, _ = _sample_fault(fleet, sc, rng)
            res = _run_trial(fleet, cfg, t0_abs, sc, dep_fault, t_fault, rng)

            counts[sc]["n_trials"] += 1
            if dep_fault is not None:
                counts[sc]["n_faulty"] += 1
                i = _label_index(res["expected"], TRUTH_LABELS)
                j = _label_index(res["faulty_label"], PRED_LABELS)
                if i >= 0 and j >= 0:
                    M[i, j] += 1
                if res["faulty_label"] == res["expected"]:
                    counts[sc]["n_correct"] += 1
                    if res["det_delay"] is not None:
                        counts[sc]["det_delays"].append(res["det_delay"])
                elif res["faulty_label"] == "NONE":
                    counts[sc]["n_missed"] += 1
                # else: wrong-label detection, already captured in confusion M
            else:
                # NOMINAL trial: any non-NONE label on any deputy is a FA
                i = _label_index("NONE", TRUTH_LABELS)
                for d, lab in res["per_dep_labels"].items():
                    j = _label_index(lab, PRED_LABELS)
                    if i >= 0 and j >= 0:
                        M[i, j] += 1
                if any(lab != "NONE" for lab in res["per_dep_labels"].values()):
                    counts[sc]["n_false_alarm"] += 1

            # also accumulate false alarms on the healthy deputies in
            # non-nominal trials
            if dep_fault is not None:
                if any(lab != "NONE" for lab in res["per_dep_labels"].values()):
                    counts[sc]["n_false_alarm"] += 1

            if (trial + 1) % 50 == 0 or trial == 0:
                print(f"  trial {trial+1}/{n_trials}", end="\r")
        print(f"  completed {counts[sc]['n_trials']} trials")

    # ── console summary ─────────────────────────────────────────────────────
    print("\n========== Isolation summary ==========")
    for sc in SCENARIOS_INJECTED:
        c = counts[sc]
        nf = max(c["n_faulty"], 1)
        Pd = c["n_correct"] / nf
        Pd_lo, Pd_hi = wilson_ci(c["n_correct"], nf)
        print(f"  {sc:8s}  trials={c['n_trials']:5d}  faulty={c['n_faulty']:5d}  "
              f"Pd={100*Pd:5.2f}% [{100*Pd_lo:.2f},{100*Pd_hi:.2f}]  "
              f"FA={c['n_false_alarm']}")

    print("\nConfusion matrix (rows=truth, cols=identified):")
    header = "          " + "  ".join(f"{l:>5s}" for l in PRED_LABELS)
    print(header)
    for i, t in enumerate(TRUTH_LABELS):
        row = "  ".join(f"{M[i,j]:>5d}" for j in range(len(PRED_LABELS)))
        print(f"  {t:6s} {row}")

    # ── save NPZ + LaTeX table + heatmap ───────────────────────────────────
    out_npz = root / "fdi_mc_isolation_results.npz"

    meta = build_metadata(
        rng_seed=seed,
        n_trials=n_trials,
        fdi_cfg=cfg,
        fault_params=dict(
            NAV_BIAS_SIGMA=NAV_BIAS_SIGMA,
            ATT_BIAS_SIGMA_DEG=ATT_BIAS_SIGMA_DEG,
            ATT_RATE_BIAS_DEG=ATT_RATE_BIAS_DEG,
            PROP_FAULT_SIGMA=PROP_FAULT_SIGMA,
            HB_NOMINAL=HB_NOMINAL,
            n_dep=N_DEP,
        ),
        script_name="apophis_swarm_fdi_mc_isolation.py",
    )

    np.savez_compressed(
        out_npz,
        confusion_truth_labels=np.array(TRUTH_LABELS),
        confusion_pred_labels=np.array(PRED_LABELS),
        confusion_matrix=M,
        scenarios=np.array(SCENARIOS_INJECTED),
        n_trials=np.array([counts[sc]["n_trials"] for sc in SCENARIOS_INJECTED]),
        n_faulty=np.array([counts[sc]["n_faulty"] for sc in SCENARIOS_INJECTED]),
        n_correct=np.array([counts[sc]["n_correct"] for sc in SCENARIOS_INJECTED]),
        n_missed=np.array([counts[sc]["n_missed"] for sc in SCENARIOS_INJECTED]),
        n_false_alarm=np.array([counts[sc]["n_false_alarm"] for sc in SCENARIOS_INJECTED]),
        # per-scenario delay arrays kept variable length via object dtype
        det_delays=np.array(
            [np.asarray(counts[sc]["det_delays"], float) for sc in SCENARIOS_INJECTED],
            dtype=object,
        ),
        metadata=np.array(meta, dtype=object),
    )
    print(f"\n[npz] wrote {out_npz.name}")

    emit_confusion_matrix_tex(out_npz, out_path=root / "fdi_mc_isolation.tex")
    _plot_confusion_heatmap(M, root / "fdi_mc_isolation_confusion.png")

    print("\nIsolation MC complete.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FDI isolation Monte-Carlo.")
    p.add_argument("--data-dir", default=None,
                   help="Directory with the NPZ bundles (default = script dir).")
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                   help=f"Trials per scenario (default {DEFAULT_N_TRIALS}).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"RNG seed (default {DEFAULT_SEED}).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    main(data_dir=args.data_dir, n_trials=args.n_trials, seed=args.seed)
