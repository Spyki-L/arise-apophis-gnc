#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_fdi_mc_peer_stress.py
===================================

Homogeneous-fleet PEER stress test.

The PEER residual is

    r_peer_i = | r_form_i - median_j(r_form_j) |
               / max( MAD_j(r_form_j - median), MAD_floor )

If the MAD floor is zero or too small and the fleet is near-homogeneous
(every deputy on a tight, similar relative orbit), the MAD collapses
toward zero and r_peer diverges for any deputy with a benign sub-metre
geometric difference.  An earlier version of the FDI sweep tripped PEER
on D2 / D5 in exactly this way.

The fix is the documented MAD floor (``PEER_MAD_FLOOR = 0.25`` in
``apophis_swarm_fdi_main.py``).  This script *demonstrates* the failure
mode and the fix: it sweeps the MAD floor across a grid of values and,
for each, reports the fraction of trials in which PEER falsely trips on
a homogeneous, fault-free fleet.

Trial structure
---------------
Per trial:

  1. Take the nominal ``r_form`` produced by ``apophis_swarm_fdi_main``
     (no faults injected, sub-metre nav).
  2. Optionally perturb r_form with small zero-mean Gaussian jitter
     (``--jitter``, default 0.05) to break tied values that the
     deterministic CW predictor would otherwise produce.
  3. Recompute r_peer with a configurable MAD floor.
  4. Run the FDI core for every deputy with FORM and PEER enabled
     (everything else at nominal) and record whether *any* deputy ends
     the encounter in a SUSPECT / FAULTED / DEAD state.

Outputs
-------
  fdi_mc_peer_stress_results.npz  : per-floor counts + reproducibility meta
  fdi_mc_peer_stress_curve.png    : false PEER-trip fraction vs MAD floor

Usage
-----
    python apophis_swarm_fdi_mc_peer_stress.py \\
        [--n-trials 200] [--jitter 0.05] [--seed 12345]
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
    _interp_scalar_series,
    _load_fleet_data,
    _estimate_global_arrival_time,
    _compute_formation_residuals,
)
from fdi_mc_tex_table import build_metadata  # noqa: E402


DEFAULT_N_TRIALS    = 200
DEFAULT_JITTER      = 0.05      # std-dev of r_form perturbation in trial loop
DEFAULT_SEED        = 12345
HB_NOMINAL          = 60.0

# MAD-floor grid: log-spaced from 1e-3 (effectively off) to ~1.0
DEFAULT_FLOORS = np.array([0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00])


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Peer residual with configurable MAD floor
# ╚══════════════════════════════════════════════════════════════════════════╝

def _compute_peer_with_floor(fleet: list[dict], mad_floor: float) -> None:
    """Re-implementation of ``_compute_peer_residuals`` with a parametric floor."""
    t_nav_list  = [f["t_nav"]  for f in fleet]
    r_form_list = [f["r_form"] for f in fleet]

    for d in range(N_DEP):
        t_nav_d   = t_nav_list[d]
        r_form_d  = r_form_list[d]
        N_nav     = t_nav_d.size
        r_peer_d  = np.zeros(N_nav)

        for k in range(N_nav):
            t_k = float(t_nav_d[k])
            vals = []
            for j in range(N_DEP):
                t_j = t_nav_list[j]
                if t_k < t_j[0] or t_k > t_j[-1]:
                    continue
                vals.append(_interp_scalar_series(t_j, r_form_list[j], t_k))
            if len(vals) < 2:
                r_peer_d[k] = 0.0
                continue
            vals = np.asarray(vals, float)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            mad_eff = max(mad, mad_floor)
            denom = max(mad_eff, 1e-9)
            r_peer_d[k] = abs(r_form_d[k] - med) / denom

        fleet[d]["r_peer"] = r_peer_d


# ╔══════════════════════════════════════════════════════════════════════════╗
#  One trial
# ╚══════════════════════════════════════════════════════════════════════════╝

def _run_one_trial(
    fleet: list[dict],
    cfg: FDIConfig,
    t0_abs: float,
    mad_floor: float,
    jitter: float,
    rng: np.random.Generator,
) -> dict:
    """Run one homogeneous-fleet trial at a given MAD floor and jitter."""
    # perturb r_form across the fleet with zero-mean jitter to break ties
    for d in range(N_DEP):
        r_f0 = fleet[d]["_r_form_nominal"]
        r_f  = r_f0.copy()
        if jitter > 0.0:
            mask = r_f0 > 0.0       # only jitter active samples
            r_f[mask] = np.maximum(0.0, r_f[mask] + rng.normal(0.0, jitter,
                                                                size=mask.sum()))
        fleet[d]["r_form"] = r_f

    _compute_peer_with_floor(fleet, mad_floor)

    any_peer_trip = False
    any_form_trip = False
    any_non_healthy = False
    n_peer_deps = 0
    n_form_deps = 0
    max_r_peer = 0.0

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
        r_form     = data["r_form"]
        r_peer     = data["r_peer"]

        t_att_rel = t_nav - t0_abs
        fdi = FDICore(cfg)
        fdi.reset(t0=float(t_nav[0]))

        hb_age = HB_NOMINAL
        form_here = False
        peer_here = False
        non_healthy_here = False

        for k in range(t_nav.size):
            t_k = float(t_nav[k])
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

            if fdi.S_peer > cfg.h_peer:
                peer_here = True
            if fdi.S_form > cfg.h_form:
                form_here = True
            if snap.state.state in (FDIState.SUSPECT, FDIState.FAULTED, FDIState.DEAD):
                non_healthy_here = True

            if r_peer[k] > max_r_peer:
                max_r_peer = float(r_peer[k])

        if peer_here:
            any_peer_trip = True
            n_peer_deps += 1
        if form_here:
            any_form_trip = True
            n_form_deps += 1
        if non_healthy_here:
            any_non_healthy = True

    return dict(
        any_peer_trip=any_peer_trip,
        any_form_trip=any_form_trip,
        any_non_healthy=any_non_healthy,
        n_peer_deps=n_peer_deps,
        n_form_deps=n_form_deps,
        max_r_peer=max_r_peer,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Plotting
# ╚══════════════════════════════════════════════════════════════════════════╝

def _plot_floor_curve(floors: np.ndarray,
                      false_peer_frac: np.ndarray,
                      false_non_healthy_frac: np.ndarray,
                      out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.semilogx(np.where(floors == 0.0, 1e-4, floors), 100 * false_peer_frac,
                "o-", color="#CC79A7", label="any deputy trips PEER channel")
    ax.semilogx(np.where(floors == 0.0, 1e-4, floors),
                100 * false_non_healthy_frac,
                "s--", color="#D55E00",
                label="any deputy reaches SUSPECT/FAULTED")
    ax.axvline(0.25, color="0.4", linestyle=":", linewidth=1.1)
    ax.text(0.25, 95, " operating point\n MAD floor = 0.25",
            color="0.3", fontsize=9, va="top")
    ax.set_xlabel("PEER MAD floor")
    ax.set_ylabel("Trials with false PEER activity [%]")
    ax.set_title("Homogeneous-fleet PEER false-alarm rate vs MAD floor")
    ax.set_ylim(-3, 105)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="upper right")
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
    jitter: float = DEFAULT_JITTER,
    seed: int = DEFAULT_SEED,
    floors: np.ndarray = DEFAULT_FLOORS,
) -> None:
    root = Path(data_dir).resolve() if data_dir is not None else SCRIPT_DIR

    print(f"\nPEER-stress MC over Apophis fleet in {root}")
    print(f"  trials/floor = {n_trials}, jitter = {jitter}, seed = {seed}")
    print(f"  floors       = {list(floors)}")

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
    # stash nominal r_form so per-trial jitter can be applied fresh
    for d in range(N_DEP):
        fleet[d]["_r_form_nominal"] = fleet[d]["r_form"].copy()

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

    n_floors = len(floors)
    n_peer_trips_per_floor   = np.zeros(n_floors, dtype=int)
    n_form_trips_per_floor   = np.zeros(n_floors, dtype=int)
    n_non_healthy_per_floor  = np.zeros(n_floors, dtype=int)
    max_r_peer_per_floor     = np.zeros(n_floors)

    for i, floor in enumerate(floors):
        print(f"\n=== MAD floor = {floor:.4f} ===")
        max_seen = 0.0
        for trial in range(n_trials):
            res = _run_one_trial(fleet, cfg, t0_abs, float(floor), jitter, rng)
            if res["any_peer_trip"]:
                n_peer_trips_per_floor[i] += 1
            if res["any_form_trip"]:
                n_form_trips_per_floor[i] += 1
            if res["any_non_healthy"]:
                n_non_healthy_per_floor[i] += 1
            if res["max_r_peer"] > max_seen:
                max_seen = res["max_r_peer"]
            if (trial + 1) % 25 == 0 or trial == 0:
                print(f"  trial {trial+1}/{n_trials}  "
                      f"peer-trip-so-far={n_peer_trips_per_floor[i]}", end="\r")
        max_r_peer_per_floor[i] = max_seen
        print(f"  PEER trip in {n_peer_trips_per_floor[i]}/{n_trials} trials, "
              f"max r_peer seen = {max_seen:.2f}")

    false_peer_frac = n_peer_trips_per_floor / max(n_trials, 1)
    false_non_healthy_frac = n_non_healthy_per_floor / max(n_trials, 1)

    print("\n========== PEER-stress summary ==========")
    for i, floor in enumerate(floors):
        print(f"  floor={floor:8.4f}  PEER-trip fraction = "
              f"{100*false_peer_frac[i]:5.2f}%  "
              f"non-HEALTHY fraction = {100*false_non_healthy_frac[i]:5.2f}%  "
              f"max r_peer = {max_r_peer_per_floor[i]:.2f}")

    meta = build_metadata(
        rng_seed=seed,
        n_trials=n_trials,
        fdi_cfg=cfg,
        fault_params=dict(
            jitter=jitter,
            floors=list(map(float, floors)),
            n_dep=N_DEP,
        ),
        script_name="apophis_swarm_fdi_mc_peer_stress.py",
    )

    out_npz = root / "fdi_mc_peer_stress_results.npz"
    np.savez_compressed(
        out_npz,
        floors=np.asarray(floors, float),
        n_trials=np.array(n_trials, dtype=int),
        n_peer_trips=n_peer_trips_per_floor,
        n_form_trips=n_form_trips_per_floor,
        n_non_healthy=n_non_healthy_per_floor,
        max_r_peer=max_r_peer_per_floor,
        metadata=np.array(meta, dtype=object),
    )
    print(f"\n[npz] wrote {out_npz.name}")

    _plot_floor_curve(np.asarray(floors), false_peer_frac, false_non_healthy_frac,
                      root / "fdi_mc_peer_stress_curve.png")

    print("\nPEER-stress MC complete.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Homogeneous-fleet PEER stress MC.")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    p.add_argument("--jitter", type=float, default=DEFAULT_JITTER,
                   help="std-dev of zero-mean noise added to nominal r_form.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    main(
        data_dir=args.data_dir,
        n_trials=args.n_trials,
        jitter=args.jitter,
        seed=args.seed,
    )
