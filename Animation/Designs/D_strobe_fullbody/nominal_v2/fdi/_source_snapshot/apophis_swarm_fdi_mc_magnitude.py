#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_fdi_mc_magnitude.py
=================================

Fault-magnitude sweep for the Apophis FDI Monte-Carlo campaign.

Where ``apophis_swarm_fdi_mc.py`` injects each fault at one fixed,
deliberately easy-to-detect amplitude (NAV at 5\sigma_\rho, ATT at 1\deg /
0.05\deg/s, PROP at 5\sigma_v), this script sweeps the magnitude and
reports detection probability $P_d$ and detection delay $p_{50}/p_{95}$
versus magnitude for each channel — i.e. the detectability curve a journal
reviewer will ask for.

Sweep grids (default)
---------------------
  NAV  bias  :  {1, 1.5, 2, 2.5, 3, 4, 5, 7, 10}  in [\sigma_\rho]
  ATT  bias  :  {0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 2.00}  in [\deg]
                rate co-bias is scaled proportionally (0.05 deg/s at 1 deg)
  PROP bias  :  {1, 1.5, 2, 3, 4, 5, 7}  in [\sigma_v]

Per (scenario, magnitude) cell we run ``--n-trials`` trials (default 2000)
with one fault deputy per trial.  Wilson 95\,\% CIs are computed
downstream by ``fdi_mc_tex_table.py``.

Outputs
-------
  fdi_mc_magnitude_results.npz   :  per-cell counts + delay stats
  fdi_mc_magnitude_pd_curve.png  :  Pd vs magnitude with CI shading
  fdi_mc_magnitude_delay.png     :  delay p50/p95 vs magnitude
  fdi_mc_magnitude.tex           :  LaTeX results table

Usage
-----
    python apophis_swarm_fdi_mc_magnitude.py [--n-trials 2000] [--seed 12345]
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
    _find_global_t0_abs,
    _interp_vec_series,
    _load_fleet_data,
)
from fdi_mc_tex_table import (  # noqa: E402
    build_metadata,
    emit_magnitude_table_tex,
    wilson_ci,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Sweep configuration
# ╚══════════════════════════════════════════════════════════════════════════╝

DEFAULT_N_TRIALS = 2000
DEFAULT_SEED     = 12345

# Each entry: (scenario_name, magnitudes, unit, units_text)
SWEEPS = [
    ("NAV",  np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]),
     "sigma_rho", r"$\sigma_\rho$"),
    ("ATT",  np.array([0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 2.00]),
     "deg", r"$^\circ$"),
    ("PROP", np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]),
     "sigma_v", r"$\sigma_v$"),
]

# Co-scale of attitude rate bias with attitude bias: 0.05 deg/s at 1 deg
ATT_RATE_PER_ANGLE = 0.05   # [deg/s per deg]

HB_NOMINAL = 60.0


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Helpers
# ╚══════════════════════════════════════════════════════════════════════════╝

def _rand_unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = la.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0])


def _sample_fault_phase(
    fleet: list[dict],
    rng: np.random.Generator,
) -> tuple[int, float, float]:
    dep_fault = int(rng.integers(0, N_DEP))
    t_nav = fleet[dep_fault]["t_nav"]
    phase = float(rng.uniform(0.1, 0.9))
    t_fault = float(t_nav[0]) + phase * (float(t_nav[-1]) - float(t_nav[0]))
    return dep_fault, t_fault, phase


# ╔══════════════════════════════════════════════════════════════════════════╗
#  One trial at one magnitude
# ╚══════════════════════════════════════════════════════════════════════════╝

def _run_trial(
    fleet: list[dict],
    cfg: FDIConfig,
    t0_abs: float,
    scenario: str,
    magnitude: float,
    rng: np.random.Generator,
) -> dict:
    """
    One MC trial.

    The fault is injected on a randomly chosen deputy at a random phase in
    [0.1, 0.9] of that deputy's nav arc.  Returns
    ``{detected: bool, delay: float | None, false_alarm: bool}``.
    """
    dep_fault, t_fault, _ = _sample_fault_phase(fleet, rng)
    expected = FaultType[scenario]

    detected = False
    delay: float | None = None
    false_alarm = False

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

        faulty = (d == dep_fault)
        hb_age = HB_NOMINAL

        for k in range(t_nav.size):
            t_k    = float(t_nav[k])
            t_relk = float(t_att_rel[k])
            inject_now = faulty and (t_k >= t_fault)

            innov = y_pre[k].copy()
            if np.any(~np.isfinite(innov)):
                innov_k = None
                S_innov_k = None
            else:
                if inject_now and scenario == "NAV":
                    innov += magnitude * sigma_meas[k]
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
                dv_est = (v_now - v_prev) - dv_cmd
                P_vv   = P_hist[k, 3:6, 3:6]
                if inject_now and scenario == "PROP":
                    sigma_v = np.sqrt(np.maximum(np.diag(P_vv), 1e-6))
                    dv_est = dv_est + magnitude * sigma_v * rng.normal(size=3)
                dv_est_H = dv_est

            sig_BR_k = _interp_vec_series(t_err, sigma_BR,   t_relk)
            omg_BR_k = _interp_vec_series(t_err, omega_BR_B, t_relk)
            if inject_now and scenario == "ATT":
                ang_rad  = np.deg2rad(magnitude)
                rate_rad = np.deg2rad(magnitude * ATT_RATE_PER_ANGLE)
                sig_BR_k = sig_BR_k + ang_rad  * _rand_unit_vec(rng)
                omg_BR_k = omg_BR_k + rate_rad * _rand_unit_vec(rng)

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
            tripped = cur_state in (FDIState.SUSPECT, FDIState.FAULTED, FDIState.DEAD)

            if faulty:
                if (t_k >= t_fault) and tripped and cur_ft == expected and not detected:
                    detected = True
                    delay = max(0.0, t_k - t_fault)
            else:
                if tripped and cur_ft is not FaultType.NONE:
                    false_alarm = True

    return dict(detected=detected, delay=delay, false_alarm=false_alarm)


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Plotting
# ╚══════════════════════════════════════════════════════════════════════════╝

CHANNEL_COLOR = {
    "NAV":  "#0072B2",
    "ATT":  "#009E73",
    "PROP": "#D55E00",
}


def _plot_pd_curves(results: dict, out_path: Path) -> None:
    """Pd vs magnitude per scenario with Wilson 95% CI shading."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for sc, mags, _, units_text in SWEEPS:
        r = results[sc]
        Pd  = np.array([nc / max(nf, 1) for nc, nf in zip(r["n_correct"], r["n_faulty"])])
        cis = [wilson_ci(nc, max(nf, 1)) for nc, nf in zip(r["n_correct"], r["n_faulty"])]
        lo  = np.array([c[0] for c in cis])
        hi  = np.array([c[1] for c in cis])
        c = CHANNEL_COLOR[sc]
        ax.plot(mags, 100 * Pd, "o-", color=c, label=sc, linewidth=1.5)
        ax.fill_between(mags, 100 * lo, 100 * hi, color=c, alpha=0.18)

    ax.set_xlabel("Fault magnitude (channel-native units)")
    ax.set_ylabel(r"Detection probability $P_d$  [%]")
    ax.set_title("FDI detectability curves with Wilson 95% confidence intervals")
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[plot] {out_path.name}")
    plt.close(fig)


def _plot_delay_curves(results: dict, out_path: Path) -> None:
    """Detection delay p50 and p95 vs magnitude per scenario."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for sc, mags, _, units_text in SWEEPS:
        r = results[sc]
        p50 = np.array(r["delay_p50_s"])
        p95 = np.array(r["delay_p95_s"])
        c = CHANNEL_COLOR[sc]
        mask = np.isfinite(p50)
        ax.plot(mags[mask], p50[mask], "o-", color=c, label=f"{sc} (p50)", linewidth=1.5)
        mask = np.isfinite(p95)
        ax.plot(mags[mask], p95[mask], "o--", color=c, label=f"{sc} (p95)",
                linewidth=1.1, alpha=0.7)

    ax.set_xlabel("Fault magnitude (channel-native units)")
    ax.set_ylabel("Detection delay [s]")
    ax.set_title("Detection delay vs fault magnitude")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
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

    print(f"\nFDI magnitude sweep over Apophis fleet in {root}")
    print(f"  trials/cell = {n_trials}, seed = {seed}")

    t0_abs = _find_global_t0_abs(root)
    fleet = _load_fleet_data(root)

    cfg = FDIConfig(
        use_nav_channel=True,
        use_att_channel=True,
        use_comm_channel=False,
        use_form_channel=False,
        use_peer_channel=False,
    )
    cfg.nu_nav   = 1.5
    cfg.leak_nav = 0.05
    cfg.h_nav    = 15.0

    rng = np.random.default_rng(seed)

    results: dict[str, dict] = {}
    n_sc = len(SWEEPS)
    n_mag_max = max(len(m) for _, m, _, _ in SWEEPS)

    # buffers sized to the maximum magnitude grid; unused cells are NaN
    mag_grid    = np.full((n_sc, n_mag_max), np.nan)
    nt_grid     = np.zeros((n_sc, n_mag_max), dtype=int)
    nf_grid     = np.zeros((n_sc, n_mag_max), dtype=int)
    nc_grid     = np.zeros((n_sc, n_mag_max), dtype=int)
    fa_grid     = np.zeros((n_sc, n_mag_max), dtype=int)
    p50_grid    = np.full((n_sc, n_mag_max), np.nan)
    p95_grid    = np.full((n_sc, n_mag_max), np.nan)
    pmean_grid  = np.full((n_sc, n_mag_max), np.nan)

    for i_sc, (sc, mags, _, _) in enumerate(SWEEPS):
        print(f"\n=== Scenario {sc} ===")
        per_cell = dict(n_trials=[], n_faulty=[], n_correct=[], n_false_alarm=[],
                        delay_p50_s=[], delay_p95_s=[], delay_mean_s=[])

        for j_m, mag in enumerate(mags):
            mag_grid[i_sc, j_m] = mag
            nc = 0
            n_fa = 0
            delays: list[float] = []
            for trial in range(n_trials):
                res = _run_trial(fleet, cfg, t0_abs, sc, mag, rng)
                if res["detected"]:
                    nc += 1
                    if res["delay"] is not None:
                        delays.append(res["delay"])
                if res["false_alarm"]:
                    n_fa += 1
                if (trial + 1) % 100 == 0:
                    print(f"  mag={mag:6.2f}  trial {trial+1}/{n_trials}", end="\r")
            print(f"  mag={mag:6.2f}  Pd={100*nc/max(n_trials,1):6.2f}%  "
                  f"FA={n_fa}/{n_trials}  delays={len(delays)}    ")

            nt_grid[i_sc, j_m] = n_trials
            nf_grid[i_sc, j_m] = n_trials   # one faulty deputy per trial
            nc_grid[i_sc, j_m] = nc
            fa_grid[i_sc, j_m] = n_fa
            if delays:
                arr = np.asarray(delays)
                p50_grid[i_sc, j_m]   = float(np.percentile(arr, 50))
                p95_grid[i_sc, j_m]   = float(np.percentile(arr, 95))
                pmean_grid[i_sc, j_m] = float(arr.mean())

            per_cell["n_trials"].append(n_trials)
            per_cell["n_faulty"].append(n_trials)
            per_cell["n_correct"].append(nc)
            per_cell["n_false_alarm"].append(n_fa)
            per_cell["delay_p50_s"].append(p50_grid[i_sc, j_m])
            per_cell["delay_p95_s"].append(p95_grid[i_sc, j_m])
            per_cell["delay_mean_s"].append(pmean_grid[i_sc, j_m])

        results[sc] = per_cell

    # ── persist ─────────────────────────────────────────────────────────────
    meta = build_metadata(
        rng_seed=seed,
        n_trials=n_trials,
        fdi_cfg=cfg,
        fault_params=dict(
            sweeps={sc: list(map(float, mags)) for sc, mags, _, _ in SWEEPS},
            ATT_RATE_PER_ANGLE=ATT_RATE_PER_ANGLE,
            n_dep=N_DEP,
        ),
        script_name="apophis_swarm_fdi_mc_magnitude.py",
    )

    out_npz = root / "fdi_mc_magnitude_results.npz"
    np.savez_compressed(
        out_npz,
        scenarios=np.array([sc for sc, _, _, _ in SWEEPS]),
        magnitudes=mag_grid,
        magnitude_units=np.array([u for _, _, u, _ in SWEEPS]),
        magnitude_units_tex=np.array([t for _, _, _, t in SWEEPS]),
        n_trials=nt_grid,
        n_faulty=nf_grid,
        n_correct=nc_grid,
        n_false_alarm=fa_grid,
        delay_p50_s=p50_grid,
        delay_p95_s=p95_grid,
        delay_mean_s=pmean_grid,
        metadata=np.array(meta, dtype=object),
    )
    print(f"\n[npz] wrote {out_npz.name}")

    emit_magnitude_table_tex(out_npz, out_path=root / "fdi_mc_magnitude.tex")
    _plot_pd_curves(results, root / "fdi_mc_magnitude_pd_curve.png")
    _plot_delay_curves(results, root / "fdi_mc_magnitude_delay.png")

    print("\nMagnitude sweep complete.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FDI magnitude sweep MC.")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    main(data_dir=args.data_dir, n_trials=args.n_trials, seed=args.seed)
