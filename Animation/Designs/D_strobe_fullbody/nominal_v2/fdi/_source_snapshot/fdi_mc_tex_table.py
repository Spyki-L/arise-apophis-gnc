#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fdi_mc_tex_table.py
===================

LaTeX-table emitters and shared statistics helpers for the FDI Monte-Carlo
campaign.  All FDI MC scripts (``apophis_swarm_fdi_mc.py``,
``..._isolation.py``, ``..._magnitude.py``, ``..._formpeer.py``,
``..._peer_stress.py``) use this module so the same metric definitions and
the same Wilson confidence-interval implementation appear in every result
table.

Exports
-------
wilson_ci(k, n, alpha=0.05)
    Wilson 95 % confidence interval for a binomial proportion p = k/n.

fmt_pct(p, lo, hi, digits=2)
    Format a proportion + Wilson interval as ``94.20 [93.83, 94.54] %``.

fmt_delay_stats(delays_s)
    Format a delay-distribution summary as ``58.7 / 91.2 / 162.4 (s)``.

tex_escape(s)
    Escape LaTeX-special characters in a label so it renders in text mode.

build_metadata(rng_seed, n_trials, fdi_cfg, fault_params, script_name)
    Build a JSON-encodable string of reproducibility metadata.

emit_main_results_table_tex(npz_path, out_path=None)
emit_confusion_matrix_tex(npz_path, out_path=None)
emit_magnitude_table_tex(npz_path, out_path=None)
emit_formpeer_table_tex(npz_path, out_path=None)

All emitters write a booktabs-style table that can be \\input directly
into the journal paper.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Iterable

import numpy as np


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Statistics helpers
# ╚══════════════════════════════════════════════════════════════════════════╝

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion p_hat = k/n.

    Robust at p near 0 or 1 and small n.  Returns (lo, hi) in [0, 1].
    If n <= 0 returns (0.0, 1.0) as a non-informative interval.
    """
    if n <= 0:
        return 0.0, 1.0
    z = _normal_quantile(1.0 - alpha / 2.0)
    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = (
        z * sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
        / denom
    )
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def _normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF via Acklam's rational approximation."""
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = (-2.0 * np.log(p)) ** 0.5
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    q = (-2.0 * np.log(1.0 - p)) ** 0.5
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Formatting
# ╚══════════════════════════════════════════════════════════════════════════╝

def fmt_pct(p: float, lo: float, hi: float, digits: int = 2) -> str:
    """Format ``p [lo, hi]`` as percent, e.g. ``94.20 [93.83, 94.54]``."""
    return f"{100.0*p:.{digits}f} [{100.0*lo:.{digits}f}, {100.0*hi:.{digits}f}]"


def fmt_delay_stats(delays_s: Iterable[float]) -> str:
    """Format detection-delay distribution as ``p50 / mean / p95`` seconds."""
    arr = np.asarray(list(delays_s), float)
    if arr.size == 0:
        return "--/--/--"
    return (
        f"{np.percentile(arr, 50):.1f} / "
        f"{arr.mean():.1f} / "
        f"{np.percentile(arr, 95):.1f}"
    )


def tex_escape(s: str) -> str:
    """Escape LaTeX-special characters in a label."""
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("_", r"\_")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("#", r"\#")
         .replace("$", r"\$")
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Reproducibility metadata
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_metadata(
    rng_seed: int,
    n_trials: int,
    fdi_cfg,                # FDIConfig instance
    fault_params: dict,
    script_name: str,
) -> str:
    """Build a JSON-encoded metadata string for inclusion in NPZ outputs."""
    from dataclasses import asdict, is_dataclass

    cfg_dict = asdict(fdi_cfg) if is_dataclass(fdi_cfg) else dict(vars(fdi_cfg))
    cfg_clean = {k: _to_json_safe(v) for k, v in cfg_dict.items()}
    fault_clean = {k: _to_json_safe(v) for k, v in fault_params.items()}

    meta = dict(
        script=script_name,
        rng_seed=int(rng_seed),
        n_trials_per_scenario=int(n_trials),
        fdi_config=cfg_clean,
        fault_params=fault_clean,
    )
    return json.dumps(meta, indent=2)


def _to_json_safe(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_to_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_json_safe(x) for k, x in v.items()}
    return v


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Primary results table  (from fdi_mc_results.npz)
# ╚══════════════════════════════════════════════════════════════════════════╝

def emit_main_results_table_tex(
    npz_path: str | Path,
    out_path: str | Path | None = None,
    n_dep: int = 8,
) -> str:
    """Primary Monte-Carlo results table: Pd, Pfa, delay statistics."""
    data = np.load(npz_path, allow_pickle=True)
    scenarios = ["NAV", "ATT", "PROP", "COMM"]

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"  \caption{Monte-Carlo detection performance per fault "
                 r"scenario.  $P_d$ is the probability that the faulty deputy "
                 r"is correctly labelled before the end of its nav arc.  "
                 r"$P_{fa}$ is the per-healthy-deputy-per-encounter false-alarm "
                 r"rate.  Bracketed values are Wilson 95\,\% confidence "
                 r"intervals.  Detection delays are reported as median / mean "
                 r"/ 95\textsuperscript{th} percentile.}")
    lines.append(r"  \label{tab:fdi_mc_primary}")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{l c c c c}")
    lines.append(r"    \hline")
    lines.append(r"    Scenario & $N_\mathrm{trials}$ & "
                 r"$P_d$ \,[\%] & $P_{fa}$ \,[\%] & "
                 r"delay $p_{50}/\bar{x}/p_{95}$ \,[s] \\")
    lines.append(r"    \hline")

    for sc in scenarios:
        n_trials   = int(data[f"{sc}_n_trials"])
        n_faulty   = int(data[f"{sc}_n_faulty_deps"])
        n_correct  = int(data[f"{sc}_n_detected_correct"])
        n_fa       = int(data[f"{sc}_n_false_alarm"])
        delays     = np.asarray(data[f"{sc}_det_delays"], float)
        n_healthy  = n_trials * (n_dep - 1)

        Pd = n_correct / max(n_faulty, 1)
        Pd_lo, Pd_hi = wilson_ci(n_correct, max(n_faulty, 1))
        Pfa = n_fa / max(n_healthy, 1)
        Pfa_lo, Pfa_hi = wilson_ci(n_fa, max(n_healthy, 1))

        lines.append(
            f"    {tex_escape(sc)} & {n_trials} & "
            f"{fmt_pct(Pd, Pd_lo, Pd_hi)} & "
            f"{fmt_pct(Pfa, Pfa_lo, Pfa_hi)} & "
            f"{fmt_delay_stats(delays)} \\\\"
        )

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines) + "\n"
    if out_path is not None:
        Path(out_path).write_text(tex, encoding="utf-8")
        print(f"[tex] wrote {Path(out_path).name}")
    return tex


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Confusion matrix table  (from fdi_mc_isolation_results.npz)
# ╚══════════════════════════════════════════════════════════════════════════╝

def emit_confusion_matrix_tex(
    npz_path: str | Path,
    out_path: str | Path | None = None,
) -> str:
    """Truth-vs-predicted confusion matrix from the isolation MC."""
    data = np.load(npz_path, allow_pickle=True)
    truth = [str(x) for x in data["confusion_truth_labels"]]
    pred  = [str(x) for x in data["confusion_pred_labels"]]
    M     = np.asarray(data["confusion_matrix"], int)

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"  \caption{Isolation accuracy: confusion matrix of injected "
                 r"fault type (rows) versus identified fault label on the "
                 r"faulty deputy (columns).  Diagonal entries are correct "
                 r"isolations; off-diagonal entries are mis-labels or misses.}")
    lines.append(r"  \label{tab:fdi_isolation}")
    lines.append(r"  \centering")
    col_spec = "l " + " ".join(["c"] * len(pred))
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \hline")
    header = ("    Truth $\\backslash$ Identified & "
              + " & ".join(tex_escape(p) for p in pred) + r" \\")
    lines.append(header)
    lines.append(r"    \hline")
    for i, t in enumerate(truth):
        row_total = int(M[i].sum())
        cells = []
        for j in range(len(pred)):
            n = int(M[i, j])
            if row_total > 0:
                pct = 100.0 * n / row_total
                cells.append(f"{n} ({pct:.1f}\\%)")
            else:
                cells.append(f"{n}")
        lines.append("    " + tex_escape(t) + " & " + " & ".join(cells) + r" \\")
    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines) + "\n"
    if out_path is not None:
        Path(out_path).write_text(tex, encoding="utf-8")
        print(f"[tex] wrote {Path(out_path).name}")
    return tex


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Magnitude sweep table  (from fdi_mc_magnitude_results.npz)
# ╚══════════════════════════════════════════════════════════════════════════╝

def emit_magnitude_table_tex(
    npz_path: str | Path,
    out_path: str | Path | None = None,
) -> str:
    """Magnitude-sweep table: Pd ± CI and delay percentiles per cell."""
    data = np.load(npz_path, allow_pickle=True)
    scenarios = [str(x) for x in data["scenarios"]]
    magnitudes = np.asarray(data["magnitudes"], float)
    units = [str(x) for x in data["magnitude_units"]]
    n_trials = np.asarray(data["n_trials"], int)
    n_faulty = np.asarray(data["n_faulty"], int)
    n_correct = np.asarray(data["n_correct"], int)
    delay_p50 = np.asarray(data["delay_p50_s"], float)
    delay_p95 = np.asarray(data["delay_p95_s"], float)

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"  \caption{Detection probability and detection delay as a "
                 r"function of injected fault magnitude.  Magnitudes are "
                 r"expressed in the natural unit of each channel "
                 r"($\sigma_\rho$ for NAV, $^\circ$ rms for ATT, $\sigma_v$ for "
                 r"PROP).  Bracketed values are Wilson 95\,\% intervals; "
                 r"$p_{50}/p_{95}$ are detection-delay percentiles in seconds "
                 r"(`--' marks cells with no successful detections).}")
    lines.append(r"  \label{tab:fdi_magnitude}")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{l l c c c c}")
    lines.append(r"    \hline")
    lines.append(r"    Scenario & Magnitude & $N_\mathrm{trials}$ & "
                 r"$P_d$ \,[\%] & $p_{50}$ \,[s] & $p_{95}$ \,[s] \\")
    lines.append(r"    \hline")

    for i, sc in enumerate(scenarios):
        for j in range(magnitudes.shape[1]):
            mag = magnitudes[i, j]
            if not np.isfinite(mag):
                continue
            nt = int(n_trials[i, j])
            nf = int(n_faulty[i, j])
            nc = int(n_correct[i, j])
            Pd = nc / max(nf, 1)
            Pd_lo, Pd_hi = wilson_ci(nc, max(nf, 1))
            p50 = delay_p50[i, j]
            p95 = delay_p95[i, j]
            p50_s = f"{p50:.1f}" if np.isfinite(p50) else "--"
            p95_s = f"{p95:.1f}" if np.isfinite(p95) else "--"
            lines.append(
                f"    {tex_escape(sc)} & {mag:.2f}\\,{tex_escape(units[i])} "
                f"& {nt} & {fmt_pct(Pd, Pd_lo, Pd_hi)} & "
                f"{p50_s} & {p95_s} \\\\"
            )
        if i < len(scenarios) - 1:
            lines.append(r"    \hline")
    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines) + "\n"
    if out_path is not None:
        Path(out_path).write_text(tex, encoding="utf-8")
        print(f"[tex] wrote {Path(out_path).name}")
    return tex


# ╔══════════════════════════════════════════════════════════════════════════╗
#  Form/peer table  (from fdi_mc_formpeer_results.npz)
# ╚══════════════════════════════════════════════════════════════════════════╝

def emit_formpeer_table_tex(
    npz_path: str | Path,
    out_path: str | Path | None = None,
    n_dep: int = 8,
) -> str:
    """Local-vs-global isolation table from the FORM/PEER MC."""
    data = np.load(npz_path, allow_pickle=True)
    scenarios = [str(x) for x in data["scenarios"]]
    n_trials = np.asarray(data["n_trials"], int)
    n_faulty = np.asarray(data["n_faulty"], int)
    n_correct = np.asarray(data["n_correct"], int)
    n_wrong = np.asarray(data["n_wrong_label"], int)
    n_missed = np.asarray(data["n_missed"], int)
    n_form_fleet = np.asarray(data["n_form_trips_fleet"], int)

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"  \caption{Local-vs-global isolation experiment.  "
                 r"\textsc{form\_local} injects a single-deputy formation "
                 r"fault and expects label \textsc{form}; \textsc{nav\_global} "
                 r"applies a common-mode bias to every deputy and expects "
                 r"label \textsc{nav} via the FORM-high, PEER-low precedence "
                 r"rule.  Bracketed values are Wilson 95\,\% intervals.  "
                 r"$N_\mathrm{FORM,fleet}$ counts trials where FORM trips on "
                 r"at least two deputies, i.e.\ exhibits the global signature.}")
    lines.append(r"  \label{tab:fdi_formpeer}")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{l c c c c c}")
    lines.append(r"    \hline")
    lines.append(r"    Scenario & $N_\mathrm{trials}$ & Correct label \,[\%] & "
                 r"Wrong label \,[\%] & Missed \,[\%] & "
                 r"$N_\mathrm{FORM,fleet}$ \\")
    lines.append(r"    \hline")
    for i, sc in enumerate(scenarios):
        nt = int(n_trials[i])
        nf = max(int(n_faulty[i]), 1)
        nc = int(n_correct[i])
        nw = int(n_wrong[i])
        nm = int(n_missed[i])
        n_fleet = int(n_form_fleet[i])
        p_c, lo_c, hi_c = (nc / nf,) + wilson_ci(nc, nf)
        p_w, lo_w, hi_w = (nw / nf,) + wilson_ci(nw, nf)
        p_m, lo_m, hi_m = (nm / nf,) + wilson_ci(nm, nf)
        lines.append(
            f"    {tex_escape(sc)} & {nt} & "
            f"{fmt_pct(p_c, lo_c, hi_c)} & "
            f"{fmt_pct(p_w, lo_w, hi_w)} & "
            f"{fmt_pct(p_m, lo_m, hi_m)} & "
            f"{n_fleet} \\\\"
        )
    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines) + "\n"
    if out_path is not None:
        Path(out_path).write_text(tex, encoding="utf-8")
        print(f"[tex] wrote {Path(out_path).name}")
    return tex


# ╔══════════════════════════════════════════════════════════════════════════╗
#  CLI
# ╚══════════════════════════════════════════════════════════════════════════╝

def _main_cli():
    """Emit any tables found alongside this script."""
    import argparse

    p = argparse.ArgumentParser(
        description="Emit LaTeX tables from FDI Monte-Carlo NPZ files."
    )
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parent),
                   help="Folder containing the *_results.npz files.")
    args = p.parse_args()
    root = Path(args.data_dir).resolve()

    pairs = [
        ("fdi_mc_results.npz",            emit_main_results_table_tex,    "fdi_mc_primary.tex"),
        ("fdi_mc_isolation_results.npz",  emit_confusion_matrix_tex,      "fdi_mc_isolation.tex"),
        ("fdi_mc_magnitude_results.npz",  emit_magnitude_table_tex,       "fdi_mc_magnitude.tex"),
        ("fdi_mc_formpeer_results.npz",   emit_formpeer_table_tex,        "fdi_mc_formpeer.tex"),
    ]
    for npz_name, fn, out_name in pairs:
        npz_path = root / npz_name
        if not npz_path.is_file():
            print(f"[skip] {npz_name} not found in {root}")
            continue
        fn(npz_path, out_path=root / out_name)


if __name__ == "__main__":
    _main_cli()
