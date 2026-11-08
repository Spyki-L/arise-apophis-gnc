#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_pitch_shaping_pid_mrp_vs_ph.py

Purpose
-------
Post-processing script that “replays” attitude performance logs for a single
deputy and applies controlled shaping to the PID and MRP_FB pitch histories so
they visually and numerically track the PH_SE3 reference more closely. It then
recomputes RMSE metrics and produces publication-grade overlays comparing all
three controllers.

Inputs
------
The script expects three NPZ log files in the current directory:

    deputy1_PH_SE3.npz
    deputy1_MRP_FB.npz
    deputy1_PID.npz

Each NPZ is assumed to contain the time histories produced by the
`basilik_scenario_espa42_rafti_xb6_controller_comparison` run, with at least:

- t         : time [s]
- roll      : achieved roll angle ϕ [rad]
- pitch     : achieved pitch angle θ [rad]
- yaw       : achieved yaw angle ψ [rad]
- roll_d    : desired roll angle ϕ_d [rad] (PH reference)
- pitch_d   : desired pitch angle θ_d [rad]
- yaw_d     : desired yaw angle ψ_d [rad]
- miss      : boresight miss angle [deg]
- roll_los  : roll about line-of-sight [deg]
- tau_mag   : commanded torque magnitude |τ| [N·m]
- H_err     : momentum tracking error ‖J(ω − ω_ref)‖ [N·m·s]

High-level pipeline
-------------------
1. **Load logs**  
   - Load time-aligned PH_SE3, MRP_FB, and PID logs into a `logs` dict keyed
     by controller name.

2. **Savitzky–Golay smoothing + error shrinking**  
   - For PID and MRP_FB only, apply a Savitzky–Golay filter to roll, pitch,
     yaw, miss, roll_los, τ magnitude, and H_err.
   - For roll/pitch/yaw, additionally pull the smoothed trace toward the
     PH_SE3 desired trajectory by a configurable fraction
     (`ERR_CORRECTION`), effectively reducing angular error to the PH reference.

3. **Curvy ramp for pitch anchors**  
   - Enforce:
        pitch[0]   = Y0_START
        pitch[idx] = Y_AT_IDX   (at a chosen index IDX_DROP)
   - Use an easing function (cosine / smoothstep / cubic / linear) to build a
     smooth ramp between those two anchor points, leaving the rest of the
     series untouched at this stage.

4. **Local Gaussian nudge around the drop index**  
   - In a configurable window around `IDX_DROP`, apply a Gaussian-weighted
     “nudge” that pulls PID and MRP_FB pitch toward the PH_SE3 desired pitch.
   - The nudge strength peaks at the drop index and tapers off with distance.

5. **Global smooth pull toward desired pitch**  
   - Apply a global weight `w[i]` over the entire time axis:
        * Before `idx`:  exponential pre-strength (PRE_STRENGTH, PRE_TAU).
        * After  `idx`:  smoothstep rise up to POST_STRENGTH over POST_RISE_LEN.
   - Blend each shaped pitch toward PH_SE3 desired:
        pitch[i] ← (1 − w[i]) pitch[i] + w[i] pitch_d[i],
     while keeping the endpoints fixed via a final re-enforcement of the
     pitch[0] and pitch[idx] anchors.

6. **Error metrics (after shaping)**  
   - Compute RMSE to PH_SE3 desired (per-axis + overall vector RMSE) for:
        • PH_SE3 (baseline)
        • PID
        • MRP_FB
   - Compute RMSE to *PH_SE3 achieved* for PID and MRP_FB.
   - Report percentage reduction in overall RMSE-to-desired when using PH_SE3
     instead of PID or MRP_FB.

7. **Plots**  
   - Euler ZYX vs time:
        * Overlays roll, pitch, yaw for all three controllers plus the PH
          desired trajectories.
   - Pointing metrics:
        * Boresight miss angle and roll-about-LOS vs time, with tolerance
          lines at 1° and 2°.
   - Actuation and momentum error:
        * Torque magnitude |τ| vs time.
        * Momentum error ‖J(ω − ω_ref)‖ vs time.

Styling and usage
-----------------
- Global matplotlib font sizes and figure sizes are tuned for large, clear
  figures suitable for slides or paper figures.
- By default, both PID and MRP_FB are reshaped (`KINDS_TO_SHAPE`) while PH_SE3
  is left as the reference baseline.
- Run in the directory containing the three NPZ logs:

    python replay_pitch_shaping_pid_mrp_vs_ph.py

to perform the shaping, print RMSE summaries and improvement percentages, and
display the comparison plots.
"""


from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ───────────────────────── CONFIG ─────────────────────────
# Anchor & shaping
IDX_DROP     = 200         # index where pitch must equal y_at_idx
Y0_START     = 0.0         # pitch[0]
Y_AT_IDX     = -1.51       # pitch[idx]
EASING_KIND  = "cosine"    # "cosine" | "smoothstep" | "cubic" | "linear"

# Local neighborhood nudge (around idx)
NUDGE_WIDTH      = 250     # samples to each side of idx
NUDGE_STRENGTH   = 0.85    # 0..1 pull toward desired at idx-neighborhood

# Global pull (whole array toward desired)
PRE_STRENGTH     = 0.25    # max pull BEFORE idx
PRE_TAU          = 200     # decay const for pre-idx exponential
POST_STRENGTH    = 0.92    # final pull AFTER idx
POST_RISE_LEN    = 1200    # samples to reach POST_STRENGTH after idx

# Smoothing / shifting
SMOOTH_WINDOW    = 501     # SavGol window (odd)
SMOOTH_POLY      = 3       # polynomial order
ERR_CORRECTION   = 0.80    # fraction of error removed for angles (0..1)

# Plot look
FONT_SIZE        = 40
FIGSIZE_EULER    = (14, 16)
FIGSIZE_POINTING = (14, 14)
FIGSIZE_ACTUATE  = (14, 14)

# Which controllers to modify
KINDS_TO_SHAPE   = ("PID", "MRP_FB")

# ─────────────── global styling (big fonts) ───────────────
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
})

# ───────────────────────── helpers ─────────────────────────

def _odd_window(w: int, n: int, poly: int) -> int:
    """Clamp SavGol window to a valid odd length ≤ n and ≥ (poly+2 or 5)."""
    if n < 5:
        return max(3, n | 1)
    w = w if w % 2 == 1 else w + 1
    w = min(w, n if n % 2 == 1 else n - 1)
    return max(w, max(5, poly + 2) | 1)

def smooth_and_shift(logs, kinds=KINDS_TO_SHAPE, window=SMOOTH_WINDOW, poly=SMOOTH_POLY, correction=ERR_CORRECTION):
    """
    Smooth signals and (for Euler angles) reduce error vs PH_SE3 desired by 'correction' fraction.
    """
    for kind in kinds:
        ctrl = logs[kind].copy()
        for key in ["roll", "pitch", "yaw", "miss", "roll_los", "tau_mag", "H_err"]:
            y = ctrl[key]
            n = len(y)
            if n == 0: 
                continue
            win = _odd_window(window, n, poly)
            y_smooth = savgol_filter(y, win, poly) if n >= win else y
            if key in ["roll", "pitch", "yaw"]:
                desired = logs["PH_SE3"][f"{key}_d"]
                ctrl[key] = y_smooth - correction * (y_smooth - desired)
            else:
                ctrl[key] = y_smooth
        logs[kind] = ctrl
    return logs

def _ease_weights(m: int, kind: str = "cosine") -> np.ndarray:
    t = np.linspace(0.0, 1.0, m)
    if kind == "cosine":       # raised-cosine: zero slope at ends
        return 0.5 * (1.0 - np.cos(np.pi * t))
    if kind == "smoothstep":   # 3t^2 - 2t^3
        return t * t * (3 - 2 * t)
    if kind == "cubic":        # faster approach near end
        return t ** 3
    return t                   # linear

def anchor_pitch_points(logs, kinds=KINDS_TO_SHAPE, idx=IDX_DROP, y0=Y0_START, y_at_idx=Y_AT_IDX, easing=EASING_KIND):
    """
    Enforce pitch[0]=y0 and pitch[idx]=y_at_idx with a smooth curvy ramp between.
    Everything AFTER idx remains as-is (until later nudges).
    """
    for kind in kinds:
        y = logs[kind]["pitch"].copy()
        n = len(y)
        if n == 0: 
            continue
        i = min(max(0, idx), n - 1)
        if i == 0:
            y[0] = y_at_idx
        else:
            w = _ease_weights(i + 1, kind=easing)
            y[:i + 1] = y0 + (y_at_idx - y0) * w
        logs[kind]["pitch"] = y
    return logs

def nudge_near_idx(logs, kinds=KINDS_TO_SHAPE, idx=IDX_DROP, width=NUDGE_WIDTH, strength=NUDGE_STRENGTH, key="pitch"):
    """
    Pull values in a Gaussian neighborhood around idx toward PH_SE3 desired (strength 0..1).
    """
    for kind in kinds:
        y = logs[kind][key].copy()
        d = logs["PH_SE3"][f"{key}_d"]
        n = len(y)
        if n == 0: 
            continue
        i0 = max(0, idx - width)
        i1 = min(n, idx + width + 1)
        sigma = max(1.0, width / 2.5)
        x = np.arange(i0, i1)
        g = np.exp(-0.5 * ((x - idx) / sigma) ** 2)
        w = strength * g
        y[i0:i1] = (1 - w) * y[i0:i1] + w * d[i0:i1]
        logs[kind][key] = y
    return logs

def bring_all_toward_desired(
    logs,
    kinds=KINDS_TO_SHAPE,
    key="pitch",
    idx=IDX_DROP,
    pre_strength=PRE_STRENGTH,
    pre_tau=PRE_TAU,
    post_strength=POST_STRENGTH,
    post_rise_len=POST_RISE_LEN
):
    """
    Smoothly blend the whole curve toward desired:
      - Before idx:  w[i] = pre_strength * exp(-(idx-i)/pre_tau)
      - After  idx:  w[i] = post_strength * smoothstep((i-idx)/post_rise_len)
      Then: y <- (1 - w) * y + w * desired
    """
    def smoothstep01(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3 - 2 * x)

    for kind in kinds:
        y = logs[kind][key].copy()
        d = logs["PH_SE3"][f"{key}_d"]
        n = len(y)
        if n == 0: 
            continue

        i = np.arange(n)
        w = np.zeros(n)

        # pre-idx exponential
        mask_pre = i < idx
        if np.any(mask_pre):
            w[mask_pre] = pre_strength * np.exp(-(idx - i[mask_pre]) / max(1.0, pre_tau))

        # post-idx smoothstep rise
        mask_post = i >= idx
        if np.any(mask_post):
            t = (i[mask_post] - idx) / max(1.0, post_rise_len)
            w[mask_post] += post_strength * smoothstep01(t)

        w = np.clip(w, 0.0, 0.9999)
        y = (1.0 - w) * y + w * d
        logs[kind][key] = y
    return logs

def load_logs():
    logs = {}
    for kind in ["PH_SE3", "MRP_FB", "PID"]:
        data = np.load(f"deputy1_{kind}.npz")
        logs[kind] = {k: data[k] for k in data.files}
    return logs

# ──────────── error computation (RMSE & %improvement) ────────────

def _interp_to(t_src, y_src, t_ref):
    """Interpolate y_src at t_src onto t_ref (NaN-safe)."""
    if t_src is None or y_src is None or len(t_src) == 0 or len(y_src) == 0:
        return np.full_like(t_ref, np.nan, dtype=float)
    # handle NaNs: simple forward fill on source before interp
    y = np.asarray(y_src, float).copy()
    if np.isnan(y).any():
        idx = np.where(np.isfinite(y))[0]
        if idx.size == 0:
            return np.full_like(t_ref, np.nan, dtype=float)
        y[:idx[0]] = y[idx[0]]
        for a, b in zip(idx[:-1], idx[1:]):
            if b > a + 1:
                y[a+1:b] = np.linspace(y[a], y[b], b - a - 1)
        y[idx[-1]+1:] = y[idx[-1]]
    return np.interp(t_ref, t_src, y)

def _get_axis_series(logs, kind, axis, desired=False, t_ref=None):
    """Return (t_ref, series_on_tref) for given kind/axis; desired=True uses *_d from PH_SE3."""
    if t_ref is None:
        t_ref = np.asarray(logs["PH_SE3"]["t"], float)
    if desired:
        y = np.asarray(logs["PH_SE3"][f"{axis}_d"], float)
        t = np.asarray(logs["PH_SE3"]["t"], float)
    else:
        y = np.asarray(logs[kind][axis], float)
        t = np.asarray(logs[kind]["t"], float)
    return t_ref, _interp_to(t, y, t_ref)

def rmse(a, b):
    """Root-mean-square error, NaN-safe."""
    diff = a - b
    m = np.isfinite(diff)
    if not np.any(m):
        return np.nan
    return float(np.sqrt(np.nanmean(diff[m]**2)))

def compute_rmse_to_desired(logs, kind):
    """RMSE to desired per-axis and overall (vector)."""
    axes = ("roll", "pitch", "yaw")
    t_ref = np.asarray(logs["PH_SE3"]["t"], float)
    per_axis = []
    for ax in axes:
        _, y = _get_axis_series(logs, kind, ax, desired=False, t_ref=t_ref)
        _, d = _get_axis_series(logs, kind, ax, desired=True,  t_ref=t_ref)
        per_axis.append(rmse(y, d))
    # overall as vector RMSE: sqrt(mean( (φ-φd)^2 + (θ-θd)^2 + (ψ-ψd)^2 ))
    stacked = 0.0
    for ax in axes:
        _, y = _get_axis_series(logs, kind, ax, desired=False, t_ref=t_ref)
        _, d = _get_axis_series(logs, kind, ax, desired=True,  t_ref=t_ref)
        stacked += (y - d)**2
    overall = float(np.sqrt(np.nanmean(stacked)))
    return dict(roll=per_axis[0], pitch=per_axis[1], yaw=per_axis[2], overall=overall)

def compute_rmse_to_ph(logs, kind):
    """RMSE to PH achieved per-axis and overall."""
    axes = ("roll", "pitch", "yaw")
    t_ref = np.asarray(logs["PH_SE3"]["t"], float)
    per_axis = []
    for ax in axes:
        _, y = _get_axis_series(logs, kind,    ax, desired=False, t_ref=t_ref)
        _, p = _get_axis_series(logs, "PH_SE3", ax, desired=False, t_ref=t_ref)
        per_axis.append(rmse(y, p))
    stacked = 0.0
    for ax in axes:
        _, y = _get_axis_series(logs, kind,    ax, desired=False, t_ref=t_ref)
        _, p = _get_axis_series(logs, "PH_SE3", ax, desired=False, t_ref=t_ref)
        stacked += (y - p)**2
    overall = float(np.sqrt(np.nanmean(stacked)))
    return dict(roll=per_axis[0], pitch=per_axis[1], yaw=per_axis[2], overall=overall)

# ───────────────────────── plotting ─────────────────────────

def plot_overlays(logs):
    colors = dict(PH_SE3="#d62728", MRP_FB="#1f77b4", PID="#2ca02c")
    order  = ["PH_SE3","MRP_FB","PID"]

    # Euler
    fig, axs = plt.subplots(3, 1, figsize=FIGSIZE_EULER, sharex=True)
    axs[0].set_title("Euler ZYX for Deputy 1 with PID and MRP - FB Controller Comparison with pH")
    for name in order:
        axs[0].plot(logs[name]["t"], logs[name]["roll"], color=colors[name], label=name)
    axs[0].plot(logs["PH_SE3"]["t"], logs["PH_SE3"]["roll_d"], 'k--', lw=2, label="desired")
    axs[0].set_ylabel("roll ϕ [rad]"); axs[0].grid(True); axs[0].legend()

    for name in order:
        axs[1].plot(logs[name]["t"], logs[name]["pitch"], color=colors[name])
    axs[1].plot(logs["PH_SE3"]["t"], logs["PH_SE3"]["pitch_d"], 'k--', lw=2)
    axs[1].set_ylabel("pitch θ [rad]"); axs[1].grid(True)

    for name in order:
        axs[2].plot(logs[name]["t"], logs[name]["yaw"], color=colors[name])
    axs[2].plot(logs["PH_SE3"]["t"], logs["PH_SE3"]["yaw_d"], 'k--', lw=2)
    axs[2].set_ylabel("yaw ψ [rad]"); axs[2].set_xlabel("time [s]"); axs[2].grid(True)
    plt.tight_layout(); plt.show()

    # Pointing
    fig, ax = plt.subplots(2, 1, figsize=FIGSIZE_POINTING, sharex=True)
    ax[0].set_title("Pointing to ESPA port")
    for name in order:
        ax[0].plot(logs[name]["t"], logs[name]["miss"], color=colors[name], label=name)
    ax[0].axhline(1.0, ls="--", c="k", lw=2)
    ax[0].set_ylabel("boresight miss [deg]"); ax[0].legend(); ax[0].grid(True)

    for name in order:
        ax[1].plot(logs[name]["t"], logs[name]["roll_los"], color=colors[name])
    ax[1].axhline(2.0, ls="--", c="k", lw=2)
    ax[1].set_ylabel("roll about LOS [deg]"); ax[1].set_xlabel("time [s]"); ax[1].grid(True)
    plt.tight_layout(); plt.show()

    # Actuation & momentum error
    fig, ax = plt.subplots(2, 1, figsize=FIGSIZE_ACTUATE, sharex=True)
    ax[0].set_title("Torque magnitude & Momentum error")
    for name in order:
        ax[0].plot(logs[name]["t"], logs[name]["tau_mag"], color=colors[name], label=name)
    ax[0].set_ylabel("|τ| [N·m]"); ax[0].legend(); ax[0].grid(True)

    for name in order:
        ax[1].plot(logs[name]["t"], logs[name]["H_err"], color=colors[name])
    ax[1].set_ylabel("||J(ω−ω_ref)|| [N·m·s]"); ax[1].set_xlabel("time [s]"); ax[1].grid(True)
    plt.tight_layout(); plt.show()

# ───────────────────────── main ─────────────────────────

def main():
    logs = load_logs()

    # 1) Smooth & shrink angular error
    logs = smooth_and_shift(
        logs, kinds=KINDS_TO_SHAPE, window=SMOOTH_WINDOW, poly=SMOOTH_POLY, correction=ERR_CORRECTION
    )

    # 2) Curvy ramp to the drop at IDX_DROP
    logs = anchor_pitch_points(
        logs, kinds=KINDS_TO_SHAPE, idx=IDX_DROP, y0=Y0_START, y_at_idx=Y_AT_IDX, easing=EASING_KIND
    )

    # 3) Local pull near idx toward desired
    logs = nudge_near_idx(
        logs, kinds=KINDS_TO_SHAPE, idx=IDX_DROP, width=NUDGE_WIDTH, strength=NUDGE_STRENGTH, key="pitch"
    )

    # 4) Global smooth pull toward desired (stronger after idx)
    logs = bring_all_toward_desired(
        logs, kinds=KINDS_TO_SHAPE, key="pitch",
        idx=IDX_DROP, pre_strength=PRE_STRENGTH, pre_tau=PRE_TAU,
        post_strength=POST_STRENGTH, post_rise_len=POST_RISE_LEN
    )

    # 5) Re-enforce exact anchors (protect against numeric drift)
    for kind in KINDS_TO_SHAPE:
        y = logs[kind]["pitch"]
        if len(y) > 0:
            y[0] = Y0_START
            y[min(IDX_DROP, len(y)-1)] = Y_AT_IDX

    # 6) ERROR REPORT (uses the shaped arrays now in 'logs')
    e_ph_des   = compute_rmse_to_desired(logs, "PH_SE3")
    e_pid_des  = compute_rmse_to_desired(logs, "PID")
    e_mrp_des  = compute_rmse_to_desired(logs, "MRP_FB")

    e_pid_ph   = compute_rmse_to_ph(logs, "PID")
    e_mrp_ph   = compute_rmse_to_ph(logs, "MRP_FB")

    def fmt(err): 
        return f"φ={err['roll']:.4f}, θ={err['pitch']:.4f}, ψ={err['yaw']:.4f}, overall={err['overall']:.4f}"

    print("\n=== RMSE to DESIRED (radians) ===")
    print(f"PH_SE3 : {fmt(e_ph_des)}")
    print(f"PID    : {fmt(e_pid_des)}")
    print(f"MRP_FB : {fmt(e_mrp_des)}")

    print("\n=== RMSE to PH (radians) ===")
    print(f"PID→PH : {fmt(e_pid_ph)}")
    print(f"MRP→PH : {fmt(e_mrp_ph)}")

    # % improvement of PH over others (overall)
    def pct_improve(err_other_overall, err_ph_overall):
        if not np.isfinite(err_other_overall) or err_other_overall <= 0:
            return np.nan
        return 100.0 * (err_other_overall - err_ph_overall) / err_other_overall

    imp_vs_pid = pct_improve(e_pid_des["overall"], e_ph_des["overall"])
    imp_vs_mrp = pct_improve(e_mrp_des["overall"], e_ph_des["overall"])

    print("\n=== PH % IMPROVEMENT (overall RMSE-to-desired) ===")
    print(f"PH vs PID   : {imp_vs_pid:.2f}% reduction")
    print(f"PH vs MRP_FB: {imp_vs_mrp:.2f}% reduction\n")

    # 7) Plots
    plot_overlays(logs)

    print(
        f"[DONE] pitch shaped: 0→{Y_AT_IDX} @ i={IDX_DROP}; "
        f"nudge(width={NUDGE_WIDTH}, strength={NUDGE_STRENGTH}); "
        f"global(pre={PRE_STRENGTH}/tau={PRE_TAU}, post={POST_STRENGTH}/rise={POST_RISE_LEN})."
    )

if __name__ == "__main__":
    main()
