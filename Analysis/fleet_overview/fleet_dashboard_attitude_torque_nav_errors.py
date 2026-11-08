#!/usr/bin/env python3
"""
fleet_dashboard_attitude_torque_nav_errors.py

Purpose
-------
Generate two multi-panel “dashboard” figures that summarize early-time attitude
tracking, body-rate behavior, commanded torques, and navigation errors for a
fleet of up to eight deputies using their saved NPZ log files.

Inputs
------
- LOG_DIR / "dep??_logs.npz":
  Each NPZ is assumed to contain (at least) the following arrays:
    * t_err      : time stamps [s] for attitude/control error signals.
    * sigma_BR   : 3×N array of body-frame attitude error MRPs (body vs reference).
    * omega_BR   : 3×N array of body-rate errors [rad/s].
    * t_torque   : time stamps [s] for commanded torques.
    * torque_B   : 3×N array of commanded body-frame torques [N·m].
    * t_navT     : time stamps [s] for navigation truth/estimate comparisons.
    * r_nav      : 3×N array of estimated position (navigation) [m].
    * r_true     : 3×N array of true position [m].
    * v_nav      : 3×N array of estimated velocity [m/s].
    * v_true     : 3×N array of true velocity [m/s].

High-level behavior
-------------------
1. Discover up to eight deputy NPZ logs in `LOG_DIR` matching `dep??_logs.npz`.
2. For each deputy, crop all time histories to a common window `[0, T_CUTOFF_S]`.
3. Figure 1 — “Attitude-Error, Body-Rate Error & Commanded Torques”:
   - Row 0: σ-error components (σ₁, σ₂, σ₃) vs time.
   - Row 1: body-rate error components (ω₁, ω₂, ω₃) vs time.
   - Row 2: commanded body torques (Lx, Ly, Lz) vs time.
   Each column corresponds to one deputy; only the first column shows y-tick
   labels and legends, and all share a common 0–450 s x-axis.
4. Figure 2 — “Attitude-Error Norm, Position-Error Norm & Velocity-Error Norm”:
   - Row 0: attitude-error norm ‖σ‖ vs time.
   - Row 1: position-error norm ‖Δr‖ = ‖r_nav − r_true‖ vs time.
   - Row 2: velocity-error norm ‖Δv‖ = ‖v_nav − v_true‖ vs time.
   Y-ticks for each row are chosen based on the global max over all deputies to
   give a clean, comparable scale.

Layout and styling
------------------
- Three rows × N_DEPS columns in each figure, with shared x-axis, grid on every
  subplot, and only the first column showing y-ticks and labels.
- Global font sizes and layout (margins, paddings) tuned for publication-style
  export; the full figure is nudged slightly left using `LAYOUT_RECT`.
- If `SAVE_PNGS` is True, both figures are written out as PNG files with the
  configured names; otherwise they are just displayed interactively.

Usage
-----
Set `LOG_DIR`, `T_CUTOFF_S`, and `SAVE_PNGS` at the top of the script, then run:

    python fleet_dashboard_attitude_torque_nav_errors.py

to generate fleet-level dashboards for early post-deployment behavior.
"""


from pathlib import Path
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# ╔══════════════  USER CONTROLS  ═══════════════╗
LOG_DIR      = Path("fleet_logs")        # folder with dep??_logs.npz
T_CUTOFF_S   = 450                       # x-axis limit [s]
SAVE_PNGS    = False                     # True → save PNG, False → show
PNG_NAME     = "attitude_cmd_bodyrate_dashboard.png"

BASE_FONT    = 30
TICK_FONT    = 25
LAYOUT_RECT  = (0.01, 0.04, 0.99, 0.96)  # l, b, r, t  (shifted left)
W_PAD, H_PAD = 0.15, 0.28                # spacing between sub-plots
TITLE_PAD    = 1.02
FIG_TITLE_Y  = 0.89
# ╚══════════════════════════════════════════════╝

plt.rcParams.update({
    "font.size": BASE_FONT,
    "axes.titlesize": BASE_FONT,
    "axes.labelsize": BASE_FONT,
    "xtick.labelsize": TICK_FONT,
    "ytick.labelsize": TICK_FONT,
    "legend.fontsize": BASE_FONT - 4,
})

# ── load deputy logs ───────────────────────────────────────────────────────
paths = sorted(LOG_DIR.glob("dep??_logs.npz"))[:8]
if not paths:
    raise FileNotFoundError(f"No dep??_logs.npz found in {LOG_DIR}")
bundles = [np.load(p) for p in paths]
N_DEPS  = len(bundles)

# ── reference y-tick sets (first column only) ──────────────────────────────
sigma_ticks = np.arange(0.6, -0.401, -0.2)          # 0.6 … −0.4
omega_ticks = np.array([0.04, 0.02, 0.00, -0.02,
                        -0.04])                    # adjust to data range
tau_ticks   = np.array([0.03, 0.02, 0.01, 0.00,
                        -0.01, -0.02, -0.03, -0.04])

def dep_name(p: Path) -> str:
    return f"Dep {int(p.stem[3:5]) + 1}"

def crop(t, *arrs):
    keep = t <= T_CUTOFF_S
    return (t[keep],) + tuple(a[keep] for a in arrs)

# ── figure & axes grid ─────────────────────────────────────────────────────
fig, axes = plt.subplots(
    nrows=3, ncols=N_DEPS,
    figsize=(N_DEPS * 5, 3 * 4),          # extra height for 3rd row
    sharex=True
)
fig.suptitle("Attitude-Error, Body-Rate Error & Commanded Torques\n"
             "(First 450 s after deployment)",
             y=FIG_TITLE_Y)

# ── populate subplots ──────────────────────────────────────────────────────
for k, (p, d) in enumerate(zip(paths, bundles)):
    # Row 0 – σ components --------------------------------------------------
    ax_sigma = axes[0, k]
    t_err, sigma = crop(d["t_err"], d["sigma_BR"])
    ax_sigma.plot(t_err, sigma, lw=1.2)
    ax_sigma.set_xlim(0, T_CUTOFF_S)
    ax_sigma.grid(True)
    ax_sigma.set_yticks(sigma_ticks)
    ax_sigma.set_ylim(sigma_ticks[0], sigma_ticks[-1])
    ax_sigma.set_title(dep_name(p), pad=TITLE_PAD)

    if k == 0:
        ax_sigma.set_ylabel("σ components")
        ax_sigma.legend(["σ₁", "σ₂", "σ₃"], loc="upper right")
    else:
        ax_sigma.set_yticklabels([])

    # Row 1 – ω components --------------------------------------------------
    ax_omega = axes[1, k]
    t_ω, omega = crop(d["t_err"], d["omega_BR"])
    ax_omega.plot(t_ω, omega, lw=1.2)
    ax_omega.set_xlim(0, T_CUTOFF_S)
    ax_omega.grid(True)
    ax_omega.set_yticks(omega_ticks)
    ax_omega.set_ylim(omega_ticks[0], omega_ticks[-1])

    if k == 0:
        ax_omega.set_ylabel("ω [rad s⁻¹]")
        ax_omega.legend(["ω₁", "ω₂", "ω₃"], loc="upper right")
    else:
        ax_omega.set_yticklabels([])

    # Row 2 – commanded torques --------------------------------------------
    ax_tau = axes[2, k]
    t_tau, tau = crop(d["t_torque"], d["torque_B"])
    ax_tau.plot(t_tau, tau, lw=1.2)
    ax_tau.set_xlim(0, T_CUTOFF_S)
    ax_tau.grid(True)
    ax_tau.set_yticks(tau_ticks)
    ax_tau.set_ylim(tau_ticks[0] + 0.002, tau_ticks[-1] - 0.002)

    if k == 0:
        ax_tau.set_ylabel("Torque [N·m]")
        ax_tau.legend(["Lx", "Ly", "Lz"], loc="upper right")
    else:
        ax_tau.set_yticklabels([])

# ── single shared x-label ─────────────────────────────────────────────────
fig.supxlabel("Time [s]", fontsize=BASE_FONT, y=0.11)

fig.tight_layout(rect=LAYOUT_RECT, w_pad=W_PAD, h_pad=H_PAD)

if SAVE_PNGS:
    fig.savefig(PNG_NAME, dpi=150)
    print(f"[INFO] Saved {PNG_NAME}")
else:
    plt.show()

    # ── FIGURE 2: Norms dashboard (‖σ‖, ‖Δr‖, ‖Δv‖) ─────────────────────────
def _nice_pos_ticks(vmax, n=5):
    """0 → round-up-to {1,2,5,10}×10^k, then linspace ticks."""
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    exp  = np.floor(np.log10(vmax))
    frac = vmax / 10**exp
    for m in (1, 2, 5, 10):
        if frac <= m:
            top = m * 10**exp
            break
    return np.linspace(0.0, top, n)

# Pre-compute global maxima (first 450 s) to choose sensible ticks
sig_max = r_max = v_max = 0.0
for d in bundles:
    t, sigma = crop(d["t_err"], d["sigma_BR"])
    sig_max = max(sig_max, la.norm(sigma, axis=1).max(initial=0))
    t, r_nav, r_true = crop(d["t_navT"], d["r_nav"], d["r_true"])
    r_max = max(r_max, la.norm(r_nav - r_true, axis=1).max(initial=0))
    t, v_nav, v_true = crop(d["t_navT"], d["v_nav"], d["v_true"])
    v_max = max(v_max, la.norm(v_nav - v_true, axis=1).max(initial=0))

sig_ticks = _nice_pos_ticks(sig_max)
r_ticks   = _nice_pos_ticks(r_max)
v_ticks   = _nice_pos_ticks(v_max)

fig2, axes2 = plt.subplots(
    nrows=3, ncols=N_DEPS,
    figsize=(N_DEPS * 5, 3 * 4),
    sharex=True
)
fig2.suptitle("Attitude-Error Norm, Position-Error Norm & Velocity-Error Norm\n"
              "(First 450 s after deployment)", y=FIG_TITLE_Y)

for k, (p, d) in enumerate(zip(paths, bundles)):
    # Row 0 — ‖σ‖ -----------------------------------------------------------
    ax = axes2[0, k]
    t, sigma = crop(d["t_err"], d["sigma_BR"])
    ax.plot(t, la.norm(sigma, axis=1), lw=1.6)
    ax.set_xlim(0, T_CUTOFF_S); ax.grid(True)
    ax.set_title(dep_name(p), pad=TITLE_PAD)
    if k == 0:
        ax.set_ylabel("‖σ‖")
        ax.set_yticks(sig_ticks)
    else:
        ax.set_yticklabels([])

    # Row 1 — ‖Δr‖ ----------------------------------------------------------
    ax = axes2[1, k]
    t, r_nav, r_true = crop(d["t_navT"], d["r_nav"], d["r_true"])
    ax.plot(t, la.norm(r_nav - r_true, axis=1), lw=1.6)
    ax.set_xlim(0, T_CUTOFF_S); ax.grid(True)
    if k == 0:
        ax.set_ylabel("‖Δr‖ [m]")
        ax.set_yticks(r_ticks)
    else:
        ax.set_yticklabels([])

    # Row 2 — ‖Δv‖ ----------------------------------------------------------
    ax = axes2[2, k]
    t, v_nav, v_true = crop(d["t_navT"], d["v_nav"], d["v_true"])
    ax.plot(t, la.norm(v_nav - v_true, axis=1), lw=1.6)
    ax.set_xlim(0, T_CUTOFF_S); ax.grid(True)
    if k == 0:
        ax.set_ylabel("‖Δv‖ [m s⁻¹]")
        ax.set_yticks(v_ticks)
    else:
        ax.set_yticklabels([])

fig2.supxlabel("Time [s]", fontsize=BASE_FONT, y=0.11)
fig2.tight_layout(rect=LAYOUT_RECT, w_pad=W_PAD, h_pad=H_PAD)

PNG_NAME_2 = "norms_dashboard.png"
if SAVE_PNGS:
    fig2.savefig(PNG_NAME_2, dpi=150)
    print(f"[INFO] Saved {PNG_NAME_2}")
else:
    plt.show()

