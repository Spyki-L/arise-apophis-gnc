#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_deploy_truth_fdi_anim_black_ops_grid_labels.py

Black-background, mission-ops style animation (manual layout).

Layout:
  - Left: 3D Hill + R–T + R–N
  - Right: 6 plots (3 rows × 2 columns)

Update requested:
  - Show deputy labels D1..D8 on:
      (1) 3D plot
      (2) R–T projection
      (3) R–N projection
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import textwrap


# -----------------------------
# Deploy UTC schedule (FROM YOUR PLOT)
# indices 0..7 correspond to D1..D8 labels
# -----------------------------
DEPLOY_UTC_FROM_PLOT = {
    0: "2029-04-13 18:05:45Z",  # D1
    1: "2029-04-13 14:54:45Z",  # D2
    2: "2029-04-13 18:34:45Z",  # D3
    3: "2029-04-13 18:25:15Z",  # D4
    4: "2029-04-13 14:13:15Z",  # D5
    5: "2029-04-13 17:04:45Z",  # D6
    6: "2029-04-13 08:31:45Z",  # D7 (earliest)
    7: "2029-04-13 15:19:15Z",  # D8
}
N_DEP = 8


# -----------------------------
# Manual panel placement (NORMALIZED FIGURE COORDS)
# Left: 3D + 2 projections
# Right: 6 plots (3 rows x 2 columns)
# -----------------------------
LEFT_X0 = 0.03
LEFT_X1 = 0.59
LEFT_W = LEFT_X1 - LEFT_X0

RIGHT_X0 = 0.60
RIGHT_X1 = 0.98
RIGHT_W = RIGHT_X1 - RIGHT_X0

COL_GAP = 0.02
COL_W = (RIGHT_W - COL_GAP) / 2.0
RX_L = RIGHT_X0 - 0.05
RX_R = RIGHT_X0 + COL_W + COL_GAP

ROW_Y0 = 0.05
ROW_H = 0.22
ROW_G = 0.02
RY_B = ROW_Y0
RY_M = ROW_Y0 + (ROW_H + ROW_G) +0.075
RY_T = ROW_Y0 + 2.0 * (ROW_H + ROW_G) + 0.15

AX_POS = {
    "ax3":   [LEFT_X0, 0.29, LEFT_W, 0.63],
    "ax_rt": [LEFT_X0, 0.05, (LEFT_W - 0.02) / 2.0, 0.21],
    "ax_rn": [LEFT_X0 + (LEFT_W - 0.02) / 2.0 + 0.02, 0.05, (LEFT_W - 0.02) / 2.0, 0.21],

    "ax_health": [RX_L, RY_T, COL_W, ROW_H],
    "ax_res":    [RX_R, RY_T, COL_W, ROW_H],
    "ax_geom":   [RX_L, RY_M, COL_W, ROW_H],
    "ax_nav":    [RX_R, RY_M, COL_W, ROW_H],
    "ax_comp":   [RX_L, RY_B, COL_W, ROW_H],
    "ax_evt":    [RX_R, RY_B, COL_W, ROW_H],
}


# -----------------------------
# Import dcm_lvlh from your codebase
# -----------------------------
def import_hill_utils():
    import sys
    script_dir = Path(__file__).resolve().parent
    root = None
    for parent in script_dir.parents:
        if (parent / "GNC").is_dir():
            root = parent
            break
    if root is None:
        root = script_dir.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from GNC.Navigation.hill_ekf import dcm_lvlh  # type: ignore[import]
    return dcm_lvlh


DCM_LVLH = import_hill_utils()


# -----------------------------
# Visual palette (black background)
# -----------------------------
STATE_COLORS = {
    "HEALTHY":  "#00ff6a",
    "SUSPECT":  "#ffcc00",
    "FAULTED":  "#ff4444",
    "DEAD":     "#888888",
    "NO_FDI":   "#33bbff",
    "DEPLOYED": "#33bbff",
    "STOWED":   "#aaaaaa",
}
GRID_RGBA = (1.0, 1.0, 1.0, 0.1)
AXIS_RGBA = (1.0, 1.0, 1.0, 0.8)
TEXT_COLOR = "white"

RTN_COLORS = {"R": "#66ccff", "T": "#ffcc66", "N": "#cc66ff"}

NU_NAV = 5.0
NU_ATT = 2.5
NU_FORM = 2.5
NU_PEER = 3.0

MEAS_ARROW_MODE = "correction"  # "correction" or "whitened" or "off"
MEAS_GATE_MODE  = "nu_nav"      # "nu_nav" or "rnav1" or "innov3" or "off"
INNOV_SIGMA_BANDS = 3.0
MEAS_ARROW_MAXLEN_FRAC = 0.10

SHOW_RANGE_SHELLS = True
SHELL_FRACS = (0.25, 0.50, 0.75)
KEEP_OUT_R_FRAC_OF_RING = 0.35

CAM_ELEV = 25.0
CAM_AZ0 = -55.0
EVENT_FREEZE_S = 120.0
FOCUS_FREEZE_S = 60.0

MAX_NAV_TICKS_IN_WINDOW = 160
NIS_99_DF3 = 11.345

# Deputy label styling (3D + 2D)
LABEL_FS_3D = 9
LABEL_FS_2D = 9
LABEL_ALPHA = 0.82
LABEL_OFF_3D_FRAC = 0.030   # of lim
LABEL_OFF_2D_FRAC = 0.035   # of proj_lim
LABEL_Z_OFF_3D_FRAC = 0.010 # of lim


# -----------------------------
# Helpers
# -----------------------------
def parse_utc(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    if "T" not in s:
        s = s.replace(" ", "T")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def to_hr(tau_s: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(tau_s, float) / 3600.0


def state_to_color(state_name: str, stowed: bool) -> str:
    if stowed:
        return STATE_COLORS["STOWED"]
    return STATE_COLORS.get(state_name, STATE_COLORS["NO_FDI"])


def brighten_rgba(rgba, factor: float = 0.35):
    r, g, b, a = rgba
    r = r + (1.0 - r) * factor
    g = g + (1.0 - g) * factor
    b = b + (1.0 - b) * factor
    return (r, g, b, a)


def interp_vec_series(t_grid: np.ndarray, data: np.ndarray, t_query: float) -> np.ndarray:
    t = np.asarray(t_grid, float).reshape(-1)
    arr = np.asarray(data, float).reshape(-1, data.shape[-1])

    if t_query <= t[0]:
        return arr[0]
    if t_query >= t[-1]:
        return arr[-1]

    j = int(np.searchsorted(t, t_query) - 1)
    dt = float(t[j + 1] - t[j])
    if dt <= 0.0:
        return arr[j]
    f = (t_query - t[j]) / dt
    return (1.0 - f) * arr[j] + f * arr[j + 1]


def interp_scalar_series(t_grid: np.ndarray, data: np.ndarray, t_query: float) -> float:
    t = np.asarray(t_grid, float).reshape(-1)
    y = np.asarray(data, float).reshape(-1)
    if t_query <= t[0]:
        return float(y[0])
    if t_query >= t[-1]:
        return float(y[-1])
    return float(np.interp(t_query, t, y))


def eci_to_hill_pos(rD_N: np.ndarray, vD_N: np.ndarray, rC_N: np.ndarray, vC_N: np.ndarray) -> np.ndarray:
    rD_N = np.asarray(rD_N, float)
    vD_N = np.asarray(vD_N, float)
    rC_N = np.asarray(rC_N, float)
    vC_N = np.asarray(vC_N, float)

    n = rD_N.shape[0]
    rho = np.zeros((n, 3), dtype=float)
    for k in range(n):
        C_HN = DCM_LVLH(rC_N[k], vC_N[k])
        rho[k] = C_HN @ (rD_N[k] - rC_N[k])
    return rho


def fit_ring_plane(pts: np.ndarray):
    P = np.asarray(pts, float).reshape(-1, 3)
    C = P.T @ P
    evals, evecs = la.eigh(C)
    idx = np.argsort(evals)
    n = evecs[:, idx[0]]
    u = evecs[:, idx[2]]
    u = u / max(la.norm(u), 1e-12)
    n = n / max(la.norm(n), 1e-12)
    v = np.cross(n, u)
    v = v / max(la.norm(v), 1e-12)

    proj = P - (P @ n).reshape(-1, 1) * n.reshape(1, 3)
    r = float(np.median(la.norm(proj, axis=1)))
    return u, v, n, float(max(r, 1.0))


def _dummy_seg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float).reshape(3)
    return np.array([[p, p]], dtype=float)


def style_2d_black(ax, title_fs=15, label_fs=11, tick_fs=10):
    ax.set_facecolor("black")
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=tick_fs)
    for sp in ax.spines.values():
        sp.set_color(AXIS_RGBA)
    ax.grid(True, color=GRID_RGBA, alpha=1.0)
    ax.title.set_fontsize(title_fs)
    ax.xaxis.label.set_fontsize(label_fs)
    ax.yaxis.label.set_fontsize(label_fs)


def style_3d_black(ax, tick_fs=12, label_fs=11):
    ax.set_facecolor("black")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.zaxis.label.set_color(TEXT_COLOR)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_tick_params(labelsize=tick_fs, colors=TEXT_COLOR)
        except Exception:
            pass

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        except Exception:
            pass
        try:
            axis._axinfo["grid"]["color"] = GRID_RGBA
            axis._axinfo["axisline"]["color"] = AXIS_RGBA
            axis._axinfo["tick"]["color"] = (1.0, 1.0, 1.0, 0.8)
        except Exception:
            pass
    try:
        ax.grid(True)
    except Exception:
        pass


def clip_vec(v: np.ndarray, max_norm: float) -> np.ndarray:
    v = np.asarray(v, float).reshape(3)
    n = float(la.norm(v))
    if n <= 1e-12:
        return np.zeros(3)
    if n <= max_norm:
        return v
    return v * (max_norm / n)


def approx_correction_drho(P6: np.ndarray, y_pre: np.ndarray, sigma_meas: np.ndarray) -> np.ndarray:
    P6 = np.asarray(P6, float).reshape(6, 6)
    y_pre = np.asarray(y_pre, float).reshape(3)
    sigma = np.maximum(np.asarray(sigma_meas, float).reshape(3), 1e-6)
    R = np.diag(sigma**2)

    P_rr = P6[0:3, 0:3]
    S = P_rr + R
    try:
        K_rr = P_rr @ la.inv(S)
    except la.LinAlgError:
        return np.zeros(3)
    return K_rr @ y_pre


def compute_nav_metrics(y_pre: np.ndarray, sigma_meas: np.ndarray, P_nav: np.ndarray | None):
    y_pre = np.asarray(y_pre, float).reshape(-1, 3)
    sigma_meas = np.asarray(sigma_meas, float).reshape(-1, 3)
    sigma_safe = np.maximum(sigma_meas, 1e-6)

    w = y_pre / sigma_safe
    innov_norm = la.norm(w, axis=1)

    n = y_pre.shape[0]
    nis = np.zeros(n, dtype=float)
    drho = np.zeros((n, 3), dtype=float)

    if P_nav is None:
        nis = np.sum(w**2, axis=1)
        drho_norm = la.norm(drho, axis=1)
        return w, innov_norm, nis, drho, drho_norm

    P_nav = np.asarray(P_nav, float).reshape(-1, 6, 6)
    if P_nav.shape[0] != n:
        nis = np.sum(w**2, axis=1)
        drho_norm = la.norm(drho, axis=1)
        return w, innov_norm, nis, drho, drho_norm

    for k in range(n):
        y = y_pre[k]
        sig = sigma_safe[k]
        R = np.diag(sig**2)
        P6 = P_nav[k]
        P_rr = P6[0:3, 0:3]
        S = P_rr + R
        try:
            nis[k] = float(y.T @ la.inv(S) @ y)
        except la.LinAlgError:
            nis[k] = float(np.sum((y / sig) ** 2))
        drho[k] = approx_correction_drho(P6, y, sig)

    drho_norm = la.norm(drho, axis=1)
    return w, innov_norm, nis, drho, drho_norm


def circle_3d(radius: float, plane: str = "xy", n: int = 200) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * np.pi, n)
    c = np.cos(th)
    s = np.sin(th)
    if plane == "xy":
        return np.c_[radius * c, radius * s, np.zeros_like(c)]
    if plane == "xz":
        return np.c_[radius * c, np.zeros_like(c), radius * s]
    if plane == "yz":
        return np.c_[np.zeros_like(c), radius * c, radius * s]
    raise ValueError("plane must be one of {'xy','xz','yz'}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent))
    ap.add_argument("--fps", type=int, default=30)

    ap.add_argument("--dt-coarse", type=float, default=120.0)
    ap.add_argument("--dt-fine", type=float, default=5.0)
    ap.add_argument("--fine-window", type=float, default=180.0)

    ap.add_argument("--pre-roll", type=float, default=180.0)
    ap.add_argument("--post-roll", type=float, default=900.0)

    ap.add_argument("--tail", type=int, default=220)
    ap.add_argument("--res-window-min", type=float, default=45.0)
    ap.add_argument("--nav-window-min", type=float, default=20.0)

    ap.add_argument("--tcm-flash-s", type=float, default=120.0)
    ap.add_argument("--meas-flash-s", type=float, default=10.0)
    ap.add_argument("--orbit-deg-per-frame", type=float, default=0.08)

    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_dir).resolve()

    # Deploy schedule -> tau seconds since earliest deploy
    deploy_dt = {d: parse_utc(DEPLOY_UTC_FROM_PLOT[d]) for d in range(N_DEP)}
    t0_utc = min(deploy_dt.values())
    deploy_tau = np.array([(deploy_dt[d] - t0_utc).total_seconds() for d in range(N_DEP)], dtype=float)
    order = np.argsort(deploy_tau)
    print("[deploy] order:", ", ".join(f"D{int(d)+1}@{DEPLOY_UTC_FROM_PLOT[int(d)]}" for d in order))

    # Identity colors (per deputy)
    cmap = plt.get_cmap("tab10")
    id_rgba = [brighten_rgba(cmap(d % 10), factor=0.35) for d in range(N_DEP)]

    # Precompute per-deputy label offset directions (spread labels around points)
    theta = np.linspace(0.0, 2.0 * np.pi, N_DEP, endpoint=False)
    lbl_u2 = np.c_[np.cos(theta), np.sin(theta)]  # for 2D labels
    lbl_u3 = np.c_[np.cos(theta), np.sin(theta), np.zeros_like(theta)]  # for 3D labels
    lbl_zsign = np.where((np.arange(N_DEP) % 2) == 0, 1.0, -1.0)

    # Load truth bundles
    fleet: list[dict] = []
    for d in range(N_DEP):
        p = root / f"apophis_rendezvous_full_D{d}.npz"
        if not p.is_file():
            raise FileNotFoundError(p)

        with np.load(p, allow_pickle=True) as npz:
            t_full = np.asarray(npz["t"], float).reshape(-1)
            rD = np.asarray(npz["rD_N"], float).reshape(-1, 3)
            vD = np.asarray(npz["vD_N"], float).reshape(-1, 3)
            rC = np.asarray(npz["rC_N"], float).reshape(-1, 3)
            vC = np.asarray(npz["vC_N"], float).reshape(-1, 3)
            rA = np.asarray(npz["rA_N"], float).reshape(-1, 3)
            impulses = np.asarray(npz["impulses"], float).reshape(-1, 4) if "impulses" in npz else np.zeros((0, 4), dtype=float)

        rho = eci_to_hill_pos(rD, vD, rC, vC)

        t0_full = float(t_full[0])
        tau_deploy = float(deploy_tau[d])
        offset = tau_deploy - t0_full
        tau_full = t_full + offset

        rng = la.norm(rho, axis=1)
        rho_dot = np.gradient(rho, tau_full, axis=0)
        spd = la.norm(rho_dot, axis=1)

        sep = la.norm(rD - rA, axis=1)
        kmin = int(np.argmin(sep))
        tau_flyby_d = float(tau_full[kmin])

        tcm_tau = np.array([], dtype=float)
        tcm_dv = np.zeros((0, 3), dtype=float)
        if impulses.size > 0:
            impulses = impulses[np.argsort(impulses[:, 0])]
            tcm_tau = impulses[:, 0] + offset
            tcm_dv = impulses[:, 1:4].copy()

        fleet.append(dict(
            d=d,
            offset=offset,
            tau=tau_full,
            rho=rho,
            rho_dot=rho_dot,
            range=rng,
            speed=spd,
            tau_deploy=tau_deploy,
            rho_attach=rho[0].copy(),
            tau_flyby=tau_flyby_d,
            tcm_tau=tcm_tau,
            tcm_dv=tcm_dv,
        ))

    tau_flyby_global = float(np.median([dep["tau_flyby"] for dep in fleet]))
    flyby_utc = t0_utc + timedelta(seconds=tau_flyby_global)
    print(f"[flyby] tau_flyby_global={tau_flyby_global:.1f} s ({flyby_utc.strftime('%Y-%m-%d %H:%M:%SZ')})")

    # Load FDI results if present
    have_fdi = True
    for dep in fleet:
        fp = root / f"apophis_swarm_FDI_results_D{dep['d']}.npz"
        if not fp.is_file():
            have_fdi = False
            break

    if have_fdi:
        import sys
        script_dir = Path(__file__).resolve().parent
        root_dir = None
        for parent in script_dir.parents:
            if (parent / "GNC").is_dir():
                root_dir = parent
                break
        if root_dir is None:
            root_dir = script_dir.parents[2]
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))

        from GNC.Navigation.fdi_core import FDIState, FaultType  # type: ignore[import]

        for dep in fleet:
            fp = root / f"apophis_swarm_FDI_results_D{dep['d']}.npz"
            with np.load(fp) as npz:
                t_fdi = np.asarray(npz["t_fdi"], float).reshape(-1)
                dep["tau_fdi"] = t_fdi + float(dep["offset"])
                dep["state_code"] = np.asarray(npz["state_code"], int).reshape(-1)
                dep["fault_code"] = np.asarray(npz["fault_code"], int).reshape(-1)
                dep["confidence"] = np.asarray(npz["confidence"], float).reshape(-1)
                dep["r_nav"]  = np.asarray(npz["r_nav"], float).reshape(-1)
                dep["r_att"]  = np.asarray(npz["r_att"], float).reshape(-1)
                dep["r_form"] = np.asarray(npz["r_form"], float).reshape(-1)
                dep["r_peer"] = np.asarray(npz["r_peer"], float).reshape(-1)

        print("[FDI] loaded apophis_swarm_FDI_results_D*.npz")
    else:
        print("[FDI] not found (health shows STOWED/DEPLOYED)")

    # Load NAV stream if present
    have_nav_any = False
    for dep in fleet:
        fn = root / f"apophis_swarm_FDI_D{dep['d']}.npz"
        if fn.is_file():
            have_nav_any = True
            with np.load(fn) as npz:
                t_nav = np.asarray(npz["t"], float).reshape(-1)
                y_pre = np.asarray(npz["y_pre"], float).reshape(-1, 3)
                sigma_meas = np.asarray(npz["sigma_meas"], float).reshape(-1, 3)
                P_nav = np.asarray(npz["P"], float).reshape(-1, 6, 6) if "P" in npz else None

            tau_nav = t_nav + float(dep["offset"])
            w_comp, innov_norm, nis, drho, drho_norm = compute_nav_metrics(y_pre, sigma_meas, P_nav)

            dep["tau_nav"] = tau_nav
            dep["y_pre"] = y_pre
            dep["sigma_meas"] = sigma_meas
            dep["w_comp"] = w_comp
            dep["innov_norm"] = innov_norm
            dep["nis"] = nis
            dep["drho"] = drho
            dep["drho_norm"] = drho_norm
            dep["P_nav"] = P_nav
        else:
            dep["tau_nav"] = None
            dep["y_pre"] = None
            dep["sigma_meas"] = None
            dep["w_comp"] = None
            dep["innov_norm"] = None
            dep["nis"] = None
            dep["drho"] = None
            dep["drho_norm"] = None
            dep["P_nav"] = None

    print("[NAV] loaded" if have_nav_any else "[NAV] not found")

    # Build tau grid with time-warp
    tau_start = -float(args.pre_roll)
    tau_end = tau_flyby_global + float(args.post_roll)

    base = np.arange(tau_start, tau_end + 0.5 * float(args.dt_coarse), float(args.dt_coarse))
    w = float(args.fine_window)

    fine_segments = []
    for dep in fleet:
        td = float(dep["tau_deploy"])
        fine_segments.append(np.arange(td - w, td + w + 0.5 * float(args.dt_fine), float(args.dt_fine)))
    for dep in fleet:
        for tt in np.asarray(dep["tcm_tau"], float).reshape(-1):
            fine_segments.append(np.arange(tt - w, tt + w + 0.5 * float(args.dt_fine), float(args.dt_fine)))
    fine_segments.append(np.arange(tau_flyby_global - w, tau_flyby_global + w + 0.5 * float(args.dt_fine), float(args.dt_fine)))

    tau_grid = np.concatenate([base] + fine_segments)
    tau_grid = np.unique(np.round(tau_grid, 6))
    tau_grid.sort()

    print(f"[anim] frames={tau_grid.size}, tau range [{tau_grid[0]:.1f}, {tau_grid[-1]:.1f}] s, fps={args.fps}")

    # Ring geometry + plot limits
    attach_pts = np.vstack([dep["rho_attach"] for dep in fleet])
    uR, vR, nR, ring_r = fit_ring_plane(attach_pts)

    all_rho = np.vstack([dep["rho"] for dep in fleet])
    lim = float(np.percentile(np.abs(all_rho), 99.0))
    lim = max(lim, 3.0 * ring_r) * 1.12
    meas_arrow_maxlen = float(MEAS_ARROW_MAXLEN_FRAC) * lim
    keep_out_r = float(max(1.0, KEEP_OUT_R_FRAC_OF_RING * ring_r))

    # Event times for camera freezing
    event_taus = []
    event_taus.extend([float(x) for x in deploy_tau])
    for dep in fleet:
        for tt in np.asarray(dep["tcm_tau"], float).reshape(-1):
            event_taus.append(float(tt))
    event_taus.append(float(tau_flyby_global))
    event_taus = np.array(sorted(set(event_taus)), dtype=float)

    # -----------------------------
    # Figure + axes
    # -----------------------------
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("black")

    ax3 = fig.add_axes(AX_POS["ax3"], projection="3d")
    ax_rt = fig.add_axes(AX_POS["ax_rt"])
    ax_rn = fig.add_axes(AX_POS["ax_rn"])

    ax_health = fig.add_axes(AX_POS["ax_health"])
    ax_res = fig.add_axes(AX_POS["ax_res"])
    ax_geom = fig.add_axes(AX_POS["ax_geom"])
    ax_nav = fig.add_axes(AX_POS["ax_nav"])
    ax_comp = fig.add_axes(AX_POS["ax_comp"])
    ax_evt = fig.add_axes(AX_POS["ax_evt"])

    style_3d_black(ax3, tick_fs=10, label_fs=11)
    for ax in (ax_rt, ax_rn, ax_health, ax_res, ax_geom, ax_nav, ax_comp, ax_evt):
        style_2d_black(ax, title_fs=12, label_fs=11, tick_fs=10)
    ax_health.grid(False)

    # Twin axis for speed on geometry panel
    ax_geom2 = ax_geom.twinx()
    style_2d_black(ax_geom2, title_fs=12, label_fs=11, tick_fs=10)
    ax_geom2.grid(False)
    ax_geom2.patch.set_alpha(0.0)
    for sp in ax_geom2.spines.values():
        sp.set_color(AXIS_RGBA)
    ax_geom2.tick_params(colors=TEXT_COLOR, labelsize=10)

    # -----------------------------
    # 3D axis setup
    # -----------------------------
    ax3.set_title(
    "Chief Hill Frame (LVLH) — Deputies / Deploy / TCM / NAV",
    color=TEXT_COLOR,
    fontsize=15,
    y=0.94,   # <-- lower than default (tweak 0.92–0.96)
    pad=0
    )
    ax3.set_xlabel("R (x) [m]", fontsize=12); ax3.set_ylabel("T (y) [m]", fontsize=12); ax3.set_zlabel("N (z) [m]", fontsize=12)
    ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim); ax3.set_zlim(-lim, lim)
    ax3.set_autoscale_on(False)
    try:
        ax3.set_box_aspect([1, 1, 1])
        ax3.set_proj_type("persp")
    except Exception:
        pass

    ax3.plot([0.0], [0.0], [0.0], marker="o", linestyle="", color="white", markersize=7, alpha=0.95)

    th = np.linspace(0.0, 2.0 * np.pi, 240)
    ring3 = (ring_r * np.cos(th)).reshape(-1, 1) * uR.reshape(1, 3) + (ring_r * np.sin(th)).reshape(-1, 1) * vR.reshape(1, 3)
    ax3.plot(ring3[:, 0], ring3[:, 1], ring3[:, 2], linewidth=2.0, alpha=0.18, color="white")

    if SHOW_RANGE_SHELLS:
        for frac in SHELL_FRACS:
            rr = float(frac) * lim
            for pl in ("xy", "xz", "yz"):
                pts = circle_3d(rr, plane=pl, n=220)
                ax3.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="white", alpha=0.1, linewidth=1.0)

    for pl in ("xy", "xz", "yz"):
        pts = circle_3d(keep_out_r, plane=pl, n=240)
        ax3.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=STATE_COLORS["FAULTED"], alpha=0.22, linewidth=1.2)

    triL = 0.12 * lim
    ax3.plot([0, triL], [0, 0], [0, 0], color=RTN_COLORS["R"], linewidth=2.0, alpha=0.90)
    ax3.plot([0, 0], [0, triL], [0, 0], color=RTN_COLORS["T"], linewidth=2.0, alpha=0.90)
    ax3.plot([0, 0], [0, 0], [0, triL], color=RTN_COLORS["N"], linewidth=2.0, alpha=0.90)
    try:
        ax3.text(triL * 1.03, 0, 0, "R", color=RTN_COLORS["R"], fontsize=12)
        ax3.text(0, triL * 1.03, 0, "T", color=RTN_COLORS["T"], fontsize=12)
        ax3.text(0, 0, triL * 1.03, "N", color=RTN_COLORS["N"], fontsize=12)
    except Exception:
        pass

    # -----------------------------
    # 2D projections
    # -----------------------------
    ax_rt.set_title("R–T Projection", color=TEXT_COLOR, fontsize=15)
    ax_rn.set_title("R–N Projection", color=TEXT_COLOR, fontsize=15)
    for ax2 in (ax_rt, ax_rn):
        ax2.axhline(0.0, color="white", alpha=0.10, linewidth=1.0)
        ax2.axvline(0.0, color="white", alpha=0.10, linewidth=1.0)
        ax2.set_aspect("equal", adjustable="box")

    ax_rt.set_xlabel("R [m]"); ax_rt.set_ylabel("T [m]")
    ax_rn.set_xlabel("R [m]"); ax_rn.set_ylabel("N [m]")

    ax_rt.plot(ring3[:, 0], ring3[:, 1], color="white", alpha=0.14, linewidth=1.2)
    ax_rn.plot(ring3[:, 0], ring3[:, 2], color="white", alpha=0.14, linewidth=1.2)

    ax_rt.add_patch(Circle((0.0, 0.0), keep_out_r, fill=False, edgecolor=STATE_COLORS["FAULTED"], alpha=0.25, linewidth=1.2))
    ax_rn.add_patch(Circle((0.0, 0.0), keep_out_r, fill=False, edgecolor=STATE_COLORS["FAULTED"], alpha=0.25, linewidth=1.2))
    ax_rt.grid(True, color=(1,1,1,0.05), alpha=0.1)
    ax_rn.grid(True, color=(1,1,1,0.05), alpha=0.1)

    # -----------------------------
    # Top HUD texts
    # -----------------------------
    focus_txt = fig.text(0.03, 0.972, "", ha="left", va="top", fontsize=15, color=TEXT_COLOR)
    time_txt = fig.text(0.985, 0.972, "", ha="right", va="top", fontsize=15, color=TEXT_COLOR)
    event_txt = fig.text(0.03, 0.845, "", ha="left", va="top", fontsize=15, color=TEXT_COLOR)

    # -----------------------------
    # Health panel
    # -----------------------------
    ax_health.set_title("Health Vector", color=TEXT_COLOR, fontsize=15)
    ax_health.set_xlim(0.0, 1.0)
    ax_health.set_ylim(-0.5, N_DEP - 0.5)
    ax_health.set_xlabel("Confidence", fontsize=12)
    ax_health.set_yticks(range(N_DEP))
    ax_health.set_yticklabels([f"D{d+1}" for d in range(N_DEP)], color=TEXT_COLOR, fontsize=12)
    bars = ax_health.barh(np.arange(N_DEP), np.zeros(N_DEP),
                          color=[STATE_COLORS["STOWED"]] * N_DEP, alpha=0.92)
    info_texts = [
        ax_health.text(0.02, d, "STOWED", va="center", ha="left", fontsize=12, color=TEXT_COLOR)
        for d in range(N_DEP)
    ]

    # -----------------------------
    # Residuals panel
    # -----------------------------
    ax_res.set_title("FDI residuals (focus: —)", color=TEXT_COLOR, fontsize=15)
    ax_res.set_ylabel("Residual", fontsize=12)
    ax_res.set_xlabel("Time Since Earliest Deploy [h]", fontsize=12)

    (ln_rnav,) = ax_res.plot([], [], linewidth=1.3, label="r_nav", color=STATE_COLORS["NO_FDI"])
    (ln_ratt,) = ax_res.plot([], [], linewidth=1.0, label="r_att", color="#66ffcc")
    (ln_rform,) = ax_res.plot([], [], linewidth=1.0, label="r_form", color="#ff66ff")
    (ln_rpeer,) = ax_res.plot([], [], linewidth=1.0, label="r_peer", color="#ffaa66")

    for y, a in [(NU_NAV, 0.25), (NU_ATT, 0.18), (NU_FORM, 0.18), (NU_PEER, 0.18)]:
        ax_res.axhline(y, linestyle="--", linewidth=1.0, alpha=a, color="white")

    now_vline_res = ax_res.axvline(0.0, color="white", linewidth=1.0, alpha=0.30)
    tcm_scatter_res = ax_res.scatter([], [], s=26, marker="x", alpha=0.9, c=[(1, 1, 1, 0.9)])

    dom_res_txt = ax_res.text(0.02, 0.92, "", transform=ax_res.transAxes,
                              ha="left", va="top", fontsize=12, color="white")

    leg_res = ax_res.legend(loc="upper right", fontsize=15, framealpha=0.12)
    leg_res.get_frame().set_facecolor("black")
    leg_res.get_frame().set_edgecolor(AXIS_RGBA)
    for t in leg_res.get_texts():
        t.set_color(TEXT_COLOR)
    ax_res.grid(True, color=(1,1,1,0.05), alpha=0.1)

    # -----------------------------
    # Geometry
    # -----------------------------
    ax_geom.set_title("Geometry (focus: —)", color=TEXT_COLOR, fontsize=15)
    ax_geom.set_xlabel("Time Since Earliest Deploy [h]", fontsize=12)
    ax_geom.set_ylabel("||rho|| [m]", fontsize=12)
    ax_geom2.set_ylabel("||rho_dot|| [m/s]", fontsize=12)

    (ln_rng,) = ax_geom.plot([], [], linewidth=1.4, label="range", color="white")
    (ln_spd,) = ax_geom2.plot([], [], linewidth=1.1, label="speed", color=RTN_COLORS["R"], alpha=0.9)
    now_vline_geom = ax_geom.axvline(0.0, color="white", linewidth=1.0, alpha=0.25)

    geom_txt = ax_geom.text(0.02, 0.86, "", transform=ax_geom.transAxes,
                            ha="left", va="top", fontsize=12, color="white")
    geom_txt.set_visible(False)

    h1, l1 = ax_geom.get_legend_handles_labels()
    h2, l2 = ax_geom2.get_legend_handles_labels()

    leg_geom = ax_geom.legend(h1 + h2, l1 + l2, loc="upper right",
                              fontsize=12, framealpha=0.12)
    leg_geom.get_frame().set_facecolor("black")
    leg_geom.get_frame().set_edgecolor(AXIS_RGBA)
    for t in leg_geom.get_texts():
        t.set_color(TEXT_COLOR)
    ax_geom.grid(True, color=(1,1,1,0.05), alpha=0.1)

    # -----------------------------
    # NAV consistency
    # -----------------------------
    ax_nav.set_title("NAV consistency (focus: —)", color=TEXT_COLOR, fontsize=15)
    ax_nav.set_ylabel("score", fontsize=12)
    ax_nav.set_xlabel("Time Since Earliest Deploy [h]", fontsize=12)
    ax_nav.axhline(NU_NAV, linestyle="--", linewidth=1.0, alpha=0.22, color="white")
    ax_nav.axhline(NIS_99_DF3, linestyle="--", linewidth=1.0, alpha=0.18, color="white")

    (ln_rnav2,) = ax_nav.plot([], [], linewidth=1.3, label="r_nav", color=STATE_COLORS["NO_FDI"])
    (ln_inorm,) = ax_nav.plot([], [], linewidth=1.0, alpha=0.9, label="innov_norm", color="white")
    (ln_nis,) = ax_nav.plot([], [], linewidth=1.0, alpha=0.9, label="NIS", color=RTN_COLORS["T"])

    now_vline_nav = ax_nav.axvline(0.0, color="white", linewidth=1.0, alpha=0.25)

    nav_txt = ax_nav.text(0.02, 0.86, "", transform=ax_nav.transAxes,
                          ha="left", va="top", fontsize=12, color="white")
    nav_txt.set_visible(False)

    leg_nav = ax_nav.legend(loc="upper right", fontsize=12, framealpha=0.12, ncol=1, labelspacing=0.3, handlelength=1.6, borderpad=0.3)
    leg_nav.get_frame().set_facecolor("black")
    leg_nav.get_frame().set_edgecolor(AXIS_RGBA)
    for t in leg_nav.get_texts():
        t.set_color(TEXT_COLOR)
    ax_nav.grid(True, color=(1,1,1,0.05), alpha=0.1)

    # -----------------------------
    # Innovation components
    # -----------------------------
    ax_comp.set_title("Innovation Components (focus: —)  W=y_pre/sigma", color=TEXT_COLOR, fontsize=15)
    ax_comp.set_ylabel("sigmas", fontsize=12)
    ax_comp.set_xlabel("Time Since Earliest Deploy [h]", fontsize=12)

    (ln_wR,) = ax_comp.plot([], [], linewidth=1.0, label="w_R", color=RTN_COLORS["R"])
    (ln_wT,) = ax_comp.plot([], [], linewidth=1.0, label="w_T", color=RTN_COLORS["T"])
    (ln_wN,) = ax_comp.plot([], [], linewidth=1.0, label="w_N", color=RTN_COLORS["N"])

    ax_comp.axhline(+INNOV_SIGMA_BANDS, linestyle="--", linewidth=1.0, alpha=0.18, color="white")
    ax_comp.axhline(-INNOV_SIGMA_BANDS, linestyle="--", linewidth=1.0, alpha=0.18, color="white")
    ax_comp.axhline(0.0, linestyle="-", linewidth=1.0, alpha=0.08, color="white")
    now_vline_comp = ax_comp.axvline(0.0, color="white", linewidth=1.0, alpha=0.22)

    leg_comp = ax_comp.legend(loc="upper right", fontsize=12, framealpha=0.12)
    leg_comp.get_frame().set_facecolor("black")
    leg_comp.get_frame().set_edgecolor(AXIS_RGBA)
    for t in leg_comp.get_texts():
        t.set_color(TEXT_COLOR)
    ax_comp.grid(True, color=(1,1,1,0.05), alpha=0.1)

    # -----------------------------
    # Events + focus state
    # -----------------------------
    ax_evt.set_title("Events + Focus State", color=TEXT_COLOR, fontsize=15)
    ax_evt.set_ylim(-0.6, 3.6)
    ax_evt.set_yticks([3, 2, 1, 0])
    ax_evt.set_yticklabels(["DEPLOY", "TCM", "NAV", "STATE"], color=TEXT_COLOR, fontsize=12)
    ax_evt.set_xlabel("Time Since Earliest Deploy [h]", fontsize=12)

    deploy_x = to_hr(deploy_tau)
    ax_evt.scatter(deploy_x, np.full_like(deploy_x, 3.0),
                   s=32, marker="^", c=id_rgba, alpha=0.90, edgecolors="none")

    tcm_x_all = []
    tcm_y_all = []
    tcm_c_all = []
    for dep in fleet:
        d = dep["d"]
        for tt in np.asarray(dep["tcm_tau"], float).reshape(-1):
            tcm_x_all.append(float(to_hr(tt)))
            tcm_y_all.append(2.0)
            tcm_c_all.append(id_rgba[d])
    if len(tcm_x_all):
        ax_evt.scatter(np.asarray(tcm_x_all), np.asarray(tcm_y_all),
                       s=26, marker="x", c=tcm_c_all, alpha=0.90)

    ax_evt.axvline(float(to_hr(tau_flyby_global)), color="white", alpha=0.20, linewidth=1.2, linestyle="--")
    ax_evt.text(float(to_hr(tau_flyby_global)), 3.25, "FLYBY",
            transform=ax_evt.get_xaxis_transform(),   # x = data, y = axes fraction
            color="white", fontsize=12,
            ha="center", va="top", alpha=0.75)

    nav_ticks = ax_evt.scatter([], [], s=10, marker="|", c=[(1, 1, 1, 0.65)])

    state_lc = LineCollection([], linewidths=5.5, alpha=0.90)
    ax_evt.add_collection(state_lc)
    state_txt = ax_evt.text(0.02, 0.02, "", transform=ax_evt.transAxes,
                            ha="left", va="bottom", fontsize=12, color="white")
    now_vline_evt = ax_evt.axvline(0.0, color="white", linewidth=1.0, alpha=0.25)
    ax_evt.grid(False)

    # -----------------------------
    # 3D + 2D artists per deputy
    # -----------------------------
    deployed = np.zeros(N_DEP, dtype=bool)
    pos_cache = np.zeros((N_DEP, 3), dtype=float)

    marker_lines = []
    halo_lines = []
    path_lines = []
    trail_collections: list[Line3DCollection] = []
    trail_pts: list[np.ndarray] = []
    trail_state: list[list[str]] = []

    rt_tail = []
    rn_tail = []
    rt_mk = []
    rn_mk = []

    # NEW: deputy labels for 3D / RT / RN
    lbl3d = []
    lbl_rt = []
    lbl_rn = []

    # Outline for readability over colored trails
    text_pe = [pe.withStroke(linewidth=2.2, foreground="black")]

    tcm_marker_lines: list[list] = [[] for _ in range(N_DEP)]
    tcm_arrow_line = []
    tcm_arrow_head = []
    tcm_timer = np.zeros(N_DEP, dtype=float)
    tcm_ptr = np.zeros(N_DEP, dtype=int)

    meas_arrow_line, = ax3.plot([], [], [], linewidth=2.4, alpha=0.85, color=STATE_COLORS["NO_FDI"])
    meas_arrow_head, = ax3.plot([], [], [], marker="o", linestyle="", markersize=5, alpha=0.9, color=STATE_COLORS["NO_FDI"])
    meas_arrow_line.set_visible(False)
    meas_arrow_head.set_visible(False)
    meas_timer = 0.0

    event_timer = 0.0

    for dep in fleet:
        d = dep["d"]
        p0 = dep["rho_attach"]

        (halo,) = ax3.plot([p0[0]], [p0[1]], [p0[2]], marker="o", linestyle="",
                           alpha=0.16, markersize=12, color=id_rgba[d])
        halo_lines.append(halo)

        (mk,) = ax3.plot([p0[0]], [p0[1]], [p0[2]], marker="^", linestyle="",
                         markersize=7, color=STATE_COLORS["STOWED"],
                         markeredgecolor=(1, 1, 1, 0.35), markeredgewidth=0.8)
        marker_lines.append(mk)

        (pl,) = ax3.plot([], [], [], linewidth=1.1, alpha=0.18, color=id_rgba[d])
        path_lines.append(pl)

        dummy = _dummy_seg(p0)
        lc = Line3DCollection(dummy, linewidths=2.4)
        lc.set_color([(0.0, 0.0, 0.0, 0.0)])
        ax3.add_collection3d(lc)
        trail_collections.append(lc)

        trail_pts.append(p0.reshape(1, 3).copy())
        trail_state.append([])
        dep["trail_dummy"] = dummy

        (lrt,) = ax_rt.plot([], [], linewidth=1.2, alpha=0.55, color=id_rgba[d])
        (lrn,) = ax_rn.plot([], [], linewidth=1.2, alpha=0.55, color=id_rgba[d])
        rt_tail.append(lrt)
        rn_tail.append(lrn)

        (mrt,) = ax_rt.plot([], [], marker="o", linestyle="", markersize=5, alpha=0.95, color=STATE_COLORS["STOWED"])
        (mrn,) = ax_rn.plot([], [], marker="o", linestyle="", markersize=5, alpha=0.95, color=STATE_COLORS["STOWED"])
        rt_mk.append(mrt)
        rn_mk.append(mrn)

        # NEW: create deputy labels (3D + RT + RN)
        lab = f"D{d+1}"
        t3 = ax3.text(p0[0], p0[1], p0[2], lab,
                      color="white", fontsize=LABEL_FS_3D, alpha=LABEL_ALPHA,
                      ha="left", va="bottom")
        t3.set_path_effects(text_pe)
        lbl3d.append(t3)

        tRT = ax_rt.text(p0[0], p0[1], lab,
                         color="white", fontsize=LABEL_FS_2D, alpha=LABEL_ALPHA,
                         ha="left", va="bottom")
        tRT.set_path_effects(text_pe)
        lbl_rt.append(tRT)

        tRN = ax_rn.text(p0[0], p0[2], lab,
                         color="white", fontsize=LABEL_FS_2D, alpha=LABEL_ALPHA,
                         ha="left", va="bottom")
        tRN.set_path_effects(text_pe)
        lbl_rn.append(tRN)

        # TCM markers
        ttaus = np.asarray(dep["tcm_tau"], float).reshape(-1)
        for _ in range(ttaus.size):
            (tm,) = ax3.plot([], [], [], marker="x", linestyle="", markersize=7,
                             alpha=0.95, color=id_rgba[d])
            tm.set_visible(False)
            tcm_marker_lines[d].append(tm)

        (al,) = ax3.plot([], [], [], linewidth=2.6, alpha=0.85, color=id_rgba[d])
        (ah,) = ax3.plot([], [], [], marker="o", linestyle="", markersize=5, alpha=0.9, color=id_rgba[d])
        al.set_visible(False); ah.set_visible(False)
        tcm_arrow_line.append(al)
        tcm_arrow_head.append(ah)

    proj_lim = [min(lim, max(2.8 * ring_r, 0.35 * lim))]

    # -----------------------------
    # Focus deputy logic
    # -----------------------------
    def snapshot_fdi(dep: dict, tau_now: float):
        if not have_fdi or ("tau_fdi" not in dep):
            return ("DEPLOYED", "NONE", 0.0, 0.0, 0.0, 0.0, 0.0)

        tau_fdi = np.asarray(dep["tau_fdi"], float).reshape(-1)
        idx = int(np.searchsorted(tau_fdi, tau_now, side="right") - 1)
        if idx < 0:
            return ("NO_FDI", "NONE", 0.0, 0.0, 0.0, 0.0, 0.0)

        st = FDIState(int(dep["state_code"][idx])).name  # type: ignore[name-defined]
        ft = FaultType(int(dep["fault_code"][idx])).name  # type: ignore[name-defined]
        conf = float(clamp01(dep["confidence"][idx]))

        r_nav = float(dep["r_nav"][idx])
        r_att = float(dep["r_att"][idx])
        r_form = float(dep["r_form"][idx])
        r_peer = float(dep["r_peer"][idx])
        return (st, ft, conf, r_nav, r_att, r_form, r_peer)

    def snapshot_nav(dep: dict, tau_now: float):
        tau_nav = dep.get("tau_nav", None)
        if tau_nav is None:
            return None
        tau_nav = np.asarray(tau_nav, float).reshape(-1)
        if tau_nav.size == 0:
            return None
        idx = int(np.searchsorted(tau_nav, tau_now, side="right") - 1)
        if idx < 0:
            return None
        w = np.asarray(dep["w_comp"], float)[idx, :]
        inorm = float(np.asarray(dep["innov_norm"], float)[idx])
        nis = float(np.asarray(dep["nis"], float)[idx]) if dep.get("nis", None) is not None else float(np.sum(w**2))
        drn = float(np.asarray(dep["drho_norm"], float)[idx]) if dep.get("drho_norm", None) is not None else 0.0
        return dict(idx=idx, w=w, innov_norm=inorm, nis=nis, drho_norm=drn)

    def pick_focus(tau_now: float, deployed_mask: np.ndarray) -> int:
        idxs = np.where(deployed_mask)[0]
        if idxs.size == 0:
            return int(order[0])

        if have_fdi:
            for target in ["FAULTED", "SUSPECT"]:
                for d in idxs:
                    st, *_ = snapshot_fdi(fleet[d], tau_now)
                    if st == target:
                        return int(d)

        best_d = int(idxs[0])
        best_score = -1e9
        for d in idxs:
            st, ft, conf, r_nav, r_att, r_form, r_peer = snapshot_fdi(fleet[d], tau_now)
            score = r_form if have_fdi else r_nav
            if score > best_score:
                best_score = score
                best_d = int(d)
        return best_d

    # -----------------------------
    # Animation update
    # -----------------------------
    tail_len = max(10, int(args.tail))
    res_window_s = float(args.res_window_min) * 60.0
    nav_window_s = float(args.nav_window_min) * 60.0

    last_focus = [-1]
    focus_freeze_timer = [0.0]

    def gate_meas_arrow(focus_dep: dict, tau_now: float, idx_nav: int) -> bool:
        if MEAS_GATE_MODE == "off":
            return True

        if have_fdi and ("tau_fdi" in focus_dep):
            st, ft, conf, r_nav, r_att, r_form, r_peer = snapshot_fdi(focus_dep, tau_now)
            if MEAS_GATE_MODE == "nu_nav":
                return bool(r_nav > NU_NAV)
            if MEAS_GATE_MODE == "rnav1":
                return bool(r_nav > 1.0)

        if focus_dep.get("innov_norm", None) is not None and idx_nav >= 0:
            inorm = float(np.asarray(focus_dep["innov_norm"], float)[idx_nav])
            if MEAS_GATE_MODE == "innov3":
                return bool(inorm > 3.0)

        return True

    def update(i: int):
        nonlocal meas_timer, event_timer

        tau_now = float(tau_grid[i])
        tau_prev = float(tau_grid[i - 1]) if i > 0 else tau_now
        dtau = max(0.0, tau_now - tau_prev)

        utc_now = (t0_utc + timedelta(seconds=tau_now)).strftime("%Y-%m-%d %H:%M:%SZ")
        ttf = tau_flyby_global - tau_now
        time_txt.set_text(f"UTC: {utc_now}   t: {tau_now/3600.0:+.2f} h   TTF: {ttf/60.0:+.1f} min")

        for dep in fleet:
            d = dep["d"]
            deployed[d] = bool(tau_now >= float(dep["tau_deploy"]))

        focus = pick_focus(tau_now, deployed)
        if focus != last_focus[0]:
            last_focus[0] = focus
            focus_freeze_timer[0] = FOCUS_FREEZE_S
            ax_res.set_title(f"FDI residuals (focus: D{focus+1})", color=TEXT_COLOR)
            ax_geom.set_title(f"Geometry (focus: D{focus+1})", color=TEXT_COLOR)
            ax_nav.set_title(f"NAV consistency (focus: D{focus+1})", color=TEXT_COLOR)
            ax_comp.set_title(f"Innov components (focus: D{focus+1})  W=y_pre/sigma", color=TEXT_COLOR)

        az_base = CAM_AZ0 + float(args.orbit_deg_per_frame) * i
        freeze = False
        if event_taus.size:
            dt_event = float(np.min(np.abs(event_taus - tau_now)))
            if dt_event <= EVENT_FREEZE_S:
                freeze = True
        if focus_freeze_timer[0] > 0.0:
            freeze = True
            focus_freeze_timer[0] = max(0.0, float(focus_freeze_timer[0] - dtau))

        az = CAM_AZ0 if freeze else az_base
        ax3.view_init(elev=CAM_ELEV, azim=az)

        # Per deputy
        for dep in fleet:
            d = dep["d"]
            tdep = float(dep["tau_deploy"])

            if tau_now < tdep:
                pos = dep["rho_attach"]
                st_name = "STOWED"
                ft_name = "NONE"
                conf = 0.0
                col = state_to_color(st_name, stowed=True)
            else:
                pos = interp_vec_series(dep["tau"], dep["rho"], tau_now)
                if have_fdi:
                    st_name, ft_name, conf, *_ = snapshot_fdi(dep, tau_now)
                else:
                    st_name, ft_name, conf = ("DEPLOYED", "NONE", 0.0)
                col = state_to_color(st_name, stowed=False)

            pos_cache[d, :] = pos

            msz = 7.0 + 7.0 * conf
            if d == focus and tau_now >= tdep:
                msz += 2.0
                edge = (1, 1, 1, 0.7)
                ewid = 1.3
            else:
                edge = (1, 1, 1, 0.35)
                ewid = 0.8

            marker_lines[d].set_color(col)
            marker_lines[d].set_markersize(msz)
            marker_lines[d].set_markeredgecolor(edge)
            marker_lines[d].set_markeredgewidth(ewid)
            marker_lines[d].set_data([pos[0]], [pos[1]])
            marker_lines[d].set_3d_properties([pos[2]])

            halo_lines[d].set_color(id_rgba[d])
            halo_lines[d].set_markersize(msz * 1.6)
            halo_lines[d].set_data([pos[0]], [pos[1]])
            halo_lines[d].set_3d_properties([pos[2]])

            rt_mk[d].set_color(col)
            rn_mk[d].set_color(col)
            rt_mk[d].set_markersize(5.0 if d != focus else 6.5)
            rn_mk[d].set_markersize(5.0 if d != focus else 6.5)
            rt_mk[d].set_data([pos[0]], [pos[1]])
            rn_mk[d].set_data([pos[0]], [pos[2]])

            # full path
            if tau_now >= tdep:
                tau_arr = dep["tau"]
                idxp = int(np.searchsorted(tau_arr, tau_now, side="right"))
                idxp = max(1, min(idxp, tau_arr.size))
                pts = dep["rho"][:idxp, :]
                path_lines[d].set_data(pts[:, 0], pts[:, 1])
                path_lines[d].set_3d_properties(pts[:, 2])
            else:
                path_lines[d].set_data([], [])
                path_lines[d].set_3d_properties([])

            # fading tail
            if tau_now >= tdep:
                trail_pts[d] = np.vstack([trail_pts[d], pos.reshape(1, 3)])
                trail_state[d].append(st_name)

                if trail_pts[d].shape[0] > tail_len:
                    trail_pts[d] = trail_pts[d][-tail_len:, :]
                    trail_state[d] = trail_state[d][-tail_len:]

                pts2 = trail_pts[d]
                if pts2.shape[0] >= 2:
                    segs = np.stack([pts2[:-1, :], pts2[1:, :]], axis=1)

                    cols = []
                    nseg = segs.shape[0]
                    for k in range(nseg):
                        sname = trail_state[d][k] if k < len(trail_state[d]) else st_name
                        base = mcolors.to_rgba(state_to_color(sname, stowed=False))
                        a = 0.10 + 0.90 * (k / max(nseg - 1, 1))
                        cols.append((base[0], base[1], base[2], a))

                    trail_collections[d].set_segments(segs)
                    trail_collections[d].set_color(cols)

                    rt_tail[d].set_data(pts2[:, 0], pts2[:, 1])
                    rn_tail[d].set_data(pts2[:, 0], pts2[:, 2])
                    rt_tail[d].set_alpha(0.55 if d != focus else 0.90)
                    rn_tail[d].set_alpha(0.55 if d != focus else 0.90)
                    rt_tail[d].set_linewidth(1.2 if d != focus else 2.0)
                    rn_tail[d].set_linewidth(1.2 if d != focus else 2.0)
                else:
                    dummy = dep.get("trail_dummy", _dummy_seg(pos))
                    trail_collections[d].set_segments(dummy)
                    trail_collections[d].set_color([(0.0, 0.0, 0.0, 0.0)])
                    rt_tail[d].set_data([], [])
                    rn_tail[d].set_data([], [])
            else:
                dummy = dep.get("trail_dummy", _dummy_seg(dep["rho_attach"]))
                trail_collections[d].set_segments(dummy)
                trail_collections[d].set_color([(0.0, 0.0, 0.0, 0.0)])
                rt_tail[d].set_data([], [])
                rn_tail[d].set_data([], [])

            # Health bars
            if tau_now < tdep:
                bars[d].set_width(0.0)
                bars[d].set_color(STATE_COLORS["STOWED"])
                info_texts[d].set_text("STOWED")
            else:
                bars[d].set_width(conf)
                bars[d].set_color(col)
                info_texts[d].set_text(f"{st_name} | {ft_name}" if have_fdi else "DEPLOYED")

            # --- NEW: update labels in 3D + RT + RN ---
            off3 = LABEL_OFF_3D_FRAC * lim
            dz3 = LABEL_Z_OFF_3D_FRAC * lim * float(lbl_zsign[d])
            p3 = pos + off3 * lbl_u3[d] + np.array([0.0, 0.0, dz3])

            # 3D text update (works reliably across mpl versions)
            lbl3d[d].set_position((float(p3[0]), float(p3[1])))
            lbl3d[d].set_3d_properties(float(p3[2]), zdir="z")

            # 2D label offsets scale with current projection limits (keeps spacing readable)
            off2 = LABEL_OFF_2D_FRAC * float(proj_lim[0])
            dxy = off2 * lbl_u2[d]

            lbl_rt[d].set_position((float(pos[0] + dxy[0]), float(pos[1] + dxy[1])))
            lbl_rn[d].set_position((float(pos[0] + dxy[0]), float(pos[2] + dxy[1])))

            # Highlight focus label a bit (operators notice focus instantly)
            if d == focus and tau_now >= tdep:
                for tt in (lbl3d[d], lbl_rt[d], lbl_rn[d]):
                    tt.set_alpha(0.98)
                    tt.set_fontsize(LABEL_FS_3D + 1)
                    tt.set_fontweight("bold")
            else:
                for tt in (lbl3d[d], lbl_rt[d], lbl_rn[d]):
                    tt.set_alpha(LABEL_ALPHA)
                    tt.set_fontsize(LABEL_FS_3D)
                    tt.set_fontweight("normal")

            # Deploy banner
            if (tau_prev < tdep <= tau_now) and (tau_now >= tdep):
                event_txt.set_text(f"DEPLOY: D{d+1} @ {DEPLOY_UTC_FROM_PLOT[d]}")
                event_timer = 25.0

            # TCM markers + pulse arrow
            ttaus = np.asarray(dep["tcm_tau"], float).reshape(-1)
            for k, tt in enumerate(ttaus):
                if (tau_now >= tt) and (tau_now >= tdep):
                    p_imp = interp_vec_series(dep["tau"], dep["rho"], float(tt))
                    tm = tcm_marker_lines[d][k]
                    tm.set_data([p_imp[0]], [p_imp[1]])
                    tm.set_3d_properties([p_imp[2]])
                    tm.set_visible(True)
                else:
                    tcm_marker_lines[d][k].set_visible(False)

            while tcm_ptr[d] < ttaus.size and float(ttaus[tcm_ptr[d]]) <= tau_now:
                tt = float(ttaus[tcm_ptr[d]])
                dv = np.asarray(dep["tcm_dv"][tcm_ptr[d]], float).reshape(3)
                if tau_prev < tt <= tau_now:
                    p_imp = interp_vec_series(dep["tau"], dep["rho"], tt)
                    dvn = float(la.norm(dv))
                    dv_hat = (dv / dvn) if dvn > 1e-9 else np.array([1.0, 0.0, 0.0])

                    arrow_len = 0.10 * lim
                    p_end = p_imp + arrow_len * dv_hat

                    tcm_timer[d] = float(args.tcm_flash_s)
                    tcm_arrow_line[d].set_data([p_imp[0], p_end[0]], [p_imp[1], p_end[1]])
                    tcm_arrow_line[d].set_3d_properties([p_imp[2], p_end[2]])
                    tcm_arrow_line[d].set_visible(True)

                    tcm_arrow_head[d].set_data([p_end[0]], [p_end[1]])
                    tcm_arrow_head[d].set_3d_properties([p_end[2]])
                    tcm_arrow_head[d].set_visible(True)

                    event_txt.set_text(f"TCM: D{d+1}  dv_H=[{dv[0]:+.3f}, {dv[1]:+.3f}, {dv[2]:+.3f}] m/s")
                    event_timer = 25.0
                tcm_ptr[d] += 1

            if tcm_timer[d] > 0.0:
                tcm_timer[d] = max(0.0, float(tcm_timer[d] - dtau))
                if tcm_timer[d] <= 0.0:
                    tcm_arrow_line[d].set_visible(False)
                    tcm_arrow_head[d].set_visible(False)

        # Flyby banner
        if (tau_prev < tau_flyby_global <= tau_now) and (tau_now >= tau_flyby_global):
            event_txt.set_text(f"FLYBY @ {flyby_utc.strftime('%Y-%m-%d %H:%M:%SZ')}")
            event_timer = 35.0

        if event_timer > 0.0:
            event_timer = max(0.0, float(event_timer - dtau))
            if event_timer <= 0.0:
                event_txt.set_text("")

        # Projection limits (smooth)
        dep_idx = np.where(deployed)[0]
        if dep_idx.size:
            mx = float(np.max(np.abs(pos_cache[dep_idx, 0])))
            my = float(np.max(np.abs(pos_cache[dep_idx, 1])))
            mz = float(np.max(np.abs(pos_cache[dep_idx, 2])))
            new_lim = 1.25 * max(mx, my, mz, 2.2 * ring_r)
            new_lim = float(np.clip(new_lim, 2.2 * ring_r, lim))
            proj_lim[0] = 0.90 * proj_lim[0] + 0.10 * new_lim

        ax_rt.set_xlim(-proj_lim[0], proj_lim[0]); ax_rt.set_ylim(-proj_lim[0], proj_lim[0])
        ax_rn.set_xlim(-proj_lim[0], proj_lim[0]); ax_rn.set_ylim(-proj_lim[0], proj_lim[0])

        # NAV arrow (focus only)
        if MEAS_ARROW_MODE != "off" and float(args.meas_flash_s) > 0.0:
            depf = fleet[focus]
            if deployed[focus] and depf.get("tau_nav", None) is not None:
                tau_nav = np.asarray(depf["tau_nav"], float).reshape(-1)
                if tau_nav.size > 0:
                    idx_now = int(np.searchsorted(tau_nav, tau_now, side="right") - 1)
                    idx_prev = int(np.searchsorted(tau_nav, tau_prev, side="right") - 1)

                    if idx_now > idx_prev and idx_now >= 0:
                        if gate_meas_arrow(depf, tau_now, idx_now):
                            p_now = interp_vec_series(depf["tau"], depf["rho"], tau_now)

                            wvec = np.asarray(depf["w_comp"], float)[idx_now, :]
                            if MEAS_ARROW_MODE == "whitened":
                                nrm = float(la.norm(wvec))
                                direction = wvec / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])
                                vec = (0.08 * lim) * direction
                            else:
                                drho = np.asarray(depf.get("drho", np.zeros((0, 3))), float)
                                if drho.size and idx_now < drho.shape[0]:
                                    vec = clip_vec(drho[idx_now], meas_arrow_maxlen)
                                else:
                                    nrm = float(la.norm(wvec))
                                    direction = wvec / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])
                                    vec = (0.08 * lim) * direction

                            p_end = p_now + vec

                            meas_timer = float(args.meas_flash_s)
                            meas_arrow_line.set_data([p_now[0], p_end[0]], [p_now[1], p_end[1]])
                            meas_arrow_line.set_3d_properties([p_now[2], p_end[2]])
                            meas_arrow_head.set_data([p_end[0]], [p_end[1]])
                            meas_arrow_head.set_3d_properties([p_end[2]])

                            meas_arrow_line.set_visible(True)
                            meas_arrow_head.set_visible(True)
                        else:
                            meas_arrow_line.set_visible(False)
                            meas_arrow_head.set_visible(False)

            if meas_timer > 0.0:
                meas_timer = max(0.0, float(meas_timer - dtau))
                if meas_timer <= 0.0:
                    meas_arrow_line.set_visible(False)
                    meas_arrow_head.set_visible(False)

        # Time windows on telemetry
        t_hr_now = float(to_hr(tau_now))
        x0r = float(to_hr(tau_now - res_window_s))
        x1r = float(to_hr(tau_now))
        x0n = float(to_hr(tau_now - nav_window_s))
        x1n = float(to_hr(tau_now))

        ax_res.set_xlim(x0r, x1r)
        ax_geom.set_xlim(x0n, x1n)
        ax_nav.set_xlim(x0n, x1n)
        ax_comp.set_xlim(x0n, x1n)
        ax_evt.set_xlim(x0n, x1n)

        now_vline_res.set_xdata([t_hr_now, t_hr_now])
        now_vline_geom.set_xdata([t_hr_now, t_hr_now])
        now_vline_nav.set_xdata([t_hr_now, t_hr_now])
        now_vline_comp.set_xdata([t_hr_now, t_hr_now])
        now_vline_evt.set_xdata([t_hr_now, t_hr_now])

        # Focus HUD
        depf = fleet[focus]
        st_name, ft_name, conf, r_nav, r_att, r_form, r_peer = snapshot_fdi(depf, tau_now)

        rngf = interp_scalar_series(depf["tau"], depf["range"], tau_now) if tau_now >= float(depf["tau_deploy"]) else float(la.norm(depf["rho_attach"]))
        spdf = interp_scalar_series(depf["tau"], depf["speed"], tau_now) if tau_now >= float(depf["tau_deploy"]) else 0.0
        navsnap = snapshot_nav(depf, tau_now)

        if navsnap is None:
            focus_txt.set_text(
                f"FOCUS: D{focus+1} | {st_name}/{ft_name} conf={conf:.2f} | "
                f"||rho||={rngf:.1f} m  ||rho_dot||={spdf:.2f} m/s"
            )
        else:
            focus_txt.set_text(
                f"FOCUS: D{focus+1} | {st_name}/{ft_name} conf={conf:.2f} | "
                f"||rho||={rngf:.1f} m  ||rho_dot||={spdf:.2f} m/s\n"
                f"innov={navsnap['innov_norm']:.2f}σ  NIS={navsnap['nis']:.2f}  ||drho||={navsnap['drho_norm']:.2f} m"
            )

        # Residual plot
        if have_fdi and ("tau_fdi" in depf):
            tau_fdi = np.asarray(depf["tau_fdi"], float).reshape(-1)
            mask = (tau_fdi >= (tau_now - res_window_s)) & (tau_fdi <= tau_now)
            tf = to_hr(tau_fdi[mask])

            rnav = np.asarray(depf["r_nav"], float)[mask]
            ratt = np.asarray(depf["r_att"], float)[mask]
            rform = np.asarray(depf["r_form"], float)[mask]
            rpeer = np.asarray(depf["r_peer"], float)[mask]

            ln_rnav.set_data(tf, rnav)
            ln_ratt.set_data(tf, ratt)
            ln_rform.set_data(tf, rform)
            ln_rpeer.set_data(tf, rpeer)

            ymax = 1.0
            for arr in (rnav, ratt, rform, rpeer):
                if arr.size:
                    ymax = max(ymax, float(np.nanmax(arr)))
            ax_res.set_ylim(0.0, max(6.0, 1.15 * ymax))

            idxc = int(np.searchsorted(tau_fdi, tau_now, side="right") - 1)
            if idxc >= 0:
                cur = {
                    "r_nav": float(depf["r_nav"][idxc]),
                    "r_att": float(depf["r_att"][idxc]),
                    "r_form": float(depf["r_form"][idxc]),
                    "r_peer": float(depf["r_peer"][idxc]),
                }
                dom_name = max(cur, key=cur.get)
                dom_val = cur[dom_name]
                dom_color = {
                    "r_nav": STATE_COLORS["NO_FDI"],
                    "r_att": "#66ffcc",
                    "r_form": "#ff66ff",
                    "r_peer": "#ffaa66",
                }[dom_name]
                dom_res_txt.set_text(f"DOM: {dom_name}={dom_val:.2f}")
                dom_res_txt.set_color(dom_color)
            else:
                dom_res_txt.set_text("")
                dom_res_txt.set_color("white")

            ttaus = np.asarray(depf["tcm_tau"], float).reshape(-1)
            inwin = ttaus[(ttaus >= (tau_now - res_window_s)) & (ttaus <= tau_now)]
            if inwin.size:
                xs = to_hr(inwin)
                ys = np.full_like(xs, 0.92 * ax_res.get_ylim()[1])
                tcm_scatter_res.set_offsets(np.c_[xs, ys])
                tcm_scatter_res.set_color([id_rgba[focus]])
            else:
                tcm_scatter_res.set_offsets(np.zeros((0, 2)))
        else:
            ln_rnav.set_data([], [])
            ln_ratt.set_data([], [])
            ln_rform.set_data([], [])
            ln_rpeer.set_data([], [])
            tcm_scatter_res.set_offsets(np.zeros((0, 2)))
            ax_res.set_ylim(0.0, 6.0)
            dom_res_txt.set_text("")

        # Geometry plot (truth range/speed)
        tau_truth = np.asarray(depf["tau"], float).reshape(-1)
        maskg = (tau_truth >= (tau_now - nav_window_s)) & (tau_truth <= tau_now)
        tg = to_hr(tau_truth[maskg])
        rng = np.asarray(depf["range"], float)[maskg]
        spd = np.asarray(depf["speed"], float)[maskg]
        ln_rng.set_data(tg, rng)
        ln_spd.set_data(tg, spd)

        ax_geom.set_ylim(0.0, max(10.0, 1.15 * float(np.nanmax(rng)) if rng.size else 10.0))
        ax_geom2.set_ylim(0.0, max(0.5, 1.15 * float(np.nanmax(spd)) if spd.size else 0.5))
        geom_txt.set_text(f"now: ||rho||={rngf:.1f} m  ||rho_dot||={spdf:.2f} m/s")

        # NAV consistency plot
        if have_fdi and ("tau_fdi" in depf):
            tau_fdi = np.asarray(depf["tau_fdi"], float).reshape(-1)
            mask2 = (tau_fdi >= (tau_now - nav_window_s)) & (tau_fdi <= tau_now)
            tf2 = to_hr(tau_fdi[mask2])
            rnv2 = np.asarray(depf["r_nav"], float)[mask2]
            ln_rnav2.set_data(tf2, rnv2)
        else:
            ln_rnav2.set_data([], [])

        if depf.get("tau_nav", None) is not None:
            tau_nav = np.asarray(depf["tau_nav"], float).reshape(-1)
            maskn = (tau_nav >= (tau_now - nav_window_s)) & (tau_nav <= tau_now)
            tn = to_hr(tau_nav[maskn])
            inorm = np.asarray(depf["innov_norm"], float)[maskn] if depf.get("innov_norm", None) is not None else np.zeros_like(tn)
            nis = np.asarray(depf["nis"], float)[maskn] if depf.get("nis", None) is not None else inorm**2
            ln_inorm.set_data(tn, inorm)
            ln_nis.set_data(tn, nis)

            ymax_nav = 1.0
            for arr in (ln_rnav2.get_ydata(), inorm, nis):
                if len(arr):
                    ymax_nav = max(ymax_nav, float(np.nanmax(arr)))
            ax_nav.set_ylim(0.0, max(6.0, 1.15 * ymax_nav))

            if navsnap is not None:
                nav_txt.set_text(f"now: r_nav={r_nav:.2f}  innov={navsnap['innov_norm']:.2f}σ  NIS={navsnap['nis']:.2f}  ||drho||={navsnap['drho_norm']:.2f} m")
            else:
                nav_txt.set_text(f"now: r_nav={r_nav:.2f}")
        else:
            ln_inorm.set_data([], [])
            ln_nis.set_data([], [])
            ax_nav.set_ylim(0.0, 6.0)
            nav_txt.set_text("")

        # Innovation components
        if depf.get("tau_nav", None) is not None and depf.get("w_comp", None) is not None:
            tau_nav = np.asarray(depf["tau_nav"], float).reshape(-1)
            maskc = (tau_nav >= (tau_now - nav_window_s)) & (tau_nav <= tau_now)
            tc = to_hr(tau_nav[maskc])
            W = np.asarray(depf["w_comp"], float)[maskc, :]
            ln_wR.set_data(tc, W[:, 0])
            ln_wT.set_data(tc, W[:, 1])
            ln_wN.set_data(tc, W[:, 2])
            wmax = float(np.nanmax(np.abs(W))) if W.size else 1.0
            ylm = max(4.0, 1.15 * wmax)
            ax_comp.set_ylim(-ylm, +ylm)
        else:
            ln_wR.set_data([], [])
            ln_wT.set_data([], [])
            ln_wN.set_data([], [])
            ax_comp.set_ylim(-5.0, +5.0)

        # Event strip: NAV ticks + focus state strip
        if depf.get("tau_nav", None) is not None:
            tau_nav = np.asarray(depf["tau_nav"], float).reshape(-1)
            maskt = (tau_nav >= (tau_now - nav_window_s)) & (tau_nav <= tau_now)
            tvis = tau_nav[maskt]
            if tvis.size:
                stride = max(1, int(np.ceil(tvis.size / MAX_NAV_TICKS_IN_WINDOW)))
                tvis = tvis[::stride]
                xs = to_hr(tvis)
                ys = np.full_like(xs, 1.0)
                nav_ticks.set_offsets(np.c_[xs, ys])
            else:
                nav_ticks.set_offsets(np.zeros((0, 2)))
        else:
            nav_ticks.set_offsets(np.zeros((0, 2)))

        if have_fdi and ("tau_fdi" in depf):
            tau_fdi = np.asarray(depf["tau_fdi"], float).reshape(-1)
            scode = np.asarray(depf["state_code"], int).reshape(-1)

            t0 = tau_now - nav_window_s
            i0 = int(np.searchsorted(tau_fdi, t0, side="left"))
            i1 = int(np.searchsorted(tau_fdi, tau_now, side="right") - 1)

            segs = []
            cols = []
            if i1 >= 0 and i0 < tau_fdi.size:
                i0 = max(0, min(i0, tau_fdi.size - 1))
                i1 = max(0, min(i1, tau_fdi.size - 1))
                if i0 <= i1:
                    for k in range(i0, i1):
                        a = max(t0, float(tau_fdi[k]))
                        b = min(tau_now, float(tau_fdi[k + 1]))
                        if b <= a:
                            continue
                        st = FDIState(int(scode[k])).name  # type: ignore[name-defined]
                        c = mcolors.to_rgba(state_to_color(st, stowed=False))
                        segs.append([[float(to_hr(a)), 0.0], [float(to_hr(b)), 0.0]])
                        cols.append(c)

                    a = max(t0, float(tau_fdi[i1]))
                    b = tau_now
                    if b > a:
                        st = FDIState(int(scode[i1])).name  # type: ignore[name-defined]
                        c = mcolors.to_rgba(state_to_color(st, stowed=False))
                        segs.append([[float(to_hr(a)), 0.0], [float(to_hr(b)), 0.0]])
                        cols.append(c)

            state_lc.set_segments(segs)
            state_lc.set_color(cols)
            state_txt.set_text(f"D{focus+1}: {st_name}/{ft_name}")
            state_txt.set_color(state_to_color(st_name, stowed=False))
        else:
            state_lc.set_segments([])
            state_lc.set_color([])
            state_txt.set_text(f"D{focus+1}: NO_FDI")
            state_txt.set_color(STATE_COLORS["NO_FDI"])

        if i == (tau_grid.size - 1):
            event_txt.set_text("SIM COMPLETE")

        return []

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=tau_grid.size,
        interval=1000.0 / max(1, int(args.fps)),
        blit=False,
        repeat=False,
    )

    if args.save:
        import shutil
        import subprocess
        import matplotlib as mpl
        from matplotlib.animation import FFMpegWriter

        out = (root / "apophis_deploy_truth_fdi_black_ops_grid_labels.mp4").resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        # ---- locate ffmpeg (system first, then imageio_ffmpeg fallback) ----
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg  # type: ignore
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg = None

        if not ffmpeg:
            raise RuntimeError(
                "MP4 save failed: ffmpeg not found.\n"
                "Install ffmpeg (conda-forge / brew / choco) OR `pip install imageio-ffmpeg`."
            )

        mpl.rcParams["animation.ffmpeg_path"] = ffmpeg

        # sanity: ffmpeg must be runnable
        try:
            subprocess.run([ffmpeg, "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            raise RuntimeError(f"ffmpeg was found but not runnable: {ffmpeg}\n{e}")

        # remove old file if it exists (avoids silent failures on some Windows setups)
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass

        print("[save] target:", out)
        print("[save] ffmpeg:", ffmpeg)

        # ---- FHD (1920x1080) enforced by ffmpeg ----
        extra = [
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-vf", "scale=1920:1080:flags=lanczos",
        ]

        writer = FFMpegWriter(
            fps=int(args.fps),
            codec="libx264",
            bitrate=8000,
            extra_args=extra,
        )

        ani.save(str(out), writer=writer, dpi=200)

        # verify file really exists and is non-trivial
        if (not out.exists()) or (out.stat().st_size < 200_000):
            raise RuntimeError(f"Save completed but output file is missing/too small: {out}")

            print("[saved]", out)



if __name__ == "__main__":
    main()
