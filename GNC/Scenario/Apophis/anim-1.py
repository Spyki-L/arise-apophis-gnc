# -*- coding: utf-8 -*-
"""
Slide 4 (Problem statement) - 3D, clean, and FHD MP4 export

Changes you asked for:
1) Callouts now have leader lines (arrows) pointing to the actual event:
   - Coverage gap -> faulty ring segment
   - Baseline distortion -> midpoint between faulty deputy and neighbor
   - Risk + -> point on keep-out bubble closest to the faulty deputy
2) Saves as MP4 in FULL HD (1920x1080) at smooth FPS (default 60 fps)

Behavior:
- Preview plays first (close the window to proceed)
- Then it renders and saves:
    slide4_problem_statement_FHD_60fps.mp4

Run:
  python slide4_problem_statement_FHD_60fps.py """

import os
import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

# If ffmpeg is not in PATH, this often fixes MP4 export automatically:
try:
    import imageio_ffmpeg  # pip install imageio-ffmpeg
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass


# -----------------------------
# EXPORT SETTINGS (FHD + smooth)
# -----------------------------
FPS = 60
DURATION_S = 7.0

# 1920x1080 = 12.8x7.2 inches @ 150 dpi
FIGSIZE_IN = (12.8, 7.2)
DPI_PREVIEW = 120
DPI_SAVE = 150

BITRATE_KBPS = 18000  # increase to 22000+ if you want even cleaner output
OUT_MP4 = "slide4_problem_statement_FHD_60fps.mp4"


# -----------------------------
# SCENE CONFIG
# -----------------------------
N_DEPUTIES = 8
FAULTY_IDX = 3
NEIGHBOR_IDX = 4

BG_SEED = 2
RING_R = 1.05
KEEP_OUT_R = 0.22

# "random" | "asteroid" | "satellite"
TARGET_KIND = "satellite"
TARGET_CHOICES = ["asteroid", "satellite"]
TARGET_SIZE = 0.25

# Panel layout
LEFT_POS = [0.05, 0.11, 0.44, 0.73]
RIGHT_POS = [0.51, 0.11, 0.44, 0.73]

# Callout label positions (figure fractions)
RISK_TEXT_POS = (0.57, 0.79)
COV_TEXT_POS  = (0.77, 0.79)
BASE_TEXT_POS = (0.62, 0.21)
FAULT_TEXT_POS = (0.90, 0.57)


# -----------------------------
# Helpers
# -----------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def ease_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 1.0 - (1.0 - t) ** 3


def ease_between(frame: int, start: int, end: int) -> float:
    if frame <= start:
        return 0.0
    if frame >= end:
        return 1.0
    return ease_out_cubic((frame - start) / float(end - start))


def smoothstep_arr(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


# -----------------------------
# Background (no white margins)
# -----------------------------
def make_star_background(ax, seed: int = 1, h: int = 900, w: int = 1600) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rng = np.random.default_rng(seed)

    y = np.linspace(0.0, 1.0, h)[:, None]
    grad = 0.04 + 0.22 * (1.0 - y) ** 1.7
    grad2 = np.repeat(grad, w, axis=1)
    img = np.dstack([grad2, grad2, grad2])
    ax.imshow(img, extent=[0, 1, 0, 1], origin="lower", aspect="auto")

    n = 1000
    xs = rng.random(n)
    ys = rng.random(n)
    sizes = rng.uniform(1, 12, n)

    cols = np.ones((n, 4))
    cols[:, :3] = 1.0
    cols[:, 3] = rng.uniform(0.03, 0.75, n)
    ax.scatter(xs, ys, s=sizes, c=cols, linewidths=0)

    ax.scatter(rng.random(40), rng.random(40),
               s=rng.uniform(20, 60, 40), c="white", alpha=0.35, linewidths=0)


# -----------------------------
# 3D styling
# -----------------------------
def style_3d_ax(ax, elev: float = 16.0, azim: float = -55.0) -> None:
    try:
        ax.set_proj_type("persp")
    except Exception:
        pass

    ax.set_xlim(-1.9, 1.5)
    ax.set_ylim(-1.4, 1.4)
    ax.set_zlim(-1.1, 1.1)

    try:
        ax.set_box_aspect((3.4, 2.8, 2.2))
    except Exception:
        pass

    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    try:
        ax.patch.set_alpha(0.0)
    except Exception:
        pass

    ax.set_facecolor((0, 0, 0, 0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_pane_color((0, 0, 0, 0))
        except Exception:
            pass
        try:
            axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        except Exception:
            pass
        try:
            axis.line.set_color((1, 1, 1, 0))
        except Exception:
            pass


# -----------------------------
# Target object (satellite or asteroid)
# -----------------------------
def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def asteroid_mesh(r0: float, n_u: int, n_v: int, seed: int):
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    uu, vv = np.meshgrid(u, v)

    a = rng.normal(scale=0.08, size=8)
    noise = (
        a[0] * np.sin(2 * vv) * np.cos(3 * uu)
        + a[1] * np.sin(3 * vv) * np.sin(2 * uu)
        + a[2] * np.cos(4 * vv)
        + a[3] * np.sin(vv) * np.cos(5 * uu)
        + a[4] * np.sin(5 * vv) * np.sin(uu)
        + a[5] * np.cos(uu) * np.cos(2 * vv)
        + a[6] * np.sin(4 * vv) * np.cos(uu)
        + a[7] * np.sin(2 * vv) * np.sin(4 * uu)
    )
    r = r0 * (1.0 + noise)
    x = r * np.cos(uu) * np.sin(vv)
    y = r * np.sin(uu) * np.sin(vv)
    z = r * np.cos(vv)
    return x, y, z


def add_target_object(ax, kind: str, seed: int):
    if kind == "asteroid":
        x, y, z = asteroid_mesh(r0=TARGET_SIZE, n_u=70, n_v=45, seed=seed)
        ls = LightSource(azdeg=320, altdeg=45)
        base = np.ones_like(x) * 0.55
        rgb = ls.shade(base, cmap=plt.cm.copper, vert_exag=1.0, blend_mode="soft")
        ax.plot_surface(x, y, z, facecolors=rgb, linewidth=0, antialiased=True, shade=False)
        ax.text(0.0, 0.0, TARGET_SIZE * 1.4, "Target", color=(1, 1, 1, 0.65), fontsize=10, ha="center")
        return

    # Satellite target: rectangular body + blue panels (fixed orientation)
    sx, sy, sz = 0.26, 0.18, 0.10
    R = rotation_matrix_xyz(math.radians(10), math.radians(25), math.radians(35))

    v = np.array([
        [-sx, -sy, -sz], [ sx, -sy, -sz], [ sx,  sy, -sz], [-sx,  sy, -sz],
        [-sx, -sy,  sz], [ sx, -sy,  sz], [ sx,  sy,  sz], [-sx,  sy,  sz],
    ], dtype=float) @ R.T

    faces = [
        [v[j] for j in [0, 1, 2, 3]],
        [v[j] for j in [4, 5, 6, 7]],
        [v[j] for j in [0, 1, 5, 4]],
        [v[j] for j in [2, 3, 7, 6]],
        [v[j] for j in [1, 2, 6, 5]],
        [v[j] for j in [3, 0, 4, 7]],
    ]
    body = Poly3DCollection(
        faces,
        facecolors=(0.78, 0.78, 0.80, 0.95),
        edgecolors=(1, 1, 1, 0.18),
        linewidths=0.6
    )
    ax.add_collection3d(body)

    panel_w, panel_h = 0.55, 0.12
    gap = 0.02
    pL = np.array([
        [-(sx + gap), -panel_h, 0.0],
        [-(sx + gap),  panel_h, 0.0],
        [-(sx + gap + panel_w),  panel_h, 0.0],
        [-(sx + gap + panel_w), -panel_h, 0.0],
    ]) @ R.T

    pR = np.array([
        [(sx + gap), -panel_h, 0.0],
        [(sx + gap),  panel_h, 0.0],
        [(sx + gap + panel_w),  panel_h, 0.0],
        [(sx + gap + panel_w), -panel_h, 0.0],
    ]) @ R.T

    pl = Poly3DCollection([pL], facecolors=(0.20, 0.50, 0.95, 0.85), edgecolors=(1, 1, 1, 0.12), linewidths=0.5)
    pr = Poly3DCollection([pR], facecolors=(0.20, 0.50, 0.95, 0.85), edgecolors=(1, 1, 1, 0.12), linewidths=0.5)
    ax.add_collection3d(pl)
    ax.add_collection3d(pr)

    ax.text(0.0, 0.0, 0.33, "Target", color=(1, 1, 1, 0.65), fontsize=10, ha="center")


# -----------------------------
# Trajectories
# -----------------------------
def generate_paths(n: int, m: int = 700):
    t = np.linspace(0.0, 1.0, m)
    phi0 = math.radians(20)  # rotate formation a bit
    phi = phi0 + 2.0 * np.pi * np.arange(n) / n

    end = np.stack([RING_R * np.cos(phi), RING_R * np.sin(phi), 0.18 * np.sin(0.7 * phi)], axis=1)
    start = np.stack([-1.75 * np.ones(n), 0.75 * np.cos(phi), 0.75 * np.sin(phi)], axis=1)

    s = smoothstep_arr(t)
    paths = []
    for i in range(n):
        p = (1.0 - s)[:, None] * start[i] + s[:, None] * end[i]
        p[:, 0] += 0.20 * np.sin(np.pi * t) * (0.4 + 0.6 * np.sin(phi[i]) ** 2)
        p[:, 1] += 0.10 * np.sin(2.0 * np.pi * t + 0.55 * i)
        p[:, 2] += 0.06 * np.cos(2.0 * np.pi * t + 0.33 * i)
        paths.append(p)
    return t, paths, end, phi


def make_faulty(paths, t, end, faulty_idx: int, neighbor_idx: int, fault_start: float = 0.60):
    m = len(t)
    start_idx = int(fault_start * (m - 1))

    mask = t > fault_start
    tau = (t[mask] - fault_start) / (1.0 - fault_start)

    drift_dir = 0.75 * (end[neighbor_idx] - end[faulty_idx]) + 0.25 * (-end[faulty_idx])
    drift_dir = drift_dir / np.linalg.norm(drift_dir)

    drift_mag = 0.70 * (tau ** 1.25)

    fpaths = [p.copy() for p in paths]
    pf = fpaths[faulty_idx]
    pf[mask] += drift_mag[:, None] * drift_dir
    fpaths[faulty_idx] = pf

    return fpaths, start_idx


# -----------------------------
# Deputies (simple 3D icon)
# -----------------------------
def create_deputy_icon(ax, color: str):
    bus, = ax.plot([], [], [], marker="s", markersize=6.0, linestyle="None",
                   markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.7, alpha=1.0, zorder=10)
    pl, = ax.plot([], [], [], lw=2.0, color=color, alpha=0.9, zorder=9)
    pr, = ax.plot([], [], [], lw=2.0, color=color, alpha=0.9, zorder=9)
    return {"bus": bus, "pl": pl, "pr": pr, "base_color": color}


def set_deputy_icon(icon, pos, yaw: float, color=None, alpha: float = 1.0, pulse: float = 0.0):
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    c = icon["base_color"] if color is None else color

    bus = icon["bus"]
    bus.set_data([x], [y])
    bus.set_3d_properties([z])
    bus.set_markerfacecolor(c)
    bus.set_alpha(alpha)
    bus.set_markersize(6.0 + 2.2 * pulse)

    u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    b = 0.03
    p = 0.08
    center = np.array([x, y, z])

    l0 = center - b * u
    l1 = center - (b + p) * u
    r0 = center + b * u
    r1 = center + (b + p) * u

    for line, a, bpt in ((icon["pl"], l0, l1), (icon["pr"], r0, r1)):
        line.set_data([a[0], bpt[0]], [a[1], bpt[1]])
        line.set_3d_properties([a[2], bpt[2]])
        line.set_color(c)
        line.set_alpha(alpha * 0.92)


# -----------------------------
# Keep-out bubble (3 circles)
# -----------------------------
def create_keepout(ax, color: str = "#ff4d6d"):
    cxy, = ax.plot([], [], [], lw=2.0, color=color, alpha=0.0, linestyle="--", zorder=6)
    cxz, = ax.plot([], [], [], lw=2.0, color=color, alpha=0.0, linestyle="--", zorder=6)
    cyz, = ax.plot([], [], [], lw=2.0, color=color, alpha=0.0, linestyle="--", zorder=6)
    return (cxy, cxz, cyz)


def update_keepout(circles, center, r: float, alpha: float):
    x0, y0, z0 = float(center[0]), float(center[1]), float(center[2])
    th = np.linspace(0.0, 2.0 * np.pi, 80)

    # XY
    x = x0 + r * np.cos(th)
    y = y0 + r * np.sin(th)
    z = np.full_like(th, z0)
    circles[0].set_data(x, y)
    circles[0].set_3d_properties(z)
    circles[0].set_alpha(alpha)

    # XZ
    x = x0 + r * np.cos(th)
    z = z0 + r * np.sin(th)
    y = np.full_like(th, y0)
    circles[1].set_data(x, y)
    circles[1].set_3d_properties(z)
    circles[1].set_alpha(alpha)

    # YZ
    y = y0 + r * np.cos(th)
    z = z0 + r * np.sin(th)
    x = np.full_like(th, x0)
    circles[2].set_data(x, y)
    circles[2].set_3d_properties(z)
    circles[2].set_alpha(alpha)


# -----------------------------
# Projection: 3D point -> overlay axes fraction
# -----------------------------
def project_to_overlay(ax3d, overlay_ax, x: float, y: float, z: float):
    x2, y2, _ = proj3d.proj_transform(x, y, z, ax3d.get_proj())
    xdisp, ydisp = ax3d.transData.transform((x2, y2))
    xo, yo = overlay_ax.transAxes.inverted().transform((xdisp, ydisp))
    return float(xo), float(yo)


# -----------------------------
# Build scene
# -----------------------------
def build_scene(dpi: int, target_kind: str, target_seed: int):
    fig = plt.figure(figsize=FIGSIZE_IN, dpi=dpi)
    fig.patch.set_facecolor("#0b0b10")

    # Background axis
    bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    make_star_background(bg, seed=BG_SEED)

    # Panel rectangles
    for pos in (LEFT_POS, RIGHT_POS):
        bg.add_patch(Rectangle(
            (pos[0], pos[1]), pos[2], pos[3],
            facecolor=(0.05, 0.06, 0.08, 0.92),
            edgecolor=(1, 1, 1, 0.22),
            linewidth=2.3
        ))

    # 3D axes
    axL = fig.add_axes(LEFT_POS, projection="3d", zorder=5)
    axR = fig.add_axes(RIGHT_POS, projection="3d", zorder=5)
    style_3d_ax(axL)
    style_3d_ax(axR)

    add_target_object(axL, target_kind, seed=target_seed)
    add_target_object(axR, target_kind, seed=target_seed)

    # Overlay axis (for text + leader lines)
    ov = fig.add_axes([0, 0, 1, 1], zorder=20)
    ov.set_axis_off()
    ov.set_xlim(0, 1)
    ov.set_ylim(0, 1)
    ov.patch.set_alpha(0.0)

    # Title + headers
    title = ov.text(0.02, 0.955, "One Bad Deputy Can Ruin Coverage or Create Collision Risk",
                    fontsize=25, fontweight="bold", color="white",
                    ha="left", va="top", alpha=0.0)
    title.set_path_effects([pe.withStroke(linewidth=6, foreground=(0, 0, 0, 0.35))])

    hL = ov.text(0.08, 0.855, "Healthy Formation", fontsize=18, fontweight="bold",
                 color="white", ha="left", va="bottom", alpha=0.0)
    hR = ov.text(0.54, 0.855, "Faulty Deputy", fontsize=18, fontweight="bold",
                 color="white", ha="left", va="bottom", alpha=0.0)
    for h in (hL, hR):
        h.set_path_effects([pe.withStroke(linewidth=4, foreground=(0, 0, 0, 0.35))])

    ulL, = ov.plot([0.08, 0.23], [0.85, 0.85], lw=3, color="white", alpha=0.0)
    ulR, = ov.plot([0.54, 0.68], [0.85, 0.85], lw=3, color="white", alpha=0.0)

    # FAULT label
    fault_label = ov.text(FAULT_TEXT_POS[0], FAULT_TEXT_POS[1], "FAULT",
                          fontsize=16, fontweight="bold", color="#ff4d6d",
                          ha="left", va="center", alpha=0.0)
    fault_label.set_path_effects([pe.withStroke(linewidth=4, foreground=(0, 0, 0, 0.35))])

    # Callouts with leader lines (arrows)
    bbox = dict(boxstyle="round,pad=0.35", fc=(0, 0, 0, 0.35), ec=(1, 1, 1, 0.20))
    arrow = dict(arrowstyle="->", lw=2.0, color="white", shrinkA=6, shrinkB=0)

    risk_anno = ov.annotate("Risk +",
                            xy=(0.0, 0.0), xycoords="axes fraction",
                            xytext=RISK_TEXT_POS, textcoords="axes fraction",
                            fontsize=13, color="white", ha="left", va="center",
                            bbox=bbox, arrowprops=arrow, alpha=0.0)

    cov_anno = ov.annotate("Coverage Gap",
                           xy=(0.0, 0.0), xycoords="axes fraction",
                           xytext=COV_TEXT_POS, textcoords="axes fraction",
                           fontsize=13, color="white", ha="left", va="center",
                           bbox=bbox, arrowprops=arrow, alpha=0.0)

    base_anno = ov.annotate("Baseline Distortion",
                            xy=(0.0, 0.0), xycoords="axes fraction",
                            xytext=BASE_TEXT_POS, textcoords="axes fraction",
                            fontsize=13, color="white", ha="left", va="center",
                            bbox=bbox, arrowprops=arrow, alpha=0.0)

    for a in (risk_anno, cov_anno, base_anno):
        a.set_path_effects([pe.withStroke(linewidth=3, foreground=(0, 0, 0, 0.25))])
        if a.arrow_patch is not None:
            a.arrow_patch.set_alpha(0.0)

    # Paths
    t, planned, end, phi = generate_paths(N_DEPUTIES, m=700)
    faulty, fault_start_idx = make_faulty(planned, t, end, FAULTY_IDX, NEIGHBOR_IDX, fault_start=0.60)

    # Palette
    colors = ["#4cc9f0", "#4895ef", "#4361ee", "#3a0ca3", "#7209b7", "#f72585", "#b5179e", "#560bad"]

    # Faint baselines
    for i, p in enumerate(planned):
        c = colors[i % len(colors)]
        axL.plot(p[:, 0], p[:, 1], p[:, 2], color=c, alpha=0.10, lw=1.6)
        axR.plot(p[:, 0], p[:, 1], p[:, 2], color=c, alpha=0.10, lw=1.6)

    # Animated traces
    animL, animR = [], []
    for i in range(N_DEPUTIES):
        c = colors[i % len(colors)]
        l1, = axL.plot([], [], [], lw=2.6, color=c, alpha=0.95)
        l2, = axR.plot([], [], [], lw=2.6, color=c, alpha=0.95)
        animL.append(l1)
        animR.append(l2)

    fault_red, = axR.plot([], [], [], lw=3.2, color="#ff4d6d", alpha=0.0)

    # Coverage ring segments
    ringL, ringR = [], []
    for i in range(N_DEPUTIES):
        th1 = phi[i] - (np.pi / N_DEPUTIES) + 0.08
        th2 = phi[i] + (np.pi / N_DEPUTIES) - 0.08
        th = np.linspace(th1, th2, 70)
        x = RING_R * np.cos(th)
        y = RING_R * np.sin(th)
        z = 0.0 * th

        segL, = axL.plot(x, y, z, color="#49f0a8", lw=5.0, alpha=0.0)
        if i == FAULTY_IDX:
            segR, = axR.plot(x, y, z, color="#ff4d6d", lw=5.5, alpha=0.0, linestyle="--")
        else:
            segR, = axR.plot(x, y, z, color="#49f0a8", lw=5.0, alpha=0.0)

        ringL.append(segL)
        ringR.append(segR)

    # Deputy icons
    iconsL = [create_deputy_icon(axL, colors[i]) for i in range(N_DEPUTIES)]
    iconsR = [create_deputy_icon(axR, colors[i]) for i in range(N_DEPUTIES)]

    # Baseline distortion line + keep-out bubble
    base_line, = axR.plot([], [], [], lw=2.5, color="#ffd166", alpha=0.0)
    keepout = create_keepout(axR, color="#ff4d6d")

    scene = dict(
        fig=fig, bg=bg, ov=ov,
        axL=axL, axR=axR,
        title=title, hL=hL, hR=hR, ulL=ulL, ulR=ulR,
        fault_label=fault_label,
        risk_anno=risk_anno, cov_anno=cov_anno, base_anno=base_anno,
        t=t, planned=planned, faulty=faulty, fault_start_idx=fault_start_idx,
        animL=animL, animR=animR, fault_red=fault_red,
        ringL=ringL, ringR=ringR,
        iconsL=iconsL, iconsR=iconsR,
        base_line=base_line, keepout=keepout,
        phi=phi,
    )
    return fig, scene


# -----------------------------
# Animation
# -----------------------------
def make_animation(fig, scene, fps: int, repeat: bool):
    m = len(scene["t"])
    n_frames = int(DURATION_S * fps)

    # Timeline
    f_title = (0, int(0.7 * fps))
    f_hdr   = (int(0.2 * fps), int(1.1 * fps))
    f_paths = (int(0.8 * fps), int(5.2 * fps))
    f_ring  = (int(2.7 * fps), int(3.4 * fps))
    f_fault = (int(3.0 * fps), int(3.6 * fps))
    f_calls = (int(3.2 * fps), int(4.4 * fps))

    axL = scene["axL"]
    axR = scene["axR"]
    ov  = scene["ov"]
    fault_start_idx = scene["fault_start_idx"]

    # Fixed point on the faulty ring segment (for coverage gap arrow tip)
    gap_theta = float(scene["phi"][FAULTY_IDX])
    gap_point = (RING_R * math.cos(gap_theta), RING_R * math.sin(gap_theta), 0.0)

    def set_anno_alpha(anno, a: float):
        anno.set_alpha(a)
        if anno.arrow_patch is not None:
            anno.arrow_patch.set_alpha(a)

    def update(frame: int):
        # Camera drift (subtle)
        cam = ease_between(frame, *f_paths)
        az = -55.0 + 10.0 * cam
        axL.view_init(elev=16.0, azim=az)
        axR.view_init(elev=16.0, azim=az)

        # Title + headers
        a_title = ease_between(frame, *f_title)
        scene["title"].set_alpha(a_title)

        a_hdr = ease_between(frame, *f_hdr)
        scene["hL"].set_alpha(a_hdr)
        scene["hR"].set_alpha(a_hdr)
        scene["ulL"].set_alpha(0.75 * a_hdr)
        scene["ulR"].set_alpha(0.75 * a_hdr)

        # Path progress
        p_path = ease_between(frame, *f_paths)
        k = max(2, int(p_path * (m - 1)))
        tk = float(scene["t"][k - 1])

        # Rings
        a_ring = ease_between(frame, *f_ring)
        for seg in scene["ringL"]:
            seg.set_alpha(0.80 * a_ring)
        for i, seg in enumerate(scene["ringR"]):
            if i == FAULTY_IDX:
                pulse = 0.70 + 0.30 * math.sin(2.0 * math.pi * frame / (0.8 * fps))
                seg.set_alpha(0.95 * a_ring * pulse)
            else:
                seg.set_alpha(0.80 * a_ring)

        # Draw traces + deputies
        for i in range(N_DEPUTIES):
            yaw = 2.0 * math.pi * tk + 0.45 * i

            # Left (healthy)
            pL = scene["planned"][i]
            scene["animL"][i].set_data(pL[:k, 0], pL[:k, 1])
            scene["animL"][i].set_3d_properties(pL[:k, 2])
            set_deputy_icon(scene["iconsL"][i], pL[k - 1], yaw, alpha=1.0, pulse=0.0)

            # Right (faulty set)
            pR = scene["faulty"][i]

            if i != FAULTY_IDX:
                scene["animR"][i].set_data(pR[:k, 0], pR[:k, 1])
                scene["animR"][i].set_3d_properties(pR[:k, 2])
                set_deputy_icon(scene["iconsR"][i], pR[k - 1], yaw, alpha=1.0, pulse=0.0)
                continue

            kk = min(k, fault_start_idx)
            scene["animR"][i].set_data(pR[:kk, 0], pR[:kk, 1])
            scene["animR"][i].set_3d_properties(pR[:kk, 2])

            if k > fault_start_idx:
                scene["fault_red"].set_alpha(0.98)
                scene["fault_red"].set_data(pR[fault_start_idx:k, 0], pR[fault_start_idx:k, 1])
                scene["fault_red"].set_3d_properties(pR[fault_start_idx:k, 2])

                pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * frame / (0.25 * fps))
                set_deputy_icon(scene["iconsR"][i], pR[k - 1], yaw, color="#ff4d6d", alpha=1.0, pulse=pulse)
            else:
                scene["fault_red"].set_alpha(0.0)
                scene["fault_red"].set_data([], [])
                scene["fault_red"].set_3d_properties([])
                set_deputy_icon(scene["iconsR"][i], pR[k - 1], yaw, alpha=1.0, pulse=0.0)

        # Fault label fade
        a_fault = ease_between(frame, *f_fault)
        scene["fault_label"].set_alpha(a_fault)

        # Baseline distortion + keepout + callouts
        a_calls = ease_between(frame, *f_calls)

        bad = scene["faulty"][FAULTY_IDX][k - 1]
        nbr = scene["faulty"][NEIGHBOR_IDX][k - 1]
        mid = 0.5 * (bad + nbr)

        # Baseline line
        scene["base_line"].set_alpha(0.95 * a_calls)
        scene["base_line"].set_data([bad[0], nbr[0]], [bad[1], nbr[1]])
        scene["base_line"].set_3d_properties([bad[2], nbr[2]])

        # Keep-out bubble
        update_keepout(scene["keepout"], nbr, r=KEEP_OUT_R, alpha=0.85 * a_calls)

        # ---- Leader lines (ARROWS) ----
        # Coverage gap -> fixed ring point
        gx, gy = project_to_overlay(axR, ov, gap_point[0], gap_point[1], gap_point[2])
        scene["cov_anno"].xy = (gx, gy)
        set_anno_alpha(scene["cov_anno"], a_calls)

        # Baseline distortion -> midpoint between faulty + neighbor
        bx, by = project_to_overlay(axR, ov, float(mid[0]), float(mid[1]), float(mid[2]))
        scene["base_anno"].xy = (bx, by)
        set_anno_alpha(scene["base_anno"], a_calls)

        # Risk + -> point on keep-out boundary toward faulty deputy
        d = bad - nbr
        dn = float(np.linalg.norm(d))
        if dn > 1e-9:
            risk_pt = nbr + (KEEP_OUT_R * d / dn)
        else:
            risk_pt = nbr
        rx, ry = project_to_overlay(axR, ov, float(risk_pt[0]), float(risk_pt[1]), float(risk_pt[2]))
        scene["risk_anno"].xy = (rx, ry)
        set_anno_alpha(scene["risk_anno"], a_calls)

        return []

    return FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / fps, blit=False, repeat=repeat)


def pick_target():
    rng = np.random.default_rng()
    if TARGET_KIND == "random":
        kind = str(rng.choice(TARGET_CHOICES))
    else:
        kind = TARGET_KIND
    seed = int(rng.integers(0, 10000))
    return kind, seed


def main():
    kind, seed = pick_target()
    print(f"[info] Target = {kind} (seed={seed})")

    # 1) PREVIEW (close window to proceed)
    fig_p, scene_p = build_scene(dpi=DPI_PREVIEW, target_kind=kind, target_seed=seed)
    _anim_p = make_animation(fig_p, scene_p, fps=FPS, repeat=True)
    plt.show()
    plt.close(fig_p)

    # 2) SAVE MP4 (fresh render)
    fig_s, scene_s = build_scene(dpi=DPI_SAVE, target_kind=kind, target_seed=seed)
    anim_s = make_animation(fig_s, scene_s, fps=FPS, repeat=False)

    writer = FFMpegWriter(
        fps=FPS,
        bitrate=BITRATE_KBPS,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    )
    anim_s.save(OUT_MP4, writer=writer)
    plt.close(fig_s)

    print("Saved:", OUT_MP4)


if __name__ == "__main__":
    main()
