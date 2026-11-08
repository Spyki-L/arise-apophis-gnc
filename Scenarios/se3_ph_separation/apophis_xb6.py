#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
espa_xb6_D1_hill_live_dashboard.py

Single-deputy (D1) live dark dashboard:
- 3D relative motion shown in the chief-centered Hill frame:
    * LVLH if chief r,v are valid
    * else fallback to chief BODY frame (shown in HUD as "BODY")
- Deputy rendered as a 3D bus + solar panel (filled + edges)
- Separation thrust vector shown during RAFTI half-sine push
- LOS line to port + boresight vector shown in 3D
- Extra 3D "pointing inset" shows a mini spacecraft at origin + LOS/boresight + axes

Plots (dark):
- H_SE3,d (D1 ONLY)
- boresight error [deg]
- LOS roll-about-LOS error [deg]
- MRP component error (sigma - sigma_d) for D1

Run (module OR full path works):
  python espa_xb6_D1_hill_live_dashboard.py --scenario "C:\\...\\basilik_scenario_espa42_rafti_xb6_fleet_se3.py"

Optional:
  --dv 0.05                 override separation dv for D1
  --no-flip-sep             if your push direction is already correct (outward)
  --keep-translation         keep deputy translation control ON (otherwise it coasts after separation)
  --save out.mp4 --fps 30    save animation (requires ffmpeg)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from Basilisk.utilities import macros


# ----------------------------- scenario import -----------------------------

def import_scenario(path_or_mod: str):
    """
    Import a scenario either by:
      - module name:  basilik_scenario_espa42_rafti_xb6_fleet_se3
      - file path:    C:\\...\\basilik_scenario_espa42_rafti_xb6_fleet_se3.py
    """
    s = path_or_mod.strip().strip('"').strip("'")
    if s.lower().endswith(".py") or os.path.sep in s or (os.path.altsep and os.path.altsep in s):
        path = os.path.abspath(s)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scenario file not found: {path}")
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for: {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(s)


# ----------------------------- math + style helpers -----------------------------

def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(la.norm(v))
    if n < eps:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return (v / n).astype(float)

def nanmax_finite(vals, default: float) -> float:
    a = np.asarray(vals, dtype=float)
    if a.size == 0 or (not np.any(np.isfinite(a))):
        return float(default)
    return float(np.nanmax(a))

def _set_dark_2d(ax):
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_color((1, 1, 1, 0.22))
    ax.tick_params(colors=(1, 1, 1, 0.75))
    ax.xaxis.label.set_color((1, 1, 1, 0.88))
    ax.yaxis.label.set_color((1, 1, 1, 0.88))
    ax.title.set_color((1, 1, 1, 0.95))

def fade_grid(ax, alpha=0.03, lw=0.6):
    """
    Reliable grid fading (works even if style overrides initial grid settings).
    """
    ax.grid(True, which="major", color=(1, 1, 1), alpha=float(alpha), linewidth=float(lw))
    # Force existing gridline artists (some styles ignore the alpha above)
    for gl in (ax.get_xgridlines() + ax.get_ygridlines()):
        gl.set_alpha(float(alpha))
        gl.set_linewidth(float(lw))

def _set_dark_3d(ax3d, grid_alpha=0.06):
    ax3d.set_facecolor("black")
    try:
        ax3d.xaxis.set_pane_color((0, 0, 0, 1))
        ax3d.yaxis.set_pane_color((0, 0, 0, 1))
        ax3d.zaxis.set_pane_color((0, 0, 0, 1))
    except Exception:
        pass
    try:
        ax3d.xaxis._axinfo["grid"]["color"] = (1, 1, 1, float(grid_alpha))
        ax3d.yaxis._axinfo["grid"]["color"] = (1, 1, 1, float(grid_alpha))
        ax3d.zaxis._axinfo["grid"]["color"] = (1, 1, 1, float(grid_alpha))
    except Exception:
        pass
    ax3d.tick_params(colors=(1, 1, 1, 0.65))
    ax3d.xaxis.label.set_color((1, 1, 1, 0.85))
    ax3d.yaxis.label.set_color((1, 1, 1, 0.85))
    ax3d.zaxis.label.set_color((1, 1, 1, 0.85))

def _set_axes_equal_3d(ax):
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    xr = abs(xlim[1] - xlim[0])
    yr = abs(ylim[1] - ylim[0])
    zr = abs(zlim[1] - zlim[0])
    r = 0.5 * max([xr, yr, zr])
    xc = 0.5 * sum(xlim)
    yc = 0.5 * sum(ylim)
    zc = 0.5 * sum(zlim)
    ax.set_xlim3d([xc - r, xc + r])
    ax.set_ylim3d([yc - r, yc + r])
    ax.set_zlim3d([zc - r, zc + r])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

def _dummy_seg():
    p0 = np.array([0.0, 0.0, 0.0], float)
    p1 = np.array([1e-6, 0.0, 0.0], float)
    return [[p0, p1]]

def dcm_to_euler_zyx(C: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw (ZYX / 3-2-1) in radians from DCM."""
    C = np.asarray(C, float)
    yaw   = math.atan2(C[1, 0], C[0, 0])
    pitch = math.asin(max(-1.0, min(1.0, -C[2, 0])))
    roll  = math.atan2(C[2, 1], C[2, 2])
    return np.array([roll, pitch, yaw], float)

def _cube_vertices_B(size_m: float) -> np.ndarray:
    h = 0.5 * float(size_m)
    return np.array([
        [-h, -h, -h],
        [ h, -h, -h],
        [ h,  h, -h],
        [-h,  h, -h],
        [-h, -h,  h],
        [ h, -h,  h],
        [ h,  h,  h],
        [-h,  h,  h],
    ], dtype=float)

_CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

def _cube_segments(verts: np.ndarray):
    return [[verts[i], verts[j]] for i, j in _CUBE_EDGES]

def _cuboid_vertices_faces_B(lx: float, ly: float, lz: float) -> Tuple[np.ndarray, List[List[int]]]:
    hx, hy, hz = 0.5 * lx, 0.5 * ly, 0.5 * lz
    v = np.array([
        [-hx, -hy, -hz],
        [ hx, -hy, -hz],
        [ hx,  hy, -hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [ hx,  hy,  hz],
        [-hx,  hy,  hz],
    ], dtype=float)
    faces = [
        [0, 1, 2, 3],  # -Z
        [4, 5, 6, 7],  # +Z
        [0, 1, 5, 4],  # -Y
        [2, 3, 7, 6],  # +Y
        [1, 2, 6, 5],  # +X
        [0, 3, 7, 4],  # -X
    ]
    return v, faces

def _cam_axis_B(cam_axis: str) -> np.ndarray:
    c = cam_axis.lower().strip()
    if c == "x":
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if c == "y":
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return np.array([0.0, 0.0, 1.0], dtype=float)


# ----------------------------- Hill frame helper -----------------------------

def chief_hill_C_HN(r_c_N: np.ndarray, v_c_N: np.ndarray, C_BN_chief: Optional[np.ndarray]) -> Tuple[np.ndarray, str]:
    """
    Return C_HN mapping vectors in N -> H (chief-centered Hill/LVLH if possible).

    LVLH if r,v allow:
      x_H = r̂
      z_H = (r×v)^
      y_H = z×x

    Else fallback:
      H := chief BODY frame (C_HN = C_BN_chief), so relative motion is expressed in chief body axes.
    """
    eps = 1e-9
    r = np.array(r_c_N, dtype=float).reshape(3)
    v = np.array(v_c_N, dtype=float).reshape(3)
    nr = float(la.norm(r))
    nv = float(la.norm(v))

    if (nr > eps) and (nv > eps):
        h = np.cross(r, v)
        nh = float(la.norm(h))
        if nh > eps:
            x = r / nr
            z = h / nh
            y = _unit(np.cross(z, x))
            C_HN = np.vstack((x, y, z)).astype(float)  # rows = H axes in N
            return C_HN, "LVLH"

    if C_BN_chief is not None:
        return np.array(C_BN_chief, dtype=float), "BODY"

    return np.eye(3, dtype=float), "N"


# ----------------------------- config -----------------------------

@dataclass
class DashConfig:
    plot_skip: int = 5
    trail_len: int = 300
    axis_lim: float = 6.0
    bore_len: float = 0.55
    thrust_len: float = 0.85
    dpi: int = 140


# ----------------------------- live dashboard -----------------------------

class D1HillDashboard:
    def __init__(self, fleet, scn_mod, cfg: DashConfig):
        self.fleet = fleet
        self.scn = scn_mod
        self.cfg = cfg

        # time histories
        self.t_hist: List[float] = []
        self.H_d1: List[float] = []
        self.bore_deg: List[float] = []
        self.los_roll_deg: List[float] = []
        self.sig_err: List[List[float]] = [[], [], []]

        # D1 relative trail in Hill frame
        self.rH_trail: List[np.ndarray] = []

        # continuity helpers for desired sigma
        self._qd_prev: Optional[np.ndarray] = None
        self._sd_prev: Optional[np.ndarray] = None

        # dark defaults
        plt.style.use("dark_background")
        plt.rcParams["figure.facecolor"] = "black"
        plt.rcParams["axes.facecolor"] = "black"

        self.fig = plt.figure(figsize=(18, 9), dpi=cfg.dpi, facecolor="black")
        gs = GridSpec(
            nrows=2, ncols=3, figure=self.fig,
            width_ratios=[1.35, 2.8, 1.35],
            height_ratios=[1.0, 1.0],
            wspace=0.08, hspace=0.18
        )

        # If you want to move plots up/down, tweak these:
        self.fig.subplots_adjust(top=0.955, bottom=0.06, hspace=0.22, wspace=0.08)

        # Left plots: H, MRP error
        self.axH = self.fig.add_subplot(gs[0, 0]); _set_dark_2d(self.axH);  fade_grid(self.axH,  alpha=0.03, lw=0.6)
        self.axSig = self.fig.add_subplot(gs[1, 0]); _set_dark_2d(self.axSig); fade_grid(self.axSig, alpha=0.03, lw=0.6)

        # Right plots: boresight, LOS roll
        self.axBore = self.fig.add_subplot(gs[0, 2]); _set_dark_2d(self.axBore); fade_grid(self.axBore, alpha=0.03, lw=0.6)
        self.axLOS  = self.fig.add_subplot(gs[1, 2]); _set_dark_2d(self.axLOS);  fade_grid(self.axLOS,  alpha=0.03, lw=0.6)

        # Center 3D: Hill relative motion
        self.ax3d = self.fig.add_subplot(gs[:, 1], projection="3d")
        _set_dark_3d(self.ax3d, grid_alpha=0.05)
        self.ax3d.set_autoscale_on(False)

        self.ax3d.set_xlabel("X_H [m]")
        self.ax3d.set_ylabel("Y_H [m]")
        self.ax3d.set_zlabel("Z_H [m]")

        lim = float(cfg.axis_lim)
        self.ax3d.set_xlim(-lim, lim)
        self.ax3d.set_ylim(-lim, lim)
        self.ax3d.set_zlim(-lim, lim)
        self.ax3d.view_init(elev=22, azim=45)
        _set_axes_equal_3d(self.ax3d)

        self.fig.suptitle("ESPA 42 Chief + D1  (Hill Frame)  Separation",
                          color=(1, 1, 1, 0.92))

        # ------------------ 2D artists ------------------
        self.lineH_d1, = self.axH.plot([], [], linewidth=1.2, alpha=0.95, label="D1")
        self.axH.set_title("H_SE3,d (D1)")
        self.axH.set_xlabel("t [s]"); self.axH.set_ylabel("H")
        self.axH.set_xlim(0, float(self.fleet.sim.T_end))
        self.axH.legend(loc="upper right", framealpha=0.15, fontsize=9)
        fade_grid(self.axH,   alpha=0.1, lw=0.5)

        self.lineBore, = self.axBore.plot([], [], linewidth=1.2, alpha=0.95, label="D1")
        self.axBore.axhline(1.0, linestyle="--", linewidth=1.0, color=(1, 1, 1, 0.22))
        self.axBore.set_title("Boresight error [deg]")
        self.axBore.set_xlabel("t [s]"); self.axBore.set_ylabel("deg")
        self.axBore.set_xlim(0, float(self.fleet.sim.T_end))
        self.axBore.set_ylim(0, 10)
        self.axBore.legend(loc="upper right", framealpha=0.15, fontsize=9)
        fade_grid(self.axBore,alpha=0.1, lw=0.5)

        self.lineLOS, = self.axLOS.plot([], [], linewidth=1.2, alpha=0.95, label="D1")
        self.axLOS.axhline(2.0, linestyle="--", linewidth=1.0, color=(1, 1, 1, 0.22))
        self.axLOS.set_title("LOS roll error [deg]")
        self.axLOS.set_xlabel("t [s]"); self.axLOS.set_ylabel("deg")
        self.axLOS.set_xlim(0, float(self.fleet.sim.T_end))
        self.axLOS.set_ylim(0, 20)
        self.axLOS.legend(loc="upper right", framealpha=0.15, fontsize=9)
        fade_grid(self.axLOS, alpha=0.1, lw=0.5)

        self.lineSig = []
        for k, lab in enumerate([r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"]):
            ln, = self.axSig.plot([], [], linewidth=1.2, alpha=0.95, label=lab)
            self.lineSig.append(ln)
        self.axSig.set_title("MRP component error (D1)")
        self.axSig.set_xlabel("t [s]"); self.axSig.set_ylabel(r"$\sigma - \sigma_d$")
        self.axSig.set_xlim(0, float(self.fleet.sim.T_end))
        self.axSig.set_ylim(-1.0, 1.0)
        self.axSig.legend(loc="upper right", framealpha=0.15, fontsize=9)
        fade_grid(self.axSig, alpha=0.1, lw=0.5)

        # ------------------ 3D (Hill) artists ------------------
        # chief hub cube at origin
        self._hub_size = 0.45
        self._hub_vertsB = _cube_vertices_B(self._hub_size)
        self.hub_wire = Line3DCollection(_dummy_seg(), linewidths=1.1, colors=[(0.2, 1.0, 1.0, 0.85)])
        self.ax3d.add_collection3d(self.hub_wire)

        # ring
        self.ring_line, = self.ax3d.plot([], [], [], linewidth=1.1, alpha=0.75, color=(1, 1, 1, 0.45))

        # port marker
        self.port_scatter = self.ax3d.scatter([], [], [], s=70, c=[(1.0, 0.2, 0.2, 0.95)])

        # deputy bus + edges + panel
        self.bus_dims = (0.10, 0.20, 0.30)  # m
        self.bus_vB, self.bus_faces_idx = _cuboid_vertices_faces_B(*self.bus_dims)

        bus_faces0 = [self.bus_vB[face] for face in self.bus_faces_idx]
        self.bus_poly = Poly3DCollection(
            bus_faces0,
            facecolors=(0.20, 1.00, 1.00, 0.16),
            edgecolors=(0.20, 1.00, 1.00, 0.55),
            linewidths=1.2
        )
        self.ax3d.add_collection3d(self.bus_poly)

        self.bus_wire = Line3DCollection(_dummy_seg(), linewidths=0.9, colors=[(0.20, 1.00, 1.00, 0.85)])
        self.ax3d.add_collection3d(self.bus_wire)

        # panel attached on +Y face
        hy = 0.5 * self.bus_dims[1]
        pLx, pLz = 0.22, 0.16
        y_attach = +hy + 0.002
        self.panel_vB = np.array([
            [-pLx, y_attach, -pLz],
            [+pLx, y_attach, -pLz],
            [+pLx, y_attach, +pLz],
            [-pLx, y_attach, +pLz],
        ], dtype=float)

        self.panel_poly = Poly3DCollection(
            [self.panel_vB],
            facecolors=(1.0, 0.55, 0.0, 0.24),
            edgecolors=(1.0, 0.55, 0.0, 0.70),
            linewidths=0.9
        )
        self.ax3d.add_collection3d(self.panel_poly)

        # trail, LOS, boresight, thrust
        self.trail_line, = self.ax3d.plot([], [], [], linewidth=0.7, alpha=0.70, color=(0.6, 0.6, 1.0, 0.85))
        self.los_line, = self.ax3d.plot([], [], [], linestyle="--", linewidth=0.7, alpha=0.75, color=(1, 1, 1, 0.70))
        self.bore_line, = self.ax3d.plot([], [], [], linewidth=0.7, alpha=0.95, color=(0.60, 0.60, 1.0, 0.95))
        self.thrust_line, = self.ax3d.plot([], [], [], linewidth=1.1, alpha=0.95, color=(1.0, 1.0, 0.2, 0.95))

        # deputy body triad in Hill
        self.triad = Line3DCollection(_dummy_seg(), linewidths=1.1)
        self.ax3d.add_collection3d(self.triad)

        # HUD text (main 3D)
        self.hud_text = self.ax3d.text2D(
            0.02, 0.98, "",
            transform=self.ax3d.transAxes,
            color=(1, 1, 1, 0.92),
            fontsize=11,
            va="top"
        )

        # ------------------ pointing inset (mini spacecraft like your snap) ------------------
        bbox = self.ax3d.get_position()
        w = 0.30 * bbox.width
        h = 0.40 * bbox.height
        x0 = bbox.x1 - w - 0.01
        y0 = bbox.y1 - h - 0.02

        self.axP = self.fig.add_axes([x0, y0, w, h], projection="3d")
        _set_dark_3d(self.axP, grid_alpha=0.04)
        self.axP.set_autoscale_on(False)
        self.axP.set_title("Pointing (Hill)", color=(1, 1, 1, 0.9), fontsize=10, pad=2)

        self.axP.set_xlim(-1.2, 1.2)
        self.axP.set_ylim(-1.2, 1.2)
        self.axP.set_zlim(-1.2, 1.2)
        self.axP.view_init(elev=22, azim=35)

        # fixed Hill axes at origin
        Lax = 1.0
        hill_segs = [[np.zeros(3), np.array([Lax, 0, 0])],
                     [np.zeros(3), np.array([0, Lax, 0])],
                     [np.zeros(3), np.array([0, 0, Lax])]]
        hill_cols = [(1, 0.2, 0.2, 0.95), (0.2, 1, 0.2, 0.95), (0.2, 0.6, 1, 0.95)]
        self.p_hill_axes = Line3DCollection(hill_segs, linewidths=0.7, colors=hill_cols)
        self.axP.add_collection3d(self.p_hill_axes)

        # mini bus+panel (rotating)
        bus_faces_inset0 = [self.bus_vB[face] for face in self.bus_faces_idx]
        self.p_bus_poly = Poly3DCollection(
            bus_faces_inset0,
            facecolors=(0.20, 1.00, 1.00, 0.16),
            edgecolors=(0.20, 1.00, 1.00, 0.65),
            linewidths=0.7
        )
        self.axP.add_collection3d(self.p_bus_poly)

        self.p_bus_wire = Line3DCollection(_dummy_seg(), linewidths=0.9, colors=[(0.20, 1.00, 1.00, 0.90)])
        self.axP.add_collection3d(self.p_bus_wire)

        self.p_panel_poly = Poly3DCollection(
            [self.panel_vB],
            facecolors=(1.0, 0.55, 0.0, 0.24),
            edgecolors=(1.0, 0.55, 0.0, 0.75),
            linewidths=0.9
        )
        self.axP.add_collection3d(self.p_panel_poly)

        # LOS + boresight (line + tip dot)
        self.p_los_line, = self.axP.plot([], [], [], linestyle="--", linewidth=0.7, alpha=0.85, color=(1, 1, 1, 0.9))
        self.p_los_tip = self.axP.scatter([], [], [], s=18, c=[(1, 1, 1, 0.9)])

        self.p_bore_line, = self.axP.plot([], [], [], linewidth=0.7, alpha=0.95, color=(0.60, 0.60, 1.0, 0.95))
        self.p_bore_tip = self.axP.scatter([], [], [], s=22, c=[(0.60, 0.60, 1.0, 0.95)])

        # body axes triad (rotating)
        self.p_body_axes = Line3DCollection(_dummy_seg(), linewidths=0.7)
        self.axP.add_collection3d(self.p_body_axes)

        # text box
        self.p_text = self.axP.text2D(
            0.67, 0.95, "",
            transform=self.axP.transAxes,
            color=(1, 1, 1, 0.9),
            fontsize=9,
            va="top"
        )

        plt.show(block=False)

    def _desired_sigma_continuous(self, Rd: np.ndarray) -> np.ndarray:
        """
        Convert desired DCM->MRP with continuity:
        - keep quaternion sign consistent
        - choose between MRP and shadow to stay close to previous
        """
        if hasattr(self.scn, "dcm_to_quat"):
            q = np.array(self.scn.dcm_to_quat(Rd), dtype=float)
            q = q / la.norm(q)
            if self._qd_prev is not None and float(np.dot(self._qd_prev, q)) < 0.0:
                q = -q

            q0 = float(q[0])
            qv = np.array(q[1:], dtype=float)
            eps = 1e-12
            if (1.0 + q0) > eps:
                s = qv / (1.0 + q0)
            else:
                s = -qv / max(eps, (1.0 - q0))

            s2 = float(np.dot(s, s))
            s_shadow = (-s / s2) if s2 > eps else s

            if self._sd_prev is None:
                sout = s
            else:
                sout = s if la.norm(s - self._sd_prev) <= la.norm(s_shadow - self._sd_prev) else s_shadow

            self._qd_prev = q
            self._sd_prev = sout
            return sout

        return np.array(self.scn.Fleet.dcm_to_mrp_safe(Rd), dtype=float)

    def update(self, t: float, Rd: np.ndarray):
        if not plt.fignum_exists(self.fig.number):
            return

        # pull controller metrics (D1 only)
        ctl = self.fleet.ctrl_deps[0]
        Hd   = float(ctl.Hd_list[-1]) if ctl.Hd_list else 0.0
        bore = float(ctl.boresight_err_hist[-1]) if ctl.boresight_err_hist else np.nan
        losr = float(ctl.roll_about_los_err_hist[-1]) if ctl.roll_about_los_err_hist else np.nan

        self.t_hist.append(float(t))
        self.H_d1.append(Hd)
        self.bore_deg.append(bore)
        self.los_roll_deg.append(losr)

        sigma = np.array(self.fleet.nav_deps[0].attOutMsg.read().sigma_BN, dtype=float)
        sig_d = self._desired_sigma_continuous(Rd)
        err = sigma - sig_d
        for k in range(3):
            self.sig_err[k].append(float(err[k]))

        th = np.array(self.t_hist, dtype=float)

        # 2D updates
        self.lineH_d1.set_data(th, np.array(self.H_d1, dtype=float))
        self.lineBore.set_data(th, np.array(self.bore_deg, dtype=float))
        self.lineLOS.set_data(th, np.array(self.los_roll_deg, dtype=float))
        for k in range(3):
            self.lineSig[k].set_data(th, np.array(self.sig_err[k], dtype=float))

        # y-lims
        Hmax = max(1e-6, float(np.max(np.asarray(self.H_d1, float))))
        self.axH.set_ylim(0, 1.08 * Hmax)

        bore_max = nanmax_finite(self.bore_deg, default=10.0)
        los_max  = nanmax_finite(self.los_roll_deg, default=20.0)
        self.axBore.set_ylim(0, max(10.0, 1.15 * bore_max))
        self.axLOS.set_ylim(0, max(20.0, 1.15 * los_max))

        se = np.asarray(self.sig_err, float)
        if se.size and np.any(np.isfinite(se)):
            m = float(np.nanmax(np.abs(se)))
            m = max(0.5, min(2.0, 1.15 * m))
            self.axSig.set_ylim(-m, m)

        # ------------------ Hill frame transform ------------------
        car_sigma = np.array(self.fleet.nav_car.attOutMsg.read().sigma_BN, dtype=float)
        C_BN_car = self.scn.mrp_to_dcm(car_sigma)   # N -> B
        C_NB_car = C_BN_car.T                       # B -> N

        r_car_N = np.array(self.fleet.nav_car.transOutMsg.read().r_BN_N, dtype=float)
        v_car_N = np.array(self.fleet.nav_car.transOutMsg.read().v_BN_N, dtype=float)

        C_HN, hill_mode = chief_hill_C_HN(r_car_N, v_car_N, C_BN_car)

        def to_H(p_N: np.ndarray) -> np.ndarray:
            return C_HN @ (np.asarray(p_N, float) - r_car_N)

        # ------------------ chief hub + ring + port in Hill ------------------
        hubH = (C_HN @ C_NB_car @ self._hub_vertsB.T).T
        self.hub_wire.set_segments(_cube_segments(hubH))

        ang = np.linspace(0, 2 * math.pi, 220)
        ringB = np.column_stack([
            self.fleet.espa_radius * np.cos(ang),
            self.fleet.espa_radius * np.sin(ang),
            np.zeros_like(ang)
        ])
        ringH = (C_HN @ C_NB_car @ ringB.T).T
        self.ring_line.set_data(ringH[:, 0], ringH[:, 1])
        self.ring_line.set_3d_properties(ringH[:, 2])

        portN = np.array(self.fleet._port_position_in_N(0), dtype=float)
        portH = to_H(portN)
        self.port_scatter._offsets3d = ([portH[0]], [portH[1]], [portH[2]])

        # ------------------ deputy state and geometry in Hill ------------------
        dep_rN = np.array(self.fleet.nav_deps[0].transOutMsg.read().r_BN_N, dtype=float)
        dep_sigma = np.array(self.fleet.nav_deps[0].attOutMsg.read().sigma_BN, dtype=float)
        C_BN_dep = self.scn.mrp_to_dcm(dep_sigma)
        C_NB_dep = C_BN_dep.T

        depH = to_H(dep_rN)
        C_HB = C_HN @ C_NB_dep  # B -> H

        bus_vH = depH + (C_HB @ self.bus_vB.T).T
        bus_facesH = [bus_vH[face] for face in self.bus_faces_idx]
        self.bus_poly.set_verts(bus_facesH)
        self.bus_wire.set_segments(_cube_segments(bus_vH))

        panel_vH = depH + (C_HB @ self.panel_vB.T).T
        self.panel_poly.set_verts([panel_vH])

        # trail
        self.rH_trail.append(depH.copy())
        if len(self.rH_trail) > self.cfg.trail_len:
            self.rH_trail = self.rH_trail[-self.cfg.trail_len:]
        tr = np.array(self.rH_trail, dtype=float)
        self.trail_line.set_data(tr[:, 0], tr[:, 1])
        self.trail_line.set_3d_properties(tr[:, 2])

        # LOS line
        losH = portH - depH
        self.los_line.set_data([depH[0], portH[0]], [depH[1], portH[1]])
        self.los_line.set_3d_properties([depH[2], portH[2]])

        # boresight vector
        camB = _cam_axis_B(getattr(self.fleet, "CAM_AXIS", "z"))
        boreH = C_HB @ camB
        bore_tip = depH + self.cfg.bore_len * _unit(boreH)
        self.bore_line.set_data([depH[0], bore_tip[0]], [depH[1], bore_tip[1]])
        self.bore_line.set_3d_properties([depH[2], bore_tip[2]])

        # separation thrust vector (during push)
        sp = self.fleet.sched[0]
        Tpush = float(sp.t_end - sp.t_start)
        Fmag = 0.0
        if (t >= sp.t_start) and (t <= sp.t_end) and (Tpush > 0):
            Fmag = float(sp.F0) * math.sin(math.pi * (t - sp.t_start) / Tpush)

        if Fmag > 0.0:
            sN = C_NB_car @ np.array(sp.s_car_B, dtype=float)  # separation direction in N
            F_N = Fmag * _unit(sN)
            Fdep_N = -F_N
            Fdep_H = C_HN @ Fdep_N
            L = self.cfg.thrust_len * (Fmag / max(1e-12, float(sp.F0)))
            thr_tip = depH + L * _unit(Fdep_H)
            self.thrust_line.set_data([depH[0], thr_tip[0]], [depH[1], thr_tip[1]])
            self.thrust_line.set_3d_properties([depH[2], thr_tip[2]])
        else:
            self.thrust_line.set_data([], [])
            self.thrust_line.set_3d_properties([])

        # body triad in Hill
        Ltri = 0.45
        xH = depH + Ltri * (C_HB @ np.array([1.0, 0.0, 0.0]))
        yH = depH + Ltri * (C_HB @ np.array([0.0, 1.0, 0.0]))
        zH = depH + Ltri * (C_HB @ np.array([0.0, 0.0, 1.0]))
        tri_segs = [[depH, xH], [depH, yH], [depH, zH]]
        tri_cols = [(1, 0.2, 0.2, 0.95), (0.2, 1, 0.2, 0.95), (0.2, 0.6, 1, 0.95)]
        self.triad.set_segments(tri_segs)
        self.triad.set_color(tri_cols)

        # ------------------ pointing inset updates (FIXED NAMES) ------------------
        # mini bus/panel rotated in Hill at origin
        bus0 = (C_HB @ self.bus_vB.T).T
        faces0 = [bus0[face] for face in self.bus_faces_idx]
        self.p_bus_poly.set_verts(faces0)
        self.p_bus_wire.set_segments(_cube_segments(bus0))

        panel0 = (C_HB @ self.panel_vB.T).T
        self.p_panel_poly.set_verts([panel0])

        u_los = _unit(losH) if la.norm(losH) > 1e-12 else np.array([1.0, 0.0, 0.0])
        u_bore = _unit(boreH)

        Llos = 1.05
        Lbore = 1.05

        # LOS line + tip
        self.p_los_line.set_data([0, Llos*u_los[0]], [0, Llos*u_los[1]])
        self.p_los_line.set_3d_properties([0, Llos*u_los[2]])
        self.p_los_tip._offsets3d = ([Llos*u_los[0]], [Llos*u_los[1]], [Llos*u_los[2]])

        # boresight line + tip
        self.p_bore_line.set_data([0, Lbore*u_bore[0]], [0, Lbore*u_bore[1]])
        self.p_bore_line.set_3d_properties([0, Lbore*u_bore[2]])
        self.p_bore_tip._offsets3d = ([Lbore*u_bore[0]], [Lbore*u_bore[1]], [Lbore*u_bore[2]])

        # body axes at origin
        xb = _unit(C_HB @ np.array([1.0, 0.0, 0.0]))
        yb = _unit(C_HB @ np.array([0.0, 1.0, 0.0]))
        zb = _unit(C_HB @ np.array([0.0, 0.0, 1.0]))
        Ltri2 = 0.75
        segs = [[np.zeros(3), Ltri2*xb],
                [np.zeros(3), Ltri2*yb],
                [np.zeros(3), Ltri2*zb]]
        cols = [(1, 0.2, 0.2, 0.95), (0.2, 1, 0.2, 0.95), (0.2, 0.6, 1, 0.95)]
        self.p_body_axes.set_segments(segs)
        self.p_body_axes.set_color(cols)

        # yaw/pitch/roll of BODY relative to Hill
        C_BH = C_HB.T
        rpy = dcm_to_euler_zyx(C_BH) * (180.0 / math.pi)
        self.p_text.set_text(
            f"3-2-1 (Hill)\n"
            f"yaw  = {rpy[2]:7.2f}°\n"
            f"pitch= {rpy[1]:7.2f}°\n"
            f"roll = {rpy[0]:7.2f}°"
        )

        # ------------------ HUD ------------------
        ang_bore_los = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u_bore, u_los))))))

        self.hud_text.set_text(
            f"t = {t:7.2f} s   Hill mode: {hill_mode}\n"
            f"r_H(D1) = [{depH[0]: .3f}, {depH[1]: .3f}, {depH[2]: .3f}] m\n"
            f"∠(boresight, LOS) = {ang_bore_los:6.3f} deg\n"
            f"Boresight err: {bore:7.3f} deg   LOS roll err: {losr:7.3f} deg\n"
            #f"Sep thrust active: {'YES' if Fmag>0 else 'no'}"
        )

        # main 3D bounds
        lim = float(self.cfg.axis_lim)
        self.ax3d.set_xlim(-lim, lim)
        self.ax3d.set_ylim(-lim, lim)
        self.ax3d.set_zlim(-lim, lim)
        _set_axes_equal_3d(self.ax3d)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ----------------------------- scenario patching -----------------------------

def configure_single_deputy_separation_and_drift(
    fleet,
    dv: Optional[float],
    flip_sep_dir: bool,
    keep_translation: bool,
):
    """
    Patch schedule + deputy controller so:
    - Separation pushes outward (optionally flips sign)
    - Deputy can coast after separation (translation control disabled)
    """
    sp = fleet.sched[0]

    # Separation sign fix:
    if flip_sep_dir:
        sp.s_car_B = -np.array(sp.s_car_B, dtype=float)

    # Override dv
    if dv is not None:
        Tpush = float(sp.t_end - sp.t_start)
        if Tpush <= 0:
            raise RuntimeError("Invalid push duration (t_end <= t_start).")
        sp.dv_cmd = float(dv)
        sp.F0 = float(fleet.deps[0].mass_kg * sp.dv_cmd * math.pi / (2.0 * Tpush))

    # Coast (disable translation shaping + keep attitude control)
    if not keep_translation:
        ctl = fleet.ctrl_deps[0]
        ctl.K_p[:] = 0.0
        ctl.D_v[:] = 0.0
        ctl.KI_p[:] = 0.0
        try:
            ctl.int_ep[:] = 0.0
        except Exception:
            pass

        # Also prevent any commanded force from allocation (keep torque mapping only)
        if hasattr(ctl, "B") and ctl.B is not None and ctl.B.size:
            Btau = ctl.B[:3, :].copy()
            ctl.B = np.vstack((Btau, np.zeros_like(Btau)))


# ----------------------------- run loop -----------------------------

def run_with_dashboard(
    fleet,
    scn,
    dash: D1HillDashboard,
    plot_skip: int,
    realtime: bool,
    save_npz: bool,
    save_video: Optional[str],
    fps: int,
):
    dt = float(fleet.dt)
    N = int(fleet.sim.T_end / dt)
    last_sep_end = float(fleet.t0_base + 0.30)

    writer = None
    if save_video:
        import shutil
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg or add it to PATH.")

        out_dir = os.path.dirname(os.path.abspath(save_video))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=int(fps), bitrate=8000, metadata={"title": "ESPA42 + D1 Hill live"})
        dash.fig.canvas.draw()
        plt.pause(0.001)
        writer.setup(dash.fig, save_video, dpi=dash.cfg.dpi)
        print(f"[video] Writing mp4: {save_video}")


    t0_wall = time.perf_counter()
    now_ns = 0

    for k in range(N):
        t = k * dt

        if realtime:
            elapsed = time.perf_counter() - t0_wall
            sleep_s = t - elapsed
            if sleep_s > 0:
                time.sleep(min(0.02, sleep_s))

        # sensors
        fleet.sns.ExecuteTaskList(now_ns)

        # separation
        fleet.sep.step(t)

        # carrier damping
        fleet.ctrl_car.step(
            t,
            fleet.nav_car.attOutMsg.read(),
            fleet.nav_car.transOutMsg.read(),
            R_d=np.eye(3),
            p_d_N=np.zeros(3),
        )

        # deputy control (D1)
        r_dep_N = np.array(fleet.nav_deps[0].transOutMsg.read().r_BN_N, float)

        # freeze translation reference after push end (if translation control exists)
        if (not fleet.p_ref_frozen[0]) and t >= last_sep_end:
            fleet.p_d_N_frozen[0] = r_dep_N.copy()
            fleet.p_ref_frozen[0] = True
        p_ref = fleet.p_d_N_frozen[0] if fleet.p_ref_frozen[0] else r_dep_N.copy()

        # pointing desired attitude
        if t >= last_sep_end:
            R_d, los_N = fleet._camera_Rd_cont(0, r_dep_N)
        else:
            R_d = scn.mrp_to_dcm(fleet.deps[0].sigma0_BN)
            los_N = None

        fleet.ctrl_deps[0].step(
            t,
            fleet.nav_deps[0].attOutMsg.read(),
            fleet.nav_deps[0].transOutMsg.read(),
            R_d=R_d,
            p_d_N=p_ref,
            los_N=los_N,
            cam_axis=fleet.CAM_AXIS,
        )

        # dashboard update
        if (k % plot_skip) == 0:
            dash.update(t, R_d)
            if writer:
                writer.grab_frame()

            if not plt.fignum_exists(dash.fig.number):
                print("[viz] Window closed - stopping run")
                break

        # dynamics
        fleet.dyn.ExecuteTaskList(now_ns)
        now_ns += int(macros.sec2nano(dt))

    if writer:
        writer.finish()
        print("[video] Done.")

    if save_npz:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"espa_xb6_D1_hill_live_{ts}"
        try:
            fleet._save_npz(base)
        except Exception as e:
            print("[npz] Could not save NPZ:", str(e))

    try:
        fleet._report(last_sep_end)
    except Exception:
        pass

    print("[DONE] D1 Hill live dashboard run complete.")


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, default="basilik_scenario_espa42_rafti_xb6_fleet_se3",
                    help="Scenario module name OR full .py path. Must define Fleet.")
    ap.add_argument("--skip", type=int, default=5, help="Plot update decimation in sim steps.")
    ap.add_argument("--trail", type=int, default=300, help="Trail length (points).")
    ap.add_argument("--axis", type=float, default=6.0, help="3D Hill axis half-range [m].")
    ap.add_argument("--borelen", type=float, default=0.55, help="Boresight vector length [m].")
    ap.add_argument("--thrustlen", type=float, default=0.85, help="Separation thrust vector max length [m].")
    ap.add_argument("--dv", type=float, default=0.03, help="Override RAFTI separation dv [m/s] for D1.")
    ap.add_argument("--no-flip-sep", action="store_true",
                    help="Do NOT flip separation direction (use if your push already drives D1 outward).")
    ap.add_argument("--keep-translation", action="store_true",
                    help="Keep deputy translation control ON (otherwise it coasts after separation).")
    ap.add_argument("--no-npz", action="store_true", help="Disable NPZ saving at end.")
    ap.add_argument("--realtime", action="store_true", help="Pace sim to ~1x real time.")
    ap.add_argument("--save", type=str, default="", help="Save mp4 animation (requires ffmpeg). Example: out.mp4")
    ap.add_argument("--fps", type=int, default=30, help="FPS for mp4 saving.")
    args = ap.parse_args()

    scn = import_scenario(args.scenario)
    if not hasattr(scn, "Fleet"):
        raise RuntimeError(f"Scenario '{args.scenario}' does not define Fleet.")

    # Build ONLY one deputy
    fleet = scn.Fleet(num_deps=1)

    # Patch separation + coasting
    configure_single_deputy_separation_and_drift(
        fleet=fleet,
        dv=float(args.dv) if args.dv is not None else None,
        flip_sep_dir=(not args.no_flip_sep),
        keep_translation=bool(args.keep_translation),
    )

    cfg = DashConfig(
        plot_skip=max(1, int(args.skip)),
        trail_len=max(10, int(args.trail)),
        axis_lim=float(args.axis),
        bore_len=float(args.borelen),
        thrust_len=float(args.thrustlen),
        dpi=140,
    )

    dash = D1HillDashboard(fleet, scn, cfg)

    save_video = args.save.strip() if args.save.strip() else None

    run_with_dashboard(
        fleet=fleet,
        scn=scn,
        dash=dash,
        plot_skip=cfg.plot_skip,
        realtime=bool(args.realtime),
        save_npz=(not args.no_npz),
        save_video=save_video,
        fps=int(args.fps),
    )

    if plt.fignum_exists(dash.fig.number):
        plt.show()


if __name__ == "__main__":
    main()
