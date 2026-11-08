#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
espa_xb6_8deps_live_dashboard.py

Live dark dashboard for:
- 3D carrier + ESPA ring + 8 deputies deploying
- Real-time plots:
    * H_SE3,d (carrier + deputies)
    * boresight error (deg)
    * LOS roll-about-LOS error (deg)
    * MRP component error (selected deputy)

Controls:
- Press keys 1..8 to select deputy for the MRP-error panel and 3D highlighting
- Close the window to stop the run early

Usage:
  python espa_xb6_8deps_live_dashboard.py
  python espa_xb6_8deps_live_dashboard.py --scenario basilik_scenario_espa42_rafti_xb6_fleet_se3
  python espa_xb6_8deps_live_dashboard.py --skip 5 --trail 300
  python espa_xb6_8deps_live_dashboard.py --save out.mp4 --fps 30

Notes:
- This script assumes your scenario module defines `Fleet` (as in the code you posted).
- For best interactivity, use an interactive Matplotlib backend (QtAgg/TkAgg).
"""

from __future__ import annotations

import argparse
import importlib
import math
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Basilisk time helper
from Basilisk.utilities import macros


# ----------------------------- small helpers -----------------------------

def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(la.norm(v))
    if n < eps:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return (v / n).astype(float)

def _set_dark_2d(ax):
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_color((1, 1, 1, 0.25))
    ax.tick_params(colors=(1, 1, 1, 0.7))
    ax.xaxis.label.set_color((1, 1, 1, 0.8))
    ax.yaxis.label.set_color((1, 1, 1, 0.8))
    ax.title.set_color((1, 1, 1, 0.9))
    ax.grid(True, color=(1, 1, 1, 0.08), linewidth=0.8)

def _set_dark_3d(ax3d):
    ax3d.set_facecolor("black")
    # Pane colors
    try:
        ax3d.xaxis.set_pane_color((0, 0, 0, 1))
        ax3d.yaxis.set_pane_color((0, 0, 0, 1))
        ax3d.zaxis.set_pane_color((0, 0, 0, 1))
    except Exception:
        pass
    # Grid styling (best effort)
    try:
        ax3d.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.08)
        ax3d.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.08)
        ax3d.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.08)
    except Exception:
        pass
    ax3d.tick_params(colors=(1, 1, 1, 0.6))
    ax3d.xaxis.label.set_color((1, 1, 1, 0.8))
    ax3d.yaxis.label.set_color((1, 1, 1, 0.8))
    ax3d.zaxis.label.set_color((1, 1, 1, 0.8))

def _set_axes_equal_3d(ax):
    """
    Make 3D axes have equal scale so cubes don't look like pancakes.
    """
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

    # Newer Matplotlib
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

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

def _cube_segments_N(vertsN: np.ndarray):
    segs = []
    for i, j in _CUBE_EDGES:
        segs.append([vertsN[i], vertsN[j]])
    return segs

def _cam_axis_B(cam_axis: str) -> np.ndarray:
    c = cam_axis.lower().strip()
    if c == "x":
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if c == "y":
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return np.array([0.0, 0.0, 1.0], dtype=float)


# --------------------------- live dashboard class ---------------------------

@dataclass
class DashConfig:
    plot_skip: int = 5            # update every N sim steps
    trail_len: int = 250          # number of points in trail
    axis_lim: float = 1.8         # +/- meters in 3D view
    bore_len: float = 0.35        # boresight vector length in 3D
    dpi: int = 140


class FleetLiveDashboard:
    def __init__(self, fleet, scenario_mod, cfg: DashConfig):
        self.fleet = fleet
        self.scn = scenario_mod
        self.cfg = cfg

        self.Ndeps = int(fleet.num_deps)
        self.selected = 0  # 0..N-1

        # History (decimated)
        self.t_hist: List[float] = []
        self.H_car: List[float] = []
        self.H_dep: List[List[float]] = [[] for _ in range(self.Ndeps)]
        self.bore_deg: List[List[float]] = [[] for _ in range(self.Ndeps)]
        self.los_roll_deg: List[List[float]] = [[] for _ in range(self.Ndeps)]
        self.sig_err: List[List[List[float]]] = [[[ ] for _ in range(3)] for _ in range(self.Ndeps)]  # [dep][k][time]
        self.r_dep_hist: List[List[np.ndarray]] = [[] for _ in range(self.Ndeps)]  # trail points

        # Desired sigma continuity helpers (optional but reduces MRP jumps)
        self._qd_prev: List[Optional[np.ndarray]] = [None] * self.Ndeps
        self._sd_prev: List[Optional[np.ndarray]] = [None] * self.Ndeps

        # Dark theme defaults
        plt.style.use("dark_background")
        plt.rcParams["figure.facecolor"] = "black"
        plt.rcParams["axes.facecolor"] = "black"

        self.fig = plt.figure(figsize=(18, 9), dpi=cfg.dpi, facecolor="black")
        gs = GridSpec(
            nrows=2, ncols=3, figure=self.fig,
            width_ratios=[1.25, 2.8, 1.25],
            height_ratios=[1.0, 1.0],
            wspace=0.08, hspace=0.18
        )

        # Left plots
        self.axH = self.fig.add_subplot(gs[0, 0])
        self.axSig = self.fig.add_subplot(gs[1, 0])
        _set_dark_2d(self.axH)
        _set_dark_2d(self.axSig)

        # Right plots
        self.axBore = self.fig.add_subplot(gs[0, 2])
        self.axLOS = self.fig.add_subplot(gs[1, 2])
        _set_dark_2d(self.axBore)
        _set_dark_2d(self.axLOS)

        # Center 3D
        self.ax3d = self.fig.add_subplot(gs[:, 1], projection="3d")
        _set_dark_3d(self.ax3d)

        self.ax3d.set_xlabel("X [m]")
        self.ax3d.set_ylabel("Y [m]")
        self.ax3d.set_zlabel("Z [m]")
        lim = float(cfg.axis_lim)
        self.ax3d.set_xlim(-lim, lim)
        self.ax3d.set_ylim(-lim, lim)
        self.ax3d.set_zlim(-lim, lim)
        self.ax3d.view_init(elev=22, azim=45)
        _set_axes_equal_3d(self.ax3d)

        self.fig.suptitle(
            "ESPA 42 Carrier  8 Deputies  Live Deployment",
            color=(1, 1, 1, 0.9)
        )

        # Color palette for deputies
        cmap = plt.cm.tab10
        self.dep_colors = [cmap(i % 10) for i in range(self.Ndeps)]

        # --------------- 2D plot artists ---------------
        # H plot (carrier + deputies)
        self.lineH_car, = self.axH.plot([], [], linewidth=2.4, alpha=0.95, label="Carrier")
        self.lineH_dep = []
        for i in range(self.Ndeps):
            lw = 2.0 if i == self.selected else 1.2
            a = 0.95 if i == self.selected else 0.55
            ln, = self.axH.plot([], [], linewidth=lw, alpha=a, color=self.dep_colors[i], label=f"D{i+1}")
            self.lineH_dep.append(ln)

        self.axH.set_title("H_SE3,d")
        self.axH.set_xlabel("t [s]")
        self.axH.set_ylabel("H")
        self.axH.set_xlim(0, float(self.fleet.sim.T_end))

        # Boresight plot
        self.lineBore = []
        for i in range(self.Ndeps):
            lw = 2.0 if i == self.selected else 1.2
            a = 0.95 if i == self.selected else 0.55
            ln, = self.axBore.plot([], [], linewidth=lw, alpha=a, color=self.dep_colors[i], label=f"D{i+1}")
            self.lineBore.append(ln)
        self.axBore.axhline(1.0, linestyle="--", linewidth=1.0, color=(1, 1, 1, 0.25))
        self.axBore.set_title("Boresight error [deg]")
        self.axBore.set_xlabel("t [s]")
        self.axBore.set_ylabel("deg")
        self.axBore.set_xlim(0, float(self.fleet.sim.T_end))
        self.axBore.set_ylim(0, 5)

        # LOS roll-about-LOS plot
        self.lineLOS = []
        for i in range(self.Ndeps):
            lw = 2.0 if i == self.selected else 1.2
            a = 0.95 if i == self.selected else 0.55
            ln, = self.axLOS.plot([], [], linewidth=lw, alpha=a, color=self.dep_colors[i], label=f"D{i+1}")
            self.lineLOS.append(ln)
        self.axLOS.axhline(2.0, linestyle="--", linewidth=1.0, color=(1, 1, 1, 0.25))
        self.axLOS.set_title("LOS roll error [deg]")
        self.axLOS.set_xlabel("t [s]")
        self.axLOS.set_ylabel("deg")
        self.axLOS.set_xlim(0, float(self.fleet.sim.T_end))
        self.axLOS.set_ylim(0, 10)

        # Sigma error (selected deputy only)
        self.lineSig = []
        sig_labels = [r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"]
        for k in range(3):
            ln, = self.axSig.plot([], [], linewidth=2.0, alpha=0.95, label=sig_labels[k])
            self.lineSig.append(ln)
        self.axSig.set_title("MRP component error (selected deputy)")
        self.axSig.set_xlabel("t [s]")
        self.axSig.set_ylabel(r"$\sigma - \sigma_d$")
        self.axSig.set_xlim(0, float(self.fleet.sim.T_end))
        self.axSig.set_ylim(-1.0, 1.0)
        self.axSig.legend(loc="upper right", framealpha=0.15, fontsize=9)

        # Keep legends light
        self.axH.legend(loc="upper right", ncol=4, framealpha=0.12, fontsize=8)
        self.axBore.legend(loc="upper right", ncol=4, framealpha=0.12, fontsize=8)
        self.axLOS.legend(loc="upper right", ncol=4, framealpha=0.12, fontsize=8)

        # --------------- 3D artists ---------------
        # Carrier hub cube
        self._hub_size = 0.45
        self._hub_vertsB = _cube_vertices_B(self._hub_size)
        self.hub_cube = Line3DCollection([], linewidths=2.0, colors=[(0.2, 1.0, 1.0, 0.9)])
        self.ax3d.add_collection3d(self.hub_cube)

        # Ring polyline + ports scatter
        self.ring_line, = self.ax3d.plot([], [], [], linewidth=2.2, alpha=0.8)
        self.ports_scatter = self.ax3d.scatter([], [], [], s=30)

        # Deputies: cube wireframes + trails + LOS + boresight
        self._dep_size = 0.18
        self._dep_vertsB = _cube_vertices_B(self._dep_size)

        self.dep_cubes: List[Line3DCollection] = []
        self.dep_trails = []
        self.dep_los_lines = []
        self.dep_bore_lines = []
        self.dep_labels = []

        for i in range(self.Ndeps):
            col = self.dep_colors[i]
            cube = Line3DCollection([], linewidths=1.4, colors=[col])
            self.ax3d.add_collection3d(cube)
            self.dep_cubes.append(cube)

            trail, = self.ax3d.plot([], [], [], linewidth=1.2, alpha=0.55, color=col)
            self.dep_trails.append(trail)

            los_ln, = self.ax3d.plot([], [], [], linestyle="--", linewidth=1.4, alpha=0.65, color=(1, 1, 1, 0.7))
            self.dep_los_lines.append(los_ln)

            bore_ln, = self.ax3d.plot([], [], [], linewidth=1.7, alpha=0.9, color=col)
            self.dep_bore_lines.append(bore_ln)

            txt = self.ax3d.text(0, 0, 0, f"D{i+1}", color=col, fontsize=9)
            self.dep_labels.append(txt)

        # Selected deputy body axes triad (single triad, moved each frame)
        self.sel_axes = Line3DCollection([], linewidths=3.0)
        self.ax3d.add_collection3d(self.sel_axes)

        # 3D HUD text
        self.hud_text = self.ax3d.text2D(
            0.02, 0.98, "",
            transform=self.ax3d.transAxes,
            color=(1, 1, 1, 0.9),
            fontsize=11,
            va="top"
        )

        # Key handler
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)

    def _on_key(self, event):
        if event.key is None:
            return
        k = event.key.strip()
        if k.isdigit():
            idx = int(k) - 1
            if 0 <= idx < self.Ndeps:
                self.selected = idx
                self._apply_selection_styling()

    def _apply_selection_styling(self):
        # Highlight selected on time series
        for i in range(self.Ndeps):
            is_sel = (i == self.selected)
            lw = 2.2 if is_sel else 1.2
            a = 0.95 if is_sel else 0.55
            self.lineH_dep[i].set_linewidth(lw)
            self.lineH_dep[i].set_alpha(a)
            self.lineBore[i].set_linewidth(lw)
            self.lineBore[i].set_alpha(a)
            self.lineLOS[i].set_linewidth(lw)
            self.lineLOS[i].set_alpha(a)

        # Highlight selected cube + boresight
        for i in range(self.Ndeps):
            is_sel = (i == self.selected)
            self.dep_cubes[i].set_linewidth(2.6 if is_sel else 1.4)
            self.dep_bore_lines[i].set_linewidth(3.0 if is_sel else 1.7)
            self.dep_bore_lines[i].set_alpha(1.0 if is_sel else 0.85)

    def _desired_sigma_continuous(self, dep_i: int, Rd: np.ndarray) -> np.ndarray:
        """
        Convert desired DCM->MRP with continuity tricks:
        - Keep quaternion sign consistent
        - Choose between MRP and shadow MRP to stay close to previous
        """
        # Use scenario's dcm_to_quat if available; else use its Fleet.dcm_to_mrp_safe
        if hasattr(self.scn, "dcm_to_quat"):
            q = np.array(self.scn.dcm_to_quat(Rd), dtype=float)
            q = q / la.norm(q)
            qprev = self._qd_prev[dep_i]
            if qprev is not None and float(np.dot(qprev, q)) < 0.0:
                q = -q

            q0 = float(q[0])
            qv = np.array(q[1:], dtype=float)
            eps = 1e-12
            if (1.0 + q0) > eps:
                s = qv / (1.0 + q0)
            else:
                s = -qv / max(eps, (1.0 - q0))

            # Shadow
            s2 = float(np.dot(s, s))
            if s2 > eps:
                s_shadow = -s / s2
            else:
                s_shadow = s

            sprev = self._sd_prev[dep_i]
            if sprev is None:
                sout = s
            else:
                # choose closer rep
                if la.norm(s - sprev) <= la.norm(s_shadow - sprev):
                    sout = s
                else:
                    sout = s_shadow

            self._qd_prev[dep_i] = q
            self._sd_prev[dep_i] = sout
            return sout

        # Fallback
        return np.array(self.scn.Fleet.dcm_to_mrp_safe(Rd), dtype=float)

    def update(self, t: float, Rd_list: List[np.ndarray]):
        """
        Called by the sim loop every plot_skip steps.
        """
        if not plt.fignum_exists(self.fig.number):
            return  # window closed

        # Pull latest energies/errors from controllers (already computed in the sim loop)
        Hc = float(self.fleet.ctrl_car.Hd_list[-1]) if self.fleet.ctrl_car.Hd_list else 0.0
        self.t_hist.append(float(t))
        self.H_car.append(Hc)

        for i in range(self.Ndeps):
            ctl = self.fleet.ctrl_deps[i]
            Hd = float(ctl.Hd_list[-1]) if ctl.Hd_list else 0.0
            bore = float(ctl.boresight_err_hist[-1]) if ctl.boresight_err_hist else np.nan
            losr = float(ctl.roll_about_los_err_hist[-1]) if ctl.roll_about_los_err_hist else np.nan
            self.H_dep[i].append(Hd)
            self.bore_deg[i].append(bore)
            self.los_roll_deg[i].append(losr)

        # Update MRP error histories (all deps), but only plot selected
        for i in range(self.Ndeps):
            sigma = np.array(self.fleet.nav_deps[i].attOutMsg.read().sigma_BN, dtype=float)
            sig_d = self._desired_sigma_continuous(i, Rd_list[i])
            err = sigma - sig_d
            for k in range(3):
                self.sig_err[i][k].append(float(err[k]))

        # ----------- 2D plots update -----------
        th = np.array(self.t_hist, dtype=float)

        # H
        self.lineH_car.set_data(th, np.array(self.H_car, dtype=float))
        for i in range(self.Ndeps):
            self.lineH_dep[i].set_data(th, np.array(self.H_dep[i], dtype=float))

        # Boresight / LOS roll
        for i in range(self.Ndeps):
            self.lineBore[i].set_data(th, np.array(self.bore_deg[i], dtype=float))
            self.lineLOS[i].set_data(th, np.array(self.los_roll_deg[i], dtype=float))

        # Sigma error (selected)
        s = self.selected
        for k in range(3):
            self.lineSig[k].set_data(th, np.array(self.sig_err[s][k], dtype=float))

        # Dynamic y-lims (cheap-ish, but keeps view usable)
        # H
        H_all = [self.H_car] + self.H_dep
        Hmax = max([max(h) if len(h) else 0.0 for h in H_all] + [1e-6])
        self.axH.set_ylim(0, 1.08 * Hmax)

        # Bore/LOS
        bore_max = np.nanmax(np.array([self.bore_deg[i] for i in range(self.Ndeps)], dtype=float))
        los_max  = np.nanmax(np.array([self.los_roll_deg[i] for i in range(self.Ndeps)], dtype=float))
        if np.isfinite(bore_max):
            self.axBore.set_ylim(0, max(5.0, 1.15 * bore_max))
        if np.isfinite(los_max):
            self.axLOS.set_ylim(0, max(10.0, 1.15 * los_max))

        # Sigma error
        se = np.array([self.sig_err[s][k] for k in range(3)], dtype=float)
        if se.size:
            m = float(np.nanmax(np.abs(se)))
            m = max(0.5, min(2.0, 1.15 * m))
            self.axSig.set_ylim(-m, m)

        # ----------- 3D update -----------
        # Carrier state
        car_sigma = np.array(self.fleet.nav_car.attOutMsg.read().sigma_BN, dtype=float)
        car_R_BN = self.scn.mrp_to_dcm(car_sigma)        # C_BN
        car_C_NB = car_R_BN.T
        car_rN = np.array(self.fleet.nav_car.transOutMsg.read().r_BN_N, dtype=float)

        # Carrier hub cube segments
        hub_vertsN = car_rN + (car_C_NB @ self._hub_vertsB.T).T
        self.hub_cube.set_segments(_cube_segments_N(hub_vertsN))

        # Ring line (in carrier body frame, then transform to N)
        ang = np.linspace(0, 2 * math.pi, 160)
        ringB = np.column_stack([
            self.fleet.espa_radius * np.cos(ang),
            self.fleet.espa_radius * np.sin(ang),
            np.zeros_like(ang)
        ])
        ringN = car_rN + (car_C_NB @ ringB.T).T
        self.ring_line.set_data(ringN[:, 0], ringN[:, 1])
        self.ring_line.set_3d_properties(ringN[:, 2])

        # Ports positions in N
        portsN = np.array([self.fleet._port_position_in_N(i) for i in range(self.Ndeps)], dtype=float)
        self.ports_scatter._offsets3d = (portsN[:, 0], portsN[:, 1], portsN[:, 2])

        # Deputies
        camB = _cam_axis_B(getattr(self.fleet, "CAM_AXIS", "z"))
        for i in range(self.Ndeps):
            dep_rN = np.array(self.fleet.nav_deps[i].transOutMsg.read().r_BN_N, dtype=float)
            dep_sigma = np.array(self.fleet.nav_deps[i].attOutMsg.read().sigma_BN, dtype=float)
            dep_R_BN = self.scn.mrp_to_dcm(dep_sigma)
            dep_C_NB = dep_R_BN.T

            # cube
            dep_vertsN = dep_rN + (dep_C_NB @ self._dep_vertsB.T).T
            self.dep_cubes[i].set_segments(_cube_segments_N(dep_vertsN))

            # trail
            self.r_dep_hist[i].append(dep_rN.copy())
            if len(self.r_dep_hist[i]) > self.cfg.trail_len:
                self.r_dep_hist[i] = self.r_dep_hist[i][-self.cfg.trail_len:]
            tr = np.array(self.r_dep_hist[i], dtype=float)
            self.dep_trails[i].set_data(tr[:, 0], tr[:, 1])
            self.dep_trails[i].set_3d_properties(tr[:, 2])

            # LOS to its port
            portN = portsN[i]
            self.dep_los_lines[i].set_data([dep_rN[0], portN[0]], [dep_rN[1], portN[1]])
            self.dep_los_lines[i].set_3d_properties([dep_rN[2], portN[2]])

            # boresight vector
            boreN = dep_C_NB @ camB
            tip = dep_rN + self.cfg.bore_len * _unit(boreN)
            self.dep_bore_lines[i].set_data([dep_rN[0], tip[0]], [dep_rN[1], tip[1]])
            self.dep_bore_lines[i].set_3d_properties([dep_rN[2], tip[2]])

            # labels (slightly offset)
            self.dep_labels[i].set_position((dep_rN[0] + 0.03, dep_rN[1] + 0.03))
            self.dep_labels[i].set_3d_properties(dep_rN[2] + 0.03)

        # Selected body axes triad
        s = self.selected
        sel_rN = np.array(self.fleet.nav_deps[s].transOutMsg.read().r_BN_N, dtype=float)
        sel_sigma = np.array(self.fleet.nav_deps[s].attOutMsg.read().sigma_BN, dtype=float)
        sel_R_BN = self.scn.mrp_to_dcm(sel_sigma)
        sel_C_NB = sel_R_BN.T

        L = 0.28
        xN = sel_rN + L * (sel_C_NB @ np.array([1.0, 0.0, 0.0]))
        yN = sel_rN + L * (sel_C_NB @ np.array([0.0, 1.0, 0.0]))
        zN = sel_rN + L * (sel_C_NB @ np.array([0.0, 0.0, 1.0]))
        tri_segs = [[sel_rN, xN], [sel_rN, yN], [sel_rN, zN]]
        tri_cols = [(1, 0.2, 0.2, 0.95), (0.2, 1, 0.2, 0.95), (0.2, 0.6, 1, 0.95)]
        self.sel_axes.set_segments(tri_segs)
        self.sel_axes.set_color(tri_cols)

        # HUD text
        bore_now = self.bore_deg[s][-1] if len(self.bore_deg[s]) else float("nan")
        los_now  = self.los_roll_deg[s][-1] if len(self.los_roll_deg[s]) else float("nan")
        self.hud_text.set_text(
            f"t = {t:7.2f} s\n"
            f"Selected: D{s+1}\n"
            f"Boresight err: {bore_now:6.3f} deg\n"
            f"LOS roll err:  {los_now:6.3f} deg"
        )

        # Enforce 3D bounds + equal aspect
        lim = float(self.cfg.axis_lim)
        self.ax3d.set_xlim(-lim, lim)
        self.ax3d.set_ylim(-lim, lim)
        self.ax3d.set_zlim(-lim, lim)
        _set_axes_equal_3d(self.ax3d)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ------------------------ custom run loop with live viz ------------------------

def run_fleet_with_dashboard(
    fleet,
    dashboard: FleetLiveDashboard,
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
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=int(fps), metadata={"title": "ESPA42 + 8 deputies live dashboard"})
            writer.setup(dashboard.fig, save_video, dpi=dashboard.cfg.dpi)
            print(f"[video] Writing mp4: {save_video}")
        except Exception as e:
            print("[video] Could not start FFMpegWriter. (Is ffmpeg installed?)")
            print("        Error:", str(e))
            writer = None

    # real-time pacing (optional)
    t0_wall = time.perf_counter()

    now_ns = 0
    for k in range(N):
        t = k * dt

        # Optional real-time pacing
        if realtime:
            target = t
            elapsed = time.perf_counter() - t0_wall
            sleep_s = target - elapsed
            if sleep_s > 0:
                time.sleep(min(0.02, sleep_s))

        # sensors (nav)
        fleet.sns.ExecuteTaskList(now_ns)

        # separation model
        fleet.sep.step(t)

        # carrier damping control
        fleet.ctrl_car.step(
            t,
            fleet.nav_car.attOutMsg.read(),
            fleet.nav_car.transOutMsg.read(),
            R_d=np.eye(3),
            p_d_N=np.zeros(3),
        )

        # deputies control
        Rd_list = [np.eye(3) for _ in range(fleet.num_deps)]
        for i in range(fleet.num_deps):
            r_dep_N = np.array(fleet.nav_deps[i].transOutMsg.read().r_BN_N, float)

            # freeze translation reference after push end
            if (not fleet.p_ref_frozen[i]) and t >= last_sep_end:
                fleet.p_d_N_frozen[i] = r_dep_N.copy()
                fleet.p_ref_frozen[i] = True
            p_ref = fleet.p_d_N_frozen[i] if fleet.p_ref_frozen[i] else r_dep_N.copy()

            # pointing desired attitude
            if t >= last_sep_end:
                R_d, los_N = fleet._camera_Rd_cont(i, r_dep_N)
            else:
                R_d = fleet.scn_mrp_to_dcm(fleet.deps[i].sigma0_BN) if hasattr(fleet, "scn_mrp_to_dcm") else fleet.scn.mrp_to_dcm(fleet.deps[i].sigma0_BN)
                los_N = None

            Rd_list[i] = R_d

            fleet.ctrl_deps[i].step(
                t,
                fleet.nav_deps[i].attOutMsg.read(),
                fleet.nav_deps[i].transOutMsg.read(),
                R_d=R_d,
                p_d_N=p_ref,
                los_N=los_N,
                cam_axis=fleet.CAM_AXIS,
            )

        # Live dashboard update (DECIMATED)
        if (k % plot_skip) == 0:
            dashboard.update(t, Rd_list)
            if writer:
                writer.grab_frame()

            # stop if window closed
            if not plt.fignum_exists(dashboard.fig.number):
                print("[viz] Window closed - stopping run")
                break

        # dynamics propagate
        fleet.dyn.ExecuteTaskList(now_ns)
        now_ns += int(macros.sec2nano(dt))

    if writer:
        writer.finish()
        print("[video] Done.")

    # Save NPZ (your scenario saver)
    if save_npz:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"espa_xb6_8deps_live_{ts}"
        fleet._save_npz(base)

    # Print report
    try:
        fleet._report(last_sep_end)
    except Exception:
        pass

    print("[DONE] Live dashboard run complete.")


# ------------------------------------ main ------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, default="basilik_scenario_espa42_rafti_xb6_fleet_se3",
                    help="Python module name that contains Fleet (no .py).")
    ap.add_argument("--skip", type=int, default=5, help="Plot update decimation in sim steps.")
    ap.add_argument("--trail", type=int, default=250, help="Trail length (points).")
    ap.add_argument("--axis", type=float, default=1.8, help="3D axis +/- range [m].")
    ap.add_argument("--borelen", type=float, default=0.35, help="Boresight vector length [m].")
    ap.add_argument("--realtime", action="store_true", help="Pace sim to ~1x real time.")
    ap.add_argument("--no-npz", action="store_true", help="Disable NPZ saving at end.")
    ap.add_argument("--save", type=str, default="", help="Save mp4 animation (requires ffmpeg). Example: out.mp4")
    ap.add_argument("--fps", type=int, default=30, help="FPS for mp4 saving.")
    args = ap.parse_args()

    # Import scenario module dynamically
    scn = importlib.import_module(args.scenario)
    if not hasattr(scn, "Fleet"):
        raise RuntimeError(f"Scenario module '{args.scenario}' does not define Fleet.")

    Fleet = scn.Fleet

    # Build fleet
    fleet = Fleet(num_deps=8)

    # Small convenience: stash scenario module + mrp_to_dcm in fleet for our loop fallback
    fleet.scn = scn
    fleet.scn_mrp_to_dcm = getattr(scn, "mrp_to_dcm", None)

    # Build dashboard
    cfg = DashConfig(
        plot_skip=max(1, int(args.skip)),
        trail_len=max(5, int(args.trail)),
        axis_lim=float(args.axis),
        bore_len=float(args.borelen),
        dpi=140
    )
    dash = FleetLiveDashboard(fleet, scn, cfg)
    dash._apply_selection_styling()

    save_video = args.save.strip() if args.save.strip() else None

    # Run
    run_fleet_with_dashboard(
        fleet=fleet,
        dashboard=dash,
        plot_skip=cfg.plot_skip,
        realtime=bool(args.realtime),
        save_npz=(not args.no_npz),
        save_video=save_video,
        fps=int(args.fps),
    )

    # Keep window alive if user wants to inspect final state
    if plt.fignum_exists(dash.fig.number):
        plt.show()


if __name__ == "__main__":
    main()

