#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cockpit_viz.py — visualization for the live FDIR mission cockpit.

Two backends share one metrics dashboard:
  * PyVista 3D scene (primary, nicer) + a matplotlib metrics window, OR
  * an all-matplotlib cockpit (3D scene + metrics in one window) as a robust fallback.

Spacecraft are PRIMITIVE shapes (boxes + camera cones), the asteroid is a primitive
lumpy ellipsoid (no CAD). The "deputy clicks a picture" inset renders the asteroid
from the imaging deputy's pose. Imported by live_mission_cockpit.py.
"""
from __future__ import annotations
import numpy as np
import numpy.linalg as la

STATE_COLOR = {"HEALTHY": "#2E7D32", "SUSPECT": "#F9A825", "FAULTED": "#C62828", "DEAD": "#555555"}


# ───────────────────────── primitive geometry ───────────────────────────────

def lumpy_ellipsoid(a=170., b=88., c=80., nu=40, nv=20, seed=1):
    """A primitive Apophis-like lumpy triaxial ellipsoid -> (verts, faces)."""
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 2*np.pi, nu); v = np.linspace(0, np.pi, nv)
    bump = 1.0 + 0.12*np.sin(3*u)[:, None] * np.cos(2*v)[None, :] + 0.06*rng.normal(0, 1, (nu, nv))
    X = a*bump*np.cos(u)[:, None]*np.sin(v)[None, :]
    Y = b*bump*np.sin(u)[:, None]*np.sin(v)[None, :]
    Z = c*bump*np.ones_like(u)[:, None]*np.cos(v)[None, :]
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    faces = []
    for i in range(nu-1):
        for j in range(nv-1):
            p = i*nv + j
            faces.append([p, p+nv, p+nv+1]); faces.append([p, p+nv+1, p+1])
    return verts, np.array(faces)


def camera_cone(apex, look, length=600., half_deg=20., n=14):
    """Line segments of a camera frustum cone from apex along look direction."""
    look = look/la.norm(look) if la.norm(look) > 0 else np.array([1., 0, 0])
    tmp = np.array([0, 0, 1.]) if abs(look[2]) < 0.9 else np.array([1., 0, 0])
    e1 = np.cross(look, tmp); e1 /= la.norm(e1); e2 = np.cross(look, e1)
    r = length*np.tan(np.radians(half_deg))
    ring = [apex + length*look + r*(np.cos(t)*e1 + np.sin(t)*e2) for t in np.linspace(0, 2*np.pi, n)]
    return apex, np.array(ring)


# ───────────────────────── matplotlib cockpit ───────────────────────────────

class MplCockpit:
    """All-matplotlib live cockpit: 3D scene + metric panels in one window."""
    def __init__(self, engine, asteroid, n_frames, show_3d=True):
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        self.plt = plt; self.engine = engine; self.ast = asteroid; self.show_3d = show_3d
        self.cov_hist = {"t": [], "onb": [], "gnd_t": [], "gnd": []}
        self.fig = plt.figure(figsize=(15, 8.5))
        gs = gridspec.GridSpec(3, 4, figure=self.fig, height_ratios=[1.4, 1, 1], width_ratios=[1.6, 1, 1, 1])
        self.ax3d = self.fig.add_subplot(gs[:, 0], projection="3d") if show_3d else None
        self.ax_cov = self.fig.add_subplot(gs[0, 1:3])
        self.ax_cmp = self.fig.add_subplot(gs[0, 3])
        self.ax_cam = self.fig.add_subplot(gs[1, 1])
        self.ax_hl = self.fig.add_subplot(gs[1, 2:])
        self.ax_log = self.fig.add_subplot(gs[2, 1:])
        self.fig.suptitle("ARISE / Apophis — live onboard FDIR autonomy cockpit", fontsize=14, weight="bold")
        self.fig.tight_layout(rect=(0, 0, 1, 0.96))

    def _draw_scene(self, st):
        ax = self.ax3d; ax.clear()
        v, f = self.ast
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, color="#9c8b7a", alpha=0.9,
                        edgecolor="none", shade=True)
        live = st["ca_all"]
        for d in range(self.engine.N):
            p = live[d]; col = STATE_COLOR[st["health"][d]]
            ax.scatter(*p, s=70, c=col, marker="s", depthshade=False, edgecolor="k", linewidth=0.5)
            ax.text(p[0], p[1], p[2]+120, f"D{d}", fontsize=7, color=col)
            if d in st["active"]:
                apex, ring = camera_cone(p, -p, length=la.norm(p)*0.6)
                for q in ring[::3]:
                    ax.plot([apex[0], q[0]], [apex[1], q[1]], [apex[2], q[2]], c=col, lw=0.4, alpha=0.5)
        if st["capture"]:
            d = st["capture"]["deputy"]; p = live[d]
            ax.plot([p[0], 0], [p[1], 0], [p[2], 0], c="#1565C0", lw=2.0)
        R = max(np.abs(np.concatenate([live[d] for d in range(self.engine.N)])).max()*1.1, 1200)
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.set_title(f"t-to-CA {st['t_go']/60:.1f} min   active {len(st['active'])}/{self.engine.N}", fontsize=10)

    def _draw_panels(self, st):
        # coverage
        ax = self.ax_cov; ax.clear()
        if self.cov_hist["t"]:
            ax.plot(self.cov_hist["t"], self.cov_hist["onb"], "-o", ms=3, color="#1565C0", label="onboard (surrogate)")
        if self.cov_hist["gnd_t"]:
            ax.plot(self.cov_hist["gnd_t"], self.cov_hist["gnd"], "s", ms=7, color="#C62828", label="ground (hi-fi)")
        ax.set_ylim(0, 1); ax.set_ylabel("CA stereo coverage"); ax.set_xlabel("frame")
        ax.set_title("predicted closest-approach coverage", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=.3)
        # compute comparison
        ax = self.ax_cmp; ax.clear()
        if self.engine.compute:
            c = self.engine.compute[-1]
            ax.bar(["onboard\nscore", "ground\nscore"], [c[1], c[2]], color=["#1565C0", "#C62828"])
            ax.set_yscale("log"); ax.set_ylabel("ms (log)")
            ax.set_title(f"compute  x{c[2]/max(c[1],1e-3):.0f} faster", fontsize=9)
        else:
            ax.text(0.5, 0.5, "compute:\n(awaiting event)", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.axis("off")
        # camera capture inset
        ax = self.ax_cam; ax.clear(); ax.axis("off")
        if st["capture"]:
            img = render_capture_mpl(self.ast, st["live"][st["capture"]["deputy"]])
            ax.imshow(img, origin="lower"); ax.set_title(f"D{st['capture']['deputy']} captured image", fontsize=8, color="#1565C0")
        else:
            ax.set_title("peer camera (idle)", fontsize=8, color="0.5")
        # health grid
        ax = self.ax_hl; ax.clear(); ax.axis("off"); ax.set_title("fleet health (FDI)", fontsize=9)
        for d in range(self.engine.N):
            x, y = d % 4, 1 - d // 4
            ax.add_patch(self.plt.Rectangle((x*0.25, y*0.4), 0.22, 0.34, color=STATE_COLOR[st["health"][d]], transform=ax.transAxes))
            ax.text(x*0.25+0.11, y*0.4+0.17, f"D{d}\n{st['health'][d][:4]}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="white", weight="bold")
        # event log
        ax = self.ax_log; ax.clear(); ax.axis("off"); ax.set_title("autonomy event log", fontsize=9, loc="left")
        tail = self.engine.log[-6:]
        for i, (fr, msg) in enumerate(tail):
            ax.text(0.01, 0.95 - i*0.16, f"f{fr}: {msg}", transform=ax.transAxes, fontsize=7.5, family="monospace")

    def update(self, st):
        self.cov_hist["t"].append(st["frame"]); self.cov_hist["onb"].append(st["cov_onb"])
        if st["event"] and st["cov_gnd"] is not None:
            self.cov_hist["gnd_t"].append(st["frame"]); self.cov_hist["gnd"].append(st["cov_gnd"])
        if self.show_3d:
            self._draw_scene(st)
        self._draw_panels(st)


def render_capture_mpl(asteroid, deputy_pos, size=120):
    """Crude painter's-render of the primitive asteroid from the deputy pose (for the inset)."""
    v, f = asteroid
    look = -deputy_pos / la.norm(deputy_pos)
    tmp = np.array([0, 0, 1.]) if abs(look[2]) < 0.9 else np.array([1., 0, 0])
    e1 = np.cross(look, tmp); e1 /= la.norm(e1); e2 = np.cross(look, e1)
    img = np.zeros((size, size))
    proj = np.stack([(v - deputy_pos) @ e1, (v - deputy_pos) @ e2], 1)
    depth = (v - deputy_pos) @ look
    sc = size / (2.2 * np.abs(proj).max() + 1e-9)
    px = (proj * sc + size/2).astype(int)
    order = np.argsort(-depth)
    for tri in f[np.argsort(-depth[f].mean(1))]:
        pts = px[tri]
        cx, cy = pts.mean(0).astype(int)
        shade = max(0.15, float(look @ (np.cross(v[tri[1]]-v[tri[0]], v[tri[2]]-v[tri[0]]))))  # crude
        rr = 4
        x0, x1 = max(0, cx-rr), min(size, cx+rr); y0, y1 = max(0, cy-rr), min(size, cy+rr)
        img[y0:y1, x0:x1] = 0.4 + 0.6*np.tanh(abs(shade)*1e-4)
    return img


# ───────────────────────── PyVista 3D scene (primary) ───────────────────────

class PvScene:
    """PyVista 3D scene: primitive lumpy-ellipsoid asteroid + deputy boxes + camera cones,
    with an offscreen render of the asteroid from the imaging deputy's pose (the capture)."""
    def __init__(self, engine, asteroid, sun_dir=(0.9, 0.37, 0.16)):
        import pyvista as pv
        self.pv = pv; self.engine = engine
        v, f = asteroid
        faces = np.hstack([np.full((len(f), 1), 3, int), f]).ravel()
        self.ast_mesh = pv.PolyData(v, faces)
        self.p = pv.Plotter(window_size=(1000, 850), title="ARISE / Apophis — 3D scene")
        self.p.set_background("#05060a")
        self.p.add_mesh(self.ast_mesh, color="#9c8b7a", smooth_shading=True, specular=0.2)
        try:
            self.p.add_light(pv.Light(position=tuple(6000*np.array(sun_dir)), light_type="scene light"))
        except Exception:
            pass
        self.box = {}
        for d in range(engine.N):
            b = pv.Cube(x_length=120, y_length=80, z_length=80)
            self.box[d] = self.p.add_mesh(b, color="#2E7D32", name=f"dep{d}")
        self.cap_p = pv.Plotter(off_screen=True, window_size=(220, 220))
        self.cap_p.add_mesh(self.ast_mesh, color="#caa", smooth_shading=True)
        self.p.add_text("", name="hud", position="upper_left", font_size=11, color="white")
        self.p.camera_position = [(0, -9000, 4000), (0, 0, 0), (0, 0, 1)]
        self.p.show(interactive_update=True, auto_close=False)

    def _col(self, state):
        return {"HEALTHY": "#2E7D32", "SUSPECT": "#F9A825", "FAULTED": "#C62828", "DEAD": "#555555"}[state]

    def update(self, st):
        cones = []
        for d in range(self.engine.N):
            p = st["ca_all"][d]
            self.box[d].SetPosition(float(p[0]), float(p[1]), float(p[2]))
            self.box[d].GetProperty().SetColor(*self.pv.Color(self._col(st["health"][d])).float_rgb)
            if d in st["active"]:
                apex, ring = camera_cone(p, -p, length=la.norm(p)*0.5, n=10)
                for q in ring:
                    cones.append([apex, q])
        if cones:
            lines = np.array(cones).reshape(-1, 3)
            self.p.add_lines(lines, color="#4488cc", width=1, name="cones")
        if st["capture"]:
            d = st["capture"]["deputy"]; p = st["ca_all"][d]
            self.p.add_lines(np.array([p, np.zeros(3)]), color="#1565C0", width=4, name="beam")
        elif "beam" in getattr(self.p, "actors", {}):
            try: self.p.remove_actor("beam")
            except Exception: pass
        self.p.actors["hud"].SetInput(
            f"t-to-CA {st['t_go']/60:5.1f} min   active {len(st['active'])}/{self.engine.N}\n"
            f"coverage onboard {st['cov_onb']:.2f}" + (f"  ground {st['cov_gnd']:.2f}" if st["cov_gnd"] else "")
            + (f"\nEVENT: {st['event']}" if st["event"] else ""))
        self.p.update()

    def capture_image(self, st):
        """Offscreen render of the asteroid from the imaging deputy's CA pose."""
        if not st["capture"]:
            return None
        p = st["ca_all"][st["capture"]["deputy"]]
        self.cap_p.camera_position = [tuple(p), (0, 0, 0), (0, 0, 1)]
        return self.cap_p.screenshot(return_img=True)

    def alive(self):
        try:
            return bool(self.p.render_window) and not self.p._closed
        except Exception:
            return True
