# -*- coding: utf-8 -*-
"""
geometry.py - physical bodies for a geometrically-correct re-dock.

ESPACylinder : the ESPA adapter as a finite CYLINDRICAL SHELL.  Its axis is the
               normal of the port plane (SVD of the port positions), radius = the
               port ring radius, with the 8 RAFTI ports on the LATERAL SURFACE,
               each facing radially OUTWARD.  Provides the surface point, outward
               normal, a staging point and capture point per port, and a
               cuboid-vs-shell collision test (a deputy may NOT pass through the
               wall into the drum - it may only touch its own port from outside).

DeputyCuboid : the 6U bus as a cuboid with a designated docking FACE (+x_body,
               where the RAFTI adaptor sits).  Provides its circumscribed radius
               (for fast broad-phase collision) and, given a body attitude,
               whether it penetrates the shell or another deputy.

Everything is in the ESPA-centred LVLH frame, SI units.  This module is the single
source of truth for the docking geometry used by rafti.py and the environments.
"""
from __future__ import annotations
import numpy as np


def _plane_normal(points: np.ndarray) -> np.ndarray:
    """Best-fit plane normal of a set of points (smallest-singular-vector)."""
    P = points - points.mean(0)
    _, _, Vt = np.linalg.svd(P)
    n = Vt[-1]
    return n / (np.linalg.norm(n) + 1e-15)


class ESPACylinder:
    def __init__(self, ports_r, ports_dir, radius=None, height=0.80,
                 wall_clear=0.02):
        self.ports_r = np.asarray(ports_r, float)          # (P,3) on the surface
        self.ports_dir = np.asarray(ports_dir, float)
        self.ports_dir /= np.linalg.norm(self.ports_dir, axis=1, keepdims=True) + 1e-15
        self.P = self.ports_r.shape[0]
        # cylinder axis = normal of the port plane (data-driven)
        self.axis = _plane_normal(self.ports_r)
        # radius = mean radial distance of ports from the axis
        self.radius = float(radius if radius is not None else
                            np.mean(np.linalg.norm(self._reject_axis(self.ports_r), axis=1)))
        self.half_h = 0.5 * float(height)
        self.wall_clear = float(wall_clear)
        # orthonormal in-plane basis (e1,e2) spanning the ring plane
        a = self.axis
        e1 = self._reject_axis(np.array([[1.0, 0, 0]]))[0]
        if np.linalg.norm(e1) < 1e-6:
            e1 = self._reject_axis(np.array([[0, 1.0, 0]]))[0]
        e1 /= np.linalg.norm(e1) + 1e-15
        self.e1 = e1
        self.e2 = np.cross(a, e1)

    # --- helpers ---------------------------------------------------------------
    def _reject_axis(self, v):
        v = np.atleast_2d(v)
        return v - np.outer(v @ self.axis, self.axis)

    def radial_dist(self, pts):
        """Perpendicular distance of point(s) from the cylinder axis."""
        return np.linalg.norm(self._reject_axis(pts), axis=-1)

    def axial_coord(self, pts):
        return np.atleast_2d(pts) @ self.axis

    # --- per-port frame + corridor points -------------------------------------
    def port_normal(self, i):
        return self.ports_dir[i]

    def surface_point(self, i):
        return self.ports_r[i]

    def capture_point(self, i, standoff):
        return self.ports_r[i] + standoff * self.ports_dir[i]

    def staging_point(self, i, r_stage):
        return self.ports_r[i] + r_stage * self.ports_dir[i]

    # --- collision -------------------------------------------------------------
    def inside_shell(self, pts, margin=0.0):
        """True where a point is INSIDE the solid drum (penetrated the wall).

        Inside == radial distance < radius - margin AND within the axial height.
        A docking deputy stops OUTSIDE (radial dist >= radius+standoff), so this
        only fires if it has driven through the shell.
        """
        pts = np.atleast_2d(pts)
        rho = self.radial_dist(pts)
        z = np.abs(self.axial_coord(pts))
        return (rho < (self.radius - margin)) & (z < self.half_h)

    def sphere_hits_shell(self, center, radius):
        """Fast broad-phase: does a sphere of `radius` at `center` overlap the drum?"""
        c = np.atleast_2d(center)
        rho = self.radial_dist(c); z = np.abs(self.axial_coord(c))
        return ((rho - radius) < (self.radius - self.wall_clear)) & (z < (self.half_h + radius))

    def body_hits_shell(self, corners):
        """Accurate: does any cuboid CORNER lie inside the solid drum?

        With the deputy oriented to face its port, the docking face touches the
        surface while the body stays radially OUTBOARD, so a proper dock does not
        trip this - only genuine wall penetration does.
        """
        return bool(np.any(self.inside_shell(np.atleast_2d(corners), margin=self.wall_clear)))

    # --- surface mesh (for plotting) ------------------------------------------
    def surface_mesh(self, n_theta=48, n_z=2):
        th = np.linspace(0, 2 * np.pi, n_theta)
        zz = np.linspace(-self.half_h, self.half_h, n_z)
        TH, ZZ = np.meshgrid(th, zz)
        X = (self.radius * np.cos(TH)[..., None] * self.e1
             + self.radius * np.sin(TH)[..., None] * self.e2
             + ZZ[..., None] * self.axis)
        return X[..., 0], X[..., 1], X[..., 2]


class DeputyCuboid:
    """6U-class bus cuboid.  dims = full (x,y,z) body extents [m]; +x_body is the
    docking face carrying the RAFTI adaptor."""
    def __init__(self, dims=(0.34, 0.20, 0.20)):
        self.dims = np.asarray(dims, float)
        self.half = 0.5 * self.dims
        self.half_diag = float(np.linalg.norm(self.half))     # circumscribed radius
        self.face_offset = self.half[0]                        # dock face along +x_body

    @staticmethod
    def attitude_pointing(dock_axis_lvlh):
        """Body->LVLH DCM that points +x_body along `dock_axis_lvlh` (the deputy
        faces its port).  Used to slave attitude to the approach for the docking
        face; full 6-DOF attitude RL is a documented extension."""
        x = np.asarray(dock_axis_lvlh, float)
        x /= np.linalg.norm(x) + 1e-15
        up = np.array([0, 0, 1.0])
        if abs(x @ up) > 0.95:
            up = np.array([0, 1.0, 0])
        y = np.cross(up, x); y /= np.linalg.norm(y) + 1e-15
        z = np.cross(x, y)
        return np.column_stack([x, y, z])

    def corners_lvlh(self, center, R_bl):
        """8 cuboid corners in LVLH given body->LVLH DCM R_bl."""
        s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return np.asarray(center) + (s * self.half) @ R_bl.T


def deputy_deputy_collision(centers, half_diag):
    """Return True if any two deputy circumscribed spheres overlap."""
    c = np.asarray(centers)
    n = c.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(c[i] - c[j]) < 2 * half_diag:
                return True, (i, j)
    return False, None


if __name__ == "__main__":
    import config_redock as C
    r, u, rad = C.load_espa_ports()
    cyl = ESPACylinder(r, u)
    dep = DeputyCuboid()
    print(f"ESPA cylinder: axis={np.round(cyl.axis,3)} radius={cyl.radius:.4f} m "
          f"height={2*cyl.half_h:.2f} m, {cyl.P} lateral ports")
    print(f"deputy cuboid: dims={dep.dims} half_diag={dep.half_diag:.3f} m")
    # deputy faces its port (attitude slaved to the inward normal)
    def body_at(i, standoff, extra):
        # deputy CENTER sits face_offset+standoff outboard so the DOCK FACE meets the port
        center = cyl.capture_point(i, standoff + dep.face_offset) + extra * cyl.port_normal(i)
        R_bl = dep.attitude_pointing(-cyl.port_normal(i))   # +x_body -> inward
        return dep.corners_lvlh(center, R_bl)
    print("5 m outboard of port 0 hits shell?", cyl.body_hits_shell(body_at(0, 0.12, 5.0)))
    print("at ESPA centre hits shell?       ", cyl.body_hits_shell(dep.corners_lvlh(np.zeros(3), np.eye(3))))
    print("docked at capture point collide? ", cyl.body_hits_shell(body_at(0, 0.12, 0.0)))
    assert cyl.body_hits_shell(dep.corners_lvlh(np.zeros(3), np.eye(3)))     # centre = collision
    assert not cyl.body_hits_shell(body_at(0, 0.12, 5.0))                    # outboard = clear
    assert not cyl.body_hits_shell(body_at(0, 0.12, 0.0))                    # docked = clear
    print("geometry self-test OK")
