# -*- coding: utf-8 -*-
"""
rafti.py - Orbit Fab RAFTI docking model on the ESPA CYLINDRICAL SHELL.

Rewritten to be geometrically correct (see geometry.py):

  * the ESPA is a finite cylinder; the 8 RAFTI ports sit on its LATERAL surface,
    each facing radially OUTWARD;
  * a deputy docks by bringing its docking FACE onto the port from OUTSIDE, along
    the port's outward normal, inside a shallow glideslope CONE, decelerating to a
    soft contact - it may never pass through the shell wall;
  * the target for the deputy CENTRE is offset outboard by (standoff + face_offset)
    so the docking face - not the body centre - meets the surface;
  * each deputy has a STAGING point on its port axis; guidance/RL first acquires
    the corridor (staging) then runs in radially (glideslope), which is what makes
    the approach trajectories physically correct;
  * the port frame (dock axis -> local +x) is retained so one policy handles all 8.

Public surface is compatible with the environments: assign(), capture_pt, dir_B,
R_port, to_port_frame(), action_to_lvlh(), capture_status(), plus new cylinder
collision helpers body_collision()/swarm_collision() and staging_pt.
"""
from __future__ import annotations
import numpy as np
from envs.geometry import ESPACylinder, DeputyCuboid, deputy_deputy_collision
from dynamics.attitude import pointing_error_deg, mrp_error_to_point

try:
    from scipy.optimize import linear_sum_assignment as _lsa
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _greedy_assignment(cost):
    C = cost.astype(float).copy(); n, m = C.shape
    rows = np.full(n, -1, int); used = np.zeros(m, bool)
    for r in np.argsort(C.min(axis=1)):
        c = int(np.argmin(np.where(used, np.inf, C[r]))); rows[r] = c; used[c] = True
    return np.arange(n), rows


class RaftiPorts:
    def __init__(self, ports_r, ports_dir, radius, rafti_cfg, deputy_dims=(0.34, 0.20, 0.20),
                 height=0.80, attitude_dcm=None):
        self.cfg = dict(rafti_cfg)
        r = np.asarray(ports_r, float); u = np.asarray(ports_dir, float)
        if attitude_dcm is not None:                       # ESPA body -> LVLH
            A = np.asarray(attitude_dcm, float); r = r @ A.T; u = u @ A.T
        self.cyl = ESPACylinder(r, u, radius=radius, height=height)
        self.dep = DeputyCuboid(deputy_dims)
        self.P = self.cyl.P
        self.radius = self.cyl.radius
        self.dir_B = self.cyl.ports_dir.copy()             # outward dock axes
        self.standoff = self.cfg["standoff_m"]
        self.r_stage = self.cfg.get("stage_range_m", 6.0)
        off = self.standoff + self.dep.face_offset         # centre offset from surface
        # deputy-CENTRE capture points and staging points (outboard on each port axis)
        self.capture_pt = np.stack([self.cyl.capture_point(i, off) for i in range(self.P)])
        self.staging_pt = np.stack([self.cyl.staging_point(i, self.r_stage + off)
                                    for i in range(self.P)])
        self.R_port = np.stack([self._port_frame(u) for u in self.dir_B])

    # port frame: columns [x=outward dock axis, y, z]
    @staticmethod
    def _port_frame(axis):
        x = axis / (np.linalg.norm(axis) + 1e-15)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(x @ ref) > 0.95:
            ref = np.array([0.0, 1.0, 0.0])
        y = np.cross(ref, x); y /= np.linalg.norm(y) + 1e-15
        return np.column_stack([x, y, np.cross(x, y)])

    # --- assignment (deputies -> distinct ports, min total range to staging) ---
    def assign(self, positions_lvlh):
        pos = np.atleast_2d(positions_lvlh)
        cost = np.linalg.norm(pos[:, None, :] - self.staging_pt[None, :, :], axis=2)
        r, c = (_lsa(cost) if _HAVE_SCIPY else _greedy_assignment(cost))
        out = np.empty(pos.shape[0], int); out[r] = c
        return out

    # --- canonicalisation ------------------------------------------------------
    def to_port_frame(self, pos_lvlh, vel_lvlh, i):
        R = self.R_port[i]
        return R.T @ (np.asarray(pos_lvlh, float) - self.capture_pt[i]), R.T @ np.asarray(vel_lvlh, float)

    def action_to_lvlh(self, a_port, i):
        return self.R_port[i] @ np.asarray(a_port, float)

    def staging_in_port_frame(self, i):
        return self.R_port[i].T @ (self.staging_pt[i] - self.capture_pt[i])   # ~ +x axis

    # --- soft-capture acceptance (port frame; +x outboard, closing is vx<0) ----
    def capture_status(self, pos_lvlh, vel_lvlh, i, sigma=None, omega=None):
        c = self.cfg
        p, v = self.to_port_frame(pos_lvlh, vel_lvlh, i)
        axial, lateral = p[0], float(np.hypot(p[1], p[2]))
        dist = float(np.linalg.norm(p)); v_ax = v[0]; v_lat = float(np.hypot(v[1], v[2]))
        cone_ang = np.degrees(np.arctan2(lateral, max(axial, 1e-9)))
        # glideslope cone with a capture ball at the apex: allowed lateral grows with
        # axial range, but never below the capture tolerance (so being AT the port,
        # where axial~0, is inside the corridor). Small inboard overshoot is allowed.
        tan_c = np.tan(np.radians(c["cone_half_angle_deg"]))
        in_cone = (axial >= -c["capture_pos_tol_m"]) and \
                  (lateral <= max(axial, 0.0) * tan_c + c["lateral_tol_m"])
        speed = np.linalg.norm(v) + 1e-12
        align_ang = np.degrees(np.arccos(np.clip(-v[0] / speed, -1, 1)))
        # a deputy that arrives and (nearly) stops at the port is a valid soft dock;
        # the axis-alignment test only applies once it is actually moving.
        aligned = (speed < 0.01) or (align_ang <= c["axis_align_tol_deg"])
        # --- 6-DOF: dock FACE must point into the port (-outward normal), low rate ---
        att_err = 0.0; rate = 0.0; att_ok = True
        if sigma is not None:
            att_err = pointing_error_deg(sigma, -self.dir_B[i])   # +x_body vs inward
            rate = float(np.linalg.norm(omega)) if omega is not None else 0.0
            att_ok = (att_err <= c.get("att_align_tol_deg", 10.0)) and \
                     (rate <= c.get("omega_capture_max", 0.02))
        captured = (dist <= c["capture_pos_tol_m"] and lateral <= c["lateral_tol_m"]
                    and (-c["v_capture_max_mps"] <= v_ax <= 0.005)
                    and v_lat <= c["v_lateral_max_mps"] and in_cone and aligned and att_ok)
        return dict(axial=axial, lateral=lateral, dist=dist, v_axial=v_ax, v_lateral=v_lat,
                    cone_ang=cone_ang, in_cone=in_cone, align_ang=align_ang,
                    att_err=att_err, rate=rate, att_ok=att_ok, captured=bool(captured))

    # --- collision: deputy cuboid (facing its port) vs the shell ---------------
    def body_collision(self, center_lvlh, i):
        R_bl = self.dep.attitude_pointing(-self.dir_B[i])          # face -> inward
        corners = self.dep.corners_lvlh(center_lvlh, R_bl)
        return self.cyl.body_hits_shell(corners)

    def swarm_collision(self, centers_lvlh):
        hit, pair = deputy_deputy_collision(centers_lvlh, self.dep.half_diag)
        return hit, pair

    # legacy name kept for the environments (now shell-aware, cuboid-accurate)
    def keepout_violation(self, pos_lvlh, i=None):
        if i is None:      # conservative sphere broad-phase when port unknown
            return bool(self.cyl.sphere_hits_shell(pos_lvlh, self.dep.half_diag)[0])
        return self.body_collision(pos_lvlh, i)


if __name__ == "__main__":
    import config_redock as C
    r, u, rad = C.load_espa_ports()
    ports = RaftiPorts(r, u, rad, C.RAFTI)
    print(f"{ports.P} ports on cylinder r={ports.radius:.3f} m; staging {ports.r_stage} m out; scipy={_HAVE_SCIPY}")
    # deputies staged on each port axis -> unique assignment
    a = ports.assign(ports.staging_pt); print("assign:", a, "unique:", len(set(a)) == ports.P)
    # a deputy at its capture point, closing 3 cm/s along -axis -> captured, no collision
    i = 0
    p = ports.capture_pt[i].copy(); v = -0.03 * ports.dir_B[i]
    st = ports.capture_status(p, v, i)
    print("docked pose captured:", st["captured"], "| body collision:", ports.body_collision(p, i))
    assert st["captured"] and not ports.body_collision(p, i)
    # driving to centre must collide
    assert ports.body_collision(np.zeros(3), i)
    print("rafti (cylinder) self-test OK")
