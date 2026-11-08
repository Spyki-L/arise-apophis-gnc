# -*- coding: utf-8 -*-
"""
docking_sensor.py - close-range relative-navigation metrology for terminal docking.

A combined vision-based relative navigation (VBN) camera + flash LIDAR, boresighted
along the deputy's docking axis at the RAFTI port.  Range-scheduled fidelity:

    range > acquisition   : no fix (target beyond acquisition / out of FOV)
    acquisition .. att_rng : bearing (camera, CIS120 plate scale) + range (LIDAR)
                             -> direct RELATIVE POSITION (port frame)
    range < att_rng        : the port pattern is resolved -> also a RELATIVE ATTITUDE
                             (dock-face-to-port pointing error) measurement

This is what a docking sensor adds over a star tracker: the star tracker knows the
deputy's INERTIAL attitude; the docking VBN measures the deputy's pose RELATIVE to the
PORT directly, which is exactly what the terminal control needs.  Camera parameters
come from the real CIS120 detector (nominal_v2) when reachable.
"""
from __future__ import annotations
import numpy as np
from flight import hardware_bridge as HWB


class DockingVBN:
    def __init__(self, rng, boresight_fov_deg=15.0, acquisition_m=200.0, att_range_m=6.0,
                 centroid_noise_px=0.2, lidar_range_frac=0.005, lidar_floor_m=0.01,
                 att_sigma_floor_deg=0.3, att_range_frac_deg_per_m=0.8, dropout_prob=0.01):
        self.rng = rng
        # camera: CIS120 (10 um pixel) + 50 mm docking-cam optics -> plate scale
        pix = HWB.HW.CIS120Detector.pixel_m if HWB.AVAILABLE else 10.0e-6
        focal = 0.050
        self.plate_scale = pix / focal                        # rad / pixel
        self.bearing_sigma = self.plate_scale * centroid_noise_px
        self.acq = acquisition_m; self.att_rng = att_range_m
        self.fov = np.radians(boresight_fov_deg)
        self.r_frac = lidar_range_frac; self.r_floor = lidar_floor_m
        self.att_floor = np.radians(att_sigma_floor_deg)
        self.att_slope = np.radians(att_range_frac_deg_per_m)
        self.drop = dropout_prob

    def measure(self, p_port, att_err_to_port_rad, rng=None):
        """p_port: deputy position in the PORT frame (relative to the capture point, +x
        outboard toward the deputy).  att_err_to_port_rad: true pointing error [rad].
        Returns a dict or None (out of range / out of FOV / dropout)."""
        r = rng or self.rng
        rho = float(np.linalg.norm(p_port))
        if rho > self.acq or r.random() < self.drop:
            return None
        # FOV: boresight is +x in the port frame (looking outboard at the deputy)
        off = np.arctan2(np.hypot(p_port[1], p_port[2]), max(abs(p_port[0]), 1e-6))
        if off > self.fov:
            return None
        # position: bearing (lateral) + LIDAR (radial), noise scales with range
        lat_sigma = self.bearing_sigma * max(rho, 0.1)
        rad_sigma = self.r_floor + self.r_frac * rho
        pos_meas = p_port + np.array([r.normal(0, rad_sigma),
                                      r.normal(0, lat_sigma), r.normal(0, lat_sigma)])
        R_pos = np.diag([rad_sigma**2, lat_sigma**2, lat_sigma**2])
        att_meas = None; R_att = None
        if rho < self.att_rng:                                # port pattern resolved
            att_sigma = self.att_floor + self.att_slope * rho
            att_meas = float(att_err_to_port_rad + r.normal(0, att_sigma))
            R_att = att_sigma**2
        return dict(pos_port=pos_meas, R_pos=R_pos, att_err=att_meas, R_att=R_att, range=rho)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(HWB.summary()[:60], "...")
    vbn = DockingVBN(rng)
    for rho in (150.0, 20.0, 2.0, 0.3):
        p = np.array([rho, 0.02 * rho, -0.01 * rho])
        m = vbn.measure(p, np.radians(5.0))
        if m is None:
            print(f"range {rho:6.1f} m -> no fix"); continue
        perr = np.linalg.norm(m["pos_port"] - p)
        print(f"range {rho:6.1f} m -> pos err {perr*100:5.2f} cm, "
              f"rel-att {'yes (%.2f deg)'%np.degrees(m['att_err']) if m['att_err'] is not None else 'no'}")
    print("docking_sensor OK")
