#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faultlab_encounter.py — encounter-geometry analytics for the fault lab windows:

  compute_cov_map(truth, stations) — the closest-approach coverage map. For each ACTIVE
     deputy it marks the asteroid facets it images (incidence + range, near-side
     visibility for the convex body = the highest-fidelity onboard test), the per-facet
     stereo count, the overall stereo coverage, the sun-lit vs shadowed hemispheres, and
     the SHADOWED facets a deputy lights with its strobe (Design-D's dark-side imaging).

  compute_compensation(truth, stations, health, I_dep, t_to_ca) — when a deputy is lost,
     how the survivors take up a bit of its share: each nearby healthy deputy nudges its
     station toward the gap (a ΔV) and re-points its camera (a torque / RW-momentum bite),
     partially recovering the lost coverage (not fully — a graceful, bounded hand-off).

Pure numpy; called by W8 (coverage/strobe) and W9 (loss-compensation) at their own cadence.
"""
from __future__ import annotations
import numpy as np
import numpy.linalg as la

INCIDENCE_MAX = 60.0
N_STEREO = 2
R_MAX = 8000.0


def latlon(cent):
    """Body-frame facet centroids -> (lon[-180,180], lat[-90,90]) for the unwrapped map."""
    x, y, z = cent[:, 0], cent[:, 1], cent[:, 2]
    r = np.maximum(la.norm(cent, axis=1), 1e-9)
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    lon = np.degrees(np.arctan2(y, x))
    return lon, lat


def sun_dir_body(truth):
    """TRUE Sun direction in the asteroid body frame at closest approach (toward the primary).
    This is the illumination TRUTH -- it drives the lit/dark facet labels and MUST stay exact."""
    i = truth["i_ca"]; rA = truth["rA"][i]; C = truth["C_A"][i]
    s = -rA / max(la.norm(rA), 1e-9)
    return C @ s


def sun_dir_body_measured(truth, rng=None):
    """MEASURED Sun direction as the coarse Sun sensor would report it -- the SAFEHOLD attitude
    reference after a star-tracker (ATT) fault.  Identity-when-absent: returns the exact
    sun_dir_body(truth) unless ARISE_SUNSENSOR_HIFI is set, in which case it is perturbed by the
    physical 1-sigma Sun-vector error from sun_sensor.py.  Never used for the illumination truth
    (that is sun_dir_body); this is purely the attitude-knowledge estimate for safe-mode logic."""
    import os
    s_true = sun_dir_body(truth)
    if os.getenv("ARISE_SUNSENSOR_HIFI", "0") in ("0", "", "false", "False"):
        return s_true
    import sys, os.path as _op
    sys.path.insert(0, _op.join(_op.dirname(_op.dirname(_op.abspath(__file__))), "modules"))
    import sun_sensor as _SS
    return _SS.measure(s_true, rng=rng)


def compute_cov_map(truth, stations, max_facets=1600):
    """stations: {d: station_vec(body frame)} for the deputies to score. Returns the map."""
    cent = truth["cent"]; norm = truth["norm"]
    nf = len(cent)
    if nf > max_facets:                                   # downsample for live rendering
        idx = np.linspace(0, nf - 1, max_facets).astype(int)
        cent = cent[idx]; norm = norm[idx]
    cmax = np.cos(np.radians(INCIDENCE_MAX))
    count = np.zeros(len(cent), int)
    per_dep = {}
    for d, st in stations.items():
        los = st[None, :] - cent                          # (F,3) deputy minus facet
        rng = la.norm(los, axis=1)
        u = los / np.maximum(rng[:, None], 1e-9)
        inc = np.einsum("fi,fi->f", u, norm)              # cos incidence
        mask = (rng < R_MAX) & (inc > cmax)               # near-side, in range, within cone
        per_dep[d] = mask
        count += mask.astype(int)
    sun = sun_dir_body(truth)
    lit = (norm @ sun) > 0.0
    dark = ~lit
    imaged = count >= 1
    strobe_lit = dark & imaged                            # dark facets a deputy lights with its strobe
    lon, lat = latlon(cent)
    return dict(cent=cent, norm=norm, count=count, per_dep=per_dep, sun=sun, lit=lit, dark=dark,
                imaged=imaged, strobe_lit=strobe_lit, stereo=count >= N_STEREO, lon=lon, lat=lat,
                n_facets=len(cent),
                cov_overall=float(np.mean(count >= N_STEREO)),
                cov_per_dep={d: float(np.mean(m)) for d, m in per_dep.items()},
                dark_lit_frac=float(np.mean(strobe_lit[dark])) if dark.any() else 0.0)


def compute_compensation(truth, stations, health, offset, I_dep, t_to_ca, alpha=0.18, k_help=3):
    """When deputies are lost, compute the survivors' partial ΔV + torque hand-off.

    stations {d: station} are the CURRENT active stations; health {d: state}; offset {d: vec}.
    Returns a plan per lost deputy + a fleet summary, or None if nothing is lost."""
    lost = [d for d, s in health.items() if s in ("DEAD", "FAULTED")]
    healthy = [d for d, s in health.items() if s == "HEALTHY"]
    if not lost or not healthy:
        return None
    T = max(float(t_to_ca), 1800.0)                       # maneuver time horizon [s] (planned with lead time)
    t_slew = 120.0                                        # camera re-point time [s]
    plans = []
    tot_dv = tot_dH = 0.0
    assign = {h: np.zeros(3) for h in healthy}            # accumulated station nudges
    for L in lost:
        sL = truth["ca_stations"][L]
        helpers = sorted(healthy, key=lambda h: la.norm(truth["ca_stations"][h] + offset.get(h, 0) - sL))[:k_help]
        rows = []
        for h in helpers:
            sh = truth["ca_stations"][h] + offset.get(h, np.zeros(3))
            d_station = alpha * (sL - sh)                  # nudge a BIT toward the gap
            assign[h] = assign[h] + d_station
            dv = 2.0 * la.norm(d_station) / T             # two-impulse CW reposition ~ 2|Δr|/T
            b_old = -sh / max(la.norm(sh), 1e-6)
            b_new = -(sh + d_station) / max(la.norm(sh + d_station), 1e-6)
            dtheta = float(np.arccos(np.clip(b_old @ b_new, -1, 1)))     # camera re-point angle
            dH = I_dep * (2.0 * dtheta / t_slew)          # rest-to-rest slew RW momentum [N·m·s]
            torque = 4.0 * I_dep * dtheta / t_slew ** 2   # peak slew torque [N·m]
            rows.append(dict(helper=h, dv=dv, dH=dH, torque=torque,
                             dtheta_deg=np.degrees(dtheta), d_station=la.norm(d_station)))
            tot_dv += dv; tot_dH += dH
        plans.append(dict(lost=L, helpers=rows))
    # recovery of the LOST deputies' footprint: how much of the gap the nudged survivors regain
    cent = truth["cent"]; norm = truth["norm"]; nf = len(cent)
    if nf > 900:
        idx = np.linspace(0, nf - 1, 900).astype(int); cent = cent[idx]; norm = norm[idx]
    cmax = np.cos(np.radians(INCIDENCE_MAX))

    def imaged_by(st):
        los = np.asarray(st)[None, :] - cent; rng = la.norm(los, axis=1)
        u = los / np.maximum(rng[:, None], 1e-9)
        return (rng < R_MAX) & (np.einsum("fi,fi->f", u, norm) > cmax)

    gap = np.zeros(len(cent), bool)
    for L in lost:
        gap |= imaged_by(truth["ca_stations"][L])               # facets the lost deputies would have seen

    def gap_cov(extra):
        cov = np.zeros(len(cent), bool)
        for h in healthy:
            cov |= imaged_by(truth["ca_stations"][h] + offset.get(h, np.zeros(3)) + extra.get(h, np.zeros(3)))
        return float(np.mean(gap & cov)) if gap.any() else 0.0

    gcov_before = gap_cov({h: np.zeros(3) for h in healthy})
    gcov_after = gap_cov(assign)
    return dict(plans=plans, lost=lost, n_helpers=len({r["helper"] for p in plans for r in p["helpers"]}),
                total_dv=tot_dv, total_dH=tot_dH, T_man=T, gap_facets=int(gap.sum()),
                gap_cov_before=gcov_before, gap_cov_after=gcov_after,
                recovered=max(0.0, gcov_after - gcov_before))


def _stereo_frac(truth, stations, max_facets=900):
    cent = truth["cent"]; norm = truth["norm"]; nf = len(cent)
    if nf > max_facets:
        idx = np.linspace(0, nf - 1, max_facets).astype(int); cent = cent[idx]; norm = norm[idx]
    if len(stations) < N_STEREO:
        return 0.0
    cmax = np.cos(np.radians(INCIDENCE_MAX)); count = np.zeros(len(cent), int)
    for st in stations:
        los = np.asarray(st)[None, :] - cent; rng = la.norm(los, axis=1)
        u = los / np.maximum(rng[:, None], 1e-9)
        count += ((rng < R_MAX) & (np.einsum("fi,fi->f", u, norm) > cmax)).astype(int)
    return float(np.mean(count >= N_STEREO))
