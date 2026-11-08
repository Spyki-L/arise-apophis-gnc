#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_case1_nac.py - CASE-1 (kinematic) tiered coverage for the SINGLE-EPOCH NAC fleet.

8 deputies, single closest-approach pass (no multi-epoch).  Each deputy = wide camera
(WAC, 0.15 m @ 750 m) + narrow camera (NAC, 0.15 m @ NAC_KM).  Coverage is classified
per facet into:

   cat 0  STEREO @ 0.15 m   (NAC, Sun- or strobe-lit, science SNR)
   cat 1  STEREO @ <= 1 m   (WAC, Sun / strobe / EARTHSHINE-lit)         [not cat 0]
   cat 2  SHAPE (visual hull / silhouette only)                          [not cat 0/1]
   cat 3  none

Illumination is REAL: Sun direction from the encounter geometry (true ~24 deg SPICE Sun),
EARTHSHINE from Earth's true direction + distance at closest approach (Lambert-sphere flux),
strobe from the XB6 capacitor model.  Camera/detector parameters are the real CIS120 +
variable-aperture optics from hardware.py.  Only OCCLUSION is optional (--occ); everything
else is physical.  Marginalised over M tumbling spin-phase poses.

Outputs (in nominal_v2/):
   case1_nac_result.json            tier coverage (overall / sunlit / dark) + CI
   case1_facet_categories.npz       per-facet BEST category + per-tier hit-probability
   outputs/plots/case1_facet_categories.png   3-D facet map coloured by category

Run locally (pure numpy; Basilisk not required):
   python simulate_case1_nac.py --m 32 --occ 0
"""
import os, sys, math, json, argparse, time, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import optimize_refly_stations as OPT
import occlusion_bvh as OCC
import hardware as HW, brdf as BRDF, detector as DET, strobe as STR

# ---------------------------------------------------------------- geometry / illumination
def load_geometry():
    """Real Sun + Earth directions and Earth distance at CA, in the inertial frame."""
    draw = os.path.join(HERE, "data", "mc", "draw_0000")
    sun = np.load(os.path.join(draw, "apophis", "attitude_truth.npz"), allow_pickle=True)["sun_hat"]
    f0 = sorted(glob.glob(os.path.join(draw, "trajectories", "apophis_rendezvous_full_D*.npz")))[0]
    d = np.load(f0, allow_pickle=True); rA = np.asarray(d["rA_N"], float)
    kca = int(np.argmin(np.linalg.norm(d["rD_N"] - rA, axis=1)))
    r_earth = -rA[kca]                                  # Apophis -> Earth (geocentric N frame)
    earth_km = np.linalg.norm(r_earth) / 1e3
    earth_hat = r_earth / np.linalg.norm(r_earth)
    # Lambert-sphere reflected flux at the body, ~gibbous phase (factor 0.5)
    EARTH_ALBEDO, EARTH_R_KM, PHASE = 0.30, 6371.0, 0.5
    esh_frac = EARTH_ALBEDO * PHASE * (EARTH_R_KM / earth_km) ** 2
    return np.asarray(sun, float), earth_hat, earth_km, esh_frac

# ---------------------------------------------------------------- one single-epoch pass
def score_pass(rs, R, mesh_data, geom, occ, esh_coadd, nac_km):
    C0, N0, areas, V, faces = mesh_data
    sun, earth_hat, earth_km, esh_frac = geom
    F = len(C0); C = C0 @ R.T; N = N0 @ R.T
    rt = OCC.MeshRT(V @ R.T, faces) if occ else None
    lit_sun = OCC.lit_facets(rt, C, N, sun) if occ else ((N @ sun) > 0)
    lit_earth = (N @ earth_hat) > 0                    # Earth-facing (earthshine candidate)

    cams = OPT.cams_for(rs); K = len(cams)
    L = np.empty((K, F, 3)); fr = np.zeros((K, F), bool)
    bR = np.full(F, np.inf); bE = np.full(F, 999.0); limb = np.zeros(F, bool)
    for i, vp in enumerate(cams):
        look = vp[None, :] - C; rng = np.linalg.norm(look, axis=1); lh = look / rng[:, None]
        ce = np.einsum('ij,ij->i', N, lh); edeg = np.degrees(np.arccos(np.clip(ce, -1, 1)))
        vis = OCC.visible_from(rt, C, N, vp) if occ else (ce > 0)
        f = vis & (ce > 0) & (rng <= OPT.MAXR); L[i] = lh; fr[i] = f
        up = f & (edeg < bE); bE[up] = edeg[up]; bR[up] = rng[up]
        limb |= (rng <= 20000) & (edeg >= 78) & (edeg <= 92)        # silhouette appearance
    # stereo pair in the 5-30 deg convergence band
    st = np.zeros(F, bool)
    for i in range(K):
        if not fr[i].any():
            continue
        for j in range(i + 1, K):
            v = fr[i] & fr[j]
            if not v.any():
                continue
            dot = np.einsum('ij,ij->i', L[i], L[j]); st |= v & (dot >= OPT.cab) & (dot <= OPT.caa)
    seen = (bE < 89) & (bR < np.inf); er = np.radians(np.clip(bE, 0, 89.9))

    # --- real radiometry (CIS120 + variable aperture) ---
    E_sun = DET.solar_irradiance(1.0)
    lit_des = DET.size_sunlit_aperture(model="hapke")
    ap_lit, t_exp, n_coadd = lit_des["aperture_m"], lit_des["t_exp_s"], lit_des["n_coadd"]
    Opix = (HW.CIS120.pixel_m / HW.FOCAL_M) ** 2
    rad = lambda inc: BRDF.radiance_over_Esun(inc, er, np.zeros(F), model="hapke")
    # sunlit SNR (science) - design point already lands SNR ~ 300-400
    snr_sun = np.where(lit_sun, lit_des["snr_total"], 0.0)
    # earthshine SNR (faint; co-add deeper at <=1 m)
    inc_e = np.arccos(np.clip(N @ earth_hat, 0, 1))
    Ne_e = DET.collected_electrons(rad(inc_e), esh_frac * E_sun, ap_lit, t_exp)
    snr_esh = np.where(lit_earth, DET.snr(Ne_e, t_exp, esh_coadd), 0.0)
    # strobe SNR (per-draw photometry, opposition), reaches ~5 km
    bh = np.arctan2(STR.BODY_RADIUS_M, np.maximum(bR, 1.0)); Om = 2 * np.pi * (1 - np.cos(bh))
    Lop = BRDF.radiance_over_Esun(er, er, np.zeros(F), model="hapke")
    Ne_s = Lop * (OPT.E_def / (Om * np.maximum(bR, 1.0) ** 2) * 0.8) * (math.pi * (HW.APERTURE_MAX_M / 2) ** 2) \
        * Opix / DET.PHOTON_ENERGY_J * HW.CIS120.qe_at_band * HW.CIS120.optical_eff
    snr_str = np.nan_to_num(OPT.snr_Ne(Ne_s))

    det_fine = (snr_sun >= HW.SNR_LIT_BAND[0]) | (snr_str >= STR.DARK_SNR_MIN)        # 0.15 m grade
    det_coarse = det_fine | (snr_esh >= 2 * STR.DARK_SNR_MIN)                          # <=1 m grade (incl earthshine)
    gsd_nac = (0.15 / (nac_km * 1000.0)) * np.maximum(bR, 1.0)
    gsd_wac = HW.IFOV_RAD * np.maximum(bR, 1.0)

    cat = np.full(F, 3, np.int8)
    c0 = st & seen & (gsd_nac <= 0.15) & det_fine
    c1 = st & seen & (gsd_wac <= 1.0) & det_coarse & ~c0
    c2 = limb & ~c0 & ~c1
    cat[c2] = 2; cat[c1] = 1; cat[c0] = 0
    lit = lit_sun
    return cat, lit, lit_earth & (snr_esh >= 2 * STR.DARK_SNR_MIN) & (cat == 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standoff", type=float, default=800.0)
    ap.add_argument("--nac_km", type=float, default=3.0)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--occ", type=int, default=0, help="ray-traced occlusion (everything else is real)")
    ap.add_argument("--esh_coadd", type=int, default=50, help="earthshine co-add frames at <=1 m")
    ap.add_argument("--targets", default="refly_targets_cma.npz")
    a = ap.parse_args()

    mm = np.load(os.path.join(HERE, "data", "radar_mesh.npz"))
    C0, N0, areas, V, faces = mm["centroids"], mm["normals"], mm["areas"], mm["vertices"], mm["faces"]
    Atot = areas.sum(); F = len(C0); mesh_data = (C0, N0, areas, V, faces)
    geom = load_geometry(); sun, earth_hat, earth_km, esh_frac = geom
    tgt = np.load(os.path.join(HERE, a.targets))["DEPUTY_TARGETS_M"]; rs0 = (OPT.C_A.T @ tgt.T).T
    rs = rs0 * (a.standoff / np.median([np.linalg.norm(r) for r in rs0]))
    print("Real geometry: Sun_hat=%s ; Earth %.0f km -> earthshine %.2f%% of sunlight ; NAC 0.15 m @ %.1f km"
          % (np.round(sun, 3), earth_km, 100 * esh_frac, a.nac_km), flush=True)

    rng = np.random.default_rng(20290413)
    af = lambda mask: 100 * areas[mask].sum() / Atot
    rows = []; best_cat = np.full(F, 3, np.int8); hit = np.zeros((4, F))
    t0 = time.time()
    for mi in range(a.m):
        p = OPT.TUM.sample_tumbler(rng); _, dc = OPT.TUM.integrate_window(p, np.array([0., 1.])); R = dc[0]
        cat, lit, esh_only = score_pass(rs, R, mesh_data, geom, bool(a.occ), a.esh_coadd, a.nac_km)
        best_cat = np.minimum(best_cat, cat)
        for k in range(4):
            hit[k] += (cat == k)
        c0 = cat == 0; c01 = cat <= 1; c012 = cat <= 2
        rows.append(dict(t015=af(c0), t1=af(c01), shape=af(c012),
                         sun015=af(c0 & lit), dark015=af(c0 & ~lit),
                         sun1=af(c01 & lit), dark1=af(c01 & ~lit), esh=af(esh_only),
                         sunlit_side=100 * areas[c0 & lit].sum() / max(areas[lit].sum(), 1e-9)))
        print("  pose %2d: 0.15m=%.1f  <=1m=%.1f  +shape=%.1f  sunlit-side=%.1f" %
              (mi, rows[-1]["t015"], rows[-1]["t1"], rows[-1]["shape"], rows[-1]["sunlit_side"]), flush=True)
    hit /= a.m
    np.savez_compressed(os.path.join(HERE, "case1_facet_categories.npz"),
                        best_category=best_cat, hit_prob=hit, areas=areas, centroids=C0)
    g = lambda key: dict(mean=float(np.mean([r[key] for r in rows])),
                         P5=float(np.percentile([r[key] for r in rows], 5)),
                         P95=float(np.percentile([r[key] for r in rows], 95)))
    summ = dict(standoff_m=a.standoff, nac_km=a.nac_km, occ=bool(a.occ), m=a.m,
                earthshine_pct=100 * esh_frac, earth_km=earth_km, sun_hat=sun.tolist(),
                stereo_0p15m=g("t015"), stereo_1m=g("t1"), plus_shape=g("shape"),
                sunlit_side_0p15m=g("sunlit_side"), sunlit_at_0p15=g("sun015"), dark_at_0p15=g("dark015"),
                sunlit_at_1m=g("sun1"), dark_at_1m=g("dark1"), earthshine_add=g("esh"),
                wall_min=(time.time() - t0) / 60)
    json.dump(summ, open(os.path.join(HERE, "case1_nac_result.json"), "w"), indent=2)
    print("\n=== CASE-1 SINGLE-EPOCH SUMMARY (real Sun+earthshine+strobe, occ=%s, M=%d) ===" % (bool(a.occ), a.m))
    print("  STEREO @0.15 m : %.1f%%  (sunlit %.0f / dark %.0f ; sunlit-side %.1f%%)" %
          (summ["stereo_0p15m"]["mean"], summ["sunlit_at_0p15"]["mean"], summ["dark_at_0p15"]["mean"], summ["sunlit_side_0p15m"]["mean"]))
    print("  STEREO @<=1 m  : %.1f%%  (sunlit %.0f / dark %.0f ; earthshine adds %.0f)" %
          (summ["stereo_1m"]["mean"], summ["sunlit_at_1m"]["mean"], summ["dark_at_1m"]["mean"], summ["earthshine_add"]["mean"]))
    print("  + VISUAL HULL  : %.1f%% shape-closed   (residual %.1f%%)" % (summ["plus_shape"]["mean"], 100 - summ["plus_shape"]["mean"]))
    print("  wrote case1_nac_result.json + case1_facet_categories.npz")

    # --- 3-D facet category map ---
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        cols = np.array([[0.13, 0.55, 0.13, 1], [0.19, 0.51, 0.74, 1], [0.99, 0.75, 0.20, 1], [0.7, 0.7, 0.7, 1]])
        fig = plt.figure(figsize=(12, 4.5))
        for sp, az in [(131, 30), (132, 210)]:
            ax = fig.add_subplot(sp, projection="3d"); pc = Poly3DCollection(V[faces], linewidths=0)
            pc.set_facecolor(cols[best_cat]); ax.add_collection3d(pc)
            mx = np.abs(V).max(); ax.set_xlim(-mx, mx); ax.set_ylim(-mx, mx); ax.set_zlim(-mx, mx)
            ax.set_box_aspect((1, 1, 1)); ax.view_init(20, az); ax.set_axis_off()
        ax = fig.add_subplot(133); from matplotlib.patches import Patch
        ax.axis("off"); ax.legend(handles=[Patch(color=cols[i], label=l) for i, l in
                                  enumerate(["0.15 m stereo", "<=1 m stereo", "visual-hull shape", "none"])], loc="center", fontsize=11)
        fig.suptitle("Apophis facet categories (best over %d poses): 0.15 m %.0f%% | <=1 m %.0f%% | shape %.0f%%"
                     % (a.m, summ["stereo_0p15m"]["mean"], summ["stereo_1m"]["mean"], summ["plus_shape"]["mean"]))
        os.makedirs(os.path.join(HERE, "outputs", "plots"), exist_ok=True)
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "outputs", "plots", "case1_facet_categories.png"), dpi=130); plt.close(fig)
        print("  wrote outputs/plots/case1_facet_categories.png")
    except Exception as e:
        print("  (plot skipped: %s)" % e)

if __name__ == "__main__":
    main()
