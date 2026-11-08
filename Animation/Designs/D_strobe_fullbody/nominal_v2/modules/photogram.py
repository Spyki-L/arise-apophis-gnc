"""
photogram.py  -  ARISE V2 photogrammetric reconstruction metric (E1)
====================================================================
Replaces the binary "two views in the 5-30 deg parallax band" gate with a real triangulation-
error model, so the MC reports the MISSION-MEANINGFUL product: a per-facet reconstruction-
accuracy distribution ("X% of facets reconstructed to <= 1 m").

For a stereo pair viewing a facet at range R with convergence (parallax) angle beta, the
along-LOS (depth) uncertainty from a sub-pixel disparity-matching error sigma_disp (pixels):

    sigma_z = R * (sigma_disp * IFOV) / (2 * sin(beta/2))           [m]

Disparity-matching precision is SNR- and texture-limited (Cramer-Rao-like):

    sigma_disp = max(sigma_floor, k_match / SNR_eff)               [px]
    SNR_eff    = harmonic mean of the two views' SNR (the weaker view dominates matching)

A pair only matches if both views clear an SNR/texture threshold AND beta is in a usable band
(too small -> ill-conditioned depth; too large -> matching/foreshortening fails).

Per facet: evaluate every valid pair, keep the BEST (smallest sigma_z).  Summaries: fraction of
(area-weighted) facets reconstructed to a set of tolerances, plus the full CDF.
"""
from __future__ import annotations
from itertools import combinations
import math
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import hardware as HW

# tunables (defensible defaults)
CONV_BAND_DEG = (5.0, 30.0)     # usable convergence band
SNR_MATCH_MIN = 10.0            # min per-view SNR to attempt feature matching
SIGMA_DISP_FLOOR_PX = 0.1       # best achievable sub-pixel matching (well-textured, high SNR)
K_MATCH = 2.0                   # sigma_disp = K_MATCH / SNR_eff  (px) above the floor
RECON_TOLS_M = (0.25, 0.5, 1.0, 2.0, 5.0)


def disparity_sigma_px(snr_a, snr_b, floor=SIGMA_DISP_FLOOR_PX, k=K_MATCH):
    """Sub-pixel disparity-matching error [px] from the two views' SNR (weaker view dominates)."""
    snr_a = np.asarray(snr_a, float); snr_b = np.asarray(snr_b, float)
    snr_eff = 2.0 / (1.0 / np.maximum(snr_a, 1e-6) + 1.0 / np.maximum(snr_b, 1e-6))  # harmonic mean
    return np.maximum(floor, k / np.maximum(snr_eff, 1e-6)), snr_eff


def triangulation_sigma_m(range_m, conv_deg, snr_a, snr_b, ifov):
    """Along-LOS depth uncertainty [m] for one stereo pair (returns np.inf if not matchable)."""
    conv = np.radians(conv_deg)
    sdisp, snr_eff = disparity_sigma_px(snr_a, snr_b)
    sig = range_m * (sdisp * ifov) / (2.0 * np.sin(conv / 2.0))
    bad = (conv_deg < CONV_BAND_DEG[0]) | (conv_deg > CONV_BAND_DEG[1]) | \
          (snr_a < SNR_MATCH_MIN) | (snr_b < SNR_MATCH_MIN)
    return np.where(bad, np.inf, sig)


def best_recon_for_facet(views, ifov, nav=None, mtf=1.0):
    """views: list of dicts {range_m, snr, look_hat(3,), [dep]} for one facet.  Returns best
    sigma_z [m] over all view pairs (convergence = angle between look directions).

    Knowledge-error model (item #2: truth vs knowledge).  The depth error combines, in quadrature
    in the triangulation numerator (all divided by 2 sin(beta/2)):
      - image matching        : R * sigma_disp * IFOV          (SNR/texture-limited; truth term)
      - BASELINE knowledge    : how well the two viewpoints' relative position is known
      - POINTING/centroiding   : R * sigma_point  (line-of-sight direction knowledge)
    Cross- vs intra-deputy are treated differently because for a SINGLE deputy imaging in a short
    burst the nav-position and attitude errors are common-mode (they cancel in the relative
    geometry), so self/motion-stereo is near-truth; only the camera's own frame clock (us) enters.
    Cross-deputy pairs carry the full independent relative-nav sigma + fleet clock skew (V*sigma_t)
    + independent pointing errors.  If nav is None -> legacy truth-geometry behaviour."""
    best = np.inf
    n = len(views)
    if n < 2:
        return best
    for a, b in combinations(range(n), 2):
        va, vb = views[a], views[b]
        if (va["snr"] < SNR_MATCH_MIN) or (vb["snr"] < SNR_MATCH_MIN):
            continue
        cosb = np.clip(np.dot(va["look_hat"], vb["look_hat"]), -1, 1)
        conv = np.degrees(np.arccos(cosb))
        if (conv < CONV_BAND_DEG[0]) or (conv > CONV_BAND_DEG[1]):
            continue
        R = 0.5 * (va["range_m"] + vb["range_m"])
        sdisp, _ = disparity_sigma_px(va["snr"], vb["snr"])
        # MTF (item #3): low contrast at Nyquist degrades sub-pixel matching ( sigma_disp ~ 1/MTF )
        sdisp = float(sdisp) / max(mtf, 1e-3)
        num2 = (R * sdisp * ifov) ** 2                            # image-matching cross-range error
        if nav is not None:
            V = float(nav.get("V", 8760.0))
            ps = nav.get("pos_sigma"); pt = nav.get("point_sigma")
            da = int(va.get("dep", 0)); db = int(vb.get("dep", 0))
            g = lambda arr, i: (float(arr[i]) if (arr is not None and 0 <= i < len(arr)) else 0.0)
            if da == db:
                # self / motion stereo: nav + attitude errors are common-mode over the burst
                dB = V * float(nav.get("clock_intra_s", 1e-6))
                pterm2 = 0.0
            else:
                dB = math.sqrt(g(ps, da) ** 2 + g(ps, db) ** 2
                               + (V * float(nav.get("clock_fleet_s", 1e-5))) ** 2)
                pterm2 = (R * g(pt, da)) ** 2 + (R * g(pt, db)) ** 2
            num2 += dB ** 2 + pterm2
        sig = math.sqrt(num2) / (2.0 * math.sin(math.radians(conv) / 2.0))
        if sig < best:
            best = sig
    return best


MTF_RESOLVE_MIN = 0.10   # below this system MTF at Nyquist, GSD-scale features are not resolved


def _hifi_on():
    return _os.getenv("ARISE_PHOTOGRAM_HIFI", "0") not in ("0", "", "false", "False")


def reconstruct(facet_views, areas, ifov, tols=RECON_TOLS_M, nav=None, mtf=1.0, incidence_deg=None):
    """facet_views: list (len n_facets) of per-facet view lists.  areas: facet areas.
    nav (optional): knowledge-error spec {pos_sigma[ndep], point_sigma[ndep], clock_fleet_s,
    clock_intra_s, V} -> reconstruct from ESTIMATED geometry (item #2); None -> truth geometry.
    mtf (item #3): system MTF at Nyquist -> degrades sub-pixel matching; if < MTF_RESOLVE_MIN the
    tier's GSD-scale features are unresolved (no reconstruction).  Returns sigma_z per facet, etc.
    ARISE_PHOTOGRAM_HIFI on -> stereophotoclinometry (multi-ray covariance + photoclinometry); off -> legacy."""
    if _hifi_on():
        return _reconstruct_spc(facet_views, areas, ifov, tols, nav, mtf, incidence_deg)
    n = len(facet_views)
    sigz = np.full(n, np.inf)
    nview = np.zeros(n, int)
    resolved = (mtf >= MTF_RESOLVE_MIN)
    for i, vws in enumerate(facet_views):
        nview[i] = len(vws)
        sigz[i] = best_recon_for_facet(vws, ifov, nav=nav, mtf=mtf) if resolved else np.inf
    areas = np.asarray(areas, float)
    Atot = areas.sum()
    fracs = {}
    for tol in tols:
        m = sigz <= tol
        fracs[tol] = float(areas[m].sum() / Atot)
    mono = float(areas[nview >= 1].sum() / Atot)
    stereo = float(areas[np.isfinite(sigz)].sum() / Atot)
    return dict(sigma_z_m=sigz, n_views=nview, area_frac_within=fracs,
                mono_frac=mono, stereo_frac=stereo,
                median_sigma_m=float(np.median(sigz[np.isfinite(sigz)])) if np.isfinite(sigz).any() else np.inf)


# ==========================================================================================
# HI-FI STEREOPHOTOCLINOMETRY (SPC) -- Cramer-Rao matching + multi-ray covariance + photoclinometry
# Refs: Gaskell/Barnouin 2020 + Palmer 2024 (PSJ, OSIRIS-REx SPC); Debella-Gilo & Kaeaeb 2011 (RSE);
# Forstner (CRLB matching); Horn 1990 / Kirk 1987 (photoclinometry / shape-from-shading).
# ==========================================================================================

def disparity_sigma_crlb(snr_eff, mtf=1.0, texture=None, window_px=None, floor=None, k=None):
    """Texture-and-noise-limited (Cramer-Rao) subpixel disparity precision [px] (Debella-Gilo & Kaeaeb
    2011; Forstner):  sigma_disp ~ 1 / (SNR * MTF * sqrt(N_window * texture_energy)).  Reduces to the
    legacy K/SNR at the reference texture/window, and adds the texture, window, and MTF dependence."""
    pg = HW.PHOTOGRAM
    texture = pg.texture_energy if texture is None else texture
    N = (pg.match_window_px if window_px is None else window_px) ** 2
    floor = pg.sigma_disp_floor_px if floor is None else floor
    k = 2.0 if k is None else k                                       # legacy K_MATCH
    # calibrate so at (texture_ref, N_ref, MTF=1) it equals the legacy k/SNR
    C = k * math.sqrt((pg.match_window_px ** 2) * pg.texture_energy)
    sig = C / (np.maximum(snr_eff, 1e-6) * max(mtf, 1e-3) * np.sqrt(N * texture))
    return np.maximum(floor, sig)


def multiray_sigma_m(views, ifov, mtf=1.0, nav=None, conv_band=None, snr_min=None):
    """Full multi-ray forward-intersection depth uncertainty [m]: build the 3x3 Fisher information from
    every matchable ray (each constrains the plane perpendicular to its LOS), invert -> the worst-axis
    (depth) 1-sigma.  Combines ALL views (bundle adjustment), not just the best pair; single/parallel
    views -> unconstrained (inf)."""
    cb = CONV_BAND_DEG if conv_band is None else conv_band
    smin = SNR_MATCH_MIN if snr_min is None else snr_min
    good = [v for v in views if v["snr"] >= smin]
    if len(good) < 2:
        return math.inf
    dirs = [np.asarray(v["look_hat"], float) / (np.linalg.norm(v["look_hat"]) + 1e-30) for v in good]
    # a view participates only if it has >=1 in-band matchable partner (foreshortening limit)
    Info = np.zeros((3, 3)); used = 0
    for i, vi in enumerate(good):
        partner = False
        for j, vj in enumerate(good):
            if i == j: continue
            conv = math.degrees(math.acos(max(-1, min(1, float(np.dot(dirs[i], dirs[j]))))))
            if cb[0] <= conv <= cb[1]:
                partner = True; break
        if not partner:
            continue
        sdisp = float(disparity_sigma_crlb(vi["snr"], mtf))
        sperp2 = (vi["range_m"] * sdisp * ifov) ** 2
        if nav is not None:                                          # add knowledge (baseline/pointing)
            V = float(nav.get("V", 8760.0)); da = int(vi.get("dep", 0))
            ps = nav.get("pos_sigma"); pt = nav.get("point_sigma")
            g = lambda arr, k: (float(arr[k]) if (arr is not None and 0 <= k < len(arr)) else 0.0)
            sperp2 += g(ps, da) ** 2 + (vi["range_m"] * g(pt, da)) ** 2 + (V * float(nav.get("clock_fleet_s", 1e-5))) ** 2
        d = dirs[i]
        Info += (np.eye(3) - np.outer(d, d)) / max(sperp2, 1e-12)
        used += 1
    if used < 2:
        return math.inf
    # depth = the reconstruction error ALONG the mean line-of-sight (what sigma_z means); combining
    # more rays (Fisher information adds) is guaranteed to tighten it vs any single pair.
    u = np.sum(np.array(dirs), axis=0); u /= np.linalg.norm(u) + 1e-30
    w, Vv = np.linalg.eigh(Info)
    uc = Vv.T @ u
    var = 0.0
    for kk in range(3):
        if w[kk] <= 1e-9 * w[-1]:
            if uc[kk] ** 2 > 1e-9:                                   # depth has an unconstrained component
                return math.inf
        else:
            var += uc[kk] ** 2 / w[kk]
    return float(math.sqrt(var))


def photoclinometry_sigma_m(view, ifov, incidence_deg, pg=None):
    """Photoclinometry / shape-from-shading height uncertainty [m] for a SINGLE image (Horn 1990;
    Kirk 1987): slope from brightness, sigma_h ~ GSD / (SNR * |dI/dslope|), dI/dslope ~ sin(incidence).
    Recovers height where stereo cannot (mono / low-parallax).  Ill-posed near normal incidence."""
    pg = pg or HW.PHOTOGRAM
    i = float(incidence_deg)
    if view["snr"] < 3.0 or i < pg.pc_incidence_min_deg or i > 175.0:
        return math.inf
    gsd = ifov * view["range_m"]
    dIdslope = max(abs(math.sin(math.radians(i))), 1e-3)            # Lambert slope sensitivity
    photon = gsd / (view["snr"] * dIdslope) * pg.pc_slope_noise_ref
    return float(max(pg.pc_systematic_frac * gsd, photon))          # BRDF/albedo systematic floor (~pixel scale)


def spc_sigma_m(stereo_sigma_m, pc_sigma_m):
    """Stereophotoclinometry: inverse-variance blend of stereo (accuracy) and photoclinometry
    (resolution) -> 'accuracy of stereo with the resolution of photoclinometry' (Gaskell; Palmer 2024).
    A mono facet (stereo=inf) is still recovered by photoclinometry."""
    inv = 0.0
    if np.isfinite(stereo_sigma_m) and stereo_sigma_m > 0: inv += 1.0 / stereo_sigma_m ** 2
    if np.isfinite(pc_sigma_m) and pc_sigma_m > 0: inv += 1.0 / pc_sigma_m ** 2
    return float(1.0 / math.sqrt(inv)) if inv > 0 else math.inf


def _reconstruct_spc(facet_views, areas, ifov, tols=RECON_TOLS_M, nav=None, mtf=1.0, incidence_deg=None):
    """Hi-fi stereophotoclinometry reconstruction (ARISE_PHOTOGRAM_HIFI): per facet combine the full
    multi-ray triangulation covariance (stereo accuracy) with photoclinometry / shape-from-shading
    (resolution; recovers mono / low-parallax facets) via an inverse-variance SPC blend."""
    n = len(facet_views)
    sigz = np.full(n, np.inf); nview = np.zeros(n, int)
    resolved = (mtf >= MTF_RESOLVE_MIN)
    inc = None if incidence_deg is None else np.asarray(incidence_deg, float)
    for i, vws in enumerate(facet_views):
        nview[i] = len(vws)
        if not resolved or len(vws) < 1:
            continue
        st = multiray_sigma_m(vws, ifov, mtf=mtf, nav=nav)
        pc = math.inf
        if inc is not None:
            bv = max(vws, key=lambda v: v["snr"])
            pc = photoclinometry_sigma_m(bv, ifov, float(inc[i]))
        sigz[i] = spc_sigma_m(st, pc)
    areas = np.asarray(areas, float); Atot = areas.sum()
    fracs = {tol: float(areas[sigz <= tol].sum() / Atot) for tol in tols}
    mono = float(areas[nview >= 1].sum() / Atot)
    stereo = float(areas[np.isfinite(sigz)].sum() / Atot)
    return dict(sigma_z_m=sigz, n_views=nview, area_frac_within=fracs, mono_frac=mono, stereo_frac=stereo,
                median_sigma_m=float(np.median(sigz[np.isfinite(sigz)])) if np.isfinite(sigz).any() else np.inf)


def _selftest():
    fails = []
    def check(name, cond, detail=""):
        print(("  PASS " if cond else "  FAIL ") + name + ("" if not detail else "  -> " + detail))
        if not cond: fails.append(name)
    ifov = HW.IFOV_RAD; pg = HW.PHOTOGRAM
    look = lambda az, el=0.2: (lambda v: v / np.linalg.norm(v))(np.array([np.cos(np.radians(az)), np.sin(np.radians(az)), el]))

    print("=" * 84)
    print("PHOTOGRAM HI-FI SELF-TEST  (SPC = stereo + photoclinometry)")
    print("  Debella-Gilo/Forstner (CRLB matching) + multi-ray covariance + Horn/Kirk (SFS) + Palmer 2024 (SPC)")
    print("=" * 84)

    # A. Cramer-Rao matching
    print("\nA. Cramer-Rao subpixel matching precision [px]:")
    for snr in (10, 30, 100):
        print(f"     SNR={snr:3d}: MTF1.0 tex0.5 -> {float(disparity_sigma_crlb(snr,1.0)):.3f}; "
              f"MTF0.4 -> {float(disparity_sigma_crlb(snr,0.4)):.3f}; tex1.0 -> {float(disparity_sigma_crlb(snr,1.0,texture=1.0)):.3f}")
    leg = 2.0 / 10.0                                                 # legacy k/SNR at SNR=10
    check("CRLB reduces to legacy K/SNR at reference (tex0.5, MTF1)", abs(float(disparity_sigma_crlb(10,1.0)) - leg) < 0.02)
    check("more texture -> tighter matching", disparity_sigma_crlb(10,1.0,texture=1.0) < disparity_sigma_crlb(10,1.0,texture=0.3))
    check("lower MTF -> looser matching", disparity_sigma_crlb(10,0.4) > disparity_sigma_crlb(10,1.0))
    check("higher SNR -> tighter (to the floor)", disparity_sigma_crlb(300,1.0) <= disparity_sigma_crlb(30,1.0))

    # B. multi-ray covariance vs best pair
    fac3 = [dict(range_m=750, snr=300, look_hat=look(0)), dict(range_m=760, snr=280, look_hat=look(12)),
            dict(range_m=770, snr=260, look_hat=look(24))]
    mr = multiray_sigma_m(fac3, ifov)                                 # all 3 views (full covariance)
    mr2 = multiray_sigma_m([fac3[0], fac3[2]], ifov)                  # widest single pair (0,24 deg)
    bp = best_recon_for_facet(fac3, ifov)                          # legacy best-pair B/H estimate
    print(f"\nB. Multi-ray covariance (3 views @ 750 m): all-views {mr:.3f} m, best-pair-only {mr2:.3f} m, "
          f"legacy B/H {bp:.3f} m")
    check("multi-ray tightens as a 3rd view is added (Fisher info adds)", mr <= mr2 + 1e-9, f"{mr:.3f} <= {mr2:.3f}")
    check("multi-ray same order as the legacy B/H estimate", 0.3 * bp < mr < 3.0 * bp, f"{mr:.3f} vs {bp:.3f}")
    check("single view -> depth unconstrained (inf)", not np.isfinite(multiray_sigma_m([fac3[0]], ifov)))
    wide = [dict(range_m=750, snr=300, look_hat=look(0)), dict(range_m=750, snr=300, look_hat=look(20))]
    narrow = [dict(range_m=750, snr=300, look_hat=look(0)), dict(range_m=750, snr=300, look_hat=look(6))]
    check("wider convergence -> smaller depth sigma", multiray_sigma_m(wide, ifov) < multiray_sigma_m(narrow, ifov))

    # C. photoclinometry (mono facet)
    v = dict(range_m=750, snr=100, look_hat=look(0))
    pc_graze = photoclinometry_sigma_m(v, ifov, 70.0); pc_norm = photoclinometry_sigma_m(v, ifov, 3.0)
    print(f"\nC. Photoclinometry (mono, R=750 m, SNR100): incidence70deg -> {pc_graze:.3f} m, "
          f"near-normal 3deg -> {'inf' if not np.isfinite(pc_norm) else '%.2f'%pc_norm}")
    check("photoclinometry recovers a MONO facet (finite height)", np.isfinite(pc_graze))
    check("SFS best at grazing incidence, ill-posed near normal", pc_graze < 5.0 and not np.isfinite(pc_norm))
    check("photoclinometry improves-or-floors with SNR",
          photoclinometry_sigma_m(dict(range_m=750,snr=100,look_hat=look(0)),ifov,20.) <=
          photoclinometry_sigma_m(dict(range_m=750,snr=10,look_hat=look(0)),ifov,20.) + 1e-12)

    # D. SPC blend + OSIRIS-REx benchmark
    st = multiray_sigma_m(fac3, ifov); pc = photoclinometry_sigma_m(fac3[0], ifov, 60.0)
    spc = spc_sigma_m(st, pc)
    spc_mono = spc_sigma_m(math.inf, pc)                            # mono facet: stereo=inf
    gsd = ifov * 750.0
    print(f"\nD. SPC blend: stereo {st:.3f} + photoclinometry {pc:.3f} -> SPC {spc:.3f} m "
          f"(GSD {gsd:.3f} m; Apophis R {pg.apophis_mean_radius_m:.0f} m -> {spc/pg.apophis_mean_radius_m*100:.3f}% radius)")
    check("SPC <= min(stereo, photoclinometry)", spc <= min(st, pc) + 1e-9)
    check("SPC recovers a mono facet that pure stereo cannot", np.isfinite(spc_mono))
    check("well-observed SPC accuracy ~ pixel scale (< 3x GSD)", spc < 3 * gsd, f"{spc:.3f} vs GSD {gsd:.3f}")
    check("SPC 3D error ~ OSIRIS-REx class (< 0.5% mean radius)", spc / pg.apophis_mean_radius_m < 0.005)

    print("\n" + "=" * 84)
    print(f"RESULT: {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}  ({len(fails)} failure(s))")
    print("=" * 84)
    return len(fails) == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
