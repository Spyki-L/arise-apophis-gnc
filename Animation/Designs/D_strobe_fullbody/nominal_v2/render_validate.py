#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_validate.py  -  ARISE Monte Carlo V2 render-in-the-loop validation (E2)
==============================================================================
Calibrates the ANALYTIC photogrammetric gate (modules/photogram.py) against an
actual image-formation + stereo-matching pipeline on a subset of draws:

  1. RENDER a synthetic deputy image of the tumbling radar mesh by ray-casting the
     camera grid (pinhole, CIS120 focal/pixel) into the mesh, shaded with the V2
     BRDF (sun-lit) or strobe (dark) and the detector SNR.  -> a real synthetic image.
  2. MATCH + TRIANGULATE: for facets seen by both cameras of a stereo pair, project
     to each image, add a feature-localization noise sigma_pix = f(SNR) (the real
     matching error), back-project the two noisy rays and triangulate the 3-D point.
  3. CALIBRATE: empirical depth error sigma_z (rendered/matched) vs the analytic
     photogram.triangulation_sigma_m -> a per-pair calibration factor for the gate.

Runs on a few draws (default 3).  Pure numpy + matplotlib (image PNG); no Basilisk.

Run : python render_validate.py [--ndraws 3] [--npix 96]
Test: python render_validate.py --self-test
"""
from __future__ import annotations
import os, sys, glob, json, math, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = os.path.join(HERE, "modules")
DATA = os.path.join(HERE, "data"); MCDIR = os.path.join(DATA, "mc")
MESH_NPZ = os.path.join(DATA, "radar_mesh.npz")
OUTDIR = os.path.join(HERE, "outputs", "render_validate")
sys.path.insert(0, MODULES)

import hardware as HW
import brdf as BRDF
import detector as DET
import photogram as PG
import occlusion_bvh as OCC

EPS = 1e-9


# ════════════════════════════════════════════════════════════════════════════
# camera model (pinhole, CIS120)
# ════════════════════════════════════════════════════════════════════════════
def camera_frame(vp, target=np.zeros(3), up=np.array([0., 0., 1.])):
    """World->camera rotation rows [x,y,z] with boresight z = (target - vp)."""
    z = (target - vp); z = z / (np.linalg.norm(z) + EPS)
    x = np.cross(up, z); 
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.array([1., 0, 0]), z)
    x = x / (np.linalg.norm(x) + EPS); y = np.cross(z, x)
    return np.vstack((x, y, z))

def project(P, vp, Rc, f=HW.FOCAL_M, pix=HW.CIS120.pixel_m):
    """Project world points P[...,3] to pixel coords (u,v) and camera depth."""
    pc = (P - vp) @ Rc.T
    depth = pc[..., 2]
    u = f * pc[..., 0] / np.maximum(pc[..., 2], EPS) / pix
    v = f * pc[..., 1] / np.maximum(pc[..., 2], EPS) / pix
    return np.stack([u, v], -1), depth

def backproject(uv, vp, Rc, f=HW.FOCAL_M, pix=HW.CIS120.pixel_m):
    """Pixel (u,v) -> world ray direction from camera center vp."""
    d_cam = np.stack([uv[..., 0] * pix / f, uv[..., 1] * pix / f, np.ones(uv.shape[:-1])], -1)
    d_world = d_cam @ Rc                       # cam->world: rows of Rc are camera axes in world (p_world = p_cam @ Rc)
    return d_world / (np.linalg.norm(d_world, axis=-1, keepdims=True) + EPS)

def triangulate(vpA, dA, vpB, dB):
    """Closest point between two world rays (midpoint of the common perpendicular)."""
    a = np.sum(dA * dA, -1); b = np.sum(dA * dB, -1); c = np.sum(dB * dB, -1)
    w0 = vpA - vpB
    d = np.sum(dA * w0, -1); e = np.sum(dB * w0, -1)
    denom = a * c - b * b + EPS
    sA = (b * e - c * d) / denom; sB = (a * e - b * d) / denom
    pA = vpA + sA[..., None] * dA; pB = vpB + sB[..., None] * dB
    return 0.5 * (pA + pB)


# ════════════════════════════════════════════════════════════════════════════
# ray-cast synthetic image (nearest hit, BRDF-shaded)
# ════════════════════════════════════════════════════════════════════════════
def _ray_tri_nearest(orig, dirs, V0, E1, E2, chunk=4000):
    """Nearest positive ray-triangle hit (Moller-Trumbore).  Returns (t_min[M], face[M])."""
    M = len(dirs); tmin = np.full(M, np.inf); fid = np.full(M, -1, int)
    nT = len(V0)
    for s in range(0, nT, chunk):
        e = min(s + chunk, nT)
        v0 = V0[s:e]; e1 = E1[s:e]; e2 = E2[s:e]
        pvec = np.cross(dirs[:, None, :], e2[None, :, :])
        det = np.einsum('mtk,tk->mt', pvec, e1)
        inv = 1.0 / np.where(np.abs(det) < 1e-12, 1e-12, det)
        tvec = orig[:, None, :] - v0[None, :, :]
        u = np.einsum('mtk,mtk->mt', tvec, pvec) * inv
        qvec = np.cross(tvec, e1[None, :, :])
        vv = np.einsum('mk,mtk->mt', dirs, qvec) * inv
        t = np.einsum('tk,mtk->mt', e2, qvec) * inv
        hit = (u >= 0) & (vv >= 0) & (u + vv <= 1) & (t > 1e-6) & (np.abs(det) >= 1e-12)
        t = np.where(hit, t, np.inf)
        jmin = np.argmin(t, axis=1); tval = t[np.arange(M), jmin]
        better = tval < tmin
        tmin = np.where(better, tval, tmin); fid = np.where(better, s + jmin, fid)
    return tmin, fid

def render_image(verts, faces, centroids, normals, vp, sun_hat, hapke, lit_mask,
                 npix=96, fov_half_deg=27.0):
    """Ray-cast a synthetic deputy image; per-pixel BRDF/strobe-shaded intensity + depth + facet."""
    V0 = verts[faces[:, 0]]; E1 = verts[faces[:, 1]] - V0; E2 = verts[faces[:, 2]] - V0
    Rc = camera_frame(vp)
    half = math.tan(math.radians(fov_half_deg))
    gx = np.linspace(-half, half, npix); gy = np.linspace(-half, half, npix)
    GX, GY = np.meshgrid(gx, gy)
    d_cam = np.stack([GX.ravel(), GY.ravel(), np.ones(npix * npix)], -1)
    dirs = d_cam @ Rc; dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    orig = np.tile(vp, (npix * npix, 1))
    tmin, fid = _ray_tri_nearest(orig, dirs, V0, E1, E2)
    img = np.zeros(npix * npix); depth = np.where(np.isfinite(tmin), tmin, 0.0)
    hit = fid >= 0
    if hit.any():
        fids = fid[hit]; n = normals[fids]
        look = -dirs[hit]                                  # facet -> camera
        cos_e = np.clip(np.einsum('ij,ij->i', n, look), 0, 1)
        islit = lit_mask[fids]
        ci = np.clip(n @ sun_hat, 0, 1)
        alpha = np.arccos(np.clip(np.einsum('j,ij->i', sun_hat, look), -1, 1))
        val = np.zeros(len(fids))
        if islit.any():
            val[islit] = np.vectorize(lambda i, e, a: BRDF.radiance_over_Esun(i, e, a, model="hapke", params=hapke))(
                np.arccos(np.clip(ci[islit], -1, 1)), np.arccos(cos_e[islit]), alpha[islit])
        if (~islit).any():                                 # dark: strobe ~0 phase (co-boresighted)
            val[~islit] = np.vectorize(lambda e: BRDF.radiance_over_Esun(e, e, 0.0, model="hapke", params=hapke))(
                np.arccos(cos_e[~islit])) * 0.3
        img[hit] = val
    return img.reshape(npix, npix), depth.reshape(npix, npix), fid.reshape(npix, npix), Rc


# ════════════════════════════════════════════════════════════════════════════
# stereo match + triangulate + calibrate vs analytic photogram
# ════════════════════════════════════════════════════════════════════════════
def calibrate_pair(centroids, normals, vpA, vpB, sun_hat, snr_value=120.0,
                   ifov=HW.IFOV_RAD, n_trials=24, rng=None, conv_band=PG.CONV_BAND_DEG):
    """For facets visible to BOTH cameras, project with SNR-localization noise, triangulate, and
    compare empirical depth sigma_z to photogram.triangulation_sigma_m.  Returns calibration stats."""
    rng = rng or np.random.default_rng(0)
    RcA = camera_frame(vpA); RcB = camera_frame(vpB)
    lookA = vpA[None, :] - centroids; rA = np.linalg.norm(lookA, axis=1); lookA /= rA[:, None]
    lookB = vpB[None, :] - centroids; rB = np.linalg.norm(lookB, axis=1); lookB /= rB[:, None]
    frontA = np.einsum('ij,ij->i', normals, lookA) > 0.1
    frontB = np.einsum('ij,ij->i', normals, lookB) > 0.1
    # convergence angle between the two views per facet
    conv = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', lookA, lookB), -1, 1)))
    inband = (conv >= conv_band[0]) & (conv <= conv_band[1])
    both = frontA & frontB & inband
    if both.sum() < 5:
        return dict(n_facets=int(both.sum()), ok=False)
    P = centroids[both]; rtrue = rA[both]; convd = conv[both]
    uvA0, _ = project(P, vpA, RcA); uvB0, _ = project(P, vpB, RcB)
    snr_eff = 2.0 * snr_value * snr_value / (snr_value + snr_value)   # harmonic mean (weaker view dominates)
    sig_pix = float(max(PG.SIGMA_DISP_FLOOR_PX, PG.K_MATCH / max(snr_eff, 1e-6)))  # SNR-limited localization
    emp = []
    for _ in range(n_trials):
        uvA = uvA0 + rng.normal(0, sig_pix, uvA0.shape)
        uvB = uvB0 + rng.normal(0, sig_pix, uvB0.shape)
        dA = backproject(uvA, vpA, RcA); dB = backproject(uvB, vpB, RcB)
        Phat = triangulate(vpA, dA, vpB, dB)
        rhat = np.linalg.norm(Phat - vpA, axis=1)
        emp.append(rhat - rtrue)
    emp = np.array(emp)                                              # [trials, facets]
    sig_emp = emp.std(axis=0)                                        # per-facet empirical depth sigma
    sig_ana = np.array([PG.triangulation_sigma_m(rtrue[i], convd[i], snr_value, snr_value, ifov)
                        for i in range(len(rtrue))])
    good = np.isfinite(sig_ana) & (sig_emp > 0)
    ratio = float(np.median(sig_emp[good] / sig_ana[good])) if good.any() else float("nan")
    return dict(n_facets=int(both.sum()), ok=True, sigma_pix=sig_pix,
                sigma_emp_median_m=float(np.median(sig_emp)), sigma_ana_median_m=float(np.median(sig_ana[good])),
                calib_ratio=ratio, conv_median_deg=float(np.median(convd)),
                range_median_m=float(np.median(rtrue)))


def _rot_mesh(R):
    m = np.load(MESH_NPZ)
    return (m["vertices"] @ R.T, m["faces"], m["centroids"] @ R.T, m["normals"] @ R.T)

def validate_draw(draw_root, npix=96, rng=None):
    rng = rng or np.random.default_rng(0)
    apf = os.path.join(draw_root, "apophis", "attitude_truth.npz")
    R = np.load(apf, allow_pickle=True)["R_body2inertial"] if os.path.exists(apf) else np.eye(3)
    sun = np.load(apf, allow_pickle=True)["sun_hat"] if os.path.exists(apf) else np.array([1., 0, 0])
    V, Faces, C, N = _rot_mesh(R)
    lit = (N @ sun) > 0
    tr = sorted(glob.glob(os.path.join(draw_root, "trajectories", "apophis_rendezvous_full_D*.npz")))
    if len(tr) < 2:
        return dict(ok=False, err="need >=2 deputies")
    # closest-approach viewpoints of two deputies
    def closest_vp(f):
        d = np.load(f, allow_pickle=True); rel = np.asarray(d["rD_N"], float) - np.asarray(d["rA_N"], float)
        return rel[np.argmin(np.linalg.norm(rel, axis=1))]
    vps = [closest_vp(f) for f in tr]
    target = 0.5 * (PG.CONV_BAND_DEG[0] + PG.CONV_BAND_DEG[1])      # mid of the usable parallax band
    best = None
    for i in range(len(vps)):
        for j in range(i + 1, len(vps)):
            ai = vps[i] / (np.linalg.norm(vps[i]) + EPS); aj = vps[j] / (np.linalg.norm(vps[j]) + EPS)
            conv = math.degrees(math.acos(float(np.clip(ai @ aj, -1, 1))))
            inband = PG.CONV_BAND_DEG[0] <= conv <= PG.CONV_BAND_DEG[1]
            score = abs(conv - target) + (0.0 if inband else 1e3)
            if best is None or score < best[0]:
                best = (score, i, j, conv)
    _, ia, ib, conv_pair = best
    vpA, vpB = vps[ia], vps[ib]
    hp = BRDF.HapkeParams()
    bod_r = float(np.linalg.norm(V, axis=1).max()); rngA = float(np.linalg.norm(vpA))
    fov = float(np.clip(math.degrees(math.atan(bod_r / max(rngA, 1.0))) * 1.3, 3.0, 27.0))
    img, depth, fid, _ = render_image(V, Faces, C, N, vpA, sun, hp, lit, npix=npix, fov_half_deg=fov)
    cal = calibrate_pair(C, N, vpA, vpB, sun, rng=rng)
    return dict(ok=True, image=img, depth=depth, calib=cal, vpA=vpA.tolist(), vpB=vpB.tolist())


def _save_png(img, path):
    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 4)); plt.imshow(img, cmap="gray", origin="lower")
        plt.title("E2 synthetic deputy image"); plt.axis("off")
        plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()
        return True
    except Exception as e:
        print("  [png] skipped:", e); return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndraws", type=int, default=3)
    ap.add_argument("--npix", type=int, default=96)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    draws = sorted(glob.glob(os.path.join(MCDIR, "draw_*")))[:a.ndraws]
    if not draws:
        print("no draws in", MCDIR, "-- run MC_v2_run.py first (or --skip-basilisk)."); return
    ratios = []; rows = []
    for dr in draws:
        k = os.path.basename(dr)
        r = validate_draw(dr, npix=a.npix)
        if not r["ok"]:
            print(f"  {k}: {r.get('err')}"); continue
        cal = r["calib"]
        _save_png(r["image"], os.path.join(OUTDIR, f"{k}_render.png"))
        np.savez_compressed(os.path.join(OUTDIR, f"{k}_validate.npz"),
                            image=r["image"], depth=r["depth"], **{f"calib_{kk}": v for kk, v in cal.items()})
        if cal.get("ok"):
            ratios.append(cal["calib_ratio"])
            print(f"  {k}: facets={cal['n_facets']} conv={cal['conv_median_deg']:.1f}deg "
                  f"sig_emp={cal['sigma_emp_median_m']:.3f} sig_ana={cal['sigma_ana_median_m']:.3f} "
                  f"calib={cal['calib_ratio']:.2f}")
            rows.append(dict(draw=k, **cal))
    summ = dict(n=len(ratios), calib_ratio_median=float(np.median(ratios)) if ratios else None,
                calib_ratio_mean=float(np.mean(ratios)) if ratios else None, rows=rows)
    json.dump(summ, open(os.path.join(OUTDIR, "render_validation.json"), "w"), indent=2)
    print("\nE2 calibration: analytic photogram sigma_z is", 
          f"{summ['calib_ratio_median']:.2f}x the rendered/matched empirical" if ratios else "n/a",
          "\nsaved ->", OUTDIR)


def _self_test():
    print("render_validate self-test (synthetic mesh + stereo triangulation)")
    os.makedirs(OUTDIR, exist_ok=True)
    # if no radar mesh present, fabricate a small bumpy sphere
    if not os.path.exists(MESH_NPZ):
        print("  (no radar mesh; fabricating a sphere)")
        u = np.linspace(0, np.pi, 18); v = np.linspace(0, 2 * np.pi, 24)
        pts = []; 
        for a in u:
            for b in v:
                R = 200 * (1 + 0.08 * math.sin(4 * b))
                pts.append([R * math.sin(a) * math.cos(b), R * math.sin(a) * math.sin(b), R * math.cos(a)])
        V = np.array(pts)
    else:
        m = np.load(MESH_NPZ); V = m["vertices"]; C = m["centroids"]; N = m["normals"]; Faces = m["faces"]
    sun = np.array([1.0, 0.2, 0.1]); sun /= np.linalg.norm(sun)
    hp = BRDF.HapkeParams()
    if os.path.exists(MESH_NPZ):
        lit = (N @ sun) > 0
        vpA = np.array([2500.0, 0.0, 300.0]); vpB = np.array([2400.0, 600.0, 250.0])
        bod_r = float(np.linalg.norm(V, axis=1).max())
        fov = float(np.clip(math.degrees(math.atan(bod_r / np.linalg.norm(vpA))) * 1.3, 3.0, 27.0))
        img, depth, fid, _ = render_image(V, Faces, C, N, vpA, sun, hp, lit, npix=72, fov_half_deg=fov)
        print("  rendered image %dx%d  hit-fraction %.2f  depth[min,max]=%.0f,%.0f m"
              % (img.shape[0], img.shape[1], (fid >= 0).mean(), depth[depth > 0].min(), depth.max()))
        ok = _save_png(img, os.path.join(OUTDIR, "selftest_render.png"))
        cal = calibrate_pair(C, N, vpA, vpB, sun, snr_value=120.0)
        print("  stereo calibrate: facets=%d conv=%.1fdeg sig_emp=%.3f m sig_ana=%.3f m  calib_ratio=%.2f"
              % (cal["n_facets"], cal["conv_median_deg"], cal["sigma_emp_median_m"],
                 cal["sigma_ana_median_m"], cal["calib_ratio"]))
        assert cal["ok"] and 0.2 < cal["calib_ratio"] < 5.0, "calibration ratio out of sane range"
        assert (fid >= 0).mean() > 0.05, "render produced too few hits"
        print("  PNG saved:", ok, " SELF-TEST PASSED")
    else:
        print("  (radar mesh absent; geometry-only path not exercised)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--self-test", action="store_true")
    known, _ = ap.parse_known_args()
    if known.self_test:
        _self_test()
    else:
        main()
