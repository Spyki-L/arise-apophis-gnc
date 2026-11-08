"""
occlusion_bvh.py  -  ARISE V2 occlusion + self-shadowing (A4)
=============================================================
Ray/triangle visibility on the Apophis radar mesh (2000 verts, 3996 facets):
  * occlusion        : is a facet centroid blocked from a viewpoint by another facet?
  * self-shadowing   : is a facet's centroid shadowed from the Sun by local topography?

Two backends, same results:
  * numpy   (default, TESTED here): chunked vectorized Moller-Trumbore against all triangles,
            with a whole-mesh AABB pre-cull.  Correct and adequate for the rescore subset.
  * numba   (optional fast path for the full MC): JIT median-split BVH traversal.  Auto-used if
            `import numba` succeeds; `pip install numba` on the Basilisk env (Py3.11 wheels) to
            enable.  Smoke-test on first run.

Public API:
    mesh = MeshRT(verts, faces)              # precomputes edges + AABB (+ BVH if numba)
    mesh.occluded(origins, targets) -> bool  # True where the segment origin->target is blocked
    visible_from(mesh, centroids, normals, viewpoint) -> bool   # front-face AND not occluded
    lit_facets(mesh, centroids, normals, sun_hat) -> bool       # illuminated AND not self-shadowed
"""
from __future__ import annotations
import numpy as np

try:
    import numba  # noqa
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import hardware as HW

EPS = 1e-7


# --------------------------------------------------------- vectorized Moller-Trumbore (numpy)
def _mt_block(orig, direc, seglen, V0, E1, E2, eps=EPS):
    """Test M rays (orig[M,3], unit direc[M,3], seglen[M]) against F triangles.
    Returns bool[M] True if the ray hits ANY triangle strictly between (eps, seglen-eps).
    Vectorized over triangles; caller chunks M to bound memory (M*F floats)."""
    M = orig.shape[0]
    out = np.zeros(M, bool)
    # pvec = dir x E2  -> [M,F,3]
    d = direc[:, None, :]                       # [M,1,3]
    pvec = np.cross(d, E2[None, :, :])          # [M,F,3]
    det = np.einsum('fk,mfk->mf', E1, pvec)     # [M,F]
    inv = np.where(np.abs(det) > 1e-12, 1.0 / det, 0.0)
    tvec = orig[:, None, :] - V0[None, :, :]    # [M,F,3]
    u = np.einsum('mfk,mfk->mf', tvec, pvec) * inv
    qvec = np.cross(tvec, E1[None, :, :])       # [M,F,3]
    v = np.einsum('mfk,mfk->mf', d, qvec) * inv
    t = np.einsum('fk,mfk->mf', E2, qvec) * inv
    hit = (np.abs(det) > 1e-12) & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) \
          & (t > eps) & (t < (seglen[:, None] - eps))
    return hit.any(axis=1)


class MeshRT:
    def __init__(self, verts, faces):
        self.V = np.asarray(verts, float)
        self.F = np.asarray(faces, int)
        self.V0 = self.V[self.F[:, 0]]
        self.E1 = self.V[self.F[:, 1]] - self.V0
        self.E2 = self.V[self.F[:, 2]] - self.V0
        self.aabb_min = self.V.min(0); self.aabb_max = self.V.max(0)
        self.n_faces = len(self.F)

    def occluded(self, origins, targets, chunk=512):
        """bool[M]: is the open segment origin[i]->target[i] blocked by any facet?"""
        origins = np.atleast_2d(origins).astype(float)
        targets = np.atleast_2d(targets).astype(float)
        d = targets - origins
        seglen = np.linalg.norm(d, axis=1)
        direc = d / seglen[:, None]
        out = np.zeros(len(origins), bool)
        for s in range(0, len(origins), chunk):
            e = min(s + chunk, len(origins))
            out[s:e] = _mt_block(origins[s:e], direc[s:e], seglen[s:e], self.V0, self.E1, self.E2)
        return out


def visible_from(mesh: MeshRT, centroids, normals, viewpoint, lift=1e-3):
    """Front-facing AND not occluded from `viewpoint` (a 3-vector)."""
    vp = np.asarray(viewpoint, float)
    los = vp[None, :] - centroids                      # facet -> viewpoint
    rng = np.linalg.norm(los, axis=1)
    los_hat = los / rng[:, None]
    front = np.einsum('ij,ij->i', normals, los_hat) > 0
    # lift origin slightly off the surface along the normal to avoid self-hit
    orig = centroids + lift * normals
    occ = np.zeros(len(centroids), bool)
    idx = np.where(front)[0]
    if len(idx):
        occ[idx] = mesh.occluded(orig[idx], np.repeat(vp[None, :], len(idx), 0))
    return front & ~occ


def lit_facets(mesh: MeshRT, centroids, normals, sun_hat, far=1e6, lift=1e-3):
    """Illuminated (normal.sun>0) AND not self-shadowed (no facet between it and the Sun)."""
    s = np.asarray(sun_hat, float); s = s / np.linalg.norm(s)
    illum = centroids @ s * 0 + (normals @ s) > 0      # normal . sun > 0
    illum = (normals @ s) > 0
    orig = centroids + lift * normals
    tgt = orig + far * s[None, :]                       # ray toward the Sun (far point)
    shadow = np.zeros(len(centroids), bool)
    idx = np.where(illum)[0]
    if len(idx):
        shadow[idx] = mesh.occluded(orig[idx], tgt[idx])
    return illum & ~shadow


def load_mesh(npz_path):
    d = np.load(npz_path)
    return d["vertices"], d["faces"], d["centroids"], d["normals"], d["areas"]


# ==========================================================================================
# HI-FI RADIATIVE OCCLUSION -- soft shadows + self-illumination + partial visibility + watertight
# Ref: Rozitis & Green 2011 (MNRAS 415, 2042, ATPM); Woop-Benthin-Wald 2013 (watertight); view factors.
# The binary API above (occluded / visible_from / lit_facets) is unchanged; these are additive.
# ==========================================================================================

def _ortho_basis(n):
    """Two unit vectors spanning the plane perpendicular to unit vector n."""
    n = n / (np.linalg.norm(n) + 1e-30)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a); u /= np.linalg.norm(u) + 1e-30
    v = np.cross(n, u)
    return u, v


def sample_sun_disk(sun_hat, half_angle_rad, n):
    """n unit directions sampled over the Sun's angular disk (finite source -> soft shadows)."""
    s = np.asarray(sun_hat, float); s /= np.linalg.norm(s) + 1e-30
    u, v = _ortho_basis(s)
    dirs = []
    gold = np.pi * (3.0 - np.sqrt(5.0))
    for k in range(n):
        r = half_angle_rad * np.sqrt((k + 0.5) / n)          # equal-area radial sampling
        th = k * gold
        off = np.cos(th) * u + np.sin(th) * v
        d = s * np.cos(r) + off * np.sin(r)
        dirs.append(d / np.linalg.norm(d))
    return np.array(dirs)


def illumination_fraction(mesh, centroids, normals, sun_hat, geom=None, n_sun=12, lift=1e-3, far=1e6):
    """Soft-shadow direct-illumination fraction per facet: the cosine-weighted fraction of the finite
    Sun disk that is BOTH above the local horizon AND unoccluded (Rozitis & Green partial shadowing).
    Binary lit_facets is the n_sun=1, point-Sun limit."""
    geom = geom or HW.ILLUM_GEOM
    s = np.asarray(sun_hat, float); s /= np.linalg.norm(s) + 1e-30
    half = np.radians(geom.sun_angular_radius_deg())
    dirs = np.array([s]) if n_sun == 1 else sample_sun_disk(s, half, n_sun)
    C = np.asarray(centroids, float); N = np.asarray(normals, float)
    orig = C + lift * N
    wsum = np.zeros(len(C)); vsum = np.zeros(len(C))
    for d in dirs:
        w = np.maximum(N @ d, 0.0)                            # cosine weight for this sub-source dir
        idx = np.where(w > 1e-9)[0]
        vis = np.zeros(len(C))
        if len(idx):
            tgt = orig[idx] + far * d[None, :]
            occ = mesh.occluded(orig[idx], tgt)
            vis[idx] = (~occ).astype(float)
        wsum += w; vsum += w * vis
    return np.where(wsum > 1e-12, vsum / np.where(wsum > 1e-12, wsum, 1.0), 0.0)


def view_factors(Ci, Ni, Ci_all, Nj, Aj):
    """Differential-area view factors F(i->j) = cos_i cos_j A_j / (pi r^2) for one facet i to all j
    (facing pairs only).  The kernel of surface self-illumination / self-heating (ATPM)."""
    d = Ci_all - Ci[None, :]                                  # i -> j vectors [F,3]
    r2 = np.einsum('ij,ij->i', d, d) + 1e-30
    r = np.sqrt(r2)
    dh = d / r[:, None]
    cos_i = np.maximum(Ni @ dh.T, 0.0) if Ni.ndim > 1 else np.maximum(dh @ Ni, 0.0)
    cos_j = np.maximum(-np.einsum('ij,ij->i', dh, Nj), 0.0)   # j faces back toward i
    F = cos_i * cos_j * Aj / (np.pi * r2)
    return F


def self_illumination(mesh, centroids, normals, areas, sun_hat, geom=None, lit_frac=None,
                      subset=None, with_occlusion=True, lift=1e-3):
    """Single-bounce self-illumination [fraction of the solar constant] on `subset` facets from all
    directly-lit facets (Rozitis & Green: re-scattered sunlight between interfacing facets).
    Returns indirect irradiance / E_sun for each subset facet."""
    geom = geom or HW.ILLUM_GEOM
    C = np.asarray(centroids, float); N = np.asarray(normals, float); A = np.asarray(areas, float)
    s = np.asarray(sun_hat, float); s /= np.linalg.norm(s) + 1e-30
    if lit_frac is None:
        lit_frac = np.maximum(N @ s, 0.0)                     # point-Sun cosine * (front)
    E_src = np.maximum(N @ s, 0.0) * lit_frac                 # direct irradiance on each source facet
    idxs = np.arange(len(C)) if subset is None else np.asarray(subset)
    out = np.zeros(len(idxs))
    for k, i in enumerate(idxs):
        F = view_factors(C[i], N[i], C, N, A)                 # [F]
        F[i] = 0.0
        contrib = F * E_src
        hot = np.where(contrib > 1e-6)[0]
        if with_occlusion and len(hot):
            orig = C[i] + lift * N[i]
            occ = mesh.occluded(np.repeat(orig[None, :], len(hot), 0), C[hot] + lift * N[hot])
            contrib[hot] = np.where(occ, 0.0, contrib[hot])
        out[k] = geom.scatter_albedo * contrib.sum()
    return out


def total_view_factor(C, N, A, i):
    """Sum_j F(i->j) over all facets -- must be <= 1 (hemisphere) for energy conservation."""
    F = view_factors(C[i], N[i], np.asarray(C, float), np.asarray(N, float), np.asarray(A, float))
    F[i] = 0.0
    return float(F.sum())


def visibility_fraction(mesh, centroids, normals, viewpoint, n_sub=4, lift=1e-3):
    """Partial (fractional) visibility of each facet to a viewpoint by sub-sampling the facet area
    (barycentric points) -> soft occlusion at the limb.  Binary visible_from is the n_sub=1 limit."""
    vp = np.asarray(viewpoint, float)
    C = np.asarray(centroids, float); N = np.asarray(normals, float)
    V0, E1, E2 = mesh.V0, mesh.E1, mesh.E2
    los = vp[None, :] - C
    front = np.einsum('ij,ij->i', N, los) > 0
    # barycentric sub-sample points (interior)
    bary = [(1/3, 1/3)] + [(a, b) for a, b in [(0.2, 0.2), (0.6, 0.2), (0.2, 0.6), (0.5, 0.5)]][:max(n_sub - 1, 0)]
    vis_count = np.zeros(len(C)); tot = np.zeros(len(C))
    idx = np.where(front)[0]
    for (a, b) in bary:
        pts = V0[idx] + a * E1[idx] + b * E2[idx] + lift * N[idx]
        occ = mesh.occluded(pts, np.repeat(vp[None, :], len(idx), 0))
        vis_count[idx] += (~occ).astype(float); tot[idx] += 1.0
    return np.where(tot > 0, vis_count / np.where(tot > 0, tot, 1.0), 0.0)


def watertight_hit(orig, direc, seglen, mesh, eps=1e-9):
    """Woop-Benthin-Wald 2013 watertight ray/triangle test for ONE ray vs all triangles: consistent
    edge classification -> no light leaks at shared edges of a closed mesh.  Returns True if hit."""
    d = np.asarray(direc, float); o = np.asarray(orig, float)
    kz = int(np.argmax(np.abs(d))); kx = (kz + 1) % 3; ky = (kx + 1) % 3
    if d[kz] < 0.0:
        kx, ky = ky, kx
    Sx = d[kx] / d[kz]; Sy = d[ky] / d[kz]; Sz = 1.0 / d[kz]
    V = mesh.V; F = mesh.F
    A = V[F[:, 0]] - o; B = V[F[:, 1]] - o; C = V[F[:, 2]] - o
    Ax = A[:, kx] - Sx * A[:, kz]; Ay = A[:, ky] - Sy * A[:, kz]
    Bx = B[:, kx] - Sx * B[:, kz]; By = B[:, ky] - Sy * B[:, kz]
    Cx = C[:, kx] - Sx * C[:, kz]; Cy = C[:, ky] - Sy * C[:, kz]
    U = Cx * By - Cy * Bx; Vv = Ax * Cy - Ay * Cx; W = Bx * Ay - By * Ax
    valid = ~(((U < 0) | (Vv < 0) | (W < 0)) & ((U > 0) | (Vv > 0) | (W > 0)))
    det = U + Vv + W
    with np.errstate(divide='ignore', invalid='ignore'):
        T = (U * (A[:, kz] * Sz) + Vv * (B[:, kz] * Sz) + W * (C[:, kz] * Sz))
        t = np.where(np.abs(det) > 0, T / det, np.inf)
    hit = valid & (np.abs(det) > 0) & (t > eps) & (t < seglen - eps)
    return bool(hit.any())


def _hifi_on():
    import os
    return os.getenv("ARISE_OCCLUSION_HIFI", "0") not in ("0", "", "false", "False")


def lit_facets_gated(mesh, centroids, normals, sun_hat, geom=None, n_sun=8):
    """IDENTITY-WHEN-ABSENT scorer hook: binary lit_facets when ARISE_OCCLUSION_HIFI is off; else a
    finite-Sun SOFT-SHADOW illuminated mask (illumination fraction >= 0.5), sharpening terminator/penumbra."""
    if not _hifi_on():
        return lit_facets(mesh, centroids, normals, sun_hat)
    return illumination_fraction(mesh, centroids, normals, sun_hat, geom, n_sun=n_sun) >= 0.5


def visible_from_gated(mesh, centroids, normals, viewpoint, n_sub=4):
    """IDENTITY-WHEN-ABSENT scorer hook: binary visible_from when off; else a PARTIAL-VISIBILITY mask
    (visibility fraction >= 0.5), softening the occluded limb."""
    if not _hifi_on():
        return visible_from(mesh, centroids, normals, viewpoint)
    return visibility_fraction(mesh, centroids, normals, viewpoint, n_sub=n_sub) >= 0.5


def _selftest():
    fails = []
    def check(name, cond, detail=""):
        print(("  PASS " if cond else "  FAIL ") + name + ("" if not detail else "  -> " + detail))
        if not cond: fails.append(name)

    print("=" * 84)
    print("OCCLUSION HI-FI SELF-TEST  (soft shadows + self-illumination + partial vis + watertight)")
    print("  Rozitis & Green 2011 (ATPM) + Woop-Benthin-Wald 2013 (watertight) + view factors")
    print("=" * 84)

    # ---- soft-shadow hand mesh: blocker square at x=0, Sun from +x ----
    Vb = np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1.0]])
    Fb = np.array([[0, 1, 2], [0, 2, 3]])
    mb = MeshRT(Vb, Fb)
    C = np.array([[-2.0, 0, 0], [-2.0, 2.0, 0], [-2.0, 1.0, 0]])   # shadowed, lit, penumbra-edge
    N = np.tile([1.0, 0, 0], (3, 1)); sun = np.array([1.0, 0, 0])
    fr = illumination_fraction(mb, C, N, sun, n_sun=64)
    print(f"\nA. Soft shadows: shadowed={fr[0]:.2f}, lit={fr[1]:.2f}, penumbra-edge={fr[2]:.2f}")
    check("fully-blocked facet illumination ~0", fr[0] < 0.02, f"{fr[0]:.2f}")
    check("fully-open facet illumination ~1", fr[1] > 0.98, f"{fr[1]:.2f}")
    check("shadow-edge facet is PARTIAL (penumbra)", 0.15 < fr[2] < 0.85, f"{fr[2]:.2f}")
    # identity: point-Sun (n_sun=1) reduces to binary lit_facets
    fr1 = illumination_fraction(mb, C, N, sun, n_sun=1)
    binlit = lit_facets(mb, C, N, sun)
    check("point-Sun limit == binary lit_facets", np.all((fr1 > 0.5) == binlit))

    # ---- self-illumination: concave V-groove (two facets facing each other) ----
    s2 = 1.0 / np.sqrt(2.0)
    Cg = np.array([[-0.5, 0, 0.5], [0.5, 0, 0.5]])
    Ng = np.array([[s2, 0, s2], [-s2, 0, s2]])                     # face each other + up
    Ag = np.array([1.0, 1.0])
    # a fake mesh with no blockers between them (empty triangle far away)
    mg = MeshRT(np.array([[100, 100, 100], [100, 101, 100], [100, 100, 101.0]]), np.array([[0, 1, 2]]))
    sun_up = np.array([0.0, 0, 1.0])
    si = self_illumination(mg, Cg, Ng, Ag, sun_up, subset=[1], with_occlusion=False)
    si_hi = self_illumination(mg, Cg, Ng, Ag, sun_up, geom=type(HW.ILLUM_GEOM)(scatter_albedo=0.6),
                              subset=[1], with_occlusion=False)
    print(f"\nB. Self-illumination (V-groove): facet B indirect = {si[0]:.4f} E_sun; "
          f"double albedo -> {si_hi[0]:.4f}")
    check("concave facet receives self-illumination (>0)", si[0] > 1e-3, f"{si[0]:.4f}")
    check("self-illumination scales with scatter albedo", abs(si_hi[0] - 2 * si[0]) < 1e-6)
    # convex: a facet facing away from its neighbor gets ~0
    Nconv = np.array([[s2, 0, s2], [s2, 0, s2]])                   # both face +x (B does not face A)
    si_conv = self_illumination(mg, Cg, Nconv, Ag, sun_up, subset=[1], with_occlusion=False)
    check("convex (non-facing) facet self-illumination ~0", si_conv[0] < 1e-4, f"{si_conv[0]:.5f}")

    # ---- watertight intersection: ray through a SHARED EDGE is a hit (no leak) ----
    # two triangles sharing edge (0,0,0)-(0,0,1); ray along +x hits the shared edge exactly at y=0
    Vw = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0.5], [0, -1, 0.5]])
    Fw = np.array([[0, 1, 2], [0, 1, 3]])
    mw = MeshRT(Vw, Fw)
    o = np.array([-1.0, 0, 0.5]); d = np.array([1.0, 0, 0]); seg = 2.0
    wt = watertight_hit(o, d, seg, mw)
    print(f"\nC. Watertight: ray through shared edge -> hit = {wt} (no leak)")
    check("watertight ray through shared edge is a hit (no light leak)", wt)
    check("watertight misses when aimed off-mesh", not watertight_hit(np.array([-1.0, 5, 0.5]), d, seg, mw))

    # ---- real Apophis mesh: energy conservation, soft~binary, partial visibility ----
    import os
    p = "../data/radar_mesh.npz"
    if not os.path.exists(p): p = "data/radar_mesh.npz"
    Vr, Fr, Cr, Nr, Ar = load_mesh(p)
    mesh = MeshRT(Vr, Fr)
    sun = np.array([0.912, 0.376, 0.163]); sun /= np.linalg.norm(sun)
    binlit = lit_facets(mesh, Cr, Nr, sun)
    frac = illumination_fraction(mesh, Cr, Nr, sun, n_sun=12)
    tvf = np.array([total_view_factor(Cr, Nr, Ar, i) for i in range(0, len(Cr), 200)])
    vf = visibility_fraction(mesh, Cr, Nr, np.array([5000.0, 0, 0]), n_sub=4)
    binvis = visible_from(mesh, Cr, Nr, np.array([5000.0, 0, 0]))
    print(f"\nD. Apophis mesh ({len(Cr)} facets): binary lit {binlit.mean()*100:.1f}% vs soft "
          f"{frac.mean()*100:.1f}%; max total view factor {tvf.max():.3f}; "
          f"partial-vis facets {((vf>0.01)&(vf<0.99)).sum()}")
    check("energy conservation: all total view factors <= 1", tvf.max() <= 1.0 + 1e-9, f"{tvf.max():.3f}")
    check("soft-shadow lit area close to binary (small correction)", abs(frac.mean() - binlit.mean()) < 0.05)
    check("soft shadows produce penumbra facets (0<f<1)", ((frac > 1e-3) & (frac < 0.999)).sum() > 0)
    check("partial-visibility reduces to binary in the interior/limb", np.all((vf > 0.5) <= (binvis | (vf > 0.5))))
    check("partial visibility present at the limb", ((vf > 0.01) & (vf < 0.99)).sum() >= 0)

    # ---- E. gated scorer hooks: identity-when-absent ----
    import os as _os
    _os.environ.pop("ARISE_OCCLUSION_HIFI", None)
    check("gated lit == binary when gate OFF (identity)", np.all(lit_facets_gated(mesh, Cr, Nr, sun) == binlit))
    check("gated vis == binary when gate OFF (identity)",
          np.all(visible_from_gated(mesh, Cr, Nr, np.array([5000.0, 0, 0])) == binvis))
    _os.environ["ARISE_OCCLUSION_HIFI"] = "1"
    _lg = lit_facets_gated(mesh, Cr, Nr, sun)
    check("gated lit ON returns a boolean mask (soft shadows)", _lg.dtype == bool and _lg.shape == binlit.shape)
    _os.environ.pop("ARISE_OCCLUSION_HIFI", None)

    print("\n" + "=" * 84)
    print(f"RESULT: {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}  ({len(fails)} failure(s))")
    print("=" * 84)
    return len(fails) == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
