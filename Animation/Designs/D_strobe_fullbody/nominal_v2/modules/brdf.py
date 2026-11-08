"""
brdf.py  -  ARISE V2 photometric BRDF (A5)  [UPGRADED to the full Hapke 2012 model]
==============================================================================
Phase-angle-dependent regolith photometry for the Apophis coverage scorer.  Upgraded from the
simplified Hapke to the modern NINE-PARAMETER Hapke (2012) model, following:

  Villa, McMahon & Nesnas, "Image Rendering and Terrain Generation of Planetary Surfaces Using
  Source-Available Tools," AAS 23-122 (Appendix A/B) -- the Hapke formulation used for
  physically-based rendering of asteroids/Moon/comets.

Models:
  * Lambert          - cosine reference.
  * Lommel-Seeliger  - single-scattering, dark regolith (no free params).
  * Hapke (2012)     - full bidirectional model with all nine terms:
      w (single-scattering albedo), b,c (double-lobe Henyey-Greenstein phase), BS0,hS (Shadow-
      Hiding Opposition Effect), theta_bar (macroscopic ROUGHNESS -> shadowing S + effective
      angles), BC0,hC (Coherent-Backscatter Opposition Effect), phi (filling factor -> porosity K).
    The previous simplified version is kept as model="hapke_simple" for comparison/ablation.

API (illumination-inclusive radiance factor):
    radiance_over_Esun(i, e, alpha, model=...) -> L / E_sun   [1/sr]   (cos(incidence) INCLUDED)

Conventions: i incidence (Sun..normal), e emission (normal..camera), alpha=g phase angle.
Defaults are representative S/Sq-complex (Apophis) values; tagged ASSUMED, override per draw.
    Refs: Hapke (2012) "Theory of Reflectance and Emittance Spectroscopy"; Sato et al. (2014);
          Villa et al. (AAS 23-122); Pravec et al. (2014) Apophis photometry.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HapkeParams:
    # --- 9 Hapke (2012) free parameters ---
    w: float = 0.23          # single-scattering albedo (S-complex; ASSUMED)
    b: float = 0.35          # HG double-lobe shape (0..1)
    c: float = 0.30          # HG backward/forward lobe strength (-1..1; >0 backscatter)
    BS0: float = 1.5         # SHOE amplitude (ASSUMED)
    hS: float = 0.05         # SHOE angular width (ASSUMED)
    theta_bar_deg: float = 20.0  # macroscopic roughness mean-slope angle (NOW APPLIED)
    BC0: float = 0.3         # CBOE amplitude (ASSUMED; small for S-type)
    hC: float = 0.02         # CBOE angular width (ASSUMED)
    phi: float = 0.30        # filling factor = 1 - porosity (ASSUMED)
    # legacy alias (old single-lobe HG asymmetry) kept for the "hapke_simple" path
    g_hg: float = -0.35


# ---------------- phase functions & opposition ----------------
def _double_hg(alpha_rad, b, c):
    """Henyey-Greenstein DOUBLE-lobe single-particle phase function p(g)  [Villa Eq 12]."""
    cg = np.cos(alpha_rad); b2 = b * b
    back = (1.0 - b2) / np.power(1.0 - 2.0 * b * cg + b2, 1.5)
    fwd = (1.0 - b2) / np.power(1.0 + 2.0 * b * cg + b2, 1.5)
    return 0.5 * (1.0 + c) * back + 0.5 * (1.0 - c) * fwd


def _hg_phase(alpha_rad, g):
    """Legacy single-term HG (for hapke_simple)."""
    ca = np.cos(alpha_rad)
    return (1.0 - g * g) / np.power(1.0 + 2.0 * g * ca + g * g, 1.5)


def _shoe(alpha_rad, BS0, hS):
    """Shadow-Hiding Opposition Effect B_S(g)  [Villa Eq 13]."""
    t = np.tan(0.5 * alpha_rad)
    return 1.0 / (1.0 + t / max(hS, 1e-9))


def _cboe(alpha_rad, hC):
    """Coherent-Backscatter Opposition Effect B_C(g)  [Villa Eq 18]."""
    x = np.tan(0.5 * alpha_rad) / max(hC, 1e-9)
    x = np.clip(x, 1e-9, 1e9)
    return (1.0 + (1.0 - np.exp(-x)) / x) / (2.0 * (1.0 + x) ** 2)


def _H(x, w):
    """Ambartsumian-Chandrasekhar H-function (Hapke 2002 approx)  [Villa Eq 16]."""
    x = np.asarray(x, float)
    r0 = (1.0 - np.sqrt(1.0 - w)) / (1.0 + np.sqrt(1.0 - w))
    with np.errstate(divide="ignore", invalid="ignore"):
        ln = np.log((1.0 + x) / np.maximum(x, 1e-12))
        inv = 1.0 - w * x * (r0 + 0.5 * (1.0 - 2.0 * r0 * x) * ln)
    return 1.0 / np.maximum(inv, 1e-9)


def _porosity_K(phi):
    """Porosity factor K from the filling factor phi  [Villa Eq 15]."""
    phi = float(np.clip(phi, 1e-3, 0.74))
    x = 1.209 * phi ** (2.0 / 3.0)
    return -np.log(1.0 - x) / x


# ---------------- macroscopic roughness (Hapke 1984) [Villa Eq 19-31] ----------------
def _roughness(i_rad, e_rad, alpha_rad, theta_bar_rad):
    """Returns (mu0e, mue, S): roughness-corrected effective cosines + shadowing function.
    Reduces to (cos i, cos e, 1) as theta_bar -> 0."""
    tb = float(theta_bar_rad)
    mu0 = np.clip(np.cos(i_rad), 1e-6, 1.0); mu = np.clip(np.cos(e_rad), 1e-6, 1.0)
    if tb < 1e-6:
        return mu0, mu, np.ones_like(np.asarray(alpha_rad, float))
    i = np.clip(i_rad, 1e-4, np.pi/2 - 1e-4); e = np.clip(e_rad, 1e-4, np.pi/2 - 1e-4)
    cg = np.clip(np.cos(alpha_rad), -1, 1)
    # azimuth psi from i,e,g  [Eq 31]
    denom = np.maximum(np.sin(e) * np.sin(i), 1e-9)
    cpsi = np.clip((cg - np.cos(e) * np.cos(i)) / denom, -1, 1)
    psi = np.arccos(cpsi)
    tanb = np.tan(tb); cott = 1.0 / max(tanb, 1e-9)
    chi = 1.0 / np.sqrt(1.0 + np.pi * tanb ** 2)                                   # [Eq 25]
    def E1(y): return np.exp(-2.0 / np.pi * cott / np.tan(y))                      # [Eq 27]
    def E2(y): return np.exp(-1.0 / np.pi * cott ** 2 / np.tan(y) ** 2)            # [Eq 28]
    def eta(y): return chi * (np.cos(y) + np.sin(y) * tanb * E2(y) / np.maximum(2.0 - E1(y), 1e-9))  # [Eq 26]
    f = np.exp(-2.0 * np.tan(0.5 * psi))                                           # [Eq 29]
    s2 = np.sin(0.5 * psi) ** 2
    E1i, E1e, E2i, E2e = E1(i), E1(e), E2(i), E2(e)
    etai, etae = eta(i), eta(e)
    i_le_e = (i <= e)
    # i <= e   [Eq 20-22]
    den_a = np.maximum(2.0 - E1e - (psi/np.pi) * E1i, 1e-9)
    mu0e_a = chi * (np.cos(i) + np.sin(i) * tanb * (cpsi * E2e + s2 * E2i) / den_a)
    mue_a = chi * (np.cos(e) + np.sin(e) * tanb * (E2e - s2 * E2i) / den_a)
    S_a = (mue_a/etae) * (mu0/etai) * chi / np.maximum(1.0 - f + f * chi * (mu0/etai), 1e-9)
    # e <= i   [Eq 23-24, 22]
    den_b = np.maximum(2.0 - E1i - (psi/np.pi) * E1e, 1e-9)
    mu0e_b = chi * (np.cos(i) + np.sin(i) * tanb * (E2i - s2 * E2e) / den_b)
    mue_b = chi * (np.cos(e) + np.sin(e) * tanb * (cpsi * E2i + s2 * E2e) / den_b)
    S_b = (mue_b/etae) * (mu0/etai) * chi / np.maximum(1.0 - f + f * chi * (mu/etae), 1e-9)
    mu0e = np.where(i_le_e, mu0e_a, mu0e_b)
    mue = np.where(i_le_e, mue_a, mue_b)
    S = np.where(i_le_e, S_a, S_b)
    return (np.clip(mu0e, 1e-6, 1.0), np.clip(mue, 1e-6, 1.0), np.clip(S, 0.0, None))


def _mu(i_rad, e_rad):
    return np.clip(np.cos(i_rad), 0.0, 1.0), np.clip(np.cos(e_rad), 1e-6, 1.0)


def lommel_seeliger(i_rad, e_rad, alpha_rad, w=0.23, g_hg=-0.35):
    mu0, mu = _mu(i_rad, e_rad); p = _hg_phase(alpha_rad, g_hg)
    return (w / (4.0 * np.pi)) * (mu0 / (mu0 + mu)) * p


def hapke_simple(i_rad, e_rad, alpha_rad, p: HapkeParams = HapkeParams()):
    """Previous simplified Hapke (single-lobe HG + SHOE + H-function; no roughness/CBOE)."""
    mu0, mu = _mu(i_rad, e_rad)
    pfun = _hg_phase(alpha_rad, p.g_hg); B = _shoe(alpha_rad, p.BS0, p.hS)
    core = (1.0 + B) * pfun + _H(mu0, p.w) * _H(mu, p.w) - 1.0
    return (p.w / (4.0 * np.pi)) * (mu0 / (mu0 + mu)) * core


def hapke(i_rad, e_rad, alpha_rad, p: HapkeParams = HapkeParams()):
    """FULL Hapke (2012) bidirectional reflectance [1/sr], illumination-inclusive  [Villa Eq 7]:
      r = (w/4pi) K (mu0e/(mu0e+mue)) [ (1+BS0 BS) p(g) + M(mu0e,mue) ] (1+BC0 BC) S(i,e,psi)."""
    i_rad = np.asarray(i_rad, float); e_rad = np.asarray(e_rad, float); alpha_rad = np.asarray(alpha_rad, float)
    K = _porosity_K(p.phi)
    tb = np.radians(p.theta_bar_deg)
    mu0e, mue, S = _roughness(i_rad, e_rad, alpha_rad, tb)
    pfun = _double_hg(alpha_rad, p.b, p.c)
    BS = _shoe(alpha_rad, p.BS0, p.hS)
    BC = _cboe(alpha_rad, p.hC)
    M = _H(mu0e / K, p.w) * _H(mue / K, p.w) - 1.0                     # IMSA w/ porosity [Eq 14]
    LS = mu0e / (mu0e + mue)                                           # LS(ie,ee)        [Eq 11]
    r = (p.w / (4.0 * np.pi)) * K * LS * ((1.0 + p.BS0 * BS) * pfun + M) * (1.0 + p.BC0 * BC) * S
    return np.clip(r, 0.0, None)


def lambert(i_rad, e_rad, alpha_rad, albedo=0.30):
    mu0, _ = _mu(i_rad, e_rad)
    return (albedo / np.pi) * mu0


def radiance_over_Esun(i_rad, e_rad, alpha_rad, model="hapke", **kw):
    if model == "hapke":
        return hapke(i_rad, e_rad, alpha_rad, kw.get("params", HapkeParams()))
    if model == "hapke_simple":
        return hapke_simple(i_rad, e_rad, alpha_rad, kw.get("params", HapkeParams()))
    if model in ("ls", "lommel", "lommel_seeliger"):
        return lommel_seeliger(i_rad, e_rad, alpha_rad, kw.get("w", 0.23), kw.get("g_hg", -0.35))
    if model == "lambert":
        return lambert(i_rad, e_rad, alpha_rad, kw.get("albedo", 0.30))
    raise ValueError("unknown BRDF model: %s" % model)


def sample_hapke(rng) -> HapkeParams:
    """Draw a 9-parameter Hapke set within plausible S/Sq bars (for the MC)."""
    return HapkeParams(
        w=float(np.clip(rng.normal(0.23, 0.04), 0.10, 0.45)),
        b=float(np.clip(rng.normal(0.35, 0.06), 0.10, 0.60)),
        c=float(np.clip(rng.normal(0.30, 0.10), -0.20, 0.70)),
        BS0=float(np.clip(rng.normal(1.5, 0.4), 0.5, 2.5)),
        hS=float(np.clip(rng.normal(0.05, 0.02), 0.01, 0.15)),
        theta_bar_deg=float(np.clip(rng.normal(20.0, 5.0), 5.0, 35.0)),
        BC0=float(np.clip(rng.normal(0.3, 0.15), 0.0, 0.8)),
        hC=float(np.clip(rng.normal(0.02, 0.01), 0.005, 0.06)),
        phi=float(np.clip(rng.normal(0.30, 0.06), 0.10, 0.55)),
    )


if __name__ == "__main__":
    d = np.radians; fails = []
    def chk(n, c): print(("  PASS " if c else "  FAIL ") + n); (fails.append(n) if not c else None)
    print("ARISE V2 BRDF (full Hapke 2012) sanity")
    print("  geometry            lambert   L-S      hapke2012  hapke_simple")
    for (i, e, a) in [(0, 0, 0), (30, 10, 25), (60, 30, 75), (80, 50, 110)]:
        print("  i=%2d e=%2d a=%3d :   %.4f  %.4f  %.4f   %.4f"
              % (i, e, a, lambert(d(i),d(e),d(a)), lommel_seeliger(d(i),d(e),d(a)),
                 hapke(d(i),d(e),d(a)), hapke_simple(d(i),d(e),d(a))))
    print()
    # 1. roughness reduces to no-roughness at theta_bar=0
    mu0e, mue, S = _roughness(d(40), d(20), d(30), 0.0)
    chk("roughness S=1 and mu0e=cos i at theta_bar=0", abs(S-1)<1e-9 and abs(mu0e-np.cos(d(40)))<1e-9)
    # 2. roughness darkens at high phase (grazing) vs smooth
    r_rough = hapke(d(70), d(40), d(100), HapkeParams(theta_bar_deg=30))
    r_smooth = hapke(d(70), d(40), d(100), HapkeParams(theta_bar_deg=0))
    print("  high-phase(100deg) rough=%.4f smooth=%.4f" % (r_rough, r_smooth))
    chk("macroscopic roughness darkens the grazing/high-phase surface", r_rough < r_smooth)
    # 3. opposition surge: brightens sharply toward alpha->0 (SHOE+CBOE)
    r0, r2 = hapke(0,0,d(0)), hapke(0,0,d(4))
    print("  opposition: hapke(a=0)=%.4f hapke(a=4deg)=%.4f  surge=%.2fx" % (r0, r2, r0/r2))
    chk("opposition surge present (a=0 brighter than a=4deg)", r0 > r2 * 1.15)
    # 4. CBOE adds extra surge at very low phase
    p_no = HapkeParams(BC0=0.0); p_cb = HapkeParams(BC0=0.6)
    chk("CBOE adds low-phase brightening", hapke(0,0,d(0.5),p_cb) > hapke(0,0,d(0.5),p_no))
    # 5. positivity/finite across geometries
    ok = all(np.isfinite(hapke(d(i),d(e),d(a))) and hapke(d(i),d(e),d(a))>=0
             for i in (0,30,60,85) for e in (0,20,50,80) for a in (0,10,60,120,160))
    chk("finite & non-negative over all geometries", ok)
    # 6. vectorized
    ii=d(np.array([10.,40.,70.])); ee=d(np.array([5.,20.,50.])); aa=d(np.array([15.,60.,120.]))
    chk("vectorized call works", np.asarray(hapke(ii,ee,aa)).shape==(3,))
    # 7. backward compat: sample_hapke -> HapkeParams usable
    import numpy as _np; hp=sample_hapke(_np.random.default_rng(0))
    chk("sample_hapke draws a usable 9-param set", isinstance(hp,HapkeParams) and _np.isfinite(hapke(d(30),d(10),d(20),hp)))
    print("\nRESULT:", "ALL PASS" if not fails else "FAILED: "+",".join(fails))
    import sys; sys.exit(0 if not fails else 1)
