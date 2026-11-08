"""
spacecraft_sbrdf.py  -  ARISE V2 spacecraft spectral-BRDF model (deputy peer-vision)
==============================================================================
A NEW module (does NOT touch the asteroid brdf.py / Hapke).  It models how an ARISE DEPUTY
(a spacecraft) reflects sunlight -- for PEER-VISION FDIR (a healthy deputy re-imaging a
suspect) and stereophotogrammetry of the fleet -- following:

  [1] A. Willison & D. Bedard, "A novel approach to modeling spacecraft spectral reflectance,"
      Advances in Space Research 58 (2016) 1318-1330.

Method (per [1]):
  * Each homogeneous material has a SPECTRAL BRDF (sBRDF) f_r(theta_i, dtheta, phi, lambda) in a
    SPECULAR-REFERENCED geometry: illumination polar angle theta_i, off-specular in-plane angle
    dtheta, and from-illumination-plane angle phi.  Here modeled as diffuse (Lambertian) + a
    specular micro-lobe with a per-material spectral reflectance (the paper uses measured LUTs;
    these physically-based curves are drop-in replacements until lab LUTs are loaded).
  * The overall spacecraft sBRDF (paper Eq 4) is the projected-area-weighted average of the
    facet material sBRDFs:   f_overall = sum_j(a2D_j * f_material_j) / sum_j(a2D_j).
  * Everything is spectral (lambda), so peer-vision can use color/filters for material ID and
    anomaly detection (a tumbling / MLI-torn / panel-cracked deputy deviates from its signature).

Materials (CanX-1 heritage + XB6): aluminum 6061-T6, triple-junction solar cell, A276 white
paint, aluminized-kapton MLI, black paint.

Run `python spacecraft_sbrdf.py` for the self-test (prints PASS/FAIL).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import numpy as np

DEG = np.pi / 180.0
H_PLANCK = 6.62607e-34; C_LIGHT = 2.998e8
E_SUN_WM2 = 1361.0                       # solar irradiance ~1 AU [W/m^2] (Apophis flyby is ~1 AU)
BAND_NM = (400.0, 1000.0)                # CIS120 imaging band


# ======================================================================================
# Spectral material library  (diffuse rho_d(lambda) + specular rho_s(lambda) + lobe width)
# ======================================================================================
@dataclass(frozen=True)
class Material:
    name: str
    rho_d_vis: float          # diffuse reflectance anchor @550 nm
    rho_s: float              # specular-lobe peak reflectance
    lobe_sigma_deg: float     # specular lobe angular width (small=mirror-like)
    spectral_slope: float     # d(reflectance)/d(micron) across the band (color)
    feature_nm: float = 0.0   # optional absorption feature centre [nm] (0 = none)
    feature_depth: float = 0.0

    def reflectance_spectrum(self, wl_nm):
        """Diffuse spectral reflectance rho_d(lambda) (normalized to rho_d_vis @550 nm)."""
        wl = np.asarray(wl_nm, float)
        r = self.rho_d_vis * (1.0 + self.spectral_slope * (wl - 550.0) / 1000.0)
        if self.feature_nm > 0:
            r = r * (1.0 - self.feature_depth * np.exp(-0.5 * ((wl - self.feature_nm) / 40.0) ** 2))
        return np.clip(r, 0.0, 1.0)

    def sbrdf(self, theta_i, dtheta, phi, wl_nm):
        """Material sBRDF [1/sr]: Lambertian diffuse + Gaussian specular lobe about the specular
        direction (off-specular angle from dtheta, phi)."""
        rho_d = self.reflectance_spectrum(wl_nm)
        off = np.sqrt(np.asarray(dtheta, float) ** 2 + np.asarray(phi, float) ** 2)   # deg from specular
        spec = self.rho_s * np.exp(-0.5 * (off / self.lobe_sigma_deg) ** 2)
        # specular reflectance is also mildly spectral (metals redder, glass bluer): reuse slope/4
        spec = spec * (1.0 + 0.25 * self.spectral_slope * (np.asarray(wl_nm, float) - 550.0) / 1000.0)
        return rho_d / np.pi + spec / (2.0 * np.pi * (self.lobe_sigma_deg * DEG) ** 2 + 1e-12)


# representative materials (physically-based; swap for measured LUTs when available)
MATERIALS: Dict[str, Material] = {
    # aluminum 6061-T6: bright, specular, gray, mild ~800 nm feature [paper Fig 8/9]
    "aluminum":  Material("aluminum 6061-T6", rho_d_vis=0.12, rho_s=0.75, lobe_sigma_deg=4.0,
                          spectral_slope=0.05, feature_nm=800.0, feature_depth=0.10),
    # triple-junction solar cell: dark, blue-weighted, cover-glass specular, thin-film features
    "solarcell": Material("TJ solar cell", rho_d_vis=0.09, rho_s=0.18, lobe_sigma_deg=3.0,
                          spectral_slope=-0.35, feature_nm=650.0, feature_depth=0.20),
    # A276 white paint: bright diffuse, nearly flat
    "whitepaint":Material("Emcore/A276 white paint", rho_d_vis=0.82, rho_s=0.04, lobe_sigma_deg=25.0,
                          spectral_slope=-0.05),
    # aluminized-kapton MLI: amber, rises to IR, specular-ish crinkled
    "mli":       Material("aluminized-kapton MLI", rho_d_vis=0.30, rho_s=0.35, lobe_sigma_deg=12.0,
                          spectral_slope=0.55),
    # black paint (Z306): very dark, flat, diffuse
    "blackpaint":Material("Z306 black paint", rho_d_vis=0.04, rho_s=0.02, lobe_sigma_deg=30.0,
                          spectral_slope=0.0),
}

# XB6 deputy material composition (area fractions; representative 6U cubesat)
XB6_COMPOSITION = {"mli": 0.35, "solarcell": 0.30, "aluminum": 0.20, "whitepaint": 0.10, "blackpaint": 0.05}
# CanX-1 EM (paper validation subject): 3 materials
CANX1_COMPOSITION = {"aluminum": 0.55, "solarcell": 0.30, "whitepaint": 0.15}
# default XB6 6U-box facets: (material, outward normal, area m^2) -- solar panels on the big faces
DEFAULT_XB6_FACETS = [
    ("solarcell", [0, 1, 0], 0.06), ("solarcell", [0, -1, 0], 0.06),
    ("mli",       [1, 0, 0], 0.03), ("mli",       [-1, 0, 0], 0.03),
    ("aluminum",  [0, 0, 1], 0.02), ("aluminum",  [0, 0, -1], 0.02),
]


# ======================================================================================
# Geometry + overall spacecraft sBRDF (paper Eq 3-4)
# ======================================================================================
def geometry_from_vectors(sun_hat, normal_hat, cam_hat):
    """Specular-referenced angles (theta_i, dtheta, phi) [deg] for a facet, from unit vectors.
    theta_i = Sun..normal; specular = mirror of Sun about normal; off-specular split into the
    in-plane (dtheta) and from-plane (phi) components toward the camera."""
    s = _u(sun_hat); n = _u(normal_hat); v = _u(cam_hat)
    ci = np.clip(np.dot(s, n), -1, 1); theta_i = np.degrees(np.arccos(ci))
    spec = _u(2 * ci * n - s)                                  # specular reflection direction
    off = np.degrees(np.arccos(np.clip(np.dot(v, spec), -1, 1)))   # total off-specular angle
    # split into in-illumination-plane (dtheta) and out-of-plane (phi)
    plane_n = _u(np.cross(s, n)) if np.linalg.norm(np.cross(s, n)) > 1e-6 else _u(np.cross(v, n) + 1e-9)
    phi = np.degrees(np.arcsin(np.clip(np.dot(v, plane_n), -1, 1)))
    dtheta = np.sign(np.dot(v, _u(np.cross(plane_n, spec)))) * np.sqrt(max(off**2 - phi**2, 0.0))
    return theta_i, dtheta, phi


def overall_sbrdf(composition, theta_i, dtheta, phi, wl_nm):
    """Composition-weighted overall spacecraft sBRDF [1/sr] at (theta_i, dtheta, phi, lambda)
    (paper Eq 4, material-fraction form: projected-area weight ~ the area fraction)."""
    wl = np.asarray(wl_nm, float); tot = np.zeros_like(wl, dtype=float); wsum = 0.0
    for key, frac in composition.items():
        tot = tot + frac * MATERIALS[key].sbrdf(theta_i, dtheta, phi, wl); wsum += frac
    return tot / max(wsum, 1e-9)


def overall_sbrdf_faceted(composition_facets, sun_hat, cam_hat, wl_nm):
    """Paper Eq 3-4 with an explicit facet list: [(material_key, normal_hat, area_m2), ...].
    Overall = sum(a2D * f_material) / sum(a2D), a2D = area * max(cos(emission), 0)."""
    wl = np.asarray(wl_nm, float); num = np.zeros_like(wl, dtype=float); den = 0.0
    v = _u(cam_hat)
    for key, nrm, area in composition_facets:
        n = _u(nrm); ce = float(np.dot(n, v))
        if ce <= 0 or float(np.dot(n, _u(sun_hat))) <= 0:   # facet must face camera AND Sun
            continue
        a2D = area * ce
        ti, dt, ph = geometry_from_vectors(sun_hat, n, v)
        num = num + a2D * MATERIALS[key].sbrdf(ti, dt, ph, wl); den += a2D
    return num / max(den, 1e-12)


# ======================================================================================
# Peer-vision: what a deputy's camera collects when imaging another deputy
# ======================================================================================
def peer_vision_signal(sun_hat, cam_hat, range_m, aperture_m, exposure_s, facets=None,
                       qe=0.55, read_noise_e=7.0, wl_nm=None):
    """Collected signal + SNR + spectral reflectance signature when a deputy images a peer.
    Sums reflected radiant intensity toward the camera over the peer's lit+visible facets
    (paper Eq 3-4): I(lambda) = sum E_sun*mu0*f_material*area*mu_e ; E_cam = I/range^2."""
    wl = np.linspace(BAND_NM[0], BAND_NM[1], 60) if wl_nm is None else np.asarray(wl_nm, float)
    facets = facets or DEFAULT_XB6_FACETS
    s_h = _u(sun_hat); v = _u(cam_hat)
    esun_per_nm = E_SUN_WM2 / (BAND_NM[1] - BAND_NM[0])
    I = np.zeros_like(wl); refl_num = np.zeros_like(wl); refl_area = 0.0
    for key, nrm, area in facets:
        n = _u(nrm); mu0 = float(np.dot(s_h, n)); mue = float(np.dot(v, n))
        if mu0 <= 0 or mue <= 0:                       # facet must be sunlit AND camera-visible
            continue
        ti, dt, ph = geometry_from_vectors(s_h, n, v)
        f = MATERIALS[key].sbrdf(ti, dt, ph, wl)       # [1/sr]
        I = I + esun_per_nm * mu0 * f * area * mue      # [W/sr/nm]
        refl_num = refl_num + f * area * mue; refl_area += area * mue
    E_cam = I / max(range_m**2, 1e-9)                  # [W/m^2/nm]
    A_ap = np.pi * (aperture_m/2)**2
    photon_E = H_PLANCK * C_LIGHT / (wl * 1e-9)
    dlam = (wl[-1] - wl[0]) / (len(wl) - 1)
    e_per_nm = E_cam * A_ap * exposure_s / photon_E * qe
    signal_e = float(np.sum(e_per_nm) * dlam)
    snr = signal_e / np.sqrt(signal_e + read_noise_e**2) if signal_e > 0 else 0.0
    overall = refl_num / max(refl_area, 1e-12)         # projected-area-weighted overall sBRDF
    sig = overall / max(overall[np.argmin(abs(wl-550))], 1e-9)
    return dict(signal_e=signal_e, snr=float(snr), wl_nm=wl, sbrdf=overall,
                reflectance_signature=sig, lit_visible_area_m2=refl_area)


def _u(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ======================================================================================
# SELF-TEST
# ======================================================================================
def _selftest():
    fails = []
    def check(name, cond, detail=""):
        print(("  PASS " if cond else "  FAIL ") + name + ("" if not detail else "  -> " + detail))
        if not cond: fails.append(name)
    print("=" * 78)
    print("SPACECRAFT sBRDF SELF-TEST  (Willison & Bedard 2016; deputy peer-vision)")
    print("=" * 78)
    wl = np.linspace(400, 1000, 60)

    # 1. Material spectral reflectance sane + distinct signatures
    print("\n1. Material diffuse reflectance @[450,550,850] nm:")
    for k, m in MATERIALS.items():
        r = m.reflectance_spectrum(np.array([450, 550, 850.0]))
        print(f"     {m.name:26s}: {r[0]:.2f} {r[1]:.2f} {r[2]:.2f}")
    check("white paint bright (>0.7), black dark (<0.1)",
          MATERIALS["whitepaint"].reflectance_spectrum(550) > 0.7 and MATERIALS["blackpaint"].reflectance_spectrum(550) < 0.1)
    check("solar cell blue-weighted (450>850 nm)",
          MATERIALS["solarcell"].reflectance_spectrum(450) > MATERIALS["solarcell"].reflectance_spectrum(850))
    check("MLI red/IR-weighted (850>450 nm)",
          MATERIALS["mli"].reflectance_spectrum(850) > MATERIALS["mli"].reflectance_spectrum(450))
    check("aluminum shows ~800 nm feature", MATERIALS["aluminum"].reflectance_spectrum(800) < MATERIALS["aluminum"].reflectance_spectrum(700))

    # 2. Specular lobe: peaks at the specular direction, falls off-specular
    print("\n2. Specular behaviour (aluminum, theta_i=30):")
    al = MATERIALS["aluminum"]
    on = al.sbrdf(30, 0.0, 0.0, 550.0); off = al.sbrdf(30, 20.0, 0.0, 550.0)
    print(f"     sBRDF at specular = {on:.3f}, 20deg off-specular = {off:.3f}")
    check("specular peak >> off-specular for a metal", on > 5 * off)
    check("diffuse paint ~ isotropic", MATERIALS["whitepaint"].sbrdf(30,0,0,550) < 3*MATERIALS["whitepaint"].sbrdf(30,30,0,550))

    # 3. Overall spacecraft sBRDF (Eq 4) + geometry from vectors
    sun = [1, 0.2, 0.6]; cam = [0.2, -0.3, 1.0]; nrm = [0, 0, 1]   # moderate phase for peer-vision
    ti, dt, ph = geometry_from_vectors(sun, nrm, cam)
    f = overall_sbrdf(XB6_COMPOSITION, ti, dt, ph, wl)
    print(f"\n3. Overall XB6 sBRDF: geometry theta_i={ti:.0f} dtheta={dt:.0f} phi={ph:.0f} ; sBRDF@550={f[np.argmin(abs(wl-550))]:.3f}/sr")
    check("overall sBRDF positive & finite", np.all(f > 0) and np.all(np.isfinite(f)))
    check("faceted algorithm (Eq 3-4) runs", overall_sbrdf_faceted([("aluminum",[0,0,1],0.02),("solarcell",[0,0.3,1],0.02)], sun, cam, wl).shape==wl.shape)

    # 4. Peer-vision: deputy imaging a peer at ~1 km with the CIS120 camera
    print("\n4. Peer-vision signal (deputy images a peer):")
    for R in [500.0, 2000.0, 5000.0]:
        s = peer_vision_signal(sun, cam, R, aperture_m=0.013, exposure_s=0.05)
        print(f"     range {R:5.0f} m: signal {s['signal_e']:.2e} e-, SNR {s['snr']:.1f}, lit+vis area {s['lit_visible_area_m2']*1e4:.0f} cm^2")
    s1 = peer_vision_signal(sun, cam, 1000.0, 0.013, 0.05)
    check("peer detectable at 1 km (SNR>5)", s1["snr"] > 5.0, f"SNR={s1['snr']:.1f}")
    check("SNR falls with range^2", peer_vision_signal(sun,cam,5000.,0.013,0.05)["snr"] < s1["snr"])

    # 5. Spectral signature differs by composition (material ID / anomaly detection)
    print("\n5. Spectral signature (material ID for FDIR):")
    healthy = peer_vision_signal(sun, cam, 1000., 0.013, 0.05)["reflectance_signature"]
    torn = [("aluminum" if k=="mli" else k, n, a) for (k,n,a) in DEFAULT_XB6_FACETS]   # MLI torn -> bare Al
    anom = peer_vision_signal(sun, cam, 1000., 0.013, 0.05, facets=torn)["reflectance_signature"]
    diff = float(np.max(np.abs(anom - healthy) / np.maximum(healthy, 1e-9)))
    print(f"     healthy vs MLI-torn spectral deviation: {100*diff:.0f}% (peer-vision can flag it)")
    check("anomalous composition gives a detectable spectral shift (>5%)", diff > 0.05, f"{100*diff:.0f}%")

    print("\n" + "=" * 78)
    print(f"RESULT: {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}  ({len(fails)} failure(s))")
    print("=" * 78)
    return len(fails) == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
