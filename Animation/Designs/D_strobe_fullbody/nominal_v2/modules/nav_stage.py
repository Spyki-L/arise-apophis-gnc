"""
nav_stage.py  -  ARISE V2 per-draw relative-navigation EKF (C1)
===============================================================
Replaces the flat 20 m delivery offset with a real, geometry-correlated navigation covariance
history that TIGHTENS into closest approach (angular measurement error -> cross-range position
error R*dtheta, which shrinks as R->0), and feeds the resulting target-direction knowledge into
the pointing gate together with the XB6 XACT attitude knowledge.

  * EKF state : relative [r(3), v(3)] of a deputy w.r.t. Apophis
  * dynamics  : constant-velocity + accel process noise (relative motion over the encounter)
  * measure   : range + two bearing angles (camera/LIDAR), R_meas from sigma_range, sigma_angle
  * pointing  : sigma_point = sqrt(adcs_knowledge^2 + (cross_range_pos_sigma / range)^2)
  * cross-deputy correlation : a shared chief broadcast injects a common-mode ephemeris error
                               (correlated across deputies) on top of each deputy's independent nav.

Runs standalone in numpy on a provided relative track; the orchestrator feeds it the Basilisk
trajectories.  For the full chain you may instead call the repo's run_nav_on_new_traj.py; this
module gives the same covariance product self-contained and per-draw.
"""
from __future__ import annotations
import os, sys
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardware as HW

ARCSEC = np.radians(1.0/3600.0)


@dataclass(frozen=True)
class NavConfig:
    sigma_range_m: float = 5.0           # ASSUMED: LIDAR/range meas noise 5 m
    sigma_angle_rad: float = 50*ARCSEC   # ASSUMED: bearing meas noise ~50 arcsec
    accel_psd: float = 1e-8              # process-noise accel PSD [m^2/s^3] (loose rel-dyn)
    chief_commonmode_m: float = 15.0     # shared chief ephemeris error (cross-deputy correlated)
    chief_corr: float = 0.6              # correlation fraction injected via the chief broadcast


def _meas_jac(r):
    """H for [range, az, el] w.r.t. position (3)."""
    x, y, z = r; rho = np.linalg.norm(r); rxy2 = x*x + y*y; rxy = np.sqrt(rxy2) + 1e-9
    Hr = np.array([x/rho, y/rho, z/rho])
    Haz = np.array([-y/rxy2, x/rxy2, 0.0])
    Hel = np.array([-x*z/(rho*rho*rxy), -y*z/(rho*rho*rxy), rxy/(rho*rho)])
    H = np.zeros((3, 6)); H[0,:3]=Hr; H[1,:3]=Haz; H[2,:3]=Hel
    return H


def _run_ekf_core(rel_track, times, cfg: NavConfig = NavConfig()):
    """rel_track[N,3] deputy-Apophis positions [m]; times[N] s. Returns pos/vel sigma history
    and the range history."""
    N = len(times)
    rel = np.asarray(rel_track, float)
    # init: rough velocity from finite diff, loose covariance
    v0 = (rel[1]-rel[0])/(times[1]-times[0]) if N > 1 else np.zeros(3)
    x = np.concatenate([rel[0], v0])
    P = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])      # 1 km pos, 10 m/s vel 1-sig start
    R = np.diag([cfg.sigma_range_m**2, cfg.sigma_angle_rad**2, cfg.sigma_angle_rad**2])
    pos_sig = np.zeros((N, 3)); vel_sig = np.zeros((N, 3)); rng_hist = np.zeros(N)
    cross_sig = np.zeros(N)
    for k in range(N):
        if k > 0:
            dt = times[k]-times[k-1]
            F = np.eye(6); F[:3,3:] = dt*np.eye(3)
            q = cfg.accel_psd
            Q = q*np.array([[dt**3/3,0,0,dt**2/2,0,0],[0,dt**3/3,0,0,dt**2/2,0],
                            [0,0,dt**3/3,0,0,dt**2/2],[dt**2/2,0,0,dt,0,0],
                            [0,dt**2/2,0,0,dt,0],[0,0,dt**2/2,0,0,dt]])
            x = F@x; P = F@P@F.T + Q
        # measurement update using the true relative position as the linearization point
        r_true = rel[k]
        H = _meas_jac(x[:3])
        S = H@P@H.T + R
        K = P@H.T@np.linalg.inv(S)
        # innovation from true geometry (sim measurement = truth + noise-free for covariance study)
        z_pred = np.array([np.linalg.norm(x[:3]),
                           np.arctan2(x[1],x[0]),
                           np.arctan2(x[2], np.linalg.norm(x[:2]))])
        z_true = np.array([np.linalg.norm(r_true),
                           np.arctan2(r_true[1],r_true[0]),
                           np.arctan2(r_true[2], np.linalg.norm(r_true[:2]))])
        x = x + K@(z_true - z_pred)
        P = (np.eye(6)-K@H)@P
        pos_sig[k] = np.sqrt(np.clip(np.diag(P)[:3], 0, None))
        vel_sig[k] = np.sqrt(np.clip(np.diag(P)[3:], 0, None))
        rng_hist[k] = np.linalg.norm(r_true)
        # true cross-LOS (perpendicular) position sigma -> drives target-direction knowledge
        los = r_true / (np.linalg.norm(r_true) + 1e-9)
        a = np.cross(los, np.array([0, 0, 1.0]))
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(los, np.array([1.0, 0, 0]))
        a = a / np.linalg.norm(a); b = np.cross(los, a)
        Ppos = P[:3, :3]
        cross_sig[k] = np.sqrt(max(a @ Ppos @ a + b @ Ppos @ b, 0.0))
    return dict(pos_sigma_m=pos_sig, vel_sigma_mps=vel_sig, range_m=rng_hist,
                crossrange_sigma_m=cross_sig)


def _startracker_hifi_on():
    return os.getenv("ARISE_STARTRACKER_HIFI", "0") not in ("0", "", "false", "False")


def pointing_error(nav_result, bus: HW.XB6Bus = None, omega_rad_s=None):
    """Combine star-tracker attitude knowledge with nav target-direction uncertainty -> total
    pointing sigma [rad] over time.

    ADCS (star-tracker) term:
      * default (identity-when-absent) : flat XACT knowledge floor bus.pointing_knowledge_arcsec.
      * ARISE_STARTRACKER_HIFI enabled : rate-dependent XACT-50 cross-boresight sigma from the
        star_tracker HFSS model (calibrated to the BCT XACT-50 datasheet).  omega_rad_s (scalar
        or array broadcastable to the track) is the body slew rate; None -> static/settled value.
    """
    bus = bus or HW.XB6
    crossrange = nav_result["crossrange_sigma_m"]                 # true cross-LOS sigma
    nav_dir = crossrange / np.maximum(nav_result["range_m"], 1.0)
    if _startracker_hifi_on():
        import star_tracker as ST
        w = 0.0 if omega_rad_s is None else np.asarray(omega_rad_s, float)
        if np.ndim(w) == 0:
            adcs = ST.active().pointing_sigma_rad(float(w))
        else:
            adcs = np.array([ST.active().pointing_sigma_rad(float(wi))
                             for wi in np.broadcast_to(w, nav_dir.shape)])
    else:
        adcs = bus.pointing_knowledge_arcsec * ARCSEC            # flat XACT heritage (identity)
    return np.sqrt(adcs**2 + nav_dir**2)


def cross_deputy_covariance(n_deputies, per_deputy_pos_sigma, cfg: NavConfig = NavConfig()):
    """Block covariance of fleet position errors: independent part + common-mode chief term."""
    common = cfg.chief_corr * cfg.chief_commonmode_m**2
    sig = np.atleast_1d(np.asarray(per_deputy_pos_sigma, float))
    if sig.size == 1:
        sig = np.full(n_deputies, sig[0])
    indep = (1-cfg.chief_corr) * sig**2
    C = np.full((n_deputies, n_deputies), common) + np.diag(indep)
    return C

def _nav_hifi_on():
    return os.getenv("ARISE_NAV_HIFI", "0") not in ("0", "", "false", "False")


def run_ekf(rel_track, times, cfg: NavConfig = NavConfig()):
    """IDENTITY-WHEN-ABSENT: the Schmidt-Kalman CONSIDER covariance + DMC (ARISE_NAV_HIFI on) else the
    optimistic EKF (off, byte-identical to the legacy covariance history)."""
    if _nav_hifi_on():
        return run_ekf_hifi(rel_track, times, cfg)
    return _run_ekf_core(rel_track, times, cfg)


# ==========================================================================================
# HI-FI NAVIGATION -- Schmidt-Kalman consider covariance + dynamic model compensation (DMC)
# Refs: Tapley/Schutz/Born 'Statistical Orbit Determination'; Woodburn & Tanygin (recursive
# consider filter); OSIRIS-REx landmark opnav.  The optimistic EKF above is the identity fallback.
# ==========================================================================================

def _dmc_accel_psd(con=None):
    """Effective process-noise acceleration PSD from the DMC Gauss-Markov unmodeled accel:
    a white-noise PSD q = 2 sigma_a^2 / tau drives a first-order Gauss-Markov acceleration (SRP,
    GM error, outgassing) -- the physically-grounded replacement for a flat accel_psd."""
    con = con or HW.NAV_CONSIDER
    return 2.0 * con.dmc_sigma_accel_ms2 ** 2 / con.dmc_tau_s


def _consider_Pcc(cfg, con, include_chief):
    """Fixed covariance of the un-estimated 'consider' parameters [b_range, b_az, b_el, chief_xyz]."""
    con = con or HW.NAV_CONSIDER
    ab = con.meas_angle_bias_arcsec * ARCSEC
    diag = [con.meas_range_bias_m ** 2, ab ** 2, ab ** 2]
    if include_chief:
        c = (con.chief_ephem_bias_m / np.sqrt(3.0)) ** 2
        diag += [c, c, c]
    else:
        diag += [0.0, 0.0, 0.0]
    return np.diag(diag)


def _skf_run(rel_track, times, cfg, con=None, include_chief=True, dmc=True):
    """Schmidt-Kalman (consider) covariance run: like run_ekf but the state covariance also carries
    the un-estimated systematic biases (range/bearing) and the chief-ephemeris common-mode, which do
    NOT average down.  Returns the CONSIDERED position 1-sigma history (the honest covariance)."""
    con = con or HW.NAV_CONSIDER
    N = len(times); rel = np.asarray(rel_track, float)
    v0 = (rel[1] - rel[0]) / (times[1] - times[0]) if N > 1 else np.zeros(3)
    x = np.concatenate([rel[0], v0])
    Pxx = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    Pcc = _consider_Pcc(cfg, con, include_chief); nc = Pcc.shape[0]
    Pxc = np.zeros((6, nc))
    R = np.diag([cfg.sigma_range_m ** 2, cfg.sigma_angle_rad ** 2, cfg.sigma_angle_rad ** 2])
    q = max(cfg.accel_psd, _dmc_accel_psd(con)) if dmc else cfg.accel_psd
    pos_sig = np.zeros((N, 3)); cross_sig = np.zeros(N); rng_hist = np.zeros(N)
    for k in range(N):
        if k > 0:
            dt = times[k] - times[k - 1]
            F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
            Q = q * np.array([[dt**3/3,0,0,dt**2/2,0,0],[0,dt**3/3,0,0,dt**2/2,0],
                              [0,0,dt**3/3,0,0,dt**2/2],[dt**2/2,0,0,dt,0,0],
                              [0,dt**2/2,0,0,dt,0],[0,0,dt**2/2,0,0,dt]])
            x = F @ x; Pxx = F @ Pxx @ F.T + Q; Pxc = F @ Pxc            # consider params constant
        Hx = _meas_jac(x[:3])                                            # 3x6 (position rows)
        Hc = np.zeros((3, nc)); Hc[:, :3] = np.eye(3); Hc[:, 3:6] = Hx[:, :3]  # meas bias + chief-through-geometry
        S = Hx @ Pxx @ Hx.T + Hx @ Pxc @ Hc.T + Hc @ Pxc.T @ Hx.T + Hc @ Pcc @ Hc.T + R
        Kx = (Pxx @ Hx.T + Pxc @ Hc.T) @ np.linalg.inv(S)
        r_true = rel[k]
        zt = np.array([np.linalg.norm(r_true), np.arctan2(r_true[1], r_true[0]),
                       np.arctan2(r_true[2], np.linalg.norm(r_true[:2]))])
        zp = np.array([np.linalg.norm(x[:3]), np.arctan2(x[1], x[0]),
                       np.arctan2(x[2], np.linalg.norm(x[:2]))])
        x = x + Kx @ (zt - zp)
        Pxx = Pxx - Kx @ (Hx @ Pxx + Hc @ Pxc.T)                         # Schmidt (consider) covariance update
        Pxc = Pxc - Kx @ (Hx @ Pxc + Hc @ Pcc)
        Pxx = 0.5 * (Pxx + Pxx.T)
        pos_sig[k] = np.sqrt(np.clip(np.diag(Pxx)[:3], 0, None)); rng_hist[k] = np.linalg.norm(r_true)
        los = r_true / (np.linalg.norm(r_true) + 1e-9)
        a = np.cross(los, [0, 0, 1.0]); a = np.cross(los, [1.0, 0, 0]) if np.linalg.norm(a) < 1e-6 else a
        a /= np.linalg.norm(a); b = np.cross(los, a); Pp = Pxx[:3, :3]
        cross_sig[k] = np.sqrt(max(a @ Pp @ a + b @ Pp @ b, 0.0))
    return dict(pos_sigma_m=pos_sig, crossrange_sigma_m=cross_sig, range_m=rng_hist)


def run_ekf_hifi(rel_track, times, cfg=None):
    """Hi-fi navigation covariance: optimistic EKF + DMC process noise + the CONSIDER (Schmidt-Kalman)
    covariance (absolute, chief-common-mode floored; and relative/deputy-deputy, measurement-bias
    floored with the chief common-mode removed)."""
    cfg = cfg or NavConfig()
    opt = _run_ekf_core(rel_track, times, cfg)                                 # optimistic (no consider)
    absr = _skf_run(rel_track, times, cfg, include_chief=True)           # absolute (with chief)
    rel = _skf_run(rel_track, times, cfg, include_chief=False)           # relative (chief cancels)
    out = dict(opt)
    out["pos_sigma_optimistic_m"] = opt["pos_sigma_m"]
    out["pos_sigma_consider_abs_m"] = absr["pos_sigma_m"]
    out["pos_sigma_consider_rel_m"] = rel["pos_sigma_m"]
    out["pos_sigma_m"] = absr["pos_sigma_m"]                             # the honest (considered) covariance
    out["crossrange_sigma_m"] = rel["crossrange_sigma_m"]               # stereo baseline uses the relative floor
    return out


def _selftest():
    fails = []
    def check(name, cond, detail=""):
        print(("  PASS " if cond else "  FAIL ") + name + ("" if not detail else "  -> " + detail))
        if not cond: fails.append(name)
    con = HW.NAV_CONSIDER
    print("=" * 84)
    print("NAV STAGE HI-FI SELF-TEST  (Schmidt-Kalman consider covariance + DMC)")
    print("  Tapley/Schutz/Born (consider filter + DMC) + Woodburn/Tanygin + OSIRIS-REx opnav")
    print("=" * 84)

    # approach 50 km -> 0.75 km
    t = np.linspace(0, 3600, 240)
    r = 50000.0 + (750.0 - 50000.0) * (t / t[-1])
    track = np.column_stack([r, 0.2 * r, 0.1 * r])
    cfg = NavConfig()
    opt = _run_ekf_core(track, t, cfg)
    hi = run_ekf_hifi(track, t, cfg)
    o_ca = np.linalg.norm(opt["pos_sigma_m"][-1])
    a_ca = np.linalg.norm(hi["pos_sigma_consider_abs_m"][-1])
    r_ca = np.linalg.norm(hi["pos_sigma_consider_rel_m"][-1])
    print(f"\nA. Position 1-sigma into CA (R={hi['range_m'][-1]:.0f} m):")
    print(f"     optimistic EKF        {o_ca:.2f} m")
    print(f"     consider (relative)   {r_ca:.2f} m   (measurement-bias floored; deputy-deputy baseline)")
    print(f"     consider (absolute)   {a_ca:.2f} m   (chief-ephemeris common-mode floored)")
    check("consider covariance >= optimistic (biases don't average down)", a_ca >= o_ca and r_ca >= o_ca)
    check("optimistic EKF is over-optimistic (< 3 m into CA)", o_ca < 3.0, f"{o_ca:.2f}")
    check("absolute nav floored near the chief ephemeris bias (~15 m)", 8.0 < a_ca < 25.0, f"{a_ca:.2f}")
    check("relative (deputy-deputy) floored near the measurement bias (~0.5-4 m)", 0.3 < r_ca < 5.0, f"{r_ca:.2f}")
    check("relative floor < absolute floor (chief common-mode cancels)", r_ca < a_ca)

    # B. bias-floor shape: bearing bias term R*b_az shrinks into CA; range/chief constant
    ab = con.meas_angle_bias_arcsec * ARCSEC
    bearing_far = hi["range_m"][5] * ab; bearing_ca = hi["range_m"][-1] * ab
    print(f"\nB. Bearing-bias cross-range term: {bearing_far:.2f} m @ {hi['range_m'][5]/1e3:.0f} km "
          f"-> {bearing_ca:.3f} m @ CA (shrinks as R->0); range bias {con.meas_range_bias_m:.1f} m (constant)")
    check("bearing-bias cross-range shrinks into CA (R*b_az)", bearing_ca < bearing_far)
    check("consider relative sigma is floored above the optimistic (range bias)", r_ca > 1.1 * o_ca, f"{r_ca:.2f} vs opt {o_ca:.2f}")

    # C. DMC process noise
    q_flat = cfg.accel_psd; q_dmc = _dmc_accel_psd(con)
    print(f"\nC. DMC process noise: Gauss-Markov accel PSD {q_dmc:.2e} m^2/s^3 (tau {con.dmc_tau_s:.0f} s, "
          f"sigma_a {con.dmc_sigma_accel_ms2:.0e}); flat baseline {q_flat:.0e}")
    check("DMC gives a physical accel PSD (>0)", q_dmc > 0)

    # D. OSIRIS-REx-class benchmark (relative baseline ~ meters at close range)
    print(f"\nD. Benchmark: relative nav {r_ca:.2f} m at {hi['range_m'][-1]:.0f} m "
          f"({r_ca/hi['range_m'][-1]*1e3:.2f} mm per m range; OSIRIS-REx landmark opnav ~ decimeters-meters)")
    check("relative nav in the meter class at close range (opnav-realistic)", r_ca < 5.0)
    check("covariance history monotone-ish into CA (shrinks then floors)",
          np.linalg.norm(hi["pos_sigma_consider_rel_m"][5]) >= r_ca)

    print("\n" + "=" * 84)
    print(f"RESULT: {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}  ({len(fails)} failure(s))")
    print("=" * 84)
    return len(fails) == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
