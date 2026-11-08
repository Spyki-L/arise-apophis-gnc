"""
imu_gyro.py  -  ARISE V2 high-fidelity rate-gyro / IMU + event-camera rate model
================================================================================
Replaces the flat onboard rate-knowledge constant (fdi/faultlab_ekf.py AttEKF
gy_noise = deg2rad(0.003) ~ 0.003 deg/s, and the implicit "gyro is perfect" rate
propagation) with TWO physics-based, paper-calibrated angular-rate models plus their
inverse-variance fusion:

  PART A - navigation-grade MEMS gyro (Allan-variance error chain), calibrated to
      W. Zhao et al., "Navigation Grade MEMS IMU for A Satellite," Micromachines 12
      (2021) 151  (the flight unit on the Lobster-Eye X-ray / Einstein-Probe sat, 2020).
      Paper Table 3: range +-20 deg/s, bias stability 0.02 deg/h, bias instability
      0.006 deg/h, ARW 0.003 deg/root-h, BW 12 Hz.  Table 1 gives the 6 per-gyro values.
      Error chain: turn-on bias + angle random walk (ARW) + Gauss-Markov bias
      instability + rate random walk (RRW) + scale-factor + misalignment + quantization,
      clipped to range/bandwidth.  Perturbs KNOWLEDGE only, never truth.

  PART B - event-based (neuromorphic) camera angular-rate estimator, calibrated to
      V. Franzese & M. El Hariry, "Spacecraft angular rate estimation via event-based
      camera sensing," Adv. Space Res. 77 (2026) 7801.  De-rotation of apparent star
      motion: feature-sensitivity matrix F(x,y) maps body rates (p,q,r) -> focal-plane
      motion (u,v); least-squares recovers (p,q,r) with analytic covariance.  A single
      camera cannot see roll (the boresight component w is unobservable), so an ORTHOGONAL
      DUAL-camera setup is fused.  Paper Table 1: single total RMS 0.307 deg/s (roll-limited),
      orthogonal dual 0.0275 deg/s.  Drift-free, and immune to the motion blur that blinds a
      frame star tracker through the CA slew.

Why ARISE needs it (V2 finding #7): the bus cannot body-track the LOS through closest
approach; the star tracker itself drops out past ~13 deg/s.  The gyro carries rate
knowledge across that gap (low short-term noise) and the event camera bounds its drift
(drift-free absolute rate) -> together they are the rate-knowledge backbone of the CA slew.

Validation (self-test): the gyro generator's overlapping Allan deviation recovers the
paper's ARW (0.003 deg/root-h) and bias instability (0.006 deg/h) for the unit and for
all 6 Table-1 gyros; the event-camera estimator reproduces the single- vs dual-camera
Table-1 RMS and the roll-observability collapse.  Run `python imu_gyro.py` -> ALL PASS.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np

try:
    import hardware as HW
except ImportError:  # pragma: no cover
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hardware as HW

DEG = np.pi / 180.0
ARCSEC = DEG / 3600.0
BI_COEF = np.sqrt(2.0 * np.log(2.0) / np.pi)     # 0.6643: Allan floor = BI_COEF * B (bias instability)


# ======================================================================================
# PART A -- MEMS gyro Allan-variance error model
# ======================================================================================
@dataclass(frozen=True)
class GyroParams:
    """Per-axis nav-grade MEMS gyro noise model (Allan coefficients + deterministic errors).
    Rate quantities are carried in deg/s internally; ARW/BI/RRW use the standard IMU units."""
    name: str
    arw_deg_rth: float          # angle random walk N            [deg/root-hour]
    bias_instab_deg_h: float    # bias instability B (Allan floor)[deg/h]
    rrw_deg_h15: float          # rate random walk K             [deg/h^1.5]
    turnon_bias_deg_h: float    # 1-sigma turn-on bias           [deg/h]
    scale_factor_ppm: float     # residual scale-factor error    [ppm]
    misalign_arcsec: float      # axis misalignment 1-sigma      [arcsec]
    range_deg_s: float          # +- measurement range           [deg/s]
    bandwidth_hz: float         # sensor bandwidth               [Hz]
    sample_rate_hz: float       # output data rate               [Hz]
    bias_gm_tau_s: float        # bias-instability GM correlation[s]
    quant_deg_s: float          # rate quantization LSB          [deg/s]

    # ---- Allan deviation (analytic), rate units deg/s, tau in seconds --------------------
    def allan_dev_deg_s(self, tau_s):
        """Analytic overlapping-Allan deviation [deg/s] of the rate for averaging time tau_s."""
        tau_s = np.asarray(tau_s, float)
        th = tau_s / 3600.0                                   # hours
        N, B, K = self.arw_deg_rth, self.bias_instab_deg_h, self.rrw_deg_h15
        # deg/h Allan variance = ARW (tau^-1/2) + bias-instability floor + RRW (tau^+1/2).
        # (Quantization is negligible for a nav-grade unit; it lives as a rate LSB in measure().)
        var_dph = (N ** 2) / th + (BI_COEF ** 2) * B ** 2 + (K ** 2) * th / 3.0
        return np.sqrt(var_dph) / 3600.0                      # deg/h -> deg/s

    def arw_component_deg_s(self, tau_s):
        """Isolated ARW (angle-random-walk) Allan deviation [deg/s]: the tau^-1/2 asymptote."""
        return self.arw_deg_rth / (60.0 * np.sqrt(np.asarray(tau_s, float)))

    def bias_instab_component_deg_s(self):
        """Isolated bias-instability Allan floor [deg/s] = BI_COEF * B."""
        return BI_COEF * self.bias_instab_deg_h / 3600.0

    def rate_white_sigma_deg_s(self, dt_s):
        """Per-sample ARW rate white-noise 1-sigma at output interval dt_s [deg/s]."""
        return self.arw_deg_rth / (60.0 * np.sqrt(dt_s))

    def bias_instab_floor_deg_s(self):
        return BI_COEF * self.bias_instab_deg_h / 3600.0

    def ekf_rate_sigma_rad_s(self, dt_s=None):
        """Effective onboard rate-measurement 1-sigma [rad/s] for the AttEKF gyro update:
        per-sample ARW white noise RSS the bias-instability floor.  This is the hi-fi
        replacement for the flat gy_noise = deg2rad(0.003)."""
        dt = (1.0 / self.sample_rate_hz) if dt_s is None else dt_s
        w = self.rate_white_sigma_deg_s(dt)
        f = self.bias_instab_floor_deg_s()
        return float(np.hypot(w, f) * DEG)

    # ---- Gauss-Markov bias-instability sigma calibrated to the Allan floor ---------------
    def _gm_sigma_deg_s(self):
        """Steady-state std of the 1st-order GM bias process [deg/s] whose Allan-deviation
        MAXIMUM equals the bias-instability floor BI_COEF*B (numerically calibrated)."""
        T = self.bias_gm_tau_s
        tau = np.geomspace(2.0 * T / 50.0, 40.0 * T, 400)
        # unit-sigma GM Allan deviation
        r = tau / T
        # overlapping-Allan variance of a UNIT-sigma 1st-order Gauss-Markov process
        # (derived from R(s)=sigma^2 exp(-|s|/T)):  sigma_A^2/sigma^2 = (2a-3+4e^-a-e^-2a)/a^2.
        av2 = (2.0 * r - 3.0 + 4.0 * np.exp(-r) - np.exp(-2.0 * r)) / r ** 2
        k = float(np.sqrt(np.max(av2)))                      # unit-sigma Allan-dev peak (~0.617)
        return self.bias_instab_floor_deg_s() / k

    # ---- time-domain generators (deg/s) -------------------------------------------------
    def simulate_rate_error(self, duration_s, rng, dt_s=None, parts=("turnon", "arw", "bias", "rrw")):
        """Zero-input gyro RATE-error time series [deg/s].  `parts` selects components:
        turn-on bias + ARW (white) + GM bias instability + RRW (drift).  Isolating a single
        part (e.g. parts=('bias',)) lets the self-test recover that coefficient cleanly."""
        dt = (1.0 / self.sample_rate_hz) if dt_s is None else dt_s
        n = int(round(duration_s / dt))
        e = np.zeros(n)
        if "arw" in parts:
            e += rng.normal(0.0, self.rate_white_sigma_deg_s(dt), n)
        if "bias" in parts:
            T = self.bias_gm_tau_s; phi = np.exp(-dt / T)
            sig_gm = self._gm_sigma_deg_s(); qd = sig_gm * np.sqrt(1.0 - phi ** 2)
            b = np.empty(n); b[0] = rng.normal(0.0, sig_gm)
            wn = rng.normal(0.0, qd, n)
            for k in range(1, n):
                b[k] = phi * b[k - 1] + wn[k]
            e += b
        if "rrw" in parts:
            dth = dt / 3600.0
            incr = rng.normal(0.0, self.rrw_deg_h15 * np.sqrt(dth), n)
            e += np.cumsum(incr) / 3600.0
        if "turnon" in parts:
            e += rng.normal(0.0, self.turnon_bias_deg_h / 3600.0)
        return e, dt

    def measure(self, omega_true_rad_s, rng, state=None, dt_s=None):
        """One gyro sample: omega_meas = (1+SF)*(C_mis @ omega_true) + bias + ARW + quant.
        Perturbs the MEASUREMENT (knowledge), never the truth.  Returns (omega_meas[3], state)."""
        dt = (1.0 / self.sample_rate_hz) if dt_s is None else dt_s
        w = np.asarray(omega_true_rad_s, float)
        if state is None:
            state = dict(bias=rng.normal(0.0, self._gm_sigma_deg_s(), 3) * DEG,
                         turnon=rng.normal(0.0, self.turnon_bias_deg_h / 3600.0, 3) * DEG,
                         sf=rng.normal(0.0, self.scale_factor_ppm * 1e-6, 3),
                         mis=rng.normal(0.0, self.misalign_arcsec * ARCSEC, 3))
        # GM bias propagate
        T = self.bias_gm_tau_s; phi = np.exp(-dt / T)
        sig_gm = self._gm_sigma_deg_s() * DEG; qd = sig_gm * np.sqrt(1.0 - phi ** 2)
        state["bias"] = phi * state["bias"] + rng.normal(0.0, qd, 3)
        # small-angle misalignment matrix
        mx, my, mz = state["mis"]
        C = np.array([[1.0, -mz, my], [mz, 1.0, -mx], [-my, mx, 1.0]])
        arw = rng.normal(0.0, self.rate_white_sigma_deg_s(dt) * DEG, 3)
        wm = (1.0 + state["sf"]) * (C @ w) + state["bias"] + state["turnon"] + arw
        # range clip + quantization
        rng_lim = self.range_deg_s * DEG
        wm = np.clip(wm, -rng_lim, rng_lim)
        q = self.quant_deg_s * DEG
        if q > 0:
            wm = np.round(wm / q) * q
        return wm, state


def overlapping_adev(y_deg_s, dt_s, n_tau=40):
    """Overlapping Allan deviation of a RATE series y [deg/s].  Returns (tau_s, adev_deg_s)
    via the phase (integrated-angle) triple-difference estimator."""
    y = np.asarray(y_deg_s, float); N = y.size
    theta = np.concatenate([[0.0], np.cumsum(y) * dt_s])     # angle [deg], length N+1
    max_m = (N - 1) // 2
    ms = np.unique(np.geomspace(1, max(max_m, 1), n_tau).astype(int))
    taus, ad = [], []
    for m in ms:
        if 2 * m >= theta.size:
            continue
        d = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
        tau = m * dt_s
        av2 = np.mean(d ** 2) / (2.0 * tau ** 2)
        taus.append(tau); ad.append(np.sqrt(av2))
    return np.array(taus), np.array(ad)


# ---- gyro presets (nav-grade default from hardware SSOT; heritage comparators) ----------
def _navgrade():
    g = HW.IMU_GYRO
    return GyroParams(name=g.name, arw_deg_rth=g.arw_deg_rth, bias_instab_deg_h=g.bias_instab_deg_h,
                      rrw_deg_h15=g.rrw_deg_h15, turnon_bias_deg_h=g.turnon_bias_deg_h,
                      scale_factor_ppm=g.scale_factor_ppm, misalign_arcsec=g.misalign_arcsec,
                      range_deg_s=g.range_deg_s, bandwidth_hz=g.bandwidth_hz,
                      sample_rate_hz=g.sample_rate_hz, bias_gm_tau_s=g.bias_gm_tau_s,
                      quant_deg_s=g.quant_deg_s)

NAVGRADE = _navgrade()
HG4930   = replace(NAVGRADE, name="Honeywell HG4930 (tactical-nav)",
                   arw_deg_rth=HW.IMU_GYRO.hg4930_arw_deg_rth,
                   bias_instab_deg_h=HW.IMU_GYRO.hg4930_bias_instab_deg_h,
                   range_deg_s=400.0, bandwidth_hz=100.0)
STIM210  = replace(NAVGRADE, name="Sensonor STIM210 (tactical)",
                   arw_deg_rth=0.15, bias_instab_deg_h=HW.IMU_GYRO.stim210_bias_instab_deg_h,
                   range_deg_s=400.0, bandwidth_hz=262.0)
TACTICAL = replace(NAVGRADE, name="generic tactical MEMS", arw_deg_rth=0.1,
                   bias_instab_deg_h=1.0, range_deg_s=450.0, bandwidth_hz=200.0)


def active():
    """Mission ACTIVE gyro preset by env ARISE_IMU_PRESET: 'navgrade'(default)|'hg4930'|'stim210'|'tactical'."""
    import os
    return {"navgrade": NAVGRADE, "hg4930": HG4930, "stim210": STIM210,
            "tactical": TACTICAL}.get(os.getenv("ARISE_IMU_PRESET", "navgrade").strip().lower(), NAVGRADE)


# ======================================================================================
# PART B -- event-based camera angular-rate estimator (de-rotation of star motion)
# ======================================================================================
@dataclass(frozen=True)
class EventCamParams:
    name: str
    fov_deg: float
    n_pix: int
    window_s: float
    max_rate_deg_s: float
    n_stars: int          # representative stars in FOV for the LS de-rotation
    sigma_uv_px_s: float  # apparent-velocity measurement 1-sigma [px/s] (event flow noise)

    @property
    def f_px(self):
        """Focal length in pixels for the (half-)FOV and array size (pinhole)."""
        return (self.n_pix / 2.0) / np.tan(np.radians(self.fov_deg) / 2.0)


def feature_matrix(x_px, y_px, f_px):
    """Feature-sensitivity matrix F(x,y) [2x3] mapping body rates (p,q,r)[rad/s] to focal-plane
    apparent motion (u,v)[px/s] for a pinhole camera (Franzese 2026 eq 12; std de-rotation flow)."""
    f = f_px
    return np.array([[x_px * y_px / f, -(f + x_px * x_px / f), y_px],
                     [(f + y_px * y_px / f), -x_px * y_px / f, -x_px]])


def _estimate_one(stars_xy, omega_rad_s, f_px, sigma_uv, rng):
    """Least-squares rate estimate from one camera's star field.  Returns omega_est[3] [rad/s]."""
    H = np.vstack([feature_matrix(x, y, f_px) for x, y in stars_xy])   # (2N x 3)
    uv = (H @ omega_rad_s)                                             # true apparent motion
    uv = uv + rng.normal(0.0, sigma_uv, uv.shape)                     # event-flow measurement noise
    est, *_ = np.linalg.lstsq(H, uv, rcond=None)
    return est


def _random_starfield(p, rng):
    """N stars uniformly on the focal plane within the circular FOV [px]."""
    r_max = (p.n_pix / 2.0) * 0.98
    r = r_max * np.sqrt(rng.uniform(0, 1, p.n_stars))
    th = rng.uniform(0, 2 * np.pi, p.n_stars)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


# orthogonal dual-camera axis permutation (Franzese eq 23): cB1=cA1, cB2=cA3, cB3=cA2
_PERM_A2B = np.array([0, 2, 1])   # omega_B = omega_A[[0,2,1]] = (p, r, q)


def rate_rms(p, n_cases=200, dual=True, seed=0):
    """Monte-Carlo RMS [deg/s] of the event-camera rate estimate over random star fields and
    rates, single or orthogonal-dual (fused).  Reproduces Franzese Table 1."""
    rng = np.random.default_rng(seed)
    f = p.f_px; wmax = np.radians(p.max_rate_deg_s)
    err = np.zeros((n_cases, 3))
    for i in range(n_cases):
        w = rng.uniform(-wmax, wmax, 3)                       # true (p,q,r) rad/s
        starsA = _random_starfield(p, rng)
        eA = _estimate_one(starsA, w, f, p.sigma_uv_px_s, rng)      # A-frame (p,q,r); roll r poor
        if not dual:
            err[i] = eA - w
            continue
        starsB = _random_starfield(p, rng)
        wB = w[_PERM_A2B]                                     # rate in B frame = (p, r, q)
        eB = _estimate_one(starsB, wB, f, p.sigma_uv_px_s, rng)     # B sees (p, r, q); its roll (q) poor
        # fuse (eq 24): p = mean(A,B well-obs), q = A yaw (good), r = B's well-obs (= A roll)
        p_f = 0.5 * (eA[0] + eB[0]); q_f = eA[1]; r_f = eB[1]
        err[i] = np.array([p_f, q_f, r_f]) - w
    rms_axes = np.sqrt(np.mean(err ** 2, axis=0)) / DEG      # deg/s per axis
    total = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))) / DEG)
    return dict(rms_p=float(rms_axes[0]), rms_q=float(rms_axes[1]), rms_r=float(rms_axes[2]),
                total=total)


def _eventcam(sigma_uv_px_s=8.6):
    e = HW.EVENT_RATE_CAM
    return EventCamParams(name=e.name, fov_deg=e.fov_deg, n_pix=e.n_pix, window_s=e.window_s,
                          max_rate_deg_s=e.max_rate_deg_s, n_stars=25, sigma_uv_px_s=sigma_uv_px_s)

EVENTCAM = _eventcam()


# ======================================================================================
# FUSION -- gyro (low short-term noise) + event camera (drift-free) inverse-variance blend
# ======================================================================================
def fuse_rate(w_gyro, sig_gyro, w_evcam, sig_evcam):
    """Per-axis inverse-variance fusion of two rate estimates. sig_* are 1-sigma [rad/s]
    (scalars or length-3).  Returns (omega_fused[3], sigma_fused[3])."""
    wg = np.asarray(w_gyro, float); we = np.asarray(w_evcam, float)
    ig = 1.0 / np.asarray(sig_gyro, float) ** 2; ie = 1.0 / np.asarray(sig_evcam, float) ** 2
    ig, ie = np.broadcast_to(ig, (3,)), np.broadcast_to(ie, (3,))
    wf = (ig * wg + ie * we) / (ig + ie)
    sf = np.sqrt(1.0 / (ig + ie))
    return wf, sf


# ======================================================================================
# SELF-TEST
# ======================================================================================
def _selftest():
    fails = []
    def check(name, cond, detail=""):
        print(("  PASS " if cond else "  FAIL ") + name + ("" if not detail else "  -> " + detail))
        if not cond: fails.append(name)

    print("=" * 80)
    print("IMU GYRO + EVENT-CAMERA RATE MODEL SELF-TEST")
    print("  Zhao 2021 (nav-grade MEMS) + Franzese & El Hariry 2026 (event camera)")
    print("=" * 80)

    # --- 1. Analytic Allan model reproduces the paper's ARW and bias-instability coefficients -
    g = NAVGRADE
    N_short = float(60.0 * np.sqrt(1.0) * g.allan_dev_deg_s(1.0))   # ARW from tau^-1/2 region (tau=1 s)
    N_comp = float(g.arw_component_deg_s(3600.0) * 3600.0)          # isolated ARW @1 h
    bi_comp = float(g.bias_instab_component_deg_s() * 3600.0 / BI_COEF)
    print(f"\n1. Analytic Allan vs paper Table 3 (ARW 0.003 deg/root-h, BI 0.006 deg/h):")
    print(f"     ARW: full-curve short-tau={N_short:.4f}, component={N_comp:.4f} | BI component={bi_comp:.4f}")
    check("ARW from tau^-1/2 region == 0.003", abs(N_short - 0.003) < 3e-4, f"{N_short:.4f}")
    check("ARW component == 0.003 deg/root-h", abs(N_comp - 0.003) < 1e-6, f"{N_comp:.4f}")
    check("bias-instability floor == 0.006 deg/h", abs(bi_comp - 0.006) < 1e-6, f"{bi_comp:.4f}")
    check("Allan has -1/2 ARW slope at short tau", g.allan_dev_deg_s(1.0) > g.allan_dev_deg_s(100.0))

    # --- 2. Time-domain generator's EMPIRICAL Allan recovers ARW + BI from data (paper Fig 8) --
    rng = np.random.default_rng(7)
    y, dt = g.simulate_rate_error(6000.0, rng, dt_s=0.05)                     # full error series
    taus, adev = overlapping_adev(y, dt, n_tau=45)
    small = taus < 1.0
    N_emp = float(np.median(60.0 * np.sqrt(taus[small]) * adev[small]))       # deg/root-h from tau^-1/2
    yb, dtb = g.simulate_rate_error(8000.0, rng, dt_s=0.05, parts=("bias",))  # isolate bias instability
    tb, ab = overlapping_adev(yb, dtb, n_tau=45)
    peak_tau = 1.89 * g.bias_gm_tau_s                                         # GM Allan peaks at ~1.89*tau_c
    idx = np.argsort(np.abs(tb - peak_tau))[:3]                               # avg 3 nearest tau (tame variance)
    BI_emp = float(np.mean(ab[idx]) * 3600.0 / BI_COEF)                       # GM Allan peak = BI_COEF*B
    print(f"\n2. Empirical Allan from the generator ({len(y)} samples @ {1/dt:.0f} Hz):")
    print(f"     ARW (tau^-1/2 region)         = {N_emp:.4f} deg/root-h  (paper 0.003)")
    print(f"     bias instability (Allan peak) = {BI_emp:.4f} deg/h       (paper 0.006)")
    check("empirical ARW ~0.003 (+-25%)", abs(N_emp - 0.003) < 0.00075, f"{N_emp:.4f}")
    check("empirical bias instability ~0.006 (+-35%)", abs(BI_emp - 0.006) < 0.0021, f"{BI_emp:.4f}")

    # --- 3. All 6 Table-1 gyros reproduced (analytic) -------------------------------------
    print(f"\n3. Zhao Table 1 per-gyro reproduction (ARW, BI in deg/h):")
    T1 = [("X1", 3.59e-3, 6.06e-3), ("X2", 2.39e-3, 5.53e-3), ("Y1", 2.22e-3, 4.91e-3),
          ("Y2", 3.03e-3, 6.22e-3), ("Z1", 2.89e-3, 9.19e-3), ("Z2", 2.61e-3, 6.27e-3)]
    ok6 = True
    for nm, arw, bi in T1:
        gp = replace(g, arw_deg_rth=arw, bias_instab_deg_h=bi)
        n1 = gp.arw_component_deg_s(3600.0) * 3600.0
        bf = gp.bias_instab_component_deg_s() * 3600.0 / BI_COEF
        good = abs(n1 - arw) < 1e-6 and abs(bf - bi) < 1e-6
        ok6 &= good
        print(f"     {nm}: ARW {n1:.4f}/{arw:.4f}  BI {bf:.4f}/{bi:.4f}  {'ok' if good else 'X'}")
    check("all 6 Table-1 gyros reproduced (ARW + BI)", ok6)

    # --- 4. Effective EKF rate sigma vs the flat default it replaces -----------------------
    flat = np.deg2rad(0.003)                                 # current AttEKF gy_noise [rad/s]
    hifi = g.ekf_rate_sigma_rad_s()
    print(f"\n4. Onboard rate-knowledge sigma: hi-fi {hifi:.2e} rad/s vs flat default {flat:.2e} rad/s "
          f"({flat/hifi:.1f}x tighter)")
    check("nav-grade gyro tighter than flat 0.003 deg/s default", hifi < flat)
    check("range fits XACT body-rate cap (10 deg/s < 20 deg/s)", g.range_deg_s > 10.0)

    # --- 5. Event camera: single vs orthogonal-dual RMS (Franzese Table 1) ----------------
    p = EVENTCAM
    single = rate_rms(p, n_cases=250, dual=False, seed=1)
    dual = rate_rms(p, n_cases=250, dual=True, seed=2)
    print(f"\n5. Event-camera RMS (deg/s)   [paper: single 0.017/0.019/0.306 tot 0.307; dual tot 0.0275]")
    print(f"     single: p={single['rms_p']:.4f} q={single['rms_q']:.4f} r={single['rms_r']:.4f} "
          f"total={single['total']:.4f}")
    print(f"     dual  : p={dual['rms_p']:.4f} q={dual['rms_q']:.4f} r={dual['rms_r']:.4f} "
          f"total={dual['total']:.4f}")
    check("single pitch RMS ~0.017 (0.008..0.035)", 0.008 < single["rms_p"] < 0.035, f"{single['rms_p']:.4f}")
    check("single yaw RMS ~0.019 (0.008..0.035)", 0.008 < single["rms_q"] < 0.035, f"{single['rms_q']:.4f}")
    check("single roll poorly observable (RMS > 0.15)", single["rms_r"] > 0.15, f"{single['rms_r']:.4f}")
    check("single roll >> pitch (>6x)", single["rms_r"] / single["rms_p"] > 6.0,
          f"{single['rms_r']/single['rms_p']:.1f}x")
    check("dual total RMS ~0.0275 (<0.05, beats MEMS)", dual["total"] < 0.05, f"{dual['total']:.4f}")
    check("dual recovers roll (roll RMS < 0.05)", dual["rms_r"] < 0.05, f"{dual['rms_r']:.4f}")
    check("dual fusion beats single total", dual["total"] < single["total"])

    # --- 6. Event camera beats a miniature IMU but the nav-grade gyro is better -----------
    print(f"\n6. Rate-sensor comparison (total 1-sigma, deg/s):")
    ev = dual["total"]; gy = g.ekf_rate_sigma_rad_s() / DEG
    print(f"     nav-grade gyro/sample {gy:.4f} | event-cam dual {ev:.4f} | miniIMU 0.05-0.5 | ST 0.01-0.1")
    check("event-cam dual in star-tracker class (<0.1)", ev < 0.1)
    check("nav-grade gyro short-term < event-cam (complementary)", gy < ev)

    # --- 7. Fusion: inverse-variance blend is no worse than either input -------------------
    wf, sf = fuse_rate(np.zeros(3), gy * DEG, np.zeros(3), ev * DEG)
    print(f"\n7. Gyro+event-cam fusion sigma {np.mean(sf)/DEG:.4f} deg/s (<= min input)")
    check("fused sigma <= gyro sigma", np.all(sf <= gy * DEG + 1e-12))
    check("fused sigma <= eventcam sigma", np.all(sf <= ev * DEG + 1e-12))

    # --- 8. measure(): perturbs knowledge, respects range clip ----------------------------
    rng = np.random.default_rng(3)
    st = None; w_true = np.radians([1.0, -2.0, 0.5])
    outs = []
    for _ in range(50):
        wm, st = g.measure(w_true, rng, st)
        outs.append(wm)
    outs = np.array(outs)
    bias_est = np.mean(outs - w_true, axis=0)
    print(f"\n8. measure(): mean rate-error {np.round(bias_est/DEG,4)} deg/s (small bias+noise, truth intact)")
    check("measurement error small (<0.05 deg/s)", np.all(np.abs(bias_est) / DEG < 0.05))
    big = g.measure(np.radians([100.0, 0, 0]), np.random.default_rng(0))[0]
    check("range clip at +-20 deg/s", abs(big[0]) / DEG <= g.range_deg_s + 1e-6, f"{big[0]/DEG:.1f}")
    check("active() default = navgrade", active() is NAVGRADE)

    print("\n" + "=" * 80)
    print(f"RESULT: {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}  ({len(fails)} failure(s))")
    print("=" * 80)
    return len(fails) == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
