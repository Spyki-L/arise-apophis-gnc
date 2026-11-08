"""
thruster_hifi.py  -  ARISE V2 high-fidelity monopropellant thruster line (MPS-130 / AF-M315E)
==============================================================================================
Replaces the flat MPS-130 execution constants in thruster.py (Isp 245 s, MIB 10 mN-s,
finite-burn loss 2%, misalign 0.5 deg) with a PHYSICS-BASED, paper-calibrated model of the
whole propellant line, so the realized dV, its uncertainty, and the health telemetry all come
from first principles.  Perturbs KNOWLEDGE/EXECUTION, never the guidance truth dV.

Subsystems (each calibrated to a source paper / standard model):
  A  nozzle & c*          - c* = sqrt(R_s Tc)/Gamma(gamma); Isp = c* CF / g0        (AFRL AF-M315E)
  B  catalyst-bed decomp  - reduced-order Arrhenius reaction advancement -> exit T, c* efficiency,
                            + global-sensitivity (Sobol) UQ                          (Kerr 2021; Robledo 2020)
  C  blowdown feed        - passive ullage blowdown -> Pc(t)/thrust/Isp decay        (sub-Newton thesis)
  D  chugging instability - injector-stiffness (dP_inj/Pc) feed-coupling criterion   (sub-Newton thesis)
  E  water-hammer / valve - Joukowsky surge + wave speed + gradual-closure (Michaud) (Joukowsky/MOC)
  F  propellant slosh     - Abramson/Dodge equivalent pendulum: freq, slosh mass, torque (NASA SP-106)
  G  thruster-line sensors- pressure/thermocouple/flow/valve telemetry (measured vs truth)
  H  integrated execution - apply_burn_hifi(): layers B..E onto the misalign/MIB/finite-burn base,
                            returns realized dV + propellant + telemetry; fleet_delta_v_hifi().

Papers: Robledo Asencio 2020 (JATM, OpenFOAM catalytic decomposition); Kerr, Karagozian, Bilyeu &
Eckhardt 2021 (J. Prop. Power 37(1), nondeterministic 1D catalyst bed); "Sub-Newton Monopropellant
Thrusters" thesis (c* eff 0.963, chugging, blowdown); Abramson NASA SP-106 / Dodge (slosh);
Joukowsky (water hammer); AFRL ASCENT fact sheet (AF-M315E properties).

Validation (self-test): AF-M315E c* ~1420 m/s and Isp ~250 s; nominal reaction completion -> c*
efficiency 0.963 (thesis); Sobol ranks propellant mass fraction & activation energy as the dominant
uncertainty drivers (Kerr); blowdown Pc decays by the design ratio; chug onsets below the injector
stiffness threshold; Joukowsky surge matches rho*a*dv; slosh frequency follows 1.841 (a/R) tanh(...).
Run `python thruster_hifi.py` -> ALL PASS.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import hardware as HW
except ImportError:  # pragma: no cover
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hardware as HW

G0 = 9.80665
RU = 8.314462                       # universal gas constant [J/mol/K]


# ======================================================================================
# A. NOZZLE & CHARACTERISTIC VELOCITY
# ======================================================================================
def gamma_function(g):
    """Vandenkerckhove Gamma(gamma) for choked-nozzle mass flow / c*."""
    return np.sqrt(g) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def cstar_ideal(Tc_K, gamma, M_g_mol):
    """Ideal characteristic velocity [m/s]: c* = sqrt(R_s Tc) / Gamma(gamma)."""
    Rs = RU / (M_g_mol * 1e-3)
    return np.sqrt(Rs * Tc_K) / gamma_function(gamma)


def thrust_coefficient_vac(eps, gamma):
    """Vacuum thrust coefficient CF for area ratio eps (solves Pe/Pc from eps, adds pressure term)."""
    G = gamma_function(gamma); g = gamma
    # invert area-ratio relation eps = G / sqrt(2g/(g-1) * (Pe/Pc)^(2/g) * (1-(Pe/Pc)^((g-1)/g)))
    def area_ratio(pepc):
        return G / np.sqrt(2 * g / (g - 1) * pepc ** (2 / g) * (1 - pepc ** ((g - 1) / g)))
    lo, hi = 1e-6, 0.9
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if area_ratio(mid) > eps:
            lo = mid
        else:
            hi = mid
    pepc = np.sqrt(lo * hi)
    cf_mom = G * np.sqrt(2 * g / (g - 1) * (1 - pepc ** ((g - 1) / g)))
    return float(cf_mom + pepc * eps)      # vacuum: Pa = 0


@dataclass(frozen=True)
class NozzlePerf:
    prop: object
    eps: float = 50.0                  # nozzle area ratio (ASSUMED for a small vac thruster)
    cf_efficiency: float = 0.97        # ASSUMED: CF efficiency (divergence + boundary layer + finite eps)

    def cstar(self, eff=True):
        c = cstar_ideal(self.prop.flame_temp_K, self.prop.gamma, self.prop.M_products_g_mol)
        return float(c * (self.prop.cstar_efficiency if eff else 1.0))

    def cf_vac(self):
        return thrust_coefficient_vac(self.eps, self.prop.gamma)

    def isp_vac_s(self, eff=True):
        return self.cstar(eff) * self.cf_vac() * self.cf_efficiency / G0

    def thrust_N(self, Pc_Pa, At_m2):
        return self.cf_vac() * Pc_Pa * At_m2

    def mdot_kg_s(self, Pc_Pa, At_m2, eff=True):
        return Pc_Pa * At_m2 / self.cstar(eff)


# ======================================================================================
# B. CATALYST-BED DECOMPOSITION (reduced-order Arrhenius reaction advancement + UQ)
# ======================================================================================
class CatalystBed:
    """1D reduced-order global-reaction model (Kerr 2021 / Robledo 2020 form): reaction completion X
    from an Arrhenius Damkohler number over the bed residence time, with the exit temperature rising
    self-consistently with X.  c* efficiency = sqrt(X) (thesis: eta_c* = 0.963 ~ sqrt(eta_cat 0.929))."""

    def __init__(self, prop=None, feed=None):
        self.prop = prop or HW.AFM315E_PROP
        self.feed = feed or HW.MPS130_FEED
        # nominal residence time from bed geometry + mass flux
        A_bed = np.pi * (self.feed.bed_diameter_m / 2.0) ** 2
        self.mdot_nom = self.feed.bed_loading_kg_m2_s * A_bed
        void_vol = self.feed.bed_void_frac * A_bed * self.feed.bed_length_m
        self.t_res_nom = void_vol * self.prop.density_kg_m3 / self.mdot_nom
        # gas-phase residence time -> sets the feed-coupled chug timescale (not the liquid fill time)
        rho_gas = self.feed.chamber_press_nom_Pa * self.prop.M_products_g_mol * 1e-3 / (RU * self.prop.flame_temp_K)
        self.t_res_gas = self.feed.bed_length_m / (self.mdot_nom / (rho_gas * self.feed.bed_void_frac * A_bed))
        # calibrate the global prefactor so the NOMINAL run reproduces the catalytic thermal eff
        self.Ea = self.prop.activation_energy_J_mol
        self._A = self._calibrate_prefactor()

    def _completion(self, Y, Ea, Ti, mdot, A):
        """Reaction completion X in [0,1] (vectorized)."""
        dTad = Y * (self.prop.flame_temp_K - Ti)
        Tref = Ti + 0.5 * dTad
        t_res = self.t_res_nom * (self.mdot_nom / mdot)      # residence ~ 1/mdot
        Da = A * np.exp(-Ea / (RU * Tref)) * t_res
        return 1.0 - np.exp(-np.clip(Da, 0, 60))

    def _calibrate_prefactor(self):
        Xtgt = self.prop.catalytic_thermal_eff              # 0.929
        Ti = self.prop.preheat_temp_K
        dTad = 1.0 * (self.prop.flame_temp_K - Ti); Tref = Ti + 0.5 * dTad
        Da_tgt = -np.log(1.0 - Xtgt)
        return float(Da_tgt / (np.exp(-self.Ea / (RU * Tref)) * self.t_res_nom))

    def exit_state(self, Y=1.0, Ea=None, Ti=None, mdot=None):
        Ea = self.Ea if Ea is None else Ea
        Ti = self.prop.preheat_temp_K if Ti is None else Ti
        mdot = self.mdot_nom if mdot is None else mdot
        X = self._completion(np.asarray(Y, float), np.asarray(Ea, float),
                             np.asarray(Ti, float), np.asarray(mdot, float), self._A)
        Texit = Ti + X * Y * (self.prop.flame_temp_K - Ti)
        eta_cstar = np.sqrt(np.clip(X, 0, 1))
        return dict(completion=X, exit_temp_K=Texit, cstar_eff=eta_cstar, t_res_s=self.t_res_nom)

    def sobol_first_order(self, N=4096, seed=0):
        """Saltelli first-order Sobol indices of the EXIT TEMPERATURE w.r.t. the uncertain inputs
        (propellant mass fraction Y, activation energy Ea, preheat Ti, mass flow mdot).  Kerr 2021:
        Y and Ea are the dominant drivers."""
        rng = np.random.default_rng(seed)
        Ea0 = self.Ea; Ti0 = self.prop.preheat_temp_K; md0 = self.mdot_nom
        lohi = dict(Y=(0.85, 1.0), Ea=(0.8 * Ea0, 1.2 * Ea0), Ti=(350.0, 450.0),
                    mdot=(0.7 * md0, 1.3 * md0))
        names = list(lohi)
        def sample(n):
            return np.column_stack([rng.uniform(*lohi[k], n) for k in names])
        def evalT(M):
            X = self._completion(M[:, 0], M[:, 1], M[:, 2], M[:, 3], self._A)
            return M[:, 2] + X * M[:, 0] * (self.prop.flame_temp_K - M[:, 2])
        A = sample(N); B = sample(N)
        yA = evalT(A); yB = evalT(B)
        varY = np.var(np.concatenate([yA, yB]))
        S = {}
        for i, nm in enumerate(names):
            AB = A.copy(); AB[:, i] = B[:, i]
            yAB = evalT(AB)
            S[nm] = float(np.mean(yB * (yAB - yA)) / varY)
        return S


# ======================================================================================
# C. BLOWDOWN FEED SYSTEM
# ======================================================================================
class Blowdown:
    """Passive blowdown: pressurant ullage expands as propellant is expelled -> tank/chamber
    pressure and thrust decay.  P = P0 (V_u0/V_u)^n."""

    def __init__(self, feed=None, prop=None):
        self.feed = feed or HW.MPS130_FEED
        self.prop = prop or HW.AFM315E_PROP
        self.V_u0 = self.feed.ullage_frac0 * self.feed.tank_volume_m3
        self.n = self.feed.pressurant_gamma

    def usable_prop_kg(self):
        """Propellant expelled before the tank pressure falls by the design blowdown ratio."""
        ratio = self.feed.blowdown_ratio
        V_uf = self.V_u0 * ratio ** (1.0 / self.n)
        return float((V_uf - self.V_u0) * self.prop.density_kg_m3)

    def pressure_curve(self, n_pts=60):
        m_use = self.usable_prop_kg()
        m = np.linspace(0, m_use, n_pts)
        V_u = self.V_u0 + m / self.prop.density_kg_m3
        P = self.feed.tank_press0_Pa * (self.V_u0 / V_u) ** self.n
        Pc = P * (self.feed.chamber_press_nom_Pa / self.feed.tank_press0_Pa)
        return dict(m_expelled_kg=m, tank_P_Pa=P, chamber_P_Pa=Pc,
                    thrust_frac=P / P[0], blowdown_ratio=float(P[0] / P[-1]))


# ======================================================================================
# D. CHUGGING (feed-coupled low-frequency instability)
# ======================================================================================
def chug_state(feed=None, bed=None, Pc_Pa=None):
    """Injector-stiffness criterion for feed-coupled 'chugging' (sub-Newton thesis): stable when
    dP_inj/Pc >= threshold.  Chug frequency ~ 1/(2 pi tau_res) (low-frequency, residence-coupled).
    Returns stability, margin, frequency, and the c* efficiency penalty when chugging."""
    feed = feed or HW.MPS130_FEED
    bed = bed or CatalystBed()
    sigma = feed.injector_stiffness                       # dP_inj / Pc
    thr = feed.chug_stiffness_threshold
    chugging = sigma < thr
    tau = bed.t_res_gas                                    # gas-phase residence -> chug coupling timescale
    f_chug = 1.0 / (2.0 * np.pi * max(tau, 1e-6))
    # efficiency penalty grows as stiffness falls below threshold (bounded)
    penalty = float(np.clip((thr - sigma) / thr, 0, 1) * 0.08) if chugging else 0.0
    return dict(stiffness=float(sigma), threshold=float(thr), chugging=bool(chugging),
                margin=float(sigma - thr), chug_freq_hz=float(f_chug), cstar_penalty=penalty)


# ======================================================================================
# E. FEED-LINE WATER HAMMER + VALVE TRANSIENT
# ======================================================================================
def wave_speed(feed=None, prop=None):
    """Pressure-wave (acoustic) speed in the liquid-filled feed line: a = sqrt((K/rho)/(1+(K D)/(E e)))."""
    feed = feed or HW.MPS130_FEED; prop = prop or HW.AFM315E_PROP
    K = feed.prop_bulk_modulus_Pa; rho = prop.density_kg_m3
    corr = 1.0 + (K * feed.line_diameter_m) / (feed.line_wall_E_Pa * feed.line_wall_thick_m)
    return float(np.sqrt((K / rho) / corr))


def water_hammer(feed=None, prop=None, dv_ms=None, closure_s=None):
    """Joukowsky surge dP = rho a dv for instantaneous valve action; gradual-closure (Michaud)
    reduction when closure time > 2L/a."""
    feed = feed or HW.MPS130_FEED; prop = prop or HW.AFM315E_PROP
    a = wave_speed(feed, prop); rho = prop.density_kg_m3
    dv = feed.valve_flow_velocity_ms if dv_ms is None else dv_ms
    tc = feed.valve_close_s if closure_s is None else closure_s
    dP_joukowsky = rho * a * dv
    t_crit = 2.0 * feed.line_length_m / a                 # pipe period
    if tc <= t_crit:
        dP_eff = dP_joukowsky                             # fast closure -> full surge
    else:
        dP_eff = dP_joukowsky * (t_crit / tc)             # Michaud gradual-closure reduction
    return dict(wave_speed_ms=a, dP_joukowsky_Pa=float(dP_joukowsky), dP_effective_Pa=float(dP_eff),
                pipe_period_s=float(t_crit), fast_closure=bool(tc <= t_crit))


def mib_inflation_frac(feed=None, prop=None):
    """Fractional MIB uncertainty added by the valve open/close water-hammer transient
    (surge relative to chamber pressure -> impulse-bit repeatability hit)."""
    feed = feed or HW.MPS130_FEED
    wh = water_hammer(feed, prop)
    return float(np.clip(wh["dP_effective_Pa"] / feed.chamber_press_nom_Pa, 0, 1) * 0.5)


# ======================================================================================
# F. PROPELLANT SLOSH (Abramson/Dodge equivalent pendulum)
# ======================================================================================
def slosh_state(slosh=None, axial_accel_ms2=0.05, amplitude_frac=0.05):
    """First lateral slosh mode as an equivalent pendulum (cylindrical tank).  Returns modal
    frequency, slosh mass fraction, pendulum length, and a representative lateral disturbance
    force/torque during a burn.  freq^2 = xi1 (a/R) tanh(xi1 h/R), xi1 = 1.841."""
    slosh = slosh or HW.PROP_SLOSH
    R = slosh.tank_radius_m; xi = slosh.first_mode_root
    h = slosh.fill_fraction * (2.0 * R)                   # liquid column height (~aspect 2R)
    a = max(axial_accel_ms2, 1e-6)
    w2 = xi * (a / R) * np.tanh(xi * h / R)
    omega = np.sqrt(w2); f = omega / (2.0 * np.pi)
    m1_frac = float((R / (2.2 * h)) * np.tanh(xi * h / R))       # first-mode slosh mass fraction
    # equivalent-pendulum length L = g_eff / omega^2 (here g_eff = axial accel)
    L_pend = a / w2
    m_liq = slosh.fill_fraction * np.pi * R ** 2 * (2.0 * R) * HW.AFM315E_PROP.density_kg_m3
    m1 = m1_frac * m_liq
    F_lat = m1 * a * amplitude_frac                       # lateral disturbance force (small angle)
    torque = F_lat * L_pend
    return dict(freq_hz=float(f), omega_rad_s=float(omega), slosh_mass_frac=m1_frac,
                slosh_mass_kg=float(m1), pend_length_m=float(L_pend),
                dist_force_N=float(F_lat), dist_torque_Nm=float(torque))


# ======================================================================================
# G. THRUSTER-LINE SENSORS (measured vs truth; truth intact)
# ======================================================================================
def read_sensors(true_Pc_Pa, true_Texit_K, true_mdot_kg_s, rng, sens=None):
    """Return measured telemetry = truth + bias + noise for the thruster-line health sensors.
    The truth is passed through untouched (perturbs KNOWLEDGE only)."""
    sens = sens or HW.THR_SENSORS
    fs = sens.press_fs_bar * 1e5
    p_sig = sens.press_pct_fs / 100.0 * fs
    Pc_m = true_Pc_Pa + sens.press_bias_pct_fs / 100.0 * fs + rng.normal(0, p_sig)
    T_m = np.clip(true_Texit_K + sens.tc_bias_K + rng.normal(0, sens.tc_sigma_K), 0, sens.tc_max_K)
    md_m = true_mdot_kg_s * (1.0 + rng.normal(0, sens.flow_pct / 100.0))
    return dict(Pc_meas_Pa=float(Pc_m), Texit_meas_K=float(T_m), mdot_meas_kg_s=float(md_m),
                Pc_true_Pa=float(true_Pc_Pa), Texit_true_K=float(true_Texit_K))


# ======================================================================================
# H. INTEGRATED EXECUTION (layers B..E onto the misalign/MIB/finite-burn base)
# ======================================================================================
def _rotate_small(vec, angle_rad, axis):
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return vec * c + np.cross(axis, vec) * s + axis * (axis @ vec) * (1 - c)


def effective_params(prop=None, feed=None):
    """Hi-fi effective execution parameters exported to the pipeline (identity-when-absent knobs):
    blowdown-averaged, c*-efficiency-and-chug-adjusted Isp; finite-burn+chug loss; MIB with the
    water-hammer inflation."""
    prop = prop or HW.AFM315E_PROP; feed = feed or HW.MPS130_FEED
    noz = NozzlePerf(prop)
    isp_eff = noz.isp_vac_s(eff=True)                     # c*-efficiency-adjusted Isp
    cg = chug_state(feed)
    isp_eff *= (1.0 - cg["cstar_penalty"])               # chug penalty (0 if stable)
    mib_nom = HW.MPS130.min_impulse_bit_Ns
    mib_eff = mib_nom * (1.0 + mib_inflation_frac(feed, prop))
    loss = HW.MPS130.finite_burn_loss_frac + cg["cstar_penalty"]
    return dict(isp_s=float(isp_eff), mib_Ns=float(mib_eff), loss_frac=float(loss),
                chugging=cg["chugging"])


def apply_burn_hifi(dv_cmd_vec, mass_kg, rng=None, prop=None, feed=None):
    """Realize a commanded dV through the hi-fi line: misalignment + MIB(water-hammer) quantization
    + finite-burn/chug loss + c*-efficiency Isp.  Returns realized dV, propellant, telemetry."""
    rng = rng or np.random.default_rng()
    prop = prop or HW.AFM315E_PROP; feed = feed or HW.MPS130_FEED
    thr = HW.MPS130
    ep = effective_params(prop, feed)
    dv_cmd = np.asarray(dv_cmd_vec, float); mag = np.linalg.norm(dv_cmd)
    if mag < 1e-12:
        return np.zeros(3), 0.0, dict(mag_cmd=0.0, mag_real=0.0, point_err_deg=0.0, n_mib=0)
    u = dv_cmd / mag
    ang = abs(rng.normal(0.0, np.radians(thr.misalignment_deg_sigma)))
    perp = np.cross(u, rng.normal(size=3))
    if np.linalg.norm(perp) < 1e-9:
        perp = np.array([1.0, 0, 0])
    u_real = _rotate_small(u, ang, perp)
    dv_step = ep["mib_Ns"] / mass_kg
    n_mib = max(1, int(round(mag / dv_step)))
    mag_q = n_mib * dv_step
    mag_real = mag_q * (1.0 - ep["loss_frac"])
    dv_real = mag_real * u_real
    m_prop = mass_kg * (1.0 - np.exp(-mag_real / (ep["isp_s"] * G0)))
    err = dict(mag_cmd=float(mag), mag_real=float(mag_real), point_err_deg=float(np.degrees(ang)),
               mag_err_mps=float(mag_real - mag), n_mib=int(n_mib), isp_eff_s=ep["isp_s"],
               chugging=ep["chugging"])
    return dv_real, float(m_prop), err


def fleet_delta_v_hifi(tcm_list, mass_kg, rng=None, prop=None, feed=None):
    rng = rng or np.random.default_rng()
    m = mass_kg; tot_real = 0.0; tot_prop = 0.0; perr = []; merr = []
    for dv in tcm_list:
        dvr, mp, e = apply_burn_hifi(dv, m, rng, prop, feed)
        tot_real += np.linalg.norm(dvr); tot_prop += mp; m -= mp
        perr.append(e["point_err_deg"]); merr.append(e["mag_err_mps"])
    return dict(dv_total_mps=float(tot_real), propellant_kg=float(tot_prop), final_mass_kg=float(m),
                mean_point_err_deg=float(np.mean(perr)), rms_mag_err_mps=float(np.sqrt(np.mean(np.square(merr)))))


# ======================================================================================
# SELF-TEST
# ======================================================================================
def _selftest():
    fails = []
    def check(name, cond, detail=""):
        print(("  PASS " if cond else "  FAIL ") + name + ("" if not detail else "  -> " + detail))
        if not cond: fails.append(name)

    prop = HW.AFM315E_PROP; feed = HW.MPS130_FEED
    print("=" * 82)
    print("MONOPROPELLANT THRUSTER HI-FI SELF-TEST  (MPS-130 / AF-M315E)")
    print("  Robledo 2020 + Kerr 2021 (catalyst bed) + sub-Newton thesis (c*/chug/blowdown)")
    print("=" * 82)

    # A. c* and Isp
    noz = NozzlePerf(prop)
    c_id = noz.cstar(eff=False); c_eff = noz.cstar(eff=True); cf = noz.cf_vac(); isp = noz.isp_vac_s()
    print(f"\nA. Nozzle/c*: c*_ideal={c_id:.0f} m/s, c*_eff={c_eff:.0f} (eta {prop.cstar_efficiency}), "
          f"CF_vac={cf:.3f}, Isp={isp:.1f} s")
    check("AF-M315E c* ideal ~1400 m/s", 1300 < c_id < 1550, f"{c_id:.0f}")
    check("Isp ~250 s (AFRL)", 235 < isp < 265, f"{isp:.1f}")
    check("CF_vac physical (1.6..2.0)", 1.6 < cf < 2.0, f"{cf:.3f}")
    hz = cstar_ideal(prop.hydrazine_flame_temp_K, 1.27, 13.0)
    check("hydrazine c* < AF-M315E c* (lower flame temp)", hz < c_id, f"N2H4 {hz:.0f} vs {c_id:.0f}")

    # B. catalyst-bed decomposition + Sobol
    bed = CatalystBed(prop, feed)
    nom = bed.exit_state()
    print(f"\nB. Catalyst bed: t_res={bed.t_res_nom*1e3:.1f} ms, completion X={float(nom['completion']):.3f}, "
          f"exit T={float(nom['exit_temp_K']):.0f} K, c* eff={float(nom['cstar_eff']):.3f}")
    check("nominal completion ~ catalytic thermal eff 0.929", abs(float(nom['completion']) - 0.929) < 0.02,
          f"{float(nom['completion']):.3f}")
    check("nominal c* efficiency ~ 0.963 (thesis)", abs(float(nom['cstar_eff']) - 0.963) < 0.01,
          f"{float(nom['cstar_eff']):.3f}")
    cold = bed.exit_state(Ti=250.0); hot = bed.exit_state(Ti=420.0)
    check("preheat helps completion (cold < hot)", float(cold['completion']) < float(hot['completion']),
          f"{float(cold['completion']):.3f} < {float(hot['completion']):.3f}")
    lean = bed.exit_state(Y=0.85)
    check("lower mass fraction lowers exit temp", float(lean['exit_temp_K']) < float(nom['exit_temp_K']))
    S = bed.sobol_first_order(N=8192, seed=1)
    ranked = sorted(S, key=lambda k: -S[k])
    print(f"   Sobol first-order (exit T): " + ", ".join(f"{k}={S[k]:.2f}" for k in ranked))
    check("Sobol: mass fraction Y & activation energy Ea dominate (Kerr)",
          set(ranked[:2]) == {"Y", "Ea"}, f"top2={ranked[:2]}")
    check("Sobol: Y+Ea >> Ti+mdot", (S["Y"] + S["Ea"]) > 3 * (abs(S["Ti"]) + abs(S["mdot"])))

    # C. blowdown
    bd = Blowdown(feed, prop); pc = bd.pressure_curve()
    print(f"\nC. Blowdown: usable prop {bd.usable_prop_kg()*1e3:.0f} g, achieved ratio "
          f"{pc['blowdown_ratio']:.2f} (design {feed.blowdown_ratio}), thrust {pc['thrust_frac'][0]:.2f}->"
          f"{pc['thrust_frac'][-1]:.2f}")
    check("tank pressure decays monotonically", np.all(np.diff(pc["tank_P_Pa"]) < 0))
    check("achieved blowdown ratio ~ design", abs(pc["blowdown_ratio"] - feed.blowdown_ratio) < 0.3,
          f"{pc['blowdown_ratio']:.2f}")
    check("thrust tracks Pc down to 1/ratio", abs(pc["thrust_frac"][-1] - 1.0 / feed.blowdown_ratio) < 0.05)

    # D. chugging
    stable = chug_state(feed, bed)
    from dataclasses import replace as _rep
    soft = chug_state(_rep(feed, injector_stiffness=0.08), bed)
    print(f"\nD. Chug: nominal stiffness {stable['stiffness']:.2f} (thr {stable['threshold']:.2f}) "
          f"chugging={stable['chugging']}; soft-injector chugging={soft['chugging']}, "
          f"f_chug={soft['chug_freq_hz']:.0f} Hz, penalty {soft['cstar_penalty']*100:.1f}%")
    check("stiff injector is stable", not stable["chugging"])
    check("soft injector chugs", soft["chugging"])
    check("chug frequency low-band (10..1000 Hz)", 10 < soft["chug_freq_hz"] < 1000, f"{soft['chug_freq_hz']:.0f}")
    check("chugging imposes a c* penalty", soft["cstar_penalty"] > 0)

    # E. water hammer
    wh = water_hammer(feed, prop)
    print(f"\nE. Water hammer: wave speed {wh['wave_speed_ms']:.0f} m/s, Joukowsky "
          f"{wh['dP_joukowsky_Pa']/1e6:.2f} MPa, effective {wh['dP_effective_Pa']/1e6:.2f} MPa "
          f"(pipe period {wh['pipe_period_s']*1e3:.2f} ms), MIB inflation {mib_inflation_frac(feed,prop)*100:.1f}%")
    a_ref = wave_speed(feed, prop)
    check("wave speed physical (600..1500 m/s)", 600 < a_ref < 1500, f"{a_ref:.0f}")
    check("Joukowsky dP = rho a dv", abs(wh["dP_joukowsky_Pa"] -
          prop.density_kg_m3 * a_ref * feed.valve_flow_velocity_ms) < 1.0)
    check("gradual-closure reduces surge below instantaneous", wh["dP_effective_Pa"] <= wh["dP_joukowsky_Pa"])
    check("valve transient inflates MIB (>0)", mib_inflation_frac(feed, prop) > 0)

    # F. slosh
    s_lo = slosh_state(axial_accel_ms2=0.02); s_hi = slosh_state(axial_accel_ms2=0.2)
    from dataclasses import replace as _rep2
    s_big = slosh_state(_rep2(HW.PROP_SLOSH, tank_radius_m=0.10), axial_accel_ms2=0.2)
    print(f"\nF. Slosh: f={s_hi['freq_hz']:.3f} Hz, slosh-mass frac {s_hi['slosh_mass_frac']:.2f}, "
          f"pend L={s_hi['pend_length_m']*1e3:.0f} mm, dist torque {s_hi['dist_torque_Nm']*1e6:.2f} uNm")
    check("slosh freq rises with axial accel", s_hi["freq_hz"] > s_lo["freq_hz"])
    check("slosh freq falls with tank radius", s_big["freq_hz"] < s_hi["freq_hz"])
    check("slosh mass fraction in (0,0.6)", 0 < s_hi["slosh_mass_frac"] < 0.6, f"{s_hi['slosh_mass_frac']:.2f}")
    check("slosh disturbance torque > 0", s_hi["dist_torque_Nm"] > 0)

    # G. sensors
    rng = np.random.default_rng(3)
    reads = [read_sensors(1.0e6, 1980.0, bed.mdot_nom, rng) for _ in range(4000)]
    Pc_m = np.array([r["Pc_meas_Pa"] for r in reads]); T_m = np.array([r["Texit_meas_K"] for r in reads])
    print(f"\nG. Sensors: Pc {np.mean(Pc_m)/1e5:.2f}+/-{np.std(Pc_m)/1e5:.2f} bar (true 10.0), "
          f"Texit {np.mean(T_m):.0f}+/-{np.std(T_m):.1f} K (true 1980)")
    fs = HW.THR_SENSORS.press_fs_bar * 1e5
    check("pressure noise ~ %FS spec", abs(np.std(Pc_m) - HW.THR_SENSORS.press_pct_fs/100*fs) < 0.15*fs)
    check("thermocouple noise ~ spec", abs(np.std(T_m) - HW.THR_SENSORS.tc_sigma_K) < 0.6)
    check("sensor truth intact (unperturbed)", reads[0]["Pc_true_Pa"] == 1.0e6)

    # H. integrated execution + fleet dV
    rng = np.random.default_rng(1)
    ep = effective_params(prop, feed)
    dvr, mp, e = apply_burn_hifi(np.array([0.30, 0, 0]), HW.XB6.wet_mass_kg, rng)
    tcms = [np.array([0.30, 0, 0]), np.array([0, 0.02, 0]), np.array([0.01, 0, 0.005])]
    fl = [fleet_delta_v_hifi(tcms, HW.XB6.wet_mass_kg, rng) for _ in range(8)]
    tot = sum(f["dv_total_mps"] for f in fl); prop_g = sum(f["propellant_kg"] for f in fl) * 1e3
    print(f"\nH. Execution: eff Isp {ep['isp_s']:.1f} s, eff MIB {ep['mib_Ns']*1e3:.2f} mN-s, "
          f"loss {ep['loss_frac']*100:.1f}%; fleet(8) dV {tot:.2f} m/s, prop {prop_g:.1f} g")
    check("realized burn < commanded (losses)", e["mag_real"] < e["mag_cmd"])
    check("effective Isp reflects c* efficiency (< ideal)", ep["isp_s"] < noz.isp_vac_s(eff=False))
    check("fleet(8) dV in V1 band (~2.0..3.0 m/s)", 1.8 < tot < 3.2, f"{tot:.2f}")
    check("propellant positive & small", 0 < prop_g < 500)

    print("\n" + "=" * 82)
    print(f"RESULT: {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}  ({len(fails)} failure(s))")
    print("=" * 82)
    return len(fails) == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
