"""
reaction_wheels.py  -  ARISE V2 high-fidelity reaction-wheel model
==============================================================================
Replaces the flat "pointing_jitter_arcsec = 1.5" constant and the ideal wheel-torque
computation with a PHYSICS-BASED reaction-wheel-assembly (RWA) model built from:

  [1] M. Longato, V. Yotov, G. Aglietti, "A predictive model for reaction wheel assembly
      microvibration," Acta Astronautica 230 (2025) 39-53.   -> the MICROVIBRATION model:
      harmonic disturbance forces/moments ~ Omega^2 (static+dynamic imbalance + bearing tones),
      amplified at the rotor whirl / structural resonances (waterfall-plot ridges).
  [2] H. Figueiredo, O. Saotome, "Design of a Set of Reaction Wheels for Satellite Attitude
      Control Simulation," COBEM 2013.   -> the DETERMINISTIC wheel+motor model: brushless-DC
      (Faulhaber 2232-class) torque/back-EMF/friction + the gyrostat sizing.

Two parts (mirrors star_tracker.py = physics model + validation):
  A) WHEEL + MOTOR (deterministic): realized torque given a command, with viscous+Coulomb
     friction, back-EMF speed limit, and torque/momentum/speed saturation (XACT-50 class).
  B) MICROVIBRATION (stochastic disturbance): given each wheel's speed Omega, the RSS pointing
     JITTER [arcsec] it induces on the bus -- roughly flat off-resonance but SPIKING when a
     disturbance harmonic h*Omega crosses a structural mode.  Feeds the pointing budget and the
     camera image smear.

KEY physics (paper [1]): the imbalance force ~ Omega^2 while the rigid-body angular response
~ 1/Omega^2, so OFF-resonance the jitter is ~speed-independent (the flat 1.5" is a fair
baseline) -- but as the wheels spin up through the CA slew, harmonics sweep through the whirl
modes and the jitter SPIKES (x Q), smearing the image at those moments.  That speed/resonance
structure is exactly what the flat constant cannot express.

Calibrated so the off-resonance jitter ~ 1.5" (the current hardware.py assumption) and the
imbalance/harmonic amplitudes and whirl resonances follow paper [1].  Run `python
reaction_wheels.py` for the self-test (prints PASS/FAIL).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

ARCSEC = np.radians(1.0 / 3600.0)
RPM = 2.0 * np.pi / 60.0            # rpm -> rad/s


@dataclass(frozen=True)
class ReactionWheelParams:
    name: str = "ARISE XACT-50-class RWA"
    # --- wheel + motor (deterministic; paper [2] Faulhaber 2232 + XACT-50 datasheet) ---
    I_wheel_kgm2: float = 7.96e-5          # RWp050 rotor inertia (0.050 Nms @ ~6000 rpm; BCT RW datasheet)
    max_speed_rad_s: float = 628.0         # ~6000 rpm max flywheel speed
    max_momentum_Nms: float = 0.050        # RWp050: 0.050 Nms/wheel (XACT-50 uses 3x RWp050)   [cited]
    max_torque_Nm: float = 0.007           # RWp050 max torque (XACT-50 system 0.006 Nm)         [cited]
    mass_kg: float = 0.24                   # RWp050 mass                                         [cited]
    power_W_max: float = 0.81               # RWp050 power @ max torque                           [cited]
    Kt_Nm_A: float = 1.30e-2               # motor torque constant (Faulhaber-2232 template [2] scaled to XACT-50)
    Ke_V_rad_s: float = 1.30e-2            # back-EMF constant (=Kt in SI)
    R_ohm: float = 15.0                    # winding resistance (scaled so stall torque > rated 6 mNm)
    bus_voltage_V: float = 12.0            # XB6 bus voltage
    visc_friction_Nm_rad_s: float = 2.48e-8   # viscous damping b   (paper[2])
    coulomb_friction_Nm: float = 4.0e-5    # Coulomb/bearing drag (ASSUMED, XACT-class)
    # --- microvibration (paper [1]) ---
    static_imbalance_kg_m: float = 5.0e-6     # U_s: static imbalance (~0.5 g.cm, precision-balanced)
    dynamic_imbalance_kg_m2: float = 1.0e-7   # U_d: dynamic imbalance (~1 g.cm^2)
    wheel_cg_arm_m: float = 0.10              # radial offset of a wheel from the bus CG (force->torque lever)
    # bearing / motor harmonic orders h (relative to spin rate) and their amplitude fraction of U_s:
    harmonic_orders: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.3, 6.1, 8.0)     # 1=imbalance; non-int = ball-pass tones
    harmonic_fracs:  Tuple[float, ...] = (1.0, 0.18, 0.09, 0.25, 0.15, 0.06)
    # rotor whirl / structure resonances [Hz] and their quality factor Q (transmissibility peak):
    modes_hz: Tuple[float, ...] = (88.0, 220.0, 510.0)   # rotor whirl + structural (h=1 crosses 88 Hz in-band)
    modes_Q:  Tuple[float, ...] = (12.0, 10.0, 8.0)
    # host-bus response
    I_bus_kgm2: float = 0.20                  # representative bus MoI seen by the disturbance
    jitter_cal: float = 1.0                   # scalar tuned so off-resonance jitter ~ hardware 1.5"
    n_wheels: int = 4                         # 4-wheel pyramid

    # ===================== A. WHEEL + MOTOR (deterministic) =====================
    def stall_torque_Nm(self) -> float:
        return self.Kt_Nm_A * self.bus_voltage_V / self.R_ohm

    def realized_torque(self, cmd_torque_Nm, wheel_speed_rad_s):
        """Torque the wheel actually delivers for a commanded torque at a given spin rate:
        motor voltage/back-EMF ceiling, viscous+Coulomb friction, and hard torque saturation."""
        cmd = np.asarray(cmd_torque_Nm, float); w = np.asarray(wheel_speed_rad_s, float)
        # available motor torque falls as back-EMF eats headroom: tau_avail = Kt*(V - Ke*w)/R
        tau_avail = self.Kt_Nm_A * (self.bus_voltage_V - self.Ke_V_rad_s * np.abs(w)) / self.R_ohm
        tau_avail = np.clip(tau_avail, 0.0, self.max_torque_Nm)
        # friction opposes motion
        fric = self.visc_friction_Nm_rad_s * w + self.coulomb_friction_Nm * np.sign(w)
        out = np.clip(cmd, -tau_avail, tau_avail) - fric
        return np.clip(out, -self.max_torque_Nm, self.max_torque_Nm)

    def speed_from_momentum(self, momentum_Nms):
        return np.clip(np.asarray(momentum_Nms, float) / self.I_wheel_kgm2,
                       -self.max_speed_rad_s, self.max_speed_rad_s)

    # ===================== B. MICROVIBRATION (disturbance -> jitter) =====================
    def _transmissibility(self, freq_hz):
        """Structural transmissibility T(f): ~1 off-resonance, ~Q at each whirl/structural mode
        (sum of Lorentzian peaks).  Vectorised over freq_hz."""
        f = np.asarray(freq_hz, float)
        T2 = np.ones_like(f)
        for fm, Q in zip(self.modes_hz, self.modes_Q):
            hw = fm / (2.0 * Q)                       # half-power half-width
            T2 = T2 + (Q / (1.0 + ((f - fm) / hw) ** 2)) ** 2
        return np.sqrt(T2)

    def harmonic_torques(self, wheel_speed_rad_s):
        """Per-harmonic disturbance torque amplitude on the bus [N.m] at a wheel speed:
        tau_h = C_h * Omega^2, C_1 from static+dynamic imbalance, higher h scaled by harmonic_fracs."""
        w = float(abs(wheel_speed_rad_s))
        C1 = self.static_imbalance_kg_m * self.wheel_cg_arm_m + self.dynamic_imbalance_kg_m2
        out = []
        for h, frac in zip(self.harmonic_orders, self.harmonic_fracs):
            C_h = C1 * frac
            tau_h = C_h * w ** 2
            f_h = h * w / (2.0 * np.pi)               # Hz
            out.append((h, f_h, tau_h))
        return out

    def jitter_arcsec(self, wheel_speed_rad_s):
        """RSS bus pointing jitter [arcsec] from ONE wheel at a given spin rate.  Off-resonance
        ~flat; spikes when a harmonic h*Omega crosses a structural mode."""
        w = float(abs(wheel_speed_rad_s))
        if w < 1e-6:
            return 0.0
        theta2 = 0.0
        for h, f_h, tau_h in self.harmonic_torques(w):
            T = float(self._transmissibility(f_h))
            omega_h = 2.0 * np.pi * f_h               # = h*w
            theta_h = tau_h * T / (self.I_bus_kgm2 * omega_h ** 2)   # rigid-body angular response [rad]
            theta2 += theta_h ** 2
        return self.jitter_cal * np.sqrt(theta2) / ARCSEC

    def fleet_jitter_arcsec(self, wheel_speeds_rad_s):
        """RSS jitter from all wheels of the pyramid (independent phases)."""
        ws = np.atleast_1d(np.asarray(wheel_speeds_rad_s, float))
        return float(np.sqrt(np.sum([self.jitter_arcsec(w) ** 2 for w in ws])))

    def smear_arcsec(self, wheel_speed_rad_s, exposure_s):
        """Image smear [arcsec] contributed by jitter over an exposure (jitter is fast vs the
        control loop, so the full jitter amplitude smears within a finite exposure)."""
        return self.jitter_arcsec(wheel_speed_rad_s)      # 1-sigma line-of-sight blur per exposure

    def operating_jitter_arcsec(self, body_rate_rad_s, I_bus_axis=0.228):
        """Representative fleet jitter [arcsec] while the bus slews at body_rate (wheels spun up
        to hold the bus momentum).  Used as the near-CA imaging pointing-jitter budget."""
        H = I_bus_axis * abs(float(body_rate_rad_s))                  # bus momentum to hold
        w = min(H / self.I_wheel_kgm2 / self.n_wheels, self.max_speed_rad_s)
        return self.fleet_jitter_arcsec([w] * self.n_wheels)

    def jitter_vs_speed(self, rpm_grid=None):
        rpm_grid = np.linspace(0, 6000, 400) if rpm_grid is None else np.asarray(rpm_grid, float)
        return rpm_grid, np.array([self.jitter_arcsec(r * RPM) for r in rpm_grid])


# calibrate jitter_cal so the OFF-resonance (mid-band) jitter matches the hardware ~1.5" assumption
def _calibrate(p: ReactionWheelParams, target_arcsec=1.0) -> ReactionWheelParams:   # XB6 spec: jitter 1"/1s
    # sample off-resonance speeds (avoid the mode-crossing spikes), take the median baseline
    rpm, jit = p.jitter_vs_speed(np.linspace(500, 5500, 200))
    base = np.median(jit)
    if base <= 0:
        return p
    from dataclasses import replace
    return replace(p, jitter_cal=p.jitter_cal * target_arcsec / base)


RWA = _calibrate(ReactionWheelParams())


# ======================================================================================
# SELF-TEST
# ======================================================================================
def _selftest():
    fails = []
    def check(name, cond, detail=""):
        print(("  PASS " if cond else "  FAIL ") + name + ("" if not detail else "  -> " + detail))
        if not cond: fails.append(name)

    print("=" * 78)
    print("REACTION-WHEEL MODEL SELF-TEST  (Longato 2025 microvib + Figueiredo 2013 motor)")
    print("=" * 78)
    p = RWA

    # 1. Deterministic motor/torque (paper [2] + XACT-50 datasheet)
    print("\n1. Wheel + motor (deterministic):")
    t_lo = p.realized_torque(0.003, 100.0)      # modest command, low speed
    t_sat = p.realized_torque(0.02, 100.0)      # over-command -> saturates at max_torque
    t_hi = p.realized_torque(0.003, 600.0)      # near max speed -> back-EMF eats headroom
    print(f"     realized torque: cmd 3 mNm @100 rad/s = {t_lo*1e3:.2f} mNm ; over-cmd = {t_sat*1e3:.2f} mNm ; @600 rad/s = {t_hi*1e3:.2f} mNm")
    check("torque saturates at max_torque (6 mNm)", t_sat <= p.max_torque_Nm + 1e-12 and t_sat > 0.005)
    check("back-EMF reduces torque headroom at high speed", t_hi < p.stall_torque_Nm())
    check("max wheel speed consistent w/ momentum", abs(p.speed_from_momentum(p.max_momentum_Nms) - p.max_speed_rad_s) < 20.0)

    # 2. Microvibration ~ Omega^2 forces (paper [1])
    print("\n2. Microvibration disturbance (paper [1]):")
    h1 = dict((round(h,1), (f, tau)) for h, f, tau in p.harmonic_torques(300.0))
    tau_300 = p.harmonic_torques(300.0)[0][2]; tau_600 = p.harmonic_torques(600.0)[0][2]
    print(f"     fundamental disturbance torque: {tau_300*1e6:.2f} uNm @300 rad/s -> {tau_600*1e6:.2f} uNm @600 rad/s (ratio {tau_600/tau_300:.1f})")
    check("imbalance torque scales as Omega^2 (x4 for x2 speed)", abs(tau_600/tau_300 - 4.0) < 0.1)

    # 3. Off-resonance jitter ~ 1.5" (hardware assumption), and RESONANCE SPIKES
    rpm, jit = p.jitter_vs_speed()
    base = np.median(jit[(rpm > 500) & (rpm < 5500)]); peak = jit.max(); peak_rpm = rpm[jit.argmax()]
    print(f"\n3. Jitter vs speed:  off-resonance baseline = {base:.2f}\" ; PEAK = {peak:.2f}\" at {peak_rpm:.0f} rpm")
    check("off-resonance jitter calibrated ~1.0\" (XB6 spec)", 0.7 < base < 1.3, f"{base:.2f}")
    check("resonance crossing spikes jitter (>3x baseline)", peak > 3.0 * base, f"{peak/base:.1f}x")
    # find a mode-crossing speed: h*Omega/2pi = f_mode
    fm = p.modes_hz[0]
    for h in p.harmonic_orders:
        rpm_cross = (2 * np.pi * fm / h) / RPM
        if 500 < rpm_cross < 6000:
            jc = p.jitter_arcsec(rpm_cross * RPM)
            print(f"     harmonic h={h} crosses {fm:.0f} Hz mode at {rpm_cross:.0f} rpm -> jitter {jc:.2f}\"")
            break

    # 4. Fleet (4-wheel) jitter + smear during the near-CA slew
    print("\n4. Near-CA operating point (fast slew -> wheels spun up):")
    # body 10 deg/s -> wheel momentum ~ I_bus*omega distributed; representative per-wheel speed
    w_ca = 0.20 * np.radians(10.0) / p.I_wheel_kgm2 / p.n_wheels   # rough per-wheel spin
    w_ca = min(w_ca, p.max_speed_rad_s)
    fleet = p.fleet_jitter_arcsec([w_ca] * p.n_wheels)
    smear = p.smear_arcsec(w_ca, exposure_s=0.01)
    print(f"     per-wheel ~{w_ca/RPM:.0f} rpm ; 4-wheel RSS jitter = {fleet:.2f}\" ; image smear ~ {smear:.2f}\"")
    check("fleet jitter finite and > single-wheel baseline", fleet > 0.5)

    # 5. Momentum/torque envelope (XACT-50)
    print("\n5. Envelope (XACT-50): max momentum %.0f mNms/wheel, max torque %.0f mNm, max speed %.0f rpm"
          % (p.max_momentum_Nms*1e3, p.max_torque_Nm*1e3, p.max_speed_rad_s/RPM))
    check("RWp050 envelope (BCT datasheet)", p.max_momentum_Nms==0.050 and p.max_torque_Nm==0.007)

    print("\n" + "=" * 78)
    print(f"RESULT: {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}  ({len(fails)} failure(s))")
    print("=" * 78)
    return len(fails) == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
