"""
isl_comms.py  -  ARISE V2 high-fidelity optical inter-satellite link (ISL) + comms model
==============================================================================
Replaces the flat "ARISE_RELNAV_SIGMA_M = 0.03 m + abstract consensus buffer" (the mission's
weakest-fidelity area, "the long pole" in SAFETY_CASE.md) with a PHYSICAL optical link budget
plus a multi-state channel, from:

  [1] Gioulis, Kamalakis & Alexandropoulos, "Comprehensive Optical Inter-Satellite Communication
      Model for LEO Constellations," Photonics 12, 392 (2025)  -> the LINK BUDGET:
        P_R = L_ch P_T ,  L_ch = (lambda/4piR)^2 nT nR G_T G_R exp(-G_T sigT^2) exp(-G_R sigR^2)
        G_T = 16/Theta_T^2 , G_R = (pi D_R/lambda)^2 , optimal Theta_T = 4 sigT ; OOK BER = Q(gamma).
  [2] Guven, Ben Yahia & Karabulut-Kurt, "Multi-State Inter-Satellite Channel Models" (Polytechnique
      Montreal) -> the CHANNEL: free-space path loss + SOLAR SCINTILLATION (Rician fading vs the
      Sun-Earth-Probe angle) + a K-state Markov chain (GOOD/DEGRADED/OUTAGE) for error bursts/outage.

Outputs for the ARISE swarm (deputies ~0.1-5 km apart, laser crosslinks):
  * link margin, BER, achievable data rate vs range/power/aperture/pointing,
  * ISL RANGING sigma (from the link SNR) -> replaces the flat 0.03 m rel-nav knowledge,
  * outage probability + availability + latency -> physical COMM/PEER fault realism + consensus.

Run `python isl_comms.py` for the self-test (prints PASS/FAIL).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from math import erfc, sqrt

Q_E = 1.602176634e-19; K_B = 1.380649e-23; C_LIGHT = 2.998e8


def Qfunc(x): return 0.5 * erfc(np.asarray(x, float) / np.sqrt(2.0))
def Qinv(p):  # inverse Q via bisection (p in (0,0.5])
    lo, hi = 0.0, 40.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if Qfunc(mid) > p: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class ISLParams:
    """Laser-terminal + channel parameters (defaults from Gioulis et al. Table 2)."""
    wavelength_m: float = 1550e-9
    P_T_W: float = 0.1               # transmit optical power (100 mW; ARISE deputy laser)
    D_R_m: float = 0.08              # receiver telescope diameter (80 mm)
    sigma_point_rad: float = 1e-6    # pointing error (1 urad)
    Theta_T_rad: float = 0.0         # transmit divergence; 0 -> optimal 4*sigma
    nT: float = 0.8                  # transmitter optical efficiency
    nR: float = 0.8                  # receiver optical efficiency
    re_dB: float = 20.0              # extinction ratio
    responsivity_A_W: float = 1.0    # pin responsivity (eta=1 @1550 -> ~1.25; use 1.0)
    Fn_dB: float = 3.0               # amplifier noise figure
    R_load_ohm: float = 50.0         # load resistance
    temp_K: float = 300.0
    target_BER: float = 1e-9         # required BER
    link_margin_req_dB: float = 3.0

    @property
    def Theta_T(self):
        return self.Theta_T_rad if self.Theta_T_rad > 0 else 4.0 * self.sigma_point_rad  # [1] Eq 16

    def G_T(self): return 16.0 / self.Theta_T ** 2                                    # [1] Eq 13
    def G_R(self): return (np.pi * self.D_R_m / self.wavelength_m) ** 2               # [1] Eq 14

    def channel_gain(self, range_m):
        """L_ch = (lambda/4piR)^2 nT nR G_T G_R exp(-G_T sigT^2) exp(-G_R sigR^2)  [1] Eq 12."""
        Lfs = (self.wavelength_m / (4.0 * np.pi * np.asarray(range_m, float))) ** 2
        GT, GR = self.G_T(), self.G_R()
        Lpt = np.exp(-GT * self.sigma_point_rad ** 2) * np.exp(-GR * self.sigma_point_rad ** 2)
        far = Lfs * self.nT * self.nR * GT * GR * Lpt
        cap = self.nT * self.nR * Lpt          # near-field: beam < aperture -> all power captured (x efficiency, pointing)
        return np.minimum(far, cap)

    def received_power_W(self, range_m):
        return self.channel_gain(range_m) * self.P_T_W                               # [1] Eq 11

    # ---- receiver sensitivity from the required BER (shot+thermal noise, OOK) ----
    def receiver_sensitivity_W(self, data_rate_bps):
        Be = data_rate_bps / 2.0                                                     # roll-off alpha=0
        Fn = 10 ** (self.Fn_dB / 10.0)
        sig_th = np.sqrt(4.0 * K_B * self.temp_K * Fn * Be / self.R_load_ohm)        # thermal current noise [A]
        re = 10 ** (self.re_dB / 10.0)
        gamma = Qinv(self.target_BER)                                                # required Q-arg [1] Eq 17
        R = self.responsivity_A_W
        # gamma = 2R (re-1)/(re+1) Pavg/(sig1+sig0); shot ~ small vs thermal near sensitivity -> sig~sig_th
        Pavg = gamma * (2.0 * sig_th) / (2.0 * R * (re - 1.0) / (re + 1.0))
        return float(Pavg)                                                           # W (avg optical)

    def link_margin_dB(self, range_m, data_rate_bps=10e9):
        return 10.0 * np.log10(self.received_power_W(range_m) / self.receiver_sensitivity_W(data_rate_bps))

    def max_data_rate_bps(self, range_m):
        """Largest data rate whose sensitivity still meets the required margin at this range."""
        PR = self.received_power_W(range_m)
        Pallow = PR / 10 ** (self.link_margin_req_dB / 10.0)                          # allowed sensitivity
        Fn = 10 ** (self.Fn_dB / 10.0); re = 10 ** (self.re_dB / 10.0); R = self.responsivity_A_W
        gamma = Qinv(self.target_BER)
        # invert Pavg(Rb): Pavg = gamma*2*sqrt(4kT Fn (Rb/2)/RL)/(2R(re-1)/(re+1))  ->  Rb ~ Pallow^2
        kfac = gamma * 2.0 / (2.0 * R * (re - 1.0) / (re + 1.0))
        # Pallow = kfac*sqrt(4kT Fn (Rb/2)/RL)  -> Rb = 2*RL/(4kT Fn) * (Pallow/kfac)^2
        Rb = 2.0 * self.R_load_ohm / (4.0 * K_B * self.temp_K * Fn) * (Pallow / kfac) ** 2
        return float(max(Rb, 0.0))

    def snr_linear(self, range_m, data_rate_bps=10e9):
        Be = data_rate_bps / 2.0; Fn = 10 ** (self.Fn_dB / 10.0)
        sig_th = np.sqrt(4.0 * K_B * self.temp_K * Fn * Be / self.R_load_ohm)
        I_sig = self.responsivity_A_W * self.received_power_W(range_m)
        return float(I_sig / max(sig_th, 1e-30))

    # ---- ISL ranging sigma from the link (strong link -> precise ranging) ----
    def ranging_sigma_m(self, range_m, mod_bw_hz=1e9):
        """Two-way laser ranging precision [m] ~ c/(4 pi f_mod sqrt(2 SNR)) (Cramer-Rao-like),
        flooring at a hardware limit.  Replaces the flat 0.03 m rel-nav knowledge."""
        snr = max(self.snr_linear(range_m), 1e-6)
        sigma = C_LIGHT / (4.0 * np.pi * mod_bw_hz * np.sqrt(2.0 * snr))
        return float(np.clip(sigma, 5e-3, 10.0))                                     # 5 mm systematic floor .. 10 m cap


# ================= multi-state channel: solar scintillation (Guven et al.) =================
def scintillation_index(sep_angle_deg, a1=1.14, theta0_deg=1.3):
    """Scintillation index m in [0,1] vs Sun-Earth-Probe (SEP) angle  [2] Eq 4.
    Small SEP (near the Sun) -> strong scintillation (m->1)."""
    phi = abs(float(sep_angle_deg))
    m = 1.0 if phi <= theta0_deg else float(np.exp(-a1 * (phi - theta0_deg)))   # strong near Sun, decays with SEP
    return float(np.clip(m, 0.0, 1.0))


def rician_K(m):
    """Rician K-factor from the scintillation index m  [2] Eq 3.  m->0 strong LoS (large K);
    m->1 heavy fading (K->0, Rayleigh-like)."""
    m = float(np.clip(m, 1e-6, 0.999999))
    s = np.sqrt(1.0 - m * m)
    return float(s / (1.0 - s))


def outage_probability(link_margin_dB, m):
    """Prob. the faded received power drops below the sensitivity (Rician fading over the margin).
    Combines the deterministic link margin with the scintillation fading depth."""
    K = rician_K(m)
    margin_lin = 10 ** (link_margin_dB / 10.0)                                        # P_R/P_sens
    thr = 1.0 / max(margin_lin, 1e-12)                                                # fade depth to sensitivity
    # Rician outage leading term P(|h|^2<thr) ~ (K+1) thr exp(-K): falls with margin AND with K (LoS)
    return float(np.clip((K + 1.0) * thr * np.exp(-K), 0.0, 1.0))


def channel_state(link_margin_dB, sep_angle_deg):
    """3-state Markov channel [2]: returns (state, outage_prob, availability, scintillation m)."""
    m = scintillation_index(sep_angle_deg)
    op = outage_probability(link_margin_dB, m)
    if op < 0.01 and link_margin_dB > 6:   state = "GOOD"
    elif op < 0.2:                          state = "DEGRADED"
    else:                                   state = "OUTAGE"
    return dict(state=state, outage_prob=op, availability=1.0 - op, scintillation_m=m,
                rician_K=rician_K(m))


def latency_s(range_m, proc_latency_s=1e-3):
    """One-way ISL latency = light time + processing."""
    return range_m / C_LIGHT + proc_latency_s


# ================= convenience summary =================
def link_summary(range_m, sep_angle_deg=90.0, data_rate_bps=10e9, p: ISLParams = ISLParams()):
    lm = float(p.link_margin_dB(range_m, data_rate_bps))
    cs = channel_state(lm, sep_angle_deg)
    return dict(range_m=range_m, link_margin_dB=lm, snr_dB=10*np.log10(p.snr_linear(range_m, data_rate_bps)),
                max_data_rate_Gbps=p.max_data_rate_bps(range_m)/1e9,
                ranging_sigma_m=p.ranging_sigma_m(range_m), latency_ms=latency_s(range_m)*1e3, **cs)


DEFAULT = ISLParams()


# ================= SELF-TEST =================
def _selftest():
    fails = []
    def check(n, c, d=""):
        print(("  PASS " if c else "  FAIL ") + n + ("" if not d else "  -> " + d)); (fails.append(n) if not c else None)
    print("=" * 78); print("OPTICAL ISL + COMMS SELF-TEST  (Gioulis 2025 link budget + Guven multi-state)"); print("=" * 78)
    p = DEFAULT

    # 1. Link-budget terms reproduce the paper's Table-2 numbers
    GT_dB = 10*np.log10(p.G_T()); GR_dB = 10*np.log10(p.G_R())
    LT_dB = 10*np.log10(np.exp(-p.G_T()*p.sigma_point_rad**2)); LR_dB = 10*np.log10(np.exp(-p.G_R()*p.sigma_point_rad**2))
    print(f"\n1. Link-budget gains/losses (paper: G_T=120 G_R=104.2 dB, L_T=-4.3 L_R=-0.11 dB):")
    print(f"     G_T={GT_dB:.1f} dB  G_R={GR_dB:.1f} dB  L_T={LT_dB:.2f} dB  L_R={LR_dB:.2f} dB  (Theta_T={p.Theta_T*1e6:.1f} urad)")
    check("G_T ~ 120 dB", abs(GT_dB-120) < 1.0)
    check("G_R ~ 104.2 dB", abs(GR_dB-104.2) < 1.0)
    check("pointing loss L_T ~ -4.3 dB (optimal Theta=4sigma)", abs(LT_dB+4.34) < 0.3)
    check("pointing loss L_R ~ -0.11 dB", abs(LR_dB+0.11) < 0.05)

    # 2. Receiver sensitivity ~ -20 dBm at 10 Gbps (paper Fig 6)
    Psens_dBm = 10*np.log10(p.receiver_sensitivity_W(10e9)/1e-3)
    print(f"\n2. Receiver sensitivity @10 Gbps: {Psens_dBm:.1f} dBm  (paper ~ -20 dBm)")
    check("receiver sensitivity ~ -20 dBm @10 Gbps", abs(Psens_dBm+20) < 6.0, f"{Psens_dBm:.1f}")

    # 3. ARISE swarm ranges (0.1-5 km): strong link, high margin, high rate, precise ranging
    print(f"\n3. ARISE swarm link (100 mW, 80 mm, 1 urad) vs range:")
    print(f"   {'range':>7s} {'margin dB':>10s} {'max Gbps':>9s} {'ranging σ':>11s} {'latency':>9s}")
    for R in [100., 500., 1000., 3000., 5000.]:
        s = link_summary(R, sep_angle_deg=90.0)
        print(f"   {R:6.0f}m {s['link_margin_dB']:9.1f} {s['max_data_rate_Gbps']:8.0f} {s['ranging_sigma_m']*1e3:8.2f} mm {s['latency_ms']:7.2f}ms")
    s1 = link_summary(1000.)
    check("km-range link has large margin (>20 dB)", s1['link_margin_dB'] > 20)
    check("km-range ranging sigma <= 3 cm (beats flat 0.03 m)", s1['ranging_sigma_m'] <= 0.03)
    check("intra-swarm margin ~constant (aperture-filling, beam<aperture)", abs(link_summary(500.)['link_margin_dB']-link_summary(5000.)['link_margin_dB']) < 1.0)
    check("far-field margin drops with range (100 km < 30 km)", link_summary(100e3)['link_margin_dB'] < link_summary(30e3)['link_margin_dB'])
    # 3b. POINTING-FRAGILE: the 4 urad beam dies if pointing degrades (e.g. during the CA slew)
    print("\n3b. Pointing sensitivity (the ISL's real risk -- a 4 urad beam):")
    for sp in [1e-6, 2e-6, 3e-6, 5e-6]:
        pp = ISLParams(sigma_point_rad=sp, Theta_T_rad=4e-6)          # FIXED 4 urad beam (design)
        print(f"     pointing {sp*1e6:.0f} urad (fixed 4 urad beam) -> link margin {pp.link_margin_dB(1000.):7.1f} dB")
    check("nominal 1 urad pointing closes the link (>10 dB)", ISLParams(sigma_point_rad=1e-6, Theta_T_rad=4e-6).link_margin_dB(1000.) > 10)
    check("degraded pointing (5 urad slew) drops the fixed-beam ISL (<0 dB)", ISLParams(sigma_point_rad=5e-6, Theta_T_rad=4e-6).link_margin_dB(1000.) < 0)

    # 4. Multi-state channel: solar scintillation vs SEP angle -> outage near the Sun
    print(f"\n4. Multi-state channel (solar scintillation vs SEP angle, R=1 km):")
    for sep in [90., 30., 10., 3., 0.]:
        cs = channel_state(s1['link_margin_dB'], sep)
        print(f"     SEP={sep:5.1f} deg -> m={cs['scintillation_m']:.2f} K={cs['rician_K']:6.1f} state={cs['state']:8s} avail={cs['availability']*100:5.1f}%")
    check("weak scintillation far from Sun (SEP=90 -> GOOD)", channel_state(s1['link_margin_dB'],90)['state']=="GOOD")
    check("strong scintillation near Sun raises outage", channel_state(s1['link_margin_dB'],2)['outage_prob'] > channel_state(s1['link_margin_dB'],90)['outage_prob'])
    check("scintillation index in [0,1] and rises toward Sun", scintillation_index(2) > scintillation_index(60))

    # 5. optimal divergence + BER
    check("optimal divergence Theta_T = 4 sigma", abs(p.Theta_T - 4*p.sigma_point_rad) < 1e-12)
    ber_close = Qfunc(np.sqrt(2*p.snr_linear(1000.)))    # rough OOK BER proxy from SNR
    check("BER effectively zero on the strong km link", ber_close < 1e-9)

    print("\n" + "=" * 78)
    print(f"RESULT: {'ALL PASS' if not fails else 'FAILED: '+', '.join(fails)}  ({len(fails)} failure(s))")
    print("=" * 78)
    return len(fails) == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
