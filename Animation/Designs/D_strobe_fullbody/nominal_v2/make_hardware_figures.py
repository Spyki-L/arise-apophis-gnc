#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_hardware_figures.py -- journal figures for the hi-fi HARDWARE models (report section).
================================================================================================
Generates H1..H4 (vector PDF + 300-dpi PNG) into outputs/reports/figs/ from the validated
nominal_v2/modules physics (no pipeline run needed):
  H1_propulsion    : MPS-130/AF-M315E line -- blowdown, water-hammer, chug, Isp ladder
  H2_isl           : optical inter-satellite link -- margin / ranging sigma / data rate / outage
  H3_imu_gyro      : nav-grade MEMS Allan model + EKF rate knowledge + ST-dropout drift + event-cam
  H4_rcs_attitude  : RCS execution (MIB staircase, misalignment scatter) + RW jitter + pointing RSS
Run:  python make_hardware_figures.py     (any numpy+matplotlib env; gates not required)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "modules"))
FIGD = os.path.join(os.path.dirname(HERE), "outputs", "reports", "figs")
os.makedirs(FIGD, exist_ok=True)

import hardware as HW
import isl_comms as ISL
import thruster_hifi as THX
import thruster as THR
import imu_gyro as IMU
import reaction_wheels as RW

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 110,
                     "axes.grid": True, "grid.alpha": 0.3})

def save(fig, name):
    fig.savefig(os.path.join(FIGD, name + ".pdf"))
    fig.savefig(os.path.join(FIGD, name + ".png"), dpi=300)
    plt.close(fig); print("  wrote", name, ".pdf/.png")

# ================= H1  PROPULSION (thruster_hifi) =================
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.4), constrained_layout=True)
bd = THX.Blowdown(); pc = bd.pressure_curve(120)
a = ax[0, 0]
a.plot(pc["m_expelled_kg"], pc["tank_P_Pa"] / 1e5, label="tank $P$")
a.plot(pc["m_expelled_kg"], pc["chamber_P_Pa"] / 1e5, label="chamber $P_c$")
a2 = a.twinx(); a2.plot(pc["m_expelled_kg"], pc["thrust_frac"], "g--", lw=1, label="thrust frac.")
a2.set_ylabel("thrust fraction $P/P_0$", color="g"); a2.grid(False)
a.set_xlabel("propellant expelled [kg]"); a.set_ylabel("pressure [bar]")
a.set_title(f"(a) blowdown: $P=P_0(V_{{u0}}/V_u)^\\gamma$, ratio {pc['blowdown_ratio']:.1f}")
a.legend(loc="center left")
a = ax[0, 1]
tc = np.logspace(-4, -0.5, 80)
dp = np.array([THX.water_hammer(closure_s=t)["dP_effective_Pa"] for t in tc]) / 1e6
wh = THX.water_hammer()
a.loglog(tc * 1e3, dp)
a.axhline(THX.water_hammer(closure_s=1e-6)["dP_effective_Pa"] / 1e6, color="r", ls=":",
          label="Joukowsky $\\rho a \\Delta v$")
a.axhline(wh["dP_effective_Pa"] / 1e6, color="k", ls="-.", lw=1,
          label=f"design surge {wh['dP_effective_Pa']/1e6:.2f} MPa (2L/a={wh['pipe_period_s']*1e3:.2f} ms)")
a.set_xlabel("valve closure time [ms]"); a.set_ylabel("surge $\\Delta P$ [MPa]")
a.set_title(f"(b) water-hammer surge $\\to$ MIB +{THX.mib_inflation_frac()*100:.0f}%"); a.legend()
a = ax[1, 0]
ep = THX.effective_params(); noz = THX.NozzlePerf(HW.AFM315E_PROP)
vals = [HW.MPS130.isp_s, noz.isp_vac_s(eff=False), ep["isp_s"]]
lbls = ["flat datasheet\n245 s", "ideal $c^*$\n(1-D equil.)", f"effective\n{ep['isp_s']:.1f} s"]
bars = a.bar(lbls, vals, color=["0.7", "tab:blue", "tab:green"])
a.set_ylim(235, max(vals) * 1.03); a.set_ylabel("vacuum $I_{sp}$ [s]")
a.set_title(f"(c) $I_{{sp}}$ ladder ($\\eta_{{c^*}}$={noz.prop.cstar_eff if hasattr(noz.prop,'cstar_eff') else 0.963:.3f}, loss {ep['loss_frac']*100:.1f}%)")
for b, v in zip(bars, vals): a.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)
a = ax[1, 1]
cg = THX.chug_state()
bars = a.bar(["operating\nstiffness", "chug onset\nthreshold"], [cg["stiffness"], cg["threshold"]],
             color=["tab:green" if not cg["chugging"] else "tab:red", "0.6"])
for b, v in zip(bars, (cg["stiffness"], cg["threshold"])):
    a.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.2f}", ha="center", fontsize=8)
a.set_ylabel("injector stiffness $\\Delta P/P_c$")
a.set_title(f"(d) chug margin {cg['margin']:.2f} ({'UNSTABLE' if cg['chugging'] else 'stable'}), "
            f"$f_{{chug}}$={cg['chug_freq_hz']:.0f} Hz")
fig.suptitle("MPS-130 / AF-M315E propulsion line -- hi-fi execution model", fontsize=10)
save(fig, "H1_propulsion")

# ================= H2  OPTICAL ISL (isl_comms) =================
p = ISL.DEFAULT
rng_m = np.logspace(np.log10(50), np.log10(5e4), 120)
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.4), constrained_layout=True)
a = ax[0, 0]
lm = np.array([p.link_margin_dB(r) for r in rng_m])
a.semilogx(rng_m / 1e3, lm); a.axhline(3, color="orange", ls="--", lw=1, label="3 dB")
a.axhline(0, color="r", ls="--", lw=1, label="0 dB")
a.plot([1.0], [p.link_margin_dB(1e3)], "ko", ms=5, label=f"design 1 km ({p.link_margin_dB(1e3):.0f} dB)")
a.set_xlabel("inter-deputy range [km]"); a.set_ylabel("link margin [dB]")
a.set_title("(a) link margin (plateau = full-aperture capture, 4 $\\mu$rad beam)"); a.legend()
a = ax[0, 1]
rs = np.array([p.ranging_sigma_m(r) for r in rng_m]) * 1e3
a.loglog(rng_m / 1e3, rs); a.axhline(30, color="0.5", ls="--", label="flat assumption 30 mm")
a.plot([1.0], [p.ranging_sigma_m(1e3) * 1e3], "ko", ms=5,
       label=f"design 1 km ({p.ranging_sigma_m(1e3)*1e3:.1f} mm)")
a.set_xlabel("range [km]"); a.set_ylabel("ranging $\\sigma$ [mm]")
a.set_title("(b) two-way ranging knowledge"); a.legend()
a = ax[1, 0]
dr = np.array([p.max_data_rate_bps(r) for r in rng_m]) / 1e9
a.loglog(rng_m / 1e3, dr)
a.set_xlabel("range [km]"); a.set_ylabel("max data rate [Gbps]")
a.set_title("(c) shot-noise rate ceiling (used at 10 Gbps; margin 33 dB)")
a = ax[1, 1]
ang = np.linspace(0.2, 20, 200)
for margin in (3.0, 6.0, 10.0, 33.0):
    po = np.array([ISL.outage_probability(margin, ISL.scintillation_index(s)) for s in ang])
    a.semilogy(ang, np.clip(po, 1e-14, 1), label=f"margin {margin:.0f} dB")
a.axvspan(0, 1.3, color="orange", alpha=0.15, label="$m=1$ inside $1.3^\\circ$ (Rayleigh)")
a.set_xlabel("Sun--Earth--probe (SEP) angle [deg]"); a.set_ylabel("outage probability")
a.set_title("(d) solar-scintillation outage (Rician fading)"); a.legend(fontsize=7)
fig.suptitle("Optical inter-satellite link -- Gioulis 2025 budget + multi-state channel", fontsize=10)
save(fig, "H2_isl")

# ================= H3  IMU / GYRO (imu_gyro) =================
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.4), constrained_layout=True)
tau = np.logspace(-2, 4, 200)
a = ax[0, 0]
for gp, c in ((IMU.NAVGRADE, "tab:blue"), (IMU.TACTICAL, "tab:orange")):
    a.loglog(tau, gp.allan_dev_deg_s(tau) * 3600.0, color=c, label=gp.name[:28])
    a.axhline(gp.bias_instab_floor_deg_s() * 3600.0, color=c, ls=":", lw=1)
rngr = np.random.default_rng(0)
_dt = 1.0 / IMU.NAVGRADE.sample_rate_hz
sim = IMU.NAVGRADE.simulate_rate_error(4000.0, rngr, dt_s=_dt)
sim_y = np.asarray(sim[0] if isinstance(sim, tuple) else sim, float)
t_ad, ad = IMU.overlapping_adev(sim_y, _dt)
a.loglog(t_ad, np.asarray(ad) * 3600.0, "k.", ms=3, label="simulated ADEV (navgrade)")
a.set_xlabel("averaging time $\\tau$ [s]"); a.set_ylabel("Allan deviation [deg/h]")
a.set_title("(a) Allan model: ARW $\\tau^{-1/2}$ + bias-instability floor"); a.legend()
a = ax[0, 1]
flat = 0.003
nav = IMU.NAVGRADE.ekf_rate_sigma_rad_s() / IMU.DEG
ev = HW.EVENT_RATE_CAM.rms_dual_deg_s
_, fused = IMU.fuse_rate(np.zeros(3), nav * IMU.DEG, np.zeros(3), ev * IMU.DEG)
fusedv = float(np.mean(fused)) / IMU.DEG
vals = [flat * 1e3, nav * 1e3, ev * 1e3, fusedv * 1e3]
lbls = ["flat\nheritage", "nav-grade\nMEMS", "event-cam\n(dual)", "fused\n(pipeline)"]
bars = a.bar(lbls, vals, color=["0.7", "tab:blue", "tab:purple", "tab:green"])
a.set_ylabel("rate knowledge 1$\\sigma$ [mdeg/s]"); a.set_title("(b) AttEKF gyro-rate budget")
for b, v in zip(bars, vals): a.text(b.get_x() + b.get_width()/2, v * 1.02, f"{v:.2f}", ha="center", fontsize=8)
a = ax[1, 0]
tdrop = np.linspace(0.01, 120, 300)
for gp, c in ((IMU.NAVGRADE, "tab:blue"), (IMU.TACTICAL, "tab:orange")):
    drift = gp.allan_dev_deg_s(tdrop) * tdrop * 3600.0
    a.loglog(tdrop, drift, color=c, label=gp.name[:28])
a.axvspan(0.3, 3.0, color="r", alpha=0.12, label="near-CA ST dropout")
a.set_xlabel("star-tracker dropout duration [s]"); a.set_ylabel("attitude drift [arcsec]")
a.set_title("(c) dead-reckoning drift through the CA slew"); a.legend()
a = ax[1, 1]
meas_s = IMU.rate_rms(IMU.EVENTCAM, n_cases=40, dual=False, seed=1)
meas_d = IMU.rate_rms(IMU.EVENTCAM, n_cases=40, dual=True, seed=1)
vals = [HW.EVENT_RATE_CAM.rms_single_deg_s, float(meas_s["total"]), HW.EVENT_RATE_CAM.rms_dual_deg_s, float(meas_d["total"])]
lbls = ["paper single\n0.307", "module single", "paper dual\n0.0275", "module dual"]
bars = a.bar(lbls, vals, color=["0.7", "tab:blue", "0.7", "tab:green"]); a.set_yscale("log")
a.set_ylabel("rate RMS [deg/s]"); a.set_title("(d) event-camera rate aid: module vs paper")
for b, v in zip(bars, vals): a.text(b.get_x() + b.get_width()/2, v * 1.1, f"{v:.3f}", ha="center", fontsize=7.5)
fig.suptitle("Rate gyro / IMU -- nav-grade MEMS Allan model (Zhao 2021) + event-camera aid", fontsize=10)
save(fig, "H3_imu_gyro")

# ================= H4  RCS EXECUTION + ATTITUDE ACTUATION =================
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.4), constrained_layout=True)
mass = HW.XB6.wet_mass_kg
ep = THX.effective_params()
a = ax[0, 0]
cmd = np.linspace(0, 8e-3, 400)
for mib, lab, c in ((HW.MPS130.min_impulse_bit_Ns, "flat MIB 10 mN-s", "0.6"),
                    (ep["mib_Ns"], f"hi-fi MIB {ep['mib_Ns']*1e3:.1f} mN-s (+water-hammer)", "tab:red")):
    step = mib / mass
    a.plot(cmd * 1e3, (np.round(cmd / step) * step) * 1e3, color=c, label=lab)
a.plot(cmd * 1e3, cmd * 1e3, "k:", lw=0.8, label="ideal")
a.set_xlabel("commanded $\\Delta v$ [mm/s]"); a.set_ylabel("realized $\\Delta v$ [mm/s]")
a.set_title("(a) MIB quantization staircase"); a.legend()
a = ax[0, 1]
rngb = np.random.default_rng(2); n = 500
errs_t, errs_m = [], []
for _ in range(n):
    out = THR.apply_burn(np.array([0.05, 0.0, 0.0]), mass, rngb)
    dv = np.asarray(out[0], float)
    errs_t.append(np.linalg.norm(dv[1:]) / 0.05 * 100)
    errs_m.append((dv[0] - 0.05) / 0.05 * 100)
a.scatter(errs_m, errs_t, s=4, alpha=0.4)
a.axvline(-HW.MPS130.finite_burn_loss_frac * 100, color="r", ls="--", lw=1, label="finite-burn loss 2%")
a.set_xlabel("magnitude error [%]"); a.set_ylabel("transverse (misalign) [%]")
a.set_title(f"(b) burn execution MC (0.5$^\\circ$ 1$\\sigma$ misalign, n={n})"); a.legend()
a = ax[1, 0]
rpm, jit = RW.RWA.jitter_vs_speed()
a.plot(rpm, jit)
a.axhline(1.0, color="0.5", ls="--", label='XB6 spec 1" / 1 s')
op = RW.RWA.operating_jitter_arcsec(np.radians(HW.STAR_TRACKER.max_slew_deg_s))
a.plot([], [], " ", label=f'operating (mid-slew): {op:.2f}"')
a.set_xlabel("wheel speed [rpm]"); a.set_ylabel("boresight jitter [arcsec RMS]")
a.set_title("(c) RW microvibration (harmonics x structural mode)"); a.legend()
a = ax[1, 1]
labels = ["star trk\n(G3x2+FSM)", "RW jitter\n(mid-slew)", "RSS\n(pipeline)", "flat\nbaseline"]
vals = [4.50, op, float(np.hypot(4.50, op)), 5.0]
bars = a.bar(labels, vals, color=["tab:blue", "tab:orange", "tab:green", "0.7"])
a.set_ylabel("pointing knowledge 1$\\sigma$ [arcsec]"); a.set_title("(d) imaging pointing budget (RSS)")
for b, v in zip(bars, vals): a.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
fig.suptitle("RCS execution + attitude actuation -- MIB / misalignment / RW jitter", fontsize=10)
save(fig, "H4_rcs_attitude")
print("[done] 4 hardware figures ->", FIGD)
