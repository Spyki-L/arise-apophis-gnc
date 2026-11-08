#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faultlab_engine.py — drives the REAL FDI core under a randomized, high-fidelity
fault campaign, for the multi-window live FDIR lab.

One `fdi_core.FDICore` per deputy (all six channels ON: NAV, ATT, COMM, PROP, FORM,
PEER). Each frame we build the genuine `step()` inputs (nominal, sub-threshold
residuals) and, for any deputy with an active fault episode, inject a faithful
perturbation into the channel that fault would actually corrupt:

    NAV  -> innovation bias (relative-nav residual / NIS rises)
    ATT  -> attitude error + body-rate anomaly (tumble)
    COMM -> heartbeat age ramps past the timeout (-> DEAD on trip)
    PROP -> thruster Δv mismatch on a burn (TCM misfire)
    FORM -> formation/task miss (drifts off its reachable slot)
    PEER -> peer-consistency outlier (Byzantine / lying self-report)

The core's own leaky-CUSUM detectors + HEALTHY->SUSPECT->FAULTED->DEAD machine then
decide — nothing is faked. On each conviction the chief-led RecoveryLadder runs, a
healthy peer re-images the suspect, the survivors are re-optimized, and the onboard
decision is graded against the high-fidelity ground oracle (compute timed).

It also maintains the subsystem truth the other windows draw: RCS thruster rosettes,
scheduled TCM burns (+ misfires), the ISL link graph, data-to-chief relays, and the
peer-imaging hand-offs. Pure numpy + this package; fully testable headless.
"""
from __future__ import annotations
import time
import numpy as np
import numpy.linalg as la

from fdi_core import FDICore, FDIConfig, FDIState, FaultType
from recovery_ladder import RecoveryLadder, FaultClass, Action

CHANNELS = ("NAV", "ATT", "COMM", "PROP", "FORM", "PEER")
FAULT_COLOR = {"NAV": "#33bbff", "ATT": "#ffcc00", "COMM": "#ff5db1",
               "PROP": "#ff7a33", "FORM": "#9b6bff", "PEER": "#00e0c6"}
# FDI fault label -> recovery consensus class, and what a peer's camera would see
FT2FC = {FaultType.NAV: FaultClass.NAV, FaultType.COMM: FaultClass.COMM,
         FaultType.ATT: FaultClass.TUMBLE, FaultType.PROP: FaultClass.THRUSTER_STUCK,
         FaultType.FORM: FaultClass.MODE, FaultType.PEER: FaultClass.UNKNOWN,
         FaultType.NONE: FaultClass.NONE}
VISION = {"NAV": FaultClass.NAV, "ATT": FaultClass.TUMBLE, "COMM": FaultClass.COMM,
          "PROP": FaultClass.THRUSTER_STUCK, "FORM": FaultClass.MODE, "PEER": FaultClass.STUCK_PANEL}


def make_cfg():
    """All six channels on; default P1 thresholds/persistence from fdi_core."""
    return FDIConfig(use_form_channel=True, use_peer_channel=True)


# ───────────────────────── fault campaign ───────────────────────────────────

class FaultEpisode:
    __slots__ = ("kind", "dep", "f0", "dur", "sev")

    def __init__(self, kind, dep, f0, dur, sev):
        self.kind = kind; self.dep = dep; self.f0 = f0; self.dur = dur; self.sev = sev

    def active(self, fi):
        return self.f0 <= fi < self.f0 + self.dur

    @property
    def f1(self):
        return self.f0 + self.dur


def build_campaign(N, F, seed=7):
    """Randomized hi-fi campaign. Guarantees one episode of EACH channel (so every
    detector is exercised) spread across the mission on distinct deputies, then adds
    random extra episodes (incl. overlap) for stress. Reproducible via `seed`."""
    rng = np.random.default_rng(seed)
    eps = []
    order = list(CHANNELS); rng.shuffle(order)
    slot0 = int(0.12 * F); span = int(0.66 * F); step = max(1, span // len(order))
    deps = list(range(N)); rng.shuffle(deps)
    for i, k in enumerate(order):
        f0 = slot0 + i * step + int(rng.integers(-step // 5 - 1, step // 5 + 1))
        dur = int(rng.integers(int(0.11 * F), int(0.17 * F)))
        sev = float(rng.uniform(6.5, 10.0))
        eps.append(FaultEpisode(k, deps[i % N], max(4, f0), max(20, dur), sev))
    for _ in range(max(2, N // 2)):                       # stress / overlap
        k = order[int(rng.integers(len(order)))]
        f0 = int(rng.integers(int(0.16 * F), int(0.82 * F)))
        dur = int(rng.integers(int(0.07 * F), int(0.13 * F)))
        eps.append(FaultEpisode(k, int(rng.integers(N)), f0, max(18, dur), float(rng.uniform(5.5, 9.0))))
    return eps


# ───────────────────────── the lab engine ───────────────────────────────────

class FaultLabEngine:
    def __init__(self, truth, onb, gnd, campaign, seed=7, reach_m=600.):
        self.tr = truth; self.N = truth["N"]; self.onb = onb; self.gnd = gnd
        self.eps = campaign
        self.fdi = [FDICore(make_cfg()) for _ in range(self.N)]
        for f in self.fdi:
            f.reset(0.0)
        self.rng = np.random.default_rng(seed + 1)
        self.recovery = RecoveryLadder(d_safe=50.)
        self.active = set(range(self.N))
        self.health = {d: "HEALTHY" for d in range(self.N)}
        self.fault = {d: "NONE" for d in range(self.N)}
        self.offset = {d: np.zeros(3) for d in range(self.N)}
        self.reach = np.full(self.N, reach_m)
        self.log = []; self.compute = []
        self.cov_onb = self.cov_gnd = None
        self._cap_hold = [0, None]
        self.decided = set()
        self._suppress = {}                                  # deputy -> frame until which a recovered fault is suppressed
        self._safehold = {}                                  # deputy -> frame at which a safe-hold is resolved/restored
        self.hb_nominal = 60.0
        F = truth["T"].size
        self.tcm_frames = {int(0.22 * F), int(0.50 * F), int(0.78 * F)}
        self.cfg0 = self.fdi[0].cfg
        # ── per-deputy onboard estimators — the FDI consumes ONLY their estimates ──
        from faultlab_ekf import NavEKF, AttEKF, chief_mean_motion
        Tg = truth["T"]; self.dt_frame = float(Tg[1] - Tg[0])
        vC = np.gradient(truth["rC"], Tg, axis=0)
        self.n_chief = chief_mean_motion(truth["rC"][Tg.size // 2], vC[Tg.size // 2])
        Hc = truth["hill"].copy(); Hc[la.norm(Hc, axis=2) > 50000.] = 0.0   # clamp pre-deploy interp artifact
        self.hill_clean = Hc
        an = np.zeros_like(Hc)                         # nominal relative acceleration (feed-forward to NavEKF)
        an[:, 1:-1] = (Hc[:, 2:] - 2 * Hc[:, 1:-1] + Hc[:, :-2]) / self.dt_frame ** 2
        an[np.abs(an) > 1e-2] = 0.0                    # ignore deploy-clamp discontinuities
        self.a_nom = an
        self.nav_ekf = []; self.att_ekf = []
        for d in range(self.N):
            kf = NavEKF(self.n_chief); kf.init(Hc[d, 0]); self.nav_ekf.append(kf)
            self.att_ekf.append(AttEKF())
        self.th_true = [np.zeros(3) for _ in range(self.N)]   # TRUE attitude error / rate (truth-only)
        self.om_true = [np.zeros(3) for _ in range(self.N)]
        self.nav_off = [np.zeros(3) for _ in range(self.N)]   # TRUE nav deviation (unmodeled accel)
        self.nav_voff = [np.zeros(3) for _ in range(self.N)]
        self.I_dep = 0.18; self.r_meas = 10.0                 # 6U inertia [kg·m²]; ISL ranging noise [m]
        # mission phases (timeline fraction) -> (name, EKF samples-per-frame "rate")
        self.phase_bounds = [(0.00, 0.10, "SEPARATION", 4), (0.10, 0.68, "COAST", 1),
                             (0.68, 0.90, "APPROACH", 2), (0.90, 1.01, "ENCOUNTER", 4)]

    # ---- coverage stations / re-optimizer (same protected objective as the cockpit) ----
    def ca_stations(self):
        return {d: self.tr["ca_stations"][d] + self.offset[d] for d in self.active}

    def _reopt(self, fi):
        t0 = time.perf_counter()
        base = {d: self.tr["ca_stations"][d] for d in self.active}
        rng = np.random.default_rng(2000 + fi)
        off = {d: self.offset[d].copy() for d in self.active}
        def cov(o):
            return self.onb.score([base[d] + o[d] for d in self.active])[0]
        best = cov(off)
        for _ in range(110):
            if not self.active:
                break
            d = list(self.active)[rng.integers(len(self.active))]
            cand = {k: v.copy() for k, v in off.items()}
            cand[d] = np.clip(off[d] + rng.normal(0, self.reach[d] * 0.3, 3), -self.reach[d], self.reach[d])
            c = cov(cand)
            if c > best:
                off = cand; best = c
        for d in self.active:
            self.offset[d] = off[d]
        return best, time.perf_counter() - t0

    # ---- nominal step() inputs (sub-threshold; HEALTHY stays HEALTHY) ----
    def _nominal_inputs(self, d, is_tcm):
        rng = self.rng; cfg = self.cfg0
        innov = rng.normal(0, 0.55, 3); S = np.eye(3)
        sigma_BR = rng.normal(0, 0.30 * cfg.sigma_sigma, 3)
        omega = rng.normal(0, 0.20 * cfg.sigma_omega, 3)
        if is_tcm:
            burn = True; dv = rng.normal(0, 0.40, 3); Pvv = np.eye(3)
        else:
            burn = False; dv = None; Pvv = None
        return dict(innov=innov, S_innov=S, dv_est_H=dv, P_vv=Pvv, burn_flag=burn,
                    sigma_BR=sigma_BR, omega_BR_B=omega, hb_age=0.5 * self.hb_nominal,
                    hb_nominal=self.hb_nominal, r_form=0.2, r_peer=0.2)

    def phase(self, fi):
        """Mission phase + EKF samples-per-frame at frame fi (separation/encounter run hot)."""
        f = fi / max(self.tr["T"].size - 1, 1)
        for a, b, name, rate in self.phase_bounds:
            if a <= f < b:
                return name, rate
        return "ENCOUNTER", 4

    def _ekf_step(self, d, fi, de, suppressed):
        """Advance TRUTH at the phase rate (a NAV fault = unmodeled accel, an ATT fault =
        unmodeled torque), simulate the measurements, and run the onboard EKFs. Returns the
        estimate-side quantities the FDI consumes plus the per-substep innovation samples and
        the truth-vs-estimate validation error. The FDI never sees truth."""
        name, n_sub = self.phase(fi)
        dt = self.dt_frame / n_sub
        r0 = self.hill_clean[d, max(fi - 1, 0)]; r1 = self.hill_clean[d, fi]
        navf = (not suppressed) and any(e.kind == "NAV" for e in de)
        attf = (not suppressed) and any(e.kind == "ATT" for e in de)
        sev = max([e.sev for e in de], default=6.0)
        rng = self.rng; kf = self.nav_ekf[d]; ak = self.att_ekf[d]
        samples = []
        for sub in range(n_sub):
            frac = (sub + 1) / n_sub
            r_true = r0 * (1 - frac) + r1 * frac
            if navf:
                a = np.array([1.0, 0.6, 0.4]) * (sev * 1.6e-4)        # unmodeled accel (stuck-thruster-like)
                self.nav_voff[d] = self.nav_voff[d] + a * dt
                self.nav_off[d] = self.nav_off[d] + self.nav_voff[d] * dt
            else:                                                     # GNC restores the station once the fault clears
                self.nav_voff[d] = self.nav_voff[d] * 0.85
                self.nav_off[d] = self.nav_off[d] * 0.90
            z_nav = r_true + self.nav_off[d] + rng.normal(0, self.r_meas, 3)
            y_nav, S_nav = kf.step(dt, z_nav, u_accel=self.a_nom[d, fi])   # filter knows the planned dynamics
            if attf:                                          # fault torque drives a tumble (integrated)
                tau = np.array([1.0, 0.6, 0.4]) * np.deg2rad(0.02) * (sev / 8.0)
                self.om_true[d] = np.clip(self.om_true[d] + (tau / self.I_dep) * dt, -np.deg2rad(2.0), np.deg2rad(2.0))
                self.th_true[d] = self.th_true[d] + self.om_true[d] * dt
            else:                                             # ADCS holding attitude: small bounded error, no drift
                self.om_true[d] = rng.normal(0, 0.15 * ak.gy_n, 3)
                self.th_true[d] = rng.normal(0, 0.12 * self.cfg0.sigma_sigma, 3)
            z_th = self.th_true[d] + rng.normal(0, ak.st_n, 3)
            z_om = self.om_true[d] + rng.normal(0, ak.gy_n, 3)
            ak.step(dt, z_th, z_om)
            samples.append((y_nav / np.sqrt(np.diag(S_nav)),
                            ak.y_st / np.sqrt(np.diag(ak.S_st)), ak.y_gy / np.sqrt(np.diag(ak.S_gy))))
        ang = np.degrees(la.norm(self.th_true[d])) % 360.0
        te = dict(nav_err=float(la.norm(kf.pos - r1)), att_err=float(la.norm(ak.theta - self.th_true[d])),
                  nav_dev=float(la.norm(self.nav_off[d])), att_true_deg=float(ang if ang <= 180 else 360 - ang))
        return dict(y_nav=y_nav, S_nav=S_nav, th=ak.theta.copy(), om=ak.omega.copy(),
                    nav_sig=np.sqrt(np.diag(S_nav)), y_st=ak.y_st.copy(), ss=np.sqrt(np.diag(ak.S_st)),
                    y_gy=ak.y_gy.copy(), sg=np.sqrt(np.diag(ak.S_gy)), samples=samples, te=te,
                    phase=name, rate=n_sub)

    def _inject(self, inp, e, fi):
        """Inject the (estimate-side) residual fault for the non-EKF channels (NAV/ATT are
        driven through the truth→EKF path in _ekf_step)."""
        frac = (fi - e.f0) / max(1, e.dur)
        s = e.sev * min(1.0, 0.35 + 1.25 * frac)
        if e.kind == "COMM":
            inp["hb_age"] = self.hb_nominal * (1.6 + 6.0 * frac)
        elif e.kind == "PROP":
            inp["burn_flag"] = True
            inp["dv_est_H"] = np.array([s * 0.55, s * 0.32, s * 0.22]); inp["P_vv"] = np.eye(3)
        elif e.kind == "FORM":
            inp["r_form"] = max(inp["r_form"], 3.2 + s * 1.6)
            inp["r_peer"] = max(inp["r_peer"], 1.8 + s * 0.7)   # a local form fault also reads peer-inconsistent
        elif e.kind == "PEER":
            inp["r_peer"] = max(inp["r_peer"], 2.2 + s * 1.05)

    # ---- ISL link graph (chief = node N) ----
    def _isl(self, fi):
        H = self.tr["hill"][:, fi, :]                       # deputy positions in chief Hill frame [m]
        links = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                rng = la.norm(H[i] - H[j]); q = float(np.clip(1.0 - rng / 16000., 0.05, 1.0))
                if self.health[i] == "DEAD" or self.health[j] == "DEAD":
                    q *= 0.0
                elif "COMM" in (self.fault[i], self.fault[j]):
                    q *= 0.25
                links.append((i, j, q))
        chief = []                                          # deputy -> chief quality (range to origin)
        for i in range(self.N):
            rng = la.norm(H[i]); q = float(np.clip(1.0 - rng / 16000., 0.05, 1.0))
            if self.health[i] == "DEAD":
                q = 0.0
            chief.append(q)
        return dict(H=H, links=links, chief=chief)

    def _relay_to_chief(self, d, isl):
        """Pick the healthiest peer to relay a misbehaving deputy's data to the chief."""
        H = isl["H"]; best = None; bestq = -1
        for p in range(self.N):
            if p == d or self.health[p] in ("DEAD", "FAULTED"):
                continue
            q = (1.0 - la.norm(H[d] - H[p]) / 16000.) * isl["chief"][p]
            if q > bestq:
                bestq = q; best = p
        return best

    def _nearest_healthy(self, d, isl):
        H = isl["H"]; best = None; bestr = 1e18
        for p in range(self.N):
            if p == d or self.health[p] in ("DEAD", "FAULTED"):
                continue
            r = la.norm(H[d] - H[p])
            if r < bestr:
                bestr = r; best = p
        return best

    def _rcs(self, d, inp, st):
        """8-thruster rosette levels [0,1]: attitude-hold baseline + detumble/burn spikes."""
        base = 0.06 + 0.04 * np.abs(np.sin(np.linspace(0, 6.28, 8) + d))
        eatt = la.norm(inp["omega_BR_B"]) / max(self.cfg0.sigma_omega, 1e-9)
        if st.state in (FDIState.SUSPECT, FDIState.FAULTED) and st.fault_type is FaultType.ATT:
            base = base + 0.6 * np.abs(self.rng.normal(0, 1, 8))            # erratic detumble
        elif inp["burn_flag"]:
            base = base + np.array([0.7, 0.7, 0, 0, 0.7, 0.7, 0, 0])       # translational TCM
        base = base + min(0.3, 0.02 * eatt)
        return np.clip(base, 0, 1)

    # ───────────────────────── main step ────────────────────────────────────
    def step(self, fi):
        t = float(self.tr["T"][fi]); t_go = self.tr["t_ca"] - t
        for d in list(self._safehold):                       # a safe-hold is temporary: restore once resolved
            if self.health[d] == "FAULTED" and fi >= self._safehold[d]:
                self.fdi[d].reset(t); self.health[d] = "HEALTHY"; self.fault[d] = "NONE"
                self.active.add(d); self.decided.discard(d); self._suppress[d] = fi + 20
                self.nav_off[d] = np.zeros(3); self.nav_voff[d] = np.zeros(3)
                self.th_true[d] = np.zeros(3); self.om_true[d] = np.zeros(3)
                self.nav_ekf[d].init(self.hill_clean[d, fi]); self.att_ekf[d] = self.att_ekf[d].__class__()
                self.log.append((fi, f"D{d+1} safe-hold cleared -> RESTORED"))
                self._safehold.pop(d, None)
        is_tcm = fi in self.tcm_frames
        act = [e for e in self.eps if e.active(fi)]
        by_dep = {}
        for e in act:
            by_dep.setdefault(e.dep, []).append(e)

        fdi = {}; rcs = {}; tcm = {}; innovd = {}; truth_est = {}
        gt_fault = {}                                       # ground-truth injected label (verification)
        newly = []                                          # deputies that just convicted
        phase_name, cadence = self.phase(fi)
        for d in range(self.N):
            inp = self._nominal_inputs(d, is_tcm)
            de = by_dep.get(d, [])
            suppressed = fi < self._suppress.get(d, -1)
            if de and not suppressed:                        # suppressed while a just-recovered fault settles
                gt_fault[d] = de[0].kind
                for e in de:
                    self._inject(inp, e, fi)                 # COMM/PROP/FORM/PEER (estimate-side residuals)
            misfire = is_tcm and any(e.kind == "PROP" for e in de)
            if self.health[d] != "DEAD":
                est = self._ekf_step(d, fi, de, suppressed)              # truth -> measurements -> EKF
                inp["innov"] = est["y_nav"]; inp["S_innov"] = est["S_nav"]   # FDI gets ONLY the estimate
                inp["sigma_BR"] = est["th"]; inp["omega_BR_B"] = est["om"]
                core = self.fdi[d]; core.step(t=t, **inp); s = core.state
                self.health[d] = s.state.name
                self.fault[d] = s.fault_type.name
                if s.state in (FDIState.FAULTED, FDIState.DEAD) and d not in self.decided:
                    newly.append(d)
                if s.state is FDIState.DEAD:
                    self.active.discard(d)
                fdi[d] = dict(S={"NAV": core.S_nav, "ATT": core.S_att, "COMM": core.S_comm,
                                 "PROP": core.S_prop, "FORM": core.S_form, "PEER": core.S_peer},
                              r={"NAV": core.r_nav, "ATT": core.r_att, "COMM": core.r_comm,
                                 "PROP": core.r_prop, "FORM": core.r_form, "PEER": core.r_peer},
                              state=s.state.name, fault=s.fault_type.name, conf=float(s.confidence))
                innovd[d] = dict(samples=est["samples"], nav=est["y_nav"], nav_sig=est["nav_sig"],
                                 att_s=est["y_st"], att_s_sig=est["ss"], att_w=est["y_gy"], att_w_sig=est["sg"])
                truth_est[d] = est["te"]
            else:
                core = self.fdi[d]
                fdi[d] = dict(S={"NAV": core.S_nav, "ATT": core.S_att, "COMM": core.S_comm,
                                 "PROP": core.S_prop, "FORM": core.S_form, "PEER": core.S_peer},
                              r={k: 0.0 for k in CHANNELS}, state="DEAD",
                              fault=self.fault[d], conf=1.0)
                z3 = np.zeros(3); o3 = np.ones(3)
                innovd[d] = dict(samples=[], nav=z3, nav_sig=o3, att_s=z3, att_s_sig=o3, att_w=z3, att_w_sig=o3)
                truth_est[d] = dict(nav_err=0., att_err=0., nav_dev=0., att_true_deg=0.)
            rcs[d] = self._rcs(d, inp, self.fdi[d].state)
            tcm[d] = dict(firing=bool(is_tcm and self.health[d] != "DEAD"), misfire=bool(misfire))

        isl = self._isl(fi)

        # ── decisions on newly convicted deputies (real recovery + peer reimage + reopt) ──
        event = None; capture = None; onb_ms = gnd_ms = None
        data_to_chief = []; imaging = []
        for d in newly:
            self.decided.add(d)
            ft = self.fdi[d].state.fault_type
            consensus = FT2FC.get(ft, FaultClass.UNKNOWN)
            peer = self._nearest_healthy(d, isl)
            relay = self._relay_to_chief(d, isl)
            kind = gt_fault.get(d, self.fault[d])
            vision = VISION.get(kind, FaultClass.UNKNOWN)
            fp = [self.ca_stations().get(k, self.tr["ca_stations"][k]) for k in range(self.N)]
            isl_logs = {d: (FaultClass.NONE if kind == "PEER" else consensus)}   # PEER => lying self-report
            res = self.recovery.run(suspect=d, consensus_label=consensus,
                                    vision_fn=lambda s, v=vision: (v, 0.92), isl_logs=isl_logs,
                                    fleet_pos=fp, fleet_vel=[np.zeros(3)] * self.N,
                                    suspect_state=(fp[d], np.zeros(3)))
            act_name = res["action"].name
            if self.fdi[d].state.state is FDIState.DEAD:      # permanent loss (e.g. comm down) -> stays DEAD
                self.active.discard(d); self.health[d] = "DEAD"; act_name = "DEAD - relay data to chief"
            elif res.get("recovered"):                       # soft fault corrected -> deputy restored, FDI re-armed
                self.fdi[d].reset(t); self.health[d] = "HEALTHY"; self.fault[d] = "NONE"
                self.active.add(d); self.decided.discard(d); self._suppress[d] = fi + 25
                self.nav_off[d] = np.zeros(3); self.nav_voff[d] = np.zeros(3)   # re-init the estimator + clear truth error
                self.th_true[d] = np.zeros(3); self.om_true[d] = np.zeros(3)
                self.nav_ekf[d].init(self.hill_clean[d, fi]); self.att_ekf[d] = self.att_ekf[d].__class__()
                act_name += " -> RECOVERED"
            elif res["action"] is Action.DISPOSE and res.get("collision_free"):
                self.active.discard(d); self.health[d] = "DEAD"
            else:                                            # SAFE_HOLD — temporary; restored after a cooldown
                self._safehold[d] = fi + 35
            _, dt_o = self._reopt(fi); onb_ms = dt_o * 1e3
            cg, dt_g, _ = self.gnd.score(list(self.ca_stations().values())); gnd_ms = dt_g * 1e3
            self.cov_gnd = cg
            if peer is not None:
                capture = dict(deputy=peer, target_dep=d); imaging.append((peer, d))
            if relay is not None:
                data_to_chief.append((d, relay))
            event = kind
            self.log.append((fi, f"D{d+1} {kind} fault -> {ft.name} isolated "
                                 f"-> {act_name}; D{(peer or 0)+1} re-images; reopt {onb_ms:.0f}ms / gnd {gnd_ms:.0f}ms"))

        # standing data-to-chief for any still-misbehaving deputy (FAULTED/DEAD)
        for d in range(self.N):
            if self.health[d] in ("FAULTED", "DEAD") and not any(x[0] == d for x in data_to_chief):
                r = self._relay_to_chief(d, isl)
                if r is not None:
                    data_to_chief.append((d, r))

        # camera-hold so the peer "click" reads in the live anim
        if capture is not None:
            self._cap_hold = [12, capture]
        elif self._cap_hold[0] > 0:
            self._cap_hold[0] -= 1; capture = self._cap_hold[1]
        if capture is not None and (capture["deputy"], capture["target_dep"]) not in imaging:
            imaging.append((capture["deputy"], capture["target_dep"]))   # persist the hand-off through the hold

        cov_o, dt_o = self.onb.score(list(self.ca_stations().values()))
        self.cov_onb = cov_o
        if event is not None:
            self.compute.append((fi, dt_o * 1e3, gnd_ms or 0.0, onb_ms or 0.0, cov_o, self.cov_gnd or cov_o))

        # focus deputy = one that is actively faulting (live signals) preferred over dead/healthy
        actf = [(self.fdi[d].state.confidence, d) for d in range(self.N) if self.health[d] in ("SUSPECT", "FAULTED")]
        if actf:
            focus = max(actf)[1]
        else:
            nonh = [(self.fdi[d].state.confidence, d) for d in range(self.N) if self.health[d] != "HEALTHY"]
            focus = max(nonh)[1] if nonh else int(np.argmax([self.fdi[d].state.confidence for d in range(self.N)]))

        return dict(
            frame=fi, t_go=t_go, active=sorted(self.active), health=dict(self.health),
            fault=dict(self.fault), cov_onb=cov_o, cov_gnd=self.cov_gnd, event=event,
            capture=capture, onb_ms=onb_ms, gnd_ms=gnd_ms,
            ca_all={d: self.tr["ca_stations"][d] + self.offset[d] for d in range(self.N)},
            log_tail=self.log[-1][1] if self.log else "",
            fdi=fdi, thresholds={"NAV": self.cfg0.h_nav, "ATT": self.cfg0.h_att, "COMM": self.cfg0.h_comm,
                                 "PROP": self.cfg0.h_prop, "FORM": self.cfg0.h_form, "PEER": self.cfg0.h_peer},
            gt_fault=gt_fault, rcs=rcs, tcm=tcm, isl=isl, imaging=imaging, innov=innovd,
            truth_est=truth_est, phase=phase_name, cadence=cadence, phase_bounds=self.phase_bounds,
            tcm_marks=sorted(self.tcm_frames), data_to_chief=data_to_chief, is_tcm=is_tcm, focus=focus)


def default_build(truth, seed=7):
    import cockpit_engine as ce
    onb = ce.OnboardCoverage(truth); gnd = ce.GroundOracle(truth)
    camp = build_campaign(truth["N"], truth["T"].size, seed=seed)
    return FaultLabEngine(truth, onb, gnd, camp, seed=seed)
