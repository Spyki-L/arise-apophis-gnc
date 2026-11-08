#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fault_library.py  (Design D / nominal_v2 / fdi — CORTEX build step 3)
=====================================================================

The **parameterized fault taxonomy** mapped to the REAL ARISE models (fidelity plan
§C.3), shared by every consumer so observation/action/reward stay identical across
fidelities:

* ``rl/fdir_env.SurrogateBackend``  — residual-domain: ``signature()`` replaces the
  generic "+ramp" with class-specific, physically-parameterized severity shapes.
* ``rl/basilisk_backend.BasiliskBackend`` — data/model-domain: ``inject()`` perturbs
  the actual streams (EKF innovations, pointing/rate, burn dV, heartbeat, station,
  peer reports) the way the fault would corrupt the real pipeline.
* the FDI test bench — faulted-bundle generation for detection/isolation campaigns.

Mapping to the real models (plan §C.3)
--------------------------------------
NAV   -> ``modules/nav_stage.py`` sensors : range/bearing bias, noise blow-up (sigma
         jump on ISL dropout), EKF divergence (innovation drift).
PROP  -> ``modules/thruster_hifi.py``     : burn scale error, pointing misalignment,
         stuck-off (missed burn), MIB-floor violation on small corrections.
ATT   -> ``guidance_v2`` RW loop          : unmodeled torque (tumble), wheel failure
         (saturation lag), all expressed as pointing/rate EXCESS beyond the healthy
         slew-lag envelope (see run_fdi_nominal.load_att_mc).
COMM  -> ``channels.py`` heartbeat        : dropout (hb_age growth), latency spikes,
         intermittent link (duty-cycled).
FORM  -> guidance station command         : corrupted commanded slot -> terminal miss.
PEER  -> ISL consensus (``channels.PeerConsensusBuffer``) : byzantine self-report
         (scaled / inverted / frozen), with the affected-neighbour set.
ENV   -> ``apophis_tumbler`` / ephemeris  : spin-state or ephemeris surprise; enters
         the OBSERVABLE channels as a correlated NAV+FORM signature (no new channel).
COMPOUND -> any ordered pair (A causes B after a delay), for the E2 abductive tests.

Each sampled fault = CLASS x MODE x magnitude x onset x duration x affected units.
Severity shapes are expressed in the SAME units the FDI residuals use (multiples of
the per-channel healthy scale), so ``signature()`` output adds directly onto healthy
residual draws; secondary channels from ``causal_model.EFFECTS`` receive an
attenuated copy (x0.5), keeping the library consistent with the E2 causal graph.

Pure numpy + stdlib.  Self-test: ``python3 fault_library.py`` -> ALL PASS.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

try:                                    # canonical taxonomy (fdi/ = same dir)
    import causal_model as cm
    CHANNELS, EFFECTS = tuple(cm.CHANNELS), {k: set(v) for k, v in cm.EFFECTS.items()}
except Exception:                       # standalone fallback (identical constants)
    CHANNELS = ("NAV", "ATT", "COMM", "PROP", "FORM", "PEER")
    EFFECTS = {"NAV": {"NAV", "FORM"}, "ATT": {"ATT"}, "COMM": {"COMM"},
               "PROP": {"PROP", "NAV", "FORM"}, "FORM": {"FORM", "PEER"}, "PEER": {"PEER"}}

FAULT_CLASSES = ("NAV", "ATT", "COMM", "PROP", "FORM", "PEER", "ENV")
SECONDARY_ATTEN = 0.5

# ── the taxonomy: class -> modes -> (physical parameter ranges, real-model hook) ──
#    mag ranges are in multiples of the channel's healthy residual scale (residual
#    domain) plus the physical parameter recorded for the data-domain injectors.
MODES = {
    "NAV": {
        "bias":       dict(mag=(0.8, 4.0),  phys=("relnav_bias_m", 0.01, 0.5)),   # vs 5 mm hifi relnav
        "noise":      dict(mag=(0.6, 3.0),  phys=("sigma_blowup_x", 2.0, 10.0)),  # ISL-dropout sigma jump
        "divergence": dict(mag=(0.5, 2.5),  phys=("drift_mps", 1e-4, 5e-3), ramp=True),
    },
    "ATT": {
        "tumble":     dict(mag=(1.0, 6.0),  phys=("torque_rate_degps", 0.05, 1.0), ramp=True),
        "wheel_fail": dict(mag=(0.8, 4.0),  phys=("lost_wheels", 1, 2)),
    },
    "COMM": {
        "dropout":    dict(mag=(1.0, 5.0),  phys=("outage_s", 300.0, 3600.0), ramp=True),
        "latency":    dict(mag=(0.5, 2.0),  phys=("latency_s", 60.0, 600.0)),
    },
    "PROP": {
        "scale":      dict(mag=(1.0, 5.0),  phys=("dv_scale_err", 0.05, 0.5)),
        "misalign":   dict(mag=(0.8, 4.0),  phys=("misalign_deg", 0.5, 10.0)),
        "stuck_off":  dict(mag=(1.5, 6.0),  phys=("missed_burns", 1, 3)),
    },
    "FORM": {
        "station":    dict(mag=(1.0, 5.0),  phys=("station_offset_m", 100.0, 2000.0), ramp=True),
    },
    "PEER": {
        "byzantine":  dict(mag=(1.0, 5.0),  phys=("report_scale", 3.0, 20.0)),
        "frozen":     dict(mag=(0.6, 2.5),  phys=("freeze_s", 600.0, 7200.0)),
    },
    "ENV": {
        "spin_change": dict(mag=(0.5, 2.5), phys=("spin_dfrac", 0.05, 0.3), ramp=True),
        "ephem_error": dict(mag=(0.5, 2.0), phys=("ephem_err_km", 0.5, 5.0), ramp=True),
    },
}
# ENV has no channel of its own: it shows up as correlated NAV+FORM evidence.
ENV_CHANNELS = {"NAV", "FORM"}


@dataclass
class FaultSpec:
    """One sampled fault: everything both backends need to inject it."""
    cls: str                  # FAULT_CLASSES member
    mode: str                 # mode within the class
    mag: float                # residual-domain severity plateau [healthy-scale units]
    phys_name: str            # physical parameter name (real-model hook)
    phys_value: float         # physical parameter value
    onset: float              # [steps or s, caller's timebase]
    duration: float           # np.inf = persistent
    duty: float = 1.0         # <1 -> intermittent (square-wave duty cycle)
    period: float = 20.0      # intermittent period [same timebase]
    ramp: bool = False        # ramp to plateau over RAMP_TIME instead of step
    ramp_time: float = 30.0
    affected: tuple = ()      # e.g. byzantine-corrupted neighbour ids
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.cls}.{self.mode}"

    # ── the primary channel this class alarms on (ENV -> NAV as leading evidence) ──
    @property
    def channel(self) -> str:
        return "NAV" if self.cls == "ENV" else self.cls

    def channels(self) -> set:
        return set(ENV_CHANNELS) if self.cls == "ENV" else set(EFFECTS[self.cls])

    # ── residual-domain signature (surrogate backend) ──────────────────────────
    def envelope(self, t) -> np.ndarray:
        """Severity multiplier in [0,1] at time-since-episode-start ``t``."""
        t = np.asarray(t, float)
        on = (t >= self.onset) & (t < self.onset + self.duration)
        env = on.astype(float)
        if self.ramp and self.ramp_time > 0:
            env = env * np.clip((t - self.onset) / self.ramp_time, 0.0, 1.0)
        if self.duty < 1.0 and self.period > 0:
            ph = ((t - self.onset) % self.period) / self.period
            env = env * (ph < self.duty)
        return env

    def signature(self, t, ch: str) -> np.ndarray:
        """Additive severity on channel ``ch`` at time(s) ``t`` (healthy-scale units)."""
        t = np.asarray(t, float)
        chans = self.channels()
        if ch not in chans:
            return np.zeros_like(t)
        atten = 1.0 if (ch == self.channel or (self.cls == "ENV" and ch in ENV_CHANNELS)) else SECONDARY_ATTEN
        sig = self.mag * atten * self.envelope(t)
        if self.cls == "PROP" and self.mode != "stuck_off":
            # burn-synchronous: PROP evidence only exists at burns (sparse channel);
            # callers gate it with their burn schedule via `burn_mask`.
            pass
        return sig

    # ── data-domain injection (Basilisk / replay backend) ──────────────────────
    def inject(self, streams: dict, t: np.ndarray, rng=None) -> dict:
        """Perturb REAL data arrays in-place-safe copy.  Recognised stream keys
        (all optional — only present ones are touched):
          y_pre (N,3) + sigma_meas (N,3)   NAV innovations / meas sigmas
          d_theta (N,) d_omega (N,)        ATT envelope-deviation streams [rad, rad/s]
          hb_age (N,)                      COMM heartbeat age [s]
          dv_real (B,3) burn_t (B,)        PROP realized burn dV
          r_form (N,)                      FORM causal residual
          peer_report (N,)                 own PEER self-report value
        Returns a NEW dict (originals untouched)."""
        rng = rng or np.random.default_rng(0)
        s = {k: np.array(v, float, copy=True) for k, v in streams.items()}
        env = self.envelope(t)
        on = env > 0
        if self.cls in ("NAV", "ENV") and "y_pre" in s:
            sg = s.get("sigma_meas", np.ones_like(s["y_pre"]))
            if self.mode == "bias":
                s["y_pre"][on] += self.phys_value * self._unit3(rng) * np.ones((on.sum(), 1))
            elif self.mode == "noise":
                s["y_pre"][on] *= self.phys_value ** 0.5
            else:                                   # divergence / env drift
                tt = (t[on] - self.onset)
                s["y_pre"][on] += (self.phys_value * tt)[:, None] * self._unit3(rng) * env[on][:, None]
        if self.cls == "ATT" and "d_theta" in s:
            tt = np.clip(t - self.onset, 0.0, None)
            if self.mode == "tumble":
                s["d_theta"] += env * np.minimum(np.pi, np.deg2rad(self.phys_value) * tt)
                if "d_omega" in s:
                    s["d_omega"] += env * np.deg2rad(self.phys_value)
            else:                                   # wheel_fail -> lag floor appears
                s["d_theta"] += env * np.deg2rad(2.0) * self.mag / 4.0
        if self.cls == "COMM" and "hb_age" in s:
            if self.mode == "dropout":
                tt = np.clip(t - self.onset, 0.0, None)
                s["hb_age"] = np.where(on, s["hb_age"] + tt * env, s["hb_age"])
            else:
                s["hb_age"] = np.where(on, s["hb_age"] + self.phys_value, s["hb_age"])
        if self.cls == "PROP" and "dv_real" in s:
            bt = s.get("burn_t", np.zeros(len(s["dv_real"])))
            benv = self.envelope(bt)
            if self.mode == "scale":
                s["dv_real"] *= (1.0 + self.phys_value * benv)[:, None]
            elif self.mode == "misalign":
                a = np.deg2rad(self.phys_value)
                R = self._small_rot(a, rng)
                idx = np.where(benv > 0)[0]
                for k in idx:
                    s["dv_real"][k] = R @ s["dv_real"][k]
            else:                                   # stuck_off: zero the first missed burns
                idx = np.where(benv > 0)[0][: int(self.phys_value)]
                s["dv_real"][idx] = 0.0
        if self.cls in ("FORM", "ENV") and "r_form" in s:
            scale = self.phys_value / 250.0 if self.cls == "FORM" else self.mag
            s["r_form"] += env * scale
        if self.cls == "PEER" and "peer_report" in s:
            if self.mode == "byzantine":
                s["peer_report"] = np.where(on, s["peer_report"] * self.phys_value, s["peer_report"])
            else:                                   # frozen
                if on.any():
                    k0 = int(np.argmax(on))
                    frozen = s["peer_report"][max(k0 - 1, 0)]
                    s["peer_report"][on] = frozen
        return s

    @staticmethod
    def _unit3(rng):
        v = rng.normal(size=3); return v / np.linalg.norm(v)

    @staticmethod
    def _small_rot(angle, rng):
        ax = FaultSpec._unit3(rng)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


@dataclass
class FaultLibrary:
    """Episode-level sampler.  Both backends call ``sample_episode``; determinism
    comes from the caller's rng.

    Curriculum / adversarial hooks (train_hpc.py sets these between stages; defaults
    reproduce the uniform library exactly):
      class_weights  {cls: w} sampling weights over ``classes`` (None = uniform)
      mag_scale      multiplies every sampled magnitude (severe > 1 > incipient)
      p_compound     default cascading-fault probability for sample_episode
    """
    classes: tuple = FAULT_CLASSES
    p_intermittent: float = 0.15
    p_transient: float = 0.15
    onset_range: tuple = (5.0, 60.0)
    horizon: float = 200.0
    class_weights: dict = field(default=None)
    mag_scale: float = 1.0
    p_compound: float = 0.1

    def _pick_class(self, rng):
        if not self.class_weights:
            return self.classes[rng.integers(len(self.classes))]
        w = np.array([max(float(self.class_weights.get(c, 0.0)), 0.0) for c in self.classes])
        w = w / w.sum() if w.sum() > 0 else np.full(len(self.classes), 1.0 / len(self.classes))
        return self.classes[rng.choice(len(self.classes), p=w)]

    def sample(self, rng, cls=None, mode=None) -> FaultSpec:
        cls = cls or self._pick_class(rng)
        modes = MODES[cls]
        mode = mode or list(modes)[rng.integers(len(modes))]
        mspec = modes[mode]
        lo, hi = mspec["mag"]
        mag = float(rng.uniform(lo, hi)) * float(self.mag_scale)
        pn, plo, phi = mspec["phys"]
        pv = float(rng.uniform(plo, phi)) if not isinstance(plo, int) else float(rng.integers(plo, phi + 1))
        onset = float(rng.uniform(*self.onset_range))
        duration = np.inf
        duty = 1.0
        u = rng.random()
        if u < self.p_transient:
            duration = float(rng.uniform(10.0, 0.5 * self.horizon))
        elif u < self.p_transient + self.p_intermittent:
            duty = float(rng.uniform(0.2, 0.7))
        return FaultSpec(cls=cls, mode=mode, mag=mag, phys_name=pn, phys_value=pv,
                         onset=onset, duration=duration, duty=duty,
                         ramp=bool(mspec.get("ramp", False)))

    def sample_episode(self, rng, n_deputies: int, fault_rate=0.6, p_compound=None):
        """-> {deputy_index: [FaultSpec, ...]} (empty dict entries omitted)."""
        p_compound = self.p_compound if p_compound is None else p_compound
        out = {}
        for i in range(n_deputies):
            if rng.random() >= fault_rate:
                continue
            specs = [self.sample(rng)]
            if rng.random() < p_compound:                       # cascading second fault
                child = self.sample(rng)
                child.onset = specs[0].onset + float(rng.uniform(10.0, 40.0))
                child.label = f"{specs[0].label}->{child.label}"
                specs.append(child)
            if specs[0].cls == "PEER":                          # byzantine needs victims
                nb = [j for j in range(n_deputies) if j != i]
                k = min(2, len(nb))
                specs[0].affected = tuple(rng.choice(nb, size=k, replace=False).tolist())
            out[i] = specs
        return out


# ══════════════════════════════ self-test ══════════════════════════════════════
def _selftest():
    rng = np.random.default_rng(7)
    lib = FaultLibrary()
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    # 1 determinism + coverage of every class/mode
    a = [FaultLibrary().sample(np.random.default_rng(s)) for s in range(60)]
    b = [FaultLibrary().sample(np.random.default_rng(s)) for s in range(60)]
    check("determinism (same seed -> same spec)",
          all(x.label == y.label and x.mag == y.mag and x.onset == y.onset for x, y in zip(a, b)))
    seen = {(s.cls, s.mode) for s in (FaultLibrary().sample(rng) for _ in range(400))}
    want = {(c, m) for c in FAULT_CLASSES for m in MODES[c]}
    check(f"taxonomy coverage ({len(seen)}/{len(want)} class.modes sampled)", seen == want)

    # 2 signature semantics
    t = np.arange(0.0, 200.0)
    sp = FaultSpec("NAV", "bias", 2.0, "relnav_bias_m", 0.1, onset=50.0, duration=np.inf)
    sig = sp.signature(t, "NAV")
    check("zero before onset", float(np.abs(sig[t < 50]).max()) == 0.0)
    check("plateau = mag after onset", np.allclose(sig[t >= 50], 2.0))
    check("secondary channel attenuated (FORM = mag/2)",
          np.allclose(sp.signature(t, "FORM")[t >= 50], 1.0))
    check("non-effect channel silent (ATT)", float(np.abs(sp.signature(t, "ATT")).max()) == 0.0)
    eff_ok = True
    for c in FAULT_CLASSES:
        spx = FaultSpec(c, list(MODES[c])[0], 1.0, "x", 1.0, onset=0.0, duration=np.inf)
        lit = {ch for ch in CHANNELS if spx.signature(np.array([10.0]), ch)[0] > 0}
        eff_ok &= (lit == spx.channels())
    check("EFFECTS consistency (all classes)", eff_ok)

    # 3 intermittent duty + transient duration
    si = FaultSpec("COMM", "latency", 1.0, "latency_s", 100.0, onset=0.0, duration=np.inf,
                   duty=0.4, period=20.0)
    frac = float((si.signature(t, "COMM") > 0).mean())
    check(f"intermittent duty ~0.4 (got {frac:.2f})", abs(frac - 0.4) < 0.05)
    st = FaultSpec("FORM", "station", 1.5, "station_offset_m", 500.0, onset=20.0, duration=30.0, ramp=True)
    check("transient silent after end", float(np.abs(st.signature(t, "FORM")[t > 60]).max()) == 0.0)

    # 4 data-domain injections (round-trip on synthetic healthy arrays)
    N = 400
    tt = np.arange(N, dtype=float)
    healthy = dict(y_pre=rng.normal(0, 1, (N, 3)), sigma_meas=np.ones((N, 3)),
                   d_theta=np.zeros(N), d_omega=np.zeros(N),
                   hb_age=np.full(N, 60.0), r_form=np.zeros(N),
                   peer_report=np.full(N, 100.0),
                   dv_real=np.tile([0.1, 0.0, 0.0], (5, 1)), burn_t=np.linspace(10, 350, 5))
    nb = FaultSpec("NAV", "bias", 2.0, "relnav_bias_m", 5.0, onset=100.0, duration=np.inf)
    out = nb.inject(healthy, tt, rng)
    m0 = np.abs(out["y_pre"][:100]).mean(); m1 = np.linalg.norm(out["y_pre"][100:], axis=1).mean()
    check(f"NAV bias shifts post-onset innovations ({m0:.2f}->{m1:.2f})", m1 > 3.0 and m0 < 1.5)
    check("inject leaves originals untouched", float(np.abs(healthy["d_theta"]).max()) == 0.0)
    at = FaultSpec("ATT", "tumble", 3.0, "torque_rate_degps", 0.5, onset=100.0, duration=np.inf, ramp=True)
    out = at.inject(healthy, tt, rng)
    check("ATT tumble grows past 1 deg", np.degrees(out["d_theta"][-1]) > 1.0 and out["d_theta"][:100].max() == 0.0)
    pr = FaultSpec("PROP", "scale", 2.0, "dv_scale_err", 0.3, onset=100.0, duration=np.inf)
    out = pr.inject(healthy, tt, rng)
    check("PROP scale hits only post-onset burns",
          np.allclose(out["dv_real"][:2], healthy["dv_real"][:2]) and
          np.allclose(np.linalg.norm(out["dv_real"][2:], axis=1), 0.13))
    so = FaultSpec("PROP", "stuck_off", 2.0, "missed_burns", 2, onset=100.0, duration=np.inf)
    out = so.inject(healthy, tt, rng)
    check("PROP stuck_off zeroes exactly 2 burns", int((np.linalg.norm(out["dv_real"], axis=1) == 0).sum()) == 2)
    pz = FaultSpec("PEER", "byzantine", 2.0, "report_scale", 10.0, onset=100.0, duration=np.inf)
    out = pz.inject(healthy, tt, rng)
    check("PEER byzantine scales self-report x10", np.allclose(out["peer_report"][100:], 1000.0))
    cd = FaultSpec("COMM", "dropout", 2.0, "outage_s", 600.0, onset=100.0, duration=np.inf, ramp=True)
    out = cd.inject(healthy, tt, rng)
    check("COMM dropout: hb_age grows monotonically", np.all(np.diff(out["hb_age"][100:]) >= 0)
          and out["hb_age"][-1] > 200.0)

    # 4b curriculum hooks
    wl = FaultLibrary(class_weights={"ATT": 1.0}, mag_scale=2.0)
    ws = [wl.sample(np.random.default_rng(s)) for s in range(40)]
    check("class_weights honored (40/40 ATT)", all(x.cls == "ATT" for x in ws))
    base = [FaultLibrary().sample(np.random.default_rng(s), cls="NAV", mode="bias").mag for s in range(20)]
    scaled = [FaultLibrary(mag_scale=2.0).sample(np.random.default_rng(s), cls="NAV", mode="bias").mag for s in range(20)]
    check("mag_scale doubles magnitudes", np.allclose(np.array(scaled), 2.0 * np.array(base)))

    # 5 episode sampler
    ep = lib.sample_episode(np.random.default_rng(3), 8, fault_rate=0.9, p_compound=0.5)
    n_compound = sum(len(v) > 1 for v in ep.values())
    peers_ok = all(len(v[0].affected) == 2 for v in ep.values() if v[0].cls == "PEER")
    check(f"episode sampler ({len(ep)} faulted deputies, {n_compound} compound)", len(ep) >= 5 and peers_ok)

    print("\nALL PASS" if ok else "\nSELF-TEST FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
