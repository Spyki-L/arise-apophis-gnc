#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basilisk_backend.py — HIGH-FIDELITY backend for fdir_env (CORTEX build step 5).

Drop-in replacement for ``fdir_env.SurrogateBackend`` (same interface: reset /
residuals / active / apply / value + attrs auth, auth0, kappa, offsta), stepping
REAL Design-D dynamics instead of synthetic noise.  Two modes:

* ``mode="replay"``  (DEFAULT, wired, self-tested) — steps through a recorded hi-fi
  MC_v2 draw (``data/mc/draw_*/``: Basilisk-pipeline nav/attitude/control products).
  The four hooks are WIRED here:
    1. _build_sim()        -> ReplaySource: loads the draw via run_fdi_nominal.load_fleet
                              (same bundles, same timebase, same ATT envelope adapter).
    2. _inject_fault()     -> fault_library.FaultSpec.inject() perturbs the RAW streams
                              (EKF innovations, pointing/rate deviation, burn dV,
                              heartbeat, peer reports) in the data domain; residuals are
                              then recomputed with the SAME formulas the FDI uses.
    3. _read_residuals()   -> the 6 channels exactly as run_fdi_nominal constructs them:
                              NAV innovation NIS-RMS, ATT envelope-deviation residual,
                              COMM hb-age ratio, PROP whitened dV-mismatch at burns,
                              FORM causal terminal-miss, PEER streaming MAD consensus —
                              normalised to surrogate severity units via surrogate_calib.
    4. _apply_action()     -> recovery semantics mirroring the surrogate (REINIT/MODE/
                              RELOAD clear soft faults, SAFEHOLD neutralises ATT/PROP,
                              ISOLATE excludes a byzantine from the consensus replay,
                              DISPOSE removes the deputy, REOPT recovers coverage), so
                              observation/action/reward are identical across fidelities.

* ``mode="live"`` — steps a LIVE Basilisk sim.  Requires the Basilisk machine; the
  construction seam is a single function (``_build_live_sim``) documented below with
  the scenario chain to reuse (run_nominal_and_hifi.py stage A/B or
  scripts/guidance/basilik_hifi_apophis_rendezvous_fleet.py).  Everything downstream
  (injection via fault_library physical params, residual formulas, action semantics)
  is SHARED with replay — wiring live mode is now only "produce the same stream dict
  from the running sim".

Episode window: the last ``horizon*dt`` seconds before TCA (default 200 x 10 s =
the FORM-window approach + encounter), where recovery actions actually matter.

Self-test (no Basilisk needed):  python3 basilisk_backend.py  -> ALL PASS.
"""
from __future__ import annotations
import os
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import numpy.linalg as la

from fdir_env import CH, FAULTS, ACT_COST, SurrogateBackend, cm  # noqa: F401

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))                      # fdi/
import run_fdi_nominal as RFN
from channels import causal_form_residual, PeerConsensusBuffer
from fault_library import FaultLibrary, FaultSpec

try:
    import Basilisk                      # noqa: F401  (only on the flight-sim machine)
    _HAVE_BASILISK = True
except Exception:
    _HAVE_BASILISK = False

SIGMA_SIGMA = np.deg2rad(0.5)
SIGMA_OMEGA = np.deg2rad(0.02)
_DEF_DRAW = _HERE.parents[1] / "data" / "mc" / "draw_0000"


def _severity_norms():
    """Real-units -> surrogate-severity normalisation (surrogate_calib.npz f_ch);
    identity when the calibration artifact is absent."""
    f = {c: 1.0 for c in CH}
    p = _HERE / "surrogate_calib.npz"
    if p.exists():
        try:
            from surrogate_calibration import CalibratedNoise
            cn = CalibratedNoise(p)
            f.update({c: cn.f.get(c, 1.0) for c in CH})
        except Exception:
            pass
    return f


class ReplaySource:
    """Recorded hi-fi draw -> per-deputy raw streams + FDI-formula residual series,
    with data-domain fault injection and consensus replay."""

    _fleet_cache = OrderedDict()                        # draw path -> (fleet, t_tca), LRU
    _cache_cap = int(os.getenv("ARISE_REPLAY_CACHE", "6"))   # per-process cap (HPC memory guard)

    def __init__(self, draw_dir, n, t0=None, dt=10.0, horizon=200):
        self.n = n; self.dt = float(dt); self.horizon = int(horizon)
        key = str(Path(draw_dir).resolve())
        cache = ReplaySource._fleet_cache
        if key in cache:
            cache.move_to_end(key)
        else:
            cache[key] = RFN.load_fleet(Path(draw_dir) / "_nav_stage")
            while len(cache) > ReplaySource._cache_cap:
                cache.popitem(last=False)
        self.fleet, self.t_tca = cache[key]
        self.t0 = float(self.t_tca - self.horizon * self.dt) if t0 is None else float(t0)
        self.norm = _severity_norms()
        self.spec = {}                                  # deputy -> [FaultSpec] (episode-step timebase)
        self.excluded = set()                           # ISOLATEd from consensus
        self._raw = [self._raw_streams(d) for d in range(n)]
        self._recompute()

    # ---- raw stream extraction (same arrays run_fdi_nominal consumes) -----------
    def _raw_streams(self, d):
        g = self.fleet[d]
        d_theta = 4.0 * np.arctan(la.norm(g["sBR"], axis=1))
        d_omega = la.norm(g["oBR"], axis=1)
        return dict(t=g["t"].copy(), y_pre=g["ypre"].copy(), sigma_meas=g["sig"].copy(),
                    rejected=g["rejected"].copy(), d_theta=d_theta, d_omega=d_omega,
                    hb_age=np.full(g["t"].size, RFN.HB_NOMINAL),
                    x=g["x"], P=g["P"], dv=g["dv"], tf=g["tf"], rC=g["rC"], vC=g["vC"],
                    rA=g["rA"], vA=g["vA"], target=g["target"])

    def _episode_env(self, spec, t_abs):
        """FaultSpec envelope evaluated on ABSOLUTE data time (spec times are episode steps)."""
        steps = (t_abs - self.t0) / self.dt
        return spec.envelope(steps)

    def _perturbed(self, d):
        """Apply this deputy's active specs to its raw streams (data domain)."""
        s = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self._raw[d].items()}
        for sp in self.spec.get(d, ()):
            steps = (s["t"] - self.t0) / self.dt
            out = sp.inject({k: s[k] for k in ("y_pre", "sigma_meas", "d_theta", "d_omega", "hb_age")},
                            steps, np.random.default_rng(17 * (d + 1)))
            s.update(out)
        return s

    # ---- residual series with the FDI's own formulas ----------------------------
    def _recompute(self):
        self.series = []
        pert = [self._perturbed(d) for d in range(self.n)]
        for d in range(self.n):
            s = pert[d]
            z = s["y_pre"] / np.maximum(s["sigma_meas"], 1e-3)
            r_nav = np.sqrt(np.einsum("ij,ij->i", z, z) / 3.0)
            bad = ~np.isfinite(r_nav) | s["rejected"]
            r_nav[bad] = 0.0                                     # FDI skips these samples
            r_att = np.tan(0.25 * s["d_theta"]) / SIGMA_SIGMA + s["d_omega"] / SIGMA_OMEGA
            r_comm = s["hb_age"] / RFN.HB_NOMINAL
            r_prop = np.zeros_like(r_nav)
            g = self.fleet[d]
            dv_pert = g["dv"].copy()
            for sp in self.spec.get(d, ()):
                if sp.cls == "PROP":
                    burns = la.norm(dv_pert, axis=1) > 0
                    bt = (s["t"] - self.t0) / self.dt
                    out = sp.inject(dict(dv_real=dv_pert[burns], burn_t=bt[burns]),
                                    bt[burns], np.random.default_rng(7))
                    dv_pert[burns] = out["dv_real"]
            for k in np.where(la.norm(g["dv"], axis=1) > 0)[0]:
                if k == 0:
                    continue
                # executed dv differs from commanded: mismatch = (measured dv) - commanded,
                # where measured dv = healthy measured dv + (dv_real - dv_cmd)
                de = (g["x"][k, 3:6] - g["x"][k - 1, 3:6]) + (dv_pert[k] - g["dv"][k]) - g["dv"][k]
                try:
                    zz = la.solve(g["P"][k, 3:6, 3:6] + 1e-12 * np.eye(3), de)
                    r_prop[k] = np.sqrt(max(float(de @ zz), 0.0) / 3.0)
                except la.LinAlgError:
                    pass
            self.series.append(dict(t=s["t"], NAV=r_nav, ATT=r_att, COMM=r_comm, PROP=r_prop,
                                    FORM=np.zeros_like(r_nav), PEER=np.zeros_like(r_nav)))
        # ---- FORM/PEER: causal consensus replay over the whole fleet ------------
        peer = PeerConsensusBuffer(n_dep=self.n, latency_s=0.0, mad_floor=0.25)
        events = sorted(((float(pert[d]["t"][k]), d, k)
                         for d in range(self.n) for k in range(pert[d]["t"].size)), key=lambda e: e[0])
        for t_k, d, k in events:
            if not ((self.t_tca - RFN.FORM_WINDOW_S) <= t_k < self.t_tca):
                continue
            g = self.fleet[d]; s = pert[d]
            ttg = max(self.t_tca - t_k, 0.0)
            w = 1.0 + 1.5 * np.clip(1.0 - ttg / RFN.FORM_WINDOW_S, 0.0, 1.0)
            miss = causal_form_residual(
                g["x"][k, 0:3], g["x"][k, 3:6], t_k, self.t_tca,
                RFN._at(s["tf"], s["rC"], t_k), RFN._at(s["tf"], s["vC"], t_k),
                RFN._at(s["tf"], s["rC"], self.t_tca), RFN._at(s["tf"], s["vC"], self.t_tca),
                RFN._at(s["tf"], s["rA"], self.t_tca), RFN._at(s["tf"], s["vA"], self.t_tca),
                station_cmd=s["target"], scale_m=1.0, w_timetogo=1.0)
            report = miss
            for sp in self.spec.get(d, ()):
                env_k = float(self._episode_env(sp, np.array([t_k]))[0])
                if sp.cls in ("FORM", "ENV") and env_k > 0:
                    miss = miss + (sp.phys_value if sp.cls == "FORM" else sp.mag * RFN.FORM_SCALE_M) * env_k
                    report = miss
                if sp.cls == "PEER" and env_k > 0:
                    report = (miss * sp.phys_value if sp.mode == "byzantine" else report)
            if d not in self.excluded:
                peer.push(d, t_k, report)
            scale = peer.fleet_scale(t_k, floor=RFN.FORM_SCALE_M)
            self.series[d]["FORM"][k] = (miss / scale) * w
            self.series[d]["PEER"][k] = peer.r_peer(d, t_k, own_value=report, scale=scale)

    # ---- stepping interface ------------------------------------------------------
    def residuals_at(self, d, step):
        t_q = self.t0 + step * self.dt
        s = self.series[d]
        k = int(np.clip(np.searchsorted(s["t"], t_q), 0, s["t"].size - 1))
        out = np.empty(len(CH))
        for ci, c in enumerate(CH):
            if c == "PROP":                                    # sparse: hold last burn's evidence
                kk = np.where((la.norm(self.fleet[d]["dv"], axis=1) > 0) & (s["t"] <= t_q))[0]
                out[ci] = s["PROP"][kk[-1]] if kk.size else 0.0
            else:
                out[ci] = s[c][k]
            out[ci] *= self.norm.get(c, 1.0)
        return np.clip(out, 0.0, 10.0)

    def inject(self, d, spec):
        self.spec.setdefault(d, []).append(spec); self._recompute()

    def clear(self, d):
        if d in self.spec:
            self.spec.pop(d); self._recompute()

    def isolate(self, d):
        self.excluded.add(d); self._recompute()


class BasiliskBackend:
    """High-fidelity backend, same interface as SurrogateBackend."""

    def __init__(self, n, dt=10.0, auth0=5.0, scenario_cfg=None, require_basilisk=False,
                 mode=None, rho=0.6):
        self.n = n; self.dt = dt; self.auth0 = auth0; self.cfg = scenario_cfg or {}
        self.rho = rho                      # byzantine corruption factor (shared with surrogate value model)
        self.mode = mode or self.cfg.get("mode", "replay")
        if self.mode == "live" and require_basilisk and not _HAVE_BASILISK:
            raise RuntimeError("Basilisk not importable: run live mode on the Basilisk "
                               "machine, or use mode='replay' (recorded hi-fi draw).")
        self.lib = FaultLibrary(horizon=float(self.cfg.get("horizon", 200)))
        self.draws = self._resolve_draws()
        self.sim = None

    def _resolve_draws(self):
        """cfg['draws']=list | cfg['draws_root']=dir of draw_* | cfg['draw']=single.
        A list -> one draw sampled per episode: domain randomization over the REAL
        dispersed MC missions (the HPC training mode)."""
        if self.cfg.get("draws"):
            return [Path(d) for d in self.cfg["draws"]]
        if self.cfg.get("draws_root"):
            root = Path(self.cfg["draws_root"])
            out = [p for p in sorted(root.glob("draw_*"))
                   if (p / "_nav_stage" / "apophis_swarm_FDI_rgazel_D0.npz").exists()]
            if out:
                return out
        return [Path(self.cfg.get("draw", _DEF_DRAW))]

    # ---- interface ---------------------------------------------------------------
    def reset(self, rng, fault_rate):
        self._draw_rng = rng
        self.kappa = np.ones(self.n); self.auth = np.full(self.n, self.auth0)
        self.offsta = np.zeros(self.n); self.removed = set(); self.isolated = set()
        self.fault = ["NONE"] * self.n; self.onset = [10 ** 9] * self.n; self.corrupt = {}
        self.specs = {}
        for i, specs in self.lib.sample_episode(rng, self.n, fault_rate=fault_rate).items():
            sp = specs[0]
            self.specs[i] = specs
            self.fault[i] = sp.cls if sp.cls != "ENV" else "NAV"
            self.onset[i] = int(sp.onset)
            if sp.cls == "PEER" and sp.affected:
                self.corrupt[i] = set(sp.affected)
        self._build_sim()                                   # HOOK 1 (wired: replay | live seam)

    def active(self, i, step):
        return (self.fault[i] != "NONE" and step >= self.onset[i]
                and i not in self.removed and not (self.fault[i] == "PEER" and i in self.isolated))

    def residuals(self, i, step, rng):
        if step != self._step_now:                          # once per ENV step (not per deputy)
            self._advance_sim()
            self._step_now = step
        specs = self.specs.get(i)
        if (specs and i in self._injected
                and all(np.isfinite(sp.duration) and step >= sp.onset + sp.duration for sp in specs)):
            self.fault[i] = "NONE"; self.specs.pop(i, None)  # transient fault expired
            self.sim.clear(i)
        if self.active(i, step) and step == self.onset[i] and i not in self._injected:
            self._inject_fault(i, self.fault[i])            # HOOK 2 (wired)
        return self._read_residuals(i)                      # HOOK 3 (wired)

    def apply(self, i, action, step):
        self.auth[i] = max(0.0, self.auth[i] - ACT_COST[action])
        return self._apply_action(i, action)                # HOOK 4 (wired)

    def value(self):
        return SurrogateBackend.contribution(self).sum()

    # ---- HOOK 1: build the sim -----------------------------------------------------
    def _build_sim(self):
        self._step_now = 0
        self._injected = set()
        if self.mode == "replay":
            draw = self.draws[self._draw_rng.integers(len(self.draws))]
            self.sim = ReplaySource(draw, self.n,
                                    dt=self.dt, horizon=int(self.cfg.get("horizon", 200)))
        else:
            self.sim = self._build_live_sim()

    def _build_live_sim(self):
        """LIVE seam (Basilisk machine only).  Reuse the validated scenario chain:
        run_nominal_and_hifi.py stages A/B (or scripts/guidance/
        basilik_hifi_apophis_rendezvous_fleet.py + scripts/sim/build_6dof_anim_data.py)
        to build chief+8 deputies with camera-tracking attitude, then return an object
        exposing the SAME dict of raw streams ReplaySource._raw_streams produces
        (y_pre/sigma_meas from the nav filter, d_theta/d_omega vs the reference,
        hb_age, dv realized vs commanded, and the ephemeris arrays) advanced by
        ``self.dt`` per _advance_sim().  Injection/residuals/actions are already
        shared with replay — no other code changes needed."""
        raise NotImplementedError("live mode: wire _build_live_sim on the Basilisk machine "
                                  "(see docstring); replay mode works everywhere.")

    def _advance_sim(self):
        pass                                                 # replay: series precomputed; live: step Basilisk by dt

    # ---- HOOK 2: inject -------------------------------------------------------------
    def _inject_fault(self, i, ftype):
        specs = self.specs.get(i) or [self.lib.sample(np.random.default_rng(101 + i), cls=ftype)]
        for sp in specs:
            self.sim.inject(i, sp)
        self._injected.add(i)

    # ---- HOOK 3: read residuals ------------------------------------------------------
    def _read_residuals(self, i):
        return self.sim.residuals_at(i, self._step_now)

    # ---- HOOK 4: apply action --------------------------------------------------------
    def _apply_action(self, i, action):
        from fdir_env import CLEARS, NEUTRAL
        f = self.fault[i]; correct = False
        if action in CLEARS and f == CLEARS[action]:
            self.fault[i] = "NONE"; self.sim.clear(i); correct = True
        elif action == "SAFEHOLD" and f in NEUTRAL["SAFEHOLD"]:
            self.fault[i] = "NONE"; self.sim.clear(i); correct = True
        elif action == "ISOLATE" and f == "PEER":
            self.isolated.add(i); self.sim.isolate(i); correct = True
        elif action == "DISPOSE" and f == "PEER":
            self.removed.add(i); self.sim.isolate(i); correct = True
        elif action == "REOPT":
            self.kappa[i] = min(1.0, self.kappa[i] + 0.1)
        return correct


# ═══════════════════════════════ self-test ═══════════════════════════════════════
def _selftest():
    ok = True
    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    draw = _DEF_DRAW
    if not (Path(draw) / "_nav_stage" / "apophis_swarm_FDI_rgazel_D0.npz").exists():
        print("no local draw -> SKIP (need data/mc/draw_*)"); return True

    src = ReplaySource(draw, 8, dt=10.0, horizon=200)
    H = 200
    sev = np.array([[src.residuals_at(d, k) for k in range(0, H, 5)] for d in range(8)])
    check(f"healthy replay stays sub-alarm (p99={np.percentile(sev, 99):.2f} < 1.5)",
          np.percentile(sev, 99) < 1.5)
    check("healthy replay nonneg + bounded", sev.min() >= 0.0 and sev.max() <= 10.0)

    # NAV bias on D2
    src.inject(2, FaultSpec("NAV", "bias", 2.0, "relnav_bias_m", 0.5, onset=50.0, duration=np.inf))
    post = np.array([src.residuals_at(2, k) for k in range(60, 200, 10)])
    pre = np.array([src.residuals_at(2, k) for k in range(0, 45, 5)])
    iN = CH.index("NAV")
    check(f"NAV bias raises NAV severity ({pre[:, iN].mean():.2f} -> {post[:, iN].mean():.2f})",
          post[:, iN].mean() > max(3 * pre[:, iN].mean(), 1.0))
    others = np.array([src.residuals_at(d, 150)[iN] for d in range(8) if d != 2])
    check("other deputies' NAV unaffected", (others < 1.5).all())

    # ATT tumble on D5
    src.inject(5, FaultSpec("ATT", "tumble", 3.0, "torque_rate_degps", 0.3, onset=40.0,
                            duration=np.inf, ramp=True))
    iA = CH.index("ATT")
    a_post = src.residuals_at(5, 150)[iA]; a_pre = src.residuals_at(5, 20)[iA]
    check(f"ATT tumble raises ATT severity ({a_pre:.2f} -> {a_post:.2f})", a_post > 1.5 and a_pre < 0.5)

    # PEER byzantine on D1 -> own PEER residual deviates; ISOLATE clears it
    src2 = ReplaySource(draw, 8, dt=10.0, horizon=200)
    iP = CH.index("PEER")
    base_peer = max(src2.residuals_at(1, k)[iP] for k in range(120, 200, 10))
    src2.inject(1, FaultSpec("PEER", "byzantine", 2.0, "report_scale", 15.0, onset=0.0,
                             duration=np.inf, affected=(0, 3)))
    byz_peer = max(src2.residuals_at(1, k)[iP] for k in range(120, 200, 10))
    check(f"byzantine raises own PEER residual ({base_peer:.2f} -> {byz_peer:.2f})",
          byz_peer > 2 * max(base_peer, 0.1))

    # backend interface + action semantics
    b = BasiliskBackend(8, scenario_cfg=dict(draw=draw, horizon=120))
    rng = np.random.default_rng(4)
    b.reset(rng, fault_rate=0.7)
    n_f = sum(f != "NONE" for f in b.fault)
    check(f"backend reset samples faults ({n_f}/8)", 1 <= n_f <= 8)
    r0 = b.residuals(0, 0, rng)
    check("backend residuals shape/finite", r0.shape == (6,) and np.isfinite(r0).all())
    i_f = next((i for i, f in enumerate(b.fault) if f in ("NAV", "COMM", "FORM")), None)
    if i_f is not None:
        for st in range(int(b.onset[i_f]) + 1):
            b.residuals(i_f, st, rng)                       # walk to onset (injects)
        act = {"NAV": "REINIT", "COMM": "RELOAD", "FORM": "MODE"}[b.fault[i_f]]
        check(f"correct recovery clears D{i_f} ({act})", b.apply(i_f, act, int(b.onset[i_f]) + 1))
    check("value() finite", np.isfinite(b.value()))

    print("\nALL PASS" if ok else "\nSELF-TEST FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
