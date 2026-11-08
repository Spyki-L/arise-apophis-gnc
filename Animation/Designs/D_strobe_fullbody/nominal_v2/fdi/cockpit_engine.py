#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cockpit_engine.py — live mission engine for the FDIR autonomy cockpit.

Streams the nominal_v2 TRUTH forward in mission time and runs the ONBOARD autonomy
(FDI + recovery + a fast surrogate re-optimizer from this package) live, injecting
scripted failures. Because the 8.76 km/s flyby makes close-imaging a sub-second
snapshot burst, the protected quantity is the PREDICTED CA-geometry stereo coverage
(what the fleet will achieve at closest approach given its current/reconfigured
stations) — exactly the deadline-aware objective the autonomy values. On each event
the onboard decision is graded against a high-fidelity GROUND oracle, and the
compute time of each is recorded — the reliability proof.

  ONBOARD coverage = incidence-based stereo on a facet SUBSET (no occlusion)  -> ms
  GROUND  coverage = full mesh + occlusion (or MC_v2_run.score_draw if wired)  -> s
Pure numpy + this package; no visualization here (testable headless).
"""
from __future__ import annotations
import os, time
import numpy as np
import numpy.linalg as la

from fdi_core import FDICore, FDIConfig, FDIState, FaultType
from recovery_ladder import RecoveryLadder, FaultClass, Action

INCIDENCE_MAX = 60.0     # deg; a facet is imaged by a deputy within this cone of its normal
N_STEREO = 2             # facet stereo-covered if >= this many deputies image it
R_MAX = 8000.0           # m


# ───────────────────────── truth loading ────────────────────────────────────

def load_truth(data_dir, anim_dir, mesh_npz, facetprob_npz, N=8, n_frames=240):
    traj = [np.load(os.path.join(data_dir, f"apophis_rendezvous_full_D{d}.npz"), allow_pickle=True)
            for d in range(N)]
    t = traj[0]["t"].astype(float)
    i0 = int(0.02 * t.size)                                   # full mission (deploy -> CA)
    idx = np.linspace(i0, t.size - 1, n_frames).astype(int)
    T = t[idx]
    rA = traj[0]["rA_N"].astype(float)[idx]
    rC = traj[0]["rC_N"].astype(float)[idx]
    vC = traj[0]["vC_N"].astype(float)[idx]
    C_A = traj[0]["C_A_N"].astype(float)[idx]
    rD = np.empty((N, T.size, 3))                              # deputies: interpolate onto T
    for d in range(N):
        td = traj[d]["t"].astype(float); rd = traj[d]["rD_N"].astype(float)
        for j in range(3):
            rD[d, :, j] = np.interp(T, td, rd[:, j])
    # chief-LVLH (Hill) relative positions over the WHOLE mission -> complete trajectory trails
    hill = np.empty((N, T.size, 3)); hill_A = np.empty((T.size, 3))
    for k in range(T.size):
        r = rC[k]; v = vC[k]; ihat = r / la.norm(r)
        khat = np.cross(r, v); khat = khat / la.norm(khat); jhat = np.cross(khat, ihat)
        Chn = np.vstack([ihat, jhat, khat])                   # rows = R, T, N
        hill_A[k] = Chn @ (rA[k] - r)
        for d in range(N):
            hill[d, k] = Chn @ (rD[d, k] - r)
    mesh = np.load(mesh_npz, allow_pickle=True)
    cent = mesh["centroids"].astype(float); norm = mesh["normals"].astype(float)
    verts = mesh["vertices"].astype(float); faces = mesh["faces"].astype(int)
    facetprob = (np.load(facetprob_npz)["facet_prob"].astype(float)
                 if os.path.exists(facetprob_npz) else np.ones(len(cent)))
    i_ca = int(np.argmin(la.norm(rC - rA, axis=1)))
    # CA-geometry stations (asteroid body frame) — the terminal formation the autonomy protects
    Cca = C_A[i_ca]; rAca = rA[i_ca]
    ca_stations = np.stack([Cca @ (rD[d, i_ca] - rAca) for d in range(N)], 0)   # (N,3)
    return dict(T=T, rD=rD, rA=rA, rC=rC, C_A=C_A, cent=cent, norm=norm, verts=verts, faces=faces,
                facetprob=facetprob, N=N, i_ca=i_ca, t_ca=float(T[i_ca]), ca_stations=ca_stations,
                hill=hill, hill_A=hill_A)


# ───────────────────────── coverage scorers ─────────────────────────────────

def _stereo_cover(stations, cent, norm, incidence_max=INCIDENCE_MAX, nmin=N_STEREO, R_max=R_MAX, occ=False):
    cmax = np.cos(np.radians(incidence_max)); nf = len(cent); cov = 0
    covered_mask = np.zeros(nf, bool)
    for s in range(nf):
        p = cent[s]; n = norm[s]; k = 0
        for st in stations:
            los = st - p; rng = la.norm(los)
            if rng > R_max or (los / rng) @ n < cmax:
                continue
            if occ:
                u = los / rng; d2 = cent - p; proj = d2 @ u
                fr = (proj > 12.0) & (proj < rng)
                if fr.any() and (la.norm(d2[fr] - np.outer(proj[fr], u), axis=1) < 6.).any():
                    continue
            k += 1
        if k >= nmin:
            covered_mask[s] = True; cov += 1
    return cov / max(nf, 1), covered_mask


class OnboardCoverage:
    """Fast onboard surrogate: facet subset, no occlusion."""
    def __init__(self, truth, n_sub=300, seed=0):
        rng = np.random.default_rng(seed)
        self.sub = rng.choice(len(truth["cent"]), size=min(n_sub, len(truth["cent"])), replace=False)
        self.cent = truth["cent"][self.sub]; self.norm = truth["norm"][self.sub]

    def score(self, stations):
        t0 = time.perf_counter()
        c, _ = _stereo_cover(list(stations), self.cent, self.norm, occ=False)
        return c, time.perf_counter() - t0


class GroundOracle:
    """High-fidelity ground scorer: full mesh + occlusion (hook for MC_v2_run.score_draw)."""
    def __init__(self, truth, scripts_dir=None):
        self.cent = truth["cent"]; self.norm = truth["norm"]; self.real = None
        if scripts_dir:
            try:
                import sys
                if scripts_dir not in sys.path:
                    sys.path.insert(0, scripts_dir)
                import MC_v2_run  # noqa
                self.real = MC_v2_run
            except Exception:
                self.real = None

    def score(self, stations):
        t0 = time.perf_counter()
        # To use the real ray-traced scorer, build the fleet/hapke args and call
        # self.real.score_draw(...) here; default high-fidelity path = full mesh + occlusion.
        c, mask = _stereo_cover(list(stations), self.cent, self.norm, occ=True)
        return c, time.perf_counter() - t0, mask


# ───────────────────────── mission engine ───────────────────────────────────

class MissionEngine:
    def __init__(self, truth, events, onboard_cov, ground, reach_m=600., dv_axis=1.0):
        self.tr = truth; self.events = {e["frame"]: e for e in events}
        self.onb = onboard_cov; self.gnd = ground
        self.N = truth["N"]; self.active = set(range(self.N))
        self.reach = np.full(self.N, reach_m); self.dv_axis = dv_axis
        self.health = {d: "HEALTHY" for d in range(self.N)}
        self.recovery = RecoveryLadder(d_safe=50.)
        self.offset = {d: np.zeros(3) for d in range(self.N)}   # CA-station re-tasking offset (body)
        self.log = []; self.compute = []
        self.cov_onb = self.cov_gnd = None

    def ca_stations(self):
        return {d: self.tr["ca_stations"][d] + self.offset[d] for d in self.active}

    def live_positions(self, fi):
        """Live deputy positions in asteroid body frame at frame fi (for the 3D animation)."""
        C = self.tr["C_A"][fi]; rA = self.tr["rA"][fi]
        out = {}
        for d in range(self.N):
            base = C @ (self.tr["rD"][d, fi] - rA)
            out[d] = base + (self.offset[d] if d in self.active else np.zeros(3))
        return out

    def _reopt(self, fi):
        t0 = time.perf_counter()
        base = {d: self.tr["ca_stations"][d] for d in self.active}
        rng = np.random.default_rng(1000 + fi)
        off = {d: self.offset[d].copy() for d in self.active}
        def cov(o):
            return self.onb.score([base[d] + o[d] for d in self.active])[0]
        best = cov(off)
        for _ in range(120):
            d = list(self.active)[rng.integers(len(self.active))]
            cand = {k: v.copy() for k, v in off.items()}
            cand[d] = np.clip(off[d] + rng.normal(0, self.reach[d] * 0.3, 3), -self.reach[d], self.reach[d])
            c = cov(cand)
            if c > best:
                off = cand; best = c
        for d in self.active:
            self.offset[d] = off[d]
        return best, time.perf_counter() - t0

    def step(self, fi):
        t_go = self.tr["t_ca"] - float(self.tr["T"][fi])
        ev = self.events.get(fi)
        event_name = None; onb_ms = gnd_ms = None; capture = None
        cov_before = self.onb.score(list(self.ca_stations().values()))[0]

        if ev:
            event_name = ev["type"]
            if ev["type"] == "DEPUTY_LOSS":
                lost = ev["dep"]; self.active.discard(lost); self.health[lost] = "FAULTED"
                _, onb_dt = self._reopt(fi); onb_ms = onb_dt * 1e3
                peer = ev.get("peer", min(self.active))
                capture = dict(deputy=peer, target_dep=lost)
                cg, gdt, _ = self.gnd.score(list(self.ca_stations().values())); gnd_ms = gdt * 1e3; self.cov_gnd = cg
                self.log.append((fi, f"D{lost+1} LOST -> onboard reopt {onb_ms:.0f}ms + D{peer+1} re-images; ground oracle {gnd_ms:.0f}ms"))
            elif ev["type"] == "ACTUATION_DEGRADE":
                self.reach[ev["dep"]] = ev.get("new_reach", 80.)
                _, onb_dt = self._reopt(fi); onb_ms = onb_dt * 1e3
                cg, gdt, _ = self.gnd.score(list(self.ca_stations().values())); gnd_ms = gdt * 1e3; self.cov_gnd = cg
                self.log.append((fi, f"D{ev['dep']+1} actuator degraded -> re-task within reach ({onb_ms:.0f}ms)"))
            elif ev["type"] == "MISBEHAVING":
                sus = ev["dep"]; st = self.ca_stations()
                fp = [st.get(d, self.tr["ca_stations"][d]) for d in range(self.N)]
                res = self.recovery.run(suspect=sus, consensus_label=FaultClass.TUMBLE,
                                        vision_fn=lambda s: (FaultClass.TUMBLE, 0.92), isl_logs={sus: FaultClass.NONE},
                                        fleet_pos=fp, fleet_vel=[np.zeros(3)]*self.N,
                                        suspect_state=(fp[sus], ev.get("vel", np.zeros(3))))
                if res["action"] is Action.DISPOSE and res.get("collision_free"):
                    self.active.discard(sus); self.health[sus] = "DEAD"
                _, onb_dt = self._reopt(fi); onb_ms = onb_dt * 1e3
                cg, gdt, _ = self.gnd.score(list(self.ca_stations().values())); gnd_ms = gdt * 1e3; self.cov_gnd = cg
                self.log.append((fi, f"D{sus+1} misbehaving -> {res['action'].name} + reopt ({onb_ms:.0f}ms)"))

        # hold the peer "camera click" on-screen for a few frames so it reads in the live anim
        if capture is not None:
            self._cap_hold = [12, capture]
        elif getattr(self, "_cap_hold", [0, None])[0] > 0:
            self._cap_hold[0] -= 1; capture = self._cap_hold[1]

        cov_o, dt_o = self.onb.score(list(self.ca_stations().values()))
        self.cov_onb = cov_o
        onb_score_ms = dt_o * 1e3
        reopt_ms = onb_ms if (ev and onb_ms is not None) else None
        onb_ms = onb_score_ms
        if ev:
            self.compute.append((fi, onb_score_ms, gnd_ms or 0.0, reopt_ms or 0.0, cov_o, self.cov_gnd or cov_o))

        return dict(frame=fi, t_go=t_go, active=sorted(self.active), health=dict(self.health),
                    cov_before=cov_before, cov_onb=cov_o, cov_gnd=self.cov_gnd, event=event_name,
                    capture=capture, onb_ms=onb_ms, gnd_ms=gnd_ms, reopt_ms=reopt_ms,
                    live=self.live_positions(fi), ca=self.ca_stations(),
                    ca_all={d: self.tr["ca_stations"][d] + self.offset[d] for d in range(self.N)},
                    log_tail=self.log[-1][1] if self.log else "")


def default_events(truth):
    F = truth["T"].size
    return [
        dict(frame=int(0.30 * F), type="DEPUTY_LOSS", dep=2, peer=1),
        dict(frame=int(0.52 * F), type="ACTUATION_DEGRADE", dep=5, new_reach=90.),
        dict(frame=int(0.74 * F), type="MISBEHAVING", dep=6, vel=np.array([2., 0, 0])),
    ]


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    B = os.path.abspath(os.path.join(HERE, "..", ".."))
    truth = load_truth(os.path.join(B, "data", "trajectories"), os.path.join(B, "data", "anim"),
                       os.path.join(HERE, "..", "data", "radar_mesh.npz"),
                       os.path.join(HERE, "..", "data", "mc_v2_results_nominal_facetprob.npz"), n_frames=120)
    print(f"truth: {truth['N']} deputies, {truth['T'].size} frames, t_ca={truth['t_ca']:.0f}s, {len(truth['cent'])} facets")
    eng = MissionEngine(truth, default_events(truth), OnboardCoverage(truth), GroundOracle(truth))
    t0 = time.perf_counter()
    for fi in range(truth["T"].size):
        s = eng.step(fi)
        if s["event"]:
            print(f"  f{fi}: {s['event']:17s} cov {s['cov_before']:.3f}->{s['cov_onb']:.3f} "
                  f"(ground {s['cov_gnd']:.3f})  onboard {s['onb_ms']:.0f}ms ground {s['gnd_ms']:.0f}ms")
    print(f"\nplayed {truth['T'].size} frames in {time.perf_counter()-t0:.1f}s")
    print("ONBOARD vs GROUND (reliability proof):")
    for c in eng.compute:
        print(f"  f{c[0]:3d} onboard-score {c[1]:5.1f}ms  ground-score {c[2]:6.0f}ms (x{c[2]/max(c[1],1e-3):.0f})  "
              f"reopt {c[3]:5.0f}ms  cov_onb={c[4]:.3f} cov_gnd={c[5]:.3f}  |agree|={abs(c[4]-c[5]):.3f}")
