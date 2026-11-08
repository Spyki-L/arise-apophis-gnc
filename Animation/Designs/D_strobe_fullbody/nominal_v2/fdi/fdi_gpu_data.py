#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fdi_gpu_data.py — the Qt-free data layer for the GPU fault lab.

It produces the live FDI state each frame and turns it into the numeric series the GPU panels
plot. It has NO Qt / pyqtgraph dependency, so it runs (and is tested) anywhere.

On your machine it drives the REAL FDI engine (the same one live_fault_lab.py uses). If that
engine or its trajectory data can't be loaded (e.g. you just want to try the UI), it falls back
to a self-contained MOCK campaign that mimics the real state shape — so the app always runs.

State contract used by the panels (a subset of the real engine's state dict):
    st["frame"]               int
    st["t_go"]                float   seconds to closest approach
    st["thresholds"][ch]      float   per-channel CUSUM trip threshold
    st["fdi"][d]["S"][ch]     float   leaky-CUSUM statistic, deputy d, channel ch
    st["fdi"][d]["state"]     str     HEALTHY | SUSPECT | FAULTED | DEAD
    st["fdi"][d]["fault"]     str     isolated channel name or "NONE"
    st["fdi"][d]["conf"]      float   detection confidence 0..1
    st["fdi"][d]["r"][ch]     float   raw residual
    st["health"][d]           str     fleet health (same vocabulary as state)
    st["gt_fault"][d]         str     ground-truth injected fault (for verification)
"""
from __future__ import annotations
import os
import sys
import numpy as np

CHANNELS = ["NAV", "ATT", "COMM", "PROP", "FORM", "PEER"]
N_DEP = 8

STATE_COLORS = {                      # mirrors cockpit_ops STATE_COLORS (hex, for the GPU side)
    "HEALTHY": "#39e08a",
    "SUSPECT": "#ffc23d",
    "FAULTED": "#ff4d4d",
    "DEAD":    "#9aa0a6",
}
CHANNEL_COLORS = {
    "NAV": "#4fb0ff", "ATT": "#ffb14e", "COMM": "#ff5d7a", "PROP": "#b98cff",
    "FORM": "#52e0c4", "PEER": "#f2f25c",
}


# ──────────────────────────────────────────────────────────────────────────────
# Mock engine — a scripted seed-7-flavoured campaign so the UI runs with no data.
# ──────────────────────────────────────────────────────────────────────────────
class MockFDIEngine:
    """Self-contained stand-in producing states with the real shape (no Basilisk needed)."""

    # (deputy index, channel, onset frame, kind)  — loosely the seed-7 campaign
    CAMPAIGN = [
        (4, "PEER", 119, "soft"), (6, "COMM", 212, "dead"), (5, "NAV", 332, "soft"),
        (0, "FORM", 472, "soft"), (3, "PEER", 499, "soft"), (1, "ATT", 563, "hold"),
        (0, "PROP", 564, "hold"), (4, "ATT", 575, "hold"), (3, "PROP", 684, "hold"),
        (1, "NAV", 812, "soft"),
    ]

    def __init__(self, n_frames=1000, seed=7):
        self.n_frames = int(n_frames)
        self.N = N_DEP
        self.rng = np.random.default_rng(seed)
        self.thr = {"NAV": 7.0, "ATT": 7.0, "COMM": 6.0, "PROP": 7.0, "FORM": 6.5, "PEER": 7.5}
        self._S = {d: {c: 0.0 for c in CHANNELS} for d in range(self.N)}
        self._dead = set()
        self._noise = {d: {c: 0.0 for c in CHANNELS} for d in range(self.N)}

    def _active(self, fi):
        """Return {(d,ch): kind} for faults whose onset has passed and not yet recovered."""
        out = {}
        for (d, ch, onset, kind) in self.CAMPAIGN:
            if fi >= onset:
                age = fi - onset
                if kind == "soft" and age > 90:        # soft faults recover
                    continue
                if kind == "hold" and age > 130:       # safe-hold then restore
                    continue
                out[(d, ch)] = kind
        return out

    def step(self, fi):
        act = self._active(fi)
        gt = {}
        for d in range(self.N):
            for c in CHANNELS:
                tgt = 0.0
                if (d, c) in act:
                    gt[d] = c
                    h = self.thr[c]
                    tgt = h * (1.25 if act[(d, c)] != "soft" else 1.08)
                # leaky integrate toward target + a little noise
                lead = 0.18 if tgt > 0 else 0.12
                self._S[d][c] += lead * (tgt - self._S[d][c])
                self._S[d][c] = max(0.0, self._S[d][c] + self.rng.normal(0, 0.05))
        # state machine (simplified but same vocabulary)
        fdi = {}
        health = {}
        for d in range(self.N):
            if (d == 6 and fi >= 212):
                self._dead.add(d)
            state, fault, conf = "HEALTHY", "NONE", 0.0
            if d in self._dead:
                state, fault, conf = "DEAD", "COMM", 1.0
            else:
                worst_c, worst_r = None, 0.0
                for c in CHANNELS:
                    r = self._S[d][c] / self.thr[c]
                    if r > worst_r:
                        worst_r, worst_c = r, c
                if worst_r >= 1.0:
                    state, fault, conf = "FAULTED", worst_c, min(1.0, worst_r / 1.3)
                elif worst_r >= 0.6:
                    state, fault, conf = "SUSPECT", worst_c, worst_r * 0.6
            fdi[d] = {
                "S": dict(self._S[d]),
                "state": state, "fault": fault, "conf": conf,
                "r": {c: self._S[d][c] * 0.6 + abs(self.rng.normal(0, 0.1)) for c in CHANNELS},
            }
            health[d] = state
        t_go = max(0.0, (self.n_frames - fi) / self.n_frames * 10.6 * 3600.0)
        return {"frame": int(fi), "t_go": t_go, "thresholds": dict(self.thr),
                "fdi": fdi, "health": health, "gt_fault": gt}


# ──────────────────────────────────────────────────────────────────────────────
# Source — real engine if possible, else the mock. Same .step(fi) -> state.
# ──────────────────────────────────────────────────────────────────────────────
class FDISource:
    def __init__(self, sim_frames=1000, seed=7, force_mock=False):
        self.sim_frames = int(sim_frames)
        self.seed = int(seed)
        self.N = N_DEP
        self.channels = list(CHANNELS)
        self.is_mock = True
        self._engine = None
        self._truth = None
        if not force_mock:
            self._try_real_engine()
        if self._engine is None:
            self._engine = MockFDIEngine(self.sim_frames, self.seed)
            self.is_mock = True
        self.n_frames = getattr(self._engine, "n_frames", self.sim_frames)

    def _try_real_engine(self):
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        try:
            import live_fault_lab as base
            engine, truth = base._build_engine(self.sim_frames, self.seed)
            # sanity: it must produce a state with the keys we rely on
            probe = engine.step(0)
            assert "fdi" in probe and "thresholds" in probe and "health" in probe
            self._engine, self._truth = engine, truth
            self.is_mock = False
            n = None
            try:
                n = int(truth["T"].size)
            except Exception:
                n = self.sim_frames
            self._engine.n_frames = n
            print("[fdi_gpu_data] using REAL FDI engine", flush=True)
        except Exception as e:
            print(f"[fdi_gpu_data] real engine unavailable ({type(e).__name__}); using MOCK campaign.",
                  flush=True)
            self._engine = None

    def step(self, fi):
        return self._engine.step(int(fi))


# ──────────────────────────────────────────────────────────────────────────────
# Extractors — turn a state dict into the arrays the panels plot.
# ──────────────────────────────────────────────────────────────────────────────
def cusum_matrix(st):
    """N×C array of S/threshold ratios (>=1 means tripped)."""
    thr = st["thresholds"]
    M = np.zeros((N_DEP, len(CHANNELS)), dtype=float)
    for d in range(N_DEP):
        Sd = st["fdi"][d]["S"]
        for j, c in enumerate(CHANNELS):
            h = thr.get(c, 1.0)
            M[d, j] = (Sd.get(c, 0.0) / h) if h else 0.0
    return M


def fleet_health(st):
    """List of (label, state_str, color) per deputy."""
    out = []
    for d in range(N_DEP):
        state = st["fdi"][d]["state"]
        out.append((f"D{d+1}", state, STATE_COLORS.get(state, "#cccccc")))
    return out


def confidence(st):
    return np.array([st["fdi"][d]["conf"] for d in range(N_DEP)], dtype=float)


def residuals(st, d):
    rd = st["fdi"][d]["r"]
    return np.array([rd.get(c, 0.0) for c in CHANNELS], dtype=float)


def cusum_ratios_for(st, d):
    thr = st["thresholds"]
    Sd = st["fdi"][d]["S"]
    return np.array([(Sd.get(c, 0.0) / thr.get(c, 1.0)) if thr.get(c, 1.0) else 0.0
                     for c in CHANNELS], dtype=float)


def worst_deputy(st):
    """Index of the deputy with the highest CUSUM ratio (for the focus panels)."""
    best, bi = -1.0, 0
    for d in range(N_DEP):
        m = max(cusum_ratios_for(st, d)) if st["fdi"][d]["state"] != "DEAD" else 1.31
        if m > best:
            best, bi = m, d
    return bi


def fleet_counts(st):
    h = [st["fdi"][d]["state"] for d in range(N_DEP)]
    return {
        "healthy": sum(s == "HEALTHY" for s in h),
        "suspect": sum(s == "SUSPECT" for s in h),
        "faulted": sum(s == "FAULTED" for s in h),
        "dead": sum(s == "DEAD" for s in h),
    }


if __name__ == "__main__":   # headless self-test (no Qt)
    src = FDISource(force_mock=True)
    print("mock source frames:", src.n_frames, "deputies:", src.N, "channels:", src.channels)
    for fi in (0, 220, 600, 999):
        st = src.step(fi)
        M = cusum_matrix(st)
        print(f"frame {fi:4d}  tripped={int((M>=1).sum())}  worst=D{worst_deputy(st)+1}  "
              f"counts={fleet_counts(st)}")
