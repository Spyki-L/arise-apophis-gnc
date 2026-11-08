# -*- coding: utf-8 -*-
"""
lambert_return.py - classical (non-RL) swarm return baseline.

Python port of eight_deputies_hcw_return.m: return 8 deputies to the ESPA (Hill-
frame origin) with a single impulsive burn, then coast under closed-form
Clohessy-Wiltshire dynamics.  The MATLAB script solved Lambert in the inertial
frame and mapped the burn into the Hill frame; the equivalent (and cleaner)
relative-frame formulation solves the CW two-point boundary value problem
directly: pick the initial velocity that drives the deputy from x0 to the origin
in time T, i.e.  v0 = -Phi_rr(T)^{-1} ... via the CW state-transition matrix.

This is the deterministic baseline the RL docking policy is compared against
(the RL adds soft-capture cone/velocity constraints the single burn ignores).
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamics.hcw import hcw_stm
import config_redock as C

# same 8 relative positions used by the MATLAB demo (km -> m)
DEPUTY_POS_KM = np.array([
    [ 1.43,  2.55, -0.27], [-3.07, -0.48,  0.09], [ 2.39, -1.76,  0.65],
    [ 2.79,  1.18, -0.41], [-2.95, -0.72, -0.71], [-0.94,  1.83,  1.95],
    [ 1.55, -2.45,  0.68], [-0.71,  1.36, -2.30]])


def cw_return_burn(x0, T, w):
    """Impulsive dv at t0 that sends relative pos x0[:3] to the origin in time T."""
    Phi = hcw_stm(T, w); Prr, Prv = Phi[:3, :3], Phi[:3, 3:]
    r0, v0 = x0[:3], x0[3:]
    v_req = np.linalg.solve(Prv, -Prr @ r0)     # r(T)=0  ->  v0_req
    return v_req - v0, v_req


def simulate(positions_m=None, T=3600.0, dt=60.0, w=None):
    w = float(w if w is not None else C.OMEGA_SURROGATE)
    P = DEPUTY_POS_KM * 1000.0 if positions_m is None else np.asarray(positions_m, float)
    N = P.shape[0]
    states = np.zeros((N, 6)); states[:, :3] = P
    dvs = np.zeros((N, 3))
    for i in range(N):
        dv, v_req = cw_return_burn(states[i], T, w)
        states[i, 3:] = v_req; dvs[i] = dv
    nsteps = int(T / dt)
    hist = np.zeros((nsteps + 1, N, 6)); hist[0] = states
    Phi = hcw_stm(dt, w)
    for k in range(nsteps):
        states = states @ Phi.T
        hist[k + 1] = states
    return dict(history=hist, dv=dvs, dv_total=np.linalg.norm(dvs, axis=1))


if __name__ == "__main__":
    res = simulate()
    final = np.linalg.norm(res["history"][-1, :, :3], axis=1)
    print("Classical CW-Lambert return baseline (port of eight_deputies_hcw_return.m)")
    print("per-deputy |dv| [m/s]:", np.round(res["dv_total"], 3))
    print("total dv [m/s]:", round(res["dv_total"].sum(), 3))
    print("final miss to origin [m]:", np.round(final, 3))
    assert np.all(final < 1.0), "single-burn CW return should reach the origin"
    print("lambert_return self-test OK")
