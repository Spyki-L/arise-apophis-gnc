# -*- coding: utf-8 -*-
"""
sixdof.py - composite 6-DOF propagator for RAFTI docking.

State per body (12):  [ r(3), v(3), sigma(3), omega(3) ]   (LVLH; MRP body<-LVLH)
Action per body (6):  [ dv(3) impulsive LVLH,  tau(3) body torque ]

  mode "hcw"      : translation via Clohessy-Wiltshire STM, attitude via rigid-body RK4.
  mode "basilisk" : delegates to BasiliskPropagator.step6dof (flight fidelity).

CORRECTED: `att_substeps` is now configurable (was hard-coded 8).  At dt=1 s with a
tiny 6U torque the attitude barely rotates per step, so 2 substeps match 8 to <1e-4
rad while running ~4x faster -> materially more RL episodes per CPU-hour.
"""
from __future__ import annotations
import numpy as np
try:
    from .hcw import HCWPropagator
    from .attitude import rk4_attitude, J_DEPUTY
except ImportError:
    import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hcw import HCWPropagator
    from attitude import rk4_attitude, J_DEPUTY

DEFAULT_ATT_SUBSTEPS = 2


class SixDofPropagator:
    def __init__(self, mode: str, omega: float, dt: float, inertia=None,
                 chief_state_fn=None, t0: float = 0.0, att_substeps: int = DEFAULT_ATT_SUBSTEPS):
        m = (mode or "hcw").lower().strip()
        if m in ("basilik", "bsk", "basilisk_hifi", "hifi", "flight"): m = "basilisk"
        if m in ("cw", "surrogate", "fast"): m = "hcw"
        self.mode = m
        self.dt = float(dt); self.omega = float(omega)
        self.att_substeps = int(att_substeps)
        self.J = np.asarray(inertia if inertia is not None else J_DEPUTY, float)
        if self.mode == "hcw":
            self.trans = HCWPropagator(omega=omega, dt=dt)
            self._bsk = None
        elif self.mode == "basilisk":
            try:
                from .basilisk_backend import BasiliskPropagator
            except ImportError:
                from basilisk_backend import BasiliskPropagator
            self._bsk = BasiliskPropagator(omega=omega, dt=dt, chief_state_fn=chief_state_fn,
                                           t0=t0, inertia=self.J)
            self.trans = None
        else:
            raise ValueError(f"unknown 6-DOF mode {mode!r} (use 'hcw' or 'basilisk')")

    def __call__(self, states, actions, dt=None):
        dt = self.dt if dt is None else float(dt)
        S = np.atleast_2d(np.asarray(states, float)).copy()
        A = np.atleast_2d(np.asarray(actions, float))
        if self.mode == "basilisk":
            return self._bsk.step6dof(S, A, dt)
        rv = S[:, :6]; dv = A[:, :3]
        rv_new = self.trans(rv, dv, dt)
        out = S.copy(); out[:, :6] = rv_new
        for i in range(S.shape[0]):
            sig, w = rk4_attitude(S[i, 6:9], S[i, 9:12], A[i, 3:6], dt, self.J,
                                  substeps=self.att_substeps)
            out[i, 6:9] = sig; out[i, 9:12] = w
        return out


def get_sixdof(mode: str, omega: float, dt: float, inertia=None,
               chief_state_fn=None, t0: float = 0.0, att_substeps: int = DEFAULT_ATT_SUBSTEPS):
    return SixDofPropagator(mode, omega, dt, inertia=inertia, chief_state_fn=chief_state_fn,
                            t0=t0, att_substeps=att_substeps)


if __name__ == "__main__":
    prop = get_sixdof("hcw", omega=1.027e-3, dt=2.0)
    s = np.zeros((2, 12)); s[:, 0] = [40.0, 60.0]
    a = np.zeros((2, 6)); a[:, 0] = -0.02; a[:, 3] = 1e-3
    s1 = prop(s, a)
    print("dpos:", np.round(s1[:, :3] - s[:, :3], 4))
    assert np.all(np.abs(s1[:, 9]) > 0)
    print("sixdof self-test OK (att_substeps=%d)" % prop.att_substeps)
