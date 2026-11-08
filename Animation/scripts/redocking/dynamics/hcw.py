# -*- coding: utf-8 -*-
"""
hcw.py - Clohessy-Wiltshire (Hill) closed-form relative-motion propagator.

Vectorised NumPy port of  propagateHCW_closedForm()  /  MultiChaserEnv.propagateHCW
from the MATLAB prototype.  Propagates a batch of relative states in the LVLH /
Hill frame of a circular reference orbit of mean motion `omega`.

State convention (per body, SI):  s = [x, y, z, vx, vy, vz]
    x - radial (R),  y - along-track (T),  z - cross-track (N).

This is the fast SURROGATE used to train the RL docking agent.  It is exact for a
circular reference orbit; during the 2029 Apophis flyby the chief is hyperbolic
wrt Earth, so flight-fidelity roll-outs should use the Basilisk backend instead
(see dynamics/basilisk_backend.py and the note in config_redock.OMEGA_SURROGATE).
"""
from __future__ import annotations
import numpy as np


def hcw_stm(dt: float, w: float) -> np.ndarray:
    """6x6 Clohessy-Wiltshire state-transition matrix for step `dt`, mean motion `w`."""
    s, c = np.sin(w * dt), np.cos(w * dt)
    Phi = np.zeros((6, 6))
    # position rows
    Phi[0, 0] = 4 - 3 * c
    Phi[0, 3] = s / w
    Phi[0, 4] = 2 * (1 - c) / w
    Phi[1, 0] = 6 * (s - w * dt)
    Phi[1, 1] = 1.0
    Phi[1, 3] = -2 * (1 - c) / w
    Phi[1, 4] = (4 * s) / w - 3 * dt
    Phi[2, 2] = c
    Phi[2, 5] = s / w
    # velocity rows
    Phi[3, 0] = 3 * w * s
    Phi[3, 3] = c
    Phi[3, 4] = 2 * s
    Phi[4, 0] = -6 * w * (1 - c)
    Phi[4, 3] = -2 * s
    Phi[4, 4] = 4 * c - 3
    Phi[5, 2] = -w * s
    Phi[5, 5] = c
    return Phi


def hcw_step(states: np.ndarray, dt: float, w: float) -> np.ndarray:
    """Propagate a batch of states one step.

    Parameters
    ----------
    states : (N,6) or (6,) array of [x,y,z,vx,vy,vz]
    dt     : step [s]
    w      : mean motion [rad/s]

    Returns
    -------
    (N,6) or (6,) propagated states.
    """
    single = states.ndim == 1
    S = np.atleast_2d(states).astype(float)          # (N,6)
    Phi = hcw_stm(dt, w)                               # (6,6)
    out = S @ Phi.T                                    # (N,6)
    return out[0] if single else out


class HCWPropagator:
    """Callable propagator with an impulsive Delta-v applied before each step.

    prop(states[N,6], dv[N,3], dt=None) -> states[N,6]
    Matches the MATLAB env.step(): v <- v + dv, then closed-form CW propagate.
    """
    name = "hcw"

    def __init__(self, omega: float, dt: float):
        self.omega = float(omega)
        self.dt = float(dt)
        self._Phi = hcw_stm(self.dt, self.omega)

    def __call__(self, states: np.ndarray, dv: np.ndarray | None = None,
                 dt: float | None = None) -> np.ndarray:
        S = np.atleast_2d(states).astype(float).copy()   # (N,6)
        if dv is not None:
            S[:, 3:6] += np.atleast_2d(dv)                # impulsive burn
        if dt is None or dt == self.dt:
            Phi = self._Phi
        else:
            Phi = hcw_stm(dt, self.omega)
        return S @ Phi.T


if __name__ == "__main__":
    # Self-test: a body released at rest on the R axis must follow the analytic
    # CW drift, and the STM must be symplectic-consistent (round-trip identity).
    w = 1.027e-3
    prop = HCWPropagator(omega=w, dt=2.0)
    s0 = np.array([[50.0, 0, 0, 0, 0, 0],
                   [0, 80.0, 0, 0, 0, 0]])
    s1 = prop(s0, dv=np.zeros((2, 3)))
    print("one step from rest:\n", np.round(s1, 6))
    # round-trip: forward dt then backward dt returns start
    Ff = hcw_stm(2.0, w); Fb = hcw_stm(-2.0, w)
    err = np.max(np.abs(Fb @ Ff - np.eye(6)))
    print(f"STM round-trip identity error = {err:.2e}")
    assert err < 1e-9, "STM round-trip failed"
    print("hcw self-test OK")
