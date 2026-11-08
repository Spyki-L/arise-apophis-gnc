"""Relative-motion propagators for the terminal docking phase.

    hcw               - Clohessy-Wiltshire closed form (fast training surrogate).
    basilisk_backend  - hi-fi relative dynamics via Basilisk (flight fidelity),
                        import-guarded so the package imports without Basilisk.
    get_propagator(name) -> callable(states[N,6], dv[N,3], dt) -> states[N,6]
"""
from .hcw import hcw_step, hcw_stm, HCWPropagator
from .sixdof import SixDofPropagator, get_sixdof

def get_propagator(name: str, omega: float, dt: float):
    name = (name or "hcw").lower()
    if name == "hcw":
        return HCWPropagator(omega=omega, dt=dt)
    if name == "basilisk":
        from .basilisk_backend import BasiliskPropagator
        return BasiliskPropagator(omega=omega, dt=dt)
    raise ValueError(f"unknown dynamics backend: {name!r} (use 'hcw' or 'basilisk')")

__all__ = ["hcw_step", "hcw_stm", "HCWPropagator", "get_propagator",
           "SixDofPropagator", "get_sixdof"]
