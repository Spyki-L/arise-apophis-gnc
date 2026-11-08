"""
ARISE - RL Re-docking package
=============================

Two-phase autonomous return-and-dock of the 8 XB6 deputies onto the ESPA ring
after the Apophis stereo-imaging pass:

    Phase A  (far / mid field)  :  existing hi-fi GNC (Lambert seed + multiple-
                                   shooting TCM in Basilisk) drives each deputy
                                   from its post-imaging dispersal back into the
                                   ESPA capture envelope (~100 m, LVLH).
    ---- handoff  (ECI -> ESPA-LVLH relative state) ----
    Phase B  (terminal docking) :  RL DDPG(+HER) agent flies each deputy from the
                                   capture envelope onto its assigned Orbit Fab
                                   RAFTI port (cone + soft-capture + axis align),
                                   using the SAME ports it separated from.

Python conversion + hardening of the MATLAB DDPG swarm-return prototype, wired into the current Apophis scenario
(scripts/guidance/basilik_hifi_apophis_rendezvous_fleet.py).
"""
__version__ = "0.1.0"
__all__ = ["dynamics", "envs", "agents", "guidance", "bridge"]
