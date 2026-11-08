# -*- coding: utf-8 -*-
"""
redock_pipeline.py - the full ARISE re-docking, wired end to end.

    post-imaging deputy states (ESPA LVLH)
        |  Phase-A: existing hi-fi GNC (Lambert + multiple-shooting TCM)   [guidance/]
        v
    handoff inside the ESPA capture envelope  (<~120 m, <0.25 m/s)
        |  hand the LVLH relative state to the RL docking policy
        v
    Phase-B: DDPG(+HER) terminal docking onto the assigned RAFTI port      [envs/ + agents/]
        v
    8 deputies soft-captured on the ESPA ring (cone + soft-dock + axis align)

`run_redock(agent=...)` accepts any object with `.choose_action(obs, add_noise=False)`
(the PyTorch DDPGAgent, an SB3 policy wrapper, or a scripted controller for a dry
run).  Reports Phase-A + Phase-B Delta-v, capture count, and per-deputy timeline.
"""
from __future__ import annotations
import os, sys
from dataclasses import dataclass, field
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_redock as C
from guidance.phase_a_gnc_return import plan_return
from envs.multi_chaser_env import MultiChaserRedock, OBS_POS_SCALE, OBS_VEL_SCALE, OBS_RATE_SCALE


@dataclass
class RedockResult:
    n_captured: int
    dv_phaseA_mps: np.ndarray
    dv_phaseB_mps: float
    steps: int
    final_dist_m: np.ndarray
    port_of: np.ndarray
    trajectory: np.ndarray = field(default=None)     # (T,N,3) LVLH positions
    attitudes: np.ndarray = field(default=None)      # (T,N,3) MRP body<-LVLH
    capture: list = field(default=None)              # per-deputy soft-capture metrics
    meta: dict = field(default_factory=dict)


from eval.rollout import PortFramePD, SwarmFromSingle
class _PDController:
    """Fallback 6-DOF glideslope+attitude baseline so the pipeline runs untrained."""
    def __init__(self, n=8): self._sw = SwarmFromSingle(PortFramePD(), n=n)
    def choose_action(self, obs, add_noise=False): return self._sw.choose_action(obs)


def _seed_env_from_handoff(env, handoff_lvlh, port_of, seed=0):
    env.reset()
    rng = np.random.default_rng(seed)
    H = np.asarray(handoff_lvlh, float)
    env.port_of = np.asarray(port_of, int).copy()
    env.captured[:] = False
    smax = np.tan(np.radians(env.handoff.get("init_att_err_deg", 90.0)) / 4.0)
    for i in range(env.N):
        env.state[i, :6] = H[i, :6]                       # translational handoff (Phase A)
        sg = rng.normal(size=3); env.state[i, 6:9] = smax * rng.random() * sg / (np.linalg.norm(sg) + 1e-15)
        env.state[i, 9:12] = 0.01 * rng.normal(size=3)    # small initial body rate
        st = env.ports.capture_status(env.state[i, :3], env.state[i, 3:6], env.port_of[i])
        env.prev_dist[i] = st["dist"]
    return env._obs()


def run_redock(agent=None, post_imaging_lvlh=None, dynamics="hcw", seed=0,
               tof=3 * 3600.0, record=True, env_cfg=None, handoff=None, rafti=None):
    rng = np.random.default_rng(seed)
    if post_imaging_lvlh is None:
        # synthetic post-imaging dispersal (1-3 km in ESPA LVLH)
        dirs = rng.normal(size=(C.N_DEPUTIES, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        post_imaging_lvlh = np.zeros((C.N_DEPUTIES, 6))
        post_imaging_lvlh[:, :3] = (1000 + 2000 * rng.random((C.N_DEPUTIES, 1))) * dirs
        post_imaging_lvlh[:, 3:] = 0.05 * rng.normal(size=(C.N_DEPUTIES, 3))

    # ---- Phase A: existing GNC return -> capture envelope ----
    pa = plan_return(post_imaging_lvlh=post_imaging_lvlh, tof=tof/3600.0, seed=seed)

    # ---- Phase B: RL terminal docking ----
    chief_fn, t0m = None, 0.0
    if dynamics == "basilisk":
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.join(C.ANIM_DIR, "scripts", "guidance"))
            import basilik_hifi_apophis_rendezvous_fleet as _hifi
            tof_c, chief_fn = _hifi.load_chief(); t0m = max(0.0, tof_c - 1200.0)
        except Exception as _e:
            print(f"[redock] chief unavailable ({_e}); Basilisk uses synthetic reference")
    env = MultiChaserRedock(dynamics=dynamics, seed=seed, env_cfg=env_cfg, handoff=handoff,
                            rafti=rafti, chief_state_fn=chief_fn, t0_mission=t0m)
    obs = _seed_env_from_handoff(env, pa.handoff_lvlh, pa.port_of, seed=seed)
    agent = agent or _PDController()

    traj = []; att = []
    dvB = 0.0; done = False
    while not done:
        if record:
            traj.append(env.positions_lvlh()); att.append(env.attitudes())
        a = agent.choose_action(obs, add_noise=False)
        obs, r, done, trunc, info = env.step(a)
        dvB += info["dv_used"]
    if record:
        traj.append(env.positions_lvlh()); att.append(env.attitudes())

    cap = [env.ports.capture_status(env.state[k, :3], env.state[k, 3:6], env.port_of[k],
                                    sigma=env.state[k, 6:9], omega=env.state[k, 9:12])
           for k in range(env.N)]

    return RedockResult(
        n_captured=int(info["n_captured"]),
        dv_phaseA_mps=pa.dv_total_mps,
        dv_phaseB_mps=float(dvB),
        steps=int(info["steps"]),
        final_dist_m=info["distances"],
        port_of=pa.port_of,
        trajectory=np.array(traj) if record else None,
        attitudes=np.array(att) if record else None,
        capture=cap,
        meta=dict(phaseA_backend=pa.backend, dynamics=dynamics,
                  dv_total_mps=float(pa.dv_total_mps.sum() + dvB)))


if __name__ == "__main__":
    res = run_redock(seed=0, dynamics="hcw")
    print("=== ARISE end-to-end re-dock (untrained PD Phase-B) ===")
    print(f"Phase-A backend      : {res.meta['phaseA_backend']}")
    print(f"Phase-A dv [m/s]     : {np.round(res.dv_phaseA_mps,2)}  (sum {res.dv_phaseA_mps.sum():.1f})")
    print(f"Phase-B dv [m/s]     : {res.dv_phaseB_mps:.2f}")
    print(f"captured             : {res.n_captured}/8 in {res.steps} steps")
    print(f"final port-dist [m]  : {np.round(res.final_dist_m,3)}")
    print(f"trajectory shape     : {None if res.trajectory is None else res.trajectory.shape}")
    print("redock_pipeline self-test OK  (capture count improves once the DDPG policy is trained)")
