# -*- coding: utf-8 -*-
"""
smoke_test.py - fast integration smoke tests for the ARISE re-docking package.

Runs the NumPy-only core (dynamics, RAFTI, envs, Phase-A, bridge, baseline) and,
if PyTorch is installed, the DDPG agent (forward + a real gradient step) and a
tiny end-to-end training/rollout.  Exit non-zero on first failure.

    python tests/smoke_test.py
"""
from __future__ import annotations
import os, sys, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = "PASS", "FAIL"
results = []

def check(name, fn):
    try:
        fn(); results.append((PASS, name)); print(f"[PASS] {name}")
    except Exception as e:
        results.append((FAIL, name)); print(f"[FAIL] {name}: {e}")
        traceback.print_exc()


# ---- NumPy-only core ----
def t_config():
    import config_redock as C
    r, u, rad = C.load_espa_ports()
    assert r.shape == (8, 3) and abs(rad - 0.5334) < 1e-3
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)

def t_hcw():
    from dynamics.hcw import HCWPropagator, hcw_stm
    p = HCWPropagator(1.027e-3, 2.0)
    s = p(np.zeros((3, 6)), np.zeros((3, 3)))
    assert s.shape == (3, 6)
    err = np.max(np.abs(hcw_stm(-2.0, 1e-3) @ hcw_stm(2.0, 1e-3) - np.eye(6)))
    assert err < 1e-9

def t_kinematics():
    from dynamics.basilisk_backend import lvlh_to_eci, eci_to_lvlh
    rr = np.array([7e6, 1e5, -2e5]); vv = np.array([-30., 7600., 12.])
    s = np.array([50., -80., 15., .03, -.02, .01])
    r, v = lvlh_to_eci(s, rr, vv); assert np.max(np.abs(eci_to_lvlh(r, v, rr, vv) - s)) < 1e-8

def t_rafti():
    import config_redock as C
    from envs.rafti import RaftiPorts
    r, u, rad = C.load_espa_ports(); ports = RaftiPorts(r, u, rad, C.RAFTI)
    a = ports.assign(ports.capture_pt + 50 * ports.dir_B); assert len(set(a)) == 8
    st = ports.capture_status(ports.capture_pt[0], -0.03 * ports.dir_B[0], 0)
    assert st["captured"]
    assert not ports.capture_status(ports.capture_pt[0], -0.3 * ports.dir_B[0], 0)["captured"]

def t_multi_env():
    from envs.multi_chaser_env import MultiChaserRedock
    e = MultiChaserRedock(seed=0); o, i = e.reset()
    assert o.shape == (96,) and e.ACTION_SIZE == 48   # 6-DOF: 12/deputy state, 6/deputy action
    o, r, d, tr, info = e.step(np.zeros(48)); assert "distances" in info

def t_single_env():
    from envs.single_dock_env import SingleDockEnv
    e = SingleDockEnv(seed=0); o = e.reset(); assert o.shape == (12,)   # 6-DOF
    e2 = SingleDockEnv(seed=0, goal_based=True); o2 = e2.reset()
    assert set(o2) == {"observation", "achieved_goal", "desired_goal"}

def t_phase_a():
    from guidance.phase_a_gnc_return import plan_return
    import config_redock as C
    X0 = np.zeros((8, 6)); X0[:, 0] = np.linspace(1000, 3000, 8); X0[:, 3] = -0.05
    res = plan_return(post_imaging_lvlh=X0, tof=6.0)
    assert res.dv_total_mps.shape == (8,) and np.all(res.dv_total_mps < 5.0)  # small, not km/s
    assert len(set(res.port_of)) == 8

def t_bridge():
    from bridge.redock_pipeline import run_redock
    res = run_redock(seed=0, dynamics="hcw")
    assert res.trajectory is not None and res.trajectory.shape[1] == 8

def t_lambert():
    from lambert_return import simulate
    res = simulate(); assert np.all(np.linalg.norm(res["history"][-1, :, :3], axis=1) < 1.0)

def t_geometry():
    import config_redock as C
    from envs.geometry import ESPACylinder, DeputyCuboid
    r, u, rad = C.load_espa_ports(); cyl = ESPACylinder(r, u); dep = DeputyCuboid()
    assert cyl.body_hits_shell(dep.corners_lvlh(np.zeros(3), np.eye(3)))       # centre = collide
    c = cyl.capture_point(0, 0.10 + dep.face_offset)
    R = dep.attitude_pointing(-cyl.port_normal(0))
    assert not cyl.body_hits_shell(dep.corners_lvlh(c, R))                     # docked = clear

def t_attitude():
    from dynamics.attitude import rk4_attitude, J_DEPUTY, pointing_error_deg, mrp_error_to_point, mrp_to_dcm
    # energy conservation (torque-free)
    s0, w0 = np.array([0.1, -0.2, 0.05]), np.array([0.02, -0.01, 0.03])
    s, w = s0.copy(), w0.copy(); E0 = 0.5 * (J_DEPUTY * w) @ w
    for _ in range(200): s, w = rk4_attitude(s, w, np.zeros(3), 1.0)
    assert abs(0.5 * (J_DEPUTY * w) @ w - E0) / E0 < 1e-3

def t_sixdof_capture():
    import config_redock as C
    from envs.rafti import RaftiPorts
    from dynamics.attitude import rk4_attitude, mrp_error_to_point, mrp_to_dcm
    r, u, rad = C.load_espa_ports()
    ports = RaftiPorts(r, u, rad, C.RAFTI, deputy_dims=C.ESPA["deputy_dims"], height=C.ESPA["height_m"])
    i = 0; pos = ports.capture_pt[i]; vel = -0.02 * ports.dir_B[i]; inward = -ports.dir_B[i]
    sg = np.zeros(3); om = np.zeros(3)
    for _ in range(80):
        eb = mrp_to_dcm(sg) @ mrp_error_to_point(sg, inward)
        sg, om = rk4_attitude(sg, om, np.clip(4000 * (np.clip(0.6 * eb, -0.05, 0.05) - om), -1, 1) * 1.2e-3, 1.0)
    assert ports.capture_status(pos, vel, i, sigma=sg, omega=om)["captured"]        # aligned -> capture
    assert not ports.capture_status(pos, vel, i, sigma=np.zeros(3), omega=np.zeros(3))["captured"]  # 180 off


# ---- torch-dependent ----
def t_agent_learn():
    import torch  # noqa
    from agents.ddpg import DDPGAgent
    ag = DDPGAgent(96, 48, batch_size=32, buffer_size=1000)
    for _ in range(60):
        s = np.random.randn(96).astype(np.float32); a = np.random.uniform(-1, 1, 48)
        ag.remember(s, a, -1.0, np.random.randn(96).astype(np.float32), False)
    a0 = ag.choose_action(np.random.randn(96), add_noise=False)
    out = ag.learn(); assert out is not None
    cl, al = out; assert np.isfinite(cl) and np.isfinite(al)
    assert a0.shape == (48,) and np.all(np.abs(a0) <= 1.0)

def t_train_tiny():
    import torch  # noqa
    from train.train_ddpg_torch import train
    ag = train(kind="single", episodes=8, curriculum=True,
               out="artifacts/_smoke", seed=0)
    assert os.path.exists("artifacts/_smoke/reward_history.npz")


check("config + real ESPA geometry", t_config)
check("HCW propagator + STM", t_hcw)
check("LVLH<->ECI kinematics", t_kinematics)
check("RAFTI ports + soft-capture", t_rafti)
check("multi-chaser env", t_multi_env)
check("single-dock env (+goal)", t_single_env)
check("Phase-A GNC return", t_phase_a)
check("end-to-end bridge", t_bridge)
def t_flight():
    import numpy as np, config_redock as C
    from envs.rafti import RaftiPorts
    from flight import config_flight as F
    from flight.gnc_loop import FlightGNC
    r, u, rad = C.load_espa_ports(); rcfg = dict(C.RAFTI); rcfg.update(F.CAPTURE_ENVELOPE)
    ports = RaftiPorts(r, u, rad, rcfg, deputy_dims=C.ESPA["deputy_dims"], height=C.ESPA["height_m"])
    r0 = ports.capture_pt[0] + 60 * ports.dir_B[0]
    s0 = np.concatenate([r0, -0.05 * ports.dir_B[0], 0.2 * np.random.default_rng(0).normal(size=3), np.zeros(3)])
    res = FlightGNC(ports, 0, agent=None, seed=0).run(s0)
    assert res["summary"]["final_phase"] in ("DOCKED", "SAFE_HOLD")   # dock or safe, never fail

check("Lambert baseline", t_lambert)
check("closed-loop flight GNC", t_flight)
check("cylinder+cuboid geometry", t_geometry)
check("attitude dynamics", t_attitude)
check("6-DOF capture (face+rate)", t_sixdof_capture)
try:
    import torch  # noqa
    check("DDPG agent forward+learn", t_agent_learn)
    check("tiny training run", t_train_tiny)
except ImportError:
    print("[SKIP] torch not installed - agent tests skipped")

n_fail = sum(1 for s, _ in results if s == FAIL)
print(f"\n=== {sum(1 for s,_ in results if s==PASS)} passed, {n_fail} failed ===")
sys.exit(1 if n_fail else 0)
