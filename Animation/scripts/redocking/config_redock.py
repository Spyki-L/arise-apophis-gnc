# -*- coding: utf-8 -*-
"""
config_redock.py - single source of truth for the ARISE RL re-docking module.
(CORRECTED: adds a REAL annealed curriculum, a rebalanced pure-state reward, and
TD3 hyper-parameters.  See HANDOFF/DIAGNOSIS for why the old run docked 0%.)
"""
from __future__ import annotations
import os, glob
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))

def _find_anim(start: str) -> str:
    d = start
    while os.path.basename(d) != "Animation":
        nd = os.path.dirname(d)
        if nd == d:
            return os.path.dirname(os.path.dirname(start))
        d = nd
    return d

ANIM_DIR  = _find_anim(_THIS)
REPO_ROOT = os.path.dirname(ANIM_DIR)
DATA_DIR  = os.path.join(REPO_ROOT, "Data")

ESPA_PORTS_NPZ_GLOBS = (
    os.path.join(ANIM_DIR, "Designs", "D_strobe_fullbody", "chief deputy seperation",
                 "outputs", "espa_xb6_8deps_*.npz"),
    os.path.join(DATA_DIR, "espa_xb6_8deps_*.npz"),
)

ART_DIR = os.environ.get("REDOCK_ART_DIR", os.path.join(_THIS, "artifacts"))

# ---------------------------------------------------------------- constants
MU_EARTH = 3.986004418e14
OMEGA_SURROGATE = 1.027e-3
N_DEPUTIES = 8

# ---------------------------------------------------------------- ESPA ports
ESPA_RADIUS_M = 0.5334
_PORTS_R_B_FALLBACK = np.array([
    [ 5.334000e-01,  0.000000e+00, 0.0],
    [ 3.771708e-01,  3.771708e-01, 0.0],
    [ 0.000000e+00,  5.334000e-01, 0.0],
    [-3.771708e-01,  3.771708e-01, 0.0],
    [-5.334000e-01,  0.000000e+00, 0.0],
    [-3.771708e-01, -3.771708e-01, 0.0],
    [ 0.000000e+00, -5.334000e-01, 0.0],
    [ 3.771708e-01, -3.771708e-01, 0.0],
], dtype=float)
_PORTS_DIR_B_FALLBACK = np.array([
    [ 1.0, 0.0, 0.0], [ 0.7071068, 0.7071068, 0.0], [ 0.0, 1.0, 0.0],
    [-0.7071068, 0.7071068, 0.0], [-1.0, 0.0, 0.0], [-0.7071068,-0.7071068, 0.0],
    [ 0.0,-1.0, 0.0], [ 0.7071068,-0.7071068, 0.0],
], dtype=float)


def load_espa_ports():
    for pattern in ESPA_PORTS_NPZ_GLOBS:
        hits = sorted(glob.glob(pattern))
        comb = [h for h in hits
                if not os.path.basename(h)[:-4].split("_")[-1].startswith("D")]
        cand = comb or hits
        for h in cand:
            try:
                d = np.load(h, allow_pickle=True)
                if "espa_ports_r_B" in d.files and "espa_ports_dir_B" in d.files:
                    r = np.asarray(d["espa_ports_r_B"], float)
                    u = np.asarray(d["espa_ports_dir_B"], float)
                    rad = float(d["espa_radius_m"]) if "espa_radius_m" in d.files \
                        else float(np.linalg.norm(r, axis=1).mean())
                    u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-15)
                    return r, u, rad
            except Exception:
                continue
    u = _PORTS_DIR_B_FALLBACK / np.linalg.norm(_PORTS_DIR_B_FALLBACK, axis=1, keepdims=True)
    return _PORTS_R_B_FALLBACK.copy(), u, ESPA_RADIUS_M

# ---------------------------------------------------------------- handoff
HANDOFF = dict(
    capture_radius_m   = 120.0,
    capture_speed_mps  = 0.25,
    envelope_min_m     = 30.0,
    envelope_max_m     = 120.0,
    init_speed_max_mps = 0.15,
)

# ---------------------------------------------------------------- RAFTI
RAFTI = dict(
    standoff_m          = 0.10,
    stage_range_m       = 6.0,
    capture_pos_tol_m   = 0.05,
    lateral_tol_m       = 0.04,
    cone_half_angle_deg = 12.0,
    v_capture_max_mps   = 0.05,
    v_lateral_max_mps   = 0.02,
    axis_align_tol_deg  = 15.0,
    att_align_tol_deg   = 10.0,
    omega_capture_max   = 0.03,
    keepout_radius_m    = 0.45,
)

# Deputy inertia [kg m^2].  Default = the ARISE separation-run principal J (mission-measured).
# nominal_v2 (modules/hardware.py XB6.inertia_diag) instead ASSUMES a uniform 6U box (36x24x11 cm)
# at 12.2 kg -> (0.07086, 0.14406, 0.19032).  Set ARISE_INERTIA=box for EXACT nominal_v2 parity
# (NB: the flight attitude PD gains Kp=0.02/Kd=0.10 were tuned to the separation-run value).
_DEPUTY_INERTIA_SEPRUN = (0.04218398, 0.20445541, 0.22767195)   # ARISE separation run (default)
_DEPUTY_INERTIA_BOX_V2 = (0.07086167, 0.14406167, 0.19032000)   # nominal_v2 uniform 6U box @ 12.2 kg
_DEPUTY_INERTIA = _DEPUTY_INERTIA_BOX_V2 if os.environ.get("ARISE_INERTIA","").lower() == "box" \
                  else _DEPUTY_INERTIA_SEPRUN
ESPA = dict(
    height_m    = 0.80,
    deputy_dims = (0.34, 0.20, 0.20),
    deputy_mass_kg = 12.2,                         # = nominal_v2 HW.XB6.wet_mass_kg (MATCH)
    deputy_inertia = _DEPUTY_INERTIA,
    deputy_inertia_box_v2 = _DEPUTY_INERTIA_BOX_V2,
)

# ---------------------------------------------------------------- ENV
ENV = dict(
    dt              = 1.0,
    max_steps       = 900,     # DEFAULT horizon; curriculum stages OVERRIDE per stage.  The old
                               # 200-step curriculum made capture UNREACHABLE (the verified PD
                               # times out ~1.7 m short at 200 steps but docks 8/8 at 700).
    att_substeps    = 2,       # attitude RK4 substeps/step (was 8; 2 is plenty at dt=1s, ~4x faster)
    max_dv_mps      = 0.05,
    bounds_radius_m = 200.0,
    max_torque_Nm   = 1.2e-3,
    dynamics        = "hcw",
    omega           = OMEGA_SURROGATE,
    collision_penalty = 50.0,
    success_bonus     = 200.0,
    dv_penalty_scale  = 2.0,
    dist_penalty_scale= 0.5,
    cone_funnel_scale = 0.15,
    brake_scale       = 0.3,
    point_funnel_scale= 0.02,
    rate_penalty_scale= 2.0,
)

# ---------------------------------------------------------------- REWARD (rebalanced)
# Pure function of the port-frame state -> IDENTICAL for real and HER-relabelled
# transitions (fixes the mixed dense/sparse critic corruption).  A successful dock
# is engineered to be the clear global optimum: bounded per-step shaping + one big bonus.
REWARD = dict(
    progress_scale   = 2.0,
    approach_cap_mps = 0.30,
    dist_shape_scale = 0.02,
    lateral_scale    = 0.05,
    att_shape_scale  = 0.01,
    rate_shape_scale = 0.5,
    brake_scale      = 0.5,
    dv_penalty_scale = 0.5,
    tau_penalty_scale= 2.0,
    step_penalty     = 0.02,
    success_bonus    = 200.0,
    collision_penalty= 50.0,
    per_step_clip    = 5.0,
)

# ---------------------------------------------------------------- CURRICULUM (annealed)
CURRICULUM_STAGES = [
    dict(name="s0_easy",   envelope=(2.0, 6.0),  init_v=0.05, steps=300,  pos=0.30, lat=0.20, v_cap=0.20, v_lat=0.12, cone=30.0, att=25.0, omega=0.12,  init_att=15.0, init_cone=25.0),
    dict(name="s1",        envelope=(4.0, 12.0), init_v=0.06, steps=450,  pos=0.25, lat=0.16, v_cap=0.15, v_lat=0.09, cone=25.0, att=20.0, omega=0.09,  init_att=35.0, init_cone=35.0),
    dict(name="s2",        envelope=(8.0, 20.0), init_v=0.08, steps=650,  pos=0.18, lat=0.12, v_cap=0.10, v_lat=0.06, cone=20.0, att=16.0, omega=0.07,  init_att=60.0, init_cone=40.0),
    dict(name="s3",        envelope=(12.0,30.0), init_v=0.10, steps=850,  pos=0.13, lat=0.09, v_cap=0.07, v_lat=0.04, cone=16.0, att=13.0, omega=0.06,  init_att=80.0, init_cone=45.0),
    dict(name="s4_flight", envelope=(15.0,40.0), init_v=0.12, steps=1000, pos=0.10, lat=0.08, v_cap=0.05, v_lat=0.03, cone=14.0, att=12.0, omega=0.055, init_att=90.0, init_cone=45.0),
]
CURRICULUM = dict(advance_at=0.60, window=100, min_eps=300, hold_last=True)


def stage_overrides(stage: dict):
    """Translate a CURRICULUM_STAGES entry into env_cfg/handoff/rafti overrides."""
    return dict(
        env_cfg=dict(max_steps=int(stage["steps"]), bounds_radius_m=200.0),
        handoff=dict(envelope_min_m=stage["envelope"][0], envelope_max_m=stage["envelope"][1],
                     init_speed_max_mps=stage["init_v"],
                     init_att_err_deg=stage["init_att"], init_cone_deg=stage["init_cone"]),
        rafti=dict(capture_pos_tol_m=stage["pos"], lateral_tol_m=stage["lat"],
                   v_capture_max_mps=stage["v_cap"], v_lateral_max_mps=stage["v_lat"],
                   cone_half_angle_deg=stage["cone"], att_align_tol_deg=stage["att"],
                   omega_capture_max=stage["omega"]),
    )

# ---------------------------------------------------------------- DDPG (legacy)
DDPG = dict(
    alpha=1e-4, beta=1e-3, gamma=0.99, tau=1e-3, layer1_size=400, layer2_size=300,
    batch_size=256, buffer_size=1_000_000, ou_theta=0.15, ou_sigma=0.20, ou_dt=1e-2,
    warmup_steps=600, her_enabled=True, her_k=4, her_strategy="future",
)

# ---------------------------------------------------------------- TD3 (corrected trainer)
TD3 = dict(
    alpha=1e-4, beta=1e-3, gamma=0.99, tau=5e-3, layer1_size=400, layer2_size=300,
    batch_size=256, buffer_size=1_000_000,
    policy_delay=2, target_noise=0.2, noise_clip=0.5,
    expl_noise=0.2, expl_noise_final=0.05,
    warmup_steps=5000, updates_per_step=1, her_k=4,
)


def ensure_art_dir():
    os.makedirs(ART_DIR, exist_ok=True)
    return ART_DIR


if __name__ == "__main__":
    r, u, rad = load_espa_ports()
    print(f"ESPA ring radius = {rad:.4f} m, {r.shape[0]} RAFTI ports")
    print("curriculum stages:", [s["name"] for s in CURRICULUM_STAGES])
    print("TD3 policy_delay:", TD3["policy_delay"], "| ENV att_substeps:", ENV["att_substeps"])
