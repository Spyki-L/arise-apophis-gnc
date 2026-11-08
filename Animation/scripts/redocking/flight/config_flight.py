# -*- coding: utf-8 -*-
"""
config_flight.py - FROZEN flight parameter database for the re-docking GNC.

Single source of truth for every limit, safety threshold, sensor spec and
actuator spec used by the flight stack.  Values are flight-representative for a
6U XB6-class deputy with cold-gas RCS and a vision-based relative-nav sensor.
Everything is SI.  Treat this file as configuration-managed (version + freeze).
"""
from __future__ import annotations
import os
import numpy as np

# Trained terminal-docking policy for the flight loop (auto-detected, zero-arg).
# Drop the best HER/single checkpoint here (or set ARISE_FLIGHT_POLICY) and
# run_flight.py / montecarlo.py use it automatically; else the verified fallback runs.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_POLICY_CKPT = os.environ.get("ARISE_FLIGHT_POLICY",
                                os.path.join(_PKG_DIR, "artifacts", "flight_policy.pt"))

def rl_policy_available():
    return bool(RL_POLICY_CKPT) and os.path.exists(RL_POLICY_CKPT)

CONFIG_VERSION = "1.0.0"
CONFIG_FROZEN  = True          # set False only in a dev branch

# --------------------------------------------------------------------------
# Control / timing
# --------------------------------------------------------------------------
GNC = dict(
    dt              = 1.0,      # [s] flight GNC control step (matches 6-DOF env)
    max_steps       = 900,      # [-] terminal-phase horizon guard
    approach_speed_max = 0.30,  # [m/s] commanded closing-speed cap (far)
    approach_speed_terminal = 0.05,  # [m/s] cap inside the terminal ball
)

# --------------------------------------------------------------------------
# Safety limits (RTA / FDIR enforce these)  -- flight requirements
# --------------------------------------------------------------------------
SAFETY = dict(
    keepout_radius_m     = 0.55,   # ESPA shell + margin (deputy body kept outside)
    cone_half_angle_deg  = 15.0,   # approach corridor half-angle
    v_approach_max_mps   = 0.30,   # hard cap on relative speed anywhere
    v_contact_max_mps    = 0.05,   # soft-capture contact speed limit (axial)
    v_lateral_max_mps    = 0.02,   # lateral speed limit at contact
    omega_max_radps      = 0.05,   # body-rate hard cap
    omega_contact_max    = 0.03,   # body rate at contact
    abort_range_m        = 150.0,  # if range exceeds this after handoff -> abort/reacquire
    collision_margin_m   = 0.05,   # RTA barrier margin off the shell
)

# --------------------------------------------------------------------------
# Sensors
# --------------------------------------------------------------------------
SENSORS = dict(
    # vision-based relative navigation (bearing always; range when close)
    vbn_bearing_sigma_rad = 1.0e-3,     # ~0.057 deg 1-sigma per axis
    vbn_bearing_bias_rad  = 3.0e-4,
    vbn_range_sigma_frac  = 0.03,       # 3% of range 1-sigma
    vbn_range_floor_m     = 0.02,       # additive floor
    vbn_range_max_m       = 200.0,      # range channel valid below this
    vbn_latency_steps     = 1,          # measurement latency (control steps)
    vbn_dropout_prob      = 0.02,       # per-step measurement dropout
    # star tracker (attitude, MRP body<-LVLH)
    st_sigma_rad          = 1.7e-4,     # ~0.01 deg 1-sigma
    st_dropout_prob       = 0.01,
    # rate gyro
    gyro_sigma_radps      = 1.0e-4,
    gyro_bias_radps       = 1.0e-5,     # slowly-varying bias
    gyro_bias_tau_s       = 300.0,
)

# --------------------------------------------------------------------------
# Actuators  (cold-gas RCS)
# --------------------------------------------------------------------------
ACTUATORS = dict(
    dv_min_impulse_mps    = 5.0e-4,     # precision-RCS min impulse bit (0.5 mm/s)
    dv_max_mps            = 0.05,       # max Delta-v per axis per step (authority)
    tau_min_impulse_Nm    = 1.0e-6,     # torque quantum (REACTION WHEEL: fine, continuous)
    tau_max_Nm            = 1.5e-3,     # max reaction-wheel torque per axis
    misalign_sigma_deg    = 1.0,        # thruster pointing misalignment 1-sigma
    scale_sigma_frac      = 0.03,       # thrust magnitude dispersion 1-sigma
    latency_steps         = 1,          # command-to-thrust latency
    isp_s                 = 60.0,       # cold-gas specific impulse
    prop_mass_kg          = 0.30,       # usable propellant per deputy
    deputy_mass_kg        = 12.2,
)

# RAFTI soft-capture envelope the mechanism provides (GNC delivers within this;
# the latches/petals do the final mm-level alignment) - looser than the aim point.
CAPTURE_ENVELOPE = dict(capture_pos_tol_m=0.10, lateral_tol_m=0.08, v_capture_max_mps=0.05,
                        v_lateral_max_mps=0.03, cone_half_angle_deg=14.0,
                        att_align_tol_deg=12.0, omega_capture_max=0.055)

ATT_CTRL = dict(Kp=0.02, Kd=0.10)   # verified full-3-DOF attitude PD (reaction wheels)

# --------------------------------------------------------------------------
# FDIR thresholds
# --------------------------------------------------------------------------
FDIR = dict(comm_loss_steps=20, 
    nav_innov_sigma_gate  = 5.0,        # innovation gate (sigmas) for measurement reject
    nav_cov_diverge_m     = 50.0,       # position 1-sigma above which nav is 'diverged'
    thruster_stuck_steps  = 5,          # consecutive off-nominal responses -> fault
    corridor_breach_steps = 3,          # consecutive cone breaches near port -> abort
    overspeed_factor      = 1.5,        # v > factor*limit near port -> abort
    max_reacquire         = 2,          # abort/reacquire attempts before SAFE_HOLD
    dv_budget_frac_abort  = 0.85,       # abort if propellant use exceeds this frac
)

# --------------------------------------------------------------------------
# Guidance handoff (from Phase-A) — capture envelope
# --------------------------------------------------------------------------
HANDOFF = dict(
    capture_radius_m   = 120.0,
    capture_speed_mps  = 0.25,
    init_cone_deg      = 45.0,
    init_att_err_deg   = 90.0,
)


NAV = dict(
    # RelativeNavEKF process noise + optional LVLH acceleration-bias augmentation.
    # On the HCW surrogate a_bias -> 0 (reduces to the CW EKF); on Basilisk it captures the
    # deputy's un-modelled J2 / third-body / SRP acceleration (carried vs a Keplerian chief),
    # so the estimate TRACKS truth instead of confidently following a wrong CW propagation.
    estimate_accel_bias = True,
    q_pos               = 1.0e-4,
    q_vel               = 1.0e-6,
    q_accel_bias        = 3.0e-6,   # random-walk on the LVLH accel-bias state
    accel_bias0_sigma   = 3.0e-2,   # initial 1-sigma on the un-modelled accel [m/s^2]
    accel_feedforward   = False,    # impulsive FF can't cancel a CONTINUOUS disturbance's offset -> use INTEGRAL
    accel_ff_gain       = 1.0,
)

INTEGRAL = dict(
    # Guidance integral action.  A constant perturbation (Basilisk J2/SRP) causes a
    # STEADY-STATE port-frame offset under proportional control (x_ss ~ b/kp) that blocks
    # capture; an integrator nulls it.  Anti-windup caps |ki*I| at clip_mps.
    enabled  = True,
    ki       = 3.0e-3,
    clip_mps = 0.03,
)


def summary():
    return (f"flight config v{CONFIG_VERSION} frozen={CONFIG_FROZEN}: "
            f"keepout {SAFETY['keepout_radius_m']} m, cone {SAFETY['cone_half_angle_deg']} deg, "
            f"v_contact {SAFETY['v_contact_max_mps']} m/s, RCS dv_max {ACTUATORS['dv_max_mps']} m/s")


if __name__ == "__main__":
    print(summary())
    assert CONFIG_FROZEN, "flight config must be frozen for a flight build"
    print("config_flight OK")
