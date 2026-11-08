# ARISE Re-docking — Architecture, Integration & Fidelity

This document explains how the deputies return to the ESPA and soft-dock onto their
Orbit Fab RAFTI ports, how it is wired into the existing Apophis GNC, and what is
faithful to the source vs. improved.

---

## 1. Mission context and the problem

At closest approach the eight XB6 (6U) deputies sit at their assigned imaging
stations in the **Apophis LVLH frame [R, T, N]**, driven there by
`scripts/guidance/basilik_hifi_apophis_rendezvous_fleet.py` (Lambert seed +
multiple-shooting TCM, Basilisk hi-fi force model). After imaging they must return
to the ESPA/chief and **re-dock**. Two regimes with very different requirements:

- **Far / mid field (km → ~100 m):** a fuel-optimal guidance problem. Deterministic,
  well-served by the Lambert + differential-correction machinery you already have.
- **Terminal docking (~100 m → contact):** a *constraint-satisfaction* problem —
  arrive on the port's approach axis, inside a shallow cone, below the soft-capture
  contact velocity, with the bus aligned. Non-convex, contact-sensitive; a natural
  fit for a learned policy. This is what the source MATLAB DDPG prototype targeted.

So the design is **two-phase**, splitting the mission at the boundary where the
cost function changes character.

---

## 2. Phase A — return with the *existing* GNC (targets reversed)

Key insight: the return leg is your **deployment guidance run backwards**. Outbound,
`basilik_hifi_apophis_rendezvous_fleet.py` drives ESPA → distinct targets in the
Apophis LVLH frame. Inbound, we keep the identical method (universal-variable
Lambert seed → multiple-shooting TCM differential corrector under the Basilisk
force model) but swap the targets to **distinct capture-envelope aim points in the
ESPA/chief LVLH frame**.

`guidance/phase_a_gnc_return.py` therefore *reuses* that module directly when it is
importable (Basilisk machine): its `lambert_uv`, `propagate`, and TCM corrector.
Off-Basilisk it falls back to a self-contained CW/Lambert relative-targeting solve
so the pipeline still runs offline. Output = one **handoff state per deputy** inside
the capture envelope (default: within 120 m of the ESPA, below 0.25 m/s relative),
plus the Phase-A Δv budget.

Assignment of deputies → ports is solved once, up front, by a Hungarian match
(minimum total distance to the capture points) so the two phases agree on which
deputy goes to which RAFTI port.

---

## 3. The handoff (ECI ↔ ESPA-LVLH)

Phase A works in ECI (Basilisk force model); Phase B works in the ESPA-centred LVLH
frame. `dynamics/basilisk_backend.py` provides the exact kinematic bridge —
`eci_to_lvlh` / `lvlh_to_eci` using the same LVLH construction as `dcm_lvlh()` in the
fleet script (R = r̂, N = (r×v)̂, T = N×R, with the rotating-frame velocity term
ω×r). These are pure-NumPy and unit-tested (round-trip < 1e-8).

---

## 4. Phase B — RL terminal docking onto RAFTI

### 4.1 RAFTI docking on the ESPA cylinder (`envs/geometry.py`, `envs/rafti.py`)
The ESPA is modelled as a finite **cylindrical shell** (`ESPACylinder`): axis =
the normal of the port plane (SVD of the real `espa_ports_r_B`), radius = the port
ring radius (0.5334 m), height 0.80 m. The 8 RAFTI ports sit on the **lateral
surface**, each facing radially **outward** (`espa_ports_dir_B`). Deputies are
**cuboids** (`DeputyCuboid`, 6U bus 0.34×0.20×0.20 m) whose +x face carries the
RAFTI adaptor. For each port we build:
- a **capture point** for the deputy CENTRE offset outboard by (standoff +
  half-length) so the docking FACE — not the body centre — meets the surface;
- a **staging point** ~6 m out on the port axis (corridor acquisition);
- a **port frame** (dock axis → local +x);
- a soft-capture test: 3-D miss ≤ 5 cm, lateral ≤ 4 cm, inside a 12° glideslope
  **cone with a capture ball at the apex**, axial contact speed within the
  **soft-capture** limit (≤ 5 cm/s closing), lateral ≤ 2 cm/s, and alignment ≤ 15°
  (auto-satisfied when the deputy arrives nearly stopped);
- **collision** = the oriented deputy cuboid penetrating the shell wall (corner
  test, so a proper dock at the surface is allowed but driving through the drum is
  not) **plus** deputy-vs-deputy overlap. The old keep-out sphere is superseded.

The deputy attitude is slaved to point its dock face at the port (`attitude_pointing`);
full 6-DOF attitude RL is a documented extension (the SE(3) separation controller
already provides the attitude capability).

### 4.2 The canonicalisation trick
Every deputy's LVLH state is expressed in **its port frame** before the policy sees
it. Result: the eight docking problems become **eight identical “dock along +x”
problems**. Two payoffs:
1. the centralised 48→24 policy (faithful to the MATLAB net sizes) sees a
   symmetric problem it can actually generalise over;
2. a **single 6→3 policy** trained on one port applies unchanged to all eight
   deputies (`eval/rollout.SwarmFromSingle`) — far more sample-efficient, and the
   natural home for HER.
The internal dynamics stay in LVLH; only the agent's I/O is canonicalised.

### 4.3 Reward
Potential-based progress to the capture point + a persistent distance pull + a
terminal speed penalty (brake near the port) + a fuel (Δv) penalty + corridor
penalty, a one-time **soft-capture bonus** (+200, carried over from
`MultiChaserEnv.m`), and collision / fly-away penalties with episode termination.
The shaping is deliberately aligned with a model-based PD terminal law (see §6), so
the optimum is reachable.

### 4.4 Environments
- `envs/multi_chaser_env.py` — `MultiChaserRedock`, the centralised 8-deputy 6-DOF
  MDP (96-D obs, 48-D action), pure-NumPy core + a `gymnasium` wrapper. Direct
  descendant of `MultiChaserEnv.m`, upgraded from "reach 15 m of the ring centre"
  to real per-port soft-capture.
- `envs/single_dock_env.py` — `SingleDockEnv`, the 12→6 canonical single-dock task,
  with a goal-based (`Dict` obs) mode for **HER** and SB3's `HerReplayBuffer`.

---

## 5. The RL agent — faithful port + the missing back-prop

The MATLAB agent is ported class-for-class: `ActorNetwork`, `CriticNetwork`,
`OUActionNoise`, `ReplayBuffer`, `DDPGAgent` → `agents/`. **Critical fix:** the
MATLAB `DDPGAgent.learn()` contained only `% (Placeholder: you would do backprop
here.)` — the networks had a forward pass but **no gradient step**, so nothing
actually trained; only the Polyak target copy ran. `agents/ddpg.py` implements the
full DDPG update (Lillicrap et al. 2016): critic MSE to the bootstrapped target,
actor policy-gradient (−Q(s,μ(s))), Adam, soft target updates, with gradient
clipping for stability. `agents/replay_buffer.py` also adds the **HER** buffer the
source RESULTS imply ("DDPG + HER") but the MATLAB code never had.

Two training entry points satisfy "both PyTorch and SB3":
- `train/train_ddpg_torch.py` — the faithful PyTorch DDPG(+HER);
- `train/train_sb3.py` — SB3 TD3/DDPG + `HerReplayBuffer` over the same envs.

---

## 6. Dynamics fidelity — and an honest note on CW

The env backend is pluggable (`dynamics/get_propagator`):
- **HCW** (`dynamics/hcw.py`) — Clohessy-Wiltshire closed form, exact for a circular
  reference orbit. Fast; used for training.
- **Basilisk** (`dynamics/basilisk_backend.py`) — Earth spherical harmonics + Sun +
  Moon + cannonball SRP via live SPICE + RKF45, the **same force model as Phase A**,
  import-guarded (run on a Basilisk machine).

Important physical caveat, consistent with your own critique of the legacy fleet
script: during the 2029 encounter the chief is on a **geocentric hyperbolic flyby**,
so a circular-orbit mean motion is only a **local surrogate**. At the ~100 m
terminal scale over a few minutes the relative motion is nearly rectilinear (tidal
terms are tiny), so HCW is a sound *training* surrogate — but flight validation and
the final policy roll-out should use the Basilisk backend. The agent and reward are
identical across backends, so this is a config switch, not a re-write.

---

## 7. Results

**End-to-end pipeline (model-based glideslope baseline).** `eval/docking_demo.py`
runs the full chain — Phase-A GNC return → LVLH handoff → Phase-B glideslope
docking (acquire the radial corridor, then run in) — standing in for the policy:
**8/8 deputies soft-captured** on their RAFTI ports at realistic tolerances (final
miss ≈ 0.05 m, no shell penetration), Phase-A ≈ 23 m/s + Phase-B ≈ 6 m/s ≈
**29 m/s total**. `artifacts/dock_trajectories.png` shows the 3-D radial approach,
the cuboids docked on the cylinder, and monotonic range convergence. This validates
the corrected geometry, dynamics, RAFTI model, assignment, handoff and bridge.

**Two capture bugs fixed during the geometry correction** (they had silently
blocked every capture, including in RL training): (1) a deputy that arrived and
stopped had an *undefined* velocity–axis alignment (→ 90°, always rejected) — now
near-zero relative velocity counts as aligned; (2) at the port the approach-cone
angle was ill-defined (lateral/≈0 axial → 90°) — the cone now carries a capture
ball at its apex.

**Classical baseline.** `lambert_return.py` (port of `eight_deputies_hcw_return.m`)
returns all deputies to the ring centre with a single CW/Lambert burn (0 m miss),
the deterministic reference the RL is compared against.

**RL training.** Both the faithful DDPG and SB3 TD3 train, checkpoint and evaluate
end-to-end (11/11 smoke tests pass, including a real gradient step). In-session, on
the fast HCW surrogate within a constrained compute budget, the policy learns to
**approach** (episode reward rises from the random baseline as it drives the
deputies inward) but does **not** reach reliable soft-capture — that convergence is
compute-bound and is exactly what `train/hpc/rl_train.slurm` is for (mirroring your
`arise_rl_kit` HPC workflow). `artifacts/reward_curve.png` shows the in-session
trend; `artifacts/dock_trajectories.png` shows the full-pipeline docking geometry.

---

## 8. What to run on the Basilisk / HPC machine
1. Point `guidance/phase_a_gnc_return.py` at the real chief via `hifi.load_chief()`
   and wire `dynamics/basilisk_backend._propagate_eci` to `_bsk_build/_bsk_run`.
2. `sbatch train/hpc/rl_train.slurm` — full DDPG (30k ep) + DDPG-HER (20k ep).
3. Validate the trained policy with `dynamics="basilisk"` and
   `eval/monte_carlo.py --trials 1000`.
4. Feed captured trajectories into the 6-DOF / animation pipeline
   (`scripts/sim/build_6dof_anim_data.py`) for the re-docking render.

---

## 9. Faithful-conversion ledger
| MATLAB source | Python | Change |
|---|---|---|
| `MultiChaserEnv.m` | `envs/multi_chaser_env.py` | + RAFTI per-port soft-capture, port-frame obs, Gym |
| `DDPGAgent.m` | `agents/ddpg.py` | **implemented the missing back-prop**; +grad clip |
| `ActorNetwork.m`/`CriticNetwork.m` | `agents/networks.py` | real `nn.Module`s, same architecture |
| `OUActionNoise.m` | `agents/ou_noise.py` | 1:1 |
| `ReplayBuffer.m` | `agents/replay_buffer.py` | 1:1 + `HERReplayBuffer` |
| `run_ddpg.m` | `train/train_ddpg_torch.py` | real learning, checkpoints, curriculum, resume |
| `rollout_swarm.m` / `run_monte_carlo.m` / `visualize_agent_performance.m` / `plot_trained_rewards.m` | `eval/*` | 1:1 |
| `eight_deputies_hcw_return.m` | `lambert_return.py` | relative-frame CW/Lambert (cleaner than the inertial round-trip) |

---

## 9b. Full 6-DOF docking (attitude pass)

The docking is now full **6-DOF**: each deputy actively slews its RAFTI face onto
the port while flying the radial glideslope, then soft-captures with both low
linear AND angular velocity - a real androgynous mate.

**State / action.** Per deputy the state is 12-D `[r, v, sigma, omega]` (LVLH
position/velocity + MRP attitude body<-LVLH + body rate) and the action is 6-D
`[dv(3) port-frame, tau(3) body torque]`.  Centralised policy: **96 -> 48**;
single-dock policy: **12 -> 6** (applied to all 8 by canonicalisation).

**Attitude dynamics (`dynamics/attitude.py`).** MRP kinematics
`sigma_dot = 1/4 B(sigma) omega` with shadow-set switching + Euler's rigid-body
equation `J omega_dot = tau - omega x J omega`, RK4 with substeps, using the real
6U inertia `J = diag(0.042, 0.204, 0.228) kg m^2` and mass 12.2 kg from the
separation run.  `dynamics/sixdof.py` composes translation (HCW or Basilisk) with
attitude into one step; the same interface serves both fidelities.

**6-DOF capture (`envs/rafti.py`).** On top of the translational soft-capture, the
deputy's dock face (+x_body, via the MRP) must point into the port within
`att_align_tol_deg` (10 deg) AND the body rate must be below `omega_capture_max`
(0.03 rad/s ~ 1.7 deg/s).  A deputy that arrives translated but mis-pointed or
spinning is rejected.

**Reward.** Adds an attitude pointing funnel (per-deg face error), a near-port
de-spin penalty, and a torque-fuel penalty; the +200 bonus now requires the full
6-DOF mate.  The observation carries the attitude error already rotated into the
BODY frame, so it maps directly to the body-torque action.

**A real finding - control cadence.** With a 2 s control step the pointing error
settles but a residual roll rate (~0.036 rad/s) survives, because the coarse
cadence cannot counter the gyroscopic `omega x J omega` coupling between the
asymmetric axes.  Dropping to a **1 s** step (and sizing the RCS torque to
1.2e-3 N.m so one command changes the rate by <~0.03 rad/s) makes the attitude
fully controllable - the 6-DOF baseline then docks 8/8 with face alignment < 2 deg
and rate < 0.03 rad/s.  This sets the control-rate / actuator-authority envelope
for the flight ADCS.

**Basilisk-ready (now wired, not a stub).** `dynamics/basilisk_backend.BasiliskPropagator.step6dof`
is the flight-fidelity 6-DOF step, mirroring your `_bsk_build`/`_bsk_run`: propagate translation under the
hi-fi force model AND attitude via a Basilisk `spacecraft` + `extForceTorque`
(commanded tau) + `gravityGradientEffector`, with `I_dep = self.inertia`,
integrated with RKF45 - mirroring `_bsk_build`/`_bsk_run` in the fleet script and
the SE(3) separation controller's attitude conventions.  It is import-guarded;
`dynamics="hcw"` is the sandbox/HPC surrogate.  Switching fidelity is a one-word
change to the env; the agent and reward are identical.

---

## 10. Correction changelog (geometry pass)
The first cut modelled the ESPA as a dimensionless point at the LVLH origin, so the
"approach" was just "drive to a point": trajectories collapsed into the ring plane
(N≈0), all deputies appeared to converge on the centre, and there was no structure
to dock to or collide with. Corrected to physical geometry:
- **`envs/geometry.py` (new)** — ESPA cylindrical shell + deputy cuboids + accurate
  cuboid-vs-shell (oriented-corner) and deputy-vs-deputy collision.
- **`envs/rafti.py`** — ports on the lateral surface, radial glideslope corridor +
  staging point, capture point offset so the dock FACE meets the surface, cone with
  an apex capture ball, low-speed alignment; cuboid collision replaces the sphere.
- **envs** — 3-D dispersal within each port's approach cone (not a plane); reward is
  a glideslope funnel (cone-excess penalty) that forces a radial approach; hard
  termination on shell/inter-deputy collision.
- **`eval/visualize.py`** — draws the cylinder, ports, approach trajectories, and the
  docked cuboids (terminal zoom).
Result: `docking_demo.py` now docks 8/8 with a physically-correct radial approach.