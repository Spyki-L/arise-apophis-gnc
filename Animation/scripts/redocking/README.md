# ARISE — RL Re-docking of the 8 Deputies onto the ESPA (Orbit Fab RAFTI)

Return the eight XB6 deputies to the ESPA ring after the Apophis stereo-imaging
pass and **soft-dock each one onto its Orbit Fab RAFTI port** — the same ports it
separated from at deployment. This package is a Python conversion + hardening of
the MATLAB DDPG swarm-return prototype
wired into the existing Apophis GNC.

## The idea in one line
**Phase A** brings the deputies back with your *existing* hi-fi GNC (Lambert seed
+ multiple-shooting TCM) — it is the deployment guidance run in reverse. **Phase B**
hands off inside the ESPA capture envelope to an **RL DDPG(+HER)** agent that flies
each deputy up its RAFTI approach cone to a soft-capture.

```
post-imaging dispersal (km)
   │  Phase A : existing hi-fi GNC, targets reversed  →  ESPA capture envelope (~100 m)
   ▼
handoff (ECI → ESPA-LVLH relative state)
   │  Phase B : DDPG(+HER) terminal docking, per-port cone + soft-capture
   ▼
8 deputies captured on the ESPA RAFTI ring
```

## Layout
```
redocking/
  config_redock.py        real ESPA/RAFTI port geometry + all parameters
  dynamics/               hcw.py (CW surrogate) · basilisk_backend.py (flight fidelity)
  dynamics/               hcw.py · attitude.py (MRP+Euler) · sixdof.py · basilisk_backend.py
  envs/                   geometry.py (cylinder+cuboids) · rafti.py · multi_chaser_env.py (96→48) · single_dock_env.py (12→6, HER)
  agents/                 networks.py · ou_noise.py · replay_buffer.py (+HER) · ddpg.py
  guidance/               phase_a_gnc_return.py   (reuses basilik_hifi_apophis_rendezvous_fleet.py)
  bridge/                 redock_pipeline.py      (Phase A → handoff → Phase B, end to end)
  train/                  train_ddpg_torch.py · train_sb3.py · hpc/*.slurm
  eval/                   rollout.py · monte_carlo.py · visualize.py · docking_demo.py
  lambert_return.py       classical CW/Lambert single-burn baseline (port of the .m demo)
  tests/smoke_test.py     11 integration tests (run this first)
  ARCHITECTURE.md         full design + integration + fidelity notes  ← read this
  LITERATURE_INDEX.md     the 35 survey PDFs, categorised
  artifacts/              trained checkpoints, reward curve, docking-trajectory figure
```

## Quick start — run locally in VS Code (no command-line arguments)

Every script has a `RUN CONFIG` block at the top and runs with **zero args** — just
open it in VS Code and press **Run** (▶).  Install once:

```
bash train/hpc/setup_rl_env.sh     # torch, gymnasium, SB3, scipy  (Basilisk: your own install)
```

**Verify your trajectories first (this is the one to run):**

```
run_local.py        ← press Run.  Does BOTH:
                       1) Phase-A: prints the per-deputy RETURN Delta-v table + phaseA_return_dv.png
                       2) Phase-B: 6-DOF DDPG final-approach docking → dock_trajectories.png
```

Or run the pieces individually (each is Run-able on its own):

```
guidance/phase_a_gnc_return.py   → per-deputy return Delta-v table (deploy + TCMs + brake)
eval/docking_demo.py             → 6-DOF terminal docking, 8/8, writes dock_trajectories.png
lambert_return.py                → classical CW/Lambert return baseline
tests/smoke_test.py              → 14 integration checks (run after install)
train/train_ddpg_torch.py        → DDPG(+HER) training (edit CFG at top: env/episodes/dynamics)
train/train_sb3.py               → SB3 TD3/DDPG + HER
eval/monte_carlo.py              → success-rate over N trials
eval/visualize.py                → reward curve + docking figure
```

### Switching to Basilisk (hi-fi) — one word
Each runnable script has `DYNAMICS = "hcw"` at the top.  Set it to `"basilisk"`:

```python
DYNAMICS = "basilisk"      # in run_local.py / eval/docking_demo.py / train_*.py
USE_BASILISK = True        # in guidance/phase_a_gnc_return.py
```

Then Phase-A uses the **real chief** (`GNC/Scenario/Apophis/chief_reference.npz`) with your
hi-fi Lambert + multiple-shooting TCM corrector, and Phase-B propagates translation in
**Basilisk** (Earth-SH + Sun + Moon + SRP, live SPICE, RKF45 — the same model as
`basilik_hifi_apophis_rendezvous_fleet.py`) with rigid-body attitude + gravity-gradient torque.
It's already wired in `dynamics/basilisk_backend.py` — nothing to implement; if Basilisk isn't
importable the scripts print a note and fall back to `hcw` so you're never blocked.

### Then train on HPC
```
sbatch train/hpc/rl_train.slurm      # full 6-DOF DDPG + DDPG-HER + SB3-TD3, then Basilisk validation
```

## Dynamics fidelity
The env has a pluggable backend, exactly like your `basilik_hifi_...` script has
basilisk/numpy/hybrid:
- `dynamics="hcw"`  — Clohessy-Wiltshire closed form. Fast; trains anywhere.
- `dynamics="basilisk"` — Earth SH + Sun + Moon + SRP via live SPICE + RKF45,
  the same force model as Phase A. Flight fidelity; **run on a Basilisk machine**.
Train on HCW, validate/fly on Basilisk with no change to the agent or reward.

See `ARCHITECTURE.md` for the physics, the integration seam, the RAFTI model, and
an honest account of what converged in-session vs. what needs the HPC run.
