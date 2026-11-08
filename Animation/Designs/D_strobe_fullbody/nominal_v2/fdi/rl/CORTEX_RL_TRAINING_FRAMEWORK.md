# CORTEX-RL — A High-Fidelity Multi-Agent RL Training Framework for the FDIR Brain

### Train a real, multi-agent (CTDE) reinforcement-learning fault-management policy — on a fast surrogate for bulk learning and on Basilisk-in-the-loop for high-fidelity — then deploy it as the Advanced Controller of the CORTEX-E stack in the Apophis mission.

*Design D · `nominal_v2/fdi/rl/` · the training framework requested to turn the E4/E7
learning *mechanism* into an actually-trained agentic brain. It reuses the CORTEX-E
epistemic stack for the observation (E0–E3 belief) and the safety layer (RTA/E4),
and produces a policy that plugs in as the AC under the VSC simplex.*

---

## 0. What this is (and what it is not)

You asked for a **real, trained, multi-agent RL brain for FDIR, trained in high
fidelity with Basilisk**. This framework is that: a PettingZoo multi-agent
environment for swarm fault management, a Stable-Baselines3 trainer (CTDE /
parameter-sharing PPO), a **Basilisk-in-the-loop backend** so the policy learns on
flight-grade dynamics, and a deploy script that runs the trained brain in the Apophis
scenario.

Honest scope: **I built and validated the framework; you train the brain.** The
environment is validated here (PettingZoo API test + rollouts); the SB3 trainer and
the Basilisk backend run on **your** machine (torch/SB3 are heavy, and Basilisk is
your local sim — neither runs in my sandbox). The RL policy is **not pre-trained** —
`train_mappo.py` is what produces it. The four Basilisk hooks are documented stubs you
wire to your Design-D scenario. Everything else is runnable now.

---

## 1. Architecture

```
                         reward = team ΔV(coverage) + per-agent shaping
   ┌───────────────────────────────────────────────────────────────────────┐
   │  fdir_env.FDIRParallelEnv  (PettingZoo ParallelEnv, one agent / deputy) │
   │    obs_i  = BELIEF (E0 residual severities · E2 conflict/mismatch ·     │
   │             E1 competence · authority · time-to-go · κ · neighbours)    │
   │    act_i  ∈ {NOOP,REINIT,MODE,RELOAD,SAFEHOLD,ISOLATE,DISPOSE,PROBE,REOPT}│
   │    RTA safety filter: unsafe (cost>authority) ⇒ VSC default, proposal    │
   │             penalised ⇒ 0 executed violations (E4/E7 guarantee)          │
   │    state() = global obs  →  centralized (MAPPO) critic                   │
   │    backend (pluggable, same interface):                                  │
   │        SurrogateBackend  (fast numpy — bulk training)                    │
   │        BasiliskBackend   (Basilisk-in-the-loop — high fidelity)          │
   └───────────────────────────────────────────────────────────────────────┘
        │  SuperSuit: parallel → vec (parameter sharing)                       ▲
        ▼                                                                      │
   train_mappo.py  (SB3 PPO, shared policy) ── model.zip ──▶ run_apophis_agent.py
```

- **`fdir_env.py`** — the environment. Observation = the CORTEX-E belief state
  (so the RL sees exactly what the epistemic stack produces); action = the recovery
  menu; reward = recovered mission coverage (team) + shaping (correct-recovery bonus,
  Δv/opportunity cost, unsafe-proposal penalty). The **RTA filter** makes every
  *executed* action safe, so the policy explores freely with **0 safety violations**.
- **`basilisk_backend.py`** — the high-fidelity backend: steps the real Basilisk fleet,
  injects real faults, reads real residuals. Four `TODO` hooks (§4).
- **`train_mappo.py`** — SB3 + PettingZoo + SuperSuit parameter-sharing PPO.
- **`run_apophis_agent.py`** — loads the trained policy and runs it (decentralized) in
  the Apophis scenario; heuristic fallback if no model/torch, so the loop always runs.

---

## 2. Multi-agent scheme — CTDE, and the honest MAPPO caveat

You chose **CTDE / MAPPO**. The SB3 + PettingZoo path realizes this as follows:

- **What ships and works today:** *parameter-sharing PPO* — SuperSuit maps the parallel
  env to a vectorized env where one shared policy controls all 8 (identical) deputies;
  decentralized execution, shared weights, coordinated by the **team reward**. This is
  the de-facto "SB3 multi-agent" recipe (≈ IPPO-with-parameter-sharing).
- **The gap to *true* MAPPO:** SB3's critic is decentralized (value net sees local obs).
  A true centralized critic reads `env.state()` (the global belief). SB3 doesn't do this
  out of the box. Two upgrade paths, both flagged in `train_mappo.py`:
  1. Subclass `ActorCriticPolicy` so the **critic input is `state()`** while the actor
     keeps local obs, and store `state()` in the rollout buffer (a ~100-line extension).
  2. Swap the trainer for a dedicated MAPPO implementation (e.g. a MAPPO repo, or
     RLlib's centralized-critic model) against the same PettingZoo env.

Start with parameter-sharing PPO (works now, already coordinates via the shared policy
+ team reward); move to the centralized critic if you need tighter chief↔deputy
coordination.

---

## 3. Observation / action / reward spec

- **Observation** (per deputy, 15-D): 6 per-channel residual severities (E0), causal
  `conflict` + `mismatch` + `competence` (E2/E1), remaining authority, time-to-go,
  off-station, coverage weight κ, and a 2-D neighbour summary (fraction flagged, mean
  severity). This is the belief the epistemic stack computes — the RL learns the
  *decision* on top of it, not the perception.
- **Action** (per deputy, Discrete-9): the recovery/probe/reconfigure menu. Each has a
  ΔV cost; `PROBE` = active diagnosis (E3), `ISOLATE`/`DISPOSE` = byzantine handling,
  `REOPT` = coverage reconfiguration.
- **Reward**: `team = Δ(fleet mission value)` (shared, drives coordination) `+` per-agent
  shaping `{+0.6 correct recovery, −0.1·cost per action, −0.2 acting on a healthy deputy,
  −0.3 unsafe proposal}`. The unsafe proposal is *penalised but never executed* — the
  RTA veto guarantees safety while the penalty teaches the policy to propose safe.

---

## 4. Basilisk integration (the four hooks)

`basilisk_backend.py` matches `SurrogateBackend`'s interface exactly, so the reward and
observation are identical across fidelities — only the dynamics change. Implement:

1. **`_build_sim()`** — construct the fleet sim from your Design-D scenario (reuse
   `scripts/guidance/basilik_hifi_apophis_rendezvous_fleet.py` +
   `scripts/sim/build_6dof_anim_data.py`, or `MC_v2_run.py`'s chain), one spacecraft per
   deputy + chief, camera-tracking attitude.
2. **`_inject_fault(i, ftype)`** — apply a Basilisk fault model (NAV = unmodeled accel,
   ATT = unmodeled torque, PROP = ΔV-execution error at the next burn, COMM = heartbeat
   freeze, FORM = station perturbation, PEER = corrupted ISL self-report).
3. **`_read_residuals(i)`** — read the deputy's onboard residuals exactly as
   `run_fdi_nominal` builds them (NAV NIS-RMS from y_pre/sigma_meas, ATT from
   sigma_BR/omega_BR, PROP ΔV-mismatch, FORM/PEER causal). This is the same residual the
   E0 conformal detector consumes, so a policy trained here transfers to flight.
4. **`_apply_action(i, action)`** — command the recovery on the sim (reinit filter, mode
   switch, safe-hold, ISL isolate, CBF-safe disposal burn, re-point/re-station), spending
   real ΔV; return whether it addressed a real fault.

---

## 5. Training recipe & compute reality

**Recommended two-stage** (even though you picked in-the-loop — you'll still want stage
A to debug the pipeline in seconds before committing sim hours):

- **A. Surrogate bulk-train** — `python train_mappo.py --backend surrogate --timesteps 2e6`.
  Millions of steps in minutes; gets the policy most of the way.
- **B. Basilisk hi-fi fine-tune** — `python train_mappo.py --backend basilisk --timesteps 2e5
  --num-vec 1`. Flight-grade dynamics; the trainer auto-limits to ≤2 slow envs.

**Compute reality (Basilisk-in-the-loop):** a Basilisk episode is minutes of sim; PPO
needs 10⁵–10⁶ environment steps. Pure in-the-loop from scratch can take **days–weeks**
without a cluster — which is exactly why stage A exists. If you truly want in-the-loop
only, budget accordingly, keep `--num-vec` small, shorten `--horizon`, and consider a
coarser Basilisk `dt`.

---

## 6. Install & run

```bash
pip install "stable-baselines3[extra]" torch supersuit pettingzoo gymnasium

# validate the env (no torch needed) — PettingZoo API test + rollout
python fdir_env.py

# dry-run the whole pipeline with the heuristic fallback (no model/torch)
python run_apophis_agent.py --backend surrogate

# train the brain
python train_mappo.py --backend surrogate --timesteps 2000000        # -> fdir_brain.zip
python train_mappo.py --backend basilisk  --timesteps 200000 --num-vec 1   # hi-fi fine-tune

# deploy the trained brain in the Apophis scenario and watch it act
python run_apophis_agent.py --model fdir_brain.zip --backend surrogate
python run_apophis_agent.py --model fdir_brain.zip --backend basilisk       # on the sim machine
```

Validated here: `fdir_env.py` passes `pettingzoo.test.parallel_api_test`; a random
rollout and the deploy loop (heuristic policy) run end-to-end (fault campaign injected,
recovery actions taken, mission value recovered, 0 unsafe executions).

---

## 7. How the trained brain plugs into CORTEX-E (deployment)

The RL policy is the **Advanced Controller (AC)** of the CORTEX-E stack: its observation
is the belief produced by E0 (conformal detection) → E2 (causal diagnosis) → E1
(competence); its action is a *proposal* routed through `rta.RTAExecutive` — the certified
VSC vetoes anything unsafe, so the learned policy is bounded exactly as in E7. `PROBE`
actions invoke E3 active diagnosis; `ISOLATE`/`DISPOSE` invoke the E5 counterfactual
recovery. So the trained brain is the *decision* layer, and every earlier phase remains
its perception, diagnosis, safety, and explanation scaffold — the certifiability argument
(learned parts non-safety-critical under the RTA boundary) carries over unchanged.

---

## 8. Honest limitations

- **The policy is untrained until you run `train_mappo.py`** — this framework produces the
  brain; it doesn't ship a trained one.
- **The Basilisk backend is four documented stubs** — I can't run Basilisk in my
  environment, so you wire and validate hooks 1–4 on your machine.
- **True MAPPO (centralized critic) is an extension** — parameter-sharing PPO ships; the
  centralized critic is a ~100-line SB3 subclass or a MAPPO-lib swap (§2).
- **The surrogate reward is a coverage/authority model** — faithful to the CORTEX-E fault
  semantics, but the *definitive* objective is the sim's true stereo-coverage, which the
  Basilisk backend's `value()` can read for the hi-fi fine-tune.
- **The obs runs a causal abduction per step** — fine for training scale here; cache or
  simplify if you push to very high throughput.
