# CORTEX-E — Epistemic Autonomy: Self-Aware, Causal, Active, Safely-Learning Swarm FDIR

### Four novel capabilities that turn CORTEX from a *verifiable* agent into an *epistemic* one — it reasons about, and acts on, its own uncertainty — unified by a conformal (distribution-free) calculus, delivered in two tiers: a flight-certifiable **[NOW]** core and a research-frontier **[MOONSHOT]** tier, under the same RTA/VSC safety spine.

*Design D · `nominal_v2/fdi/` · the capability-expansion + novelty plan layered on
`AGENTIC_NEUROSYMBOLIC_FDIR_PLAN.md` (CORTEX) and `CORTEX_FLIGHT_READY_PLAN.md`. Builds on the
master plan's already-named cross-cutting method ("phase-scheduled conformal sequential detection")
and elevates it to the unifying spine. Honest novelty assessment throughout — every component has a
precedent in some domain; the contribution is the **instantiation for deadline-coupled swarm FDIR,
the unification under one epistemic/conformal calculus, the self-referential twist, and the
flight-certifiable [NOW] realization.*

**Codename:** **CORTEX-E** — the *Epistemic core*. Tag line: *an FDIR organ that puts itself inside
the fault model it manages, and knows what it does not know — with a guarantee.*

---

## 0. Executive summary

CORTEX (so far) is a *verifiable agent*: advisory perception + a temporal-logic-checkable behavior
tree + a bounded deadline-value planner, all bounded by the RTA simplex, mission-agnostic across
regimes. It is safe and explainable, but epistemically *flat* — it trusts its detectors, reasons by
pattern not mechanism, senses passively, and never improves in flight.

CORTEX-E adds the dimension that makes autonomy *truly* novel here: **the agent reasons about its own
knowledge.** Four capabilities, which are not four features but four faces of one idea — *acting under,
and to reduce, uncertainty*:

1. **Metacognitive self-FDIR** — uncertainty about *its own beliefs*: the agent puts itself in the
   fault model, calibrates its competence, and **abstains to the VSC with a distribution-free
   guarantee** when it is out of its depth.
2. **Causal & counterfactual reasoning** — uncertainty about *mechanism*: a structural causal model of
   how faults propagate (fault → dynamics → residual → objective) isolates novel/compound faults by
   *explanation*, and plans recovery as *counterfactual interventions* `do(a)` scored by `V(τ)`.
3. **Active information-seeking diagnosis** — uncertainty about *what to sense*: when the fault
   hypothesis is ambiguous, the fleet **reconfigures its own measurement/crosslink graph to maximize
   information gain**, traded against mission value, with a submodular `(1−1/e)` guarantee and a
   structural diagnosability condition.
4. **Safe in-flight learning** — uncertainty about *its own models*: the agent improves its advisory and
   planner-heuristic layers **online, under RTA, with a regret/competence certificate** that the safety
   case is preserved — turning the simplex boundary into a license to learn in flight.

The unifier — and the reason this reads as *method*, not engineering — is a **conformal epistemic
calculus**: one distribution-free, finite-sample layer (`conformal.py`) that calibrates detection,
abstention, decision risk, and learning competence on a violently non-stationary encounter. It is the
master plan's "phase-scheduled conformal" promise, generalized from the detector to the whole agentic
loop.

Per your direction this is **two-tier**. The **[NOW]** tier keeps every capability inside the
flight-certifiable, RTA-bounded, Class-A/B envelope of the flight-ready plan (conformal-calibrated
detection + abstention; a *fixed* causal graph; greedy myopic active sensing; offline-trained,
online-*shaped* advisory layers with a frozen safety core). The **[MOONSHOT]** tier is the publishable
frontier (online causal discovery + full counterfactual planning; dual-control active diagnosis with a
diagnosability theorem; safe online RL with an `Õ(√T)` regret bound; and the unified conformal guarantee
over the closed loop). Critically, **metacognition makes the system *safer*, not riskier** — it adds a
competence-triggered VSC handoff that the current simplex lacks.

---

## 1. The unifying idea — epistemic autonomy and the conformal spine

### 1.1 Why these four are one thing

A fault-management agent is, at bottom, an *inference machine under uncertainty*. CORTEX today treats
that uncertainty implicitly (CUSUM thresholds, point labels, fixed sensing, frozen models). CORTEX-E
makes uncertainty a **first-class, manipulable object** in the belief base, and gives the agent four
verbs over it: **quantify** it (metacognition), **explain** it (causal), **reduce** it by acting
(active diagnosis), and **reduce** it over time (safe learning). One noun, four verbs — that is the
architecture, not a checklist.

### 1.2 The conformal spine (`conformal.py`)

Conformal prediction gives **distribution-free, finite-sample** validity under mere exchangeability —
exactly the rigor a safety case wants and exactly what a learned or agentic component otherwise lacks.
The spine wraps the whole loop with one calculus:

- **Detection** — replace the hand-tuned leaky-CUSUM thresholds (`fdi_core.py`) with **conformal,
  phase-scheduled** thresholds: a guaranteed false-alarm rate (ARL₀) per mission phase, on
  non-stationary data, by conditioning the nonconformity score on the phase (`phase_schedule`).
- **Perception** — conformal **abstention** sets on the nets' outputs (Capability I).
- **Decision** — conformal **risk control** on the planner's committed actions (bounded probability a
  committed plan is value-negative).
- **Learning** — conformal **competence** monitors that gate online updates (Capability IV).

The same finite-sample guarantee thus covers detection-to-learning, which no prior swarm-FDIR system
offers. Phase-conditioning is the key to honoring exchangeability across the encounter (the violent
non-stationarity is *between* phases; within a phase the residuals are exchangeable enough).

---

## 2. Capability I — Metacognitive self-FDIR (the agent in its own fault model)

### 2.1 Idea

The boldest, most genuinely-new move: **the autonomy is a monitored subsystem of its own FDIR.** A
metacognition layer watches the agent's *epistemic* state — not the spacecraft's — and asks "is my
current belief trustworthy?" It fuses three self-signals: **(a) out-of-distribution** perception
(inputs unlike training), **(b) internal conflict** (detectors, consensus, vision, and the causal model
disagree), and **(c) model mismatch** (innovations inconsistent with every hypothesis). It maps these to
a calibrated **competence** `c ∈ [0,1]` with a conformal guarantee on its meaning, and when competence
falls below a phase-set threshold it **voluntarily abstains** — emitting a new RTA verdict
`AC_INCOMPETENT` that hands control to the certified VSC, with a logged, language-grade explanation of
*why it could not proceed*.

### 2.2 Integration

New `metacognition.py` reads `belief_base` + `perception` uncertainty + the causal residuals; it adds a
`competence` field to the belief base and a self-trust trace to `agent_trace.py`. The handoff is one
new enum value in `rta.RTAReason` (`AC_INCOMPETENT`) and one branch in `RTAExecutive.decide()` —
**a minimal, certifiable change** that *strengthens* the simplex (the AC can now recuse itself, not just
be overruled). W4 renders a live "self-trust" gauge and the abstention reason.

### 2.3 Tiers

- **[NOW]** — OOD score (e.g., Mahalanobis/energy on perception features) + a conflict score (disagreement
  among the independent sources, generalizing `recovery_ladder.fuse`) + conformal calibration of the
  abstention threshold → `AC_INCOMPETENT` handoff. Deterministic, bounded, Class-C-monitored feeding a
  Class-A/B handoff. Flight-certifiable.
- **[MOONSHOT]** — a learned **competence model** (predicts its own decision error from belief features) and
  **meta-self-awareness** (reasoning about the *cost/benefit* of gathering more evidence vs acting now —
  the bridge to Capability III), with a formal calibration guarantee.

### 2.4 Novelty (honest)

Computational self-awareness / MAPE-K and recent *agentic metacognition with failure-handoff* exist —
but for IT/business agents, with **human** handoff, no safety certificate, no conformal calibration, and
no swarm. The new parts: a **safety-critical, conformal-calibrated, RTA-coupled (machine) handoff** in a
multi-agent FDIR organ, and the **self-referential** framing (the agent inside the fault model it runs).

---

## 3. Capability II — Causal & counterfactual reasoning

### 3.1 Idea

Replace pattern-matching residual→label isolation with **mechanism**. A structural causal model (SCM)
encodes how a root-cause fault propagates: `fault → dynamics (regime STM) → measurement residual →
fleet-objective impact`. Isolation becomes **abduction** — the minimal set of root causes that best
explains the observed residual pattern (actual-causality, Halpern–Pearl style) — which **handles novel
and compound faults** the enumerated labels miss, and yields a real *explanation*, not a guess. Recovery
becomes **counterfactual planning**: the planner scores each candidate action as a do-intervention
`V(τ | do(a))` on the SCM, i.e., "what would coverage be *if* I disposed D4 and re-imaged with D7?" —
the same value function, now mechanistically grounded.

### 3.2 Integration

New `causal_model.py` defines the SCM over the fleet (nodes: faults, states, residuals, `κ_i` objective
sensitivities), wired to the mission-agnostic `DynamicsModel`/`ObjectiveModel` so the causal mechanism is
**profile-portable**. `belief_base` gains a posterior over root causes (not just a label); `planner`'s
candidate-simulation calls the SCM's counterfactual query instead of the raw surrogate. Counterfactual
explanations attach to every decision (`agent_trace`), and double as counterfactual explanations of
behavior-tree/temporal-logic outcomes.

### 3.3 Tiers

- **[NOW]** — a **fixed, hand-specified causal graph** (faults→residuals derived from the known GNC physics)
  + abductive isolation by minimal-cost explanation + counterfactual recovery via the existing surrogate
  as the structural equations. Bounded inference, deterministic, flight-certifiable.
- **[MOONSHOT]** — **online causal discovery** (refine the graph from flight data, ground-validated before
  upload), full do-calculus counterfactual planning, and counterfactual explanations for MITL/temporal-logic
  violations of the behavior tree.

### 3.4 Novelty (honest)

SCM/counterfactual root-cause analysis is established in IT observability and industrial process
monitoring; actual-causality discovery in stochastic systems is recent. **None couples a causal fault
model to spacecraft relative-dynamics propagation and to counterfactual *recovery* scored by a
deadline-indexed mission objective, across regimes.** The dynamics-grounded SCM + counterfactual-`V(τ)`
coupling is the new contribution.

---

## 4. Capability III — Active information-seeking diagnosis (the fleet acts to learn)

### 4.1 Idea

When belief is ambiguous (high entropy over root-cause hypotheses, or two explanations tie), CORTEX
today waits for more passive evidence. CORTEX-E **acts to disambiguate**: it treats the fleet's
**measurement/crosslink graph as a design variable** and reconfigures it — re-task a healthy peer to
re-image the suspect, open a specific ISL session, nudge a relative geometry — to **maximize expected
information gain** about the fault, *traded against mission value* (probe only if the information is worth
the science/▵v it costs). Because mutual-information sensor selection is **submodular**, a greedy schedule
gets the `(1−1/e)` optimality guarantee; a **structural diagnosability condition** on the resulting graph
says when a fault is identifiable at all.

### 4.2 Integration

New `active_diagnosis.py` computes expected-information-gain over candidate measurement-graph edits using
the SCM (Capability II) and drives `isl_scheduler.py` (which already schedules crosslinks) and the
peer-imaging hand-off. It adds a **`PROBE`** action class to the planner's menu, scored by
`expected_info_gain − λ·mission_cost(τ)` — so active sensing is *just another action* the deadline-value
planner weighs, and the CBF/`peer_cbf` keeps probe maneuvers safe.

### 4.3 Tiers

- **[NOW]** — **greedy myopic** information-gain over the discrete ISL/re-image re-tasking options, with the
  submodular `(1−1/e)` bound and the value tradeoff; bounded, event-driven on the AC tier. Flight-certifiable.
- **[MOONSHOT]** — **dual-control / optimal-input design** (actively perturb a relative trajectory to excite a
  fault signature) and a **formal diagnosability theorem** on the time-varying fleet measurement graph
  (when can `f` faults be isolated given the crosslink topology) — the structural-diagnosability MOONSHOT
  the master plan's Pillar II calls for, now *active*.

### 4.4 Novelty (honest)

Active fault diagnosis (probing inputs) exists for single systems; multi-agent active perception has the
`(1−1/e)` submodular result; sensor-selection-for-diagnosability exists for discrete-event systems.
**None makes a swarm reconfigure its own crosslink graph to disambiguate a fault under a deadline-coupled
science-cost tradeoff, tied to a diagnosability condition.** The value-aware, FDIR-coupled, fleet-graph
instantiation is new.

---

## 5. Capability IV — Safe in-flight learning (the simplex as a license to learn)

### 5.1 Idea

Onboard learning is normally banned from flight because it cannot be verified. The RTA boundary
**inverts** that: since the VSC bounds every action, the agent can **safely learn the parts that are not
safety-critical** — the advisory perception priors and the planner's value-shaping heuristics — *in
flight*, because no update can ever produce an unsafe action (the simplex catches it) and a **conformal
competence gate** rejects updates that degrade calibrated performance. Safe-RL theory supplies the other
half: under safety + sufficient excitation, **`Õ(√T)` regret** is achievable, and safety constraints can
even provide "free exploration." The result is a **lifelong-learning autonomy with a safety certificate**
— it gets better over a long mission without ever risking it.

### 5.2 Integration

New `learning.py` runs a ground+onboard loop: onboard it does **bounded online shaping** (bandit /
online-convex updates) of the advisory layers; the safety-critical VSC/RTA/CBF and the belief/BT core are
**frozen** (no learning in the Class-A/B path). All learned artifacts are version-controlled,
ground-validated, uploadable configuration items (per the flight-ready plan's data plane). The conformal
spine provides the competence gate; `agent_trace` logs every accepted/rejected update for audit.

### 5.3 Tiers

- **[NOW]** — **online shaping of advisory priors only** (perception confidence, planner heuristic weights)
  with a conformal competence gate and a hard rollback; frozen safety core; ground re-validation before any
  weight upload. Stays inside the flight-ready envelope.
- **[MOONSHOT]** — **safe online RL** of the planner's value-shaping with a proven `Õ(√T)` regret bound and a
  formal *safety-preservation* theorem (the closed loop never leaves the VSC-safe set during learning),
  plus test-time policy switching by remaining ▵v/science budget.

### 5.4 Novelty (honest)

Safe online RL with regret bounds is a current control-theory frontier; offline-then-upload is standard
practice. **None instantiates safe *in-flight* learning of an FDIR agent's advisory/planner layers,
bounded by an RTA simplex and gated by a conformal competence certificate, with the safety case provably
preserved.** The "simplex-as-license-to-learn" framing for flight autonomy is the contribution.

---

## 6. The two-tier synthesis

| Capability | [NOW] — flight-certifiable | [MOONSHOT] — frontier | Touches |
|---|---|---|---|
| I Metacognition | OOD+conflict score → conformal abstention → `AC_INCOMPETENT` VSC handoff | learned competence model + meta-self-awareness (sense-vs-act value) | `metacognition.py`, `rta.py`, `belief_base.py` |
| II Causal | fixed causal graph; abductive isolation; counterfactual recovery via surrogate | online causal discovery; full do-calculus; counterfactual TL explanations | `causal_model.py`, `belief_base.py`, `planner.py` |
| III Active diagnosis | greedy `(1−1/e)` info-gain over ISL/re-image, value-traded | dual-control input design; fleet diagnosability theorem | `active_diagnosis.py`, `isl_scheduler.py`, `planner.py` |
| IV Safe learning | online shaping of advisory priors, conformal-gated, frozen safety | safe online RL, `Õ(√T)` regret, safety-preservation proof | `learning.py`, `perception.py`, `planner.py` |
| Spine | conformal phase-scheduled detection + abstention + decision risk | unified conformal guarantee over the closed loop | `conformal.py`, `fdi_core.py` |

The **[NOW]** column is a complete, self-contained upgrade that lives inside the
`CORTEX_FLIGHT_READY_PLAN.md` Class-A/B envelope: everything is advisory, a proposal, or RTA-bounded; the
conformal layer supplies the distribution-free false-alarm guarantee the safety case wants; metacognition
*adds* a safety path. The **[MOONSHOT]** column is the publishable science.

---

## 7. Integration map (additive, identity-when-absent)

```
nominal_v2/fdi/
  conformal.py        the epistemic spine: phase-scheduled conformal calibration (detection/abstention/risk)
  metacognition.py    competence model + OOD/conflict/mismatch → AC_INCOMPETENT abstention      [Cap I]
  causal_model.py     SCM over fault→dynamics→residual→objective; abduction + counterfactuals   [Cap II]
  active_diagnosis.py expected-info-gain measurement-graph design; PROBE action; diagnosability [Cap III]
  learning.py         ground+onboard safe-learning loop for advisory/planner layers, conformal-gated [Cap IV]
```

All five bolt onto the CORTEX core through the existing seams: `belief_base` gains
`{competence, cause_posterior, info_value}` fields; `planner` gains the `PROBE` action and the
counterfactual `V(τ|do(a))` evaluator; `rta` gains the `AC_INCOMPETENT` verdict; `fdi_core` thresholds are
fed by `conformal.py`; `isl_scheduler`, `peer_cbf`, `recovery_ladder`, and the mission-agnostic
`DynamicsModel`/`ObjectiveModel` are reused unchanged. Each is gated by an `ARISE_EPISTEMIC_*` env flag
(identity-when-absent), so CORTEX-E degrades exactly to CORTEX when disabled — which is also the clean
ablation baseline. **W4 finally renders the epistemic state**: self-trust gauge, live causal graph,
information-gain heatmap, and the learning/competence trend.

---

## 8. Flight-safety story (unchanged spine, strengthened)

Every new capability is, by construction, one of: **advisory** (perception/causal priors, competence
score), a **proposal** (counterfactual plans, PROBE maneuvers — all RTA-gated), or **RTA-bounded
learning** (advisory layers only, frozen safety core). So the `SAFETY_CASE.md` top claim and the
`CORTEX_FLIGHT_READY_PLAN.md` Class-A/B mapping are untouched — and metacognition *improves* the safety
argument by adding a competence-triggered VSC handoff (`AC_INCOMPETENT`). The conformal spine converts the
SAFETY_CASE's qualitative false-alarm claim into a **distribution-free, finite-sample guarantee**. New
verification obligations (all [NOW]-tier, all bounded): conformal coverage validity under phase-conditioned
exchangeability; determinism/WCET of the abstention + abduction + greedy info-gain (they are O(N)–O(N²)
bounded); and the rollback safety of online shaping. None enters the safety-critical actuation path.

---

## 9. Novelty & prior-art positioning

| Capability | Closest prior art | The precise gap CORTEX-E fills |
|---|---|---|
| I Metacognition | computational self-awareness / MAPE-K; agentic metacognition with *human* failure-handoff (2025); conformal abstention for perception (2025) | safety-critical, conformal-calibrated, **RTA (machine) handoff** in a **swarm FDIR** organ; the **self-referential** fault model |
| II Causal | SCM root-cause in IT/industrial observability; actual-causality discovery in stochastic systems (2025) | causal model grounded in **relative-orbit dynamics** + **counterfactual recovery** scored by a **deadline-indexed objective**, across regimes |
| III Active diagnosis | single-system active fault diagnosis / optimal input design; multi-agent active perception `(1−1/e)`; sensor-selection diagnosability (Petri nets, 2024) | a **swarm reconfiguring its crosslink graph** to disambiguate a fault under a **deadline/science-cost tradeoff**, tied to a **fleet diagnosability** condition |
| IV Safe learning | safe online RL with `Õ(√T)` regret (2024–25); offline-then-upload practice | **in-flight** learning of an FDIR agent's advisory/planner layers, **RTA-bounded + conformal-gated**, with the **safety case preserved** |
| Spine | conformal anomaly detection / process monitoring (2025); the master plan's phase-scheduled conformal | one **distribution-free guarantee across the whole agentic loop** (detection→abstention→decision→learning) for a non-stationary encounter |

**The four unifying novelties** (the part that is "nothing like before"): (1) **epistemic unification** —
detection, diagnosis, sensing, and learning as four operations on one uncertainty object; (2) the
**conformal epistemic calculus** giving finite-sample guarantees end-to-end; (3) **self-reference** — the
agent inside its own fault model with a competence-triggered safe handoff; (4) **deadline/value coupling**
of all four to the mission objective. Venue: the systems result fits **AIAA JGCD**; the epistemic-autonomy
method (conformal spine + self-referential FDIR) is a **Science Robotics / NeurIPS / L4DC**-class
contribution.

---

## 10. Validation (extends the F10 / CORTEX harness)

- **Competence calibration** — abstention coverage matches the conformal target; on injected OOD/novel
  faults the agent abstains to VSC *before* erring; 0 unsafe.
- **Causal isolation** — vs the CUSUM baseline on **compound/novel** faults: higher isolation accuracy +
  a valid explanation; counterfactual-recovery value vs the reflex/CORTEX planner.
- **Active diagnosis** — information-gain and time-to-disambiguation vs passive waiting; empirical
  diagnosability vs the structural condition; science cost of probing accounted.
- **Safe learning** — regret curve (approaching `Õ(√T)`) and end-of-mission competence gain, with **zero
  safety-constraint violations** throughout (the certificate, demonstrated).
- **Spine** — measured ARL₀/false-alarm meets the conformal target across all mission phases.
- **Ablations** — CORTEX-E vs CORTEX (epistemic flags off) on every metric; per-capability leave-one-out.

---

## 11. Roadmap (E-phases, layered on the A-phases; [NOW] first)

| Phase | Scope | Tier | Exit artifact |
|---|---|---|---|
| **E0 Conformal spine** | `conformal.py`; phase-scheduled conformal thresholds replace static `ν/h` | NOW | measured ARL₀ meets target across phases (the master-plan P2 result) |
| **E1 Metacognition** | OOD+conflict competence + `AC_INCOMPETENT` handoff | NOW | calibrated abstention; 0 unsafe under injected OOD |
| **E2 Causal core** | fixed-graph SCM; abductive isolation; counterfactual recovery | NOW | isolation win on compound faults + explanations |
| **E3 Active diagnosis** | greedy `(1−1/e)` info-gain `PROBE` over ISL/re-image | NOW | faster disambiguation vs passive; value-traded |
| **E4 Safe shaping** | online shaping of advisory priors, conformal-gated, frozen core | NOW | competence gain, 0 safety violations, clean rollback |
| **E5 Counterfactual+discovery** | do-calculus planning; online causal discovery (ground-validated) | MOONSHOT | counterfactual planner; refined graph |
| **E6 Dual-control + diagnosability** | optimal-input active diagnosis; fleet diagnosability theorem | MOONSHOT | theorem + active-probe demo |
| **E7 Safe online RL** | regret-bounded learning of planner value-shaping + safety-preservation proof | MOONSHOT | `Õ(√T)` regret, proof, 0 violations |
| **E8 Unified guarantee + paper** | conformal guarantee over the closed loop; the epistemic-autonomy manuscript | MOONSHOT | end-to-end guarantee + manuscript |

Critical path: **E0→E1→E2→E3→E4** is the complete flight-certifiable epistemic upgrade (and already a
strong paper). E5–E8 are the frontier. E0–E2 need no new hardware and run against the current `nominal_v2`
products in numpy, like P0/P1.

---

## 12. Risks, limitations, honesty

- **Conformal exchangeability** is violated by the encounter's non-stationarity; **phase-conditioning**
  restores it within a phase, but transitions need care (widen sets at phase boundaries). State this; it is
  the spine's main assumption.
- **Causal-graph correctness** — a wrong [NOW] graph mis-isolates; mitigate by deriving it from validated
  GNC physics and cross-checking against the CUSUM detector (disagreement → metacognitive abstention). The
  graph is advisory; the VSC still bounds actuation.
- **Active-probe cost** — probing spends science/▵v; the value tradeoff (`λ`) must be tuned so the agent
  never probes itself past the deadline. Conservative default: probe only when `info_value ≫ cost`.
- **Online-learning distribution shift** — guard with the conformal competence gate + hard rollback +
  ground re-validation before any upload; never learn in the safety path.
- **Don't over-claim "self-aware."** This is calibrated competence estimation + a safe handoff, not
  consciousness. The novelty is the *safety-critical, guaranteed, swarm* instantiation — say exactly that.
- **Compute** — abduction and info-gain are bounded (O(N²) over a 9-node fleet) and event-driven on the AC
  tier; the [MOONSHOT] discovery/RL is ground-trained, onboard-inference only.

---

## 13. References (verify before submission)

Conformal / distribution-free: Dual-Threshold Conformal Prediction for autonomous perception abstention
(2025); Conformal Decision Theory (ICRA 2024); State-Dependent Conformal Perception Bounds for
Neuro-Symbolic Verification (2025); Safe OOD-Adaptive MPC with conformalized NN ensembles (L4DC 2025);
Adaptive/Online Conformal Anomaly Detection; Distribution-Free Process Monitoring with Conformal
Prediction (2025). Metacognition / self-awareness: Agentic Metacognition — self-aware agent for failure
prediction & handoff (2025); the Notion of Self-aware Computing; MAPE-K (IBM autonomic computing);
meta-self-awareness (receding-horizon hybrid planning). Causal: structural causal models &
counterfactual root-cause (observability/industrial; EP4310618B1, 2024); Efficient Discovery of Actual
Causality in Stochastic Systems (2025); Counterfactual Explanations for MITL Violations (2024); causality
in XAI (2025). Active diagnosis: Joint Observer-Gain and Input Design for Active Fault Diagnosis (2024);
distributionally-robust active fault diagnosis; IMAS² info-theoretic multi-agent selection with `(1−1/e)`
(2025); Resilient Active Information Acquisition with Teams of Robots; Optimal Sensor Selection for
Diagnosability (Petri nets, 2024). Safe learning: Improved Regret Bounds for Safe RL (RLC 2025);
Foundations of Safe Online RL in the LQR — `Õ(√T)` regret & free exploration (2024); online CMDP with
anytime constraints (ICML 2025). Carry over `AGENTIC_NEUROSYMBOLIC_FDIR_PLAN.md`,
`CORTEX_FLIGHT_READY_PLAN.md`, `SAFETY_CASE.md`, and the master-plan references.
