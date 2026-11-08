# Epistemic Autonomy for Onboard Swarm Fault Management: A Lightweight Neuro-Symbolic Agent with a Unified Distribution-Free Guarantee

*CORTEX-E · Design D / ARISE-Apophis · manuscript draft synthesizing phases E0–E8.
Companion artifacts: the design plan (`AGENTIC_NEUROSYMBOLIC_FDIR_PLAN.md`), the
flight-readiness plan (`CORTEX_FLIGHT_READY_PLAN.md`), the capability plan
(`CORTEX_ADVANCED_CAPABILITIES.md`), and the fourteen runnable, self-tested modules
in `nominal_v2/fdi/`. Numbers below are produced by those modules' self-tests and the
real-fleet calibration; every claim is executable.*

---

## Abstract

Onboard fault management for a spacecraft swarm on a one-shot, deadline-bound
encounter must be autonomous, distributed, and — because it is learning-enabled —
certifiable. We present **CORTEX-E**, a lightweight neuro-symbolic agent that treats
fault management as *inference under uncertainty* and gives the agent four operations
over that uncertainty: **quantify** it (metacognitive self-assessment), **explain**
it (causal abduction), **reduce it by acting** (active, information-seeking
diagnosis), and **reduce it over time** (safe in-flight learning). The unifying
method is a **conformal epistemic calculus**: a single distribution-free, finite-
sample layer that calibrates detection, abstention, decision, and learning. Its
capstone is a **unified closed-loop guarantee** — on a healthy fleet the probability
of *any* spurious epistemic action is bounded by the union of the per-stage conformal
levels, distribution-free, with zero safety violations enforced by a run-time-
assurance simplex. Calibrated on 28,036 real navigation residuals from a flown
10.3-hour rendezvous arc, the detector holds a measured false-alarm rate ≤ 0.2%
(ARL₀ ≳ 500) with 0 false alarms over the arc, while detecting an injected drift at
100% power in 7–20 samples. Causal abduction isolates *both* causes on 11/15 compound
faults where a precedence rule isolates 0; active diagnosis resolves passively-
ambiguous pairs in a single probe; a dual-control input separates faults that are
otherwise indistinguishable (0 → 4.5σ); a formal diagnosability theorem states when
isolation is possible; and safe online learning attains Õ(√T) regret with provable
forward invariance and 0 violations. The system-level guarantee holds at 2.2% ≤ 5.0%
and is distribution-free. We argue the design is flight-certifiable under NASA
NPR 7150.2 Class A/B because the run-time-assurance boundary makes every learned and
agentic component non-safety-critical.

---

## 1. Introduction

The ARISE concept flies an eight-deputy CubeSat swarm through a hyperbolic close flyby
of (99942) Apophis (≈8.76 km/s, ≈10 km miss, fixed closest-approach epoch). The
science objective — whole-body stereo coverage — is one-shot and deadline-bound, so a
fault that is detected late, isolated wrongly, or recovered unsafely is unrecoverable.
Prior onboard fault-detection-and-isolation (FDI) for this mission is a competent but
*offline, deputy-centric, detection-and-isolation* post-processor: six residual
channels (NAV, ATT, COMM, PROP, FORM, PEER), each a leaky-CUSUM under a
HEALTHY→SUSPECT→FAULTED→DEAD machine. It has three limitations as an *agent*: it holds
no persistent belief, it isolates by a single-label precedence rule that cannot name a
compound fault, and it neither knows when it is out of its depth nor acts to resolve
ambiguity nor improves in flight.

CORTEX-E reframes the problem: a fault-management agent is an inference machine under
uncertainty, and the useful autonomy is the ability to *reason about that
uncertainty*. This yields four capabilities — metacognition, causal reasoning, active
diagnosis, and safe learning — unified by one distribution-free calculus and bounded
by a run-time-assurance (RTA) simplex that makes the whole thing certifiable. The
contribution is not any single mechanism (each has a precedent in another domain) but
their **instantiation for deadline-coupled swarm FDIR, their unification under one
conformal calculus with a system-level guarantee, and their flight-certifiable
realization**, demonstrated on real flown data.

---

## 2. Related work and positioning

| Line of work | Representative | What it lacks vs CORTEX-E |
|---|---|---|
| Neuro-symbolic spacecraft AI | hierarchical NS-AI for maneuver + anomaly (AMOS 2025) | single-agent; no recovery; no deadline objective; no RTA simplex; no system guarantee |
| Onboard executives | MEXEC (Europa Clipper / M2020), PLEXIL | fault management = "stop and safe"; no learned perception; no objective-coupled severity |
| Run-time assurance | STARS/SARAS, ASIF, RTA-RL (2024–25) | single-agent control loops; not a fleet FDIR organ; no agentic deliberator |
| Conformal / distribution-free | conformal anomaly detection, conformal abstention (2024–25) | single-stage; not a whole-loop guarantee for a swarm |
| Metacognition / self-awareness | agentic metacognition w/ human handoff (2025), MAPE-K | not safety-critical, not conformal, not RTA-coupled; not a swarm |
| Causal fault diagnosis | SCM root-cause in IT/industrial (2024–25) | not grounded in relative-orbit dynamics; no counterfactual recovery vs a deadline objective |
| Active fault diagnosis | optimal-input AFD (single system); multi-agent active perception | not a swarm reconfiguring its crosslink graph under a deadline/science tradeoff |
| Safe online RL | Õ(√T)-regret safe RL (2024–25) | not in-flight FDIR-agent learning, RTA-bounded + conformal-gated, safety-case-preserving |
| System-level diagnosis | PMC t-diagnosability | classic; here reconnected to a byzantine crosslink fleet and to active resolution |

The unoccupied point is the *combination*: agentic deliberation + verifiable symbolic
core + advisory neural perception + RTA/CBF safety + a distribution-free calculus that
spans detection→abstention→diagnosis→learning with a system-level guarantee.

---

## 3. The conformal epistemic calculus (the spine)

Conformal prediction gives distribution-free, finite-sample validity under mere
exchangeability: a conformal p-value is super-uniform, `P(p ≤ α) ≤ α`, with no
Gaussian/residual-model assumption. CORTEX-E uses this as one calculus across the
loop. **Detection** replaces hand-tuned CUSUM thresholds with a phase-scheduled
(Mondrian) conformal test martingale; by Ville's inequality, alarming at `M ≥ 1/α`
bounds the whole-stream false-alarm probability by α (so ARL₀ ≥ ~1/α).
**Abstention**, **decision risk**, and **learning-rollback** reuse the same
p-value/martingale machinery. Phase-conditioning restores exchangeability across the
non-stationary encounter — a necessity we demonstrate: a single static calibration
false-alarms at 100% as the healthy residual scale rises into the encounter, while the
phase-conditioned calibrator holds 0.8% (module `conformal.py`, test T4).

---

## 4. Methods

### 4.1 Phase-scheduled conformal detection (E0), calibrated on flown data

`conformal.py` provides Mondrian split-conformal calibration, a mixture-of-power
conformal test martingale, and a drop-in `PhaseScheduledConformalDetector` wired into
`fdi_core` (additive, identity-when-absent: the legacy self-test remains 7/7).
Calibrated on **28,036 real NAV innovation residuals** from the flown ~10.3-hour
nominal arc (`conformal_calibrate.py`, cross-deputy split), it measures **0 false
alarms over the arc** for all held-out deputies, a **bootstrap false-alarm rate ≤
0.20% per phase** (ARL₀ ≳ 500 vs the 100 target), and detects an injected NAV drift at
**100% power in 7–20 samples**. The real residual is well-whitened and stationary
across phases (mean 0.48, p99 1.55), an honest finding: for a consistent filter the
conformal layer *confirms* health and supplies the distribution-free guarantee the
static threshold lacked.

### 4.2 Metacognitive self-FDIR (E1)

`metacognition.py` puts the agent in its own fault model. A competence
`c = (1−conflict)(1−mismatch)(1−ood)` is computed from disagreement among independent
sources (conflict), incoherent multi-channel firing (mismatch), and out-of-
distribution perception (ood); a conformally-calibrated threshold gives a guaranteed
false-abstention rate (measured 1.9% ≤ 2%). Below threshold the agent abstains to the
certified VSC via a new `AC_INCOMPETENT` verdict in `rta.py`. Crucially a *clear*
coherent fault stays competent (c = 0.95, the agent acts); abstention fires only on
conflict/novelty/OOD — so metacognition *strengthens* the simplex (the AC can recuse
itself, not only be overruled).

### 4.3 Causal abduction (E2)

`causal_model.py` replaces the single-label precedence rule with a structural causal
model (`fault → GNC physics → channel residuals`) and *abductive* isolation: the
minimal cause-set explaining the observed anomalies. It recovers **both** causes on
11/15 compound fault pairs, correctly **flags ambiguity** on the 2 observationally-
equivalent pairs, and collapses the 2 masked pairs — every explanation valid — where
the precedence rule recovers both on 0/15. Ambiguity (a lying PEER is observationally
identical to a FORM fault) and novelty (uncovered anomalies) are emitted as the
conflict/mismatch signals E1 consumes.

### 4.4 Active information-seeking diagnosis (E3)

`active_diagnosis.py` resolves ambiguity by acting: it selects measurement-graph edits
(peer re-image, ISL cross-range) maximizing expected information gain about the
hypothesis, traded against mission cost, with the submodular (1−1/e) greedy guarantee
and a structural diagnosability check. On the `{NAV,FORM,PEER}` ambiguity it picks a
`lying`-measuring probe (0.66 bits) over a useless one (0.00), collapsing hypothesis
entropy **1.0 → 0.29 bits in one probe**; it declines to probe when the information is
not worth the cost.

### 4.5 Safe in-flight learning (E4, rigorized in E7)

`learning.py` tunes advisory layers online with a no-regret learner; every action is
RTA-filtered so the executed action is safe regardless of what the learner proposes (0
violations even under a reckless config), and a *left-tailed* reuse of the E0 test
martingale on the reward stream gives a guaranteed false-rollback rate and a clean
rollback to the ground-validated baseline (detected an injected drift in 21 steps).
`safe_rl.py` rigorizes this as a LinUCB/OFUL contextual bandit: **Õ(√T) safe-regret**
(measured ratio R(4T)/R(T) = 1.11), **forward invariance** (barrier ≥ 0, max|s| =
1.000 = s_max), free exploration (θ̂→θ*, error 0.006), and **test-time policy
switching by remaining Δv budget** that never exceeds budget.

### 4.6 Counterfactual recovery and causal discovery (E5)

`causal_planner.py` scores each recovery action by `V(τ | do(a))` on the SCM. On a
byzantine-corruption scenario, `do(ISOLATE liar)` yields **+0.70** (peers recover)
while a myopic rule sees only the liar's own **−0.50** loss — so the counterfactual
planner recovers V = 4.00 where the myopic planner keeps the liar at 3.30.
`causal_discovery.py` refines the causal graph from flight data with a Wilson-lower-
bound ground-validation gate: it recovers the fixed edges at F1 = 1.0, discovers a
hidden `ATT→NAV` edge, rejects spurious edges (precision 1.0), and the discovered graph
corrects a diagnosis the fixed graph over-diagnoses.

### 4.7 Dual control and the diagnosability theorem (E6)

`dual_control.py` designs a safety-bounded excitation input that separates passively-
indistinguishable hypotheses: a NAV vs PROP fault identical at rest becomes **0 →
4.5σ** separable under a commanded test burn, closing the gap E3 flags.
`diagnosability.py` states and verifies the exact fleet condition: fault sets `F, F'`
are indistinguishable iff `N(F △ F') ⊆ F ∪ F'`; the fleet is f-diagnosable iff
`N ≥ 2f+1` and no such pair exists. `K_N` is f-diagnosable iff `N ≥ 2f+1`;
`(2f+1)`-connectivity is sufficient; a shared single witness defeats isolation
(exhibited) — `N ≥ 2f+1` alone is not sufficient.

---

## 5. The unified closed-loop guarantee (E8)

**Theorem (informal).** Let the epistemic loop compose E0 detection (level
`α_detect`), E1 abstention (`α_abstain`), and E4 rollback (`α_rollback`), each a
conformal/Ville-bounded test. On a healthy fleet the probability that the loop takes
*any* spurious epistemic action satisfies, by the union bound and the distribution-
free validity of each stage,

    P(spurious | healthy) ≤ α_detect + α_abstain + α_rollback,

with no distributional assumption; and every executed action passes the certified
VSC/RTA simplex, so there are 0 safety violations.

**Verification (`cortex_epistemic.py`, 6/6).** Composing all modules over 1,500
healthy and 480 fault episodes: the family-wise spurious-action rate is **2.20% ≤
5.0%** union bound (per-stage: detect 0.53%, abstain 1.67%, rollback 0.00%); it holds
**2.60% ≤ 5.0% under a heavy-tailed healthy distribution** (distribution-free);
single-fault diagnosis efficacy is **100%**; **100%** of ambiguous episodes are
flagged (abstain or probe) — never silently wrong; and there are **0 safety
violations**. This is the "conformal epistemic calculus" made concrete: detection,
abstention, and learning-rollback share one calculus, so the closed loop inherits a
finite-sample, system-level guarantee.

---

## 6. Results summary

| Phase | Module | Headline result |
|---|---|---|
| E0 | `conformal.py`, `conformal_calibrate.py` | spine 6/6; real arc: 0 FA, FA ≤ 0.2% (ARL₀ ≳ 500), drift 100% @7–20 samples; static-calib 100% FA vs Mondrian 0.8% |
| E1 | `metacognition.py` (+`rta` `AC_INCOMPETENT`) | 6/6; false-abstain 1.9% ≤ 2%; clear fault competent (0.95) |
| E2 | `causal_model.py` | 6/6; compound: both causes 11/15 vs precedence 0/15 |
| E3 | `active_diagnosis.py` | 6/6; ambiguity 1.0→0.29 bits in one probe; (1−1/e) ratio 1.0 |
| E4 | `learning.py` | 5/5; 0 violations; gate detects drift @21, clean rollback |
| E5 | `causal_planner.py`, `causal_discovery.py` | 5/5 + 5/5; do(ISOLATE) +0.70 vs myopic −0.50; discovery F1 1.0 + hidden edge |
| E6 | `dual_control.py`, `diagnosability.py` | 5/5 + 6/6; 0→4.5σ separation; `K_N` f-diagnosable iff N≥2f+1 |
| E7 | `safe_rl.py` | 6/6; Õ(√T) regret (ratio 1.11), invariance max|s|=1.000, θ err 0.006 |
| E8 | `cortex_epistemic.py` | 6/6; family-wise spurious 2.2% ≤ 5% union; distribution-free; 0 violations |

---

## 7. Flight-readiness and certification

The RTA simplex is the certifiability lever. NASA's position is that learning-enabled
components cannot be exhaustively verified; CORTEX-E does not need to, because every
learned and agentic component is a *proposal* the certified Verified-Safe Controller
can veto. Under NASA NPR 7150.2 + NASA-STD-8739.8, the safety-critical set collapses to
the VSC + Decision Module + CBF + the determinism of the belief/behavior-tree core — a
few thousand lines of deterministic, fixed-point, bounded-WCET code — with the planner
and perception as Class-C *monitored* components. The two-processor mapping (rad-hard
GR712/GR740 for the VSC; a rad-tolerant Versal AI Edge for the AC) and the delta-
certification argument (the safety-critical surface is mission-profile-independent) are
detailed in `CORTEX_FLIGHT_READY_PLAN.md`. The [NOW]-tier phases (E0–E4) live entirely
inside that Class-A/B envelope; the [MOONSHOT]-tier phases (E5–E8) are the research
frontier.

---

## 8. Limitations (honest)

Every component has a precedent in another domain; the contribution is the combination,
the unification, and the flight-certifiable instantiation — not a new base algorithm.
The real-data calibration is currently on the NAV channel (the attitude/formation logs
were absent from the snapshot); ATT/FORM/PEER calibrate identically once their streams
are present. The conformal guarantee assumes within-phase exchangeability, restored by
phase-conditioning; transitions widen the sets. The causal graph is hand-specified in
the [NOW] tier and discovered/ground-validated in [MOONSHOT]. The compute budgets are
targets pending PIL/HIL on the actual rad-hard parts. And "self-aware" here means
calibrated competence estimation with a safe handoff — not consciousness; the novelty
is the safety-critical, guaranteed, swarm instantiation.

---

## 9. Conclusion

Treating onboard swarm fault management as inference under uncertainty — with a single
conformal calculus spanning detection, abstention, diagnosis, and learning, all bounded
by a run-time-assurance simplex — yields an agent that is at once more capable (it
quantifies, explains, actively resolves, and safely learns) and *more* certifiable (the
learned parts are non-safety-critical by construction), with a system-level distribution-
free guarantee and a demonstrated path to a NASA Class-A/B flight build. The fourteen
runnable modules make every claim executable and every result reproducible.

---

## References (verify before submission)

Conformal / distribution-free: dual-threshold conformal abstention (2025); conformal
decision theory (ICRA 2024); distribution-free process monitoring with conformal
prediction (2025). Metacognition: agentic metacognition with failure handoff (2025);
self-aware computing; MAPE-K. Causal: structural causal models & counterfactual
root-cause (2024); actual causality in stochastic systems (2025). Active diagnosis:
optimal-input active fault diagnosis (2024); IMAS² multi-agent selection with (1−1/e)
(2025); sensor selection for diagnosability (Petri nets, 2024). Safe learning: safe RL
regret bounds (RLC 2025); safe online RL in the LQR, Õ(√T) (2024). Diagnosis theory:
Preparata–Metze–Chien / Hakimi–Amin system-level diagnosis. Run-time assurance: Cofer
et al. (NFM 2020); STARS (2025). Assurance: NASA NPR 7150.2D; NASA-STD-8739.8; NASA
Autonomy V&V Roadmap 2045. Carry over the ARISE master-plan and
`Peer_RelNav_Technical_Document.pdf` references, and the companion CORTEX plans.
