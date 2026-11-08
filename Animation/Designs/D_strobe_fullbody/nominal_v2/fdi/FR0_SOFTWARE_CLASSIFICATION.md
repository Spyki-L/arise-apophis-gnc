# FR0 — Software Classification & Safety-Criticality Record

*Design D · `nominal_v2/fdi/` · the NPR 7150.2D software classification and the NASA-STD-8739.8 safety-criticality determination for the CORTEX FDIR flight software. This is FR0 of `CORTEX_FLIGHT_READY_PLAN.md` §8 — the assurance baseline that gates FR1 onward.*

## 1. Governing standards

- **NPR 7150.2D** — NASA Software Engineering Requirements (software classification A–F; required activities per class).
- **NASA-STD-8739.8** — Software Assurance & Software Safety Standard (independent safety-criticality assessment; SA/IV&V).
- **NASA Software Engineering Handbook** — classification & safety-criticality guidance.

## 2. Project software classification

| Attribute | Determination |
|---|---|
| Mission | Robotic chief–deputy swarm (Apophis 2029), **no crew** |
| Usage / criticality | Mission-critical onboard autonomy; loss-of-function → loss of science / possible loss of vehicle |
| Human rating | Non-human-rated |
| **NPR 7150.2 class** | **Class B** (non-human-rated, mission-critical) for the project; the **safety-critical** designation (per §3) is applied to the subset that can command actuation |
| Complexity / autonomy | Learning-enabled + agentic → the LEC assurance argument of §4 applies |

## 3. Safety-criticality determination (NASA-STD-8739.8)

A component is **safety-critical** iff it can **command actuation** (or guarantee that the safety loop runs). The determination is recorded per-component in the worksheet **`FR0_CRITICALITY_ASSESSMENT.xlsx`**; summary:

| Tier | Components | Class | Safety-critical |
|---|---|---|---|
| Certified spine | VSC (SC-01), RTA/Decision Module (SC-02), CBF filter (SC-03), cFS partitioning (SC-12) | **A/B** | **Yes** — the minimal certifiable surface |
| Reflex / safety-related | Behavior-tree core (SC-04), Belief base (SC-05), FDI detector (SC-06), MissionProfile config (SC-11) | **B** | Deterministic; feeds the safety decision |
| Monitored (learned/agentic) | GOAP planner (SC-07), Perception nets (SC-08), RL brain (SC-09), Re-optimizer (SC-10) | **C** | **No** — proposals the VSC can veto |

## 4. Tailoring rationale — why the certifiable surface is small (the crux)

Conventional guidance holds that learning-enabled components (LECs) are not certifiable by current methods. The **run-time-assurance (simplex) architecture** makes every learned/agentic component **non-safety-critical by construction** (the RTA boundary), so the Class-A/B safety-critical surface collapses from "an AI" to a few thousand lines of **deterministic, fixed-point, bounded-WCET** classical code:

- RTA + VSC + CBF (SC-01..03) — machine-checked in `FORMAL_VERIFICATION.md` (FR2 P1/P2/P3) and ported to MISRA-C in `csrc/vsc_spine.c` (FR1).
- Behavior-tree core (SC-04) — LTL spec verified (FR2 P4).
- Planner (SC-07, Class C) — WCET-cap proven (FR2 P5) but remains a *monitored* proposal.

The learned parts (SC-08..10) may only **raise a hypothesis / propose an action**; they never actuate. This is the classification-time realisation of the `SAFETY_CASE.md` argument.

## 5. Determination summary

- **Class B** project; **Class A/B safety-critical** applied to SC-01, SC-02, SC-03, SC-12; **Class C monitored** for all learned/agentic components.
- The safety-critical set is **minimal and classical** and already carries machine-checked evidence (FR2) + a MISRA-C reference build (FR1).
- **Open FR0 item:** the designations here are the developer's initial assessment; **NASA-STD-8739.8 requires an independent assessor to ratify them** (an FR6 IV&V-adjacent activity). This record + the worksheet are what that assessor reviews.

## 6. Baseline artifacts (this FR0 package)

```
FR0_SOFTWARE_CLASSIFICATION.md    this record
FR0_CRITICALITY_ASSESSMENT.xlsx   per-component independent-assessment worksheet (12 components)
FR0_SDP_SMP_SVP.md                SDP / SMP-SA / SVP plan skeletons mapped to NPR 7150.2 / 8739.8
```
