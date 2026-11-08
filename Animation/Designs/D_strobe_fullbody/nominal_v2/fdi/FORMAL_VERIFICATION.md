# FR2 — Formal Verification (machine-checked safety properties)

*Design D · `nominal_v2/fdi/` · turns the **numerically-demonstrated** properties of `SAFETY_CASE.md` §5 (the F9 T4/T5 sweeps) into **machine-checked** results, and closes the two software-gated FR2 items (behavior-tree LTL spec, planner WCET cap) by implementing and verifying the reactive core and the bounded planner. Pure software — **no target hardware required.***

**Reproduce:** `python3 formal/verify_all.py` — exit 0 iff all properties are proved.
Dependencies: `z3-solver`, `sympy`, `numpy`. The verified modules (`rta.py`, `flight_readiness.py`, `peer_cbf.py`, `behavior_tree.py`, `planner.py`) are numpy/stdlib-only; the proofs run anywhere.

**Result: 5 / 5 properties PROVED.**

| Property | Method | Replaces / closes | Result |
|---|---|---|---|
| P1 · RTA simplex soundness | exhaustive over real `rta.py` (2⁵) + Z3 boolean theorem + concrete corroboration | F9 T5 (2000 samples) | **PROVED** |
| P2 · CBF forward-invariance | exact `sympy` identity + Z3 ∀-proof over ℝ + numeric bridge + fixed-point check | F9 T4 (300×300 sweep) | **PROVED** |
| P3 · Decision determinism / totality | exhaustive over real `rta.py` (2⁵) | F9 T1 (extended) | **PROVED** |
| P4 · Behavior-tree temporal-logic spec | exhaustive safety invariants + explicit-state liveness + Z3 BMC | FR2 BT-LTL item (was outstanding) | **PROVED** |
| P5 · Planner WCET cap | exhaustive input sweep + adversarial stress + Z3 ranking function | FR2 planner-WCET item (was outstanding) | **PROVED** |

---

## P1 — RTA (simplex) soundness

**Theorem.** For every input, the Decision Module in `rta.py` executes the AC proposal **only when all five guards hold** (competent ∧ well-formed ∧ valid ∧ within-deadline ∧ safety-predicate); otherwise it executes the VSC fallback. Therefore **no action that fails the safety predicate is ever executed, for any safety predicate.**

**Method.** Exhaustive execution of the real `RTAExecutive.decide()` over its entire 2⁵ guard space (**0 violations / 32**); Z3 shows `(accept ∧ ¬safe)` and `(accept ∧ ¬all-guards)` **UNSAT**; concrete reach+separation predicate over 5000 proposals → `executed_unsafe = 0`. Stronger than T5: exhausts the decision-logic input space and holds for *any* predicate.

## P2 — CBF forward-invariance (collision keep-out)

**Theorem.** With `h(r)=|r|²−d²`, step `r_{k+1}=r_k+v·dt`, if `C:=2·r·v+γ·h ≥ 0`, `0≤γ·dt≤1`, `h(r_k)≥0`, then `h(r_{k+1})≥0` — `{h≥0}` is forward-invariant.

**Method.** Exact `sympy` identity `h_{k+1}=h_k(1−γdt)+dt·C+dt²|v|²` (residual **0**); Z3 ∀-proof over ℝ that hypotheses ∧ `¬(h_{k+1}≥0)` are **UNSAT**; numeric bridge to the real `cbf_velocity_filter` (min h ≈ 0 vs −2500 unfiltered) + float32==float64 verdicts. Stronger than T4: a proof over the whole continuous state space.

## P3 — Decision-policy determinism / totality

**Theorem / method.** `decide()` is deterministic, total, and single-branch — exhaustive over 2⁵ cases: **0** non-deterministic, **0** no-action, all 6 enumerated reasons reachable.

## P4 — Behavior-tree temporal-logic spec

**Module verified:** `behavior_tree.py` — the deterministic, bounded-depth reactive core (belief → one safe command per 10 Hz tick; priority: collision → `CBF_SAFE_HOLD`; fault → `RECOVER` if recoverable & in-deadline else `SAFE_HOLD`; else re-opt `DELIBERATE`; else `NOMINAL`).

**Theorem.** The BT satisfies its **safety** invariants and a bounded-settling **liveness** spec.

**Method.**
- *Safety (LTL `G φ` ⇒ per-tick invariant, exhaustive over 2⁵ beliefs):* S1 collision ⇒ `CBF_SAFE_HOLD`; S2 uncorrectable fault ⇒ `SAFE_HOLD`; S3 `NOMINAL` ⇒ no hazard; S4 fault ⇒ acted on; S5 determinism. **0 violations / 32.**
- *Liveness (closed loop, benign/fair environment), two ways:* complete **explicit-state** reachability — L1 collision cleared ≤1 tick, L2 SAFE-SETTLED reached ≤ K=4 ticks, L3 SAFE-SETTLED absorbing (`G(settled→G settled)`), **0 violations**; and a **Z3 bounded model-check** encoding the transition relation over K steps — "never settled within K" is **UNSAT** for all symbolic starts.

## P5 — Planner WCET cap (bounded work / termination)

**Module verified:** `planner.py` — GOAP-lite STRIPS best-first recovery planner over the finite action menu, admissible heuristic, **hard iteration + depth caps**, warm-started from the contingency table (anytime → returns the best plan found so far on cap).

**Theorem.** For every input the planner (a) terminates within `max_iters` expansions, (b) returns a plan of length ≤ `max_depth`, (c) always returns a well-formed, feasible proposal (or the warm-start on cap) — never hangs, never raises. Hence its worst-case work is bounded and WCET-cappable.

**Method.**
- *Exhaustive sweep* — every 4-deputy fault configuration (7⁴ = **2401**): `iters ≤ cap`, `len ≤ depth`, feasible, and the plan **clears every fault** (0 bound-violations, 0 wrong-plans; worst 18/300 iters, 4/12 length).
- *Adversarial stress* — 8×PEER at caps {1,2,5,300}: anytime returns a feasible warm-start, `iters ≤ cap`, cap-hit handled.
- *Termination (Z3 ranking function)* — the loop counter increments by 1 and the loop exits at `counter==cap`, so #iterations ≤ cap (UNSAT to exceed).

---

## What this upgrades

- `SAFETY_CASE.md` §5 — **RTA set-invariance**, **CBF forward-invariance**, and the **decision-policy termination/safety** claim are now machine-checked (P1/P2/P3).
- The two previously-open FR2 sub-items — **behavior-tree LTL model-check** and **planner WCET-cap proof** — are now **closed** (P4/P5): the reactive core (`behavior_tree.py`) and the bounded planner (`planner.py`) were implemented per `AGENTIC_NEUROSYMBOLIC_FDIR_PLAN.md` and verified.
- **Gap checklist FR2** — all *software* sub-items are now green.

## Honest scope — the one remaining FR2 item

- **On-target WCET (FR3)** is the only outstanding FR2-adjacent item, and it is **hardware-gated** — this suite bounds *logic and iteration counts*, not *cycles on the GR712/GR740*. Measured WCET needs PIL on the target part (FR3). The `SAFETY_CASE.md §4` WCET table remains an estimate until then.
- Note: `behavior_tree.py` / `planner.py` are reference implementations of the designed reflex/deliberation tiers; their integration WCET on the flight OBC is likewise an FR3 (on-target) measurement, not a logic item.

## Files

```
behavior_tree.py                  reactive core (deterministic, bounded) — verified by P4
planner.py                        GOAP-lite bounded/anytime planner — verified by P5
formal/verify_rta_soundness.py    P1
formal/verify_cbf_invariance.py   P2
formal/verify_determinism.py      P3
formal/verify_behavior_tree.py    P4 — safety invariants + explicit-state liveness + Z3 BMC
formal/verify_planner_wcet.py     P5 — exhaustive + adversarial + Z3 ranking function
formal/verify_all.py              runs all five; exit 0 iff every property PROVED
```

## Provenance

Z3 4.16.0, sympy 1.14. The proofs contain no randomness; numeric bridges use fixed seeds. `verify_all.py` returns 0 (all proved) and is safe to wire into CI as a certification gate.
