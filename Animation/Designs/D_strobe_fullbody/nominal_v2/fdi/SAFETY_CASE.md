# Flight-Readiness Safety Case — onboard swarm autonomy (F9)

*Design D · `nominal_v2/fdi/` · the assurance argument for flying the F2–F8 autonomy package.
Every claim below is backed by an executable check in `flight_readiness.py` / `run_selftest_f9.py`
(all 7/7) and the compute-budget figure `figs/wcet_compute_budget.png`.*

---

## 1. Top claim

> **The onboard autonomy package can be flown because it cannot make the spacecraft unsafe: a
> certified, bounded, deterministic Verified Safe Controller (VSC) is always available and always
> in control of safety, while the high-performance Advanced Controller (AC) — including the
> learning-based perception and the optimizer — can only ever *improve* on that baseline, never
> override it.**

This is a **run-time-assurance (simplex) safety argument**: it does not require formally verifying
the AC (which would be intractable for an optimizer + learned perception). It requires only that
(a) the VSC is safe and sufficient on its own, and (b) the Decision Module that arbitrates between
them is sound. Both are established below.

---

## 2. Argument structure

```
                       ┌─────────────────────────── Decision Module (sound) ──────────────────────────┐
   AC proposal ───────▶│  validity?  ─no─▶ VSC │  deadline met?  ─no─▶ VSC │  safety predicate?  ─no─▶ VSC │──▶ actuate
   (optimizer /        │      │yes                     │yes                         │yes                      │
    perception, AI)    │      └────────────────────────┴────────────  AC accepted  ─┴──────────────────────▶ │
                       └──────────────────────────────────────────────────────────────────────────────────┘
   VSC  =  contingency-table lookup + CBF-safe defaults + safe-hold   (certified, bounded, deterministic)
```

Sub-claims and the evidence for each:

| # | Sub-claim | Evidence (F9 test) | Result |
|---|-----------|--------------------|--------|
| A | The Decision Module is **sound** — no unsafe AC proposal is ever actuated | T5 RTA soundness (2000 mixed proposals) | **0** unsafe executed; 989 safe accepted, 1011 rejected→VSC |
| B | The flight path is **deterministic** (reproducible, no data-dependent branching of timing) | T1 determinism | bit-identical output for identical input+seed |
| C | The flight path has **bounded memory** (O(1), no growth over the mission) | T2 bounded memory | footprint identical at 1k and 100k steps |
| D | The flight path has **bounded WCET** that fits every cadence | T3 WCET + compute budget | all layers fit (see §4) |
| E | The safety-critical math is **fixed-point feasible** (no float64 dependence) | T6 fixed-point | 0 float32-vs-float64 verdict mismatches |
| F | The collision CBF is **forward-invariant** (the keep-out is never breached when enforced) | T4 CBF invariance (300×300 sweep) | min h ≥ −1e-6 filtered vs −2500 unfiltered |
| G | The system **degrades gracefully** — AC failure leaves a safe action | T7 graceful degradation | AC exception → VSC safe action, no crash |

With A established, B–G establish that the *VSC itself* is a competent, bounded, certifiable
controller — so the system is safe whether the AC succeeds, fails, or is slow.

---

## 3. The two-processor mapping (the "minimal compute upgrade")

- **VSC** — the certified baseline (FDI bank, FORM-tube margin, PEER-CBF, contingency-table lookup,
  safe-hold) runs on a **rad-hard OBC** (GR712/GR740-class LEON, or Cobham/Vorago ARM). All of it is
  O(1)–O(N²) in the 9-node fleet, bounded memory, fixed-point feasible (§2 C/E), and deterministic
  (§2 B). It needs no new theory and is what flight qualification certifies.
- **AC** — the optimizer (Tier-2 re-optimization, spin re-point) and the learned perception
  (spin-state, visual fault ID) run on a **rad-tolerant performance module** (AMD Zynq
  UltraScale+/Kria-class SoC-FPGA). It is monitored, not trusted: timing/validity/safety are all
  checked by the Decision Module (§2 A).

This is the entire hardware delta to today's XB6 avionics.

---

## 4. Compute budget (measured, scaled, vs cadence)

Per-call work is constant per tick (independent of mission time), so WCET is bounded. Measured in the
numpy sandbox, scaled to the target processors (VSC ≈ 30× slower than the sandbox x86 for the bounded
math; AC ≈ 3× for FP), against the cadence deadline:

| Layer | Processor | onboard WCET | cadence budget | margin |
|---|---|---|---|---|
| FDI step | VSC | ~3 ms | 100 ms (10 Hz) | ~30× |
| FORM tube | VSC | ~4 ms | 1000 ms (1 Hz) | ~250× |
| PEER-CBF | VSC | ~13 ms | 100 ms (10 Hz) | ~8× |
| ISL scheduler | VSC | ~41 ms | 1000 ms (1 Hz) | ~24× |
| Re-optimizer (Tier-2) | AC | ~1.0 s | 60 s (event window) | ~60× |

Every layer fits with large margin (`figs/wcet_compute_budget.png`). The heavy re-optimization is
**event-driven**, not per-tick, and runs in the seconds-to-minutes reconfiguration window — never in
the final-seconds snapshot burst, where the contingency table (Tier-1, instant lookup) governs.

---

## 5. Formal properties

- **CBF forward-invariance.** For the collision barrier `h = ‖r_ij‖² − d_safe²`, enforcing the
  discrete condition `2 r·v + γ h ≥ 0` keeps `h_{k+1} ≥ (1−γΔt) h_k ≥ 0` for `γΔt ≤ 1` — the keep-out
  set is forward-invariant. T4 verifies this numerically over a 300-IC × 300-step sweep with a
  minimal-norm safety filter: the filtered trajectories never breach (min h ≥ −1e-6), the unfiltered
  ones collapse (min h = −2500). The connectivity and consistency CBFs are the same construction.
- **RTA set-invariance.** Because the Decision Module reverts to the VSC on any failed check, the
  closed loop never leaves the VSC-safe set; T5 demonstrates soundness over 2000 adversarial proposals.
- **Decision-policy termination/safety.** The decision policy is a finite behaviour tree over a finite
  action menu with a bounded-iteration optimizer — it is model-checkable for safety/liveness (the
  bounded structure is in place; a formal model-check is a P-level item, see §7).

---

## 6. V&V status and TRL

- **Unit + property tests:** the whole package is green — P0/P1 7/7, channels 6/6, F2 6/6, F3 7/7,
  F4 7/7, F5 7/7, F6 6/6, F7 7/7, F8 8/8, F9 7/7.
- **Integration:** F8 demonstrates an end-to-end whole-mission run (deputy loss, actuation
  degradation, spin change, misbehaving deputy, deadline hold) with zero ground intervention, zero
  unsafe actions, and the mission objective preserved.
- **TRL:** the algorithms and the integrated simulation place the package at **TRL 4→5**; the
  determinism/WCET/fixed-point/RTA evidence here advances the *flight-software* aspect toward
  **TRL 5→6**. Full TRL 6 needs PIL/HIL on the actual target parts (§7).

---

## 7. What full flight qualification still requires (honest maturity)

This safety case argues the *architecture* is flyable; it is not a completed qualification. Remaining,
in priority order:

1. **PIL/HIL on the target processors** — port the VSC to the rad-hard OBC and the AC to the Zynq
   part; measure true WCET and validate the fixed-point build (the scaling factors in §4 are estimates).
2. **Formal model-check** of the decision policy and the RTA switch (the structure is bounded and
   amenable; this turns §5's argument into a machine-checked proof).
3. **The comms/ISL model** — the PEER-CBF connectivity/consistency and the ISL-scheduler results are
   only as strong as the crosslink model; a measured RF/link budget is the long pole (flagged
   throughout).
4. **Certification campaign (F10)** — the full Monte-Carlo over all failure modes × phases × Byzantine
   counts with CIs, plus the radiation/fault-containment analysis for the chosen parts.
5. **Independent V&V** and the FMECA/fault-tree linkage required by mission assurance.

None of these change the architecture; they are the evidence-generation that converts a TRL-5/6
flight-software build into a qualified one.
