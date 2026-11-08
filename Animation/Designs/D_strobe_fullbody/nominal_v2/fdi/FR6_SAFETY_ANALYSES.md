# FR6 — Safety & Mission-Assurance Analyses (FMECA · FTA · GSN)

*Design D · `nominal_v2/fdi/` · the software-and-analysis portion of `CORTEX_FLIGHT_READY_PLAN.md` §6 / FR6, buildable without hardware. Ties together `SAFETY_CASE.md`, `CERTIFICATION_DOSSIER.md`, and the `FORMAL_VERIFICATION.md` (FR2) proofs into the three artifacts a mission-assurance review expects.*

These are the **initial analyses for independent V&V review** — the structure and the argument; the IV&V sign-off itself (an independent human team) is the remaining FR6 item.

| Artifact | File | What it shows |
|---|---|---|
| FMECA | `FMECA.xlsx` | 18 failure modes across all tiers → local/system effect, detection, RTA/VSC containment, criticality, **residual class** |
| Fault tree | `fault_tree.svg` / `.png` | Top event = unsafe actuation; every path requires a certified-spine barrier to fail |
| Assurance case | `assurance_case_gsn.svg` / `.png` | GSN: the `SAFETY_CASE` top claim decomposed to strategy → sub-goals → evidence, with the outstanding items marked undeveloped |

All three tell the **same story three ways**: the learned/agentic tier is contained by construction, so the residual safety risk lives only in the small certified spine (addressed by FR2) and in the on-target hardware evidence (FR3/FR4/FR5).

---

## 1. FMECA (`FMECA.xlsx`)

**Method.** Each failure mode is scored `Severity (1–5) × Likelihood (1–5)` = **Criticality** (and `× Detectability` = RPN), *unmitigated*; then a **Residual class** is assigned *after* the RTA/VSC/CBF containment:

- **NEGLIGIBLE** — proof- or fallback-excluded (green).
- **PERFORMANCE** — reduced to degraded coverage/authority only, never a safety event (amber).
- **SAFETY** — a residual safety risk remains (red).

**Key finding (the headline row on the *Method & Legend* sheet).** After containment, **every AC-monitored (learned) failure mode — perception mis-classification, planner divergence, an unsafe RL proposal, optimizer failure — collapses to PERFORMANCE.** The RL brain's worst case (F10: proposes an unsafe action) has residual severity 1 because the RTA vetoes it (FR2·P1; observed live as 7,345 proposed → 0 executed). The only **SAFETY**-residual rows are:

- **config-management** — a bad uploaded contingency table or net/profile (F03, F18) → gated by the FR7 conformance contract + the CBF; and
- **on-target hardware** — SEU on either partition and a partition breach (F13, F14, F17) → gated by rad-hard parts, EDAC, and cFS partitioning, whose *evidence* is FR3/FR4/FR5.

That is the quantitative statement of the safety case: **none of the residual safety risk is the RL policy.**

## 2. Fault tree (`fault_tree.svg`)

**Top event:** unsafe actuation (collision / keep-out breach / dV-authority violation / loss of vehicle). It decomposes (OR) into four ways it could occur, and **each requires a certified-spine barrier to fail**:

1. *An unsafe AC action is executed* — needs `AC proposes unsafe` **AND** `RTA fails to veto`. The second is **excluded** by FR2·P1 (exhaustive + Z3 soundness).
2. *The VSC baseline is itself unsafe* — a bad contingency-table entry or safe-hold; bounded by config-management + the CBF, and certified/deterministic (F9).
3. *The CBF admits a breach* — needs `keep-out approached` **AND** `CBF invariance fails`. The second is **excluded** by FR2·P2 (Z3 ∀-proof + fixed-point).
4. *The spine is not run / is corrupted* — partition breach or an SEU in the VSC math. These are the **residual** basic events — the on-target FR3/FR4/FR5 evidence.

**Cut-set reading:** the RL/AC inputs (blue) cannot reach the top event alone — the RTA gate and the CBF cut those paths. The minimal cut sets that remain are all **hardware/config**, not the learned policy.

## 3. Assurance case — GSN (`assurance_case_gsn.svg`)

Goal-Structuring-Notation decomposition of the `SAFETY_CASE.md` top claim (**G1**), under the run-time-assurance strategy **S1** ("certify the monitor, not the model"), into sub-goals each discharged by a **solution (evidence)** node:

- **G2** decision-module soundness → **FR2·P1**  ·  **G3** CBF forward-invariance → **FR2·P2**
- **G4** determinism / bounded-memory / fixed-point → **F9 T1/T2/T6 + FR2·P3**  ·  **G5** graceful degradation → **F9 T7**
- **G6** learned/AC tier contained → **FMECA + FTA** (this package)  ·  **G7** detection & fault-tolerance → **CERTIFICATION_DOSSIER**

The three **undeveloped goals** (GSN diamonds) make the gap explicit and honest: **G8** on-target WCET/fixed-point (FR3), **G9** AC SEU + HIL (FR4/FR5), **G10** independent V&V + sign-off (FR6). None is RL training.

---

## Consolidated conclusion

Across the FMECA (residual class), the FTA (cut sets), and the GSN (undeveloped goals), the outstanding **safety** evidence is the same short list: **on-target PIL/HIL (FR3–FR5)**, **independent V&V (FR6)**, and **config-management (FR7)** — plus the standing **RF/ISL link-budget** long pole flagged in `SAFETY_CASE.md` §7. The learned RL layer contributes only to *performance* branches everywhere.

## Honest scope (what these do **not** yet close)

- **Independent V&V** — these are the developer's analyses; NASA-STD-8739.8 requires an independent team to audit them. This package is what they would review.
- **Quantitative likelihoods** — the `L`/`D` scores are engineering first-pass judgments for ranking, to be refined against the reliability data / radiation cross-sections during FR4/FR6.
- **On-target evidence** — every red/undeveloped item resolves only with hardware (FR3/FR4/FR5).

## Files

```
FMECA.xlsx                    18 failure modes + Method & Legend sheet (0 formula errors)
fault_tree.svg / .png         top-event fault tree with the cut-set argument
assurance_case_gsn.svg / .png GSN assurance case (goals → strategy → evidence → undeveloped)
FR6_SAFETY_ANALYSES.md        this cover document
```
Evidence referenced: `FORMAL_VERIFICATION.md` (FR2), `SAFETY_CASE.md` (F9), `CERTIFICATION_DOSSIER.md` (F10), `CORTEX_FLIGHT_READY_PLAN.md` §6/§8.
