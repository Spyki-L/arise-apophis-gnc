# FR0 — SDP / SMP-SA / SVP Plan Skeletons

*Design D · `nominal_v2/fdi/` · the three governing plans required by NPR 7150.2D + NASA-STD-8739.8, as **baselined skeletons** (section structure, required content, and the status/evidence already in hand). Full plans are authored against these outlines. Status key: 🟢 evidenced now · 🟡 partial · ⬜ outstanding.*

---

## A. Software Development Plan (SDP) — NPR 7150.2D

| § | Section | Content | Status |
|---|---|---|---|
| A.1 | Scope & classification | Class B; safety-critical subset per `FR0_SOFTWARE_CLASSIFICATION.md` | 🟢 |
| A.2 | Lifecycle model | Incremental; certified-spine-first (FR1→FR2→FR3) | 🟢 (defined) |
| A.3 | Requirements & bidirectional traceability | reqs → design → code → tests, safety-critical reqs tagged | 🟡 (proofs + tests exist; formal trace matrix outstanding) |
| A.4 | Architecture & design | Two-processor simplex (VSC rad-hard / AC rad-tolerant); `SAFETY_CASE.md` §3 | 🟢 |
| A.5 | Coding standards | **MISRA-C:2012** for the VSC/RTA/CBF spine; no dynamic alloc, bounded loops | 🟢 (`csrc/vsc_spine.c`, `-Werror` clean) · ⬜ licensed MISRA checker sign-off |
| A.6 | Configuration management | version-controlled CIs incl. **uploadable artifacts** (net weights, tables, profiles) | 🟡 (delta-cert hashes in `DELTA_CERT_RECORD.md`) |
| A.7 | Reviews & audits | peer/inspection reviews; coverage gates | ⬜ (process) |
| A.8 | Build & reproducibility | deterministic safety-critical binary | 🟢 (numpy/stdlib + C; hashes recorded) |

## B. Software Management & Assurance Plan (SMP / SA) — NASA-STD-8739.8

| § | Section | Content | Status |
|---|---|---|---|
| B.1 | SA organisation & independence | developer SA + **independent** assessor/IV&V | 🟡 (developer analyses built; IV&V team outstanding) |
| B.2 | Safety-criticality assessment | independent determination | 🟢 worksheet (`FR0_CRITICALITY_ASSESSMENT.xlsx`) · ⬜ independent ratification |
| B.3 | Software safety analyses | FMECA, fault tree, hazard analysis | 🟢 `FMECA.xlsx`, `fault_tree.svg` (FR6) |
| B.4 | Assurance case | GSN claims→evidence | 🟢 `assurance_case_gsn.svg` (FR6) |
| B.5 | LEC assurance argument | certify the monitor, not the model | 🟢 `SAFETY_CASE.md` §2 + `FORMAL_VERIFICATION.md` |
| B.6 | FMECA/FTA → mission-assurance linkage | FTA top event ↔ project fault tree | 🟡 (FTA built; project-FTA linkage outstanding) |
| B.7 | Independent V&V | independent audit of VSC/RTA/CBF | ⬜ (FR6) |

## C. Software V&V Plan (SVP) — NPR 7150.2D §5 / 8739.8

| § | Section | Content | Status |
|---|---|---|---|
| C.1 | Unit & property test | green suites | 🟢 P0/P1, channels, F2–F9, + `csrc` MC/DC suite (20/20) |
| C.2 | Coverage targets | statement/branch (Class B); **MC/DC for VSC/RTA/CBF** | 🟢 MC/DC demonstrated for `rta_decide` (5/5) · 🟡 full MC/DC instrumentation |
| C.3 | Formal methods | RTA soundness, CBF invariance, BT LTL, planner WCET | 🟢 `FORMAL_VERIFICATION.md` 5/5 (Z3 + sympy) |
| C.4 | Fixed-point validation | float64 parity of the safety verdict | 🟢 `csrc` (0 false-safe) |
| C.5 | Static WCET | loop-bound / op-count analysis | 🟢 `csrc/FR3_STATIC_WCET.md` · ⬜ measured on GR712/GR740 (PIL) |
| C.6 | PIL / HIL | on-target WCET, SEU, closed-loop brassboard | ⬜ (FR3/FR4/FR5, hardware) |
| C.7 | Monte-Carlo certification | detection/isolation/recovery, 3 regimes | 🟡 `CERTIFICATION_DOSSIER.md` (real-fleet + surrogate) · ⬜ 3-regime expansion |
| C.8 | Mission-agnostic conformance | C1–C6 delta-cert | 🟢 `fr7_conformance.py` + `DELTA_CERT_RECORD.md` |

---

## Traceability at a glance (what FR0 already points to)

- **Design/safety:** `SAFETY_CASE.md`, `CORTEX_FLIGHT_READY_PLAN.md`
- **Formal (FR2):** `FORMAL_VERIFICATION.md` + `formal/` (5/5 proved)
- **MISRA-C spine (FR1) + WCET (FR3-sw):** `csrc/`
- **Safety analyses (FR6):** `FMECA.xlsx`, `fault_tree.svg`, `assurance_case_gsn.svg`, `FR6_SAFETY_ANALYSES.md`
- **Conformance (FR7):** `DELTA_CERT_RECORD.md`

**Outstanding (by design):** independent V&V ratification (FR6), and everything on-target — PIL/HIL/radiation (FR3/FR4/FR5) — plus the licensed MISRA static-analysis sign-off.
