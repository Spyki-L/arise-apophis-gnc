# CORTEX FDIR — Assurance Status (FR0–FR7, without-HIL build)

*Design D · `nominal_v2/fdi/` · one-glance map of the flight-readiness roadmap after the software-and-analysis build. Status: 🟢 evidenced now · 🟡 partial · 🔴 hardware-gated (needs the bench).*

| Phase | What | Status | Artifacts |
|---|---|---|---|
| **FR0** | Assurance baseline | 🟢 built (independent sign-off pending) | `FR0_SOFTWARE_CLASSIFICATION.md`, `FR0_CRITICALITY_ASSESSMENT.xlsx` (12 components), `FR0_SDP_SMP_SVP.md` |
| **FR1** | Certifiable-core freeze | 🟢 reference build (MISRA sign-off pending) | `csrc/vsc_spine.{c,h}` (RTA+CBF+VSC, `-Werror` clean), `csrc/test_vsc_spine.c` (20/20; MC/DC 5/5; fixed-point 0 false-safe) |
| **FR2** | Formal verification | 🟢 **5/5 proved** | `FORMAL_VERIFICATION.md`, `formal/verify_all.py`, `behavior_tree.py`, `planner.py` |
| **FR3** | Fixed-point + WCET on OBC | 🟡 static done · 🔴 measured cycles | `csrc/FR3_STATIC_WCET.md` (loop-bound + empirical anchor); measured on GR712/GR740 = hardware |
| **FR4** | AC on FPGA + radiation | 🔴 hardware | — |
| **FR5** | System HIL brassboard | 🔴 hardware | — |
| **FR6** | Cert campaign + assurance case | 🟢 developer analyses built · 🔴 IV&V | `FMECA.xlsx`, `fault_tree.svg`, `assurance_case_gsn.svg`, `FR6_SAFETY_ANALYSES.md` |
| **FR7** | Mission-agnostic delta-cert | 🟢 automated | `fr7_conformance.py`, `DELTA_CERT_RECORD.md`, `delta_cert_record.json` |

## This build (FR0 / FR1 / FR3-sw / FR7)

- **FR0** — Class B project; Class A/B safety-critical applied only to the RTA/VSC/CBF spine + partitioning; every **learned/agentic** component is Class-C monitored (non-safety-critical by the RTA boundary). Worksheet has 12 components ready for independent ratification.
- **FR1** — the safety spine is a **MISRA-C reference** (`vsc_spine.c`): RTA switch, CBF margin/filter (float64 **and** fixed-point Q16.16), contingency VSC. Compiles clean under `-std=c99 -Wall -Wextra -pedantic -Werror`; 20/20 unit tests, **MC/DC** demonstrated for all 5 RTA conditions, and the **fixed-point verdict has zero false-safe** vs float64.
- **FR3 (reachable)** — a **static WCET argument**: every spine loop is bounded (==3), no recursion/alloc → WCET is a finite constant; empirical anchor projects **< 0.1 µs/call** on LEON, > 10⁵× inside the 10 Hz budget. Only *measured cycles on the part* remain (PIL).
- **FR7** — the **§6.8 conformance contract** as an automated C1–C6 suite over `apophis` / `apophis_test4` / `toy_inspect`. The delta-cert record proves **one binary across missions** (identical core SHAs → FR2 proofs carry over) and honestly flags **C3 (author the Tier-1 contingency table)** and **C6 (comms latency/topology bounds)** as the per-mission delta.

## What genuinely remains (all hardware or independent-human)

- **FR3/FR4/FR5** — on-target WCET, SEU/radiation, and the HIL brassboard (the bench).
- **FR6** — an **independent** V&V team to ratify the FR0 criticality assessment and audit the safety analyses.
- **FR1/FR3** — the **licensed MISRA static-analysis** and target static-WCET tooling sign-off.

Reproduce this build: `cd csrc && make check` · `python3 fr7_conformance.py` · `python3 formal/verify_all.py` (all exit 0).
