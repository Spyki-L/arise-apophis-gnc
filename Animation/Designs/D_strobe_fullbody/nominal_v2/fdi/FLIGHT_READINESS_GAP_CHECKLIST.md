# ARISE CORTEX FDIR — Flight-Readiness Gap Checklist

One-page status against the **FR0–FR7** roadmap (`CORTEX_FLIGHT_READY_PLAN.md` §8), `SAFETY_CASE.md` §7, and `CERTIFICATION_DOSSIER.md`.

**Bottom line:** the *architecture* is flight-ready (TRL ≈ 5, safety case complete as an argument); flight *qualification* is an outstanding V&V campaign. The learned RL layer is deliberately **non-safety-critical**.

Legend: 🟢 Evidenced (done) · 🟡 Partial (argued / estimated, not proven on target) · 🔴 Outstanding

| Phase | Gate / deliverable | Status | Evidence on hand · what's still needed |
|---|---|---|---|
| **FR0** | Assurance baseline — sw classification + independent criticality assessment; SDP/SMP/V&V plans | 🟡 Partial | Safety-critical (Class A/B) vs monitored (Class C) split **defined** (CORTEX §1.2). NASA plan set + independent criticality sign-off not yet produced. |
| **FR1** | Certifiable-core freeze — VSC + RTA + CBF + belief/BT to Class-A/B code; static analysis; unit MC/DC | 🟡 Partial | Logic implemented; unit/property suites **green** (P0/P1, channels, F2–F9 = 7/7–8/8). Still numpy — no MISRA-C freeze, no MC/DC. |
| **FR2** | Formal verification — BT LTL model-check; RTA soundness; CBF invariance (per regime); determinism; planner WCET-cap | 🟡 Partial | Numerically **shown**: RTA 0/2000 unsafe (F9 T5), CBF 300×300 sweep (T4), determinism/bounded-mem/fixed-point (T1/T2/T6). Machine-checked *proofs* outstanding. |
| **FR3** | Fixed-point + WCET on OBC — MISRA build of VSC; deterministic WCET on GR712/GR740 (PIL); fixed-point validation | 🔴 Outstanding | WCET only **estimated** (sandbox ×30 scaling, SAFETY_CASE §4). No target-part PIL, no MISRA build. ← the TRL 5→6 step. |
| **FR4** | AC on FPGA + radiation — port perception/planner/RL to Zynq/Versal; SEU/TMR/scrub; RTA catches injected faults (HIL) | 🔴 Outstanding | Nothing on target FPGA; radiation campaign not started. **The RL net is characterized here** (inference WCET / SEU) — only if it is flown. |
| **FR5** | System HIL brassboard — closed-loop VSC (rad-hard) + AC (FPGA); fault + SEU injection; 0 unsafe | 🔴 Outstanding | End-to-end demonstrated in **sim only** (F8 whole-mission, 0 unsafe, objective preserved). No brassboard HIL. |
| **FR6** | Cert campaign + assurance case — expanded MC (3 regimes, agent-vs-reflex); FMECA/FTA; sw-safety; GSN; IV&V | 🟡 Partial | F10 dossier on real-fleet + surrogate: Pd≈1.0 / Pfa=0, Byzantine f≤3, 66.6% gap recovered / 98.8% beat-τ*. Full 3-regime MC, FMECA/FTA, GSN, IV&V outstanding. |
| **FR7** | Mission-agnostic delta-cert — port a 2nd profile through the conformance contract; bounded re-cert | 🟡 Partial | Mechanism **exists + exercised**: MissionProfile adapter, toy_inspect + the new 4-deputy apophis_test4 on the same core. Formal delta-cert record not produced. |

### The RL layer (why it isn't on the critical path)

The RL recovery brain is **Class-C monitored (non-safety-critical)**. It gates **no** FR phase: safety is invariant to policy quality (0 *executed* unsafe across base/hc/hc2, including hc2's 7,345 proposals). More fine-tuning is **not** on the critical path; better recovery needs an architectural change (centralized-critic MAPPO / per-class value heads) + richer ATT/PROP telemetry. If flown, the RL enters at **FR4** as net WCET/SEU characterization — never as a training task.

### Critical path & long poles

- **Critical path:** FR0 → FR1 → FR2 → FR3 → FR5 → FR6 (FR4 parallels FR3; FR7 after FR6).
- **Hardware is the gate:** PIL/HIL on GR712/GR740 + Zynq/Versal (FR3–FR5) is the TRL 5→6 work.
- **Standing long pole:** the measured RF/ISL link budget (feeds the connectivity/consistency CBFs) — flagged across all three source docs.
- **Detection caveat (honest):** PROP detection Pd ≈ 0.98 with high, variable latency (sparse TCM-burn evidence) — a real limitation of the *detector*, independent of the RL.
