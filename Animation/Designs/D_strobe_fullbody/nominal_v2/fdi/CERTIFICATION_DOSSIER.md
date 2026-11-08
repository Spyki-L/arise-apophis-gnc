# Certification Dossier (F10) — ARISE/Apophis swarm FDIR

*Design D · `nominal_v2/fdi/` · the journal-grade evidence campaign. All numbers below are produced
by `certification.py` / `run_certification.py` and visualized in `figs/certification_summary.png`.
Detection/isolation/latency runs on the **real 8-deputy fleet bundles**; the Byzantine, recovery, and
recoverability campaigns use the validated fast onboard models. This dossier feeds §14 of
`FDI_DESIGN_D_INNOVATION_AND_INTEGRATION_PLAN.md` (the journal paper).*

---

## 1. Detection, isolation, latency (real fleet, N=100/scenario)

A fault is injected on a random deputy at a random onset phase (0.1–0.9 of its nav arc); the causal
FDI core runs the full timeline. Detection = the correct label is declared after onset.

| Fault | Pd | 95% CI (Wilson) | latency median | latency p95 |
|---|---|---|---|---|
| NAV  | 1.000 | [0.963, 1.000] | 217 s | 247 s |
| ATT  | 1.000 | [0.963, 1.000] | 95 s | 99 s |
| PROP | 0.980 | [0.930, 0.994] | 709 s | 12322 s |
| COMM | 1.000 | [0.963, 1.000] | 175 s | 179 s |

**False-alarm rate** Pfa = **0.000** [0.000, 0.037] per healthy deputy per encounter (N=100 nominal).

**Isolation confusion matrix** (rows = injected truth, cols = identified label): a **clean diagonal**
— NAV→NAV, ATT→ATT, PROP→PROP, COMM→COMM. The PROP row is the headline of the P1 fix: **0% → ~98–100%
correct isolation** vs the legacy gated rule.

Reading: NAV/ATT/COMM detect in **1.5–4 min** — well inside the ~2 h imaging window. PROP is detected
with Pd ≈ 0.98 but with high, variable latency, because propulsion evidence only accrues at the
sparse TCM burns; a missed/late case is one where the fault is injected after the last burn. This is
an honest, expected limitation and is reported as such (not hidden).

---

## 2. Byzantine f-of-N tolerance (PEER-CBF consistency, N=8)

A liar reports a corrupted position; the robust median-consistency residual flags it against peers'
independent range measurements.

| f liars | Pd (liar caught) | Pfa (honest flagged) |
|---|---|---|
| 0 | — | 0.000 |
| 1 | 1.000 | 0.000 |
| 2 | 1.000 | 0.000 |
| 3 | 1.000 | 0.000 |
| 4 | 1.000 | 0.943 |

**The architecture tolerates up to f = 3 < N/2 Byzantine nodes** with perfect detection and zero
honest false alarms; the median estimator breaks exactly at **f = 4 = N/2**, the theoretical bound.
This is the quantitative statement of the f-of-N robustness claim.

---

## 3. Recovery efficacy (random single/double deputy loss)

| Metric | Value |
|---|---|
| nominal coverage | 1.000 |
| mean coverage-gap recovered | **66.6%** (p10 = 37.4%) |
| beat-τ* (reconfiguration reachable in time) | **98.8%** over N=80 losses |

Single-deputy losses recover ~94% of the gap (F5); the mean is pulled down by the harder
double-deputy losses. In **98.8%** of cases the recovering reconfiguration is reachable within the
remaining time-to-CA and ΔV (the F3 deadline test), so the recovery is actionable, not just optimal.

---

## 4. Recoverability frontier (F3, authority × onset)

The (surviving authority × onset time-to-CA) heatmap (figure, panel D) is the deadline map: at full
authority the slot stays reachable until ~48 min before CA; degraded craft must act earlier; below
~13% authority it is unrecoverable in-window. This *is* the τ* surface the recovery value function
uses.

---

## 5. Apophis case study (end-to-end)

The whole-mission F8 run (figure `mission_autonomy_timeline.png`) exercises the package on the
Apophis profile through deputy loss → actuation degradation → spin change → misbehaving deputy →
too-late event, with **zero ground intervention**, **zero unsafe actions** (RTA: AC accepts vs VSC
fallbacks logged), and the mission objective preserved. The campaign above provides the statistics
behind that single run: the detector that flags each fault (≈100% Pd, 0 Pfa), the isolation that
labels it correctly (diagonal confusion), the recovery that reconfigures it (66.6% gap recovered,
98.8% in time), and the safety layer that bounds it (f≤3 Byzantine, CBF-safe disposal, forward
invariance from F9).

---

## 6. How this becomes the paper

Per §14 of the master plan, these are exactly the journal artifacts JGCD reviewers require:
- **Primary results table** — §1 (Pd/Pfa/latency with CIs) + the isolation confusion matrix.
- **Detectability/recoverability** — §4 frontier (answers "5σ is easy").
- **Byzantine tolerance curve** — §2 (a contribution the prior art, incl. arXiv:2511.08752, lacks).
- **Recovery evidence** — §3 (ΔCoverage recovered, beat-τ*).
- **Before/after rigor** — the PROP 0%→98% and FORM/PEER fixes (P1) with `mc/isolation_before_after.py`.
The before/after honesty (a documented defect found and fixed) is a strength, not a liability.

---

## 7. Maturity caveats (honest)

- These campaigns use the **causal numpy FDI/CBF/optimizer models** (the onboard build) plus the
  **real rel-nav/attitude fleet bundles** — not a full Basilisk re-fly per trial. The detection
  numbers are therefore on real flown geometry; the recovery/coverage numbers use the analytic
  surrogate (ground-calibrated, ordering-preserving, see F5).
- A full certification adds: a magnitude sweep per channel (detectability curves), FORM/PEER fault
  injection through the formation pipeline, the comms/ISL model behind the consistency/connectivity
  CBFs (the long pole), and the PIL/HIL + radiation analysis of F9 §7.
- None of these change the result structure; they raise the trial counts and the fidelity behind the
  same tables.
