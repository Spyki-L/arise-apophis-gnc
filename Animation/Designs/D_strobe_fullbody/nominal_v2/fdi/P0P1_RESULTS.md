# P0 + P1 — built, run, and validated

*Design D · `nominal_v2/fdi/` · 2026-06-22*

This is the implementation + evidence record for **Phase 0 (causal rehome)** and
**Phase 1 (defect fixes)** of the FDI integration plan. Everything below was
executed and validated against the **real 8-deputy fleet bundles** (numpy; no
Basilisk needed — the FDI layer is filter-agnostic).

## What was built

| File | Role |
|---|---|
| `fdi_core.py` | Causal, self-contained engine (no `GNC` import). Same `FDICore`/`FDIConfig`/`FDIState`/`FaultType`/`step()` API as the original → drop-in. Adds the P1 PROP CUSUM + FORM/PEER episode latch, plus `legacy_isolation` to reproduce the old behaviour for before/after. |
| `channels.py` | Causal residual builders replacing the 3 acausal shortcuts: `predicted_tca` (commanded epoch, not truth min-range), `causal_form_residual` (own estimate → commanded station), `PeerConsensusBuffer` (streaming, latency-aware, MAD-floored consensus that also supplies the causal FORM scale). |
| `run_fdi_nominal.py` | Causal onboard-style driver: processes all deputy samples in **global time order** so the peer consensus is strictly causal; writes `health_history.npz`. |
| `run_selftest_p01.py` | 7 deterministic core proofs (the canonical unit test). |
| `mc/isolation_before_after.py`, `mc/build_cache.py` | Real-fleet before/after isolation confusion matrix (legacy vs fixed). |

## Validation results (all executed)

**1. Core unit proofs — `run_selftest_p01.py`: 7/7 PASS**
```
PASS T1 healthy -> HEALTHY/NONE
PASS T2 PROP: FIXED=PROP/FAULTED  LEGACY=NONE   (the 0% defect)
PASS T3 NAV: both -> NAV
PASS T4 FORM_LOCAL latch: FIXED=FORM  LEGACY=PEER   (the 99.5% mislabel)
PASS T5 NAV_GLOBAL (FORM high, PEER quiet) -> NAV
PASS T6 PEER_ONLY (PEER high, FORM quiet) -> PEER
PASS T7 causality: state@160 independent of steps>160
```

**2. Channel builders — `channels.py` self-test: 6/6 PASS**
(causal PEER homogeneous-fleet safe; true outlier scores high; latency excludes
un-arrived reports; causal FORM finite; predicted_tca = commanded epoch.)

**3. P1 PROP fix on REAL fleet data — `mc/isolation_before_after.py` (120 trials/scenario):**
```
                    NONE   NAV   ATT  PROP  COMM | diag%
 BEFORE (legacy) PROP   120     0     0     0     0 |   0.0     <-- reproduces fdi_mc_isolation.tex
 AFTER  (fixed)  PROP     1     0     0   119     0 |  99.2
 NAV/ATT/COMM diagonals: 100% in both legacy and fixed.
```
**PROP isolation: 0.0% → 99.2% on real data.**

**4. Causal driver on REAL nominal fleet — `run_fdi_nominal.py`: 8/8 HEALTHY**
(both with FORM/PEER **off** and **on**) → the causal, onboard-style run
reproduces the offline healthy-fleet result with zero false alarms.

## Key implementation notes

- **PROP fix.** A leaky CUSUM `S_prop` accumulates Δv-mismatch evidence **at
  burns** and only gently leaks between sparse burns (no `ν` subtraction when
  there is no burn — mirrors the NAV no-measurement branch). PROP is then a
  first-class label (`prop_first_class`), no longer gated behind nav/att/form.
- **FORM/PEER fix.** Detection *timing* uses the instantaneous candidate (early
  PEER warning preserved); the settled *label* (at FAULTED) is the precedence
  over the per-episode **latched** channel set, so an early PEER-only reading no
  longer overrides a FORM trip that happened earlier in the same episode.
- **Causal FORM scale.** The nominal predicted miss is ~8.4 km — a large but
  **common-mode** CW-vs-Basilisk modelling offset. The offline code cancels it
  with a whole-timeline fleet median (acausal). The causal driver derives the
  same scale from the **consensus buffer** (median of peer misses received so
  far), so a homogeneous fleet → r_form ≈ 1 and stays HEALTHY, **causally**.
  This is the P0 placeholder; the principled replacement is the chance-
  constrained **terminal-reachability FORM (Pillar III / P3)**, which normalises
  by predicted uncertainty instead.

## How to run (on the Basilisk/Windows machine or any numpy env)

```bat
cd Animation\Designs\D_strobe_fullbody\nominal_v2\fdi
python run_selftest_p01.py                                  REM 7/7 core proofs
python channels.py                                          REM 6/6 channel proofs
python run_fdi_nominal.py --data <dir with the 8 deputy bundles>      REM causal nominal -> 8/8 healthy
python mc\isolation_before_after.py --data <bundle dir> --n 300       REM PROP 0% -> ~99%
```
The bundle dir is wherever the per-deputy `apophis_swarm_FDI_rgazel_D*.npz`,
`apophis_rendezvous_full_D*.npz`, and `dep0*_logs.npz` live (e.g.
`GNC/Scenario/Apophis/`, or a Design D nav-stage once regenerated).

## Sandbox caveat encountered (for the next session)
The OneDrive-mounted folder served **stale/truncated/corrupted** copies of
just-edited `.py` files to the sandbox shell (a `snap = R` tail corruption, and
Python importing a pre-edit `.pyc`). The editor (Read/Write) copies are
canonical and correct. Validation here was therefore run from a **sandbox-local
`/tmp` copy** with `python -B` (no bytecode cache). New files synced fine;
edited files lagged. The deliverables in this folder are the canonical,
validated logic.

## Status vs roadmap
- **P0 (rehome + causal):** DONE — core + channels + driver, healthy on nominal causally.
- **P1 (defect fixes):** DONE — PROP 0→99.2% (real data) + FORM/PEER latch (proven); `legacy_isolation` preserved for before/after.
- **Next:** P3 reachability FORM (replaces the consensus-median scale placeholder), then P2 conformal thresholds; a real-data FORM_LOCAL/NAV_GLOBAL before/after needs the ramp-injection harness (synthetic already proven here).
