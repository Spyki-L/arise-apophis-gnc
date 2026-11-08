# FDI package — snapshot + Design D integration root

This folder brings the **multi-channel Fault Detection & Isolation (FDI)** work
into the Design D `nominal_v2` package so it can become an **onboard, active,
whole-mission fault-management system** for the chief and the 8 deputies.

The master plan for that is **`FDI_DESIGN_D_INNOVATION_AND_INTEGRATION_PLAN.md`**
(in this folder). Read it first.

## Implemented so far — P0 + P1 (DONE, validated on real fleet data)

See **`P0P1_RESULTS.md`** for the full evidence. New, runnable modules:

- `fdi_core.py` — causal, self-contained engine (drop-in API) + the P1 fixes
  (PROP first-class CUSUM; FORM/PEER settle-time latch) + `legacy_isolation`
  switch for before/after.
- `channels.py` — causal builders (predicted-TCA, own-estimate FORM, streaming
  Byzantine-ready `PeerConsensusBuffer` that also supplies the causal FORM scale).
- `run_fdi_nominal.py` — causal, global-time-order onboard-style driver.
- `run_selftest_p01.py` — 7/7 core proofs; `mc/isolation_before_after.py` — real-data confusion.

Validated: core proofs **7/7**, channel proofs **6/6**, causal driver **8/8 HEALTHY**
on nominal, and **PROP isolation 0.0% → 99.2%** on the real fleet (the documented
defect, fixed). The original `_source_snapshot/` files are unchanged.

## Provenance

Copied **2026-06-22** from `ARISE Arch/GNC/` (md5-verified faithful copies).
The originals remain the canonical, runnable versions in their original
location; this is a working snapshot + the new design root.

```
_source_snapshot/                       (faithful copies — do NOT run in place yet, see "Imports")
  fdi_core.py                           the 6-channel leaky-CUSUM FDI engine (filter-agnostic, numpy-only)
  apophis_swarm_fdi_main.py             OFFLINE integration: reads saved NPZ bundles, runs FDI over the arc, makes journal figs
  apophis_swarm_fdi_mc.py               baseline fault-injection MC (NAV/ATT/PROP/COMM), Pd/Pfa/delay
  apophis_swarm_fdi_mc_isolation.py     full confusion-matrix isolation MC
  apophis_swarm_fdi_mc_magnitude.py     fault-magnitude detectability sweep (Wilson CIs)
  apophis_swarm_fdi_mc_formpeer.py      FORM/PEER local-vs-global isolation experiment
  apophis_swarm_fdi_mc_peer_stress.py   homogeneous-fleet PEER false-alarm (MAD-floor) stress
  apophis_swarm_fdi_mc_plot.py          re-plot helper for the MC NPZ
  fdi_mc_tex_table.py                   LaTeX results-table generator
  FDI_anim.py                           FDI animation / visualization

FDI_Related_Work_and_Novelty.docx       prior-art review + candid novelty assessment (the bar to beat)
FDI_Journal_Revision_Roadmap.docx       reviewer-anticipating revision plan (JGCD target)
Peer_RelNav_Technical_Document.pdf       peer relative-navigation (the PEER/FORM data source)

reference_figs/                          current-performance visuals + result tables (for the writeup)
  fdi_mc_isolation.tex, fdi_mc_formpeer.tex, *.png
```

Large stale `.npz` data bundles (`apophis_swarm_FDI_*`, `fdi_mc_results.npz`, …)
were **intentionally not copied** — they are offline products of the old GNC run
and will be **regenerated from the Design D pipeline** (the `nominal_v2`
trajectories / nav / 6-DOF / separation / coverage products) per the plan.

## Current capability (honest baseline)

- **Detection** of NAV/ATT and **NAV_GLOBAL** common-mode is strong (≈100% in the MCs).
- **Two known defects** the plan fixes and proves:
  1. **PROP isolation ≈ 0%** — injected PROP faults label `NONE` 100% of the time
     (the `prop_bad AND (nav/att/form bad)` gate makes a *pure* prop fault un-isolable).
  2. **FORM_LOCAL settles on the wrong label 99.5%** — PEER fires first and wins the
     settled label; the local-vs-global rule only works in the global direction.
- **Scope gaps** the plan closes: it is **offline/acausal** (uses truth TCA + whole-timeline
  fleet median), **deputy-centric** (no real chief↔deputy consensus, no comms/Byzantine model),
  **detection/isolation only** (no Recovery — the missing "R"), and **active only on the
  near-flyby nav arc** (not the separation transient or the full mission).

## Imports (before running the snapshot in this new location)

The snapshot keeps its original imports (`from GNC.Navigation.fdi_core import …`
and `from apophis_swarm_fdi_main import …`). To run it from here, either keep
`ARISE Arch/` on `PYTHONPATH`, or wait for **Phase 0** of the plan, which
rehomes the engine as a causal `fdi/` module wired to `nominal_v2` signal taps.
