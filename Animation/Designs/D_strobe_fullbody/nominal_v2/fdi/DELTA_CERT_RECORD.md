# ARISE CORTEX FDIR — Delta-Certification Record (FR7)

*§6.8 conformance contract · generated 2026-07-08T00:35:18Z · `fr7_conformance.py`*

## Profile-independent safety core (one binary across missions)

| core file | SHA-256 (16) |
|---|---|
| `rta.py` | `7b529f65354714a9` |
| `behavior_tree.py` | `cdb7e8bffbfc19ee` |
| `planner.py` | `d3aa681b62d5bea2` |
| `peer_cbf.py` | `38d4875c97b6584f` |
| `csrc/vsc_spine.c` | `8139ce32fc58c147` |
| `csrc/vsc_spine.h` | `7a3e8e5cbe715952` |

> These SHAs are identical across every profile evaluated -> one safety-critical binary across missions. The FR2 machine-checked invariants (RTA soundness, CBF invariance, BT LTL spec, planner WCET cap) therefore carry over UNCHANGED (see FORMAL_VERIFICATION.md).

## Per-profile conformance (C1–C6)

### ARISE_Apophis_DesignD  ·  8 deputies  ·  `profiles/apophis.json`

| Obligation | Status | Detail |
|---|---|---|
| **C1** | PASS | DynamicsModel: mean_motion=0.0001 rad/s (CW-Hill); keep-out CBF = peer_cbf (profile-independent form). Regime-CBF certification = author obligation (§6.9.3). |
| **C2** | PASS | ActuationModel: dV budget=10.0 m/s bounded -> safety predicate + reach well-defined. |
| **C3** | PENDING | VSC: contingency-table entries=0 (safe-hold + CBF-safe disposal already in code). Tier-1 table is profile DATA — author per §6.9.3 (currently stub). |
| **C4** | PASS | ObjectiveModel: metric='facet_stereo_coverage', bounded target=0.85 in [0,1], tiers=[0.15, 1.0] (performance, not certified). |
| **C5** | PASS | DeadlineModel: form_window=3600.0 s; t_go derivable (inf permitted). |
| **C6** | PENDING | CommsModel: ISL declared ('S-band+crosslink'); quantitative latency/topology bounds not yet in profile schema — supply per §6.9 (declaration only). |

**Verdict:** DELTA REQUIRED before flight: C3, C6

### ARISE_Apophis_Test4  ·  4 deputies  ·  `profiles/apophis_test4.json`

| Obligation | Status | Detail |
|---|---|---|
| **C1** | PASS | DynamicsModel: mean_motion=0.0001 rad/s (CW-Hill); keep-out CBF = peer_cbf (profile-independent form). Regime-CBF certification = author obligation (§6.9.3). |
| **C2** | PASS | ActuationModel: dV budget=10.0 m/s bounded -> safety predicate + reach well-defined. |
| **C3** | PENDING | VSC: contingency-table entries=0 (safe-hold + CBF-safe disposal already in code). Tier-1 table is profile DATA — author per §6.9.3 (currently stub). |
| **C4** | PASS | ObjectiveModel: metric='facet_stereo_coverage', bounded target=0.85 in [0,1], tiers=[0.15, 1.0] (performance, not certified). |
| **C5** | PASS | DeadlineModel: form_window=3600.0 s; t_go derivable (inf permitted). |
| **C6** | PENDING | CommsModel: ISL declared ('S-band+crosslink'); quantitative latency/topology bounds not yet in profile schema — supply per §6.9 (declaration only). |

**Verdict:** DELTA REQUIRED before flight: C3, C6

### GEO_Inspection_Swarm  ·  4 deputies  ·  `profiles/toy_inspect.json`

| Obligation | Status | Detail |
|---|---|---|
| **C1** | PASS | DynamicsModel: mean_motion=7.29e-05 rad/s (CW-Hill); keep-out CBF = peer_cbf (profile-independent form). Regime-CBF certification = author obligation (§6.9.3). |
| **C2** | PASS | ActuationModel: dV budget=50.0 m/s bounded -> safety predicate + reach well-defined. |
| **C3** | PENDING | VSC: contingency-table entries=0 (safe-hold + CBF-safe disposal already in code). Tier-1 table is profile DATA — author per §6.9.3 (currently stub). |
| **C4** | PASS | ObjectiveModel: metric='viewsphere_coverage', bounded target=0.95 in [0,1], tiers=[0.05] (performance, not certified). |
| **C5** | PASS | DeadlineModel: form_window=7200.0 s; t_go derivable (inf permitted). |
| **C6** | PENDING | CommsModel: ISL declared ('crosslink'); quantitative latency/topology bounds not yet in profile schema — supply per §6.9 (declaration only). |

**Verdict:** DELTA REQUIRED before flight: C3, C6

## Safety-transfer statement

Per §6.8: a profile that discharges C1–C6 with the CBF (C1) and VSC (C3) certified for its regime inherits the SAFETY_CASE top claim unchanged, because the simplex argument is profile-independent and the only profile-specific safety objects are the CBF and the VSC.

## Delta to certify (any profile): C3, C6

Bringing a new mission onboard is a DELTA-CERT: re-verify the profile data + the regime VSC/CBF (the pending items), re-run this suite, re-baseline the Monte-Carlo — WITHOUT re-certifying the agent core.
