# FR3 (reachable part) — Static WCET analysis of the VSC safety spine

*Design D · `nominal_v2/fdi/csrc/` · the software-side of FR3: a **static worst-case-execution-time argument** for the MISRA-C safety spine (`vsc_spine.c`). The only FR3 item this does **not** close is the **measured cycles on the actual GR712/GR740** — that is hardware-gated (PIL). This analysis is the loop-bound / op-count layer that precedes and frames that measurement.*

## Why a WCET exists (the static argument)

Every safety-spine function satisfies the flight-code WCET preconditions **by construction**:

- **No recursion, no dynamic allocation, no unbounded loops.** Every loop bound is the compile-time constant **3** (the spatial dimension). There are no `while`/data-dependent loops.
- **Straight-line branch structure** in `rta_decide` (≤ 5 compares) and `vsc_contingency` (1 compare + 1 table load).
- **Fixed-point path** (`cbf_margin_fixed`) uses only int64 mul/shift/add — no FPU, deterministic timing.

Therefore each function's WCET is a **finite constant**, `O(1)` in the fleet size — the property flight qualification requires.

## Static bound + empirical anchor

Conservative LEON3FT @ 100 MHz cycle model (int op ≈ 1, FP mul/add ≈ 1–4, FP div ≈ 25 cyc); the empirical column is `bench_wcet.c` on this host scaled ×30 (the `SAFETY_CASE.md §4` x86→LEON factor). Cadence budget = 100 ms (10 Hz safety loop).

| Function | Loop bound | Worst-case ops | Static WCET est. | Empirical (LEON est.) | Margin vs 100 ms |
|---|---|---|---|---|---|
| `rta_decide` | none (≤5 branches) | ~8 int | < 0.10 µs | 0.036 µs | > 10⁶× |
| `cbf_margin` (f64) | 3 | ~16 FP | < 0.30 µs | 0.077 µs | > 3×10⁵× |
| `cbf_filter` (f64) | 3 (+1 div) | ~25 FP + 1 div | < 0.70 µs | 0.088 µs | > 1.4×10⁵× |
| `cbf_margin_fixed` | 3 | ~20 int64 | < 0.25 µs | 0.062 µs | > 4×10⁵× |
| `vsc_contingency` | none | ~3 int | < 0.05 µs | 0.028 µs | > 2×10⁶× |

Even the most conservative static estimate (the `cbf_filter` division) sits **~150,000×** inside the 10 Hz budget. The full spine tick (all functions once) is **< 1.5 µs static** vs 100 ms — consistent with the `SAFETY_CASE.md §4` provisional table (FDI ~3 ms, CBF ~13 ms include the fleet-loop and are budgeted separately; this spine is the per-pair inner kernel).

## What remains hardware-gated (the true FR3 item)

- **Measured cycles on the GR712RC / GR740** via PIL — pipeline, cache, and memory-latency effects that a host cannot reproduce.
- **Static-WCET tooling on the target** (aiT / Bound-T / OTAWA) with the loop-bound annotations above, then a MISRA-C static-analysis sign-off (needs the licensed checker; the code is written to MISRA-C:2012 mandatory/required style and compiles clean under `-std=c99 -Wall -Wextra -pedantic -Werror`).

Reproduce: `cd csrc && make check` (unit suite) and `./bench_wcet` (timing).
