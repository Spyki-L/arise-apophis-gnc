# Live FDIR Fault Lab — six-window run guide

**Run this file:** `live_fault_lab.py`  (press Run in Visual Studio — **no arguments**;
`numpy` + `matplotlib` auto-install on first run).

A full-blown, multi-window LIVE fault lab. It drives the **real FDI core** (one
`fdi_core.FDICore` per deputy, all six channels) under a **randomized, high-fidelity
fault campaign** and animates **nine synchronized windows** off a single, deliberately
**slow** mission clock, so you can watch every fault be detected, isolated and acted on.

**Truth vs estimate.** Faults perturb a **true** state (an unmodeled accel for NAV, an
unmodeled torque for ATT, a sensor/link error for the rest). Each deputy runs onboard
**EKFs** — a 6-state Clohessy-Wiltshire relative-orbit filter (NAV) and a 6-state attitude
filter (ATT) — and **the FDI consumes only their estimates**. Truth is used solely to
generate the measurements and to *validate* (the truth-vs-estimate strip in W7). The EKFs
run at the **mission phase rate** — **separation and encounter sample fast, coast slow** —
and the **TCM burns are marked**.

## The faults (each injected into the channel it really corrupts)

The campaign is randomized (seeded) but **guarantees one episode of every channel** so
each detector is exercised, then adds random extra/overlapping episodes for stress:

| Fault | How it's injected (high-fidelity) | Detector that should trip | Typical action |
|-------|-----------------------------------|---------------------------|----------------|
| **NAV**  | bias on the rel-nav innovation (NIS ↑) | NAV CUSUM | REINIT → **recovered** |
| **ATT**  | attitude error + body-rate (tumble)    | ATT CUSUM | SAFE_HOLD |
| **COMM** | heartbeat age ramps past the timeout   | COMM CUSUM | **DEAD** → data relayed to chief |
| **PROP** | thruster Δv mismatch on a burn (**TCM misfire**) | PROP CUSUM | SAFE_HOLD |
| **FORM** | formation/task miss (off its slot) + peer-inconsistent | FORM (+PEER) CUSUM | MODE_SWITCH → **recovered** |
| **PEER** | Byzantine: peer-inconsistent + a lying ISL self-report | PEER CUSUM | SAFE_HOLD |

Nothing is faked: the core's own leaky-CUSUM detectors and HEALTHY→SUSPECT→FAULTED→DEAD
machine make every call. Correctable faults (NAV/COMM-glitch/FORM-mode) are **recovered**
and the deputy returns to HEALTHY; uncorrectable ones are safe-held; a true comm loss goes
**DEAD** and a healthy peer relays its science to the chief and re-images its target.

## The nine windows (all synced to one slow clock)

1. **W1 · MISSION OPS** — the complete deputy trajectories in the chief Hill frame
   (fading trails, D1–D8), fleet health, predicted-CA coverage (onboard vs ground),
   peer camera capture, onboard-vs-ground compute, event log.
2. **W2 · FDI CORE SCOPE** — the **8×6 live CUSUM detector matrix** (every deputy ×
   every channel) vs thresholds, deputy state lamps, fleet counts, and an
   **injected → detected** verification ticker. *This is the "is the core working" view.*
3. **W3 · SUBSYSTEMS** — per-deputy **RCS thruster rosettes**, **TCM burns and MISFIRES**,
   propellant, attitude error and thrust residual, plus a TCM timeline.
4. **W4 · AUTONOMY BRAIN** — a live **reasoning connectome** of how the executive thinks:
   it wires many independent information streams — the six FDI detectors plus fleet
   **consensus**, peer **vision**, the suspect's **ISL self-report** (distrusted when
   Byzantine), collision-safety **CBF** and the deadline-aware coverage **value** — into a
   pulsing **decision core**, forms a **belief** (the fault hypothesis + confidence), and
   **commits** an action. The active thought-path lights up with signal packets flowing
   along the synapses, the fleet belief state shows below, and a live **chain-of-thought**
   narrates the inference ("D6: NAV evidence 26× threshold · isolate = NAV · consensus +
   vision agree (0.91) · self-report consistent → REINIT").
5. **W5 · DEPUTY DEEP-DIVE** — the worst-off deputy in full: six CUSUM traces vs
   thresholds, residual bars, RCS rosette, ISL links, confidence, and the decision log.
6. **W6 · ISL COMMS NETWORK** — the link graph (chief + 8 deputies, laid out by their
   true chief-frame geometry), live ISL quality, the **DATA-TO-CHIEF relays** a
   misbehaving deputy's data is routed through, downlinks, and **peer-imaging hand-offs**.
7. **W7 · INNOVATION MONITOR** — the **live NAV and ATT EKF innovation sequences** for the
   most-anomalous deputy: the relative-nav innovation `y_k` (3 comp + NIS), the
   attitude-error innovation `σ_BR`, and the body-rate innovation `ω_BR` — each normalized
   by its own covariance √S (±1σ / ±3σ), **phase-paced** (dense in separation/encounter,
   sparse in coast), **TCM-marked**, with a **truth-vs-estimate** strip proving the
   estimate tracked truth while the fault drove the *true* state off. A fault drives the
   innovation out of its bound — the true filter-level fault signature, as it happens.
8. **W8 · COVERAGE & STROBE** — the closest-approach coverage mapped onto the asteroid two
   ways: a **3-D body** coloured by stereo coverage with the **Sun** direction and the
   **strobes lighting the shadowed dark side** (Design-D dark-side imaging) and the camera
   lines, and an **unwrapped lat-lon map** of the same, with sub-/anti-solar points. Plus
   per-deputy and overall coverage and the fraction of the dark side the strobes light.
9. **W9 · LOSS COMPENSATION** — when a deputy is lost, the survivors take up a *bit* of its
   share: which healthy deputies nudge their station toward the gap (a **ΔV**) and re-point
   their camera (a **torque / RW-momentum** bite), shown as a schematic + per-helper bars,
   with the **fraction of the lost footprint recovered** and the fleet ΔV/torque budget.

## How to run

1. Open `live_fault_lab.py` in Visual Studio (your Basilisk Python env) and press **Run**.
2. The nine windows open **each in their own process** and **auto-tile across your two
   monitors** (6 on the 2K, 3 on the 1080p) — nothing to drag. They render in PARALLEL
   across your CPU cores, so playback is smooth and any window stays draggable.
3. The whole mission plays over ~`PLAYBACK_SECONDS` (wall-clock); each window keeps pace
   by dropping frames if it falls behind, so it stays real-time and in lock-step.

**Smoothness / CPU.** matplotlib renders on the CPU (it cannot use the GPU); running one
window per process is how all your CPU cores are used in parallel. If your machine has few
cores, set `MULTIPROCESS = False` for a single-process fallback.

**Two-monitor layout.** Defaults assume the 1080p is to the RIGHT of the 2K. If it's
elsewhere, edit `SCREEN_SECOND` at the top: LEFT → `x=-1920`; ABOVE → `y=-1080`; BELOW →
`y=1440`. Set `AUTO_ARRANGE = False` to place them yourself.

Knobs at the top (no args): `SIM_FRAMES` (mission resolution — validated clean to ~1000;
higher may need re-tuning), `PLAYBACK_SECONDS` (how long the mission plays; raise to slow
down), `MULTIPROCESS`, `SEED` (different = different campaign), `LOOP_FOREVER`, `SHOW`
(toggle windows), and the `SCREEN_MAIN`/`SCREEN_SECOND` monitor geometry.

## Verification (printed + written to `faultlab_report.txt` on exit)

A headless pass of the default campaign (seed 7, 300 frames) gives:

```
chan   detected  episodes  median latency [frames]
NAV    YES       2         10
ATT    YES       2         1
COMM   YES       1         8
PROP   YES       2         6
FORM   YES       1         3
PEER   YES       2         1
ALL SIX DETECTED: True
onboard re-score ~4.3 ms  vs  ground ~400 ms  (~93x faster onboard)
```

Each injected fault trips **its own** channel's CUSUM (the truest core check), the state
machine reaches all four states, and the onboard decisions are graded against the
high-fidelity ground oracle (≈90× faster onboard). Change `SEED` to re-verify on a fresh
random campaign.

## Files

`live_fault_lab.py` (run this) · `faultlab_engine.py` (real FDI core per deputy +
randomized campaign + subsystem/ISL/decision model) · `faultlab_windows.py` (W2 scope,
W5 deep-dive) · `faultlab_subsystems.py` (W3 subsystems, W6 ISL network) ·
`faultlab_blockdiagram.py` (W4 architecture) · `faultlab_innovations.py` (W7 innovations) ·
`faultlab_coverage.py` (W8 coverage/strobe) · `faultlab_loss.py` (W9 loss-compensation) ·
`faultlab_ekf.py` (the per-deputy NAV/ATT EKFs) · `faultlab_encounter.py` (coverage-map +
compensation analytics) · `cockpit_ops.py` (W1 mission ops, reused).
The autonomy itself is the validated F0–F10 package in this folder; the lab only injects
faults and visualizes the real core's response.

## Troubleshooting

- *Too fast / too slow* → raise / lower `PLAYBACK_SECONDS` (the whole-mission wall-clock length).
- *Screen too small for six windows* → set some entries in `SHOW` to `False`.
- *Windows stack on top of each other* → your screen isn't 1920×1080; just drag them
  apart (the tiling offsets are a best-effort hint).
- *Want a specific fault sequence* → set a `SEED` you like (the console prints the
  campaign), or edit `faultlab_engine.build_campaign`.
