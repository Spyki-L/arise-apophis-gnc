# Live FDIR Mission Cockpit — run guide

**Run this file:** `live_mission_cockpit.py`  (just press Run in Visual Studio — **no arguments**).

It plays your **nominal_v2 mission datasets** forward in time, runs the **onboard autonomy**
(FDI + recovery + the fast surrogate re-optimizer from this package) **live**, injects scripted
deputy failures, and on each event grades the onboard decision against a **high-fidelity ground
oracle** — animating everything and recording the **onboard-vs-ground compute time**.

## What you'll see

A **black mission-ops cockpit** (the `FDI_anim` aesthetic — neon palette, glow labels), default
`AESTHETIC = True`:

- **3D chief-LVLH (Hill) view — the COMPLETE deputy trajectories.** All 8 deputies are drawn as
  fading neon trails from dispense (co-located on the chief) fanning out to the ~10–13 km imaging
  formation, colour-coded by FDI state, with `D1…D8` labels at each head. The chief sits at the
  origin; **Apophis** is shown as a bearing on the box edge during the long approach (with live
  range) and drops into the box at closest approach. Two synced **R-T and R-N projections** sit
  below. When a deputy is lost, a healthy neighbour fires a dashed-blue **capture line-of-sight**
  onto the gap and **"clicks a picture"** (held on-screen ~12 frames).
- **Fleet health grid** (HEALTHY/SUSPECT/FAULTED/DEAD) — the live FDI state of all 8.
- **Predicted closest-approach coverage** — the stereo coverage the fleet will achieve at CA, live;
  it dips when a deputy fails and recovers after the onboard reconfiguration. The red squares are
  the **high-fidelity ground** scores at each event.
- **Peer camera capture** — the rendered snapshot the re-tasked deputy takes of the asteroid.
- **Onboard-vs-ground compute** — onboard score (ms) vs ground score (ms), log scale, with the
  live speed-up, and the **autonomy event log**.

Set `AESTHETIC = False` to fall back to the older split metrics-dashboard (optionally with a
PyVista photoreal window via `USE_PYVISTA = True`).

## How to run

1. Open `live_mission_cockpit.py` in Visual Studio (your Basilisk Python 3.11 env).
2. Press **Run**. That's it — no command-line arguments. Required packages (`numpy`, `matplotlib`)
   **auto-install on first run**, so you don't need to pip-install anything.

Knobs are at the top of the file (no args): `AESTHETIC` (the mission-ops cockpit, default on),
`N_FRAMES`, `SEC_PER_FRAME`, `LOOP_FOREVER`, `USE_PYVISTA` (optional photoreal window), and the
data paths (auto-resolved relative to the script).

## The reliability proof (onboard vs ground)

The script protects the **predicted CA-geometry stereo coverage** (the deadline-aware objective —
the 8.76 km/s flyby makes close-imaging a sub-second snapshot burst, so what matters is what the
fleet *will* achieve at CA given its current/reconfigured stations).

- **Onboard** = a fast analytic stereo surrogate on a facet subset — **~4 ms**.
- **Ground** = the full mesh + occlusion scorer — **~400 ms** (≈90× slower here).

On every event the cockpit shows the two agree to within ~**0.06–0.08** coverage (the onboard
surrogate is slightly optimistic, matching the documented ~1.4× Design-D calibration band), i.e.
**the fast onboard decision is reliable**. At the end it prints a summary and writes
`cockpit_compute_log.csv`.

**To use the *real* ray-traced ground oracle** (`MC_v2_run.score_draw` — detector + strobe +
occlusion-BVH + Hapke BRDF + photogrammetry): in `cockpit_engine.py → GroundOracle.score`, build
the `fleet`/`hapke` arguments and call `self.real.score_draw(...)` (the import is already wired via
`SCRIPTS_DIR`). The ground time then goes to **seconds–minutes**, and the speed-up to **>10,000×** —
the strongest version of the proof. The default (full mesh + occlusion) runs everywhere without
Basilisk so the cockpit always animates.

## Data it uses (from nominal_v2)

`data/trajectories/apophis_rendezvous_full_D*.npz` (positions, asteroid attitude), `data/anim/…`
(6-DOF), `nominal_v2/data/radar_mesh.npz` (shape → coverage facets), and
`nominal_v2/data/mc_v2_results_nominal_facetprob.npz` (coverage prior). If these are missing, run
the nominal_v2 pipeline first (`run_nominal_and_hifi.py`); the cockpit prints a clear message and
the paths are at the top of the run file.

## Scripted scenario (edit in `cockpit_engine.default_events`)

1. **D3 lost** → onboard re-optimizes the survivors + **D2 re-images** the gap (camera capture);
   graded against the ground oracle.
2. **D6 actuator degraded** → re-task within its reduced reach.
3. **D7 misbehaving** → recovery ladder (safe-hold / CBF-safe disposal) + re-opt.

## Files

`live_mission_cockpit.py` (run this) · `cockpit_ops.py` (the black mission-ops cockpit: complete
Hill-frame trajectories + telemetry, `FDI_anim` style) · `cockpit_engine.py` (mission player +
onboard/ground scorers + timing + Hill-frame relative trajectories) · `cockpit_viz.py` (optional
PyVista photoreal scene + legacy matplotlib dashboard + primitive shapes + camera render).
The autonomy itself is the validated F0–F10 package in this folder.

## Troubleshooting

- *"trajectory data not found"* → fix the paths in CONFIG or run nominal_v2 first.
- *PyVista errors* → set `USE_PYVISTA = False` (or uninstall); the matplotlib fallback always works.
- *Animation too fast/slow* → adjust `SEC_PER_FRAME` / `N_FRAMES`.
