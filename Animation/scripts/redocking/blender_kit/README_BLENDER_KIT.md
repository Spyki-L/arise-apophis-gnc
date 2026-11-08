# ARISE · Apophis — Blender 4K Mission Film Kit

Photoreal Cycles film of the full mission arc (150 s @ 30 fps, 3840×2160),
in the style of `Apophis Mission_Inital.mp4`. Every trajectory, attitude,
tumble state, strobe exposure and capture time comes from the saved mission
datasets — nothing is hand-animated.

## Quick start (no Blender install needed — the script bootstraps it)
Every script here runs with **plain Python** (press Run in your IDE).
`build_mission_film.py` looks for Blender (PATH, `%ProgramFiles%\Blender Foundation`,
`BLENDER_EXE` env var, or `blender_kit/_blender/`); if none is found it downloads a
portable Blender 4.2.3 into `blender_kit/_blender/` (~350 MB, once) and re-launches
itself inside it.

1. `python export_blender_data.py`  — only if the sim data changed (already run)
2. `python build_mission_film.py`   — builds + saves `arise_mission.blend` (already run)
3. `render_4k.bat`                  — the 4K render, resumable  ← **start here**
4. `python assemble.py`             — frames → `ARISE_mission_blender_4K.mp4`

Preview one shot first (~2 min):
`python build_mission_film.py --render s4 --lowres --samples 24`

Frames are reused on re-run (crash-safe resume). If render settings change
(e.g. after a `--lowres` preview) the old frames are purged automatically, so a
preview can never contaminate the final render.

## CAD models I can use (drop into assets/, any of .blend .glb .fbx .obj .stl)
| file stem   | used for                              | fallback if missing            |
|-------------|---------------------------------------|--------------------------------|
| `deputy`    | the 6U XB6 CubeSat — auto-scaled to 0.34 m | parametric 6U stand-in |
| `carrier`   | white carrier / kick stage (shot 1)   | white cylinder + engine stand-in |
| `espa_ring` | chief ESPA ring with RAFTI ports      | procedural ring (0.5334 m, 8 ports) |

**Status of your supplied CAD** (verified by import test):
* `cubesat_web_decomp.glb` → copied to **`deputy.glb`** ✔ (41 meshes, 160k tris)
* `espa_web_decomp.glb` → copied to **`espa_ring.glb`** ✔ (41 meshes, 71k tris)
* `cubesat_web.glb` / `espa_web.glb` — Draco-compressed, fail headless import (not used)
* `*.stp` — Blender has no STEP importer (not used; the GLBs carry the same geometry)
* `Earth.stl` / `earth.obj` — not needed; the kit's NASA-textured sphere looks better
* `carrier.*` — **not supplied**, shot 1 uses the stand-in. Drop `carrier.glb` in
  `assets/` and re-run step 2 to upgrade it.

Apophis needs **no CAD** — the real radar shape model is exported to
`data/apophis.obj` automatically. Earth day/night textures (NASA Blue/Black
Marble) are bundled in `assets/`.

## Shots (all data-driven)
S1 0–20s Earth departure · S2 20–40s real ECI trajectories (chief white /
Apophis red) · S3 40–65s separation (real 6-DOF + attitudes) · S4 65–105s
CA ±2 s strobe pass (real tumble, 136 real exposures as light flashes) ·
S5 105–120s STM return arcs · S6 120–140s RL dock (slow-mo captures at the
real capture times) · S7 140–150s finale + real mission stats.


## Running on HPC (recommended — ~2 h instead of ~50)
See `hpc/README_HPC.md`: a 30-task SLURM array (account `jekan`, partition `standard`)
renders the 4500 frames in parallel, then `assemble_film.slurm` builds the mp4.
The `.blend` has its textures packed, so it's a single self-contained file to rsync.

## Render time — read this before committing to a long run
The film is **4500 frames** (0–4499) = 150 s at 30 fps.

| shot | frames | count |
|---|---|---|
| s1 departure | 0–599 | 600 |
| s2 orbits | 600–1199 | 600 |
| s3 separation | 1200–1949 | 750 |
| s4 imaging | 1950–3149 | 1200 |
| s5 return | 3150–3599 | 450 |
| s6 dock | 3600–4199 | 600 |
| s7 finale | 4200–4499 | 300 |

Your console prints the device on every run:
`[render] engine=CYCLES device=GPU samples=96 res=3840x2160@100%`
If that says **device=CPU**, run `python check_gpu.py` — on CPU a 4K/96-sample frame
takes ~40–55 s, i.e. **50–70 hours** for the whole film.

### Speed levers (each is a one-line change at the top of build_mission_film.py)
| change | effect | quality |
|---|---|---|
| `--samples 32` (or `SAMPLES = 32`) | ~3× faster → ~17 h on CPU | near-identical; the denoiser carries it in space scenes |
| GPU enabled | 3–6× faster than CPU | identical |
| `ENGINE = "BLENDER_EEVEE_NEXT"` | minutes, not hours | ~90 % of the look for space; no path-traced GI |
| render shot by shot | spread over sessions | identical (`--render s4`) |

Recommended if you're short on time: **`--samples 32` + GPU**, or Eevee for a
same-day cut. `use_persistent_data` is already on, which keeps the 160+ imported CAD
objects resident between frames instead of re-syncing them every frame.

Renders resume automatically — you can stop and restart any time, and changing
settings purges mismatched frames so a preview never contaminates the final.

## Render settings & time
Cycles, 96 samples, OpenImageDenoise, AgX. GPU auto-detected (OPTIX→CUDA→HIP→
METAL→oneAPI), CPU fallback. ~6–12 s/frame on a modern GPU → 8–15 h for all
4500 frames; halve with `--samples 48`. PNG frames land in `render/` and
existing frames are SKIPPED on re-run (crash-safe resume). Preview a shot:
`blender -b -P build_mission_film.py -- --render s4 --lowres --samples 24`

Tweaks: captions on/off `CAPTIONS` in build_mission_film.py; engine via
`ENGINE = "BLENDER_EEVEE_NEXT"` for a fast draft pass.
