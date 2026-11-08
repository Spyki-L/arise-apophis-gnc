# ARISE · Apophis — Google Veo kit (TV-meeting cinematic)

Everything needed to generate a ~72 s cinematic in **Google Veo 3.1** (Gemini app,
Flow, or Vertex AI). Nine 8-second clips, then assemble in any editor.

## Why this kit exists
Veo cannot read CAD files. So your CAD has been **rendered into reference images**
("ingredients") that Veo *can* ingest — that's what locks the spacecraft design so it
looks the same in every clip. The real mission trajectories were likewise rendered
into plates you can feed in or drop straight into the edit.

## Folders
| folder | what it is |
|---|---|
| `01_prompts/` | 9 ready-to-paste prompts, one per shot, + the shared style/negative suffix |
| `02_reference_frames/` | 10 stills pulled from your existing render — use as ingredients to match its look |
| `03_cad/` | **Rendered CAD plates (use these in Veo)** + the original .glb/.obj for reference |
| `04_trajectory_plates/` | Real-trajectory graphics: orbits, separation, imaging formation, return/dock |
| `05_data/` | Mission facts, caption strip, voice-over paragraph, numbers as JSON |
| `06_edit/` | Shot list, timeline order, edit + troubleshooting notes |

## How to run it (Flow or Gemini app)
1. Open `01_prompts/01_shot_earth_departure.txt`, copy the PROMPT block.
2. Attach the ingredient image(s) named at the top of that file
   (Flow: "Ingredients to Video" / "Frames to Video"; Gemini: attach image).
3. Generate at **16:9, 1080p or 4K**. Veo caps each generation at 8 s.
4. Repeat for prompts 02 → 09. Regenerate any clip you don't like — they're independent.
5. Assemble in the order in `06_edit/EDIT_PLAN.md`, burn in the captions from
   `05_data/MISSION_FACTS.md`, add music, export 4K.

## Two tips that matter most
- **Reuse the same ingredient image** for every clip showing the same hardware —
  that's what keeps the CubeSat consistent shot to shot.
- **Don't ask Veo for on-screen text.** It garbles it. Add captions in the edit;
  the exact wording is already written for you in `05_data/MISSION_FACTS.md`.

## Accuracy note
Veo output is *artistic*, not simulation-accurate — fine for a TV meeting. The
physically exact version is the Blender kit in `../blender_kit/` (real datasets,
frame-exact); use that when accuracy matters.
