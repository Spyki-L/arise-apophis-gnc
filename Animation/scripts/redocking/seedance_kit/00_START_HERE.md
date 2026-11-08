# Seedance 2.5 — add the RE-DOCKING phase to your existing mission video

Goal: generate ONE clip of the eight CubeSats docking back onto the ESPA ring,
in the same look as `Apophis Mission_Inital.mp4`, then splice it into that video.
Nothing else about the original video changes.

## Why these particular files
Your existing video never shows the ESPA ring or any docking — so that hardware
cannot be referenced from the video itself. It comes from your CAD, rendered here
into reference images Seedance can actually read. The CubeSat, meanwhile, IS in your
video, so it's referenced from your own footage to guarantee a match.

## Step 1 — upload (Multi Reference mode, this exact order)
Reference Images:
| order | file | becomes |
|---|---|---|
| 1 | `03_reference_images/img1_cubesat_clean.png` | `@Image1` — your CubeSat, straight from your video |
| 2 | `03_reference_images/img2_espa_ring_with_8_docked.png` | `@Image2` — the end state to aim at |
| 3 | `03_reference_images/img3_espa_ring_empty_ports.png` | `@Image3` — the ring with empty ports |
| 4 | `03_reference_images/img4_cubesat_cad_hero.png` | `@Image4` — CubeSat CAD, second angle |

Reference Videos:
| order | file | becomes |
|---|---|---|
| 1 | `02_reference_video/ref1_cubesat_beauty_5s.mp4` | `@Video1` — locks style, lighting, motion |
| 2 | `02_reference_video/ref2_fleet_asteroid_6s.mp4` | `@Video2` — fleet-in-space look |

Totals: 11.5 s of video (limit 30 s) and 1.8 MB (limit 200 MB) — well inside limits.

## Step 2 — prompt
Paste `01_prompt/PROMPT_redock_main.txt` (everything under the dashed line).
Settings: 16:9, highest resolution, 20–30 s.

Which prompt to use, in order of preference:
1. `PROMPT_redock_main.txt` — detailed: hardware, 4-beat choreography, camera,
   lighting, physics, and a long negative list. Start here.
2. `PROMPT_redock_SHORT.txt` — ~180-word version if the box rejects long prompts,
   or as a second attempt if the detailed one produces something over-complicated.
3. `PROMPT_redock_splitA_approach.txt` + `...splitB_capture.txt` — two shorter clips
   if a single 30 s generation drifts. Turn on **Return Last Frame** for A, then
   upload that returned frame with B and add "continue seamlessly from @Image5".

`ACCURACY_NOTES.md` lists the real numbers from the simulation (approach 24–40 m,
capture at ≤0.05 m/s, real docking order D1→D8→D7→D6→D2→D3→D4→D5) and explains why
the prompt is worded the way it is — useful if you want to tune it further.

## Step 3 — splice it in
Save the generated file as `docking_clip.mp4` in this folder, copy
`Apophis Mission_Inital.mp4` here too, then:

    python append_docking.py            # adds it to the end  ← recommended
    python append_docking.py --at 62    # or inserts it at 62 s, before the
                                        # labelled CubeSat outro

Output: `ARISE_mission_with_docking.mp4` — 4K, fps matched, one clean re-encode.
(Both modes tested and verified.)

## If the clip comes out wrong
- spacecraft changes shape between shots → add "identical rigid spacecraft, no design changes"
- it invents Earth or the asteroid → they're already in the NEGATIVE list; repeat them in the prompt
- docking too fast → add "extremely slow final approach, almost stationary"
- ring looks wrong → put `img3` first in the upload order so it becomes `@Image1`
