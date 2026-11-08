# What the real re-docking actually looks like
(from your `dock_rollout.npz` — use these if you want the clip to match the sim,
or to sanity-check what Seedance gives back)

| quantity | real value |
|---|---|
| deputies | 8 × 6U CubeSat, 0.34 × 0.20 × 0.20 m |
| chief | ESPA ring, radius 0.5334 m, height 0.80 m |
| docking ports | 8, evenly spaced at 0°/45°/90°/…/315° around the ring |
| start of the dock phase | deputies 24–40 m out (mean ≈ 35 m) |
| final standoff | ~0.88–0.90 m from ring centre — i.e. hard against the port collar |
| closing speed at capture | ≤ 0.05 m/s — centimetres per second |
| lateral drift allowed | ≤ 0.03 m/s |
| attitude error allowed | ≤ 12° |
| docking ORDER (real) | D1 → D8 → D7 → D6 → D2 → D3 → D4 → D5 |
| spacing between captures | 2 close together, pause, 3 more, then two long gaps |
| all captured | 8 / 8 |

## Why the prompt says what it says
* "a few centimetres per second" and "almost still" come from the real 0.05 m/s cap —
  this is the single most common thing AI video gets wrong (it docks like a car parking).
* "uneven intervals, two arrive close together, then a pause" mirrors the real capture
  order above, which is clustered rather than evenly spaced.
* "settles with a tiny rebound" matches a soft-capture latch, not a collision.
* The ring is described as *squat and cylindrical with 8 ports* because that is the
  actual ESPA geometry from your CAD — generic "space station ring" prompts drift into
  a large thin torus, which is wrong.

## If you want it even closer to the sim
Add this line to the prompt:
"The CubeSats dock in this order: first one at the near port, then two more in quick
succession on the far side, a pause, then the rest one at a time with the last one
arriving noticeably later."
