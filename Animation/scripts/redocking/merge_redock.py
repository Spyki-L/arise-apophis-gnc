"""
merge_redock.py — swap Final_Video.mp4's return/re-dock tail for the dashboard clip.

Final_Video.mp4 analysis (measured, not assumed):
    PHASE 3 · RETURN  starts at  85 s
    PHASE 4 · RE-DOCK starts at 100 s
    ends 128.9 s, still mid-dock, WITHOUT the MISSION COMPLETE card

So the default is: keep Final_Video 0-85 s, then append the new 60 s clip
(return 20 s + re-dock 32 s + end card 8 s)  ->  145 s total.

Put both files next to this script, then:
    python merge_redock.py                 # cut at 85 s (default)
    python merge_redock.py --at 100        # keep Final_Video's return, swap only the dock
    python merge_redock.py --xfade 1.0     # 1 s crossfade instead of a hard cut
"""
import os, sys, subprocess, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "Final_Video.mp4")
CLIP = os.path.join(HERE, "thesis_arc_out", "videos", "ARISE_returndock_4K.mp4")
OUT  = os.path.join(HERE, "ARISE_final_with_dashboard_redock.mp4")

CUT = 85.0
if "--at" in sys.argv:    CUT = float(sys.argv[sys.argv.index("--at") + 1])
XF = 0.0
if "--xfade" in sys.argv: XF = float(sys.argv[sys.argv.index("--xfade") + 1])

ff = shutil.which("ffmpeg")
if not ff:
    try:
        import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise SystemExit("ffmpeg not found — pip install imageio-ffmpeg")
for p, what in ((BASE, "Final_Video.mp4"), (CLIP, "ARISE_returndock_4K.mp4")):
    if not os.path.exists(p):
        raise SystemExit(f"missing {what}\n  expected at: {p}")

V = ("scale=3840:2160:force_original_aspect_ratio=decrease,"
     "pad=3840:2160:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1,format=yuv420p")

if XF > 0:
    fc = (f"[0:v]{V},trim=0:{CUT + XF},setpts=PTS-STARTPTS[a];"
          f"[1:v]{V}[b];"
          f"[a][b]xfade=transition=fade:duration={XF}:offset={CUT}[v]")
else:
    fc = (f"[0:v]{V},trim=0:{CUT},setpts=PTS-STARTPTS[a];"
          f"[1:v]{V}[b];[a][b]concat=n=2:v=1:a=0[v]")

print(f"[merge] Final_Video 0–{CUT:.1f}s  +  dashboard return/re-dock"
      + (f"  (xfade {XF}s)" if XF else "  (hard cut)"))
subprocess.check_call([ff, "-y", "-i", BASE, "-i", CLIP, "-filter_complex", fc,
                       "-map", "[v]", "-c:v", "libx264", "-preset", "medium",
                       "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT])
print("->", OUT)
