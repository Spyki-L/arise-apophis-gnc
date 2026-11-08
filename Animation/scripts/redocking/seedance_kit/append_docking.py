"""
append_docking.py — stitch the Seedance re-docking clip onto the original mission video.

1. Put the generated clip in this folder as  docking_clip.mp4
2. python append_docking.py           -> appends it to the end
   python append_docking.py --at 62   -> inserts it at 62 s instead (before the
                                          labelled CubeSat outro)
Output: ARISE_mission_with_docking.mp4 (4K, matched fps, single re-encode)
"""
import os, sys, subprocess, shutil, glob

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = None
for c in [os.path.join(HERE, "Apophis Mission_Inital.mp4")] + glob.glob(os.path.join(HERE, "*.mp4")):
    if "docking_clip" not in os.path.basename(c) and os.path.exists(c):
        SRC = c; break
CLIP = os.path.join(HERE, "docking_clip.mp4")
OUT = os.path.join(HERE, "ARISE_mission_with_docking.mp4")

ff = shutil.which("ffmpeg")
if not ff:
    try:
        import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise SystemExit("ffmpeg not found — install it, or: pip install imageio-ffmpeg")
if not SRC or not os.path.exists(SRC):
    raise SystemExit("Original video not found — copy 'Apophis Mission_Inital.mp4' next to this script.")
if not os.path.exists(CLIP):
    raise SystemExit("Put your generated clip here and name it 'docking_clip.mp4'")

at = None
if "--at" in sys.argv:
    at = float(sys.argv[sys.argv.index("--at") + 1])

V = "scale=3840:2160:force_original_aspect_ratio=decrease,pad=3840:2160:(ow-iw)/2:(oh-ih)/2,fps=30000/1001,setsar=1"
if at is None:                                   # append at the end
    cmd = [ff, "-y", "-i", SRC, "-i", CLIP, "-filter_complex",
           f"[0:v]{V}[a];[1:v]{V}[b];[a][b]concat=n=2:v=1:a=0[v]",
           "-map", "[v]", "-c:v", "libx264", "-crf", "17", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
else:                                            # insert at a timestamp
    cmd = [ff, "-y", "-i", SRC, "-i", CLIP, "-filter_complex",
           f"[0:v]{V},trim=0:{at},setpts=PTS-STARTPTS[a];"
           f"[0:v]{V},trim={at},setpts=PTS-STARTPTS[c];"
           f"[1:v]{V}[b];[a][b][c]concat=n=3:v=1:a=0[v]",
           "-map", "[v]", "-c:v", "libx264", "-crf", "17", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
print("[append] rendering…")
subprocess.check_call(cmd)
print("->", OUT)
