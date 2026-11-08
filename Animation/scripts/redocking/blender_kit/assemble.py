"""Assemble render/frame_####.png into ARISE_mission_blender_4K.mp4 (H.264 CRF 16)."""
import os, subprocess, shutil
here = os.path.dirname(os.path.abspath(__file__))
ff = shutil.which("ffmpeg")
if not ff:
    try:
        import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise SystemExit("ffmpeg not found — install it or `pip install imageio-ffmpeg`")
out = os.path.join(here, "ARISE_mission_blender_4K.mp4")
subprocess.check_call([ff, "-y", "-framerate", "30",
    "-i", os.path.join(here, "render", "frame_%04d.png"),
    "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
    "-movflags", "+faststart", out])
print("->", out)
