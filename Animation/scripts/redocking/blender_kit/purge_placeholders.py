"""Delete zero-byte placeholder PNGs left by an interrupted render (they would
otherwise be skipped on resume, leaving holes in the film)."""
import os
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render")
n = 0
if os.path.isdir(here):
    for f in os.listdir(here):
        p = os.path.join(here, f)
        if f.endswith(".png") and os.path.getsize(p) == 0:
            os.remove(p); n += 1
print(f"[purge] removed {n} placeholder frame(s)")
