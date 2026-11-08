"""
check_gpu.py — is Cycles actually using your GPU?  Run:  python check_gpu.py
(Standalone: does NOT import the film script.)
"""
import os, sys, glob, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))

def _find_blender():
    c = []
    if os.environ.get("BLENDER_EXE"): c.append(os.environ["BLENDER_EXE"])
    w = shutil.which("blender")
    if w: c.append(w)
    c += glob.glob(os.path.join(HERE, "_blender", "**", "blender.exe"), recursive=True)
    c += glob.glob(os.path.join(HERE, "_blender", "**", "blender"), recursive=True)
    c += sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender *\blender.exe"), reverse=True)
    c += sorted(glob.glob("/Applications/Blender*.app/Contents/MacOS/Blender"), reverse=True)
    c += ["/usr/bin/blender", "/usr/local/bin/blender", "/snap/bin/blender"]
    for x in c:
        if x and os.path.isfile(x): return x
    return None

try:
    import bpy
except ModuleNotFoundError:
    exe = _find_blender()
    if not exe:
        sys.exit("Blender not found — run build_mission_film.py once to install it.")
    sys.exit(subprocess.call([exe, "-b", "--factory-startup", "-P", os.path.abspath(__file__)]))

print("\n================ CYCLES DEVICE REPORT ================")
prefs = bpy.context.preferences.addons["cycles"].preferences
found = {}
for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
    try:
        prefs.compute_device_type = backend
        prefs.get_devices()
        devs = [d for d in prefs.devices if d.type == backend]
        if devs: found[backend] = [d.name for d in devs]
    except Exception:
        pass
if found:
    for b, names in found.items():
        print(f"  AVAILABLE {b}: {', '.join(names)}")
    best = next(b for b in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI") if b in found)
    print(f"\n  -> GPU rendering IS possible. Best backend: {best}")
    print("     build_mission_film.py enables this automatically at render time.")
    print("     If your render still says device=CPU, open Blender GUI once:")
    print("     Edit > Preferences > System > Cycles Render Devices > tick your GPU.")
else:
    print("  No GPU backend detected — Cycles will use the CPU.")
    print("  -> update your GPU driver, or enable the GPU in")
    print("     Edit > Preferences > System > Cycles Render Devices.")
cpu = [d.name for d in prefs.devices if d.type == "CPU"]
print(f"\n  CPU device: {cpu[0] if cpu else 'n/a'}")
print("  Your render log prints the device in use as:  [render] engine=... device=...")
print("======================================================\n")
