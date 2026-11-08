# -*- coding: utf-8 -*-
"""
build_mission_film.py — ARISE · APOPHIS 4K cinematic (Blender 4.x, Cycles).

Builds the full 150 s mission film from the REAL exported datasets
(data/mission_bundle.json — run export_blender_data.py first):

  S1 Earth departure · S2 orbit overview (real ECI tracks) · S3 separation
  (real 6-DOF) · S4 Apophis strobe pass (real tumble + 136 real exposures)
  S5 return · S6 RL dock (real rollout, slow-mo captures) · S7 finale

CAD: drop your models in assets/ — deputy.(blend|glb|fbx|obj), carrier.*,
espa_ring.* — otherwise clean procedural stand-ins are built. Earth textures
(assets/earth_day.jpg, earth_night.jpg) are bundled.

Run — EITHER of these works:
  python build_mission_film.py                           → self-bootstraps: finds
        (or downloads) Blender and re-launches itself inside it. Just press Run.
  blender -b -P build_mission_film.py                    → builds + saves .blend
  blender -b -P build_mission_film.py -- --render all    → renders PNG frames (resumable)
  blender -b -P build_mission_film.py -- --render s4 --lowres --samples 24
Open arise_mission.blend in the GUI to tweak anything.
"""
import json, os, sys, math

# ---------------------------------------------------------------- bootstrap
# Running under plain Python? `bpy` only exists INSIDE Blender, so locate a
# Blender install (or download a portable one next to this script) and re-launch.
BLENDER_VERSION = "4.2.3"        # portable build fetched if none is installed

def _find_blender():
    import glob, shutil
    cands = []
    if os.environ.get("BLENDER_EXE"):
        cands.append(os.environ["BLENDER_EXE"])
    w = shutil.which("blender")
    if w:
        cands.append(w)
    here = os.path.dirname(os.path.abspath(__file__))
    cands += glob.glob(os.path.join(here, "_blender", "**", "blender.exe"), recursive=True)
    cands += glob.glob(os.path.join(here, "_blender", "**", "blender"), recursive=True)
    cands += sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender *\blender.exe"),
                    reverse=True)
    cands += sorted(glob.glob(r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe"))
    cands += sorted(glob.glob("/Applications/Blender*.app/Contents/MacOS/Blender"), reverse=True)
    cands += ["/usr/bin/blender", "/usr/local/bin/blender", "/snap/bin/blender"]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK if os.name != "nt" else os.F_OK):
            return c
    return None

def _install_blender():
    """Download + extract a portable Blender into blender_kit/_blender/."""
    import urllib.request, zipfile, tarfile, tempfile, platform
    here = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.join(here, "_blender")
    os.makedirs(dest, exist_ok=True)
    v = BLENDER_VERSION; short = ".".join(v.split(".")[:2])
    sysname = platform.system()
    if sysname == "Windows":
        fn = f"blender-{v}-windows-x64.zip"
    elif sysname == "Linux":
        fn = f"blender-{v}-linux-x64.tar.xz"
    else:
        print("[bootstrap] macOS: please install Blender from https://www.blender.org/download/\n"
              "            then re-run, or set BLENDER_EXE=/Applications/Blender.app/Contents/MacOS/Blender")
        sys.exit(1)
    url = f"https://download.blender.org/release/Blender{short}/{fn}"
    tmp = os.path.join(tempfile.gettempdir(), fn)
    if not os.path.exists(tmp) or os.path.getsize(tmp) < 100_000_000:
        print(f"[bootstrap] downloading Blender {v} (~350 MB, one time)\n            {url}")
        def _prog(b, bs, total):
            if total > 0 and b % 200 == 0:
                pct = min(100, b * bs * 100 // total)
                print(f"\r[bootstrap] {pct:3d}%", end="", flush=True)
        urllib.request.urlretrieve(url, tmp, _prog); print()
    print("[bootstrap] extracting…")
    if fn.endswith(".zip"):
        with zipfile.ZipFile(tmp) as z: z.extractall(dest)
    else:
        with tarfile.open(tmp) as t: t.extractall(dest)
    exe = _find_blender()
    if not exe:
        print("[bootstrap] extraction finished but blender executable not found in", dest)
        sys.exit(1)
    print("[bootstrap] Blender ready:", exe)
    return exe

try:
    import bpy                                            # inside Blender → go on
except ModuleNotFoundError:
    import subprocess
    exe = _find_blender() or _install_blender()
    argv = sys.argv[1:]
    cmd = [exe, "-b", "-P", os.path.abspath(__file__)]
    if argv:
        cmd += ["--"] + argv
    print("[bootstrap] launching:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    sys.exit(subprocess.call(cmd))

from mathutils import Vector, Quaternion, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = json.load(open(os.path.join(HERE, "data", "mission_bundle.json")))
ASSETS = os.path.join(HERE, "assets")
FPS = BUNDLE["fps"]; SHOTS = BUNDLE["shots"]
ENGINE = "CYCLES"            # or "BLENDER_EEVEE_NEXT"
SAMPLES = 96
CAPTIONS = True
SHOT_GAP = 0.0               # all shots share the origin (float precision!);
                             # isolation is purely per-frame visibility keys

# ---------------------------------------------------------------- small utils
def C(name):
    c = bpy.data.collections.get(name)
    if not c:
        c = bpy.data.collections.new(name); bpy.context.scene.collection.children.link(c)
    return c

def link(obj, coll):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj); return obj

def key(obj, path, frame, value):
    setattr(obj, path, value); obj.keyframe_insert(data_path=path, frame=frame)

def key_vec(obj, path, frame, v):
    setattr(obj, path, Vector(v)); obj.keyframe_insert(data_path=path, frame=frame)

def key_quat(obj, frame, q):
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = Quaternion((q[0], q[1], q[2], q[3]))
    obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

def linearize(obj):
    ad = obj.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'

def new_cam(name, coll, lens=50.0):
    cd = bpy.data.cameras.new(name); cd.lens = lens; cd.clip_end = 1e9; cd.clip_start = 0.05
    cam = bpy.data.objects.new(name, cd); link(cam, coll); return cam

def bind_cam(cam, frame):
    m = bpy.context.scene.timeline_markers.new(cam.name, frame=frame); m.camera = cam

def track(obj, target):
    con = obj.constraints.new('TRACK_TO')
    con.target = target; con.track_axis = 'TRACK_NEGATIVE_Z'; con.up_axis = 'UP_Y'

def empty(name, coll, loc=(0,0,0)):
    e = bpy.data.objects.new(name, None); e.location = loc; link(e, coll); return e

# ---------------------------------------------------------------- materials
def mat_new(name):
    m = bpy.data.materials.new(name); m.use_nodes = True
    return m, m.node_tree.nodes, m.node_tree.links, m.node_tree.nodes["Principled BSDF"]

def set_in(node, names, val):
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = val; return

def mat_metal(name, col=(0.75,0.76,0.78,1), rough=0.35):
    m, n, l, p = mat_new(name)
    p.inputs["Base Color"].default_value = col
    p.inputs["Metallic"].default_value = 1.0
    p.inputs["Roughness"].default_value = rough
    return m

def mat_plain(name, col, rough=0.6, emit=0.0, emit_col=None):
    m, n, l, p = mat_new(name)
    p.inputs["Base Color"].default_value = col
    p.inputs["Roughness"].default_value = rough
    set_in(p, ["Emission Color"], emit_col or col)
    set_in(p, ["Emission Strength"], emit)
    return m

def mat_solarcell(name="solar_cell"):
    m, n, l, p = mat_new(name)
    p.inputs["Base Color"].default_value = (0.06, 0.14, 0.30, 1)
    p.inputs["Metallic"].default_value = 0.4
    p.inputs["Roughness"].default_value = 0.18
    set_in(p, ["Coat Weight", "Clearcoat"], 0.6)
    return m

def mat_emission(name, col, strength):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = col; em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs[0]); return m

def mat_earth():
    m = bpy.data.materials.new("earth"); m.use_nodes = True
    nt = m.node_tree; nodes, links = nt.nodes, nt.links
    p = nodes["Principled BSDF"]
    p.inputs["Roughness"].default_value = 0.85
    set_in(p, ["Specular IOR Level", "Specular"], 0.3)
    tex_c = nodes.new("ShaderNodeTexImage")
    tex_c.image = bpy.data.images.load(os.path.join(ASSETS, "earth_day.jpg"))
    tex_n = nodes.new("ShaderNodeTexImage")
    tex_n.image = bpy.data.images.load(os.path.join(ASSETS, "earth_night.jpg"))
    tc = nodes.new("ShaderNodeTexCoord"); mp = nodes.new("ShaderNodeMapping")
    links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    for t in (tex_c, tex_n):
        t.projection = 'SPHERE'
        links.new(mp.outputs["Vector"], t.inputs["Vector"])
    links.new(tex_c.outputs["Color"], p.inputs["Base Color"])
    # night lights only on the dark side: dot(normal, sun) < 0
    geo = nodes.new("ShaderNodeNewGeometry")
    dot = nodes.new("ShaderNodeVectorMath"); dot.operation = 'DOT_PRODUCT'
    sd = BUNDLE.get("sun_dir", [1, 0.3, 0.2])
    dot.inputs[1].default_value = (sd[0], sd[1], sd[2])
    links.new(geo.outputs["Normal"], dot.inputs[0])
    ramp = nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = -0.15; ramp.inputs["From Max"].default_value = 0.05
    ramp.inputs["To Min"].default_value = 1.0;   ramp.inputs["To Max"].default_value = 0.0
    links.new(dot.outputs["Value"], ramp.inputs["Value"])
    mul = nodes.new("ShaderNodeMath"); mul.operation = 'MULTIPLY'
    links.new(ramp.outputs["Result"], mul.inputs[0]); mul.inputs[1].default_value = 4.0
    gain = nodes.new("ShaderNodeMixColor") if hasattr(nodes, "xxx") else nodes.new("ShaderNodeMix")
    try:
        gain.data_type = 'RGBA'
        links.new(tex_n.outputs["Color"], gain.inputs[7]); gain.inputs[6].default_value = (0,0,0,1)
        links.new(ramp.outputs["Result"], gain.inputs[0])
        set_in(p, ["Emission Color"], (1,1,1,1))
        links.new(gain.outputs[2], p.inputs["Emission Color"])
    except Exception:
        links.new(tex_n.outputs["Color"], p.inputs["Emission Color"])
    links.new(mul.outputs["Value"], p.inputs["Emission Strength"])
    return m

def mat_apophis():
    m = bpy.data.materials.new("apophis"); m.use_nodes = True
    nt = m.node_tree; nodes, links = nt.nodes, nt.links
    p = nodes["Principled BSDF"]
    p.inputs["Roughness"].default_value = 0.95
    tex = nodes.new("ShaderNodeTexNoise"); tex.inputs["Scale"].default_value = 18.0
    tex.inputs["Detail"].default_value = 8.0
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.085, 0.075, 0.065, 1)
    ramp.color_ramp.elements[1].color = (0.19, 0.17, 0.145, 1)
    links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], p.inputs["Base Color"])
    bump = nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.6
    links.new(tex.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], p.inputs["Normal"])
    return m

def world_stars():
    w = bpy.context.scene.world or bpy.data.worlds.new("space")
    bpy.context.scene.world = w; w.use_nodes = True
    nt = w.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 220.0; tex.inputs["Detail"].default_value = 2.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = 'CONSTANT'
    ramp.color_ramp.elements[0].position = 0.0; ramp.color_ramp.elements[0].color = (0,0,0,1)
    e1 = ramp.color_ramp.elements[1]; e1.position = 0.795; e1.color = (1,1,1,1)
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = 1.6
    nt.links.new(ramp.outputs["Color"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], bg.inputs["Strength"])
    bg.inputs["Color"].default_value = (1,1,1,1)
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

# ---------------------------------------------------------------- asset import
def try_import_asset(stem, coll):
    """Import assets/<stem>.(blend|glb|gltf|fbx|obj|stl) if present -> single object/empty parent."""
    for ext in ("blend", "glb", "gltf", "fbx", "obj", "stl"):
        p = os.path.join(ASSETS, f"{stem}.{ext}")
        if not os.path.exists(p):
            continue
        before = set(bpy.data.objects)
        try:
            if ext == "blend":
                with bpy.data.libraries.load(p) as (src, dst):
                    dst.objects = list(src.objects)
                for ob in dst.objects:
                    if ob: bpy.context.scene.collection.objects.link(ob)
            elif ext in ("glb", "gltf"):
                bpy.ops.import_scene.gltf(filepath=p)
            elif ext == "fbx":
                bpy.ops.import_scene.fbx(filepath=p)
            elif ext == "obj":
                bpy.ops.wm.obj_import(filepath=p)
            elif ext == "stl":
                bpy.ops.wm.stl_import(filepath=p)
        except Exception as e:
            print(f"[assets] import failed for {p}: {e}")
            continue
        new = [o for o in bpy.data.objects if o not in before and o.type in ('MESH','EMPTY')]
        if not new:
            continue
        root = empty(f"{stem}_root", coll)
        for o in new:
            if o.parent is None:
                o.parent = root
            link(o, coll)
        print(f"[assets] imported {stem}.{ext} ({len(new)} objects)")
        return root
    return None

def normalize_to_length(root, target_len, axis=0):
    """Uniform-scale an imported asset so its longest bbox axis = target_len metres."""
    import mathutils
    bpy.context.view_layer.update()
    mins = [1e30]*3; maxs = [-1e30]*3
    for o in root.children_recursive if hasattr(root, "children_recursive") else root.children:
        if o.type != 'MESH': continue
        for corner in o.bound_box:
            wc = o.matrix_world @ Vector(corner)
            for i in range(3):
                mins[i] = min(mins[i], wc[i]); maxs[i] = max(maxs[i], wc[i])
    ext = max(1e-6, max(maxs[i]-mins[i] for i in range(3)))
    root.scale = (target_len/ext,)*3

# ---------------------------------------------------------------- procedural models
def build_deputy_proc(coll, dims):
    """Parametric 6U: alu frame + solar faces + NAC aperture on +X."""
    L, W, H = dims
    root = empty("deputy_proc", coll)
    bpy.ops.mesh.primitive_cube_add(size=1)
    body = bpy.context.active_object; body.name = "dep_body"
    body.scale = (L/2*0.98, W/2*0.98, H/2*0.98)
    body.data.materials.append(mat_solarcell())
    link(body, coll); body.parent = root
    bpy.ops.mesh.primitive_cube_add(size=1)
    frame = bpy.context.active_object; frame.name = "dep_frame"
    frame.scale = (L/2, W/2, H/2)
    mod = frame.modifiers.new("wire", 'WIREFRAME'); mod.thickness = 0.012; mod.use_even_offset = False
    frame.data.materials.append(mat_metal("alu", (0.8,0.81,0.83,1), 0.3))
    link(frame, coll); frame.parent = root
    bpy.ops.mesh.primitive_cylinder_add(radius=min(W,H)*0.22, depth=0.03, vertices=24)
    nac = bpy.context.active_object; nac.name = "dep_nac"
    nac.rotation_euler = (0, math.pi/2, 0); nac.location = (L/2+0.012, 0, 0)
    nac.data.materials.append(mat_plain("lens", (0.02,0.02,0.03,1), 0.1))
    link(nac, coll); nac.parent = root
    return root

def build_ring_proc(coll):
    ring = BUNDLE["ring"]; R, Hh = ring["radius"], ring["height"]/2
    root = empty("espa_proc", coll)
    bpy.ops.mesh.primitive_cylinder_add(radius=R, depth=Hh*2, vertices=48)
    cyl = bpy.context.active_object; cyl.name = "espa_cyl"
    solid = cyl.modifiers.new("shell", 'SOLIDIFY'); solid.thickness = -0.05
    cyl.data.materials.append(mat_metal("espa_alu", (0.55,0.57,0.6,1), 0.45))
    link(cyl, coll); cyl.parent = root
    for i, (pr, pd) in enumerate(zip(ring["ports"], ring["dirs"])):
        bpy.ops.mesh.primitive_cylinder_add(radius=0.055, depth=0.05, vertices=24)
        p = bpy.context.active_object; p.name = f"rafti_{i}"
        d = Vector(pd); p.location = Vector(pr) + d*0.02
        p.rotation_mode = 'QUATERNION'
        p.rotation_quaternion = d.to_track_quat('Z', 'Y')
        p.data.materials.append(mat_plain(f"rafti_m{i}", (0.9,0.35,0.45,1), 0.4, emit=0.0,
                                          emit_col=(1.0,0.3,0.45,1)))
        link(p, coll); p.parent = root
    return root

def build_carrier_proc(coll):
    root = empty("carrier_proc", coll)
    bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=3.2, vertices=48)
    b = bpy.context.active_object; b.name = "carrier_body"
    b.rotation_euler = (0, math.pi/2, 0)
    b.data.materials.append(mat_plain("carrier_white", (0.85,0.85,0.87,1), 0.35))
    link(b, coll); b.parent = root
    bpy.ops.mesh.primitive_cylinder_add(radius=0.75, depth=0.5, vertices=48)
    n = bpy.context.active_object; n.name = "carrier_engine"
    n.rotation_euler = (0, math.pi/2, 0); n.location = (-1.85, 0, 0)
    n.data.materials.append(mat_metal("engine", (0.35,0.34,0.33,1), 0.5))
    link(n, coll); n.parent = root
    return root

def get_deputy_template(coll):
    dims = BUNDLE["ring"]["deputy_dims"]
    root = try_import_asset("deputy", coll)
    if root:
        normalize_to_length(root, dims[0]); return root
    return build_deputy_proc(coll, dims)

def clone(root, name, coll):
    import bpy
    new_root = empty(name, coll)
    for o in (root.children_recursive if hasattr(root, "children_recursive") else root.children):
        c = o.copy()
        if o.data: c.data = o.data
        link(c, coll)
        if o.parent == root:
            c.parent = new_root
        else:
            for oo in coll.objects:
                if oo.name.startswith(o.parent.name):
                    c.parent = oo; break
    new_root.scale = root.scale
    root.hide_render = True; root.hide_viewport = True
    return new_root

# ---------------------------------------------------------------- instancing
def make_template_collection(name, builder):
    tc = bpy.data.collections.get(name)
    if tc is None:
        tc = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(tc)
        builder(tc)
        lc = bpy.context.view_layer.layer_collection.children.get(name)
        if lc: lc.exclude = True                      # template hidden from render
    return tc

def instance_of(tpl_coll, name, coll, scale=1.0):
    e = bpy.data.objects.new(name, None)
    e.instance_type = 'COLLECTION'; e.instance_collection = tpl_coll
    e.empty_display_size = 0.2; e.scale = (scale,)*3
    link(e, coll); return e

def shot_show(objs, f0, f1):
    """Restrict render visibility of objects to [f0,f1]."""
    for o in objs:
        for fr, hide in ((f0-1, True), (f0, False), (f1, False), (f1+1, True)):
            o.hide_render = hide; o.keyframe_insert("hide_render", frame=fr)
            o.hide_viewport = hide; o.keyframe_insert("hide_viewport", frame=fr)

def caption(text, cam, coll, f0, f1, size=0.043, y=-0.36, sub=False):
    if not CAPTIONS: return None
    curve = bpy.data.curves.new(f"cap_{text[:12]}", type='FONT')
    curve.body = text; curve.size = size; curve.align_x = 'CENTER'
    ob = bpy.data.objects.new(f"cap_{text[:12]}", curve)
    ob.data.materials.append(mat_emission(f"capm_{text[:10]}", (1,1,1,1), 2.0 if not sub else 1.2))
    link(ob, coll); ob.parent = cam
    ob.location = (0, y, -(cam.data.lens / 24.0)); ob.scale = (1, 1, 1)
    shot_show([ob], f0, f1)
    return ob

# ---------------------------------------------------------------- shots
def fill_light(coll):
    ld = bpy.data.lights.new("fill", 'SUN'); ld.energy = 0.35
    ld.color = (0.55, 0.65, 0.9); ld.angle = math.radians(15)
    lo = bpy.data.objects.new("fill", ld); link(lo, coll)
    lo.rotation_euler = (math.radians(65), 0, math.radians(210))
    return lo

def sun_light(coll, name="Sun"):
    sd = Vector(BUNDLE.get("sun_dir", [1, 0.3, 0.2])).normalized()
    ld = bpy.data.lights.new(name, 'SUN'); ld.energy = 5.5; ld.angle = math.radians(0.53)
    lo = bpy.data.objects.new(name, ld); link(lo, coll)
    lo.rotation_mode = 'QUATERNION'
    lo.rotation_quaternion = (-sd).to_track_quat('-Z', 'Y')
    return lo

def make_earth(coll, radius, loc):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=96, ring_count=64, location=loc)
    e = bpy.context.active_object; e.name = f"Earth_{coll.name}"
    bpy.ops.object.shade_smooth()
    e.data.materials.append(mat_earth())
    e.rotation_euler = (math.radians(23), 0, math.radians(-100))
    link(e, coll)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius*1.014, segments=64, ring_count=48, location=loc)
    a = bpy.context.active_object; a.name = f"Atmo_{coll.name}"
    bpy.ops.object.shade_smooth()
    ma = bpy.data.materials.new("atmo"); ma.use_nodes = True
    nt = ma.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.35, 0.55, 1.0, 1); em.inputs["Strength"].default_value = 2.2
    fr = nt.nodes.new("ShaderNodeFresnel"); fr.inputs["IOR"].default_value = 1.25
    nt.links.new(fr.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs[0])
    ma.blend_method = 'BLEND'
    a.data.materials.append(ma); a.visible_shadow = False
    link(a, coll)
    return [e, a]

def shot_departure(dep_tpl, car_tpl):
    f0, f1 = SHOTS["s1_departure"]; coll = C("S1_departure")
    org = Vector((0*SHOT_GAP, 0, 0))
    earth = make_earth(coll, 100.0, org + Vector((-260, 260, -70)))
    car = instance_of(car_tpl, "carrier", coll, scale=1.15)
    car.location = org + Vector((10.5, 9, -3.4))
    car.rotation_euler = (math.radians(6), math.radians(14), math.radians(52))
    cam = new_cam("cam_s1", coll, lens=42)
    tgt = empty("s1_aim", coll, org + Vector((2, 8, -2.5)))
    key_vec(cam, "location", f0, org + Vector((9, -22, 2.5)))
    key_vec(cam, "location", f1, org + Vector((11.5, -13, 0.8)))
    track(cam, tgt); bind_cam(cam, f0); linearize(cam)
    shot_show([car] + earth, f0, f1)
    caption("ARISE · APOPHIS MISSION", cam, coll, f0+30, f1, size=0.065, y=0.385)
    caption("eight-deputy autonomous reconnaissance fleet", cam, coll, f0+60, f1,
            size=0.036, y=0.30, sub=True)
    return [coll]

def shot_orbits():
    f0, f1 = SHOTS["s2_orbits"]; coll = C("S2_orbits")
    org = Vector((1*SHOT_GAP, 0, 0)); S = 20.0
    earth = make_earth(coll, S, org)
    def curve_of(pts, name, col, width):
        cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
        cu.bevel_depth = width; cu.bevel_resolution = 3
        sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
        for i, p in enumerate(pts):
            sp.points[i].co = (org.x + p[0]*S, org.y + p[1]*S, org.z + p[2]*S, 1)
        ob = bpy.data.objects.new(name, cu)
        ob.data.materials.append(mat_emission(name + "_m", col, 14))
        link(ob, coll)
        cu.bevel_factor_mapping_end = 'SPLINE'
        key(cu, "bevel_factor_end", f0, 0.0); key(cu, "bevel_factor_end", f1 - 60, 1.0)
        for fc in cu.animation_data.action.fcurves:
            for kp in fc.keyframe_points: kp.interpolation = 'LINEAR'
        return ob
    cw = curve_of(BUNDLE["orbit"]["chief"], "chief_track", (1, 1, 1, 1), 0.30)
    apts = [p for p in BUNDLE["orbit"]["apophis"]
            if (p[0]**2 + p[1]**2 + p[2]**2) ** 0.5 < 12.0]
    if len(apts) < 10:                       # safety: keep the approach tail
        apts = BUNDLE["orbit"]["apophis"][-120:]
    ca = curve_of(apts, "apophis_track", (1, 0.18, 0.14, 1), 0.45)
    BUNDLE["orbit"]["apophis"] = apts
    end = BUNDLE["orbit"]["apophis"][-1]
    bpy.ops.mesh.primitive_uv_sphere_add(radius=2.0,
        location=org + Vector((end[0]*S, end[1]*S, end[2]*S)))
    mk = bpy.context.active_object; mk.name = "apophis_marker"
    mk.data.materials.append(mat_emission("mk", (1, 0.2, 0.15, 1), 8)); link(mk, coll)
    cam = new_cam("cam_s2", coll, lens=35)
    cpts = BUNDLE["orbit"]["chief"]
    cen = [sum(p[i] for p in cpts)/len(cpts)*S for i in range(3)]
    tgt = empty("s2_aim", coll, org + Vector(cen)*0.6)
    key_vec(cam, "location", f0, org + Vector(cen)*0.6 + Vector((20, -285, 130)))
    key_vec(cam, "location", f1, org + Vector(cen)*0.6 + Vector((110, -245, 100)))
    track(cam, tgt); bind_cam(cam, f0); linearize(cam)
    shot_show([cw, ca, mk] + earth, f0, f1)
    caption("REAL MISSION TRAJECTORIES — CHIEF (WHITE) · APOPHIS (RED)", cam, coll,
            f0+20, f1, size=0.029, y=-0.36, sub=True)
    return [coll]

def keyframe_fleet(dep_tpl, coll, seq_pos, seq_quat, f0, f1, org, scale=1.0,
                   dep_scale=1.0, name="dep"):
    n = len(seq_pos); deps = []
    for k in range(8):
        d = instance_of(dep_tpl, f"{name}{k+1}", coll, scale=dep_scale)
        step = max(1, n // (f1 - f0 + 1) if n > (f1-f0+1) else 1)
        for i in range(n):
            fr = f0 + i * (f1 - f0) / (n - 1)
            p = seq_pos[i][k]
            key_vec(d, "location", round(fr), org + Vector((p[0]*scale, p[1]*scale, p[2]*scale)))
            if seq_quat is not None:
                key_quat(d, round(fr), seq_quat[i][k])
        linearize(d); deps.append(d)
    return deps

def shot_separation(dep_tpl, ring_tpl):
    f0, f1 = SHOTS["s3_separation"]; coll = C("S3_separation")
    org = Vector((2*SHOT_GAP, 0, 0))
    ring = instance_of(ring_tpl, "espa_s3", coll); ring.location = org
    earth3 = make_earth(coll, 60.0, org + Vector((350, -520, -160)))
    deps = keyframe_fleet(dep_tpl, coll, BUNDLE["separation"]["pos"],
                          BUNDLE["separation"]["quat"], f0, f1, org)
    cam = new_cam("cam_s3", coll, lens=45)
    tgt = empty("s3_aim", coll, org)
    key_vec(cam, "location", f0, org + Vector((4.2, -5.2, 1.9)))
    key_vec(cam, "location", f1, org + Vector((15, -18, 6.5)))
    track(cam, tgt); bind_cam(cam, f0); linearize(cam)
    shot_show(deps + [ring] + earth3, f0, f1)
    caption("PHASE 1 · SEPARATION — ESPA RAFTI DEPLOY (REAL 6-DOF)", cam, coll,
            f0+15, f1, size=0.029, sub=True)
    return [coll]

def shot_imaging(dep_tpl):
    f0, f1 = SHOTS["s4_imaging"]; coll = C("S4_imaging")
    org = Vector((3*SHOT_GAP, 0, 0)); POS_SC = 0.12; DEP_SC = 95.0
    bpy.ops.wm.obj_import(filepath=os.path.join(HERE, "data", "apophis.obj"))
    apo = bpy.context.active_object; apo.name = "Apophis"
    bpy.ops.object.shade_smooth()
    apo.data.materials.append(mat_apophis()); link(apo, coll)
    apo.location = org
    qs = BUNDLE["imaging"]["tumble_quat"]; n = len(qs)
    for i, q in enumerate(qs):
        fr = f0 + i * (f1 - f0) / (n - 1)
        key_quat(apo, round(fr), q)
    linearize(apo)
    deps = []; offs = BUNDLE["imaging"]["offsets"]
    for k in range(8):
        d = instance_of(dep_tpl, f"img_dep{k+1}", coll, scale=DEP_SC)
        o = offs[k]
        d.location = org + Vector((o[0]*POS_SC, o[1]*POS_SC, o[2]*POS_SC))
        q = BUNDLE["imaging"]["aim_quat"][k]
        d.rotation_mode = 'QUATERNION'; d.rotation_quaternion = q
        deps.append(d)
    # REAL strobe exposures -> light flashes
    lights = []
    for e in BUNDLE["imaging"]["events"]:
        k = e["k"]; fr = e["frame"]
        ld = bpy.data.lights.new(f"strobe_{fr}_{k}", 'POINT')
        ld.energy = 0.0; ld.shadow_soft_size = 2.0; ld.color = (0.85, 0.92, 1.0)
        lo = bpy.data.objects.new(f"strobe_{fr}_{k}", ld)
        lo.location = deps[k].location; link(lo, coll)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=11.0, location=deps[k].location)
        sp = bpy.context.active_object; sp.name = f"spark_{fr}_{k}"
        sp.data.materials.append(mat_emission(f"sparkm_{fr}_{k}", (0.9, 0.95, 1, 1), 700))
        link(sp, coll)
        for f, sc in ((fr-1, 0.001), (fr, 1.0), (fr+2, 0.001)):
            sp.scale = (sc,)*3; sp.keyframe_insert("scale", frame=f)
        lights.append(sp)
        for f, en in ((fr-1, 0.0), (fr, 6.0e7), (fr+2, 0.0)):
            ld.energy = en; ld.keyframe_insert("energy", frame=f)
        lights.append(lo)
    cam = new_cam("cam_s4", coll, lens=50)
    tgt = empty("s4_aim", coll, org)
    key_vec(cam, "location", f0, org + Vector((900, -1350, 360)))
    key_vec(cam, "location", (f0+f1)//2, org + Vector((260, -980, 240)))
    key_vec(cam, "location", f1, org + Vector((-520, -820, 130)))
    track(cam, tgt); bind_cam(cam, f0); linearize(cam)
    shot_show(deps + [apo] + lights, f0, f1)
    caption("PHASE 2 · CLOSE APPROACH — 136 REAL STROBE EXPOSURES · CA ±2 s",
            cam, coll, f0+15, f1, size=0.029, sub=True)
    caption("spacecraft enlarged · formation compressed for visibility",
            cam, coll, f0+15, f1, size=0.022, y=-0.365, sub=True)
    return [coll]

def shot_return(dep_tpl, ring_tpl):
    f0, f1 = SHOTS["s5_return"]; coll = C("S5_return")
    org = Vector((4*SHOT_GAP, 0, 0))
    ring = instance_of(ring_tpl, "espa_s5", coll, scale=30.0); ring.location = org
    deps = keyframe_fleet(dep_tpl, coll, BUNDLE["ret"]["pos"], None, f0, f1, org,
                          dep_scale=25.0, name="ret_dep")
    cam = new_cam("cam_s5", coll, lens=40)
    tgt = empty("s5_aim", coll, org)
    key_vec(cam, "location", f0, org + Vector((300, -480, 150)))
    key_vec(cam, "location", f1, org + Vector((42, -80, 24)))
    track(cam, tgt); bind_cam(cam, f0); linearize(cam)
    shot_show(deps + [ring], f0, f1)
    caption("PHASE 3 · RETURN — RELATIVE STM TRANSFERS (REAL ARCS)", cam, coll,
            f0+15, f1, size=0.029, sub=True)
    caption("spacecraft & ring enlarged for visibility", cam, coll, f0+15, f1,
            size=0.016, y=-0.408, sub=True)
    return [coll]

def shot_dock(dep_tpl, ring_tpl):
    f0, f1 = SHOTS["s6_dock"]; coll = C("S6_dock")
    org = Vector((5*SHOT_GAP, 0, 0))
    ring = instance_of(ring_tpl, "espa_s6", coll); ring.location = org
    deps = keyframe_fleet(dep_tpl, coll, BUNDLE["dock"]["pos"], BUNDLE["dock"]["quat"],
                          f0, f1, org, name="dock_dep")
    flashes = []
    for k, fr in enumerate(BUNDLE["dock"]["capture_frames"]):
        ld = bpy.data.lights.new(f"cap_{k}", 'POINT')
        ld.energy = 0.0; ld.color = (0.35, 1.0, 0.6); ld.shadow_soft_size = 0.3
        lo = bpy.data.objects.new(f"cap_{k}", ld); link(lo, coll)
        lo.parent = deps[k]
        for f, en in ((fr-1, 0.0), (fr, 9000.0), (fr+6, 0.0)):
            ld.energy = en; ld.keyframe_insert("energy", frame=f)
        flashes.append(lo)
    cam = new_cam("cam_s6", coll, lens=50)
    tgt = empty("s6_aim", coll, org)
    key_vec(cam, "location", f0, org + Vector((20, -26, 9)))
    key_vec(cam, "location", f1, org + Vector((3.8, -5.2, 1.8)))
    track(cam, tgt); bind_cam(cam, f0); linearize(cam)
    shot_show(deps + [ring] + flashes, f0, f1)
    caption("PHASE 4 · RE-DOCK — RL POLICY · REAL ROLLOUT · 8/8 CAPTURED",
            cam, coll, f0+15, f1, size=0.029, sub=True)
    return [coll]

def shot_finale(dep_tpl, ring_tpl):
    f0, f1 = SHOTS["s7_finale"]; coll = C("S7_finale")
    org = Vector((6*SHOT_GAP, 0, 0))
    ring = instance_of(ring_tpl, "espa_s7", coll); ring.location = org
    deps = []
    lastq = BUNDLE["dock"]["quat"][-1]; lastp = BUNDLE["dock"]["pos"][-1]
    for k in range(8):
        d = instance_of(dep_tpl, f"fin_dep{k+1}", coll)
        p = lastp[k]; d.location = org + Vector(p)
        d.rotation_mode = 'QUATERNION'; d.rotation_quaternion = lastq[k]
        deps.append(d)
    cam = new_cam("cam_s7", coll, lens=62)
    tgt = empty("s7_aim", coll, org)
    key_vec(cam, "location", f0, org + Vector((4.2, -6.2, 2.0)))
    key_vec(cam, "location", f1, org + Vector((-2.6, -5.6, 2.4)))
    track(cam, tgt); bind_cam(cam, f0); linearize(cam)
    shot_show(deps + [ring], f0, f1)
    st = BUNDLE["stats"]
    caption("MISSION COMPLETE", cam, coll, f0+20, f1, size=0.065, y=0.36)
    caption(f"{st['images']} STROBE IMAGES · {st['coverage']:.1f}% COVERAGE · "
            f"{st['captures']}/8 CAPTURED · ΔV {st['dv_return']:.1f} m/s",
            cam, coll, f0+45, f1, size=0.030, y=0.275, sub=True)
    return [coll]

# ---------------------------------------------------------------- render setup
def configure(engine=ENGINE, lowres=False, samples=SAMPLES):
    s = bpy.context.scene
    s.render.engine = engine
    s.render.fps = FPS
    s.frame_start = 0; s.frame_end = SHOTS["s7_finale"][1]
    s.render.resolution_x = 3840; s.render.resolution_y = 2160
    if lowres:
        s.render.resolution_percentage = 25
    if engine == "CYCLES":
        s.cycles.samples = samples
        s.cycles.use_denoising = True
        try: s.cycles.denoiser = 'OPENIMAGEDENOISE'
        except Exception: pass
        s.cycles.device = 'GPU'
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for dt in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
                try:
                    prefs.compute_device_type = dt
                    prefs.get_devices()
                    if any(d.type == dt for d in prefs.devices):
                        for d in prefs.devices: d.use = True
                        print(f"[render] GPU backend: {dt}"); break
                except Exception: continue
            else:
                s.cycles.device = 'CPU'
        except Exception:
            s.cycles.device = 'CPU'
    try:
        s.view_settings.view_transform = 'AgX'
        s.view_settings.look = 'AgX - Medium High Contrast'
    except Exception: pass
    # big multi-frame win: keep the scene resident between frames instead of
    # re-syncing 160+ imported CAD objects every single frame
    try: s.render.use_persistent_data = True
    except Exception: pass
    print(f"[render] engine={s.render.engine} device={getattr(s.cycles,'device','n/a')} "
          f"samples={getattr(s.cycles,'samples','n/a')} "
          f"res={s.render.resolution_x}x{s.render.resolution_y}@{s.render.resolution_percentage}%")
    s.render.image_settings.file_format = 'PNG'
    s.render.use_overwrite = False; s.render.use_placeholder = True
    s.render.filepath = os.path.join(HERE, "render", "frame_")

def main():
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    lowres = "--lowres" in argv
    samples = int(argv[argv.index("--samples")+1]) if "--samples" in argv else SAMPLES
    global engine_samples; engine_samples = samples
    bpy.ops.wm.read_factory_settings(use_empty=True)
    world_stars()
    base = C("base"); sun_light(base); fill_light(base)
    def _mk_deputy(c):
        root = try_import_asset("deputy", c)
        if root:
            normalize_to_length(root, BUNDLE["ring"]["deputy_dims"][0])
        else:
            build_deputy_proc(c, BUNDLE["ring"]["deputy_dims"])
    def _mk_ring(c):
        root = try_import_asset("espa_ring", c)
        if root:
            normalize_to_length(root, 1.25)   # OD hugging the real 0.5334 m port circle
        else:
            build_ring_proc(c)
    def _mk_carrier(c):
        if not try_import_asset("carrier", c):
            build_carrier_proc(c)
    dep_tpl = make_template_collection("TPL_deputy", _mk_deputy)
    ring_tpl = make_template_collection("TPL_ring", _mk_ring)
    car_tpl = make_template_collection("TPL_carrier", _mk_carrier)
    shot_departure(dep_tpl, car_tpl)
    shot_orbits()
    shot_separation(dep_tpl, ring_tpl)
    shot_imaging(dep_tpl)
    shot_return(dep_tpl, ring_tpl)
    shot_dock(dep_tpl, ring_tpl)
    shot_finale(dep_tpl, ring_tpl)
    configure(lowres=lowres, samples=samples)
    blend = os.path.join(HERE, "arise_mission.blend")
    try:                       # embed textures so the .blend is portable (HPC transfer)
        bpy.ops.file.pack_all()
        print("[build] external resources packed into the .blend")
    except Exception as e:
        print("[build] pack_all skipped:", e)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    print(f"[build] saved {blend}")
    if "--render" in argv:
        what = argv[argv.index("--render")+1]
        s = bpy.context.scene
        if what != "all" and any(what in k for k in SHOTS):
            kk = [k for k in SHOTS if what in k][0]
            s.frame_start, s.frame_end = SHOTS[kk]
        if "--frames" in argv:                       # explicit range (render_4k.bat)
            i = argv.index("--frames")
            s.frame_start, s.frame_end = int(argv[i+1]), int(argv[i+2])
        rd = os.path.join(HERE, "render"); os.makedirs(rd, exist_ok=True)
        def _rm(fp):
            try:
                os.remove(fp); return True
            except Exception as e:
                print(f"[render] could not delete {os.path.basename(fp)}: {e}"); return False
        for f in os.listdir(rd):                     # purge crash placeholders
            fp = os.path.join(rd, f)
            if f.endswith(".png") and os.path.getsize(fp) == 0:
                _rm(fp)
        # resume safety: frames rendered at DIFFERENT settings (e.g. a --lowres
        # preview) must not be reused by a full-quality run — purge on mismatch
        sig = f"{s.render.resolution_x}x{s.render.resolution_y}@" \
              f"{s.render.resolution_percentage}%/{engine_samples}"
        sig_p = os.path.join(rd, "_render_settings.txt")
        try:
            old = open(sig_p).read().strip() if os.path.exists(sig_p) else sig
        except Exception:
            old = sig
        if old != sig:
            n_gone = sum(_rm(os.path.join(rd, f))
                         for f in os.listdir(rd) if f.endswith(".png"))
            print(f"[render] settings changed ({old} -> {sig}) — purged {n_gone} old frame(s)")
        try:
            open(sig_p, "w").write(sig)
        except Exception:
            pass
        print(f"[render] settings {sig} · frames {s.frame_start}-{s.frame_end} -> {rd}")
        bpy.ops.render.render(animation=True)
        print("[render] done")

if __name__ == "__main__":
    main()
