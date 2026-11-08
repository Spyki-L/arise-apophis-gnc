# -*- coding: utf-8 -*-
"""
make_redock_film.py — standalone 4K clip of PHASE-A RETURN + TERMINAL RE-DOCK
+ MISSION COMPLETE card, in the dashboard style, on a pure-black starfield that
matches Final_Video.mp4 so the two can be merged.

Press Run (no CLI args). Reuses everything from make_mission_film.py — same real
datasets, same 3D scene, same encoder/worker/resume machinery.

Output: thesis_arc_out/videos/ARISE_returndock_4K.mp4
"""
from __future__ import annotations
import os, sys, json, time, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("REDOCK_DIR", HERE)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_mission_film as M
import film_gui as G

# ================================================================ CONFIG
CONFIG = dict(
    SECONDS   = 60,        # return + dock + end card
    SPLIT     = (20, 32, 8),   # seconds: return, dock, finale
    FPS       = 30,
    WIDTH     = 3840,
    WORKERS   = None,      # None -> auto
    CRF       = 17,
    PREVIEW   = False,     # True -> 960x540 draft
    BLACK_BG  = True,      # match Final_Video's pure-black space
    BLENDER_LOOK = True,   # full-bleed scene + Final_Video's caption style,
                           # no dashboard cards / title bar / HUD labels
    NAME      = "ARISE_returndock_4K.mp4",
)

# ---------------------------------------------------------------- Final_Video look
# Measured directly from Final_Video.mp4 (3840x2160):
#   main caption : cap-height 53 px, RGB(209,209,209), baseline 0.0731 from bottom
#   sub caption  : 38 px tall,       RGB(205,205,205), baseline 0.0162 from bottom
# At 16x9 in / dpi=W/16, 1 pt = dpi/72 px  ->  53 px cap ~= 21.8 pt, sub ~= 12.0 pt
CAP_COL, CAP_PT, CAP_Y = "#d1d1d1", 19.5, 0.0754
SUB_COL, SUB_PT, SUB_Y = "#cdcdcd", 12.0, 0.0157

CAPTIONS_BL = {          # exactly the strings Final_Video uses
    "RETURN": ("PHASE 3 · RETURN — RELATIVE STM TRANSFERS (REAL ARCS)",
               "spacecraft & ring enlarged for visibility"),
    "DOCK":   ("PHASE 4 · RE-DOCK — RL POLICY · REAL ROLLOUT · 8/8 CAPTURED",
               None),
}

def _caption_bl(fig, main, sub=None):
    fig.text(0.5, CAP_Y, main, color=CAP_COL, fontsize=CAP_PT, ha="center",
             va="baseline")
    if sub:
        fig.text(0.5, SUB_Y, sub, color=SUB_COL, fontsize=SUB_PT, ha="center",
                 va="baseline")

def _finale_bl(fig, D, sp):
    """MISSION COMPLETE card in Final_Video's two-line style."""
    st = D["stats"] if "stats" in D else None
    cov = D["cov"]["whole"]; img = int(D["nimg"].sum())
    cap = int(np.asarray(D["dock"]["n_captured"]))
    dv = float(np.asarray(D["ret"]["dv_total"]))
    u = sp["u"]; a = float(np.clip(u * 2.5, 0, 1))
    fig.text(0.5, 0.60, "MISSION COMPLETE", color="#e8e8e8", fontsize=CAP_PT * 2.6,
             ha="center", va="baseline", alpha=a)
    a2 = float(np.clip((u - 0.2) * 2.5, 0, 1))
    fig.text(0.5, 0.53,
             f"{img} STROBE IMAGES · {cov:.1f}% COVERAGE · {cap}/8 CAPTURED · ΔV {dv:.1f} m/s",
             color=CAP_COL, fontsize=CAP_PT * 0.95, ha="center", va="baseline", alpha=a2)

def render_frame_bl(fig, D, sp):
    """Full-bleed 3D scene + Final_Video captions. No cards, no HUD."""
    fig.clear(); fig.patch.set_facecolor("black")
    _bg_black(fig)
    if sp.get("kind") == "finale":
        _finale_bl(fig, D, sp)
    else:
        sp2 = dict(sp); sp2["hero_rect"] = [0.0, 0.0, 1.0, 1.0]
        ax = M.scene_3d(fig, D, sp2)
        ax.set_axis_off()                       # no grid box — Blender shows none
        for t in list(fig.texts):               # drop scene_3d's corner label
            if t.get_text().startswith("FRAME ·"):
                t.remove()
        main, sub = CAPTIONS_BL.get(sp.get("phase"), (None, None))
        if main: _caption_bl(fig, main, sub)
    M.fade_overlay(fig, sp.get("fade", 0.0))

# ---------------------------------------------------------------- black background
_STAR_CACHE = {}
def _bg_black(fig):
    """Pure black + crisp starfield — matches the Blender film's background."""
    ax = fig.add_axes([0, 0, 1, 1], zorder=-20); ax.set_axis_off()
    ax.set_facecolor("black")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="black", zorder=-21))
    if "stars" not in _STAR_CACHE:
        # tuned against Final_Video: sparse, ~1.5-2.5 px points (Blender renders
        # far fewer/smaller stars than the dashboard's default field)
        r = np.random.default_rng(7); n = 190
        _STAR_CACHE["stars"] = (r.random(n), r.random(n),
                                0.18 + 0.40 * r.random(n) ** 2,
                                0.45 + 0.55 * r.random(n) ** 2)
    x, y, s, a = _STAR_CACHE["stars"]
    ax.scatter(x, y, s=s, c=[(1, 1, 1, ai) for ai in a], lw=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ---------------------------------------------------------------- spec building
def build_specs_segment(D, total, fps, split):
    """Full spec list, then keep only RETURN + DOCK + finale."""
    ret_s, dock_s, fin_s = split
    tiny = 0.1
    M.SEGS = [("TITLE", tiny), ("SEPARATION", tiny), ("CRUISE", tiny),
              ("IMAGING", tiny), ("RETURN", float(ret_s)),
              ("DOCK", float(dock_s)), ("FINALE", float(fin_s))]
    dur_sum = 4 * tiny + ret_s + dock_s + fin_s
    build_total = int(round(total * dur_sum / (ret_s + dock_s + fin_s)))
    specs = M.build_specs(D, build_total, fps)
    keep = [sp for sp in specs
            if sp.get("phase") in ("RETURN", "DOCK") or sp.get("kind") == "finale"]
    # rebuild the fade on the first/last second so it tops and tails cleanly
    n = len(keep); f = int(fps)
    for i, sp in enumerate(keep):
        sp["fade"] = (1 - i / f) if i < f else ((i - (n - 1 - f)) / f if i > n - 1 - f else 0.0)
    return keep

# ---------------------------------------------------------------- main
def main():
    argv = sys.argv[1:]
    cfg = dict(CONFIG)
    if cfg["PREVIEW"]:
        cfg["WIDTH"] = 960; cfg["FPS"] = 15; cfg["CRF"] = 23
    cfg["_DPI"] = cfg["WIDTH"] / 25.6 * 1.6      # 3840 -> 240 dpi on a 16x9 fig
    cfg["_DPI"] = int(round(cfg["WIDTH"] / 16))
    outdir = os.path.join(HERE, "thesis_arc_out", "videos")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, ("PREVIEW_" if cfg["PREVIEW"] else "") + cfg["NAME"])
    total = int(cfg["SECONDS"] * cfg["FPS"])

    if cfg["BLACK_BG"]:
        M.bg_space = _bg_black                    # match Final_Video
    if cfg["BLENDER_LOOK"]:
        M.render_frame = render_frame_bl          # full-bleed, Final_Video captions

    if len(argv) >= 4 and argv[0] == "--worker":
        i0, i1, chunk = int(argv[1]), int(argv[2]), argv[3]
        prog = argv[4] if len(argv) > 4 else None
        D = M.load_all(os.path.join(HERE, "thesis_arc_out", "arc_npz"))
        specs = build_specs_segment(D, total, cfg["FPS"], cfg["SPLIT"])
        M.render_range(D, specs, i0, i1, chunk, cfg["FPS"], cfg["_DPI"], cfg["CRF"], prog=prog)
        return

    print(f"[redock] {cfg['WIDTH']}x{int(cfg['WIDTH']*9/16)} @ {cfg['FPS']} fps · "
          f"{cfg['SECONDS']} s · {total} frames · black bg={cfg['BLACK_BG']}")
    M._ensure_imageio()
    D = M.load_all(os.path.join(HERE, "thesis_arc_out", "arc_npz"))
    specs = build_specs_segment(D, total, cfg["FPS"], cfg["SPLIT"])
    phases = {}
    for sp in specs:
        k = sp.get("phase") or sp["kind"]; phases[k] = phases.get(k, 0) + 1
    print(f"[redock] {len(specs)} frames -> " +
          " · ".join(f"{k} {v} ({v/cfg['FPS']:.0f}s)" for k, v in phases.items()))

    fig = plt.figure(figsize=(16, 9), dpi=cfg["_DPI"])
    t0 = time.time(); M.render_frame(fig, D, specs[len(specs)//2]); fig.canvas.draw()
    spf = time.time() - t0; plt.close(fig)
    nw = cfg["WORKERS"] or max(1, (os.cpu_count() or 4) - 2)
    nw = int(min(nw, 10, len(specs) // 30 or 1))
    print(f"[redock] ~{spf:.2f}s/frame -> est {spf*len(specs)/nw/60:.0f} min on {nw} worker(s)")

    if nw == 1:
        M.render_range(D, specs, 0, len(specs), out, cfg["FPS"], cfg["_DPI"], cfg["CRF"], tag="1/1")
    else:
        tmp = os.path.join(outdir, f"_rd_chunks_{len(specs)}f_{cfg['_DPI']}dpi_{nw}w")
        os.makedirs(tmp, exist_ok=True)
        b = np.linspace(0, len(specs), nw + 1).astype(int)
        chunks = [os.path.join(tmp, f"c_{i:02d}.mp4") for i in range(nw)]
        progs = [c + ".prog" for c in chunks]
        todo = [i for i in range(nw) if not os.path.exists(chunks[i] + ".ok")]
        env = dict(os.environ); env["REDOCK_DIR"] = HERE
        procs = {}
        for i in todo:
            procs[i] = subprocess.Popen([sys.executable, os.path.abspath(__file__),
                                         "--worker", str(b[i]), str(b[i+1]),
                                         chunks[i], progs[i]], env=env)
            time.sleep(1.0)
        failed = None
        while any(p.poll() is None for p in procs.values()):
            time.sleep(6); done = 0
            for i in range(nw):
                if os.path.exists(chunks[i] + ".ok"): done += int(b[i+1] - b[i]); continue
                try: done += int(open(progs[i]).read().split()[0])
                except Exception: pass
            print(f"[redock] {done}/{len(specs)} frames ({100*done/len(specs):.0f}%)", flush=True)
            bad = [i for i, p in procs.items() if p.poll() not in (None, 0)]
            if bad:
                failed = bad
                for p in procs.values():
                    if p.poll() is None: p.terminate()
                break
        if failed is None:
            failed = [i for i, p in procs.items() if p.returncode not in (0, None)]
        if failed:
            raise RuntimeError(f"worker(s) {sorted(failed)} failed — lower WORKERS and re-run")
        M._concat(chunks, out)
        for c in chunks + progs + [c + ".ok" for c in chunks]:
            try: os.remove(c)
            except Exception: pass
        try:
            os.remove(os.path.join(tmp, "concat.txt")); os.rmdir(tmp)
        except Exception: pass
    print(f"[done] -> {out}")

if __name__ == "__main__":
    main()
