#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_faultlab_film.py — 4K FILM of the ARISE/Apophis FDIR FAULT LAB.

Renders the SAME dashboard as live_fault_lab.py (LAYOUT="custom": your saved
fdi_layout.json positions, titles, hidden panels, font scale) driven by the real
FDI core over the full randomized fault campaign — but offline, frame-exact, at
3840x2160 H.264.

Press Run (no CLI args).  The script re-launches itself as worker processes;
the deterministic engine (seed) is fast-forwarded in each worker, so chunks are
bit-identical to a serial render.  Finished chunks are kept and reused if a run
is interrupted (resume).  Edit CONFIG below.
"""
from __future__ import annotations
import os, sys, json, time, subprocess
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.environ.get("FDI_OVERRIDE_PATH"):          # dev hook: shadow modules for testing
    sys.path.insert(0, os.environ["FDI_OVERRIDE_PATH"])

# ================================================================ CONFIG
CONFIG = dict(
    WIDTH      = 3840,     # 3840 = 4K UHD (16:9 locked)
    FPS        = 30,
    DUP        = 4,        # video frames per sim frame → 1000 sim frames ≈ 133 s film
    SIM_FRAMES = 1000,     # mission resolution (matches live_fault_lab)
    SEED       = 7,        # same seed = same fault campaign as the live lab
    FONT_SCALE = None,     # None → use fdi_layout.json font_scale; or override, e.g. 1.35
    WORKERS    = None,     # None → auto (cpu-2, capped 10); 1 = serial
    CRF        = 17,       # x264 quality (17 ≈ visually lossless)
    PREVIEW    = False,    # True → 960x540 / 15 fps quick draft
    OUT        = None,     # default: ./figs/videos
    NAME       = "ARISE_FDIR_fault_lab_4K.mp4",
)
# =================================================================

import live_fault_lab as LFL          # window factory + data paths (no GUI started)
import fdi_layout as FL


def build_dashboard(engine, truth, dpi, font_scale=None):
    """One 16:9 figure with the saved custom layout, ready for refresh()."""
    FL.install_font_patch()
    d = FL.load(FL.layout_path(HERE), LFL.GRID_ORDER)
    shown = {n: bool(d["shown"].get(n, True) and LFL.SHOW.get(n, True)) for n in LFL.GRID_ORDER}
    fs = float(font_scale if font_scale else d["font_scale"])
    # 25.6 x 14.4 in — the SAME canvas proportions the live window was designed on
    # (2560x1440 @ FONT_DPI≈100), so point-sized text keeps its intended relative size.
    fig = plt.figure(figsize=(25.6, 14.4), dpi=dpi)
    fig.set_facecolor("black"); fig.patch.set_facecolor("black")
    builder = FL.TileBuilder(fig, LFL._make_window, engine, truth, LFL.GRID_ORDER)
    builder.build(d["layout"], d["titles"], shown, fs)
    # footer strip — at the very bottom edge, below every tile
    fig.text(0.006, 0.012, "ARISE / APOPHIS · FDIR FAULT LAB — real FDI core · randomized fault campaign",
             color="#39d9ff", fontsize=10.5 * fs / 1.25, weight="bold")
    foot = fig.text(0.994, 0.012, "", color="#9ab0cc", fontsize=9.5 * fs / 1.25,
                    ha="right", family="monospace")
    return fig, builder, d["titles"], fs, foot


def render_range(i0, i1, path, cfg, prog=None, tag=""):
    engine, truth = LFL._build_engine(cfg["SIM_FRAMES"], cfg["SEED"])
    fig, builder, titles, fs, foot = build_dashboard(engine, truth, cfg["_DPI"], cfg["FONT_SCALE"])
    st = None
    for fi in range(i0):                       # deterministic fast-forward (no drawing)
        st = engine.step(fi)
    import imageio.v2 as iio
    w = iio.get_writer(path, fps=cfg["FPS"], codec="libx264", quality=None,
        macro_block_size=1, pixelformat="yuv420p",
        ffmpeg_params=["-crf", str(cfg["CRF"]), "-preset", "veryfast",
                       "-rc-lookahead", "20", "-threads", "2",
                       "-profile:v", "high", "-level", "5.1"])
    t0 = time.time()
    for j, fi in enumerate(range(i0, i1)):
        st = engine.step(fi)
        builder.refresh(st, titles, fs)
        nflt = len(st.get("gt_fault", {}))
        foot.set_text(f"frame {fi+1:4d}/{cfg['SIM_FRAMES']}  ·  t-to-CA {st['t_go']/3600:5.2f} h"
                      f"  ·  active faults {nflt}  ·  seed {cfg['SEED']}")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        hh, ww = buf.shape[0] // 2 * 2, buf.shape[1] // 2 * 2
        frame = np.ascontiguousarray(buf[:hh, :ww])
        try:
            for _ in range(cfg["DUP"]):
                w.append_data(frame)
        except OSError as e:
            raise RuntimeError("ffmpeg encoder pipe died (memory pressure?) — "
                               "lower CONFIG WORKERS and re-run") from e
        if prog and (j % 10 == 0 or fi == i1 - 1):
            try:
                with open(prog, "w") as f:
                    f.write(f"{j+1} {i1-i0}")
            except Exception:
                pass
        if tag and j % 25 == 0:
            print(f"  [{tag}] {j+1}/{i1-i0} ({(time.time()-t0)/(j+1):.2f}s/frame)", flush=True)
    w.close(); plt.close(fig)
    open(path + ".ok", "w").write("done")


def _concat(chunks, out):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    lst = os.path.join(os.path.dirname(chunks[0]), "concat.txt")
    with open(lst, "w") as f:
        for c in chunks:
            f.write("file '%s'\n" % c.replace("\\", "/"))
    r = subprocess.run([ff, "-y", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", "-movflags", "+faststart", out],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg concat failed:\n" + r.stderr.decode()[-800:])


def main():
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        i0, i1 = int(sys.argv[2]), int(sys.argv[3])
        cfg = json.loads(os.environ["FAULTLAB_CFG"])
        render_range(i0, i1, sys.argv[4], cfg, prog=sys.argv[5] if len(sys.argv) > 5 else None)
        return

    cfg = dict(CONFIG)
    if cfg["PREVIEW"]:
        cfg["WIDTH"] = 960; cfg["FPS"] = 15; cfg["DUP"] = 2; cfg["CRF"] = 23
    cfg["_DPI"] = cfg["WIDTH"] / 25.6          # 3840 -> 150 dpi (exact 3840x2160)
    outdir = cfg["OUT"] or os.path.join(HERE, "figs", "videos")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, ("PREVIEW_" if cfg["PREVIEW"] else "") + cfg["NAME"])
    total = int(cfg["SIM_FRAMES"])
    secs = total * cfg["DUP"] / cfg["FPS"]
    print(f"[faultlab-film] {cfg['WIDTH']}x{int(cfg['WIDTH']*9/16)} @ {cfg['FPS']} fps · "
          f"{total} sim frames ×{cfg['DUP']} → {secs:.0f} s film", flush=True)

    # probe one frame for an ETA
    engine, truth = LFL._build_engine(cfg["SIM_FRAMES"], cfg["SEED"])
    fig, builder, titles, fs, foot = build_dashboard(engine, truth, cfg["_DPI"], cfg["FONT_SCALE"])
    st = engine.step(0)
    t0 = time.time(); builder.refresh(st, titles, fs); fig.canvas.draw()
    spf = time.time() - t0; plt.close(fig)

    nw = cfg["WORKERS"] or max(1, (os.cpu_count() or 4) - 2)
    nw = int(min(nw, 10, total // 20 or 1))
    print(f"[faultlab-film] ~{spf:.2f}s/frame → est {spf*total/nw/60:.0f} min on {nw} worker(s)", flush=True)

    if nw == 1:
        render_range(0, total, out, cfg, tag="1/1")
    else:
        tmp = os.path.join(outdir, f"_chunks_{total}f_{cfg['_DPI']}dpi_{nw}w")
        os.makedirs(tmp, exist_ok=True)
        bounds = np.linspace(0, total, nw + 1).astype(int)
        chunks = [os.path.join(tmp, f"chunk_{i:02d}.mp4") for i in range(nw)]
        progs = [c + ".prog" for c in chunks]
        todo = [i for i in range(nw) if not os.path.exists(chunks[i] + ".ok")]
        if len(todo) < nw:
            print(f"[faultlab-film] resume: {nw-len(todo)}/{nw} chunks already complete", flush=True)
        env = dict(os.environ); env["FAULTLAB_CFG"] = json.dumps(cfg)
        procs = {}
        for i in todo:                       # staggered spawn — no encoder burst
            procs[i] = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker",
                                         str(bounds[i]), str(bounds[i+1]), chunks[i], progs[i]], env=env)
            time.sleep(1.2)
        failed = None
        while any(p.poll() is None for p in procs.values()):
            time.sleep(8); done = 0
            for i in range(nw):
                if os.path.exists(chunks[i] + ".ok"):
                    done += int(bounds[i+1] - bounds[i]); continue
                try:
                    done += int(open(progs[i]).read().split()[0])
                except Exception:
                    pass
            print(f"[faultlab-film] {done}/{total} frames ({100*done/total:.0f}%)", flush=True)
            bad = [i for i, p in procs.items() if p.poll() not in (None, 0)]
            if bad:                          # abort everything immediately, keep finished chunks
                failed = bad
                for p in procs.values():
                    if p.poll() is None:
                        p.terminate()
                break
        if failed is None:
            failed = [i for i, p in procs.items() if p.returncode not in (0, None)]
        if failed:
            raise RuntimeError(f"worker(s) {sorted(failed)} failed — re-run with CONFIG "
                               f"WORKERS={max(2, nw//2)}; finished chunks are reused (resume).")
        _concat(chunks, out)
        for c in chunks + progs + [c + ".ok" for c in chunks]:
            try: os.remove(c)
            except Exception: pass
        try:
            os.remove(os.path.join(tmp, "concat.txt")); os.rmdir(tmp)
        except Exception: pass
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
