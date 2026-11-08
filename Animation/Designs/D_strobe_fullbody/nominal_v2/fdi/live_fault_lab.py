#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_fault_lab.py — RUN THIS (no arguments).

A full-blown LIVE fault lab for the ARISE/Apophis onboard FDIR. It drives the REAL FDI
core (one fdi_core.FDICore per deputy, all six channels) under a randomized, high-fidelity
fault campaign — faults perturb a TRUE state, the FDI consumes only the onboard EKF
estimates — and shows the whole autonomy live.

ONE WINDOW (default), ALL NINE VIEWS, MAIN SCREEN — no flicker, nothing to arrange
----------------------------------------------------------------------------------
By default (LAYOUT = "single") every view is a panel inside a SINGLE maximized window on
your main screen, in a clean 3×3 grid: Mission Ops · Autonomy Brain · FDI Scope /
Innovation Monitor · Subsystems · Coverage & Strobe / Deputy Deep-Dive · ISL Network ·
Loss-Compensation. One figure + one draw loop = no flicker, no window juggling. The text is
shrunk (low figure DPI) so nothing overlaps. The mission plays over ~PLAYBACK_SECONDS
(wall-clock); if it falls behind it drops frames to stay real-time.

(matplotlib renders on the CPU — it cannot use the GPU. Two other layouts exist if you want
them: LAYOUT="consolidated" = two windows, one per monitor; LAYOUT="separate" = nine windows.)

Just press Run in Visual Studio. No command-line arguments; numpy + matplotlib auto-install.
"""
from __future__ import annotations
import os, sys, importlib, subprocess


def _ensure(pkg, mod=None):
    try:
        return importlib.import_module(mod or pkg)
    except ImportError:
        print(f"[setup] installing {pkg} (one-time)…", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(mod or pkg)


for _p in ("numpy", "matplotlib"):
    _ensure(_p)

# ─────────────────────────────── CONFIG (edit here, no args) ─────────────────
LAYOUT           = "custom"     # "custom"  = clean per-WINDOW layout (fdi_layout.json) — recommended;
                                # "elements"= ONE window placed by your ELEMENT layout (fdi_elements.json,
                                #             from capture_element_layout.py + fdi_element_editor.py — every
                                #             graph + text positioned individually). Falls back to 3×3.
                                # "custom"  = ONE window from a WINDOW-level design (fdi_layout.json,
                                #             arranged with interactive_fault_lab.py);
                                # "single"  = ALL panels in ONE window, fixed 3×3 grid;
                                # "consolidated" = 2 windows (1/monitor); "separate" = 9 windows
FONT_DPI         = 92           # base text size (higher = bigger; try 70–95). Saved font scales multiply on top.
SIM_FRAMES       = 1000         # mission resolution (validated false-alarm-free to ~1000)
PLAYBACK_SECONDS = 330          # the ~10.6 h mission plays over this long (wall-clock) ≈ 5.5 min; raise to slow further
SEED             = 7            # fault-campaign seed — change for a different randomized campaign
LOOP_FOREVER     = False        # replay continuously
SAVE_SUMMARY     = True         # save a final snapshot to figs/
SHOW = dict(ops=True, scope=True, subsystems=True, architecture=True, deputy=True, isl=True,
            innovations=True, coverage=True, loss=True)

# ── screen placement (pixel coords in the Windows virtual desktop) ────────────
# Main = your 2K. (Only used to size/position the window; you can also just maximize it.)
SCREEN_MAIN   = dict(x=0,    y=0, w=2560, h=1440)   # 2K main screen
SCREEN_SECOND = dict(x=2560, y=0, w=1920, h=1080)   # 1080p (only used by LAYOUT="consolidated")
TASKBAR_PX    = 48
# ─────────────────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN_D = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_TRAJ = os.path.join(DESIGN_D, "data", "trajectories")
DATA_ANIM = os.path.join(DESIGN_D, "data", "anim")
MESH_NPZ = os.path.join(HERE, "..", "data", "radar_mesh.npz")
FACET_NPZ = os.path.join(HERE, "..", "data", "mc_v2_results_nominal_facetprob.npz")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

GRID_ORDER = ["ops", "architecture", "scope", "innovations", "subsystems", "coverage", "deputy", "isl", "loss"]
MAIN_PANELS = ["ops", "architecture", "scope", "innovations", "subsystems", "coverage"]
SECOND_PANELS = ["deputy", "isl", "loss"]
# short panel headers used when views are tiled into one window (the full titles overflow a 1/9 cell)
SHORT_TITLES = {"ops": "W1 · MISSION OPS", "architecture": "W4 · AUTONOMY BRAIN", "scope": "W2 · FDI SCOPE",
                "innovations": "W7 · INNOVATION MONITOR", "subsystems": "W3 · SUBSYSTEMS",
                "coverage": "W8 · COVERAGE & STROBE", "deputy": "W5 · DEPUTY DEEP-DIVE",
                "isl": "W6 · ISL NETWORK", "loss": "W9 · LOSS COMPENSATION"}


def _place(fig, x, y, w, h):
    try:
        m = fig.canvas.manager
        try:
            m.window.wm_geometry(f"{int(w)}x{int(h)}+{int(x)}+{int(y)}")
        except Exception:
            m.window.setGeometry(int(x), int(y), int(w), int(h))
    except Exception:
        pass


def _maximize(fig):
    try:
        m = fig.canvas.manager
        try:
            m.window.state("zoomed")          # TkAgg on Windows
        except Exception:
            m.window.showMaximized()          # Qt
    except Exception:
        pass


def _grid(scr, cols, rows, pad=6):
    tw = scr["w"] // cols
    th = (scr["h"] - TASKBAR_PX) // rows
    return [(scr["x"] + c * tw + pad, scr["y"] + r * th + pad, tw - 2 * pad, th - 2 * pad)
            for r in range(rows) for c in range(cols)]


def _make_window(name, engine, truth):
    if name == "ops":
        import cockpit_ops as m; return m.OpsCockpit(engine, truth, truth["T"].size)
    if name == "architecture":
        import faultlab_brain as m; return m.AutonomyBrainWindow(engine)
    if name == "scope":
        import faultlab_windows as m; return m.ChannelScopeWindow(engine)
    if name == "deputy":
        import faultlab_windows as m; return m.DeputyDeepDiveWindow(engine)
    if name == "subsystems":
        import faultlab_subsystems as m; return m.SubsystemsWindow(engine)
    if name == "isl":
        import faultlab_subsystems as m; return m.ISLNetworkWindow(engine)
    if name == "innovations":
        import faultlab_innovations as m; return m.InnovationMonitorWindow(engine)
    if name == "coverage":
        import faultlab_coverage as m; return m.CoverageStrobeWindow(engine)
    if name == "loss":
        import faultlab_loss as m; return m.LossCompensationWindow(engine)
    raise ValueError(name)


def _embed(name, engine, truth, subfig):
    """Build a window but make its internal `plt.figure(...)` return THIS subfigure,
    so the (unchanged) window draws into one panel of a combined window."""
    import matplotlib.pyplot as plt
    real = plt.figure
    plt.figure = lambda *a, **k: subfig
    try:
        w = _make_window(name, engine, truth)
    finally:
        plt.figure = real
    if LAYOUT != "separate":           # replace the long per-window title with a short panel header
        try:
            subfig.suptitle(SHORT_TITLES.get(name, name.upper()), color="#33bbff", fontsize=11,
                            weight="bold", y=0.985)
        except Exception:
            pass
    return w


def _build_engine(sim_frames, seed):
    import cockpit_engine as ce
    import faultlab_engine as fe
    truth = ce.load_truth(DATA_TRAJ, DATA_ANIM, MESH_NPZ, FACET_NPZ, n_frames=sim_frames)
    return fe.default_build(truth, seed=seed), truth


def _loop(plt, figs, wins, engine, truth, place_fn):
    import time
    plt.show(block=False)
    try:
        plt.pause(0.5)
    except Exception:
        pass
    place_fn()
    try:
        plt.pause(0.2)
    except Exception:
        pass
    place_fn()
    F = truth["T"].size; spf = max(PLAYBACK_SECONDS, 1) / F
    while True:
        t0 = time.time(); last = -1; st = None
        while True:
            target = min(int((time.time() - t0) / spf), F - 1)
            if target > last:
                for fi in range(last + 1, target + 1):
                    st = engine.step(fi)
                last = target
                for n, w in wins:
                    try:
                        w.update(st)
                    except Exception as e:
                        print(f"[{n}] update error: {e}", flush=True)
                for f in figs:
                    try:
                        f.canvas.draw_idle()
                    except Exception:
                        pass
            try:
                plt.pause(0.004)
            except Exception:
                return
            if last >= F - 1:
                break
        if not LOOP_FOREVER:
            break
        engine, truth = _build_engine(SIM_FRAMES, SEED)
    if SAVE_SUMMARY:
        try:
            os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
            for i, f in enumerate(figs):
                f.savefig(os.path.join(HERE, "figs", f"faultlab_window{i+1}.png"), dpi=110, facecolor="black")
        except Exception:
            pass
    print("[faultlab] mission complete — close the window to exit.")
    try:
        plt.ioff(); plt.show()
    except Exception:
        pass


def _run_single_window(plt, engine, truth):
    """ALL nine views as panels in ONE maximized window on the main screen (3×3 grid)."""
    fig = plt.figure(figsize=(SCREEN_MAIN["w"] / 100.0, SCREEN_MAIN["h"] / 100.0), dpi=FONT_DPI)
    fig.set_facecolor("black"); fig.patch.set_facecolor("black")
    try:
        fig.canvas.manager.set_window_title("ARISE / APOPHIS  —  FDIR FAULT LAB  (all views)")
    except Exception:
        pass
    sf = fig.subfigures(3, 3, wspace=0.015, hspace=0.03)
    pos = {n: sf[i // 3, i % 3] for i, n in enumerate(GRID_ORDER)}
    for s in pos.values():
        try:
            s.patch.set_facecolor("black"); s.set_facecolor("black")
        except Exception:
            pass
    wins = [(n, _embed(n, engine, truth, pos[n])) for n in GRID_ORDER if SHOW.get(n)]
    print(f"[faultlab] one window, {len(wins)} panels (3×3), maximized on the main screen. "
          f"If text looks big/small, tune FONT_DPI at the top.", flush=True)
    _loop(plt, [fig], wins, engine, truth,
          lambda: (_place(fig, SCREEN_MAIN["x"], SCREEN_MAIN["y"], SCREEN_MAIN["w"], SCREEN_MAIN["h"]), _maximize(fig)))


def _run_consolidated(plt, engine, truth):
    """Two windows, one per monitor (6 panels on the 2K, 3 on the 1080p)."""
    figA = plt.figure(figsize=(25.6, 14.4), dpi=FONT_DPI); figA.set_facecolor("black"); figA.patch.set_facecolor("black")
    figB = plt.figure(figsize=(19.2, 10.8), dpi=FONT_DPI); figB.set_facecolor("black"); figB.patch.set_facecolor("black")
    for f, t in ((figA, "FDIR — MAIN (2K)"), (figB, "FDIR — AUX (1080p)")):
        try:
            f.canvas.manager.set_window_title(t)
        except Exception:
            pass
    sfA = figA.subfigures(2, 3, wspace=0.02, hspace=0.04)
    sfB = figB.subfigures(2, 2, wspace=0.02, hspace=0.04)
    posA = {n: sfA[i // 3, i % 3] for i, n in enumerate(MAIN_PANELS)}
    posB = {n: sfB[i // 2, i % 2] for i, n in enumerate(SECOND_PANELS)}
    for s in list(posA.values()) + list(posB.values()) + [sfB[1, 1]]:
        try:
            s.patch.set_facecolor("black"); s.set_facecolor("black")
        except Exception:
            pass
    wins = [(n, _embed(n, engine, truth, sf)) for n, sf in list(posA.items()) + list(posB.items()) if SHOW.get(n)]
    _loop(plt, [figA, figB], wins, engine, truth,
          lambda: (_place(figA, **{k: SCREEN_MAIN[k] for k in ("x", "y", "w", "h")}),
                   _place(figB, **{k: SCREEN_SECOND[k] for k in ("x", "y", "w", "h")})))


def _run_separate(plt, engine, truth):
    """Nine separate windows in one process, tiled across the two screens."""
    import time
    slots = {}
    for n, g in zip([n for n in MAIN_PANELS if SHOW.get(n)], _grid(SCREEN_MAIN, 3, 2)):
        slots[n] = g
    for n, g in zip([n for n in SECOND_PANELS if SHOW.get(n)], _grid(SCREEN_SECOND, 2, 2)):
        slots[n] = g
    names = [n for n in GRID_ORDER if SHOW.get(n) and n in slots]
    wins = {n: _make_window(n, engine, truth) for n in names}
    plt.show(block=False)
    try:
        plt.pause(0.4)
    except Exception:
        pass
    for n, w in wins.items():
        _place(w.fig, *slots[n])
    F = truth["T"].size; spf = max(PLAYBACK_SECONDS, 1) / F
    t0 = time.time(); last = -1; st = None
    while True:
        target = min(int((time.time() - t0) / spf), F - 1)
        if target > last:
            for fi in range(last + 1, target + 1):
                st = engine.step(fi)
            last = target
            for n, w in wins.items():
                try:
                    w.update(st); w.fig.canvas.draw_idle()
                except Exception as e:
                    print(f"[{n}] err: {e}")
        try:
            plt.pause(0.004)
        except Exception:
            break
        if last >= F - 1:
            break
    plt.ioff(); plt.show()


def _loop_custom(plt, fig, builder, engine, truth, design, place_fn):
    """Live, paced playback for the custom (saved-design) single-window layout."""
    import time
    titles, font_scale = design["titles"], design["font_scale"]
    plt.show(block=False)
    try:
        plt.pause(0.5)
    except Exception:
        pass
    place_fn()
    try:
        plt.pause(0.2)
    except Exception:
        pass
    place_fn()
    F = truth["T"].size; spf = max(PLAYBACK_SECONDS, 1) / F
    while True:
        t0 = time.time(); last = -1
        while True:
            target = min(int((time.time() - t0) / spf), F - 1)
            if target > last:
                st = None
                for fi in range(last + 1, target + 1):
                    st = engine.step(fi)
                last = target
                builder.refresh(st, titles, font_scale)
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass
            try:
                plt.pause(0.004)
            except Exception:
                return
            if last >= F - 1:
                break
        if not LOOP_FOREVER:
            break
        engine, truth = _build_engine(SIM_FRAMES, SEED)
        builder.engine, builder.truth = engine, truth
        fig.clear()
        builder.build(design["layout"], titles, design["shown"], font_scale)
    if SAVE_SUMMARY:
        try:
            os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
            fig.savefig(os.path.join(HERE, "figs", "faultlab_custom.png"), dpi=110, facecolor="black")
        except Exception:
            pass
    print("[faultlab] mission complete — close the window to exit.")
    try:
        plt.ioff(); plt.show()
    except Exception:
        pass


def _run_custom_window(plt, engine, truth):
    """ALL panels in ONE window, placed by YOUR saved design (fdi_layout.json): positions,
    sizes, titles, hidden panels and font size. Arrange it with interactive_fault_lab.py and
    press Save; if no design is saved yet this falls back to a clean 3×3."""
    import fdi_layout as FL
    FL.install_font_patch()
    d = FL.load(FL.layout_path(HERE), GRID_ORDER)
    if not d["found"]:
        print("[faultlab] no fdi_layout.json yet — using the default 3×3. Arrange one with "
              "interactive_fault_lab.py (press Save), then re-run.", flush=True)
    # a panel hidden in the design OR switched off in SHOW stays hidden
    shown = {n: bool(d["shown"].get(n, True) and SHOW.get(n, True)) for n in GRID_ORDER}
    design = dict(layout=d["layout"], titles=d["titles"], shown=shown, font_scale=d["font_scale"])

    fig = plt.figure(figsize=(SCREEN_MAIN["w"] / 100.0, SCREEN_MAIN["h"] / 100.0), dpi=FONT_DPI)
    fig.set_facecolor("black"); fig.patch.set_facecolor("black")
    try:
        fig.canvas.manager.set_window_title("ARISE / APOPHIS  —  FDIR FAULT LAB  (custom layout)")
    except Exception:
        pass
    builder = FL.TileBuilder(fig, _make_window, engine, truth, GRID_ORDER)
    builder.build(design["layout"], design["titles"], design["shown"], design["font_scale"])
    print(f"[faultlab] custom layout · {sum(shown.values())} panels · font ×{design['font_scale']:.2f}"
          + ("" if d["found"] else " (default 3×3 — no saved design)"), flush=True)
    _loop_custom(plt, fig, builder, engine, truth, design,
                 lambda: (_place(fig, SCREEN_MAIN["x"], SCREEN_MAIN["y"], SCREEN_MAIN["w"], SCREEN_MAIN["h"]),
                          _maximize(fig)))


def _loop_elements(plt, fig, wins, win_axes, engine, truth, cat, per_win_ov, font_scale, place_fn):
    """Live playback for the element layout: place every graph each frame + apply text/font edits."""
    import time
    import fdi_layout as FL
    import fdi_elements as E
    plt.show(block=False)
    try:
        plt.pause(0.5)
    except Exception:
        pass
    place_fn()
    try:
        plt.pause(0.2)
    except Exception:
        pass
    place_fn()
    F = truth["T"].size; spf = max(PLAYBACK_SECONDS, 1) / F
    while True:
        t0 = time.time(); last = -1
        while True:
            target = min(int((time.time() - t0) / spf), F - 1)
            if target > last:
                st = None
                for fi in range(last + 1, target + 1):
                    st = engine.step(fi)
                last = target
                FL.set_scale(font_scale); FL.scale_on()
                old_fs = plt.rcParams.get("font.size", 10); plt.rcParams["font.size"] = 10 * font_scale
                try:
                    for n, w in wins:
                        try:
                            w.update(st)
                        except Exception as e:
                            print(f"[{n}] update error: {e}", flush=True)
                finally:
                    FL.scale_off(); plt.rcParams["font.size"] = old_fs
                E.apply_graphs(win_axes, cat)          # hold each graph at its saved box
                for _n, _axl in win_axes.items():       # apply your text edits within each window only
                    _ov = per_win_ov.get(_n)
                    if _ov:
                        for _ax in _axl:
                            E.apply_text(_ax, _ov, font_scale)
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass
            try:
                plt.pause(0.004)
            except Exception:
                return
            if last >= F - 1:
                break
        if not LOOP_FOREVER:
            break
    print("[faultlab] mission complete — close the window to exit.")
    try:
        plt.ioff(); plt.show()
    except Exception:
        pass


def _run_elements_window(plt, engine, truth):
    """ONE window, every GRAPH and TEXT placed by your saved element layout (fdi_elements.json).
    Make it with capture_element_layout.py (capture the real positions) → fdi_element_editor.py
    (rearrange) → here. Falls back to the fixed 3×3 if no element layout exists yet."""
    import fdi_elements as E
    import fdi_layout as FL
    path = E.layout_path(HERE)
    if not os.path.exists(path):
        print("[faultlab] no fdi_elements.json yet — run capture_element_layout.py then "
              "fdi_element_editor.py. Using the fixed 3×3 single window for now.", flush=True)
        return _run_single_window(plt, engine, truth)
    cat = E.load(path)
    FL.install_font_patch()
    font_scale = cat.get("font_scale", 1.0)

    fig = plt.figure(figsize=(SCREEN_MAIN["w"] / 100.0, SCREEN_MAIN["h"] / 100.0), dpi=FONT_DPI)
    fig.set_facecolor("black"); fig.patch.set_facecolor("black")
    try:
        fig.canvas.manager.set_window_title("ARISE / APOPHIS  —  FDIR FAULT LAB  (element layout)")
    except Exception:
        pass

    # build ALL windows flat into this one figure, tracking each window's axes (creation order)
    win_axes = {}; wins = []
    real = plt.figure
    plt.figure = lambda *a, **k: fig
    try:
        for n in GRID_ORDER:
            if not SHOW.get(n):
                continue
            before = list(fig.axes)
            w = _make_window(n, engine, truth)
            win_axes[n] = [ax for ax in fig.axes if ax not in before]
            wins.append((n, w))
    finally:
        plt.figure = real
    try:
        fig.suptitle("")
    except Exception:
        pass

    per_win_ov = E.text_overrides(cat)          # {window: {orig: {...}}} — applied per window, not globally
    E.apply_graphs(win_axes, cat)
    for _t in cat.get("texts", []):             # panel titles were per-window suptitles (one figure can
        if _t.get("title"):                     #   only hold one) → redraw each as static figure text
            try:
                fig.text(_t["pos"][0], _t["pos"][1], _t.get("text", ""), color=_t.get("color", "#33ddff"),
                         fontsize=_t.get("fontsize", 12) * font_scale, ha=_t.get("ha", "left"),
                         va=_t.get("va", "top"), weight="bold", zorder=5)
            except Exception:
                pass
    print(f"[faultlab] element layout · {len(cat['graphs'])} graphs · font ×{font_scale:.2f}", flush=True)
    _loop_elements(plt, fig, wins, win_axes, engine, truth, cat, per_win_ov, font_scale,
                   lambda: (_place(fig, SCREEN_MAIN["x"], SCREEN_MAIN["y"], SCREEN_MAIN["w"], SCREEN_MAIN["h"]),
                            _maximize(fig)))


def main():
    if not os.path.isdir(DATA_TRAJ):
        raise SystemExit(f"[faultlab] trajectory data not found at {DATA_TRAJ}\n"
                         f"           run the nominal_v2 pipeline first or fix the CONFIG paths.")
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    plt.ion()
    print(f"[faultlab] {SIM_FRAMES} frames · plays over ~{PLAYBACK_SECONDS}s · layout='{LAYOUT}'", flush=True)
    engine, truth = _build_engine(SIM_FRAMES, SEED)
    if LAYOUT == "separate":
        _run_separate(plt, engine, truth)
    elif LAYOUT == "consolidated":
        _run_consolidated(plt, engine, truth)
    elif LAYOUT == "custom":
        _run_custom_window(plt, engine, truth)
    elif LAYOUT == "elements":
        _run_elements_window(plt, engine, truth)
    else:
        _run_single_window(plt, engine, truth)
    print("[faultlab] done.")


if __name__ == "__main__":
    main()
