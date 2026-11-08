#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interactive_fault_lab.py — the STATIC LAYOUT DESIGNER for the FDIR fault lab.

This does NOT run the mission. It shows the nine views as a single frozen, representative
snapshot so you can calmly ARRANGE them: drag panels around, resize them, rename their headers
and scale the text — then SAVE. Your design is written to `fdi_layout.json`.

Then run the real simulation with that design:  open **live_fault_lab.py** (it defaults to
LAYOUT = "custom") and press Run — it renders the live mission using exactly the layout,
titles, hidden panels and font size you saved here.

WORKFLOW
--------
  1) Run THIS file.  →  arrange / rename / resize / set font  →  press **Save**.
  2) Run live_fault_lab.py  →  the mission plays inside your saved design.

CONTROLS  (buttons along the bottom; keyboard shortcut in [brackets])
  • [e]  Edit Layout ....... enter/leave edit mode. In edit mode each panel is a movable box:
                             drag the body to move · drag the bottom-right corner to resize.
  • [+] / [-]  Font + / − .. make all text bigger / smaller (saved with the design).
  • [r]  Rename ............ rename the selected panel's header (click a panel to select it).
  • [h]  Hide  /  [a] Show all  remove a panel you don't want / bring them all back.
  • [s]  Save  /  [l] Load  /  [0] Reset .. save the design · reload it · reset to 3×3.
  • close the window (or [q]) to finish.

The snapshot is frozen at frame PREVIEW_FRAME purely so you can see what each panel looks like.
matplotlib renders on the CPU. Just press Run — numpy + matplotlib auto-install.
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

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import live_fault_lab as base   # engine + window factory (reused; nothing duplicated)
import fdi_layout as FL         # shared tile layout / save-load / builder

# ─────────────────────────────── CONFIG (edit here, no args) ─────────────────
SIM_FRAMES    = 1000      # mission resolution (matches live_fault_lab)
SEED          = 7         # fault-campaign seed (matches live_fault_lab)
PREVIEW_FRAME = 600       # which mission frame to freeze for the preview snapshot
BASE_DPI      = 80        # base text size; Font +/- multiplies on top and is saved with the design
LAYOUT_FILE   = FL.layout_path(HERE)
SCREEN        = base.SCREEN_MAIN

PANELS = list(FL.DEFAULT_ORDER)

import matplotlib   # noqa: E402


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class LayoutDesigner:
    def __init__(self, plt, engine, truth):
        self.plt = plt
        self.engine = engine
        self.truth = truth
        self.closed = False
        self.edit = False
        self.sel = None
        self.drag = None
        self._bg = None

        FL.install_font_patch()

        # design state (load a saved design if present, else defaults)
        d = FL.load(LAYOUT_FILE, PANELS)
        self.layout = d["layout"]
        self.titles = d["titles"]
        self.shown = d["shown"]
        self.font_scale = d["font_scale"]

        # the single frozen frame everything previews
        self.last_state = FL.representative_state(engine, truth["T"].size, PREVIEW_FRAME)

        self.fig = plt.figure(figsize=(SCREEN["w"] / 100.0, SCREEN["h"] / 100.0), dpi=BASE_DPI)
        self.fig.set_facecolor("black"); self.fig.patch.set_facecolor("black")
        try:
            self.fig.canvas.manager.set_window_title(
                "FDIR FAULT LAB  —  LAYOUT DESIGNER (static · arrange & save, then run live_fault_lab.py)")
        except Exception:
            pass

        self._dim = None
        self._boxes, self._labels, self._handles = {}, {}, {}
        self._ghost = self._ghost_lbl = self._ghost_handle = None
        self._ctrl_axes = set()
        self._widgets = []
        self._status = None
        self.builder = None

        for _key in ("keymap.save", "keymap.fullscreen", "keymap.yscale", "keymap.xscale",
                     "keymap.grid", "keymap.grid_minor", "keymap.pan", "keymap.zoom",
                     "keymap.home", "keymap.all_axes"):
            try:
                plt.rcParams[_key] = []
            except Exception:
                pass

        self._connect()
        self.build()

    # ───────────────────────── build / content ──────────────────────────────
    def build(self):
        self.fig.clear()
        self._widgets = []
        self._ctrl_axes = set()
        self._boxes, self._labels, self._handles = {}, {}, {}
        self._dim = self._ghost = self._ghost_lbl = self._ghost_handle = None
        self.builder = FL.TileBuilder(self.fig, base._make_window, self.engine, self.truth, PANELS)
        self.builder.build(self.layout, self.titles, self.shown, self.font_scale)
        self.tiles = self.builder.tiles
        self._make_controls()

    def refresh(self):
        if self.last_state is not None and self.builder is not None:
            self.builder.refresh(self.last_state, self.titles, self.font_scale)

    # ───────────────────────── controls (bottom strip) ──────────────────────
    def _make_controls(self):
        from matplotlib.widgets import Button
        specs = [("Edit Layout", self._toggle_edit), ("Font −", self.font_down),
                 ("Font +", self.font_up), ("Rename", self.rename), ("Hide", self.hide),
                 ("Show all", self.show_all), ("Save", self.save), ("Load", self.load),
                 ("Reset", self.reset)]
        x = 0.005
        w, h, y, gap = 0.086, 0.044, 0.013, 0.004
        self._edit_btn = None
        for label, cb in specs:
            ax = self.fig.add_axes([x, y, w, h]); ax.set_zorder(50)
            b = Button(ax, label, color="#15151c", hovercolor="#2b2b3a")
            b.label.set_color("white"); b.label.set_fontsize(9.5)
            b.on_clicked(cb)
            for sp in ax.spines.values():
                sp.set_color("#3a3a4a")
            self._widgets.append(b); self._ctrl_axes.add(ax)
            if label == "Edit Layout":
                self._edit_btn = b
            x += w + gap
        sax = self.fig.add_axes([x + 0.01, y, 0.995 - (x + 0.01), h])
        sax.axis("off"); sax.set_zorder(50); self._ctrl_axes.add(sax)
        self._status = sax.text(0.0, 0.5, self._help_text(), ha="left", va="center",
                                color="#8aa0b4", fontsize=9.0, family="monospace")

    def _help_text(self):
        m = "EDIT" if self.edit else "VIEW"
        sel = self.titles.get(self.sel, "—") if self.sel else "—"
        return (f"[{m}]  font ×{self.font_scale:.2f}  ·  selected: {sel}   "
                f"·  Edit → drag to move, drag corner to resize  ·  Save writes fdi_layout.json "
                f"→ run live_fault_lab.py")

    def _set_status(self, msg):
        if self._status is not None:
            self._status.set_text(msg + "    " + self._help_text())
            self.fig.canvas.draw_idle()

    # ───────────────────────── font ─────────────────────────────────────────
    def font_up(self, *_):
        self.font_scale = round(min(self.font_scale * 1.12, 3.0), 3)
        self._apply_font()

    def font_down(self, *_):
        self.font_scale = round(max(self.font_scale / 1.12, 0.5), 3)
        self._apply_font()

    def _apply_font(self):
        if not self.edit:
            self.refresh()
        for lbl in self._labels.values():
            lbl.set_fontsize(11 * self.font_scale)
        self._set_status(f"font ×{self.font_scale:.2f}")
        self.fig.canvas.draw_idle()

    # ───────────────────────── rename / hide ────────────────────────────────
    def rename(self, *_):
        if not self.sel:
            self._set_status("click a panel first, then Rename")
            return
        new = self._ask_string(f"New title for {self.sel}:", self.titles.get(self.sel, self.sel))
        if not new or not new.strip():
            return
        new = new.strip()
        self.titles[self.sel] = new
        if self.edit and self.sel in self._labels:
            self._labels[self.sel].set_text(new)
        elif self.sel in self.tiles:
            self.tiles[self.sel]["header"].set_text(new)
        self.fig.canvas.draw_idle()
        self._set_status(f"renamed → {new}")

    def hide(self, *_):
        if not self.sel:
            self._set_status("click a panel first, then Hide")
            return
        if sum(self.shown.values()) <= 1:
            self._set_status("can't hide the last panel")
            return
        self.shown[self.sel] = False
        gone, self.sel = self.sel, None
        self._rebuild_keep_mode()
        self._set_status(f"hid {gone}")

    def show_all(self, *_):
        self.shown = {n: True for n in PANELS}
        self._rebuild_keep_mode()
        self._set_status("all panels shown")

    def _rebuild_keep_mode(self):
        was_edit = self.edit
        if was_edit:
            self._teardown_overlay()
        self.build()
        self.refresh()
        if was_edit:
            self._build_overlay()
        self.fig.canvas.draw_idle()

    def _ask_string(self, prompt, initial):
        try:
            import tkinter.simpledialog as sd
            parent = getattr(self.fig.canvas.manager, "window", None)
            return sd.askstring("FDIR Layout Designer", prompt, initialvalue=initial, parent=parent)
        except Exception:
            print(f"[rename] {prompt}  (type in console, blank = cancel)")
            try:
                s = input("> ")
                return s if s.strip() else None
            except Exception:
                return None

    # ───────────────────────── save / load / reset ──────────────────────────
    def save(self, *_):
        try:
            FL.save(LAYOUT_FILE, self.layout, self.titles, self.shown, self.font_scale)
            self._set_status(f"SAVED → {os.path.basename(LAYOUT_FILE)}  (now run live_fault_lab.py)")
        except Exception as e:
            self._set_status(f"save failed: {e}")

    def load(self, *_):
        d = FL.load(LAYOUT_FILE, PANELS)
        self.layout, self.titles, self.shown, self.font_scale = d["layout"], d["titles"], d["shown"], d["font_scale"]
        if self.edit:
            self._toggle_edit()
        self.build(); self.refresh(); self.fig.canvas.draw_idle()
        self._set_status("design reloaded" if d["found"] else "no saved design yet")

    def reset(self, *_):
        self.layout = FL.default_layout(PANELS)
        self.titles = dict(FL.DEFAULT_TITLES)
        self.shown = {n: True for n in PANELS}
        self.font_scale = 1.0
        if self.edit:
            self._toggle_edit()
        self.build(); self.refresh(); self.fig.canvas.draw_idle()
        self._set_status("reset to default 3×3")

    # ───────────────────────── edit mode ────────────────────────────────────
    def _toggle_edit(self, *_):
        if self.edit:
            self._teardown_overlay()
            self.edit = False
            if self._edit_btn:
                self._edit_btn.label.set_text("Edit Layout")
            self.build(); self.refresh(); self.fig.canvas.draw_idle()
            self._set_status("layout applied to preview")
        else:
            self.edit = True
            if self._edit_btn:
                self._edit_btn.label.set_text("Done")
            self._build_overlay(); self.fig.canvas.draw_idle()
            self._set_status("EDIT MODE — drag panels / corners, then press Done")

    def _build_overlay(self):
        from matplotlib.patches import Rectangle
        tf = self.fig.transFigure
        self._dim = Rectangle((0, FL.CTRL_TOP), 1, 1 - FL.CTRL_TOP, transform=tf,
                              facecolor="black", alpha=0.5, zorder=100)
        self.fig.add_artist(self._dim)
        for n in PANELS:
            if not self.shown.get(n, True):
                continue
            x, y, w, h = self.layout[n]
            box = Rectangle((x, y), w, h, transform=tf, fill=True, facecolor=(0.10, 0.16, 0.22, 0.55),
                            edgecolor=FL.ACCENT, lw=1.8, ls=(0, (6, 3)), zorder=101)
            self.fig.add_artist(box)
            lbl = self.fig.text(x + w / 2, y + h / 2, self.titles.get(n, n), ha="center", va="center",
                                color="white", fontsize=11 * self.font_scale, weight="bold", zorder=103)
            hd = Rectangle((x + w - 0.014, y), 0.014, 0.022, transform=tf, facecolor=FL.ACCENT,
                           edgecolor="white", lw=0.6, zorder=102)
            self.fig.add_artist(hd)
            self._boxes[n], self._labels[n], self._handles[n] = box, lbl, hd
        self._ghost = Rectangle((0, 0), 0, 0, transform=tf, fill=True, facecolor=(0.20, 0.55, 0.85, 0.40),
                                edgecolor="#ffd24a", lw=2.2, ls="--", zorder=140, animated=True, visible=False)
        self.fig.add_artist(self._ghost)
        self._ghost_handle = Rectangle((0, 0), 0.014, 0.022, transform=tf, facecolor="#ffd24a",
                                       edgecolor="white", lw=0.6, zorder=141, animated=True, visible=False)
        self.fig.add_artist(self._ghost_handle)
        self._ghost_lbl = self.fig.text(0, 0, "", ha="center", va="center", color="white",
                                        fontsize=11 * self.font_scale, weight="bold",
                                        zorder=142, animated=True, visible=False)

    def _teardown_overlay(self):
        for art in [self._dim, self._ghost, self._ghost_handle, self._ghost_lbl]:
            try:
                art.remove()
            except Exception:
                pass
        for d in (self._boxes, self._labels, self._handles):
            for art in d.values():
                try:
                    art.remove()
                except Exception:
                    pass
        self._boxes, self._labels, self._handles = {}, {}, {}
        self._dim = self._ghost = self._ghost_handle = self._ghost_lbl = None

    def _set_box(self, n, rect):
        x, y, w, h = rect
        self._boxes[n].set_bounds(x, y, w, h)
        self._labels[n].set_position((x + w / 2, y + h / 2))
        self._handles[n].set_xy((x + w - 0.014, y))

    # ───────────────────────── mouse / keys ─────────────────────────────────
    def _connect(self):
        c = self.fig.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("motion_notify_event", self.on_motion)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("key_press_event", self.on_key)
        c.mpl_connect("close_event", lambda e: setattr(self, "closed", True))

    def _frac(self, event):
        if event.x is None or event.y is None:
            return None
        return event.x / self.fig.bbox.width, event.y / self.fig.bbox.height

    def _tile_at(self, fx, fy):
        hit = None
        for n in PANELS:
            if not self.shown.get(n, True):
                continue
            x, y, w, h = self.layout[n]
            if x <= fx <= x + w and y <= fy <= y + h:
                hit = n
        return hit

    def on_press(self, event):
        if event.inaxes in self._ctrl_axes:
            return
        fr = self._frac(event)
        if fr is None:
            return
        fx, fy = fr
        n = self._tile_at(fx, fy)
        self.sel = n
        self._set_status(f"selected: {self.titles.get(n, '—') if n else '—'}")
        if not (self.edit and n):
            return
        x, y, w, h = self.layout[n]
        on_handle = (x + w - 0.02 <= fx <= x + w + 0.005) and (y - 0.005 <= fy <= y + 0.03)
        self.drag = dict(name=n, mode="resize" if on_handle else "move",
                         offx=fx - x, offy=fy - y, orig=[x, y, w, h], rect=[x, y, w, h])
        for art in (self._boxes[n], self._labels[n], self._handles[n]):
            art.set_visible(False)
        self._ghost.set_visible(True); self._ghost_handle.set_visible(True); self._ghost_lbl.set_visible(True)
        self._ghost_lbl.set_text(self.titles.get(n, n))
        self.fig.canvas.draw()
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._update_ghost(self.layout[n])

    def on_motion(self, event):
        if not (self.edit and self.drag):
            return
        fr = self._frac(event)
        if fr is None:
            return
        fx, fy = fr
        x0, y0, w0, h0 = self.drag["orig"]
        if self.drag["mode"] == "move":
            nx = _clamp(fx - self.drag["offx"], 0.0, 1.0 - w0)
            ny = _clamp(fy - self.drag["offy"], FL.CTRL_TOP, 1.0 - h0)
            rect = [nx, ny, w0, h0]
        else:
            right = _clamp(fx, x0 + FL.MINW, 1.0)
            bottom = _clamp(fy, FL.CTRL_TOP, y0 + h0 - FL.MINH)
            rect = [x0, bottom, right - x0, (y0 + h0) - bottom]
        self.drag["rect"] = rect
        self._update_ghost(rect)

    def _update_ghost(self, rect):
        x, y, w, h = rect
        self._ghost.set_bounds(x, y, w, h)
        self._ghost_handle.set_xy((x + w - 0.014, y))
        self._ghost_lbl.set_position((x + w / 2, y + h / 2))
        cv = self.fig.canvas
        cv.restore_region(self._bg)
        self.fig.draw_artist(self._ghost)
        self.fig.draw_artist(self._ghost_handle)
        self.fig.draw_artist(self._ghost_lbl)
        cv.blit(self.fig.bbox)

    def on_release(self, event):
        if not (self.edit and self.drag):
            return
        n = self.drag["name"]
        self.layout[n] = FL.snap(self.drag["rect"])
        self.drag = None
        self._ghost.set_visible(False); self._ghost_handle.set_visible(False); self._ghost_lbl.set_visible(False)
        for art in (self._boxes[n], self._labels[n], self._handles[n]):
            art.set_visible(True)
        self._set_box(n, self.layout[n])
        self.fig.canvas.draw_idle()
        self._set_status(f"moved {n} — press Done to apply, Save to keep")

    def on_key(self, event):
        k = (event.key or "").lower()
        if k == "e":
            self._toggle_edit()
        elif k in ("+", "=", "up"):
            self.font_up()
        elif k in ("-", "_", "down"):
            self.font_down()
        elif k == "r":
            self.rename()
        elif k == "h":
            self.hide()
        elif k == "a":
            self.show_all()
        elif k in ("s", "ctrl+s"):
            self.save()
        elif k == "l":
            self.load()
        elif k == "0":
            self.reset()
        elif k == "escape" and self.edit:
            self._toggle_edit()

    # ───────────────────────── show (static) ────────────────────────────────
    def run(self):
        plt = self.plt
        plt.show(block=False)
        try:
            plt.pause(0.4)
        except Exception:
            pass
        base._place(self.fig, SCREEN["x"], SCREEN["y"], SCREEN["w"], SCREEN["h"])
        base._maximize(self.fig)
        try:
            plt.pause(0.2)
        except Exception:
            pass
        self.refresh()
        self.fig.canvas.draw_idle()
        print("[designer] static preview ready · arrange + press Save, then run live_fault_lab.py", flush=True)
        while not self.closed:
            try:
                plt.pause(0.05)
            except Exception:
                break
        print("[designer] closed — if you pressed Save, run live_fault_lab.py (LAYOUT='custom').")


def main():
    if not os.path.isdir(base.DATA_TRAJ):
        raise SystemExit(f"[designer] trajectory data not found at {base.DATA_TRAJ}\n"
                         f"           run the nominal_v2 pipeline first or fix the CONFIG paths.")
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    plt.ion()
    print(f"[designer] static layout designer · preview frame {PREVIEW_FRAME} · "
          f"save → fdi_layout.json", flush=True)
    engine, truth = base._build_engine(SIM_FRAMES, SEED)
    LayoutDesigner(plt, engine, truth).run()
    print("[designer] done.")


if __name__ == "__main__":
    main()
