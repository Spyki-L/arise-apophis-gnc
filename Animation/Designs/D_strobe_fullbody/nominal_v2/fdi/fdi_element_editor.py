#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fdi_element_editor.py — STATIC element-level layout editor for the FDIR fault lab.

Wireframe of every GRAPH (blank box) and TEXT label, so you can arrange the dashboard down to
the individual element, then run live_fault_lab.py (LAYOUT="elements") in that layout.

FEATURES
  • drag a box to MOVE · drag its corner to RESIZE · drag a label to move it
  • Ctrl-click to MULTI-SELECT boxes/labels; drag any selected item to move them ALL together
  • ALIGNMENT GUIDES: pink lines appear and the box SNAPS when its edges/centres line up with
    neighbours — so you can tell at a glance if a box is aligned
  • text inside a box AUTO-WRAPS / shrinks so it never crosses the box boundary; labels inside a
    box move + scale with the box when you move/resize it (a label you drag yourself keeps its spot)
  • Font −/+ changes the font of ALL selected labels (or everything if none selected)
  • Hide Text toggle → show only the boxes to arrange them cleanly, then toggle back
  • double-click a label to edit its words / size · arrow keys nudge the selection · Ctrl+Z undo
  • the controls are a separate floating window you can drag aside

Controls window + keys: [s]ave · [l]oad · [+/-] font · [e]dit text · [t]oggle text · [a] select-all
  · [Esc] clear selection · [Ctrl+Z] undo · arrow keys nudge.

INPUT: fdi_elements.json (from capture_element_layout.py). Missing → a small sample opens.
Press Run in Visual Studio. numpy + matplotlib auto-install.
"""
from __future__ import annotations
import os, sys, importlib, subprocess, copy


def _ensure(pkg):
    try:
        return importlib.import_module(pkg)
    except ImportError:
        print(f"[setup] installing {pkg}…", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(pkg)


for _p in ("numpy", "matplotlib"):
    _ensure(_p)

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fdi_elements as E

ELEMENTS_FILE = E.layout_path(HERE)
HANDLE = 0.012
MINW, MINH = 0.03, 0.02
SNAP_TOL = 0.006
ACCENT, SELC, GUIDEC = "#39d9ff", "#ffd24a", "#ff5db1"


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ── text auto-fit-to-box (wrap + shrink) — validated by agent 1 ──────────────
def fit_text_to_box(fig, text, box_w_px, box_h_px, base_fontsize,
                    family="DejaVu Sans", min_fontsize=4.0, pad_px=2.0):
    avail_w = max(1.0, box_w_px - 2.0 * pad_px)
    avail_h = max(1.0, box_h_px - 2.0 * pad_px)
    if text is None or text.strip() == "":
        return (text, base_fontsize)
    base_fontsize = float(base_fontsize); min_fontsize = float(min_fontsize)
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", fontsize=base_fontsize, family=family, ha="left", va="top")
    try:
        REF = max(base_fontsize, 1.0)

        def ext_at_ref(s):
            probe.set_text(s); e = probe.get_window_extent(renderer); return e.width, e.height
        words = text.split()
        wseen = {}; word_wpp = []
        for w in words:
            wpp = wseen.get(w)
            if wpp is None:
                ww, _ = ext_at_ref(w); wpp = ww / REF; wseen[w] = wpp
            word_wpp.append(wpp)
        sp_w, _ = ext_at_ref("x x"); x_w, _ = ext_at_ref("x")
        space_wpp = max(0.0, (sp_w - 2.0 * x_w) / REF)
        _, two_h = ext_at_ref("Xg\nXg"); _, one_h = ext_at_ref("Xg")
        line_h_pp = (two_h - one_h) / REF; first_h_pp = one_h / REF

        def wrap_est(fs):
            lines = []; cur_w = 0.0; cur = []; overflow = False; sw = space_wpp * fs
            for w, wpp in zip(words, word_wpp):
                ww = wpp * fs
                if ww > avail_w:
                    overflow = True
                if not cur:
                    cur = [w]; cur_w = ww
                else:
                    nw = cur_w + sw + ww
                    if nw <= avail_w:
                        cur.append(w); cur_w = nw
                    else:
                        lines.append(" ".join(cur)); cur = [w]; cur_w = ww
            if cur:
                lines.append(" ".join(cur))
            return lines, overflow

        def height_est(n, fs):
            return 0.0 if n <= 0 else (first_h_pp + (n - 1) * line_h_pp) * fs

        def fits_est(fs):
            lines, ov = wrap_est(fs)
            return ((not ov) and height_est(len(lines), fs) <= avail_h), lines

        def q(x):
            return round(x * 2.0) / 2.0
        hi = q(base_fontsize); lo = q(min_fontsize)
        ok_hi, lines_hi = fits_est(hi)
        if ok_hi:
            chosen, chosen_lines = hi, lines_hi
        else:
            best_fs, best_lines = None, None
            steps = int(round((hi - lo) / 0.5)); grid = [q(lo + i * 0.5) for i in range(steps + 1)]
            i0, i1 = 0, len(grid) - 1
            while i0 <= i1:
                mid = (i0 + i1) // 2; ok, lines = fits_est(grid[mid])
                if ok:
                    best_fs, best_lines = grid[mid], lines; i0 = mid + 1
                else:
                    i1 = mid - 1
            if best_fs is not None:
                chosen, chosen_lines = best_fs, best_lines
            else:
                lines, _ = wrap_est(min_fontsize); chosen, chosen_lines = min_fontsize, lines
        disp = "\n".join(chosen_lines); fs = chosen
        for _ in range(6):
            probe.set_text(disp); probe.set_fontsize(fs); e = probe.get_window_extent(renderer)
            if (e.width <= avail_w and e.height <= avail_h) or fs <= min_fontsize:
                break
            fs = max(min_fontsize, q(fs - 0.5)); lines, _ = wrap_est(fs); disp = "\n".join(lines)
        return (disp, fs)
    finally:
        probe.remove()


# ── alignment guides + snap — validated by agent 2 ───────────────────────────
def alignment_guides(moving_rect, other_rects, tol=SNAP_TOL):
    mx, my, mw, mh = moving_rect
    mxl = {"left": mx, "centerX": mx + mw / 2.0, "right": mx + mw}
    xp = [("left", "left"), ("centerX", "centerX"), ("right", "right"), ("left", "right"), ("right", "left")]
    myl = {"bottom": my, "centerY": my + mh / 2.0, "top": my + mh}
    yp = [("bottom", "bottom"), ("centerY", "centerY"), ("top", "top"), ("bottom", "top"), ("top", "bottom")]

    def nx(r):
        return {"left": r[0], "centerX": r[0] + r[2] / 2.0, "right": r[0] + r[2]}

    def ny(r):
        return {"bottom": r[1], "centerY": r[1] + r[3] / 2.0, "top": r[1] + r[3]}
    bx = None
    for o in other_rects:
        nl = nx(o)
        for mf, nf in xp:
            d = abs(mxl[mf] - nl[nf])
            if d <= tol + 1e-12 and (bx is None or d < bx[0]):
                bx = (d, mf, nl[nf], o)
    by = None
    for o in other_rects:
        nl = ny(o)
        for mf, nf in yp:
            d = abs(myl[mf] - nl[nf])
            if d <= tol + 1e-12 and (by is None or d < by[0]):
                by = (d, mf, nl[nf], o)
    nx_, ny_ = mx, my; guides = []
    if bx is not None:
        _, mf, tgt, nb = bx; nx_ = mx + (tgt - mxl[mf])
        guides.append({"orient": "v", "pos": tgt, "span": (min(ny_, nb[1]), max(ny_ + mh, nb[1] + nb[3]))})
    if by is not None:
        _, mf, tgt, nb = by; ny_ = my + (tgt - myl[mf])
        guides.append({"orient": "h", "pos": tgt, "span": (min(nx_, nb[0]), max(nx_ + mw, nb[0] + nb[2]))})
    return [nx_, ny_, mw, mh], guides


def _sample_catalog():
    cat = {"font_scale": 1.0, "graphs": [], "texts": []}
    for k, w in enumerate(E.ORDER):
        r, c = divmod(k, 3)
        X = 0.01 + c * 0.332; Y = 0.69 - r * 0.335
        cat["graphs"].append({"id": f"{w}/0", "window": w, "rect": [X + 0.012, Y + 0.05, 0.30, 0.18],
                              "label": f"{E.SHORT[w]} · graph 0"})
        cat["texts"].append({"id": f"{w}/t0", "window": w, "pos": [X + 0.02, Y + 0.24],
                             "text": E.SHORT[w], "orig": E.SHORT[w], "fontsize": 11,
                             "color": "#33ddff", "ha": "left", "va": "top", "dynamic": False, "title": True})
        cat["texts"].append({"id": f"{w}/t1", "window": w, "pos": [X + 0.03, Y + 0.20],
                             "text": "value 0.00 nominal", "orig": "value 0.00 nominal", "fontsize": 9,
                             "color": "#9fb0c0", "ha": "left", "va": "top", "dynamic": True, "title": False})
    return cat


class ElementEditor:
    def __init__(self, plt):
        self.plt = plt
        self.closed = False
        self.drag = None
        self._bg = None
        self._bg_size = None
        self.sel = set()                      # set of ("graph"|"text", id)
        self.hide_text = False
        self._undo = []
        self._redo = []
        self._scene_bg = None
        self._scene_size = None
        self._fitcache = {}
        if os.path.exists(ELEMENTS_FILE):
            self.cat = E.load(ELEMENTS_FILE); self._loaded = True
        else:
            self.cat = _sample_catalog(); self._loaded = False
        self.font_scale = self.cat.get("font_scale", 1.0)
        self.parent = {}; self.rel = {}
        self._reparent()

        self.fig = plt.figure(figsize=(19.2, 10.8), dpi=80)
        self.fig.patch.set_facecolor("black")
        try:
            self.fig.canvas.manager.set_window_title("FDIR — ELEMENT LAYOUT EDITOR (drag · multi-select · align)")
        except Exception:
            pass
        for _k in ("keymap.save", "keymap.fullscreen", "keymap.yscale", "keymap.xscale", "keymap.grid",
                   "keymap.all_axes", "keymap.home", "keymap.back", "keymap.forward"):
            try:
                plt.rcParams[_k] = []
            except Exception:
                pass
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1); self.ax.axis("off"); self.ax.set_facecolor("black")

        self.ctrl = plt.figure(figsize=(8.4, 1.0), dpi=90)
        self.ctrl.patch.set_facecolor("#15151c")
        try:
            self.ctrl.canvas.manager.set_window_title("Controls — drag me out of the way")
        except Exception:
            pass
        self._widgets = []
        self._build_controls()
        self._connect()
        self.draw()

    # ---- element helpers ---------------------------------------------------
    def _graph(self, gid):
        return next((g for g in self.cat["graphs"] if g["id"] == gid), None)

    def _text(self, tid):
        return next((t for t in self.cat["texts"] if t["id"] == tid), None)

    def _containing_graph(self, pos):
        x, y = pos; best, barea = None, 1e18
        for g in self.cat["graphs"]:
            gx, gy, gw, gh = g["rect"]
            if gx <= x <= gx + gw and gy <= y <= gy + gh and gw * gh < barea:
                barea, best = gw * gh, g["id"]
        return best

    def _reparent(self):
        self.parent = {}; self.rel = {}
        for t in self.cat["texts"]:
            self._reparent_one(t)

    def _reparent_one(self, t):
        gid = self._containing_graph(t["pos"]); self.parent[t["id"]] = gid
        if gid:
            gx, gy, gw, gh = self._graph(gid)["rect"]
            self.rel[t["id"]] = [(t["pos"][0] - gx) / gw if gw else 0.0,
                                 (t["pos"][1] - gy) / gh if gh else 0.0]

    def _apply_graph_move(self, gid, old, new):
        dx, dy = new[0] - old[0], new[1] - old[1]
        for t in self.cat["texts"]:
            if self.parent.get(t["id"]) == gid:
                t["pos"] = [t["pos"][0] + dx, t["pos"][1] + dy]

    def _apply_graph_resize(self, gid, old, new):
        ox, oy, ow, oh = old; nx, ny, nw, nh = new
        fscale = ((nw / ow if ow else 1.0) * (nh / oh if oh else 1.0)) ** 0.5
        for t in self.cat["texts"]:
            if self.parent.get(t["id"]) == gid:
                rx, ry = self.rel.get(t["id"], [0.0, 0.0])
                t["pos"] = [nx + rx * nw, ny + ry * nh]
                t["fontsize"] = max(2.0, t.get("fontsize", 9) * fscale)

    def _overlaps(self, eps=0.004):
        """ids of graph boxes that overlap another box (so you can guarantee no overlap)."""
        gs = self.cat["graphs"]; over = set()
        for i in range(len(gs)):
            xi, yi, wi, hi = gs[i]["rect"]
            for j in range(i + 1, len(gs)):
                xj, yj, wj, hj = gs[j]["rect"]
                if xi < xj + wj - eps and xj < xi + wi - eps and yi < yj + hj - eps and yj < yi + hi - eps:
                    over.add(gs[i]["id"]); over.add(gs[j]["id"])
        return over

    # ---- undo --------------------------------------------------------------
    def _push_undo(self):
        self._undo.append(copy.deepcopy(self.cat))
        if len(self._undo) > 40:
            self._undo.pop(0)
        self._redo.clear()                       # a fresh edit invalidates the redo chain

    def _restore(self, cat, msg):
        self.cat = cat
        self.font_scale = self.cat.get("font_scale", 1.0)
        self.sel = set(); self._fitcache.clear(); self._reparent(); self.draw()
        self._set_status(msg)

    def undo(self, *_):
        if self._undo:
            self._redo.append(copy.deepcopy(self.cat))
            self._restore(self._undo.pop(), "undo")

    def redo(self, *_):                          # bring back what undo removed (Ctrl+Y / Ctrl+Shift+Z)
        if self._redo:
            self._undo.append(copy.deepcopy(self.cat))
            self._restore(self._redo.pop(), "redo")

    # ---- drawing -----------------------------------------------------------
    def _is_title(self, t):
        """Panel header (W1/W2/…) — kept visible when text is hidden."""
        import re
        return bool(t.get("title")) or bool(re.match(r"\s*W\d", t.get("text", "")))

    def _fit(self, t, fs):
        """display (text, fontsize) for a label, wrapped to stay inside its parent box."""
        gid = self.parent.get(t["id"])
        base = t.get("fontsize", 9) * fs
        if not gid:
            return t["text"], base
        g = self._graph(gid)
        if not g:
            return t["text"], base
        gx, gy, gw, gh = g["rect"]
        avail_w = (gx + gw - t["pos"][0]) * self.fig.bbox.width
        avail_h = (t["pos"][1] - gy) * self.fig.bbox.height
        if avail_w < 6 or avail_h < 6:
            return t["text"], base
        key = (t["text"], round(avail_w), round(avail_h), round(base, 1))
        if key not in self._fitcache:
            self._fitcache[key] = fit_text_to_box(self.fig, t["text"], avail_w, avail_h, base)
        return self._fitcache[key]

    def draw(self):
        import matplotlib.pyplot as plt
        ax = self.ax
        ax.clear(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_facecolor("black")
        fs = self.font_scale
        over = self._overlaps()                       # boxes that overlap → drawn red
        for g in self.cat["graphs"]:
            x, y, w, h = g["rect"]; sel = ("graph", g["id"]) in self.sel; ov = g["id"] in over
            ec = SELC if sel else ("#ff5555" if ov else ACCENT)
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=ec, lw=2.2 if (sel or ov) else 1.2))
            ax.add_patch(plt.Rectangle((x + w - HANDLE, y), HANDLE, HANDLE, color=SELC if sel else ACCENT,
                                       ec="white", lw=0.5))
            ax.text(x + 0.004, y + h - 0.006, g["label"], color="#5fb0d8", fontsize=7 * fs,
                    ha="left", va="top", family="monospace")
        self._txtart = {}
        for t in self.cat["texts"]:
            if self.hide_text and not self._is_title(t):   # Hide Text keeps the W1/W2… panel titles
                continue
            x, y = t["pos"]; sel = ("text", t["id"]) in self.sel
            disp, dfs = self._fit(t, fs)
            self._txtart[t["id"]] = ax.text(
                x, y, disp, color=SELC if sel else t.get("color", "#cccccc"), fontsize=dfs,
                ha=t.get("ha", "left"), va=t.get("va", "top"),
                bbox=dict(boxstyle="square,pad=0.04", fc="none", ec=SELC if sel else "none", lw=0.8))
        try:                                     # render once + cache the scene for fast drag-start blits
            self.fig.canvas.draw()
            self._scene_bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            self._scene_size = tuple(self.fig.bbox.size)
        except Exception:
            self._scene_bg = None
        self._set_status()

    # ---- floating controls -------------------------------------------------
    def _build_controls(self):
        from matplotlib.widgets import Button
        specs = [("Font −", self.font_down), ("Font +", self.font_up), ("Edit Text", self.edit_selected_text),
                 ("Hide Text", self.toggle_text), ("Undo", self.undo), ("Redo", self.redo),
                 ("Save", self.save), ("Load", self.load)]
        x = 0.006
        for label, cb in specs:
            axb = self.ctrl.add_axes([x, 0.45, 0.112, 0.42])
            b = Button(axb, label, color="#1b1b24", hovercolor="#2b2b3a")
            b.label.set_color("white"); b.label.set_fontsize(9.0); b.on_clicked(cb)
            self._widgets.append(b); x += 0.1235
        self._sax = self.ctrl.add_axes([0.008, 0.05, 0.984, 0.30]); self._sax.axis("off")
        self._status = self._sax.text(0, 0.5, "", ha="left", va="center", color="#8aa0b4",
                                      fontsize=8.5, family="monospace")

    def _set_status(self, msg=""):
        src = "loaded" if self._loaded else "SAMPLE (run capture_element_layout.py)"
        ng = sum(1 for k in self.sel if k[0] == "graph"); nt = sum(1 for k in self.sel if k[0] == "text")
        hid = " · TEXT HIDDEN" if self.hide_text else ""
        nov = len(self._overlaps()); ovs = f" · ⚠ {nov} boxes OVERLAP (red)" if nov else " · no overlaps ✓"
        self._status.set_text(f"{msg}  [{src}] · font ×{self.font_scale:.2f} · selected {ng} box / {nt} text{hid}{ovs}\n"
                              f"Ctrl-click=multi · drag=move all selected · corner=resize · pink=aligned · red=overlap · "
                              f"keys s l + - e t a Esc · Ctrl+Z/Y")
        try:
            self.ctrl.canvas.draw_idle()
        except Exception:
            pass

    # ---- font / text / hide / save / load ----------------------------------
    def _sel_texts(self):
        return [self._text(k[1]) for k in self.sel if k[0] == "text"]

    def font_up(self, *_):
        self._font_step(1.1)

    def font_down(self, *_):
        self._font_step(1 / 1.1)

    def _font_step(self, f):
        self._push_undo()
        sel_t = self._sel_texts()
        if sel_t:                                  # scale ONLY the selected labels
            for t in sel_t:
                t["fontsize"] = max(2.0, t.get("fontsize", 9) * f)
        else:                                      # nothing selected → global
            self.font_scale = round(_clamp(self.font_scale * f, 0.4, 3.0), 3); self.cat["font_scale"] = self.font_scale
        self._fitcache.clear(); self.draw()

    def toggle_text(self, *_):
        self.hide_text = not self.hide_text; self.draw()

    def edit_selected_text(self, *_):
        sel_t = [k[1] for k in self.sel if k[0] == "text"]
        if not sel_t:
            self._set_status("select a text first, then Edit Text"); return
        if len(sel_t) == 1:
            self._edit_text(sel_t[0])
        else:                                      # multi: just set a common font size
            sz = self._ask("Font size for selected labels (pt):", "")
            try:
                if sz and sz.strip():
                    self._push_undo()
                    for tid in sel_t:
                        self._text(tid)["fontsize"] = float(sz)
                    self._fitcache.clear(); self.draw()
            except Exception:
                pass

    def _edit_text(self, tid):
        t = self._text(tid)
        if not t:
            return
        self._push_undo()
        new = self._ask("Text:", t["text"])
        if new is not None and new.strip():
            t["text"] = new.strip()
        sz = self._ask("Font size (pt):", str(round(t.get("fontsize", 9), 1)))
        try:
            if sz and sz.strip():
                t["fontsize"] = float(sz)
        except Exception:
            pass
        self._fitcache.clear(); self.draw()

    def _ask(self, prompt, initial):
        try:
            import tkinter.simpledialog as sd
            return sd.askstring("Element editor", prompt, initialvalue=initial,
                                parent=getattr(self.fig.canvas.manager, "window", None))
        except Exception:
            return None

    def save(self, *_):
        self.cat["font_scale"] = self.font_scale
        try:
            E.save(ELEMENTS_FILE, self.cat); self._set_status(f"SAVED → {os.path.basename(ELEMENTS_FILE)}")
        except Exception as e:
            self._set_status(f"save failed: {e}")

    def load(self, *_):
        if os.path.exists(ELEMENTS_FILE):
            self.cat = E.load(ELEMENTS_FILE); self._loaded = True
            self.font_scale = self.cat.get("font_scale", 1.0); self.sel = set()
            self._fitcache.clear(); self._reparent(); self.draw()
        else:
            self._set_status("no fdi_elements.json yet")

    # ---- mouse / keys ------------------------------------------------------
    def _connect(self):
        c = self.fig.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("motion_notify_event", self.on_motion)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("key_press_event", self.on_key)
        c.mpl_connect("close_event", lambda e: setattr(self, "closed", True))
        c.mpl_connect("resize_event", lambda e: (setattr(self, "_bg", None), self._fitcache.clear(), self.draw()))
        try:
            self.ctrl.canvas.mpl_connect("key_press_event", self.on_key)
        except Exception:
            pass

    def _ctrl_held(self, ev):
        m = getattr(ev, "modifiers", None)
        if m is not None:
            return ("ctrl" in m) or ("control" in m)
        ge = getattr(ev, "guiEvent", None)
        if ge is not None and hasattr(ge, "state"):
            return bool(ge.state & 0x0004)
        return False

    def _frac(self, ev):
        if ev.x is None:
            return None
        return ev.x / self.fig.bbox.width, ev.y / self.fig.bbox.height

    def _text_at(self, ev):
        best, barea = None, 1e18
        for t in self.cat["texts"]:
            art = self._txtart.get(t["id"])
            if art is None:
                continue
            try:
                bb = art.get_window_extent()
            except Exception:
                continue
            if bb.x0 <= ev.x <= bb.x1 and bb.y0 <= ev.y <= bb.y1 and bb.width * bb.height < barea:
                barea, best = bb.width * bb.height, t["id"]
        return best

    def _graph_at(self, fx, fy):                   # smallest-area containing graph (consistent w/ reparent)
        best, barea = None, 1e18
        for g in self.cat["graphs"]:
            x, y, w, h = g["rect"]
            if x <= fx <= x + w and y <= fy <= y + h and w * h < barea:
                barea, best = w * h, g["id"]
        return best

    def _hit(self, fx, fy, ev):
        for g in self.cat["graphs"]:               # resize handle first
            x, y, w, h = g["rect"]
            if (x + w - HANDLE - 0.004 <= fx <= x + w + 0.006) and (y - 0.004 <= fy <= y + HANDLE + 0.006):
                return ("graph", g["id"], "resize")
        tid = self._text_at(ev)        # _text_at only sees drawn texts → in hide mode, just the titles
        if tid is not None:
            return ("text", tid, "move")
        gid = self._graph_at(fx, fy)
        if gid is not None:
            return ("graph", gid, "move")
        return None

    def on_press(self, ev):
        fr = self._frac(ev)
        if fr is None:
            return
        fx, fy = fr
        hit = self._hit(fx, fy, ev)
        ctrl = self._ctrl_held(ev)
        if hit is None:
            if not ctrl:
                self.sel = set(); self.draw()
            return
        kind, hid, mode = hit
        key = (kind, hid)
        if getattr(ev, "dblclick", False) and kind == "text":
            self.sel = {key}; self.draw(); self._edit_text(hid); return
        if ctrl:
            self.sel.symmetric_difference_update({key}); self.draw(); return
        if key not in self.sel:
            self.sel = {key}; self.draw()        # refresh cached scene with the new selection
        self._primary = key
        if mode == "resize" and kind == "graph":
            self._begin_resize(hid)
        else:
            self._begin_move(fx, fy)

    # ---- drag: resize one graph -------------------------------------------
    def _begin_resize(self, gid):
        self._push_undo()
        self.drag = {"mode": "resize", "id": gid, "orig": list(self._graph(gid)["rect"]), "cur": None}
        self._make_ghosts()

    # ---- drag: move all selected ------------------------------------------
    def _begin_move(self, fx, fy):
        self._push_undo()
        mg = [k[1] for k in self.sel if k[0] == "graph"]
        mt = [k[1] for k in self.sel if k[0] == "text" and self.parent.get(k[1]) not in mg]
        self._orig_g = {g: list(self._graph(g)["rect"]) for g in mg}
        self._orig_t = {t: list(self._text(t)["pos"]) for t in mt}
        self.drag = {"mode": "gmove", "fx0": fx, "fy0": fy, "dxdy": (0.0, 0.0)}
        self._make_ghosts()

    def _make_ghosts(self):
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        for _t in getattr(self, "_ghosts", []):       # drop any prior ghosts (re-entrant resize path)
            try:
                _t[2].remove()
            except Exception:
                pass
        for _ln in getattr(self, "_guides", []):
            try:
                _ln.remove()
            except Exception:
                pass
        self._ghosts = []
        d = self.drag
        if d["mode"] == "resize":
            r = d["orig"]
            gh = Rectangle((r[0], r[1]), r[2], r[3], fill=False, edgecolor=SELC, lw=2.0, ls="--", animated=True)
            self.ax.add_patch(gh); self._ghosts.append(("g", d["id"], gh))
        else:
            for gid, r in self._orig_g.items():
                gh = Rectangle((r[0], r[1]), r[2], r[3], fill=False, edgecolor=SELC, lw=2.0, ls="--", animated=True)
                self.ax.add_patch(gh); self._ghosts.append(("g", gid, gh))
            for tid, p in self._orig_t.items():
                mk = Rectangle((p[0], p[1] - 0.016), 0.05, 0.018, fill=False, edgecolor=SELC, lw=1.2, ls="--", animated=True)
                self.ax.add_patch(mk); self._ghosts.append(("t", tid, mk))
        self._guides = []
        for _ in range(2):
            ln = Line2D([], [], color=GUIDEC, lw=1.0, animated=True, visible=False)
            self.ax.add_line(ln); self._guides.append(ln)
        if self._scene_bg is None or tuple(self.fig.bbox.size) != self._scene_size:
            self.fig.canvas.draw()
            self._scene_bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            self._scene_size = tuple(self.fig.bbox.size)
        self._bg = self._scene_bg
        self._bg_size = self._scene_size
        self._paint_ghosts()

    def on_motion(self, ev):
        if not self.drag:
            return
        fr = self._frac(ev)
        if fr is None:
            return
        fx, fy = fr; d = self.drag
        if tuple(self.fig.bbox.size) != self._bg_size:      # window resized mid-drag → re-grab bg
            self._make_ghosts(); return
        guides = []
        if d["mode"] == "resize":
            ox, oy, w, h = d["orig"]
            nw = _clamp(fx - ox, MINW, 1 - ox); nh = _clamp((oy + h) - fy, MINH, oy + h)
            d["cur"] = [ox, (oy + h) - nh, nw, nh]
            self._ghosts[0][2].set_bounds(*d["cur"])
        else:
            dx = fx - d["fx0"]; dy = fy - d["fy0"]
            pk = self._primary
            if pk[0] == "graph" and pk[1] in self._orig_g:
                pr = self._orig_g[pk[1]]; moved = [pr[0] + dx, pr[1] + dy, pr[2], pr[3]]
                others = [g["rect"] for g in self.cat["graphs"] if g["id"] not in self._orig_g]
                snapped, guides = alignment_guides(moved, others)
                dx += snapped[0] - moved[0]; dy += snapped[1] - moved[1]
            dx = _clamp(dx, -1, 1); dy = _clamp(dy, -1, 1)
            d["dxdy"] = (dx, dy)
            for kind, eid, art in self._ghosts:
                if kind == "g":
                    r = self._orig_g[eid]; art.set_bounds(_clamp(r[0] + dx, 0, 1 - r[2]), _clamp(r[1] + dy, 0, 1 - r[3]), r[2], r[3])
                else:
                    p = self._orig_t[eid]; art.set_xy((p[0] + dx, p[1] + dy - 0.016))
        for i, ln in enumerate(self._guides):
            if i < len(guides):
                gd = guides[i]
                if gd["orient"] == "v":
                    ln.set_data([gd["pos"], gd["pos"]], [gd["span"][0], gd["span"][1]])
                else:
                    ln.set_data([gd["span"][0], gd["span"][1]], [gd["pos"], gd["pos"]])
                ln.set_visible(True)
            else:
                ln.set_visible(False)
        self._paint_ghosts()

    def _paint_ghosts(self):
        cv = self.fig.canvas
        if self._bg is None:
            cv.draw_idle(); return
        cv.restore_region(self._bg)
        for _, _, art in self._ghosts:
            self.ax.draw_artist(art)
        for ln in self._guides:
            if ln.get_visible():
                self.ax.draw_artist(ln)
        cv.blit(self.fig.bbox)

    def on_release(self, ev):
        if not self.drag:
            return
        d = self.drag
        if d["mode"] == "resize" and d.get("cur"):
            g = self._graph(d["id"]); old = list(g["rect"]); new = [round(v, 4) for v in d["cur"]]
            g["rect"] = new; self._apply_graph_resize(d["id"], old, new)
        elif d["mode"] == "gmove":
            dx, dy = d["dxdy"]
            for gid, r in self._orig_g.items():
                g = self._graph(gid); old = list(g["rect"])
                new = [round(_clamp(r[0] + dx, 0, 1 - r[2]), 4), round(_clamp(r[1] + dy, 0, 1 - r[3]), 4), r[2], r[3]]
                g["rect"] = new; self._apply_graph_move(gid, old, new)
            for tid, p in self._orig_t.items():
                t = self._text(tid); t["pos"] = [round(_clamp(p[0] + dx, 0, 1), 4), round(_clamp(p[1] + dy, 0, 1), 4)]
                self._reparent_one(t)
        self.drag = None; self._fitcache.clear(); self.draw()

    def on_key(self, ev):
        k = (ev.key or "").lower()
        step = 0.01 if "shift" in k else 0.003
        if k in ("+", "="):
            self.font_up()
        elif k in ("-", "_"):
            self.font_down()
        elif k == "s":
            self.save()
        elif k == "l":
            self.load()
        elif k == "e":
            self.edit_selected_text()
        elif k == "t":
            self.toggle_text()
        elif k in ("ctrl+a", "a"):
            self.sel = set(("graph", g["id"]) for g in self.cat["graphs"]) | \
                       set(("text", t["id"]) for t in self.cat["texts"]); self.draw()
        elif k in ("escape", "esc"):
            self.sel = set(); self.draw()
        elif k in ("ctrl+z", "z"):
            self.undo()
        elif k in ("ctrl+y", "ctrl+shift+z", "y"):
            self.redo()
        elif k.endswith(("left", "right", "up", "down")) and self.sel:
            self._nudge(("left" in k) * -step + ("right" in k) * step,
                        ("down" in k) * -step + ("up" in k) * step)

    def _nudge(self, dx, dy):
        self._push_undo()
        mg = [k[1] for k in self.sel if k[0] == "graph"]
        for gid in mg:
            g = self._graph(gid); old = list(g["rect"])
            g["rect"] = [old[0] + dx, old[1] + dy, old[2], old[3]]; self._apply_graph_move(gid, old, g["rect"])
        for k in self.sel:
            if k[0] == "text" and self.parent.get(k[1]) not in mg:
                t = self._text(k[1]); t["pos"] = [t["pos"][0] + dx, t["pos"][1] + dy]
        self._fitcache.clear(); self.draw()

    def run(self):
        plt = self.plt
        try:
            self.fig.canvas.draw(); self.ctrl.canvas.draw()
        except Exception:
            pass
        try:
            self.fig.canvas.manager.window.state("zoomed")
        except Exception:
            pass
        try:
            self.ctrl.canvas.manager.window.wm_geometry("+40+40")
        except Exception:
            pass
        self.draw()
        print("[editor] ready · controls float in their own window · Save → fdi_elements.json", flush=True)
        plt.show()                              # event-driven (blocking) — no busy loop, no flicker
        print("[editor] closed.")


def main():
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt          # NOTE: no plt.ion() — that caused the flicker/lag
    ElementEditor(plt).run()


if __name__ == "__main__":
    main()
