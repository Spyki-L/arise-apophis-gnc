#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fdi_elements.py — element-level layout for the FDIR fault lab (graphs + text, not whole windows).

Schema (fdi_elements.json):
{
  "font_scale": 1.0,
  "graphs": [ {"id":"ops/0", "window":"ops", "rect":[x,y,w,h], "label":"W1 · ops graph 0"} , ... ],
  "texts":  [ {"id":"ops/t3", "window":"ops", "pos":[x,y], "text":"FLEET HEALTH (FDI)",
               "orig":"FLEET HEALTH (FDI)", "fontsize":11.0, "color":"#33bbff",
               "ha":"left", "va":"baseline", "dynamic":false} , ... ]
}
All rects/positions are ABSOLUTE figure fractions (0..1, y up).

Three jobs:
  • capture(fig, order)         — read every graph (Axes) + text from a built dashboard.
  • render_schematic(cat, path) — draw the blank-boxes + text picture (no chart content).
  • apply(...)                  — in the live run, move each graph to its saved rect and apply
                                  the text string / font edits.
This module is matplotlib-only and has NO dependency on the engine, so the editor runs anywhere.
"""
from __future__ import annotations
import json
import os

# Stable window order = how live_fault_lab builds them (W1..W9).
ORDER = ["ops", "architecture", "scope", "innovations", "subsystems", "coverage", "deputy", "isl", "loss"]
SHORT = {"ops": "W1 MISSION OPS", "architecture": "W4 AUTONOMY BRAIN", "scope": "W2 FDI SCOPE",
         "innovations": "W7 INNOVATION", "subsystems": "W3 SUBSYSTEMS", "coverage": "W8 COVERAGE",
         "deputy": "W5 DEPUTY", "isl": "W6 ISL", "loss": "W9 LOSS"}


def layout_path(here):
    return os.path.join(here, "fdi_elements.json")


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        cat = json.load(f)
    cat.setdefault("font_scale", 1.0)
    cat.setdefault("graphs", [])
    cat.setdefault("texts", [])
    return cat


def save(path, cat):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cat, f, indent=2)


# ─────────────────────────── capture ────────────────────────────────────────
def _abs_rect(artist, fig):
    """Absolute figure-fraction rect (x, y, w, h) of any artist with a window extent."""
    bb = artist.get_window_extent()
    inv = fig.transFigure.inverted()
    (x0, y0) = inv.transform((bb.x0, bb.y0))
    (x1, y1) = inv.transform((bb.x1, bb.y1))
    return [float(min(x0, x1)), float(min(y0, y1)), float(abs(x1 - x0)), float(abs(y1 - y0))]


def graph_extent(ax, fig):
    """(extent_rect, axes_rect) absolute figure-fraction. extent_rect = the TRUE visual footprint:
    the axes' tight bbox unioned with its tick labels, axis labels, title, offset and ax.text
    children — so the box encloses everything the graph draws. axes_rect = ax.get_position().
    fig must already be drawn (capture draws once). 3D/odd axes fall back to axes_rect."""
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    pos = ax.get_position(); axes_rect = [pos.x0, pos.y0, pos.width, pos.height]
    if getattr(ax, "name", "") == "3d":          # 3D axes (W1 trajectory, W8 asteroid): get_tightbbox
        r = _abs_rect(ax, fig)                    #   is unreliable → use the rendered box (proven path)
        return r, r
    bxs = []
    try:
        tb = ax.get_tightbbox(renderer)
        if tb is not None:
            bxs.append(tb)
    except Exception:
        return axes_rect, axes_rect
    arts = list(getattr(ax, "texts", [])) + [getattr(ax, "title", None),
            ax.xaxis.label, ax.yaxis.label, getattr(ax.xaxis, "offsetText", None)]
    for art in arts:
        if art is None:
            continue
        try:
            if str(art.get_text()) == "" or not art.get_visible():
                continue
            bb = art.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bb is not None and bb.width and bb.height:
            bxs.append(bb)
    if not bxs:
        return axes_rect, axes_rect
    x0 = min(b.x0 for b in bxs); y0 = min(b.y0 for b in bxs)
    x1 = max(b.x1 for b in bxs); y1 = max(b.y1 for b in bxs)
    (fx0, fy0) = inv.transform((x0, y0)); (fx1, fy1) = inv.transform((x1, y1))
    ext = [min(fx0, fx1), min(fy0, fy1), abs(fx1 - fx0), abs(fy1 - fy0)]
    # FORCE the footprint to contain the axes rect so axes_margins is always well-defined
    # (scattered data labels can otherwise pull the tight bbox off the axes → invalid margins).
    ux0 = min(ext[0], axes_rect[0]); uy0 = min(ext[1], axes_rect[1])
    ux1 = max(ext[0] + ext[2], axes_rect[0] + axes_rect[2]); uy1 = max(ext[1] + ext[3], axes_rect[1] + axes_rect[3])
    ext = [ux0, uy0, ux1 - ux0, uy1 - uy0]
    if not all(v == v and abs(v) != float("inf") for v in ext) or ext[2] <= 1e-6 or ext[3] <= 1e-6:
        return axes_rect, axes_rect              # guard degenerate / non-finite footprints
    return ext, axes_rect


def axes_margins(extent_rect, axes_rect):
    """axes_rect expressed as fractions [mx,my,mw,mh] WITHIN extent_rect."""
    ex, ey, ew, eh = extent_rect; ax_, ay_, aw_, ah_ = axes_rect
    ew = ew if ew > 1e-12 else 1e-12; eh = eh if eh > 1e-12 else 1e-12
    cl = lambda v, lo, hi: max(lo, min(hi, v))
    return [cl((ax_ - ex) / ew, 0.0, 1.0), cl((ay_ - ey) / eh, 0.0, 1.0),
            cl(aw_ / ew, 1e-6, 1.0), cl(ah_ / eh, 1e-6, 1.0)]


def axes_from_extent(extent_rect, margins):
    """The [x,y,w,h] for ax.set_position so the axes' footprint fills extent_rect."""
    ex, ey, ew, eh = extent_rect; mx, my, mw, mh = margins
    return [ex + mx * ew, ey + my * eh, mw * ew, mh * eh]


def _looks_title(s):
    """The panel headers read 'W1 · MISSION OPS', 'W2 FDI SCOPE', … — i.e. start with W<digit>."""
    import re
    return bool(re.match(r"\s*W\d", s or ""))


def _texts_of(sf, axlist):
    """(text_artist, is_panel_header) for the MEANINGFUL text of a window: the panel header
    (subfigure suptitle) plus free/section labels (ax.text) and graph titles. EXCLUDES axis tick
    labels and x/y axis labels — those live in the axis machinery (not ax.texts) and are sized
    GLOBALLY via the font control."""
    out = [(t, True) for t in getattr(sf, "texts", [])]      # subfigure-level = panel header
    for ax in axlist:
        out.extend((t, False) for t in getattr(ax, "texts", []))
        ttl = getattr(ax, "title", None)
        if ttl is not None and (ttl.get_text() or "").strip():
            out.append((ttl, False))
    return out


def capture(fig, win_subfigs, min_text_len=1, max_texts_per_window=80):
    """Build a catalog from a drawn dashboard.

    win_subfigs : dict window_name -> the SubFigure (or Figure) that window drew into.
    Axis ticks / axis labels are skipped (edit those globally via the font control).
    Returns the catalog dict. Call fig.canvas.draw() before this so extents are valid.
    """
    import matplotlib.text as mtext
    graphs, texts = [], []
    for w in ORDER:
        sf = win_subfigs.get(w)
        if sf is None:
            continue
        axlist = list(getattr(sf, "axes", []))
        for i, ax in enumerate(axlist):
            ext, arect = graph_extent(ax, fig)          # box = full footprint (axes + ticks + labels + title)
            graphs.append({"id": f"{w}/{i}", "window": w, "rect": [float(v) for v in ext],
                           "axes_margins": axes_margins(ext, arect),
                           "label": f"{SHORT.get(w, w)} · graph {i}"})
        seen = 0
        for tobj, is_header in _texts_of(sf, axlist):
            s = (tobj.get_text() or "").strip()
            if len(s) < min_text_len or seen >= max_texts_per_window:
                continue
            try:
                r = _abs_rect(tobj, fig)
            except Exception:
                continue
            seen += 1
            _fs = float(tobj.get_fontsize())
            texts.append({"id": f"{w}/t{seen}", "window": w, "pos": [r[0], r[1] + r[3]],
                          "text": s, "orig": s, "fontsize": _fs, "fontsize0": _fs,
                          "color": _to_hex(tobj.get_color()), "ha": "left", "va": "top",
                          "dynamic": _looks_dynamic(s), "title": bool(is_header or _looks_title(s))})
    return {"font_scale": 1.0, "graphs": graphs, "texts": texts}


def _looks_dynamic(s):
    return any(ch.isdigit() for ch in s)        # values/frames/etc. → "example" text


def _to_hex(c):
    try:
        import matplotlib.colors as mc
        r, g, b, _ = mc.to_rgba(c)
        return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
    except Exception:
        return "#cccccc"


# ─────────────────────────── schematic picture ──────────────────────────────
def render_schematic(cat, path, figsize=(25.6, 14.4), dpi=80, show_text=True):
    """Draw the blank-boxes + text layout (no chart content) and save a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor("black")
    fs = cat.get("font_scale", 1.0)
    for g in cat["graphs"]:
        x, y, w, h = g["rect"]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="#39d9ff", lw=1.3))
        ax.text(x + 0.004, y + h - 0.006, g["label"], color="#5fb0d8", fontsize=7 * fs,
                ha="left", va="top", family="monospace")
    if show_text:
        for t in cat["texts"]:
            x, y = t["pos"]
            ax.text(x, y, t["text"], color=t.get("color", "#cccccc"),
                    fontsize=t.get("fontsize", 9) * fs, ha=t.get("ha", "left"), va=t.get("va", "top"))
    fig.savefig(path, facecolor="black")
    plt.close(fig)
    return path


# ─────────────────────────── live apply ─────────────────────────────────────
def apply_graphs(win_axes, cat):
    """Position every axes so its footprint fills its saved box. win_axes: window -> [Axes,...]."""
    by_id = {g["id"]: g for g in cat["graphs"]}
    for w, axlist in win_axes.items():
        for i, ax in enumerate(axlist):
            g = by_id.get(f"{w}/{i}")
            if not g:
                continue
            try:
                m = g.get("axes_margins")
                if m and m[0] + m[2] <= 1.001 and m[1] + m[3] <= 1.001 and m[2] > 0.01 and m[3] > 0.01:
                    ax.set_position(axes_from_extent(g["rect"], m))   # valid margins → inset within box
                else:
                    ax.set_position(g["rect"])                         # degenerate/missing → fill the box (visible)
            except Exception:
                pass


def text_overrides(cat):
    """{window: {orig_string: {'text':..|None, 'fontsize':..|None}}} for the live run — only for
    labels you actually EDITED (renamed or resized). Dynamic labels (live values the window writes
    each frame) are never overridden, and unedited labels are left alone, so nothing is clobbered."""
    out = {}
    for t in cat["texts"]:
        if t.get("dynamic"):
            continue
        f0 = t.get("fontsize0", t.get("fontsize"))
        edited_text = t.get("text") != t.get("orig")
        edited_fs = abs(float(t.get("fontsize", f0)) - float(f0)) > 1e-6
        if not (edited_text or edited_fs):
            continue
        out.setdefault(t["window"], {})[t["orig"]] = {
            "text": t["text"] if edited_text else None,
            "fontsize": t["fontsize"] if edited_fs else None}
    return out


def apply_text(subfig, overrides_for_window, font_scale=1.0):
    """After a window's update(), rewrite matched static labels + scale fonts (best-effort)."""
    import matplotlib.text as mtext
    if not overrides_for_window:
        return
    for t in subfig.findobj(mtext.Text):
        ov = overrides_for_window.get((t.get_text() or "").strip())
        if ov:
            if ov.get("text") is not None:
                t.set_text(ov["text"])
            if ov.get("fontsize"):
                t.set_fontsize(ov["fontsize"] * font_scale)
