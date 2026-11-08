#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_element_layout.py — RUN THIS ONCE on your machine to read the REAL current positions of
every graph and text in the FDIR fault lab, so the element editor starts from your actual layout.

It builds the nine windows exactly as live_fault_lab.py does (single-window 3×3), steps one
representative mission frame so the labels exist, then records:
  • every GRAPH (Axes) — as an absolute figure-fraction box, and
  • every TEXT label — its words, position, font size and colour,
into **fdi_elements.json**, and also saves **fdi_elements_preview.png** (the blank-boxes + text
picture) so you can see the captured layout immediately.

No window pops up (it renders off-screen). Then run `fdi_element_editor.py` to rearrange, and
`live_fault_lab.py` with LAYOUT="elements" to fly it.

Just press Run in Visual Studio.
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import live_fault_lab as base
import fdi_elements as E

PREVIEW_FRAME = 600          # which mission frame to freeze for the captured snapshot


def main():
    if not os.path.isdir(base.DATA_TRAJ):
        raise SystemExit(f"[capture] trajectory data not found at {base.DATA_TRAJ}\n"
                         f"          run the nominal_v2 pipeline first or fix the CONFIG paths.")
    import matplotlib
    matplotlib.use("Agg")           # off-screen — no window
    import matplotlib.pyplot as plt

    print("[capture] building the nine windows (single-window 3×3) …", flush=True)
    engine, truth = base._build_engine(base.SIM_FRAMES, base.SEED)

    fig = plt.figure(figsize=(base.SCREEN_MAIN["w"] / 100.0, base.SCREEN_MAIN["h"] / 100.0),
                     dpi=base.FONT_DPI)
    fig.set_facecolor("black"); fig.patch.set_facecolor("black")
    sf = fig.subfigures(3, 3, wspace=0.015, hspace=0.03)
    pos = {n: sf[i // 3, i % 3] for i, n in enumerate(base.GRID_ORDER)}
    for s in pos.values():
        try:
            s.patch.set_facecolor("black"); s.set_facecolor("black")
        except Exception:
            pass
    wins = []
    for n in base.GRID_ORDER:
        if base.SHOW.get(n):
            wins.append((n, base._embed(n, engine, truth, pos[n])))

    target = min(PREVIEW_FRAME, truth["T"].size - 1)
    print(f"[capture] stepping to frame {target} for representative labels …", flush=True)
    st = None
    for fi in range(target + 1):
        st = engine.step(fi)
    for n, w in wins:
        try:
            w.update(st)
        except Exception as e:
            print(f"[capture] {n} update warning: {e}", flush=True)
    fig.canvas.draw()

    cat = E.capture(fig, {n: pos[n] for n, _ in wins})
    out_json = E.layout_path(HERE)
    if os.path.exists(out_json):                 # KEEP your existing arrangement: the fresh capture has
        try:                                     #   correct margins; copy your saved boxes + text onto it
            old = E.load(out_json)
            old_g = {g["id"]: g for g in old.get("graphs", [])}
            for g in cat["graphs"]:
                og = old_g.get(g["id"])
                if og and "rect" in og:
                    g["rect"] = og["rect"]        # your box; axes_margins stay fresh (the fixed values)
            old_t = {t["id"]: t for t in old.get("texts", [])}
            for t in cat["texts"]:
                ot = old_t.get(t["id"])
                if ot:
                    for k in ("pos", "text", "fontsize"):
                        if k in ot:
                            t[k] = ot[k]
            cat["font_scale"] = old.get("font_scale", cat.get("font_scale", 1.0))
            print("[capture] merged your saved layout — kept positions + text edits, refreshed margins.")
        except Exception as e:
            print(f"[capture] could not merge existing layout ({e}); wrote a fresh capture.")
    E.save(out_json, cat)
    out_png = os.path.join(HERE, "fdi_elements_preview.png")
    E.render_schematic(cat, out_png, figsize=(base.SCREEN_MAIN["w"] / 100.0, base.SCREEN_MAIN["h"] / 100.0),
                       dpi=base.FONT_DPI)
    print(f"[capture] DONE — {len(cat['graphs'])} graphs + {len(cat['texts'])} text labels captured.")
    print(f"          layout : {out_json}")
    print(f"          preview: {out_png}")
    print("          next   : run fdi_element_editor.py to rearrange, then live_fault_lab.py (LAYOUT='elements').")


if __name__ == "__main__":
    main()
