# Custom layouts for the FDIR Fault Lab — design once, run live

Two scripts, one shared design file. You **arrange** the nine panels in a static designer, press
**Save**, then **run the live mission inside that design**.

```
interactive_fault_lab.py   →  (static designer: move / resize / rename / font)  →  Save
                                              │
                                              ▼  fdi_layout.json
live_fault_lab.py (LAYOUT="custom")  →  runs the real simulation in your saved layout
fdi_layout.py              →  shared layout engine used by both (don't run directly)
```

## 1. Design the layout — `interactive_fault_lab.py`
Open it in Visual Studio and **Run**. It shows the nine views as a single **frozen snapshot**
(no mission playing) so you can arrange them calmly.

| Action | How (button · key) |
|---|---|
| **Move / resize a panel** | **Edit Layout** `[e]` → drag a panel to move, drag its **bottom-right corner** to resize → **Done** `[e]`. |
| **Bigger / smaller text** | **Font +** `[+]` / **Font −** `[-]`. |
| **Rename a header** | Click a panel to select it, then **Rename** `[r]`. |
| **Remove / restore a panel** | **Hide** `[h]` · **Show all** `[a]`. |
| **Save / Load / Reset** | **Save** `[s]` · **Load** `[l]` · **Reset** `[0]`. |
| **Finish** | close the window or `[q]`. |

**Press Save** when you're happy — it writes **`fdi_layout.json`** next to the scripts.
(The snapshot is frozen at `PREVIEW_FRAME` just so each panel shows representative content.)

## 2. Run the mission in your design — `live_fault_lab.py`
It now defaults to **`LAYOUT = "custom"`**, which reads `fdi_layout.json` and plays the live
mission using your saved positions, sizes, titles, hidden panels and font size. Just **Run** it.

- No saved design yet? It falls back to a clean 3×3 and tells you so.
- The other layouts still exist: set `LAYOUT` to `"single"` (fixed 3×3), `"consolidated"`
  (2 monitors) or `"separate"` (9 windows) at the top of `live_fault_lab.py`.
- `FONT_DPI` is the base text size; your saved font scale multiplies on top of it.
- A panel switched off in `SHOW` **or** hidden in the designer stays hidden.

## Notes
- **What you see is what you get** — the designer and the live runner build the tiles with the
  same code (`fdi_layout.py`), so the live mission matches your preview exactly.
- **Tiles are independent** — overlapping is allowed (the later panel draws on top). Spread them
  out for a clean grid, or pile unused ones in a corner and **Hide** them.
- **`DEPUTY / STATE`** (the per-deputy state column that used to sit on the left of **W2 · FDI
  Scope**) is **hidden** because it duplicated **W1 · Fleet Health**; the CUSUM matrix widened to
  fill the space. Restore it with `SHOW_DEPUTY_STATE = True` at the top of `faultlab_windows.py`.
- Headers are bold and sit in a reserved strip above each panel (the window's own long title is
  dropped), so nothing overlaps.
- To start clean, delete `fdi_layout.json`.
