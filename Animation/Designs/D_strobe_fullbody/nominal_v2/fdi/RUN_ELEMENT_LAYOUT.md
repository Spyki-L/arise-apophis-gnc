# Element-level layout — move/resize every graph, edit every label, then fly it

Edit the FDIR dashboard down to the individual **graph** and **text** — not just whole windows —
in a fast static editor, save it, and have `live_fault_lab.py` place everything exactly where you
put it. No live rendering while you edit; it's a clean wireframe of boxes + text.

```
capture_element_layout.py   →  fdi_elements.json (+ fdi_elements_preview.png)   [run once, your machine]
fdi_element_editor.py        →  drag/resize boxes, edit text + fonts  →  Save     [static, runs anywhere]
live_fault_lab.py (LAYOUT="elements")  →  the live mission, every graph/text in your positions
```

## 1. Capture your real layout — `capture_element_layout.py`
Press Run once. It builds the nine windows off-screen, reads the **current position of every
graph and every text label**, and writes:
- **`fdi_elements.json`** — the editable layout, and
- **`fdi_elements_preview.png`** — a blank-boxes + text picture so you can see what was captured.

(No window opens; it just writes the two files. Re-run it any time to re-capture from scratch.)

## 2. Rearrange — `fdi_element_editor.py`
Press Run. Every graph is a blank box, every label is its text (dynamic values show an example so
you can check for overlaps). The **controls are a separate floating window** you can drag aside.

| Action | How |
|---|---|
| **Move a graph** | drag the box — labels inside it move with it |
| **Resize a graph** | drag its **bottom-right corner**; inside labels scale + auto-**wrap** so they never cross the box edge |
| **Multi-select** | **Ctrl-click** boxes/labels to add/remove; **drag any selected item to move them all together** |
| **Align** | while dragging, a **pink guide line** appears and the box **snaps** when its edges/centres line up with a neighbour |
| **Move a label** | drag the text (it re-homes to whatever box it lands in) |
| **Edit a label** | **double-click** it → new words / font size |
| **Font of selected labels** | select labels, then **Font −/+** (changes just those; with nothing selected it scales everything) |
| **Hide Text** | toggle to show **only the boxes** (the W1/W2… panel titles stay visible — and you can still **move/edit them** in this mode); toggle back and the rest of the text returns, wrapped/scaled to where the boxes are |
| **Overlap check** | any two boxes that overlap are drawn **red**, and the controls show the overlap count — so you can guarantee no overlap in the final animation |
| **Nudge** | arrow keys move the selection (Shift = larger step) |
| **Undo / Redo** | **Undo** / **Redo** buttons, or **Ctrl+Z** / **Ctrl+Y** |
| **Save** | **Save** → writes `fdi_elements.json` |

Keys: `s` save · `l` load · `+/-` font · `e` edit · `t` hide-text · `a` select-all · `Esc` clear · `Ctrl+Z` undo.
Selected elements are highlighted yellow; the floating controls show the selection count.

## 3. Fly it — `live_fault_lab.py`
It already defaults to **`LAYOUT = "elements"`**. Press Run: it builds the real mission and places
**every graph at the rect you set**, applies your **text edits and font sizes**, and plays. If no
`fdi_elements.json` exists yet it falls back to the fixed 3×3 and tells you.

## What carries into the live run — and the one limit
- **Graph position & size** — fully honored. Move/resize/space-out every graph to kill overlaps. ✓
- **Text wording & font size** — your relabels and per-label sizes are applied each frame; the
  global Font ± is applied too. ✓
- **One limit (matplotlib):** a graph's *own* auto-updating labels are redrawn inside that graph
  every frame, so a text label **moves with its graph**, not independently of it. Moving the graph
  is the reliable way to reposition its labels; standalone headers you relabel/resize as above.

## Notes
- Element ids are `window/index` (e.g. `scope/0`) and `window/tN` for text — stable between
  capture and the live run because both build the windows in the same order.
- Re-capture (`capture_element_layout.py`) only if the underlying windows change; your edits live
  in `fdi_elements.json`, so back it up if you want to keep a design before re-capturing.
- This is separate from the window-level designer (`interactive_fault_lab.py` → `fdi_layout.json`,
  `LAYOUT="custom"`) and the GPU lab (`fdi_gpu_lab.py`); all three coexist.
- I validated the editor, the schema, and the position/text apply logic here; the **capture** step
  and the **live elements run** execute on your machine (they need the real engine), so run capture
  first and tell me if the captured boxes look off — easy to tune.
