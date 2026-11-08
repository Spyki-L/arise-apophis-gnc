# GPU Fault Lab — `fdi_gpu_lab.py` (increment 1 of the GPU rebuild)

A from-scratch, **fast** FDIR dashboard on **PySide6 + pyqtgraph** (GPU-composited) instead of
matplotlib. Every panel — each chart and each text note — is its own **dock**: drag it to move,
drag its edges to resize, drag it out to **float** it as its own window, tab it, or close it —
all with **zero latency**, because Qt composites the move/resize natively (nothing re-renders
while you drag). The live FDI mission streams into the charts at high frame-rate.

This is **increment 1**: the framework + the core FDI panels wired to the real engine. The
remaining panels get ported onto this same framework next (see "Roadmap").

## Run (Visual Studio: just press Run)
Open **`fdi_gpu_lab.py`** and press Run. It **auto-installs** PySide6 + pyqtgraph on first launch
(one-time, ~a minute; PySide6 is ~100 MB), then opens — no manual `pip install` needed.

Optional command-line flags if you launch from a terminal:
```bat
python fdi_gpu_lab.py            REM real FDI engine if available, else a built-in mock campaign
python fdi_gpu_lab.py --mock     REM force the mock campaign (no trajectory data needed)
python fdi_gpu_lab.py --selftest REM build everything off-screen and exit 0 — a quick smoke test
```
Tip: the very first time, watch the Run console for `[setup] installing …` lines while the GPU
stack downloads. If you ever want a quick check without the window, run `--selftest`: it builds
every panel and steps the data headless, then prints `… — OK`.

## Controls (toolbar)
`⏸ Pause / ▶ Run` · `⟲ Restart` · `Slower` / `Faster` · `Add Note` · `Font −` / `Font +` ·
`Save Layout` · `Load Layout`.
Drag a dock's **title bar** to move it; drag its **edges/corner** to resize; drag it **off** to
float; the **✕** closes it. **Add Note** drops in an editable text panel — click in and type.
`Save Layout` writes `fdi_gpu_layout.json` (dock arrangement + notes + font); `Load Layout`
restores it.

## What's in increment 1
Five live panels, each a movable/resizable dock, driven by the real FDI state:

| Panel | Shows |
|---|---|
| **Fleet Health** | per-deputy state cell (HEALTHY / SUSPECT / FAULTED / DEAD), colour-coded |
| **CUSUM Scope** | deputy × channel heatmap of S / threshold (the W2 detector matrix, GPU heatmap) |
| **CUSUM Traces** | live S/threshold traces (all six channels) for the worst-off deputy vs the trip line |
| **Confidence** | detection confidence per deputy |
| **Residuals** | raw residuals for the focus deputy |

Plus any number of **Notes** (editable text), and a global **font** control.

## Architecture (so the rest is easy to add)
- **`fdi_gpu_data.py`** — Qt-free data layer. `FDISource` drives the **real** engine
  (`live_fault_lab._build_engine`) when it's importable, else a built-in **mock** campaign, so the
  app always runs. Extractors (`cusum_matrix`, `fleet_health`, `confidence`, `residuals`, …) turn
  the engine state into plain arrays. This module is unit-tested headlessly.
- **`fdi_gpu_lab.py`** — the Qt app. Each panel is a small widget with `.update_state(st)`. The
  main window puts each panel in a `pyqtgraph.dockarea.Dock`; a `QTimer` steps the source and
  fans the state out to every panel. Layout persists via `DockArea.saveState()`.

**To add a panel:** write a widget with `update_state(st)` (read what you need from the state via
a `fdi_gpu_data` extractor) and add it in `_build_default_docks()`. That's it — it's automatically
movable/resizable/closable and gets the live feed.

## Roadmap (next increments)
1. **Innovation Monitor (W7)** — NAV/ATT EKF innovation traces (needs an innovations extractor).
2. **Subsystems (W3)** — RCS / propellant / attitude-error per deputy (bar/line docks).
3. **ISL Network (W6)** — link graph via a `pyqtgraph.GraphItem` (nodes = deputies, edges = links).
4. **Autonomy Brain (W4)** — the SENSE→DECIDE connectome as a `GraphItem` with pulsing edges.
5. **Coverage & Strobe (W8)** and **Mission Ops 3-D (W1)** — 3-D asteroid / trajectories via
   `pyqtgraph.opengl.GLViewWidget` (true GPU 3-D), each in its own dock.
6. **Per-element editing inside a panel** — once the panel set is in, individual labels/items can
   be made draggable on a `GraphicsScene` overlay; the dock model already gives per-chart movement.

## Honest status / caveats
- I built and **unit-tested the data layer** and the layout-save/load logic, and syntax-checked the
  app, but I could **not run the Qt GUI in my environment** (no display/GL there). So treat
  increment 1 as "ready to run on your machine" — please run `--selftest`, then the GUI, and tell
  me anything that misbehaves; GUI polish is fast to iterate from here.
- The mock campaign mirrors the real state shape, so the panels look right even without Basilisk;
  point it at the real engine just by running normally (it auto-detects).
- The old matplotlib labs (`live_fault_lab.py`, `interactive_fault_lab.py`) are untouched and still
  work; this GPU lab is a separate, parallel track.
