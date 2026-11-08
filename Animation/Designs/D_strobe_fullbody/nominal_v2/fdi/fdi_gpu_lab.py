#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fdi_gpu_lab.py — GPU-accelerated, fully-rearrangeable FDIR fault lab (increment 1).

A from-scratch, **fast** dashboard built on PySide6 + pyqtgraph instead of matplotlib. Every
panel (each chart, each text note) is its own **dock** you can drag, resize, re-tile, tear off
into its own floating window, or close — with **zero latency**, because Qt composites it
natively (nothing is re-rendered while you drag). The live FDI mission streams into the charts
at high frame-rate.

This is the FIRST INCREMENT of the GPU rebuild: the framework + the core FDI panels wired to
the real engine. More panels (3-D asteroid, autonomy brain, ISL graph, coverage map) get ported
on top of this same framework next.

RUN
  pip install PySide6 pyqtgraph          # one time
  python fdi_gpu_lab.py                   # uses the real FDI engine if available, else a built-in
                                          #   mock campaign so it always runs
  python fdi_gpu_lab.py --mock            # force the mock campaign
  python fdi_gpu_lab.py --selftest        # construct everything off-screen and exit (smoke test)

USE
  • Toolbar:  ▶ Run / ⏸ Pause · Speed · Add Note · Font − / + · Save Layout · Load Layout · Panels▾
  • Drag a dock's title bar to move it; drag its edges to resize; drag it out to float it; ✕ to close.
  • Save Layout writes fdi_gpu_layout.json (dock arrangement + notes + fonts); Load restores it.
  • Notes are editable text panels — click in and type. Add as many as you like.
"""
from __future__ import annotations
import os
import sys
import json
import time
import ast
import importlib
import subprocess
from collections import deque


def _ensure(pip_name, import_name=None):
    """Import a package, pip-installing it on first run if it's missing — so you can just press
    Run in Visual Studio with no manual `pip install`."""
    mod = import_name or pip_name
    try:
        return importlib.import_module(mod)
    except ImportError:
        print(f"[setup] installing {pip_name} (one-time; PySide6 is ~100 MB, give it a minute)…",
              flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        return importlib.import_module(mod)


# auto-install the GPU stack on first run, then import it
np = _ensure("numpy")
_ensure("PySide6")
_ensure("pyqtgraph")

from PySide6 import QtCore, QtGui, QtWidgets    # noqa: E402
import pyqtgraph as pg                           # noqa: E402
from pyqtgraph.dockarea import DockArea, Dock    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fdi_gpu_data as DATA   # Qt-free data layer (real engine or mock)   # noqa: E402

LAYOUT_FILE = os.path.join(HERE, "fdi_gpu_layout.json")
CH = DATA.CHANNELS
N = DATA.N_DEP

pg.setConfigOptions(antialias=True, background="#0b0b0e", foreground="#c7d2dd",
                    imageAxisOrder="row-major")


# ──────────────────────────────────────────────────────────────────────────────
# Panels — each is a self-contained widget with .update_state(st) and .kind
# ──────────────────────────────────────────────────────────────────────────────
class CusumScopePanel(pg.PlotWidget):
    """W2-style heatmap: deputy (rows) × channel (cols), coloured by S / threshold."""
    kind = "cusum_scope"

    def __init__(self):
        super().__init__(title="FDI CUSUM SCOPE — S / threshold")
        self.img = pg.ImageItem()
        self.addItem(self.img)
        cmap = pg.ColorMap(pos=[0.0, 0.6, 1.0, 1.3],
                           color=[(18, 70, 40), (40, 160, 70), (255, 185, 50), (255, 60, 60)])
        self.img.setLookupTable(cmap.getLookupTable(0.0, 1.3, 256))
        self.img.setLevels([0.0, 1.3])
        self.getAxis("bottom").setTicks([[(i + 0.5, c) for i, c in enumerate(CH)]])
        self.getAxis("left").setTicks([[(i + 0.5, f"D{N - i}") for i in range(N)]])
        self.setXRange(0, len(CH), padding=0)
        self.setYRange(0, N, padding=0)
        self.getViewBox().setMouseEnabled(False, False)
        self.hideButtons()

    def update_state(self, st):
        M = DATA.cusum_matrix(st)            # (N, C); row 0 = D1
        self.img.setImage(M[::-1, :], levels=[0.0, 1.3], autoLevels=False)


class TracePanel(pg.PlotWidget):
    """W5-style CUSUM traces (S/threshold over time) for the worst-off deputy."""
    kind = "cusum_traces"

    def __init__(self, history=180):
        super().__init__(title="CUSUM TRACES — focus deputy")
        self.H = history
        self.hist = {c: deque(maxlen=history) for c in CH}
        self.addLegend(offset=(10, 5))          # legend must exist BEFORE the curves it lists
        self.curves = {}
        for c in CH:
            self.curves[c] = self.plot(pen=pg.mkPen(DATA.CHANNEL_COLORS[c], width=2), name=c)
        self.addLine(y=1.0, pen=pg.mkPen("#ffffff", width=1, style=QtCore.Qt.DashLine))
        self.setYRange(0.0, 1.4)
        self.setLabel("left", "S / h")
        self.setLabel("bottom", "frames ago")

    def update_state(self, st):
        d = DATA.worst_deputy(st)
        self.setTitle(f"CUSUM TRACES — D{d+1} ({st['fdi'][d]['state']})")
        ratios = DATA.cusum_ratios_for(st, d)
        for j, c in enumerate(CH):
            self.hist[c].append(float(min(ratios[j], 1.39)))
            ys = list(self.hist[c])
            xs = list(range(-len(ys) + 1, 1))
            self.curves[c].setData(xs, ys)


class BarPanel(pg.PlotWidget):
    """Generic per-deputy or per-channel bar chart (confidence / residuals)."""

    def __init__(self, mode="confidence"):
        self.mode = mode
        self.kind = f"bar_{mode}"
        title = "DETECTION CONFIDENCE — per deputy" if mode == "confidence" else "RESIDUALS — focus deputy"
        super().__init__(title=title)
        n = N if mode == "confidence" else len(CH)
        self.bars = pg.BarGraphItem(x=np.arange(n), height=np.zeros(n), width=0.7, brush="#39d9ff")
        self.addItem(self.bars)
        if mode == "confidence":
            self.getAxis("bottom").setTicks([[(i, f"D{i+1}") for i in range(N)]])
            self.setYRange(0, 1.0)
            self.setXRange(-0.6, N - 0.4, padding=0)
        else:
            self.getAxis("bottom").setTicks([[(i, c) for i, c in enumerate(CH)]])
            self.setYRange(0, 3.0)
            self.setXRange(-0.6, len(CH) - 0.4, padding=0)
        self.getViewBox().setMouseEnabled(False, False)
        self.hideButtons()

    def update_state(self, st):
        if self.mode == "confidence":
            h = DATA.confidence(st)
            brushes = [DATA.STATE_COLORS[st["fdi"][d]["state"]] for d in range(N)]
            self.bars.setOpts(height=h, brushes=brushes)
        else:
            d = DATA.worst_deputy(st)
            self.setTitle(f"RESIDUALS — D{d+1}")
            r = DATA.residuals(st, d)
            self.bars.setOpts(height=r, brushes=[DATA.CHANNEL_COLORS[c] for c in CH])


class FleetHealthPanel(QtWidgets.QWidget):
    """W1-style fleet health: a coloured cell per deputy."""
    kind = "fleet_health"

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background:#0b0b0e;")
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(6)
        self.cells = []
        self._base_pt = 11
        for i in range(N):
            lab = QtWidgets.QLabel(f"D{i+1}\n—")
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setFont(QtGui.QFont("Arial", self._base_pt, QtGui.QFont.Bold))
            lab.setStyleSheet("color:#000; background:#39e08a; border-radius:6px; padding:6px;")
            grid.addWidget(lab, i // 4, i % 4)
            self.cells.append(lab)

    def set_font_scale(self, s):
        for lab in self.cells:
            f = lab.font(); f.setPointSizeF(self._base_pt * s); lab.setFont(f)

    def update_state(self, st):
        for d in range(N):
            state = st["fdi"][d]["state"]
            col = DATA.STATE_COLORS[state]
            self.cells[d].setText(f"D{d+1}\n{state}")
            self.cells[d].setStyleSheet(
                f"color:#000; background:{col}; border-radius:6px; padding:6px;")


class NotePanel(QtWidgets.QTextEdit):
    """An editable free-text note (move it, resize it, type in it, scale its font)."""
    kind = "note"

    def __init__(self, text="New note — click and type…"):
        super().__init__()
        self.setStyleSheet("background:#101016; color:#dfe7ee; border:1px solid #2a2a38;")
        self._base_pt = 13
        self.setFont(QtGui.QFont("Arial", self._base_pt))
        self.setText(text)

    def set_font_scale(self, s):
        f = self.font(); f.setPointSizeF(self._base_pt * s); self.setFont(f)

    def update_state(self, st):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────────────────
class FDIGpuLab(QtWidgets.QMainWindow):
    def __init__(self, force_mock=False, sim_frames=1000, seed=7):
        super().__init__()
        self.setWindowTitle("ARISE / APOPHIS — FDIR FAULT LAB · GPU (drag, resize, tear-off any panel)")
        self.resize(1680, 980)
        self.source = DATA.FDISource(sim_frames, seed, force_mock=force_mock)
        self.frame = 0
        self.playing = True
        self.speed = 2.0           # mission frames advanced per tick
        self.font_scale = 1.0
        self._fps_t = time.time(); self._fps_n = 0

        self.area = DockArea()
        self.setCentralWidget(self.area)

        # registry of live panels (those with update_state)
        self.panels = []            # list of (dock, widget)
        self.notes = []             # NotePanel widgets
        self._build_default_docks()
        self._build_toolbar()
        self._build_statusbar()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)        # ~30 Hz UI; charts are GPU-composited

    # ---- docks -------------------------------------------------------------
    def _add_panel_dock(self, name, widget, size=(420, 320), position="right", relto=None):
        dock = Dock(name, size=size, closable=True)
        dock.addWidget(widget)
        self.area.addDock(dock, position, relto)
        if hasattr(widget, "update_state"):
            self.panels.append((dock, widget))
        return dock

    def _build_default_docks(self):
        d1 = self._add_panel_dock("Fleet Health", FleetHealthPanel(), size=(360, 240), position="left")
        d2 = self._add_panel_dock("CUSUM Scope", CusumScopePanel(), size=(460, 320), position="right", relto=d1)
        self._add_panel_dock("CUSUM Traces", TracePanel(), size=(560, 320), position="bottom", relto=d2)
        d4 = self._add_panel_dock("Confidence", BarPanel("confidence"), size=(360, 240), position="bottom", relto=d1)
        self._add_panel_dock("Residuals", BarPanel("residuals"), size=(360, 240), position="right", relto=d4)

    def _make_note(self, text="New note — click and type…"):
        nt = NotePanel(text)
        nt.set_font_scale(self.font_scale)
        self.notes.append(nt)
        self._add_panel_dock(f"Note {len(self.notes)}", nt, size=(300, 160), position="right")
        return nt

    # ---- toolbar / status --------------------------------------------------
    def _act(self, tb, text, slot):
        a = QtGui.QAction(text, self)
        a.triggered.connect(slot)
        tb.addAction(a)
        return a

    def _build_toolbar(self):
        tb = QtWidgets.QToolBar("controls")
        tb.setStyleSheet("QToolBar{background:#15151c;spacing:4px;padding:4px;}"
                         "QToolButton{color:#dfe7ee;padding:5px 10px;}"
                         "QToolButton:hover{background:#2b2b3a;border-radius:4px;}")
        self.addToolBar(tb)
        self.act_play = self._act(tb, "⏸ Pause", self._toggle_play)
        self._act(tb, "⟲ Restart", self._restart)
        tb.addSeparator()
        self._act(tb, "Slower", lambda: self._set_speed(self.speed / 1.5))
        self._act(tb, "Faster", lambda: self._set_speed(self.speed * 1.5))
        tb.addSeparator()
        self._act(tb, "Add Note", lambda: self._make_note())
        self._act(tb, "Font −", lambda: self._set_font(self.font_scale / 1.12))
        self._act(tb, "Font +", lambda: self._set_font(self.font_scale * 1.12))
        tb.addSeparator()
        self._act(tb, "Save Layout", self.save_layout)
        self._act(tb, "Load Layout", self.load_layout)

    def _build_statusbar(self):
        self.status = self.statusBar()
        self.status.setStyleSheet("color:#8aa0b4;background:#0b0b0e;")
        self._set_status()

    def _set_status(self, fps=None):
        src = "MOCK campaign" if self.source.is_mock else "REAL engine"
        extra = f" · {fps:.0f} fps" if fps else ""
        self.status.showMessage(
            f"{src} · frame {self.frame}/{self.source.n_frames} · speed ×{self.speed:.1f} · "
            f"font ×{self.font_scale:.2f}{extra}   —   drag any dock to move/resize/tear-off")

    # ---- controls ----------------------------------------------------------
    def _toggle_play(self):
        self.playing = not self.playing
        self.act_play.setText("⏸ Pause" if self.playing else "▶ Run")

    def _restart(self):
        self.source = DATA.FDISource(self.source.sim_frames, self.source.seed,
                                     force_mock=self.source.is_mock)
        self.frame = 0

    def _set_speed(self, s):
        self.speed = float(np.clip(s, 0.25, 16.0)); self._set_status()

    def _set_font(self, s):
        self.font_scale = float(np.clip(s, 0.5, 3.0))
        for _, w in self.panels:
            if hasattr(w, "set_font_scale"):
                w.set_font_scale(self.font_scale)
        for nt in self.notes:
            nt.set_font_scale(self.font_scale)
        self._set_status()

    # ---- live loop ---------------------------------------------------------
    def _tick(self):
        if self.playing:
            self.frame = int(min(self.frame + self.speed, self.source.n_frames - 1))
            try:
                st = self.source.step(self.frame)
            except Exception as e:
                self.status.showMessage(f"engine error: {e}")
                return
            for dock, w in self.panels:
                try:
                    w.update_state(st)
                except Exception as e:
                    print(f"[panel {dock.name()}] update error: {e}", flush=True)
            if self.frame >= self.source.n_frames - 1:
                self.playing = False
                self.act_play.setText("▶ Run")
        self._fps_n += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            self._set_status(fps=self._fps_n / (now - self._fps_t))
            self._fps_t = now; self._fps_n = 0

    # ---- layout persistence ------------------------------------------------
    def save_layout(self, *_):
        try:
            state = self.area.saveState()
            blob = {
                "dock_state": repr(state),          # ast.literal_eval-safe round-trip
                "font_scale": self.font_scale,
                "speed": self.speed,
                "notes": [nt.toPlainText() for nt in self.notes],
            }
            with open(LAYOUT_FILE, "w", encoding="utf-8") as f:
                json.dump(blob, f, indent=2)
            self.status.showMessage(f"saved → {os.path.basename(LAYOUT_FILE)}")
        except Exception as e:
            self.status.showMessage(f"save failed: {e}")

    def load_layout(self, *_):
        if not os.path.exists(LAYOUT_FILE):
            self.status.showMessage("no saved layout yet")
            return
        try:
            with open(LAYOUT_FILE, "r", encoding="utf-8") as f:
                blob = json.load(f)
            # recreate notes to match the saved count (dock_state references them by name)
            for nt in list(self.notes):
                pass
            self._set_font(blob.get("font_scale", 1.0))
            self.speed = float(blob.get("speed", self.speed))
            saved_notes = blob.get("notes", [])
            while len(self.notes) < len(saved_notes):
                self._make_note(saved_notes[len(self.notes)])
            for nt, txt in zip(self.notes, saved_notes):
                nt.setText(txt)
            state = ast.literal_eval(blob["dock_state"])
            self.area.restoreState(state)
            self.status.showMessage("layout restored")
        except Exception as e:
            self.status.showMessage(f"load failed ({e}) — keeping current layout")

    def reset_layout(self, *_):
        self.status.showMessage("reset: close/reopen via a fresh run for now")  # docks are user-managed


# ──────────────────────────────────────────────────────────────────────────────
def main():
    force_mock = "--mock" in sys.argv
    selftest = "--selftest" in sys.argv
    if selftest:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = FDIGpuLab(force_mock=force_mock)
    if selftest:
        # construct + step a few frames off-screen, then exit 0
        for fi in (0, 200, 600, 999):
            st = win.source.step(fi)
            for _, w in win.panels:
                w.update_state(st)
        print("[selftest] constructed", len(win.panels), "live panels +",
              len(win.notes), "notes; data source:",
              "mock" if win.source.is_mock else "real", "— OK")
        return 0
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
