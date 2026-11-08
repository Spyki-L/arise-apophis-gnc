#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import use_style, C, SERIES, FULL_W

DATA = Path("/sessions/cool-hopeful-franklin/mnt/Animation/Designs/D_strobe_fullbody/nominal_v2/data/mc/draw_0000/_nav_stage")
FIGS = Path(__file__).resolve().parent.parent / "figs"
use_style(serif=True)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.9))
for d in range(8):
    x = np.load(DATA / f"apophis_swarm_FDI_rgazel_D{d}.npz")["x_true"]
    a1.plot(x[:, 1], x[:, 0], lw=0.8, color=SERIES[d], label=f"D{d}")
    a2.plot(x[:, 1], x[:, 2], lw=0.8, color=SERIES[d])
    a1.plot(x[-1, 1], x[-1, 0], "o", ms=3.2, color=SERIES[d])
    a2.plot(x[-1, 1], x[-1, 2], "o", ms=3.2, color=SERIES[d])
for a in (a1, a2):
    a.plot(0, 0, "k*", ms=9)
    a.set_xlabel("along-track $T$ [m]")
a1.set_ylabel("radial $R$ [m]"); a2.set_ylabel("cross-track $N$ [m]")
a1.legend(ncol=2, fontsize=6.2, loc="lower right", handlelength=1.2)
a1.set_title("(a) $R$–$T$ plane", fontsize=8.5); a2.set_title("(b) $N$–$T$ plane", fontsize=8.5)
fig.subplots_adjust(wspace=0.28)
fig.savefig(FIGS / "fig_fleet_traj.pdf"); fig.savefig(FIGS / "fig_fleet_traj.png", dpi=150)
print("wrote fig_fleet_traj")
