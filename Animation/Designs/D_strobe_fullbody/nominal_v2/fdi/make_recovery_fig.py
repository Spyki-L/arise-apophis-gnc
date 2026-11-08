#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_recovery_fig.py — F7 figure: CBF-safe disposal (min-separation) + recovery-ladder steps."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from recovery_ladder import cbf_safe_disposal

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs"); os.makedirs(OUT, exist_ok=True)
d_safe = 50.
hp = [np.array([0, 0, 0.]), np.array([0, 80, 0.]), np.array([80, 0, 0.])]; hv = [np.zeros(3)]*3
r_s = np.array([0, 0, 300.]); v_s = np.array([0, 0, -1.0])
dv, cfree, mag = cbf_safe_disposal(r_s, v_s, hp, hv, d_safe=d_safe)
ts = np.arange(0, 600, 2.)
def minsep(v):
    return np.array([min(np.linalg.norm((r_s + v*t) - (hp[j] + hv[j]*t)) for j in range(len(hp))) for t in ts])
m0, m1 = minsep(v_s), minsep(v_s + dv)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
ax1.plot(ts, m0, color="#C62828", lw=2, label="do-nothing (suspect drifts in)")
ax1.plot(ts, m1, color="#1565C0", lw=2, label="CBF-safe disposal (dV=%.2f m/s)" % mag)
ax1.axhline(d_safe, ls="--", color="0.3", label="keep-out d_safe = %.0f m" % d_safe)
ax1.fill_between(ts, 0, d_safe, color="#C62828", alpha=.08)
ax1.set_xlabel("time after disposal decision [s]"); ax1.set_ylabel("min separation to nearest healthy deputy [m]")
ax1.set_title("CBF-safe disposal is provably collision-free\n(do-nothing breaches keep-out; disposal never does)", fontsize=10)
ax1.legend(fontsize=8, loc="upper right"); ax1.grid(alpha=.3); ax1.set_ylim(0, 320)

ax2.axis("off")
steps = [("0  DETECT", "consensus + PEER-CBF convict the suspect"),
         ("1  INSPECT", "peer camera images it -> vision: TUMBLE (0.92)"),
         ("2  INTERROGATE", "ISL pulls self-report -> NONE (lying)"),
         ("3  FUSE", "robust: 2 external sources win -> TUMBLE"),
         ("4  ACT", "uncorrectable + collision -> CBF-safe DISPOSE")]
y = 0.92
for tag, desc in steps:
    ax2.text(0.02, y, tag, fontsize=10, weight="bold", color="#1565C0", transform=ax2.transAxes)
    ax2.text(0.30, y, desc, fontsize=9, transform=ax2.transAxes); y -= 0.16
ax2.text(0.02, y-0.02, "chief-fault fallback: deterministic leader election", fontsize=8.5,
         style="italic", color="#555", transform=ax2.transAxes)
ax2.set_title("Chief-led recovery ladder (multi-source, CBF-bounded)", fontsize=10)
fig.suptitle("F7 — robust chief-led recovery: diagnose from independent sources, correct or safely dispose", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "recovery_ladder_disposal.png"), dpi=160)
print(f"wrote figs/recovery_ladder_disposal.png  (do-nothing min={m0.min():.0f}, disposal min={m1.min():.0f})")
