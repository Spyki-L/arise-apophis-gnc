# -*- coding: utf-8 -*-
"""
film_gui.py — Grafana-style card/gauge presentation layer for the ARISE mission film.

Dark-navy dashboard cards, arc-gauge KPI row with true end-scales, phase chip
list, gradient area charts with live value chips. All numbers are the real
telemetry carried in each frame spec; this module only draws.

make_mission_film.render_frame() delegates here:  film_gui.render_frame(M, fig, D, sp)
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge

# ---------------------------------------------------------------- palette
BG      = "#0c111d"
CARD_FC = (0.078, 0.100, 0.160, 0.96)
CARD_EC = (0.19, 0.25, 0.38, 0.95)
CARD_HL = (0.55, 0.65, 0.85, 0.10)         # top highlight
DIVIDER = (1, 1, 1, 0.07)
GREEN   = "#3ddc84"
TRACK   = (0.145, 0.180, 0.270, 1.0)
INK     = "#e8eefb"
SUB     = "#8b96ad"
FAINT   = "#5b6478"
CYAN    = "#4fc3f7"
VIOLET  = "#b39dfb"
MAGENTA = "#e57bd8"
AMBER   = "#ffce6a"
RED     = "#ff5d6c"
MONO    = "monospace"

def _sh(lw=3, a=0.5):
    return [pe.withStroke(linewidth=lw, foreground=(0, 0, 0, a))]

# ---------------------------------------------------------------- primitives
def card(fig, x, y, w, h, title=None, right=None, dot=GREEN, tfs=7.8, divider=False):
    ax = fig.add_axes([x, y, w, h]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.012, 0.015), 0.976, 0.97,
        boxstyle="round,pad=0.006,rounding_size=0.04", transform=ax.transAxes,
        facecolor=CARD_FC, edgecolor=CARD_EC, lw=1.0, zorder=1))
    ax.plot([0.05, 0.95], [0.982, 0.982], color=CARD_HL, lw=1.0,
            transform=ax.transAxes, solid_capstyle="round")          # top sheen
    if title:
        ax.add_patch(Circle((0.052, 0.908), 0.0135, color=dot, transform=ax.transAxes,
                            zorder=3))
        ax.add_patch(Circle((0.052, 0.908), 0.024, color=dot, alpha=0.18,
                            transform=ax.transAxes, zorder=2))
        ax.text(0.092, 0.908, title, color=INK, fontsize=tfs, weight="bold",
                va="center", transform=ax.transAxes, zorder=3)
    if right:
        ax.text(0.948, 0.908, right, color=SUB, fontsize=tfs - 1.8, ha="right",
                va="center", transform=ax.transAxes, zorder=3)
    if divider:
        ax.plot([0.045, 0.955], [0.845, 0.845], color=DIVIDER, lw=0.9,
                transform=ax.transAxes)
    return ax

def gauge_card(fig, x, y, w, h, title, value, unit, frac, lo="0", hi="", accent=GREEN,
               note=""):
    """KPI card: 240° arc gauge, quarter ticks, true end-scale labels, big value."""
    ax = card(fig, x, y, w, h, title=title, tfs=7.2)
    cx, cy, R = 0.5, 0.40, 0.335
    th0, sweep = 210.0, 240.0
    ax.add_patch(Wedge((cx, cy), R, th0 - sweep, th0, width=0.080,
                       facecolor=TRACK, edgecolor="none", transform=ax.transAxes))
    for q in (0.25, 0.50, 0.75):                       # quarter ticks on the track
        a = np.radians(th0 - sweep * q)
        r1, r2 = R - 0.095, R - 0.115
        ax.plot([cx + r1 * np.cos(a), cx + r2 * np.cos(a)],
                [cy + r1 * np.sin(a), cy + r2 * np.sin(a)],
                color=(1, 1, 1, 0.22), lw=0.8, transform=ax.transAxes)
    f = float(np.clip(frac, 0.0, 1.0))
    rm = R - 0.040                                     # arc mid-radius (rounded caps)
    a0 = np.radians(th0)
    ax.add_patch(Circle((cx + rm * np.cos(a0), cy + rm * np.sin(a0)), 0.040,
                        facecolor=accent if f > 0.004 else TRACK, edgecolor="none",
                        transform=ax.transAxes))
    if f > 0.004:
        ax.add_patch(Wedge((cx, cy), R, th0 - sweep * f, th0, width=0.080,
                           facecolor=accent, edgecolor="none", transform=ax.transAxes))
        a1 = np.radians(th0 - sweep * f)
        ax.add_patch(Circle((cx + rm * np.cos(a1), cy + rm * np.sin(a1)), 0.040,
                            facecolor=accent, edgecolor="none", transform=ax.transAxes))
        ax.add_patch(Circle((cx + rm * np.cos(a1), cy + rm * np.sin(a1)), 0.014,
                            color="white", alpha=0.95, transform=ax.transAxes))
    vfs = 14.5 if len(str(value)) <= 6 else (11.5 if len(str(value)) <= 8 else 9.5)
    ax.text(cx, cy + 0.035, value, color=accent, fontsize=vfs, weight="bold",
            ha="center", va="center", transform=ax.transAxes, path_effects=_sh(2))
    ax.text(cx, cy - 0.135, unit, color=SUB, fontsize=6.0, ha="center", va="center",
            transform=ax.transAxes)
    if note:
        ax.text(cx, cy - 0.235, note, color=FAINT, fontsize=5.0, ha="center",
                va="center", transform=ax.transAxes)
    # end-scale labels at the true arc ends
    for ang, txt, hal in ((th0, lo, "center"), (th0 - sweep, hi, "center")):
        a = np.radians(ang)
        ax.text(cx + (R + 0.065) * np.cos(a), cy + (R + 0.065) * np.sin(a), txt,
                color=FAINT, fontsize=5.0, ha=hal, va="center", transform=ax.transAxes)
    return ax

def chip_row(ax, y, h, label, sub, state):
    """state: 'done' | 'active' | 'todo' — template list rows with ✓ for done."""
    if state == "active":
        fc, ec, sq, lc = (0.10, 0.40, 0.23, 0.97), GREEN, GREEN, "white"
    elif state == "done":
        fc, ec, sq, lc = (0.070, 0.115, 0.155, 0.95), (0.16, 0.34, 0.24, 0.9), \
                         (0.18, 0.55, 0.35), (0.62, 0.78, 0.68)
    else:
        fc, ec, sq, lc = (0.070, 0.095, 0.150, 0.9), (0.20, 0.26, 0.38, 0.9), \
                         (0.24, 0.30, 0.43), SUB
    ax.add_patch(FancyBboxPatch((0.045, y), 0.91, h,
        boxstyle="round,pad=0.004,rounding_size=0.02", transform=ax.transAxes,
        facecolor=fc, edgecolor=ec, lw=1.1 if state == "active" else 0.9))
    ax.add_patch(Rectangle((0.068, y + h * 0.20), 0.042, h * 0.60,
        facecolor=sq, edgecolor="none", transform=ax.transAxes))
    ax.text(0.135, y + h * 0.65, label, color=lc, fontsize=6.9, weight="bold",
            va="center", transform=ax.transAxes)
    ax.text(0.135, y + h * 0.27, sub, color=(0.80, 0.92, 0.86, 0.9) if state == "active"
            else FAINT, fontsize=5.2, va="center", transform=ax.transAxes)
    if state == "done":
        ax.text(0.905, y + h * 0.48, "✓", color=GREEN, fontsize=8, ha="center",
                va="center", weight="bold", transform=ax.transAxes)

def area_chart(ax, x, y, w, h, series, mark_u, color, mark_text="", ticks=None):
    """Gradient-fill chart with y-scale labels and a live value chip at the marker."""
    s = np.asarray(series, float)
    if len(s) < 2:
        return
    lo, hi = float(s.min()), float(s.max()); rng = hi - lo + 1e-12
    X = x + w * np.linspace(0, 1, len(s)); Y = y + h * (s - lo) / rng
    for g in range(5):                                            # grid + y labels
        gy = y + h * g / 4
        ax.plot([x, x + w], [gy, gy], color=(1, 1, 1, 0.05), lw=0.6,
                transform=ax.transAxes)
    ax.text(x - 0.012, y, f"{lo:,.0f}", color=FAINT, fontsize=4.8, ha="right",
            va="center", transform=ax.transAxes)
    ax.text(x - 0.012, y + h, f"{hi:,.0f}", color=FAINT, fontsize=4.8, ha="right",
            va="center", transform=ax.transAxes)
    for i, a in enumerate((0.18, 0.11, 0.05)):
        ax.fill_between(X, y, Y - i * 0.004 * h, color=color, alpha=a,
                        transform=ax.transAxes, lw=0)
    ax.plot(X, Y, color=color, lw=1.6, transform=ax.transAxes,
            solid_capstyle="round")
    mi = int(np.clip(mark_u, 0, 1) * (len(s) - 1))
    ax.plot([X[mi], X[mi]], [y, y + h], color=(1, 1, 1, 0.16), lw=0.8,
            ls=(0, (3, 3)), transform=ax.transAxes)
    ax.plot([X[mi]], [Y[mi]], "o", ms=5, color="white", mec=color, mew=1.2,
            transform=ax.transAxes)
    if mark_text:                                                # live value chip
        tx = min(max(X[mi] + 0.02, x + 0.05), x + w - 0.16)
        ty = min(Y[mi] + 0.10, y + h - 0.05)
        ax.add_patch(FancyBboxPatch((tx - 0.012, ty - 0.045), 0.19, 0.10,
            boxstyle="round,pad=0.004,rounding_size=0.015", transform=ax.transAxes,
            facecolor=(0.05, 0.07, 0.12, 0.95), edgecolor=(0.35, 0.45, 0.65, 0.8),
            lw=0.8, zorder=5))
        ax.text(tx, ty, mark_text, color=INK, fontsize=5.4, family=MONO,
                va="center", transform=ax.transAxes, zorder=6)
    if ticks:
        for txf, tt in ticks:
            ax.text(x + w * txf, y - 0.055, tt, color=FAINT, fontsize=5.0,
                    ha="center", transform=ax.transAxes)

def hbar(ax, x, y, w, h, frac, color, lab=None, val=None, fs=6.0):
    for g in (0.25, 0.5, 0.75):
        ax.plot([x + w * g] * 2, [y - 0.004, y + h + 0.004], color=(1, 1, 1, 0.06),
                lw=0.6, transform=ax.transAxes)
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes,
        facecolor=(1, 1, 1, 0.055), edgecolor=(1, 1, 1, 0.12), lw=0.6))
    f = float(np.clip(frac, 0, 1))
    if f > 0.005:
        ax.add_patch(Rectangle((x, y), w * f, h, transform=ax.transAxes,
            facecolor=color, edgecolor="none", alpha=0.92))
        ax.add_patch(Rectangle((x + w * f - 0.004, y), 0.004, h,
            transform=ax.transAxes, facecolor="white", alpha=0.5))
    if lab:
        ax.text(x - 0.015, y + h / 2, lab, color=SUB, fontsize=fs, ha="right",
                va="center", transform=ax.transAxes)
    if val:
        ax.text(x + w + 0.015, y + h / 2, val, color=INK, fontsize=fs, va="center",
                transform=ax.transAxes, family=MONO)

def led(ax, x, y, ok=True, r=0.010):
    c = GREEN if ok else AMBER
    ax.add_patch(Circle((x, y), r * 2.1, color=c, alpha=0.16, transform=ax.transAxes))
    ax.add_patch(Circle((x, y), r, color=c, ec="white", lw=0.3, transform=ax.transAxes))

# ---------------------------------------------------------------- layout
# MINIMAL = only the 3D scene + the deputy camera feeds (no KPI gauges, no
# phase list, no event log, no FDIR panel). Set False for the full dashboard.
MINIMAL = True

KPI_Y, KPI_H = 0.775, 0.175
LEFT_X, LEFT_W = 0.005, 0.215
HERO_X, HERO_W = 0.226, 0.435
RIGHT_X, RIGHT_W = 0.667, 0.328
ROW_Y, ROW_H = 0.035, 0.725

# ---------------------------------------------------------------- sections
def top_strip(fig, sp):
    fig.text(0.007, 0.978, "ARISE · APOPHIS MISSION", color=INK, fontsize=11.5,
             weight="bold", va="center", path_effects=_sh(2))
    fig.text(0.168, 0.978, "FULL MISSION ARC · FLIGHT-DATA RECONSTRUCTION",
             color=FAINT, fontsize=6.0, va="center")
    fig.text(0.845, 0.978, sp.get("clock", ""), color=GREEN, fontsize=8.5, ha="right",
             va="center", family=MONO, weight="bold")
    fig.text(0.995, 0.978, sp.get("clock2", ""), color=SUB, fontsize=7.0, ha="right",
             va="center", family=MONO)
    fig.text(0.995, 0.006, "BASILISK 6-DOF · RADAR SHAPE MODEL · TD3/BC POLICY · ALL TELEMETRY FROM SAVED DATA",
             color=(0.32, 0.38, 0.52), fontsize=4.8, ha="right")

def kpi_row(fig, sp):
    k = sp["kpi"]; W = (0.99 - 0.005) / 6.0
    cov_ok = k["cov"] >= 99.0; done = k["cap"] == 8
    items = [
        (k["rng_lab"], k["rng_val"], k["rng_unit"], k["rng_frac"], CYAN,
         "0", k.get("rng_hi", ""), "live range" if "RANGE" in k["rng_lab"] else "separation"),
        ("FLEET SPREAD", k["spr_val"], k["spr_unit"], k["spr_frac"], CYAN,
         "0", k.get("spr_hi", ""), "max deputy–chief"),
        ("ΔV USED", f"{k['dv']:.1f}", "m/s · fleet", np.clip(k["dv"] / 60.0, 0, 1),
         GREEN if k["dv"] < 50 else AMBER, "0", "60", "cumulative commanded"),
        ("IMAGES", f"{k['img']}", "strobe exposures", k["img"] / max(k["img_tot"], 1),
         AMBER, "0", f"{k['img_tot']}", "CA ±2 s pass"),
        ("COVERAGE", f"{k['cov']:.1f}" if k["cov"] > 0 else "—", "% whole-body stereo",
         k["cov"] / 100.0, GREEN if cov_ok else CYAN, "0", "100",
         "objective ≥ 99 %"),
        ("DOCKED", f"{k['cap']}/8", "RAFTI captures", k["cap"] / 8.0,
         GREEN if done else VIOLET, "0", "8", "terminal RL policy"),
    ]
    for i, (t, v, u, f, c, lo, hi, note) in enumerate(items):
        gauge_card(fig, 0.005 + i * W, KPI_Y, W - 0.004, KPI_H, t, v, u, f,
                   lo=lo, hi=hi, accent=c, note=note)

def phase_card(fig, sp):
    ax = card(fig, LEFT_X, ROW_Y + 0.42, LEFT_W, ROW_H - 0.42, "MISSION PHASE",
              right="timeline")
    names = [("SEPARATION", "ESPA RAFTI deploy"), ("CRUISE", "formation phasing"),
             ("IMAGING", "CA ±2 s strobe pass"), ("RETURN", "STM transfer"),
             ("RE-DOCK", "RL terminal guidance")]
    pi = sp["phase_idx"]
    for i, (nm, sub) in enumerate(names):
        st = "active" if i == pi else ("done" if i < pi else "todo")
        chip_row(ax, 0.70 - i * 0.165, 0.135, nm, sub, st)
    return ax

def side_card(fig, sp):
    ax = card(fig, LEFT_X, ROW_Y, LEFT_W, 0.41, sp.get("side_title", "STATUS"),
              divider=True)
    rows = sp.get("side_rows")
    if rows:
        for i, (lab, val, cc) in enumerate(rows[:6]):
            y = 0.72 - i * 0.125
            ax.text(0.07, y, lab, color=SUB, fontsize=6.2, va="center",
                    transform=ax.transAxes)
            ax.text(0.93, y, val, color=cc or INK, fontsize=7.0, weight="bold",
                    ha="right", va="center", transform=ax.transAxes, family=MONO)
    lines = sp.get("log_lines")
    if lines:
        y0 = 0.74 if not rows else 0.26
        for i, (msg, cc) in enumerate(lines[-(6 if not rows else 2):]):
            ax.text(0.055, y0 - i * 0.115, msg[:52], color=cc, fontsize=4.9,
                    family=MONO, va="center", transform=ax.transAxes, clip_on=True)
    return ax

def hero_card(fig, M, D, sp, rect=None):
    HX, HY, HW, HH = rect if rect else (HERO_X, ROW_Y, HERO_W, ROW_H)
    ax = card(fig, HX, HY, HW, HH, sp["banner"],
              right="CHIEF LVLH (RTN)", dot=sp.get("banner_dot", GREEN), tfs=8.0)
    ax.plot([0.03, 0.97], [0.868, 0.868], color=DIVIDER, lw=0.9, transform=ax.transAxes)
    pad_x, pad_y = 0.012, 0.02
    sp = dict(sp)
    sp["hero_rect"] = [HX + pad_x, HY + pad_y,
                       HW - 2 * pad_x, HH - 2 * pad_y - 0.045]
    M.scene_3d(fig, D, sp)
    if sp["phase"] == "IMAGING":
        fig.text(HX + HW - 0.018, HY + HH - 0.112,
                 f"IMAGES {sp.get('nimg_done', 0):03d}/{int(D['nimg'].sum()):03d}",
                 color=AMBER, fontsize=8, ha="right", weight="bold", family=MONO,
                 path_effects=_sh(2))
        fig.text(HX + HW - 0.018, HY + HH - 0.134,
                 f"NAC FOV {M.NAC_FOV_HALF_DEG*2:.0f}° · TIME ×{max(sp.get('rate',1),0.1):.1f}",
                 color=SUB, fontsize=5.6, ha="right")
    elif sp.get("rate", 1.0) < 0.55:
        fig.text(HX + HW - 0.018, HY + HH - 0.112,
                 f"TIME ×{max(sp['rate'], 0.1):.1f}", color=AMBER, fontsize=7.5,
                 ha="right", weight="bold", family=MONO, path_effects=_sh(2))

# ---------------------------------------------------------------- right column
def right_sep(fig, M, D, sp):
    ax = card(fig, RIGHT_X, ROW_Y + 0.37, RIGHT_W, ROW_H - 0.37, "DEPLOYMENT ΔV",
              right="cumulative · m/s", divider=True)
    dv = sp["dv"]; mx = max(float(D["sep"]["dv"].max()), 1e-6)
    for k in range(8):
        y = 0.72 - k * 0.082
        hbar(ax, 0.15, y, 0.58, 0.042, dv[k] / mx, M.COLS[k], lab=f"D{k+1}",
             val=f"{dv[k]:.3f}")
    ax.text(0.73 + 0.015, 0.78, f"max {mx:.3f}", color=FAINT, fontsize=4.8,
            transform=ax.transAxes)
    ax.text(0.07, 0.045, "thrusters firing" if sp["thr_on"] else "coast",
            color=AMBER if sp["thr_on"] else SUB, fontsize=5.8, family=MONO,
            transform=ax.transAxes)
    ax2 = card(fig, RIGHT_X, ROW_Y, RIGHT_W, 0.36, "SEPARATION STATUS", divider=True)
    for i, (lab, val) in enumerate([("deputies deployed", f"{sp['ndep']} / 8"),
                                    ("stack", "ESPA ring · XB6 6U"),
                                    ("push-off ΔV", "0.28 m/s ea"),
                                    ("attitude control", "PD detumble · RW")]):
        y = 0.68 - i * 0.16
        ax2.text(0.07, y, lab, color=SUB, fontsize=6.4, transform=ax2.transAxes)
        ax2.text(0.93, y, val, color=INK, fontsize=7.2, weight="bold", ha="right",
                 transform=ax2.transAxes, family=MONO)

def right_cruise(fig, M, D, sp):
    ax = card(fig, RIGHT_X, ROW_Y + 0.37, RIGHT_W, ROW_H - 0.37, "RANGE TO APOPHIS",
              right="log₁₀ km", divider=True)
    area_chart(ax, 0.13, 0.17, 0.79, 0.58, np.log10(np.maximum(D["mission"]["apr"] / 1e3, 1.0)),
               sp["u"], CYAN, mark_text=f"{sp['ap_rng_km']:,.0f} km",
               ticks=[(0.0, "t0"), (1.0, "CA")])
    ax.text(0.5, 0.048, f"closing · CA in {sp['tca_h']:.1f} h", color=INK, fontsize=6.4,
            ha="center", family=MONO, transform=ax.transAxes)
    ax2 = card(fig, RIGHT_X, ROW_Y, RIGHT_W, 0.36, "FLEET SPREAD", right="km",
               divider=True)
    area_chart(ax2, 0.13, 0.26, 0.79, 0.48, D["spread_series"], sp["u"], MAGENTA,
               mark_text=f"{sp['spread_km']:.1f} km")
    ax2.text(0.5, 0.075, "strobe formation phasing", color=FAINT, fontsize=5.6,
             ha="center", family=MONO, transform=ax2.transAxes)

def right_imaging(fig, M, D, sp):
    ax = card(fig, RIGHT_X, ROW_Y + 0.315, RIGHT_W, ROW_H - 0.315, "DEPUTY CAMERA FEEDS",
              right="NAC · live")
    cov = D["cov"]
    fig.text(RIGHT_X + 0.016, ROW_Y + 0.315 + (ROW_H - 0.315) - 0.052,
             f"SOLAR 1361 W/m² · ALBEDO 0.23 · EARTHSHINE {cov['earthshine']:.2f}% · DARK SIDE VIA STROBE",
             color=(0.50, 0.58, 0.72), fontsize=4.6)
    fw, fh = 0.152, 0.166; gx, gy = 0.008, 0.012
    x0 = RIGHT_X + 0.010; y0 = ROW_Y + 0.325
    last = sp.get("last_ev", {})
    for j, k in enumerate((0, 2, 5, 7)):
        M.camera_feed(fig, D, x0 + (j % 2) * (fw + gx), y0 + (1 - j // 2) * (fh + gy),
                      fw, fh, k, sp["pos"], sp["apo_cad"], sp["ap_R"], sp["u"],
                      ev=last.get(k), flash=sp.get("strobe", [None] * 8)[k])
    ax2 = card(fig, RIGHT_X, ROW_Y, RIGHT_W, 0.305, "MISSION AUTONOMY · FDIR",
               right="8/8 nominal", divider=True)
    snr = D["mission"]["snr"]
    for k in range(8):
        cx = 0.085 + (k % 4) * 0.235; cy = 0.60 - (k // 4) * 0.30
        led(ax2, cx, cy, ok=True, r=0.015)
        ax2.text(cx, cy - 0.115, f"D{k+1}", color=INK, fontsize=5.6, ha="center",
                 weight="bold", transform=ax2.transAxes)
        ax2.text(cx, cy - 0.205, f"{snr[k]:.0f}", color=GREEN, fontsize=4.8,
                 ha="center", family=MONO, transform=ax2.transAxes)
    ax2.text(0.07, 0.83, "GNC · CAM · COM watchdogs", color=FAINT, fontsize=4.8,
             transform=ax2.transAxes)
    ax2.text(0.93, 0.83, "SNR dB-Hz", color=FAINT, fontsize=4.8, ha="right",
             transform=ax2.transAxes)

def right_return(fig, M, D, sp):
    ret = D["ret"]
    ax = card(fig, RIGHT_X, ROW_Y + 0.37, RIGHT_W, ROW_H - 0.37, "RETURN ΔV BUDGET",
              right="per deputy · m/s", divider=True)
    dvp = np.asarray(ret["dv_per_deputy"], float); mx = dvp.max() + 1e-9
    for k in range(8):
        y = 0.72 - k * 0.082
        hbar(ax, 0.15, y, 0.58, 0.042, dvp[k] / mx, M.COLS[k], lab=f"D{k+1}",
             val=f"{dvp[k]:.2f}")
    ax.text(0.07, 0.045, f"fleet total {float(ret['dv_total']):.2f} m/s", color=SUB,
            fontsize=5.8, family=MONO, transform=ax.transAxes)
    ax2 = card(fig, RIGHT_X, ROW_Y, RIGHT_W, 0.36, "RETURN STATUS", divider=True)
    for i, (lab, val) in enumerate([("guidance", "relative STM targeting"),
                                    ("target", "ESPA ring · own port"),
                                    ("transfer", "two-impulse · ½ period"),
                                    ("progress", f"{sp['pct']} %")]):
        y = 0.68 - i * 0.16
        ax2.text(0.07, y, lab, color=SUB, fontsize=6.4, transform=ax2.transAxes)
        ax2.text(0.93, y, val, color=INK, fontsize=7.2, weight="bold", ha="right",
                 transform=ax2.transAxes, family=MONO)

def right_dock(fig, M, D, sp):
    ax = card(fig, RIGHT_X, ROW_Y, RIGHT_W, ROW_H, "AUTONOMOUS DOCKING · RL POLICY π(o)",
              right="actor 400×300 · TD3/BC", dot=VIOLET, divider=True)
    d = sp["focus"]; a = sp["act"]; nrm = sp["nrm"]; raw = sp["raw"]; mode = sp["mode"]
    ax.text(0.06, 0.812, f"D{d+1}", color=M.COLS[d], fontsize=11, weight="bold",
            transform=ax.transAxes)
    ax.text(0.145, 0.815, "SECURED" if mode == "DOCKED" else "ENGAGED", color=GREEN,
            fontsize=6.2, weight="bold", transform=ax.transAxes)
    ax.text(0.94, 0.815, f"CAPTURED {sp['ncap']}/8",
            color=GREEN if sp["ncap"] == 8 else INK, fontsize=7.0, weight="bold",
            ha="right", transform=ax.transAxes, family=MONO)
    for k in range(8):
        led(ax, 0.475 + k * 0.042, 0.775, ok=bool(sp["caps"][k]), r=0.008)
    ax.plot([0.045, 0.955], [0.745, 0.745], color=DIVIDER, lw=0.9, transform=ax.transAxes)
    ax.text(0.06, 0.712, "OBSERVATION o(t)", color=CYAN, fontsize=6.0, weight="bold",
            transform=ax.transAxes)
    labs = [("axial", f"{raw['ax']:6.2f} m", CYAN), ("lateral", f"{raw['lat']:6.2f} m", CYAN),
            ("closing", f"{raw['cv']:6.3f} m/s", GREEN), ("lat vel", f"{raw['lv']:6.3f} m/s", GREEN),
            ("att err", f"{raw['ae']:5.1f} °", AMBER), ("rate", f"{raw['rt']:6.3f} r/s", VIOLET)]
    for j, (lb, vv, cc) in enumerate(labs):
        y = 0.668 - j * 0.040
        ax.text(0.06, y + 0.010, lb, color=(0.80, 0.86, 0.95), fontsize=5.1,
                transform=ax.transAxes)
        hbar(ax, 0.175, y, 0.135, 0.013, nrm[j], cc)
        ax.text(0.345, y + 0.005, vv, color=INK, fontsize=4.8, family=MONO,
                transform=ax.transAxes)
    ax.text(0.55, 0.712, "COMMAND a(t)", color=GREEN, fontsize=6.0, weight="bold",
            transform=ax.transAxes)
    outs = [("Δv R", a[0] * 0.05, "m/s", GREEN), ("Δv T", a[1] * 0.05, "m/s", GREEN),
            ("Δv N", a[2] * 0.05, "m/s", GREEN), ("τ x", a[3] * 1.2, "mN·m", VIOLET),
            ("τ y", a[4] * 1.2, "mN·m", VIOLET), ("τ z", a[5] * 1.2, "mN·m", VIOLET)]
    for j, (lb, vv, un, cc) in enumerate(outs):
        y = 0.668 - j * 0.040
        ax.text(0.55, y + 0.010, lb, color=(0.80, 0.86, 0.95), fontsize=5.1,
                transform=ax.transAxes)
        bx, bw = 0.625, 0.15
        ax.add_patch(Rectangle((bx, y), bw, 0.013, transform=ax.transAxes,
            facecolor=(1, 1, 1, 0.055), edgecolor=(1, 1, 1, 0.12), lw=0.5))
        half = bw / 2; f = float(np.clip(a[j], -1, 1)) * half
        ax.add_patch(Rectangle((bx + half + min(f, 0), y), abs(f), 0.013,
            transform=ax.transAxes, facecolor=cc, alpha=0.92))
        ax.plot([bx + half] * 2, [y - 0.003, y + 0.016], color=(1, 1, 1, 0.35), lw=0.6,
                transform=ax.transAxes)
        ax.text(0.94, y + 0.005, f"{vv:+7.3f} {un}", color=INK, fontsize=4.6,
                ha="right", family=MONO, transform=ax.transAxes)
    for i, mm in enumerate(("APPROACH", "GLIDESLOPE", "TERMINAL", "CAPTURE", "DOCKED")):
        on = mm == mode; cx = 0.135 + i * 0.185
        ax.add_patch(FancyBboxPatch((cx - 0.082, 0.372), 0.164, 0.040,
            boxstyle="round,pad=0.004,rounding_size=0.015", transform=ax.transAxes,
            facecolor=(0.10, 0.40, 0.23, 0.97) if on else (0.070, 0.095, 0.150, 0.9),
            edgecolor=GREEN if on else (0.20, 0.26, 0.38), lw=1.0))
        ax.text(cx, 0.392, mm, color="white" if on else SUB, fontsize=4.8,
                weight="bold", ha="center", va="center", transform=ax.transAxes)
    ax.text(0.06, 0.330, "RAFTI CAPTURE CRITERIA", color=AMBER, fontsize=5.8,
            weight="bold", transform=ax.transAxes)
    ax.text(0.94, 0.330, "flight-stage tolerances", color=FAINT, fontsize=4.4,
            ha="right", transform=ax.transAxes)
    crit = [("axial standoff", raw["ax"], 0.10, "m", 6.0),
            ("lateral offset", raw["lat"], 0.08, "m", 2.0),
            ("closing speed", raw["cv"], 0.05, "m/s", 0.15),
            ("lateral speed", raw["lv"], 0.03, "m/s", 0.06),
            ("attitude error", raw["ae"], 12.0, "°", 40.0),
            ("body rate", raw["rt"], 0.055, "rad/s", 0.12)]
    for i, (lb, v, tol, un, vis) in enumerate(crit):
        y = 0.288 - i * 0.032
        ok = v <= tol if mode != "DOCKED" else True
        led(ax, 0.075, y + 0.004, ok=ok, r=0.007)
        ax.text(0.105, y, lb, color=(0.80, 0.86, 0.95), fontsize=4.9,
                transform=ax.transAxes)
        hbar(ax, 0.30, y - 0.003, 0.36, 0.011, min(v / vis, 1.0),
             GREEN if ok else AMBER)
        ax.plot([0.30 + 0.36 * min(tol / vis, 1.0)] * 2, [y - 0.007, y + 0.013],
                color=AMBER, lw=1.0, transform=ax.transAxes)
        ax.text(0.94, y, f"{v:7.3f} ≤ {tol:g} {un}", color=GREEN if ok else AMBER,
                fontsize=4.4, ha="right", family=MONO, transform=ax.transAxes)
    ax.plot([0.045, 0.955], [0.085, 0.085], color=DIVIDER, lw=0.9, transform=ax.transAxes)
    ax.text(0.06, 0.062, "POLICY TRACE", color=VIOLET, fontsize=4.9, weight="bold",
            transform=ax.transAxes)
    ax.text(0.06, 0.032, sp["trace"][:80], color=GREEN, fontsize=4.7, family=MONO,
            transform=ax.transAxes, clip_on=True)

RIGHT = {"SEPARATION": right_sep, "CRUISE": right_cruise, "IMAGING": right_imaging,
         "RETURN": right_return, "DOCK": right_dock}

# ---------------------------------------------------------------- minimal mode
M_Y, M_H = 0.015, 0.938            # vertical band under the title strip
HERO_MIN   = (0.004, M_Y, 0.566, M_H)      # 3D scene when feeds are shown
HERO_FULL  = (0.004, M_Y, 0.992, M_H)      # 3D scene alone
FEED_X0, FEED_W, FEED_H, FEED_GAP = 0.589, 0.190, 0.235, 0.013
FEED_HDR, FEED_PAD = 0.105, 0.022          # header band / bottom pad inside the card

def feeds_minimal(fig, M, D, sp):
    """The four deputy NAC feeds, enlarged, in their own card."""
    cw = 2 * FEED_W + FEED_GAP + 0.024
    ch = FEED_HDR + 2 * FEED_H + FEED_GAP + FEED_PAD
    cy = M_Y + (M_H - ch) / 2
    ax = card(fig, FEED_X0 - 0.010, cy, cw, ch, "DEPUTY CAMERA FEEDS",
              right="NAC · live", dot=AMBER)
    cov = D["cov"]
    fig.text(FEED_X0 + 0.001, cy + ch - 0.072,
             f"SOLAR 1361 W/m² · ALBEDO 0.23 · EARTHSHINE {cov['earthshine']:.2f}%"
             f" · DARK SIDE VIA STROBE", color=(0.50, 0.58, 0.72), fontsize=5.0)
    last = sp.get("last_ev", {})
    y0 = cy + FEED_PAD
    for j, k in enumerate((0, 2, 5, 7)):
        M.camera_feed(fig, D,
                      FEED_X0 + (j % 2) * (FEED_W + FEED_GAP),
                      y0 + (1 - j // 2) * (FEED_H + FEED_GAP),
                      FEED_W, FEED_H, k, sp["pos"], sp["apo_cad"], sp["ap_R"], sp["u"],
                      ev=last.get(k), flash=sp.get("strobe", [None] * 8)[k])

def render_minimal(M, fig, D, sp):
    top_strip(fig, sp)
    if sp["phase"] == "IMAGING":
        hero_card(fig, M, D, sp, rect=HERO_MIN)
        feeds_minimal(fig, M, D, sp)
    else:
        hero_card(fig, M, D, sp, rect=HERO_FULL)

# ---------------------------------------------------------------- entry point
def render_frame(M, fig, D, sp):
    fig.clear(); fig.patch.set_facecolor(BG)
    M.bg_space(fig)
    if sp["kind"] == "title":
        M.card_title(fig, D, sp); M.fade_overlay(fig, sp.get("fade", 0.0)); return
    if sp["kind"] == "finale":
        M.card_finale(fig, D, sp); M.fade_overlay(fig, sp.get("fade", 0.0)); return
    if MINIMAL:
        render_minimal(M, fig, D, sp)
        M.fade_overlay(fig, sp.get("fade", 0.0))
        return
    top_strip(fig, sp)
    kpi_row(fig, sp)
    phase_card(fig, sp)
    side_card(fig, sp)
    hero_card(fig, M, D, sp)
    RIGHT[sp["phase"]](fig, M, D, sp)
    M.fade_overlay(fig, sp.get("fade", 0.0))
