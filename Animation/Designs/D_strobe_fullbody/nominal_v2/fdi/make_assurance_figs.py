#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild the CORTEX assurance-case (GSN) and fault-tree figures with larger fonts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Wedge, Rectangle
from pathlib import Path
plt.rcParams.update({"font.family": "DejaVu Sans"})
OUT = Path(__file__).resolve().parent

NAVY="#1f3b6e"; BLUE="#cfe0f3"; BLUE_E="#3a5a8c"; PUR="#e4d7f2"; PUR_E="#7a5aa8"
GREY="#e6e6e6"; GREY_E="#7a7a7a"; CREAM="#fdf0cf"; CREAM_E="#c79a2e"
GRN="#bfe6c8"; GRN_E="#2f8f52"; ORA="#f5c06a"; ORA_E="#c07f1e"
RED="#f2a19a"; RED_E="#b0392e"; SALM="#f6cbc5"
TITLE=23; BIG=15; MED=13; SMALL=12; GATE=13; LEG=13; FOOT=11

def rbox(ax,cx,cy,w,h,text,fc,ec,fs=MED,weight="normal",lw=1.7,tc="black"):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.02,rounding_size=0.10",fc=fc,ec=ec,lw=lw,zorder=2))
    ax.text(cx,cy,text,ha="center",va="center",fontsize=fs,weight=weight,color=tc,zorder=3,linespacing=1.25)
def para(ax,cx,cy,w,h,text,fc=PUR,ec=PUR_E,fs=MED):
    sk=0.35; pts=[(cx-w/2+sk,cy-h/2),(cx+w/2,cy-h/2),(cx+w/2-sk,cy+h/2),(cx-w/2,cy+h/2)]
    ax.add_patch(Polygon(pts,closed=True,fc=fc,ec=ec,lw=1.7,zorder=2))
    ax.text(cx,cy,text,ha="center",va="center",fontsize=fs,zorder=3,linespacing=1.25)
def diamond(ax,cx,cy,w,h,text,fc=CREAM,ec=CREAM_E,fs=SMALL):
    ax.add_patch(Polygon([(cx,cy+h/2),(cx+w/2,cy),(cx,cy-h/2),(cx-w/2,cy)],closed=True,fc=fc,ec=ec,lw=1.8,zorder=2))
    ax.text(cx,cy,text,ha="center",va="center",fontsize=fs,zorder=3,linespacing=1.22)
def circ(ax,cx,cy,r,text,fc=GRN,ec=GRN_E,fs=SMALL):
    ax.add_patch(Circle((cx,cy),r,fc=fc,ec=ec,lw=1.7,zorder=2))
    ax.text(cx,cy,text,ha="center",va="center",fontsize=fs,zorder=3,linespacing=1.22)
def gate(ax,cx,cy,kind,r=0.44):
    ax.add_patch(Rectangle((cx-r,cy-0.16),2*r,0.16,fc=GREY,ec=GREY_E,lw=1.5,zorder=2))
    ax.add_patch(Wedge((cx,cy-0.16),r,0,180,fc=GREY,ec=GREY_E,lw=1.5,zorder=2))
    ax.text(cx,cy-0.01,kind,ha="center",va="center",fontsize=GATE,weight="bold",zorder=3)
def line(ax,p0,p1,lw=1.5,arm=False):
    ax.annotate("",xy=p1,xytext=p0,zorder=1,arrowprops=dict(arrowstyle="-|>" if arm else "-",color=NAVY,lw=lw,shrinkA=0,shrinkB=0))
def swatch(ax,x,y,fc,ec,txt,fs=LEG):
    ax.add_patch(FancyBboxPatch((x,y-0.17),0.55,0.34,boxstyle="round,pad=0.01,rounding_size=0.06",fc=fc,ec=ec,lw=1.5))
    ax.text(x+0.72,y,txt,fontsize=fs,va="center",ha="left")

# ══════════ GSN ══════════
def gsn():
    fig,ax=plt.subplots(figsize=(16.5,11.8)); ax.set_xlim(0,16.5); ax.set_ylim(0,11.8); ax.axis("off")
    ax.text(0.2,11.45,"ARISE CORTEX FDIR  ·  Assurance Case (GSN)",fontsize=TITLE,weight="bold",color=NAVY,ha="left")
    rbox(ax,8.3,10.1,5.9,1.15,"G1 · The onboard autonomy\ncannot make the spacecraft unsafe\n(SAFETY_CASE top claim)",BLUE,BLUE_E,fs=BIG,weight="bold")
    rbox(ax,13.9,10.3,4.8,0.72,"C1 · RTA simplex: only VSC+RTA+CBF\nare safety-critical (Class A/B)",GREY,GREY_E,fs=SMALL)
    rbox(ax,13.9,9.25,4.8,0.6,"C2 · Design D · Apophis · XB6 deputies",GREY,GREY_E,fs=SMALL)
    para(ax,8.3,8.5,6.1,0.98,"S1 · Argue by run-time assurance:\ncertify the MONITOR (VSC/RTA/CBF),\nnot the model",fs=MED)
    rbox(ax,13.9,8.5,4.8,0.8,"J1 · Cofer et al. NFM 2020;\nNASA Autonomy V&V Roadmap",CREAM,CREAM_E,fs=SMALL)
    line(ax,(8.3,9.525),(8.3,8.99),arm=True)
    line(ax,(11.25,10.2),(11.5,10.3),arm=True); line(ax,(11.25,9.8),(11.5,9.25),arm=True)
    line(ax,(11.35,8.5),(11.5,8.5),arm=True)
    gx=[1.55,4.27,6.99,9.71,12.43,15.15]; gy=6.6
    gt=["G2 · Decision Module\nis SOUND (no unsafe\nproposal executed)",
        "G3 · Collision\nkeep-out is\nFORWARD-INVARIANT",
        "G4 · Deterministic,\nbounded-memory,\nfixed-point path",
        "G5 · Graceful\ndegradation (AC fail\n→ safe action)",
        "G6 · Learned/AC tier\nCONTAINED (Class-C,\nadvisory only)",
        "G7 · Detection &\nf-tolerance\nadequate"]
    for x,t in zip(gx,gt):
        rbox(ax,x,gy,2.55,1.2,t,BLUE,BLUE_E,fs=SMALL,weight="bold")
        line(ax,(8.3,8.01),(x,gy+0.62),lw=1.2,arm=True)
    sy=4.7
    st=["Sn1\nFR2·P1\nexhaustive\n+ Z3","Sn2\nFR2·P2\nsympy +\nZ3 ∀","Sn3\nF9 T1/T2/\nT6 + FR2·P3",
        "Sn4\nF9 T7\ngraceful","Sn5\nFMECA +\nFTA (FR6)","Sn6\nCERT\nDOSSIER"]
    for x,t in zip(gx,st):
        circ(ax,x,sy,0.72,t,fs=11); line(ax,(x,gy-0.6),(x,sy+0.72),lw=1.2,arm=True)
    dy=1.7
    diamond(ax,4.27,dy,3.9,2.05,"G8 · WCET & fixed-point\nverified on GR712/GR740\n(FR3, on-target)",fs=10.5)
    diamond(ax,9.71,dy,3.9,2.05,"G9 · AC SEU + partitioning\non FPGA / HIL brassboard\n(FR4 / FR5)",fs=10.5)
    diamond(ax,14.0,dy,3.9,2.05,"G10 · Independent V&V\n+ FMECA/FTA sign-off\n(FR6)",fs=10.5)
    line(ax,(4.27,gy-0.6),(4.27,dy+1.02),lw=1.2,arm=True)
    line(ax,(9.71,gy-0.6),(9.71,dy+1.02),lw=1.2,arm=True)
    line(ax,(12.43,gy-0.6),(13.8,dy+0.85),lw=1.2,arm=True)
    ax.text(8.3,3.15,"TO BE DEVELOPED  (outstanding: the FR3/FR4/FR5/FR6 gap;\nnone is RL training)",fontsize=MED,weight="bold",color=ORA_E,ha="center",va="center",linespacing=1.25,bbox=dict(facecolor="white",edgecolor="none",pad=2))
    ly=0.42
    swatch(ax,0.3,ly,BLUE,BLUE_E,"Goal")
    ax.add_patch(Polygon([(2.75,ly-0.17),(3.4,ly-0.17),(3.3,ly+0.17),(2.65,ly+0.17)],closed=True,fc=PUR,ec=PUR_E,lw=1.5)); ax.text(3.55,ly,"Strategy",fontsize=LEG,va="center")
    swatch(ax,5.4,ly,GREY,GREY_E,"Context / Justification")
    ax.add_patch(Circle((9.7,ly),0.18,fc=GRN,ec=GRN_E,lw=1.5)); ax.text(9.98,ly,"Solution (evidence)",fontsize=LEG,va="center")
    ax.add_patch(Polygon([(13.0,ly+0.18),(13.28,ly),(13.0,ly-0.18),(12.72,ly)],closed=True,fc=CREAM,ec=CREAM_E,lw=1.5)); ax.text(13.4,ly,"Undeveloped goal",fontsize=LEG,va="center")
    fig.savefig(OUT/"fdi_assurance_gsn.png",dpi=200,bbox_inches="tight",pad_inches=0.15); plt.close(fig); print("wrote gsn")

# ══════════ Fault tree ══════════
def fault_tree():
    fig,ax=plt.subplots(figsize=(16.5,11)); ax.set_xlim(0,16.5); ax.set_ylim(0,11); ax.axis("off")
    ax.text(0.2,10.6,"ARISE CORTEX FDIR  ·  Fault Tree",fontsize=TITLE,weight="bold",color=NAVY,ha="left")
    rbox(ax,8.3,10.0,14.6,0.95,"TOP EVENT\nUnsafe actuation · collision / keep-out breach · dV-authority violation · loss of vehicle",SALM,RED_E,weight="bold",fs=13.5)
    gate(ax,8.3,9.15,"OR"); line(ax,(8.3,9.52),(8.3,9.1))
    cx=[2.3,6.3,10.3,14.3]; iey=8.4
    line(ax,(8.3,8.78),(8.3,8.95)); line(ax,(2.3,8.95),(14.3,8.95))
    ie=["IE1 · Unsafe AC action\nis EXECUTED","IE2 · VSC baseline\nitself unsafe","IE3 · CBF admits a\nkeep-out breach","IE4 · Safety spine not\nrun / corrupted"]
    kinds=["AND","OR","AND","OR"]
    b1=[("AC proposes an unsafe\naction (RL / planner /\nperception)\ncredible input",BLUE,BLUE_E),
        ("Contingency-table entry\nunsafe · config-mgmt\n(FR7) + CBF gate",ORA,ORA_E),
        ("Keep-out approached\ncredible geometry",BLUE,BLUE_E),
        ("Partition breach ·\nAC stalls VSC\n(cFS partitioning)\nFR5 OUTSTANDING",RED,RED_E)]
    b2=[("RTA fails to veto it\nEXCLUDED · FR2·P1\n(exhaustive + Z3\nsoundness)",GRN,GRN_E),
        ("Safe-hold action unsafe\ncertified /\ndeterministic (F9)",ORA,ORA_E),
        ("CBF invariance fails\nEXCLUDED · FR2·P2\n(Z3 for-all +\nfixed-point)",GRN,GRN_E),
        ("SEU flips a VSC\nsafety bit · rad-hard +\nEDAC · FR3/4 OUTSTANDING",RED,RED_E)]
    for i,x in enumerate(cx):
        line(ax,(x,8.95),(x,iey+0.48))
        rbox(ax,x,iey,3.5,0.85,ie[i],CREAM,CREAM_E,fs=SMALL,weight="bold")
        gate(ax,x,7.5,kinds[i]); line(ax,(x,iey-0.43),(x,7.45))
        rbox(ax,x,6.25,3.6,1.35,b1[i][0],b1[i][1],b1[i][2],fs=SMALL)
        line(ax,(x,7.24),(x,6.95))
        rbox(ax,x,4.55,3.6,1.35,b2[i][0],b2[i][1],b2[i][2],fs=SMALL)
        line(ax,(x,5.575),(x,5.25))
    swatch(ax,0.6,3.35,BLUE,BLUE_E,"Credible input (cut by a barrier)")
    swatch(ax,8.6,3.35,GRN,GRN_E,"Excluded by FR2 machine-checked proof")
    swatch(ax,0.6,2.75,ORA,ORA_E,"Bounded (fixed-point / config / certified)")
    swatch(ax,8.6,2.75,RED,RED_E,"Residual: on-target outstanding (FR3/4/5)")
    note=("The RL / AC inputs (blue) cannot reach the top event alone: the RTA gate (FR2·P1) and the CBF (FR2·P2) cut those paths.\n"
          "The only residual paths are on-target hardware (red), the FR3/FR4/FR5 HIL/PIL gap, plus config-management. None is the RL policy.")
    ax.add_patch(FancyBboxPatch((0.9,1.15),14.7,0.95,boxstyle="round,pad=0.02,rounding_size=0.08",fc="#eef4fb",ec=BLUE_E,lw=1.3))
    ax.text(8.25,1.62,note,fontsize=11,style="italic",ha="center",va="center",color="#20304a",linespacing=1.35)
    fig.savefig(OUT/"fdi_fault_tree.png",dpi=200,bbox_inches="tight",pad_inches=0.15); plt.close(fig); print("wrote fault_tree")

if __name__=="__main__":
    gsn(); fault_tree()
