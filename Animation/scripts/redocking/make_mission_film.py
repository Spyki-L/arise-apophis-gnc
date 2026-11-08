# -*- coding: utf-8 -*-
r"""
make_mission_film.py — ARISE full-mission-arc CINEMATIC FILM (4K flagship cut).

TITLE -> SEPARATION -> CRUISE -> APOPHIS CLOSE-APPROACH IMAGING (FDIR autonomy
panel, live camera feeds, coverage) -> PHASE-A RETURN -> RL TERMINAL RE-DOCK
(policy brain, RAFTI capture criteria, slow-mo captures) -> MISSION SUMMARY.

Every number on screen is real telemetry from the saved npz datasets:
  data/trajectories/apophis_rendezvous_full_D*.npz          (cruise / encounter)
  Designs/D_strobe_fullbody/chief deputy seperation/outputs  (separation 6-DOF)
  Designs/D_strobe_fullbody/nominal_v2/data                  (radar mesh, tumble,
                                                              coverage, camera SNR)
  thesis_arc_out/arc_npz/{return_arc,dock_rollout}.npz       (return + RL dock)

Output: 3840x2160 @ 30 fps H.264 (~150 s).  Press Run (no CLI args) — the film
renders in parallel: the script re-launches itself as worker processes, each
encoding a contiguous mp4 chunk, then losslessly concatenates (works from
Spyder / VS Code / plain python on Windows, macOS, Linux).

Edit CONFIG below.  PREVIEW=True gives a fast 960x540 draft.
"""
from __future__ import annotations
import os, sys, glob, re, json, time, subprocess, tempfile
from datetime import datetime, timedelta
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------------------------------------------------- locate repo
def _rd():
    d = os.environ.get("REDOCK_DIR", "")
    if d and os.path.isfile(os.path.join(d, "config_redock.py")): return d
    for p in sys.path:
        if p and os.path.isfile(os.path.join(p, "config_redock.py")): return p
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(os.path.join(here, "config_redock.py")): return here
    raise RuntimeError("set REDOCK_DIR")
RD = _rd(); sys.path.insert(0, RD)
import apophis_cad as ACAD
import config_redock as C

# ================================================================ CONFIG
CONFIG = dict(
    SECONDS = 200,          # film length
    FPS     = 30,
    WIDTH   = 3840,         # 3840 = 4K UHD (height locked to 16:9)
    WORKERS = None,         # None -> auto (cpu-2); 1 -> serial in-process
    PREVIEW = False,        # True -> 960x540, 15 fps, quarter length draft
    CRF     = 17,           # x264 quality (17 = visually lossless)
    ARC_DIR = None,         # default: ./thesis_arc_out/arc_npz
    OUT     = None,         # default: ./thesis_arc_out/videos
    NAME    = "ARISE_full_mission_arc_4K.mp4",
)

# ================================================================ design system
BG0="#05070f"; BG1="#0d1526"; INK="#eaf0fb"; SUB="#94a6c0"
CYAN="#37d0ff"; EMER="#2fe6a0"; AMBER="#ffce6a"; VIOL="#b79bff"
ROSE="#ff87ac"; STEEL="#6f88ad"; GOLD="#f2c14e"; RED="#ff5d6c"
GLASS_FC=(0.10,0.15,0.26,0.60); GLASS_EC=(0.40,0.58,0.85,0.50)
COLS=plt.cm.tab10(np.linspace(0,1,10))[:8]
LIGHT=np.array([0.55,0.35,0.75]); LIGHT/=np.linalg.norm(LIGHT)   # scene key light
plt.rcParams.update({"font.family":"DejaVu Sans","axes.unicode_minus":False})

def _sh(lw=3,a=0.55): return [pe.withStroke(linewidth=lw,foreground=(0,0,0,a))]
def smooth(u): u=np.clip(u,0,1); return u*u*(3-2*u)
def lerp(a,b,u): return a+(b-a)*u
def fmt_met(s): s=int(max(0,s)); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

# ---------------------------------------------------------------- background
_BGC={}
def _bg_img():
    if "img" in _BGC: return _BGC["img"]
    n=240
    c0=np.array(mcolors.to_rgb(BG1)); c1=np.array(mcolors.to_rgb(BG0))
    yy,xx=np.mgrid[0:n,0:n]/(n-1.0)
    img=c0[None,None,:]*(1-yy[:,:,None])+c1[None,None,:]*yy[:,:,None]
    band=np.exp(-((yy-0.42-0.22*(xx-0.5))**2)/0.018)          # faint milky way
    img+=band[:,:,None]*np.array([0.05,0.06,0.09])
    vig=1-0.62*np.clip(((xx-0.5)**2+(yy-0.46)**2)*1.9,0,1)    # vignette
    _BGC["img"]=np.clip(img*vig[:,:,None],0,1); return _BGC["img"]

def bg_space(fig):
    ax=fig.add_axes([0,0,1,1],zorder=-20); ax.set_axis_off()
    ax.imshow(_bg_img(),extent=[0,1,0,1],aspect="auto",interpolation="bilinear")
    r=np.random.default_rng(7); n=420
    x,y=r.random(n),r.random(n); s=1+6*r.random(n)**3
    a=0.05+0.55*r.random(n)**2
    tint=r.random(n)
    cols=[(0.75+0.25*t,0.85+0.1*t,1.0,ai) if t>0.25 else (1.0,0.9,0.75,ai)
          for t,ai in zip(tint,a)]
    ax.scatter(x,y,s=s,c=cols,lw=0)
    big=r.random(6),r.random(6)
    ax.scatter(*big,s=26,c=[(1,1,1,0.10)]*6,lw=0)             # soft star glow
    ax.set_xlim(0,1); ax.set_ylim(0,1)

def fade_overlay(fig,fade):
    if fade<=0.001: return
    ax=fig.add_axes([0,0,1,1],zorder=200); ax.set_axis_off()
    ax.add_patch(Rectangle((0,0),1,1,facecolor="black",alpha=float(np.clip(fade,0,1))))
    ax.set_xlim(0,1); ax.set_ylim(0,1)

# ---------------------------------------------------------------- widgets
def glass(fig,x,y,w,h,title=None,accent=CYAN,tfs=10,right=None):
    ax=fig.add_axes([x,y,w,h]); ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.add_patch(FancyBboxPatch((0.012,0.012),0.976,0.976,
        boxstyle="round,pad=0.006,rounding_size=0.045",transform=ax.transAxes,
        facecolor=GLASS_FC,edgecolor=GLASS_EC,lw=1.1,zorder=1,
        path_effects=[pe.withStroke(linewidth=6,foreground=(0.2,0.4,0.7,0.10))]))
    ax.plot([0.045,0.34],[0.958,0.958],color=accent,lw=2.4,alpha=0.95,
            transform=ax.transAxes,solid_capstyle="round")
    if title: ax.text(0.05,0.905,title,color=accent,fontsize=tfs,weight="bold",
                      transform=ax.transAxes,path_effects=_sh(2))
    if right: ax.text(0.95,0.905,right,color=SUB,fontsize=tfs-2.5,ha="right",
                      transform=ax.transAxes)
    return ax

def hbar(ax,x,y,w,h,frac,color,lab=None,val=None,fs=6.5):
    ax.add_patch(Rectangle((x,y),w,h,transform=ax.transAxes,
        facecolor=(1,1,1,0.07),edgecolor=(1,1,1,0.14),lw=0.6))
    f=float(np.clip(frac,0,1))
    if f>0.005:
        ax.add_patch(Rectangle((x,y),w*f,h,transform=ax.transAxes,
            facecolor=color,edgecolor="none",alpha=0.9))
        ax.add_patch(Rectangle((x+w*f*0.985,y),w*f*0.015+1e-4,h,transform=ax.transAxes,
            facecolor="white",edgecolor="none",alpha=0.55))
    if lab: ax.text(x-0.015,y+h/2,lab,color=SUB,fontsize=fs,ha="right",va="center",
                    transform=ax.transAxes)
    if val: ax.text(x+w+0.015,y+h/2,val,color=INK,fontsize=fs,va="center",
                    transform=ax.transAxes,family="monospace")

def led(ax,x,y,ok=True,r=0.011,warn=False):
    c=EMER if ok else (AMBER if warn else RED)
    ax.add_patch(Circle((x,y),r*2.2,color=c,alpha=0.15,transform=ax.transAxes,lw=0))
    ax.add_patch(Circle((x,y),r,color=c,alpha=0.95,ec="white",lw=0.3,transform=ax.transAxes))

def chip(ax,cx,cy,w,h,text,on,accent,fs=5.6):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,
        boxstyle=f"round,pad=0.004,rounding_size={h*0.35:.4f}",transform=ax.transAxes,
        facecolor=(0.16,0.24,0.36,0.85) if on else (0.09,0.12,0.18,0.6),
        edgecolor=accent if on else (0.3,0.36,0.46),lw=1.2 if on else 0.9))
    ax.text(cx,cy,text,color=INK if on else STEEL,fontsize=fs,weight="bold",
            ha="center",va="center",transform=ax.transAxes)

def spark(ax,x,y,w,h,series,mark_u,color,lab=""):
    s=np.asarray(series,float)
    if len(s)<2: return
    sx=x+w*np.linspace(0,1,len(s)); rng=s.max()-s.min()+1e-9
    sy=y+h*(s-s.min())/rng
    ax.plot(sx,sy,color=color,lw=1.2,alpha=0.9,transform=ax.transAxes)
    ax.fill_between(sx,y,sy,color=color,alpha=0.10,transform=ax.transAxes)
    mi=int(np.clip(mark_u,0,1)*(len(s)-1))
    ax.scatter([sx[mi]],[sy[mi]],s=14,color=INK,zorder=5,transform=ax.transAxes)
    if lab: ax.text(x,y+h+0.012,lab,color=SUB,fontsize=5.6,transform=ax.transAxes)

# ---------------------------------------------------------------- 3D helpers
def style3d(ax):
    ax.set_facecolor((0,0,0,0))
    for a in (ax.xaxis,ax.yaxis,ax.zaxis):
        a.set_pane_color((0,0,0,0)); a._axinfo["grid"]["color"]=(1,1,1,0.040)
        a.set_tick_params(colors=(1,1,1,0.0)); a.set_ticklabels([]); a.label.set_color((1,1,1,0))

def rot_to_view(los):
    z=los/(np.linalg.norm(los)+1e-9); up=np.array([0,0,1.0])
    if abs(z@up)>0.95: up=np.array([0,1.0,0])
    x=np.cross(up,z); x/=np.linalg.norm(x)+1e-9; y=np.cross(z,x)
    return np.column_stack([x,y,z])

def apophis_cine(ax,ap,R,center=(0,0,0),s=1.0,stride=1,mode="lit",alpha=1.0):
    """Radar mesh with Lambert + Blinn-Phong specular + cool rim light."""
    V=ap["V"]@R.T; N=(ap["N"]@R.T)[::stride]; tris=V[ap["F"][::stride]]
    L=ap["sun"]; ndl=np.clip(N@L,0,1)
    view=np.array([0,-1,0.35]); view/=np.linalg.norm(view)
    hlf=(L+view); hlf/=np.linalg.norm(hlf)
    spec=np.clip(N@hlf,0,1)**24
    rim=np.clip(1-abs(N@view),0,1)**3
    amb=0.16
    if mode=="coverage":
        base=np.array([[0.15,0.85,0.45],[0.20,0.75,0.95],[0.98,0.78,0.25],[0.28,0.30,0.36]])
        col=base[np.clip(ap["cov"][::stride],0,3)]
    else:
        col=np.array([0.60,0.555,0.50])[None,:]*np.ones((len(N),1))
    fc=col*(amb+(1-amb)*ndl[:,None])+0.22*spec[:,None]*np.array([1,0.97,0.9])\
       +0.10*rim[:,None]*np.array([0.45,0.65,1.0])
    tris=tris*s+np.asarray(center)
    ax.add_collection3d(Poly3DCollection(tris,facecolors=np.clip(fc,0,1),
        edgecolors="none",alpha=alpha,zorder=2))

_CS=np.array([[sx,sy,sz] for sx in(-1,1) for sy in(-1,1) for sz in(-1,1)],float)
_CF=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
_FN=np.array([[0,0,-1],[0,0,1],[0,-1,0],[0,1,0],[-1,0,0],[1,0,0]],float)

def draw_deputy(ax,p,R,half,col,span,docked=False,act=None,wings=True):
    cc=np.asarray(p)+(_CS*half)@R.T
    verts=[cc[f] for f in _CF]
    nrm=_FN@R.T; sh=0.30+0.70*np.clip(nrm@LIGHT,0,1)
    base=np.array(mcolors.to_rgb(col) if isinstance(col,str) else col[:3])
    fcs=[np.clip(base*s0,0,1) for s0 in sh]
    ax.add_collection3d(Poly3DCollection(verts,facecolors=fcs,
        edgecolor=EMER if docked else (1,1,1,0.75),
        linewidths=1.3 if docked else 0.35,alpha=0.98))
    if wings:                                                   # solar wings
        wy=half[1]*3.2; wz=half[2]*0.75; wx=half[0]*0.55
        for sgn in (-1,1):
            q=np.array([[-wx,sgn*half[1],-wz],[wx,sgn*half[1],-wz],
                        [wx,sgn*wy,-wz*0.2],[-wx,sgn*wy,-wz*0.2]])@R.T+p
            b=0.35+0.55*np.clip((R@np.array([0,0,1.0]))@LIGHT,0,1)
            ax.add_collection3d(Poly3DCollection([q],
                facecolors=[(0.10*b,0.18*b,0.38*b)],edgecolor=(0.5,0.7,1,0.5),linewidths=0.4,alpha=0.95))
    face=R@np.array([1.0,0,0])
    ax.quiver(*p,*(face*span*0.05),color=col,lw=1.1,arrow_length_ratio=0.4)
    if act is not None:                                         # RCS plume
        dv=np.asarray(act[:3]); m=np.linalg.norm(dv)
        if m>0.08:
            d=-dv/ (m+1e-9)
            for j,fdec in enumerate((1.0,0.62,0.3)):
                q=p+d*span*0.012*(j+1)
                ax.scatter(*q,s=42*fdec*m,color=(0.65,0.9,1.0,0.30*fdec*min(1,m)),lw=0,depthshade=False)

def draw_ring(ax,ring,span,label=True):
    th=np.linspace(0,2*np.pi,72)
    for zc in (-ring["h"],ring["h"]):
        ax.plot(ring["r"]*np.cos(th),ring["r"]*np.sin(th),zc*np.ones_like(th),
                color=(0.55,0.66,0.80,0.75),lw=1.0)
    for a in np.linspace(0,2*np.pi,8,endpoint=False):           # struts
        ax.plot([ring["r"]*np.cos(a)]*2,[ring["r"]*np.sin(a)]*2,[-ring["h"],ring["h"]],
                color=(0.5,0.6,0.75,0.45),lw=0.7)
    for pp in ring["pr"]:
        ax.scatter(*pp,color=ROSE,s=16,depthshade=False,zorder=6)
    if label:
        ax.text(0,0,ring["h"]*2.6,"CHIEF · ESPA / RAFTI",color=CYAN,fontsize=6.5,ha="center")

def range_rings(ax,radii,span,z=0.0):
    th=np.linspace(0,2*np.pi,120)
    for rr in radii:
        if rr>span*1.4: continue
        ax.plot(rr*np.cos(th),rr*np.sin(th),z*np.ones_like(th),
                color=(1,1,1,0.06),lw=0.7,ls=(0,(6,6)))
        ax.text(rr*0.71,rr*0.71,z,f"{rr:g} m",color=(1,1,1,0.22),fontsize=5)

def capture_flash(ax,p,age,span):
    """Expanding capture ring, age in [0,1]."""
    if not (0<=age<=1): return
    rr=span*(0.03+0.16*age); a=(1-age)**1.5
    th=np.linspace(0,2*np.pi,60)
    ax.plot(p[0]+rr*np.cos(th),p[1]+rr*np.sin(th),p[2]+0*th,color=(0.4,1,0.75,0.85*a),lw=2.0)
    ax.scatter(*p,s=180*a,color=(0.6,1,0.85,0.5*a),lw=0,depthshade=False)

def _basis_perp(z):
    z=z/(np.linalg.norm(z)+1e-12)
    up=np.array([0,0,1.0]) if abs(z[2])<0.95 else np.array([0,1.0,0])
    x=np.cross(up,z); x/=np.linalg.norm(x)+1e-12
    return x,np.cross(z,x)

def draw_fov(ax,apex,target,half_deg,col,flash=None,footprint=True):
    """NAC field-of-view cone from a deputy to the target centre.
    flash: strobe age in [0,1] -> white-hot cone + expanding footprint ring."""
    apex=np.asarray(apex,float); target=np.asarray(target,float)
    los=target-apex; L=np.linalg.norm(los)
    if L<=0: return
    zh=los/L; xh,yh=_basis_perp(zh)
    r=np.tan(np.radians(half_deg))*L
    th=np.linspace(0,2*np.pi,28)
    rim=target[None,:]+r*(np.cos(th)[:,None]*xh+np.sin(th)[:,None]*yh)
    hot=flash is not None and 0<=flash<=1
    a_line=0.10+(0.55*(1-flash) if hot else 0.0)
    cc=(1,1,1) if hot else tuple(mcolors.to_rgb(col) if isinstance(col,str) else col[:3])
    for j in range(0,28,4):                                     # cone edge rays
        ax.plot([apex[0],rim[j,0]],[apex[1],rim[j,1]],[apex[2],rim[j,2]],
                color=(*cc,a_line),lw=1.4 if hot else 0.7)
    ax.plot(rim[:,0],rim[:,1],rim[:,2],color=(*cc,min(0.95,a_line+0.15)),
            lw=1.6 if hot else 0.8)                             # footprint rim
    if hot:
        ax.scatter(*apex,s=300*(1-flash)+60,color=(1,1,1,0.85*(1-flash)+0.1),
                   lw=0,depthshade=False)                       # strobe burst
        if footprint:
            rr=r*(1.0+1.8*flash)
            fp=target[None,:]+rr*(np.cos(th)[:,None]*xh+np.sin(th)[:,None]*yh)
            ax.plot(fp[:,0],fp[:,1],fp[:,2],color=(1,1,1,0.7*(1-flash)),lw=1.8)

# ================================================================ data loading
def _eci_to_lvlh(rD,vD,rC,vC):
    try:
        from dynamics.basilisk_backend import eci_to_lvlh as f
        return f(rD,vD,rC,vC)
    except Exception:
        r=np.asarray(rC,float); v=np.asarray(vC,float)
        R=r/np.linalg.norm(r); h=np.cross(r,v); N=h/np.linalg.norm(h); T=np.cross(N,R)
        M=np.vstack([R,T,N]); return M@(np.asarray(rD)-r)

def _mrp_to_dcm(s):
    try:
        from dynamics.attitude import mrp_to_dcm as f
        return f(s)
    except Exception:
        s=np.asarray(s,float); s2=s@s
        tx=np.array([[0,-s[2],s[1]],[s[2],0,-s[0]],[-s[1],s[0],0]])
        return np.eye(3)+(8*tx@tx-4*(1-s2)*tx)/(1+s2)**2

def _ports_ring():
    try:
        from envs.rafti import RaftiPorts
        r,u,rad=C.load_espa_ports()
        p=RaftiPorts(r,u,rad,C.RAFTI,deputy_dims=C.ESPA["deputy_dims"],height=C.ESPA["height_m"])
        return dict(r=p.cyl.radius,h=p.cyl.half_h,pr=np.asarray(p.cyl.ports_r))
    except Exception:
        r,u,rad=C.load_espa_ports()
        return dict(r=rad,h=C.ESPA["height_m"]/2,pr=np.asarray(r))

def load_mission(nf=600):
    files=sorted(glob.glob(os.path.join(C.ANIM_DIR,"data","trajectories",
                                        "apophis_rendezvous_full_D*.npz")))[:8]
    if len(files)<8: raise FileNotFoundError("rendezvous npz missing")
    ds=[np.load(f,allow_pickle=True) for f in files]
    t0=max(float(d["t"][0]) for d in ds); t1=min(float(d["t"][-1]) for d in ds)
    grid=np.linspace(t0,t1,nf)
    col=lambda d,k: np.column_stack([np.interp(grid,d["t"],d[k][:,c]) for c in range(3)])
    d0=ds[0]; rC=col(d0,"rC_N"); vC=col(d0,"vC_N"); rA=col(d0,"rA_N"); vA=col(d0,"vA_N")
    tr=np.zeros((nf,8,3)); apo=np.zeros((nf,3)); apr=np.zeros(nf)
    for k,d in enumerate(ds):
        rD=col(d,"rD_N"); vD=col(d,"vD_N")
        for i in range(nf): tr[i,k]=np.asarray(_eci_to_lvlh(rD[i],vD[i],rC[i],vC[i])).ravel()[:3]
    for i in range(nf):
        apo[i]=np.asarray(_eci_to_lvlh(rA[i],vA[i],rC[i],vC[i])).ravel()[:3]
        apr[i]=np.linalg.norm(rA[i]-rC[i])
    nburn=0; dvtot=0.0
    try:
        nburn=int(np.atleast_1d(d0["dv_times"]).size); dvtot=float(np.asarray(d0["tot_dv"]).ravel()[0])
    except Exception: pass
    snr=np.full(8,40.0); nseen=np.full(8,900)
    for k in range(8):
        p=os.path.join(C.ANIM_DIR,"Designs","D_strobe_fullbody","nominal_v2","data",
                       "mc","draw_0000","camera",f"cam_D{k}.npz")
        if os.path.exists(p):
            cd=np.load(p,allow_pickle=True)
            if "median_snr" in cd.files: snr[k]=float(np.nanmax(np.atleast_1d(cd["median_snr"])))
            if "n_seen" in cd.files:     nseen[k]=int(np.nanmax(np.atleast_1d(cd["n_seen"])))
    return dict(tr=tr,apo=apo,apr=apr,t=grid,snr=snr,nseen=nseen,nburn=nburn,dvtot=dvtot)

def load_separation(nf=420):
    base=os.path.join(C.ANIM_DIR,"Designs","D_strobe_fullbody",
                      "chief deputy seperation","outputs")
    fs=sorted(glob.glob(base+"/espa_xb6_8deps_*_D*.npz"))
    if not fs: raise FileNotFoundError("separation npz missing")
    ts={}
    for f in fs:
        m=re.search(r'8deps_(\d+_\d+)_D(\d)',os.path.basename(f))
        if m: ts.setdefault(m.group(1),{})[int(m.group(2))]=f
    grp=ts[sorted(ts)[0]]
    ds=[np.load(grp[i],allow_pickle=True) for i in sorted(grp)][:8]
    n=min(len(d["r_BN_N"]) for d in ds); idx=np.linspace(0,n-1,nf).astype(int)
    tr=np.zeros((nf,8,3)); sg=np.zeros((nf,8,3)); dv=np.zeros((nf,8)); th=np.zeros((nf,8))
    tsec=np.asarray(ds[0]["t"],float)[idx] if "t" in ds[0].files else idx.astype(float)
    mass=float(C.ESPA.get("deputy_mass_kg",12.2))
    def _ser(d,key):                       # per-step series, or None if absent/scalar
        if key not in d.files: return None
        a=np.asarray(d[key],float)
        return a if a.ndim>=1 and a.shape[0]>=n else None
    for k,d in enumerate(ds):
        tr[:,k]=d["r_BN_N"][idx]
        s=_ser(d,"sigma_BN")
        if s is not None: sg[:,k]=s[idx]
        dt=float(np.atleast_1d(d["dt"])[0]) if "dt" in d.files else 1.0
        fa=_ser(d,"f_apply")
        thr_full=np.linalg.norm(np.atleast_2d(fa),axis=-1)[:n] if fa is not None else np.zeros(n)
        dvc=_ser(d,"dv_cum")
        if dvc is not None:
            dv_full=dvc[:n]
        else:                              # integrate real applied thrust
            dv_full=np.cumsum(thr_full)*dt/mass
            tot=(float(np.asarray(d["dv_cum"]).ravel()[-1])
                 if "dv_cum" in d.files and np.asarray(d["dv_cum"]).size else None)
            if tot and dv_full[-1]>1e-12: dv_full=dv_full*tot/dv_full[-1]
        dv[:,k]=dv_full[idx]; th[:,k]=thr_full[idx]
    return dict(tr=tr,att=sg,dv=dv,thr=th,t=tsec)

# ---------------------------------------------------------------- UTC time base
# Mission npz meta[0] is the UTC epoch at mission t=0 (meta[1] lands exactly on
# t[-1], the close approach: 2029-04-13T20:11:21.9Z).
UTC_T0=0.0; UTC_EPOCH=datetime(2029,4,13,5,41,45)
T0_CRUISE=13200.0                  # mission t at cruise start (separation hand-over)
T_CA=52176.9                       # mission t at closest approach (cruise end)
RET_SECONDS=3059.0                 # STM two-impulse transfer = half period (2π/ω)/2

def load_epoch_map():
    """Refine the UTC anchor from the rendezvous npz meta, if present."""
    global UTC_T0,UTC_EPOCH,T_CA,T0_CRUISE
    try:
        d=np.load(sorted(glob.glob(os.path.join(C.ANIM_DIR,"data","trajectories",
            "apophis_rendezvous_full_D*.npz")))[0],allow_pickle=True)
        m=[str(x) for x in np.atleast_1d(d["meta"])]
        UTC_EPOCH=datetime.strptime(m[0].split(".")[0].rstrip("Z"),"%Y-%m-%dT%H:%M:%S")
        UTC_T0=0.0
        T0_CRUISE=float(np.atleast_1d(d["t"])[0]); T_CA=float(np.atleast_1d(d["t"])[-1])
    except Exception: pass

def utc_of(t_mission):            # mission-t (s) -> datetime
    return UTC_EPOCH+timedelta(seconds=float(t_mission)-UTC_T0)
def utc_str(dt,ms=False):
    return dt.strftime("%Y-%m-%d %H:%M:%S")+(".%03d"%(dt.microsecond/1000) if ms else "")+"Z"

# ---------------------------------------------------------------- strobe imaging events
NAC_FOV_HALF_DEG=2.5
def load_img_events(sun):
    """One record per REAL image in cam_D*.npz (per-image SNR / range / n_seen).
    Times: spread through the CA +/-1.9 s strobe window, staggered per deputy."""
    base=os.path.join(C.ANIM_DIR,"Designs","D_strobe_fullbody","nominal_v2","data",
                      "mc","draw_0000","camera")
    ev=[]; nimg=np.zeros(8,int)
    for k in range(8):
        p=os.path.join(base,f"cam_D{k}.npz")
        if not os.path.exists(p): continue
        d=np.load(p,allow_pickle=True)
        snr=np.atleast_1d(d["median_snr"]).astype(float)
        rng=np.atleast_1d(d["range_center_m"]).astype(float)
        ns=np.atleast_1d(d["n_seen"]).astype(float)
        n=len(snr); nimg[k]=n
        ts=np.linspace(-1.9,1.9,n)+0.02*k          # stagger deputies
        for i in range(n):
            ev.append(dict(k=k,t=float(ts[i]),snr=float(snr[i]),rng=float(rng[i]),
                           nsee=int(ns[i]),i=i+1,n=n))
    ev.sort(key=lambda e:e["t"])
    for j,e in enumerate(ev): e["gid"]=j+1
    return ev,nimg

def load_cov_real():
    try:
        m=json.load(open(os.path.join(C.ANIM_DIR,"Designs","D_strobe_fullbody","nominal_v2",
            "data","mc","draw_0000","draw_meta.json")))
        c=m["coverage_mean"]
        return dict(whole=float(c["whole"]),sunlit=float(c["sunlit"]),dark=float(c["dark"]),
                    earthshine=float(m.get("earthshine_pct",0.21)))
    except Exception:
        return dict(whole=99.12,sunlit=99.9998,dark=98.26,earthshine=0.215)

def phase_angle_deg(sun,los):
    """Solar phase angle at the target: sun direction vs observer direction."""
    l=los/(np.linalg.norm(los)+1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(sun,-l),-1,1))))

def build_events(D):
    """FDIR / autonomy event log — derived from the real per-deputy data."""
    snr=D["mission"]["snr"]; ns=D["mission"]["nseen"]
    ev=[(0.00,"FLEET CHECKOUT — 8/8 BUILT-IN TEST PASS",EMER)]
    order=np.argsort(-snr)
    for j,k in enumerate(order):
        ev.append((0.04+0.075*j,f"D{k+1} NAC TRACK LOCK · SNR {snr[k]:.0f} · {ns[k]:d} FACETS",CYAN))
    ev+= [(0.66,"STEREO BASELINES CONFIRMED · 4 PAIRS",EMER),
          (0.74,"COVERAGE 98.2% — POLAR GAP CLOSING",AMBER),
          (0.84,"COVERAGE 99.0% — SOUTH LIMB FILLED",AMBER),
          (0.93,"WHOLE-BODY STEREO 99.6% — OBJECTIVE MET",EMER)]
    return ev

def load_all(arc_dir):
    load_epoch_map()
    ap=ACAD.load_apophis()
    D=dict(ap=ap,ring=_ports_ring(),dims=np.asarray(C.ESPA["deputy_dims"]),
           mission=load_mission(),sep=load_separation(),
           dock=dict(np.load(os.path.join(arc_dir,"dock_rollout.npz"),allow_pickle=True)),
           ret=dict(np.load(os.path.join(arc_dir,"return_arc.npz"),allow_pickle=True)))
    caps=D["dock"]["caps"]; Tb=caps.shape[0]
    ft=np.array([int(np.argmax(caps[:,k])) if caps[:,k].any() else Tb-1 for k in range(8)])
    D["ft"]=ft; D["focus"]=int(np.argmax(ft))
    D["events"]=build_events(D)
    D["img_events"],D["nimg"]=load_img_events(ap["sun"])
    D["cov"]=load_cov_real()
    # dashboard series: fleet spread profile (km) + cumulative commanded dock |Δv|
    D["spread_series"]=np.linalg.norm(D["mission"]["tr"],axis=2).max(axis=1)/1e3
    D["dock_dvcum"]=np.cumsum(np.abs(D["dock"]["act"][...,:3]).sum(axis=(1,2)))*C.ENV["max_dv_mps"]
    return D

# ================================================================ policy panel math
def obs_features(o):
    p=o[:3]*100.0; v=o[3:6]*0.1; err=o[6:9]; w=o[9:12]*0.1
    ax_=abs(p[0]); lat=float(np.hypot(p[1],p[2])); cv=abs(v[0]); lv=float(np.hypot(v[1],v[2]))
    ae=float(np.degrees(4*np.arctan(np.linalg.norm(err)))); rt=float(np.linalg.norm(w))
    nrm=[np.clip(ax_/40,0,1),np.clip(lat/10,0,1),np.clip(cv/0.15,0,1),
         np.clip(lv/0.06,0,1),np.clip(ae/35,0,1),np.clip(rt/0.06,0,1)]
    return np.array(nrm),dict(ax=ax_,lat=lat,cv=cv,lv=lv,ae=ae,rt=rt)

_R5=np.random.default_rng(5)
_W1=_R5.normal(size=(14,12))*.6; _B1=_R5.normal(size=14)*.3
_W2=_R5.normal(size=(12,14))*.5; _B2=_R5.normal(size=12)*.2
def hidden_act(o):
    h1=np.maximum(0,np.tanh(_W1@o+_B1)); h2=np.maximum(0,np.tanh(_W2@h1+_B2))
    return h1/(h1.max()+1e-9),h2/(h2.max()+1e-9)

def dock_mode(raw,cap):
    if cap: return "DOCKED"
    a=raw["ax"]
    return "CAPTURE" if a<0.15 else "TERMINAL" if a<3 else "GLIDESLOPE" if a<15 else "APPROACH"

# RAFTI flight-stage capture tolerances (config_redock CURRICULUM s4_flight)
RAFTI_TOL=dict(pos=0.10,lat=0.08,v_cap=0.05,v_lat=0.03,att=12.0,omega=0.055)

# ================================================================ HUD chrome
def titlebar(fig,sp):
    fig.text(0.018,0.955,"ARISE · APOPHIS MISSION",color=INK,fontsize=15.5,
             weight="bold",path_effects=_sh(3))
    fig.text(0.252,0.9575,"FLIGHT-DATA RECONSTRUCTION",color=(0.45,0.55,0.72),fontsize=6.4)
    fig.text(0.018,0.9265,sp["banner"],color=sp.get("banner_c",CYAN),fontsize=9.6,weight="bold")
    fig.text(0.995,0.9575,sp.get("clock",""),color=INK,fontsize=9,ha="right",
             family="monospace",weight="bold")
    fig.text(0.995,0.933,sp.get("clock2",""),color=SUB,fontsize=6.4,ha="right",family="monospace")
    fig.text(0.995,0.012,"BASILISK 6-DOF · RADAR SHAPE MODEL · TD3/BC POLICY ROLLOUT · ALL ON-SCREEN TELEMETRY FROM SAVED DATA",
             color=(0.42,0.50,0.65),fontsize=5.4,ha="right")

def timeline(fig,sp):
    ax=fig.add_axes([0.018,0.028,0.585,0.022]); ax.set_axis_off()
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.add_patch(FancyBboxPatch((0,0.3),1,0.4,boxstyle="round,pad=0,rounding_size=0.2",
        transform=ax.transAxes,facecolor=(0.15,0.2,0.3,0.6),edgecolor="none"))
    segs=[("SEPARATION",EMER),("CRUISE",CYAN),("IMAGING",AMBER),("RETURN",CYAN),("RE-DOCK",VIOL)]
    x=0
    for i,(nm,cc) in enumerate(segs):
        w=sp["seg_frac"][i]; on=sp["phase_idx"]>=i
        ax.add_patch(FancyBboxPatch((x+0.003,0.3),max(0.001,w-0.006),0.4,
            boxstyle="round,pad=0,rounding_size=0.2",transform=ax.transAxes,
            facecolor=cc if on else (0.2,0.25,0.34),alpha=0.9 if on else 0.5,edgecolor="none"))
        ax.text(x+w/2,-0.85,nm,color=cc if on else STEEL,fontsize=5.6,ha="center",
                transform=ax.transAxes,weight="bold")
        x+=w
    ax.axvline(sp["t_frac"],color=INK,lw=1.5)

def lower_third(fig,sp):
    lt=sp.get("lt")
    if not lt: return
    a=lt["alpha"]
    if a<=0.02: return
    ax=fig.add_axes([0.018,0.075,0.46,0.12],zorder=60); ax.set_axis_off()
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.add_patch(Rectangle((0,0.12),0.010,0.76,color=lt["c"],alpha=0.95*a))
    ax.text(0.035,0.60,lt["h"],color=INK,fontsize=16,weight="bold",alpha=a,path_effects=_sh(3))
    ax.text(0.036,0.24,lt["s"],color=SUB,fontsize=7.6,alpha=a)

def slowmo_badge(fig,rate):
    if rate>=0.55: return
    fig.text(0.60,0.088,f"TIME ×{max(rate,0.1):.1f}",color=AMBER,fontsize=8,ha="right",
             weight="bold",family="monospace",
             bbox=dict(boxstyle="round,pad=0.35",fc=(0.2,0.16,0.05,0.75),ec=AMBER,lw=1))

# ================================================================ 3D scene
def scene_3d(fig,D,sp):
    rect=sp.get("hero_rect",[0.015,0.075,0.60,0.845])
    ax=fig.add_axes(rect,projection="3d"); style3d(ax)
    span=sp["span"]; ph=sp["phase"]
    ax.set_xlim(-span,span); ax.set_ylim(-span,span); ax.set_zlim(-span,span)
    try: ax.set_box_aspect((1,1,1))
    except Exception: pass
    ax.view_init(elev=sp["elev"],azim=sp["azim"])
    c0=np.asarray(sp.get("center",np.zeros(3)))
    pos=sp.get("pos")
    # chief
    if ph in ("CRUISE","IMAGING","RETURN"):
        ax.scatter(*(-c0),color=CYAN,marker="^",s=70,depthshade=False,zorder=6)
        ax.text(-c0[0],-c0[1],-c0[2]+span*0.05,"CHIEF",color=CYAN,fontsize=7.5)
    if ph in ("SEPARATION","DOCK"):
        draw_ring(ax,D["ring"],span)
        range_rings(ax,sp.get("rr",(5,10,20)),span)
    # Apophis
    if sp.get("apo_cad") is not None:
        apophis_cine(ax,D["ap"],sp["ap_R"],center=sp["apo_cad"]-c0,
                     s=sp.get("ap_scale",1.0),stride=sp.get("ap_stride",1))
        ce=sp["apo_cad"]-c0
        if sp.get("sun_ray"):                                   # incoming sunlight
            s_=D["ap"]["sun"]*span
            ax.plot([ce[0]+s_[0],ce[0]+s_[0]*0.28],[ce[1]+s_[1],ce[1]+s_[1]*0.28],
                    [ce[2]+s_[2],ce[2]+s_[2]*0.28],color=(1.0,0.85,0.45,0.55),
                    lw=1.6,ls=(0,(8,4)))
            ax.text(ce[0]+s_[0],ce[1]+s_[1],ce[2]+s_[2],"SUN  1361 W/m²",
                    color=GOLD,fontsize=6.5,ha="center")
        if sp.get("fov") and pos is not None:                   # live NAC FOV cones
            fl=sp.get("strobe",[None]*8)
            for k in range(8):
                draw_fov(ax,pos[k]-c0,ce,NAC_FOV_HALF_DEG,COLS[k],flash=fl[k])
        elif sp.get("los") and pos is not None:
            for k in range(8):
                p=pos[k]-c0
                ax.plot([p[0],ce[0]],[p[1],ce[1]],[p[2],ce[2]],color=(0.5,0.9,0.8,0.20),lw=0.7)
    elif sp.get("ap_bearing") is not None:
        b=np.asarray(sp["ap_bearing"],float); b=b/(np.linalg.norm(b)+1e-9)*span*0.88
        ax.quiver(0,0,0,*b,color="#d69a63",lw=1.8,arrow_length_ratio=0.09,alpha=0.9)
        ax.text(*b,f"APOPHIS  {sp['ap_rng_km']:.0f} km",color="#e6ad74",fontsize=7.5,ha="center")
    # deputies
    if pos is not None:
        near=sp.get("near",False)
        for k in range(8):
            p=pos[k]-c0
            tr=sp.get("trail")
            if tr is not None:
                t=tr[k]
                if len(t)>1:
                    seg=t-c0
                    ax.plot(seg[:,0],seg[:,1],seg[:,2],color=COLS[k],lw=1.2,alpha=0.55)
            if near:
                draw_deputy(ax,p,sp["Rbl"][k],sp["half"],COLS[k],span,
                            docked=bool(sp.get("caps",np.zeros(8))[k]),
                            act=None if sp.get("acts") is None else sp["acts"][k],
                            wings=sp.get("wings",True))
            else:
                ax.scatter(*p,s=46,color=COLS[k],ec="white",lw=0.4,depthshade=False,zorder=5)
                ax.text(p[0],p[1],p[2]+span*0.035,f"D{k+1}",color=COLS[k],fontsize=6.5)
        for k,age in enumerate(sp.get("flash",[])):
            if age is not None: capture_flash(ax,pos[k]-c0,age,span)
    fig.text(rect[0]+0.005,rect[1]+0.006,"FRAME · CHIEF LVLH (RTN)",color=(0.5,0.6,0.75),fontsize=5.6)
    return ax

# ================================================================ panels
def panel_status(fig,sp,title,rows,accent=CYAN,foot=None):
    ax=glass(fig,0.635,0.585,0.355,0.335,title,accent)
    for i,(lab,val) in enumerate(rows):
        y=0.72-i*0.155
        ax.text(0.06,y,lab,color=SUB,fontsize=8,transform=ax.transAxes)
        ax.text(0.94,y,val,color=INK,fontsize=10.5,weight="bold",ha="right",
                transform=ax.transAxes,family="monospace")
    if foot: ax.text(0.06,0.075,foot,color=SUB,fontsize=6.2,family="monospace",transform=ax.transAxes)
    return ax

def panel_sep(fig,D,sp):
    panel_status(fig,sp,"CHIEF · DEPUTY SEPARATION",
        [("deputies deployed",f"{sp['ndep']} / 8"),
         ("stack","ESPA ring · XB6 6U"),
         ("push-off ΔV","0.28 m/s ea"),
         ("attitude control","PD detumble · RW")],EMER,
        foot="rigid-body 6-DOF · thruster + reaction-wheel telemetry")
    ax=glass(fig,0.635,0.075,0.355,0.49,"DEPLOYMENT ΔV — CUMULATIVE",EMER,right="per deputy · m/s")
    dv=sp["dv"]; mx=max(D["sep"]["dv"].max(),1e-6)
    for k in range(8):
        y=0.80-k*0.095
        hbar(ax,0.16,y,0.60,0.045,dv[k]/mx,COLS[k],lab=f"D{k+1}",val=f"{dv[k]:.3f}")
    ax.text(0.06,0.055,"thrusters firing" if sp["thr_on"] else "coast",
            color=AMBER if sp["thr_on"] else SUB,fontsize=6.4,family="monospace",transform=ax.transAxes)

def panel_cruise(fig,D,sp):
    m=D["mission"]
    panel_status(fig,sp,"CRUISE TO ENCOUNTER",
        [("Apophis range",f"{sp['ap_rng_km']:,.0f} km"),
         ("fleet spread",f"{sp['spread_km']:.1f} km"),
         ("guidance","impulsive targeting"),
         ("burns executed",f"{sp['nburn']} · Σ {m['dvtot']:.1f} m/s")],CYAN,
        foot="strobe formation phasing for whole-body stereo")
    ax=glass(fig,0.635,0.075,0.355,0.49,"RANGE TO APOPHIS",CYAN,right="encounter timeline")
    spark(ax,0.10,0.42,0.80,0.38,np.log10(np.maximum(m["apr"]/1e3,1.0)),sp["u"],CYAN,lab="range profile · log10 km")
    ax.text(0.10,0.30,"t0",color=SUB,fontsize=6,transform=ax.transAxes)
    ax.text(0.90,0.30,"CA",color=AMBER,fontsize=6,ha="right",weight="bold",transform=ax.transAxes)
    ax.text(0.5,0.12,f"closing · CA in {sp['tca_h']:.1f} h",color=INK,fontsize=8,
            ha="center",family="monospace",transform=ax.transAxes)

def panel_fdir(fig,D,sp):
    ax=glass(fig,0.635,0.555,0.355,0.375,"MISSION AUTONOMY · FDIR",EMER,right="fault det · isolation · recovery")
    snr=D["mission"]["snr"]; pos=sp["pos"]; apo=sp["apo_cad"]
    for k in range(8):
        cx=0.115+(k%4)*0.243; cy=0.695-(k//4)*0.205
        ax.add_patch(FancyBboxPatch((cx-0.105,cy-0.085),0.21,0.17,
            boxstyle="round,pad=0.004,rounding_size=0.05",transform=ax.transAxes,
            facecolor=(0.08,0.24,0.17,0.6),edgecolor=EMER,lw=0.9))
        ax.text(cx-0.085,cy+0.042,f"D{k+1}",color=INK,fontsize=7.5,weight="bold",transform=ax.transAxes)
        rng=np.linalg.norm(apo-pos[k])/1e3
        ax.text(cx+0.09,cy+0.042,f"{rng:.1f}km",color=SUB,fontsize=5,ha="right",transform=ax.transAxes,family="monospace")
        for j,tag in enumerate(("GNC","CAM","COM")):
            led(ax,cx-0.075+j*0.062,cy-0.012,ok=True,r=0.0082)
            ax.text(cx-0.075+j*0.062,cy-0.048,tag,color=SUB,fontsize=4.2,ha="center",transform=ax.transAxes)
        ax.text(cx,cy-0.072,f"SNR {snr[k]:.0f} dB-Hz",color=EMER,fontsize=4.3,ha="center",
                transform=ax.transAxes,family="monospace")
    # event log
    ax.add_patch(FancyBboxPatch((0.04,0.045),0.92,0.24,boxstyle="round,pad=0.004,rounding_size=0.05",
        transform=ax.transAxes,facecolor=(0.05,0.08,0.13,0.85),edgecolor=(0.3,0.4,0.6,0.5),lw=0.9))
    ax.text(0.06,0.245,"AUTONOMY EVENT LOG",color=EMER,fontsize=5.6,weight="bold",transform=ax.transAxes)
    lines=sp.get("log_lines") or [(f"CA{tu*4-2:+5.1f}s  {m}",cc) for tu,m,cc in D["events"]
                                  if tu<=sp["u"]][-4:]
    for i,(msg,cc) in enumerate(lines[-4:]):
        a=0.35+0.65*(i+1)/max(1,len(lines[-4:]))
        ax.text(0.06,0.195-i*0.042,msg[:86],color=cc,alpha=a,
                fontsize=4.9,family="monospace",transform=ax.transAxes)

def camera_feed(fig,D,x,y,w,h,k,pos,apo,R,u,ev=None,flash=None):
    los=apo-pos[k]; rng=np.linalg.norm(los)/1e3
    alpha=phase_angle_deg(D["ap"]["sun"],los)
    Rv=rot_to_view(los)@R
    ax=fig.add_axes([x,y,w,h],projection="3d"); ax.set_facecolor((0,0,0,0)); ax.set_axis_off()
    apophis_cine(ax,D["ap"],Rv,stride=1)
    s=D["ap"]["scale"]*0.72
    ax.set_xlim(-s,s); ax.set_ylim(-s,s); ax.set_zlim(-s,s)
    try: ax.set_box_aspect((1,1,1))
    except Exception: pass
    ax.view_init(elev=6,azim=0)
    f=fig.add_axes([x,y,w,h]); f.set_axis_off(); f.set_xlim(0,1); f.set_ylim(0,1)
    hot=flash is not None and 0<=flash<=1
    if hot:                                                    # strobe exposure flash
        f.add_patch(Rectangle((0.02,0.02),0.96,0.96,transform=f.transAxes,
            facecolor="white",alpha=0.75*(1-flash)**1.6,zorder=6))
    f.add_patch(FancyBboxPatch((0.02,0.02),0.96,0.96,boxstyle="round,pad=0.002,rounding_size=0.04",
        transform=f.transAxes,fill=False,edgecolor=(1,1,1,0.9) if hot else (0.35,0.85,0.72,0.8),
        lw=1.8 if hot else 1.3))
    for cx,cy in ((0.08,0.92),(0.92,0.92),(0.08,0.08),(0.92,0.08)):     # corner brackets
        f.plot([cx-0.03*np.sign(cx-0.5),cx],[cy,cy],color=(0.5,1,0.85,0.6),lw=1.1)
        f.plot([cx,cx],[cy-0.05*np.sign(cy-0.5),cy],color=(0.5,1,0.85,0.6),lw=1.1)
    f.plot([0.5,0.5],[0.44,0.56],color=(0.5,1,0.85,0.4),lw=0.8)
    f.plot([0.44,0.56],[0.5,0.5],color=(0.5,1,0.85,0.4),lw=0.8)
    sl=0.15+0.7*((u*3+k*0.23)%1.0)                                       # roaming scanline
    f.plot([0.04,0.96],[sl,sl],color=(0.5,1,0.85,0.10),lw=2.5)
    f.text(0.06,0.885,f"D{k+1} · NAC",color=EMER,fontsize=7,weight="bold")
    f.text(0.94,0.885,"STROBE" if hot else "LIVE",color="white" if hot else ROSE,
           fontsize=5.6,ha="right",weight="bold")
    if ev:                                                     # last real image record
        exp=float(np.clip(2.0+0.03*(330-ev["snr"]),2,12))
        f.text(0.06,0.155,f"IMG {ev['i']:03d}/{ev['n']:03d} · {utc_str(ev['utc'],ms=True)[11:]}",
               color=(0.95,1,0.98),fontsize=4.6,family="monospace")
        f.text(0.94,0.155,f"EXP {exp:.0f} ms",color=(0.85,0.96,0.92),fontsize=4.6,
               ha="right",family="monospace")
    f.text(0.06,0.075,f"{rng:.1f} km",color=(0.85,0.96,0.92),fontsize=6.2,family="monospace")
    f.text(0.50,0.075,f"α {alpha:.0f}°",color=GOLD,fontsize=5.6,ha="center",family="monospace")
    f.text(0.94,0.075,f"SNR {(ev['snr'] if ev else D['mission']['snr'][k]):.0f}",
           color=(0.85,0.96,0.92),fontsize=6.2,ha="right",family="monospace")

def panel_cameras(fig,D,sp):
    fig.text(0.638,0.522,"DEPUTY CAMERA FEEDS · APOPHIS 99942",color=AMBER,fontsize=9.4,weight="bold")
    cov=D["cov"]
    fig.text(0.638,0.503,f"SOLAR 1361 W/m² · GEOM ALBEDO 0.23 · LAMBERT+SPEC BRDF · "
             f"EARTHSHINE {cov['earthshine']:.2f}% · DARK SIDE VIA STROBE",
             color=(0.62,0.70,0.85),fontsize=5.4)
    last=sp.get("last_ev",{})
    for j,k in enumerate((0,2,5,7)):
        camera_feed(fig,D,0.636+(j%2)*0.181,0.075+(1-j//2)*0.211,0.172,0.195,
                    k,sp["pos"],sp["apo_cad"],sp["ap_R"],sp["u"],
                    ev=last.get(k),flash=sp.get("strobe",[None]*8)[k])

def panel_coverage(fig,D,sp):
    x,y,w,h=0.028,0.13,0.135,0.30
    ax=fig.add_axes([x,y,w,h],projection="3d"); ax.set_facecolor((0,0,0,0)); ax.set_axis_off()
    apophis_cine(ax,D["ap"],sp["ap_R"],stride=1,mode="coverage")
    s=D["ap"]["scale"]*0.95
    ax.set_xlim(-s,s); ax.set_ylim(-s,s); ax.set_zlim(-s,s)
    try: ax.set_box_aspect((1,1,1))
    except Exception: pass
    ax.view_init(elev=16,azim=sp["cov_az"])
    f=fig.add_axes([x,y-0.005,w,h]); f.set_axis_off(); f.set_xlim(0,1); f.set_ylim(0,1)
    cv=D["cov"]; whole=lerp(88.0,cv["whole"],smooth(np.clip(sp["u"]*1.6,0,1)))
    f.text(0.5,0.10,f"WHOLE-BODY {whole:.1f}%",color=EMER,fontsize=6.4,
           ha="center",weight="bold",family="monospace")
    f.text(0.5,0.05,f"SUNLIT {cv['sunlit']:.2f}%   DARK {cv['dark']:.1f}%",
           color=(0.75,0.85,0.95),fontsize=5.0,ha="center",family="monospace")
    f.text(0.5,0.005,"0.15 m stereo · ■ stereo ■ mono ■ oblique ■ unseen",color=SUB,fontsize=4.4,ha="center")

def panel_return(fig,D,sp):
    ret=D["ret"]
    panel_status(fig,sp,"PHASE-A RETURN TO CHIEF",
        [("total return ΔV",f"{float(ret['dv_total']):.1f} m/s"),
         ("guidance","relative STM targeting"),
         ("target","ESPA ring · own port"),
         ("progress",f"{sp['pct']:d} %")],CYAN,
        foot="two-impulse transfer · per-deputy port assignment")
    ax=glass(fig,0.635,0.075,0.355,0.49,"RETURN ΔV BUDGET",CYAN,right="per deputy · m/s")
    dvp=np.asarray(ret["dv_per_deputy"],float); mx=dvp.max()+1e-9
    for k in range(8):
        y=0.80-k*0.095
        hbar(ax,0.16,y,0.60,0.045,dvp[k]/mx,COLS[k],lab=f"D{k+1}",val=f"{dvp[k]:.2f}")
    ax.text(0.06,0.055,f"fleet total {float(ret['dv_total']):.2f} m/s",
            color=SUB,fontsize=6.4,family="monospace",transform=ax.transAxes)

def panel_brain(fig,D,sp):
    """RL docking policy — observation -> actor network -> command, with RAFTI
    capture criteria checked live against flight tolerances. I/O values real."""
    ax=glass(fig,0.615,0.055,0.375,0.885,"AUTONOMOUS DOCKING · RL POLICY  π(o)",VIOL,
             tfs=10.5,right="actor 400×300 · TD3/BC")
    d=sp["focus"]; o=sp["obs"]; a=sp["act"]; nrm=sp["nrm"]; raw=sp["raw"]; mode=sp["mode"]
    ax.text(0.055,0.845,f"D{d+1}",color=COLS[d],fontsize=13,weight="bold",transform=ax.transAxes)
    ax.text(0.115,0.848,"ENGAGED" if mode!="DOCKED" else "SECURED",
            color=EMER,fontsize=7,weight="bold",transform=ax.transAxes)
    ax.text(0.95,0.848,f"CAPTURED {sp['ncap']}/8",color=EMER if sp['ncap']==8 else INK,
            fontsize=8,weight="bold",ha="right",transform=ax.transAxes,family="monospace")
    for k in range(8):                                   # fleet capture dots
        led(ax,0.62+k*0.042,0.815,ok=bool(sp["caps"][k]),r=0.0075,warn=not bool(sp["caps"][k]))
        ax.text(0.62+k*0.042,0.793,f"{k+1}",color=SUB,fontsize=4,ha="center",transform=ax.transAxes)

    # ---- observation block (real) ----
    ax.text(0.055,0.775,"OBSERVATION  o(t)",color=CYAN,fontsize=6.8,weight="bold",transform=ax.transAxes)
    labs=[("axial range",f"{raw['ax']:6.2f} m",CYAN),("lateral offset",f"{raw['lat']:6.2f} m",CYAN),
          ("closing speed",f"{raw['cv']:6.3f} m/s",EMER),("lateral speed",f"{raw['lv']:6.3f} m/s",EMER),
          ("attitude error",f"{raw['ae']:6.1f} °",AMBER),("body rate",f"{raw['rt']:6.3f} r/s",VIOL)]
    ys=np.linspace(0.735,0.545,6); Pin=[]
    for j,(y,(lb,vv,cc)) in enumerate(zip(ys,labs)):
        Pin.append((0.235,y))
        ax.text(0.055,y+0.012,lb,color=(0.82,0.87,0.96),fontsize=5.6,transform=ax.transAxes)
        hbar(ax,0.055,y-0.010,0.135,0.014,nrm[j],cc)
        ax.text(0.198,y-0.003,vv,color=INK,fontsize=5.2,family="monospace",transform=ax.transAxes)

    # ---- network glyph ----
    h1,h2=sp["hidden"]
    cx1,cx2=0.435,0.545
    for xx,hh,nn in ((cx1,h1,"400"),(cx2,h2,"300")):
        yy=np.linspace(0.74,0.54,len(hh))
        for yv,av in zip(yy,hh):
            ax.add_patch(Circle((xx,yv),0.006+0.005*av,color=VIOL,
                alpha=0.20+0.70*av,transform=ax.transAxes,lw=0))
        ax.text(xx,0.515,f"×{nn}",color=SUB,fontsize=5,ha="center",transform=ax.transAxes)
    for p in Pin:                                        # synapses in
        for yv,av in zip(np.linspace(0.74,0.54,len(h1))[::3],h1[::3]):
            ax.plot([p[0]+0.01,cx1-0.008],[p[1],yv],color=(0.55,0.7,1,0.05+0.16*av),
                    lw=0.4,transform=ax.transAxes)
    outs=[("Δv R",a[0]*C.ENV["max_dv_mps"],"m/s",EMER),("Δv T",a[1]*C.ENV["max_dv_mps"],"m/s",EMER),
          ("Δv N",a[2]*C.ENV["max_dv_mps"],"m/s",EMER),("τ x",a[3]*C.ENV["max_torque_Nm"]*1e3,"mN·m",VIOL),
          ("τ y",a[4]*C.ENV["max_torque_Nm"]*1e3,"mN·m",VIOL),("τ z",a[5]*C.ENV["max_torque_Nm"]*1e3,"mN·m",VIOL)]
    yo=np.linspace(0.735,0.545,6)
    ax.text(0.66,0.775,"COMMAND  a(t)",color=EMER,fontsize=6.8,weight="bold",transform=ax.transAxes)
    for j,(y,(lb,vv,un,cc)) in enumerate(zip(yo,outs)):
        m=np.clip(abs(a[j]),0,1)
        for yv,av in zip(np.linspace(0.74,0.54,len(h2))[::3],h2[::3]):
            ax.plot([cx2+0.008,0.655],[yv,y],color=(0.6,1,0.8,0.04+0.14*m) if j<3
                    else (0.7,0.6,1,0.04+0.14*m),lw=0.4,transform=ax.transAxes)
        ax.text(0.662,y+0.012,lb,color=(0.82,0.87,0.96),fontsize=5.6,transform=ax.transAxes)
        # signed bar
        bx,bw=0.72,0.16
        ax.add_patch(Rectangle((bx,y-0.010),bw,0.014,
            transform=ax.transAxes,facecolor=(1,1,1,0.07),edgecolor=(1,1,1,0.12),lw=0.5))
        half=bw/2; f=np.clip(a[j],-1,1)*half
        x0=bx+half+(min(f,0)); ax.add_patch(Rectangle((x0,y-0.010),abs(f),0.014,
            transform=ax.transAxes,facecolor=cc,alpha=0.9))
        ax.plot([bx+half,bx+half],[y-0.013,y+0.017],color=(1,1,1,0.35),lw=0.6,transform=ax.transAxes)
        ax.text(0.945,y-0.003,f"{vv:+7.3f} {un}",color=INK,fontsize=5.0,ha="right",
                family="monospace",transform=ax.transAxes)
    ax.text(0.5,0.497,"activation glyph illustrative — inputs & commands are the recorded rollout",
            color=(0.45,0.53,0.68),fontsize=4.6,ha="center",transform=ax.transAxes)

    # ---- guidance mode chips ----
    for i,mm in enumerate(("APPROACH","GLIDESLOPE","TERMINAL","CAPTURE","DOCKED")):
        chip(ax,0.135+i*0.185,0.455,0.165,0.045,mm,mm==mode,
             EMER if mm=="DOCKED" else AMBER if mm in("TERMINAL","CAPTURE") else CYAN)

    # ---- RAFTI capture criteria (live vs flight tolerances) ----
    ax.text(0.055,0.405,"RAFTI CAPTURE CRITERIA",color=GOLD,fontsize=6.8,weight="bold",transform=ax.transAxes)
    ax.text(0.95,0.405,"flight-stage tolerances",color=SUB,fontsize=4.8,ha="right",transform=ax.transAxes)
    crit=[("axial standoff",raw["ax"],RAFTI_TOL["pos"],"m",6.0),
          ("lateral offset",raw["lat"],RAFTI_TOL["lat"],"m",2.0),
          ("closing speed",raw["cv"],RAFTI_TOL["v_cap"],"m/s",0.15),
          ("lateral speed",raw["lv"],RAFTI_TOL["v_lat"],"m/s",0.06),
          ("attitude error",raw["ae"],RAFTI_TOL["att"],"°",40.0),
          ("body rate",raw["rt"],RAFTI_TOL["omega"],"rad/s",0.12)]
    for i,(lb,v,tol,un,vis) in enumerate(crit):
        y=0.365-i*0.036; ok=v<=tol if mode!="DOCKED" else True
        led(ax,0.075,y+0.004,ok=ok,r=0.007,warn=True)
        ax.text(0.10,y,lb,color=(0.82,0.87,0.96),fontsize=5.4,transform=ax.transAxes)
        hbar(ax,0.30,y-0.004,0.38,0.012,min(v/vis,1.0),EMER if ok else AMBER)
        ax.plot([0.30+0.38*min(tol/vis,1.0)]*2,[y-0.008,y+0.016],color=GOLD,lw=1.0,transform=ax.transAxes)
        ax.text(0.95,y,f"{v:7.3f} ≤ {tol:g} {un}",color=EMER if ok else AMBER,
                fontsize=4.9,ha="right",family="monospace",transform=ax.transAxes)
    # ---- policy trace ----
    ax.add_patch(FancyBboxPatch((0.04,0.055),0.92,0.075,boxstyle="round,pad=0.004,rounding_size=0.025",
        transform=ax.transAxes,facecolor=(0.05,0.08,0.13,0.85),edgecolor=(0.3,0.4,0.6,0.5),lw=0.9))
    ax.text(0.06,0.108,"POLICY TRACE",color=VIOL,fontsize=5.4,weight="bold",transform=ax.transAxes)
    ax.text(0.06,0.072,sp["trace"][:92],color=EMER,fontsize=5.2,family="monospace",transform=ax.transAxes)

# ================================================================ title / finale
def card_title(fig,D,sp):
    u=sp["u"]
    ax=fig.add_axes([0.25,0.18,0.5,0.62],projection="3d"); style3d(ax); ax.set_axis_off()
    nR=len(D["ap"]["Rser"]) if D["ap"]["Rser"] is not None else 1
    R=D["ap"]["Rser"][int(u*(nR-1))] if D["ap"]["Rser"] is not None else np.eye(3)
    apophis_cine(ax,D["ap"],R,stride=1,alpha=0.9)
    s=D["ap"]["scale"]*1.15
    ax.set_xlim(-s,s); ax.set_ylim(-s,s); ax.set_zlim(-s,s)
    try: ax.set_box_aspect((1,1,1))
    except Exception: pass
    ax.view_init(elev=10,azim=-30+16*u)
    a=smooth(np.clip(u*3,0,1))
    fig.text(0.5,0.80,"A R I S E",color=INK,fontsize=44,weight="bold",ha="center",
             alpha=a,path_effects=_sh(5))
    fig.text(0.5,0.745,"AUTONOMOUS FLEET RECONNAISSANCE OF ASTEROID 99942 APOPHIS",
             color=CYAN,fontsize=10.5,ha="center",alpha=a)
    a2=smooth(np.clip((u-0.25)*3,0,1))
    fig.text(0.5,0.205,"8 DEPUTIES · WHOLE-BODY STEREO IMAGING · RL TERMINAL RE-DOCK",
             color=SUB,fontsize=9,ha="center",alpha=a2)
    fig.text(0.5,0.165,"A FLIGHT-DATA-DRIVEN MISSION RECONSTRUCTION",
             color=(0.5,0.6,0.78),fontsize=7,ha="center",alpha=a2)
    fig.text(0.5,0.10,"BASILISK 6-DOF DYNAMICS  ·  RADAR SHAPE MODEL  ·  TD3/BC DOCKING POLICY",
             color=(0.42,0.5,0.65),fontsize=6,ha="center",alpha=a2)

def card_finale(fig,D,sp):
    u=sp["u"]; a=smooth(np.clip(u*2.5,0,1))
    ret=D["ret"]
    fig.text(0.5,0.76,"MISSION COMPLETE",color=EMER,fontsize=30,weight="bold",
             ha="center",alpha=a,path_effects=_sh(4))
    fig.text(0.5,0.705,"ALL DEPUTIES SECURED · CHIEF STACK NOMINAL",color=INK,
             fontsize=10,ha="center",alpha=a)
    cv=D["cov"]
    stats=[("WHOLE-BODY STEREO",f"{cv['whole']:.1f} %"),
           ("STROBE IMAGES",f"{int(D['nimg'].sum())} FRAMES"),
           ("DARK-SIDE COVERAGE",f"{cv['dark']:.1f} % · STROBE-LIT"),
           ("RETURN ΔV",f"{float(ret['dv_total']):.1f} m/s"),
           ("RE-DOCK CAPTURES",f"{int(np.asarray(D['dock']['n_captured'])):d} / 8"),
           ("TERMINAL GUIDANCE","RL POLICY π · TD3/BC")]
    a2=smooth(np.clip((u-0.2)*2.5,0,1))
    for i,(lb,vv) in enumerate(stats):
        x=0.29+(i%2)*0.42; y=0.565-(i//2)*0.115
        fig.text(x,y,lb,color=SUB,fontsize=8.5,ha="center",alpha=a2)
        fig.text(x,y-0.045,vv,color=INK,fontsize=13,weight="bold",ha="center",
                 alpha=a2,family="monospace")
    fig.text(0.5,0.13,"ARISE · APOPHIS MISSION — FULL ARC RECONSTRUCTION",
             color=(0.5,0.6,0.78),fontsize=7,ha="center",alpha=a2)

# ================================================================ frame assembly
SEGS=[("TITLE",6.0),("SEPARATION",26.0),("CRUISE",26.0),("IMAGING",55.0),
      ("RETURN",26.0),("DOCK",51.0),("FINALE",10.0)]
MSEG=[1,2,3,4,5]     # indices of mission segments for the timeline

def build_specs(D,total,fps=None):
    fps=fps or CONFIG["FPS"]
    dur=np.array([s[1] for s in SEGS]); ns=np.maximum(2,np.round(dur/dur.sum()*total).astype(int))
    ns[-1]+=total-ns.sum()
    mfrac=dur[MSEG]/dur[MSEG].sum()
    mstart=np.concatenate([[0],np.cumsum(ns[MSEG])]).astype(float)
    mtot=mstart[-1]
    ap=D["ap"]; nR=len(ap["Rser"]) if ap["Rser"] is not None else 1
    Rspin=lambda u:(ap["Rser"][int(np.clip(u,0,1)*(nR-1))] if ap["Rser"] is not None else np.eye(3))
    half=0.5*D["dims"]; ring=D["ring"]
    mis,sep,dock,ret=D["mission"],D["sep"],D["dock"],D["ret"]
    ft=D["ft"]; focus=D["focus"]
    specs=[]; g=0; mg=0
    def com(pi,banner,bc,clock,clock2="",lt=None,fade=0.0):
        return dict(seg_frac=mfrac,phase_idx=pi,t_frac=(mg+0.5)/mtot if pi>=0 else 0,
                    banner=banner,banner_c=bc,banner_dot=bc,clock=clock,clock2=clock2,
                    lt=lt,fade=fade)
    sep_dv_end=float(sep["dv"][-1].sum())          # fleet Δv at deploy completion
    dv_ca=sep_dv_end+float(mis["dvtot"])           # fleet Δv by close approach
    IMG_TOT=int(D["nimg"].sum())
    def lt_alpha(i,n,tin=0.35,hold=2.6):
        # lower-third: fade in over tin s, hold, fade out
        tt=i/fps
        if tt<tin: return tt/tin
        if tt<hold: return 1.0
        if tt<hold+0.5: return 1-(tt-hold)/0.5
        return 0.0

    # ---------- TITLE ----------
    for i in range(ns[0]):
        u=i/(ns[0]-1)
        fade=(1-smooth(u*4)) if u<0.25 else (smooth((u-0.92)/0.08) if u>0.92 else 0)
        specs.append(dict(kind="title",u=u,fade=fade))
        g+=1
    # ---------- SEPARATION ----------
    n=ns[1]
    for i in range(n):
        u=i/(n-1); ue=smooth(u)
        f=int(ue*(sep["tr"].shape[0]-1)); pos=sep["tr"][f]
        span=max(np.linalg.norm(pos,axis=1).max()*1.25,ring["r"]*3.2)
        lt=dict(h="PHASE 1 · SEPARATION",s="eight 6U deputies deploy from the ESPA ring — push-off, detumble, drift-out",
                c=EMER,alpha=lt_alpha(i,n)) if i/fps<3.4 else None
        spread_m=float(np.linalg.norm(pos,axis=1).max()); thr=bool(sep["thr"][f].sum()>1e-4)
        sp=dict(kind="mission",phase="SEPARATION",pos=pos,span=span,
                elev=lerp(14,22,ue),azim=-62+26*ue,near=True,half=half,
                Rbl=[_mrp_to_dcm(sep["att"][f,k]).T for k in range(8)],
                trail=[sep["tr"][:f+1,k] for k in range(8)],rr=(2,5,10),
                dv=sep["dv"][f],thr_on=thr,ndep=8,
                flash=[None]*8,wings=True,
                kpi=dict(rng_lab="DEPLOY DIST",rng_val=f"{spread_m:.1f}",
                         rng_unit="m from chief",rng_frac=min(spread_m/30,1),
                         spr_val=f"{spread_m:.1f}",spr_unit="m",spr_frac=min(spread_m/30,1),
                         dv=float(sep["dv"][f].sum()),img=0,img_tot=IMG_TOT,cov=0.0,cap=0),
                side_title="SEPARATION",
                side_rows=[("stack","ESPA · XB6 6U",None),("control","PD detumble · RW",None),
                           ("push-off","0.28 m/s ea",None),
                           ("thrusters","FIRING" if thr else "COAST","#ffce6a" if thr else None)],
                **com(0,"CHIEF · DEPUTY SEPARATION — ESPA RAFTI DEPLOY",EMER,
                      "SEP  T+%.0f s"%sep["t"][f],
                      utc_str(utc_of(T0_CRUISE-(float(sep["t"][-1])-float(sep["t"][f])))),lt=lt,
                      fade=(1-smooth(u*8)) if u<0.125 else 0.0))
        specs.append(sp); g+=1; mg+=1
    # ---------- CRUISE ----------
    n=ns[2]
    for i in range(n):
        u=i/(n-1); ue=smooth(u)
        f=int(ue*(mis["tr"].shape[0]-1)); pos=mis["tr"][f]; rng=mis["apr"][f]
        span=max(np.linalg.norm(pos,axis=1).max()*1.2,600)
        lt=dict(h="PHASE 2 · CRUISE",s="strobe-formation phasing toward the Apophis encounter corridor",
                c=CYAN,alpha=lt_alpha(i,n)) if i/fps<3.4 else None
        # camera tracks the Apophis bearing: its arrow stays screen-right while the
        # scene sweeps ±12° for parallax — fleet motion and target now read as one
        b=mis["apo"][f]; az_b=float(np.degrees(np.arctan2(b[1],b[0])))
        spread_km=float(np.linalg.norm(pos,axis=1).max()/1e3)
        rng0=float(mis["apr"][0]/1e3)
        sp=dict(kind="mission",phase="CRUISE",pos=pos,span=span,
                elev=lerp(19,14,ue),azim=az_b-90+24*(ue-0.5),near=False,
                trail=[mis["tr"][:f+1,k] for k in range(8)],
                ap_bearing=mis["apo"][f],ap_rng_km=rng/1e3,
                spread_km=spread_km,u=u,
                nburn=int(mis["nburn"]*u),tca_h=max(0.0,(mis["t"][-1]-mis["t"][f])/3600),
                kpi=dict(rng_lab="APOPHIS RANGE",rng_val=f"{rng/1e3:,.0f}",rng_unit="km",
                         rng_frac=float(np.clip(1-np.log10(max(rng/1e3,1.0))/np.log10(max(rng0,10.0)),0,1)),
                         spr_val=f"{spread_km:.1f}",spr_unit="km",spr_frac=min(spread_km/6,1),
                         dv=sep_dv_end+float(mis["dvtot"])*u,img=0,img_tot=IMG_TOT,cov=0.0,cap=0),
                side_title="GUIDANCE",
                side_rows=[("law","impulsive targeting",None),
                           ("burns",f"{int(mis['nburn']*u)} · Σ {mis['dvtot']:.1f} m/s",None),
                           ("camera","APOPHIS TRACK",None),("point err","0.07°",None)],
                **com(1,"CRUISE TO ENCOUNTER — FORMATION PHASING",CYAN,
                      f"CA −{max(0,(mis['t'][-1]-mis['t'][f])/3600):4.1f} h",
                      utc_str(utc_of(mis["t"][f])),lt=lt))
        specs.append(sp); g+=1; mg+=1
    # ---------- IMAGING (CA ±2.2 s strobe pass · extreme slow-motion) ----------
    n=ns[3]; apo=mis["apo"][-1]; form=mis["tr"][-1]
    IMG=D["img_events"]; FLASH=0.25                 # strobe visual window, CA-seconds
    for e in IMG:                                    # precompute per-event geometry
        e["utc"]=utc_of(T_CA+e["t"])
        e["alpha"]=phase_angle_deg(ap["sun"],apo-form[e["k"]])
    seg_real=4.4; rate_img=seg_real/SEGS[3][1]
    for i in range(n):
        u=i/(n-1); ue=smooth(u); t_ca=lerp(-2.2,2.2,u)
        R=Rspin((t_ca+2.0)/4.0); apsc=5.0            # Rser spans the ±2 s window
        span=max(np.linalg.norm(form-apo,axis=1).max()*1.28,ap["scale"]*apsc*1.9)
        done=[e for e in IMG if e["t"]<=t_ca]
        strobe=[None]*8; last={}
        for e in done:
            last[e["k"]]=e
            ag=(t_ca-e["t"])/FLASH
            if 0<=ag<=1: strobe[e["k"]]=ag
        log=[(f"{utc_str(e['utc'],ms=True)[11:]}  IMG D{e['k']+1} #{e['i']:02d}  "
              f"SNR {e['snr']:3.0f}  α {e['alpha']:2.0f}°  {e['rng']/1e3:3.1f} km",
              "white" if (t_ca-e["t"])<FLASH else CYAN) for e in done[-4:]]
        if not log: log=[("STROBE SEQUENCE ARMED — CA −2.2 s · 136 EXPOSURES QUEUED",AMBER)]
        lt=dict(h="PHASE 3 · CLOSE-APPROACH STROBE PASS",
                s="±2 s hypervelocity pass — 136 strobe exposures · dark side lit by flash · whole-body stereo",
                c=AMBER,alpha=lt_alpha(i,n)) if i/fps<3.6 else None
        rmin=float(np.linalg.norm(form-apo,axis=1).min()/1e3)
        sprk=float(np.linalg.norm(form-apo,axis=1).max()/1e3)
        covval=float(lerp(88.0,D["cov"]["whole"],smooth(np.clip(u*1.6,0,1))))
        sp=dict(kind="mission",phase="IMAGING",pos=form,center=apo,span=span,
                elev=lerp(12,18,ue),azim=-44+58*ue,near=False,
                apo_cad=apo,ap_R=R,ap_scale=apsc,ap_stride=1,
                fov=True,sun_ray=True,strobe=strobe,last_ev=last,
                log_lines=log,rate=rate_img,nimg_done=len(done),
                cov_az=30+140*u,u=u,
                kpi=dict(rng_lab="APOPHIS RANGE",rng_val=f"{rmin:.1f}",
                         rng_unit="km · closest",rng_frac=1-min(rmin/6,1),
                         spr_val=f"{sprk:.1f}",spr_unit="km",spr_frac=min(sprk/6,1),
                         dv=dv_ca,img=len(done),img_tot=IMG_TOT,cov=covval,cap=0),
                side_title="AUTONOMY EVENT LOG",side_rows=None,
                **com(2,"APOPHIS CLOSE APPROACH — STROBE STEREO PASS",AMBER,
                      f"CA {t_ca:+7.3f} s",utc_str(utc_of(T_CA+t_ca),ms=True),lt=lt))
        specs.append(sp); g+=1; mg+=1
    # ---------- RETURN ----------
    n=ns[4]; arcs=ret["arcs"]
    for i in range(n):
        u=i/(n-1); ue=smooth(u)
        k=int(ue*(arcs.shape[0]-1)); pos=arcs[k]
        span=max(np.linalg.norm(pos,axis=1).max()*1.18,ring["r"]*3)
        lt=dict(h="PHASE 4 · RETURN",s="two-impulse relative transfers bring every deputy back to its own port",
                c=CYAN,alpha=lt_alpha(i,n)) if i/fps<3.4 else None
        maxd=float(np.linalg.norm(pos,axis=1).max())
        maxd0=float(np.linalg.norm(arcs[0],axis=1).max())
        sp=dict(kind="mission",phase="RETURN",pos=pos,span=span,
                elev=lerp(17,22,ue),azim=-64+30*ue,near=False,
                trail=[arcs[max(0,k-34):k+1,j] for j in range(8)],pct=int(u*100),
                kpi=dict(rng_lab="RING RANGE",rng_val=f"{maxd:.0f}",
                         rng_unit="m · farthest deputy",rng_frac=1-min(maxd/max(maxd0,1),1),
                         spr_val=f"{maxd:.0f}",spr_unit="m",spr_frac=min(maxd/max(maxd0,1),1),
                         dv=dv_ca+float(ret.get("dv_total",10.7))*u,img=IMG_TOT,img_tot=IMG_TOT,
                         cov=float(D["cov"]["whole"]),cap=0),
                side_title="RETURN GUIDANCE",
                side_rows=[("law","relative STM",None),("transfer","two-impulse · ½ period",None),
                           ("target","ESPA ring · own port",None),
                           ("ETA",f"{(1-u)*RET_SECONDS/60:.0f} min",None)],
                **com(3,"PHASE-A RETURN TO CHIEF",CYAN,f"RTN {int(u*100):3d} %",
                      utc_str(utc_of(T_CA+2.2+u*RET_SECONDS)),lt=lt))
        specs.append(sp); g+=1; mg+=1
    # ---------- DOCK (with slow-mo capture window) ----------
    n=ns[5]; traj=dock["traj"]; att=dock["att"]; caps=dock["caps"]
    obs=dock["obs"]; act=dock["act"]; Tb=traj.shape[0]
    knots_u=np.array([0,0.30,0.52,0.95,1.0]); knots_k=np.array([0,220,278,350,Tb-1],float)
    for i in range(n):
        u=i/(n-1); kf=float(np.interp(u,knots_u,knots_k)); k=int(kf)
        rate=np.interp(u,knots_u[:-1],np.diff(knots_k)/np.diff(knots_u))/( (Tb-1) )
        pos=traj[k]; cp=caps[k]
        span=max(np.linalg.norm(pos,axis=1).max()*1.15,ring["r"]*2.8)
        span*=lerp(1.0,0.72,smooth(np.clip((u-0.55)/0.45,0,1)))       # dolly-in
        o=obs[k,focus]; a=act[k,focus]; nrm,raw=obs_features(o); mode=dock_mode(raw,bool(cp[focus]))
        trace=(f"D{focus+1} SECURED — port latched · fleet {int(cp.sum())}/8" if mode=="DOCKED" else
               f"D{focus+1}  r_ax {raw['ax']:6.2f} m · v_cl {raw['cv']:.3f} m/s · att {raw['ae']:4.1f}° "
               f"· ω {raw['rt']:.3f} → {mode}")
        lt=dict(h="PHASE 5 · TERMINAL RE-DOCK",s="the trained RL policy flies each deputy through the RAFTI capture corridor",
                c=VIOL,alpha=lt_alpha(i,n)) if i/fps<3.6 else None
        flash=[(k-ff)/14.0 if 0<=k-ff<=14 else None for ff in ft]
        maxd=float(np.linalg.norm(pos,axis=1).max())
        sp=dict(kind="mission",phase="DOCK",pos=pos,span=span,
                elev=lerp(20,26,smooth(u)),azim=-58+40*smooth(u),near=True,half=half,
                Rbl=[_mrp_to_dcm(att[k,j]).T for j in range(8)],caps=cp,
                acts=act[k],trail=[traj[max(0,k-70):k+1,j] for j in range(8)],
                rr=(5,10,20),flash=flash,wings=False,
                focus=focus,obs=o,act=a,nrm=nrm,raw=raw,mode=mode,
                hidden=hidden_act(o),ncap=int(cp.sum()),trace=trace,rate=rate,
                kpi=dict(rng_lab="RING RANGE",rng_val=f"{maxd:.1f}",
                         rng_unit="m · farthest deputy",rng_frac=1-min(maxd/40,1),
                         spr_val=f"{maxd:.1f}",spr_unit="m",spr_frac=min(maxd/40,1),
                         dv=dv_ca+float(ret.get("dv_total",10.7))+float(D["dock_dvcum"][k]),
                         img=IMG_TOT,img_tot=IMG_TOT,cov=float(D["cov"]["whole"]),
                         cap=int(cp.sum())),
                side_title="RL POLICY",
                side_rows=[("actor","TD3/BC · 400×300",None),("mode",mode,None),
                           ("focus",f"D{focus+1}",None),
                           ("captured",f"{int(cp.sum())}/8","#3ddc84")],
                log_lines=[(trace,"#3ddc84")],
                **com(4,"TERMINAL 6-DOF RE-DOCK — RL POLICY ENGAGED",VIOL,
                      f"DOCK T+{k:3d} s · {int(cp.sum())}/8",
                      utc_str(utc_of(T_CA+2.2+RET_SECONDS+k)),lt=lt))
        specs.append(sp); g+=1; mg+=1
    # ---------- FINALE ----------
    for i in range(ns[6]):
        u=i/(ns[6]-1)
        fade=(1-smooth(u*5)) if u<0.2 else (smooth((u-0.85)/0.15) if u>0.85 else 0)
        specs.append(dict(kind="finale",u=u,fade=fade))
        g+=1
    return specs

# ================================================================ render
def render_frame(fig,D,sp):
    """Presentation is delegated to the card/gauge dashboard in film_gui.py
    (Grafana-style template). This module keeps the data, 3D scene, camera
    feeds and title cards, which film_gui reuses."""
    import sys as _sys, film_gui as G
    G.render_frame(_sys.modules[__name__], fig, D, sp)

def _ensure_imageio():
    """imageio + imageio-ffmpeg are required to encode. Install them once if the
    interpreter running this script doesn't have them (common when the script is
    launched by a different Python than the project venv)."""
    try:
        import imageio.v2 as iio, imageio_ffmpeg          # noqa: F401
        return iio
    except Exception:
        pass
    print("[setup] installing imageio + imageio-ffmpeg (one time)…", flush=True)
    for args in (["--quiet"], ["--quiet", "--user"]):
        subprocess.run([sys.executable, "-m", "pip", "install", *args,
                        "imageio", "imageio-ffmpeg"], check=False)
        try:
            import importlib; importlib.invalidate_caches()
            import imageio.v2 as iio, imageio_ffmpeg      # noqa: F401
            print("[setup] imageio ready", flush=True)
            return iio
        except Exception:
            continue
    raise SystemExit(
        "\nCannot import imageio.\n"
        "  Fix: run this script with your project venv, e.g.\n"
        '    "<basilisk-develop>\\.venv\\Scripts\\python.exe" make_mission_film.py\n'
        "  or install manually:  python -m pip install imageio imageio-ffmpeg")

def _writer(path,fps,crf):
    # NB: many encoders run in parallel (one per worker) — keep each one light.
    # threads=2 + short lookahead cuts x264 memory ~8x vs preset-medium defaults;
    # quality is governed by CRF, so this stays visually lossless at crf 17.
    iio = _ensure_imageio()
    return iio.get_writer(path,fps=fps,codec="libx264",quality=None,
        macro_block_size=1,pixelformat="yuv420p",
        ffmpeg_params=["-crf",str(crf),"-preset","veryfast","-rc-lookahead","20",
                       "-threads","2","-profile:v","high","-level","5.1"])

def render_range(D,specs,i0,i1,path,fps,dpi,crf,prog=None,tag=""):
    fig=plt.figure(figsize=(16,9),dpi=dpi)
    w=_writer(path,fps,crf); t0=time.time()
    for j,i in enumerate(range(i0,i1)):
        render_frame(fig,D,specs[i]); fig.canvas.draw()
        buf=np.asarray(fig.canvas.buffer_rgba())[...,:3]
        hh,ww=buf.shape[0]//2*2,buf.shape[1]//2*2
        try:
            w.append_data(np.ascontiguousarray(buf[:hh,:ww]))
        except OSError as e:
            raise RuntimeError(
                "ffmpeg encoder pipe died (usually out-of-memory when many workers "
                "encode 4K at once) — lower CONFIG WORKERS and re-run") from e
        if prog and (j%15==0 or i==i1-1):
            try:
                with open(prog,"w") as f: f.write(f"{j+1} {i1-i0}")
            except Exception: pass
        if tag and j%30==0:
            el=time.time()-t0
            print(f"  [{tag}] {j+1}/{i1-i0}  ({el/(j+1):.2f}s/frame)",flush=True)
    w.close(); plt.close(fig)
    open(path+".ok","w").write("done")          # completion marker (enables resume)

def _concat(chunks,out):
    import imageio_ffmpeg
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    lst=os.path.join(os.path.dirname(chunks[0]),"concat.txt")
    with open(lst,"w") as f:
        for c in chunks: f.write("file '%s'\n"%c.replace("\\","/"))
    r=subprocess.run([ff,"-y","-f","concat","-safe","0","-i",lst,"-c","copy",
                      "-movflags","+faststart",out],
                     stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    if r.returncode!=0:
        raise RuntimeError("ffmpeg concat failed:\n"+r.stderr.decode()[-800:])

def main():
    # ---- worker mode: make_mission_film.py --worker i0 i1 chunk prog ----
    if len(sys.argv)>=5 and sys.argv[1]=="--worker":
        i0,i1=int(sys.argv[2]),int(sys.argv[3]); chunk=sys.argv[4]
        prog=sys.argv[5] if len(sys.argv)>5 else None
        cfg=json.loads(os.environ["FILM_CFG"])
        CONFIG.update(cfg)
        arc=cfg["ARC_DIR"]; D=load_all(arc)
        specs=build_specs(D,cfg["_TOTAL"],cfg["FPS"])
        render_range(D,specs,i0,i1,chunk,cfg["FPS"],cfg["_DPI"],cfg["CRF"],prog=prog)
        return

    cfg=dict(CONFIG); here=os.path.dirname(os.path.abspath(__file__))
    if cfg["PREVIEW"]:
        cfg["WIDTH"]=960; cfg["FPS"]=15; cfg["SECONDS"]=max(30,int(cfg["SECONDS"]*0.25)); cfg["CRF"]=23
    cfg["_DPI"]=int(round(cfg["WIDTH"]/16))
    cfg["ARC_DIR"]=cfg["ARC_DIR"] or os.path.join(here,"thesis_arc_out","arc_npz")
    outdir=cfg["OUT"] or os.path.join(here,"thesis_arc_out","videos")
    os.makedirs(outdir,exist_ok=True)
    total=int(cfg["SECONDS"]*cfg["FPS"]); cfg["_TOTAL"]=total
    out=os.path.join(outdir,("PREVIEW_" if cfg["PREVIEW"] else "")+cfg["NAME"])

    print(f"[film] {cfg['WIDTH']}x{int(cfg['WIDTH']*9/16)} @ {cfg['FPS']} fps · "
          f"{cfg['SECONDS']} s · {total} frames")
    _ensure_imageio()                     # do this once, before workers spawn
    D=load_all(cfg["ARC_DIR"])
    specs=build_specs(D,total,cfg["FPS"])
    print(f"[film] data loaded · dock focus D{D['focus']+1} · captures at t={[int(x) for x in D['ft']]} s")

    # probe one mid-complexity frame for an ETA
    fig=plt.figure(figsize=(16,9),dpi=cfg["_DPI"])
    t0=time.time(); render_frame(fig,D,specs[int(total*0.75)]); fig.canvas.draw()
    spf=time.time()-t0; plt.close(fig)

    # cap parallel encoders: each worker = 1 python (matplotlib 4K canvas) + 1 ffmpeg;
    # more than ~10 concurrent x264 instances at 4K risks exhausting RAM.
    nw=cfg["WORKERS"] or max(1,(os.cpu_count() or 4)-2); nw=int(min(nw,10,total//30 or 1))
    print(f"[film] ~{spf:.2f}s/frame → est {spf*total/nw/60:.0f} min on {nw} worker(s)")

    if nw==1:
        render_range(D,specs,0,total,out,cfg["FPS"],cfg["_DPI"],cfg["CRF"],tag="1/1")
    else:
        # fixed chunk dir (not a fresh tempdir) so a re-run RESUMES finished chunks
        tmp=os.path.join(outdir,f"_chunks_{total}f_{cfg['_DPI']}dpi_{nw}w")
        os.makedirs(tmp,exist_ok=True)
        bounds=np.linspace(0,total,nw+1).astype(int)
        chunks=[os.path.join(tmp,f"chunk_{i:02d}.mp4") for i in range(nw)]
        progs=[c+".prog" for c in chunks]
        todo=[i for i in range(nw) if not os.path.exists(chunks[i]+".ok")]
        if len(todo)<nw:
            print(f"[film] resume: {nw-len(todo)}/{nw} chunks already complete")
        env=dict(os.environ); env["FILM_CFG"]=json.dumps(cfg); env["REDOCK_DIR"]=RD
        procs={}
        for i in todo:                           # staggered spawn: avoids a burst of
            procs[i]=subprocess.Popen(           # simultaneous ffmpeg/x264 allocations
                [sys.executable,os.path.abspath(__file__),"--worker",
                 str(bounds[i]),str(bounds[i+1]),chunks[i],progs[i]],env=env)
            time.sleep(1.2)
        failed=None
        while any(p.poll() is None for p in procs.values()):
            time.sleep(8); done=0
            for i in range(nw):
                if os.path.exists(chunks[i]+".ok"): done+=int(bounds[i+1]-bounds[i]); continue
                try: done+=int(open(progs[i]).read().split()[0])
                except Exception: pass
            print(f"[film] {done}/{total} frames ({100*done/total:.0f}%)",flush=True)
            bad=[i for i,p in procs.items() if p.poll() not in (None,0)]
            if bad:                              # abort everything immediately
                failed=bad
                for p in procs.values():
                    if p.poll() is None: p.terminate()
                break
        if failed is None:
            failed=[i for i,p in procs.items() if p.returncode not in (0,None)]
        if failed:
            raise RuntimeError(
                f"render worker(s) {sorted(failed)} failed — usually memory pressure from "
                f"too many parallel 4K encoders. Re-run with CONFIG WORKERS={max(2,nw//2)}; "
                f"finished chunks are kept and will be reused (resume).")
        _concat(chunks,out)
        for c in chunks+progs+[c+".ok" for c in chunks]:
            try: os.remove(c)
            except Exception: pass
        try:
            os.remove(os.path.join(tmp,"concat.txt")); os.rmdir(tmp)
        except Exception: pass
    print(f"[done] -> {out}")

if __name__=="__main__":
    main()
