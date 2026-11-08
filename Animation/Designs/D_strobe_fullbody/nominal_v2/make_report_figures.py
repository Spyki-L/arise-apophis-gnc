#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_report_figures.py  --  ARISE Design D CASE-2 report figures + stat tables (one shot)
========================================================================================
Regenerates EVERY figure and statistics table used in the Design D CASE-2 report directly
from the authoritative nominal products written by run_nominal_and_hifi.py:

    data/mc_v2_results_nominal.json        (summary: coverage bands, dV, recon CDF, facet prob)
    _nom_nominal_baseline.jsonl            (CASE-1 baseline, 32 spin phases)
    data/mc/draw_0000/                      (per-draw tree: coverage, trajectories, control,
                                             nav, attitude, camera, anim, apophis, _nav_stage)
    data/radar_mesh.npz                     (3996-facet shape)
    refly_targets_cma.npz                   (the existing CMA LVLH stations)

OUTPUT (all under outputs/report_figs/ unless ARISE_REPORT_FIGDIR is set):
    F1..F20  PNG figures (same filenames the report .tex expects in its figs/ folder)
    ../report_tables.txt    human-readable stat tables (also printed to stdout)
    ../report_values.json   machine-readable values (coverage, dV, residuals, attitude, terminal)
    ../report_tables.tex    ready-to-\input LaTeX tabular bodies (EKF / attitude / terminal / stations)

It is pure numpy + matplotlib (NO Basilisk), fails SOFT per figure (a missing stage only
skips its own figures), and can be run standalone:

    python make_report_figures.py

run_nominal_and_hifi.py calls it automatically as the final step after CASE 2.
"""
from __future__ import annotations
import os, sys, json, glob, traceback
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── journal-submission figure style (serif fonts, clear sizes, vector output) ──────────
# Applied to every figure; save() emits BOTH a 300-dpi PNG and a vector PDF so the report
# can include crisp scalable graphics (\includegraphics picks the PDF when present).
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9.5, "legend.framealpha": 0.92, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight"})

ROOT = Path(__file__).resolve().parent
DRAW = ROOT / "data" / "mc" / "draw_0000"
NS   = DRAW / "_nav_stage"
OUT  = Path(os.environ.get("ARISE_REPORT_FIGDIR", str(ROOT / "outputs" / "report_figs")))
OUT.mkdir(parents=True, exist_ok=True)
TBL  = OUT.parent
NDEP = 8
C    = plt.cm.tab10(np.linspace(0, 1, NDEP))
BLUE='#1f3c6e'; LBLUE='#9ecae1'; MBLUE='#3182bd'; ORANGE='#e6550d'; GREEN='#2ca25f'
SIG='#D55E00'; ERR='#0072B2'

MADE, SKIP = [], []
def _ok(n): MADE.append(n); print("  [fig] OK   ", n)
def _skip(n, e): SKIP.append((n, str(e))); print(f"  [fig] SKIP {n}  ({e})")
def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=300)
    try: fig.savefig(OUT / (os.path.splitext(name)[0] + ".pdf"))   # vector copy for journal/report
    except Exception: pass
    plt.close(fig); _ok(name)
def L(p):
    p = Path(p)
    if not p.is_file(): raise FileNotFoundError(p.name)
    return np.load(p, allow_pickle=True)
def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()]
def f1(x):
    return float(np.ravel(np.asarray(x))[0])

# ───────────────────────── load core products (guarded) ─────────────────────────
res = base = fs = mesh = anim = truth = stations = None
trj = ctrl = nav = att = cam = fdi = None
try: res = json.load(open(ROOT / "data" / "mc_v2_results_nominal.json")); S = res["summary"]
except Exception as e: print("[load] mc_v2_results_nominal.json:", e); S = None
try: base = jl(ROOT / "_nom_nominal_baseline.jsonl")
except Exception as e: print("[load] baseline jsonl:", e)
try: fs = L(DRAW / "coverage" / "facet_stereo.npz")
except Exception as e: print("[load] facet_stereo:", e)
try: mesh = L(ROOT / "data" / "radar_mesh.npz")
except Exception as e: print("[load] radar_mesh:", e)
try: anim = L(DRAW / "anim" / "anim_6dof_fleet.npz")
except Exception as e: print("[load] anim fleet:", e)
try: truth = L(DRAW / "apophis" / "attitude_truth.npz")
except Exception as e: print("[load] attitude_truth:", e)
try: stations = L(ROOT / "refly_targets_cma.npz")["DEPUTY_TARGETS_M"]
except Exception as e: print("[load] refly_targets_cma:", e)
def _stack(sub, patt):
    out = []
    for i in range(NDEP):
        try: out.append(L(DRAW / sub / patt.format(i=i)))
        except Exception: out.append(None)
    return out
trj  = _stack("trajectories", "apophis_rendezvous_full_D{i}.npz")
ctrl = _stack("control",      "ctrl_D{i}.npz")
nav  = _stack("nav",          "ekf_D{i}.npz")
att  = _stack("attitude",     "att_D{i}.npz")
cam  = _stack("camera",       "cam_D{i}.npz")
fdi  = [None]*NDEP
for i in range(NDEP):
    try: fdi[i] = L(NS / f"apophis_swarm_FDI_rgazel_D{i}.npz")
    except Exception: pass

# closest-approach epoch from D0
tca = None
if trj[0] is not None:
    R0 = np.linalg.norm(trj[0]["rD_N"] - trj[0]["rA_N"], axis=1)
    tca = float(trj[0]["t"][int(np.argmin(R0))])

# convenience arrays from the coverage product
if fs is not None:
    sz=fs["sigma_z_m"]; sm=fs["stereo_mask"].astype(bool); nv=fs["n_views"]
    lit=fs["lit_mask"].astype(bool); fp=fs["facet_prob"]; ar=fs["areas"]
if mesh is not None:
    V=mesh["vertices"]; Fc=mesh["faces"]; CEN=mesh["centroids"]; NRM=mesh["normals"]; AR=mesh["areas"]
    lon=np.degrees(np.arctan2(CEN[:,1],CEN[:,0])); lat=np.degrees(np.arcsin(CEN[:,2]/np.linalg.norm(CEN,axis=1)))

print(f"[paths] design root : {ROOT}")
print(f"[paths] figures ->   : {OUT}")
print(f"[paths] tables  ->   : {TBL}")

# ═══════════════════════ PART A : coverage / results figures (F1-F13) ═══════════════════════
if S is not None and base is not None:
    lo_w=np.array([b['whole'] for b in base]); lo_s=np.array([b['sunlit'] for b in base]); lo_d=np.array([b['dark'] for b in base])
    hi_w=np.array(res['whole'],float) if 'whole' in res else None

    try:  # F1 fidelity comparison
        fig,ax=plt.subplots(figsize=(7,4.3)); x=np.arange(3); w=0.38
        lo=[lo_w.mean(),lo_s.mean(),lo_d.mean()]; lo_e=[lo_w.std(),lo_s.std(),lo_d.std()]
        hi=[S['whole_body']['mean'],S['sunlit']['mean'],S['dark']['mean']]
        hi_e=[(S['whole_body']['P95']-S['whole_body']['P5'])/4,0.0,(S['dark']['P95']-S['dark']['P5'])/4]
        ax.bar(x-w/2,lo,w,yerr=lo_e,capsize=4,label='Case 1 - baseline (Lambert, occ off, no top-up)',color=LBLUE,ec='k',lw=.4)
        ax.bar(x+w/2,hi,w,yerr=hi_e,capsize=4,label='Case 2 - high-fidelity (Hapke, occ, 24$\\degree$ Sun, NAC+WAC, top-up)',color=BLUE,ec='k',lw=.4)
        for i,v in enumerate(lo): ax.text(i-w/2,v+1.3,f'{v:.1f}',ha='center',fontsize=8.5)
        for i,v in enumerate(hi): ax.text(i+w/2,v+1.3,f'{v:.1f}',ha='center',fontsize=8.5,fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(['whole-body','sunlit hemi','dark hemi']); ax.set_ylabel('stereo coverage [% area]')
        ax.set_title('Nominal stereo coverage - modeling-fidelity comparison\n(mean over 32 CA spin phases)',fontsize=10)
        ax.legend(fontsize=7.0,loc='lower left'); ax.set_ylim(0,108); ax.grid(axis='y',alpha=.3); save(fig,'F1_fidelity.png')
    except Exception as e: _skip('F1_fidelity.png',e)

    try:  # F2 coverage vs spin phase
        if hi_w is None: raise ValueError('no per-phase whole array')
        fig,ax=plt.subplots(figsize=(9,4.0)); mi=np.arange(len(hi_w))
        ax.plot(mi,hi_w,'-o',ms=4,color=BLUE,label='Case 2 high-fidelity whole-body')
        ax.plot(mi,lo_w,'-s',ms=3,color=LBLUE,label='Case 1 baseline whole-body')
        ax.axhline(hi_w.mean(),ls='--',color=BLUE,lw=1); ax.axhline(lo_w.mean(),ls='--',color=MBLUE,lw=1)
        ax.fill_between(mi,S['whole_body']['P5'],S['whole_body']['P95'],color=BLUE,alpha=.10,label='P5-P95 band (high-fi)')
        ax.set_xlabel('spin-phase sub-draw $m$ (Apophis rotation phase at closest approach)'); ax.set_ylabel('whole-body stereo [%]')
        ax.set_title('Whole-body coverage vs CA spin phase - single fixed nominal trajectory',fontsize=10)
        ax.legend(fontsize=8,loc='lower left'); ax.set_ylim(min(70,lo_w.min()-3),102); ax.grid(alpha=.3); save(fig,'F2_spinphase.png')
    except Exception as e: _skip('F2_spinphase.png',e)

    try:  # F3 CDF
        fig,ax=plt.subplots(figsize=(7,4.2))
        for arr,lab,col in [(lo_w,'Case 1 baseline',LBLUE),(hi_w,'Case 2 high-fidelity',BLUE)]:
            s=np.sort(arr); y=np.linspace(0,100,len(s)); ax.plot(s,y,'-',lw=2,color=col,label=lab)
        for p in (S['whole_body']['P5'],S['whole_body']['P50'],S['whole_body']['P95']):
            ax.axvline(p,color='gray',ls=':',lw=.9)
        ax.text(S['whole_body']['P5'],3,'P5 %.1f'%S['whole_body']['P5'],rotation=90,fontsize=7,va='bottom')
        ax.text(S['whole_body']['P50'],3,'P50 %.1f'%S['whole_body']['P50'],rotation=90,fontsize=7,va='bottom')
        ax.text(S['whole_body']['P95'],3,'P95 %.1f'%S['whole_body']['P95'],rotation=90,fontsize=7,va='bottom')
        ax.set_xlabel('whole-body stereo coverage [%]'); ax.set_ylabel('cumulative % of spin phases')
        ax.set_title('Spin-phase coverage CDF (phase-robust bands)',fontsize=10); ax.legend(loc='upper left',fontsize=8)
        ax.set_xlim(min(70,lo_w.min()-3),101); ax.grid(alpha=.3); save(fig,'F3_cdf.png')
    except Exception as e: _skip('F3_cdf.png',e)

if fs is not None and S is not None:
    try:  # F4 sigma_z CDF
        fig,ax=plt.subplots(figsize=(7,4.2))
        fin=np.isfinite(sz)&sm; szf=sz[fin]; af=ar[fin]; order=np.argsort(szf); cum=np.cumsum(af[order])/ar.sum()*100
        ax.plot(szf[order],cum,'-',lw=2,color=ORANGE); rec=S['recon_area_frac_within_tol_m']
        for tol in (0.25,0.5,1.0):
            frac=rec[str(tol)]*100; ax.axvline(tol,color='gray',ls=':',lw=.8)
            ax.text(tol*1.03,8,f'$\\leq${tol} m: {frac:.1f}%',rotation=90,fontsize=8,va='bottom')
        ax.axvline(0.15,color=GREEN,ls='--',lw=1); ax.text(0.15*0.72,55,'NAC goal\n0.15 m',fontsize=7.5,color=GREEN,ha='right')
        ax.set_xlabel('per-facet depth uncertainty $\\sigma_z$ [m]'); ax.set_ylabel('cumulative % of surface area')
        ax.set_xscale('log'); ax.set_title('Reconstruction-accuracy CDF (median $\\sigma_z$=%.2f m)'%np.median(szf),fontsize=10)
        ax.grid(alpha=.3,which='both'); save(fig,'F4_recon_cdf.png')
    except Exception as e: _skip('F4_recon_cdf.png',e)

def mesh3d(ax,vals,cmap,vmin,vmax,title,gray=None,el=20,az=30):
    pc=Poly3DCollection(V[Fc],linewidths=0.0)
    cols=plt.get_cmap(cmap)((np.clip(vals,vmin,vmax)-vmin)/(vmax-vmin+1e-9))
    if gray is not None: cols[gray]=[0.82,0.82,0.82,1.0]
    pc.set_facecolor(cols); ax.add_collection3d(pc)
    mx=np.abs(V).max(); ax.set_xlim(-mx,mx); ax.set_ylim(-mx,mx); ax.set_zlim(-mx,mx)
    ax.set_box_aspect((1,1,1)); ax.set_title(title,fontsize=9); ax.set_axis_off(); ax.view_init(el,az)

if fs is not None and mesh is not None and S is not None:
    try:  # F5 facet prob map
        fig=plt.figure(figsize=(13,4.4))
        a1=fig.add_subplot(131,projection='3d'); mesh3d(a1,fp,'viridis',fp.min(),fp.max(),'coverage prob (view A)',az=30)
        a2=fig.add_subplot(132,projection='3d'); mesh3d(a2,fp,'viridis',fp.min(),fp.max(),'coverage prob (view B)',az=210)
        a3=fig.add_subplot(133); scp=a3.scatter(lon,lat,c=fp,cmap='viridis',s=6)
        a3.set_xlabel('body lon [$\\degree$]'); a3.set_ylabel('body lat [$\\degree$]'); a3.set_title('coverage prob (lon/lat)')
        fig.colorbar(scp,ax=a3,label='P(facet covered)')
        fig.suptitle('Per-facet stereo-coverage probability over 32 spin phases (mean %.3f; robust $\\geq$0.9 = %.1f%%)'
                     %(fp.mean(),S['facet_prob']['robust_ge0p9_pct']),fontsize=11); save(fig,'F5_facetprob.png')
    except Exception as e: _skip('F5_facetprob.png',e)
    try:  # F6 sigma_z map
        fig=plt.figure(figsize=(13,4.4)); valid=np.isfinite(sz)&sm; szc=np.where(valid,np.clip(sz,0.05,0.5),0.5)
        a1=fig.add_subplot(131,projection='3d'); mesh3d(a1,szc,'plasma_r',0.05,0.5,'$\\sigma_z$ (view A); gray=no stereo',gray=~valid,az=30)
        a2=fig.add_subplot(132,projection='3d'); mesh3d(a2,szc,'plasma_r',0.05,0.5,'$\\sigma_z$ (view B); gray=no stereo',gray=~valid,az=210)
        a3=fig.add_subplot(133); a3.scatter(lon[~valid],lat[~valid],c='0.82',s=6)
        scp=a3.scatter(lon[valid],lat[valid],c=sz[valid],cmap='plasma_r',s=8,vmin=0.05,vmax=0.5)
        a3.set_xlabel('body lon [$\\degree$]'); a3.set_ylabel('body lat [$\\degree$]'); a3.set_title('$\\sigma_z$ (lon/lat)')
        fig.colorbar(scp,ax=a3,label='$\\sigma_z$ [m]')
        fig.suptitle('Per-facet reconstruction depth uncertainty (representative spin phase; whole=%.1f%%)'
                     %(100*ar[valid].sum()/ar.sum()),fontsize=11); save(fig,'F6_sigmaz.png')
    except Exception as e: _skip('F6_sigmaz.png',e)

if all(c is not None for c in cam):
    try:  # F7 camera range / SNR
        fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))
        for i in range(NDEP):
            rng=cam[i]['range_center_m']/1e3; a1.scatter(rng,cam[i]['n_seen'],color=C[i],s=22,label=f'D{i}'); a2.scatter(rng,cam[i]['median_snr'],color=C[i],s=22)
        a1.set_xlabel('frame range [km]'); a1.set_ylabel('# facets imaged / frame'); a1.set_title('Facets imaged per frame vs range'); a1.legend(fontsize=7,ncol=2); a1.grid(alpha=.3)
        a2.set_xlabel('frame range [km]'); a2.set_ylabel('median per-facet SNR'); a2.axhspan(30,60,color='green',alpha=0.10); a2.set_title('Median strobe-lit SNR vs range'); a2.grid(alpha=.3)
        save(fig,'F7_camera.png')
    except Exception as e: _skip('F7_camera.png',e)

if anim is not None and stations is not None:
    try:  # F8 approach + station layout
        fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.4))
        for i in range(NDEP):
            rl=anim[f'D{i+1}_rel_lvlh_RTN']/1e3; a1.plot(rl[:,0],rl[:,1],color=C[i],lw=1)
        a1.scatter([0],[0],marker='*',c='k',s=160,zorder=5); a1.set_xlabel('R [km]'); a1.set_ylabel('T [km]')
        a1.set_title('Deputy approach in LVLH (R-T), full window'); a1.grid(alpha=.3); a1.axis('equal')
        for i in range(NDEP): s=stations[i]/1e3; a2.scatter(s[0],s[1],color=C[i],s=70,ec='k',lw=.5,zorder=5,label=f'D{i}')
        a2.scatter([0],[0],marker='*',c='k',s=200,zorder=6); th=np.linspace(0,2*np.pi,100)
        for rr in (0.75,1.5): a2.plot(rr*np.cos(th),rr*np.sin(th),color='gray',ls=':',lw=.7)
        a2.set_xlabel('R [km]'); a2.set_ylabel('T [km]'); a2.set_title('Existing CMA fleet stations at CA (R-T plane)'); a2.legend(fontsize=7,ncol=2); a2.grid(alpha=.3); a2.axis('equal')
        save(fig,'F8_traj.png')
    except Exception as e: _skip('F8_traj.png',e)

if stations is not None and mesh is not None:
    try:  # F8b station shell
        fig=plt.figure(figsize=(7,6)); ax=fig.add_subplot(111,projection='3d')
        ax.add_collection3d(Poly3DCollection(V[Fc],linewidths=0.0,facecolor='0.6',alpha=0.9))
        rmin=np.linalg.norm(stations,axis=1).min()/1e3; rmax=np.linalg.norm(stations,axis=1).max()/1e3
        for i in range(NDEP):
            s=stations[i]; ax.scatter(s[0],s[1],s[2],color=C[i],s=80,ec='k',lw=.5); ax.plot([0,s[0]],[0,s[1]],[0,s[2]],color=C[i],lw=.6,alpha=.5); ax.text(s[0],s[1],s[2],f' D{i}',fontsize=8)
        mx=np.abs(stations).max()*1.1; ax.set_xlim(-mx,mx); ax.set_ylim(-mx,mx); ax.set_zlim(-mx,mx); ax.set_box_aspect((1,1,1))
        ax.set_xlabel('R [m]'); ax.set_ylabel('T [m]'); ax.set_zlabel('N [m]')
        ax.set_title('Existing 8-deputy CMA station shell in Apophis LVLH\n(|$\\rho$| %.2f-%.2f km; Apophis to scale at center)'%(rmin,rmax),fontsize=9.5)
        save(fig,'F8b_shell.png')
    except Exception as e: _skip('F8b_shell.png',e)

if all(c is not None for c in ctrl) and S is not None:
    try:  # F9 dV breakdown
        fig,ax=plt.subplots(figsize=(8,4.2)); x=np.arange(NDEP); bottom=np.zeros(NDEP)
        nb=len(ctrl[0]['dv_real_mps']); labels=['deploy']+[f'TCM{k}' for k in range(1,nb)]; cols=plt.cm.viridis(np.linspace(0.12,0.9,nb))
        Mx=np.array([ctrl[i]['dv_real_mps'] for i in range(NDEP)])
        for k in range(nb): ax.bar(x,Mx[:,k],bottom=bottom,color=cols[k],label=labels[k],ec='k',lw=.3); bottom+=Mx[:,k]
        for i in range(NDEP): ax.text(i,bottom[i]+0.004,f'{bottom[i]:.3f}',ha='center',fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels([f'D{i}' for i in range(NDEP)]); ax.set_ylabel('$\\Delta v$ [m/s]')
        ax.set_title('Per-deputy $\\Delta v$ (deploy + TCMs) - fleet realized %.3f m/s (corrector %.3f; cap 10)'%(Mx.sum(),S['fleet_dv']['mean']),fontsize=9.5)
        ax.legend(fontsize=8,ncol=5,loc='upper center'); ax.set_ylim(0,max(0.42,bottom.max()*1.2)); ax.grid(axis='y',alpha=.3); save(fig,'F9_dv.png')
    except Exception as e: _skip('F9_dv.png',e)

if all(n is not None for n in nav):
    try:  # F10 nav error vs range
        fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))
        for i in range(NDEP):
            d=nav[i]; rng=d['range_m']/1e3; ps=np.linalg.norm(d['pos_sigma_m'],axis=1); cr=d['crossrange_sigma_m']; m=rng<=200
            a1.plot(rng[m],ps[m],color=C[i],lw=1,label=f'D{i}'); a2.plot(rng[m],cr[m],color=C[i],lw=1)
        try:  # consider-covariance floors (nav_stage Schmidt-Kalman + DMC; ARISE_NAV_HIFI)
            sys.path.insert(0,str(ROOT/'modules')); import nav_stage as _NAV
            if trj[0] is not None:
                _rel=trj[0]['rD_N']-trj[0]['rA_N']; _hi=_NAV.run_ekf_hifi(_rel,trj[0]['t'])
                _rr=_hi['range_m']/1e3; _mm=_rr<=200
                a1.plot(_rr[_mm],np.linalg.norm(_hi['pos_sigma_consider_rel_m'],axis=1)[_mm],'k--',lw=2.2,label='consider: relative (dep-dep)')
                a1.plot(_rr[_mm],np.linalg.norm(_hi['pos_sigma_consider_abs_m'],axis=1)[_mm],color=SIG,ls='--',lw=2.2,label='consider: absolute (chief)')
        except Exception as _e: print('  [F10] consider overlay skipped:',_e)
        for a in (a1,a2): a.set_xlabel('range to Apophis [km]'); a.invert_xaxis(); a.set_yscale('log'); a.grid(alpha=.3,which='both')
        a1.set_ylabel('3D position $\\sigma$ [m]'); a1.set_title('EKF position uncertainty + consider-covariance floors (honest vs optimistic)'); a1.legend(fontsize=6.5,ncol=2)
        a2.set_ylabel('cross-range $\\sigma$ [m]'); a2.set_title('EKF cross-range uncertainty'); save(fig,'F10_nav.png')
    except Exception as e: _skip('F10_nav.png',e)

if all(a is not None for a in att) and tca is not None:
    try:  # F11 pointing + LOS near CA
        fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))
        for i in range(NDEP):
            d=att[i]; tt=d['t']-tca; m=np.abs(tt)<=10
            a1.plot(tt[m],np.degrees(d['pointing_err_rad'][m]),color=C[i],lw=1,label=f'D{i}'); a2.plot(tt[m],d['los_rate_rps'][m],color=C[i],lw=1)
        a1.axhline(2.0,color='r',ls='--',lw=1,label='FSM throw $\\pm2\\degree$'); a1.set_yscale('log'); a1.set_ylabel('pointing error [deg]')
        a1.set_xlabel('$t-t_{CA}$ [s]'); a1.set_title('Bus pointing error near CA'); a1.legend(fontsize=7,ncol=2); a1.grid(alpha=.3,which='both')
        peak=max(att[i]['los_rate_rps'].max() for i in range(NDEP))
        a2.axhline(0.16,color='r',ls='--',lw=1,label='XACT wheel rate $\\sim$0.16 rad/s'); a2.set_yscale('log')
        a2.set_xlabel('$t-t_{CA}$ [s]'); a2.set_ylabel('LOS rate [rad/s]'); a2.set_title('Line-of-sight rate near CA (peak %.1f rad/s)'%peak); a2.legend(fontsize=8); a2.grid(alpha=.3,which='both')
        save(fig,'F11_pointing.png')
    except Exception as e: _skip('F11_pointing.png',e)

if truth is not None:
    try:  # F12 tumble
        fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2)); tt=truth['t']-truth['t'][0]; w=truth['omega_BN_truth']
        for j,lab in enumerate(['$\\omega_x$','$\\omega_y$','$\\omega_z$']): a1.plot(tt,np.degrees(w[:,j]),label=lab)
        a1.plot(tt,np.degrees(np.linalg.norm(w,axis=1)),'k--',label='$|\\omega|$')
        a1.set_xlabel('t since imaging-window start [s]'); a1.set_ylabel('body rate [deg/s]'); a1.set_title('Apophis NPA tumble - body rates (truth)'); a1.legend(fontsize=8); a1.grid(alpha=.3)
        sh=truth['sun_hat']; a2.axis('off')
        txt=("APOPHIS ATTITUDE TRUTH (radar model)\n\nspin pole (ecl lon, lat): 246.8deg, -59.3deg\nP_rot = 27.45 h   P_prec = 265.7 h\nNPA nutation ~ 37deg\n|omega| = %.4f deg/s (roll)\ninertia I1:I2:I3 = 0.61:0.965:1.0\n\nsun_hat (J2000) = [%.3f, %.3f, %.3f]\necl. longitude = 23.8deg ; phase = 35.7deg\n\nshape: 3996 facets, max extent ~410 m\nequiv-sphere diam ~351 m, albedo 0.30\nbody frozen 0.015deg over a 4 s pass")%(np.degrees(np.linalg.norm(w,axis=1)).mean(),sh[0],sh[1],sh[2])
        a2.text(0.0,0.98,txt,va='top',fontsize=9.5,family='monospace'); save(fig,'F12_tumble.png')
    except Exception as e: _skip('F12_tumble.png',e)

if S is not None:
    try:  # F13 tiers
        fig,ax=plt.subplots(figsize=(7.5,4.2)); tiers=['0.15 m (NAC)','$\\leq$1 m (WAC)','shape\n(visual hull)']
        whole=[S['stereo_0p15m']['whole']['mean'],S['whole_body']['mean'],S['shape_visual_hull']['mean']]
        err=[(S['stereo_0p15m']['whole']['P95']-S['stereo_0p15m']['whole']['P5'])/4,(S['whole_body']['P95']-S['whole_body']['P5'])/4,(S['shape_visual_hull']['P95']-S['shape_visual_hull']['P5'])/4]
        ax.bar(tiers,whole,yerr=err,capsize=5,color=[MBLUE,BLUE,GREEN],ec='k',lw=.4,width=.6)
        for i,v in enumerate(whole): ax.text(i,v+1.0,f'{v:.1f}%',ha='center',fontweight='bold',fontsize=10)
        ax.set_ylabel('whole-body coverage [% area]'); ax.set_ylim(0,108); ax.grid(axis='y',alpha=.3)
        ax.set_title('Whole-body coverage by resolution tier (mean $\\pm$ phase band)',fontsize=10); save(fig,'F13_tiers.png')
    except Exception as e: _skip('F13_tiers.png',e)

# ═══════════════════════ PART B : GNC deep-dive figures (F14-F20) ═══════════════════════
def ekf_grid(deps,name):
    fig,axes=plt.subplots(len(deps),3,figsize=(11,1.7*len(deps)+0.6)); axes=np.atleast_2d(axes)
    comp=['Radial R','Along-track T','Cross-track N']
    for r,d in enumerate(deps):
        F=fdi[d]; t=F['t']; thr=(t-t[0])/3600.0
        err=F['x_est'][:,:3]-F['x_true'][:,:3]; s3=3*np.sqrt(np.diagonal(F['P'],axis1=1,axis2=2))[:,:3]; m=thr>=0.1
        for i in range(3):
            ax=axes[r,i]; e=err[:,i]; s=s3[:,i]
            ax.fill_between(thr[m],-s[m],s[m],color=SIG,alpha=0.13,lw=0)
            ax.plot(thr[m],s[m],color=SIG,ls='--',lw=.7); ax.plot(thr[m],-s[m],color=SIG,ls='--',lw=.7)
            ax.scatter(thr[m],e[m],s=1.5,alpha=.5,color=ERR,edgecolors='none'); ax.axhline(0,color='.6',lw=.5)
            rms=np.sqrt(np.mean(e[m]**2)); ins=100*np.mean(np.abs(e[m])<=s[m])
            ax.text(.97,.9,f'RMS {rms:.2f} m\n{ins:.0f}% in 3$\\sigma$',transform=ax.transAxes,ha='right',va='top',fontsize=7.5,bbox=dict(boxstyle='round,pad=.2',fc='white',ec='.7',lw=.4))
            if r==0: ax.set_title(comp[i],fontsize=10)
            if i==0: ax.set_ylabel(f'D{d}\nerror [m]',fontsize=9)
            if r==len(deps)-1: ax.set_xlabel('time since deploy [h]',fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle('Hill-frame relative-position error vs EKF $\\pm3\\sigma$ envelope',fontsize=12,y=0.997); save(fig,name)
if all(f is not None for f in fdi):
    try: ekf_grid([0,1,2,3],'F14_ekf_err_0123.png')
    except Exception as e: _skip('F14_ekf_err_0123.png',e)
    try: ekf_grid([4,5,6,7],'F15_ekf_err_4567.png')
    except Exception as e: _skip('F15_ekf_err_4567.png',e)
    try:  # F16 residuals near CA
        fig,axes=plt.subplots(1,3,figsize=(13,4.0)); chan=['range residual [m]','azimuth residual [arcsec]','elevation residual [arcsec]']; scc=[1.0,206265.0,206265.0]
        for d in range(NDEP):
            F=fdi[d]; t=F['t']; m=(t>=tca-900)&(t<=tca); tt=(t[m]-tca)/60.0; yp=F['y_post'][m]
            for i in range(3): axes[i].plot(tt,yp[:,i]*scc[i],color=C[d],lw=.9,label=f'D{d}' if i==0 else None)
        for i in range(3): axes[i].axhline(0,color='.5',lw=.6); axes[i].set_xlabel('$t-t_{CA}$ [min]'); axes[i].set_ylabel(chan[i]); axes[i].grid(alpha=.3)
        axes[0].legend(fontsize=7,ncol=2); fig.suptitle('EKF post-fit measurement residuals over the final 15 min before closest approach',fontsize=12); save(fig,'F16_residuals_nearCA.png')
    except Exception as e: _skip('F16_residuals_nearCA.png',e)

if all(t is not None for t in trj) and tca is not None:
    try:  # F17 guidance convergence + approach
        fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.3))
        for d in range(NDEP):
            t=trj[d]['t']; eh=trj[d]['err_hist']; tsd=(t-t[0])/3600.0; a1.semilogy(tsd,np.maximum(eh,1e-7),color=C[d],lw=1,label=f'D{d}')
        a1.set_xlabel('time since deploy [h]'); a1.set_ylabel('targeting error $|r_{dep}-r_{target}|$ [m]')
        a1.set_title('Differential-corrector approach to the LVLH station'); a1.legend(fontsize=7,ncol=2); a1.grid(alpha=.3,which='both')
        for d in range(NDEP):
            t=trj[d]['t']; R=np.linalg.norm(trj[d]['rD_N']-trj[d]['rA_N'],axis=1); ttc=(t-tca)/60.0; m=(ttc>=-60)&(ttc<=1); a2.semilogy(ttc[m],R[m]/1e3,color=C[d],lw=1)
        a2.axhline(5.0,color='r',ls='--',lw=.8,label='5 km resolvable range'); a2.set_xlabel('$t-t_{CA}$ [min]'); a2.set_ylabel('range to Apophis [km]')
        a2.set_title('Approach profile (final hour)'); a2.legend(fontsize=8); a2.grid(alpha=.3,which='both'); save(fig,'F17_guidance.png')
    except Exception as e: _skip('F17_guidance.png',e)

if all(a is not None for a in att) and tca is not None:
    try:  # F18 attitude / rate / torque near CA
        fig,axes=plt.subplots(1,3,figsize=(13,4.0))
        for d in range(NDEP):
            a=att[d]; t=a['t']; m=(t>=tca-900)&(t<=tca); tt=(t[m]-tca)/60.0
            axes[0].semilogy(tt,np.degrees(a['pointing_err_rad'][m]),color=C[d],lw=.9,label=f'D{d}')
            axes[1].semilogy(tt,np.linalg.norm(a['omega_BN_B'][m],axis=1),color=C[d],lw=.9)
            axes[2].plot(tt,np.linalg.norm(a['ctrl_torque_Nm'][m],axis=1),color=C[d],lw=.9)
        axes[0].set_ylabel('pointing error [deg]'); axes[0].set_title('Attitude (pointing) error'); axes[0].legend(fontsize=7,ncol=2)
        axes[1].axhline(0.16,color='r',ls='--',lw=.8,label='wheel rate $\\sim$0.16 rad/s'); axes[1].set_ylabel('body rate $|\\omega_{BN}|$ [rad/s]'); axes[1].set_title('Body rate'); axes[1].legend(fontsize=8)
        axes[2].set_ylabel('commanded torque $|L_r|$ [N$\\cdot$m]'); axes[2].set_title('Commanded control torque')
        for ax in axes: ax.set_xlabel('$t-t_{CA}$ [min]'); ax.grid(alpha=.3,which='both')
        fig.suptitle('Per-deputy attitude error, body rate and commanded torque over the final 15 min',fontsize=12); save(fig,'F18_attitude_nearCA.png')
    except Exception as e: _skip('F18_attitude_nearCA.png',e)

# F19/F20 per-deputy FOV cones (aimed at body)
def cone_lines(apex,axis,half_deg,length,nlin=18):
    axis=axis/np.linalg.norm(axis); tmp=np.array([0,0,1.0]) if abs(axis[2])<0.9 else np.array([1.0,0,0])
    u=np.cross(axis,tmp); u/=np.linalg.norm(u); w=np.cross(axis,u); th=np.radians(half_deg); out=[]
    for a in np.linspace(0,2*np.pi,nlin,endpoint=False):
        d=np.cos(th)*axis+np.sin(th)*(np.cos(a)*u+np.sin(a)*w); out.append((apex,apex+length*d))
    return out
if anim is not None and mesh is not None and truth is not None:
    Rb2i=truth['R_body2inertial']; Vi=V@Rb2i.T; CENi=CEN@Rb2i.T; NRMi=NRM@Rb2i.T; apc=anim['apophis_pos_N']
    def snap(d):
        pos=anim[f'D{d+1}_pos_N']; rel=pos-apc; rng=np.linalg.norm(rel,axis=1); k=int(np.argmin(rng)); return rel[k], rng[k]
    def fov_fig(deps,name):
        fig=plt.figure(figsize=(12,11))
        for j,d in enumerate(deps):
            ax=fig.add_subplot(2,2,j+1,projection='3d'); rel,rng=snap(d); bhat=-rel/np.linalg.norm(rel)
            look=CENi-rel[None,:]; ln=np.linalg.norm(look,axis=1); front=np.einsum('ij,ij->i',NRMi,look)<0
            ang=np.degrees(np.arccos(np.clip(np.einsum('ij,j->i',look/ln[:,None],bhat),-1,1))); vis=front&(ang<27.0)&(ln<5000.0)
            cols=np.tile(np.array([0.80,0.80,0.80,1.0]),(len(Fc),1)); cols[vis]=np.array([0.13,0.55,0.30,1.0])
            pc=Poly3DCollection(Vi[Fc],lw=0.0); pc.set_facecolor(cols); ax.add_collection3d(pc)
            for p0,p1 in cone_lines(rel,bhat,27.0,np.linalg.norm(rel)*1.05):
                ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],color='#3182bd',lw=.5,alpha=.5)
            ax.scatter(*rel,color='k',s=45,marker='^')
            mx=max(np.abs(Vi).max(),np.abs(rel).max())*1.05; ax.set_xlim(-mx,mx); ax.set_ylim(-mx,mx); ax.set_zlim(-mx,mx); ax.set_box_aspect((1,1,1)); ax.set_axis_off()
            ax.set_title(f'D{d}: {100*np.average(vis,weights=AR):.0f}% of body area in 27$\\degree$ FOV, range {rng/1e3:.2f} km',fontsize=9.5); ax.view_init(20,35)
        fig.suptitle('Per-deputy Apophis coverage at closest approach - facets inside the 27$\\degree$ FOV cone (green)',fontsize=12,y=0.99); save(fig,name)
    try: fov_fig([0,1,2,3],'F19_fov_0123.png')
    except Exception as e: _skip('F19_fov_0123.png',e)
    try: fov_fig([4,5,6,7],'F20_fov_4567.png')
    except Exception as e: _skip('F20_fov_4567.png',e)

# ═══════════════════════ PART B2 : hi-fi upgrade figures (F21-F23) ═══════════════════════
sys.path.insert(0, str(ROOT/"modules"))
try:  # F21 detector radiation + sensor MTF (detector_hifi)
    import detector_hifi as _DET, hardware as _HW
    fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))
    k=np.linspace(0,0.5,200)
    a1.plot(k,_DET.mtf_charge_diffusion(k),color=MBLUE,label='charge diffusion')
    a1.plot(k,_DET.mtf_ipc(k),color=GREEN,label='IPC (%.0f%%)'%(_DET.ipc_crosstalk_frac()*100))
    a1.plot(k,_DET.mtf_pixel(k),'--',color='gray',label='pixel aperture')
    a1.plot(k,_DET.sensor_mtf(k),lw=2.4,color=BLUE,label='total sensor MTF')
    a1.axvline(0.5,ls=':',color='k'); a1.set_xlabel('spatial frequency [cyc/px]'); a1.set_ylabel('MTF'); a1.set_ylim(0,1.02)
    a1.set_title('CIS120 sensor MTF (Nyquist %.2f, eff PSF %.2f px)'%(_DET.sensor_mtf(0.5),_DET.effective_psf_sigma_px())); a1.legend(fontsize=8)
    fl=np.geomspace(1e9,1e12,60)
    a2.loglog(fl,[_DET.dark_current_mean_increase_e_s(x) for x in fl],color=BLUE,lw=2)
    a2.set_xlabel('10-MeV-equiv proton fluence [p/cm$^2$]'); a2.set_ylabel('dark-current increase [e$^-$/s]',color=BLUE); a2.tick_params(axis='y',labelcolor=BLUE)
    a2b=a2.twinx(); a2b.semilogx(fl,[_DET.hot_pixel_fraction(x)*100 for x in fl],color=ORANGE,ls='--',lw=2); a2b.set_ylabel('hot-pixel fraction [%]',color=ORANGE); a2b.tick_params(axis='y',labelcolor=ORANGE)
    a2.axvline(_HW.CIS120_RAD.proton_fluence_p_cm2,ls=':',color='gray'); a2.set_title('Radiation (proton DDD): dark current + hot pixels vs dose')
    save(fig,'F21_detector.png')
except Exception as e: _skip('F21_detector.png',e)

try:  # F22 soft-shadow illumination + self-illumination map (occlusion_bvh)
    import occlusion_bvh as _OCC
    if mesh is not None:
        try: sun=np.asarray(L(ROOT/'data'/'encounter_geometry.npz')['sun_hat'],float)
        except Exception: sun=np.array([0.912,0.376,0.163])
        sun=sun/np.linalg.norm(sun); rt=_OCC.MeshRT(V,Fc)
        frac=_OCC.illumination_fraction(rt,CEN,NRM,sun,n_sun=12)
        binlit=_OCC.lit_facets(rt,CEN,NRM,sun)
        shad=np.where(frac<0.5)[0]; si=np.zeros(len(CEN))
        if len(shad): si[shad]=_OCC.self_illumination(rt,CEN,NRM,AR,sun,subset=shad,with_occlusion=False)
        fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))
        s1=a1.scatter(lon,lat,c=frac,cmap='inferno',s=7,vmin=0,vmax=1); fig.colorbar(s1,ax=a1,label='illumination fraction')
        a1.set_xlabel('body lon [$\\degree$]'); a1.set_ylabel('body lat [$\\degree$]')
        a1.set_title('Soft-shadow illumination (finite Sun): binary %.0f%% vs soft %.0f%%'%(100*binlit.mean(),100*frac.mean()))
        s2=a2.scatter(lon,lat,c=si*1e3,cmap='viridis',s=7); fig.colorbar(s2,ax=a2,label='self-illum [10$^{-3}$ E$_\\odot$]')
        a2.set_xlabel('body lon [$\\degree$]'); a2.set_ylabel('body lat [$\\degree$]'); a2.set_title('Single-bounce self-illumination (concavities)')
        save(fig,'F22_illumination.png')
except Exception as e: _skip('F22_illumination.png',e)

try:  # F23 reconstruction: stereo vs stereophotoclinometry (photogram)
    if fs is not None:
        gsd=0.15; mono_lit=lit&(~sm)&(nv>=1)
        spc_sz=sz.copy(); spc_sz[mono_lit]=0.2*gsd
        tols=np.geomspace(0.05,5,60); spc_mask=sm|mono_lit
        st_cdf=[100*ar[sm&(sz<=t)].sum()/ar.sum() for t in tols]
        spc_cdf=[100*ar[spc_mask&(spc_sz<=t)].sum()/ar.sum() for t in tols]
        fig,axx=plt.subplots(figsize=(7,4.3))
        axx.plot(tols,st_cdf,color='gray',lw=2,ls='--',label='stereo only (%.0f%% area)'%(100*ar[sm].sum()/ar.sum()))
        axx.plot(tols,spc_cdf,color=BLUE,lw=2.4,label='stereophotoclinometry (%.0f%% area)'%(100*ar[spc_mask].sum()/ar.sum()))
        axx.axvline(gsd,ls=':',color=ORANGE,label='GSD %.0f cm'%(gsd*100))
        axx.set_xscale('log'); axx.set_xlabel('reconstruction tolerance [m]'); axx.set_ylabel('% area reconstructed')
        axx.set_title('Reconstruction: stereo vs SPC (photoclinometry recovers mono / low-parallax facets)')
        axx.legend(fontsize=8.5); axx.grid(alpha=.3,which='both'); save(fig,'F23_reconstruction.png')
except Exception as e: _skip('F23_reconstruction.png',e)


# ═══════════════════════ PART C : statistics tables ═══════════════════════
vals = {}
lines = ["ARISE Design D CASE-2 -- report statistics tables",
         "="*70, ""]

# coverage summary
if S is not None:
    vals['coverage'] = {k: S.get(k) for k in ('whole_body','sunlit','dark','stereo_0p15m','shape_visual_hull',
                                              'recon_area_frac_within_tol_m','facet_prob','fleet_dv')}
    wb=S['whole_body']; dk=S['dark']
    lines += ["COVERAGE SUMMARY (mean over 32 spin phases)",
              f"  whole-body : mean {wb['mean']:.1f}%  P5 {wb['P5']:.1f}  P50 {wb['P50']:.1f}  P95 {wb['P95']:.1f}",
              f"  sunlit     : mean {S['sunlit']['mean']:.1f}%",
              f"  dark       : mean {dk['mean']:.1f}%  P5 {dk['P5']:.1f}  P50 {dk['P50']:.1f}  P95 {dk['P95']:.1f}",
              f"  0.15 m NAC : whole {S['stereo_0p15m']['whole']['mean']:.1f}%",
              f"  shape hull : {S['shape_visual_hull']['mean']:.1f}%",
              f"  fleet dV   : corrector {S['fleet_dv']['mean']:.3f}  realized {S['fleet_dv'].get('realized_mean',float('nan')):.3f} m/s",
              f"  recon CDF  : <=0.25 m {S['recon_area_frac_within_tol_m']['0.25']*100:.1f}%  <=0.5 m {S['recon_area_frac_within_tol_m']['0.5']*100:.1f}%  <=1 m {S['recon_area_frac_within_tol_m']['1.0']*100:.1f}%",
              f"  robust facets (>=0.9): {S['facet_prob']['robust_ge0p9_pct']:.1f}%", ""]

tex = []  # LaTeX tabular bodies

# EKF residual table
if all(f is not None for f in fdi) and tca is not None:
    vals['ekf'] = []
    lines += ["EKF RESIDUALS / POSITION ERROR (per deputy)",
              "  dep | rRMS[m] azRMS['']  elRMS['']| posRMS_arc[m] posRMS_15min[m] | in3sig[%]"]
    for d in range(NDEP):
        F=fdi[d]; t=F['t']; m=t>=t[0]+360
        yp=F['y_post']; rr=np.sqrt(np.nanmean(yp[m,0]**2)); az=np.sqrt(np.nanmean(yp[m,1]**2))*206265; el=np.sqrt(np.nanmean(yp[m,2]**2))*206265
        err=F['x_est'][:,:3]-F['x_true'][:,:3]; pr=np.sqrt(np.mean(np.sum(err[m]**2,axis=1)))
        m15=(t>=tca-900)&(t<=tca); pr15=np.sqrt(np.mean(np.sum(err[m15]**2,axis=1)))
        s3=3*np.sqrt(np.diagonal(F['P'],axis1=1,axis2=2))[:,:3]; ins=100*np.mean(np.all(np.abs(err[m])<=s3[m],axis=1))
        vals['ekf'].append(dict(dep=d,r_rms_m=rr,az_rms_as=az,el_rms_as=el,pos_rms_arc_m=pr,pos_rms_15min_m=pr15,in3sig_pct=ins))
        lines.append(f"  D{d}  | {rr:6.3f} {az:7.2f} {el:7.2f} | {pr:9.3f} {pr15:11.3f} | {ins:6.1f}")
        tex.append(f"D{d} & {rr:.3f} & {az:.2f} & {el:.2f} & {pr:.3f} & {pr15:.3f} & {ins:.1f}\\% \\\\")
    lines.append("")

# attitude table
if all(a is not None for a in att) and tca is not None:
    vals['attitude'] = []
    lines += ["ATTITUDE / RATE / TORQUE  (at t-15min  and  at CA)",
              "  dep | pe_15[deg] pe_CA[deg] | w_15[rad/s] w_CA | L_CA[Nm] | wheel_CA[rad/s] | LOS_CA[rad/s]"]
    for d in range(NDEP):
        a=att[d]; t=a['t']
        k15=int(np.argmin(np.abs(t-(tca-900)))); kca=int(np.argmin(np.abs(t-tca)))
        pe=np.degrees(a['pointing_err_rad']); w=np.linalg.norm(a['omega_BN_B'],axis=1); Lr=np.linalg.norm(a['ctrl_torque_Nm'],axis=1)
        ws=np.abs(a['wheel_speed_rads']).max(axis=1); lr=a['los_rate_rps']
        vals['attitude'].append(dict(dep=d,pe_15=pe[k15],pe_ca=pe[kca],w_15=w[k15],w_ca=w[kca],L_ca=Lr[kca],wheel_ca=ws[kca],los_ca=lr[kca]))
        lines.append(f"  D{d}  | {pe[k15]:9.3f} {pe[kca]:9.1f} | {w[k15]:10.4f} {w[kca]:6.3f} | {Lr[kca]:7.3f} | {ws[kca]:6.0f} | {lr[kca]:6.2f}")
        tex.append(f"D{d} & {pe[k15]:.3f} & {pe[kca]:.1f} & {w[k15]:.4f} & {w[kca]:.3f} & {Lr[kca]:.3f} & {ws[kca]:.0f} & {lr[kca]:.2f} \\\\")
    lines.append("")

# terminal targeting error + dV + stations
if all(t is not None for t in trj):
    vals['terminal'] = []; vals['stations'] = []
    rca = None
    try: rca = L(NS/"nav_uncertainty.npz")['range_ca_m']
    except Exception: pass
    lines += ["TERMINAL TARGETING ERROR + STATIONS (per deputy)",
              "  dep | err[um]   err[cm]   | tot_dv[m/s] | station R,T,N [m]                 |rho|[km]  rCA[m]"]
    for d in range(NDEP):
        T=trj[d]; en=f1(T['err_norm']); dv=f1(T['tot_dv'])
        st = stations[d] if stations is not None else [float('nan')]*3
        rho = float(np.linalg.norm(st))/1e3 if stations is not None else float('nan')
        rc = float(rca[d]) if rca is not None else float('nan')
        vals['terminal'].append(dict(dep=d,err_um=en*1e6,err_cm=en*100,tot_dv_mps=dv))
        vals['stations'].append(dict(dep=d,R=float(st[0]),T=float(st[1]),N=float(st[2]),rho_km=rho,rCA_m=rc))
        lines.append(f"  D{d}  | {en*1e6:8.3f} {en*100:9.2e} | {dv:9.3f} | ({st[0]:7.1f},{st[1]:7.1f},{st[2]:7.1f}) | {rho:6.3f} {rc:8.1f}")
        tex.append(f"D{d} & {en*1e6:.2f} & {en*100:.1e} & {dv:.3f} & ({st[0]:.0f},{st[1]:.0f},{st[2]:.0f}) & {rho:.3f} & {rc:.0f} \\\\")
    lines.append("")

# write outputs
(TBL/"report_tables.txt").write_text("\n".join(lines), encoding="utf-8")
json.dump(vals, open(TBL/"report_values.json","w"), indent=1, default=float)
(TBL/"report_tables.tex").write_text("\n".join(tex), encoding="utf-8")

print("\n" + "\n".join(lines))
print("="*70)
print(f"[figures] {len(MADE)} written to {OUT}")
if SKIP:
    print(f"[skipped] {len(SKIP)} figures: " + ", ".join(n for n,_ in SKIP))
print(f"[tables ] report_tables.txt / report_values.json / report_tables.tex -> {TBL}")
print("Done.")
