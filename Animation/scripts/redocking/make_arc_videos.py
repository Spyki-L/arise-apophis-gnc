# -*- coding: utf-8 -*-
"""make_arc_videos.py — individual + combined MP4s of the full thesis arc from saved npz.
Reads: real imaging npz (deploy->CA) + return_arc.npz + dock_rollout.npz."""
import os, sys, glob, numpy as np

def _redock_dir():
    d=os.environ.get("REDOCK_DIR","")
    if d and os.path.isfile(os.path.join(d,"config_redock.py")): return d
    for p in sys.path:
        if p and os.path.isfile(os.path.join(p,"config_redock.py")): return p
    raise RuntimeError("no redocking dir")

RD=_redock_dir(); sys.path.insert(0,RD)
# reuse the cockpit renderer + helpers
import importlib.util
_CK = os.environ.get("COCKPIT_PY", os.path.join(RD,"run_redock_cockpit.py"))
spec=importlib.util.spec_from_file_location("rrc",_CK); rrc=importlib.util.module_from_spec(spec); spec.loader.exec_module(rrc)

import config_redock as C
from dynamics.basilisk_backend import eci_to_lvlh
from dynamics.attitude import mrp_to_dcm
from envs.rafti import RaftiPorts

def _ports():
    r,u,rad=C.load_espa_ports()
    p=RaftiPorts(r,u,rad,C.RAFTI,deputy_dims=C.ESPA["deputy_dims"],height=C.ESPA["height_m"])
    return p, dict(r=p.cyl.radius,h=p.cyl.half_h,pr=p.cyl.ports_r,pd=p.cyl.ports_dir)

def load_mission_timeseries(nframes=90):
    files=sorted(glob.glob(os.path.join(C.ANIM_DIR,"data","trajectories","apophis_rendezvous_full_D*.npz")))
    if len(files)<C.N_DEPUTIES: return None
    ds=[np.load(f,allow_pickle=True) for f in files[:C.N_DEPUTIES]]
    # common time grid (deputy npz can differ in length/range) -> interpolate
    t0=min(float(d["t"][0]) for d in ds); t1=min(float(d["t"][-1]) for d in ds)  # full staggered deploy -> CA
    grid=np.linspace(t0,t1,nframes)
    def col(d,key): return np.column_stack([np.interp(grid,d["t"],d[key][:,c]) for c in range(3)])
    tracks=np.zeros((nframes,C.N_DEPUTIES,3)); apo=np.zeros((nframes,3)); apr=np.zeros(nframes)
    d0=ds[0]; rC=col(d0,"rC_N"); vC=col(d0,"vC_N"); rA=col(d0,"rA_N"); vA=col(d0,"vA_N")
    for k,d in enumerate(ds):
        rD=col(d,"rD_N"); vD=col(d,"vD_N")
        for fi in range(nframes):
            tracks[fi,k]=eci_to_lvlh(rD[fi],vD[fi],rC[fi],vC[fi])[:3]
    for fi in range(nframes):
        apo[fi]=eci_to_lvlh(rA[fi],vA[fi],rC[fi],vC[fi])[:3]; apr[fi]=np.linalg.norm(rA[fi]-rC[fi])
    return dict(tracks=tracks,apo=apo,apr=apr,t=grid)

def _Rbls(A,N): return [mrp_to_dcm(A[k]).T for k in range(N)]

def build_mission_frames(mis, obs0, focus, gk0=0):
    N=C.N_DEPUTIES; F=mis["tracks"].shape[0]; frames=[]
    for f in range(F):
        pos=mis["tracks"][f]; s=f/(F-1); span=max(np.linalg.norm(pos,axis=1).max()*1.2,500.0)
        rng=mis["apr"][f]; inbox=rng<span*1.4; apk=mis["apo"][f]
        phase=("DEPLOY / CRUISE" if s<0.4 else "CRUISE / APPROACH" if s<0.88 else "IMAGING @ CLOSEST APPROACH")
        frames.append(dict(pos=pos,att=np.zeros((N,3)),caps=np.zeros(N,bool),Rbl=_Rbls(np.zeros((N,3)),N),
            trail=[mis["tracks"][:f+1,j] for j in range(N)],span=span,elev=20,azim=-62+0.22*(gk0+f),
            phase=phase,t=f"t+{(mis['t'][f]-mis['t'][0])/3600:.1f}h",
            apophis=(apk if inbox else None),ap_bearing=(None if inbox else apk),ap_alpha=1.0,ap_rng=rng/1e3,
            sightlines=(s>0.85 and inbox),k=gk0+f,brain_active=False,obs_focus=obs0,act_focus=np.zeros(6),
            hidden=rrc.brain_activations(obs0),mode="APPROACH",dstate=["IMAGING"]*N,
            cot=f"{phase} · Apophis {rng/1e3:.0f} km · fleet imaging · redock brain STANDBY"))
    return frames

def build_return_frames(ret, pb0, formation, obs0, focus, gk0):
    N=C.N_DEPUTIES; arcs=ret["arcs"]; F=arcs.shape[0]; frames=[]
    ring_r=_ports()[1]["r"]
    for k in range(F):
        s=k/(F-1); pos=arcs[k]; span=max(np.linalg.norm(pos,axis=1).max()*1.18,ring_r*3)
        frames.append(dict(pos=pos,att=np.zeros((N,3)),caps=np.zeros(N,bool),Rbl=_Rbls(np.zeros((N,3)),N),
            trail=[arcs[max(0,k-40):k+1,j] for j in range(N)],span=span,elev=20,azim=-62+0.22*(gk0+k),
            phase="PHASE-A RETURN TO CHIEF",t=f"return {int(s*100)}%",apophis=None,ap_alpha=0,ap_rng=0,
            k=gk0+k,brain_active=False,obs_focus=obs0,act_focus=np.zeros(6),hidden=rrc.brain_activations(obs0),
            mode="APPROACH",dstate=["RETURN"]*N,
            cot=f"Phase-A GNC return · dv {ret.get('dv_total',0):.1f} m/s · zoom {span/1000:.1f} km -> ring · brain STANDBY"))
    return frames

def build_dock_frames(dk, focus, gk0, dock_frames=200):
    N=C.N_DEPUTIES; traj=dk["traj"]; att=dk["att"]; caps=dk["caps"]; obs=dk["obs"]; act=dk["act"]; port_of=dk["port_of"]
    p,ring=_ports(); Tb=traj.shape[0]; stride=max(1,Tb//dock_frames); idx=list(range(0,Tb,stride))
    if idx[-1]!=Tb-1: idx.append(Tb-1)
    frames=[]
    for gi,k in enumerate(idx):
        pos=traj[k]; A=att[k]; cp=caps[k]; span=max(np.linalg.norm(pos,axis=1).max()*1.15,ring["r"]*2.6)
        o=obs[k,focus]; a=act[k,focus]; nrm,raw=rrc.obs_features(o); mode=rrc.dock_mode(raw,bool(cp[focus]))
        ds=[rrc.dock_mode(rrc.obs_features(obs[k,j])[1],bool(cp[j])) for j in range(N)]
        cot=(f"D{focus+1}: range {raw[0]:.1f}m closing {raw[2]:.03f}m/s cone {raw[6]:.0f}deg att {raw[4]:.0f}deg "
             f"| dv=[{a[0]:+.02f},{a[1]:+.02f},{a[2]:+.02f}] -> {mode}")
        frames.append(dict(pos=pos,att=A,caps=cp,Rbl=_Rbls(A,N),trail=[traj[max(0,k-70):k+1,j] for j in range(N)],
            span=span,elev=22,azim=-62+0.2*(gk0+gi),phase="TERMINAL 6-DOF RE-DOCK",t=f"{k}s",apophis=None,ap_alpha=0,
            ap_rng=0,k=gk0+gi,brain_active=True,obs_focus=o,act_focus=a,hidden=rrc.brain_activations(o),
            mode=mode,dstate=ds,cot=cot))
    return frames

def _ensure_imageio():
    try:
        import imageio.v2  # noqa
        import imageio_ffmpeg  # noqa
        return True
    except Exception:
        import subprocess, sys
        print("[setup] installing imageio + imageio-ffmpeg for MP4 export (one-time)...")
        try:
            subprocess.run([sys.executable,"-m","pip","install","-q","imageio","imageio-ffmpeg"],check=True)
            import importlib, imageio; importlib.invalidate_caches(); importlib.reload(imageio)
            import imageio.v2, imageio_ffmpeg  # noqa
            return True
        except Exception as e:
            print(f"[setup] auto-install failed ({e}); writing GIF instead."); return False

def render_video(frames, focus, dims, ring, out_mp4, fps=25, dpi=100):
    """Render frames to MP4 via imageio's ffmpeg writer (auto-installs imageio), GIF fallback."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ck=rrc.Cockpit(focus,dims,ring); ck.fig.set_dpi(dpi)
    if not _ensure_imageio():
        from PIL import Image
        imgs=[]
        for fr in frames:
            ck.render(fr); ck.fig.canvas.draw()
            imgs.append(Image.fromarray(np.asarray(ck.fig.canvas.buffer_rgba())[...,:3]))
        gif=out_mp4[:-4]+".gif"; imgs[0].save(gif,save_all=True,append_images=imgs[1:],duration=int(1000/min(fps,18)),loop=0)
        print("wrote",os.path.basename(gif),"(",len(frames),"frames )"); plt.close(ck.fig); return
    try:
        import imageio.v2 as iio
        w=iio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8,
                         pixelformat="yuv420p", macro_block_size=1)
        for fr in frames:
            ck.render(fr); ck.fig.canvas.draw()
            buf=np.asarray(ck.fig.canvas.buffer_rgba())[..., :3]
            h,wd=buf.shape[:2]; buf=np.ascontiguousarray(buf[:h//2*2, :wd//2*2])
            w.append_data(buf)
        w.close(); print("wrote",os.path.basename(out_mp4),"(",len(frames),"frames )")
    except Exception as e:
        from PIL import Image
        imgs=[]
        for fr in frames:
            ck.render(fr); ck.fig.canvas.draw()
            imgs.append(Image.fromarray(np.asarray(ck.fig.canvas.buffer_rgba())[...,:3]))
        gif=out_mp4[:-4]+".gif"; imgs[0].save(gif,save_all=True,append_images=imgs[1:],duration=int(1000/min(fps,18)),loop=0)
        print(f"mp4 failed ({e}); wrote GIF",os.path.basename(gif))
    plt.close(ck.fig)

def _focus(dk):
    N=C.N_DEPUTIES
    return int(np.argmax([np.argmax(dk["caps"][:,k]) if dk["caps"][:,k].any() else len(dk["caps"]) for k in range(N)]))

def make_all(out_dir, imaging_frames=90, return_np=None, dock_np=None, quality=1.0, which=("00","01","02","03")):
    os.makedirs(out_dir,exist_ok=True)
    p,ring=_ports(); dims=C.ESPA["deputy_dims"]
    dk=dict(np.load(dock_np,allow_pickle=True)) if dock_np and os.path.exists(dock_np) else None
    ret=dict(np.load(return_np,allow_pickle=True)) if return_np and os.path.exists(return_np) else None
    focus=_focus(dk) if dk is not None else 6
    obs0=dk["obs"][0,focus] if dk is not None else np.zeros(12)
    mis=load_mission_timeseries(int(imaging_frames*quality))
    formation=mis["tracks"][-1] if mis is not None else None
    fps=25
    need_m=("01" in which) or ("00" in which); need_r=("02" in which) or ("00" in which); need_d=("03" in which) or ("00" in which)
    mf=build_mission_frames(mis,obs0,focus) if (need_m and mis is not None and mis["tracks"].shape[0]>1) else []
    rf=build_return_frames(ret,dk["traj"][0] if dk is not None else None, formation, obs0, focus, len(mf)) if (need_r and ret is not None) else []
    df=build_dock_frames(dk,focus,len(mf)+len(rf),max(2,int(200*quality))) if (need_d and dk is not None) else []
    jobs={"01":("01_imaging",mf),"02":("02_return",rf),"03":("03_terminal_dock",df),
          "00":("00_full_arc",mf+rf+df)}
    for key in which:
        name,frames=jobs[key]
        if not frames: print("skip",name,"(no data)"); continue
        render_video(frames,focus,dims,ring,os.path.join(out_dir,name+".mp4"),fps=fps,dpi=100)

# ================= CONFIG (edit; no CLI args) =================
CONFIG = dict(
    ARC_DIR       = None,   # folder with return_arc.npz + dock_rollout.npz (default: ./thesis_arc_out/arc_npz)
    OUT_DIR       = None,   # where to write the mp4s (default: <ARC_DIR>/../videos)
    WHICH         = ("00","01","02","03"),   # 00=combined, 01=imaging, 02=return, 03=dock
    IMAGING_FRAMES= 90,
    QUALITY       = 1.0,
)
def main():
    HERE=os.path.dirname(os.path.abspath(__file__))
    arc = CONFIG["ARC_DIR"] or os.path.join(HERE,"thesis_arc_out","arc_npz")
    out = CONFIG["OUT_DIR"] or os.path.join(os.path.dirname(arc),"videos")
    make_all(out, imaging_frames=CONFIG["IMAGING_FRAMES"], quality=CONFIG["QUALITY"],
             return_np=os.path.join(arc,"return_arc.npz"), dock_np=os.path.join(arc,"dock_rollout.npz"),
             which=CONFIG["WHICH"])
    print("[done] videos ->",out)

if __name__=="__main__":
    main()
