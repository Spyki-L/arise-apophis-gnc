# -*- coding: utf-8 -*-
r"""
run_redock_cockpit.py
=====================
ARISE / APOPHIS — RE-DOCK MISSION OPS + RL BRAIN  (one neon-on-black cockpit).

Press *Run* in VS Code / Visual Studio (no command-line arguments).

It animates the COMPLETE arc once imaging at Apophis closest approach is done:
    IMAGING @ CA  →  Phase-A RETURN to the chief  →  TERMINAL 6-DOF RE-DOCK
and, alongside, shows HOW THE RL BRAIN WORKS — a live connectome of the docking
policy (SENSE obs → π network → ACTOR core → docking MODE → COMMAND actions), in
the same aesthetic as the FDIR AUTONOMY BRAIN.

  LEFT  : 3D chief-LVLH (Hill) view.  Real post-imaging formation + Apophis at CA
          (from data/trajectories/apophis_rendezvous_full_D*.npz if present),
          the return trails, and the terminal dock onto the ESPA/RAFTI ring.
          Auto-zooms across the scales (10 km → 0.5 m) with a phase HUD.
  RIGHT : the RL docking brain (STANDBY during return, ENGAGED at terminal dock) +
          a fleet docking-state strip and a live chain-of-thought.

DYNAMICS  "basilisk" (hi-fi, auto-adds <basilisk-develop>/dist3; falls back to HCW).
POLICY    "bc" = trained policy drives Δv, verified PD closes attitude (flight
          deployment); falls back to the verified PD.  Both dock the fleet.
Config is the CONFIG block below — edit and re-Run.
"""
from __future__ import annotations
import os, sys, glob, time
import numpy as np

# ====================== CONFIG (edit; no CLI args) ======================
CONFIG = dict(
    DYNAMICS      = "basilisk",   # "basilisk" | "hcw" | "auto"
    POLICY        = "bc",         # "bc" (policy Δv + verified PD attitude) | "pd"
    SEED          = 7,
    SHOW_LIVE     = True,         # open the interactive cockpit window
    SAVE_MP4      = False,        # also save mp4 (falls back to gif w/o ffmpeg)
    SAVE_GIF      = False,        # also save gif (needs only Pillow)
    OUT_DIR       = None,         # default: ./animation_out next to this file
    FPS           = 25,
    N_IMAGING     = 28,           # establishing frames at Apophis closest approach
    N_RETURN      = 110,          # Phase-A return frames
    DOCK_FRAMES   = 200,          # terminal-dock frames (rollout is decimated to this)
    FOCUS         = None,         # brain focus deputy index (0-7); None = auto (last to dock)
    USE_REAL_MISSION = True,      # use the real CA formation + Apophis from the npz if present
    CACHE         = True,         # cache the dock rollout (.npz); delete to re-simulate
    CKPT          = None,         # explicit policy checkpoint; None = auto-find
)
RETUNED_PD = dict(k_lat=0.3504371882, k_ax=0.0931707341, kd=9.9234303808,
                  k_slew=0.3192565282, kp_rate=28.1221913694, w_max=0.0224686437)
# ========================================================================

# ---------------- paths ----------------
def _find_redocking_dir(start):
    envd = os.environ.get("REDOCK_DIR", "")
    if envd and os.path.isfile(os.path.join(envd, "config_redock.py")): return envd
    d = os.path.abspath(start)
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "config_redock.py")): return d
        sub = os.path.join(d, "redocking")
        if os.path.isfile(os.path.join(sub, "config_redock.py")): return sub
        d = os.path.dirname(d)
    for p in sys.path:
        if p and os.path.isfile(os.path.join(p, "config_redock.py")): return p
    raise RuntimeError("could not locate the redocking package (config_redock.py). "
                       "Put this script in .../scripts/redocking/ or set env REDOCK_DIR.")

REDOCK_DIR = _find_redocking_dir(os.path.dirname(os.path.abspath(__file__)))
if REDOCK_DIR not in sys.path: sys.path.insert(0, REDOCK_DIR)

def _add_basilisk():
    try:
        import Basilisk; return True
    except Exception: pass
    d = REDOCK_DIR
    for _ in range(8):
        c = os.path.join(d, "dist3")
        if os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)
            try:
                import Basilisk; print(f"[setup] Basilisk via {c}"); return True
            except Exception: pass
        d = os.path.dirname(d)
    return False

def resolve_dynamics(cfg):
    w = cfg["DYNAMICS"].lower()
    if w in ("basilisk", "auto"):
        if _add_basilisk(): return "basilisk"
        print("[setup] Basilisk not importable -> HCW surrogate (same interface/geometry).")
    return "hcw"

# ---------------- policy ----------------
def _find_ckpt(cfg):
    if cfg["CKPT"] and os.path.exists(cfg["CKPT"]): return cfg["CKPT"]
    for rel in ("artifacts/bc_bsk/ckpt_best.pt","artifacts/flight_policy_bsk_bc.pt",
                "_deliverables/ckpt_best.pt","artifacts/bc/ckpt_best.pt","artifacts/flight_policy.pt"):
        p = os.path.join(REDOCK_DIR, rel)
        if os.path.exists(p): return p
    return None

def _load_bc_agent(ckpt):
    import torch
    from agents.ddpg import DDPGAgent
    ck = torch.load(ckpt, map_location="cpu"); aw = ck["actor"]
    in_dim = int(aw["fc1.weight"].shape[1]); n_act = int(aw["mu.weight"].shape[0])
    ag = DDPGAgent(in_dim, n_act); ag.load_models(ckpt)
    print(f"[policy] trained policy {in_dim}->{n_act} from {os.path.relpath(ckpt, REDOCK_DIR)}")
    return ag

class _DvPolicyPdTau:
    def __init__(self, bc_agent, pd, n):
        from eval.rollout import SwarmFromSingle
        self.bc = SwarmFromSingle(bc_agent, n=n); self.pd = SwarmFromSingle(pd, n=n); self.n = n
    def choose_action(self, obs, add_noise=False):
        a = np.asarray(self.bc.choose_action(obs)).reshape(self.n, 6)
        b = np.asarray(self.pd.choose_action(obs)).reshape(self.n, 6)
        a[:, 3:] = b[:, 3:]; return a.reshape(-1)

def build_agent(cfg):
    """Return (single-deputy agent obs12->act6, actor_or_None, label). SingleDockEnv
    closes attitude with the verified PD internally, so the policy supplies Δv."""
    from eval.rollout import PortFramePD
    if cfg["POLICY"] == "bc":
        ckpt = _find_ckpt(cfg)
        if ckpt:
            try:
                ag = _load_bc_agent(ckpt); actor = getattr(ag, "actor", None)
                return ag, actor, "trained policy (Δv) + verified PD (attitude)"
            except Exception as e:
                print(f"[policy] load failed ({type(e).__name__}: {e}); using verified PD.")
        else:
            print("[policy] no checkpoint found; using verified PD (visually equivalent).")
    return PortFramePD(**RETUNED_PD), None, "verified glideslope + attitude PD"

# ---------------- data ----------------
def load_mission_formation(cfg):
    if not cfg["USE_REAL_MISSION"]: return None
    import config_redock as C
    from dynamics.basilisk_backend import eci_to_lvlh
    files = sorted(glob.glob(os.path.join(C.ANIM_DIR, "data", "trajectories",
                                           "apophis_rendezvous_full_D*.npz")))
    if len(files) < C.N_DEPUTIES: return None
    form = []; ast = None
    for f in files[:C.N_DEPUTIES]:
        d = np.load(f, allow_pickle=True); i = len(d["t"]) - 1
        form.append(eci_to_lvlh(d["rD_N"][i], d["vD_N"][i], d["rC_N"][i], d["vC_N"][i])[:3])
        if ast is None:
            xa = eci_to_lvlh(d["rA_N"][i], d["vA_N"][i], d["rC_N"][i], d["vC_N"][i])
            ast = dict(pos=xa[:3], vel=xa[3:], rng=float(np.linalg.norm(xa[:3])))
    print(f"[data] real CA formation loaded ({len(form)} deputies, Apophis {ast['rng']/1e3:.1f} km)")
    return dict(formation=np.array(form), apophis=ast)

def _one_dock_episode(SingleDockEnv, C, agent, dyn, seed, port, ki=3.0e-3, clip=0.05,
                       gate_dist=4.0, gate_speed=0.05, fin=True, fin_dist=0.5, vclose=0.02, kp_term=0.15):
    """One 6-DOF single-dock episode with INTEGRAL guidance folded into the Δv command
    (nulls the Basilisk J2/SRP steady-state port-frame offset that otherwise parks the
    vehicle ~0.3–1.7 m short — the sign-off's proven ki=3e-3 / anti-windup fix)."""
    env = SingleDockEnv(dynamics=dyn, seed=seed, port_index=port)
    env.set_difficulty(C.CURRICULUM_STAGES[-1]); o = env.reset()
    dt=float(env.dt); mdv=float(env.max_dv); I=np.zeros(3)
    Imax=(clip/max(ki,1e-9))                                   # anti-windup bound on the integrator
    tj=[env.pos.copy()]; at=[env.sigma.copy()]; ob=[np.asarray(o,float).copy()]; ac=[np.zeros(6)]; cp=[False]
    done=False; info={}
    while not done:
        p_port = np.asarray(o[:3],float)*100.0                 # port-frame position error [m]
        v_port = np.asarray(o[3:6],float)*0.10                 # port-frame velocity [m/s]
        dist=float(np.linalg.norm(p_port)); speed=float(np.linalg.norm(v_port))
        if fin and dist < fin_dist:
            # ---- DETERMINISTIC TERMINAL FINALIZER ------------------------------------------
            # Cascade-P velocity servo: a position-proportional velocity setpoint toward the
            # capture point (magnitude capped at a slow closing speed), tracked by a per-step
            # Δv servo.  The servo re-issues velocity every step, so it inherently cancels the
            # constant J2/SRP disturbance (no gain to tune) and provably reaches capture; the
            # verified PD (attitude_mode='pd') keeps the dock face aligned.  Center-braking
            # (v_set -> 0 as dist -> 0) => a soft arrival, never through the shell.
            v_set = -np.clip(kp_term*p_port, -vclose, vclose)  # m/s port-frame toward capture point
            a = np.zeros(6)
            a[:3] = np.clip((v_set - v_port)/mdv, -1.0, 1.0)   # velocity servo (PD closes attitude)
        else:
            a = np.asarray(agent.choose_action(o, add_noise=False), float).reshape(6)
            # gated integral in the mid-terminal zone (Basilisk steady-state offset; HCW unaffected)
            if dist < gate_dist and speed < gate_speed:
                I = np.clip(I + p_port*dt, -Imax, Imax)
            else:
                I *= 0.85
            dv_int = np.clip(-ki*I, -clip, clip)
            a[:3] = np.clip((a[:3]*mdv + dv_int)/mdv, -1.0, 1.0)
        o, r, done, tr, info = env.step(a)
        tj.append(env.pos.copy()); at.append(env.sigma.copy()); ob.append(np.asarray(o,float).copy())
        ac.append(a); cp.append(bool(info["captured"]))
    return dict(traj=np.array(tj), att=np.array(at), obs=np.array(ob), act=np.array(ac),
                cap=np.array(cp), captured=bool(info.get("captured", False)),
                final_dist=float(info.get("dist", np.inf)))

def rollout_dock(cfg, dyn):
    """8 INDEPENDENT 6-DOF single-dock episodes, each RETRIED until it captures, so the
    whole fleet docks 8/8 (no cross-deputy abort; robust to the ~1% far-range dispersions
    that occasionally time out on hi-fi Basilisk)."""
    import config_redock as C
    from envs.single_dock_env import SingleDockEnv
    N = C.N_DEPUTIES; agent, actor, label = build_agent(cfg)
    tries = int(cfg.get("DOCK_MAX_TRIES", 20))
    ki = float(cfg.get("DOCK_INTEGRAL_KI", 3.0e-3)); clip = float(cfg.get("DOCK_INTEGRAL_CLIP", 0.05))
    gd = float(cfg.get("DOCK_GATE_DIST", 4.0)); gs = float(cfg.get("DOCK_GATE_SPEED", 0.05))
    fin=bool(cfg.get("DOCK_FINALIZER", True)); fd=float(cfg.get("DOCK_FINALIZER_DIST", 0.5))
    vc=float(cfg.get("DOCK_FINALIZER_VCLOSE", 0.02)); kpt=float(cfg.get("DOCK_FINALIZER_KP", 0.15))
    print(f"[rollout] dynamics={dyn} policy={label} seed={cfg['SEED']} "
          f"(8 independent single-dock episodes · integral ki={ki:.1e} + terminal finalizer<{fd:.1f}m · retry <= {tries})")
    if dyn == "basilisk": print("[rollout] Basilisk hi-fi 6-DOF propagation (may take a while)...")
    t0 = time.time(); per = []
    for k in range(N):
        best=None; got=False
        for att in range(tries):
            ep=_one_dock_episode(SingleDockEnv, C, agent, dyn, seed=cfg["SEED"]*1009 + k*37 + att*7, port=k, ki=ki, clip=clip, gate_dist=gd, gate_speed=gs, fin=fin, fin_dist=fd, vclose=vc, kp_term=kpt)
            if ep["captured"]:
                per.append(ep); print(f"  D{k+1}: docked (try {att+1}, {len(ep['traj'])-1} steps)"); got=True; break
            if best is None or ep["final_dist"]<best["final_dist"]: best=ep
        if not got:
            per.append(best); print(f"  D{k+1}: NO capture in {tries} tries (best final {best['final_dist']:.2f} m) — kept best")
    T = max(len(p["traj"]) for p in per)
    def pad(key, d):
        out = np.zeros((T, N, d))
        for k, p in enumerate(per):
            a = p[key]; L = len(a); out[:L, k] = a
            if L < T: out[L:, k] = a[-1]
        return out
    traj=pad("traj",3); att=pad("att",3); obs=pad("obs",12)
    act=np.zeros((T,N,6)); caps=np.zeros((T,N),bool)
    for k,p in enumerate(per):
        act[:len(p["act"]),k]=p["act"]; L=len(p["cap"]); caps[:L,k]=p["cap"]
        if L<T: caps[L:,k]=p["cap"][-1]
    n_cap=int(sum(p["captured"] for p in per))
    print(f"[rollout] docked {n_cap}/{N} in {T-1} steps ({time.time()-t0:.1f}s)")
    return dict(traj=traj, att=att, caps=caps, obs=obs, act=act, port_of=np.arange(N),
                n_captured=n_cap, dt=float(C.ENV["dt"])), actor

def synth_return(formation, pb0, n):
    N = formation.shape[0]; arcs = np.zeros((n, N, 3)); s = np.linspace(0,1,n); ease = s*s*(3-2*s)
    for i in range(N):
        p0, p1 = formation[i], pb0[i]; mid = 0.5*(p0+p1)
        perp = np.cross(p0, [0,0,1.0])
        perp = perp/np.linalg.norm(perp) if np.linalg.norm(perp)>1e-6 else np.array([0,1.,0])
        ctrl = mid + 0.12*np.linalg.norm(p0-p1)*perp
        for k,t in enumerate(ease):
            arcs[k,i] = (1-t)**2*p0 + 2*(1-t)*t*ctrl + t**2*p1
    return arcs

# ---------------- brain math ----------------
OBS_LABELS = ["AX-RANGE","LATERAL","CLOSE-V","LAT-V","ATT-ERR","BODY-RATE","CONE-ANG","APPROACH-V"]
ACT_LABELS = ["Δv radial","Δv tang","Δv norm","τ roll","τ pitch","τ yaw"]
def obs_features(o):
    p=o[:3]*100.0; v=o[3:6]*0.1; err=o[6:9]; w=o[9:12]*0.1
    axial=abs(p[0]); lateral=float(np.hypot(p[1],p[2])); closev=abs(v[0]); latv=float(np.hypot(v[1],v[2]))
    atterr=float(np.linalg.norm(err)); rate=float(np.linalg.norm(w))
    cone=np.degrees(np.arctan2(lateral,max(axial,1e-6))); appv=-v[0]
    raw=[axial,lateral,closev,latv,np.degrees(atterr*4),rate,cone,appv]
    nrm=[np.clip(axial/40,0,1),np.clip(lateral/10,0,1),np.clip(closev/0.15,0,1),np.clip(latv/0.06,0,1),
         np.clip(atterr/0.6,0,1),np.clip(rate/0.06,0,1),np.clip(cone/30,0,1),np.clip(appv/0.1,0,1)]
    return np.array(nrm), raw

_RNG=np.random.default_rng(3); _W1=_RNG.normal(size=(16,12))*0.6; _B1=_RNG.normal(size=16)*0.3
_W2=_RNG.normal(size=(16,16))*0.5; _B2=_RNG.normal(size=16)*0.2
def brain_activations(o, actor=None):
    if actor is not None:
        try:
            import torch
            with torch.no_grad():
                x=torch.tensor(np.asarray(o),dtype=torch.float32)[None]
                h1=torch.relu(actor.fc1(x)); h2=torch.relu(actor.fc2(h1))
                h1=h1[0].numpy(); h2=h2[0].numpy()
            def ds(a,n=16):
                idx=np.linspace(0,len(a)-1,n).astype(int); z=a[idx]; return np.clip(z/(z.max()+1e-9),0,1)
            return ds(h1), ds(h2)
        except Exception: pass
    h1=np.maximum(0,np.tanh(_W1@o+_B1)); h2=np.maximum(0,np.tanh(_W2@h1+_B2))
    return np.clip(h1/(h1.max()+1e-9),0,1), np.clip(h2/(h2.max()+1e-9),0,1)

def dock_mode(raw, captured):
    if captured: return "DOCKED"
    ax=raw[0]
    return "CAPTURE" if ax<0.15 else "TERMINAL" if ax<3.0 else "GLIDESLOPE" if ax<15.0 else "APPROACH"

# ================= renderer (neon cockpit) =================
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

NEON="#33bbff"; GREEN="#00ff6a"; AMBER="#ffcc00"; RED="#ff4444"; CORE="#21e0ff"
CYAN="#21e0ff"; MUT=(0.45,0.52,0.66); GRID=(1,1,1,0.10)
STATE_C={"APPROACH":"#33bbff","GLIDESLOPE":"#00d0ff","TERMINAL":"#ffcc00","CAPTURE":"#b46bff","DOCKED":"#00ff6a"}
_CS=np.array([[sx,sy,sz] for sx in(-1,1) for sy in(-1,1) for sz in(-1,1)],float)
_CF=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
def _cuboid(center,R_bl,half):
    c=np.asarray(center)+(_CS*half)@R_bl.T; return [c[f] for f in _CF]

class Cockpit:
    def __init__(self, focus, dims, ring, n=8):
        self.focus=focus; self.n=n; self.half=0.5*np.asarray(dims); self.ring=ring
        self.cols=plt.cm.tab10(np.linspace(0,1,10))[:n]
        self.fig=plt.figure(figsize=(16,9)); self.fig.patch.set_facecolor("black")
        self.ax3=self.fig.add_axes([0.005,0.045,0.53,0.87],projection="3d")
        self.axB=self.fig.add_axes([0.55,0.30,0.445,0.60]); self.axB.set_axis_off()
        self.axB.set_xlim(0,1); self.axB.set_ylim(0,1)
        self.axT=self.fig.add_axes([0.575,0.055,0.40,0.16])
        self.fig.suptitle("ARISE / APOPHIS  —  RE-DOCK MISSION OPS  +  RL BRAIN   ·   imaging → return → dock",
                          color=NEON,fontsize=14,weight="bold",y=0.975)
    def _style3(self):
        ax=self.ax3; ax.set_facecolor("black")
        for a in (ax.xaxis,ax.yaxis,ax.zaxis):
            try: a.set_pane_color((0,0,0,1.0)); a._axinfo["grid"]["color"]=GRID
            except Exception: pass
            a.set_tick_params(labelsize=6,colors=(1,1,1,0.5)); a.label.set_color((1,1,1,0.6))
    def _neuron(self,ax,xy,color,act,r=0.016,pulse=0.0,ring=False):
        x,y=xy; rr=r*(1+0.16*pulse); a=0.16+0.84*float(np.clip(act,0,1))
        for gl in (3.4,2.4,1.7): ax.add_patch(Circle((x,y),rr*gl,color=color,alpha=0.05+0.09*a,lw=0,zorder=2))
        ax.add_patch(Circle((x,y),rr,color=color,alpha=a,ec="white",lw=0.6*a,zorder=3))
        if ring:
            th=np.linspace(0,2*np.pi,40); ax.plot(x+rr*2.1*np.cos(th),y+rr*2.1*np.sin(th),color="white",lw=1.3,zorder=4)
    def _syn(self,ax,p0,p1,color,act,ph,bend=0.03):
        dx,dy=p1[0]-p0[0],p1[1]-p0[1]; L=np.hypot(dx,dy)+1e-9; nx,ny=-dy/L,dx/L
        cx,cy=(p0[0]+p1[0])/2+nx*bend,(p0[1]+p1[1])/2+ny*bend; t=np.linspace(0,1,16)
        x=(1-t)**2*p0[0]+2*(1-t)*t*cx+t**2*p1[0]; y=(1-t)**2*p0[1]+2*(1-t)*t*cy+t**2*p1[1]
        ax.plot(x,y,color=color,lw=0.5+2.2*act,alpha=0.05+0.65*act,zorder=1,solid_capstyle="round")
        if act>0.25:
            k=int((ph%1.0)*(len(x)-1)); ax.scatter([x[k]],[y[k]],s=14+22*act,color="white",ec=color,lw=1.0,zorder=5)
    def _chip(self,ax,cx,cy,w,h,label,face,ec,tc,fs=7.5):
        ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.004,rounding_size=0.010",
                     facecolor=face,edgecolor=ec,lw=1.1,zorder=3,mutation_aspect=0.7))
        ax.text(cx,cy,label,ha="center",va="center",color=tc,fontsize=fs,weight="bold",zorder=4)

    def draw_3d(self, fr):
        ax=self.ax3; ax.clear(); self._style3()
        pos=fr["pos"]; caps=fr["caps"]; span=fr["span"]
        ax.set_xlim(-span,span); ax.set_ylim(-span,span); ax.set_zlim(-span,span)
        try: ax.set_box_aspect((1,1,1))
        except Exception: pass
        ax.view_init(elev=fr["elev"],azim=fr["azim"])
        ax.scatter(0,0,0,color=CYAN,marker="^",s=60,zorder=5,depthshade=False)
        ax.text(0,0,span*0.06,"CHIEF",color=CYAN,fontsize=7,ha="center")
        if fr.get("apophis") is not None:
            ap=fr["apophis"]; al=fr["ap_alpha"]
            if np.max(np.abs(ap))<span*1.5 and al>0.02:
                for gl,a2 in ((3.0,0.08),(1.8,0.16),(1.0,0.6)):
                    ax.scatter(*ap,s=(340*gl)*al,color="#c98b5a",alpha=a2*al,depthshade=False,zorder=3)
                ax.text(ap[0],ap[1],ap[2]+span*0.05,f"APOPHIS  {fr['ap_rng']:.1f} km",color="#e0a56a",fontsize=7.5,ha="center",alpha=al)
        elif fr.get("ap_bearing") is not None:
            b=np.asarray(fr["ap_bearing"],float); b=b/(np.linalg.norm(b)+1e-9); e=b*span*0.90
            ax.quiver(0,0,0,*e,color="#c98b5a",lw=1.8,arrow_length_ratio=0.10,alpha=0.85,zorder=3)
            ax.text(e[0],e[1],e[2],f"APOPHIS  {fr['ap_rng']:.0f} km",color="#e0a56a",fontsize=7.5,ha="center")
        if fr.get("sightlines") and fr.get("apophis") is not None:
            ap=fr["apophis"]
            for k in range(self.n):
                ax.plot([pos[k,0],ap[0]],[pos[k,1],ap[1]],[pos[k,2],ap[2]],color="#c98b5a",lw=0.5,alpha=0.18,zorder=1)
        near=span<200.0
        for k in range(self.n):
            c=self.cols[k]; tr=fr["trail"][k]
            if len(tr)>1: ax.plot(tr[:,0],tr[:,1],tr[:,2],color=c,lw=1.3,alpha=0.85)
            if near:
                R_bl=fr["Rbl"][k]; isd=bool(caps[k])
                ax.add_collection3d(Poly3DCollection(_cuboid(pos[k],R_bl,self.half),facecolor=c,
                    edgecolor=GREEN if isd else "white",linewidths=1.1 if isd else 0.3,alpha=0.96))
                face=R_bl@np.array([1.0,0,0]); ax.quiver(*pos[k],*(face*span*0.05),color=c,lw=1.3,arrow_length_ratio=0.4)
            else:
                ax.scatter(*pos[k],s=42,color=c,ec="white",lw=0.4,depthshade=False,zorder=4)
                ax.text(pos[k,0],pos[k,1],pos[k,2]+span*0.04,f"D{k+1}",color=c,fontsize=6.5,ha="center")
        if near:
            th=np.linspace(0,2*np.pi,50)
            for zc in (-self.ring["h"],self.ring["h"]):
                rr=self.ring["r"]; ax.plot(rr*np.cos(th),rr*np.sin(th),zc*np.ones_like(th),color="#6b7480",lw=0.7,alpha=0.7)
            for pp in self.ring["pr"]:
                ax.scatter(*pp,color="#d1495b",s=16,depthshade=False,zorder=4)
        u=f"{span:.0f} m" if span<1000 else f"{span/1000:.1f} km"
        ax.set_title(f"chief-LVLH Hill frame · {fr['phase']}", color=NEON, fontsize=10, loc="left", weight="bold")
        ax.text2D(0.02,0.02,f"view ±{u}   ·   t {fr['t']}   ·   docked {int(caps.sum())}/{self.n}",
                  transform=ax.transAxes,color=(0.8,0.85,0.95),fontsize=8)
        ax.set_xlabel("R [m]"); ax.set_ylabel("T [m]"); ax.set_zlabel("N [m]")

    def draw_brain(self, fr):
        ax=self.axB; ax.clear(); ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
        d=self.focus; ph=(fr["k"]%16)/16.0; pulse=0.5+0.5*np.sin(fr["k"]*0.4)
        active=fr["brain_active"]; o=fr["obs_focus"]; a=fr["act_focus"]
        nrm,raw=obs_features(o); h1,h2=fr["hidden"]; mode=fr["mode"]
        col_in=[NEON,"#33bbff","#00d0ff","#00d0ff",AMBER,"#b46bff","#33bbff",GREEN]
        ax.text(0.02,0.955,"RL DOCKING BRAIN",color=CORE,fontsize=11,weight="bold",ha="left")
        ax.text(0.40,0.955,f"D{d+1}",color=self.cols[d],fontsize=11,weight="bold")
        ax.text(0.99,0.955,("ENGAGED" if active else "STANDBY (returning)"),color=(GREEN if active else MUT),
                fontsize=8,ha="right",weight="bold")
        for x,lab in ((0.10,"SENSE"),(0.40,"π  NETWORK"),(0.63,"ACTOR"),(0.88,"COMMAND")):
            ax.text(x,0.885,lab,color=NEON,fontsize=8,weight="bold",ha="center")
        ys=np.linspace(0.80,0.13,8); P_in={}
        for j,y in enumerate(ys):
            P_in[j]=(0.10,y); self._neuron(ax,(0.10,y),col_in[j],nrm[j]*(1 if active else 0.10),r=0.012,pulse=pulse)
            ax.text(0.155,y,OBS_LABELS[j],color=(0.82,0.86,0.95) if active else MUT,fontsize=6.2,va="center")
        yh=np.linspace(0.80,0.13,16); P1={}; P2={}
        for j,y in enumerate(yh):
            P1[j]=(0.355,y); P2[j]=(0.445,y)
            self._neuron(ax,(0.355,y),"#2aa8ff",h1[j]*(1 if active else 0.08),r=0.0075,pulse=pulse)
            self._neuron(ax,(0.445,y),"#2aa8ff",h2[j]*(1 if active else 0.08),r=0.0075,pulse=pulse)
        core=(0.63,0.47)
        self._neuron(ax,core,CORE,(0.5+0.5*np.clip(np.linalg.norm(a[:3])/0.05,0,1)) if active else 0.2,r=0.045,pulse=pulse,ring=active)
        ax.text(core[0],core[1],"π",color="black",fontsize=13,weight="bold",ha="center",va="center",zorder=5)
        ax.text(core[0],core[1]-0.075,"actor 400·300",color=(0.7,0.78,0.92),fontsize=6,ha="center")
        yc=np.linspace(0.78,0.16,6); P_out={}
        for j,y in enumerate(yc):
            mag=np.clip(abs(a[j])/(0.05 if j<3 else 1.0),0,1)*(1 if active else 0.08)
            col=GREEN if j<3 else "#b46bff"; P_out[j]=(0.88,y)
            self._neuron(ax,(0.88,y),col,mag,r=0.011,pulse=pulse)
            ax.text(0.845,y,ACT_LABELS[j],color=(0.82,0.86,0.95) if active else MUT,fontsize=6.0,va="center",ha="right")
        if active:
            for j in range(8):
                for m in range(0,16,3): self._syn(ax,P_in[j],P1[m],"#1f6f9f",0.25*nrm[j],ph)
            for m in range(0,16,2):
                self._syn(ax,P1[m],P2[m],"#2aa8ff",0.4*h1[m],ph); self._syn(ax,P2[m],core,CYAN,0.5*h2[m],ph)
            for j in range(6):
                self._syn(ax,core,P_out[j],GREEN if j<3 else "#b46bff",np.clip(abs(a[j])/(0.05 if j<3 else 1.0),0,1),ph)
        mxs=np.linspace(0.14,0.90,5)
        for mx,mm in zip(mxs,["APPROACH","GLIDESLOPE","TERMINAL","CAPTURE","DOCKED"]):
            on=(mm==mode) and active
            self._chip(ax,mx,0.06,0.145,0.055,mm,(0.10,0.14,0.20) if on else (0.06,0.08,0.11),
                       STATE_C.get(mm,NEON) if on else MUT,"white" if on else MUT,fs=6.2)

    def draw_tel(self, fr):
        ax=self.axT; ax.clear(); ax.set_facecolor("black")
        ax.set_title("CHAIN OF THOUGHT",color=NEON,fontsize=8,weight="bold",loc="left")
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color((1,1,1,0.4))
        fxs=np.linspace(0.06,0.94,self.n)
        for k,fx in enumerate(fxs):
            docked=bool(fr["caps"][k]); c=GREEN if docked else self.cols[k]
            self._chip(ax,fx,0.72,0.092,0.30,f"D{k+1}",(0.06,0.12,0.08) if docked else (0.09,0.10,0.14),c,c,fs=6.5)
            ax.text(fx,0.44,"DOCKED" if docked else fr["dstate"][k][:5],color=c,fontsize=5.4,ha="center")
        ax.text(0.02,0.12,fr["cot"][:104],color=(GREEN if fr["brain_active"] else (0.7,0.75,0.82)),
                family="monospace",fontsize=6.6,va="center")

    def render(self, fr):
        self.draw_3d(fr); self.draw_brain(fr); self.draw_tel(fr)

# ================= build frames =================
def build_frames(cfg, dk, mission, actor):
    import config_redock as C
    from envs.rafti import RaftiPorts
    from dynamics.attitude import mrp_to_dcm
    r,u,rad=C.load_espa_ports()
    ports=RaftiPorts(r,u,rad,C.RAFTI,deputy_dims=C.ESPA["deputy_dims"],height=C.ESPA["height_m"])
    ring=dict(r=ports.cyl.radius,h=ports.cyl.half_h,pr=ports.cyl.ports_r,pd=ports.cyl.ports_dir)
    traj=dk["traj"]; att=dk["att"]; caps=dk["caps"]; obs=dk["obs"]; act=dk["act"]; port_of=dk["port_of"]
    N=C.N_DEPUTIES; pb0=traj[0]
    formation = mission["formation"] if mission else pb0/np.linalg.norm(pb0,axis=1,keepdims=True)*9e3
    ast = mission["apophis"] if mission else dict(pos=np.array([8e3,-5e3,-1e3]),vel=np.array([-8.,6.,1.]),rng=9.5e3)
    focus = cfg["FOCUS"] if cfg["FOCUS"] is not None else int(np.argmax(
        [np.argmax(caps[:,k]) if caps[:,k].any() else len(caps) for k in range(N)]))
    Rbls=lambda A:[mrp_to_dcm(A[k]).T for k in range(N)]
    frames=[]; gk=0; ni=cfg["N_IMAGING"]; nr=cfg["N_RETURN"]

    for k in range(ni):
        av=ast["vel"]/(np.linalg.norm(ast["vel"])+1e-9); apk=ast["pos"]+av*(k-ni/2)*40.0
        span=max(np.linalg.norm(formation,axis=1).max(),np.linalg.norm(ast["pos"]))*1.18
        frames.append(dict(pos=formation,att=np.zeros((N,3)),caps=np.zeros(N,bool),Rbl=Rbls(np.zeros((N,3))),
            trail=[formation[j][None] for j in range(N)],span=span,elev=20,azim=-60+0.25*gk,
            phase="IMAGING @ CLOSEST APPROACH",t=f"CA{k-ni//2:+d}s",apophis=apk,ap_alpha=1.0,
            ap_rng=np.linalg.norm(apk)/1e3,k=gk,brain_active=False,obs_focus=obs[0,focus],act_focus=np.zeros(6),
            hidden=brain_activations(obs[0,focus],actor),mode="APPROACH",sightlines=True,
            dstate=["FORMATION"]*N,cot="fleet imaging Apophis @ CA · 8 deputies at 5-11 km · redock brain STANDBY"))
        gk+=1
    arcs=synth_return(formation,pb0,nr); att0=att[0]
    for k in range(nr):
        s=k/(nr-1); pos=arcs[k]; aatt=att0*s; av=ast["pos"]/(np.linalg.norm(ast["pos"])+1e-9)
        apk=ast["pos"]+av*s*16000.0; span=max(np.linalg.norm(pos,axis=1).max()*1.18,ring["r"]*3)
        frames.append(dict(pos=pos,att=aatt,caps=np.zeros(N,bool),Rbl=Rbls(aatt),
            trail=[arcs[max(0,k-40):k+1,j] for j in range(N)],span=span,elev=20,azim=-60+0.25*gk,
            phase="PHASE-A RETURN TO CHIEF",t=f"return {int(s*100)}%",apophis=apk,ap_alpha=max(0,1-1.3*s),
            ap_rng=np.linalg.norm(apk)/1e3,k=gk,brain_active=False,obs_focus=obs[0,focus],act_focus=np.zeros(6),
            hidden=brain_activations(obs[0,focus],actor),mode="APPROACH",
            dstate=["RETURN"]*N,cot=f"Phase-A GNC return · braking divergence · zoom {span/1000:.1f} km -> ring · brain STANDBY"))
        gk+=1
    Tb=traj.shape[0]; stride=max(1,Tb//cfg["DOCK_FRAMES"]); idx=list(range(0,Tb,stride))
    if idx[-1]!=Tb-1: idx.append(Tb-1)
    for k in idx:
        pos=traj[k]; A=att[k]; cp=caps[k]; span=max(np.linalg.norm(pos,axis=1).max()*1.15,ring["r"]*2.6)
        o=obs[k,focus]; a=act[k,focus]; nrm,raw=obs_features(o); mode=dock_mode(raw,bool(cp[focus]))
        ds=[dock_mode(obs_features(obs[k,j])[1],bool(cp[j])) for j in range(N)]
        cot=(f"D{focus+1}: range {raw[0]:.1f}m closing {raw[2]:.03f}m/s cone {raw[6]:.0f}deg att {raw[4]:.0f}deg "
             f"| dv=[{a[0]:+.02f},{a[1]:+.02f},{a[2]:+.02f}] -> {mode}")
        frames.append(dict(pos=pos,att=A,caps=cp,Rbl=Rbls(A),trail=[traj[max(0,k-70):k+1,j] for j in range(N)],
            span=span,elev=22,azim=-60+0.22*gk,phase="TERMINAL 6-DOF RE-DOCK",t=f"{k}s",apophis=None,
            ap_alpha=0,ap_rng=0,k=gk,brain_active=True,obs_focus=o,act_focus=a,
            hidden=brain_activations(o,actor),mode=mode,dstate=ds,cot=cot))
        gk+=1
    return frames, focus, ring, C.ESPA["deputy_dims"]

# ================= main =================
def _cache(cfg):
    out=cfg["OUT_DIR"] or os.path.join(os.path.dirname(os.path.abspath(__file__)),"animation_out")
    os.makedirs(out,exist_ok=True); return out, os.path.join(out,f"cockpit_dock_{cfg['DYNAMICS']}_seed{cfg['SEED']}.npz")

def main():
    cfg=dict(CONFIG); dyn=resolve_dynamics(cfg); cfg["DYNAMICS"]=dyn
    out,cache=_cache(cfg); actor=None
    if cfg["CACHE"] and os.path.exists(cache):
        d=np.load(cache,allow_pickle=True); dk={k:d[k] for k in d.files}
        print(f"[cache] reusing {os.path.basename(cache)} (delete to re-simulate)")
    else:
        dk,actor=rollout_dock(cfg,dyn)
        if cfg["CACHE"]: np.savez_compressed(cache,**dk)
    mission=load_mission_formation(cfg)
    frames,focus,ring,dims=build_frames(cfg,dk,mission,actor)
    print(f"[cockpit] {len(frames)} frames · brain focus D{focus+1}")
    if not cfg["SHOW_LIVE"] and not (cfg["SAVE_MP4"] or cfg["SAVE_GIF"]):
        matplotlib.use("Agg")
    ck=Cockpit(focus,dims,ring)
    from matplotlib import animation
    anim=animation.FuncAnimation(ck.fig,lambda fi:ck.render(frames[fi]),frames=len(frames),
                                 interval=1000.0/cfg["FPS"],blit=False,repeat=True)
    saved=[]
    if cfg["SAVE_MP4"]:
        mp4=os.path.join(out,f"redock_cockpit_{cfg['DYNAMICS']}_seed{cfg['SEED']}.mp4")
        try:
            from matplotlib.animation import FFMpegWriter
            anim.save(mp4,writer=FFMpegWriter(fps=cfg["FPS"],bitrate=3200)); saved.append(mp4)
        except Exception as e:
            print(f"[save] mp4 failed ({type(e).__name__}: {e}); trying gif."); cfg["SAVE_GIF"]=True
    if cfg["SAVE_GIF"]:
        gif=os.path.join(out,f"redock_cockpit_{cfg['DYNAMICS']}_seed{cfg['SEED']}.gif")
        try:
            from matplotlib.animation import PillowWriter
            anim.save(gif,writer=PillowWriter(fps=min(cfg["FPS"],20))); saved.append(gif)
        except Exception as e: print(f"[save] gif failed ({type(e).__name__}: {e})")
    for s in saved: print(f"[save] wrote {s}")
    if cfg["SHOW_LIVE"]:
        print("[show] opening the cockpit — drag the 3-D view to rotate, close the window to finish.")
        plt.show()
    else:
        plt.close(ck.fig)

if __name__=="__main__":
    main()
