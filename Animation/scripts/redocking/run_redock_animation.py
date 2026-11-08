# -*- coding: utf-8 -*-
r"""
run_redock_animation.py
=======================
Complete 6-DOF ARISE re-docking animation on the **Basilisk hi-fi** dynamics.

Just press *Run* in VS Code / Visual Studio (no command-line arguments needed).
It rolls out the terminal re-dock of the ESPA fleet onto the RAFTI ports using the
SAME verified pipeline as the V&V campaigns (`MultiChaserRedock` on the Basilisk
6-DOF backend), then opens an interactive 3-D window showing:

  * the ESPA drum + 8 RAFTI ports (outward normals),
  * each 6U deputy cuboid flying in AND slewing its docking face onto its port,
  * fading approach trails, an orbiting camera and an auto-zoom to soft-capture,
  * live telemetry: range-to-port, dock-face pointing error, and docked count.

DYNAMICS
  Default "basilisk" = the flight-fidelity force model (Earth J2..N + Sun + Moon
  from SPICE + cannonball SRP, RKF45) exactly as in the hi-fi V&V.  Needs Basilisk
  importable locally (this script auto-adds <basilisk-develop>/dist3 to sys.path).
  If Basilisk is not found it AUTOMATICALLY falls back to the fast HCW surrogate
  (same interface, same geometry) so the animation always runs.

POLICY
  "bc" = the trained/behavior-cloned docking policy drives translation (Delta-v)
  while the verified attitude PD closes the docking-face pointing (exactly how the
  policy is deployed in flight).  Falls back to the pure verified PD if the policy
  checkpoint / PyTorch are unavailable.  Both dock the full fleet.

Everything is controlled by the CONFIG block just below — edit and re-Run.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

# =====================================================================
# CONFIG  — edit here, then press Run.  No command-line arguments used.
# =====================================================================
CONFIG = dict(
    SCENE          = "fleet",     # "fleet" = all 8 deputies | "single" = one deputy
    DYNAMICS       = "basilisk",  # "basilisk" (hi-fi) | "hcw" (fast) | "auto" (basilisk, else hcw)
    POLICY         = "bc",        # "bc" (trained policy + verified PD attitude) | "pd" (verified PD only)
    SEED           = 7,           # RNG seed for the dispersed start geometry
    SHOW_LIVE      = True,        # open the interactive 3-D window (plt.show)
    SAVE_MP4       = False,       # also write an .mp4 (auto-falls back to .gif if no ffmpeg)
    SAVE_GIF       = False,       # also write an animated .gif (needs only Pillow)
    OUT_DIR        = None,        # where to save mp4/gif (default: ./animation_out next to this file)
    FPS            = 25,          # animation frames per second
    TARGET_FRAMES  = 220,         # decimate the rollout to ~this many frames (speed vs smoothness)
    TRAIL          = 70,          # length of the fading approach trail (frames)
    ROTATE_CAM     = True,        # slowly orbit the camera during the approach
    AUTO_ZOOM      = True,        # zoom from the wide approach in to soft-capture
    CACHE          = True,        # cache the rollout to .npz and reuse it (delete to re-run the sim)
    CKPT           = None,        # explicit policy checkpoint; None = auto-find the BC policy
)
# Verified PD teacher gains, re-tuned for Basilisk (pd_tune/best_gains.json).
RETUNED_PD = dict(k_lat=0.3504371882, k_ax=0.0931707341, kd=9.9234303808,
                  k_slew=0.3192565282, kp_rate=28.1221913694, w_max=0.0224686437)
# =====================================================================


# ------------------------------------------------------------------ paths
def _find_redocking_dir(start: str) -> str:
    """Locate the `redocking` package (this file may live inside it or beside it)."""
    envd = os.environ.get("REDOCK_DIR", "")
    if envd and os.path.isfile(os.path.join(envd, "config_redock.py")):
        return envd
    d = os.path.abspath(start)
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "config_redock.py")):
            return d
        sub = os.path.join(d, "redocking")
        if os.path.isfile(os.path.join(sub, "config_redock.py")):
            return sub
        d = os.path.dirname(d)
    for p in sys.path:
        if p and os.path.isfile(os.path.join(p, "config_redock.py")):
            return p
    raise RuntimeError("could not locate the redocking package (config_redock.py). "
                       "Put this script in .../scripts/redocking/ or set env REDOCK_DIR.")

REDOCK_DIR = _find_redocking_dir(os.path.dirname(os.path.abspath(__file__)))
if REDOCK_DIR not in sys.path:
    sys.path.insert(0, REDOCK_DIR)


def _add_basilisk_to_path(verbose=True):
    """Make `import Basilisk` work locally with no env vars: try as-installed, else
    add <...>/dist3 (Basilisk's build output) discovered by walking up the tree."""
    try:
        import Basilisk  # noqa: F401
        return True
    except Exception:
        pass
    cands = []
    d = REDOCK_DIR
    for _ in range(8):
        cands.append(os.path.join(d, "dist3"))
        d = os.path.dirname(d)
    cands.append(os.environ.get("BASILISK_DIST3", ""))
    for c in cands:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)
            try:
                import Basilisk  # noqa: F401
                if verbose:
                    print(f"[setup] Basilisk found via {c}")
                return True
            except Exception:
                continue
    return False


# ------------------------------------------------------------------ policy
def _load_bc_agent(ckpt):
    """Load the behavior-cloned docking policy (12->6) into a DDPGAgent."""
    import torch
    from agents.ddpg import DDPGAgent
    ck = torch.load(ckpt, map_location="cpu"); aw = ck["actor"]
    in_dim = int(aw["fc1.weight"].shape[1]); n_act = int(aw["mu.weight"].shape[0])
    ag = DDPGAgent(in_dim, n_act); ag.load_models(ckpt)
    print(f"[policy] loaded trained policy {in_dim}->{n_act} from {os.path.relpath(ckpt, REDOCK_DIR)}")
    return ag, in_dim


def _find_ckpt(cfg):
    if cfg["CKPT"] and os.path.exists(cfg["CKPT"]):
        return cfg["CKPT"]
    for rel in ("artifacts/bc_bsk/ckpt_best.pt", "artifacts/flight_policy_bsk_bc.pt",
                "_deliverables/ckpt_best.pt", "artifacts/bc/ckpt_best.pt",
                "artifacts/flight_policy.pt"):
        p = os.path.join(REDOCK_DIR, rel)
        if os.path.exists(p):
            return p
    return None


class _DvPolicyPdTau:
    """Faithful flight deployment: the trained policy commands translation (Delta-v),
    the verified attitude PD commands the docking-face pointing (torque)."""
    def __init__(self, bc_agent, pd, n, goal_dim=0):
        from eval.rollout import SwarmFromSingle
        self.bc = SwarmFromSingle(bc_agent, n=n, goal_dim=goal_dim)
        self.pd = SwarmFromSingle(pd, n=n)
        self.n = n
    def choose_action(self, obs, add_noise=False):
        a_bc = np.asarray(self.bc.choose_action(obs)).reshape(self.n, 6)
        a_pd = np.asarray(self.pd.choose_action(obs)).reshape(self.n, 6)
        out = a_bc.copy(); out[:, 3:] = a_pd[:, 3:]     # dv from policy, tau from verified PD
        return out.reshape(-1)


class _SingleDvPdTau:
    """Single-deputy version of the Delta-v-policy / PD-attitude split."""
    def __init__(self, bc_agent, pd):
        self.bc = bc_agent; self.pd = pd
    def choose_action(self, obs, add_noise=False):
        a_bc = np.asarray(self.bc.choose_action(obs, add_noise=False)).reshape(6)
        a_pd = np.asarray(self.pd.choose_action(obs)).reshape(6)
        out = a_bc.copy(); out[3:] = a_pd[3:]
        return out


def _build_agent(cfg, N):
    """Return (agent, label) for the chosen policy, with safe fallback to the PD."""
    from eval.rollout import PortFramePD, SwarmFromSingle
    pd = PortFramePD(**RETUNED_PD)
    if cfg["POLICY"] == "bc":
        ckpt = _find_ckpt(cfg)
        if ckpt is not None:
            try:
                agent, _ = _load_bc_agent(ckpt)
                if N > 1:
                    return _DvPolicyPdTau(agent, pd, n=N), "trained policy (dv) + verified PD (attitude)"
                return _SingleDvPdTau(agent, pd), "trained policy (dv) + verified PD (attitude)"
            except Exception as e:
                print(f"[policy] could not load trained policy ({type(e).__name__}: {e}); using verified PD.")
        else:
            print("[policy] no policy checkpoint found; using verified PD (visually equivalent, MSE~1e-5).")
    if N > 1:
        return SwarmFromSingle(pd, n=N), "verified glideslope + attitude PD"
    return pd, "verified glideslope + attitude PD"


# ------------------------------------------------------------------ rollout
def resolve_dynamics(cfg):
    want = cfg["DYNAMICS"].lower()
    if want in ("basilisk", "auto"):
        if _add_basilisk_to_path():
            return "basilisk"
        if want == "basilisk":
            print("[setup] Basilisk not importable here -> falling back to the fast HCW surrogate "
                  "(same interface/geometry).")
        else:
            print("[setup] Basilisk not importable here -> using HCW.")
        return "hcw"
    return "hcw"


def rollout(cfg):
    """Run ONE re-dock episode and record the full 6-DOF history."""
    import config_redock as C
    dyn = resolve_dynamics(cfg)
    N = 8 if cfg["SCENE"] == "fleet" else 1
    ov = C.stage_overrides(C.CURRICULUM_STAGES[-1])          # flight-grade regime
    agent, label = _build_agent(cfg, N)
    print(f"[rollout] scene={cfg['SCENE']} dynamics={dyn} policy={label} seed={cfg['SEED']}")
    if dyn == "basilisk":
        print("[rollout] running the Basilisk hi-fi 6-DOF propagation (this can take a while)...")
    t_start = time.time()

    if N > 1:
        from envs.multi_chaser_env import MultiChaserRedock
        env = MultiChaserRedock(dynamics=dyn, seed=cfg["SEED"],
                                env_cfg=ov["env_cfg"], handoff=ov["handoff"], rafti=ov["rafti"])
        obs, info = env.reset()
        traj, att, caps, dvs = [env.positions_lvlh().copy()], [env.attitudes().copy()], \
                               [env.captured.copy()], [0.0]
        done = False
        while not done:
            a = agent.choose_action(obs, add_noise=False)
            obs, r, done, tr, info = env.step(a)
            traj.append(env.positions_lvlh().copy()); att.append(env.attitudes().copy())
            caps.append(env.captured.copy()); dvs.append(info.get("dv_used", 0.0))
        port_of = env.port_of.copy()
        n_captured = int(info["n_captured"])
    else:
        from envs.single_dock_env import SingleDockEnv
        env = SingleDockEnv(dynamics=dyn, seed=cfg["SEED"]); env.set_difficulty(C.CURRICULUM_STAGES[-1])
        obs = env.reset()
        traj, att, caps, dvs = [env.pos.copy()[None]], [env.sigma.copy()[None]], \
                               [np.array([False])], [0.0]
        done = False; info = {}
        while not done:
            a = agent.choose_action(obs, add_noise=False)
            obs, r, done, tr, info = env.step(a)
            traj.append(env.pos.copy()[None]); att.append(env.sigma.copy()[None])
            caps.append(np.array([bool(info["captured"])]))
            dvs.append(float(np.linalg.norm(a[:3]) * env.max_dv))
        port_of = np.array([env.port])
        n_captured = int(info.get("captured", False))

    data = dict(traj=np.array(traj), att=np.array(att), caps=np.array(caps),
                dv_step=np.array(dvs), port_of=np.asarray(port_of, int),
                dt=float(C.ENV["dt"]), n_captured=n_captured, N=N,
                dyn=dyn, policy=label, seed=cfg["SEED"])
    print(f"[rollout] captured {n_captured}/{N} in {data['traj'].shape[0]-1} steps "
          f"({time.time()-t_start:.1f}s wall)")
    return data


def _cache_path(cfg):
    out = cfg["OUT_DIR"] or os.path.join(os.path.dirname(os.path.abspath(__file__)), "animation_out")
    os.makedirs(out, exist_ok=True)
    return out, os.path.join(out, f"redock_{cfg['SCENE']}_{cfg['DYNAMICS']}_seed{cfg['SEED']}.npz")


def get_rollout(cfg):
    out, cache = _cache_path(cfg)
    if cfg["CACHE"] and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        print(f"[cache] reusing {os.path.basename(cache)} (delete it to re-run the sim)")
        return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}, out
    data = rollout(cfg)
    if cfg["CACHE"]:
        np.savez_compressed(cache, **data)
    return data, out


# ------------------------------------------------------------------ geometry helpers
_CUBE_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
_CUBE_FACES = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]


def _cuboid_faces(center, R_bl, half):
    c = np.asarray(center) + (_CUBE_SIGNS * half) @ R_bl.T
    return [c[f] for f in _CUBE_FACES]


# ------------------------------------------------------------------ animation
def build_animation(data, cfg):
    import matplotlib
    if not cfg["SHOW_LIVE"]:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import config_redock as C
    from envs.rafti import RaftiPorts
    from dynamics.attitude import mrp_to_dcm, pointing_error_deg

    r, u, rad = C.load_espa_ports()
    ports = RaftiPorts(r, u, rad, C.RAFTI, deputy_dims=C.ESPA["deputy_dims"], height=C.ESPA["height_m"])
    cyl, dep = ports.cyl, ports.dep
    half = dep.half

    traj, att, caps = data["traj"], data["att"], data["caps"]
    port_of = data["port_of"]; N = int(data["N"]); dt = float(data["dt"])
    T = traj.shape[0]
    stride = max(1, T // int(cfg["TARGET_FRAMES"]))
    frames = list(range(0, T, stride))
    if frames[-1] != T - 1:
        frames.append(T - 1)
    cols = plt.cm.tab10(np.linspace(0, 1, 10))[:N]

    # ---- precompute telemetry series ----
    cap_pt = ports.capture_pt[port_of]                       # (N,3) target centres
    rng = np.linalg.norm(traj - cap_pt[None, :, :], axis=2)  # (T,N) range to capture
    perr = np.zeros((T, N))
    for t in range(T):
        for k in range(N):
            perr[t, k] = pointing_error_deg(att[t, k], -ports.dir_B[port_of[k]])
    docked = caps.sum(axis=1)                                # (T,)
    tsec = np.arange(T) * dt

    # ---- figure ----
    fig = plt.figure(figsize=(15.5, 8.4)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 2, width_ratios=[2.35, 1.0], hspace=0.42, wspace=0.16,
                          left=0.02, right=0.975, top=0.93, bottom=0.07)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    axR = fig.add_subplot(gs[0, 1]); axP = fig.add_subplot(gs[1, 1]); axC = fig.add_subplot(gs[2, 1])
    fig.suptitle("ARISE 6-DOF re-docking - Basilisk hi-fi" if data["dyn"] == "basilisk"
                 else "ARISE 6-DOF re-docking - HCW surrogate",
                 fontsize=15, fontweight="bold")

    # static drum + ports
    X, Y, Z = cyl.surface_mesh(n_theta=60, n_z=2)
    ax3d.plot_surface(X, Y, Z, color="#9aa7b4", alpha=0.20, linewidth=0, shade=True)
    ang = np.linspace(0, 2 * np.pi, 60)
    for zc in (-cyl.half_h, cyl.half_h):
        ring = (cyl.radius * np.cos(ang)[:, None] * cyl.e1 + cyl.radius * np.sin(ang)[:, None] * cyl.e2
                + zc * cyl.axis)
        ax3d.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#6b7480", lw=0.8, alpha=0.7)
    for i in range(cyl.P):
        p = cyl.surface_point(i); n = cyl.port_normal(i); assigned = i in port_of
        ax3d.scatter(*p, color="#d1495b" if assigned else "#b8b8b8",
                     s=26 if assigned else 12, depthshade=False)
        ax3d.quiver(*p, *(n * 0.55), color="#d1495b" if assigned else "#c4c4c4",
                    lw=1.1, arrow_length_ratio=0.35)

    # static telemetry curves
    for k in range(N):
        axR.semilogy(tsec, rng[:, k], color=cols[k], lw=1.3)
        axP.plot(tsec, perr[:, k], color=cols[k], lw=1.3)
    axR.axhline(C.RAFTI["capture_pos_tol_m"], color="k", ls="--", lw=0.8)
    axR.set_ylabel("range to port [m]"); axR.set_title("Approach range", fontsize=10); axR.grid(alpha=0.25)
    axP.axhline(C.RAFTI["att_align_tol_deg"], color="k", ls="--", lw=0.8)
    axP.set_ylabel("dock-face err [deg]"); axP.set_title("Attitude slew (6-DOF)", fontsize=10); axP.grid(alpha=0.25)
    axC.fill_between(tsec, docked, step="pre", color="#2E8B57", alpha=0.35)
    axC.plot(tsec, docked, color="#2E8B57", lw=1.6, drawstyle="steps-pre")
    axC.set_ylim(0, N + 0.5); axC.set_ylabel("docked"); axC.set_xlabel("time [s]")
    axC.set_title("Soft-captures", fontsize=10); axC.grid(alpha=0.25)
    curR = axR.axvline(0, color="#333", lw=1.0); curP = axP.axvline(0, color="#333", lw=1.0)
    curC = axC.axvline(0, color="#333", lw=1.0)

    ax3d.set_xlabel("R [m]"); ax3d.set_ylabel("T [m]"); ax3d.set_zlabel("N [m]")
    try:
        ax3d.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    span0 = float(np.linalg.norm(traj[0], axis=1).max()) * 1.15

    dyn_artists = []

    def update(fi):
        t = frames[fi]
        while dyn_artists:
            a = dyn_artists.pop()
            try:
                a.remove()
            except Exception:
                pass
        t0 = max(0, t - int(cfg["TRAIL"]))
        for k in range(N):
            seg = traj[t0:t + 1, k]
            if len(seg) > 1:
                ln, = ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=cols[k], lw=1.5, alpha=0.9)
                dyn_artists.append(ln)
            R_bl = mrp_to_dcm(att[t, k]).T                   # body -> LVLH
            is_dock = bool(caps[t, k])
            pc = Poly3DCollection(_cuboid_faces(traj[t, k], R_bl, half),
                                  facecolor=cols[k], edgecolor="#1f7a3d" if is_dock else "k",
                                  linewidths=1.2 if is_dock else 0.3, alpha=0.96)
            ax3d.add_collection3d(pc); dyn_artists.append(pc)
            face = R_bl @ np.array([1.0, 0, 0])              # +x_body (dock face) in LVLH
            q = ax3d.quiver(*traj[t, k], *(face * 0.9), color=cols[k], lw=1.4, arrow_length_ratio=0.35)
            dyn_artists.append(q)

        if cfg["AUTO_ZOOM"]:
            span = max(float(np.linalg.norm(traj[t], axis=1).max()) * 1.15, cyl.radius * 3.0)
        else:
            span = span0
        ax3d.set_xlim(-span, span); ax3d.set_ylim(-span, span); ax3d.set_zlim(-span, span)
        if cfg["ROTATE_CAM"]:
            ax3d.view_init(elev=22, azim=(-60 + 90 * fi / max(1, len(frames) - 1)))

        curR.set_xdata([tsec[t], tsec[t]]); curP.set_xdata([tsec[t], tsec[t]]); curC.set_xdata([tsec[t], tsec[t]])
        ax3d.set_title(f"t = {tsec[t]:6.1f} s   |   step {t:4d}/{T-1}   |   docked {int(docked[t])}/{N}",
                       fontsize=11)
        return dyn_artists

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000.0 / cfg["FPS"], blit=False, repeat=True)
    return fig, anim, plt


# ------------------------------------------------------------------ main
def main():
    cfg = dict(CONFIG)
    data, out_dir = get_rollout(cfg)
    fig, anim, plt = build_animation(data, cfg)

    saved = []
    if cfg["SAVE_MP4"]:
        mp4 = os.path.join(out_dir, f"redock_{cfg['SCENE']}_{cfg['DYNAMICS']}_seed{cfg['SEED']}.mp4")
        try:
            from matplotlib.animation import FFMpegWriter
            anim.save(mp4, writer=FFMpegWriter(fps=cfg["FPS"], bitrate=2600)); saved.append(mp4)
        except Exception as e:
            print(f"[save] mp4 failed ({type(e).__name__}: {e}); writing a GIF instead.")
            cfg["SAVE_GIF"] = True
    if cfg["SAVE_GIF"]:
        gif = os.path.join(out_dir, f"redock_{cfg['SCENE']}_{cfg['DYNAMICS']}_seed{cfg['SEED']}.gif")
        try:
            from matplotlib.animation import PillowWriter
            anim.save(gif, writer=PillowWriter(fps=min(cfg["FPS"], 20))); saved.append(gif)
        except Exception as e:
            print(f"[save] gif failed ({type(e).__name__}: {e})")
    for s in saved:
        print(f"[save] wrote {s}")

    if cfg["SHOW_LIVE"]:
        print("[show] opening interactive 3-D window - drag to rotate, close it to finish.")
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
