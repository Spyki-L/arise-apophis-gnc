# -*- coding: utf-8 -*-
"""visualize.py - reward curves + 3D docking trajectories.
Ports plot_trained_rewards.m and visualize_agent_performance.m to matplotlib."""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_redock as C


def _movavg(x, w=50):
    if len(x) < 1: return x
    w = min(w, len(x)); c = np.cumsum(np.insert(x, 0, 0))
    return (c[w:] - c[:-w]) / w


def plot_rewards(npz_path, out_png, kind="multi"):
    d = np.load(npz_path); r = d["reward"]; s = d.get("success")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(r, lw=0.6, alpha=0.35, color="#4C72B0", label="episode")
    ma = _movavg(r, 50)
    ax[0].plot(np.arange(len(ma)) + 50, ma, lw=2.0, color="#C44E52", label="moving avg (50)")
    ax[0].set_xlabel("episode"); ax[0].set_ylabel("total reward")
    ax[0].set_title("DDPG re-docking: reward per episode"); ax[0].legend(); ax[0].grid(alpha=.3)
    if s is not None:
        sm = _movavg(s.astype(float), 50)
        ax[1].plot(np.arange(len(sm)) + 50, sm, lw=2.0, color="#55A868")
        ax[1].set_xlabel("episode")
        ax[1].set_ylabel("deputies captured" if kind == "multi" else "capture rate")
        ax[1].set_title("Docking success (moving avg 50)"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    print("wrote", out_png)


def _draw_cuboid(ax, center, R_bl, dims, color):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    h = 0.5 * np.asarray(dims)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    c = np.asarray(center) + (signs * h) @ R_bl.T
    faces = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
    ax.add_collection3d(Poly3DCollection([c[f] for f in faces], facecolor=color,
                                         edgecolor="k", linewidths=0.3, alpha=0.9))


def _draw_cylinder(ax, cyl, alpha=0.18):
    X, Y, Z = cyl.surface_mesh(n_theta=60, n_z=2)
    ax.plot_surface(X, Y, Z, color="#9aa7b4", alpha=alpha, linewidth=0, shade=True)
    # port markers + outward normals
    for p, n in zip(cyl.ports_r, cyl.ports_dir):
        ax.scatter(*p, color="#d1495b", s=14)
        ax.quiver(*p, *(n * 0.9), color="#d1495b", lw=1.0, arrow_length_ratio=0.3)


def _equal_3d(ax, span, center=(0, 0, 0)):
    ax.set_xlim(center[0]-span, center[0]+span)
    ax.set_ylim(center[1]-span, center[1]+span)
    ax.set_zlim(center[2]-span, center[2]+span)
    try: ax.set_box_aspect((1, 1, 1))
    except Exception: pass


def plot_trajectories(traj, port_of, out_png, title="ARISE re-docking", attitudes=None):
    """traj: (T,N,3) LVLH positions; port_of: (N,) assigned ports.
    Draws the ESPA cylinder + ports, the approach trajectories, a terminal zoom
    with docked deputy cuboids, and range-vs-time."""
    from envs.rafti import RaftiPorts
    from envs.geometry import DeputyCuboid
    from dynamics.attitude import pointing_error_deg
    r, u, rad = C.load_espa_ports()
    ports = RaftiPorts(r, u, rad, C.RAFTI, deputy_dims=C.ESPA["deputy_dims"], height=C.ESPA["height_m"])
    cyl, dep = ports.cyl, ports.dep
    N = traj.shape[1]; cols = plt.cm.tab10(np.linspace(0, 1, N))

    if attitudes is not None:
        fig = plt.figure(figsize=(11.6, 10.6))          # 2x2 grid
        axA = fig.add_subplot(2, 2, 1, projection="3d")
        axB = fig.add_subplot(2, 2, 2, projection="3d")
        axC = fig.add_subplot(2, 2, 3)
        axD = fig.add_subplot(2, 2, 4)
    else:
        fig = plt.figure(figsize=(16.2, 5.2))
        axA = fig.add_subplot(1, 3, 1, projection="3d")
        axB = fig.add_subplot(1, 3, 2, projection="3d")
        axC = fig.add_subplot(1, 3, 3)

    # ---- A: wide approach ----
    _draw_cylinder(axA, cyl)
    for i in range(N):
        axA.plot(traj[:, i, 0], traj[:, i, 1], traj[:, i, 2], color=cols[i], lw=1.3)
        axA.scatter(*traj[0, i], color=cols[i], s=22, marker="o")     # start
    axA.set_title("Approach to ESPA ring (LVLH)")
    axA.set_xlabel("R [m]"); axA.set_ylabel("T [m]"); axA.set_zlabel("N [m]")
    span = float(np.abs(traj).max()) * 1.05
    _equal_3d(axA, span)

    # ---- B: terminal zoom with docked cuboids ----
    _draw_cylinder(axB, cyl, alpha=0.25)
    for i in range(N):
        seg = traj[:, i]; near = seg[np.linalg.norm(seg, axis=1) < 3.0]
        if len(near): axB.plot(near[:, 0], near[:, 1], near[:, 2], color=cols[i], lw=1.4)
        pth = int(port_of[i])
        center = cyl.capture_point(pth, ports.standoff + dep.face_offset)
        R_bl = dep.attitude_pointing(-cyl.port_normal(pth))
        _draw_cuboid(axB, center, R_bl, dep.dims, cols[i])
    axB.set_title("Terminal soft-capture (cuboids on RAFTI ports)")
    axB.set_xlabel("R [m]"); axB.set_ylabel("T [m]"); axB.set_zlabel("N [m]")
    _equal_3d(axB, 1.4)

    # ---- C: range to assigned port vs time ----
    for i in range(N):
        cp = ports.capture_pt[int(port_of[i])]
        d = np.linalg.norm(traj[:, i] - cp, axis=1)
        axC.plot(d, color=cols[i], lw=1.2, label=f"D{i+1}->P{int(port_of[i])+1}")
    axC.axhline(C.RAFTI["capture_pos_tol_m"], color="k", ls="--", lw=0.8, label="capture tol")
    axC.set_yscale("log"); axC.set_xlabel("control step")
    axC.set_ylabel("range to capture point [m]"); axC.grid(alpha=.3)
    axC.set_title("Range vs time"); axC.legend(fontsize=6, ncol=2)

    if attitudes is not None:
        for i in range(N):
            pth = int(port_of[i]); inward = -cyl.port_normal(pth)
            errs = [pointing_error_deg(attitudes[t, i], inward) for t in range(attitudes.shape[0])]
            axD.plot(errs, color=cols[i], lw=1.2)
        axD.axhline(C.RAFTI["att_align_tol_deg"], color="k", ls="--", lw=0.8, label="align tol")
        axD.set_xlabel("control step"); axD.set_ylabel("dock-face pointing error [deg]")
        axD.set_title("Attitude slew to port (6-DOF)"); axD.grid(alpha=.3); axD.legend(fontsize=7)

    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out_png, dpi=130); plt.close(fig)
    print("wrote", out_png)


if __name__ == "__main__":
    # ===== RUN CONFIG (press Run) =====
    CFG = dict(ckpt=None, rewards="artifacts/local_train/reward_history.npz",
               out="artifacts", env="her")
    # ==================================
    import sys
    if len(sys.argv) > 1:
        ap = argparse.ArgumentParser()
        ap.add_argument("--ckpt", default=CFG["ckpt"]); ap.add_argument("--rewards", default=CFG["rewards"])
        ap.add_argument("--out", default=CFG["out"]); ap.add_argument("--env", default=CFG["env"])
        a = ap.parse_args(); CFG.update(ckpt=a.ckpt, rewards=a.rewards, out=a.out, env=a.env)
    os.makedirs(CFG["out"], exist_ok=True)
    if CFG["rewards"] and os.path.exists(CFG["rewards"]):
        plot_rewards(CFG["rewards"], os.path.join(CFG["out"], "reward_curve.png"), kind=CFG["env"])
    if CFG["ckpt"] and os.path.exists(CFG["ckpt"]):
        # infer policy dims from the checkpoint (DDPG 12/15 or TD3 export both load)
        import torch
        from agents.ddpg import DDPGAgent
        ck = torch.load(CFG["ckpt"], map_location="cpu"); aw = ck["actor"]
        in_dim = int(aw["fc1.weight"].shape[1]); act = int(aw["mu.weight"].shape[0])
        agent = DDPGAgent(in_dim, act); agent.load_models(CFG["ckpt"])
        try:
            from bridge.redock_pipeline import run_redock
            res = run_redock(agent=agent, dynamics="hcw", seed=1)
            plot_trajectories(res.trajectory, res.port_of,
                              os.path.join(CFG["out"], "dock_trajectories.png"), attitudes=res.attitudes)
        except Exception as e:
            print(f"[visualize] trajectory render skipped ({type(e).__name__}: {e}); reward curve written.")
