# -*- coding: utf-8 -*-
"""
verify_all.py - ONE-CLICK verification plot suite (VS Code: press Run).

Generates every figure you need to sign off the re-docking, into artifacts/:
    1. mission_dv_budget.png        deploy (npz) + return (Phase-A) per deputy + fleet
    2. phaseA_return_dv.png         return Delta-v per deputy + handoff geometry
    3. phaseA_return_trajectories.png   3-D return paths (imaging station -> ESPA envelope)
    4. dock_trajectories.png        6-DOF terminal docking (approach, cuboids, range, attitude)
    5. dock_softcapture.png         per-deputy contact conditions vs RAFTI tolerances
    6. reward_curve.png             DDPG training status (if a run exists)
and prints a pass/fail verification summary.

Set DYNAMICS below ("hcw" fast | "basilisk" hi-fi).  No command-line arguments.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import config_redock as C
from guidance import phase_a_gnc_return as pa
from bridge.redock_pipeline import run_redock
from eval.rollout import PortFramePD, SwarmFromSingle
from eval.visualize import plot_trajectories, plot_rewards

# ==================== RUN CONFIG (edit here) ====================
DYNAMICS = "hcw"      # "hcw" | "basilisk"
SEED     = 2
# ===============================================================
ART = C.ensure_art_dir()
cols = plt.cm.tab10(np.linspace(0, 1, 8))


def _deploy_dv():
    files = pa._deputy_npz()
    dv = []
    for f in files[:8]:
        d = np.load(f, allow_pickle=True); dv.append(float(d["tot_dv"][0]) if "tot_dv" in d else np.nan)
    return np.array(dv) if dv else np.full(8, np.nan)


def plot_budget(res_pa, dvB_fleet):
    deploy = _deploy_dv(); ret = res_pa.dv_total_mps
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(1, 9)
    ax.bar(x, deploy, label=f"deploy (npz)  Σ={np.nansum(deploy):.1f}", color="#8da0cb")
    ax.bar(x, ret, bottom=deploy, label=f"return (Phase-A)  Σ={ret.sum():.1f}", color="#fc8d62")
    ax.set_xlabel("deputy"); ax.set_ylabel("Delta-v [m/s]")
    ax.set_title(f"Mission Delta-v budget per deputy  (Phase-B docking ≈ {dvB_fleet:.1f} m/s fleet)")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    for i in range(8):
        ax.text(x[i], deploy[i] + ret[i] + 0.03, f"{deploy[i]+ret[i]:.2f}", ha="center", fontsize=8)
    p = os.path.join(ART, "mission_dv_budget.png"); fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print("  wrote", p)


def plot_return_trajectories(res_pa):
    X0 = res_pa.meta["X0"]; hand = res_pa.handoff_lvlh
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
    for i in range(len(X0)):
        pts = np.linspace(X0[i, :3], hand[i, :3], 30)     # free-drift straight-line return
        ax.plot(pts[:, 0] / 1e3, pts[:, 1] / 1e3, pts[:, 2] / 1e3, color=cols[i], lw=1.3)
        ax.scatter(*(X0[i, :3] / 1e3), color=cols[i], s=30, marker="o")     # imaging station
        ax.scatter(*(hand[i, :3] / 1e3), color=cols[i], s=45, marker="*")   # handoff
    ax.scatter(0, 0, 0, color="red", marker="P", s=120, label="ESPA")
    ax.set_title("Phase-A return trajectories (imaging o -> handoff *) [LVLH]")
    ax.set_xlabel("R [km]"); ax.set_ylabel("T [km]"); ax.set_zlabel("N [km]"); ax.legend()
    p = os.path.join(ART, "phaseA_return_trajectories.png"); fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print("  wrote", p)


def plot_softcapture(res):
    cap = res.capture; R = C.RAFTI
    metrics = [("3-D miss [m]", [c["dist"] for c in cap], R["capture_pos_tol_m"]),
               ("axial |v| [m/s]", [abs(c["v_axial"]) for c in cap], R["v_capture_max_mps"]),
               ("lateral |v| [m/s]", [c["v_lateral"] for c in cap], R["v_lateral_max_mps"]),
               ("body rate [rad/s]", [c["rate"] for c in cap], R["omega_capture_max"]),
               ("face align [deg]", [c["att_err"] for c in cap], R["att_align_tol_deg"])]
    fig, axs = plt.subplots(2, 3, figsize=(13, 8)); axs = axs.ravel()   # 2x3 grid (5 metrics + 1 blank)
    for ax, (name, vals, tol) in zip(axs, metrics):
        ax.bar(range(1, 9), vals, color=cols)
        ax.axhline(tol, color="r", ls="--", lw=1.2, label=f"tol {tol:g}")
        ax.set_title(name, fontsize=11); ax.set_xlabel("deputy"); ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")
    for ax in axs[len(metrics):]:
        ax.axis("off")
    fig.suptitle("RAFTI soft-capture conditions at contact vs tolerances (all bars below red = PASS)")
    p = os.path.join(ART, "dock_softcapture.png"); fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print("  wrote", p)


def main():
    print(f"\n[verify] dynamics = {DYNAMICS}\n")
    print("[1/4] Phase-A return...")
    pa.USE_BASILISK = (DYNAMICS == "basilisk")
    res_pa = pa.plan_return(files=pa._deputy_npz())
    pa._plot(res_pa, os.path.join(ART, "phaseA_return_dv.png"))
    plot_return_trajectories(res_pa)

    print("[2/4] Phase-B 6-DOF docking...")
    from eval.docking_demo import DEMO
    pol = SwarmFromSingle(PortFramePD(), n=8)
    res = run_redock(agent=pol, dynamics=DYNAMICS, seed=SEED, record=True, **DEMO)
    plot_trajectories(res.trajectory, res.port_of, os.path.join(ART, "dock_trajectories.png"),
                      attitudes=res.attitudes,
                      title=f"ARISE 6-DOF re-docking: {res.n_captured}/8 soft-captured")
    plot_softcapture(res)

    print("[3/4] Mission Delta-v budget...")
    plot_budget(res_pa, res.dv_phaseB_mps)

    print("[4/4] Training curve...")
    rew = sorted(glob.glob(os.path.join(ART, "*", "reward_history.npz")))
    if rew:
        plot_rewards(rew[-1], os.path.join(ART, "reward_curve.png"), kind="her")
    else:
        print("  (no training run yet; skip)")

    # ---- verification summary ----
    R = C.RAFTI
    print("\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  Phase-A return   : {res_pa.dv_total_mps.min():.2f}-{res_pa.dv_total_mps.max():.2f} m/s/deputy, "
          f"{res_pa.dv_total_mps.sum():.1f} m/s fleet  ({res_pa.backend})")
    print(f"  Phase-B docked   : {res.n_captured}/8 deputies soft-captured  ({DYNAMICS})")
    ok = all(c["captured"] for c in res.capture)
    worst = dict(miss=max(c["dist"] for c in res.capture),
                 vax=max(abs(c["v_axial"]) for c in res.capture),
                 vlat=max(c["v_lateral"] for c in res.capture),
                 rate=max(c["rate"] for c in res.capture),
                 att=max(c["att_err"] for c in res.capture))
    print(f"  Soft-capture     : worst miss {worst['miss']:.3f}/{R['capture_pos_tol_m']}m, "
          f"axial {worst['vax']:.3f}/{R['v_capture_max_mps']}, lateral {worst['vlat']:.3f}/{R['v_lateral_max_mps']} m/s")
    print(f"                     worst rate {worst['rate']:.3f}/{R['omega_capture_max']} rad/s, "
          f"align {worst['att']:.2f}/{R['att_align_tol_deg']} deg  -> {'ALL PASS' if ok else 'CHECK'}")
    print(f"  Total mission dv : deploy {np.nansum(_deploy_dv()):.1f} + return {res_pa.dv_total_mps.sum():.1f} "
          f"+ dock {res.dv_phaseB_mps:.1f} = {np.nansum(_deploy_dv())+res_pa.dv_total_mps.sum()+res.dv_phaseB_mps:.1f} m/s fleet")
    print("=" * 70)
    print(f"  6 figures in: {ART}\n")


if __name__ == "__main__":
    main()
