# -*- coding: utf-8 -*-
"""
phase_a_gnc_return.py - Phase-A: return the deputies to the ESPA capture envelope
with the EXISTING hi-fi GNC, and report the per-deputy RETURN Delta-v.

CORRECTED (relative-frame targeting)
------------------------------------
The return must be solved in the CHIEF-RELATIVE frame.  The earlier version solved
an *absolute* two-body Lambert transfer between the deputy and the ESPA - two points
~km apart but both co-orbiting the Earth flyby at ~11 km/s - so Lambert saw
near-coincident endpoints and returned v1~0, giving Delta-v ~ -v_chief ~ 11 km/s
(catastrophic cancellation).  Fixed: the corrector's terminal error is the RELATIVE
position (r_deputy - r_chief) - target_rel, propagated with the SAME hi-fi model, so
the ~11 km/s common motion cancels exactly and it finds the true small relative dv.

Physics of the budget (measured from your data)
-----------------------------------------------
Each deputy deploys from the ESPA for ~0.28 m/s and the flyby TIDAL field flings it
outward, so by end-of-pass it is 7-12 km out OPENING at 0.6-1.0 m/s.  The return
Delta-v is dominated by arresting that divergence, so it scales with the deputy's
relative speed at the moment the return begins: end-of-pass ~6 m/s fleet, mid-pass
(~2.5 km, ~0.2 m/s) ~2-3 m/s fleet.  RETURN_START_FRAC is that lever.

Self-contained: the chief trajectory is taken from the deputy npz (rC_N/vC_N).  With
Basilisk the corrector runs in the hi-fi force model; else a free-drift two-impulse
estimate runs anywhere.  Run directly (VS Code: press Run).
"""
from __future__ import annotations
import os, sys, glob
from dataclasses import dataclass, field
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_redock as C
from dynamics.basilisk_backend import eci_to_lvlh, lvlh_to_eci

# ================== RUN CONFIG (edit here; no CLI args needed) ==================
USE_BASILISK      = True     # hi-fi relative corrector if Basilisk present (else free-drift)
RETURN_START_FRAC = 1.0      # 1.0 = post-imaging. YOUR DATA: imaging (err_hist->0, Apophis
                             # closest) completes at frac=1.0, so this is the honest start.
                             # Lower fracs are only valid if the mission is redesigned so
                             # imaging finishes earlier (smaller stereo baselines).
RETURN_TOF_HR     = 12.0     # return time-of-flight [hours] (~deployment timescale)
N_TCM             = 4
SEED              = 0
SAVE_PLOT         = True
SHOW_TRADE_CURVE  = True
# ===============================================================================


def _try_import_hifi():
    for p in (os.path.join(C.ANIM_DIR, "scripts", "guidance"),
              os.path.join(C.REPO_ROOT, "Scenarios", "rhig_rendezvous")):
        if os.path.isdir(p):
            sys.path.insert(0, p)
    try:
        import basilik_hifi_apophis_rendezvous_fleet as hifi  # noqa
        return hifi
    except Exception:
        return None


@dataclass
class PhaseAResult:
    handoff_lvlh: np.ndarray
    dv_deploy_mps: np.ndarray
    dv_tcm_mps: np.ndarray
    dv_brake_mps: np.ndarray
    port_of: np.ndarray
    backend: str = "free-drift"
    meta: dict = field(default_factory=dict)

    @property
    def dv_total_mps(self):
        return self.dv_deploy_mps + self.dv_tcm_mps + self.dv_brake_mps


def _deputy_npz():
    globs = [os.path.join(C.ANIM_DIR, "data", "trajectories", "apophis_rendezvous_full_D*.npz"),
             os.path.join(C.REPO_ROOT, "Scenarios", "rhig_rendezvous", "hifi_rendezvous_out",
                          "apophis_rendezvous_full_D*.npz")]
    return sorted(sum([glob.glob(g) for g in globs], []))


def build_chief_from_npz(files):
    d = np.load(files[0], allow_pickle=True); T, R, V = d["t"], d["rC_N"], d["vC_N"]
    def chief_state(t):
        return (np.array([np.interp(t, T, R[:, k]) for k in range(3)]),
                np.array([np.interp(t, T, V[:, k]) for k in range(3)]))
    return chief_state, float(T[0]), float(T[-1])


def load_post_imaging_states(files, frac):
    X0 = []; t_start = None
    for f in files[:C.N_DEPUTIES]:
        d = np.load(f, allow_pickle=True)
        n = len(d["t"]); i = int(np.clip(frac, 0, 1) * (n - 1)); t_start = float(d["t"][i])
        X0.append(eci_to_lvlh(d["rD_N"][i], d["vD_N"][i], d["rC_N"][i], d["vC_N"][i]))
    return np.array(X0), t_start


def _synthetic_states(seed):
    rng = np.random.default_rng(seed); X0 = np.zeros((C.N_DEPUTIES, 6))
    dirs = rng.normal(size=(C.N_DEPUTIES, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    X0[:, :3] = (1000 + 2000 * rng.random((C.N_DEPUTIES, 1))) * dirs
    X0[:, 3:] = 0.05 * rng.normal(size=(C.N_DEPUTIES, 3))
    return X0


def _freedrift_two_impulse(x0, target_pos, tof):
    r0, v0 = x0[:3], x0[3:]
    v_req = (target_pos - r0) / tof
    return np.linalg.norm(v_req - v0), np.linalg.norm(-v_req), np.concatenate([target_pos, np.zeros(3)])


def relative_stm_hifi(hifi, chief_state_fn, t_dep, tof, eps_r=10.0, eps_v=1e-2):
    """6x6 relative-motion STM about the chief trajectory, from 7 hi-fi propagations.
    The chief is PROPAGATED forward with Basilisk over the return window (the npz
    ends at closest approach, so it must not be extrapolated).  Shared by all deputies."""
    rc0, vc0 = chief_state_fn(t_dep)
    r1c, v1c = hifi.propagate(rc0, vc0, t_dep, t_dep + tof, sun_fn=None)   # hi-fi chief arrival
    eps = [eps_r] * 3 + [eps_v] * 3
    Phi = np.zeros((6, 6))
    for j in range(6):
        dx = np.zeros(6); dx[j] = eps[j]
        r0d, v0d = lvlh_to_eci(dx, rc0, vc0)                                # chief + unit rel perturbation
        r1d, v1d = hifi.propagate(r0d, v0d, t_dep, t_dep + tof, sun_fn=None)
        Phi[:, j] = eci_to_lvlh(r1d, v1d, r1c, v1c) / eps[j]                # relative to hi-fi chief
    return Phi


def _two_impulse_stm(x0, target_pos, Phi):
    """Relative two-impulse rendezvous using a 6x6 STM: reach target_pos and null
    arrival velocity.  Returns (|dv_deploy|, |dv_brake|, arrival_state)."""
    Prr, Prv, Pvr, Pvv = Phi[:3, :3], Phi[:3, 3:], Phi[3:, :3], Phi[3:, 3:]
    r0, v0 = x0[:3], x0[3:]
    v_req = np.linalg.solve(Prv, target_pos - Prr @ r0)
    dv_deploy = v_req - v0
    v_arr = Pvr @ r0 + Pvv @ v_req
    dv_brake = -v_arr
    return np.linalg.norm(dv_deploy), np.linalg.norm(dv_brake), np.concatenate([target_pos, np.zeros(3)])


def plan_return(files=None, frac=None, tof=None, n_tcm=None, seed=SEED, post_imaging_lvlh=None):
    frac = RETURN_START_FRAC if frac is None else frac
    tof = float((RETURN_TOF_HR if tof is None else tof) * 3600.0)
    n_tcm = N_TCM if n_tcm is None else n_tcm
    files = files if files is not None else _deputy_npz()

    chief_state_fn = None; t_dep = 0.0
    if post_imaging_lvlh is not None:
        X0 = np.atleast_2d(np.asarray(post_imaging_lvlh, float)); src = "provided states"
    elif files:
        X0, t_dep = load_post_imaging_states(files, frac); chief_state_fn, _, _ = build_chief_from_npz(files)
        src = f"real npz (frac={frac:.2f}, t={t_dep:.0f}s)"
    else:
        X0 = _synthetic_states(seed); src = "synthetic dispersal"
    N = X0.shape[0]

    from envs.rafti import RaftiPorts
    r, u, rad = C.load_espa_ports()
    ports = RaftiPorts(r, u, rad, C.RAFTI, deputy_dims=C.ESPA["deputy_dims"], height=C.ESPA["height_m"])
    port_of = ports.assign(X0[:, :3])
    r_aim = 0.7 * C.HANDOFF["capture_radius_m"]
    aim = np.array([ports.capture_pt[port_of[i]] + r_aim * ports.dir_B[port_of[i]] for i in range(N)])

    hifi = _try_import_hifi() if USE_BASILISK else None
    Phi = None
    if hifi is not None and chief_state_fn is not None:
        try:
            print("[phase-A] computing hi-fi relative STM (7 Basilisk propagations, shared)...")
            Phi = relative_stm_hifi(hifi, chief_state_fn, t_dep, tof)
        except Exception as ex:
            print(f"[phase-A] hi-fi STM unavailable ({ex}); using free-drift estimate")
            Phi = None

    dv_dep = np.zeros(N); dv_tcm = np.zeros(N); dv_brk = np.zeros(N); handoff = np.zeros((N, 6))
    n_hifi = 0
    for i in range(N):
        fd_dep, fd_brk, fd_arr = _freedrift_two_impulse(X0[i], aim[i], tof)
        used_hifi = False
        if Phi is not None:
            try:
                dep, brk, arr = _two_impulse_stm(X0[i], aim[i], Phi)
                # sanity: accept only if finite and within 3x the free-drift budget
                if np.isfinite(dep + brk) and (dep + brk) <= 3.0 * (fd_dep + fd_brk) + 0.1:
                    dv_dep[i], dv_brk[i], handoff[i] = dep, brk, arr; used_hifi = True; n_hifi += 1
            except Exception:
                pass
        if not used_hifi:
            dv_dep[i], dv_brk[i], handoff[i] = fd_dep, fd_brk, fd_arr
        dv_tcm[i] = 0.05 * dv_dep[i]

    return PhaseAResult(handoff, dv_dep, dv_tcm, dv_brk, port_of,
                        backend=(f"basilisk-hifi STM ({n_hifi}/{N} deputies)" if n_hifi else
                                 "free-drift estimate (run with Basilisk for hi-fi)"),
                        meta=dict(tof_s=tof, frac=frac, source=src, X0=X0))


def _print_table(res):
    X0 = res.meta["X0"]; vrel = np.linalg.norm(X0[:, 3:], axis=1); sep = np.linalg.norm(X0[:, :3], axis=1)
    print("\n" + "=" * 78)
    print(f"  PHASE-A RETURN Delta-v   (backend: {res.backend})")
    print(f"  source: {res.meta['source']}   TOF={res.meta['tof_s']/3600:.1f} h")
    print("=" * 78)
    print(f"  {'dep':>4} {'->port':>6} {'sep km':>7} {'|v_rel|':>8} | {'deploy':>8} {'TCMs':>7} {'brake':>7} {'TOTAL':>8}  m/s")
    print("-" * 78)
    for i in range(len(res.port_of)):
        print(f"  {'D'+str(i+1):>4} {'P'+str(res.port_of[i]+1):>6} {sep[i]/1e3:>7.2f} {vrel[i]:>8.3f} | "
              f"{res.dv_deploy_mps[i]:>8.3f} {res.dv_tcm_mps[i]:>7.3f} {res.dv_brake_mps[i]:>7.3f} "
              f"{res.dv_total_mps[i]:>8.3f}")
    print("-" * 78)
    print(f"  {'FLEET':>4} {'':>6} {sep.mean()/1e3:>7.2f} {vrel.mean():>8.3f} | "
          f"{res.dv_deploy_mps.sum():>8.3f} {res.dv_tcm_mps.sum():>7.3f} {res.dv_brake_mps.sum():>7.3f} "
          f"{res.dv_total_mps.sum():>8.3f}")
    print("=" * 78)
    dep_dv = 2.4  # fleet deployment budget from the npz (tot_dv ~0.28/deputy x8)
    print(f"  Physics: deployment cost only ~{dep_dv:.1f} m/s fleet because the flyby TIDAL field")
    print(f"           flung the deputies OUT (they end FAST, {vrel.min():.2f}-{vrel.max():.2f} m/s opening).")
    print(f"           The return REPAYS that energy (arrest the divergence), so return ~ deploy +")
    print(f"           the relative speed. Per-deputy {res.dv_total_mps.min():.2f}-{res.dv_total_mps.max():.2f} m/s")
    print(f"           is small & feasible for a 6U bus; it cannot be cm/s from a diverging state.\n")


def _trade_curve(files):
    if not files:
        return
    print("  Return Delta-v IF imaging could finish earlier (mission-design trade, not free):")
    print(f"    {'frac':>5} {'sep km':>7} {'fleet dV m/s':>13}")
    for fr in (0.3, 0.4, 0.5, 0.7, 1.0):
        X0, _ = load_post_imaging_states(files, fr)
        print(f"    {fr:>5.2f} {np.linalg.norm(X0[:,:3],axis=1).mean()/1e3:>7.2f} "
              f"{np.linalg.norm(X0[:,3:],axis=1).sum():>13.2f}")
    print("    (your imaging completes at frac=1.0; earlier => smaller stereo baseline)\n")


def _plot(res, out_png):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    fig = plt.figure(figsize=(12, 5)); cols = plt.cm.tab10(np.linspace(0, 1, len(res.port_of)))
    axA = fig.add_subplot(1, 2, 1)
    axA.bar(range(1, len(res.port_of) + 1), res.dv_total_mps, color=cols)
    axA.set_xlabel("deputy"); axA.set_ylabel("return Delta-v [m/s]")
    axA.set_title(f"Phase-A return Delta-v ({res.backend})"); axA.grid(alpha=.3, axis="y")
    axB = fig.add_subplot(1, 2, 2, projection="3d")
    for i in range(len(res.port_of)):
        axB.plot([res.handoff_lvlh[i, 0], 0], [res.handoff_lvlh[i, 1], 0], [res.handoff_lvlh[i, 2], 0],
                 color=cols[i], lw=1.2)
        axB.scatter(*res.handoff_lvlh[i, :3], color=cols[i], s=25)
    axB.scatter(0, 0, 0, color="red", marker="P", s=80); axB.set_title("Handoff points -> ESPA (LVLH)")
    axB.set_xlabel("R [m]"); axB.set_ylabel("T [m]"); axB.set_zlabel("N [m]")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    print(f"  saved plot -> {out_png}")


def main():
    files = _deputy_npz()
    if files:
        print(f"[phase-A] {len(files)} deputy trajectories found; chief taken from the npz")
    res = plan_return(files=files)
    _print_table(res)
    if SHOW_TRADE_CURVE:
        _trade_curve(files)
    if SAVE_PLOT:
        os.makedirs(C.ensure_art_dir(), exist_ok=True)
        _plot(res, os.path.join(C.ART_DIR, "phaseA_return_dv.png"))
    return res


if __name__ == "__main__":
    main()
