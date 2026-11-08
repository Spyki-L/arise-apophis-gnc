#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apophis_swarm_peer_relnav_v3.py
================================

Cascade-Anchored MOS-CI for Deployment-Ordered Asteroid Rendezvous Swarms
-------------------------------------------------------------------------

A novel architecture for peer-to-peer relative navigation in deep space:

  1. Single absolute anchor: only the CHIEF carries DSN/GPS-grade tracking.
     No deputy has any independent absolute reference.

  2. Chief-sighting anchor:  each deputy carries an optical sensor that
     images the chief.  Because the chief sits at the origin of the Hill
     frame BY CONSTRUCTION, a chief sighting is exactly a direct measurement
     of the observing deputy's own Hill position — the cleanest possible
     anchor in deep space (no DSN station needed for the deputy).

  3. Deployment-ordered information hierarchy:  deputies deploy
     SEQUENTIALLY (D0 first, ..., D7 last).  Earlier deputies converge to
     tight covariances long before late deputies exist.  When a junior
     deputy deploys, it bootstraps off the senior deputies, who serve as
     "information intermediaries" between the chief and the new arrival.

  4. Bootstrap CI burst on deployment:  the instant a new deputy comes
     online, it fans out an MOS-CI fusion against every currently-active
     senior.  Within one step its covariance jumps from sigma~1500 m to
     roughly sigma_senior + sigma_meas — no slow convergence transient.

  5. MOS-CI math (from v2, unchanged):  marginalised observable-subspace
     CI on the 3-D position subspace, re-embedded into 6-D via the
     conditional Gaussian update.

Differences vs. v2
------------------
  * gps_update REPLACED by chief_observation_update (3-D position only)
  * Deputies activated sequentially based on bundle deployment time
  * Bootstrap CI burst at each deputy's first active step
  * New 3-D truth-vs-estimate trajectory plot for all deputies

Author: Harish Vernekar (2026)
"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

# Locate ARISE root
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = None
for parent in SCRIPT_DIR.parents:
    if (parent / "GNC").is_dir():
        ROOT = parent
        break
if ROOT is None:
    ROOT = SCRIPT_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the heavy machinery from v2
from GNC.Navigation.hill_ekf import dcm_lvlh, hill_rate, cw_state_transition
from GNC.Navigation.crosslink_scheduler import CrosslinkScheduler
from GNC.Scenario.Apophis.apophis_swarm_peer_relnav_v2 import (
    polar_to_cartesian_drho,
    polar_to_cartesian_jacobian,
    polar_meas_covariance_in_cartesian,
    oif_ci_fuse,
    peer_range_pred_and_jac,
    peer_bearing_pred_and_jac,
    generate_peer_measurement,
    load_apophis_bundle,
    make_state_fn,
    hill_state_from_eci,
    eci_to_hill_traj,
    build_burn_sequence,
    PeerRelNavConfig as _BaseConfig,
    PeerNavSnapshot,
    PeerRelNavEKF as _BaseEKF,
    NIS_GATE_PEER as V2_NIS_GATE_PEER,
)

# ──────────────────────────────────────────────────────────────────────────────
#                              CONFIG  (v3 defaults)
# ──────────────────────────────────────────────────────────────────────────────

N_DEP       = 8
MEAS_DECIM  = 20             # dt_nav ~ 20 s
CROP_HR     = 0.0            # do NOT crop; we want the very earliest deployment

# Peer sensor noise (1σ)
RANGE_SIGMA   = 5.0          # [m]
AZ_SIGMA_DEG  = 0.3          # [deg]
EL_SIGMA_DEG  = 0.3          # [deg]

# Chief observation (deputy optical sighting of chief at Hill origin).
# Composite noise = chief inertial uncertainty (~1 m) + deputy optical (~5 m).
CHIEF_OBS_SIGMA_POS = 5.0    # [m]
CHIEF_OBS_DROPOUT   = 0.05   # occasional optical miss

# Bootstrap CI burst on deployment
BOOTSTRAP_CI_BURST_N = 4     # fuse against up to N tightest seniors at deploy
BOOTSTRAP_NO_DROPOUT = True  # bootstrap measurements never drop

# Peer link reliability (in normal operation)
P_PEER_DROPOUT = 0.05
P_PEER_OUTLIER = 0.01
OUTLIER_SCALE  = 6.0

# Gating thresholds
NIS_GATE_PEER  = V2_NIS_GATE_PEER       # chi2(3) @ 99.5%
NIS_GATE_CHIEF = 11.34                  # chi2(3) @ 99.0%

# MOS-CI fusion controls
USE_CI_FUSION       = True
CI_EVERY_N_STEPS    = 1
CI_NEES_TARGET      = 3.0
CI_NEES_PENALTY     = 0.10
CI_REJECT_DET_RATIO = 1.05

# Event-triggered ISL broadcast
ISL_TRIGGER_MAHAL  = 0.3
ISL_TRIGGER_TR_REL = 0.05

# Plotting
SAVE_FIGS = False
FIG_DIR   = SCRIPT_DIR / "peer_relnav_v3_figs"

# Authoritative chief-truth file (overrides bundle's rC_N/vC_N).
# Schema:  times (T,)   positions (T,3)   velocities (T,3)
CHIEF_REFERENCE_NAME = "chief_reference.npz"


def _find_chief_reference() -> Optional[Path]:
    """
    Locate chief_reference.npz.  Searched in order:
      1.  <ROOT>/Data/
      2.  <ROOT.parent>/Data/   (repo root)
      3.  the script directory
      4.  <ROOT>/GNC/Scenario/GEO_Sample/   (legacy)
    Returns the first match, or None.
    """
    candidates = []
    if ROOT is not None:
        candidates += [ROOT / "Data", ROOT.parent / "Data",
                       ROOT / "GNC" / "Scenario" / "GEO_Sample"]
    candidates += [SCRIPT_DIR]
    for d in candidates:
        p = Path(d) / CHIEF_REFERENCE_NAME
        if p.is_file():
            return p
    return None


def load_chief_reference(path: Path) -> Dict[str, np.ndarray]:
    """
    Load chief_reference.npz and remap its keys (`times`, `positions`,
    `velocities`) onto the bundle convention (`t`, `rC_N`, `vC_N`) so it
    plugs straight into make_state_fn().

    Units:  the file stores positions in km and velocities in km/s
    (geocentric, ||r|| ~ 42-53 thousand km).  Deputy bundles are in
    SI (m, m/s).  We auto-detect the scale by comparing the first
    position magnitude against a 100 km threshold and convert if needed.
    """
    with np.load(path, allow_pickle=True) as z:
        t  = np.asarray(z["times"],      float).reshape(-1)
        r  = np.asarray(z["positions"],  float).reshape(-1, 3)
        v  = np.asarray(z["velocities"], float).reshape(-1, 3)

    # Auto-detect units.  Anything chief-like with ||r|| < 1e5 must be km.
    # (Earth GEO ~ 42164 km vs 4.2e7 m; we flag based on a safe boundary.)
    r0_mag = float(np.linalg.norm(r[0]))
    if r0_mag < 1e5:
        # km, km/s  ->  m, m/s
        r = r * 1000.0
        v = v * 1000.0
        print(f"[v3] chief_reference: detected km units, converted to m "
              f"(||r0||: {r0_mag:.1f} km -> {r0_mag*1000:.1f} m)")
    return dict(t=t, rC_N=r, vC_N=v)


# ──────────────────────────────────────────────────────────────────────────────
#                       v3 EKF: subclass + chief obs update
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PeerRelNavConfigV3(_BaseConfig):
    sigma_chief_obs: float = CHIEF_OBS_SIGMA_POS


class PeerRelNavEKFV3(_BaseEKF):
    """
    Cascade-Anchored EKF.  Adds:
      * chief_observation_update(...)  — direct 3-D position measurement
                                         from sighting the chief at Hill origin.
      * chief_obs_count                — diagnostic counter.
    """

    def __init__(self, dep_id, cfg, chief_state_fn):
        super().__init__(dep_id, cfg, chief_state_fn)
        self.chief_obs_count = 0
        self.deploy_t: Optional[float] = None
        self.deploy_step: Optional[int] = None
        self.bootstrap_done = False
        self.has_been_updated = False  # any measurement at all yet?

    def reset(self, x0, P0=None):
        super().reset(x0, P0)
        self.chief_obs_count = 0
        self.bootstrap_done = False
        self.has_been_updated = False

    def chief_observation_update(self, t, z_chief_pos,
                                 x_true_for_nees: Optional[np.ndarray] = None):
        """
        Direct 3-D position measurement of this deputy from chief sighting.

            z = ρ_i + v,    v ~ N(0, σ_chief^2 · I_3)
            H = [I_3 | 0_3]   (3×6)

        The deputy infers its own ρ_i because the chief sits at the
        origin of the chief's Hill frame by definition; the noise rolls
        up the deputy's optical sensor accuracy and the chief's
        inertial-state uncertainty (which is small thanks to DSN).
        """
        z = np.asarray(z_chief_pos, float).reshape(3)
        H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
        R = (self.cfg.sigma_chief_obs**2) * np.eye(3)

        zhat = self.x_pred[:3]
        y = z - zhat
        S = H @ self.P_pred @ H.T + R

        try:
            nis = float(y @ la.solve(S, y))
        except la.LinAlgError:
            nis = float(y @ y)

        if nis > NIS_GATE_CHIEF:
            self.x = self.x_pred.copy(); self.P = self.P_pred.copy()
            return

        K = self.P_pred @ H.T @ la.inv(S)
        I6 = np.eye(6)
        self.x = self.x_pred + K @ y
        KH = K @ H
        if self.cfg.use_joseph:
            self.P = (I6 - KH) @ self.P_pred @ (I6 - KH).T + K @ R @ K.T
        else:
            self.P = (I6 - KH) @ self.P_pred
        self.P = 0.5 * (self.P + self.P.T)
        self.chief_obs_count += 1
        self.has_been_updated = True

        nees = None
        if x_true_for_nees is not None:
            e = self.x - x_true_for_nees
            try:
                nees = float(e @ la.solve(self.P + 1e-12*np.eye(6), e))
            except la.LinAlgError:
                nees = float("nan")
            self.nees_history.append(nees)

        self.history.append(PeerNavSnapshot(
            t=t, dep_id=self.dep_id, x=self.x.copy(), P=self.P.copy(),
            innov=y.copy(), S=S.copy(), nees=nees, meas_type="chief_obs"))

        self.x_pred = self.x.copy(); self.P_pred = self.P.copy()


# ──────────────────────────────────────────────────────────────────────────────
#                           simulation driver
# ──────────────────────────────────────────────────────────────────────────────

def run_peer_relnav_v3_sim(data_dir: Path,
                            N_dep: int = N_DEP,
                            meas_decim: int = MEAS_DECIM,
                            chief_ref_path: Optional[Path] = None
                            ) -> Dict[str, np.ndarray]:
    data_dir = Path(data_dir)

    # ── load chief truth (authoritative) ──
    if chief_ref_path is None:
        chief_ref_path = _find_chief_reference()
    if chief_ref_path is None:
        raise FileNotFoundError(
            f"Could not find {CHIEF_REFERENCE_NAME} in any standard location.")
    print(f"\n[v3] chief truth source: {chief_ref_path}")
    chief_ref = load_chief_reference(chief_ref_path)

    # ── load all deputy bundles ──
    bundles = []
    for d in range(N_dep):
        p = data_dir / f"apophis_rendezvous_full_D{d}.npz"
        if not p.is_file():
            raise FileNotFoundError(f"Missing: {p}")
        bundles.append(load_apophis_bundle(p))

    # ── deployment times per deputy (= bundle's first time stamp) ──
    t_deploy = np.array([float(bd["t"][0]) for bd in bundles])
    t_deploy_end = np.array([float(bd["t"][-1]) for bd in bundles])

    deploy_order = np.argsort(t_deploy).tolist()  # earliest first

    t_sim_start = float(t_deploy[deploy_order[0]])  # first deputy deploys
    t_sim_end   = float(min(t_deploy_end))           # latest possible end

    if CROP_HR > 0:
        t_sim_start += CROP_HR * 3600.0

    # Use earliest-deployed deputy's data as the reference time grid
    data_ref = bundles[deploy_order[0]]
    t_full = np.asarray(data_ref["t"], float).reshape(-1)
    idx_start = int(np.searchsorted(t_full, t_sim_start))
    idx_end   = int(np.searchsorted(t_full, t_sim_end, side="right")) - 1
    idx_sel = np.arange(idx_start, idx_end + 1, meas_decim)
    t_arr = t_full[idx_sel]
    N = len(t_arr)
    dt_nav = float(np.median(np.diff(t_arr)))

    # Step index at which each deputy comes online
    deploy_step = np.array(
        [int(np.searchsorted(t_arr, t_deploy[d])) for d in range(N_dep)],
        dtype=int)
    deploy_step = np.clip(deploy_step, 0, N - 1)

    print(f"\n[Cascade MOS-CI v3]")
    print(f"  Sim window: [{t_sim_start:.0f}, {t_sim_end:.0f}] s "
          f"({(t_sim_end - t_sim_start)/3600:.2f} h)")
    print(f"  Nav grid:   {N} steps, dt = {dt_nav:.1f} s")
    print(f"  Deployment order (earliest first):")
    for d in deploy_order:
        dt_rel = (t_deploy[d] - t_sim_start) / 60.0
        print(f"    D{d}:  t+{dt_rel:6.2f} min   (step k={deploy_step[d]})")

    # ── chief state interpolator (from chief_reference.npz, authoritative) ──
    chief_state_fn = make_state_fn(chief_ref, "rC_N", "vC_N")

    # Pre-evaluate chief truth on the sim grid so deputy ECI -> Hill uses it
    rC_grid = np.zeros((N, 3)); vC_grid = np.zeros((N, 3))
    for k in range(N):
        rC_grid[k], vC_grid[k] = chief_state_fn(float(t_arr[k]))

    # Stash chief truth in results for plotting
    chief_truth_inertial = dict(t=t_arr.copy(), r=rC_grid.copy(), v=vC_grid.copy())

    # ── per-deputy truth + burns ──
    # Deputy positions/velocities come from the bundles, but the chief
    # state used in the ECI->Hill conversion is now the authoritative
    # chief_reference, NOT the bundle's own rC_N/vC_N.
    x_true_all = np.zeros((N_dep, N, 6))
    dv_all     = np.zeros((N_dep, N, 3))
    for d in range(N_dep):
        bd = bundles[d]
        t_full_d = np.asarray(bd["t"], float).reshape(-1)
        rD = np.asarray(bd["rD_N"], float).reshape(-1, 3)
        vD = np.asarray(bd["vD_N"], float).reshape(-1, 3)
        idx_d = np.clip(np.searchsorted(t_full_d, t_arr), 0, len(t_full_d) - 1)
        rD_grid = rD[idx_d]
        vD_grid = vD[idx_d]
        # Use AUTHORITATIVE chief (rC_grid, vC_grid) for the Hill conversion
        x_true_all[d] = eci_to_hill_traj(rD_grid, vD_grid, rC_grid, vC_grid)
        # Burn aggregation: assign each impulse to the sim step whose
        # window [t_arr[k]-dt_nav/2, t_arr[k]+dt_nav/2) contains it,
        # SUMMING all burns inside the window.  Fixes the previous
        # point-sampling that dropped 19/20 burns when MEAS_DECIM=20.
        imp = np.asarray(bd["impulses"], float).reshape(-1, 4)
        if imp.size > 0:
            t_burn = imp[:, 0]; dv_burn = imp[:, 1:4]
            half = 0.5 * dt_nav
            for b in range(len(t_burn)):
                k = int(np.searchsorted(t_arr, t_burn[b] + half)) - 1
                if 0 <= k < N and abs(t_arr[k] - t_burn[b]) <= half:
                    dv_all[d, k] += dv_burn[b]
        n_caught = int(np.sum(np.linalg.norm(dv_all[d], axis=1) > 0))
        n_burns = int(imp.shape[0])
        if n_burns > 0:
            print(f"    D{d}: {n_burns:3d} burns -> {n_caught:3d} sim-steps with non-zero ΔV "
                  f"(captured magnitude: {np.linalg.norm(dv_all[d].sum(axis=0)):.3f} m/s)")

    # ── per-deputy EKFs (constructed but only initialised on deploy) ──
    cfg = PeerRelNavConfigV3(dt=dt_nav)
    ekfs: List[PeerRelNavEKFV3] = [
        PeerRelNavEKFV3(d, cfg, chief_state_fn) for d in range(N_dep)
    ]
    # initial-state biases: realistic deployment uncertainty
    initial_x = []
    for d in range(N_dep):
        ks = deploy_step[d]
        bias = np.array([200.0, -150.0, 80.0, 0.06, -0.04, 0.02])
        initial_x.append(x_true_all[d, ks] + bias * (1.0 + 0.1 * d))

    sched = CrosslinkScheduler(N_dep)
    rng = np.random.default_rng(42)
    sigma_az_rad = np.deg2rad(AZ_SIGMA_DEG)
    sigma_el_rad = np.deg2rad(EL_SIGMA_DEG)

    active = np.zeros(N_dep, dtype=bool)
    bootstrap_log: List[Tuple[float, int, List[int]]] = []
    chief_obs_log_count = np.zeros(N_dep, dtype=int)

    x_est_hist  = np.full((N_dep, N, 6), np.nan)
    P_diag_hist = np.full((N_dep, N, 6), np.nan)
    nees_hist   = np.full((N_dep, N), np.nan)
    omega_hist: List[Tuple[float, int, int, float]] = []
    active_hist = np.zeros((N_dep, N), dtype=bool)

    print(f"\n[OIF-CI / Cascade] Running ...")
    for k in range(N):
        t = float(t_arr[k])

        # 1) Activate newly-deployed deputies and run bootstrap CI burst
        for d in range(N_dep):
            if not active[d] and k >= deploy_step[d]:
                ekfs[d].reset(initial_x[d])
                ekfs[d].deploy_t = t
                ekfs[d].deploy_step = k
                active[d] = True
                seniors = [s for s in range(N_dep) if active[s] and s != d]
                # tightest position covariance first
                seniors.sort(key=lambda s: np.trace(ekfs[s].P[:3, :3]))
                used = []
                for s in seniors[:BOOTSTRAP_CI_BURST_N]:
                    z_polar, valid = generate_peer_measurement(
                        rng, x_true_all[d, k], x_true_all[s, k],
                        RANGE_SIGMA, sigma_az_rad, sigma_el_rad,
                        p_dropout=(0.0 if BOOTSTRAP_NO_DROPOUT else P_PEER_DROPOUT),
                        p_outlier=0.0)
                    if not valid:
                        continue
                    ekfs[d].x_pred = ekfs[d].x.copy()
                    ekfs[d].P_pred = ekfs[d].P.copy()
                    ekfs[d].oif_ci_fuse(t, z_polar,
                                        ekfs[s].x.copy(), ekfs[s].P.copy(), s,
                                        x_true_for_nees=x_true_all[d, k],
                                        nees_weight=CI_NEES_PENALTY)
                    used.append(s)
                    if ekfs[d].omega_history:
                        omega_hist.append((t, d, s, ekfs[d].omega_history[-1]))
                ekfs[d].bootstrap_done = True
                bootstrap_log.append((t, d, used))
                trP = np.sqrt(np.trace(ekfs[d].P[:3, :3]))
                print(f"  [t+{(t-t_arr[0])/60:6.2f} min] D{d} DEPLOYED  "
                      f"bootstrap seniors={used}  -> sigma_pos~{trP:.1f} m")

        # 2) Predict active deputies
        for d in range(N_dep):
            if not active[d]:
                continue
            if not ekfs[d].has_been_updated and ekfs[d].chief_obs_count == 0 and ekfs[d].peer_update_count == 0 and ekfs[d].ci_count == 0:
                # First step after deployment + bootstrap: pred = current
                ekfs[d].x_pred = ekfs[d].x.copy()
                ekfs[d].P_pred = ekfs[d].P.copy()
                ekfs[d].t_last = t
            else:
                ekfs[d].predict(t, burn_dv_H=dv_all[d, k])

        # 3) Chief-observation update for each active deputy
        for d in range(N_dep):
            if not active[d]:
                continue
            if rng.random() < CHIEF_OBS_DROPOUT:
                continue
            z_chief = (x_true_all[d, k, :3] +
                       CHIEF_OBS_SIGMA_POS * rng.normal(size=3))
            ekfs[d].chief_observation_update(
                t, z_chief, x_true_for_nees=x_true_all[d, k])
            chief_obs_log_count[d] += 1

        # 4) Peer measurements + MOS-CI between active pairs
        pairs = sched.pairs_for_step(k)
        for (i, j) in pairs:
            if not (active[i] and active[j]):
                continue
            z_ij, valid_ij = generate_peer_measurement(
                rng, x_true_all[i, k], x_true_all[j, k],
                RANGE_SIGMA, sigma_az_rad, sigma_el_rad)
            z_ji, valid_ji = generate_peer_measurement(
                rng, x_true_all[j, k], x_true_all[i, k],
                RANGE_SIGMA, sigma_az_rad, sigma_el_rad)

            # Event-triggered broadcast
            if ekfs[i].should_broadcast():
                ekfs[i].mark_broadcast(t)
            if ekfs[j].should_broadcast():
                ekfs[j].mark_broadcast(t)

            x_j_post = (ekfs[j].x_last_broadcast.copy()
                        if ekfs[j].x_last_broadcast is not None else ekfs[j].x.copy())
            P_j_post = (ekfs[j].P_last_broadcast.copy()
                        if ekfs[j].P_last_broadcast is not None else ekfs[j].P.copy())
            x_i_post = (ekfs[i].x_last_broadcast.copy()
                        if ekfs[i].x_last_broadcast is not None else ekfs[i].x.copy())
            P_i_post = (ekfs[i].P_last_broadcast.copy()
                        if ekfs[i].P_last_broadcast is not None else ekfs[i].P.copy())

            # Schmidt-augmented EKF peer update
            if valid_ij:
                ekfs[i].peer_update(t, z_ij, x_j_post, P_j_post, j,
                                    x_true_for_nees=x_true_all[i, k])
            if valid_ji:
                ekfs[j].peer_update(t, z_ji, x_i_post, P_i_post, i,
                                    x_true_for_nees=x_true_all[j, k])

            # MOS-CI peer-state fusion (rate-limited)
            ci_time_ok = (k % CI_EVERY_N_STEPS == 0)
            if USE_CI_FUSION and ci_time_ok:
                if valid_ij:
                    ekfs[i].oif_ci_fuse(t, z_ij, x_j_post, P_j_post, j,
                                        x_true_for_nees=x_true_all[i, k],
                                        nees_weight=CI_NEES_PENALTY)
                    if ekfs[i].omega_history:
                        omega_hist.append((t, i, j, ekfs[i].omega_history[-1]))
                if valid_ji:
                    ekfs[j].oif_ci_fuse(t, z_ji, x_i_post, P_i_post, i,
                                        x_true_for_nees=x_true_all[j, k],
                                        nees_weight=CI_NEES_PENALTY)
                    if ekfs[j].omega_history:
                        omega_hist.append((t, j, i, ekfs[j].omega_history[-1]))

        # 5) Record
        for d in range(N_dep):
            active_hist[d, k] = active[d]
            if active[d]:
                x_est_hist[d, k] = ekfs[d].x.copy()
                P_diag_hist[d, k] = np.diag(ekfs[d].P)
                e = ekfs[d].x - x_true_all[d, k]
                try:
                    nees_hist[d, k] = float(e @ la.solve(
                        ekfs[d].P + 1e-12*np.eye(6), e))
                except la.LinAlgError:
                    pass

    # ── reporting ──
    t_hr = (t_arr - t_arr[0]) / 3600.0

    print("\n" + "=" * 92)
    print("  CASCADE-ANCHORED MOS-CI  —  RESULTS")
    print("=" * 92)
    print(f"\n┌─── Per-Deputy Cascade Summary ────────────────────────────────────────────────────────────┐")
    print(f"│ {'Dep':>3} │ {'deploy@':>8} │ {'RMS pos':>8} │ {'RMS vel':>8} │ "
          f"{'ChiefObs':>8} │ {'Peers':>5} │ {'CI':>4} │ {'CIrej':>5} │ "
          f"{'<NEES>':>7} │ {'<ω>':>5} │ {'σ_pos_f':>8} │")
    print(f"│{'─'*3:>3}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*5}─┼─"
          f"{'─'*4}─┼─{'─'*5}─┼─{'─'*7}─┼─{'─'*5}─┼─{'─'*8}─│")
    for d in range(N_dep):
        ks = deploy_step[d]
        # mask = active & past first 0.05 h after deploy
        mask_d = active_hist[d].copy()
        mask_d[:ks + max(1, int(180.0 / dt_nav))] = False
        if mask_d.sum() == 0:
            mask_d = active_hist[d]
        err = x_est_hist[d] - x_true_all[d]
        rms_pos = float(np.sqrt(np.nanmean(np.sum(err[mask_d, :3]**2, axis=1))))
        rms_vel = float(np.sqrt(np.nanmean(np.sum(err[mask_d, 3:]**2, axis=1))))
        nees_avg = float(np.nanmean(nees_hist[d, mask_d]))
        omega_avg = (float(np.mean(ekfs[d].omega_history))
                     if ekfs[d].omega_history else float("nan"))
        sigma_pos_f = float(np.sqrt(np.nansum(P_diag_hist[d, -1, :3])))
        deploy_min = (t_deploy[d] - t_sim_start) / 60.0
        print(f"│  D{d} │ {deploy_min:7.2f}m │ {rms_pos:8.2f} │ {rms_vel:8.4f} │ "
              f"{ekfs[d].chief_obs_count:8d} │ {ekfs[d].peer_update_count:5d} │ "
              f"{ekfs[d].ci_count:4d} │ {ekfs[d].ci_reject_count:5d} │ "
              f"{nees_avg:7.2f} │ {omega_avg:5.2f} │ {sigma_pos_f:8.2f} │")
    print(f"└───────────────────────────────────────────────────────────────────────────────────────────┘")

    return dict(
        t_arr=t_arr, t_hr=t_hr,
        x_true_all=x_true_all, x_est_hist=x_est_hist,
        P_diag_hist=P_diag_hist, nees_hist=nees_hist,
        omega_hist=omega_hist, active_hist=active_hist,
        deploy_step=deploy_step, t_deploy=t_deploy,
        N_dep=N_dep, ekfs=ekfs,
    )


# ──────────────────────────────────────────────────────────────────────────────
#                                 plotting
# ──────────────────────────────────────────────────────────────────────────────

def _mask_per_deputy(res, d):
    m = res["active_hist"][d].copy()
    return m


def plot_results_v3(res):
    t_hr        = res["t_hr"]
    x_true_all  = res["x_true_all"]
    x_est_hist  = res["x_est_hist"]
    P_diag      = res["P_diag_hist"]
    nees_hist   = res["nees_hist"]
    omega_hist  = res["omega_hist"]
    deploy_step = res["deploy_step"]
    N_dep       = res["N_dep"]

    if SAVE_FIGS:
        FIG_DIR.mkdir(parents=True, exist_ok=True)

    colors = plt.cm.tab10(np.linspace(0, 1, N_dep))
    labels = ["R", "T", "N"]

    # 1) covariance trace evolution
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Cascade MOS-CI: position covariance trace evolution", fontsize=14)
    for d in range(N_dep):
        m = _mask_per_deputy(res, d)
        ax.plot(t_hr[m],
                np.sqrt(np.nansum(P_diag[d, m, :3], axis=1)),
                lw=0.9, color=colors[d], label=f"D{d}", alpha=0.85)
        ax.axvline(t_hr[deploy_step[d]], color=colors[d], ls=":", lw=0.6, alpha=0.5)
    ax.set_ylabel("sqrt(tr(P_pos)) [m]"); ax.set_xlabel("Time [h]")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "cov_trace.png", dpi=150)
    plt.show()

    # 2) per-deputy errors vs 3sigma
    for d in range(N_dep):
        m = _mask_per_deputy(res, d)
        if not m.any():
            continue
        err = x_est_hist[d] - x_true_all[d]
        three_sigma = 3.0 * np.sqrt(np.maximum(P_diag[d], 0.0))
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        fig.suptitle(f"Cascade MOS-CI — Deputy D{d}: errors vs 3σ", fontsize=14)
        tt = t_hr[m]
        for i in range(3):
            ax = axes[0, i]
            ax.scatter(tt, err[m, i], s=4, alpha=0.5)
            ax.plot(tt,  three_sigma[m, i], "r--", lw=1)
            ax.plot(tt, -three_sigma[m, i], "r--", lw=1)
            ax.set_ylabel(f"ρ_{labels[i]} [m]"); ax.grid(True, alpha=0.3)
            ax.set_title(f"Position {labels[i]}")
        for i in range(3):
            ax = axes[1, i]
            ax.scatter(tt, err[m, 3+i], s=4, alpha=0.5)
            ax.plot(tt,  three_sigma[m, 3+i], "r--", lw=1)
            ax.plot(tt, -three_sigma[m, 3+i], "r--", lw=1)
            ax.set_ylabel(f"ρ̇_{labels[i]} [m/s]"); ax.set_xlabel("Time [h]")
            ax.grid(True, alpha=0.3); ax.set_title(f"Velocity {labels[i]}")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if SAVE_FIGS: fig.savefig(FIG_DIR / f"dep{d}_err_3sig.png", dpi=150)
        plt.show()

    # 3) NEES per deputy
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Cascade MOS-CI: NEES per deputy (target ~ 6)", fontsize=14)
    for d in range(N_dep):
        m = _mask_per_deputy(res, d)
        ax.plot(t_hr[m], nees_hist[d, m], lw=0.7, color=colors[d],
                label=f"D{d}", alpha=0.7)
    ax.axhline(6.0, color="k", ls=":", lw=1, label="dof = 6")
    ax.axhline(12.6, color="r", ls=":", lw=1, label="chi2(6) 95%")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    ax.set_ylabel("NEES"); ax.set_xlabel("Time [h]")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "nees.png", dpi=150)
    plt.show()

    # 4) omega histogram
    if omega_hist:
        ws = np.array([w for (_, _, _, w) in omega_hist])
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle("Cascade MOS-CI: omega* distribution")
        ax.hist(ws, bins=30, edgecolor="k", alpha=0.7)
        ax.axvline(np.mean(ws), color="r", ls="--", lw=1,
                   label=f"mean = {np.mean(ws):.3f}")
        ax.set_xlabel("omega*"); ax.set_ylabel("count")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        if SAVE_FIGS: fig.savefig(FIG_DIR / "omega.png", dpi=150)
        plt.show()

    # 5) 3-D combined truth vs estimate, all deputies on one axis
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("Cascade MOS-CI: TRUTH vs ESTIMATE (all deputies, Hill frame)",
                 fontsize=14)
    for d in range(N_dep):
        m = _mask_per_deputy(res, d)
        if not m.any():
            continue
        xt = x_true_all[d, m]; xe = x_est_hist[d, m]
        ax.plot(xt[:, 0], xt[:, 1], xt[:, 2],
                "-", color=colors[d], lw=1.3, alpha=0.85)
        ax.plot(xe[:, 0], xe[:, 1], xe[:, 2],
                "--", color=colors[d], lw=1.0, alpha=0.6)
        ax.scatter(xt[0, 0],  xt[0, 1],  xt[0, 2],
                   color=colors[d], s=40, marker="o", edgecolor="k")
        ax.scatter(xt[-1, 0], xt[-1, 1], xt[-1, 2],
                   color=colors[d], s=80, marker="*", edgecolor="k",
                   label=f"D{d}")
    ax.scatter(0, 0, 0, color="k", s=140, marker="X",
               label="Chief (origin)")
    ax.set_xlabel("R [m]"); ax.set_ylabel("T [m]"); ax.set_zlabel("N [m]")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "all_3d.png", dpi=150)
    plt.show()

    # 6) per-deputy 3-D panels
    rows, cols = 2, 4
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Cascade MOS-CI: per-deputy 3-D truth (solid) vs estimate (dashed)",
                 fontsize=14)
    for d in range(N_dep):
        ax = fig.add_subplot(rows, cols, d + 1, projection="3d")
        m = _mask_per_deputy(res, d)
        if not m.any():
            ax.set_title(f"D{d} (no data)")
            continue
        xt = x_true_all[d, m]; xe = x_est_hist[d, m]
        ax.plot(xt[:, 0], xt[:, 1], xt[:, 2], "b-",  lw=1.3, label="truth")
        ax.plot(xe[:, 0], xe[:, 1], xe[:, 2], "r--", lw=1.0, label="estimate")
        ax.scatter(0, 0, 0, color="k", s=60, marker="X")
        ax.set_xlabel("R [m]"); ax.set_ylabel("T [m]"); ax.set_zlabel("N [m]")
        ax.set_title(f"D{d}  (deploys k={deploy_step[d]})")
        ax.legend(fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGS: fig.savefig(FIG_DIR / "per_dep_3d.png", dpi=150)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
#                                 entry point
# ──────────────────────────────────────────────────────────────────────────────

def _find_data_dir() -> Path:
    candidates = [SCRIPT_DIR, ROOT.parent if ROOT else SCRIPT_DIR]
    for d in candidates:
        if (d / "apophis_rendezvous_full_D0.npz").is_file():
            return d
    raise FileNotFoundError(
        "Cannot find apophis_rendezvous_full_D0.npz in any known location.\n"
        "Searched: " + ", ".join(str(c) for c in candidates))


if __name__ == "__main__":
    data_dir = _find_data_dir()
    print(f"Using data from: {data_dir}")
    results = run_peer_relnav_v3_sim(data_dir)
    plot_results_v3(results)
