#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_refly_stations.py - GA re-optimizer for the tighter-standoff fleet.

Searches 8 deputy LVLH stations to MAXIMIZE whole-body stereo coverage at a target GSD,
subject to: body clearance >= MIN_CLEAR (min range to the ~205 m body) and inter-deputy
separation >= SEP_MIN.  CASE-1 fidelity: fast kinematic flyby (linear relative motion at
the flyby v_rel) + the fixed coverage scorer (occlusion off, per-draw strobe + smooth P_det).

Geometry facts used (all deputies share the flyby v_rel):
  * inter-deputy separation is CONSTANT over the pass = |station_i - station_j|
  * each deputy's body min-range = station distance to the velocity axis (perp component)
So a station built as rho*b_plane_dir + tau*v_hat has min-range = rho exactly.

Usage:
  python optimize_refly_stations.py            # evolve (re-run to continue; checkpointed)
  python optimize_refly_stations.py --final     # validate best on full mesh + write targets npz
"""
import os, sys, glob, math, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "modules"))
import hardware as HW, brdf as BRDF, detector as DET, strobe as STR, apophis_tumbler as TUM

# ---- tunables ----
POP = 32; TBUDGET = 35.0; M_FIT = 3
RHO = (700.0, 850.0)          # min-range shell [m] -> GSD floor 0.14-0.17 m
SEP_MIN = 150.0               # inter-deputy clearance [m]
GSD_OBJ = 0.15                # objective GSD [m]
NFR = 7
STATE = os.path.join(HERE, "refly_ga_state.npz")

# ---- flyby kinematics from the nominal draw ----
DRAW = os.path.join(HERE, "data", "mc", "draw_0000")
ds = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(DRAW, "trajectories", "apophis_rendezvous_full_D*.npz")))]
t = np.asarray(ds[0]["t"], float); rel0 = ds[0]["rD_N"] - ds[0]["rA_N"]; kca = int(np.argmin(np.linalg.norm(rel0, axis=1)))
vrel = (ds[0]["vD_N"] - ds[0]["vA_N"])[kca]; vmag = np.linalg.norm(vrel); vhat = vrel / vmag
tmp = np.array([1.0, 0, 0]) if abs(vhat[0]) < 0.9 else np.array([0, 1.0, 0])
e1 = np.cross(vhat, tmp); e1 /= np.linalg.norm(e1); e2 = np.cross(vhat, e1)
C_A = np.asarray(ds[0]["C_A_N"][kca], float)        # inertial->LVLH (rows R,T,N) at CA
sun = np.load(os.path.join(DRAW, "apophis", "attitude_truth.npz"), allow_pickle=True)["sun_hat"]

MAXR = HW.GSD_LIMIT_M / HW.IFOV_RAD
cab = math.cos(math.radians(30)); caa = math.cos(math.radians(5))
E_def = STR.dark_side_budget(target_snr=STR.DARK_SNR_MIN, range_m=HW.DESIGN_STANDOFF_M)["E_pulse_opt_J"]
Opix = (HW.CIS120.pixel_m / HW.FOCAL_M) ** 2; det = HW.CIS120
qd = det.dark_current_e_s() * 1e-5; q = (det.full_well_e / 2 ** det.adc_bits) / math.sqrt(12.)
snr_Ne = lambda Ne: Ne / np.sqrt(Ne + qd + det.read_noise_e**2 + det.dsnu_e**2 + q**2 + (det.prnu_frac * Ne)**2)
erf = np.vectorize(math.erf); Lref = BRDF.radiance_over_Esun(0.17, 0.17, 0., model="hapke"); TGT = 45.; PW = 6.

def load_mesh(sub=4):
    m = np.load(os.path.join(HERE, "data", "radar_mesh.npz"))
    idx = np.arange(0, len(m["centroids"]), sub)
    return m["centroids"][idx], m["normals"][idx], m["areas"][idx]

def stations(genes):
    th = genes[:, 0]; rho = np.clip(genes[:, 1], *RHO); tau = genes[:, 2]
    return rho[:, None]*(np.cos(th)[:, None]*e1[None, :] + np.sin(th)[:, None]*e2[None, :]) + tau[:, None]*vhat[None, :]

def cams_for(rs):
    cams = []
    for r_s in rs:
        a = vmag**2; b = 2*r_s@vrel; c = r_s@r_s - MAXR**2; disc = b*b - 4*a*c
        if disc <= 0:
            continue
        t1 = (-b - math.sqrt(disc))/(2*a); t2 = (-b + math.sqrt(disc))/(2*a)
        for tt in np.linspace(t1, t2, NFR):
            cams.append(r_s + vrel*tt)
    return np.array(cams)

def coverage(rs, C0, N0, areas, R, h, gsd):
    C = C0@R.T; N = N0@R.T; lit = (N@sun) > 0; cams = cams_for(rs); K = len(cams); F = len(C0); Atot = areas.sum()
    if K < 2:
        return 0.0
    L = np.empty((K, F, 3)); fr = np.zeros((K, F), bool); bE = np.full(F, 9.9); bR = np.full(F, np.inf)
    for i, vp in enumerate(cams):
        look = vp[None, :] - C; rng = np.linalg.norm(look, axis=1); lh = look/rng[:, None]; ce = np.einsum('ij,ij->i', N, lh)
        f = (ce > 0) & (rng <= MAXR); L[i] = lh; fr[i] = f
        e = np.arccos(np.clip(ce, -1, 1)); up = f & (e < bE); bE[up] = e[up]; bR[up] = rng[up]
    st = np.zeros(F, bool)
    for i in range(K):
        if not fr[i].any():
            continue
        for j in range(i+1, K):
            v = fr[i] & fr[j]
            if not v.any():
                continue
            dot = np.einsum('ij,ij->i', L[i], L[j]); st |= v & (dot >= cab) & (dot <= caa)
    seen = bE < np.radians(89); er = np.clip(bE, 0, np.radians(89.9))
    Lsmp = BRDF.radiance_over_Esun(er, er, np.zeros(F), model="hapke", params=h)
    bh = np.arctan2(STR.BODY_RADIUS_M, np.maximum(bR, 1.)); Om = 2*np.pi*(1-np.cos(bh))
    Lr = BRDF.radiance_over_Esun(0.17, 0.17, 0., model="hapke", params=h); Efix = E_def*(Lref/max(Lr, 1e-9))*(TGT/STR.DARK_SNR_MIN)
    Ne = Lsmp*(Efix/(Om*np.maximum(bR, 1.)**2)*0.8)*(math.pi*(HW.APERTURE_MAX_M/2)**2)*Opix/DET.PHOTON_ENERGY_J*det.qe_at_band*det.optical_eff
    fs = np.where(lit, 308., snr_Ne(Ne)); pdet = 0.5*(1+erf((fs - STR.DARK_SNR_MIN)/(math.sqrt(2)*PW)))
    w = st.astype(float)*seen*pdet*((2e-4*bR) <= gsd)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return 100*(areas*w).sum()/Atot

def sep_min(rs):
    s = 1e9
    for i in range(8):
        for j in range(i+1, 8):
            s = min(s, np.linalg.norm(rs[i]-rs[j]))
    return s

C0, N0, areas = load_mesh(4)
def fitness(genes, mfit=M_FIT):
    rs = stations(genes); sp = sep_min(rs); pen = max(0., SEP_MIN - sp)*0.5
    rng = np.random.default_rng(1); cov = 0.
    for _ in range(mfit):
        p = TUM.sample_tumbler(rng); _, dc = TUM.integrate_window(p, np.array([0., 1.])); R = dc[0]; h = BRDF.sample_hapke(rng)
        cov += coverage(rs, C0, N0, areas, R, h, GSD_OBJ)
    return cov/mfit - pen

def seed_pop(rng):
    pop = np.zeros((POP, 8, 3))
    # half: spread b-plane rings (good prior); half: random
    for k in range(POP):
        if k < POP//2:
            az = np.linspace(0, 2*np.pi, 8, endpoint=False) + rng.normal(0, 0.2, 8)
            pop[k, :, 0] = az; pop[k, :, 1] = rng.uniform(720, 800, 8); pop[k, :, 2] = rng.normal(0, 600, 8)
        else:
            pop[k, :, 0] = rng.uniform(0, 2*np.pi, 8); pop[k, :, 1] = rng.uniform(*RHO, 8); pop[k, :, 2] = rng.uniform(-1500, 1500, 8)
    return pop

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--final", action="store_true"); a = ap.parse_args()
    rng = np.random.default_rng(0)
    if os.path.exists(STATE):
        z = np.load(STATE); pop = z["pop"]; gen = int(z["gen"]); best = z["best"]; bestf = float(z["bestf"])
    else:
        pop = seed_pop(rng); gen = 0; best = pop[0]; bestf = -1e9

    if a.final:
        C0f, N0f, areasf = load_mesh(1); rs = stations(best)
        rngv = np.random.default_rng(7); gsds = [0.15, 0.2, 0.3, 0.5, 1.0]; cov = np.zeros(len(gsds))
        for _ in range(8):
            p = TUM.sample_tumbler(rngv); _, dc = TUM.integrate_window(p, np.array([0., 1.])); R = dc[0]; h = BRDF.sample_hapke(rngv)
            cov += np.array([coverage(rs, C0f, N0f, areasf, R, h, g) for g in gsds])
        cov /= 8
        rho = np.clip(best[:, 1], *RHO)
        tgt_lvlh = (C_A @ rs.T).T                   # inertial rel -> LVLH [R,T,N] (m)
        np.savez(os.path.join(HERE, "refly_targets_750m_opt.npz"), DEPUTY_TARGETS_M=tgt_lvlh)
        print("GA best fitness (CASE-1 cov@0.15m, M=%d): %.1f%%" % (M_FIT, bestf))
        print("min-range per deputy [m]:", np.round(rho).astype(int).tolist(), " clearance:", int(rho.min()-205), "m")
        print("inter-deputy min separation [m]: %.0f (limit %.0f)" % (sep_min(rs), SEP_MIN))
        print("coverage vs GSD:", {g: round(c, 1) for g, c in zip(gsds, cov)})
        print("DEPUTY_TARGETS_KM = np.array([")
        for v in tgt_lvlh:
            print("    [%6.3f, %6.3f, %6.3f]," % tuple(v/1e3))
        print("])\nwrote refly_targets_750m_opt.npz")
        return

    t0 = time.time(); ngen = 0
    while time.time() - t0 < TBUDGET:
        fits = np.array([fitness(g) for g in pop]); gi = int(np.argmax(fits))
        if fits[gi] > bestf:
            bestf = float(fits[gi]); best = pop[gi].copy()
        newp = [best.copy()]
        while len(newp) < POP:
            # tournament selection via cached fits
            a1, b1 = rng.integers(POP), rng.integers(POP); p1 = pop[a1] if fits[a1] > fits[b1] else pop[b1]
            a2, b2 = rng.integers(POP), rng.integers(POP); p2 = pop[a2] if fits[a2] > fits[b2] else pop[b2]
            mask = rng.random((8, 3)) < 0.5; child = np.where(mask, p1, p2).copy()
            child[:, 0] += rng.normal(0, 0.3, 8); child[:, 1] += rng.normal(0, 40, 8); child[:, 2] += rng.normal(0, 300, 8)
            child[:, 1] = np.clip(child[:, 1], *RHO)
            newp.append(child)
        pop = np.array(newp); gen += 1; ngen += 1
    np.savez(STATE, pop=pop, gen=gen, best=best, bestf=bestf)
    print("evolved +%d gens (total %d), best CASE-1 cov@0.15m = %.1f%%" % (ngen, gen, bestf))

if __name__ == "__main__":
    main()
