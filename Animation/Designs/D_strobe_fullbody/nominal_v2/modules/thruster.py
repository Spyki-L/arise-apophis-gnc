"""
thruster.py  -  ARISE V2 finite-burn / thruster fidelity (C2), Aerojet MPS-130
==============================================================================
The Basilisk guidance already gives the TRUE corrector dV (tot_dv).  This module adds the
execution-error layer the idealized impulse hides, using the MPS-130 (AF-M315E) parameters in
hardware.py:

  * thrust-vector misalignment (0.5 deg 1-sigma) -> dV pointing error
  * minimum-impulse-bit quantization (~10 mN-s)  -> dV magnitude granularity
  * finite-burn / cosine loss (~2%)              -> realized < commanded magnitude
  * propellant via the rocket equation (Isp 245 s)

apply_burn(dv_cmd_vec, mass) -> realized dV vector, propellant used, execution error.
fleet_delta_v(tcm_list, mass) -> realized total dV + propellant + error stats for the MC tail.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardware as HW

G0 = 9.80665


def _rotate_small(vec, angle_rad, axis):
    """Rodrigues rotation of vec by angle about unit axis."""
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return vec * c + np.cross(axis, vec) * s + axis * (axis @ vec) * (1 - c)


def apply_burn(dv_cmd_vec, mass_kg, rng=None, thr: HW.MPS130Thruster = None):
    """Realize a commanded dV vector through the MPS-130 error model."""
    rng = rng or np.random.default_rng()
    thr = thr or HW.MPS130
    # identity-when-absent hi-fi execution knobs (exported by thruster_hifi.py under
    # ARISE_THRUSTER_HIFI); unset -> exact flat MPS-130 constants.
    isp_s = float(os.getenv("ARISE_THRUSTER_ISP_S", thr.isp_s))
    mib_Ns = float(os.getenv("ARISE_THRUSTER_MIB_NS", thr.mib_quantization_Ns))
    loss_frac = float(os.getenv("ARISE_THRUSTER_LOSS_FRAC", thr.finite_burn_loss_frac))
    dv_cmd = np.asarray(dv_cmd_vec, float)
    mag = np.linalg.norm(dv_cmd)
    if mag < 1e-12:
        return np.zeros(3), 0.0, dict(mag_cmd=0.0, mag_real=0.0, point_err_deg=0.0, n_mib=0)
    u = dv_cmd / mag

    # 1) misalignment: tilt by a Gaussian angle about a random perpendicular axis
    ang = abs(rng.normal(0.0, np.radians(thr.misalignment_deg_sigma)))
    perp = np.cross(u, rng.normal(size=3))
    if np.linalg.norm(perp) < 1e-9:
        perp = np.array([1.0, 0, 0])
    u_real = _rotate_small(u, ang, perp)

    # 2) MIB magnitude quantization: dV step = MIB / mass
    dv_step = mib_Ns / mass_kg
    n_mib = max(1, int(round(mag / dv_step)))
    mag_q = n_mib * dv_step

    # 3) finite-burn / cosine loss
    mag_real = mag_q * (1.0 - loss_frac)

    dv_real = mag_real * u_real
    # propellant via rocket equation
    m_prop = mass_kg * (1.0 - np.exp(-mag_real / (isp_s * G0)))
    err = dict(mag_cmd=float(mag), mag_real=float(mag_real),
               point_err_deg=float(np.degrees(ang)),
               mag_err_mps=float(mag_real - mag), n_mib=int(n_mib))
    return dv_real, float(m_prop), err


def fleet_delta_v(tcm_list, mass_kg, rng=None, thr: HW.MPS130Thruster = None):
    """tcm_list: list of commanded dV vectors (m/s) for one deputy (deploy + TCMs).
    Returns realized total |dV|, propellant, and execution-error summary."""
    rng = rng or np.random.default_rng()
    thr = thr or HW.MPS130
    m = mass_kg; tot_real = 0.0; tot_prop = 0.0; perr = []; merr = []
    for dv in tcm_list:
        dvr, mp, e = apply_burn(dv, m, rng, thr)
        tot_real += np.linalg.norm(dvr); tot_prop += mp; m -= mp
        perr.append(e["point_err_deg"]); merr.append(e["mag_err_mps"])
    return dict(dv_total_mps=float(tot_real), propellant_kg=float(tot_prop),
                final_mass_kg=float(m), mean_point_err_deg=float(np.mean(perr)),
                rms_mag_err_mps=float(np.sqrt(np.mean(np.square(merr)))))


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    print("ARISE V2 thruster (MPS-130) sanity")
    print("  Isp %.0f s, MIB %.0f mN-s, misalign %.1f deg, finite-burn loss %.0f%%"
          % (HW.MPS130.isp_s, HW.MPS130.min_impulse_bit_Ns*1e3, HW.MPS130.misalignment_deg_sigma,
             HW.MPS130.finite_burn_loss_frac*100))
    # single commanded 0.30 m/s deploy on a 12.2 kg deputy
    dvc = np.array([0.30, 0.0, 0.0])
    samples = [apply_burn(dvc, HW.XB6.wet_mass_kg, rng) for _ in range(2000)]
    pe = np.array([s[2]["point_err_deg"] for s in samples])
    me = np.array([s[2]["mag_err_mps"] for s in samples])
    print("  0.30 m/s burn: pointing err mean %.3f deg (1-sig in ~0.5),  mag err mean %.4f m/s"
          % (pe.mean(), me.mean()))
    print("  dV step (MIB/mass) = %.4f mm/s ; propellant/burn ~ %.4f g"
          % (HW.MPS130.mib_quantization_Ns/HW.XB6.wet_mass_kg*1e3, samples[0][1]*1e3))
    # fleet of 8 deputies, ~0.30 m/s each + a couple of small TCMs
    tcms = [np.array([0.30,0,0]), np.array([0,0.02,0]), np.array([0.01,0,0.005])]
    fl = [fleet_delta_v(tcms, HW.XB6.wet_mass_kg, rng) for _ in range(8)]
    tot = sum(f["dv_total_mps"] for f in fl); prop = sum(f["propellant_kg"] for f in fl)
    print("  fleet(8): realized total dV %.3f m/s, propellant %.1f g (MPS-130 capacity is huge)"
          % (tot, prop*1e3))
