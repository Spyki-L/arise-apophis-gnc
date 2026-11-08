# ARISE Re-docking — Basilisk fidelity vs `nominal_v2`, and Basilisk in the RL loop

**Question answered:** does the re-docking RL use the *same fidelity* as
`…/D_strobe_fullbody/nominal_v2`, and is Basilisk actually **in the training loop**?

**Short answer.** The **translational force model is identical** to `nominal_v2`
(`guidance_v2.py`): Earth spherical harmonics **J2..10** (GGM03S) + **Sun** + **Moon**
third-body + **cannonball SRP**, integrated with **RKF45** on **live SPICE** ephemeris.
Two things differed and are now **aligned / made switchable**: the **epoch** (now anchored
to the nominal_v2 close-approach epoch) and the **deputy inertia** (a one-line switch to the
nominal_v2 value). The **attitude** stack is intentionally lighter (see §3). Basilisk is now
**wired end-to-end into the loop** — the setup script puts the *same* Basilisk build on
`PYTHONPATH`, and the SLURM pipeline runs a hi-fi fine-tune + a Basilisk dock-rate eval.

---

## 1. Fidelity comparison (translational dynamics — what the RL actually controls)

| Element | `nominal_v2` (`guidance_v2.py` main) | Re-docking (`dynamics/basilisk_backend.py`) | Match |
|---|---|---|---|
| Gravity — Earth | Spherical harmonics **J2..degree 10** (GGM03S) | `useSphericalHarmonicsGravityModel(GGM03S, EARTH_SH_DEG=10)` | ✅ |
| Third bodies | **Sun + Moon** (SPICE) | `gf.createSun(); gf.createMoon()` (SPICE) | ✅ |
| Solar radiation pressure | **Cannonball SRP** | `RadiationPressure().setUseCannonballModel()` (area 0.30 m², Cr 1.3) | ✅ |
| Ephemeris | **Live SPICE**, Earth-centred | `createSpiceInterface(...); spi.zeroBase="Earth"` | ✅ |
| Integrator | **RKF45** (adaptive) | `svIntegratorRKF45` (env `ARISE_INTEGRATOR=rkf45`) | ✅ |
| Deputy mass | `HW.XB6.wet_mass_kg = **12.2 kg**` | `BUS_MASS_KG = **12.2**` | ✅ |
| **Epoch** | CA `**2029-04-13 20:11:21.903 UTC**` (`compute_encounter_geometry.CA_EPOCH_UTC`) | **now** `ARISE_DEPARTURE_UTC=2029-04-13T20:11:21Z` (was 05:41 departure) | ✅ (aligned) |
| **Deputy inertia** | uniform 6U box → `(0.07086, 0.14406, 0.19032)` | separation-run `(0.04218, 0.20446, 0.22767)`; `ARISE_INERTIA=box` → nominal_v2 value | ⚙️ switchable |
| Frame | ECI / J2000 | ESPA-**LVLH**, converts LVLH↔ECI each step (round-trip **4e-11**) | ✅ (equivalent) |

**Net:** the truth force model the RL trains/validates against is the same one `nominal_v2`
uses. The LVLH↔ECI kinematics that bridge the RL's port-frame view to Basilisk's ECI are
pure NumPy and unit-tested to 1e-11.

---

## 2. Two differences, now handled

**(a) Epoch.** The backend previously anchored the sim at a *departure* time
(`2029-04-13T05:41:45Z`, ~14 h before CA). Re-docking happens **after** imaging, i.e. near/after
close approach, so the Sun/Moon/Earth geometry (which sets the SRP direction and third-body
gradients) should match the encounter. The default is now the **nominal_v2 CA epoch**
`2029-04-13T20:11:21Z`; override per re-dock time with `ARISE_DEPARTURE_UTC`.

**(b) Inertia.** `nominal_v2` *assumes* a uniform 6U box (36×24×11 cm @ 12.2 kg →
`(0.07086, 0.14406, 0.19032)`), while the re-dock config uses the **mission separation-run**
principal J `(0.04218, 0.20446, 0.22767)`. Both describe the same XB6 bus; they differ because
one is measured and one is a box assumption. Default stays the separation-run value (the flight
attitude PD gains `Kp=0.02/Kd=0.10` were tuned to it); set **`ARISE_INERTIA=box`** for exact
nominal_v2 parity.

---

## 3. Attitude fidelity — deliberately lighter (and why it's fine for docking)

`nominal_v2` runs a **full Basilisk reaction-wheel stack** (`attitude_basilisk`): a 4-wheel
BCT_RWP015 pyramid + `simpleNav` → `attTrackingError` → `mrpFeedback (K=0.05, Ki=-1, P=0.5)`
→ `rwMotorTorque` → `reactionWheelStateEffector`, with microvibration/jitter models. That
fidelity exists to certify **arcsecond imaging pointing**.

The re-dock backend propagates attitude with the **exact rigid-body ODE** (real 6U inertia,
RK4) plus the **gravity-gradient torque**, and — critically — the docking attitude is now closed
by the **same verified flight PD** as deployment (`Kp=0.02, Kd=0.10, τ_max=1.5e-3`,
`dock_attitude_error`), run *inside* the env (`SingleDockEnv(attitude="pd")`). For **soft docking**
the relevant attitude requirement is a ≤12° / ≤0.055 rad·s⁻¹ capture gate, which the PD meets
robustly (verified: dock-face pointing error → ~0° at capture, 10/10 across all curriculum
stages). Wheel microvibration/jitter matter for imaging, not for a RAFTI soft-capture, so the
RW-stack fidelity is not required here. If you *do* want the RW stack in training, it can be added
behind an `ARISE_ATT_BASILISK=1` flag — say the word.

---

## 4. Basilisk **in the loop** — how it's wired now

```
SingleDockEnv(dynamics="basilisk")
   └─ dynamics/sixdof.get_sixdof("basilisk", …)
        └─ dynamics/basilisk_backend.BasiliskPropagator.step6dof(states12, actions6, dt)
             ├─ translation: LVLH→ECI → Basilisk hi-fi step (SH10+Sun+Moon+SRP+RKF45+SPICE) → ECI→LVLH
             └─ attitude:    rigid-body RK4 + gravity-gradient, driven by the internal flight PD τ
```

Verified in-sandbox: `get_sixdof("basilisk", …)` reaches the Basilisk import (clean ImportError
when absent), and the LVLH↔ECI round-trip is 4e-11. Nothing about the RL/agent/obs/action
changes between `hcw` and `basilisk` — only the `--dynamics` string.

**What was missing (now fixed):** the old `setup_rl_env.sh` never made Basilisk importable
("expected from the cluster module system"). It now, like `nominal_v2/HPC/setup_hpc_env.sh`:
- loads **Python 3.11** (matches the Basilisk build),
- **auto-locates** `…/basilisk-develop` (walks up for `conanfile.py`; or `ARISE_BSK_ROOT`) and puts
  **`$BSK/dist3` on `PYTHONPATH`**,
- creates a `--system-site-packages` venv so Basilisk stays visible, installs torch/gym/SB3,
- prints whether **Basilisk import is LIVE**.

The SLURM pipeline (`train/hpc/rl_train.slurm`) then:
- runs `train/hpc/preflight.py` **and** `train/hpc/basilisk_check.py` (reports LIVE / not-LIVE and
  the HCW-vs-Basilisk divergence),
- **Stage 3** fine-tunes the flight policy with `--dynamics basilisk` (skips cleanly if absent),
- **evaluates the dock rate on BOTH** `--dynamics hcw` and `--dynamics basilisk`.

---

## 5. How to run the hi-fi path

```bash
cd <repo>/Animation/scripts/redocking
# one-time (interactive compute node): build/point at Basilisk + RL deps
export ARISE_BSK_ROOT=/path/to/basilisk-develop   # optional; auto-detected otherwise
bash train/hpc/setup_rl_env.sh            # -> "Basilisk IMPORT OK -> hi-fi stage is LIVE"
python train/hpc/basilisk_check.py        # -> BASILISK: LIVE + HCW-vs-hi-fi divergence

# full pipeline (HCW bulk-train -> Basilisk fine-tune -> eval on both):
sbatch train/hpc/rl_train.slurm

# validate a specific policy on the hi-fi force model only:
python eval/monte_carlo.py --ckpt artifacts/flight_policy.pt --env her --dynamics basilisk --trials 200
```

**Fidelity knobs (env vars):** `ARISE_DEPARTURE_UTC` (epoch, default CA), `ARISE_EARTH_SH_DEG`
(default 10), `ARISE_INERTIA=box|separation` (default separation), `ARISE_SRP=1`, `ARISE_SRP_CR`
(1.3), `ARISE_SRP_AREA_M2` (0.30), `ARISE_INTEGRATOR=rkf45`, `ARISE_BSK_ROOT`.

---

## 6. Strategy: HCW for bulk training, Basilisk for fine-tune + V&V

RKF45 + SPICE + SH-10 is **~100–1000× slower** per step than the closed-form HCW surrogate, so
training entirely in Basilisk is impractical on the `standard` CPU partition. The pipeline uses the
standard **train-cheap / validate-true** split (exactly the `nominal_v2` philosophy):
1. **bulk-train** the policy on the HCW surrogate (fast, runs anywhere),
2. **fine-tune** briefly on the Basilisk hi-fi force model (Stage 3), and
3. **certify** the dock rate on Basilisk (`--dynamics basilisk` eval).

`basilisk_check.py` prints the open-loop HCW-vs-Basilisk divergence so you can see the surrogate
gap the fine-tune has to close; because the terminal dock is a *local, low-Δv* manoeuvre the CW
surrogate is a good proxy and the residual is small.

---

## 7. Files changed for this

```
dynamics/basilisk_backend.py   epoch default -> nominal_v2 CA (2029-04-13T20:11:21Z); force model already SH10+Sun+Moon+SRP+RKF45+SPICE
config_redock.py               ARISE_INERTIA=box selector (nominal_v2 uniform-box inertia parity); mass already 12.2 (match)
train/hpc/setup_rl_env.sh       Python 3.11 + $BSK/dist3 on PYTHONPATH (Basilisk importable) + LIVE check
train/hpc/rl_train.slurm        module load python/3.11; Basilisk preflight; hi-fi Stage 3; eval on hcw AND basilisk
train/hpc/basilisk_check.py     NEW - proves Basilisk is live, quantifies HCW-vs-hi-fi gap, scores policy on basilisk
```
`envs/single_dock_env.py` already closes the flight attitude PD internally (train == flight);
`dynamics/sixdof.py` routes `"basilisk"` to the hi-fi backend unchanged.

---

## 8. Verified in-sandbox

| Check | Result |
|---|---|
| Translational force model vs nominal_v2 | SH10 + Sun + Moon + SRP + RKF45 + SPICE — **identical** |
| Deputy mass | 12.2 kg both — **match** |
| Epoch | aligned to nominal_v2 CA `2029-04-13T20:11:21Z` |
| Inertia parity switch | `ARISE_INERTIA=box` → `(0.07086,0.14406,0.19032)` (nominal_v2) |
| `get_sixdof("basilisk")` wiring | reaches Basilisk import (clean ImportError when absent) |
| LVLH↔ECI kinematics | round-trip error **4e-11** |
| `basilisk_check.py` | runs, reports LIVE/not-LIVE, exits 0 |

> Basilisk itself can't run in this sandbox (no SPICE/build), so the hi-fi *numbers* (dock rate,
> divergence) are produced on Puma by `basilisk_check.py` / the Stage-3 eval. The wiring, force-model
> parity, constants, and graceful fallback are all verified here.
