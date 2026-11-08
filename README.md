# ARISE: Apophis swarm GNC on Basilisk

Simulation code for ARISE, a reusable swarm of eight XB6 CubeSats that ride an ESPA-class carrier out to asteroid (99942) Apophis for its April 2029 Earth flyby. The deputies separate from Orbit Fab RAFTI ports on the ring, damp their separation transients, fly relative guidance to imaging stations around the asteroid, navigate off each other and off Apophis, and then come back and re-dock on the same ports they left from.

Everything is built on top of [Basilisk](https://github.com/AVSLab/basilisk) (AVS Lab, CU Boulder). This repo holds the mission-specific scenarios, controllers, filters and tooling, not Basilisk itself.

I wrote this as part of my graduate work in the SpaceTREx Lab at the University of Arizona. It's research code: some scripts are polished, others are the experiment that produced one figure. The folder layout below tells you which is which.

## Layout

| Folder | What's in it |
|---|---|
| `GNC/` | Importable package. `Control/` (fleet attitude, port-Hamiltonian and SE(3) controllers), `Guidance/`, `Navigation/` (Hill-frame EKF, swarm EKF), and `Scenario/Apophis/` with the relative-navigation and FDI experiments |
| `Scenarios/` | Basilisk runners grouped by method: Lambert and global Δv optimization, single-impulse HCW/MROE guidance, LVLH station keeping, RHIG rendezvous, ESPA SE(3) separation, Layer-1 RAFTI damping, 6-DOF attitude and line-of-sight pointing, close-approach ephemeris |
| `sim_core/` | Small reusable sim loop and mean relative orbit element (MROE) helpers |
| `Analysis/` | Post-processing and plotting for the scenarios |
| `Animation/scripts/guidance/` | High-fidelity fleet rendezvous guidance used by the mission pipeline |
| `Animation/scripts/redocking/` | Swarm return and re-docking: Lambert return (Phase A), DDPG/TD3(+HER) docking agents (Phase B), HCW and Basilisk 6-DOF backends, a flight-software loop, Monte Carlo eval and Blender film tooling |
| `Animation/Designs/D_strobe_fullbody/nominal_v2/` | Nominal mission run and its hardware models (star tracker, IMU, reaction wheels, thrusters, sun sensor, ISL comms, timing) |
| `.../nominal_v2/fdi/` | Fault detection, isolation and recovery for the swarm: fault library, innovation monitors, conformal calibration, behavior-tree executive, formal checks and the Monte Carlo campaigns behind the FDIR paper |
| `Apophis_misc/lib/` | The original `MiniSim` and `MROE` modules that several scenarios still import |
| `Utils/` | `.npz` to Excel round-trip |

## Getting started

1. Build Basilisk from source following its install guide, then activate its virtual environment.
2. `pip install -r requirements.txt` (the RL and dashboard packages are only needed for `redocking/` and `fdi/`).
3. Run scripts from the repo root so `GNC` resolves as a package.

Most experiments read trajectory bundles instead of re-simulating every time. Those `.npz` files are not in the repo (a single fleet run is tens of megabytes), so generate them first:

```
python Scenarios/rhig_rendezvous/basilik_rhig_apophis_rendezvous_fleet.py
```

That writes `apophis_rendezvous_full_D{0..7}.npz` and `viz_inputs_D{0..7}.npz`. Copy or point the `GNC/Scenario/Apophis/` scripts at them; each of those scripts loads bundles from its own directory. Ephemerides are also left out: get `de430.bsp` from the NAIF generic kernels and the Apophis SPK (`20099942.bsp`) from JPL Horizons.

## Known rough edges

- Scripts were developed on Windows, and a couple import `gnc` in lower case. That works on a case-insensitive filesystem and fails on Linux until you change it to `GNC`.
- `basilik_` is a long-standing typo in file names. Renaming would break imports and saved paths, so it stays.
- Several scripts have tuning toggles at the top of the file rather than CLI flags. Read the header block before running.
- HPC (Slurm) job files are not included; the runners all work on a single machine, just slower.

## Citing

If this code helps your work, please cite the related paper: [ADD: citation for the ARISE / FDIR paper once it is out]

## License

MIT, see `LICENSE`. Basilisk is distributed separately under its ISC license.
