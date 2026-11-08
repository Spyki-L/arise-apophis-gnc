# ARISE Re-docking — Flight Software Package

Flight-representative onboard GNC for the autonomous re-dock of an XB6-class deputy
onto its ESPA RAFTI port. Closed loop, estimated-state-in-the-loop, with runtime
assurance, FDIR, and an autonomy executive around the validated 6-DOF docking core.

## 1. Architecture (closed loop, per deputy)
```
        TRUE state
           │  sensors (VBN bearing+range, star tracker, gyro; noise/bias/latency/dropout)
           │  docking VBN/LIDAR (cm-level relative pose, range-scheduled, close-in)
           │  avionics (coarse Sun sensor, CSAC clock, optical ISL peer ranging)
           ▼
      navigation ── relative-translation EKF (HCW STM + innovation gating
           │          + direct docking-VBN pose fix at close range)
           │        + attitude/gyro-bias estimator (trust-ST, shadow-safe,
           │          Sun-vector aided through star-tracker dropouts)
           ▼  ESTIMATED state
       executive (mission state machine) ── phase + guidance mode
           ▼
    guidance/control ── trained RL policy | verified glideslope+3-DOF-attitude fallback
           ▼
        RTA ── runtime-assurance safety filter (keep-out, cone, speed, rate, contact)
           ▼
       FDIR ── fault detect/isolate/recover (thruster, nav, corridor, overspeed, prop, comm)
           ▼
     actuators ── RCS (Δv, min-impulse-bit, misalignment, dispersion, fault) +
                  reaction wheels (attitude, fine torque)
           ▼
      6-DOF dynamics (HCW surrogate | Basilisk hi-fi) → next TRUE state
           ▼
      telemetry (event + health log)
```
Modules: `config_flight, sensors, docking_sensor, avionics, navigation, controller,
rta, fdir, executive, actuators, telemetry, gnc_loop` (hardware bound through
`hardware_bridge`). Entry points: `run_flight.py` (one deputy + timeline),
`montecarlo.py` (dispersion V&V).

## 2. Interfaces (ICD, per control step, dt = 1 s)
| From → To | Signal | Units |
|---|---|---|
| Sensors → Nav | VBN bearing (az,el), range | rad, m |
| Sensors → Nav | star-tracker MRP σ_BL, gyro ω_B | – , rad/s |
| Docking VBN/LIDAR → Nav | relative pos (port frame) + R, rel-attitude (close-in) | m, deg |
| Sun sensor → Att est | Sun unit-vector (body), ST-dropout backup | – |
| CSAC clock → Executive | onboard time + drift/sync σ | s |
| Optical ISL → FDIR/fleet | peer range, comm health, keep-out alert | m, – |
| Nav → Executive/Control | r̂,v̂ (LVLH), σ̂,ω̂ | m, m/s, –, rad/s |
| Control → RTA | Δv_cmd (LVLH), τ_cmd (body) | m/s, N·m |
| RTA → Actuators | Δv_safe, τ_safe | m/s, N·m |
| Actuators → Dynamics | Δv_real (RCS), τ_real (RWA) | m/s, N·m |
| All → Telemetry | phase, mode, faults, health | – |

## 3. Safety (runtime assurance + FDIR)
The RTA filter makes the guidance command — learned or not — provably safe each step:
keep-out (deputy body stays outside the ESPA shell + margin except through its own
corridor), approach cone, hard speed cap, soft-capture contact-speed cap (axial-only,
preserving lateral authority), and body-rate cap; it projects unsafe commands to the
safe set and raises ABORT on a hard breach. FDIR isolates thruster-stuck, nav-diverged,
corridor-breach, contact-overspeed, propellant-low, and comm-degraded (persistent ISL
loss) faults and recommends SAFE_HOLD or ABORT. The executive then aborts→retreats→
reacquires (bounded) or holds.
**Invariant: every case ends DOCKED or in a SAFE terminal state — never collision.**

## 4. Verification & Validation (`montecarlo.py`)
Dispersion + fault-injection campaign (handoff state/attitude, sensor/actuator seeds,
15% stuck-thruster injection). 200-case result with the **full avionics suite**
(real nominal_v2 hardware + docking VBN/LIDAR, model-based fallback controller, HCW
surrogate):

| Metric | Result | Requirement | Status |
|---|---|---|---|
| Collision / failure outcomes | 0 % | 0 % | **PASS** |
| DOCKED or SAFE-HOLD on every case | 100 % | 100 % | **PASS** |
| Autonomous DOCK rate | 90.5 % | (RL improves) | strong |
| Captured contact distance | mean 9.4 cm, max 10.0 cm | ≤ envelope | **PASS** |
| Contact face-alignment | < 1° | ≤ 12° | **PASS** |
| Propellant per dock | mean 7.0 %, max 22.3 % | < 100 % | **PASS** |

The docking VBN/LIDAR direct pose fix (vs the bearing-only relative-nav) raised the
autonomous dock rate from ~70–75 % to **90.5 %**; the remaining 9.5 % safely hold
(FDIR/executive) rather than force a risky dock — the correct flight behavior. Swapping
the fallback for the HPC-trained RL policy (`controller.FlightController(agent=...)`)
raises it further.


## 7. Hardware fidelity (nominal_v2 traceability)
When the ARISE V2 hardware modules are reachable (`Designs/D_strobe_fullbody/
nominal_v2/modules`), the flight loop wires the REAL parts via `flight/hardware_bridge.py`
- otherwise it falls back to self-contained analytic models, so it always runs.

| Function | Real model (nominal_v2) | Key parameters |
|---|---|---|
| Bus / inertia | Blue Canyon **XB6** (6U) | wet 12.2 kg, J from box inertia |
| Translation | Aerojet **MPS-130** (AF-M315E) `thruster.apply_burn` | Isp 245 s, MIB 10 mN·s, misalign 0.5°, finite-burn loss |
| Attitude | **XACT-50** reaction wheels `ReactionWheelParams.realized_torque` | τ_max 7 mN·m, h 50 mN·m·s, back-EMF + friction + saturation |
| Attitude sensing | BCT **NST Gen-3 ×2** star tracker `measure_error` | ~1″ 3-axis, rate-dependent streak, availability limit |
| Rate sensing | Nav-grade **IMU gyro** `measure` | ARW + bias-instability (Gauss-Markov) + RRW + SF/misalign |
| Docking metrology | **Docking VBN + flash LIDAR** (`docking_sensor`, CIS120 10 µm / 50 mm optics) | range-scheduled: bearing+LIDAR pose to ~cm, rel-attitude < 6 m; FOV 15°, acq 200 m |
| Coarse attitude | ARISE **coarse Sun sensor** `sun_sensor.measure` | Sun-vector 0.094°; nulls 2 observable DOF through ST dropouts |
| Timing / PNT | Microchip **SA.45s CSAC** `pnt_timing` | onboard time, Allan/aging drift (~ns/100 s), ~1 ns fleet sync |
| Fleet comms | **Optical ISL** `isl_comms` | link-budget ranging σ (~0.5 cm), latency, outage → keep-out + comm-health |

Presets follow the same env vars as nominal_v2 (`ARISE_ST_PRESET`, `ARISE_IMU_PRESET`,
`ARISE_THRUSTER_HIFI`). Effect on the V&V: with the MPS-130's 245 s Isp the per-dock
propellant drops from ~20 % (placeholder cold-gas) to **~4–7 %**, and the XB6/XACT-50
attitude chain holds sub-degree face alignment. The **docking VBN/LIDAR** supplies a
direct relative-pose fix at close range (EKF `update_position`, `H=[I,0]`) that collapses
terminal nav error to near zero and resolves relative attitude inside 6 m; the **Sun
sensor** bounds attitude drift during star-tracker dropouts (observable perp-to-Sun DOF
held to ~0.1°); the **CSAC** provides the onboard time reference and inter-deputy sync;
the **optical ISL** provides peer ranging, an ISL-coordinated collision-avoidance
keep-out, and comm-health into FDIR (persistent loss → COMM_DEGRADED → SAFE_HOLD). All
avionics fall back to flight-representative analytic models when nominal_v2 is not
reachable, so the package always runs.

## 5. Flight-representative modeling choices (documented approximations)
- **Attitude via reaction wheels** (fine torque) + **translation via RCS** (min impulse
  bit) — the standard cubesat ADCS split; this is why terminal precision is achievable.
- **3-DOF attitude regulation** to a fixed dock reference, error computed via DCMs to
  avoid the MRP singularity at the ~180° docking attitude; the star tracker is trusted
  directly (0.01°) to avoid shadow-set blending.
- **Soft-capture envelope** (~10 cm): the GNC delivers within the mechanism's latch
  range; the RAFTI latches perform the final mm-level alignment.
- RCS command conditioned in the LVLH frame (a body-thruster allocation is the next
  refinement); dt = 1 s control (the attitude-controllability envelope from §V&V).

## 6. Run
```
python flight/run_flight.py          # one deputy, prints outcome + flight_timeline.png
python flight/montecarlo.py          # dispersion + fault V&V campaign + requirements
```
The terminal-docking RL policy is **auto-detected**: drop the trained checkpoint at
`artifacts/flight_policy.pt` (or set `ARISE_FLIGHT_POLICY`) and both entry points use it
via `controller.load_flight_agent`; otherwise the verified glideslope fallback runs.
`montecarlo.py --ckpt <path>` evaluates a specific policy. Set `DYNAMICS="basilisk"` for
hi-fi dynamics. Not yet flight-certified (no DO-178C artifacts, WCET, or HW-in-the-loop)
— this is a flight-representative software architecture and V&V harness.
