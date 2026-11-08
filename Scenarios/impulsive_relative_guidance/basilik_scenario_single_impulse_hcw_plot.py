"""
basilik_scenario_single_impulse_hcw_plot.py
────────────────────────────────────────────

Purpose
-------
Minimal Basilisk scenario that demonstrates a single-impulse Hill–Clohessy–
Wiltshire (HCW) maneuver:

- One deputy is released at a specified deploy time,
- Receives a single HCW-designed LVLH ΔV 1 second later,
- Then coasts for 5.80 hours to reach a commanded RTN offset relative to a chief,
- Finally, the script plots the inertial XY trajectories of chief and deputy.

The script is written against the current Basilisk API — no deprecated
`addEvent` / `setActive` calls — and uses a `SysModel`-based impulse
scheduler instead.

Scenario setup
--------------
User parameters at the top:

- `DEPLOY_EPOCH_S`      : deploy time [s] since simulation epoch.
- `BURN_EPOCH_S`        : burn time [s] = DEPLOY_EPOCH_S + 1 s.
- `RENDEZVOUS_EPOCH_S`  : end time [s]; total coast ≈ 5.80 h.
- `R_F_LVLH_KM`         : desired final relative offset in LVLH [km].
        e.g. [ +2.65, +3.77, −1.49 ] km.
- `M_DEP`               : deputy mass [kg].
- `DYN_RATE_NS`         : dynamics step, default 1 Hz.

Orbit model:
- Both chief and deputy start on the same simple **heliocentric circular**
  orbit (toy model):

    * Central body: Sun, using `orbitalMotion.MU_SUN`.
    * Semi-major axis: `a_c = 4.5e11` m (≈ 3 AU).
    * Initial state:
        r0_N = [a_c, 0, 0] m,
        v0_N = [0, a_c n, 0] m/s,
      where `n = sqrt(MU_SUN / a_c³)` is the mean motion.

Maneuver design (HCW)
---------------------
The HCW helper:

    hcw_initial_vel(rf_L_m, n, dt) -> v_imp_L

computes the LVLH velocity at t = 0 that drives:

- initial relative position:  ρ(0) = 0,
- to a final offset:          ρ(dt) = rf_L_m (in LVLH),
- under linear HCW dynamics over time dt.

Given:

- `rf_L_m` = `R_F_LVLH_KM * 1e3` (desired final offset in metres),
- `n`      = chief mean motion (from the heliocentric orbit),
- `dt`     = `RENDEZVOUS_EPOCH_S − BURN_EPOCH_S`,

the script computes:

- `v_imp_L`  : LVLH ΔV vector [m/s],
- transforms LVLH → inertial using a simple RTN frame at t=0:
      r̂ = r0/‖r0‖
      ĥ = (r0×v0)/‖r0×v0‖
      t̂ = ĥ×r̂
      C_NL = [ r̂ t̂ ĥ ]
- and finally:

      dv_N = C_NL @ v_imp_L

The printed line:

    |ΔV| (LVLH) = xxx m/s

reports the LVLH impulse magnitude.

Impulse modeling (finite burn)
------------------------------
An `ExtForceTorque` effector is added to the deputy:

- For a 1-second effective burn:

      F_imp_N = dv_N * M_DEP   [N]

- The effector is controlled by a small `SysModel`:

    class ImpulseScheduler(sysModel.SysModel):
        - At `BURN_EPOCH_S` (in simulation nanoseconds), sets:
              eff.extForce_N = F_imp_N,
          for exactly one step; otherwise:
              eff.extForce_N = [0,0,0].

- This approximates an impulsive ΔV using a 1 s boxcar thrust.

Simulation structure
--------------------
- Sim core:
    sim  = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("proc")
    sim.CreateNewTask("dyn", DYN_RATE_NS)

- Two spacecraft:
    * Chief: mass 200 kg, identity inertia, initial state (r0_N, v0_N).
    * Deputy: mass `M_DEP`, identity inertia, same initial state.

- Both are added to the "dyn" task.

- A `ProgressPrinter` `SysModel` is also attached to "dyn" to print the
  current simulation time every 0.25 h of simulated time.

Recording and output
--------------------
- Chief and deputy states are recorded via:

    chief_rec = chief.scStateOutMsg.recorder()
    dep_rec   = dep.scStateOutMsg.recorder()

- After running the simulation from t = 0 to `RENDEZVOUS_EPOCH_S`:

    sim.ConfigureStopTime(macros.sec2nano(RENDEZVOUS_EPOCH_S))
    sim.ExecuteSimulation()

- The script builds inertial position histories (in km):

    rC_km, rD_km

from the recorders and then produces a single plot:

- ECI XY trajectories:
    * Chief: X vs Y (solid line).
    * Deputy: X vs Y (solid line).
    * Filled markers at t=0 for visual reference
      (chief: circle, deputy: x).

Plot details:

- Aspect ratio set to equal for a clean XY view:
    plt.gca().set_aspect("equal", "box")
- Labels:  x_N [km], y_N [km]
- Title:   “ECI XY Trajectories”
- Simple legend and `tight_layout()`.

Usage
-----
- Tune `DEPLOY_EPOCH_S`, `BURN_EPOCH_S`, `RENDEZVOUS_EPOCH_S`, and
  `R_F_LVLH_KM` as desired.
- Optionally replace the heliocentric toy orbit with a more realistic chief
  ephemeris.
- Run:

    python basilik_scenario_single_impulse_hcw_plot.py

- The script will:
    * Print the LVLH impulse magnitude,
    * Run the Basilisk simulation to the rendezvous epoch,
    * Plot the chief and deputy inertial XY trajectories showing the effect of
      the single HCW ΔV.
"""


from __future__ import annotations
import sys, numpy as np, numpy.linalg as la
from Basilisk.utilities   import macros, SimulationBaseClass, orbitalMotion, unitTestSupport
from Basilisk.simulation  import spacecraft, extForceTorque
from Basilisk.architecture import sysModel

# ─────────── USER PARAMETERS ───────────────────────────────────────────────
DEPLOY_EPOCH_S      = 10_023.0
BURN_EPOCH_S        = DEPLOY_EPOCH_S + 1.0
RENDEZVOUS_EPOCH_S  = 31_170.0                 # 5.80 h coast
R_F_LVLH_KM         = np.array([2.65, 3.77, -1.49])   # desired final offset
M_DEP               = 12.0                    # kg
DYN_RATE_NS         = macros.sec2nano(1.0)    # 1-Hz dynamics step

# ─────────── HCW HELPER ────────────────────────────────────────────────────
def hcw_initial_vel(rf_L_m: np.ndarray, n: float, dt: float) -> np.ndarray:
    """Initial LVLH velocity that drives r0=0 ➜ rf_L in time dt (HCW)."""
    sn, cs = np.sin(n*dt), np.cos(n*dt)
    A = np.array([[ sn/n,           2*(1-cs)/n ],
                  [ -2*(1-cs)/n,    (4*sn - 3*n*dt)/n ]])
    vx0, vy0 = la.solve(A, rf_L_m[:2])
    vz0 = n*rf_L_m[2] / sn
    return np.array([vx0, vy0, vz0])

# ─────────── BUILD BASILISK SIMULATION SHELL ───────────────────────────────
sim  = SimulationBaseClass.SimBaseClass()
proc = sim.CreateNewProcess("proc")
sim.CreateNewTask("dyn", DYN_RATE_NS)          # dynamics task “dyn”

col = lambda v: [[v[0]], [v[1]], [v[2]]]       # 3×1 column helper

# chief + deputy: simple heliocentric circular orbit (replace with real ephem)
mu_sun = orbitalMotion.MU_SUN                  # m³ s⁻²
a_c    = 4.5e11                                # m  (semi-major axis ≈ 3 AU)
n      = np.sqrt(mu_sun / a_c**3)              # rad s⁻¹  (mean motion)
r0_N   = np.array([a_c, 0., 0.])               # inertial position [m]
v0_N   = np.array([0., a_c*n, 0.])             # inertial velocity [m s⁻¹]

def make_spacecraft(tag: str, mass: float):
    sc = spacecraft.Spacecraft(); sc.ModelTag = tag; sc.hub.mHub = mass
    sc.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d([1,0,0, 0,1,0, 0,0,1])
    sc.hub.r_CN_NInit, sc.hub.v_CN_NInit = col(r0_N), col(v0_N)
    sc.hub.sigma_BNInit   = [[0.],[0.],[0.]]; sc.hub.omega_BN_BInit = [[0.],[0.],[0.]]
    sim.AddModelToTask("dyn", sc); return sc

chief = make_spacecraft("chief", 200.0)
dep   = make_spacecraft("deputy", M_DEP)

# ─────────── COMPUTE ΔV (HCW) ──────────────────────────────────────────────
dt_coast = RENDEZVOUS_EPOCH_S - BURN_EPOCH_S
v_imp_L  = hcw_initial_vel(R_F_LVLH_KM*1e3, n, dt_coast)         # m s⁻¹ (LVLH)
r_hat = r0_N/la.norm(r0_N); h_hat = np.cross(r0_N,v0_N); h_hat/=la.norm(h_hat)
t_hat = np.cross(h_hat, r_hat); C_NL = np.column_stack((r_hat,t_hat,h_hat))
dv_N  = C_NL @ v_imp_L                                           # m s⁻¹ (inertial)
print(f"|ΔV| (LVLH) = {la.norm(v_imp_L):.3f} m/s", flush=True)

# ─────────── IMPULSE EFFECTOR (1-s FINITE FORCE) ───────────────────────────
F_imp_N = dv_N * M_DEP                                           # N (Δt = 1 s)
FT = extForceTorque.ExtForceTorque(); FT.ModelTag = "oneImpulse"
FT.extForce_N = [0., 0., 0.]
dep.addDynamicEffector(FT); sim.AddModelToTask("dyn", FT)

class ImpulseScheduler(sysModel.SysModel):
    """Turn the force on for exactly one step at BURN_EPOCH_S."""
    def __init__(self, eff, burn_sec, force_vec):
        super().__init__(); self.eff, self.burn_ns = eff, macros.sec2nano(burn_sec)
        self.force, self.fired = force_vec, False
    def UpdateState(self, now_ns):
        if not self.fired and now_ns >= self.burn_ns:
            self.eff.extForce_N = self.force; self.fired = True
        else:
            self.eff.extForce_N = [0., 0., 0.]
sim.AddModelToTask("dyn", ImpulseScheduler(FT, BURN_EPOCH_S, F_imp_N.tolist()))

# ─────────── HEARTBEAT MODULE FOR CONSOLE FEEDBACK ─────────────────────────
class ProgressPrinter(sysModel.SysModel):
    def __init__(self, step_hrs=0.25):
        super().__init__(); self.step_ns = macros.sec2nano(step_hrs*3600); self.next_ns = self.step_ns
    def UpdateState(self, now_ns):
        if now_ns >= self.next_ns:
            print(f"  … simulation t = {now_ns*macros.NANO2SEC/3600:.2f} h", flush=True)
            self.next_ns += self.step_ns
sim.AddModelToTask("dyn", ProgressPrinter())

# ─────────── STATE RECORDERS ───────────────────────────────────────────────
chief_rec = chief.scStateOutMsg.recorder(); dep_rec = dep.scStateOutMsg.recorder()
sim.AddModelToTask("dyn", chief_rec); sim.AddModelToTask("dyn", dep_rec)

# ─────────── RUN SIMULATION ───────────────────────────────────────────────
sim.InitializeSimulation()
print("Simulation started — please wait …", flush=True)
sim.ConfigureStopTime(macros.sec2nano(RENDEZVOUS_EPOCH_S))
sim.ExecuteSimulation()
print("Simulation complete.", flush=True)

# ─────────── PLOT (import AFTER run to avoid backend overhead) ─────────────
import matplotlib.pyplot as plt

rC_km = np.vstack([chief_rec.r_BN_N[:,j] for j in range(3)]).T / 1e3
rD_km = np.vstack([ dep_rec.r_BN_N[:,j] for j in range(3)]).T / 1e3

plt.figure(figsize=(6,6))
plt.plot(rC_km[:,0], rC_km[:,1], label="Chief"); plt.plot(rD_km[:,0], rD_km[:,1], label="Deputy")
plt.scatter(*rC_km[0,:2], c="k", marker="o"); plt.scatter(*rD_km[0,:2], c="k", marker="x")
plt.xlabel("$x_N$  [km]"); plt.ylabel("$y_N$  [km]"); plt.gca().set_aspect("equal")
plt.title("ECI XY Trajectories"); plt.legend(); plt.tight_layout(); plt.show()
