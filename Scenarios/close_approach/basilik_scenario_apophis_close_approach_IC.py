"""
basilik_scenario_apophis_close_approach.py

Purpose
-------
Basilisk + SPICE scenario to analyze the 2028–2029 Earth–Apophis encounter:

1. Load NAIF kernels plus a custom SPK for (99942) Apophis.
2. Propagate Earth, Sun, and Apophis with Basilisk’s SPICE interface over a
   multi-year window.
3. Find the epoch and distance of the minimum Earth–Apophis separation.
4. Detect Apophis’s **equatorial-plane crossing** (Z_eq ≈ 0 in the heliocentric
   frame) and annotate it.
5. Produce two publication-style plots:
   - Earth–Apophis distance vs time, with markers for closest approach and
     the equatorial-plane crossing.
   - Heliocentric XY projection of Earth and Apophis trajectories.

What the script does
--------------------
- Builds a minimal Basilisk simulation:

  * SimulationBaseClass.SimBaseClass()
  * One process (“DynProc”) with a single task (“DynTask”) stepping at
    `STEP_SEC`.
  * A `SpiceInterface` configured with:
      - SPICE data path (`SPICE_DIR`),
      - leap seconds, planetary constants/masses, DE430, and Apophis SPK
        (`20099942.bsp`),
      - `UTCCalInit` aligned to `START_UTC`.
  * Registers planets:
      - "Earth", "Sun", "20099942" (Apophis) with `zeroBase = "Sun"`.
  * Uses an `EphemerisConverter` to convert SPICE states into ephemeris
    messages for all bodies, then attaches recorders for:
      - Earth ephemeris (`earthRec`),
      - Apophis ephemeris (`apophisRec`).

- Runs the simulation for:

      DURATION_DAYS × 86400 seconds

  at the specified step size.

Post-processing
---------------
After the run:

1. Extract states:
   - Earth heliocentric positions:  rE(t)  [m]
   - Apophis heliocentric positions: rA(t) [m]
   - Timestamps: `times` [s] relative to `START_UTC`.

2. Closest approach:
   - Compute Earth–Apophis separation:
        dEA(t) = ‖rA(t) − rE(t)‖.
   - Find `iMin = argmin dEA`.
   - Convert to:
        tClose (UTC ISO string),
        dCloseKm = dEA[iMin] / 1e3.
   - Print a short summary block to stdout.

3. Equatorial-plane crossing:
   - Use Apophis heliocentric Z-component, zA(t):
        zA = rA[:, 2].
   - Detect zero-crossings by sign changes between samples.
   - Choose the crossing index nearest the closest-approach index `iMin`.
   - Linearly interpolate between samples to refine:
        tCross (s), dCross (km).
   - If no crossing occurs in the window, the crossing marker is omitted.

Plots
-----
Figure 1 – Earth–Apophis distance vs time
- x-axis: days since `START_UTC`.
- y-axis: distance [km].
- Curves / markers:
    * Thick line for dEA(t).
    * Red circle marking the closest approach.
    * Blue square marking the equatorial-plane crossing (if detected).
- Styling:
    * Large fonts (labels and title ~ 50, ticks ~ 45).
    * Legend in the upper-left corner.
    * A text box to the right containing:
          *** Apophis–Earth Close Approach ***
          UTC epoch : <tClose>
          Distance  : <dCloseKm> km
      placed just outside the right plot margin, with a white background.

Figure 2 – Heliocentric XY projection
- Plots:
    * Earth XY track in Gm (X_E/1e9, Y_E/1e9).
    * Apophis XY track in Gm (X_A/1e9, Y_A/1e9).
- Axis options:
    * `axis("equal")` to preserve geometry.
    * Labels X [Gm], Y [Gm].
    * Title: “Heliocentric XY Projection”.
    * Grid, large fonts, and legend.

Usage
-----
- Adjust `START_UTC`, `DURATION_DAYS`, and `STEP_SEC` at the top as needed
  for different windows around the 2029 close approach.
- Ensure the SPICE kernels listed in `KERNEL_LIST` (especially
  `20099942.bsp`) are present in Basilisk’s `EphemerisData` directory.
- Run:

    python basilik_scenario_apophis_close_approach.py

- The script will:
    * Run the SPICE-based propagation,
    * Print the close-approach epoch and distance,
    * Mark the equatorial-plane crossing (if any),
    * Display both figures if `SHOW_PLOTS` is True.
"""


import os
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt

# — Basilisk imports —————————————————————————————————————————————
from Basilisk import __path__ as bskPath
from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.utilities.simIncludeGravBody import gravBodyFactory
from Basilisk.simulation import spiceInterface
from Basilisk.simulation import ephemerisConverter

# ——— 0. Simulation parameters —————————————————————————————————————
START_UTC     = "2028 JAN 01 00:00:00"   # UTC (TDB) epoch
DURATION_DAYS = 365 * 1.5                 
STEP_SEC      = 10                
SHOW_PLOTS    = True

# ——— 0a. SPICE kernel folder & list ——————————————————————————————
SPICE_DIR = os.path.join(bskPath[0], "supportData", "EphemerisData") + os.sep
KERNEL_LIST = [
    "naif0012.tls",        # leap seconds
    "pck00010.tpc",        # planetary constants & masses
    "de-403-masses.tpc",   # mass table
    "de430.bsp",           # planetary ephemeris
    "20099942.bsp"         # (99942) Apophis SPK
]

# ——— 1. Build the simulation “skeleton” ———————————————————————————
scSim   = SimulationBaseClass.SimBaseClass()
dynProc = scSim.CreateNewProcess("DynProc")
dynProc.addTask(scSim.CreateNewTask("DynTask", macros.sec2nano(STEP_SEC)))

# ——— 2. Register Earth & Sun as gravitating bodies ————————————————
gravFactory = gravBodyFactory()
gravFactory.createBodies("earth", "sun")

# ——— 3. Configure the SPICE interface —————————————————————————————
spice = spiceInterface.SpiceInterface()

# 3a. Tell SPICE where the kernels live
spice.SPICEDataPath = SPICE_DIR

# 3b. ***CRITICAL*** — align the internal epoch with our START_UTC
from datetime import datetime
spice.UTCCalInit = datetime.strptime(START_UTC, "%Y %b %d %H:%M:%S")\
                           .strftime("%Y %B %d, %H:%M:%S.0 (UTC)")

# 3c. Load kernels and register bodies
for kern in KERNEL_LIST:
    spice.loadSpiceKernel(kern, SPICE_DIR)

spice.addPlanetNames(["Earth", "Sun", "20099942"])   # Apophis as NAIF‑ID string
spice.zeroBase = "Sun"
scSim.AddModelToTask("DynTask", spice)

# ——— 4. Convert SPICE states → EphemerisMsgs ——————————————————————
ephemConv = ephemerisConverter.EphemerisConverter()
ephemConv.ModelTag = "EphemConv"
for msg in spice.planetStateOutMsgs:
    ephemConv.addSpiceInputMsg(msg)
scSim.AddModelToTask("DynTask", ephemConv)

# ——— 5. Set up recorders ——————————————————————————————————————————
earthRec   = ephemConv.ephemOutMsgs[0].recorder()  # Earth
apophisRec = ephemConv.ephemOutMsgs[2].recorder()  # Apophis (3rd entry)

scSim.AddModelToTask("DynTask", earthRec)
scSim.AddModelToTask("DynTask", apophisRec)

# ——— 6. Run the simulation ——————————————————————————————————————
scSim.InitializeSimulation()
scSim.ConfigureStopTime(macros.sec2nano(DURATION_DAYS * 86400))
scSim.ExecuteSimulation()

# ——— 7. Post‑process: distance curve & closest‑approach ————————————
rE    = earthRec.r_BdyZero_N                   # [m]
rA    = apophisRec.r_BdyZero_N                 # [m]
times = earthRec.times() * macros.NANO2SEC      # [s]

dEA  = np.linalg.norm(rA - rE, axis=1)          # [m]
iMin = np.argmin(dEA)

tClose   = datetime.strptime(START_UTC, "%Y %b %d %H:%M:%S") \
           + timedelta(seconds=float(times[iMin]))
dCloseKm = dEA[iMin] / 1e3                      # [km]

print("\n*** Apophis–Earth Close Approach ***")
print(f"UTC epoch : {tClose.isoformat()}")
print(f"Distance  : {dCloseKm:,.1f} km\n")

# ——— Prepare plotting variables ——————————————————————————————————
days     = times / 86400               # days since START_UTC
dist_km  = dEA / 1e3                   # distance in km
xE_gm    = rE[:,0] / 1e9               # Earth X in Gm
yE_gm    = rE[:,1] / 1e9               # Earth Y in Gm
xA_gm    = rA[:,0] / 1e9               # Apophis X in Gm
yA_gm    = rA[:,1] / 1e9               # Apophis Y in Gm

# ——— 7a. Locate Apophis’ equatorial-plane crossing ——————————————
zA = rA[:, 2]                                # heliocentric Z [m]
sign_flip_idx = np.where(np.signbit(zA[:-1]) != np.signbit(zA[1:]))[0]

if sign_flip_idx.size:                       # at least one crossing
    # pick the crossing nearest the close-approach epoch
    iCross = sign_flip_idx[np.argmin(np.abs(sign_flip_idx - iMin))]

    # (optional) refine the epoch with linear interpolation
    frac    =  zA[iCross] / (zA[iCross] - zA[iCross+1])
    tCross  = times[iCross]   + frac*(times[iCross+1] - times[iCross])
    dCross  = dist_km[iCross] + frac*(dist_km[iCross+1] - dist_km[iCross])
else:
    iCross, tCross, dCross = None, None, None   # no crossing in window


# -------------------------------
#  Figure 1: Distance vs. Time w/ Equatorial Crossing
# -------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

# 1) The separation curve
ax.plot(days, dist_km, linewidth=3.5, label="Separation")

# 2) Closest‐approach marker
ax.plot(days[iMin], dist_km[iMin], "ro", markersize=10, label="Closest Approach")

# 3) Equatorial‐plane crossing marker
if iCross is not None:
    ax.plot(tCross/86400, dCross, "bs", markersize=10, label="Equatorial Crossing")

# labels, title, grid
ax.set_xlabel(f"Days since {START_UTC.split()[0]}", fontsize=50)
ax.set_ylabel("Distance [km]",               fontsize=50)
ax.set_title("Apophis–Earth Distance vs. Time", fontsize=50)
ax.grid(True)
ax.tick_params(axis="both", labelsize=45)

# move legend into upper‐left corner
ax.legend(loc="upper left", fontsize=45)

# leave room for title+annotations
fig.subplots_adjust(top=0.85, right=0.80)

# closest‐approach text box (upper‐right, just inside new right margin)
info_str = (
    "*** Apophis–Earth Close Approach ***\n"
    f"UTC epoch : {tClose.isoformat()}\n"
    f"Distance  : {dCloseKm:,.1f} km"
)
ax.text(
    1.02, 0.95, info_str,
    transform=ax.transAxes,
    fontsize=50, va="top", ha="left",
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
    clip_on=False
)

# -------------------------------
#  Figure 2: Heliocentric XY Projection
# -------------------------------
plt.figure(figsize=(12, 8))
plt.plot(xE_gm, yE_gm, label="Earth",   linewidth=3.5)
plt.plot(xA_gm, yA_gm, label="Apophis", linewidth=3.5)

plt.axis("equal")
plt.xlabel("X [Gm]", fontsize=50)
plt.ylabel("Y [Gm]", fontsize=50)
plt.title("Heliocentric XY Projection", fontsize=30)
plt.grid(True)
plt.xticks(fontsize=45)
plt.yticks(fontsize=45)
plt.legend(fontsize=45)

if SHOW_PLOTS:
    plt.show()


