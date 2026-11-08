"""
basilik_scenario_apophis_close_approach.py

Purpose
-------
Use Basilisk's SPICE interface to propagate Earth, Sun, and asteroid (99942) Apophis
over a user-defined window around the 2029 encounter, identify the epoch and distance
of the closest Earth–Apophis approach, and generate quick-look plots.

Overview
--------
1. Locate and load NAIF kernels plus a custom SPK for Apophis.
2. Build a minimal Basilisk simulation with:
   - a gravity body factory (Earth, Sun),
   - a SPICE interface task running at fixed step,
   - an ephemeris converter that turns SPICE states into EphemerisMsgs.
3. Record Earth and Apophis position histories over the full propagation interval.
4. Post-process the recorded states to:
   - compute Earth–Apophis separation vs. time,
   - find the minimum-distance epoch,
   - print the closest-approach UTC and range.
5. Generate two diagnostic plots:
   - Earth–Apophis distance vs. time,
   - heliocentric XY trajectories of Earth and Apophis.

Usage
-----
Run as a stand-alone script after setting START_UTC, DURATION_DAYS, and STEP_SEC.
Requires a working Basilisk install and the Apophis SPK file (20099942.bsp) in
the Basilisk EphemerisData directory.
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
START_UTC     = "2029 JAN 01 00:00:00"   # UTC (TDB) epoch
DURATION_DAYS = 365 * 1                 
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
spice.zeroBase = "Earth"
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
rE    = earthRec.r_BdyZero_N                 # [m] ← fixed
rA    = apophisRec.r_BdyZero_N               # [m] ← fixed
times = earthRec.times() * macros.NANO2SEC      # [s]

dEA  = np.linalg.norm(rA - rE, axis=1)          # [m]
iMin = np.argmin(dEA)

tClose   = datetime.strptime(START_UTC, "%Y %b %d %H:%M:%S") \
           + timedelta(seconds=float(times[iMin]))
dCloseKm = dEA[iMin] / 1e3                      # km

print("\n*** Apophis–Earth Close Approach ***")
print(f"UTC epoch : {tClose.isoformat()}")
print(f"Distance  : {dCloseKm:,.1f} km\n")

# ——— 8. Quick‑look plots ——————————————————————————————————————————
plt.figure()
plt.plot(times / 86400, dEA / 1e3, linewidth=1.2)
plt.plot(times[iMin] / 86400, dEA[iMin] / 1e3, "ro")
plt.xlabel(f"Days since {START_UTC.split()[0]}")
plt.ylabel("Distance [km]")
plt.title("Apophis–Earth Distance vs. Time")
plt.grid(True)

plt.figure()
plt.plot(rE[:, 0] / 1e9, rE[:, 1] / 1e9, label="Earth")
plt.plot(rA[:, 0] / 1e9, rA[:, 1] / 1e9, label="Apophis")
plt.axis("equal")
plt.xlabel("X [Gm]")
plt.ylabel("Y [Gm]")
plt.title("Heliocentric XY Projection")
plt.legend()

if SHOW_PLOTS:
    plt.show()
