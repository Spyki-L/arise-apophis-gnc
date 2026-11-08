#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_hifi.py  --  ARISE Design D: run the FULL high-fidelity pipeline with ALL 12 gates ON.
================================================================================================
HOW TO USE IN VISUAL STUDIO:
    Open this file and press Run (the green |> triangle).  No command-line arguments needed.

What it does: sets every high-fidelity environment gate to ON, then launches
run_nominal_and_hifi.py.  It auto-selects the Basilisk Python 3.11 (which has numpy + Basilisk);
if Visual Studio is already using that interpreter it runs in-process, otherwise it re-launches
the pipeline under it.  Each gate is identity-when-absent, so comment out any single gate below
to A/B that one upgrade against the baseline.
"""
import os, sys, subprocess, runpy

HERE  = os.path.dirname(os.path.abspath(__file__))
PY311 = r"C:\Program Files\Python311\python.exe"                              # the Python that has Basilisk
DIST3 = r"C:\Basilisk\basilisk-develop\dist3"   # Basilisk source build
DRIVER = os.path.join(HERE, "run_nominal_and_hifi.py")
os.chdir(HERE)

# ---- the 12 high-fidelity gates (set to "0" to disable one for an A/B) ----
GATES = {
    "ARISE_STARTRACKER_HIFI": "1",   # rate-dependent XACT-50 star tracker
    "ARISE_FSM_HIFI":         "1",   # fast-steering-mirror step-stare
    "ARISE_RW_HIFI":          "1",   # reaction-wheel microvibration
    "ARISE_ISL_HIFI":         "1",   # optical inter-satellite link budget
    "ARISE_SUNSENSOR_HIFI":   "1",   # coarse Sun sensor
    "ARISE_IMU_HIFI":         "1",   # nav-grade MEMS gyro Allan model
    "ARISE_THRUSTER_HIFI":    "1",   # MPS-130 / AF-M315E line
    "ARISE_TIMING_HIFI":      "1",   # CSAC + two-way time transfer
    "ARISE_DETECTOR_HIFI":    "1",   # CIS120 PSF/MTF + radiation + PRNU/DSNU
    "ARISE_OCCLUSION_HIFI":   "1",   # soft shadows + self-illumination
    "ARISE_PHOTOGRAM_HIFI":   "1",   # stereophotoclinometry
    "ARISE_NAV_HIFI":         "1",   # Schmidt-Kalman consider covariance + DMC
    "ARISE_RATEAID_EVENTCAM": "1",   # fuse the drift-free event-camera rate into the gyro
}
for k, v in GATES.items():
    os.environ[k] = v
os.environ["PYTHONPATH"] = DIST3 + os.pathsep + os.environ.get("PYTHONPATH", "")

n_on = sum(1 for k, v in GATES.items() if k.endswith("_HIFI") and v == "1")
print("=" * 78)
print(" ARISE Design D  --  HIGH-FIDELITY run  (%d/12 gates ON)" % n_on)
print("=" * 78, flush=True)

if not os.path.exists(DRIVER):
    sys.exit("[error] run_nominal_and_hifi.py not found next to this file: %s" % DRIVER)

# Run under the Basilisk Python 3.11.  If VS launched us under it already -> in-process (so the
# VS debugger/console attaches directly); otherwise re-launch the driver under it.
same = os.path.normcase(os.path.abspath(sys.executable)) == os.path.normcase(os.path.abspath(PY311))
if os.path.exists(PY311) and not same:
    print("[hifi] launching pipeline under Basilisk Python 3.11 ...\n", flush=True)
    raise SystemExit(subprocess.call([PY311, DRIVER], cwd=HERE, env=os.environ))
else:
    if not os.path.exists(PY311):
        print("[hifi] Basilisk Python 3.11 not found at the expected path; using the current "
              "interpreter (%s). Heavy stages still use whatever run_nominal_and_hifi.py selects.\n"
              % sys.executable, flush=True)
    print("[hifi] running run_nominal_and_hifi.py in-process ...\n", flush=True)
    sys.argv = [DRIVER]                                   # no command-line arguments
    runpy.run_path(DRIVER, run_name="__main__")
