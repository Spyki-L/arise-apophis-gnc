#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FR2 · run every machine-checked property; exit 0 only if ALL are PROVED."""
import sys, os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
mods = ["verify_rta_soundness.py", "verify_cbf_invariance.py", "verify_determinism.py",
        "verify_behavior_tree.py", "verify_planner_wcet.py"]
rc = 0
for m in mods:
    print(); r = subprocess.run([sys.executable, os.path.join(HERE, m)]); rc |= r.returncode
print("\n" + "=" * 74)
print("FR2 FORMAL VERIFICATION:  " + ("ALL PROPERTIES PROVED" if rc == 0 else "*** SOME FAILED ***"))
print("=" * 74)
sys.exit(rc)
