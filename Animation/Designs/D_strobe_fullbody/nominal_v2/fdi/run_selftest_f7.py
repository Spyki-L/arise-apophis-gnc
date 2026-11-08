#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_selftest_f7.py — F7 proofs: chief-led recovery ladder + CBF-safe disposal.

  T1 detect: consensus + PEER-CBF convict the suspect
  T2 robust fusion: a lying ISL self-report cannot override the two external sources
  T3 honest fusion: all sources agree
  T4 a correctable fault is recovered by a commanded correction
  T5 uncorrectable + collision risk -> CBF-safe DISPOSE (collision_free)
  T6 disposal is necessary (do-nothing collides) AND sufficient (cleared) over the horizon
  T7 chief-fault -> deterministic leader election
"""
import numpy as np
from recovery_ladder import (RecoveryLadder, FaultClass, Action, fuse,
                            cbf_safe_disposal, elect_leader, _min_separation)
ok = 0; tot = 0
def check(name, cond, extra=""):
    global ok, tot; tot += 1
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else "")); ok += int(bool(cond))

lad = RecoveryLadder(d_safe=50.)
res = lad.run(suspect=3, consensus_label=FaultClass.MODE,
              vision_fn=lambda s: (FaultClass.MODE, 0.9), isl_logs={3: FaultClass.MODE})
check("T1 detect: consensus convicts the suspect", any("detect" in m for _, m in res["log"]))

f2, _ = fuse(FaultClass.TUMBLE, FaultClass.TUMBLE, FaultClass.NONE)
check("T2 robust fuse: lying ISL self-report ignored", f2 is FaultClass.TUMBLE, f"fused={f2.name}")

f3, c3 = fuse(FaultClass.COMM, FaultClass.COMM, FaultClass.COMM)
check("T3 honest fuse: all sources agree", f3 is FaultClass.COMM and c3 > 0.9)

check("T4 correctable fault -> recovered", res["action"] is Action.MODE_SWITCH and res["recovered"],
      f"action={res['action'].name}")

fleet_pos = [np.array([0, 0, 0.]), np.array([0, 80, 0.]), np.array([0, 0, 300.]), np.array([80, 0, 0.])]
fleet_vel = [np.zeros(3), np.zeros(3), np.array([0, 0, -1.0]), np.zeros(3)]
lad2 = RecoveryLadder(d_safe=50.)
res2 = lad2.run(suspect=2, consensus_label=FaultClass.TUMBLE,
                vision_fn=lambda s: (FaultClass.TUMBLE, 0.92), isl_logs={2: FaultClass.NONE},
                fleet_pos=fleet_pos, fleet_vel=fleet_vel,
                suspect_state=(fleet_pos[2], fleet_vel[2]))
check("T5 uncorrectable + collision -> CBF-safe DISPOSE", res2["action"] is Action.DISPOSE and res2["collision_free"],
      f"dV={np.linalg.norm(res2['dv']):.2f} m/s collision_free={res2['collision_free']}")

hp = [fleet_pos[j] for j in (0, 1, 3)]; hv = [fleet_vel[j] for j in (0, 1, 3)]
do_nothing = _min_separation(fleet_pos[2], fleet_vel[2], hp, hv, 600., 5.)
after = _min_separation(fleet_pos[2], fleet_vel[2] + res2["dv"], hp, hv, 600., 5.)
check("T6 disposal necessary (do-nothing collides) AND sufficient (cleared)",
      do_nothing <= 50. and after > 50., f"do-nothing={do_nothing:.1f} -> {after:.1f} m")

res3 = RecoveryLadder().run(suspect=None, consensus_label=FaultClass.NONE,
                            vision_fn=lambda s: (FaultClass.NONE, 0), isl_logs={},
                            chief_faulted=True, healthy=[2, 4, 5, 0])
check("T7 chief-fault -> deterministic leader election", res3["new_leader"] == 0 and res3["action"] is Action.TAKEOVER,
      f"new_leader=D{res3['new_leader']}")

print(f"\n{ok}/{tot} passed")
