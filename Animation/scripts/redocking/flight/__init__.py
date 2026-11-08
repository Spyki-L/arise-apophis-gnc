"""
ARISE re-docking FLIGHT package
===============================

Flight-representative onboard GNC stack around the validated RL/model-based docking
core.  Closed loop (est-state in the loop, not truth):

    sensors -> navigation (EKF) -> executive (state machine) -> guidance/control
            -> runtime-assurance safety filter -> FDIR -> actuators -> dynamics

Modules
-------
    config_flight   frozen parameter database (limits, thresholds, sensor/actuator specs)
    sensors         VBN relative-nav (bearing+range), star tracker, gyro (noise/bias/latency/dropout)
    navigation      relative-translation EKF + attitude estimator
    controller      trained-RL policy | verified model-based fallback + command conditioning
    rta             runtime-assurance safety filter (keep-out, cone, velocity/contact limits)
    fdir            fault detection / isolation / recovery (abort, retreat, safe-mode)
    executive       autonomy mission state machine
    telemetry       flight event + health logging
    gnc_loop        integrated closed-loop flight GNC (per-deputy)
    montecarlo      dispersion + fault-injection V&V campaign
"""
__version__ = "1.0.0-flight"
