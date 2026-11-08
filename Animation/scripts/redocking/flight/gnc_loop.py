# -*- coding: utf-8 -*-
"""
gnc_loop.py - integrated closed-loop flight GNC for ONE deputy's re-dock.

Per control step:
    TRUE state --sensors--> measurements --nav(EKF)--> ESTIMATED state
    executive(estimate, RTA, FDIR) -> phase + guidance mode
    controller/guidance(ESTIMATE) -> command [dv, tau]
    RTA safety filter -> safe command
    RCS actuator -> REALIZED command (quantised, dispersed, fault)
    FDIR(monitor) -> faults + recovery recommendation
    6-DOF dynamics(TRUE, realized) -> next TRUE state
    telemetry.log

Guidance never sees truth; capture is physical (true state).  This is the
flight-representative loop used for V&V (montecarlo.py) and demo (run_flight.py).
Note: the RCS model conditions the LVLH command vector directly (a body-thruster
allocation is a documented refinement).
"""
from __future__ import annotations
import numpy as np
import config_redock as C
from dynamics import get_sixdof
from flight import config_flight as F
from flight.sensors import VBNSensor, StarTracker, RateGyro
from flight.navigation import RelativeNavEKF, AttitudeEstimator
from flight.controller import FlightController
from flight.rta import RTA
from flight.fdir import FDIR
from flight.executive import Executive
from flight.telemetry import Telemetry
from flight import hardware_bridge as HWB
from flight.docking_sensor import DockingVBN
from flight.avionics import SunSensorHead, OnboardClock, ISLLink
from dynamics.attitude import mrp_to_dcm


class FlightGNC:
    def __init__(self, ports, port, agent=None, seed=0, fault_axis=None, dynamics="hcw",
                 chief_state_fn=None):
        self.ports = ports; self.port = port; self.rng = np.random.default_rng(seed)
        self.dt = F.GNC["dt"]
        self.vbn = VBNSensor(self.rng); self.st = StarTracker(self.rng); self.gyro = RateGyro(self.rng)
        self.ctrl = FlightController(ports, agent=agent)
        self.rta = RTA(ports, dt=self.dt); self.fdir = FDIR(); self.exec = Executive()
        from flight.actuators import RCS
        self.rcs = RCS(self.rng, fault_axis=fault_axis, dt=self.dt)
        self.dock_vbn = DockingVBN(self.rng)          # close-range relative-pose metrology
        self.sun = SunSensorHead(self.rng)            # coarse attitude backup / safe-mode
        self.clock = OnboardClock(self.rng, dt=self.dt)   # CSAC onboard time
        self.isl = ISLLink(self.rng)                  # optical ISL peer ranging + comm health
        _sr = np.array([0.35, -0.55, 0.76]); self.sun_ref = _sr / np.linalg.norm(_sr)  # Sun line (LVLH)
        self.peer_sep_m = 12.0                        # nominal nearest-neighbour separation
        self.fleet_keepout_m = 3.0                    # ISL-coordinated collision-avoidance keep-out
        self.J = HWB.deputy_inertia()          # real XB6 wet-mass inertia when available
        self.prop = get_sixdof(dynamics, C.OMEGA_SURROGATE, self.dt, inertia=self.J,
                               chief_state_fn=chief_state_fn)
        self.tm = Telemetry(port)

    def _guidance(self, mode, r_est, v_est, sigma_est, omega_est):
        if mode == "approach":
            return self.ctrl.command(r_est, v_est, sigma_est, omega_est, self.port)
        if mode == "retreat":
            n_out = r_est / (np.linalg.norm(r_est) + 1e-9)
            dv = np.clip(0.03 * n_out - 0.5 * v_est, -F.ACTUATORS["dv_max_mps"], F.ACTUATORS["dv_max_mps"])
            return dv, np.zeros(3)
        if mode == "hold":
            return np.clip(-v_est, -F.ACTUATORS["dv_max_mps"], F.ACTUATORS["dv_max_mps"]), -0.5 * omega_est
        return np.zeros(3), np.zeros(3)          # idle

    def run(self, true_state0, max_steps=None):
        s = np.asarray(true_state0, float).copy()      # [r,v,sigma,omega] (LVLH truth)
        # nav init: noisy knowledge of the handoff state
        x0 = np.concatenate([s[:3] + self.rng.normal(scale=8, size=3),
                             s[3:6] + self.rng.normal(scale=0.05, size=3)])
        nav = RelativeNavEKF(x0, omega=C.OMEGA_SURROGATE)
        att = AttitudeEstimator(sigma0=s[6:9] + 0.02 * self.rng.normal(size=3))
        dv_prev = np.zeros(3); rta_status = "NOMINAL"; fdir_action = "NOMINAL"
        n_st_drop = n_sun_aid = n_isl_out = n_ca_alert = 0
        traj = []; max_steps = max_steps or F.GNC["max_steps"]

        for step in range(max_steps):
            r_t, v_t, sig_t, om_t = s[:3], s[3:6], s[6:9], s[9:12]
            # --- sensing (from truth) ---
            m_vbn = self.vbn.measure(r_t)
            m_st = self.st.measure(sig_t, om_t)
            m_gyro = self.gyro.measure(om_t, self.dt)
            # --- navigation (estimate) ---
            nav.predict(self.dt, dv_prev); nav.update(m_vbn)
            # docking VBN/LIDAR: direct relative-pose fix at close range
            p_port_true, _ = self.ports.to_port_frame(r_t, v_t, self.port)
            att_err_true = self.ports.capture_status(r_t, v_t, self.port, sigma=sig_t,
                                                     omega=om_t)["att_err"]
            m_dock = self.dock_vbn.measure(p_port_true, np.radians(att_err_true))
            dock_att_meas = None
            if m_dock is not None:
                R = self.ports.R_port[self.port]
                pos_lvlh = self.ports.capture_pt[self.port] + R @ m_dock["pos_port"]
                nav.update_position(pos_lvlh, R @ m_dock["R_pos"] @ R.T)
                dock_att_meas = m_dock["att_err"]
            # coarse Sun-vector backup when the star tracker drops out
            m_sun = self.sun.measure(mrp_to_dcm(sig_t) @ self.sun_ref)
            sig_est, _ = att.step(m_gyro, self.dt, m_st,
                                  sun_meas_body=m_sun, sun_ref=self.sun_ref)
            if m_st is None:
                n_st_drop += 1
                if att.st_ok is False: n_sun_aid += 1
            # optical ISL peer ranging + comm-health (fleet collision-avoidance aiding)
            _pr, isl_h = self.isl.peer_range(self.peer_sep_m)
            if not isl_h["available"]:
                n_isl_out += 1                          # comm outage -> stale peer knowledge
            elif _pr < self.fleet_keepout_m:
                n_ca_alert += 1                         # ISL-coordinated collision-avoidance alert
            comm_ok = isl_h["available"]
            r_est, v_est = nav.x[:3].copy(), nav.x[3:].copy()
            om_est = m_gyro - att.bias
            self.clock.tick()                          # advance onboard time
            p_est, _ = self.ports.to_port_frame(r_est, v_est, self.port); dist_est = np.linalg.norm(p_est)
            # --- true capture + geometry (for executive/FDIR) ---
            st_true = self.ports.capture_status(r_t, v_t, self.port, sigma=sig_t, omega=om_t)
            # --- executive ---
            phase, mode = self.exec.step(dist_est, st_true["captured"], rta_status, fdir_action, self.dt)
            traj.append(r_t.copy())
            if self.exec.done:
                self.tm.log(step, step * self.dt, phase, self.ctrl.mode, dist_est, rta_status,
                            self.fdir.active, self.rcs.prop_frac, nav.pos_sigma)
                break
            # --- guidance/control (on estimate) ---
            dv_cmd, tau_cmd = self._guidance(mode, r_est, v_est, sig_est, om_est)
            # perturbation feed-forward: cancel the EKF-estimated un-modelled LVLH accel
            # (nav.b) so the loop stops chasing the continuous J2/SRP drift a step late.
            _navc = getattr(F, 'NAV', {})
            if _navc.get('accel_feedforward', True):
                dv_cmd = dv_cmd - float(_navc.get('accel_ff_gain', 1.0)) * nav.b * self.dt
            # --- RTA safety filter ---
            dv_safe, tau_safe, rta_status = self.rta.filter(dv_cmd, tau_cmd, r_est, v_est, om_est, self.port)
            # --- actuator ---
            dv_real, tau_real = self.rcs.apply(dv_safe, tau_safe)
            # --- FDIR ---
            faults, fdir_action = self.fdir.monitor(dv_safe, dv_real, nav.pos_sigma, dist_est,
                                                    st_true["in_cone"], st_true["v_axial"], self.rcs.prop_frac,
                                                    comm_ok=comm_ok)
            # --- dynamics (truth) ---
            act = np.concatenate([dv_real, tau_real])[None, :]
            s = self.prop(s[None, :], act, self.dt)[0]
            dv_prev = dv_real
            self.tm.log(step, step * self.dt, phase, self.ctrl.mode, dist_est, rta_status,
                        faults, self.rcs.prop_frac, nav.pos_sigma)

        if not self.exec.done:
            # window expired without a terminal decision -> SAFE-HOLD abort.  The executive's
            # timeout intent (steps>max_steps -> ABORT -> RETREAT -> SAFE_HOLD) never fires
            # because that threshold equals the loop horizon; a non-converged approach must end
            # SAFE (RTA holds keep-out every step, so nothing has breached), not be frozen
            # mid-APPROACH and scored FAILED.
            self.exec._to("SAFE_HOLD")
        summ = self.tm.summary()
        summ.update(final_phase=self.exec.phase, captured=(self.exec.phase == "DOCKED"),
                    nav_final_err=float(np.linalg.norm(nav.x[:3] - s[:3])),
                    contact=self.ports.capture_status(s[:3], s[3:6], self.port, sigma=s[6:9], omega=s[9:12]),
                    avionics=dict(st_dropouts=n_st_drop, sun_aided_steps=n_sun_aid,
                                  clock_err_ns=float(self.clock.time_error_s * 1e9),
                                  clock_sync_ns=float(self.clock.sync_sigma * 1e9),
                                  isl_outages=n_isl_out, ca_alerts=n_ca_alert))
        return dict(summary=summ, trajectory=np.array(traj), final_state=s)


if __name__ == "__main__":
    from envs.rafti import RaftiPorts
    rr, uu, rad = C.load_espa_ports()
    rcfg = dict(C.RAFTI); rcfg.update(F.CAPTURE_ENVELOPE)
    ports = RaftiPorts(rr, uu, rad, rcfg, deputy_dims=C.ESPA["deputy_dims"], height=C.ESPA["height_m"])
    port = 0
    # handoff: on-corridor, ~60 m out, small closing speed, mild attitude error
    r0 = ports.capture_pt[port] + 60 * ports.dir_B[port]
    s0 = np.concatenate([r0, -0.05 * ports.dir_B[port], 0.2 * np.random.default_rng(0).normal(size=3), np.zeros(3)])
    gnc = FlightGNC(ports, port, agent=None, seed=0)     # fallback controller
    res = gnc.run(s0)
    S = res["summary"]
    print(f"phase={S['final_phase']} captured={S['captured']} steps={S['steps']} "
          f"min_dist={S['min_dist']:.2f} RTA_limited={S['n_rta_limited']} prop={100*S['prop_used_frac']:.1f}% "
          f"nav_err={S['nav_final_err']:.2f}m")
    print("contact:", {k: round(S['contact'][k], 4) for k in ('dist', 'v_axial', 'att_err', 'captured')})
    print("avionics:", {k: (round(v,2) if isinstance(v,float) else v) for k,v in S['avionics'].items()})
    print("flight gnc_loop OK")
