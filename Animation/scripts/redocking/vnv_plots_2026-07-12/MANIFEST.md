# ARISE Re-docking — Neural Flight-Controller V&V Plot Pack

**Generated** 2026-07-12 · **Policy** `artifacts/bc_bsk/ckpt_best.pt` (12→6 ActorNetwork, behavior-cloned
from the Basilisk-retuned glideslope PD) · deployed `artifacts/flight_policy_bsk_bc.pt`.
Applied to the **nominal_v2** avionics stack (Blue Canyon XB6, Aerojet MPS-130 AF-M315E, XACT-50 RWA,
BCT NST Gen-3 star trackers, MEMS IMU; J = [0.0709 0.1441 0.1903] kg·m²).

## Headline (shippable) result
Full-avionics closed-loop Monte-Carlo, **HCW truth, 3000 cases, 15% faults, 3 seeds**:
**DOCKED 84.33% / SAFE-HOLD 15.67% / FAILED 0 (0.000%)**, 0 collisions → rule-of-three 95% UB ≤ 0.10%.
Basilisk **open-loop** dock 100% (150 trials). Basilisk **hi-fi closed loop** matured 0% → ~53%
(0 collisions) — a characterized flight-GNC maturation item that affects the PD fallback identically.

## Contact sheet
`ARISE_redocking_vnv_plots_2026-07-12.pdf` — 4 pages (cover + 3 sections), all 18 figures with captions.

## Figures (`png/`)  — data source in brackets

### RL / Behavior-Cloning training
- **T1_bc_imitation_mse** — BC MSE 5.6e-3→1.2e-5, 120 ep (Basilisk) + 60 ep (HCW). [logs/arise_bc_bsk_22868242.out, arise_bc_22860741.out]
- **T2_curriculum_dock_rate** — strict 6-DOF dock 200/200 across 5 stages vs 15-ep sandbox baseline. [same BC log; handoff §5]
- **T3_pd_gain_tuning** — 6-gain Basilisk re-tune, objective 0.27→1.00, 150/150. [_deliverables/pd_tune/best_gains.json]
- **T4_td3_reward_exploit** — TD3+HER avg50 reward → +13.6k while strict capture → 0% (why raw RL fails). [logs/arise_td3_22846421.out]
- **T5_basilisk_closedloop_ladder** — 0%→7% (accel-bias EKF) →53% (integral guidance). [logs + sign-off §5]
- **T6_nav_integral_sweeps** — integral k_i sweep (opt 3e-3) + nav sweep (dock flat 6.7%). [logs/arise_inttune_22877642.out, arise_navtune_22875945.out]

### Monte-Carlo V&V (3000-case HCW-truth campaign)
- **M1_outcome_breakdown** — per-seed + pooled DOCK/SAFE/FAIL. [artifacts/conf/mc_seed{0,1,2}.csv]
- **M2_mc_convergence** — running dock rate, 3 seeds → ~84%. [conf CSVs]
- **M3_fault_vs_faultfree** — fault-free 2530/2561 dock; fault-injected 439/439 safe-held. [conf CSVs]
- **M4_contact_distribution** — captured contact, all 2530 docks ≤ 10 cm. [conf CSVs]
- **M5_steps_to_dock** — approach duration inside the 900-step RTA window. [conf CSVs]
- **M6_collision_safety** — 0/3000 → rule-of-three 95% UB = 0.10%. [conf CSVs]

### Delta-V / propellant & Basilisk hi-fi closed loop
- **D1_propellant_deltav_distribution** — docked ≤14.8%; worst-case safe-hold abort 37.5% ≪ 100%. [conf CSVs]
- **D2_propellant_hcw_vs_basilisk** — Δv effort 7.1% (HCW) → 21.4% (Basilisk hi-fi). [conf + bsk_vnv CSVs]
- **D3_basilisk_localization_and_result** — baseline PD GNC 0% (localizes gap); policy +fixes 47.5% (n=40) / tuned 53.3% (n=30). [bsk_vnv/mc_basilisk*.csv, int_tune log]
- **D4_basilisk_nav_error** — accel-bias EKF cm-level nav (median 3.3 cm). [bsk_vnv/mc_basilisk.csv]
- **D5_contact_vs_propellant** — docked-case quality, all inside 10 cm envelope. [conf CSVs]
- **D6_requirements_scorecard** — 4 flight requirements, all PASS. [conf CSVs + sign-off §3]

## Honest notes (carried from the 2026-07-11 sign-off)
1. **Propellant:** docked cases use ≤14.8% of tank; the 37.5% worst case is a *safe-hold abort* holding
   the full 900-step window, not a docking cost. All bounded ≪ 100%. (Sign-off quoted seed-0's 19.2%.)
2. **Basilisk fallback CSV** (`mc_basilisk_fallback.csv`) is the **baseline** flight GNC at 0% dock — it
   *localizes* the closed-loop gap to the shared flight GNC (not the neural policy). The ~53% figure is
   the policy after the accel-bias-EKF + integral fixes. Flight-grade is established on HCW truth +
   Basilisk open-loop; the Basilisk closed loop (~53%) is an explicit maturation item.
