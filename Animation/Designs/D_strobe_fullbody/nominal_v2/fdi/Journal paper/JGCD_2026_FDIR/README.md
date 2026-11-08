# JGCD 2026 FDIR Journal Paper

**Deputy-Centric Fault Detection, Isolation, and Safe Learned Recovery for
SmallSat Swarms at Time-Critical Encounters**

- `main.tex` + `References.bib` + `Figures/` compile with the included
  `new-aiaa.cls`. On Overleaf, delete the two local shims
  (`newtxtext.sty`, `newtxmath.sty`); they exist only for offline TeX Live
  installs that lack the newtx fonts. `main.pdf` is the compiled paper
  (35 pages).
- All quantitative claims were regenerated for this manuscript, not copied
  from earlier documents. `campaigns/` holds the exact scripts:
  - `exp_apophis_mc.py` / `exp_run_part.py`: detection (N=200/class),
    isolation before/after (N=150, identical seeds), magnitude sweeps (N=80/pt)
    on the real 8-deputy fleet bundles (`data/mc/draw_0000/_nav_stage`).
  - `exp_formpeer.py` / `exp_fp_part.py`: causal FORM/PEER campaigns
    (local fault, common-mode absorbed, common-mode with ISL outage,
    homogeneous-fleet MAD-floor stress).
  - `exp_byz.py`: Byzantine f-of-N sweep, fleets N=4..32.
  - `exp_regimes.py`: LEO / GEO / NRHO transfer + sensor-grade sweep.
  - `prep_figdata.py`: nav-error/RMS extraction + nominal causal FDI histories.
  - `figures_src/`: every figure in `Figures/*.pdf` regenerates from these
    scripts (one shared journal style, Okabe-Ito colorblind-safe palette).
  - `results/`: raw campaign outputs (JSON/npz) the tables were typed from.
- Detection whitening note: campaign scripts floor the reported per-axis sigma
  at 1e-6 (numerical guard only). An earlier internal campaign
  (`certification.py`) used a 1e-3 floor, which under-weighted the angle axes;
  the paper's numbers use the corrected guard, which is why NAV latency (116 s)
  and the NAV detectability knee (~2 sigma) improve on the certification
  dossier values.
- RL evidence: training scalars extracted from `tb_fdir.zip` into
  `campaigns/results/tb_scalars.csv`; evaluation numbers from the three
  `fdir_brain_hpc*_metrics.json` files (base + two fine-tune ablations).
