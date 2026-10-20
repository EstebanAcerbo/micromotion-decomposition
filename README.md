# Decomposition of micromotion signals into physically interpretable perturbation modes via lattice-based simulations

**E. Acerbo, M. I. Bellotti, F. J. Bonetto**  
Centro Atómico Bariloche & Instituto Balseiro, CNEA-CONICET, Argentina

Companion data and code repository for the manuscript submitted to the *Journal of the Royal Society Interface* (rsif-2026-0404).

---

## Overview

This repository contains the experimental data, simulation code, and analysis scripts supporting the above manuscript. The code implements a finite-domain (FD) impedance solver for Electric Cell-substrate Impedance Spectroscopy (ECIS) electrodes, builds perturbation libraries representing distinct physical micromotion mechanisms, and reconstructs experimental impedance fluctuations as sequences of discrete perturbation modes.

The simulation engine was developed for a specific laboratory setup (ECIS, 350 µm circular gold electrodes) and is not intended as a general-purpose tool. Precomputed results are provided in `results/` so that figures and tables can be reproduced without rerunning the full simulation pipeline.

Enquiries about code logic and operation: esteban.acerbo@ib.edu.ar

---

## Requirements

- Python 3.8+
- numpy, scipy, matplotlib
- Intel MKL via the pardiso sparse solver (for the FD engine; not required to inspect precomputed results)

All internal paths are relative. Scripts should be run from the repository root or from within their own directory as indicated in each script's docstring.

---

## Repository Structure

### `simulation_engine/`

Core simulation engine and library generation scripts.

| File | Description |
|------|-------------|
| `engine.py` | Shared simulation kernel. Giaever–Keese FD formulation on a square RecCell geometry (n = 20 pixels). MDCKII parameters: α = 19.4, Rb = 14.4, Cm = 4.95 × 10⁻⁶ F, CPE A = 1.396 × 10⁻⁸, N = 0.928. |
| `shift_library.py` | Library A: lateral displacement sweep (861 modes, lower-triangle dx ∈ [0, 40] px). |
| `shift2_library.py` | Library C: alternating-row brick-wall shear (21 modes, δ = 0–0.5). |
| `library_B_rb.py` | Library B: L-shaped corner junction extension with Rb scaling (126 modes). |
| `library_D_rb.py` | Library D: three-parameter (α, Rb, Cm) percentage perturbation (up to 1728 modes). |
| `run_libraries.py` | Batch launcher for all library builders. |
| `baseline.py` | Computes the reference impedance Z₀ at 924 Hz. |
| `culture_construction_vectorized.py`, `SolveMatrix.py`, `Calc_Curr.py`, `CPE_GK_Funs.py`, `ConstantsFile.py` | FD simulation kernel: electrode geometry construction, sparse linear system (pardiso), current integration, CPE model, physical constants. |
| `gk_sensitivity.py` | GK parameter sensitivity analysis. |
| `check_domain_boundary.py`, `check_shift_boundary.py`, `compare_libB.py`, `lib_verify.py` | Diagnostic and verification scripts (not required to reproduce paper results). |

### `experimental_signal_analysis/`

Raw experimental data files and preprocessing pipeline.

| File | Description |
|------|-------------|
| `eb04_sc.dat` | Seeding (MDCKII cells attaching to electrode, 1 kHz) |
| `eb06_sc.dat` | Confluent monolayer (500 Hz) |
| `eb11_sc.dat` | Confluent monolayer (primary analysis electrode, 1 kHz) |
| `eb13_sc.dat` | Wound healing (scratch assay, 1 kHz) |
| `eb15_sc.dat` | Post-healing (one day after confluence restored, 1 kHz) |
| `eb18_sc.dat`, `eb19_sc.dat`, `eb20_sc.dat` | Bare electrode controls (no cells) |
| `ConfluentCH3Data.dat` | Multi-frequency spectrum of a confluent MDCKII monolayer |
| `FDeb*_sc.npz` | Precomputed fractal dimension data from prior analysis |
| `load_experimental.py` | Parser for the .dat format; calibrated Z(t) extraction; group-adaptive detrending; Welch PSD. |
| `phase1_analysis.py` | Full Phase 1 characterisation: detrending, window sensitivity, PSD, two-timescale decomposition. |
| `fractal_dim.py` | Fractal dimension module: Higuchi, MR path length, MR box count, R/S methods with windowed uncertainty. |

### `reconstruction/`

Reconstruction algorithms that decompose each experimental time step into perturbation modes.

| File | Description |
|------|-------------|
| `reconstruct_greedy.py` | Greedy nearest-prototype decomposition (primary method in the manuscript). Selects one mode per library in order B → A → D → C to minimise the residual dZ. |
| `reconstruct_viterbi.py` | Viterbi-path reconstruction with temporal continuity regularisation via dynamic programming. |

### `plot_libraries/`

| File | Description |
|------|-------------|
| `plot_libraries.py` | dR/dX scatter plots and mode distribution histograms for all libraries. |
| `phase_space_coverage.py` | Experimental dZ range overlaid against combined library coverage. |

### `plot_reconstruction/`

| File | Description |
|------|-------------|
| `plot_reconstruction.py` | Time series of selected modes and reconstruction residuals. |
| `plot_mode_trajectories.py` | Mode trajectory in library phase space over time. |
| `plotrec.py`, `report_figures_eb11.py`, `result_eb11_plot.py` | Additional reconstruction visualisation scripts. |

### `latex_table_generator/`

| File | Description |
|------|-------------|
| `generate_fractal_table.py` | Runs fractal analysis across all electrodes × methods, saves `results/fractal_results.npz`, outputs LaTeX booktabs table. |

### `Report/`

Manuscript source and figure generation scripts.

| File | Description |
|------|-------------|
| `manuscript.tex` | Manuscript LaTeX source (as submitted). |
| `manuscript.pdf` | Compiled manuscript. |
| `bibliography.bib` | Bibliography. |
| `images/figure1–9.png` | Manuscript figures as submitted. |
| `fractal_plots.py` | Fractal dimension figures: log-log curves, per-window fits, reduced summary. |
| `libraries_plots.py` | Library characterisation figures: perturbation spectra, phase-space coverage, baseline comparison. |
| `baseline_tabla.py` | Baseline impedance comparison table and figure. |
| `fig_linear_vs_ma.py` | Linear vs. moving-average detrending comparison figure. |
| `fig_window_comparison.py` | Window size comparison figure. |
| `fig_window_sensitivity.py` | Window sensitivity figure. |
| `fig_libD_edges.py` | Library D edge-case visualisation. |

### `results/`

Precomputed numerical outputs. These files allow figure and table reproduction without running the simulation pipeline.

| File | Description |
|------|-------------|
| `library_A_shift.npz` | 861 lateral displacement modes |
| `library_B_rb.npz` | 126 L-corner Rb extension modes |
| `library_C_brickwall.npz` | 21 alternating-row shear modes |
| `library_D_rb.npz` | 216 three-parameter perturbation modes |
| `phase1_experimental.npz` | Detrended R(t), X(t), PSD, two-timescale decomposition for all electrodes |
| `reconstruct_v3_eb*.npz` | Greedy reconstruction outputs |
| `reconstruct_v4_eb*.npz` | Viterbi reconstruction outputs |
| `fractal_results.npz` | Fractal dimension estimates (all electrodes, methods, signals) |
| `baseline.npz` | Simulated Z₀ spectrum |
| `gk_sensitivity_data.npz` | GK parameter sensitivity analysis data |

---

## Reproducing the analysis

```bash
# Phase 1 — Experimental characterisation
python experimental_signal_analysis/phase1_analysis.py

# Fractal dimension table (Table 1 in manuscript)
python latex_table_generator/generate_fractal_table.py

# Reconstruction (per electrode)
python reconstruction/reconstruct_greedy.py --stem eb11 --savefig
python reconstruction/reconstruct_greedy.py --stem eb04 --savefig
python reconstruction/reconstruct_greedy.py --stem eb13 --savefig
python reconstruction/reconstruct_greedy.py --stem eb15 --savefig

# Figures
python Report/fractal_plots.py
python Report/libraries_plots.py
python Report/baseline_tabla.py
```

Rebuilding the perturbation libraries from scratch requires Intel MKL pardiso and significant compute time (Library D: ~2 days on a single workstation). The precomputed `.npz` files in `results/` are provided for this reason.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this code or data, please cite the accompanying manuscript:

> E. Acerbo, M. I. Bellotti, F. J. Bonetto, "Decomposition of micromotion signals into physically interpretable perturbation modes via lattice-based simulations," *Journal of the Royal Society Interface* (2026).
