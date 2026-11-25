# Effect of Colored Noise on Coupled Thermoacoustic Oscillators

This repository contains the code used to produce the results of

> R. Rai, Y. Patil, L. Kabiraj, A. Saurabh, C. Meena,
> **Effect of Colored Noise on Coupled Thermoacoustic Oscillators**,

The study considers two horizontal Rijke tubes, modelled with a Galerkin reduced-order model, interacting through time-delay and dissipative coupling. Each tube is forced by independent additive noise: white noise or power-normalized Ornstein–Uhlenbeck (OU) noise with correlation times τ<sub>c</sub>/T<sub>0</sub> = 0.1, 0.5, 1, 2 and 5. The code computes the amplitude response (amplitude death and limit-cycle oscillations) and the coherence factor (coherence resonance) of the coupled system.

## Repository structure

```
coupled_TO_colored_noise/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── noise_calibration.py            reference white-noise intensity Γ0
│   ├── fig01_bifurcation_single_tube.py
│   ├── fig03_noise_psd.py
│   ├── fig04_Prms_vs_Ktau.py
│   ├── fig05_Prms_vs_tau.py
│   ├── fig06_heatmap_tau_Ktau.py
│   ├── fig07_coherence_timedelay.py
│   ├── fig08_Prms_vs_detuning.py
│   ├── fig09_heatmap_detuning_Kd.py
│   └── fig10_coherence_dissipative.py
├── plotting scripts/
│   ├── plot_fig04.py                   one-parameter plot (template for Figs. 4, 5, 8)
│   └── plot_fig09.py                   two-parameter map (template for Figs. 6, 9)
├── data/
└── output/
```

Each script is self-contained: it includes the model, the noise generation, and the analysis needed for its figure. Scripts must be run from inside the repository, because they locate `data/` and `output/` relative to their own position.

## Installation

Python 3.12 was used. Install the dependencies with

```bash
pip install -r requirements.txt
```

## Reproducing the results

| Figure | Script | Command | Output |
|---|---|---|---|
| Γ<sub>0</sub> calibration | `scripts/noise_calibration.py` | `python scripts/noise_calibration.py` | `data/noise_calibration/gamma0.json` |
| Fig. 1 | `scripts/fig01_bifurcation_single_tube.py` | `python scripts/fig01_bifurcation_single_tube.py` | `data/fig01/`, `output/` |
| Fig. 2 | – | schematic, no code | – |
| Fig. 3 | `scripts/fig03_noise_psd.py` | `python scripts/fig03_noise_psd.py` | `data/fig03/`, `output/` |
| Fig. 4 | `scripts/fig04_Prms_vs_Ktau.py` | `python scripts/fig04_Prms_vs_Ktau.py` | `data/fig04/` |
| Fig. 5 | `scripts/fig05_Prms_vs_tau.py` | `python scripts/fig05_Prms_vs_tau.py` | `data/fig05/` |
| Fig. 6 | `scripts/fig06_heatmap_tau_Ktau.py` | `python scripts/fig06_heatmap_tau_Ktau.py` | `data/fig06/` |
| Fig. 7 | `scripts/fig07_coherence_timedelay.py` | see below | `data/fig07/` |
| Fig. 8 | `scripts/fig08_Prms_vs_detuning.py` | `python scripts/fig08_Prms_vs_detuning.py` | `data/fig08/` |
| Fig. 9 | `scripts/fig09_heatmap_detuning_Kd.py` | `python scripts/fig09_heatmap_detuning_Kd.py` | `data/fig09/` |
| Fig. 10 | `scripts/fig10_coherence_dissipative.py` | see below | `data/fig10/` |

The scripts for Figs. 1 and 3 compute the data and save the figure. The scripts for Figs. 4–10 compute and save the data only; the figures are drawn from the saved data with the scripts in `plotting scripts/` (see [Plotting](#plotting)).

### Noise intensity

All noisy simulations use the reference white-noise intensity Γ<sub>0</sub> (Sec. 2.3 of the paper). If `data/noise_calibration/gamma0.json` exists, the
scripts read Γ<sub>0</sub> from it; otherwise they use the built-in value `GAMMA0_DEFAULT`. Running `noise_calibration.py` first is therefore optional. Γ<sub>0</sub> can also be set directly with `--gamma0`.

### Figures 4, 5 and 8 (one-parameter response)

By default these scripts compute σ = 0.05, 0.1 and 0.2 and the noise-free reference, with 50 realizations each. Noise amplitudes can be run separately:

```bash
python scripts/fig04_Prms_vs_Ktau.py --sigmas 0.05
python scripts/fig04_Prms_vs_Ktau.py --sigmas 0.1 0.2 --skip-no-noise
```

### Figures 6 and 9 (two-parameter maps)

These are the most expensive computations (41 × 41 grid × 6 noise types × 10 realizations per σ). Running one noise amplitude at a time is recommended:

```bash
python scripts/fig06_heatmap_tau_Ktau.py --sigmas                      # no-noise maps only
python scripts/fig06_heatmap_tau_Ktau.py --sigmas 0.05 --skip-no-noise
python scripts/fig06_heatmap_tau_Ktau.py --sigmas 0.4 --skip-no-noise
```

For Fig. 6 the script computes two no-noise maps: with different initial conditions in the two tubes (Fig. 6b) and with identical initial conditions (Fig. 6a).

### Figures 7 and 10 (coherence factor)

Each script has two parameter sweeps:

```bash
# Fig. 7: time-delay coupling (K_tau = 0.2, tau = 0.1)
python scripts/fig07_coherence_timedelay.py --sweep sigma --K 0.47 0.51 0.54   # Fig. 7a, 7b
python scripts/fig07_coherence_timedelay.py --sweep K                         # Fig. 7c, 7d

# Fig. 10: dissipative coupling (K_d = 0.4, omega_B/omega_A = 0.95)
python scripts/fig10_coherence_dissipative.py --sweep sigma --K 0.50 0.54 0.59  # Fig. 10a, 10b
python scripts/fig10_coherence_dissipative.py --sweep K                        # Fig. 10c, 10d
```

The σ sweep gives β versus σ (panel a). Panel b uses, for each noise colour, the σ at which the mean β is largest. The K sweep gives β versus K and versus τ<sub>c</sub>/T<sub>0</sub> at σ = 0.05 (panels c and d).

### Common options

| Option | Meaning |
|---|---|
| `--t-max` | simulation time (default 5000; 30000 for Fig. 1) |
| `--n-runs` | number of noise realizations |
| `--n-jobs` | number of parallel processes (default: all cores) |
| `--gamma0` | override the reference noise intensity Γ<sub>0</sub> |
| `--grid-step` | coarser grid for Figs. 6 and 9 (testing only) |
| `--plot-only` | redraw Figs. 1 and 3 from saved data |

For a quick check that a script runs, use a short simulation and few realizations, for example

```bash
python scripts/fig04_Prms_vs_Ktau.py --t-max 500 --n-runs 2 --sigmas 0.05
python scripts/fig06_heatmap_tau_Ktau.py --t-max 300 --n-runs 2 --grid-step 8 --sigmas 0.05
```

Results from such short runs are not meaningful; they only test the code.

## Plotting

The folder `plotting scripts/` contains two example plotting scripts. They read the saved arrays from `data/` and save the figure as a PDF in `output/`. Because the folder name contains a space, put the path in quotes when running them:

```bash
python "plotting scripts/plot_fig04.py"    # Fig. 4 -> output/fig04_Prms_vs_Ktau.pdf
python "plotting scripts/plot_fig09.py"    # Fig. 9 -> output/fig09_heatmap_detuning_Kd.pdf
```

**One-parameter plot (`plot_fig04.py`).** Draws P′<sub>rms</sub> of oscillator A versus K<sub>τ</sub> for σ = 0.05, 0.1 and 0.2 (panels a–c), with standard-error bars, and P′<sub>rms</sub> at K<sub>τ</sub> = 0.1 versus τ<sub>c</sub>/T<sub>0</sub> for all three σ (panel d). It serves as a template for the other one-parameter figures: Fig. 5 (P′<sub>rms</sub> versus τ) and Fig. 8 (P′<sub>rms</sub> versus ω<sub>B</sub>/ω<sub>A</sub>) have the same layout and data format, and only the settings at the top of the script need to be changed (data folder, x-axis values and label, the value shown in panel d, and the file-name prefix).

**Two-parameter plot (`plot_fig09.py`).** Draws the P′<sub>rms</sub> maps in the (ω<sub>B</sub>/ω<sub>A</sub>, K<sub>d</sub>) plane: the no-noise map (panel a) and the maps for σ = 0.05, 0.1, 0.2, 0.4 (rows) and τ<sub>c</sub>/T<sub>0</sub> = 0, 0.1, 1, 2, 5 (columns), on a common discrete colour scale. It serves as a template for Fig. 6, the (τ, K<sub>τ</sub>) maps under time-delay coupling; the changes needed (data folder, axes, and the two no-noise panels of Fig. 6) are listed in the header of the script.

## Data files

The saved arrays follow these naming conventions:

| Part of file name | Meaning |
|---|---|
| `Pa_`, `Pb_` | oscillator A, oscillator B |
| `wn` | white noise (τ<sub>c</sub>/T<sub>0</sub> = 0) |
| `cn1` … `cn5` | OU noise with τ<sub>c</sub>/T<sub>0</sub> = 0.1, 0.5, 1, 2, 5 |
| `inten1` … `inten4` | σ = 0.05, 0.1, 0.2, 0.4 |
| `nn` | no noise |
| `mean`, no tag | mean P′<sub>rms</sub> over realizations |
| `std` | standard deviation over realizations |

The error bars in the paper are standard errors of the mean, `std / sqrt(n_runs)`. Each script's header describes its output files and array shapes in detail.

In the coherence-factor folders (`data/fig07/`, `data/fig10/`), the `*_wn_power.npy` files hold β for white noise and the `*_cn_power.npy` files hold β for the five OU noises (one column per τ<sub>c</sub>/T<sub>0</sub>). `intensity.npy` and `K_values.npy` give the swept parameter values.

