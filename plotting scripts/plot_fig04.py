"""
Plot Figure 4: P'_rms vs time-delay coupling strength K_tau.

Reads the data saved by scripts/fig04_Prms_vs_Ktau.py from data/fig04/ and
saves the figure to output/fig04.pdf.

Panels
    (a)-(c) P'_rms vs K_tau for sigma = 0.05, 0.1, 0.2, for no noise, white
            noise and OU noise (tau_c/T0 = 0.1, 0.5, 1, 2, 5). Error bars are
            the standard error of the mean = std / sqrt(n_runs).
    (d)     P'_rms at K_tau = 0.1 vs tau_c/T0 for the three noise amplitudes.

Usage
    python plotting/plot_fig04.py

This script is a demonstration of how the one-parameter response plots in the
paper are made. Figs. 5 and 8 have the same layout and file format, so the
same code can be used after changing the settings in the "Settings" block:

    Fig. 5 (P'_rms vs delay time tau), data from scripts/fig05_Prms_vs_tau.py
        DATA_DIR    = REPO_ROOT / "data" / "fig05"
        X_VALUES    = np.round(np.arange(0, 1.02, 0.02), 2)
        X_LABEL     = r"$\\tau$", with ticks at 0, 0.5, 1
        X_SELECTED  = 0.5   (panel d), title r"$\\tau = 0.5$"
        file names are the same as for Fig. 4 (Pa_delay_Kt_...)

    Fig. 8 (P'_rms vs detuning omega_B/omega_A), data from scripts/fig08_Prms_vs_detuning.py
        DATA_DIR    = REPO_ROOT / "data" / "fig08"
        X_VALUES    = np.round(np.arange(0.8, 1.21, 0.01), 2)
        X_LABEL     = r"$\\omega_B/\\omega_A$", with ticks at 0.8, 1, 1.2
        X_SELECTED  = 1.1   (panel d), title r"$\\omega_B/\\omega_A = 1.1$"
        PREFIX      = "Pa_delay_Kd"
        the no-noise files exist only with the inten1 tag, so load
        Pa_delay_Kd_nn_inten1.npy for all three panels

Axis limits and the positions of the sigma labels in panel (d) may also need
small adjustments to fit the data of each figure.
"""

import shutil
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # save the figure only, never show it
import matplotlib.pyplot as plt


# Settings

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "fig04"
OUTPUT_DIR = REPO_ROOT / "output"
OUTFILE = OUTPUT_DIR / "fig04.pdf"

X_VALUES = np.round(np.arange(0, 0.21, 0.01), 2)   # K_tau values of the data
X_SELECTED = 0.1                                    # K_tau shown in panel (d)
PREFIX = "Pa_delay_Kt"                              # file-name prefix (oscillator A)
N_RUNS = 50                                         # realizations, for the standard error

labels = [r"$No\ Noise$", r"$\tau_c / T_0 = 0$", r"$\tau_c / T_0 = 0.1$",
          r"$\tau_c / T_0 = 0.5$", r"$\tau_c / T_0 = 1$", r"$\tau_c / T_0 = 2$",
          r"$\tau_c / T_0 = 5$"]
markers = ["o", "s", "^", "D", "v", "P", "X"]
noise_names = ["nn", "wn", "cn1", "cn2", "cn3", "cn4", "cn5"]

suffixes = ["_inten1.npy", "_inten2.npy", "_inten3.npy"]   # sigma = 0.05, 0.1, 0.2
panel_titles = [r"$\sigma = 0.05$", r"$\sigma = 0.1$", r"$\sigma = 0.2$"]
panel_markers = [r"$(a)$", r"$(b)$", r"$(c)$"]


def latex_available():
    """Use LaTeX text rendering (as in the paper) if it works, otherwise mathtext."""
    if shutil.which("latex") is None:
        print("LaTeX not found: using matplotlib mathtext instead.")
        return False
    try:
        with plt.rc_context({"text.usetex": True}):
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$K_{\tau}$")
            fig.canvas.draw()
            plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        print("LaTeX rendering failed: using matplotlib mathtext instead.")
        return False


def load(name, suffix, std=False):
    """Load one saved array, e.g. Pa_delay_Kt_wn_inten1.npy or Pa_delay_Kt_std_wn_inten1.npy."""
    middle = "_std_" if std else "_"
    return np.load(DATA_DIR / f"{PREFIX}{middle}{name}{suffix}")


# Figure

def main():
    plt.rcParams["text.usetex"] = latex_available()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(1, 4, 3, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(1, 4, 4)
    axes = [ax1, ax2, ax3]

    # Panels (a)-(c): P'_rms vs K_tau for each noise amplitude
    for ax, suffix, title, marker_label in zip(axes, suffixes, panel_titles, panel_markers):
        for name, label, marker in zip(noise_names, labels, markers):
            data = load(name, suffix)
            sem = load(name, suffix, std=True) / np.sqrt(N_RUNS)
            line, = ax.plot(X_VALUES, data, marker=marker, linestyle="-", linewidth=1.5,
                            markersize=6, label=label, clip_on=False)
            ax.errorbar(X_VALUES, data, yerr=sem, fmt="none", ecolor=line.get_color(),
                        capsize=4, elinewidth=2)
        ax.set_title(title, fontsize=24, pad=15)
        ax.text(0.87, 0.92, marker_label, transform=ax.transAxes, fontsize=24,
                fontweight="bold", verticalalignment="top")
        ax.set_xlim(0, 0.2)
        ax.set_ylim(0, 0.005)
        ax.set_xticks([0, 0.1, 0.2], [r"$0$", r"$0.1$", r"$0.2$"])
        ax.set_yticks([0, 0.002, 0.004], [r"$0$", r"$2$", r"$4$"])
        ax.tick_params(axis="both", which="major", labelsize=26, length=8, width=3)
        ax.set_xlabel(r"$K_{\tau}$", fontsize=26)

    ax1.set_ylabel(r"$P_{rms}^{\prime} (\times 10^{-3})$", fontsize=26, labelpad=14)
    ax2.tick_params(axis="y", labelleft=False)
    ax3.tick_params(axis="y", labelleft=False)

    # Panel (d): P'_rms at the selected K_tau vs noise colour, for all three sigma
    k_idx = np.argmin(np.abs(X_VALUES - X_SELECTED))
    noisy = noise_names[1:]                       # wn, cn1 ... cn5 (no-noise case excluded)
    tc_labels = [r"$0$", r"$0.1$", r"$0.5$", r"$1$", r"$2$", r"$5$"]
    x = np.arange(len(tc_labels))

    for suffix, marker, sigma_label in zip(suffixes, ["o", "s", "^"],
                                           [r"$\sigma=0.05$", r"$\sigma=0.1$", r"$\sigma=0.2$"]):
        values = np.array([load(name, suffix)[k_idx] for name in noisy])
        ax4.plot(x, values, marker=marker, linestyle="-", linewidth=1.5, markersize=7,
                 clip_on=False, label=sigma_label)

    ax4.set_title(rf"$K_\tau = {X_VALUES[k_idx]:.1f}$", fontsize=24, pad=15)
    ax4.text(0.87, 0.92, r"$(d)$", transform=ax4.transAxes, fontsize=24,
             fontweight="bold", verticalalignment="top")
    ax4.set_xlabel(r"$\tau_c/T_0$", fontsize=26)
    ax4.set_xticks(x, tc_labels)
    ax4.set_yticks([0, 0.002], [r"$0$", r"$2$"])
    ax4.tick_params(axis="both", which="major", labelsize=26, length=8, width=3)
    ax4.set_xlim(-0.3, 5.3)
    ax4.set_ylim(-0.0001, 0.002)

    ax4.text(0.6, 0.22, r"$\sigma=0.05$", transform=ax4.transAxes, fontsize=22)
    ax4.text(0.6, 0.39, r"$\sigma=0.1$", transform=ax4.transAxes, fontsize=22)
    ax4.text(0.6, 0.72, r"$\sigma=0.2$", transform=ax4.transAxes, fontsize=22)

    # Shared legend above the panels
    handles, legend_labels = ax1.get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=22, frameon=True, loc="upper center",
               bbox_to_anchor=(0.51, 1.08), ncol=7, columnspacing=1.2, handletextpad=0.4)

    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.20, top=0.80, wspace=0.15)
    fig.savefig(OUTFILE, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUTFILE}")


if __name__ == "__main__":
    main()
