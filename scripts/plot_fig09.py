"""
Plot Figure 9: P'_rms of oscillator A in the (omega_B/omega_A, K_d) plane
under dissipative coupling.

Reads the data saved by scripts/fig09_heatmap_detuning_Kd.py from data/fig09/
and saves the figure to output/fig09.pdf.

Layout (5 x 5 grid)
    row 1   (a)      no noise (sigma = 0), green dot Y marks K_d = 0.4,
                     omega_B/omega_A = 0.95 (used for Fig. 10); colour bar
    rows 2-5 (b)-(u) sigma = 0.05, 0.1, 0.2, 0.4 (rows) and
                     tau_c/T0 = 0, 0.1, 1, 2, 5 (columns)

All panels share one discrete colour scale (5 levels from 0 to the largest
P'_rms over all panels).


This script is a demonstration of how the two-parameter maps in the paper are
made. Fig. 6 (time-delay coupling) uses the same data format and can be
plotted with the same code after changing the "Settings" block:

    DATA_DIR    = REPO_ROOT / "data" / "fig06"
    X_VALUES    = np.round(np.arange(0, 2.05, 0.05), 2)     # tau
    Y_VALUES    = np.round(np.arange(0, 0.81, 0.02), 2)     # K_tau
    axis labels r"$\\tau$" and r"$K_\\tau$", x ticks at 0, 1, 2
    marker      green dot X at tau = 0.1, K_tau = 0.2

    Fig. 6 has two no-noise panels in the first row:
        (a) identical initial conditions:  Pa_delay_2D_mean_nn_inten1_in_phase.npy
        (b) different initial conditions:  Pa_delay_2D_mean_nn_inten1_out_phase.npy
    so the first row draws two maps instead of one, the marker goes in panel (b),
    and the panel letters of the noisy maps start at (c).
"""

import shutil
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # save the figure only, never show it
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.transforms import ScaledTranslation


# Settings

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "fig09"
OUTPUT_DIR = REPO_ROOT / "output"
OUTFILE = OUTPUT_DIR / "fig09.pdf"

Y_VALUES = np.round(np.arange(0, 0.81, 0.02), 2)     # K_d
X_VALUES = np.round(np.arange(0.8, 1.21, 0.01), 2)   # omega_B / omega_A

NO_NOISE_FILE = "Pa_dissi_2D_mean_nn_inten1.npy"
PREFIX = "Pa_2D_mean"            # noisy maps: Pa_2D_mean_{noise}_inten{i}.npy

# Columns: noise types shown and their tau_c/T0 titles.
# File names: wn = 0, cn1 = 0.1, cn2 = 0.5, cn3 = 1, cn4 = 2, cn5 = 5
noise_names = ["wn", "cn1", "cn3", "cn4", "cn5"]
titles = [r"$\tau_c / T_0 = 0$", r"$\tau_c / T_0 = 0.1$", r"$\tau_c / T_0 = 1$",
          r"$\tau_c / T_0 = 2$", r"$\tau_c / T_0 = 5$"]

# Rows: noise amplitudes
suffixes = ["_inten1.npy", "_inten2.npy", "_inten3.npy", "_inten4.npy"]
sigma_labels = [r"$\sigma = 0$", r"$\sigma = 0.05$", r"$\sigma = 0.1$",
                r"$\sigma = 0.2$", r"$\sigma = 0.4$"]

MARKER_POINT = (0.95, 0.4)       # (omega_B/omega_A, K_d) used for Fig. 10
MARKER_TEXT = "Y"
N_LEVELS = 5                     # discrete colour levels


def latex_available():
    """Use LaTeX text rendering (as in the paper) if it works, otherwise mathtext."""
    if shutil.which("latex") is None:
        print("LaTeX not found: using matplotlib mathtext instead.")
        return False
    try:
        with plt.rc_context({"text.usetex": True}):
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$K_d$")
            fig.canvas.draw()
            plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        print("LaTeX rendering failed: using matplotlib mathtext instead.")
        return False


def show_map(ax, arr, cmap, norm):
    return ax.imshow(arr, origin="lower", aspect="auto",
                     extent=[X_VALUES[0], X_VALUES[-1], Y_VALUES[0], Y_VALUES[-1]],
                     cmap=cmap, norm=norm)


def panel_letter(ax, text):
    ax.text(0.45, 0.95, text, transform=ax.transAxes, fontsize=30,
            fontweight="bold", family="sans-serif", verticalalignment="top")


def shift_tick_labels(fig, labels, shifts, axis):
    """Move individual tick labels by a number of points (to avoid overlaps)."""
    for label, shift in zip(labels, shifts):
        dx, dy = (shift / 72, 0) if axis == "x" else (0, shift / 72)
        label.set_transform(label.get_transform()
                            + ScaledTranslation(dx, dy, fig.dpi_scale_trans))


# Figure

def main():
    plt.rcParams["text.usetex"] = latex_available()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all maps
    no_noise = np.load(DATA_DIR / NO_NOISE_FILE)
    noisy = [[np.load(DATA_DIR / f"{PREFIX}_{name}{suffix}") for name in noise_names]
             for suffix in suffixes]

    # Common discrete colour scale over all panels
    Pmax = max(no_noise.max(), max(arr.max() for row in noisy for arr in row))
    bounds = np.linspace(0, Pmax, N_LEVELS + 1)
    cmap = plt.get_cmap("inferno", N_LEVELS)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(5, 5, figsize=(28, 20), sharex=True, sharey=True)

    # Row 1: (a) no noise, with the marker for Fig. 10; the other cells stay empty
    ax = axes[0, 0]
    im = show_map(ax, no_noise, cmap, norm)
    ax.plot(*MARKER_POINT, marker="o", markersize=15, markerfacecolor="green",
            markeredgecolor="green", markeredgewidth=2)
    ax.text(0.4, 0.55, MARKER_TEXT, transform=ax.transAxes, fontsize=38,
            fontweight="bold", verticalalignment="top")
    panel_letter(ax, "(a)")
    for j in range(1, 5):
        axes[0, j].axis("off")

    # Rows 2-5: (b)-(u), one row per sigma, one column per noise colour
    letters = iter("bcdefghijklmnopqrstu")
    for row, row_maps in enumerate(noisy):
        for col, arr in enumerate(row_maps):
            ax = axes[row + 1, col]
            im = show_map(ax, arr, cmap, norm)
            if row == 0:
                ax.set_title(titles[col], fontsize=30, pad=12)
            panel_letter(ax, f"({next(letters)})")

    # Ticks and axis labels
    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xticks([0.8, 1.0, 1.2], [r"$0.8$", r"$1$", r"$1.2$"])
            ax.set_yticks([0, 0.4, 0.8], [r"$0$", r"$0.4$", r"$0.8$"])
            ax.tick_params(axis="both", labelsize=32, length=6, width=1.5, direction="out")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\omega_B / \omega_A$", fontsize=32)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$K_d$", fontsize=32)

    # Row labels (noise amplitude)
    for y, label in zip([0.86, 0.67, 0.49, 0.31, 0.12], sigma_labels):
        fig.text(0.01, y, label, rotation=90, fontsize=32, va="center")

    # Colour bar in the empty part of the first row
    cbar_ax = fig.add_axes([0.50, 0.88, 0.43, 0.028])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
                        boundaries=bounds, ticks=bounds)
    cbar.set_ticklabels([rf"${b * 1e3:.1f}$".replace(".0$", "$") for b in bounds])
    cbar.set_label(r"$P_{rms}^{\prime}(\times10^{-3})$", fontsize=32)
    cbar.ax.tick_params(labelsize=32, length=8, width=1.3)

    plt.subplots_adjust(top=0.95, bottom=0.07, left=0.07, right=0.96,
                        hspace=0.08, wspace=0.08)

    # Larger gap between the first row and the rest
    extra_gap = 0.025
    for row in range(1, 5):
        for col in range(5):
            pos = axes[row, col].get_position()
            axes[row, col].set_position([pos.x0, pos.y0 - extra_gap, pos.width, pos.height])

    # Move outer tick labels inwards so neighbouring panels do not overlap
    for ax in axes[-1, :]:
        labels = ax.get_xticklabels()
        if len(labels) == 3:
            shift_tick_labels(fig, labels, [12, 0, -12], axis="x")
    for ax in axes[:, 0]:
        labels = ax.get_yticklabels()
        if len(labels) == 3:
            shift_tick_labels(fig, labels, [9, 0, -9], axis="y")

    fig.savefig(OUTFILE, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUTFILE}")


if __name__ == "__main__":
    main()
