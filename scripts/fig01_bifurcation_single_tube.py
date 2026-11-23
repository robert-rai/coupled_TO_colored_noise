"""
Figure 1: Bifurcation diagram of a single horizontal Rijke tube.


Panels
    (a) P'_rms vs heater power K (forward and backward sweeps)
    (b) acoustic pressure time series
    (c) amplitude spectrum of the acoustic pressure

"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend: save figures, never show them
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed
from numba import njit



# Parameters

T_MAX = 30000.0       # total simulation time (non-dimensional)
DT = 0.001            # integration time step
TRANSIENT_FRAC = 0.8  # P'_rms computed over the final 20% of the signal

N = 10                # number of Galerkin modes
gamma = 1.4           # specific heat ratio
Ma = 0.01             # Mach number
tf = 0.2              # thermal time lag
xf = 0.25             # heater location
C1, C2 = 0.1, 0.06    # damping coefficients

K_VALUES = np.round(np.arange(0, 1.01, 0.01), 2)  # heater power sweep
K_TIMESERIES = 0.62   # heater power for panels (b) and (c)
BASE_SEED = 12345     # base random seed for reproducible initial conditions

# Bistable region drawn in panel (a)
SADDLE_NODE_K = 0.52
HOPF_K = 0.61

# Output locations (relative to the repository root)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "fig01"
OUTPUT_DIR = REPO_ROOT / "output"



# Model

@njit
def heat_release(uf):
    """Modified King's law nonlinearity."""
    return np.sqrt(np.abs(1.0 / 3.0 + uf)) - 1.0 / np.sqrt(3.0)


def zeta_func(j):
    """Modal damping coefficient zeta_j, Eq. (6), with omega_j = j*pi."""
    return (1.0 / (2.0 * np.pi)) * (C1 * j + C2 * np.sqrt(1.0 / j))


modes = np.arange(1, N + 1)
zeta = np.array([zeta_func(j) for j in modes], dtype=np.float64)
phi = np.cos(np.pi * modes * xf)                                    # velocity mode shapes at xf
modal_coeff = gamma * Ma * np.sin(np.pi * modes * xf) / (modes * np.pi)  # pressure at xf


@njit
def rhs(eta, etadot, zeta_j, j, qdot, K):
    """Right-hand side of the modal acceleration equation, Eq. (5)."""
    return (-2 * zeta_j * j * np.pi * etadot
            - (j * np.pi) ** 2 * eta
            - np.pi * j * K * np.sin(j * np.pi * xf) * qdot)


@njit
def simulate(K, nt, dt, backward, seed):
    """
    Integrate the single Rijke tube with RK4 and return the acoustic
    pressure P'(t) at the heater location.

    backward = False : small initial conditions (forward sweep)
    backward = True  : large initial conditions (backward sweep), so the
                       system starts on the limit-cycle branch
    """
    np.random.seed(seed)

    eta = np.zeros((N, nt))
    etadot = np.zeros((N, nt))

    if backward:
        eta[:, 0] = np.random.random(N) + 1.0
        etadot[:, 0] = np.random.random(N) + 1.0
    else:
        eta[:, 0] = 1e-5 * np.random.random(N)
        etadot[:, 0] = 1e-5 * np.random.random(N)

    for i in range(1, nt):
        # Delayed modal amplitudes at t - tf (linear interpolation)
        i_delay = (i * dt - tf) / dt
        i0 = int(np.floor(i_delay))
        i1 = i0 + 1
        if i0 >= 0 and i1 < i:
            w = i_delay - i0
            eta_delay = (1 - w) * eta[:, i0] + w * eta[:, i1]
        else:
            eta_delay = eta[:, 0].copy()

        uf = np.sum(eta_delay * phi)
        qdot = heat_release(uf)

        # RK4 step for each Galerkin mode
        for j in range(N):
            e0 = eta[j, i - 1]
            v0 = etadot[j, i - 1]

            k1_e = v0
            k1_v = rhs(e0, v0, zeta[j], j + 1, qdot, K)

            k2_e = v0 + 0.5 * dt * k1_v
            k2_v = rhs(e0 + 0.5 * dt * k1_e, v0 + 0.5 * dt * k1_v, zeta[j], j + 1, qdot, K)

            k3_e = v0 + 0.5 * dt * k2_v
            k3_v = rhs(e0 + 0.5 * dt * k2_e, v0 + 0.5 * dt * k2_v, zeta[j], j + 1, qdot, K)

            k4_e = v0 + dt * k3_v
            k4_v = rhs(e0 + dt * k3_e, v0 + dt * k3_v, zeta[j], j + 1, qdot, K)

            eta[j, i] = e0 + dt * (k1_e + 2 * k2_e + 2 * k3_e + k4_e) / 6
            etadot[j, i] = v0 + dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

    p = np.zeros(nt)
    for j in range(N):
        p -= modal_coeff[j] * etadot[j, :]
    return p


def p_rms(p):
    """RMS pressure over the final (1 - TRANSIENT_FRAC) of the signal."""
    return np.sqrt(np.mean(p[int(TRANSIENT_FRAC * len(p)):] ** 2))



# Computation

def compute_sweep(nt, backward, n_jobs):
    """P'_rms for every K in K_VALUES, forward or backward."""
    offset = 10_000 if backward else 0

    def one(idx, K):
        return p_rms(simulate(K, nt, DT, backward, BASE_SEED + offset + idx))

    rms = Parallel(n_jobs=n_jobs)(
        delayed(one)(idx, K) for idx, K in enumerate(K_VALUES)
    )
    return np.array(rms)


def compute_timeseries_and_spectrum(nt):
    """Time series window and amplitude spectrum for panels (b) and (c)."""
    p = simulate(K_TIMESERIES, nt, DT, True, BASE_SEED + 20_000)
    t = np.arange(nt) * DT

    # Keep only the window plotted in panel (b) to keep saved data small
    window = (t >= 150) & (t <= 170)

    fft_p = np.fft.fft(p)
    freqs = np.fft.fftfreq(nt, DT)
    omega = 2 * np.pi * freqs[: nt // 2]
    amplitude = 2 * np.abs(fft_p[: nt // 2]) / nt
    dominant_omega = omega[np.argmax(amplitude)]

    # Keep only the range plotted in panel (c)
    band = omega <= 2 * np.pi
    return t[window], p[window], omega[band], amplitude[band], dominant_omega



# Plotting

def latex_available():
    """
    Use LaTeX text rendering (as in the paper) if a working LaTeX installation
    is found; otherwise fall back to matplotlib's built-in mathtext.
    """
    if shutil.which("latex") is None:
        print("LaTeX not found: using matplotlib mathtext instead.")
        return False
    try:
        with plt.rc_context({"text.usetex": True}):
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$P'_{rms}$")
            fig.canvas.draw()
            plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        print("LaTeX rendering failed: using matplotlib mathtext instead.")
        return False


def boxed_label(ax, x, y, text, color="#ffdddd"):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=25, color="black",
            bbox=dict(facecolor=color, edgecolor="black",
                      boxstyle="round,pad=0.4", alpha=0.8))


def make_figure(data, outfile):
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 3, figure=fig)

    # (a) Bifurcation diagram
    ax1 = fig.add_subplot(gs[:, :2])
    ax1.scatter(data["K_values"], data["rms_forward"], marker=".",
                label=r"$forward$", s=110, clip_on=False)
    ax1.scatter(data["K_values"], data["rms_backward"], marker="+",
                label=r"$backward$", s=90, clip_on=False)
    ax1.set_xlabel(r"$K$", fontsize=34)
    ax1.set_ylabel(r"$P'_{rms}$", fontsize=34)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, None)
    ax1.set_yticks([0.001, 0.003, 0.005])
    ax1.tick_params(axis="both", labelsize=30)

    ax1.axvspan(SADDLE_NODE_K, HOPF_K, alpha=0.25, color="gray", zorder=0)
    ax1.axvline(x=SADDLE_NODE_K, color="gray", linewidth=2)
    ax1.axvline(x=HOPF_K, color="gray", linewidth=2)
    ax1.legend(fontsize=22)
    ax1.text(0.02, 0.95, r"$(a)$", transform=ax1.transAxes, fontsize=25,
             verticalalignment="top")

    boxed_label(ax1, 0.13, 0.4, r"$Subthreshold$")
    boxed_label(ax1, 0.48, 0.4, r"$Bistable$")
    boxed_label(ax1, 0.72, 0.4, r"$Limit\ Cycle$")

    arrow = dict(arrowstyle="-|>", color="black", linewidth=2, shrinkA=5, shrinkB=5)
    box = dict(facecolor="lightblue", edgecolor="black",
               boxstyle="round,pad=0.4", alpha=0.8)
    ax1.annotate(r"$Saddle\ Node\ Point$", xy=(0.52, 0.0034), xycoords="data",
                 xytext=(0.11, 0.7), textcoords=ax1.transAxes,
                 fontsize=25, bbox=box, arrowprops=arrow)
    ax1.annotate(r"$Hopf\ Point$", xy=(0.62, 0.00001), xycoords="data",
                 xytext=(0.7, 0.1), textcoords=ax1.transAxes,
                 fontsize=25, bbox=box, arrowprops=arrow)

    # (b) Time series
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(data["t_window"], data["p_window"], "k")
    ax2.set_xlim(150, 170)
    ax2.set_ylim(-0.01, 0.01)
    ax2.set_xticks([150, 170])
    ax2.set_yticks([-0.01, 0, 0.01])
    ax2.tick_params(axis="both", labelsize=30)
    ax2.set_xlabel(r"$t$", fontsize=34)
    ax2.set_ylabel(r"$P'$", fontsize=34)
    ax2.text(0.05, 0.96, r"$(b)$", transform=ax2.transAxes, fontsize=25,
             verticalalignment="top")

    # (c) Amplitude spectrum
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(data["omega"], data["amplitude"], "k")
    ax3.set_xlim(0, 2 * np.pi)
    ax3.set_xticks([0, np.pi, 2 * np.pi])
    ax3.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax3.set_yticks([0, 0.003, 0.006])
    ax3.set_yticklabels([r"$0$", r"$0.003$", r"$0.006$"])
    ax3.tick_params(axis="both", labelsize=30)
    ax3.set_xlabel(r"$\omega$", fontsize=34)
    ax3.set_ylabel(r"$Amplitude$", fontsize=20, labelpad=10)
    ax3.text(0.05, 0.95, r"$(c)$", transform=ax3.transAxes, fontsize=25,
             verticalalignment="top")

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)



# Main

def main():
    parser = argparse.ArgumentParser(description="Reproduce Fig. 1 (single Rijke tube).")
    parser.add_argument("--plot-only", action="store_true",
                        help="skip simulations and plot from data/fig01/fig01_data.npz")
    parser.add_argument("--t-max", type=float, default=T_MAX,
                        help=f"simulation time (default {T_MAX}); use a small value for a quick test")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="number of parallel jobs (default: all cores)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "fig01_data.npz"

    plt.rcParams["text.usetex"] = latex_available()

    if args.plot_only:
        data = dict(np.load(data_file))
        print(f"Loaded data from {data_file}")
    else:
        nt = int(args.t_max / DT)
        print(f"Number of time steps: {nt}")

        print("Forward sweep ...")
        rms_forward = compute_sweep(nt, backward=False, n_jobs=args.n_jobs)
        print("Backward sweep ...")
        rms_backward = compute_sweep(nt, backward=True, n_jobs=args.n_jobs)
        print(f"Time series and spectrum at K = {K_TIMESERIES} ...")
        t_w, p_w, omega, amp, omega0 = compute_timeseries_and_spectrum(nt)
        print(f"Dominant frequency omega_0 = {omega0:.4f}")

        data = dict(K_values=K_VALUES, rms_forward=rms_forward,
                    rms_backward=rms_backward, t_window=t_w, p_window=p_w,
                    omega=omega, amplitude=amp, dominant_omega=omega0)
        np.savez_compressed(data_file, **data)
        print(f"Saved data to {data_file}")

    outfile = OUTPUT_DIR / "fig01_bifurcation_single_tube.pdf"
    make_figure(data, outfile)
    print(f"Saved figure to {outfile}")


if __name__ == "__main__":
    main()
