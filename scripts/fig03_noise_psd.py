"""
Figure 3: Power spectral density of white noise and power-normalized
Ornstein-Uhlenbeck (OU) noise for several correlation times.


Noise model
    Scaled white noise:  eps_sigma(t) = sigma * eps_0(t),
                         <eps_sigma(t) eps_sigma(t')> = sigma^2 Gamma_0 delta(t - t')
    OU noise (Eq. 11):   d(xi)/dt = -xi / tau_c + sqrt(D) / tau_c * eps_sigma(t)
    Normalization (Eq. 17), so that white and OU noise carry the same power
    in the band [w1, w2] around the natural frequency w0 (Eq. 16):
        D(tau_c) = tau_c (w2 - w1) / (atan(w2 tau_c) - atan(w1 tau_c))

    Euler-Maruyama discretization (Appendix A):
        dW_k      = sqrt(sigma^2 Gamma_0 dt) * N(0, 1)
        xi_{k+1}  = xi_k - (dt / tau_c) xi_k + (sqrt(D) / tau_c) dW_k

The script also checks numerically that every noise has the same power in
the band, which is the purpose of the normalization.

Output
    data/fig03/fig03_data.npz
    output/fig03_noise_psd.pdf

"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numba import njit
from scipy.signal import welch



# Parameters

T_MAX = 5000.0              # length of the generated noise signal
DT = 0.001                  # time step
SIGMA = 0.05                # noise amplitude shown in Fig. 3
GAMMA0_DEFAULT = 5.761620e-01  # reference white-noise intensity (noise_calibration.py)

W0 = 3.2546899891190257     # natural (LCO) frequency at K = 0.8, from Fig. 1
T0 = 2 * np.pi / W0         # acoustic time period
TAU_RATIOS = np.array([0.1, 0.5, 1, 2, 5])  # tau_c / T0 (white noise is tau_c = 0)
BAND_RATIO = 1.2            # bandwidth  delta_w / w0
W1 = W0 - BAND_RATIO * W0 / 2
W2 = W0 + BAND_RATIO * W0 / 2

NPERSEG = 900000            # Welch segment length
SEED = 7                    # random seed

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "fig03"
OUTPUT_DIR = REPO_ROOT / "output"
GAMMA0_FILE = REPO_ROOT / "data" / "noise_calibration" / "gamma0.json"



# Noise generation

def normalization_D(tau_c, w1=W1, w2=W2):
    """Normalization coefficient D(tau_c)"""
    return tau_c * (w2 - w1) / (np.arctan(w2 * tau_c) - np.arctan(w1 * tau_c))


@njit
def ou_euler_maruyama(dW, D, tau_c, dt):
    """OU process xi(t) driven by the white-noise increments dW """
    n = dW.shape[0]
    xi = np.zeros(n)
    a = dt / tau_c
    b = np.sqrt(D) / tau_c
    for k in range(n - 1):
        xi[k + 1] = xi[k] - a * xi[k] + b * dW[k]
    return xi


def generate_noises(Gamma0, sigma, nt, seed):
    """
    Return the white noise eps_sigma(t) = dW/dt and the OU noises xi(t)
    for every tau_c in TAU_RATIOS * T0, all driven by the same increments.
    """
    rng = np.random.default_rng(seed)
    dW = np.sqrt(sigma**2 * Gamma0 * DT) * rng.standard_normal(nt)
    white = dW / DT
    colored = np.empty((len(TAU_RATIOS), nt))
    for i, ratio in enumerate(TAU_RATIOS):
        tau_c = ratio * T0
        colored[i] = ou_euler_maruyama(dW, normalization_D(tau_c), tau_c, DT)
    return white, colored


def psd_omega(x, nperseg):
    """One-sided Welch PSD as a function of angular frequency."""
    f, P = welch(x, fs=1 / DT, nperseg=nperseg, scaling="density", detrend="constant")
    return 2 * np.pi * f, P / (2 * np.pi)


def band_power(omega, S):
    """Power of a signal within [W1, W2] from its PSD."""
    band = (omega >= W1) & (omega <= W2)
    return np.trapezoid(S[band], omega[band])



# Plotting

def latex_available():
    """Use LaTeX rendering if it works, otherwise matplotlib mathtext."""
    if shutil.which("latex") is None:
        return False
    try:
        with plt.rc_context({"text.usetex": True}):
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$\omega/\omega_0$")
            fig.canvas.draw()
            plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        print("LaTeX rendering failed: using matplotlib mathtext instead.")
        return False


def make_figure(data, outfile):
    fig = plt.figure(figsize=(10, 4))
    labels = ["0"] + [f"{r:g}" for r in data["tau_ratios"]]
    for i, label in enumerate(labels):
        plt.plot(data["omega"] / W0, np.log(data["S"][i]), linewidth=1.5,
                 label=rf"${label}$")

    plt.axvline(W1 / W0, linestyle="--", linewidth=1)
    plt.axvline(W2 / W0, linestyle="--", linewidth=1)
    plt.axvspan(W1 / W0, W2 / W0, color="red", alpha=0.1)
    plt.xlabel(r"$\omega/\omega_0$", fontsize=24)
    plt.ylabel(r"$PSD$", fontsize=24)
    plt.tick_params(axis="both", labelsize=24)
    plt.xticks([0, 1, 2], [r"$0$", r"$1$", r"$2$"])
    plt.xlim(0, 2)
    plt.legend(fontsize=19, loc="center left", bbox_to_anchor=(1.02, 0.45))

    fig.patches.append(Rectangle((0.3, 0.6), 0.16, 0.08, transform=fig.transFigure,
                                 fill=False, linewidth=1.5))
    fig.text(0.31, 0.63, r"$\Delta \omega /\omega_0 = 1.2$", fontsize=18,
             ha="left", va="center")
    fig.text(0.67, 0.9, r"$\tau_c/T_0$", fontsize=18, ha="left", va="center")
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)



# Main

def load_gamma0():
    """Read Gamma_0 from noise_calibration.py output, or use the default."""
    if GAMMA0_FILE.exists():
        with open(GAMMA0_FILE) as f:
            g = json.load(f)["Gamma0"]
        print(f"Gamma_0 = {g:.6e} (from {GAMMA0_FILE})")
        return g
    print(f"Gamma_0 = {GAMMA0_DEFAULT:.6e} (default value used in the paper)")
    return GAMMA0_DEFAULT


def main():
    parser = argparse.ArgumentParser(description="Reproduce Fig. 3 (noise PSD).")
    parser.add_argument("--plot-only", action="store_true",
                        help="plot from data/fig03/fig03_data.npz")
    parser.add_argument("--t-max", type=float, default=T_MAX)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--rate-units", action="store_true",
                        help="plot the PSD of xi(t) itself instead of the per-step "
                             "increments xi(t)*dt used for the published figure")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "fig03_data.npz"
    plt.rcParams["text.usetex"] = latex_available()

    if args.plot_only:
        data = dict(np.load(data_file))
        print(f"Loaded data from {data_file}")
    else:
        nt = int(args.t_max / DT)
        Gamma0 = load_gamma0()
        white, colored = generate_noises(Gamma0, args.sigma, nt, SEED)

        # The published figure shows the noise as the increment added per
        # time step (xi * dt); --rate-units shows xi(t) itself.
        scale = 1.0 if args.rate_units else DT
        signals = [white * scale] + [c * scale for c in colored]

        nperseg = min(NPERSEG, nt)
        spectra = [psd_omega(s, nperseg) for s in signals]
        omega = spectra[0][0]

        # Check of Eq. (16): equal power in the band [W1, W2]
        p_white = band_power(*spectra[0])
        print(f"Band [{W1:.3f}, {W2:.3f}], power relative to white noise:")
        for label, (om, S) in zip(["white"] + [f"tau_c/T0 = {r:g}" for r in TAU_RATIOS], spectra):
            print(f"  {label:<14}: {band_power(om, S) / p_white:.4f}")

        keep = omega <= 2 * W0
        data = dict(omega=omega[keep],
                    S=np.array([S[keep] for _, S in spectra]),
                    tau_ratios=TAU_RATIOS, sigma=args.sigma, Gamma0=Gamma0,
                    rate_units=args.rate_units)
        np.savez_compressed(data_file, **data)
        print(f"Saved data to {data_file}")

    outfile = OUTPUT_DIR / "fig03_noise_psd.pdf"
    make_figure(data, outfile)
    print(f"Saved figure to {outfile}")


if __name__ == "__main__":
    main()
