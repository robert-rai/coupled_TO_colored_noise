"""
Figure 8 data: P'_rms vs frequency detuning omega_B / omega_A.

Two Rijke tubes (K_A = K_B = 0.8) with dissipative coupling only
(K_tau = 0, K_d = 0.2), Eq. (8). Tube B is detuned: its modal frequencies are
omega_j^B = (omega_B/omega_A) * j * pi, with omega_B/omega_A = 0.8, 0.81, ..., 1.2. Each tube receives an independent noise.
For every noise amplitude sigma, the white noise and five OU noises
(tau_c/T0 = 0.1, 0.5, 1, 2, 5) are simulated over n_runs realizations.
Within one realization all six noise types are built from the same white-noise
increments, so differences between them come from the noise colour only.

This script only computes and saves data (no plotting).

Output files in data/fig08/ (one value per omega_B/omega_A = 0.8, 0.81, ..., 1.2):
    Pa_delay_Kd_{name}_inten{i}.npy       mean P'_rms of tube A
    Pb_delay_Kd_{name}_inten{i}.npy       mean P'_rms of tube B
    Pa_delay_Kd_std_{name}_inten{i}.npy   standard deviation over realizations (tube A)
    Pb_delay_Kd_std_{name}_inten{i}.npy   standard deviation over realizations (tube B)
    Pa_delay_Kd_nn_inten1.npy             no-noise mean P'_rms of tube A
    Pa_delay_Kd_std_nn_inten1.npy         no-noise standard deviation (tube A, ddof = 0)
    (as in the original notebook, the no-noise files carry the inten1 tag only)

    name: wn (white noise), cn1 ... cn5 (tau_c/T0 = 0.1, 0.5, 1, 2, 5)
    i:    1, 2, 3 for sigma = 0.05, 0.1, 0.2 (see INTENSITY_INDEX)

    The standard error of the mean is std / sqrt(n_runs).


"""

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numba import njit


# Parameters

T_MAX = 5000.0
DT = 0.001
TRANSIENT_FRAC = 0.8          # P'_rms over the final 20% of the signal

N = 10                        # Galerkin modes
gamma = 1.4
Ma = 0.01
tf = 0.2
xf = 0.25
C1, C2 = 0.1, 0.06
K_HEATER = 0.8                # K_A = K_B

KD = 0.2                      # dissipative coupling strength
RATIO_VALUES = np.round(np.arange(0.8, 1.21, 0.01), 2)  # omega_B / omega_A

SIGMAS = [0.05, 0.1, 0.2]     # noise amplitudes in the paper
INTENSITY_INDEX = {0.05: 1, 0.1: 2, 0.2: 3, 0.4: 4}  # sigma -> "inten" number in file names
N_RUNS = 50                   # noise realizations per sigma
GAMMA0_DEFAULT = 5.761620e-01 # reference white-noise intensity

W0 = 3.2546899891190257       # natural frequency at K = 0.8 (Fig. 1)
T0 = 2 * np.pi / W0
TAU_RATIOS = np.array([0.1, 0.5, 1, 2, 5])  # tau_c / T0 of the OU noises
W1 = W0 - 0.6 * W0            # band of width 1.2 * w0 around w0
W2 = W0 + 0.6 * W0
NOISE_NAMES = ["wn", "cn1", "cn2", "cn3", "cn4", "cn5"]

BASE_SEED = 8000

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "fig08"
GAMMA0_FILE = REPO_ROOT / "data" / "noise_calibration" / "gamma0.json"



# Model:

@njit
def heat_release(uf):
    return np.sqrt(np.abs(1.0 / 3.0 + uf)) - 1.0 / np.sqrt(3.0)


def zeta_func(j):
    return (1.0 / (2.0 * np.pi)) * (C1 * j + C2 * np.sqrt(1.0 / j))


modes = np.arange(1, N + 1)
zeta = np.array([zeta_func(j) for j in modes], dtype=np.float64)
phi = np.cos(np.pi * modes * xf)
modal_coeff = gamma * Ma * np.sin(np.pi * modes * xf) / (modes * np.pi)


@njit
def rhs(eta, etadot, zeta_j, j, etadot_other, qdot, Kd, K, ratio):
    """
    Modal acceleration with dissipative coupling with K_tau = 0.
    ratio = omega_B / omega_A scales the modal frequencies: omega_j = ratio * j * pi
    .
    """
    return (-2 * zeta_j * ratio * j * np.pi * etadot
            - (ratio * j * np.pi) ** 2 * eta
            - np.pi * j * K * np.sin(j * np.pi * xf) * qdot
            + Kd * (etadot_other - etadot))


@njit
def simulate(Kd, ratio, K1, K2, noise_a, noise_b, dt, seed):
    """
    RK4 integration of the dissipatively coupled tubes. Tube B has modal
    frequencies scaled by ratio = omega_B / omega_A. The coupling is
    instantaneous and uses the partner's current RK4 stage values.
    noise_a / noise_b are the noise increments added to every modal velocity
    of tube A / B at each step. Returns the acoustic pressures of A and B.
    """
    np.random.seed(seed)
    nt = noise_a.shape[0]

    eta1 = np.zeros((N, nt))
    eta2 = np.zeros((N, nt))
    etadot1 = np.zeros((N, nt))
    etadot2 = np.zeros((N, nt))
    eta1[:, 0] = np.random.random(N)
    eta2[:, 0] = np.random.random(N)
    etadot1[:, 0] = np.random.random(N)
    etadot2[:, 0] = np.random.random(N)

    for i in range(1, nt):
        tn = i * dt

        # Thermal delay t - tf (heat release)
        i_delay = (tn - tf) / dt
        i0 = int(np.floor(i_delay))
        i1 = i0 + 1
        if i0 >= 0 and i1 < nt:
            w = i_delay - i0
            eta_delay1 = (1 - w) * eta1[:, i0] + w * eta1[:, i1]
            eta_delay2 = (1 - w) * eta2[:, i0] + w * eta2[:, i1]
        else:
            eta_delay1 = eta1[:, 0].copy()
            eta_delay2 = eta2[:, 0].copy()

        qdot1 = heat_release(np.sum(eta_delay1 * phi))
        qdot2 = heat_release(np.sum(eta_delay2 * phi))

        for j in range(N):
            e1, v1 = eta1[j, i - 1], etadot1[j, i - 1]
            e2, v2 = eta2[j, i - 1], etadot2[j, i - 1]

            k1_e1 = v1
            k1_e2 = v2
            k1_v1 = rhs(e1, v1, zeta[j], j + 1, v2, qdot1, Kd, K1, 1.0)
            k1_v2 = rhs(e2, v2, zeta[j], j + 1, v1, qdot2, Kd, K2, ratio)

            k2_e1 = v1 + 0.5 * dt * k1_v1
            k2_e2 = v2 + 0.5 * dt * k1_v2
            k2_v1 = rhs(e1 + 0.5 * dt * k1_e1, v1 + 0.5 * dt * k1_v1, zeta[j], j + 1,
                        v2 + 0.5 * dt * k1_v2, qdot1, Kd, K1, 1.0)
            k2_v2 = rhs(e2 + 0.5 * dt * k1_e2, v2 + 0.5 * dt * k1_v2, zeta[j], j + 1,
                        v1 + 0.5 * dt * k1_v1, qdot2, Kd, K2, ratio)

            k3_e1 = v1 + 0.5 * dt * k2_v1
            k3_e2 = v2 + 0.5 * dt * k2_v2
            k3_v1 = rhs(e1 + 0.5 * dt * k2_e1, v1 + 0.5 * dt * k2_v1, zeta[j], j + 1,
                        v2 + 0.5 * dt * k2_v2, qdot1, Kd, K1, 1.0)
            k3_v2 = rhs(e2 + 0.5 * dt * k2_e2, v2 + 0.5 * dt * k2_v2, zeta[j], j + 1,
                        v1 + 0.5 * dt * k2_v1, qdot2, Kd, K2, ratio)

            k4_e1 = v1 + dt * k3_v1
            k4_e2 = v2 + dt * k3_v2
            k4_v1 = rhs(e1 + dt * k3_e1, v1 + dt * k3_v1, zeta[j], j + 1,
                        v2 + dt * k3_v2, qdot1, Kd, K1, 1.0)
            k4_v2 = rhs(e2 + dt * k3_e2, v2 + dt * k3_v2, zeta[j], j + 1,
                        v1 + dt * k3_v1, qdot2, Kd, K2, ratio)

            eta1[j, i] = e1 + dt * (k1_e1 + 2 * k2_e1 + 2 * k3_e1 + k4_e1) / 6
            eta2[j, i] = e2 + dt * (k1_e2 + 2 * k2_e2 + 2 * k3_e2 + k4_e2) / 6
            etadot1[j, i] = v1 + dt * (k1_v1 + 2 * k2_v1 + 2 * k3_v1 + k4_v1) / 6 + noise_a[i]
            etadot2[j, i] = v2 + dt * (k1_v2 + 2 * k2_v2 + 2 * k3_v2 + k4_v2) / 6 + noise_b[i]

    p1 = np.zeros(nt)
    p2 = np.zeros(nt)
    for j in range(N):
        p1 -= modal_coeff[j] * etadot1[j, :]
        p2 -= modal_coeff[j] * etadot2[j, :]
    return p1, p2


def p_rms(p):
    return np.sqrt(np.mean(p[int(TRANSIENT_FRAC * len(p)):] ** 2))



# White-noise increments and power-normalized OU noise

def normalization_D(tau_c):
    """Eq. (17)."""
    return tau_c * (W2 - W1) / (np.arctan(W2 * tau_c) - np.arctan(W1 * tau_c))


@njit
def ou_euler_maruyama(dW, D, tau_c, dt):
    """OU process xi(t) driven by white-noise increments dW."""
    n = dW.shape[0]
    xi = np.zeros(n)
    a = dt / tau_c
    b = np.sqrt(D) / tau_c
    for k in range(n - 1):
        xi[k + 1] = xi[k] - a * xi[k] + b * dW[k]
    return xi


def noise_increments(noise_idx, sigma, Gamma0, nt, run, oscillator):
    """
    Per-step noise increment for one oscillator.
    noise_idx = 0: white noise, increment dW = sigma*sqrt(Gamma0*dt)*N(0,1)
    noise_idx > 0: OU noise with tau_c = TAU_RATIOS[noise_idx-1]*T0, increment xi*dt
    The same seed (run, oscillator) is used for every noise type and sigma.
    """
    rng = np.random.default_rng([BASE_SEED, run, oscillator])
    dW = sigma * np.sqrt(Gamma0 * DT) * rng.standard_normal(nt)
    if noise_idx == 0:
        return dW
    tau_c = TAU_RATIOS[noise_idx - 1] * T0
    return ou_euler_maruyama(dW, normalization_D(tau_c), tau_c, DT) * DT



# Computation

def one_task(noise_idx, run, sigma, Gamma0, nt):
    """P'_rms of tubes A and B over all omega_B/omega_A, for one noise type and realization."""
    if sigma == 0:
        noise_a = np.zeros(nt)
        noise_b = np.zeros(nt)
    else:
        noise_a = noise_increments(noise_idx, sigma, Gamma0, nt, run, 0)
        noise_b = noise_increments(noise_idx, sigma, Gamma0, nt, run, 1)

    rms_a = np.empty(len(RATIO_VALUES))
    rms_b = np.empty(len(RATIO_VALUES))
    for k, ratio in enumerate(RATIO_VALUES):
        pa, pb = simulate(KD, ratio, K_HEATER, K_HEATER, noise_a, noise_b, DT, BASE_SEED + run)
        rms_a[k] = p_rms(pa)
        rms_b[k] = p_rms(pb)
    return rms_a, rms_b


def intensity_tag(sigma):
    """File-name tag for a noise amplitude, e.g. sigma = 0.05 -> 'inten1'."""
    for key, idx in INTENSITY_INDEX.items():
        if np.isclose(sigma, key):
            return f"inten{idx}"
    return f"sigma{sigma:g}"


def compute_sigma(sigma, Gamma0, nt, n_runs, n_jobs):
    """All noise types and realizations for one sigma; saves .npy files."""
    print(f"sigma = {sigma}: {6 * n_runs} tasks of {len(RATIO_VALUES)} simulations")
    tasks = [(n_idx, run) for run in range(n_runs) for n_idx in range(6)]
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(one_task)(n_idx, run, sigma, Gamma0, nt) for n_idx, run in tasks
    )
    # shape (n_runs, 6 noise types, n_ratio)
    rms_a = np.array([o[0] for o in out]).reshape(n_runs, 6, len(RATIO_VALUES))
    rms_b = np.array([o[1] for o in out]).reshape(n_runs, 6, len(RATIO_VALUES))

    rms_pa_mean = rms_a.mean(axis=0)
    rms_pb_mean = rms_b.mean(axis=0)
    rms_pa_std = rms_a.std(axis=0, ddof=1)
    rms_pb_std = rms_b.std(axis=0, ddof=1)

    tag = intensity_tag(sigma)
    for n_idx, name in enumerate(NOISE_NAMES):
        np.save(DATA_DIR / f"Pa_delay_Kd_{name}_{tag}.npy", rms_pa_mean[n_idx])
        np.save(DATA_DIR / f"Pb_delay_Kd_{name}_{tag}.npy", rms_pb_mean[n_idx])
        np.save(DATA_DIR / f"Pa_delay_Kd_std_{name}_{tag}.npy", rms_pa_std[n_idx])
        np.save(DATA_DIR / f"Pb_delay_Kd_std_{name}_{tag}.npy", rms_pb_std[n_idx])
    print(f"Saved Pa/Pb_delay_Kd_*_{tag}.npy to {DATA_DIR}")


def compute_no_noise(nt, n_runs, n_jobs, sigmas):
    """Deterministic reference, averaged over random initial conditions."""
    print(f"No noise: {n_runs} tasks of {len(RATIO_VALUES)} simulations")
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(one_task)(0, run, 0.0, 0.0, nt) for run in range(n_runs)
    )
    rms_pa_values_nn = np.array([o[0] for o in out])
    rms_pa_mean_nn = rms_pa_values_nn.mean(axis=0)
    rms_std_nn = rms_pa_values_nn.std(axis=0)  # ddof = 0, as in the original notebook

    # As in the original notebook, stored once with the inten1 tag
    np.save(DATA_DIR / "Pa_delay_Kd_nn_inten1.npy", rms_pa_mean_nn)
    np.save(DATA_DIR / "Pa_delay_Kd_std_nn_inten1.npy", rms_std_nn)
    print(f"Saved Pa_delay_Kd_nn_inten1.npy to {DATA_DIR}")



# Main

def load_gamma0(override):
    if override is not None:
        return override
    if GAMMA0_FILE.exists():
        with open(GAMMA0_FILE) as f:
            return json.load(f)["Gamma0"]
    return GAMMA0_DEFAULT


def main():
    parser = argparse.ArgumentParser(description="Compute the data for Fig. 8 (P'_rms vs omega_B/omega_A).")
    parser.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS,
                        help=f"noise amplitudes to compute (default {SIGMAS})")
    parser.add_argument("--skip-no-noise", action="store_true",
                        help="do not compute the deterministic reference")
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--t-max", type=float, default=T_MAX)
    parser.add_argument("--gamma0", type=float, default=None,
                        help="override the reference intensity Gamma_0")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    nt = int(args.t_max / DT)
    Gamma0 = load_gamma0(args.gamma0)
    print(f"Time steps: {nt}, Gamma_0 = {Gamma0:.6e}, runs per sigma: {args.n_runs}")

    if not args.skip_no_noise:
        compute_no_noise(nt, args.n_runs, args.n_jobs, args.sigmas)
    for sigma in args.sigmas:
        compute_sigma(sigma, Gamma0, nt, args.n_runs, args.n_jobs)


if __name__ == "__main__":
    main()
