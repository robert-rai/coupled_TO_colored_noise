"""
Figure 9 data: P'_rms in the (omega_B/omega_A, K_d) plane under dissipative coupling.


Two Rijke tubes (K_A = K_B = 0.8) with dissipative coupling only (K_tau = 0),
Eq. (8), with tube B detuned (omega_j^B = (omega_B/omega_A) * j * pi), on the grid
    K_d             = 0, 0.02, ..., 0.8   (41 values, rows)
    omega_B/omega_A = 0.8, 0.81, ..., 1.2 (41 values, columns)

Two kinds of runs:
    noise      white noise and five OU noises (tau_c/T0 = 0.1, 0.5, 1, 2, 5),
               n_runs realizations per sigma, independent noise in each tube
               (Fig. 9b-u). Within one realization all six noise types are
               built from the same white-noise increments.
    no noise   n_runs random initial conditions (Fig. 9a)

This script only computes and saves data (no plotting).

Output files in data/fig09/, each an array of shape (41, 41) = (K_d, omega_B/omega_A):
    Pa_2D_mean_{name}_inten{i}.npy        mean P'_rms of tube A
    Pb_2D_mean_{name}_inten{i}.npy        mean P'_rms of tube B
    Pa_2D_std_{name}_inten{i}.npy         standard deviation over realizations (tube A)
    Pb_2D_std_{name}_inten{i}.npy         standard deviation over realizations (tube B)
    Pa_dissi_2D_mean_nn_inten1.npy        no noise, mean P'_rms of tube A
    Pa_dissi_2D_std_nn_inten1.npy         no noise, standard deviation (ddof = 0)

    name: wn (white noise), cn1 ... cn5 (tau_c/T0 = 0.1, 0.5, 1, 2, 5)
    i:    1, 2, 3, 4 for sigma = 0.05, 0.1, 0.2, 0.4 (see INTENSITY_INDEX)

    Note: the noisy-case file names are the same as for Fig. 6, so the two
    figures are kept in separate folders (data/fig06/ and data/fig09/).


Note: the full computation is expensive (41 x 41 grid x 6 noises x n_runs
simulations per sigma). Running one sigma at a time is recommended.
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

KD_VALUES = np.round(np.arange(0, 0.81, 0.02), 2)       # rows
RATIO_VALUES = np.round(np.arange(0.8, 1.21, 0.01), 2)  # columns, omega_B / omega_A

SIGMAS = [0.05, 0.1, 0.2, 0.4]  # noise amplitudes in Fig. 9
INTENSITY_INDEX = {0.05: 1, 0.1: 2, 0.2: 3, 0.4: 4}  # sigma -> "inten" number in file names
N_RUNS = 10                   # realizations (noise or initial conditions)
GAMMA0_DEFAULT = 5.761620e-01 # reference white-noise intensity

W0 = 3.2546899891190257       # natural frequency at K = 0.8 (Fig. 1)
T0 = 2 * np.pi / W0
TAU_RATIOS = np.array([0.1, 0.5, 1, 2, 5])  # tau_c / T0 of the OU noises
W1 = W0 - 0.6 * W0            # band of width 1.2 * w0 around w0
W2 = W0 + 0.6 * W0
NOISE_NAMES = ["wn", "cn1", "cn2", "cn3", "cn4", "cn5"]

BASE_SEED = 9000

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "fig09"
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
    """OU process xi(t) driven by white-noise increments dW, Eq. (11)."""
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

def run_point(Kd, ratio, noise_a, noise_b, seed):
    """P'_rms of tubes A and B at one (K_d, omega_B/omega_A) grid point."""
    pa, pb = simulate(Kd, ratio, K_HEATER, K_HEATER, noise_a, noise_b, DT, seed)
    return p_rms(pa), p_rms(pb)


def ic_seed(run, i, j):
    """Seed for the random initial conditions of one simulation."""
    return BASE_SEED + 100_000 * run + 100 * i + j


def noise_row_task(noise_idx, run, i, sigma, Gamma0, nt, ratio_values, kd_values):
    """One noise type and realization, one K_d row, all detuning values."""
    noise_a = noise_increments(noise_idx, sigma, Gamma0, nt, run, 0)
    noise_b = noise_increments(noise_idx, sigma, Gamma0, nt, run, 1)
    rms = np.empty((2, len(ratio_values)))
    for j, ratio in enumerate(ratio_values):
        rms[:, j] = run_point(kd_values[i], ratio, noise_a, noise_b, ic_seed(run, i, j))
    return rms


def no_noise_point_task(i, j, nt, n_runs, ratio_values, kd_values):
    """No noise: n_runs initial conditions at one grid point (mean and std, ddof=0)."""
    zeros = np.zeros(nt)
    rms = np.empty((2, n_runs))
    for run in range(n_runs):
        rms[:, run] = run_point(kd_values[i], ratio_values[j], zeros, zeros,
                                ic_seed(run, i, j))
    return rms.mean(axis=1), rms.std(axis=1)


def intensity_tag(sigma):
    """File-name tag for a noise amplitude, e.g. sigma = 0.05 -> 'inten1'."""
    for key, idx in INTENSITY_INDEX.items():
        if np.isclose(sigma, key):
            return f"inten{idx}"
    return f"sigma{sigma:g}"


def compute_sigma(sigma, Gamma0, nt, n_runs, n_jobs, kd_values, ratio_values):
    n_kd, n_ratio = len(kd_values), len(ratio_values)
    tasks = [(n_idx, run, i) for run in range(n_runs) for n_idx in range(6)
             for i in range(n_kd)]
    print(f"sigma = {sigma}: {len(tasks)} tasks of {n_ratio} simulations")
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(noise_row_task)(n_idx, run, i, sigma, Gamma0, nt, ratio_values, kd_values)
        for n_idx, run, i in tasks
    )
    rms_pa = np.empty((n_runs, 6, n_kd, n_ratio))
    rms_pb = np.empty((n_runs, 6, n_kd, n_ratio))
    for (n_idx, run, i), r in zip(tasks, out):
        rms_pa[run, n_idx, i] = r[0]
        rms_pb[run, n_idx, i] = r[1]

    rms_pa_mean = rms_pa.mean(axis=0)
    rms_pb_mean = rms_pb.mean(axis=0)
    rms_pa_std = rms_pa.std(axis=0, ddof=1)
    rms_pb_std = rms_pb.std(axis=0, ddof=1)

    tag = intensity_tag(sigma)
    for n_idx, name in enumerate(NOISE_NAMES):
        np.save(DATA_DIR / f"Pa_2D_mean_{name}_{tag}.npy", rms_pa_mean[n_idx])
        np.save(DATA_DIR / f"Pb_2D_mean_{name}_{tag}.npy", rms_pb_mean[n_idx])
        np.save(DATA_DIR / f"Pa_2D_std_{name}_{tag}.npy", rms_pa_std[n_idx])
        np.save(DATA_DIR / f"Pb_2D_std_{name}_{tag}.npy", rms_pb_std[n_idx])
    print(f"Saved P*_2D_*_{tag}.npy to {DATA_DIR}")


def compute_no_noise(nt, n_runs, n_jobs, kd_values, ratio_values):
    n_kd, n_ratio = len(kd_values), len(ratio_values)
    print(f"No noise: {n_kd * n_ratio} tasks of {n_runs} simulations")
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(no_noise_point_task)(i, j, nt, n_runs, ratio_values, kd_values)
        for i in range(n_kd) for j in range(n_ratio)
    )
    mean = np.array([o[0] for o in out]).reshape(n_kd, n_ratio, 2)
    std = np.array([o[1] for o in out]).reshape(n_kd, n_ratio, 2)
    np.save(DATA_DIR / "Pa_dissi_2D_mean_nn_inten1.npy", mean[:, :, 0])
    np.save(DATA_DIR / "Pa_dissi_2D_std_nn_inten1.npy", std[:, :, 0])
    print(f"Saved Pa_dissi_2D_*_nn_inten1.npy to {DATA_DIR}")



# Main

def load_gamma0(override):
    if override is not None:
        return override
    if GAMMA0_FILE.exists():
        with open(GAMMA0_FILE) as f:
            return json.load(f)["Gamma0"]
    return GAMMA0_DEFAULT


def main():
    parser = argparse.ArgumentParser(description="Compute the data for Fig. 9 ((omega_B/omega_A, K_d) maps).")
    parser.add_argument("--sigmas", type=float, nargs="*", default=SIGMAS,
                        help=f"noise amplitudes to compute (default {SIGMAS}); "
                             "give no values to skip the noisy runs")
    parser.add_argument("--skip-no-noise", action="store_true",
                        help="do not compute the no-noise map")
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--t-max", type=float, default=T_MAX)
    parser.add_argument("--gamma0", type=float, default=None,
                        help="override the reference intensity Gamma_0")
    parser.add_argument("--grid-step", type=int, default=1,
                        help="use every n-th grid value (for quick tests only)")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    nt = int(args.t_max / DT)
    Gamma0 = load_gamma0(args.gamma0)
    kd_values = KD_VALUES[::args.grid_step]
    ratio_values = RATIO_VALUES[::args.grid_step]
    print(f"Time steps: {nt}, Gamma_0 = {Gamma0:.6e}, realizations: {args.n_runs}, "
          f"grid: {len(kd_values)} x {len(ratio_values)}")

    if not args.skip_no_noise:
        compute_no_noise(nt, args.n_runs, args.n_jobs, kd_values, ratio_values)
    for sigma in args.sigmas:
        compute_sigma(sigma, Gamma0, nt, args.n_runs, args.n_jobs, kd_values, ratio_values)


if __name__ == "__main__":
    main()
