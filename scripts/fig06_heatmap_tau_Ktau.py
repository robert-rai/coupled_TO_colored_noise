"""
Figure 6 data: P'_rms in the (tau, K_tau) plane under time-delay coupling.


Two identical Rijke tubes (K_A = K_B = 0.8) with time-delay coupling only
(K_d = 0), Eq. (8), on the grid
    K_tau = 0, 0.02, ..., 0.8   (41 values, rows)
    tau   = 0, 0.05, ..., 2     (41 values, columns)
At tau = 0 the coupling is instantaneous (simulate_instantaneous).

Three kinds of runs:
    noise      white noise and five OU noises (tau_c/T0 = 0.1, 0.5, 1, 2, 5),
               n_runs realizations per sigma, independent noise in each tube
               (Fig. 6c-v). Within one realization all six noise types are
               built from the same white-noise increments.
    out_phase  no noise, different random initial conditions (Fig. 6b)
    in_phase   no noise, identical initial conditions in both tubes (Fig. 6a)

This script only computes and saves data (no plotting).

Output files in data/fig06/, each an array of shape (41, 41) = (K_tau, tau):
    Pa_2D_mean_{name}_inten{i}.npy             mean P'_rms of tube A
    Pb_2D_mean_{name}_inten{i}.npy             mean P'_rms of tube B
    Pa_2D_std_{name}_inten{i}.npy              standard deviation over realizations (tube A)
    Pb_2D_std_{name}_inten{i}.npy              standard deviation over realizations (tube B)
    Pa_delay_2D_mean_nn_inten1_out_phase.npy   no noise, different initial conditions
    Pa_delay_2D_std_nn_inten1_out_phase.npy
    Pa_delay_2D_mean_nn_inten1_in_phase.npy    no noise, identical initial conditions
    Pa_delay_2D_std_nn_inten1_in_phase.npy

    name: wn (white noise), cn1 ... cn5 (tau_c/T0 = 0.1, 0.5, 1, 2, 5)
    i:    1, 2, 3, 4 for sigma = 0.05, 0.1, 0.2, 0.4 (see INTENSITY_INDEX)


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

KT_VALUES = np.round(np.arange(0, 0.81, 0.02), 2)    # rows
TAU_VALUES = np.round(np.arange(0, 2.05, 0.05), 2)   # columns

SIGMAS = [0.05, 0.1, 0.2, 0.4]  # noise amplitudes in Fig. 6
INTENSITY_INDEX = {0.05: 1, 0.1: 2, 0.2: 3, 0.4: 4}  # sigma -> "inten" number in file names
N_RUNS = 10                   # realizations (noise or initial conditions)
GAMMA0_DEFAULT = 5.761620e-01 # reference white-noise intensity

W0 = 3.2546899891190257       # natural frequency at K = 0.8 (Fig. 1)
T0 = 2 * np.pi / W0
TAU_RATIOS = np.array([0.1, 0.5, 1, 2, 5])  # tau_c / T0 of the OU noises
W1 = W0 - 0.6 * W0            # band of width 1.2 * w0 around w0
W2 = W0 + 0.6 * W0
NOISE_NAMES = ["wn", "cn1", "cn2", "cn3", "cn4", "cn5"]

BASE_SEED = 6000

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "fig06"
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
def rhs(eta, etadot, zeta_j, j, etadot_other_delayed, qdot, Kt, K):
    """Modal acceleration with time-delay coupling with K_d = 0."""
    return (-2 * zeta_j * j * np.pi * etadot
            - (j * np.pi) ** 2 * eta
            - np.pi * j * K * np.sin(j * np.pi * xf) * qdot
            + Kt * (etadot_other_delayed - etadot))


@njit
def simulate(Kt, tau, K1, K2, noise_a, noise_b, dt, seed, identical_ic):
    """
    RK4 integration of the coupled tubes. noise_a / noise_b are the noise
    increments added to every modal velocity of tube A / B at each step.
    identical_ic = True gives both tubes the same initial state.
    Returns the acoustic pressures of tubes A and B.
    """
    np.random.seed(seed)
    nt = noise_a.shape[0]

    eta1 = np.zeros((N, nt))
    eta2 = np.zeros((N, nt))
    etadot1 = np.zeros((N, nt))
    etadot2 = np.zeros((N, nt))
    if identical_ic:
        # Same initial state for both tubes (symmetry-preserving case, Fig. 6a)
        v = np.random.random(N)
        eta1[:, 0] = v
        eta2[:, 0] = v
        etadot1[:, 0] = v
        etadot2[:, 0] = v
    else:
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

        # Coupling delay t - tau
        i_delayC = (tn - tau) / dt
        i0C = int(np.floor(i_delayC))
        i1C = i0C + 1
        if i0C >= 0 and i1C < nt:
            wC = i_delayC - i0C
            etadot_delay1 = (1 - wC) * etadot1[:, i0C] + wC * etadot1[:, i1C]
            etadot_delay2 = (1 - wC) * etadot2[:, i0C] + wC * etadot2[:, i1C]
        else:
            etadot_delay1 = etadot1[:, 0]
            etadot_delay2 = etadot2[:, 0]

        qdot1 = heat_release(np.sum(eta_delay1 * phi))
        qdot2 = heat_release(np.sum(eta_delay2 * phi))

        for j in range(N):
            e1, v1 = eta1[j, i - 1], etadot1[j, i - 1]
            e2, v2 = eta2[j, i - 1], etadot2[j, i - 1]
            d2, d1 = etadot_delay2[j], etadot_delay1[j]

            k1_e1 = v1
            k1_e2 = v2
            k1_v1 = rhs(e1, v1, zeta[j], j + 1, d2, qdot1, Kt, K1)
            k1_v2 = rhs(e2, v2, zeta[j], j + 1, d1, qdot2, Kt, K2)

            k2_e1 = v1 + 0.5 * dt * k1_v1
            k2_e2 = v2 + 0.5 * dt * k1_v2
            k2_v1 = rhs(e1 + 0.5 * dt * k1_e1, v1 + 0.5 * dt * k1_v1, zeta[j], j + 1, d2, qdot1, Kt, K1)
            k2_v2 = rhs(e2 + 0.5 * dt * k1_e2, v2 + 0.5 * dt * k1_v2, zeta[j], j + 1, d1, qdot2, Kt, K2)

            k3_e1 = v1 + 0.5 * dt * k2_v1
            k3_e2 = v2 + 0.5 * dt * k2_v2
            k3_v1 = rhs(e1 + 0.5 * dt * k2_e1, v1 + 0.5 * dt * k2_v1, zeta[j], j + 1, d2, qdot1, Kt, K1)
            k3_v2 = rhs(e2 + 0.5 * dt * k2_e2, v2 + 0.5 * dt * k2_v2, zeta[j], j + 1, d1, qdot2, Kt, K2)

            k4_e1 = v1 + dt * k3_v1
            k4_e2 = v2 + dt * k3_v2
            k4_v1 = rhs(e1 + dt * k3_e1, v1 + dt * k3_v1, zeta[j], j + 1, d2, qdot1, Kt, K1)
            k4_v2 = rhs(e2 + dt * k3_e2, v2 + dt * k3_v2, zeta[j], j + 1, d1, qdot2, Kt, K2)

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


@njit
def simulate_instantaneous(Kt, K1, K2, noise_a, noise_b, dt, seed, identical_ic):
    """
    Same as simulate() but for tau = 0: the coupling uses the partner's
    current RK4 stage value instead of a stored (delayed) value.
    """
    np.random.seed(seed)
    nt = noise_a.shape[0]

    eta1 = np.zeros((N, nt))
    eta2 = np.zeros((N, nt))
    etadot1 = np.zeros((N, nt))
    etadot2 = np.zeros((N, nt))
    if identical_ic:
        # Same initial state for both tubes (symmetry-preserving case, Fig. 6a)
        v = np.random.random(N)
        eta1[:, 0] = v
        eta2[:, 0] = v
        etadot1[:, 0] = v
        etadot2[:, 0] = v
    else:
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
            k1_v1 = rhs(e1, v1, zeta[j], j + 1, v2, qdot1, Kt, K1)
            k1_v2 = rhs(e2, v2, zeta[j], j + 1, v1, qdot2, Kt, K2)

            k2_e1 = v1 + 0.5 * dt * k1_v1
            k2_e2 = v2 + 0.5 * dt * k1_v2
            k2_v1 = rhs(e1 + 0.5 * dt * k1_e1, v1 + 0.5 * dt * k1_v1, zeta[j], j + 1,
                        v2 + 0.5 * dt * k1_v2, qdot1, Kt, K1)
            k2_v2 = rhs(e2 + 0.5 * dt * k1_e2, v2 + 0.5 * dt * k1_v2, zeta[j], j + 1,
                        v1 + 0.5 * dt * k1_v1, qdot2, Kt, K2)

            k3_e1 = v1 + 0.5 * dt * k2_v1
            k3_e2 = v2 + 0.5 * dt * k2_v2
            k3_v1 = rhs(e1 + 0.5 * dt * k2_e1, v1 + 0.5 * dt * k2_v1, zeta[j], j + 1,
                        v2 + 0.5 * dt * k2_v2, qdot1, Kt, K1)
            k3_v2 = rhs(e2 + 0.5 * dt * k2_e2, v2 + 0.5 * dt * k2_v2, zeta[j], j + 1,
                        v1 + 0.5 * dt * k2_v1, qdot2, Kt, K2)

            k4_e1 = v1 + dt * k3_v1
            k4_e2 = v2 + dt * k3_v2
            k4_v1 = rhs(e1 + dt * k3_e1, v1 + dt * k3_v1, zeta[j], j + 1,
                        v2 + dt * k3_v2, qdot1, Kt, K1)
            k4_v2 = rhs(e2 + dt * k3_e2, v2 + dt * k3_v2, zeta[j], j + 1,
                        v1 + dt * k3_v1, qdot2, Kt, K2)

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

def run_point(Kt, tau, noise_a, noise_b, seed, identical_ic):
    """P'_rms of tubes A and B at one (K_tau, tau) grid point."""
    if tau == 0:
        pa, pb = simulate_instantaneous(Kt, K_HEATER, K_HEATER, noise_a, noise_b,
                                        DT, seed, identical_ic)
    else:
        pa, pb = simulate(Kt, tau, K_HEATER, K_HEATER, noise_a, noise_b,
                          DT, seed, identical_ic)
    return p_rms(pa), p_rms(pb)


def ic_seed(run, i, j):
    """Seed for the random initial conditions of one simulation."""
    return BASE_SEED + 100_000 * run + 100 * i + j


def noise_row_task(noise_idx, run, i, sigma, Gamma0, nt, tau_values, kt_values):
    """One noise type and realization, one K_tau row, all tau values."""
    noise_a = noise_increments(noise_idx, sigma, Gamma0, nt, run, 0)
    noise_b = noise_increments(noise_idx, sigma, Gamma0, nt, run, 1)
    rms = np.empty((2, len(tau_values)))
    for j, tau in enumerate(tau_values):
        rms[:, j] = run_point(kt_values[i], tau, noise_a, noise_b,
                              ic_seed(run, i, j), False)
    return rms


def no_noise_point_task(i, j, nt, n_runs, identical_ic, tau_values, kt_values):
    """No noise: n_runs initial conditions at one grid point (mean and std, ddof=0)."""
    zeros = np.zeros(nt)
    rms = np.empty((2, n_runs))
    for run in range(n_runs):
        rms[:, run] = run_point(kt_values[i], tau_values[j], zeros, zeros,
                                ic_seed(run, i, j), identical_ic)
    return rms.mean(axis=1), rms.std(axis=1)


def intensity_tag(sigma):
    """File-name tag for a noise amplitude, e.g. sigma = 0.05 -> 'inten1'."""
    for key, idx in INTENSITY_INDEX.items():
        if np.isclose(sigma, key):
            return f"inten{idx}"
    return f"sigma{sigma:g}"


def compute_sigma(sigma, Gamma0, nt, n_runs, n_jobs, kt_values, tau_values):
    n_kt, n_tau = len(kt_values), len(tau_values)
    tasks = [(n_idx, run, i) for run in range(n_runs) for n_idx in range(6)
             for i in range(n_kt)]
    print(f"sigma = {sigma}: {len(tasks)} tasks of {n_tau} simulations")
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(noise_row_task)(n_idx, run, i, sigma, Gamma0, nt, tau_values, kt_values)
        for n_idx, run, i in tasks
    )
    rms_pa = np.empty((n_runs, 6, n_kt, n_tau))
    rms_pb = np.empty((n_runs, 6, n_kt, n_tau))
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


def compute_no_noise(nt, n_runs, n_jobs, identical_ic, kt_values, tau_values):
    label = "in_phase" if identical_ic else "out_phase"
    n_kt, n_tau = len(kt_values), len(tau_values)
    print(f"No noise ({label}): {n_kt * n_tau} tasks of {n_runs} simulations")
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(no_noise_point_task)(i, j, nt, n_runs, identical_ic, tau_values, kt_values)
        for i in range(n_kt) for j in range(n_tau)
    )
    mean = np.array([o[0] for o in out]).reshape(n_kt, n_tau, 2)
    std = np.array([o[1] for o in out]).reshape(n_kt, n_tau, 2)
    np.save(DATA_DIR / f"Pa_delay_2D_mean_nn_inten1_{label}.npy", mean[:, :, 0])
    np.save(DATA_DIR / f"Pa_delay_2D_std_nn_inten1_{label}.npy", std[:, :, 0])
    print(f"Saved Pa_delay_2D_*_nn_inten1_{label}.npy to {DATA_DIR}")


# Main
def load_gamma0(override):
    if override is not None:
        return override
    if GAMMA0_FILE.exists():
        with open(GAMMA0_FILE) as f:
            return json.load(f)["Gamma0"]
    return GAMMA0_DEFAULT


def main():
    parser = argparse.ArgumentParser(description="Compute the data for Fig. 6 ((tau, K_tau) maps).")
    parser.add_argument("--sigmas", type=float, nargs="*", default=SIGMAS,
                        help=f"noise amplitudes to compute (default {SIGMAS}); "
                             "give no values to skip the noisy runs")
    parser.add_argument("--skip-no-noise", action="store_true",
                        help="do not compute the two no-noise maps")
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
    kt_values = KT_VALUES[::args.grid_step]
    tau_values = TAU_VALUES[::args.grid_step]
    print(f"Time steps: {nt}, Gamma_0 = {Gamma0:.6e}, realizations: {args.n_runs}, "
          f"grid: {len(kt_values)} x {len(tau_values)}")

    if not args.skip_no_noise:
        compute_no_noise(nt, args.n_runs, args.n_jobs, False, kt_values, tau_values)
        compute_no_noise(nt, args.n_runs, args.n_jobs, True, kt_values, tau_values)
    for sigma in args.sigmas:
        compute_sigma(sigma, Gamma0, nt, args.n_runs, args.n_jobs, kt_values, tau_values)


if __name__ == "__main__":
    main()
