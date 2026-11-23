"""
Figure 7 data: coherence factor beta under time-delay coupling.


Two identical Rijke tubes with time-delay coupling (K_tau = 0.2, tau = 0.1,
K_d = 0), operated below the saddle-node point of the coupled system, driven
by independent white or OU noise (tau_c/T0 = 0.1, 0.5, 1, 2, 5). For each
simulation the coherence factor beta = H_p * f_p / delta_f is obtained from a
Lorentzian fit to the Welch spectrum of the acoustic pressure (Sec. 3.1.2).


  --sweep sigma   beta vs noise amplitude sigma = 0, 0.02, ..., 0.94
                  at fixed heater power K (default K = 0.54)       -> Fig. 7a, 7b
                  For Fig. 7b run it for K = 0.47, 0.51 and 0.54 and take,
                  for each noise colour, the sigma at which mean beta peaks.

  --sweep K       beta vs heater power K = 0.45, 0.47, 0.49, 0.51
                  at fixed sigma = 0.05                              -> Fig. 7c, 7d

This script only computes and saves data (no plotting).

Output (file names as in the original notebooks; the two sweeps use the same
names, so each run is written to its own folder):

  data/fig07/sigma_sweep_K<K>/      e.g. data/fig07/sigma_sweep_K0.54/
  data/fig07/K_sweep_sigma<sigma>/  e.g. data/fig07/K_sweep_sigma0.05/

    betaA_wn_power.npy       mean beta, tube A, white noise          shape (n,)
    betaA_cn_power.npy       mean beta, tube A, OU noises            shape (n, 5)
    betaB_wn_power.npy       mean beta, tube B, white noise
    betaB_cn_power.npy       mean beta, tube B, OU noises
    betaA_std_wn_power.npy   standard deviations over realizations (ddof = 1)
    betaA_std_cn_power.npy
    betaB_std_wn_power.npy
    betaB_std_cn_power.npy
    intensity.npy            sigma values (sigma sweep only)
    K_values.npy             K values (K sweep only)

    n = number of sigma values or K values; the 5 columns of the *_cn_* files
    are tau_c/T0 = 0.1, 0.5, 1, 2, 5.


"""

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from scipy.optimize import curve_fit
from scipy.signal import welch


# Parameters
T_MAX = 5000.0
DT = 0.001
IC_SCALE = 1e-3               # initial conditions uniform in [0, 1e-3)

N = 10                        # Galerkin modes
gamma = 1.4
Ma = 0.01
tf = 0.2
xf = 0.25
C1, C2 = 0.1, 0.06

KT = 0.2                      # time-delay coupling strength
TAU = 0.1                     # coupling delay

# Sigma sweep (Fig. 7a, 7b)
SIGMA_VALUES = np.round(np.arange(0, 0.95, 0.02), 2)
K_SIGMA_SWEEP = [0.54]

# K sweep (Fig. 7c, 7d)
K_VALUES = [0.45, 0.47, 0.49, 0.51]
SIGMA_K_SWEEP = 0.05

N_RUNS = 50                   # noise realizations
GAMMA0_DEFAULT = 5.761620e-01 # reference white-noise intensity

W0 = 3.2546899891190257       # natural frequency at K = 0.8 (Fig. 1)
T0 = 2 * np.pi / W0
TAU_RATIOS = np.array([0.1, 0.5, 1, 2, 5])  # tau_c / T0 of the OU noises
W1 = W0 - 0.6 * W0            # band of width 1.2 * w0 around w0
W2 = W0 + 0.6 * W0

# Coherence factor
F0_GUESS = 0.5                # expected peak frequency (omega / 2 pi), search band 0.5-1.5 x F0_GUESS
NPERSEG = 300000              # Welch segment length
COHERENCE_FRAC = 0.5          # spectrum computed over the final 50% of the signal

BASE_SEED = 7000

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "fig07"
GAMMA0_FILE = REPO_ROOT / "data" / "noise_calibration" / "gamma0.json"


# Model

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
def simulate(Kt, tau, K1, K2, noise_a, noise_b, dt, seed):
    """
    RK4 integration of the coupled tubes. noise_a / noise_b are the noise
    increments added to every modal velocity of tube A / B at each step.
    Returns the acoustic pressures of tubes A and B.
    """
    np.random.seed(seed)
    nt = noise_a.shape[0]

    eta1 = np.zeros((N, nt))
    eta2 = np.zeros((N, nt))
    etadot1 = np.zeros((N, nt))
    etadot2 = np.zeros((N, nt))
    eta1[:, 0] = IC_SCALE * np.random.random(N)
    eta2[:, 0] = IC_SCALE * np.random.random(N)
    etadot1[:, 0] = IC_SCALE * np.random.random(N)
    etadot2[:, 0] = IC_SCALE * np.random.random(N)

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



# Noise 

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


def noise_from_normals(z, noise_idx, sigma, Gamma0):
    """
    Per-step noise increment built from standard normal numbers z.
    noise_idx = 0: white noise, dW = sigma*sqrt(Gamma0*dt)*z
    noise_idx > 0: OU noise with tau_c = TAU_RATIOS[noise_idx-1]*T0, increment xi*dt
    """
    dW = sigma * np.sqrt(Gamma0 * DT) * z
    if noise_idx == 0:
        return dW
    tau_c = TAU_RATIOS[noise_idx - 1] * T0
    return ou_euler_maruyama(dW, normalization_D(tau_c), tau_c, DT) * DT



# Coherence factor

def lorentzian(f, A, f0, df):
    return A / (1 + ((f - f0) / (df / 2)) ** 2)


def coherence_factor(p, dt, f0=None):
    """
    beta = H_p * f_p / delta_f from a Lorentzian fit to the dominant peak of
    the Welch spectrum. Returns beta, f_p, delta_f, H_p.
    """
    p = p - np.mean(p)
    p = p[int(COHERENCE_FRAC * len(p)):]  # remove transient

    nperseg = min(NPERSEG, len(p))
    f, Pxx = welch(p, fs=1 / dt, window="hann", nperseg=nperseg,
                   noverlap=nperseg // 2, scaling="density")

    if f0 is not None:
        mask = (f >= 0.5 * f0) & (f <= 1.5 * f0)
        f_band, P_band = f[mask], Pxx[mask]
    else:
        f_band, P_band = f, Pxx
    if len(P_band) == 0:
        raise ValueError("Frequency band is empty.")

    # Initial guesses from the highest point and its half-power width
    peak_idx = np.argmax(P_band)
    fp_guess = f_band[peak_idx]
    Hp_guess = P_band[peak_idx]

    initial_width = 0.5 * fp_guess
    initial_mask = (f_band >= fp_guess - initial_width) & (f_band <= fp_guess + initial_width)
    f_initial = f_band[initial_mask]
    P_initial = P_band[initial_mask]

    indices = np.where(P_initial >= Hp_guess / 2)[0]
    if len(indices) > 1:
        df_guess = f_initial[indices[-1]] - f_initial[indices[0]]
    else:
        df_guess = 0.05 * fp_guess
    df_guess = max(df_guess, 1e-6)

    # Lorentzian fit around the peak
    fit_width = max(0.1 * fp_guess, 2.0 * df_guess)
    fit_mask = (f_band >= fp_guess - fit_width) & (f_band <= fp_guess + fit_width)
    try:
        popt, _ = curve_fit(
            lorentzian, f_band[fit_mask], P_band[fit_mask],
            p0=[Hp_guess, fp_guess, df_guess],
            bounds=([0, 0.8 * fp_guess, 1e-6], [np.inf, 1.2 * fp_guess, 2 * fp_guess]),
            max_nfev=10000,
        )
        Hp, fp, df = popt
    except RuntimeError:
        print("Lorentzian fit failed, using direct FWHM")
        Hp, fp, df = Hp_guess, fp_guess, df_guess

    return Hp * (fp / df), fp, df, Hp


# Computation
def beta_all_noises(K, sigma, zA, zB, Gamma0, ic_seed):
    """beta of tubes A and B for the white noise and the five OU noises."""
    betaA = np.empty(6)
    betaB = np.empty(6)
    for n_idx in range(6):
        noise_a = noise_from_normals(zA, n_idx, sigma, Gamma0)
        noise_b = noise_from_normals(zB, n_idx, sigma, Gamma0)
        pa, pb = simulate(KT, TAU, K, K, noise_a, noise_b, DT, ic_seed + n_idx)
        betaA[n_idx] = coherence_factor(pa, DT, f0=F0_GUESS)[0]
        betaB[n_idx] = coherence_factor(pb, DT, f0=F0_GUESS)[0]
    return betaA, betaB


def sigma_sweep_task(run, s_idx, K, Gamma0, nt):
    """
    One realization, one sigma.
    """
    rng = np.random.default_rng(run + 200)
    zA = rng.normal(0.0, 1.0, nt)
    zB = rng.normal(0.0, 1.0, nt)
    ic_seed = BASE_SEED + 1000 * run + 10 * s_idx
    return beta_all_noises(K, SIGMA_VALUES[s_idx], zA, zB, Gamma0, ic_seed)


def K_sweep_task(run, K, sigma, Gamma0, nt):
    """
    One realization, one K.
    """
    rng = np.random.default_rng([BASE_SEED, run])
    zA = rng.standard_normal(nt)
    zB = rng.standard_normal(nt)
    return beta_all_noises(K, sigma, zA, zB, Gamma0, BASE_SEED)


def save_results(out_dir, betaA, betaB):
    """
    betaA, betaB: arrays of shape (n_runs, n_values, 6), column 0 = white noise.
    Saves mean and std over realizations in the original file format.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "betaA_wn_power": betaA[:, :, 0].mean(axis=0),
        "betaA_cn_power": betaA[:, :, 1:].mean(axis=0),
        "betaB_wn_power": betaB[:, :, 0].mean(axis=0),
        "betaB_cn_power": betaB[:, :, 1:].mean(axis=0),
        "betaA_std_wn_power": betaA[:, :, 0].std(axis=0, ddof=1),
        "betaA_std_cn_power": betaA[:, :, 1:].std(axis=0, ddof=1),
        "betaB_std_wn_power": betaB[:, :, 0].std(axis=0, ddof=1),
        "betaB_std_cn_power": betaB[:, :, 1:].std(axis=0, ddof=1),
    }
    for name, arr in files.items():
        np.save(out_dir / f"{name}.npy", arr)


def run_sigma_sweep(K, Gamma0, nt, n_runs, n_jobs):
    tasks = [(run, s) for run in range(n_runs) for s in range(len(SIGMA_VALUES))]
    print(f"Sigma sweep at K = {K}: {len(tasks)} tasks of 6 simulations")
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(sigma_sweep_task)(run, s, K, Gamma0, nt) for run, s in tasks
    )
    betaA = np.array([o[0] for o in out]).reshape(n_runs, len(SIGMA_VALUES), 6)
    betaB = np.array([o[1] for o in out]).reshape(n_runs, len(SIGMA_VALUES), 6)
    out_dir = DATA_DIR / f"sigma_sweep_K{K:g}"
    save_results(out_dir, betaA, betaB)
    np.save(out_dir / "intensity.npy", SIGMA_VALUES)
    print(f"Saved to {out_dir}")


def run_K_sweep(K_values, sigma, Gamma0, nt, n_runs, n_jobs):
    tasks = [(run, K) for run in range(n_runs) for K in K_values]
    print(f"K sweep at sigma = {sigma}: {len(tasks)} tasks of 6 simulations")
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(K_sweep_task)(run, K, sigma, Gamma0, nt) for run, K in tasks
    )
    betaA = np.array([o[0] for o in out]).reshape(n_runs, len(K_values), 6)
    betaB = np.array([o[1] for o in out]).reshape(n_runs, len(K_values), 6)
    out_dir = DATA_DIR / f"K_sweep_sigma{sigma:g}"
    save_results(out_dir, betaA, betaB)
    np.save(out_dir / "K_values.npy", np.array(K_values))
    print(f"Saved to {out_dir}")



# Main
def load_gamma0(override):
    if override is not None:
        return override
    if GAMMA0_FILE.exists():
        with open(GAMMA0_FILE) as f:
            return json.load(f)["Gamma0"]
    return GAMMA0_DEFAULT


def main():
    parser = argparse.ArgumentParser(description="Compute the data for Fig. 7 (coherence factor, time-delay coupling).")
    parser.add_argument("--sweep", choices=["sigma", "K"], required=True,
                        help="sigma: beta vs noise amplitude (Fig. 7a, b); "
                             "K: beta vs heater power (Fig. 7c, d)")
    parser.add_argument("--K", type=float, nargs="+", default=None,
                        help=f"sigma sweep: heater power(s), one run each (default {K_SIGMA_SWEEP}); "
                             f"K sweep: K values (default {K_VALUES})")
    parser.add_argument("--sigma", type=float, default=SIGMA_K_SWEEP,
                        help=f"noise amplitude for the K sweep (default {SIGMA_K_SWEEP})")
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--t-max", type=float, default=T_MAX)
    parser.add_argument("--gamma0", type=float, default=None,
                        help="override the reference intensity Gamma_0")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    nt = int(args.t_max / DT)
    Gamma0 = load_gamma0(args.gamma0)
    print(f"Time steps: {nt}, Gamma_0 = {Gamma0:.6e}, realizations: {args.n_runs}")

    if args.sweep == "sigma":
        for K in (args.K or K_SIGMA_SWEEP):
            run_sigma_sweep(K, Gamma0, nt, args.n_runs, args.n_jobs)
    else:
        run_K_sweep(args.K or K_VALUES, args.sigma, Gamma0, nt, args.n_runs, args.n_jobs)


if __name__ == "__main__":
    main()
