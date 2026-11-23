"""
Calibration of the reference white-noise intensity Gamma_0



Calibration criterion:
    The rms acoustic pressure induced by the reference white noise in an
    uncoupled Rijke tube at K = 0 equals the rms amplitude of the
    deterministic limit cycle at K = 0.8:

        P_rms^noise(K = 0) = P_rms^LCO(K = 0.8)

Noise convention used here:
    Additive white noise eps_0(t) with <eps_0(t) eps_0(t')> = Gamma_0 delta(t - t')
    is applied to every modal velocity equation. With the Euler-Maruyama scheme,
    each time step adds the increment

        dW_i = sqrt(Gamma_0 * dt) * N(0, 1)

    to every modal velocity etadot_j.

Procedure:
    1. Simulate the deterministic tube at K = 0.8 and compute P_rms^LCO.
    2. For each of n realizations, drive the tube at K = 0 with unit-intensity
       white noise (Gamma = 1) and compute the induced P_rms.
       At K = 0 the system is linear, so P_rms scales as sqrt(Gamma), and the
       amplitude factor needed to hit the target is  c = P_rms^LCO / P_rms.
    3. Gamma_0 = (mean of c)^2.
    4. Verify with a fresh realization at Gamma_0.

Output:
    data/noise_calibration/gamma0.json  (read by the figure scripts)

"""

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numba import njit



# Parameters

T_MAX = 10000.0       # total simulation time (non-dimensional)
DT = 0.001            # integration time step
TRANSIENT_FRAC = 0.8  # P'_rms computed over the final 20% of the signal

N = 10                # number of Galerkin modes
gamma = 1.4           # specific heat ratio
Ma = 0.01             # Mach number
tf = 0.2              # thermal time lag
xf = 0.25             # heater location
C1, C2 = 0.1, 0.06    # damping coefficients

K_LCO = 0.8           # heater power of the reference limit cycle
N_REALIZATIONS = 200  # number of noise realizations for the calibration
SIGMA_CHECK = [0.05, 0.1, 0.2, 0.4]  # noise amplitudes used in the paper
BASE_SEED = 2024      # base random seed

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "noise_calibration"



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
phi = np.cos(np.pi * modes * xf)
modal_coeff = gamma * Ma * np.sin(np.pi * modes * xf) / (modes * np.pi)


@njit
def rhs(eta, etadot, zeta_j, j, qdot, K):
    """Right-hand side of the modal acceleration equation, Eq. (5)."""
    return (-2 * zeta_j * j * np.pi * etadot
            - (j * np.pi) ** 2 * eta
            - np.pi * j * K * np.sin(j * np.pi * xf) * qdot)


@njit
def simulate(K, dW, dt, seed):
    """
    Integrate a single Rijke tube with RK4 for the deterministic part and
    Euler-Maruyama for the additive noise.

    dW[i] is the noise increment added to every modal velocity at step i
    (pass an array of zeros for the deterministic case).
    Returns the acoustic pressure P'(t).
    """
    np.random.seed(seed)
    nt = dW.shape[0]

    eta = np.zeros((N, nt))
    etadot = np.zeros((N, nt))
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

        qdot = heat_release(np.sum(eta_delay * phi))

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
            etadot[j, i] = (v0 + dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
                            + dW[i])

    p = np.zeros(nt)
    for j in range(N):
        p -= modal_coeff[j] * etadot[j, :]
    return p


def p_rms(p):
    """RMS pressure over the final (1 - TRANSIENT_FRAC) of the signal."""
    return np.sqrt(np.mean(p[int(TRANSIENT_FRAC * len(p)):] ** 2))


def white_increments(Gamma, nt, dt, rng):
    """Euler-Maruyama increments of white noise with intensity Gamma."""
    return np.sqrt(Gamma * dt) * rng.standard_normal(nt)



# Calibration steps

def lco_rms(nt):
    """Step 1: deterministic limit-cycle amplitude at K = 0.8."""
    return p_rms(simulate(K_LCO, np.zeros(nt), DT, BASE_SEED))


def calibration_factor(target_rms, nt, idx):
    """Step 2: amplitude factor for one unit-intensity noise realization."""
    rng = np.random.default_rng(BASE_SEED + 1 + idx)
    dW = white_increments(1.0, nt, DT, rng)
    rms_unit = p_rms(simulate(0.0, dW, DT, BASE_SEED + 1 + idx))
    return target_rms / rms_unit, rms_unit


def main():
    parser = argparse.ArgumentParser(description="Calibrate the reference white-noise intensity Gamma_0.")
    parser.add_argument("--t-max", type=float, default=T_MAX,
                        help=f"simulation time (default {T_MAX})")
    parser.add_argument("--n-real", type=int, default=N_REALIZATIONS,
                        help=f"number of noise realizations (default {N_REALIZATIONS})")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="number of parallel jobs (default: all cores)")
    args = parser.parse_args()

    nt = int(args.t_max / DT)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Number of time steps: {nt}")

    # Step 1
    target = lco_rms(nt)
    print(f"LCO P_rms at K = {K_LCO}: {target:.6e}")

    # Step 2
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(calibration_factor)(target, nt, idx) for idx in range(args.n_real)
    )
    factors = np.array([r[0] for r in results])
    mean_factor = factors.mean()
    std_factor = factors.std(ddof=1) if len(factors) > 1 else 0.0
    print(f"Calibration factor: mean = {mean_factor:.6f}, std = {std_factor:.6f}")

    # Step 3
    Gamma0 = mean_factor ** 2
    increment_std = np.sqrt(Gamma0 * DT)
    print(f"Reference white-noise intensity Gamma_0 = {Gamma0:.6e}")
    print(f"Per-step increment std sqrt(Gamma_0 dt) = {increment_std:.6e}")

    # Step 4: verification and response at the paper's noise amplitudes
    rng = np.random.default_rng(BASE_SEED + 100_000)
    unit = rng.standard_normal(nt)
    check = {}
    for sigma in [1.0] + SIGMA_CHECK:
        dW = sigma * np.sqrt(Gamma0 * DT) * unit
        rms = p_rms(simulate(0.0, dW, DT, BASE_SEED + 100_000))
        check[str(sigma)] = rms
        print(f"sigma = {sigma:<5}: noise-induced P_rms(K=0) = {rms:.6e} "
              f"(ratio to LCO = {rms / target:.4f})")

    out = dict(
        Gamma0=Gamma0,
        increment_std=increment_std,
        dt=DT,
        t_max=args.t_max,
        n_realizations=args.n_real,
        lco_rms_K08=target,
        calibration_factor_mean=mean_factor,
        calibration_factor_std=std_factor,
        verification_rms_by_sigma=check,
        base_seed=BASE_SEED,
    )
    out_file = DATA_DIR / "gamma0.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved calibration to {out_file}")


if __name__ == "__main__":
    main()
