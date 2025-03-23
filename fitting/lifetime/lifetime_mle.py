import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def fit_lifetime_mle(dt, distance, peak, diameter, plot=False):
    """
    Performs Maximum Likelihood Estimation (MLE) to fit an exponential decay function
    to photon arrival times (dt), weighted by a quadratic distance scheme.
    """
    valid_idx = dt > peak
    dt_valid = dt[valid_idx] - peak
    distance_valid = distance[valid_idx]

    radius = diameter / 2.0
    weights = 0.2 + (1 - (distance_valid / (radius ** 2)))

    sum_w = np.sum(weights)
    sum_wt = np.sum(weights * dt_valid)

    def neg_log_likelihood(lmbd):
        if lmbd <= 0:
            return np.inf
        return - (sum_w * np.log(lmbd) - lmbd * sum_wt)

    lambda_guess = 1.0 / np.mean(dt_valid) if np.mean(dt_valid) != 0 else 1.0

    result = minimize(
        neg_log_likelihood,
        x0=[lambda_guess],
        method='L-BFGS-B',
        bounds=[(1e-12, None)]
    )

    lambda_mle = result.x[0]
    tau_mle = 1.0 / lambda_mle

    if plot:
        bins = np.linspace(0, dt_valid.max(), 30)
        hist, edges = np.histogram(dt_valid, bins=bins, weights=weights)
        bin_centers = (edges[:-1] + edges[1:]) / 2

        fitted_curve = lambda_mle * np.exp(-lambda_mle * bin_centers) * np.sum(hist) * (bin_centers[1] - bin_centers[0])

        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0], alpha=0.6, label='Weighted Histogram')
        plt.plot(bin_centers, fitted_curve, 'r-', label=f'Exp Fit (tau={tau_mle:.1f})')
        plt.xlabel("Time since peak")
        plt.ylabel("Weighted count")
        plt.title("MLE Exponential Fit")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return tau_mle


# -----------------------------------------------
# Example usage
# -----------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    n_photons = 1000
    true_tau = 500  # True exponential decay lifetime
    peak_time = 200  # Start time of event
    diameter = 2.0

    # Simulate photon arrival times (exponential decay + peak shift)
    dt = np.random.exponential(scale=true_tau, size=n_photons) + peak_time

    # Simulate distances from the localization center (uniform for now)
    distance = np.random.uniform(0, 1, size=n_photons)

    # Run MLE estimation
    estimated_tau = fit_lifetime_mle(
        dt=dt,
        distance=distance,
        peak=peak_time,
        diameter=diameter,
        plot=True
    )

    print(f"True tau: {true_tau:.2f} ms")
    print(f"Estimated tau (MLE): {estimated_tau:.2f} ms")