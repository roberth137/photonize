import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp_decay(t, A, tau):
    """Exponential decay function without offset."""
    return A * np.exp(-t / tau)

def fit_lifetime_LQ(dt,
                         distance,
                         peak,
                         bin_size=150,
                         diameter=4.5,
                         base=1.0,
                         distance_weight=None,
                         plot=False):
    """
    Fits an exponential decay function to photon arrival times (dt),
    optionally using distance-based quadratic weights.

    Parameters
    ----------
    dt : np.ndarray
        Array of photon arrival times.
    distance : np.ndarray
        Array of distances of each photon from the localization center.
    peak : float
        Start time (only dt > peak are used).
    diameter : float
        Diameter of the localization region (used for distance weighting).
    bin_size : float, optional
        Bin size for the histogram (default is 150).
    base : float, optional
        Base weight added to each photon (default is 1.0).
    distance_weight : float or None, optional
        Weighting coefficient for distance. If None, no distance weighting is applied.
    plot : bool, optional
        If True, plot the weighted histogram and exponential fit.

    Returns
    -------
    tau : float
        The fitted decay lifetime.
    """
    # Select valid photons after the peak
    valid_idx = dt > peak
    dt_valid = dt[valid_idx] - peak
    distance_valid = distance[valid_idx]

    # Compute weights
    if distance_weight is not None:
        radius = diameter / 2.0
        weights = base + distance_weight * (1 - (distance_valid / (radius ** 2)))
    else:
        weights = np.ones_like(dt_valid) * base

    # Build weighted histogram
    max_dt = dt_valid.max()
    bins = np.arange(0, max_dt + bin_size, bin_size)
    hist, bin_edges = np.histogram(dt_valid, bins=bins, weights=weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Filter out zero bins
    nonzero = hist > 0
    xdata = bin_centers[nonzero]
    ydata = hist[nonzero]

    # Initial parameter guess
    A_guess = np.max(ydata)
    tau_guess = (xdata.max() - xdata.min()) / 2.0
    initial_guess = (A_guess, tau_guess)

    try:
        popt, pcov = curve_fit(exp_decay, xdata, ydata, p0=initial_guess)
    except RuntimeError as e:
        print("Fitting failed:", e)
        return 100.0  # Default tau if fit fails

    if plot:
        x_fit = np.linspace(0, xdata.max(), 300)
        y_fit = exp_decay(x_fit, *popt)

        plt.figure(figsize=(8, 5))
        plt.bar(xdata, ydata, width=bin_size * 0.9, alpha=0.6, label='Weighted Histogram')
        plt.plot(x_fit, y_fit, 'r-', label='Fitted Exponential')
        plt.xlabel('Time since peak')
        plt.ylabel('Weighted Count')
        plt.title('Exponential Fit to Arrival Times (Weighted)' if distance_weight else 'Unweighted Fit')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return popt[1]  # Return only tau

if __name__ == "__main__":
    # Simulate photon data
    np.random.seed(42)
    n_photons = 500
    true_tau = 280  # true decay lifetime in ms

    # Simulated arrival times after a peak (exponential decay)
    dt = np.random.exponential(scale=true_tau, size=n_photons)
    dt += 100  # simulate a peak at 100 ms

    # Simulated distances from event center (uniform in a circular region)
    distance = np.random.uniform(0, 1, size=n_photons)  # normalized

    # Call the function
    tau_fit = fit_lifetime_LQ(
        dt=dt,
        distance=distance,
        peak=200,
        diameter=2.0,
        bin_size=100,
        plot=True
    )

    print(f"Fitted lifetime (tau): {tau_fit:.2f} ms")