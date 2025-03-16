import numpy as np
import pandas as pd
import get_photons
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from numba import njit


def avg_lifetime(loc_photons, peak, dt_offset=0):
    '''
    Returns the average lifetime value, starting at the peak
    '''
    arr_times = np.copy(loc_photons.dt)
    arr_times_normalized = arr_times[arr_times > peak]
    lifetime = np.mean(arr_times_normalized-peak)
    return lifetime

#@njit
def avg_lifetime_weighted(dt, distance, peak, diameter):
    """
    Uses quadratic weights for lifetimes depending on photons' distance from the center of localization.

    Parameters
    ----------
    dt : np.ndarray
        Time difference array (photon timestamps).
    distance : np.ndarray
        Distance of each photon from the center of the localization.
    peak : int
        The peak time after which the lifetime calculation starts.
    diameter : float
        The diameter of the localization region.

    Returns
    -------
    float32
        The weighted average lifetime of the photons.
    """
    radius = diameter / 2
    n = len(dt)

    weighted_sum = 0.0
    weight_total = 0.0

    for i in range(n):
        if dt[i] > peak:  # Only consider after the peak
            ap_dt = dt[i] - peak
            ap_weight = 0.2 + (1 - (distance[i] / (radius ** 2)))  # Quadratic weighting with 0.2 offset
            weighted_sum += ap_dt * ap_weight
            weight_total += ap_weight

    lifetime = weighted_sum / weight_total if weight_total > 0 else 0.0
    return np.float32(lifetime)


def exp_decay(t, A, tau):
    """Exponential decay function."""
    return A * np.exp(-t / tau)


def fit_weighted_exponential(dt, distance, peak, diameter, bin_size=150, plot=False):
    """
    Fits an exponential decay function to arrival time values (dt) weighted by a
    quadratic distance weighting scheme. Only dt values > peak are used (with peak subtracted).

    Parameters
    ----------
    dt : np.ndarray
        Array of photon arrival times.
    distance : np.ndarray
        Array of distances for each photon from the center of localization.
    peak : float or int
        The start time (peak) after which dt values are considered.
    diameter : float
        The diameter of the localization region (used to compute weights).
    bin_size : float, optional
        Bin size for the weighted histogram (default is 20).
    plot : bool, optional
        If True, plots the weighted histogram and the fitted exponential.

    Returns
    -------
    popt : tuple
        Optimal parameters (A, tau, offset) of the exponential decay.
    pcov : 2D array
        The estimated covariance of popt.
    """
    # Only consider dt values greater than the peak and shift them to start at zero
    valid_idx = dt > peak
    dt_valid = dt[valid_idx] - peak
    distance_valid = distance[valid_idx]

    # Compute weights using quadratic weighting with a 0.2 offset.
    # radius is half the diameter.
    radius = diameter / 2.0
    weights = 0.2 + (1 - (distance_valid / (radius ** 2)))

    # Create a weighted histogram of dt values
    max_dt = dt_valid.max()
    bins = np.arange(0, max_dt + bin_size, bin_size)
    hist, bin_edges = np.histogram(dt_valid, bins=bins, weights=weights)

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Remove bins with zero count to avoid issues in fitting
    nonzero = hist > 0
    xdata = bin_centers[nonzero]
    ydata = hist[nonzero]

    # Initial guess for parameters:
    # A_guess ~ maximum weighted count,
    # tau_guess ~ half the range of the time bins,
    # offset_guess ~ minimum weighted count.
    A_guess = np.max(ydata)
    tau_guess = (xdata.max() - xdata.min()) / 2.0
    initial_guess = (A_guess, tau_guess)

    # Fit the exponential function to the weighted histogram data.
    try:
        popt, pcov = curve_fit(exp_decay, xdata, ydata, p0=initial_guess)

    except RuntimeError as e:
        print("Fitting failed:", e)
        # Return some defaults, e.g. tau=100
        popt = [0, 100, 0]
        pcov = np.zeros((3, 3))

    if plot:
        # Generate fitted curve for a smooth plot.
        x_fit = np.linspace(0, xdata.max(), 300)
        y_fit = exp_decay(x_fit, *popt)

        plt.figure(figsize=(8, 5))
        plt.bar(xdata, ydata, width=bin_size * 0.9, alpha=0.6, label='Weighted Histogram')
        plt.plot(x_fit, y_fit, 'r-', label='Fitted Exponential')
        plt.xlabel('Time since peak')
        plt.ylabel('Weighted Count')
        plt.title('Weighted Exponential Fit to Arrival Times')
        plt.legend()
        plt.show()

    return popt[1]#, pcov


def mle_exponential_lifetime(dt, distance, peak, diameter, plot=False):
    """
    Performs Maximum Likelihood Estimation (MLE) to fit an exponential decay function
    to photon arrival times (dt), weighted by a quadratic distance scheme.

    Parameters
    ----------
    dt : np.ndarray
        Array of photon arrival times.
    distance : np.ndarray
        Array of distances for each photon from the center of localization.
    peak : float or int
        The start time (peak) after which dt values are considered.
    diameter : float
        The diameter of the localization region (used to compute weights).
    plot : bool, optional
        If True, plots the histogram of arrival times with the MLE-estimated exponential decay.

    Returns
    -------
    tau_mle : float
        Estimated exponential decay lifetime using MLE.
    """
    # Only consider dt values greater than the peak and shift them to start at zero
    valid_idx = dt > peak
    dt_valid = dt[valid_idx] - peak
    distance_valid = distance[valid_idx]

    # Compute weights using quadratic weighting with a 0.2 offset.
    radius = diameter / 2.0
    weights = 0.2 + (1 - (distance_valid / (radius ** 2)))

    # Precompute sums for efficiency
    sum_w = np.sum(weights)
    sum_wt = np.sum(weights * dt_valid)

    # Negative log-likelihood (NLL) for the weighted exponential distribution
    #
    # For each data point i:
    #   log(f_i) = log(lambda) - lambda * t_i
    # Weighted sum: sum_i w_i * log(f_i)
    # NLL = - sum_i w_i * log(f_i)
    # => NLL = - sum_i w_i [ log(lambda) - lambda * t_i ]
    # => NLL = - ( sum_i w_i ) log(lambda ) + lambda * sum_i w_i t_i
    def neg_log_likelihood(lmbd):
        # Ensure lambda > 0
        if lmbd <= 0:
            return np.inf
        return - (sum_w * np.log(lmbd) - lmbd * sum_wt)

    # Initial guess: lambda ~ 1 / mean(dt)
    if np.mean(dt_valid) == 0:
        # fallback in case all dt_valid are 0 (unlikely but safe)
        lambda_guess = 1.0
    else:
        lambda_guess = 1.0 / np.mean(dt_valid)

    # Use scipy's 'minimize' with a bound to ensure lambda > 0
    result = minimize(
        neg_log_likelihood,
        x0=[lambda_guess],
        method='L-BFGS-B',
        bounds=[(1e-12, None)]
    )

    # Extract the MLE estimate of lambda
    lambda_mle = result.x[0]
    tau_mle = 1.0 / lambda_mle

    return tau_mle


def calibrate_peak_events(event_photons):
    '''
    Parameters
    ----------
    All photons of the current fov that arrive during events
    Returns
    -------
    Position of arrival time histogram peak
    '''
    counts, bins = np.histogram(event_photons.dt, bins=np.arange(0, 2500))
    print('len photons for calib_peak: ', len(event_photons))
    return np.argmax(counts)


if __name__ == "__main__":
    # Generate synthetic dt and distance data for demonstration.
    np.random.seed(0)
    n = 1000
    # Simulate arrival times (exponentially distributed) in arbitrary units
    dt = np.random.exponential(scale=100, size=n)
    # Simulate distances uniformly (in same arbitrary units)
    distance = np.random.uniform(0, 50, size=n)

    # Define the 'peak' time and diameter of the localization region.
    peak = 50
    diameter = 100  # arbitrary units

    # Fit the weighted exponential; use default bin size of 20 and plot the result.
    popt, pcov = fit_weighted_exponential(dt, distance, peak, diameter, bin_size=20, plot=True)

    print("Fitted parameters:")
    print("Amplitude (A):", popt[0])
    print("Tau:", popt[1])
    print("Offset:", popt[2])