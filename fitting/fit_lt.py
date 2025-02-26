import numpy as np
import pandas as pd
import get_photons
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit



def avg_lifetime_sergi_40(loc_photons, peak, dt_offset=0):
    '''
    Fit lifetimes of individual localizations with 40mhz laser frequency
    Parameters
    ----------
    loc_photons : all photons from one localization
    peak : position of the maximum of arrival times for this pick of
    localizations, calibrated from calibrate_peak_locs()
    dt_offset : the offset from the peak where arrival times are considered
    for fitting the lifetime.The default is 50.

    Returns
    -------
    average arrival time of photons, in units of arrival time bin size.
    Usually 10ps

    '''

    counts, bins = np.histogram(loc_photons.dt, bins=np.arange(0, 2500))
    background = np.sum(counts[-300:]) / 300
    counts_bgsub = counts - background
    weights = np.arange(1, (2500 - (peak + dt_offset)))
    considered_bgsub = counts_bgsub[(peak + dt_offset):2500]
    #if len(loc_photons) < 70:
    #    print('\nphotons for fitting: ', len(loc_photons))
    #    print('good photons: ', sum(considered_bgsub))
    lifetime = np.sum(np.multiply(considered_bgsub, weights)) / np.sum(considered_bgsub)
    return lifetime

def avg_lifetime_no_bg_40(loc_photons, peak, dt_offset=0):
    '''
    Not considering bg
    '''

    #counts, bins = np.histogram(loc_photons.dt, bins=np.arange(0, 2500))
    #counts_bgsub = counts - background
    #weights = np.arange(1, (2500 - (peak + dt_offset)))
    #considered_bgsub = counts_bgsub[(peak + dt_offset):2500]
    #if len(loc_photons) < 70:
    #    print('\nphotons for fitting: ', len(loc_photons))
    #    print('good photons: ', sum(considered_bgsub))
    #lifetime = np.sum(np.multiply(considered_bgsub, weights)) / np.sum(considered_bgsub)
    arr_times = np.copy(loc_photons.dt)
    arr_times_normalized = arr_times[arr_times > peak]
    lifetime = np.mean(arr_times_normalized-peak)
    return lifetime

def avg_lifetime_gauss_w_40(loc_photons, peak, diameter, sigma=1 ,dt_offset=0):
    '''
    use weights for fluorophores position
    '''
    radius = diameter/2
    after_peak = loc_photons[loc_photons.dt>peak]
    ap_dt = np.copy(after_peak.dt)
    ap_dt = ap_dt.astype('int64')
    ap_dist = np.copy(after_peak.distance)
    ap_dt -= peak
    ap_weight = np.exp(-(ap_dist/sigma**2))
    weighted_dt = np.multiply(ap_dt,ap_weight)
    lifetime = np.sum(weighted_dt)/np.sum(ap_weight)
    return lifetime

def mean_arrival(loc_photons, diameter):
    return np.mean(loc_photons.dt)

#@njit
def avg_lifetime_weighted_40(dt, distance, peak, diameter):
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


def exp_decay(t, A, tau, offset):
    """Exponential decay function."""
    return A * np.exp(-t / tau) + offset


def fit_weighted_exponential(dt, distance, peak, diameter, bin_size=100, plot=False):
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
    offset_guess = np.min(ydata)
    initial_guess = (A_guess, tau_guess, offset_guess)

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

def avg_lifetime_weighted_40_old(loc_photons, peak, diameter):
    '''
    uses quadratic weights for lifetimes depending on photons distance from center of localization
    '''
    radius = diameter/2
    after_peak = loc_photons[loc_photons.dt>peak]
    ap_dt = np.copy(after_peak.dt)
    ap_dt = ap_dt.astype('int64')
    ap_dist = np.copy(after_peak.distance)
    ap_dt -= peak
    ap_weight = (0.2+(1-ap_dist/(radius**2)))
    weighted_dt = np.multiply(ap_dt,ap_weight)
    lifetime = np.sum(weighted_dt)/np.sum(ap_weight)
    return lifetime.astype(np.float32)


def avg_lifetime_sergi_80(loc_photons, peak, dt_offset=50):
    '''
    Fit lifetimes of individual localizations with 80mhz laser frequency
    Parameters
    ----------
    loc_photons : all photons from one localization
    peak : position of the maximum of arrival times for this pick of
    localizations, calibrated from calibrate_peak_locs()
    dt_offset : the offset from the peak where arrival times are considered
    for fitting the lifetime.The default is 50.

    Returns
    -------
    average arrival time of photons, in units of arrival time bin size.
    Usually 10ps

    '''
    counts, bins = np.histogram(loc_photons.dt, bins=np.arange(0, 1250))
    background = np.sum(counts[-300:]) / 300
    counts_bgsub = counts - background
    weights = np.arange(1, (1250 - (peak + dt_offset)))
    considered_bgsub = counts_bgsub[(peak + dt_offset):1250]
    lifetime = np.sum(np.multiply(considered_bgsub, weights)) / np.sum(considered_bgsub)
    return lifetime

def calibrate_peak_locs(locs_group, pick_photons, offset,
                   box_side_length, int_time):
    '''
    Parameters
    ----------
    locs_group : localizations of this pick as pd dataframe
    pick_photons : photons of this pick as pd dataframe
    offset : how many offsetted frames
    Returns
    -------
    Position of arrival time histogram peak
    '''
    group_photons = pd.DataFrame()
    for i in range(len(locs_group)):
        phot_loc = get_photons.photons_of_one_localization(locs_group.iloc[i], pick_photons, offset,
                                                           box_side_length, int_time)
        group_photons = pd.concat([group_photons, phot_loc],
                                  ignore_index=True)
    counts, bins = np.histogram(group_photons.dt, bins=np.arange(0, 2500))
    print('len photons for calib_peak: ', len(group_photons))
    return np.argmax(counts)

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