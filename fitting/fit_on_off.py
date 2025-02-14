import numpy as np
import ruptures as rpt
from ruptures.exceptions import BadSegmentationParameters


def get_on_off_dur(photons, bin_size, smoothing_size):
    bins = np.arange(min(photons.ms), max(photons.ms) + bin_size, bin_size)
    counts, _ = np.histogram(photons, bins=bins)
    smoothed_counts_1 = lee_filter_1d(counts, smoothing_size)
    model = "l2"  # Least squares cost function
    algo = rpt.Binseg(model=model, min_size=1, jump=1).fit(smoothed_counts_1)
    try:
        change_points = algo.predict(n_bkps=2)  # Detect 2 change points (for on and off)
        change_points_trans = np.array(change_points)
        start = (change_points_trans[0] - 1.5) * bin_size + bins[0]
        end = (change_points_trans[1] + 0.5) * bin_size + bins[0]
        duration = (change_points_trans[1] - change_points_trans[0]) * bin_size
    except BadSegmentationParameters:
        # Handle the case where there aren't enough data points
        start = min(photons.ms)
        end = max(photons.ms)
        duration = end - start

    return start, end, duration

def lee_filter_1d(data, window_size=5):
    """
    Applies the Lee filter to 1D data for noise reduction.

    Parameters:
        data (numpy.ndarray): 1D array of data to filter.
        window_size (int): Size of the sliding window (must be odd).

    Returns:
        numpy.ndarray: Smoothed data after applying the Lee filter.
    """
    # Ensure the window size is odd
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    # Calculate the local mean and variance in the sliding window
    padded_data = np.pad(data, pad_width=window_size // 2, mode='reflect')
    local_mean = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    local_var = np.convolve(padded_data ** 2, np.ones(window_size) / window_size, mode='valid') - local_mean ** 2

    # Estimate the noise variance (assume it's uniform across the data)
    noise_var = np.mean(local_var)

    # Apply the Lee filter
    result = local_mean + (local_var / (local_var + noise_var)) * (data - local_mean)
    return result