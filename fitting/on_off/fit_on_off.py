import numpy as np
import ruptures as rpt
from ruptures.exceptions import BadSegmentationParameters


def get_on_off_dur(photons, bin_size_ms=10, smoothing_size=5, jump=1):
    """
    Estimate the start, end, and duration of an event based on photon arrival times,
    using histogram binning, smoothing, and change point detection.

    Parameters
    ----------
    photons : object with attribute 'ms'
        Photon data, where `photons.ms` is an array-like of photon arrival times in milliseconds.
    bin_size_ms : int or float
        Bin width for creating the time histogram (in ms).
    smoothing_size : int
        Window size for Lee filter smoothing (must be odd).

    Returns
    -------
    start : float
        Estimated start time of the event (in ms).
    end : float
        Estimated end time of the event (in ms).
    duration : float
        Estimated duration of the event (in ms).

    Notes
    -----
    - Uses the 'Binseg' algorithm from the `ruptures` package with the L2 model
      to find two change points corresponding to "on" and "off" transitions.
    - Falls back to min/max of photon times if segmentation fails.
    """
    bins = np.arange(min(photons.ms), max(photons.ms) + bin_size_ms, bin_size_ms)
    counts, _ = np.histogram(photons.ms, bins=bins)
    smoothed_counts = lee_filter_1d(counts, smoothing_size)

    model = "l2"
    algo = rpt.Binseg(model=model, min_size=1, jump=jump).fit(smoothed_counts)
    try:
        change_points = np.array(algo.predict(n_bkps=2))
        #change_points_trans = np.array(change_points)

        start = (change_points[0] - 1.5) * bin_size_ms + bins[0]
        end = (change_points[1] + 0.5) * bin_size_ms + bins[0]
        duration = (change_points[1] - change_points[0]) * bin_size_ms
    except BadSegmentationParameters:
        start = min(photons.ms)
        end = max(photons.ms)
        duration = end - start

    return start, end, duration


def lee_filter_1d(data, window_size=5):
    """
    Applies the Lee filter to 1D data for noise reduction.

    Parameters
    ----------
    data : numpy.ndarray
        1D array of data to filter.
    window_size : int
        Size of the sliding window (must be odd).

    Returns
    -------
    numpy.ndarray
        Smoothed data after applying the Lee filter.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    padded_data = np.pad(data, pad_width=window_size // 2, mode='reflect')
    local_mean = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    local_var = np.convolve(padded_data ** 2, np.ones(window_size) / window_size, mode='valid') - local_mean ** 2
    noise_var = np.mean(local_var)

    result = local_mean + (local_var / (local_var + noise_var)) * (data - local_mean)
    return result

def simulate_event_ms_trace(seed=42, bg_rate=3, diameter=4.5, dur_all=600, dur_eve=300, brightness_eve=1.5):
    np.random.seed(seed)
    area = (diameter / 2) * np.pi
    bg_photon_rate = bg_rate * area  # photons / 200ms / single_pixel * area_in_pixel

    # 1. Simulate Background
    n_bg_expected = int((dur_all / 200) * bg_photon_rate)
    bg_arrival_times = np.sort(np.random.uniform(0, dur_all, size=n_bg_expected))
    # 2. Simulate Event
    event_start = (dur_all - dur_eve) / 2
    event_end = event_start + dur_eve

    n_eve_expected = int(dur_eve * brightness_eve)
    event_arrival_times = np.sort(np.random.uniform(event_start, event_end, size=n_eve_expected))

    all_arrival_times = np.sort(np.concatenate([bg_arrival_times, event_arrival_times]))

    return all_arrival_times

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    class PhotonMock:
        """Mock photon object with .ms attribute"""
        def __init__(self, ms_array):
            self.ms = ms_array

    bin_size_ms = 10

    all_arrival_times = simulate_event_ms_trace()

    photons = PhotonMock(ms_array=all_arrival_times)

    start, end, duration = get_on_off_dur(photons, bin_size_ms=bin_size_ms, smoothing_size=5)

    print(f"Estimated ON time: {start:.1f} ms")
    print(f"Estimated OFF time: {end:.1f} ms")
    print(f"Estimated duration: {duration:.1f} ms")

    # Optional plot for visual inspection
    bins = np.arange(min(photons.ms), max(photons.ms) + bin_size_ms , bin_size_ms)
    counts, _ = np.histogram(photons.ms, bins=bins)
    smoothed = lee_filter_1d(counts, 5)

    plt.figure(figsize=(8, 5))
    plt.bar(bins[:-1], counts, width=bin_size_ms, label='Original Histogram', alpha=0.5)
    plt.bar(bins[:-1], smoothed, width=bin_size_ms, label='Smoothed (Lee Filter)', alpha=0.5)
    plt.axvline(start, color='green', linestyle='--', label='Start (ON)')
    plt.axvline(end, color='red', linestyle='--', label='End (OFF)')
    plt.title("Photon Arrival Histogram with Change Point Detection")
    plt.xlabel("Time (ms)")
    plt.ylabel("Photon Count")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
