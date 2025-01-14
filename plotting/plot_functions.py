import matplotlib.pyplot as plt
import numpy as np
import get_photons
import fitting
import ruptures as rpt
from scipy.ndimage import gaussian_filter
from plotting import group_events, all_events_photons, diameter

def hist_ms_event(i):
    this_event = group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event,
                                                all_events_photons,
                                                diameter,
                                                200)
    bin_size = 10
    bins = np.arange(min(this_event_photons.ms), max(this_event_photons.ms) + bin_size, bin_size)
    counts, _ = np.histogram(this_event_photons, bins=bins)
    #smoothed_counts_0 = lee_filter_1d(counts, 5)
    smoothed_counts_1 = lee_filter_1d(counts, 5)
    smoothed_counts_2 = lee_filter_1d(counts, 7)
    bg = np.sum((smoothed_counts_1[:(int(200/bin_size))])
                + np.sum(smoothed_counts_1[:-(int(200/bin_size))])/400)
    smoothed_counts_3 = lee_filter_1d(counts, 11)
    plt.figure(figsize=(8, 6))
    #plt.bar(bins[:-1], smoothed_counts_0, width=bin_size, color='red', alpha=0.5)
    plt.bar(bins[:-1], smoothed_counts_1, width=bin_size, color='blue', alpha=0.5)
    plt.bar(bins[:-1], smoothed_counts_2, width=bin_size, color='orange', alpha=0.5)

    # Fit a step function using change point detection
    model = "l2"  # Least squares cost function
    algo = rpt.Binseg(model=model).fit(smoothed_counts_1)
    change_points = algo.predict(n_bkps=2)  # Detect 2 change points (for on and off)

    change_points_trans = [(x * bin_size + bins[0]) for x in change_points]
    print('change points raw: ', change_points)
    print('change points transf: ', change_points_trans)

    start, end, threshold = detect_signal_threshold(smoothed_counts_1, bg, bin_size)#
    start_after = (start*bin_size)+bins[0]
    end_after = (end*bin_size)+bins[0]

    #plt.plot([], [], ' ', label=f'Total number of photons: {len(this_event_photons)}')
    plt.plot([], [], ' ', label=f'dist_frames:: {this_event.end_ms-this_event.start_ms}')
    plt.plot([], [], ' ', label=f'dist_thresh: {end_after-start_after}')
    plt.plot([], [], ' ', label=f'dist_rupt: {change_points_trans[1]-change_points_trans[0]}')
    plt.plot([], [], ' ', label=f'num_frames: {this_event.num_frames}')
    plt.plot([], [], ' ', label=f'Lifetime: {this_event.lifetime}')
    plt.axvline(this_event.start_ms, color='red')
    plt.axvline(this_event.end_ms, color='red')
    plt.axvline(change_points_trans[0], color='green')
    plt.axvline(change_points_trans[1], color='green')
    plt.axvline(start_after, color='magenta')
    plt.axvline(end_after, color='magenta')
    plt.axhline(threshold, color='blue')
    plt.axhline(5, color='green')
    plt.title("Histogram of ms")
    plt.xlabel("ms of arrival")
    # plt.ylabel("Y Position")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')  # Adjust the legend position if needed
    plt.show()


def detect_signal_threshold(data, bg, bin_size):
    # Calculate basic statistics
    mean = np.mean(data[int((200/bin_size)):-int((200/bin_size))])
    threshold = mean/2

    # Vectorized approach to find the start and end points
    above_threshold = data > threshold
    start = np.argmax(above_threshold)
    end_candidates = np.where(above_threshold[start:] == False)[0]
    end = start + end_candidates[0] if len(end_candidates) > 0 else len(data) - 1

    return start, end, threshold


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

def plot_all_dt(all_events_photons):
    bin_size = 10
    bins = np.arange(min(all_events_photons.dt), max(all_events_photons.dt) + bin_size, bin_size)
    plt.figure(figsize=(8, 6))
    plt.hist(all_events_photons['dt'], bins=bins)
    plt.plot([], [], ' ', label=f'Total number of photons: {len(all_events_photons)}')
    plt.title("Histogram of dt values")
    plt.xlabel("arrival time ")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')  # Adjust the legend position if needed
    plt.show()


def scatter_event(i):
    this_event = group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event, all_events_photons, diameter)
    print(this_event_photons)

    prev_x = this_event.x
    prev_y = this_event.y
    new_x, new_y = fitting.event_position_w_bg(this_event, this_event_photons, diameter, False)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(this_event_photons['x'],
                          this_event_photons['y'],
                          c=this_event_photons['ms'],
                          cmap='viridis')
    cbar = plt.colorbar(scatter)
    cbar.set_label('ms value', rotation=270, labelpad=15)
    plt.plot(prev_x, prev_y, 'o', label=f'Prev: ({prev_x}, {prev_y})', color='blue')
    plt.plot(new_x, new_y, '^', label=f'New Pos: ({new_x}, {new_y})', color='red')
    plt.plot([], [], ' ', label=f'Total number of photons: {len(this_event_photons)}')
    plt.title("Scatter Plot of x, y Positions from DataFrame")
    plt.title('Data Points with Legend')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.yscale("log")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')  # Adjust the legend position if needed
    plt.show()


def hist_dt_event(i):
    this_event = group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event, all_events_photons, diameter)

    bin_size = 5
    bins = np.arange(min(this_event_photons.dt), max(this_event_photons.dt) + bin_size, bin_size)
    plt.figure(figsize=(8, 6))
    plt.hist(this_event_photons['dt'], bins=bins)
    plt.plot([], [], ' ', label=f'Total number of photons: {len(this_event_photons)}')
    plt.title("Histogram of dt values")
    plt.xlabel("arrival time ")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')  # Adjust the legend position if needed
    plt.show()


def hist_x_event(i):
    this_event = group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event, all_events_photons, diameter)
    print(this_event_photons)
    bin_size = 0.05
    bins = np.arange(min(this_event_photons.x), max(this_event_photons.x) + bin_size, bin_size)
    plt.figure(figsize=(8, 6))
    plt.hist(this_event_photons['x'],
             bins=bins)
    plt.title("Histogram of x position")
    plt.xlabel("X Position")
    # plt.ylabel("Y Position")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()