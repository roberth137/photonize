import matplotlib.pyplot as plt
import numpy as np
import get_photons
import fitting
import ruptures as rpt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import plotting
from plotting import group_events, pick_photons, diameter, peak_arrival_time


def hist_ms_event(i):
    this_event = group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event,
                                                pick_photons,
                                                diameter,
                                                400)
    bin_size = 10
    bins = np.arange(min(this_event_photons.ms), max(this_event_photons.ms) + bin_size, bin_size)
    counts, _ = np.histogram(this_event_photons, bins=bins)
    smoothed_counts_2 = lee_filter_1d(counts, 5)
    plt.figure(figsize=(8, 6))
    #plt.bar(bins[:-1], smoothed_counts_0, width=bin_size, color='red', alpha=0.5)
    plt.bar(bins[:-1], counts, width=bin_size, color='blue', alpha=0.5)
    plt.bar(bins[:-1], smoothed_counts_2, width=bin_size, color='orange', alpha=0.5)

    # Fit a step function using change point detection
    model = "l2"  # Least squares cost function
    algo = rpt.Binseg(model=model, min_size=1, jump=1).fit(smoothed_counts_2)
    change_points = algo.predict(n_bkps=2)  # Detect 2 change points (for on and off)
    change_points_trans = np.array(change_points)
    change_points_trans[0] = (change_points_trans[0] - 1.5)*bin_size + bins[0]
    change_points_trans[1] = (change_points_trans[1] + 0.5)*bin_size + bins[0]


    #plt.plot([], [], ' ', label=f'Total number of photons: {len(this_event_photons)}')
    plt.plot([], [], ' ', label=f'duration_ms: {this_event.end_ms-this_event.start_ms}')
    plt.plot([], [], ' ', label=f'number_photons: {len(this_event_photons)}')
    #plt.plot([], [], ' ', label=f'brightness: {this_event.brightness:.3f}')
    plt.plot([], [], ' ', label=f'Lifetime: {this_event.lifetime:.3f}')
    plt.axvline(this_event.start_ms+30, color='red')
    plt.axvline(this_event.end_ms-30, color='red')
    #plt.axvline(change_points_trans[0], color='green')
    #plt.axvline(change_points_trans[1], color='green')
    plt.title("Histogram of ms")
    plt.xlabel("ms of arrival")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')  # Adjust the legend position if needed
    plt.show()

def hist_noise_dt_event(i):
    this_event = group_events.iloc[i]

    more_ms = 400
    this_event_photons = get_photons.crop_event(this_event,
                                                pick_photons,
                                                diameter,
                                                more_ms)
    bounds_first = [(this_event.start_ms_fr - more_ms), this_event.s_ms_new]
    bounds_second = [this_event.e_ms_new, (this_event.end_ms_fr + more_ms)]

    filtered_photons = this_event_photons[
        ((this_event_photons['ms'] >= bounds_first[0])
        & (this_event_photons['ms'] <= bounds_first[1]))
        | ((this_event_photons['ms'] >= bounds_second[0])
        & (this_event_photons['ms'] <= bounds_second[1]))
    ]
    bg_lt = np.mean(filtered_photons.dt)-peak_arrival_time
    bin_size = 5
    bins = np.arange(min(this_event_photons.dt), max(this_event_photons.dt) + bin_size, bin_size)

    # Main plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(this_event_photons['dt'], bins=bins, alpha=0.5, color='orange', label="All photons")
    ax.hist(filtered_photons['dt'], bins=bins, alpha=1, color='blue', label="Filtered photons")
    ax.plot([], [], ' ', label=f'Event photons: {len(this_event_photons)-len(filtered_photons)}')
    ax.plot([], [], ' ', label=f'Bg photons: {len(filtered_photons)}')
    ax.plot([], [], ' ', label=f'Eve_lt: {this_event.lifetime}, BG_lt: {bg_lt}')
    ax.axvline(plotting.peak_arrival_time, color='red', label="Peak arrival time")
    ax.set_title("Histogram of dt values")
    ax.set_xlabel("arrival time")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')

    # Inset plot for hist_ms_event
    inset_ax = inset_axes(ax, width="30%", height="30%", loc="upper right")  # Adjust size and position
    bins_ms = np.arange(min(this_event_photons.ms), max(this_event_photons.ms) + 10, 10)
    counts, _ = np.histogram(this_event_photons.ms, bins=bins_ms)
    inset_ax.bar(bins_ms[:-1], counts, width=10, color='blue', alpha=0.5)
    inset_ax.axvline(this_event.s_ms_new, color='green', linestyle='--', label='Start')
    inset_ax.axvline(this_event.e_ms_new, color='green', linestyle='--', label='End')
    inset_ax.set_title("ms Histogram", fontsize=10)  # Inset title
    inset_ax.tick_params(axis='both', which='major', labelsize=8)  # Smaller ticks for inset

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
    plt.axvline(plotting.peak_arrival_time, color='red')
    plt.title("Histogram of dt values")
    plt.xlabel("arrival time ")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')  # Adjust the legend position if needed
    plt.show()


def scatter_event(i):
    this_event = group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event, pick_photons, diameter)
    print(this_event_photons)

    prev_x = this_event.x
    prev_y = this_event.y
    x_array = this_event_photons['x'].to_numpy()
    y_array = this_event_photons['y'].to_numpy()
    new_x, new_y, sx, sy = fitting.event_position(x_array, y_array, False)
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

    this_event_photons = get_photons.crop_event(this_event, pick_photons, diameter)

    bin_size = 5
    bins = np.arange(min(this_event_photons.dt), max(this_event_photons.dt) + bin_size, bin_size)
    plt.figure(figsize=(8, 6))
    plt.hist(this_event_photons['dt'], bins=bins)
    plt.plot([], [], ' ', label=f'Total number of photons: {len(this_event_photons)}')
    plt.plot([], [], ' ', label=f'Lifetime: {this_event.lifetime:.3f}')
    plt.title("Histogram of dt values")
    plt.xlabel("arrival time ")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')  # Adjust the legend position if needed
    plt.show()


def hist_x_event(i):
    this_event = group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event, pick_photons, diameter)
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

if __name__ == "__main__":
    # Try calling one of your functions to produce the plot:
    scatter_event(1)