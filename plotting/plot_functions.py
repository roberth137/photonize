import matplotlib.pyplot as plt
import numpy as np
import get_photons
import fitting
from plotting import group_events, all_events_photons, diameter

def hist_ms_event(i):
    this_event = group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event,
                                                all_events_photons,
                                                diameter,
                                                200)
    print(this_event_photons)
    bin_size = 5
    bins = np.arange(min(this_event_photons.ms), max(this_event_photons.ms) + bin_size, bin_size)
    counts, _ = np.histogram(this_event_photons, bins=bins)
    plt.figure(figsize=(8, 6))
    plt.bar(bins[:-1], counts, width=bin_size)
    start, end, _ = detect_signal_threshold(counts, 1)#
    start_after = (start*bin_size)+bins[0]
    end_after = (end*bin_size)+bins[0]

    plt.plot([], [], ' ', label=f'Total number of photons: {len(this_event_photons)}')
    plt.plot([], [], ' ', label=f'Start_ms: {this_event.start_ms}, End_ms: {this_event.end_ms}')
    plt.plot([], [], ' ', label=f'Start_thresh: {start}, End_thresh: {end}')
    plt.plot([], [], ' ', label=f'Start_after: {start_after}, End_after: {end_after}')
    plt.plot([], [], ' ', label=f'Lifetime: {this_event.lifetime}')
    plt.axvline(this_event.start_ms, color='red')
    plt.axvline(this_event.end_ms, color='red')
    plt.axvline(start_after, color='magenta')
    plt.axvline(end_after, color='magenta')
    plt.title("Histogram of ms")
    plt.xlabel("ms of arrival")
    # plt.ylabel("Y Position")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')  # Adjust the legend position if needed
    plt.show()


def detect_signal_threshold(data, threshold_factor=2):
    # Calculate basic statistics
    mean = np.mean(data)
    std_dev = np.std(data)
    threshold = mean + threshold_factor * std_dev

    # Vectorized approach to find the start and end points
    above_threshold = data > threshold
    start = np.argmax(above_threshold)
    end_candidates = np.where(above_threshold[start:] == False)[0]
    end = start + end_candidates[0] if len(end_candidates) > 0 else len(data) - 1

    return start, end, threshold


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