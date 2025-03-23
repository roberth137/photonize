import matplotlib
matplotlib.use('Qt5Agg') # for command line plotting
import matplotlib.pyplot as plt
plt.ion() # turn on interactive mode
import numpy as np
import get_photons
import plotting as _p
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def hist_dt_event(i):
    this_event = _p.group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event, _p.pick_photons, _p.diameter)

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
    #plt.savefig(f"plot_dt_{i}.png")

def hist_noise_dt_event(i):
    this_event = _p.group_events.iloc[i]

    more_ms = 400
    this_event_photons = get_photons.crop_event(this_event,
                                                _p.pick_photons,
                                                _p.diameter,
                                                more_ms)
    bounds_first = [(this_event.start_ms_fr - more_ms), this_event.s_ms_new]
    bounds_second = [this_event.e_ms_new, (this_event.end_ms_fr + more_ms)]

    filtered_photons = this_event_photons[
        ((this_event_photons['ms'] >= bounds_first[0])
        & (this_event_photons['ms'] <= bounds_first[1]))
        | ((this_event_photons['ms'] >= bounds_second[0])
        & (this_event_photons['ms'] <= bounds_second[1]))
    ]
    bg_lt = np.mean(filtered_photons.dt)-_p.peak_arrival_time
    bin_size = 5
    bins = np.arange(min(this_event_photons.dt), max(this_event_photons.dt) + bin_size, bin_size)

    # Main plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(this_event_photons['dt'], bins=bins, alpha=0.5, color='orange', label="All photons")
    ax.hist(filtered_photons['dt'], bins=bins, alpha=1, color='blue', label="Filtered photons")
    ax.plot([], [], ' ', label=f'Event photons: {len(this_event_photons)-len(filtered_photons)}')
    ax.plot([], [], ' ', label=f'Bg photons: {len(filtered_photons)}')
    ax.plot([], [], ' ', label=f'Eve_lt: {this_event.lifetime}, BG_lt: {bg_lt}')
    ax.axvline(_p.peak_arrival_time, color='red', label="Peak arrival time")
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


if __name__ == "__main__":
    # Plot the histogram for the first event in _p.group_events
    plt.ioff()
    hist_dt_event(1)
    plt.show(block=True)