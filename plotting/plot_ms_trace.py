import matplotlib
matplotlib.use('Qt5Agg') # for command line plotting
import matplotlib.pyplot as plt
plt.ion() # turn on interactive mode
import numpy as np
import get_photons
import fitting
import plotting as _p



def hist_ms_event(i):
    """
    Plots the ms_histogram for the i-th event in group_events.

    """
    this_event = _p.group_events.iloc[i]

    # Extract relevant values
    start_ms = this_event["start_ms"]
    end_ms = this_event["end_ms"]

    # Simulate or retrieve photons for this event
    this_event_photons = get_photons.crop_event(
        this_event,
        _p.pick_photons,
        _p.diameter,
        more_ms=400
    )

    start_ms_calc, end_ms_calc, duration_ms_calc = fitting.get_on_off_dur(this_event_photons)

    # For convenience, treat these photons as an array of ms times
    bin_size = 10
    bins = np.arange(min(this_event_photons.ms),
                     max(this_event_photons.ms) + bin_size,
                     bin_size)

    counts, _ = np.histogram(this_event_photons, bins=bins)
    smoothed_counts_2 = fitting.lee_filter_1d(counts, 5)
    # Plot
    plt.figure(figsize=(8, 6))
    # Original histogram
    plt.bar(bins[:-1], counts, width=bin_size, alpha=0.5)
    # Smoothed histogram
    plt.bar(bins[:-1], smoothed_counts_2, width=bin_size, alpha=0.5)

    # Annotate the plot
    duration_ms = end_ms - start_ms
    plt.plot([], [], ' ', label=f'duration_ms: {duration_ms}')
    plt.plot([], [], ' ', label=f'number_photons: {len(this_event_photons)}')
    if "lifetime_10ps" in this_event:
        plt.plot([], [], ' ', label=f'Lifetime: {this_event["lifetime_10ps"]:.3f}')

    plt.axvline(start_ms, color='red')
    plt.axvline(end_ms, color='red')
    plt.title("Histogram of ms")
    plt.xlabel("ms of arrival")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    #plt.savefig(f"hist_event_{i}.png")

def hist_idw_ms_event(i):
    """
        Plots the histogram for the i-th event in group_events.
        Uses inverse distance weighting from event.x, event.y to determine on/off times.

        :param i: Index of the event in the group_events list/DataFrame.
        """
    # Retrieve the event
    this_event = _p.group_events.iloc[i]

    # Extract relevant values from the event
    start_ms = this_event["start_ms"]
    end_ms = this_event["end_ms"]
    lifetime = this_event["lifetime_10ps"]
    event_x = this_event["x"]
    event_y = this_event["y"]

    # Retrieve photons for this event (assumes photons have attributes ms, x, y)
    this_event_photons = get_photons.crop_event(
        this_event,
        _p.pick_photons,
        _p.diameter,
        more_ms=400
    )

    # Compute inverse distance weights for each photon.
    # Here, a small epsilon is added to avoid division by zero.
    epsilon = 1e-6
    distances = np.sqrt((this_event_photons.x - event_x) ** 2 + (this_event_photons.y - event_y) ** 2)
    weights = (1 * (1-distances/(_p.diameter/2)))

    # Define histogram bins (using ms times)
    bin_size = 10
    bins = np.arange(min(this_event_photons.ms),
                     max(this_event_photons.ms) + bin_size,
                     bin_size)

    # Compute the weighted histogram of ms times using the computed weights.
    counts, _ = np.histogram(this_event_photons.ms, bins=bins, weights=weights)

    # Optionally, smooth the weighted histogram (using your existing lee_filter_1d)
    smoothed_counts = fitting.lee_filter_1d(counts, 5)

    # Plot the weighted histogram and its smoothed version.
    plt.figure(figsize=(8, 6))
    plt.bar(bins[:-1], counts, width=bin_size, alpha=0.5, label='Weighted histogram')
    plt.bar(bins[:-1], smoothed_counts, width=bin_size, alpha=0.5, label='Smoothed weighted histogram')

    # Annotate the plot with event information.
    duration_ms = end_ms - start_ms
    plt.plot([], [], ' ', label=f'duration_ms: {duration_ms}')
    plt.plot([], [], ' ', label=f'number_photons: {len(this_event_photons)}')
    plt.plot([], [], ' ', label=f'Lifetime: {lifetime:.3f}')

    # Mark original start and end times (if desired)
    plt.axvline(start_ms, color='red', linestyle='--', label=f'start_ms: {start_ms}')
    plt.axvline(end_ms, color='red', linestyle='--', label=f'end_ms: {end_ms}')


    plt.title("Histogram of ms (with IDW on/off times)")
    plt.xlabel("ms of arrival")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    #plt.show()


if __name__ == "__main__":
    # Plot the histogram for the first event in _p.group_events
    plt.ioff()
    #hist_ms_event(1)
    hist_ms_event(3)
    plt.show(block=True)

