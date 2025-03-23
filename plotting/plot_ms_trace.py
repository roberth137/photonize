import matplotlib
matplotlib.use('Qt5Agg') # for command line plotting
import matplotlib.pyplot as plt
plt.ion() # turn on interactive mode
import numpy as np
import ruptures as rpt
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
    lifetime = this_event["lifetime"]

    # Simulate or retrieve photons for this event
    this_event_photons = get_photons.crop_event(
        this_event,
        _p.pick_photons,
        _p.diameter,
        more_ms=400
    )

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

    # Fit a step function using change point detection
    model = "l2"
    algo = rpt.Binseg(model=model, min_size=1, jump=1).fit(smoothed_counts_2)
    change_points = algo.predict(n_bkps=2)  # for 2 change points
    change_points_trans = np.array(change_points, dtype=float)

    # Shift change points to match bin centers, approximate
    change_points_trans[0] = (change_points_trans[0] - 1.5) * bin_size + bins[0]
    change_points_trans[1] = (change_points_trans[1] + 0.5) * bin_size + bins[0]

    # Annotate the plot
    duration_ms = end_ms - start_ms
    plt.plot([], [], ' ', label=f'duration_ms: {duration_ms}')
    plt.plot([], [], ' ', label=f'number_photons: {len(this_event_photons)}')
    plt.plot([], [], ' ', label=f'Lifetime: {lifetime:.3f}')

    plt.axvline(start_ms+30, color='red')
    plt.axvline(end_ms-30, color='red')

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
    lifetime = this_event["lifetime"]
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

    # Compute weighted quantiles (here using the 10th and 90th percentiles as on/off times)
    ms_values = np.array(this_event_photons.ms)
    on_time = weighted_quantile(ms_values, 0.1, weights)
    off_time = weighted_quantile(ms_values, 0.9, weights)

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

    # Mark the weighted on/off times
    plt.axvline(on_time, color='green', linestyle='-', label='On time (10th percentile)')
    plt.axvline(off_time, color='blue', linestyle='-', label='Off time (90th percentile)')

    plt.title("Histogram of ms (with IDW on/off times)")
    plt.xlabel("ms of arrival")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    #plt.show()


def weighted_quantile(values, quantile, weights, eps=1e-6):
    """
    Compute the weighted quantile of the given 1D array.

    :param values: Array of values (or Pandas Series).
    :param quantile: Quantile to compute (e.g. 0.1 for the 10th percentile).
    :param weights: Weights corresponding to each value (or Pandas Series).
    :param eps: A small value added to the denominator to avoid division by zero.
    :return: The weighted quantile value.
    """

    # Convert to NumPy arrays so that np.argsort and indexing work as expected.
    values_array = np.asarray(values)
    weights_array = np.asarray(weights)

    # Sort by the values
    sorter = np.argsort(values_array)
    values_sorted = values_array[sorter]
    weights_sorted = weights_array[sorter]

    # Normalize cumulative weights
    cum_weights = np.cumsum(weights_sorted)
    norm_cum_weights = cum_weights / (cum_weights[-1] + eps)

    # Interpolate to get the quantile value
    return np.interp(quantile, norm_cum_weights, values_sorted)


if __name__ == "__main__":
    # Plot the histogram for the first event in _p.group_events
    plt.ioff()
    #hist_ms_event(1)
    hist_idw_ms_event(3)
    plt.show(block=True)

