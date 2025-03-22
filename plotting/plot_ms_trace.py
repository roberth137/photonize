import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import get_photons
import fitting
import plotting as _p


def hist_ms_event(i):
    """
    Plots the histogram for the i-th event in group_events.

    :param i: Index of the event in the group_events list/DataFrame
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

    plt.show()


if __name__ == "__main__":
    # Plot the histogram for the first event in _p.group_events
    hist_ms_event(1)
