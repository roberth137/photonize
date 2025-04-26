import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate as s



def simulate_and_fit_events(event_stats, diameter=s.fitting_diameter, random_seed=42):
    """
    Given a DataFrame of event parameters, simulate fluorophore + background for each event,
    perform fits (with and without background), and return:
      x_fit_pure, y_fit_pure    -> Fitted positions w/o background correction
      x_fit_w_bg, y_fit_w_bg    -> Fitted positions w/ background correction
      distance_pure, distance_w_bg -> Distances of fitted positions from (0, 0)
    """
    np.random.seed(random_seed)

    # Preallocate arrays for storing the fitted positions
    n = len(event_stats)
    x_fit_w_bg = np.empty(n, dtype=float)
    y_fit_w_bg = np.empty(n, dtype=float)
    x_fit_pure = np.empty(n, dtype=float)
    y_fit_pure = np.empty(n, dtype=float)

    print("First 5 event stats:")
    pd.set_option('display.max_columns', None)
    print(event_stats.head(5))

    # Iterate over DataFrame rows
    for i, row in event_stats.iterrows():
        # Extract parameters for the event from the row
        num_photons = int(row['photons'])
        sigma_psf = (row['sx'] + row['sy']) / 2.0
        brightness = row['brightness']
        binding_time = row['binding_time']
        bg_rate = row['bg']

        # Simulate fluorophore
        x_fluo, y_fluo = s.simulate_fluorophore(
            binding_time=binding_time,
            brightness=brightness,
            sigma_psf=sigma_psf,
            camera_error=s.camera_error,
            subpixel=s.subpixel
        )
        # Simulate background
        x_bg, y_bg = s.simulate_background(
            num_pixels=s.num_pixels,
            binding_time_ms=binding_time,
            bg_rate=bg_rate,
            subpixel=s.subpixel
        )

        # Perform COM fit without background correction
        pos_no_bg = s.analyze_sim_event(
            x_fluo, y_fluo,
            x_bg, y_bg,
            x_entry=s.x_ref, y_entry=s.y_ref,
            diameter=diameter,
            consider_bg=False
        )

        # Perform COM fit with background correction
        pos_with_bg = s.analyze_sim_event(
            x_fluo, y_fluo,
            x_bg, y_bg,
            x_entry=s.x_ref, y_entry=s.y_ref,
            diameter=diameter,
            consider_bg=True
        )

        # Store fitted positions
        x_fit_pure[i], y_fit_pure[i] = pos_no_bg
        x_fit_w_bg[i], y_fit_w_bg[i] = pos_with_bg

    # Calculate distances from (0, 0) – or use s.x_ref, s.y_ref if needed
    _, _, distance_pure = s.distance_to_point(x_fit_pure, y_fit_pure, x_ref=0, y_ref=0)
    _, _, distance_w_bg = s.distance_to_point(x_fit_w_bg, y_fit_w_bg, x_ref=0, y_ref=0)

    indices = np.where(distance_w_bg > 20)[0].astype(np.int32)
    print(indices)
    for i in indices:
        print(event_stats.iloc[i])

    return distance_pure, distance_w_bg


def freedman_diaconis_bins(data, data_range=None):
    """
    Calculate the optimal number of bins for a histogram using the Freedman-Diaconis rule.

    Parameters
    ----------
    data : array-like
        The data to be histogrammed.
    data_range : (float, float) or None
        If not None, a 2-tuple specifying the lower and upper range of the bins.
        If None, it uses the actual min and max of the data.

    Returns
    -------
    bins : int
        The number of bins to use.
    """
    data = np.asarray(data)
    n = len(data)

    # Edge case: if there's no data or only one point, default to 1 bin
    if n <= 1:
        return 1

    # Compute the interquartile range (IQR)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25

    # Freedman-Diaconis bin width
    bin_width = 2.0 * iqr / np.cbrt(n)

    # Edge case: if the IQR is zero (all data points are identical), fallback to 1 bin
    if bin_width <= 0:
        return 1

    # Determine the actual data range to cover
    if data_range is None:
        data_min = data.min()
        data_max = data.max()
    else:
        data_min, data_max = data_range

    # Compute number of bins based on Freedman-Diaconis bin width
    data_span = data_max - data_min
    bins = int(np.ceil(data_span / bin_width))

    # At least 1 bin
    return max(bins, 1)


def plot_results(distance_pure, distance_w_bg):
    """
    Plot two histograms:
      1) Error distances with background correction
      2) Error distances without background correction

    Parameters
    ----------
    distance_pure : array-like
        Error distances without background correction.
    distance_w_bg : array-like
        Error distances with background correction.
    """
    # Combine both datasets to figure out a common x-range and bin size
    combined_data = np.concatenate((distance_pure, distance_w_bg))

    # Compute the global min and max to limit the x-axis dynamically
    x_min = 0#np.min(combined_data)
    x_max = 0.5#np.max(combined_data)

    # Calculate the number of bins using Freedman–Diaconis for the combined data
    fd_bins = freedman_diaconis_bins(combined_data, data_range=(x_min, x_max))

    print(f"Global min: {x_min:.5f}, Global max: {x_max:.5f}")
    print(f"FD bins: {fd_bins}")
    print(f"Max dist (with BG): {max(distance_w_bg):.5f}")
    print(f"Max dist (pure): {max(distance_pure):.5f}")

    # Create the plot
    plt.figure(figsize=(12, 5))

    # Left subplot: with background
    plt.subplot(1, 2, 1)
    plt.hist(distance_w_bg, bins=fd_bins, range=(x_min, x_max), color='purple', alpha=0.7)
    plt.xlabel('Error distance (pixels)')
    plt.ylabel('Counts')
    plt.title(f'Error w/ BG correction (mean: {np.mean(distance_w_bg):.5f})')

    # Right subplot: without background
    plt.subplot(1, 2, 2)
    plt.hist(distance_pure, bins=fd_bins, range=(x_min, x_max), color='green', alpha=0.7)
    plt.xlabel('Error distance (pixels)')
    plt.ylabel('Counts')
    plt.title(f'Error w/o BG correction (mean: {np.mean(distance_pure):.5f})')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Option A: Load event statistics from an HDF5 file
    #event_stats = pd.read_hdf('simulate/sim_experiments_stats/2green_test.hdf5')
    diameter = 4
    # Option B: Or generate 10000 event statistics instead (uncomment if desired)
    event_stats = s.simulate_event_stats(n_events=10000)
    event_stats = pd.DataFrame(event_stats)  # Convert structured array to a DataFrame


    dist_pure, dist_w_bg = simulate_and_fit_events(event_stats, diameter)

    plot_results(dist_pure, dist_w_bg)