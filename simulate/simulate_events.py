import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate as s



def simulate_and_fit_events(event_stats, diameter=s.fitting_diameter):
    """
    Given a DataFrame of event parameters, simulate fluorophore + background for each event,
    perform fits (with and without background), and return:
      x_fit_pure, y_fit_pure    -> Fitted positions w/o background correction
      x_fit_w_bg, y_fit_w_bg    -> Fitted positions w/ background correction
      distance_pure, distance_w_bg -> Distances of fitted positions from (0, 0)
    """

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
    # Compute a common upper bound for the histogram range
    hist_dist_bound = 1#(np.max(distance_pure) + np.max(distance_w_bg)) / 2f
    print(f'Max dist with bg: {max(distance_w_bg)}')
    print(f'Max dist pure: {max(distance_pure)}')


    # Create the plot
    plt.figure(figsize=(12, 5))

    # Left subplot: with background
    plt.subplot(1, 2, 1)
    plt.hist(distance_w_bg, bins=30, range=(0, hist_dist_bound), color='purple', alpha=0.7)
    plt.xlabel('Error distance (pixels)')
    plt.ylabel('Counts')
    plt.title(f'Error w/ BG correction (mean: {np.mean(distance_w_bg):.5f})')

    # Right subplot: without background
    plt.subplot(1, 2, 2)
    plt.hist(distance_pure, bins=30, range=(0, hist_dist_bound), color='green', alpha=0.7)
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