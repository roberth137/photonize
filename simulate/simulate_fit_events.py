import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate as s


def simulate_and_fit_events(event_stats,
                             method: str = 'com',
                             diameter: float = s.fitting_diameter,
                             random_seed: int = 42):
    """
    Given a DataFrame of event parameters, simulate fluorophore + background for each event,
    perform a center-of-mass fit using either background correction or not,
    and return the fitted positions and background rates.

    Parameters
    ----------
    event_stats : pd.DataFrame
        DataFrame containing 'photons', 'sx', 'sy', 'brightness', 'binding_time', and 'bg'.
    method : {'com', 'no_bg'}
        Choose 'com' for plain center of mass, 'no_bg' otherwise.
    diameter : float
        Diameter (in pixels) for the fitting region.
    random_seed : int
        Seed for reproducible simulation.

    Returns
    -------
    x_fit : np.ndarray
        Fitted x positions for each event.
    y_fit : np.ndarray
        Fitted y positions for each event.
    bg_rates : np.ndarray
        Background rates used for each event.
    """
    np.random.seed(random_seed)

    n = len(event_stats)
    x_fit = np.empty(n, dtype=float)
    y_fit = np.empty(n, dtype=float)
    bg_rates = event_stats['bg'].astype(float).values

    use_bg = True if method == 'with_bg' else False

    print(f"Selected method: {'Background correction' if use_bg else 'No background'}")
    pd.set_option('display.max_columns', None)
    print("First 5 event stats:")
    print(event_stats.head(5))

    for i, row in event_stats.iterrows():
        num_photons = int(row['photons'])
        sigma_psf = (row['sx'] + row['sy']) / 2.0
        brightness = row['brightness']
        binding_time = row['binding_time']
        bg_rate = row['bg']

        # simulate fluorophore and background
        x_fluo, y_fluo = s.simulate_fluorophore(
            binding_time=binding_time,
            brightness=brightness,
            sigma_psf=sigma_psf,
            camera_error=s.camera_error,
            subpixel=s.subpixel
        )
        x_bg, y_bg = s.simulate_background(
            num_pixels=s.num_pixels,
            binding_time_ms=binding_time,
            bg_rate=bg_rate,
            subpixel=s.subpixel
        )

        # fit COM once using the selected method
        x0, y0 = s.analyze_sim_event(
            x_fluo, y_fluo,
            x_bg, y_bg,
            x_entry=s.x_ref, y_entry=s.y_ref,
            diameter=diameter,
            consider_bg=use_bg
        )

        x_fit[i], y_fit[i] = x0, y0

    return x_fit, y_fit, bg_rates


def plot_results(distance, method: str = 'com'):
    """
    Plot histogram of error distances for a single fitting method.

    Parameters
    ----------
    distance : array-like
        Error distances for each event.
    method : {'com', 'no_bg'}
        Label for the plot title.
    """
    # Define range based on data
    x_min, x_max = 0.0, np.percentile(distance, 95)
    bins = freedman_diaconis_bins(distance, data_range=(x_min, x_max))

    plt.figure(figsize=(6, 5))
    plt.hist(distance, bins=bins, range=(x_min, x_max), alpha=0.7)
    label = method
    plt.xlabel('Error distance (pixels) up to 95th percentile displayed')
    plt.ylabel('Counts')
    plt.title(f"Method: {label}  (mean: {np.mean(distance):.5f})")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Generate or load event statistics
    event_stats = s.simulate_event_stats(n_events=1000)
    event_stats = pd.DataFrame(event_stats)

    # Choose fitting method:
    method = 'com'
    diameter = 4

    # Run simulation and fitting
    x_fit, y_fit, bg_rates = simulate_and_fit_events(event_stats, method, diameter)

    # Compute distances from reference
    _, _, distances = s.filter_points_by_radius(x_fit, y_fit, x_ref=s.x_ref, y_ref=s.y_ref)

    # Plot results
    plot_results(distances, method)
