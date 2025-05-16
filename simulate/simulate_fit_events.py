import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate as s


def simulate_and_fit_events(
    event_stats: pd.DataFrame,
    method: str = 'com',
    diameter: float = s.fitting_diameter,
    random_seed: int = 42
):
    """
    Given a DataFrame of event parameters, simulate fluorophore + background for each event,
    delegate localization to analyze_sim_event (COM, full‐EM MLE, or fixed‐σ fixed‐B MLE),
    and return the fitted positions plus background counts in the ROI.

    Parameters
    ----------
    event_stats : pd.DataFrame
        Must contain columns 'photons', 'sx', 'sy', 'brightness', 'binding_time', 'bg'.
    method : {'com', 'mle', 'mle_fixed'}
        'com'       → center‐of‐mass
        'mle'       → full EM Gaussian+uniform‐BG fit
        'mle_fixed' → EM fit with fixed σ & fixed B (fits only μ)
    diameter : float
        ROI diameter (same units as x,y).
    random_seed : int
        For reproducible simulation.

    Returns
    -------
    x_fit : np.ndarray of shape (n,)
    y_fit : np.ndarray of shape (n,)
    bg_counts : np.ndarray of shape (n,)
        Number of background photons inside the ROI for each event.
    """
    np.random.seed(random_seed)
    n = len(event_stats)
    x_fit     = np.full(n, np.nan, float)
    y_fit     = np.full(n, np.nan, float)
    bg_fit     = np.full(n, np.nan, float)
    bg_counts = np.zeros(n, int)

    method = method.lower()
    if method not in ('com', 'mle', 'mle_fixed', 'mle_once', 'pass', 'com_twice'):
        raise ValueError("Method must be one of 'com', 'mle', or 'mle_fixed'.")

    for i, row in event_stats.iterrows():
        # --- parameters for this event ---
        sigma_psf  = (row['sx'] + row['sy']) / 2.0
        binding_ms = row['binding_time']
        brightness = row['brightness']
        bg_rate    = float(row['bg'])
        x_ref, y_ref = row['delta_x'], row['delta_y']
        radius = diameter / 2.0

        # --- simulate signal and background photons ---
        x_fluo, y_fluo = s.simulate_fluorophore(
            binding_time=binding_ms,
            brightness=brightness,
            sigma_psf=sigma_psf,
            camera_error=s.camera_error,
            subpixel=s.subpixel
        )
        x_bg, y_bg = s.simulate_background(
            num_pixels=s.num_pixels,
            binding_time_ms=binding_ms,
            bg_rate=bg_rate,
            subpixel=s.subpixel
        )

        # --- count background photons inside the ROI ---
        dx_bg = x_bg - x_ref
        dy_bg = y_bg - y_ref
        mask_bg = dx_bg*dx_bg + dy_bg*dy_bg <= radius*radius
        bg_counts[i] = mask_bg.sum()

        # --- delegate to analyze_sim_event ---
        x0, y0, bg0 = s.analyze_sim_event(
            x_fluo, y_fluo,
            x_bg,   y_bg,
            x_entry=x_ref, y_entry=y_ref,
            sigma=sigma_psf,
            bg_rate=bg_rate,
            diameter=diameter,
            method=method,
            binding_time=binding_ms
        )

        x_fit[i], y_fit[i], bg_fit[i] = x0, y0, bg0

    return x_fit, y_fit, bg_fit


def plot_results(distances: np.ndarray, diameter: int, method: str = 'com'):
    """
    Plot histogram of localization error distances for a single fitting method.

    Parameters
    ----------
    distances : array-like
        Error distances for each event.
    method : {'com', 'mle'}
        Label for the plot title.
    """
    # Truncate at 95th percentile for display
    x_min, x_max = 0.0, np.percentile(distances, 95)
    bins = s.freedman_diaconis_bins(distances, data_range=(x_min, x_max))

    plt.figure(figsize=(6, 5))
    plt.hist(distances, bins=bins, range=(x_min, x_max), alpha=0.7)
    mean_dist = np.nanmean(distances)

    plt.xlabel(f'Error until 95th percentile, total events: {len(distances)}')
    plt.ylabel('Counts')
    plt.title(f"Method: {method.upper()}, diameter: {diameter}, mean error: {mean_dist:.5f} px)")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Generate or load event statistics
    event_stats = s.simulate_event_stats(n_events=1000)
    event_stats = pd.DataFrame(event_stats)

    # Choose fitting method and diameter
    method = 'mle'  # 'com' , 'mle' , 'mle_fixed'
    diameter = 6.0

    # Run simulation and fitting
    x_fit, y_fit, bg_rates = simulate_and_fit_events(event_stats, method, diameter)

    distances = np.hypot(x_fit, y_fit)

    # Compute distances from reference center
    _, _, distances = s.filter_points_by_radius(
        x_fit, y_fit,
        x_ref=0, y_ref=0,
        max_dist=np.inf
    )

    # Plot results
    plot_results(distances, diameter, method)
