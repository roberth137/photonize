import numpy as np
import matplotlib.pyplot as plt


def simulate_filtered_events(n_events=1000, batch_size=1000):
    """
    Simulate events until we have n_events that pass the thresholds.

    Each event is characterized by:
      - binding_time (ms): Exponential distribution (scale ~697.07 ms)
      - sx: Normal (mean ~1.06892 px, std ~0.11833)
      - sy: Normal (mean ~1.08432 px, std ~0.12250)
      - bg: Normal (mean ~20, std ~25.73) then clipped to non-negative values
      - brightness: Normal (mean ~0.91946, std ~0.60204) then clipped to >0
      - photons: Normal (mean ~564.78, std ~615.97) then clipped to >= 101

    Only events with:
      - binding_time >= 30 ms
      - sx >= 0.55 px
      - sy >= 0.55 px
      - brightness >= 0.1
      - photons >= 101

    are accepted.

    Parameters:
      n_events (int): Desired number of valid events.
      batch_size (int): Number of events to simulate in each batch.

    Returns:
      events (np.ndarray): Structured array with fields
          'binding_time', 'sx', 'sy', 'bg', 'brightness', 'photons'
          containing n_events valid events.
    """

    # Define thresholds
    thresh_binding_time = 30
    thresh_sx = 0.55
    thresh_sy = 0.55
    thresh_brightness = 0.1
    thresh_photons = 101

    # List to accumulate valid events
    events_list = []

    binding_lognorm_mean, binding_lognorm_sigma = lognormal_params_from_mean_std(700, 600)
    bright_lognorm_mean, bright_lognorm_sigma = lognormal_params_from_mean_std(0.9, 0.6)
    photons_lognorm_mean, phot_lognorm_sigma = lognormal_params_from_mean_std(560, 600)


    while True:
        # Simulate a batch of events:
        binding_time = np.clip(np.random.lognormal(mean=binding_lognorm_mean,
                                                 sigma=binding_lognorm_sigma,
                                                 size=batch_size), 50, None)

        sx = np.random.normal(loc=1.06892, scale=0.118329, size=batch_size)
        sy = np.random.normal(loc=1.084316, scale=0.122499, size=batch_size)
        bg = np.clip(np.random.normal(loc=3, scale=1, size=batch_size), 0, None)
        brightness = np.clip(np.random.lognormal(mean=bright_lognorm_mean,
                                                 sigma=bright_lognorm_sigma,
                                                 size=batch_size), 0.001, None)
        photons = np.clip(np.random.lognormal(mean=photons_lognorm_mean,
                                              sigma=phot_lognorm_sigma,
                                              size=batch_size), 101, None)

        # Create a boolean mask for events that pass all thresholds
        mask = (
                (binding_time >= thresh_binding_time) &
                (sx >= thresh_sx) &
                (sy >= thresh_sy) &
                (brightness >= thresh_brightness) &
                (photons >= thresh_photons)
        )

        # If any events pass, collect them
        if np.any(mask):
            valid_events = np.column_stack((binding_time[mask],
                                            sx[mask],
                                            sy[mask],
                                            bg[mask],
                                            brightness[mask],
                                            photons[mask]))
            events_list.append(valid_events)

        # Concatenate all valid events so far
        all_events = np.vstack(events_list) if events_list else np.empty((0, 6))

        if all_events.shape[0] >= n_events:
            # We have enough events; break the loop.
            all_events = all_events[:n_events, :]
            break

    # Create a structured array for clarity
    dtype = [('binding_time', 'f4'),
             ('sx', 'f4'),
             ('sy', 'f4'),
             ('bg', 'f4'),
             ('brightness', 'f4'),
             ('photons', 'f4')]
    events = np.array([tuple(row) for row in all_events], dtype=dtype)
    return events

def lognormal_params_from_mean_std(mean, std):
    """
    Given an arithmetic mean and std (both > 0),
    compute the (mu, sigma) parameters for np.random.lognormal.
    """
    var = std**2
    # The factor inside the log
    factor = 1.0 + var / (mean**2)
    sigma = np.sqrt(np.log(factor))
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma


def plot_event_histograms(events):
    """
    Plot histograms of each event parameter.

    Parameters:
      events (np.ndarray): Structured array with fields
          'binding_time', 'sx', 'sy', 'bg', 'brightness', 'photons'
    """
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    ax = axes.ravel()

    ax[0].hist(events['binding_time'], bins=50, color='C0', alpha=0.7)
    ax[0].set_title("Binding Time (ms)")

    ax[1].hist(events['sx'], bins=50, color='C1', alpha=0.7)
    ax[1].set_title("sx (pixels)")

    ax[2].hist(events['sy'], bins=50, color='C2', alpha=0.7)
    ax[2].set_title("sy (pixels)")

    ax[3].hist(events['bg'], bins=50, color='C3', alpha=0.7)
    ax[3].set_title("Background (bg)")

    ax[4].hist(events['brightness'], bins=50, color='C4', alpha=0.7)
    ax[4].set_title("Brightness (phot/ms)")

    ax[5].hist(events['photons'], bins=50, color='C5', alpha=0.7)
    ax[5].set_title("Photons")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    n_events_desired = 1000
    events = simulate_filtered_events(n_events=n_events_desired, batch_size=1000)

    print(f"Simulated {len(events)} valid events:")
    print("Means:")
    print(" Binding time (ms):", np.mean(events['binding_time']))
    print(" sx (pixels):      ", np.mean(events['sx']))
    print(" sy (pixels):      ", np.mean(events['sy']))
    print(" Background (bg):  ", np.mean(events['bg']))
    print(" Brightness:       ", np.mean(events['brightness']))
    print(" Photons:          ", np.mean(events['photons']))

    # Plot histograms of the filtered event parameters
    plot_event_histograms(events)