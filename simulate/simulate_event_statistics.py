import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba.core.cgutils import printf

import simulate as s  # Ensure that s has the constants like s.binding_time_mean, s.binding_time_std, etc.

# SET EVENT SIMULATION PARAMETERS (mean, std, even if not gaussian distributed)
binding_time_mean, binding_time_std  = 400, 300 # in ms
sx_mean, sx_std = 1.07, 0.13
sy_mean, sy_std = 1.07, 0.13
brightness_mean, brightness_std = 0.92, 0.6
bg_rate_mean, bg_rate_std = 4, 1.6
delta_x_mean, delta_x_std = 0, 0.1
delta_y_mean, delta_y_std = 0, 0.1

def lognormal_params_from_mean_std(mean, std):
    """
    Given an arithmetic mean and standard deviation (both > 0),
    compute the (mu, sigma) parameters for np.random.lognormal.
    """
    var = std ** 2
    factor = 1.0 + var / (mean ** 2)
    sigma = np.sqrt(np.log(factor))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return mu, sigma


def simulate_event_stats(seed=42,
                         n_events=10000,
                         binding_mean=binding_time_mean, binding_std=binding_time_std,
                         brightness_mean=brightness_mean, brightness_std=brightness_std,
                         sx_mean=sx_mean, sx_std=sx_std,
                         sy_mean=sy_mean, sy_std=sy_std,
                         bg_mean=bg_rate_mean, bg_std=bg_rate_std,
                         delta_x_mean=delta_x_mean, delta_x_std=delta_x_std,
                         delta_y_mean=delta_y_mean, delta_y_std=delta_y_std):
    """
    Simulate n_events directly with distributions that ensure valid values.

    The parameters are set as follows:
      - binding_time (ms): Lognormal (from mean=binding_mean, std=binding_std), clipped >= 50 ms
      - sx: Normal (mean=sx_mean, std=sx_std); nearly all values should exceed ~0.5 px
      - sy: Normal (mean=sy_mean, std=sy_std)
      - bg: Normal (mean=bg_mean, std=bg_std), clipped >= 0
      - brightness: Lognormal (from mean=brightness_mean, std=brightness_std), clipped >= 0.1
      - photons: Derived from binding_time * brightness
      - delta_x: Normal (mean=delta_x_mean, std=delta_x_std)
      - delta_y: Normal (mean=delta_y_mean, std=delta_y_std)

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns:
          ['binding_time', 'sx', 'sy', 'bg', 'brightness',
           'photons', 'delta_x', 'delta_y']
    """
    np.random.seed(seed=seed)
    # 1) Compute lognormal parameters
    binding_mu, binding_sigma = lognormal_params_from_mean_std(binding_mean, binding_std)
    brightness_mu, brightness_sigma = lognormal_params_from_mean_std(brightness_mean, brightness_std)

    # 2) Draw from lognormal / normal distributions
    brightness = np.clip(np.random.lognormal(mean=brightness_mu,
                                             sigma=brightness_sigma,
                                             size=n_events), 0.1, None)
    binding_time = np.clip(np.random.lognormal(mean=binding_mu,
                                               sigma=binding_sigma,
                                               size=n_events), 50, None)
    photons = binding_time * brightness
    sx = np.random.normal(loc=sx_mean, scale=sx_std, size=n_events)
    sy = np.random.normal(loc=sy_mean, scale=sy_std, size=n_events)
    bg = np.clip(np.random.normal(loc=bg_mean, scale=bg_std, size=n_events), 0, None)
    delta_x = np.random.normal(loc=delta_x_mean, scale=delta_x_std, size=n_events)
    delta_y = np.random.normal(loc=delta_y_mean, scale=delta_y_std, size=n_events)

    # 3) Build a pandas DataFrame directly
    df = pd.DataFrame({
        'binding_time': binding_time,
        'sx': sx,
        'sy': sy,
        'bg': bg,
        'brightness': brightness,
        'photons': photons,
        'delta_x': delta_x,
        'delta_y': delta_y
    })
    return df


def save_simulate_events(filename,
                         seed, n_events=10000,
                         binding_mean=binding_time_mean, binding_std=binding_time_std,
                         brightness_mean=brightness_mean, brightness_std=brightness_std,
                         sx_mean=sx_mean, sx_std=sx_mean,
                         sy_mean=sy_mean, sy_std=sy_std,
                         bg_mean=bg_rate_mean, bg_std=bg_rate_std,
                         delta_x_mean=delta_x_mean, delta_x_std=delta_x_std,
                         delta_y_mean=delta_y_mean, delta_y_std=delta_y_std):
    """
    Simulate a dataset of events and save them in HDF5 format.

    Parameters
    ----------
    filename : str
        HDF5 filename for saving the dataset.
    n_events : int
        Number of events to simulate.
    ...
    (Other parameter docstrings omitted for brevity.)

    Saves
    -----
    An HDF5 file with a single key ("events") containing a pandas DataFrame.
    """
    # Simulate the event stats (pandas DataFrame)
    df_events = simulate_event_stats(seed, n_events,
        binding_mean, binding_std,
        brightness_mean, brightness_std,
        sx_mean, sx_std,
        sy_mean, sy_std,
        bg_mean, bg_std,
        delta_x_mean, delta_x_std,
        delta_y_mean, delta_y_std
    )

    # Save to HDF5
    df_events.to_hdf(filename, key='events', mode='w')
    print(f"Saved {len(df_events)} events to {filename} in HDF5 format.")


def plot_event_histograms(df):
    """
    Plot histograms for each event parameter in a DataFrame.
    """
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    ax = axes.ravel()

    ax[0].hist(df['binding_time'], bins=50, alpha=0.7)
    ax[0].set_title("Binding Time (ms)")

    ax[1].hist(df['sx'], bins=50, alpha=0.7)
    ax[1].set_title("sx (pixels)")

    ax[2].hist(df['sy'], bins=50, alpha=0.7)
    ax[2].set_title("sy (pixels)")

    ax[3].hist(df['bg'], bins=50, alpha=0.7)
    ax[3].set_title("Background (bg)")

    ax[4].hist(df['brightness'], bins=50, alpha=0.7)
    ax[4].set_title("Brightness (phot/ms)")

    ax[5].hist(df['photons'], bins=50, alpha=0.7)
    ax[5].set_title("Photons")

    ax[6].hist(df['delta_x'], bins=50, alpha=0.7)
    ax[6].set_title("Delta X")

    ax[7].hist(df['delta_y'], bins=50, alpha=0.7)
    ax[7].set_title("Delta Y")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    random_seed = 42
    n_events = 100000
    save_data = True
    if save_data: save_name = 'less_photons.hdf5'

    # SET EVENT SIMULATION PARAMETERS (mean, std, even if not gaussian distributed)
    binding_time_mean, binding_time_std = 300, 200  # in ms
    sx_mean, sx_std = 1.07, 0.13
    sy_mean, sy_std = 1.07, 0.13
    brightness_mean, brightness_std = 0.9, 0.6
    bg_rate_mean, bg_rate_std = 3, 1
    delta_x_mean, delta_x_std = 0, 0.3
    delta_y_mean, delta_y_std = 0, 0.3

    # Generate a dataset of event statistics
    df_events = simulate_event_stats(seed=random_seed,
                                     n_events=n_events,
                                     binding_mean=binding_time_mean, binding_std=binding_time_std,
                                     brightness_mean=brightness_mean, brightness_std=brightness_std,
                                     sx_mean=sx_mean, sx_std=sx_std,
                                     sy_mean=sy_mean, sy_std=sy_std,
                                     bg_mean=bg_rate_mean, bg_std=bg_rate_std,
                                     delta_x_mean=delta_x_mean, delta_x_std=delta_x_std,
                                     delta_y_mean=delta_y_mean, delta_y_std=delta_y_std)

    print(f"Simulated {len(df_events)} events.")
    print("Means:")
    print(" Binding time (ms):", df_events['binding_time'].mean())
    print(" sx (pixels):      ", df_events['sx'].mean())
    print(" sy (pixels):      ", df_events['sy'].mean())
    print(" Background (bg):  ", df_events['bg'].mean())
    print(" Brightness:       ", df_events['brightness'].mean())
    print(" Photons:          ", df_events['photons'].mean())
    print(" Delta X:          ", df_events['delta_x'].mean())
    print(" Delta Y:          ", df_events['delta_y'].mean())

    # Plot histograms of the event parameters
    plot_event_histograms(df_events)

    #save_dataset
    if save_data:
        save_simulate_events(
        filename=save_name,
        seed=42,
        n_events=n_events,
        binding_mean=binding_time_mean, binding_std=binding_time_std,
        brightness_mean=brightness_mean, brightness_std=brightness_std,
        sx_mean=sx_mean, sx_std=sx_std,
        sy_mean=sy_mean, sy_std=sy_std,
        bg_mean=bg_rate_mean, bg_std=bg_rate_std,
        delta_x_mean=delta_x_mean, delta_x_std=delta_x_std,
        delta_y_mean=delta_y_mean, delta_y_std=delta_y_std
        )