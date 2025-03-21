import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # sets the global seed

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


def simulate_event_stats(n_events=1000000):
    """
    Simulate n_events directly with distributions that ensure valid values.

    The parameters are set as follows:
      - binding_time (ms): Lognormal (from mean=700, std=600), then clipped to a minimum of 50 ms
      - sx: Normal (mean=1.06892, std=0.118329); nearly all values exceed 0.55 px
      - sy: Normal (mean=1.084316, std=0.122499); nearly all values exceed 0.55 px
      - bg: Normal (mean=3, std=1), clipped to be non-negative
      - brightness: Lognormal (from mean=0.9, std=0.6), clipped to a minimum of 0.1
      - photons: Lognormal (from mean=560, std=600), clipped to a minimum of 101
      - delta_x: Lognormal (from mean=1.219247e6, std=7.297928e5), clipped to a minimum of 145
      - delta_y: Normal (mean=0.000737, std=0.04155), clipped to the range [-0.326751, 0.36564]

    Returns:
      events (np.ndarray): A structured array with fields:
          'binding_time', 'sx', 'sy', 'bg', 'brightness', 'photons', 'delta_x', 'delta_y'
    """
    # Compute parameters for the lognormal distributions
    binding_mu, binding_sigma = lognormal_params_from_mean_std(700, 600)
    brightness_mu, brightness_sigma = lognormal_params_from_mean_std(0.9, 0.6)
    photons_mu, photons_sigma = lognormal_params_from_mean_std(560, 600)

    # Direct simulation of each parameter:
    binding_time = np.clip(np.random.lognormal(mean=binding_mu,
                                               sigma=binding_sigma,
                                               size=n_events), 50, None)
    sx = np.random.normal(loc=1.06892, scale=0.118329, size=n_events)
    sy = np.random.normal(loc=1.084316, scale=0.122499, size=n_events)
    bg = np.clip(np.random.normal(loc=3, scale=1, size=n_events), 0, None)
    brightness = np.clip(np.random.lognormal(mean=brightness_mu,
                                             sigma=brightness_sigma,
                                             size=n_events), 0.1, None)
    photons = np.clip(np.random.lognormal(mean=photons_mu,
                                          sigma=photons_sigma,
                                          size=n_events), 101, None)
    # Simulate delta_x and _y using a lognormal distribution.
    delta_x = np.random.normal(loc=0, scale=0.05,size=n_events)
    delta_y = np.random.normal(loc=0, scale=0.05,size=n_events)

    # Create a structured array for clarity
    dtype = [('binding_time', 'f4'),
             ('sx', 'f4'),
             ('sy', 'f4'),
             ('bg', 'f4'),
             ('brightness', 'f4'),
             ('photons', 'f4'),
             ('delta_x', 'f4'),
             ('delta_y', 'f4')]
    events = np.array(list(zip(binding_time, sx, sy, bg, brightness, photons, delta_x, delta_y)),
                      dtype=dtype)
    return events


def plot_event_histograms(events):
    """
    Plot histograms for each event parameter.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    ax = axes.ravel()

    ax[0].hist(events['binding_time'], bins=50, alpha=0.7)
    ax[0].set_title("Binding Time (ms)")

    ax[1].hist(events['sx'], bins=50, alpha=0.7)
    ax[1].set_title("sx (pixels)")

    ax[2].hist(events['sy'], bins=50, alpha=0.7)
    ax[2].set_title("sy (pixels)")

    ax[3].hist(events['bg'], bins=50, alpha=0.7)
    ax[3].set_title("Background (bg)")

    ax[4].hist(events['brightness'], bins=50, alpha=0.7)
    ax[4].set_title("Brightness (phot/ms)")

    ax[5].hist(events['photons'], bins=50, alpha=0.7)
    ax[5].set_title("Photons")

    ax[6].hist(events['delta_x'], bins=50, alpha=0.7)
    ax[6].set_title("Delta X")

    ax[7].hist(events['delta_y'], bins=50, alpha=0.7)
    ax[7].set_title("Delta Y")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    events = simulate_event_stats(n_events=10000)

    print(f"Simulated {len(events)} events:")
    print("Means:")
    print(" Binding time (ms):", np.mean(events['binding_time']))
    print(" sx (pixels):      ", np.mean(events['sx']))
    print(" sy (pixels):      ", np.mean(events['sy']))
    print(" Background (bg):  ", np.mean(events['bg']))
    print(" Brightness:       ", np.mean(events['brightness']))
    print(" Photons:          ", np.mean(events['photons']))
    print(" Delta X:          ", np.mean(events['delta_x']))
    print(" Delta Y:          ", np.mean(events['delta_y']))

    # Plot histograms of the event parameters
    plot_event_histograms(events)

