import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # For reproducibility


# -------------------------------
# 1. Simulate event statistics
# -------------------------------
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
      - sx: Normal (mean=1.06892, std=0.118329)
      - sy: Normal (mean=1.084316, std=0.122499)
      - bg: Normal (mean=3, std=1), clipped to be non-negative
      - brightness: Lognormal (from mean=0.9, std=0.6), clipped to a minimum of 0.1
      - photons: Lognormal (from mean=560, std=600), clipped to a minimum of 101
      - delta_x: Normal (loc=0, scale=0.05)  [for illustration]
      - delta_y: Normal (loc=0, scale=0.05)  [for illustration]

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
    # For illustration, delta_x and delta_y are simulated as normals
    delta_x = np.random.normal(loc=0, scale=0.05, size=n_events)
    delta_y = np.random.normal(loc=0, scale=0.05, size=n_events)

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


# Generate 1000 event statistics
event_stats = simulate_event_stats(n_events=10000)

# Optionally, you can inspect the first few rows:
print("First 5 event stats:")
print(event_stats[:5])

# -------------------------------
# 2. Use event stats as parameters for simulation
# -------------------------------
# Here we assume that the simulation and fitting functions are defined in a module "simulate"
# and that the fitting routine "analyze_sim_event" and functions "simulate_fluorophore" and
# "simulate_background" are available. For this example, we simulate a single fluorophore and
# background per event and then perform a COM fit with and without background correction.

import simulate as s  # This module should contain simulate_fluorophore, simulate_background, analyze_sim_event

# Preallocate arrays for storing the fitted positions
n = len(event_stats)
x_fit_w_bg = np.empty(n)
y_fit_w_bg = np.empty(n)
x_fit_pure = np.empty(n)
y_fit_pure = np.empty(n)

# Loop over each event in our event stats dataset
for i, event in enumerate(event_stats):
    # Extract parameters for the event
    num_photons = int(event['photons'])
    # Use the average of sx and sy as the effective PSF width (you can choose differently)
    sigma_psf = (event['sx'] + event['sy']) / 2.0
    binding_time_ms = event['binding_time']
    bg_rate_true = event['bg']

    # Use the event parameters in the simulation functions:
    x_fluo, y_fluo = s.simulate_fluorophore(num_photons=num_photons,
                                            sigma_psf=sigma_psf,
                                            camera_error=s.camera_error,
                                            min_cam_binning=s.subpixel)

    x_bg, y_bg = s.simulate_background(num_pixels=s.num_pixels,
                                       binding_time_ms=binding_time_ms,
                                       bg_rate_true=bg_rate_true,
                                       subpixel=s.subpixel)

    # Perform COM fit without background correction
    pos_no_bg = s.analyze_sim_event(x_fluo, y_fluo,
                                    x_bg, y_bg,
                                    x_entry=s.x_ref, y_entry=s.y_ref,
                                    diameter=s.fitting_diameter,
                                    consider_bg=False)
    # Perform COM fit with background correction
    pos_with_bg = s.analyze_sim_event(x_fluo, y_fluo,
                                      x_bg, y_bg,
                                      x_entry=s.x_ref, y_entry=s.y_ref,
                                      diameter=s.fitting_diameter,
                                      consider_bg=True)

    # If no photons were found within the ROI, skip this event.
    if pos_no_bg[0] is None or pos_with_bg[0] is None:
        continue

    x_fit_pure[i], y_fit_pure[i] = pos_no_bg
    x_fit_w_bg[i], y_fit_w_bg[i] = pos_with_bg

# -------------------------------
# 3. Plot the fitted positions
# -------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(x_fit_w_bg, bins=30, color='purple', alpha=0.7)
plt.xlabel('Fitted X with bg (pixels)')
plt.ylabel('Counts')
plt.title(f'Error w bg correction (std: {np.std(x_fit_w_bg)})')

plt.subplot(1, 2, 2)
plt.hist(x_fit_pure, bins=30, color='green', alpha=0.7)
plt.xlabel('Fitted X without bg (pixels)')
plt.ylabel('Counts')
plt.title(f'Error without bg correction (std: {np.std(x_fit_pure)})')

plt.tight_layout()
plt.show()
