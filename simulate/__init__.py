import numpy as np

# SET TEST PARAMETERS
num_pixels = 8  # 8x8 area => coordinates range from -4 to +4
binding_time_ms = 300  # Binding time in ms
brightness = 1 # Fluorophore brightness in photons/ms
bg_rate_true = 2 # True background counts in photons/pixel/200ms
bg_rate_meas = 2 # Measured background value for fitting
sigma_psf = 1.1  # Gaussian width (standard deviation)
camera_error = 2#0.29 # Camera error when assigning pixels
min_cam_binning = 1/16 # Smallest unit camera can theoretically resolve (12 bit ~ 4096 pixel)
fitting_diameter = 4# diameter considered for fitting
max_dist = fitting_diameter/2 # radius from point of interest to be considered
x_ref, y_ref = 0, 0

# SET EVENT SIMULATION PARAMETERS (mean, std, even if not gaussian distributed)
binding_time_mean, binding_time_std  = 700, 500 # in ms
sx_mean, sx_std = 1.07, 0.13
sy_mean, sy_std = 1.07, 0.13
brightness_mean, brightness_std = 0.92, 0.6
bg_rate_true_mean, bg_rate_true_std = 3.3, 1.6
bg_rate_meas_mean, bg_rate_meas_std = 3.3, 1.6
delta_x_mean, delta_x_std = 0, 0.1
delta_y_mean, delta_y_std = 0, 0.1


num_photons = brightness * binding_time_ms
fit_area = max_dist**2 * np.pi

__all__ = ["num_pixels", "binding_time_ms", "brightness", "bg_rate_true", "bg_rate_meas",
           "sigma_psf", "camera_error", "min_cam_binning", "fitting_diameter",
           "num_photons", "max_dist", "x_ref", "y_ref", "fit_area",
           "binding_time_mean", "binding_time_std",
           "sx_mean", "sx_std",
           "sy_mean", "sy_std",
           "brightness_mean", "brightness_std",
           "bg_rate_true_mean", "bg_rate_true_std",
           "bg_rate_meas_mean", "bg_rate_meas_std",
           "delta_x_mean", "delta_x_std",
           "delta_y_mean", "delta_y_std"]

from simulate.analyze_event import distance_to_point, analyze_sim_event
from simulate.simulate_bg import simulate_background
from simulate.simulate_fluorophore import simulate_fluorophore
from simulate.simulate_event_statistics import simulate_event_stats