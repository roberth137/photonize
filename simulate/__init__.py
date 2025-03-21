import numpy as np

# SET TEST PARAMETERS
num_pixels = 8  # 8x8 area => coordinates range from -4 to +4
binding_time_ms = 300  # Binding time in ms
brightness = 1 # Fluorophore brightness in photons/ms
bg_rate_true = 2 # True background counts in photons/pixel/200ms
bg_rate_meas = 2 # Measured background value for fitting
sigma_psf = 1.1  # Gaussian width (standard deviation)
camera_error = 0.29 # Camera error when assigning pixels
min_cam_binning = 1/16 # Smallest unit camera can theoretically resolve (12 bit ~ 4096 pixel)
fitting_diameter = 4.5 # diameter considered for fitting
max_dist = fitting_diameter/2 # radius from point of interest to be considered
x_ref, y_ref = 0, 0

# SET SIMULATION PARAMETERS

num_photons = brightness * binding_time_ms
fit_area = max_dist**2 * np.pi

__all__ = ["num_pixels",
           "binding_time_ms",
           "brightness",
           "bg_rate_true",
           "bg_rate_meas",
           "sigma_psf",
           "camera_error",
           "min_cam_binning",
           "fitting_diameter",
           "num_photons",
           "max_dist",
           "x_ref", "y_ref",
           "fit_area"]

from simulate.analyze_event import distance_to_point, analyze_sim_event
from simulate.simulate_bg import simulate_background
from simulate.simulate_fluorophore import simulate_fluorophore
from simulate.simulate_event_statistics import simulate_event_stats