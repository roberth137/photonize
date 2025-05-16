import numpy as np

# SET TEST PARAMETERS
num_pixels = 8  # 8x8 area => coordinates range from -4 to +4
binding_time_ms = 300  # Binding time in ms
brightness = 1 # Fluorophore brightness in photons/ms
bg_rate = 4 # True background counts in photons/pixel/200ms
sigma_psf = 1.1  # Gaussian width (standard deviation)
camera_error = 0.29 # 0.29 # Camera error when assigning pixels
subpixel = 16 # Smallest division unit camera can theoretically resolve (12 bit ~ 4096 pixel corresponding to 256 pixel -> 4096 / 256 = 16)
fitting_diameter = 4.5 #  diameter considered for fitting
x_ref, y_ref = 0, 0

num_photons = brightness * binding_time_ms
max_dist = fitting_diameter/2 # radius from point of interest to be considered
fit_area = max_dist**2 * np.pi

__all__ = (["num_pixels", "binding_time_ms", "brightness", "bg_rate",
           "sigma_psf", "camera_error", "subpixel", "fitting_diameter",
           "num_photons", "max_dist", "x_ref", "y_ref", "fit_area"])

from simulate.analyze_sim_event import analyze_sim_event
from simulate.simulate_bg import simulate_background
from simulate.simulate_fluorophore import simulate_fluorophore
from simulate.simulate_event_statistics import simulate_event_stats
from simulate.plot_event import plot_event
from simulate.simulate_utils import filter_points_by_radius, freedman_diaconis_bins
from simulate.simulate_fit_events import simulate_and_fit_events