num_pixels = 8  # 8x8 area => coordinates range from -4 to +4
binding_time_ms = 300  # Binding time in ms
brightness = 2 # Fluorophore brightness in photons/ms
bg_rate = 4 # Background counts in photons/pixel/200ms
sigma_psf = 1.1  # Gaussian width (standard deviation)
camera_error = 0.29 # Camera error when assigning pixels
min_cam_binning = 1/16 # Smallest unit camera can theoretically resolve (12 bit ~ 4096 pixel)
fitting_diameter = 4 # diameter considered for fitting
max_dist = fitting_diameter/2 # radius from point of interest to be considered

num_photons = brightness * binding_time_ms

__all__ = ["num_pixels",
           "binding_time_ms",
           "brightness",
           "bg_rate",
           "sigma_psf",
           "camera_error",
           "min_cam_binning",
           "fitting_diameter",
           "num_photons",
           "max_dist"]

from simulate.analyze_event import distance_to_point
from simulate.simulate_bg import simulate_background
from simulate.simulate_fluorophore import simulate_fluorophore
from simulate.simulate_event_statistics import simulate_filtered_events