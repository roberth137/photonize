import matplotlib
matplotlib.use('Qt5Agg')
from utilities import helper
import get_photons
import fitting

# Set filenames and parameters
events_filename = '/Users/roberthollmann/Desktop/resi-flim/t/orig58_pf_event_legend.hdf5'
photons_filename = '/Users/roberthollmann/Desktop/resi-flim/t/orig58_index.hdf5'
drift_filename = '/Users/roberthollmann/Desktop/resi-flim/t/orig58_drift.txt'
diameter = 4.5
pick_group = 1
more_ms = 400

# Loading data to memory
print("Loading events and photons...")
events = helper.process_input(events_filename, 'locs')
photons = helper.process_input(photons_filename, 'photons')
drift = helper.process_input(drift_filename, 'drift')

# Load photons of events
group_events = events[events.group == pick_group]
group_events.reset_index(drop=True, inplace=True)

peak_arrival_time = fitting.calibrate_peak_events(photons[:500000])
pick_photons = get_photons.get_pick_photons(group_events,
                                            photons,
                                            drift,
                                            10,
                                            diameter,
                                            200)


print('peak arrival time is: ', peak_arrival_time)
# Make loaded data available throughout the package
__all__ = ["events",
           "photons",
           "drift",
           "group_events",
           "pick_photons",
           "diameter",
           "peak_arrival_time"]

# Make plotting functions available (after making data available, otherwise circular import)
from plotting_2.plot_functions import (plot_all_dt,
                                       scatter_event,
                                       hist_dt_event,
                                       hist_x_event,
                                       hist_noise_dt_event)