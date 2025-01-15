import matplotlib
matplotlib.use('Qt5Agg')
import helper
import get_photons

# Set filenames and parameters
events_filename = 't/NUPS_pf_more_F20_event.hdf5'
photons_filename = 't/NUP20k_index.hdf5'
drift_filename = 't/drift_20k.txt'
diameter = 4
pick_group = 0

# Loading data to memory
print("Loading events and photons...")
events = helper.process_input(events_filename, 'locs')
photons = helper.process_input(photons_filename, 'photons')
drift = helper.process_input(drift_filename, 'drift')

# Load photons of events
group_events = events[events.group == pick_group]
group_events.reset_index(drop=True, inplace=True)

pick_photons = get_photons.get_pick_photons(group_events,
                                            photons,
                                            drift,
                                            10,
                                            diameter,
                                            200)
all_events_photons = get_photons.photons_of_many_events(group_events,
                                                        pick_photons,
                                                        diameter,
                                                        400)

# Make loaded data available throughout the package
__all__ = ["events",
           "photons",
           "drift",
           "group_events",
           "pick_photons",
           "all_events_photons",
           "diameter"]

# Make plotting functions available (after making data available, otherwise circular import)
from .plot_functions import (hist_ms_event,
                             plot_all_dt,
                             scatter_event,
                             hist_dt_event,
                             hist_x_event)