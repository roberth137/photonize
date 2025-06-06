from utilities import helper
import get_photons
import fitting

# Set filenames and parameters
events_filename = '/Users/roberthollmann/Desktop/resi-flim/t/orig58_pf_event.hdf5'
photons_filename = '/Users/roberthollmann/Desktop/resi-flim/t/orig58_index.hdf5'
drift_filename = '/Users/roberthollmann/Desktop/resi-flim/t/orig58_drift.txt'
diameter = 4.5
pick_group = 4
more_ms = 400

# Loading data to memory
print("Loading events and photons...")
events = helper.process_input(events_filename, 'locs')
photons = helper.process_input(photons_filename, 'photons')
drift = helper.process_input(drift_filename, 'drift')

# Load photons of events
group_events = events[events.group == pick_group]
group_events.reset_index(drop=True, inplace=True)

peak_arrival_time = fitting.calibrate_peak_arrival(photons[:500000])
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
from plotting.plot_ms_trace import hist_ms_event, hist_idw_ms_event
from plotting.plot_dt import hist_dt_event