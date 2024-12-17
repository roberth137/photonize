import event
import get_photons
import helper
import matplotlib.pyplot as plt

events_filename = 't/'
photons_filename = ''
drift_filename = ''
event_number = 1

events = helper.process_input(events_filename, 'locs')
photons = helper.process_input(photons_filename, 'photons')
drift = helper.process_input(drift_filename, 'drift')

this_event = events.iloc[event_number]

diameter = 6
pick_photons = get_photons.photons_of_picked_area(events, photons_filename, )

all_events_photons = get_photons.photons_of_many_events(events, pick_photons,
                                                                diameter)
this_event_photons = get_photons.crop_event(this_event, all_events_photons, diameter)

plt.figure(figsize=(8, 6))
plt.scatter(this_event_photons['x'], this_event_photons['y'], c='blue', alpha=0.5, s=20)
plt.title("Scatter Plot of x, y Positions from DataFrame")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()