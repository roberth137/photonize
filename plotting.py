import event
import get_photons
import helper
import matplotlib.pyplot as plt
import numpy as np

events_filename = 't/orig_7mle_pf_event.hdf5'
photons_filename = 't/10x_photons.hdf5'
drift_filename = 't/orig_7mle_drift.txt'

events = helper.process_input(events_filename, 'locs')
photons = helper.process_input(photons_filename, 'photons')
drift = helper.process_input(drift_filename, 'drift')

diameter = 6
group_events = events[events.group == 1]
group_events.reset_index(drop=True, inplace=True)

pick_photons = get_photons.get_pick_photons(group_events, photons, drift, 4, diameter, 200)

all_events_photons = get_photons.photons_of_many_events(group_events, pick_photons,diameter)



def plot_event(i):
    this_event = group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event, all_events_photons, diameter)
    print(this_event_photons)

    plt.figure(figsize=(8, 6))
    plt.scatter(this_event_photons['x'], this_event_photons['y'], c='blue', alpha=0.5, s=20)
    plt.title("Scatter Plot of x, y Positions from DataFrame")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
def hist_x_event(i): 
    this_event = group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event, all_events_photons, diameter)
    print(this_event_photons)
    bin_size=0.05
    bins = np.arange(min(this_event_photons.x), max(this_event_photons.x) + bin_size, bin_size)
    plt.figure(figsize=(8, 6))
    plt.hist(this_event_photons['x'], 
             bins=bins)
    plt.title("Histogram of x position")
    plt.xlabel("X Position")
    #plt.ylabel("Y Position")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()