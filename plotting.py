import event
import fitting
import get_photons
import helper
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

events_filename = 't/orig58_pf_event.hdf5'
photons_filename = 't/orig58_index.hdf5'
drift_filename = 't/orig58_drift.txt'

events = helper.process_input(events_filename, 'locs')
photons = helper.process_input(photons_filename, 'photons')
drift = helper.process_input(drift_filename, 'drift')

diameter = 15
group_events = events[events.group == 1]
group_events.reset_index(drop=True, inplace=True)

pick_photons = get_photons.get_pick_photons(group_events, photons, drift, 10, diameter, 200)

all_events_photons = get_photons.photons_of_many_events(group_events, pick_photons,diameter)



def plot_event(i):
    this_event = group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event, all_events_photons, diameter)
    print(this_event_photons)

    prev_x = this_event.x
    prev_y = this_event.y
    new_x, new_y = fitting.event_position(this_event, this_event_photons, diameter,False)
    new_x_bg, new_y_bg = fitting.event_position_w_bg(this_event, this_event_photons, diameter, False)

    plt.figure(figsize=(8, 8))
    plt.scatter(this_event_photons['x'], this_event_photons['y'], c='orange', alpha=0.5, s=20)
    plt.plot(prev_x, prev_y, 'o', label=f'Prev: ({prev_x}, {prev_y})', color='blue')
    #plt.plot(new_x, new_y, 's', label=f'New: ({new_x}, {new_y})', color='green')
    plt.plot(new_x_bg, new_y_bg, '^', label=f'New BG: ({new_x_bg}, {new_y_bg})', color='red')
    plt.plot([], [], ' ', label=f'Total number of photons: {len(this_event_photons)}')
    plt.title("Scatter Plot of x, y Positions from DataFrame")
    plt.title('Data Points with Legend')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')  # Adjust the legend position if needed
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