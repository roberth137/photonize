import event
import fitting
import get_photons
import helper
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

events_filename = 't/orig58_pf_eventtest.hdf5'
photons_filename = 't/orig58_index.hdf5'
drift_filename = ('t/orig58_drift.txt')

events = helper.process_input(events_filename, 'locs')
photons = helper.process_input(photons_filename, 'photons')
drift = helper.process_input(drift_filename, 'drift')

diameter = 4
group_events = events[events.group == 0]
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
                                                        200)


def hist_ms_event(i):
    this_event = group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event,
                                                all_events_photons,
                                                diameter,
                                                200)
    print(this_event_photons)
    bin_size=5
    bins = np.arange(min(this_event_photons.ms), max(this_event_photons.ms) + bin_size, bin_size)
    plt.figure(figsize=(8, 6))
    plt.hist(this_event_photons['ms'],
             bins=bins)
    plt.plot([], [], ' ', label=f'Total number of photons: {len(this_event_photons)}')
    plt.plot([], [], ' ', label=f'Start_ms: {this_event.start_ms}, End_ms: {this_event.end_ms}')
    plt.axvline(this_event.start_ms, color='red')
    plt.axvline(this_event.end_ms, color='red')
    plt.title("Histogram of ms")
    plt.xlabel("ms of arrival")
    #plt.ylabel("Y Position")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')  # Adjust the legend position if needed
    plt.show()



def plot_all_dt(all_events_photons):

    bin_size = 10
    bins = np.arange(min(all_events_photons.dt), max(all_events_photons.dt) + bin_size, bin_size)
    plt.figure(figsize=(8, 6))
    plt.hist(all_events_photons['dt'], bins=bins)
    plt.plot([], [], ' ', label=f'Total number of photons: {len(all_events_photons)}')
    plt.title("Histogram of dt values")
    plt.xlabel("arrival time ")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')  # Adjust the legend position if needed
    plt.show()

def plot_event(i):
    this_event = group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event, all_events_photons, diameter)
    print(this_event_photons)

    prev_x = this_event.x
    prev_y = this_event.y
    new_x, new_y = fitting.event_position_w_bg(this_event, this_event_photons, diameter, False)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(this_event_photons['x'],
                this_event_photons['y'],
                c=this_event_photons['ms'],
                cmap='viridis')
    cbar = plt.colorbar(scatter)
    cbar.set_label('ms value', rotation=270, labelpad=15)
    plt.plot(prev_x, prev_y, 'o', label=f'Prev: ({prev_x}, {prev_y})', color='blue')
    plt.plot(new_x, new_y, '^', label=f'New Pos: ({new_x}, {new_y})', color='red')
    plt.plot([], [], ' ', label=f'Total number of photons: {len(this_event_photons)}')
    plt.title("Scatter Plot of x, y Positions from DataFrame")
    plt.title('Data Points with Legend')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.yscale("log")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')  # Adjust the legend position if needed
    plt.show()

def hist_dt_event(i):
    this_event = group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event, all_events_photons, diameter)

    bin_size = 5
    bins = np.arange(min(this_event_photons.dt), max(this_event_photons.dt) + bin_size, bin_size)
    plt.figure(figsize=(8, 6))
    plt.hist(this_event_photons['dt'], bins=bins)
    plt.plot([], [], ' ', label=f'Total number of photons: {len(this_event_photons)}')
    plt.title("Histogram of dt values")
    plt.xlabel("arrival time ")
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