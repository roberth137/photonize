import pandas as pd
import numpy as np
import helper
import get_photons
import fitting

input_events = '/Users/roberthollmann/Desktop/resi-flim/t/c3.hdf5'
input_photons = '/Users/roberthollmann/Desktop/resi-flim/t/orig58_index.hdf5'
drift_file = '/Users/roberthollmann/Desktop/resi-flim/t/orig58_drift.txt'
fluorophore = 'Cy3'
offset = 10
diameter = 4.5
int_time = 200

events = helper.process_input(input_events, 'locs')
photons = helper.process_input(input_photons, 'photons')
drift = helper.process_input(drift_file, 'drift')

peak_arrival_time = fitting.calibrate_peak_events(photons[:1000000])
max_dt = max(photons[:1000000].dt)

# Parameters
bin_size = 20  # Bin size for histogramming (in the same units as dt)
bins = np.arange(peak_arrival_time, max_dt, bin_size)

output_file = "histogram_data.csv"

histograms = pd.DataFrame()

for group in events['group'].unique():
    events_group = events[events.group == group]
    print('this is group: ', group)
    print('number of events in group: ', len(events_group))
    pick_photons = get_photons.get_pick_photons(events_group, photons,
                                            drift, offset,
                                            box_side_length=diameter,
                                            int_time=int_time)
    # now photons are undrifted

    for i, event in events_group.iterrows():
        event_photons = get_photons.crop_cylinder(event.x,
                                             event.y,
                                             event.s_ms_new,
                                             event.e_ms_new,
                                             diameter,
                                             pick_photons)
        #print('len event_photons: ', len(event_photons))
        hist, _ = np.histogram(event_photons.dt, bins=bins)

        hist = hist / hist.max() if hist.max() > 0 else hist
        hist_df = pd.DataFrame([hist])

        histograms = pd.concat([histograms, hist_df], axis=0, ignore_index=True)

    histograms['label'] = fluorophore

    print(histograms.head())



# 1. Read in events (only 1 group to start)

# 2. Undrift photons

# 3. Iterate events
    # For every event: Get photons
    # Histogram photons
    # Add counts to dataframe

# 4. Label data

# 5. Save file
