##### This is a script preparing event data for classifying events based on their decay histogram
##### PREDICTION

import pandas as pd
import numpy as np
import helper
import get_photons
import fitting
import h5py

folder = '/Users/roberthollmann/Desktop/resi-flim/ml/event_data/'

input_events = f'{folder}cy3_200ms_fp_event_f.hdf5'
input_photons = f'{folder}cy3_59_index.hdf5'
drift_file = f'{folder}cy3_200ms_drift.txt'
# For start: R1 Cy3 is 0, R2 A550 is 1, R4, A565 is 2
fluorophore_name = 'cy3_1402_4p5'
#fluorophore_number = 0
offset = 10
diameter = 4.5
int_time = 200

output_filename = f'{fluorophore_name}_histogram.hdf5'

events = helper.process_input(input_events, 'locs')
photons = helper.process_input(input_photons, 'photons')
drift = helper.process_input(drift_file, 'drift')

peak_arrival_time = fitting.calibrate_peak_events(photons[:1000000])
max_dt = max(photons[:1000000].dt)

#events = events[events.group == 0]
# Parameters
bin_size = 20  # Bin size for histogramming (in the same units as dt)
bins = np.arange(peak_arrival_time, max_dt, bin_size)
print(f'peak arrival time: {peak_arrival_time}')
print(f'first bins: {bins[:5]}, . . . last bins: {bins[-5:]}')

num_bins = len(bins) - 1  # np.histogram returns bins+1 edges

# Define column names
column_names = [f'bin_{i}' for i in range(num_bins)] + ['event']

# Initialize an empty DataFrame with pre-defined columns
histograms = pd.DataFrame(columns=column_names)

# List to store individual histograms (faster than `pd.concat()` in loop)
histogram_list = []

# Initialize delta tracking variables
#delta_x, delta_y, delta_phot, delta_phot_2 = 0, 0, 0, 0
delta_phot = 0
missmatch = []
for group in events['group'].unique():
    events_group = events[events.group == group]

    print('_______________')
    print('this is group: ', group)
    print('number of events in group: ', len(events_group))

    pick_photons = get_photons.get_pick_photons(
        events_group, photons, drift, offset,
        diameter=diameter, int_time=int_time
    )

    print(f'event photons | filtered photons')

    # Process all events in this group
    for i, event in events_group.iterrows():
        event_photons = get_photons.crop_event(event, pick_photons, diameter)

        if i % 4 == 0:
            print(f'{int(event.photons)}  |  {len(event_photons)}')
        #if event.photons != len(event_photons):
        #    print(event)
        #    missmatch.append(event.event)
        delta_phot += (len(event_photons) - event.photons)
        # Compute histogram
        hist, _ = np.histogram(event_photons.dt, bins=bins)
        hist = np.log1p(hist) / 3
        hist = np.append(hist, event.event)#hist.append(event.event)
        # Append to list as dictionary (efficient for DataFrame creation)
        histogram_list.append({
            #'event_number': event.event,  # Assuming event.event holds the event number
            **dict(zip(column_names, hist))  # Histogram values mapped to columns
        })

# Convert list to DataFrame in one operation (much faster)
histograms = pd.DataFrame(histogram_list, columns=column_names)
histograms['event'] = histograms['event'].astype(np.int32)

# Display the first few rows
print(histograms.head())

print(f'now {delta_phot/len(events)} more photons per event')
print(missmatch)

labels = list(histograms.keys())
df_pd = histograms.reindex(columns=labels, fill_value=1)
hist_pd = df_pd.to_records(index=False)

hf = h5py.File(output_filename, 'w')
hf.create_dataset('hist', data=hist_pd)
hf.close()