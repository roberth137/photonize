# This is a file for preparing fluorescent decay data for ML applications
# Every decay gets histogrammed and this histogram values saved in a hdf5
# with the corresponding label

import pandas as pd
import numpy as np
import helper
import get_photons
import fitting
import h5py

input_events = '/Users/roberthollmann/Desktop/resi-flim/int_test/cy3_200ms_fp_event.hdf5'
input_photons = '/Users/roberthollmann/Desktop/resi-flim/int_test/cy3_59_index.hdf5'
drift_file = '/Users/roberthollmann/Desktop/resi-flim/int_test/cy3_200ms_drift.txt'
# For start: R1 Cy3 is 0, R2 A550 is 1, R4, A565 is 2
fluorophore_name = 'a565'
fluorophore_number = 2
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


histograms = pd.DataFrame()

delta_x, delta_y = 0, 0
delta_phot = 0
delta_phot_2 = 0
for group in events['group'].unique():
    events_group = events[events.group == group]
    print('_______________')
    print('this is group: ', group)
    print('number of events in group: ', len(events_group))
    pick_photons = get_photons.get_pick_photons(events_group, photons,
                                            drift, offset,
                                            box_side_length=diameter,
                                            int_time=int_time)
    # now photons are undrifted
    #print(f'event bounds [x],[y]: [{min(events_group.x):.2f}, {max(events_group.x):.2f}], [{min(events_group.y):.2f}, {max(events_group.y):.2f}]')
    #print(f'photon bounds [x],[y]: [{min(pick_photons.x):.2f}, {max(pick_photons.x):.2f}], [{min(pick_photons.y):.2f}, {max(pick_photons.y):.2f}]')
    print(f'event photons | filtered photons')
    for i, event in events_group.iterrows():
        event_photons = get_photons.crop_event(event, pick_photons, diameter, 300)
        if i % 4 ==0: print(f'{int(event.photons)}  |  {len(event_photons)}')
        x_new, y_new = fitting.event_position(event, event_photons, diameter, False)
        #if i % 5 ==0:print(f'delta_x: {(event.x-x_new):.2f}, delta_y: {(event.y-y_new):.2f}')
        delta_x += (event.x-x_new)
        delta_y += (event.y-y_new)
        #print('len event_photons: ', len(event_photons))
        hist, _ = np.histogram(event_photons.dt, bins=bins)

        delta_phot += event.photons - len(event_photons)
        delta_phot_2 += (event.photons - len(event_photons))**2
        hist  = np.log1p(hist)/3
        hist_df = pd.DataFrame([hist])

        histograms = pd.concat([histograms, hist_df], axis=0, ignore_index=True)
avg_delta_x = delta_x/len(events)
avg_delta_y = delta_y/len(events)
avg_delta_phot = delta_phot/len(events)
std_phot = np.sqrt(delta_phot_2/len(events))
print(f'delta_x: {(event.x-x_new):.2f}, delta_y: {(event.y-y_new):.2f}')
print(f'delta phot mean: {avg_delta_phot}, std_phot: {std_phot}')

histograms['label'] = fluorophore_number

print(histograms.head())

with h5py.File(output_filename, 'w') as h5f:
    h5f.create_dataset('hist', data=histograms)

print("Histograms saved as .hdf5 successfully.")