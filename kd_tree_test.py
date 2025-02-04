import time

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

events = pd.read_hdf('t/orig58_pf_event.hdf5', key='locs')
photons = pd.read_hdf('t/orig58_index.hdf5', key='photons')

photons = photons[::100]

events = events[events.group==0]


def process_localizations_kd(localizations, photons, radius=2):
    # Step 1: Build spatial index on all photons
    photon_coords = photons[['x', 'y']].to_numpy()
    photon_tree = cKDTree(photon_coords)

    # Preallocate array for results
    avg_dt_results = np.empty(len(localizations), dtype=np.float32)

    # Step 2: Loop through localizations (ordered by time)
    for i, loc in localizations.iterrows():
        # Get time window and localization position
        start_ms, end_ms = loc['start_ms'], loc['end_ms']
        loc_x, loc_y = loc['x'], loc['y']

        # Step 3: Filter photons by time window
        time_filtered_photons = photons[(photons['ms'] >= start_ms) & (photons['ms'] <= end_ms)]

        # If no photons found in the time window, skip
        if time_filtered_photons.empty:
            avg_dt_results[i] = np.nan
            continue

        # Step 4: Get global indices within 2-pixel radius using KD-tree
        global_indices = photon_tree.query_ball_point([loc_x, loc_y], r=radius)

        # Step 5: Map global indices to filtered DataFrame using `.index`
        valid_indices = time_filtered_photons.index.intersection(global_indices)

        # Step 6: Get the corresponding rows of nearby photons
        nearby_photons = photons.loc[valid_indices]

        # Step 7: Compute average `dt` for the current localization
        if len(nearby_photons) > 0:
            avg_dt_results[i] = nearby_photons['dt'].mean()
        else:
            avg_dt_results[i] = np.nan  # No photons found in range

    # Step 8: Store the results back in the localizations DataFrame
    localizations['avg_dt'] = avg_dt_results
    return localizations

def process_localizations(localizations, photons, radius=2):
    # Step 1: Build spatial index on photons
    photon_coords = photons[['x', 'y']].to_numpy()
    photon_tree = cKDTree(photon_coords)

    # Preallocate array for results
    avg_dt_results = np.empty(len(localizations), dtype=np.float32)

    # Step 2: Loop through localizations (ordered by time)
    for i, loc in localizations.iterrows():
        # Get time window and localization position
        start_ms, end_ms = loc['start_ms'], loc['end_ms']
        loc_x, loc_y = loc['x'], loc['y']

        # Step 3: Filter photons by time window
        time_filtered_photons = photons[(photons['ms'] >= start_ms) & (photons['ms'] <= end_ms)]
        pos_filtered_photons = time_filtered_photons[(time_filtered_photons['x']>(loc_x-2))
                                                     &(time_filtered_photons['x']<(loc_x+2))
                                                     &(time_filtered_photons['y']>(loc_y-2))
                                                     &(time_filtered_photons['y']<(loc_y+2))]

        # Step 4: Query photons within 2-pixel radius
        #indices = photon_tree.query_ball_point([loc_x, loc_y], r=radius)
        #nearby_photons = time_filtered_photons.iloc[indices]

        # Step 5: Compute average `dt` for the current localization
        if len(pos_filtered_photons) > 0:
            avg_dt_results[i] = pos_filtered_photons['dt'].mean()
        else:
            avg_dt_results[i] = np.nan  # No photons found in range

    # Step 6: Store the results back in the localizations DataFrame
    localizations['avg_dt'] = avg_dt_results
    return localizations

start_kd = time.time()
events = process_localizations_kd(events, photons)
end_kd = time.time()
print('kd done')

start_base = time.time()
events_base = process_localizations(events, photons)
end_base = time.time()

print(f'kd took {end_kd-start_kd}s'
      f'base took {end_base-start_base}s')
