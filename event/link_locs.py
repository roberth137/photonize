#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:21:46 2024

This module reads in picasso localizations and tags them with their event

@author: roberthollmann
"""

import numpy as np
import pandas as pd
from utilities import helper


#@njit
def are_nearby(x1, y1, x2, y2, threshold):
    """
    Determines if 2 points are nearby based on Euclidean distance
    Numba optimized

    Input:
        coordinates: x1, y1, x2, y2 (float)
        threshold: max allowed distance
    Output: True if distance is smaller than threshold, False if larger
    """
    dx = x1 - x2
    dy = y1 - y2
    return (dx * dx + dy * dy) <= (threshold * threshold)

#@njit
def link_group(group_frames, group_x, group_y, max_distance, max_dark_frames):
    """
    Link localizations to events within a group

    Parameters:
    group_frames : 1D array, localizations frames (int)
    group_x : 1D array, localizations x-coordinates (float)
    group_y : 1D array, localizations y-coordinates (float)
    lpx_p_lpy : 1D array, distance threshold for every localization (typically proximity * lpx+lpy)
    max_dark_frames : int, max number of frames that can be skipped while still considering localizations as part the same event.

    Returns:
    event_ids: np.ndarray, A 1D array of event IDs for each localization. Localizations connected into the same event share
        the same event ID.
    """
    n = len(group_frames) #len localizations
    event_ids = np.zeros(n, dtype=np.int32)
    current_event_id = 0

    # Loop over all group localizations
    for i in range(n):

        if event_ids[i] == 0:  # If not yet assigned to an event
            current_event_id += 1
            event_ids[i] = current_event_id

        # Compare with localizations in the next frames within max_dark_frames
        for j in range(i + 1, n):
            if group_frames[j] > group_frames[i] + 1 + max_dark_frames:
                break  # Stop if frames are beyond the allowed gap

            if group_frames[i] < group_frames[j] <= group_frames[i] + 1 + max_dark_frames:
                if are_nearby(group_x[i], group_y[i], group_x[j], group_y[j], max_distance[i]):
                    event_ids[j] = event_ids[i]  # Assign the same event ID
                    break

    return event_ids


def link_locs_by_group(localizations_dset,
                          filter_single=True,
                          proximity=2,
                          max_dark_frames=1):
    """
    Connect localizations in adjacent or nearby frames (up to max_dark_frames) into events, iterating group by group.

    Parameters
    ----------
    localizations_dset : pd.DataFrame
        DataFrame of localizations with columns: frame, group, x, y, etc.
    filter_single : bool, default=True
        If True, remove single localizations that cannot be connected to any multi-localization event.
    proximity : float, default=2
        Multiplier for the maximum distance threshold for connecting localizations.
    max_dark_frames : int, default=1
        Maximum number of frames allowed to be skipped when connecting localizations.

    Returns
    -------
    pd.DataFrame
        A copy of the localizations with additional columns:
          - 'event': an integer label for the connected event
          - 'count': total number of localizations in that event
    """
    localizations = localizations_dset.copy()
    localizations.insert(1, 'event', 0)  # Initialize event column

    # Group by 'group' and iterate
    grouped = localizations.groupby('group', sort=False)

    # Global event counter
    global_event_id = 0
    for group_id, group_df in grouped:
        # Convert necessary columns to NumPy arrays for Numba
        frames = group_df['frame'].to_numpy()
        x_coords = group_df['x'].to_numpy()
        y_coords = group_df['y'].to_numpy()
        lpx_p_lpy = (group_df['lpx'] + group_df['lpy']).to_numpy()

        # Sort by frame to ensure temporal order
        sort_indices = np.argsort(frames)
        frames = frames[sort_indices]
        x_coords = x_coords[sort_indices]
        y_coords = y_coords[sort_indices]
        lpx_p_lpy = lpx_p_lpy[sort_indices]

        # Call Numba-optimized function to assign event IDs
        group_event_ids = link_group(frames,
                                     x_coords,
                                     y_coords,
                                     proximity * lpx_p_lpy,
                                     max_dark_frames)

        # Map local group event IDs to global event IDs
        unique_local_event_ids = np.unique(group_event_ids)
        local_to_global_map = {local_id: global_event_id + idx + 1 for idx, local_id in enumerate(unique_local_event_ids)}
        global_event_id += len(unique_local_event_ids)

        # Assign the global event IDs back to the original DataFrame
        localizations.loc[group_df.index[sort_indices], 'event'] = [local_to_global_map[e] for e in group_event_ids]

    # Count the number of localizations per event
    localizations.insert(2, 'count', localizations['event'].map(localizations['event'].value_counts()))

    # Optionally filter out single-localization events
    if filter_single:
        locs_before = len(localizations)
        localizations = localizations[localizations['count'] > 1].reset_index(drop=True)
        locs_after = len(localizations)
        print(f'removed {locs_before-locs_after} single frame localizations.')

    return localizations

def link_locs_picasso(localizations_file,
                             filter_single=True,
                             proximity=2,
                             max_dark_frames=1,
                             suffix=''):
    localizations = helper.process_input(localizations_file, 'locs')
    localizations = link_locs_by_group(localizations,
                                          filter_single=filter_single,
                                          proximity=proximity,
                                          max_dark_frames=max_dark_frames)
    helper.dataframe_to_picasso(localizations, localizations_file, f'_event_tagged{suffix}')
