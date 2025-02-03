#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:21:46 2024

This module reads in picasso localizations and tags them with their event

@author: roberthollmann
"""

import numpy as np
import pandas as pd
import helper
from numba import njit
import time

def connect_locs_picasso_new(localizations_file, filter_single=True, proximity=2, max_dark_frames=1, suffix=''):
    localizations = helper.process_input(localizations_file, 'locs')
    localizations = connect_locs_by_group(localizations, filter_single=filter_single, proximity=proximity, max_dark_frames=max_dark_frames)
    helper.dataframe_to_picasso(localizations, localizations_file, f'event_tagged{suffix}')
#@njit
def are_nearby(x1, y1, x2, y2, threshold):
    dx = x1 - x2
    dy = y1 - y2
    return (dx * dx + dy * dy) <= (threshold * threshold)

#@njit
def connect_group(group_frames, group_x, group_y, lpx_p_lpy, max_dark_frames):
    n = len(group_frames)
    event_ids = np.zeros(n, dtype=np.int32)
    current_event_id = 0

    for i in range(n):
        if event_ids[i] == 0:  # If not yet assigned to an event
            current_event_id += 1
            event_ids[i] = current_event_id

        # Compare with localizations in the next frames within max_dark_frames
        for j in range(i + 1, n):
            if group_frames[j] > group_frames[i] + 1 + max_dark_frames:
                break  # Stop if frames are beyond the allowed gap

            if group_frames[i] < group_frames[j] <= group_frames[i] + 1 + max_dark_frames:
                if are_nearby(group_x[i], group_y[i], group_x[j], group_y[j], lpx_p_lpy[i]):
                    event_ids[j] = event_ids[i]  # Assign the same event ID

    return event_ids

def connect_locs_by_group(localizations_dset,
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
        group_event_ids = connect_group(frames, x_coords, y_coords, proximity * lpx_p_lpy, max_dark_frames)

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


def connect_locs(localizations_dset, filter_single=True, box_side_length=5):
    """

    Parameters
    ----------
    localizations_dset : picked localizations
    filter_single : default= True
        - if single localizations which cant be connected to an event are dropped
    box_side_length : The default is 5.

    Returns
    -------
    localizations with additional columns event and count

    """
    localizations = helper.process_input(localizations_dset, 'locs').copy()
    event_column = np.zeros(len(localizations), dtype=int)
    event_counter = 0

    grouped = localizations.groupby('group', sort=False)

    for group_id, group_df in grouped:
        idxs = group_df.index.to_numpy()

    for i in np.arange(len(localizations), dtype=int):
        has_follower = False

        if event_column[i] == 0:  # not connected to previous event -> new event
            event_counter += 1
            event_column[i] = event_counter

        frame = localizations.frame.iloc[i]
        group = localizations.group.iloc[i]
        locs_next_frame = localizations[(localizations.frame == frame + 1)
                                        & (localizations.group == group)]
        follower_index = None
        if len(locs_next_frame) != 0:
            has_follower, follower_index = return_nearby(
                localizations.iloc[i], locs_next_frame)

        if has_follower:
            event_column[follower_index] = event_column[i]  # set to same event

    if 'event' in localizations.columns:
        localizations = localizations.drop('event', axis=1)

    event_counts = count_localizations(event_column)
    localizations.insert(4, 'event', event_column)
    localizations.insert(5, 'count', event_counts)
    helper.calculate_total_photons(localizations, 5)
    if filter_single:
        filtered_locs = filter_unique_events(localizations)
        return filtered_locs
    else:
        return localizations
    
def return_nearby(this_localization, locs_next_frame):
    """
    Parameters
    ----------
    this_localization : one localization
    locs_next_frame : all locs in next frame

    Returns
    -------
    If the localization has a successor in the next frame
    """
    
    has_next = False
    locs_next = locs_next_frame.copy()
    max_distance = 3*(this_localization.lpx+this_localization.lpy)
    
    x_distance = (locs_next['x'].to_numpy() - this_localization.x)
    y_distance = (locs_next['y'].to_numpy() - this_localization.y)
    total_distance_sq = np.square(x_distance) + np.square(y_distance)

    locs_next['distance'] = total_distance_sq
    radius_sq = max_distance**2

    adjacent = locs_next[
        locs_next.distance < radius_sq]

    if len(adjacent) == 1:
        #print('similar loc in next frame.')
        has_next = True
        return has_next, adjacent.index.values
    
    elif len(adjacent) > 1:
        #print('too many locs')
        #print('max_distance: ', max_distance)
        #print(adjacent)
        #print('\n-----returning smallest element: ')
        #print(adjacent.loc[adjacent['distance'].idxmin()])
        return has_next, adjacent['distance'].idxmin()
    
    else:
        #print('Number of locs in next frame were: ', len(locs_next), 
        #      'no similar loc in next frame.')
        return has_next, float('nan')
    
    
    
def filter_unique_events(localizations):
    """Filters a DataFrame, removing localizations 
    that are not connected to other events.
    
    Args:
      localizations: The DataFrame to filter.
    Returns:
      A new DataFrame with the filtered rows.
    """
    
    # Extract unique event values and their counts
    event_counts = localizations['event'].value_counts()
    
    # Identify events that occur only once
    unique_events = event_counts[event_counts == 1].index
    
    # Filter out rows with unique events
    filtered_locs = localizations[~localizations['event'].isin(unique_events)]
    
    print('removed ', (len(localizations)-len(filtered_locs)), 'single frame localizations.')
    return filtered_locs


def calculate_total_photons(localizations, box_side_length):
    if {'photons', 'bg'}.issubset(localizations.columns):
        photons_arr = localizations['photons'].to_numpy()
        bg_arr = localizations['bg'].to_numpy()
        total_photons = photons_arr + (bg_arr * box_side_length ** 2)
        localizations.insert(5, 'total_photons', total_photons)
        return localizations
    else:
        raise ValueError("DataFrame must contain 'photons', 'bg', and 'roi' columns.")

def count_localizations(events):
    """
    Count the number of localizations per event.

    Parameters:
    events (numpy array): Array of event labels.

    Returns:
    numpy array: Array where each element represents the number of localizations per event.
    """
    # Get unique events and their counts
    unique_events, counts = np.unique(events, return_counts=True)

    # Create an array mapping events to counts
    number_locs = np.zeros_like(events, dtype=int)

    # Fill number_locs array
    for event, count in zip(unique_events, counts):
        number_locs[events == event] = count

    return number_locs

def connect_locs_to_picasso(localizations_file, box_side_length=5):
    localizations = helper.process_input(localizations_file, 'locs')
    event_column = np.zeros(len(localizations), dtype=int)
    event_counter = 0
    
    for i in np.arange(len(localizations), dtype=int): 
        has_follower = False
        
        if event_column[i] == 0: #not connected to previous event -> new event
            event_counter +=1 
            event_column[i] = event_counter
                    
        frame = localizations.frame.iloc[i]
        group = localizations.group.iloc[i]
        locs_next_frame = localizations[(localizations.frame == frame+1)
                                        &(localizations.group == group)]

        follower_index = None
        if len(locs_next_frame) != 0:
            has_follower, follower_index = return_nearby(
                localizations.iloc[i], locs_next_frame)
        
        if has_follower: 
            event_column[follower_index] = event_column[i] #set to same event

    event_counts = count_localizations(event_column)
    localizations.insert(4, 'event', event_column)
    localizations.insert(5, 'count', event_counts)
    helper.calculate_total_photons(localizations, 5)
    #filtered_locs = filter_unique_events(locs_event_tagged)
    helper.dataframe_to_picasso(localizations, localizations_file, '_event_tagged')