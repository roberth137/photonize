#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:21:46 2024

This module reads in picasso localizations and tags them with their event

@author: roberthollmann
"""

import numpy as np
import helper

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
    
    if 'event' in localizations.columns:
        localizations = localizations.drop('event', axis=1)


    event_counts = count_localizations(event_column)
    localizations.insert(4, 'event', event_column)
    localizations.insert(5, 'count', event_counts)
    helper.calculate_total_photons(localizations, 5)
    if filter_single:
        filtered_locs = filter_unique_events(localizations)
        return filtered_locs
    else: return localizations
    
def connect_locs_by_group(localizations_dset, filter_single=True, box_side_length=5):
    """
    Connect localizations in adjacent frames into events, iterating group by group.

    Parameters
    ----------
    localizations_dset : pd.DataFrame
        DataFrame of localizations with (at least) columns:
          - frame (int)
          - group (int)
          - x, y (floats, or whatever coordinates you use)
          - ...
    filter_single : bool, default=True
        If True, remove single localizations that cannot be connected to any multi-localization event.
    box_side_length : float, default=5
        Maximum distance threshold for connecting localizations in consecutive frames.

    Returns
    -------
    pd.DataFrame
        A copy of the localizations with additional columns:
          - 'event': an integer label for the connected event
          - 'count': total number of localizations in that event
          - possibly other columns as needed
    """
    localizations = localizations_dset.copy()
    
    # Prepare an array to store the event ID for each localization
    event_column = np.zeros(len(localizations), dtype=int)
    
    # Global counter that increments each time we discover a new event
    event_counter = 0
    
    # Group by 'group' (which implies spatial proximity) and iterate
    grouped = localizations.groupby('group', sort=False)
    print(f'connecting {len(localizations)} localizations of {len(grouped)} groups.')
    for group_id, group_df in grouped:
        if group_id % 10 == 0 and group_id != 0: print(f'{group_id} of {len(grouped)} groups connected.')
        # Extract the row indices for this group (so we can assign event IDs correctly)
        idxs = group_df.index.to_numpy()
        
        # Sort the group’s localizations by frame to process in temporal order
        group_df_sorted = group_df.sort_values(by='frame')
        sorted_idxs = group_df_sorted.index.to_numpy()
        
        # We’ll keep track of each row’s event ID in this group separately at first
        group_event_ids = np.zeros(len(sorted_idxs), dtype=int)
        
        # Loop through localizations in ascending frame order
        for i, row_idx in enumerate(sorted_idxs):
            if group_event_ids[i] == 0:
                # If this localization does not yet belong to an event, start a new event
                event_counter += 1
                group_event_ids[i] = event_counter
            
            # Compare with localizations in the next frame for the same group
            current_frame = group_df.loc[row_idx, 'frame']
            
            # We only look at localizations in the subsequent positions (since sorted by frame)
            # that have frame == current_frame + 1
            j = i + 1
            while j < len(sorted_idxs):
                next_row_idx = sorted_idxs[j]
                next_frame = group_df.loc[next_row_idx, 'frame']
                
                # If the next localization’s frame is more than 1 ahead, break early
                if next_frame > current_frame + 1:
                    break
                
                # If the next localization’s frame is exactly current_frame + 1, check distance
                if next_frame == current_frame + 1:
                    # Check if they are within distance
                    if are_nearby(group_df.loc[row_idx], group_df.loc[next_row_idx], box_side_length):
                        # Assign the same event ID
                        group_event_ids[j] = group_event_ids[i]
                j += 1
        
        # Now, assign these group_event_ids back to the main event_column array
        for k, row_idx in enumerate(sorted_idxs):
            event_column[row_idx] = group_event_ids[k]
    
    # Insert the 'event' column
    localizations['event'] = event_column
    
    # Optionally count how many localizations belong to each event
    # (a simple way is to do value_counts on 'event' and then map back)
    event_counts = localizations['event'].value_counts()
    localizations['count'] = localizations['event'].map(event_counts)
    
    # (Optional) Filter out single-localization events
    if filter_single:
        localizations = localizations[localizations['count'] > 1]
        localizations = localizations.reset_index(drop=True)
    
    # Return your final result
    return localizations
    
# We will define a helper function to check if two localizations are “nearby”
def are_nearby(loc1, loc2, threshold):
    """
    Returns True if loc1 and loc2 are within 'threshold' distance in (x,y), else False.
    """
    dx = loc1['x'] - loc2['x']
    dy = loc1['y'] - loc2['y']
    return (dx*dx + dy*dy) <= (threshold * threshold)
    
    
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