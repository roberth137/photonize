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
    localizations = helper.process_input(localizations_dset, 'locs')
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