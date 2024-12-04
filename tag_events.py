#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:21:46 2024

This module reads in picasso localizations and tags them with their event

@author: roberthollmann
"""

import pandas as pd
import numpy as np
import core

def connect_locs(localizations):
    #localizations = pd.read_hdf(localizations_file, key='locs')
    event = np.zeros(len(localizations), dtype=int)
    event_counter = 0
    
    for i in np.arange(len(localizations), dtype=int): 
        has_follower = False
        
        if event[i] == 0: #not connected to previous event -> new event
            event_counter +=1 
            event[i] = event_counter
                    
        frame = localizations.frame.iloc[i]
        locs_next_frame = localizations[(localizations.frame == frame+1)]
        
        if len(locs_next_frame) != 0:
            has_follower, follower_index = return_nearby(
                localizations.iloc[i], locs_next_frame)
        
        if has_follower: 
            event[follower_index] = event[i] #set to same event
    
    if 'event' in localizations.columns:
        localizations = localizations.drop('event', axis=1)   
    
    localizations.insert(4, 'event', event)
    #filtered_locs = filter_unique_events(locs_event_tagged)
    return localizations
    


def connect_locs_to_picasso(localizations_file):
    localizations = pd.read_hdf(localizations_file, key='locs')
    event = np.zeros(len(localizations), dtype=int)
    event_counter = 0
    
    for i in np.arange(len(localizations), dtype=int): 
        has_follower = False
        
        if event[i] == 0: #not connected to previous event -> new event
            event_counter +=1 
            event[i] = event_counter
                    
        frame = localizations.frame.iloc[i]
        locs_next_frame = localizations[(localizations.frame == frame+1)]
        
        if len(locs_next_frame) != 0:
            has_follower, follower_index = return_nearby(
                localizations.iloc[i], locs_next_frame)
        
        if has_follower: 
            event[follower_index] = event[i] #set to same event
            
    localizations.insert(4, 'event', event)
    #filtered_locs = filter_unique_events(locs_event_tagged)
    core.dataframe_to_picasso(localizations, localizations_file, '_eve')
    
    
    
def return_nearby(localization, locs_next_frame):
    '''

    Parameters
    ----------
    localization : one localization
    locs_next_frame : all locs in next frame

    Returns
    -------
    If the localization has a successor in the next frame

    '''
    
    has_next = False
    
    locs_next = locs_next_frame.copy()
    
    max_distance = (localization.lpx+localization.lpy)
    
    x_distance = (locs_next['x'].to_numpy() - localization.x)
    y_distance = (locs_next['y'].to_numpy() - localization.y)
    
    total_distance_sq = np.square(x_distance) + np.square(y_distance)

    locs_next['distance'] = total_distance_sq
    
    radius_sq = max_distance**2
    
    #print('max_distance:', radius_sq)
    #print(locs_next.distance)
    
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
      event_column: The name of the column containing the events to filter.
    
    Returns:
      A new DataFrame with the filtered rows.
    """
    
    # Extract unique event values and their counts
    event_counts = localizations['event'].value_counts()
    
    # Identify events that occur only once
    unique_events = event_counts[event_counts == 1].index
    
    # Filter out rows with unique events
    filtered_locs = localizations[~localizations['event'].isin(unique_events)]
    
    print('removed ', (len(localizations)-len(filtered_locs)), 'localizations.')
    return filtered_locs



    
    
    