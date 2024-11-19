#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:08:39 2024

@author: roberthollmann

This is a python script to connect individual localizations to events. 
It should be used only with filtered localizations
"""

import pandas as pd
import numpy as np
import core

class binding_event:
    def __init__(self,
                 x,
                 y,
                 number_locs,
                 frames,
                 start_ms, 
                 end_ms, 
                 num_photons,
                 bg):
        self.x = x
        self.y = y
        self.num_photons = num_photons
        self.frames = frames
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.num_photons = num_photons
        self.bg = bg
        
        
        
def connect_locs(localizations_file):
    localizations = pd.read_hdf(localizations_file, key='locs')
    #grouped_locs = localizations.copy
    #create event array 
    event = np.zeros(len(localizations), dtype=int)
    event_counter = 0
    deleted = False
    num_deleted = 0
    #loop over all localizations:
    for i in np.arange(len(localizations), dtype=int):
        print('\n\n----- NEW LOOP -----')
        print('loop number: ', i, ' starting...')
        print('deleted localizations: ', num_deleted)
        
        if deleted:
            num_deleted += 1
            event_counter -=1
            deleted = False
        position_counter = i - num_deleted
        
        if i == len(localizations)-1: print('last round')
        print('current event array: ', event[i-2:i+5])
        has_follower = False
        
        print('loop number: ', i, ' starting...')
        print('setting position: ', position_counter , '\n')
        
        if event[i] == 0: #not connected to previous event -> new event
            event_counter +=1 
            print('updated event counter: ', event_counter)
            new_event = True 
            event[i] = event_counter
            print('new event. event number: ', event[i])
            
        else:
            print('position ', i, ' already set with value: ', event[i], '\n') 
        
        frame = localizations.frame.iloc[i]
        print('Current frame: ', frame)
        print('collecting localizations in next frame...')
        frame = localizations.frame.iloc[i]
        locs_next_frame = localizations[(localizations.frame == frame+1)]
        
        
        if len(locs_next_frame) == 0:
            print('\nno localization in frame ', frame+1)
            pass
        
        else: has_follower, follower_index = return_nearby(
            localizations.iloc[i], locs_next_frame)
        
        if has_follower: 
            event[follower_index] = event[i] #set to same event
            print('follower index is: ', follower_index)
            print('connected to previous. 1new event_arr: ', event[i-2:i+5])
        if new_event and not has_follower:
            localizations.drop(index=i)
            np.delete(event, i)
            print('deleted standalone localization.')
            deleted = True
        #else: print('no follower nearby')
        print('----- END LOOP -----')
    locs_event_tagged = pd.DataFrame({'frame': localizations.frame,
                                 'x': localizations.x, 
                                 'y': localizations.y, 
                                 'photons': localizations.photons, 
                                 'event': event,
                                 'lifetime': localizations.lifetime,
                                 'lpx': localizations.lpx,
                                 'lpy': localizations.lpy,
                                 'bg': localizations.bg,
                                 'ellipticity': localizations.ellipticity,
                                 'group': localizations.group,
                                 'sx': localizations.sx,
                                 'sy': localizations.sy})
    filtered_locs = filter_unique_events(locs_event_tagged)
    core.dataframe_to_picasso(filtered_locs, localizations_file, '_eve')
    
        
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

  return filtered_locs

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
    max_distance = localization.lpx+localization.lpy
    
    x_distance = (locs_next['x'].to_numpy() - localization.x)
    y_distance = (locs_next['y'].to_numpy() - localization.y)
    
    total_distance_sq = np.square(x_distance) + np.square(y_distance)

    locs_next['distance'] = total_distance_sq
    
    radius_sq = max_distance**2
    
    print('max_distance:', radius_sq)
    print(locs_next.distance)
    
    loc = locs_next[
        locs_next.distance < radius_sq]
    
    
    if len(loc) == 1:
        print('similar loc in next frame.')
        has_next = True
        return has_next, loc.index.values
    
    elif len(loc) > 1:
        print('too many locs')
        return has_next, float('nan')
    
    else:
        print('Number of locs in next frame were: ', len(locs_next), 
              'no similar loc in next frame.')
        return has_next, float('nan')
        