#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:09:10 2024

@author: roberthollmann

Module that outputs locs dataframes to picasso format
"""

import pandas as pd
import helper
import fitting
from event import event_bounds, tag_events


def locs_to_events_to_picasso(localizations_file, 
                              offset, box_side_length, int_time):
    '''
    Converts a DataFrame of localizations into a list of Event objects.

    Returns:
        list of Event: List of Event objects.
    '''
    localizations = pd.read_hdf(localizations_file, key='locs')
    # Validate required columns
    required_columns = {'frame', 'x', 'y', 'photons', 'bg', 'lpx', 'lpy', }
    if not required_columns.issubset(localizations.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Tag localizations with event number and group them
    localizations_eve = tag_events.connect_locs(localizations)
    grouped = localizations_eve.groupby('event')

    events = pd.DataFrame()
    
    for event, group in grouped:
        group = group.reset_index(drop=True)
        #print('_____________________________________________________START')
        #print('calculating event number: ', event)
        # Compute event properties
        first = group.iloc[0]
        last = group.iloc[-1]
        
        
        #print('group photons values: \n', group['photons'])
        #print('max phot index: ',group['photons'].idxmax())
        #print('len group', len(group))
        
        
        peak_event = group.iloc[group['photons'].idxmax()]
        start_ms, end_ms = event_bounds.get_ms_bounds(
            group, offset, int_time)
        #print('peak event is: ', '\n', peak_event)
        
        event_data = {'frame': peak_event['frame'],
                 'event': first['event'], 
                 'x': fitting.avg_photon_weighted(group, 'x'),
                 'y': fitting.avg_photon_weighted(group, 'y'),
                 'photons': peak_event['photons'],
                 'start_ms': start_ms,
                 'end_ms': end_ms,
                 'lpx': fitting.avg_photon_weighted(group, 'lpx'),
                 'lpy': fitting.avg_photon_weighted(group, 'lpy'),
                 'num_frames': (last['frame']-first['frame'])+1,
                 'start_frame': first['frame'],
                 'end_frame': last['frame'],
                 'bg': fitting.avg_photon_weighted(group, 'bg'),
                 'sx': fitting.avg_photon_weighted(group, 'sx'),
                 'sy': fitting.avg_photon_weighted(group, 'sy'),
                 'net_gradient': fitting.avg_photon_weighted(group, 'net_gradient'),
                 'ellipticity': fitting.avg_photon_weighted(group, 'ellipticity'),
                 'group': first['group']
                 }
        event = pd.DataFrame(event_data, index=[0])
        events = pd.concat([events, event], 
                                  ignore_index=True)
    helper.dataframe_to_picasso(events, localizations_file,
                              extension='_locs_connected')
    #return events