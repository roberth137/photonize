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
from event import tag_events
from event import event_bounds
import helper
from fitting.locs_average import avg_photon_weighted
    
    

def locs_to_events(localizations_file, offset, box_side_length, int_time):
    """
    Converts a DataFrame of localizations into a list of Event objects.

    Returns:
        list of Event: List of Event objects.
    """
    # Validate required columns
    localizations = helper.process_input(localizations_file, 
                                         dataset='locs')
    required_columns = {'frame', 'x', 'y', 'photons', 'bg', 'lpx', 'lpy', }
    if not required_columns.issubset(localizations.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Tag localizations with event number and group them
    localizations_eve = tag_events.connect_locs(localizations)
    grouped = localizations_eve.groupby('event')


    events = pd.DataFrame()
    
    for event, eve_group in grouped:
        eve_group = eve_group.reset_index(drop=True)
        #print('_____________________________________________________START')
        #print('calculating event number: ', event)
        # Compute event properties
        first = eve_group.iloc[0]
        last = eve_group.iloc[-1]
        
        peak_event = eve_group.iloc[eve_group['photons'].idxmax()]
        start_ms, end_ms = event_bounds.get_ms_bounds(
            eve_group, offset, int_time)

        event_duration = (1 + ((last.frame - first.frame) / offset))  # * int_time ## start_1st frame to end_last frame
        measured_frames = len(eve_group)  # * int_time
        overlap = measured_frames / event_duration
        total_photons_estimate = eve_group['total_photons'].sum() / overlap

        average_bg = eve_group['bg'].mean()

        event_data = {'frame': peak_event['frame'],
                 'event': first['event'], 
                 'x': avg_photon_weighted(eve_group, 'x'),
                 'y': avg_photon_weighted(eve_group, 'y'),
                 'photons': peak_event['photons'],
                 'total_photons': total_photons_estimate,
                 'start_ms': start_ms,
                 'end_ms': end_ms,
                 'duration_ms': (end_ms-start_ms),
                 'lpx': avg_photon_weighted(eve_group, 'lpx'),
                 'lpy': avg_photon_weighted(eve_group, 'lpy'),
                 'num_frames': (last['frame']-first['frame'])+1,
                 'start_frame': first['frame'],
                 'end_frame': last['frame'],
                 'bg': average_bg,
                 'sx': avg_photon_weighted(eve_group, 'sx'),
                 'sy': avg_photon_weighted(eve_group, 'sy'),
                 'net_gradient': avg_photon_weighted(eve_group, 'net_gradient'),
                 'ellipticity': avg_photon_weighted(eve_group, 'ellipticity'),
                 'group': first['group']
                 }
        event = pd.DataFrame(event_data, index=[0])
        events = pd.concat([events, event], 
                                  ignore_index=True)
    events = events.astype({'frame': 'uint32'})
        
    print('Linked ', len(localizations), ' locs to ', 
          len(events), 'events.')
    print('_______________________________________________')
    return events

   

def locs_to_events_to_picasso(localizations_file, 
                              offset, box_side_length, int_time):
    """
    Converts a DataFrame of localizations into a list of Event objects.

    Returns:
        list of Event: List of Event objects.
    """
    localizations = helper.process_input(localizations_file, 'locs')
    # Validate required columns
    required_columns = {'frame', 'x', 'y', 'photons', 'bg', 'lpx', 'lpy', }
    if not required_columns.issubset(localizations.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Tag localizations with event number and group them
    localizations_eve = tag_events.connect_locs(localizations)
    grouped = localizations_eve.groupby('event')

    events = pd.DataFrame()
    for event, eve_group in grouped:
        eve_group = eve_group.reset_index(drop=True)
        #print('_____________________________________________________START')
        #print('calculating event number: ', event)
        # Compute event properties
        first = eve_group.iloc[0]
        last = eve_group.iloc[-1]
        #photons_array = np.ones(len(group))
        #for i in range(len(group)):
        #    photons_array[i] = group.iloc[i].photons

        peak_event = eve_group.iloc[eve_group['photons'].idxmax()]
        start_ms, end_ms = event_bounds.get_ms_bounds(
            eve_group, offset, int_time)
        event_duration = (1 + ((last.frame - first.frame)/offset)) #* int_time ## start_1st frame to end_last frame
        measured_frames = len(eve_group) #* int_time
        overlap = measured_frames/event_duration
        total_photons_estimate = eve_group['total_photons'].sum()/overlap
        average_bg = eve_group['bg'].mean()

        #print('_____________________________________')
        #print('FRAME: first -- last -- duration')
        #print('_____', first.frame, ' -- ', last.frame, ' -- ', (last.frame-first.frame+1))
        #print('peak event is: ', peak_event.frame)
        #print('MS:    first -- last -- duration')
        #print('_____', start_ms, end_ms, (end_ms-start_ms))
        #print('PHOTONS: first -- max -- last ')
        #print(photons_array)
        
        event_data = {'frame': peak_event['frame'],
                 'event': first['event'], 
                 'x': avg_photon_weighted(eve_group, 'x'),
                 'y': avg_photon_weighted(eve_group, 'y'),
                 'photons': peak_event['photons'],
                 'total_photons': total_photons_estimate,
                 'start_ms': start_ms,
                 'end_ms': end_ms,
                 'duration_ms': (end_ms - start_ms),
                 'lpx': avg_photon_weighted(eve_group, 'lpx'),
                 'lpy': avg_photon_weighted(eve_group, 'lpy'),
                 'num_frames': (last['frame']-first['frame'])+1,
                 'start_frame': first['frame'],
                 'end_frame': last['frame'],
                 'bg': average_bg,
                 'sx': avg_photon_weighted(eve_group, 'sx'),
                 'sy': avg_photon_weighted(eve_group, 'sy'),
                 'net_gradient': avg_photon_weighted(eve_group, 'net_gradient'),
                 'ellipticity': avg_photon_weighted(eve_group, 'ellipticity'),
                 'group': first['group']
                 }
        event = pd.DataFrame(event_data, index=[0])
        events = pd.concat([events, event], 
                                  ignore_index=True)
    helper.dataframe_to_picasso(events, localizations_file,
                              extension='_locs_to_events')
    #return events


        
def event_average(localizations_file):
    """
    Parameters
    ----------
    localizations_file : localizations_file, tagged with event

    Returns
    -------
    localizations file with each row the average of events

    """
    localizations = pd.read_hdf(localizations_file, key='locs')
    averaged = pd.DataFrame()#{'frame':'','x':'','y':'', 'photons':'',
                             #'event':'','lifetime':'','lpx':'','lpy':'',
                             #'bg':'','ellipticity':'','group':'',
                             #'sx':'','sy':''})
    for e in set(localizations.event):
        #print('averaging event; ', e)
        locs_event = localizations[localizations.event == e]
        duration_frames = len(locs_event)
        #print(locs_event)
        new_event = {'frame': int(np.floor(avg_photon_weighted(locs_event, 'frame'))),
        'x': avg_photon_weighted(locs_event, 'x'),
        'y': avg_photon_weighted(locs_event, 'y'),
        'photons': max(locs_event['photons']),
        'event': e,
        'lifetime': avg_photon_weighted(locs_event, 'lifetime'),
        'duration_fr': duration_frames,
        'lpx': avg_photon_weighted(locs_event, 'lpx'),
        'lpy': avg_photon_weighted(locs_event, 'lpy'),
        'bg': avg_photon_weighted(locs_event, 'bg'),
        'ellipticity': avg_photon_weighted(locs_event, 'ellipticity'),
        'net_gradient': avg_photon_weighted(locs_event, 'net_gradient'),
        'sx': avg_photon_weighted(locs_event, 'sx'),
        'sy': avg_photon_weighted(locs_event, 'sy')}
        new_event_df = pd.DataFrame([new_event])

        averaged = pd.concat([averaged, new_event_df])

    helper.dataframe_to_picasso(averaged, localizations_file, 'event_averaged')
