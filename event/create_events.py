"""
Created on Tue Nov 19 15:08:39 2024

@author: roberthollmann

This is a python script to create events from linked localizations.
One event has the key coordinates: start_ms_frame, end_ms_frame, x and y.
It should be used only with filtered localizations
"""
import pandas as pd
import numpy as np
from event import link_locs
from utilities import helper
from fitting.locs_average import avg_photon_weighted
    
    

def locs_to_events(localizations_file, offset, int_time, max_dark_frames=1, proximity=2, filter_single=True):
    """
    Connects a DataFrame of Localization to Events (linked Locs)
    Input:
        localizations: rendered, filtered, picked
        offset: offset used for frame video
        int_time:
        max_dark_frames: Number of frames where no loc is found that can be skipped
        proximity: max distance between adjacent locs, in units of lpx+lpy
        filter_single: Filters single localizaitions that cant be connected to an event

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
    localizations_eve = link_locs.link_locs_by_group(localizations,
                                                         max_dark_frames=max_dark_frames,
                                                         proximity=proximity,
                                                         filter_single=filter_single)
    grouped = localizations_eve.groupby('event')
    events = pd.DataFrame()
    
    for event, eve_group in grouped:
        eve_group = eve_group.reset_index(drop=True)

        # Compute event properties
        first = eve_group.iloc[0]
        last = eve_group.iloc[-1]
        
        peak_loc = eve_group.iloc[eve_group['photons'].idxmax()]
        start_ms_fr = (first.frame/offset)*int_time
        end_ms_fr = (last.frame/offset + 1)*int_time
        event_data = {'frame': peak_loc['frame'],
                 'event': first['event'], 
                 'x': avg_photon_weighted(eve_group, 'x'),
                 'y': avg_photon_weighted(eve_group, 'y'),
                 'photons': peak_loc['photons'],
                 'start_ms_fr': start_ms_fr,
                 'end_ms_fr': end_ms_fr,
                 'num_frames': (last['frame'] - first['frame']) + 1,
                 'lpx': peak_loc['lpx'],
                 'lpy': peak_loc['lpy'],
                 'bg': np.mean(eve_group['bg']),
                 'sx': avg_photon_weighted(eve_group, 'sx'),
                 'sy': avg_photon_weighted(eve_group, 'sy'),
                 'group': first['group'],
                 'net_gradient': peak_loc['net_gradient'],
                 'ellipticity': peak_loc['ellipticity'],
                 'start_frame': first['frame'],
                 'end_frame': last['frame']
                 }
        event = pd.DataFrame(event_data, index=[0])
        events = pd.concat([events, event], 
                                  ignore_index=True)
    events = events.astype({'frame': 'uint32',
                            'event': 'uint32',
                            'x': 'float32',
                            'y': 'float32',
                            'photons': 'float32',
                            'start_ms_fr': 'float32',
                            'end_ms_fr': 'float32',
                            'lpx': 'float32',
                            'lpy': 'float32',
                            'num_frames': 'uint32',
                            'start_frame': 'uint32',
                            'end_frame': 'uint32',
                            'bg': 'float32',
                            'sx': 'float32',
                            'sy': 'float32',
                            'net_gradient': 'float32',
                            'ellipticity': 'float32'})

    print('Linked ', len(localizations), ' locs to ', 
          len(events), 'events.')
    return events

   

def locs_to_events_to_picasso(localizations_file, 
                              offset, box_side_length, int_time):
    """

    OLD FUNCTION


    Converts a DataFrame of localizations into a list of Event objects and saves it as picasso file

    Returns:
        list of Event: List of Event objects.
    """
    localizations = helper.process_input(localizations_file, 'locs')
    # Validate required columns
    required_columns = {'frame', 'x', 'y', 'photons', 'bg', 'lpx', 'lpy', }
    if not required_columns.issubset(localizations.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Tag localizations with event number and group them
    localizations_eve = link_locs.link_locs_by_group(localizations, False)
    grouped = localizations_eve.groupby('event')

    events = pd.DataFrame()
    for event, eve_group in grouped:
        eve_group = eve_group.reset_index(drop=True)

        # Compute event properties
        first = eve_group.iloc[0]
        last = eve_group.iloc[-1]

        peak_event = eve_group.iloc[eve_group['photons'].idxmax()]

        #start_ms, end_ms = event_bounds.get_ms_bounds(
        #    eve_group, offset, int_time)
        start_ms = (first.frame/offset)*int_time
        end_ms = (last.frame/offset + 1)*int_time
        duration_ms = (end_ms-start_ms)

        event_duration = (1 + ((last.frame - first.frame)/offset)) #* int_time ## start_1st frame to end_last frame
        measured_frames = len(eve_group) #* int_time
        overlap = measured_frames/event_duration
        total_photons_estimate = eve_group['total_photons'].sum()/overlap
        average_bg = eve_group['bg'].mean()
        bg = average_bg*duration_ms/int_time

        event_data = {'frame': peak_event['frame'],
                 'event': first['event'], 
                 'x': avg_photon_weighted(eve_group, 'x'),
                 'y': avg_photon_weighted(eve_group, 'y'),
                 'photons': peak_event['photons'],
                 'total_photons': total_photons_estimate,
                 'start_ms': start_ms,
                 'end_ms': end_ms,
                 'duration_ms': duration_ms,
                 'lpx': avg_photon_weighted(eve_group, 'lpx'),
                 'lpy': avg_photon_weighted(eve_group, 'lpy'),
                 'num_frames': (last['frame']-first['frame'])+1,
                 'start_frame': first['frame'],
                 'end_frame': last['frame'],
                 'bg': bg,
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

        
def event_average(localizations_file):
    """
    OLD FUNCTION


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