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

class Event:
    def __init__(self, number, x, y, photons, lifetime, 
                 start_frame, end_frame, lpx, lpy, duration, frame, bg):
        self.number = number
        self.x = x
        self.y = y
        self.photons = photons
        self.lifetime = lifetime
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.lpx = lpx
        self.lpy = lpy
        self.duration = duration
        self.frame = frame
        self.bg = bg

    def __repr__(self):
        return (f'Event(number= {self.number}, x={self.x}, y={self.y}, '
                f'photons={self.photons}, lifetime={self.lifetime},'
                f'start_frame={self.start_frame}, end_frame={self.end_frame},'
                f'lpx={self.lpx}, lpy={self.lpy}, '
                f'duration={self.duration}, frame={self.frame}, '
                f'bg={self.bg})')

def create_events_from_localizations(localizations):
    '''
    Converts a DataFrame of localizations into a list of Event objects.

    Returns:
        list of Event: List of Event objects.
    '''
    # Validate required columns
    required_columns = {'frame', 'x', 'y', 'photons', 'bg', 'event',
                        'lifetime', 'lpx', 'lpy', }
    if not required_columns.issubset(localizations.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Group localizations by 'event_number'
    grouped = localizations.groupby('event')

    events = []
    for event, group in grouped:
        # Compute event properties
        first = group.iloc[0]
        last = group.iloc[-1]
        
        number = first.event
        x = avg_photon_weighted(group, 'x')  # Average position (x)
        y = avg_photon_weighted(group, 'y')  # Average position (y)
        photons = max(group['photons'])  # Total photons
        lifetime = avg_photon_weighted(group, 'lifetime')
        lpx = avg_photon_weighted(group, 'lpx')
        lpy = avg_photon_weighted(group, 'lpy')
        start_frame = first.frame # Earliest frame in the group
        end_frame = last.frame  # Latest frame in the group
        bg = avg_photon_weighted(group, 'bg')  # Average background
        duration = (end_frame-start_frame+1) 
        frame = (end_frame-start_frame)/2 

        # Create the Event object
        event = Event(
            number=number,
            x=x,
            y=y,
            photons=photons,
            lifetime=lifetime,
            start_frame=start_frame,
            end_frame=end_frame,
            lpx=lpx,
            lpy=lpy,
            bg=bg,
            duration=duration,
            frame=frame
        )
        events.append(event)

    return events



# Example usage
#if __name__ == "__main__":
#    # Example DataFrame
#    data = {
#        'frame': [1, 2, 2, 3, 3, 4],
#        'x': [10, 11, 12, 20, 21, 22],
#        'y': [15, 16, 16, 25, 26, 26],
#        'number_photons': [100, 120, 130, 200, 210, 220],
#        'background': [5, 5, 5, 10, 10, 10],
#        'event_number': [1, 1, 1, 2, 2, 2]
#    }
#    localizations_df = pd.DataFrame(data)

    # Create events from localizations
#events = create_events_from_localizations(localizations)

    # Print events
    #for event in events:
     #   print(event)
        
        
        
#########################33333333#########################        

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
        
def event_average(localizations_file):
    '''
    Parameters
    ----------
    localizations_file : localizations_file, tagged with event

    Returns
    -------
    localizations file with each row the average of events

    '''
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
        '''
            {'frame': avg_photon_weighted(locs_event, 'frame'),
            'x': avg_photon_weighted(locs_event, 'x'),
            'y': avg_photon_weighted(locs_event, 'y'),
            'photons': max(locs_event['photons']),
            'lifetime': avg_photon_weighted(locs_event, 'lifetime'),
            'lpx': avg_photon_weighted(locs_event, 'lpx'),
            'lpy': avg_photon_weighted(locs_event, 'lpy'),
            'bg': avg_photon_weighted(locs_event, 'bg'),
            'ellipticity': avg_photon_weighted(locs_event, 'ellipticity'),
            'sx': avg_photon_weighted(locs_event, 'sx'),
            'sy': avg_photon_weighted(locs_event, 'sy')}])
        '''
        #print(averaged)
    core.dataframe_to_picasso(averaged, localizations_file, 'event_averaged')
        
        
        
def avg_photon_weighted(localizations, column):
    '''

    Parameters
    ----------
    localizations : localizations where one column should be averaged over 
    with photons weight 
    column : column to be averaged, e.g. 'x' or 'lifetime'

    Returns
    -------
    average : sum over i: value[i]*photons[i]/total_photons

    '''
    localizations = localizations.reset_index(drop=True)
    column_sum = 0 
    total_photons = 0
    for i in np.arange(len(localizations), dtype=int): 
        loc_photons = localizations.loc[i, 'photons']
        column_sum += (localizations.loc[i, column] * loc_photons)
        total_photons += loc_photons
    average = column_sum/total_photons
    #print('average value is: ', average)
    return average
        

    
    
        