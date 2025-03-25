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
from typing import Union


def locs_to_events(
        localizations_file: Union[str, pd.DataFrame],
        offset: float,
        int_time: float,
        max_dark_frames: int = 1,
        proximity: float = 2,
        filter_single: bool = True
) -> pd.DataFrame:
    """
    Connects a DataFrame of localizations to events (linked localizations) and
    computes event-level summary metrics.

    Parameters
    ----------
    localizations_file : Union[str, pd.DataFrame]
        Either the path to a file containing localizations or a DataFrame itself.
        Must contain the columns: {'frame', 'x', 'y', 'photons', 'bg', 'lpx', 'lpy'}.
    offset : float
        The offset used for scaling the 'frame' column into real time (often used when frames start at 0).
    int_time : float
        The integration time per frame (e.g., in milliseconds).
    max_dark_frames : int, optional
        Number of consecutive frames without any localization that can be skipped
        when linking localizations. Default is 1.
    proximity : float, optional
        Maximum distance (in units of lpx + lpy) between adjacent localizations
        to be considered part of the same event. Default is 2.
    filter_single : bool, optional
        Whether to filter out (exclude) single localizations that cannot be connected
        to an event. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame of computed events with columns such as:
        ['event', 'frame', 'x', 'y', 'photons', 'bg', 'lpx', 'lpy', 'sx', 'sy',
         'start_frame', 'end_frame', 'start_ms_fr', 'end_ms_fr', 'num_frames',
         'net_gradient', 'ellipticity', 'group'].
    """
    # --- STEP 1: Load and validate the input DataFrame ---
    localizations = helper.process_input(localizations_file, dataset='locs')
    required_cols = {'frame', 'x', 'y', 'photons', 'bg', 'lpx', 'lpy'}
    missing = required_cols - set(localizations.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # --- STEP 2: Link localizations into events ---
    # The link_locs_by_group function must add an 'event' column indicating the event ID
    localizations_eve = link_locs.link_locs_by_group(
        localizations,
        max_dark_frames=max_dark_frames,
        proximity=proximity,
        filter_single=filter_single
    )

    # --- STEP 3: Build event-level records ---
    event_records = []
    grouped = localizations_eve.groupby('event')

    for event_id, eve_group in grouped:
        # Make sure indexing is clean
        eve_group = eve_group.reset_index(drop=True)

        # Basic references
        first_loc = eve_group.iloc[0]
        last_loc = eve_group.iloc[-1]

        # Identify the peak localization (max photons)
        peak_idx = eve_group['photons'].idxmax()
        peak_loc = eve_group.loc[peak_idx]

        # Compute weighted means (assuming you have an avg_photon_weighted function)
        x_weighted = avg_photon_weighted(eve_group, 'x')
        y_weighted = avg_photon_weighted(eve_group, 'y')
        sx_weighted = avg_photon_weighted(eve_group, 'sx')
        sy_weighted = avg_photon_weighted(eve_group, 'sy')

        # Convert frames to time (ms) - adjust by offset
        start_ms = (first_loc.frame / offset) * int_time
        end_ms = ((last_loc.frame / offset) + 1) * int_time

        # Accumulate event data in a dict
        event_data = {
            'event': np.uint32(first_loc['event']),
            'frame': np.uint32(peak_loc['frame']),
            'x': np.float32(x_weighted),
            'y': np.float32(y_weighted),
            'photons': np.float32(peak_loc['photons']),
            'bg': np.float32(eve_group['bg'].mean()),
            'lpx': np.float32(peak_loc['lpx']),
            'lpy': np.float32(peak_loc['lpy']),
            'sx': np.float32(sx_weighted),
            'sy': np.float32(sy_weighted),
            'group': first_loc.get('group', np.nan),
            'num_frames': np.uint32((last_loc['frame'] - first_loc['frame']) + 1),
            'start_frame': np.uint32(first_loc['frame']),
            'end_frame': np.uint32(last_loc['frame']),
            'net_gradient': np.float32(peak_loc.get('net_gradient', np.nan)),
            'ellipticity': np.float32(peak_loc.get('ellipticity', np.nan)),
            'start_ms_fr': np.float32(start_ms),
            'end_ms_fr': np.float32(end_ms),
        }
        event_records.append(event_data)

    # --- STEP 4: Build the events DataFrame ---
    events = pd.DataFrame(event_records)

    # --- STEP 5: Final print and return ---
    print(f"Linked {len(localizations)} locs to {len(events)} events.")
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