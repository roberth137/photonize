#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:41:27 2024

This module reads in event tagged localizations and defines event bins:
    
    These event bins define the start_ms, end_ms, x, y and other attributes
    of an event 
    
@author: roberthollmann
"""
import numpy as np

def get_ms_bounds(locs_event, offset, int_time):
    """
    Parameters
    ----------
    locs_event : all localizations of an event
    offset
    int_time

    Returns
    -------
    start_ms of an event

    """
    first_loc = locs_event.iloc[0] # now a pd Series 
    last_loc = locs_event.iloc[-1]
    average_photons = (locs_event['total_photons'].sum()/len(locs_event))
    
    start_ms_first = (first_loc['frame']/offset) * int_time
    start_ms_last = (last_loc['frame']/offset) * int_time
    
    if len(locs_event) <3:

        beg_on_fraction = (first_loc['total_photons']/average_photons)
        end_on_fraction = (last_loc['total_photons']/average_photons)

    else:
        second_loc = locs_event.iloc[1]
        second_last_loc = locs_event.iloc[-2]

        difference_time = 1/offset

        #avg_top3 = (locs_event['total_photons'].nlargest(3).sum())/3

        on_fraction_first = (first_loc['total_photons'] / average_photons)
        on_fraction_second = (second_loc['total_photons'] / average_photons)

        on_fraction_last = (last_loc['total_photons'] / average_photons)
        on_fraction_second_last = (second_last_loc['total_photons'] / average_photons)

        on_sec_in_first = on_fraction_second - difference_time
        beg_on_fraction = (on_fraction_first+on_sec_in_first)/2

        on_sec_last_in_last = on_fraction_second_last - difference_time
        end_on_fraction = (on_fraction_last+on_sec_last_in_last)/2



    start_ms_event = start_ms_first + (1-beg_on_fraction) * int_time
    end_ms_event = start_ms_last + end_on_fraction * int_time
    
    return np.floor(start_ms_event), np.ceil(end_ms_event)


def get_start_ms(locs_event, offset, int_time):
    """
    Parameters
    ----------
    locs_event : all localizations of an event
    offset
    int_time

    Returns
    -------
    start_ms of an event

    """
    first_loc = locs_event.iloc[0] # now a pd Series 
    max_photons = max(locs_event['photons'])
    #print('num_locs: ', len(locs_event))
    
    #print(first_loc)
    #print('max_photons: ', max_photons)
    
    start_ms_first = (first_loc['frame']/offset) * int_time
    #print('start_ms first: ', start_ms_first)
    
    on_fraction = (first_loc['photons']/max_photons)
    print('on_fraction first: ', on_fraction)
    
    start_ms_event = start_ms_first + (1 - on_fraction) * int_time
    #print('start_ms event', start_ms_event)
    
    return start_ms_event


def get_end_ms(locs_event, offset, int_time):
    """
    Parameters
    ----------
    locs_event : all localizations of an event
    offset
    int_time

    Returns
    -------
    end_ms of an event

    """
    last_loc = locs_event.iloc[-1] # now a pd Series 
    max_photons = max(locs_event['photons'])
    start_ms_last = (last_loc['frame']/offset) * int_time

    on_fraction = (last_loc['photons']/max_photons)
    print('on_fraction last: ', on_fraction)
    
    end_ms_event = start_ms_last + (on_fraction * int_time)

    return end_ms_event