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
    '''
    Parameters
    ----------
    event : all localizations of an event

    Returns
    -------
    start_ms of an event

    '''
    first_loc = locs_event.iloc[0] # now a pd Series 
    last_loc = locs_event.iloc[-1]
    max_photons = max(locs_event['photons'])
    #num_locs = len(locs_event)
    
    #print('num_locs: ', num_locs)
    
    #print(first_loc)
    #print('max_photons: ', max_photons)
    
    start_ms_first = (first_loc['frame']/offset) * int_time
    #print('start_ms first: ', start_ms_first)
    start_ms_last = (last_loc['frame']/offset) * int_time
    
    on_fraction_first = (first_loc['photons']/max_photons)
    on_fraction_last = (last_loc['photons']/max_photons)
    #print('on_ fraction: ', on_fraction_first)
    
    start_ms_event = start_ms_first + (1 - on_fraction_first) * int_time
    end_ms_event = start_ms_last + on_fraction_last * int_time
    #print('start_ms event', start_ms_event)
    #print('end_ms_event: ', end_ms_event)
    #print('on time in ms: ', (end_ms_event - start_ms_event))
    
    return np.floor(start_ms_event), np.ceil(end_ms_event)





def get_start_ms(locs_event, offset, int_time):
    '''
    Parameters
    ----------
    event : all localizations of an event

    Returns
    -------
    start_ms of an event

    '''
    first_loc = locs_event.iloc[0] # now a pd Series 
    max_photons = max(locs_event['photons'])
    num_locs = len(locs_event)
    print('num_locs: ', num_locs)
    
    print(first_loc)
    print('max_photons: ', max_photons)
    
    start_ms_first = (first_loc['frame']/offset) * int_time
    print('start_ms first: ', start_ms_first)
    
    on_fraction = (first_loc['photons']/max_photons)
    print('on_ fraction: ', on_fraction)
    
    start_ms_event = start_ms_first + (1 - on_fraction) * int_time
    print('start_ms event', start_ms_event)
    
    return start_ms_event



def get_end_ms(locs_event, offset, int_time):
    '''
    Parameters
    ----------
    event : all localizations of an event

    Returns
    -------
    end_ms of an event

    '''
    last_loc = locs_event.iloc[-1] # now a pd Series 
    max_photons = max(locs_event['photons'])
    num_locs = len(locs_event)
    print('num_locs; ', num_locs)
    
    print(last_loc)
    print('max_photons: ', max_photons)
    
    start_ms_last = (last_loc['frame']/offset) * int_time
    print('start_ms_last: ', start_ms_last)
    
    on_fraction = (last_loc['photons']/max_photons)
    print('on_fraction: ', on_fraction)
    
    end_ms_event = start_ms_last + (on_fraction * int_time)
    print('end_ms_event: ', end_ms_event)
    
    return end_ms_event

