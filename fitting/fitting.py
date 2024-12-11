#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 01:23:12 2024

@author: roberthollmann
"""
import numpy as np

def avg_lifetime_sergi_40(loc_photons, peak, dt_offset=0):
    '''
    Fit lifetimes of individual localizations with 40mhz laser frequency
    Parameters
    ----------
    loc_photons : all photons from one localization 
    peak : position of the maximum of arrival times for this pick of 
    localizations, calibrated from calibrate_peak()
    offset : the offset from the peak where arrival times are considered
    for fitting the lifetime.The default is 50.

    Returns
    -------
    average arrival time of photons, in units of arrival time bin size.
    Usually 10ps 

    '''
    
    counts, bins = np.histogram(loc_photons.dt, bins=np.arange(0,2500))
    background = np.sum(counts[-300:])/300
    counts_bgsub = counts - background
    weights = np.arange(1,(2500-(peak+dt_offset)))
    considered_bgsub = counts_bgsub[(peak+dt_offset):2500]
    if len(loc_photons) < 70:
        print('\nphotons for fitting: ', len(loc_photons))
        print('considered bg is: ', sum(considered_bgsub))
    lifetime = np.sum(np.multiply(considered_bgsub, weights))/np.sum(considered_bgsub)
    return lifetime

def avg_lifetime_sergi_80(loc_photons, peak, dt_offset=50):
    '''
    Fit lifetimes of individual localizations with 80mhz laser frequency
    Parameters
    ----------
    loc_photons : all photons from one localization 
    peak : position of the maximum of arrival times for this pick of 
    localizations, calibrated from calibrate_peak()
    offset : the offset from the peak where arrival times are considered
    for fitting the lifetime.The default is 50.

    Returns
    -------
    average arrival time of photons, in units of arrival time bin size.
    Usually 10ps 

    '''
    counts, bins = np.histogram(loc_photons.dt, bins=np.arange(0,1250))
    background = np.sum(counts[-300:])/300
    counts_bgsub = counts - background
    weights = np.arange(1,(1250-(peak+dt_offset)))
    considered_bgsub = counts_bgsub[(peak+dt_offset):1250]
    lifetime = np.sum(np.multiply(considered_bgsub, weights))/np.sum(considered_bgsub)
    return lifetime


def avg_of_roi(localization, phot_locs, box_side_length):
    '''
    Parameters
    ----------
    phot_locs : photons of one localization as pd dataframe
    
    Returns
    -------
    - x position
    - y position
    
    Position is calculated by adding up all photons in the circular
    surrounding of the localization. 
    Background gets subtracted

    '''
    fit_area = (box_side_length/2)**2
    number_phot = (len(phot_locs)-fit_area*localization.bg)
    bg = localization.bg*fit_area
    x_pos = (np.sum(phot_locs.x) - bg*localization.x)/number_phot
    y_pos = (np.sum(phot_locs.y) - bg*localization.y)/number_phot
    return x_pos, y_pos

        