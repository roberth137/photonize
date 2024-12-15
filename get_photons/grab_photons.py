#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:19:22 2024

This module is returns original photons from localization data

@author: roberthollmann
"""
import pandas as pd
import get_photons

def get_pick_photons(
        locs_group, photons, drift, offset,
        box_side_length, int_time):
    '''
    Parameters
    ----------
    locs_group : localizations of this pick (group) as pd dataframe
    photons : photons as pd dataframe
    drift : drift as pd dataframe
    integration time: camera integration time
    box_side_length: size of the PSF in pixels

    Returns
    -------
    All driftcorrected photons in the area
    of the pick +- box_side_length/2
    '''
    # set dimensions of the region and crop photons
    # -0.53125 because: -> see undrift (pixel conversion)
    dr_x, dr_y = max(abs(drift.x)), max(abs(drift.y))
    min_x, max_x, min_y, max_y = get_photons.min_max_box(locs_group, box_side_length)
    phot_cr = get_photons.crop_photons(photons,
                                (min_x - 0.53125 - dr_x),
                                (max_x - 0.53125 + dr_x),
                                (min_y - 0.53125 - dr_y),
                                (max_y - 0.53125 + dr_y))
    print('number of cropped photons: ', len(phot_cr))
    # undrift photons
    phot_cr_und = get_photons.undrift_photons(phot_cr, drift, offset, int_time)
    # crop photons again after drift
    phot_cr_und_cr = get_photons.crop_photons(phot_cr_und,
                                       min_x, max_x, min_y, max_y)
    print('number of cropped-undrifted-cropped photons: ',
          len(phot_cr_und_cr))
    return phot_cr_und_cr


def photons_of_picked_area(localizations_file, photons_file,
                         drift_file, offset, box_side_length, 
                         int_time):
    '''

    Parameters
    ----------
    localizations_file : picasso .hdf5 of picked localizations
    photons_file : photons file as .hdf5
    drift_file : picasso generated drift file .hdf5
    offset : 
    box_side_length : 
    int_time :

    Returns
    -------
    pick_photons : photons in the area of all picked localizations
    over the whole imaging time 

    '''
    localizations = pd.read_hdf(localizations_file, key='locs')
    photons = pd.read_hdf(photons_file, key='photons')
    drift = pd.read_csv(drift_file, delimiter=' ',names =['x','y'])
    drift = drift[::offset]
    pick_photons = crop_undrift_crop(localizations, photons, drift, 
                                    offset, box_side_length, 
                                    int_time)
    return pick_photons

def photons_of_many_events(events, photons, diameter):
    #photons_all_events = pd.DataFrame()
    photons_all_events = pd.concat(
        [get_photons.crop_event(event, photons, diameter) for event in events.itertuples(index=False)],
        ignore_index=True
    )
    #for i in range(len(events)):
    #    photons_this_event = get_photons.crop_event(events.iloc[i], photons, diameter)
    #    photons_all_events = pd.concat([photons_all_events, photons_this_event],
    #                                   ignore_index=True)
    return photons_all_events



def photons_of_many_picked_localizations(
        localizations_file, photons_file,
        drift_file, offset, box_side_length, 
        int_time):
    '''
    Parameters
    ----------
    localizations_file : picasso .hdf5 of picked localizations
    photons_file : photons file as .hdf5
    drift_file : picasso generated drift file .hdf5
    offset : 
    box_side_length : 
    integration_time : 

    Returns
    -------
    pick_photons : photons corresponding to all picked localizations,
    some may be multiple in case of offset 

    '''
    localizations = pd.read_hdf(localizations_file, key='locs')
    photons = pd.read_hdf(photons_file, key='photons')
    drift = pd.read_csv(drift_file, delimiter=' ',names =['x','y'])
    drift = drift[::offset]
    pick_photons = crop_undrift_crop(localizations, photons, drift, 
                                    offset, box_side_length, 
                                    int_time)
    photons_locs = pd.DataFrame()
    for i in range(len(localizations)):
        photons = photons_of_one_localization(
                        localizations.iloc[i], pick_photons, offset, 
                        box_side_length, int_time)
        photons_locs = pd.concat([photons_locs, photons], 
                                  ignore_index=True)
    print('number of photons corresponding to locs: ', len(photons_locs))
    return photons_locs
    

def crop_undrift_crop(
        locs_group, photons, drift, offset,
        box_side_length, int_time):
    '''
    Parameters
    ----------
    locs_group : localizations of this pick (group) as pd dataframe
    photons : photons as pd dataframe
    drift : drift as pd dataframe
    integration time: camera integration time
    box_side_length: size of the PSF in pixels
    
    Returns
    -------
    All driftcorrected photons in the area 
    of the pick +- box_side_length/2
    '''
    # set dimensions of the region and crop photons 
    # -0.53125 because: -> see undrift (pixel conversion)
    dr_x, dr_y = max(abs(drift.x)), max(abs(drift.y))
    min_x, max_x, min_y, max_y = get_photons.min_max_box(locs_group, box_side_length)
    phot_cr = get_photons.crop_photons(photons,
                          (min_x-0.53125-dr_x),
                          (max_x-0.53125+dr_x),
                          (min_y-0.53125-dr_y),
                          (max_y-0.53125+dr_y))
    print('number of cropped photons: ', len(phot_cr))
    # undrift photons 
    phot_cr_und = (get_photons.undrift_photons(phot_cr, drift, offset, int_time))
    # crop photons again after drift 
    phot_cr_und_cr = get_photons.crop_photons(phot_cr_und,
                                       min_x, max_x, min_y, max_y)
    print('number of cropped-undrifted-cropped photons: ', 
          len(phot_cr_und_cr))
    return phot_cr_und_cr


def photons_of_one_localization(localization, pick_photons, offset, box_side_length=5, int_time=200):
    '''
    Returns photons of localization
    IN: 
    - localization (picasso format)
    - picked photons (dataframe format, undrifted)
    - box_size in pixel 
    - int time in ms
    OUT:
    - photons (dataframe format)
    '''
    photons_cylinder, _ = get_photons.crop_cylinder(localization,
            pick_photons, offset, box_side_length, int_time)
    
    return photons_cylinder


