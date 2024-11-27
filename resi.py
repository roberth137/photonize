# RESI functions
import numpy as np
import pandas as pd
import core
import get_photons
#import scipy as sp
#import matplotlib.pyplot as plt
#import math
#import h5py
#from pathlib import Path
#import copy
#import matplotlib as mpl
#import scipy.io as sio
#from scipy.optimize import curve_fit
#import shutil

def locs_lt_to_picasso_40(localizations_file, photons_file, 
                       drift_file, offset, box_side_length=5,
                       integration_time=200, fitting='avg'):
    '''
    tagging list of picked localizations with lifetime (40mhz pulsed)
    and returning as picasso files
    IN:
    - list of picked localizations (picasso hdf5 file with 'group' column)
    - list of photons (hdf5 file)
    - drift (txt file)
    - offset (how many offsetted frames)
    OUT: 
    - picasso hdf5 file tagged with lifetime 
    - yaml file 
    '''
    #read in files
    localizations = pd.read_hdf(localizations_file, key='locs')
    total_localizations = len(localizations)
    photons = pd.read_hdf(photons_file, key='photons')
    print(len(photons), ' photons and ', total_localizations,
          'localization read in')
    drift = pd.read_csv(drift_file, delimiter=' ',names =['x','y']) 
    lifetime = np.ones(len(localizations))
    lt_photons = np.ones(len(localizations), dtype=int)
    counter = 0
    # iterating over every pick in file
    for g in set(localizations['group']):
        locs_group = localizations[(localizations.group == g)]
        print(len(locs_group), 'localizations in current group.')
        pick_photons = get_photons.crop_undrift_crop(locs_group, photons, 
                                        drift, offset,
                                        box_side_length, integration_time)
        peak_arrival_time = calibrate_peak(locs_group, pick_photons, 
                                           offset, box_side_length, 
                                           integration_time)
        # iterating over every localization in pick
        for i in range(counter, counter+len(locs_group)):
            if i == 0: print('fitting lifetime of ', len(locs_group),
                             ' localizations.') 
            phot_loc = get_photons.photons_of_one_localization(
                    locs_group.iloc[i-counter], 
                    pick_photons,offset, 
                    box_side_length, integration_time)
            if i % 200 == 0:print('200 fitted. Number of photons',
                                  ' in last fit: ', len(phot_loc))
            lifetime[i] = avg_lifetime_sergi_40(phot_loc, 
                                                      peak_arrival_time)
            lt_photons[i] = len(phot_loc)
        counter += len(locs_group)
    localizations['lifetime'] = lifetime
    localizations['lt_photons'] = lt_photons
    core.dataframe_to_picasso(
        localizations, localizations_file)
    print(len(localizations), 'localizations tagged with lifetime')
    
    
def events_lt_avg_pos(event_file, photons_file, 
                       drift_file, offset, radius=5,
                       integration_time=200, fitting='avg'):
    '''
    tagging list of picked localizations with lifetime 
    and returning as picasso files
    IN:
    - list of picked localizations (picasso hdf5 file with 'group' column)
    - list of photons (hdf5 file)
    - drift (txt file)
    - offset (how many offsetted frames)
    OUT: 
    - picasso hdf5 file tagged with lifetime 
    - yaml file 
    '''
    #read in files
    events = pd.read_hdf(event_file, key='locs')
    total_events = len(events)
    photons = pd.read_hdf(photons_file, key='photons')
    
    print(len(photons), ' photons and ', total_events,
          'localization read in')
    drift = pd.read_csv(drift_file, delimiter=' ',names =['x','y'])  
    
    lifetime = np.ones(len(events))
    lt_photons = np.ones(len(events), dtype=int)
    x_position = np.ones(len(events))
    y_position = np.ones(len(events))
    counter = 0
    # iterating over every pick in file
    for g in range(1):#set(events['event']):
        #locs_group = localizations[(localizations.group == g)]
        #print(len(locs_group), 'localizations in current group.')
        pick_photons = get_pick_photons(events, photons, 
                                        drift, offset,
                                        box_side_length=radius, integration_time=integration_time)
        peak_arrival_time = calibrate_peak(events, pick_photons, 
                                           offset, box_side_length=radius, 
                                           integration_time=integration_time)
        
        # iterating over every localization in pick
        for i in range(counter, counter+len(events)):
            #if i == 0: print('fitting lifetime of ', len(locs_group),
            #                 ' localizations.') 
            my_event = events.iloc[i-counter]
            phot_event = pd.DataFrame(data=core.crop_event
                                    (my_event, pick_photons, radius))
            if i % 200 == 0:print('200 fitted. Number of photons',
                                  ' in last fit: ', len(phot_event))
            x, y = avg_of_roi(my_event, phot_event, radius)
            x_position[i] = x
            y_position[i] = y
            lifetime[i] = avg_lifetime_sergi_40(phot_event, 
                                                      peak_arrival_time)
            lt_photons[i] = len(phot_event)
        counter += len(events)
    events['x'] = x_position
    events['y'] = y_position
    events['lifetime'] = lifetime
    events['lt_photons'] = lt_photons
    core.dataframe_to_picasso(
        events, event_file, '_lt_avgPos_noBg')
    print(len(events), 'events tagged with lifetime and'
          ' fitted with avg x,y position.') 
    
    
def locs_lt_to_picasso_80(localizations_file, photons_file, 
                       drift_file, offset, box_side_length=5,
                       integration_time=200, fitting='avg'):
    '''
    tagging list of picked localizations with lifetime (40mhz pulsed)
    and returning as picasso files
    IN:
    - list of picked localizations (picasso hdf5 file with 'group' column)
    - list of photons (hdf5 file)
    - drift (txt file)
    - offset (how many offsetted frames)
    OUT: 
    - picasso hdf5 file tagged with lifetime 
    - yaml file 
    '''
    #read in files
    localizations = pd.read_hdf(localizations_file, key='locs')
    total_localizations = len(localizations)
    photons = pd.read_hdf(photons_file, key='photons')
    print(len(photons), ' photons and ', total_localizations,
          'localization read in')
    drift = pd.read_csv(drift_file, delimiter=' ',names =['x','y']) 
    lifetime = np.ones(len(localizations))
    lt_photons = np.ones(len(localizations), dtype=int)
    counter = 0
    # iterating over every pick in file
    for g in set(localizations['group']):
        locs_group = localizations[(localizations.group == g)]
        print(len(locs_group), 'localizations in current group.')
        pick_photons = get_pick_photons(locs_group, photons, 
                                        drift, offset,
                                        box_side_length, integration_time)
        #peak_arrival_time = calibrate_peak(locs_group, pick_photons, 
                                           #offset, box_side_length, 
                                           #integration_time)
        # iterating over every localization in pick
        for i in range(counter, counter+len(locs_group)):
            if i == 0: print('fitting lifetime of ', len(locs_group),
                             ' localizations.') 
            phot_loc = get_photons.photons_of_one_localization(locs_group.iloc[i-counter], 
                                   pick_photons,offset, 
                                   box_side_length, integration_time)
            if i % 200 == 0:print('200 fitted. Number of photons',
                                  ' in last fit: ', len(phot_loc))
            lifetime[i] = avg_lifetime_sergi_80(phot_loc, 80, 0)
            lt_photons[i] = len(phot_loc)
        counter += len(locs_group)
    localizations['lifetime'] = lifetime
    localizations['lt_photons'] = lt_photons
    core.dataframe_to_picasso(
        localizations, localizations_file)
    print(len(localizations), 'localizations tagged with lifetime')
    


def locs_lt_avg_pos(localizations_file, photons_file, 
                       drift_file, offset, box_side_length=5,
                       integration_time=200, fitting='avg'):
    '''
    tagging list of picked localizations with lifetime 
    and returning as picasso files
    IN:
    - list of picked localizations (picasso hdf5 file with 'group' column)
    - list of photons (hdf5 file)
    - drift (txt file)
    - offset (how many offsetted frames)
    OUT: 
    - picasso hdf5 file tagged with lifetime 
    - yaml file 
    '''
    #read in files
    localizations = pd.read_hdf(localizations_file, key='locs')
    total_localizations = len(localizations)
    photons = pd.read_hdf(photons_file, key='photons')
    print(len(photons), ' photons and ', total_localizations,
          'localization read in')
    drift = pd.read_csv(drift_file, delimiter=' ',names =['x','y'])    
    lifetime = np.ones(len(localizations))
    lt_photons = np.ones(len(localizations), dtype=int)
    x_position = np.ones(len(localizations))
    y_position = np.ones(len(localizations))
    counter = 0
    # iterating over every pick in file
    for g in set(localizations['group']):
        locs_group = localizations[(localizations.group == g)]
        print(len(locs_group), 'localizations in current group.')
        pick_photons = get_pick_photons(locs_group, photons, 
                                        drift, offset,
                                        box_side_length, integration_time)
        peak_arrival_time = calibrate_peak(locs_group, pick_photons, 
                                           offset, box_side_length, 
                                           integration_time)
        # iterating over every localization in pick
        for i in range(counter, counter+len(locs_group)):
            if i == 0: print('fitting lifetime of ', len(locs_group),
                             ' localizations.') 
            one_loc = locs_group.iloc[i-counter]
            phot_loc = pd.DataFrame(data=core.crop_cylinder
                                    (one_loc, pick_photons,offset, 
                                       box_side_length, integration_time))
            if i % 200 == 0:print('200 fitted. Number of photons',
                                  ' in last fit: ', len(phot_loc))
            x, y = avg_of_roi(one_loc, phot_loc, box_side_length)
            x_position[i] = x
            y_position[i] = y
            lifetime[i] = avg_lifetime_sergi_40(phot_loc, 
                                                      peak_arrival_time)
            lt_photons[i] = len(phot_loc)
        counter += len(locs_group)
    localizations['x'] = x_position
    localizations['y'] = y_position
    localizations['lifetime'] = lifetime
    localizations['lt_photons'] = lt_photons
    core.dataframe_to_picasso(
        localizations, localizations_file, '_lt_avgPos_noBg')
    print(len(localizations), 'localizations tagged with lifetime and'
          ' fitted with avg x,y position.') 
    
    
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
    fit_area = np.pi*(box_side_length/2)**2
    number_phot = (len(phot_locs)-fit_area*localization.bg)
    bg = localization.bg*fit_area
    x_pos = (np.sum(phot_locs.x) - bg*localization.x)/number_phot
    y_pos = (np.sum(phot_locs.y) - bg*localization.y)/number_phot
    return x_pos, y_pos
    



'''
def photons_of_picked_area(localizations_file, photons_file,
                         drift_file, offset, box_side_length, 
                         integration_time):
    '''
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
    pick_photons : photons in the area of all picked localizations
    over the whole imaging time 
    '''
'''
    localizations = pd.read_hdf(localizations_file, key='locs')
    photons = pd.read_hdf(photons_file, key='photons')
    drift = pd.read_csv(drift_file, delimiter=' ',names =['x','y'])
    drift = drift[::offset]
    pick_photons = get_pick_photons(localizations, photons, drift, 
                                    offset, box_side_length, 
                                    integration_time)
    return pick_photons


   '''
   
'''
def photons_of_many_picked_localizations(
        localizations_file, photons_file,
        drift_file, offset, box_side_length, 
        integration_time):
    '''
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
'''
    localizations = pd.read_hdf(localizations_file, key='locs')
    photons = pd.read_hdf(photons_file, key='photons')
    drift = pd.read_csv(drift_file, delimiter=' ',names =['x','y'])
    drift = drift[::offset]
    pick_photons = get_pick_photons(localizations, photons, drift, 
                                    offset, box_side_length, 
                                    integration_time)
    photons_locs = pd.DataFrame()
    for i in range(len(localizations)):
        photons = photons_of_one_localization(
                        localizations.iloc[i], pick_photons, offset, 
                        box_side_length, integration_time)
        photons_locs = pd.concat([photons_locs, photons], 
                                  ignore_index=True)
    print('number of photons corresponding to locs: ', len(photons_locs))
    return photons_locs
'''    
    
    
def calibrate_peak(locs_group, pick_photons, offset,
                   box_side_length, integration_time):
    '''
    Parameters
    ----------
    locs_group : localizations of this pick as pd dataframe
    pick_photons : photons of this pick as pd dataframe
    offset : how many offsetted frames
    Returns
    -------
    Position of arrival time histogram peak
    '''
    group_photons = pd.DataFrame()
    for i in range(len(locs_group)):
        phot_loc = get_photons.photons_of_one_localization(locs_group.iloc[i], pick_photons, offset, 
                               box_side_length, integration_time)
        group_photons = pd.concat([group_photons, phot_loc], 
                                  ignore_index=True)
    counts, bins = np.histogram(group_photons.dt, bins=np.arange(0, 2500))
    return np.argmax(counts)
    


def get_pick_photons(
        locs_group, photons, drift, offset,
        box_side_length, integration_time):
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
    min_x, max_x, min_y, max_y = core.min_max_box(locs_group, box_side_length)
    phot_cr = core.crop_photons(photons,
                          (min_x-0.53125-dr_x),
                          (max_x-0.53125+dr_x),
                          (min_y-0.53125-dr_y),
                          (max_y-0.53125+dr_y))
    print('number of cropped photons: ', len(phot_cr))
    # undrift photons 
    phot_cr_und = core.undrift(phot_cr, drift, offset, integration_time)
    # crop photons again after drift 
    phot_cr_und_cr = core.crop_photons(phot_cr_und, 
                                       min_x, max_x, min_y, max_y)
    print('number of cropped-undrifted-cropped photons: ', 
          len(phot_cr_und_cr))
    return phot_cr_und_cr

    
def avg_lifetime_sergi_40(loc_photons, peak, offset=50):
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
    return np.sum(np.multiply(counts_bgsub[(peak+50):2000], np.arange(1,(2000-(peak+49)))))/np.sum(counts_bgsub[(peak+50):2000])

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



def loc_boundaries(localization, offset, 
                            box_side_length, integration_time):
    x_min = localization.x - (box_side_length/2)
    x_max = x_min + box_side_length
    y_min = localization.y - (box_side_length/2)
    y_max = y_min + box_side_length
    ms_min = (localization.frame/offset) * integration_time
    ms_max = ms_min + integration_time
    return x_min, x_max, y_min, y_max, ms_min, ms_max

'''
def photons_of_one_localization(localization, pick_photons, offset, box_side_length=5, integration_time=200):
    '''
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
'''
    photons_cylinder = core.crop_cylinder(localization,
            pick_photons, offset, box_side_length, integration_time)
    
    return photons_cylinder
    '''
    
    
    
    
    
    
    