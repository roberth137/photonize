import numpy as np
import pandas as pd
import get_photons


def avg_lifetime_sergi_40(loc_photons, peak, dt_offset=0):
    '''
    Fit lifetimes of individual localizations with 40mhz laser frequency
    Parameters
    ----------
    loc_photons : all photons from one localization
    peak : position of the maximum of arrival times for this pick of
    localizations, calibrated from calibrate_peak_locs()
    offset : the offset from the peak where arrival times are considered
    for fitting the lifetime.The default is 50.

    Returns
    -------
    average arrival time of photons, in units of arrival time bin size.
    Usually 10ps

    '''

    counts, bins = np.histogram(loc_photons.dt, bins=np.arange(0, 2500))
    background = np.sum(counts[-300:]) / 300
    counts_bgsub = counts - background
    weights = np.arange(1, (2500 - (peak + dt_offset)))
    considered_bgsub = counts_bgsub[(peak + dt_offset):2500]
    #if len(loc_photons) < 70:
    #    print('\nphotons for fitting: ', len(loc_photons))
    #    print('good photons: ', sum(considered_bgsub))
    lifetime = np.sum(np.multiply(considered_bgsub, weights)) / np.sum(considered_bgsub)
    return lifetime


def avg_lifetime_sergi_80(loc_photons, peak, dt_offset=50):
    '''
    Fit lifetimes of individual localizations with 80mhz laser frequency
    Parameters
    ----------
    loc_photons : all photons from one localization
    peak : position of the maximum of arrival times for this pick of
    localizations, calibrated from calibrate_peak_locs()
    offset : the offset from the peak where arrival times are considered
    for fitting the lifetime.The default is 50.

    Returns
    -------
    average arrival time of photons, in units of arrival time bin size.
    Usually 10ps

    '''
    counts, bins = np.histogram(loc_photons.dt, bins=np.arange(0, 1250))
    background = np.sum(counts[-300:]) / 300
    counts_bgsub = counts - background
    weights = np.arange(1, (1250 - (peak + dt_offset)))
    considered_bgsub = counts_bgsub[(peak + dt_offset):1250]
    lifetime = np.sum(np.multiply(considered_bgsub, weights)) / np.sum(considered_bgsub)
    return lifetime

def calibrate_peak_locs(locs_group, pick_photons, offset,
                   box_side_length, int_time):
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
                                                           box_side_length, int_time)
        group_photons = pd.concat([group_photons, phot_loc],
                                  ignore_index=True)
    counts, bins = np.histogram(group_photons.dt, bins=np.arange(0, 2500))
    print('len photons for calib_peak: ', len(group_photons))
    return np.argmax(counts)

def calibrate_peak_events(event_photons):
    '''
    Parameters
    ----------
    All photons of the current fov that arrive during events
    Returns
    -------
    Position of arrival time histogram peak
    '''
    counts, bins = np.histogram(event_photons.dt, bins=np.arange(0, 2500))
    print('len photons for calib_peak: ', len(event_photons))
    return np.argmax(counts)