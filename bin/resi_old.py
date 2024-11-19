#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 20:31:38 2024

@author: roberthollmann
"""
import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import math
import h5py
from pathlib import Path
import copy
#import matplotlib as mpl
#import scipy.io as sio
from scipy.optimize import curve_fit
#import shutil


def phot_of_locs(localizations_file, photons_file, drift_file, bsl=5, integration_time=200):
    '''
    returning photons for list of localizations
    IN:
    - list of localizations (hdf5)
    - list of photons (hdf5)
    - drift (txt)
    OUT: 
    - photons as pandas dataframe, tagged with localization number 
    '''
    localizations = pd.read_hdf(localizations_file, key='locs')
    photons = pd.read_hdf(photons_file, key='photons')
    drift = pd.read_csv(drift_file, delimiter=' ',names =['x','y'])
    dr_x, dr_y = max(abs(drift.x)), max(abs(drift.y))
    for g in range((localizations['group'].iloc[-1])+1):
        # grab photons for every group
        locs_group = localizations[(localizations.group == g)]
        min_x, max_x, min_y, max_y = get_min_max(locs_group)
        phot_crop = photons[(photons.x > (min_x-0.5-(bsl/2)-dr_x)) &
                (photons.x < (max_x-0.5+(bsl/2)+dr_x)) &
                (photons.y > (min_y-0.5-(bsl/2)-dr_y)) & 
                (photons.y < (max_y-0.5+(bsl/2)+dr_y))]
        phot_cr_und = undrift(phot_crop, drift, integration_time)
        phot_cr_und_cr = phot_cr_und[(phot_cr_und.x > (min_x-0.5-(bsl/2))) & 
                (phot_cr_und.x < (max_x-0.5+(bsl/2))) &
                (phot_cr_und.y > (min_y-0.5-(bsl/2))) & 
                (phot_cr_und.y < (max_y-0.5+(bsl/2)))]
        for i in range(len(locs_group)):
            if i == 0:
                loc_photons = get_photons(locs_group.iloc[i], phot_cr_und_cr, box_size=bsl, integration_time=integration_time)
                locs_photons = pd.DataFrame(data=loc_photons)
                loc_tag = (np.ones(len(loc_photons))*i).astype(int)
                locs_photons.loc[:, ['localization']] = loc_tag
            else:
                loc_photons2 = get_photons(locs_group.iloc[i], phot_cr_und_cr, box_size=bsl, integration_time=integration_time)
                loc_tag = (np.ones(len(loc_photons2))*i).astype(int)
                loc_photons2.loc[:, ['localization']] = loc_tag
                locs_photons = pd.concat([locs_photons, loc_photons2], ignore_index=True)
    return locs_photons



def photons_to_picasso(photons_dataframe, integration_time=200):
    phot_frame = copy.copy(photons_dataframe)
    frame_arr, dummy = np.ones(len(phot_frame)), np.ones(len(phot_frame))
    phot_frame = phot_frame.reset_index(drop=True)
    for i in range(len(photons_dataframe)):
        frame_arr[i] = int(math.floor(phot_frame.ms[i]/integration_time))
    phot_frame.insert(0, "frame", frame_arr)
    phot_frame.insert(3, "photons", frame_arr)
    phot_frame.insert(4, "sx", dummy)
    phot_frame.insert(5, "sy", dummy)
    phot_frame.insert(6, "bg", dummy)
    phot_frame.insert(7, "lpx", dummy)
    phot_frame.insert(8, "lpy", dummy)
    phot_frame.insert(9, "ellipticity", frame_arr)
    phot_frame.insert(10, "net_gradient", dummy)
    hdf5_fname = 'photons_to_picasso.hdf5'
    path = str(Path.cwd())
    labels = list(phot_frame.keys())
    df_picasso = phot_frame.reindex(columns=labels, fill_value=1)
    locs = df_picasso.to_records(index = False)
    # Saving data
    hf = h5py.File(path + '/' + hdf5_fname, 'w')
    hf.create_dataset('locs', data=locs)
    hf.close()


def lin_dec(x, a, b):
    return a*x + b


def locs_lt_to_dataframe(localizations_file, photons_file, drift_file, fitting='avg', bsl=5, integration_time=200):
    '''
    tagging list of picked localizations with lifetime and returning as picasso files
    IN:
    - list of picked localizations (picasso hdf5 file with 'group' column)
    - list of photons (hdf5 file)
    - drift (txt file)
    OUT: 
    - picasso hdf5 file tagged with lifetime 
    - yaml file 
    '''
    #read in files
    localizations = pd.read_hdf(localizations_file, key='locs')
    photons = pd.read_hdf(photons_file, key='photons')
    print(len(photons), ' photons read in')
    drift = pd.read_csv(drift_file, delimiter=' ',names =['x','y'])
    drift = drift[::4]
    dr_x, dr_y = max(abs(drift.x)), max(abs(drift.y))
    lifetime = np.ones(len(localizations))
    lt_fit_photons = np.ones(len(localizations))
    counter = 0
    for g in range((localizations['group'].iloc[-1])+1):
        # grab photons for every group
        locs_group = localizations[(localizations.group == g)]
        min_x, max_x, min_y, max_y = get_min_max(locs_group)
        phot_crop = photons[(photons.x > (min_x-0.5-(bsl/2)-dr_x)) &
                (photons.x < (max_x-0.5+(bsl/2)+dr_x)) &
                (photons.y > (min_y-0.5-(bsl/2)-dr_y)) & 
                (photons.y < (max_y-0.5+(bsl/2)+dr_y))]
        phot_cr_und = undrift(phot_crop, drift, integration_time)
        phot_cr_und_cr = phot_cr_und[(phot_cr_und.x > (min_x-0.5-(bsl/2))) & 
                (phot_cr_und.x < (max_x-0.5+(bsl/2))) &
                (phot_cr_und.y > (min_y-0.5-(bsl/2))) & 
                (phot_cr_und.y < (max_y-0.5+(bsl/2)))]
        for i in range(len(locs_group)):
            if i == 0: print('fitting lifetime of ', len(locs_group), ' localizations.') 
            elif i % 50 == 0: print('50 fitted.')
            phot_loc = get_photons(locs_group.iloc[i], phot_cr_und_cr, 5 ,200)
            lifetime[counter] = avg_lifetime_sergi_40(phot_loc)
            lt_fit_photons[counter] = len(phot_loc)
            counter +=1
    localizations['lifetime'] = np.array(lifetime)
    localizations['lt_fit_photons'] = np.array(lt_fit_photons, dtype='uint16')
    print(len(localizations), 'localizations tagged with lifetime')
    return localizations

def fit_lifetime_log(loc_photons, dt_units_ps=10, d_points=8, x_offs=1):
    dt_counts, dt_bins, _ = plt.hist(loc_photons.dt[(loc_photons.dt<2000)&(loc_photons.dt>200)], bins=50)
    binsize = dt_bins[2]-dt_bins[1]
    x_axis = range(d_points)
    max_pos = np.argmax(dt_counts)
    dt_counts_cut = dt_counts[max_pos+x_offs:max_pos+d_points+x_offs]
    if np.all(dt_counts_cut >1):
        log_dt_counts_cut = np.log(dt_counts_cut)
    else:
        log_dt_counts_cut = (np.ones(d_points)*d_points)-np.arange(d_points)
        print('err with hist')
    p_lin, _ = curve_fit(lin_dec, np.arange(len(log_dt_counts_cut)), log_dt_counts_cut)
    return -(1/p_lin[0])*binsize*dt_units_ps

def get_min_max(localizations):
    '''
    IN: -list of localizations/driftfile as pandas dataframe
    OUT: -  min_x, max_x, min_y, max_y
    '''
    return min(localizations.x), max(localizations.x), min(localizations.y), max(localizations.y)

def get_photons(localization, photons_file, offset=1, box_size=5, integration_time=200):
    '''
    Returns photons of localization
    IN: 
    - localization (picasso format)
    - photons (dataframe format, undrifted)
    - box_size in pixel 
    - int time in ms
    OUT:
    - photons (dataframe format)
    '''
    x_min, x_max = (localization.x-(box_size/2)-0.5), (localization.x+(box_size/2)-0.5)
    y_min, y_max = (localization.y-(box_size/2)-0.5), (localization.y+(box_size/2)-0.5)
    start_ms = ((localization.frame/offset)*integration_time)
    ms_min, ms_max = start_ms, start_ms+integration_time
    photons_loc = photons_file[(photons_file.x>x_min)&(photons_file.x<x_max)&
                        (photons_file.y>y_min)&(photons_file.y<y_max)&
                        (photons_file.ms>ms_min)&(photons_file.ms<ms_max)]
    return photons_loc

def undrift(photons_index, drift_file, offset, integration_time=200):
    '''
    IN: 
    - photon_index - list of all photons (x, y, dt, ms) as pandas dataframe
    - drift_file - drift positions for all frames (picasso generated) as pandas dataframe
    - integration_time - camera integration time 
    OUT: 
    undrifted photons_index as dataframe
    '''
    # create frame array
    #frame_arr = np.ones(len(photons_index))
    ms_index = np.copy(photons_index.ms)
    frames = np.floor(ms_index/(integration_time*offset)).astype(int)
    for x in range(1,len(frames), 50000000):
        print(frames[x])
    last_frame = np.max(frames)
    for i in range(len(frames)):
        if frames[i] == last_frame: 
            frames[i] = (last_frame-1)
    #create numpy arrays to speed up 
    num_phot = len(photons_index)
    undrifted_x = np.copy(photons_index.x)
    undrifted_y = np.copy(photons_index.y) 
    drift_x = np.copy(drift_file.x)
    drift_y = np.copy(drift_file.y)
    drift_x_array = np.ones(num_phot)
    drift_y_array = np.ones(num_phot)
    for i in range(num_phot):
        frame = frames[i]
        drift_x_array[i] = drift_x[frame]
        drift_y_array[i] = drift_y[frame]
        if i == 0: print('start undrifting')
        elif i %10000000 == 0: print('10mio undrifted')
    #apply drift 
    undrifted_x -= drift_x_array
    undrifted_y -= drift_y_array
    #create and return new dataframe 
    photons_index_new = pd.DataFrame({'x': undrifted_x, 'y': undrifted_y, 'dt': photons_index.dt
                                      , 'ms': photons_index.ms})
    return photons_index_new 

def avg_lifetime_sergi_40(loc_photons, peak, offset=50):
    counts, bins = np.histogram(loc_photons.dt, bins=np.arange(0,2500))
    background = np.sum(counts[-300:])/300
    counts_bgsub = counts - background
    return np.sum(np.multiply(counts_bgsub[(peak+50):2000], np.arange(1,(2000-(peak+49)))))/np.sum(counts_bgsub[(peak+50):2000])

def avg_lifetime_sergi_80(loc_photons, peak, offset=50):
    counts, bins = np.histogram(loc_photons.dt, bins=np.arange(0,1250))
    background = (np.sum(counts[:(peak-50)])+np.sum(counts[1150:]))/(peak+50)
    counts_bgsub = counts - background
    
    return np.sum(np.multiply(counts_bgsub[(peak+50):1150], np.arange(1,(2000-(peak+49)))))/np.sum(counts_bgsub[(peak+50):2000])

def avg_lifetime_sergi_cut(loc_photons, start_hist=110, end_hist=2500, offset=50):
    counts, bins = np.histogram(loc_photons.dt, bins=np.arange(start_hist,end_hist))
    background = np.sum(counts[-300:])/300
    counts_bgsub = counts - background
    return np.sum(np.multiply(counts_bgsub, np.arange(1,len(bins))))/np.sum(counts_bgsub)