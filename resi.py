# RESI functions
import numpy as np
#import scipy as sp
#import matplotlib.pyplot as plt
import pandas as pd
#import math
import h5py
from pathlib import Path
#import copy
#import matplotlib as mpl
#import scipy.io as sio
#from scipy.optimize import curve_fit
import shutil


def locs_lt_to_picasso(localizations_file, photons_file, 
                       drift_file, offset, fitting='avg', 
                       box_side_length=5, integration_time=200):
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
    photons = pd.read_hdf(photons_file, key='photons')
    print(len(photons), ' photons read in')
    drift = pd.read_csv(drift_file, delimiter=' ',names =['x','y'])
    drift = drift[::offset]
    lifetime = np.ones(len(localizations))
    lt_photons = np.ones(len(localizations))
    counter = 0
    for g in range((localizations['group'].iloc[-1])+1):
        # grab photons for every group
        locs_group = localizations[(localizations.group == g)]
        print(len(locs_group), 'localizations in first group.')
        pick_photons = get_pick_photons(locs_group, photons, 
                                        drift, offset,
                                        box_side_length, integration_time)
        '''
        for i in range(len(locs_group)):
            if i == 0:
                phot_loc = get_photons(locs_group.iloc[i], pick_photons, offset, 5 ,200)
                group_photons = pd.DataFrame(data=phot_loc)
            else:
                phot_loc = get_photons(locs_group.iloc[i], pick_photons, offset, 5 ,200)
                group_photons = pd.concat([group_photons, phot_loc], ignore_index=True)
        peak = calibrate_dt(group_photons)
        '''
        peak = calibrate_peak(locs_group, pick_photons, offset, 
                              box_side_length, integration_time)
        for i in range(len(locs_group)):
            if i == 0: print('fitting lifetime of ', len(locs_group), ' localizations.') 
            phot_loc = get_photons(locs_group.iloc[i], pick_photons, offset, 5 ,200)
            if i % 50 == 0:print('50 fitted. Number of photons in last fit ', len(phot_loc))
            lifetime[counter] = avg_lifetime_sergi_40(phot_loc, peak)
            lt_photons[counter] = len(phot_loc)
            if i ==0: print('fitting ', len(phot_loc), ' photons.')
            counter +=1
    localizations['lifetime'] = lifetime
    localizations['lt_photons'] = lt_photons
    dataframe_to_picasso(localizations, localizations_file)
    print(len(localizations), 'localizations tagged with lifetime')
    
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
        '''
        if i == 0:
            phot_loc = get_photons(locs_group.iloc[i], pick_photons, offset, 
                                   box_side_length, integration_time)
            group_photons = pd.DataFrame(data=phot_loc)
        else:
            phot_loc = get_photons(locs_group.iloc[i], pick_photons, offset, 5 ,200)
            group_photons = pd.concat([group_photons, phot_loc], ignore_index=True)
        '''
        phot_loc = get_photons(locs_group.iloc[i], pick_photons, offset, 
                               box_side_length, integration_time)
        group_photons = pd.concat([group_photons, phot_loc], ignore_index=True)
    peak = calibrate_dt(group_photons)
    return peak
    
    
    
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
    print('running "get_pick_photons()".')
    # set dimensions of the region and crop photons 
    # -0.5 because position 0 on camera means pixel 1 
    dr_x, dr_y = max(abs(drift.x)), max(abs(drift.y))
    min_x, max_x, min_y, max_y = get_min_max(locs_group)
    print('dimensions of pick are min_x,max_x, min_y,max_y', 
          min_x, max_x, min_y, max_y)
    print('box side length is: ', box_side_length)
    print('drift values are dr_x, dr_y', dr_x, dr_y)
    phot_crop = photons[(photons.x > (min_x-0.5-(box_side_length/2)-dr_x))
                        &(photons.x < (max_x-0.5+(box_side_length/2)+dr_x))
                        &(photons.y > (min_y-0.5-(box_side_length/2)-dr_y))
                        &(photons.y < (max_y-0.5+(box_side_length/2)+dr_y))]
    print('number of cropped photons: ', len(phot_crop))
    # undrift photons 
    phot_cr_und = undrift(phot_crop, drift, offset, integration_time)
    # crop photons again after drift 
    phot_cr_und_cr = phot_cr_und[
        (phot_cr_und.x > (min_x-0.5-(box_side_length/2))) 
        &(phot_cr_und.x < (max_x-0.5+(box_side_length/2))) 
        &(phot_cr_und.y > (min_y-0.5-(box_side_length/2))) 
        &(phot_cr_und.y < (max_y-0.5+(box_side_length/2)))]
    return phot_cr_und_cr
    
    
    
def dataframe_to_picasso(dataframe, filename):
    '''

    Parameters
    ----------
    dataframe : dataframe in picasso format (with all necessary columns)
    filename : name with which the file will be saved
    
    DO: takes a dataframe and saves it to picasso format
    The corresponding yaml file has to be in the same directory and will be copied
    '''
    path = str(Path.cwd())
    labels = list(dataframe.keys())
    df_picasso = dataframe.reindex(columns=labels, fill_value=1)
    locs = df_picasso.to_records(index = False)
    # Saving data
    yaml_old = (path + '/' + filename[:-4] + 'yaml')
    yaml_new = (yaml_old[:-5] + '_lt.yaml')
    shutil.copyfile(yaml_old, yaml_new) 
    hf = h5py.File(path + '/' + filename[:-5] +'_lt.hdf5', 'w')
    hf.create_dataset('locs', data=locs)
    hf.close()
    print('dataframe succesfully saved in picasso format.')


def calibrate_dt(photons):
    counts, bins = np.histogram(photons.dt, bins=np.arange(0, 2500))
    return np.argmax(counts)

    
def avg_lifetime_sergi_40(loc_photons, peak, offset=50):
    counts, bins = np.histogram(loc_photons.dt, bins=np.arange(0,2500))
    background = np.sum(counts[-300:])/300
    counts_bgsub = counts - background
    return np.sum(np.multiply(counts_bgsub[(peak+50):2000], np.arange(1,(2000-(peak+49)))))/np.sum(counts_bgsub[(peak+50):2000])


def get_min_max(localizations):
    '''
    IN: -list of localizations/driftfile as pandas dataframe
    OUT: -  min_x, max_x, min_y, max_y
    '''
    return min(localizations.x), max(localizations.x), min(localizations.y), max(localizations.y)

def get_photons(localization, photons_file, offset, box_size=5, integration_time=200):
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
    frames = np.floor(ms_index/(integration_time)).astype(int)
    for x in range(1,len(frames), 50000000):
        print('frames vector has value: min, max',
              min(frames), max(frames))
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