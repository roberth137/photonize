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
    drift = drift[::offset]     
    lifetime = np.ones(len(localizations))
    lt_photons = np.ones(len(localizations), dtype=int)
    counter = 0
    # iterating over every pick in file
    for g in range((localizations['group'].iloc[-1])+1):
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
            phot_loc = photons_of_one_localization(locs_group.iloc[i-counter], 
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
    dataframe_to_picasso(localizations, localizations_file)
    print(len(localizations), 'localizations tagged with lifetime')
    
    
    
def photons_of_picked_area(localizations_file, photons_file,
                         drift_file, offset, box_side_length, 
                         integration_time):
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
    localizations = pd.read_hdf(localizations_file, key='locs')
    photons = pd.read_hdf(photons_file, key='photons')
    drift = pd.read_csv(drift_file, delimiter=' ',names =['x','y'])
    drift = drift[::offset]
    pick_photons = get_pick_photons(localizations, photons, drift, 
                                    offset, box_side_length, 
                                    integration_time)
    return pick_photons
    
def photons_of_many_picked_localizations(
        localizations_file, photons_file,
        drift_file, offset, box_side_length, 
        integration_time):
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
        phot_loc = photons_of_one_localization(locs_group.iloc[i], pick_photons, offset, 
                               box_side_length, integration_time)
        group_photons = pd.concat([group_photons, phot_loc], 
                                  ignore_index=True)
    counts, bins = np.histogram(pick_photons.dt, bins=np.arange(0, 2500))
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
    # -0.5 because position 0 on camera means pixel 1 
    dr_x, dr_y = max(abs(drift.x)), max(abs(drift.y))
    min_x, max_x, min_y, max_y = get_min_max(locs_group)
    phot_crop = photons[(photons.x > (min_x-0.53125-(box_side_length/2)-dr_x))
                        &(photons.x < (max_x-0.53125+(box_side_length/2)+dr_x))
                        &(photons.y > (min_y-0.53125-(box_side_length/2)-dr_y))
                        &(photons.y < (max_y-0.53125+(box_side_length/2)+dr_y))]
    print('number of cropped photons: ', len(phot_crop))
    # undrift photons 
    phot_cr_und = undrift(phot_crop, drift, offset, integration_time)
    # crop photons again after drift 
    phot_cr_und_cr = phot_cr_und[
        (phot_cr_und.x > (min_x-(box_side_length/2))) 
        &(phot_cr_und.x < (max_x+(box_side_length/2))) 
        &(phot_cr_und.y > (min_y-(box_side_length/2))) 
        &(phot_cr_und.y < (max_y+(box_side_length/2)))]
    print('number of cropped-undrifted-cropped photons: ', 
          len(phot_cr_und_cr))
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

def photons_of_one_localization(localization, pick_photons, offset, box_side_length=5, integration_time=200):
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
    x_min = (localization.x-(box_side_length/2))
    x_max = x_min + box_side_length
    y_min = (localization.y-(box_side_length/2))
    y_max = y_min + box_side_length
    ms_min = ((localization.frame/offset)*integration_time)
    ms_max = ms_min + integration_time
    photons_loc = pick_photons[
        (pick_photons.x>x_min)&(pick_photons.x<x_max)
        &(pick_photons.y>y_min)&(pick_photons.y<y_max)
        &(pick_photons.ms>ms_min)&(pick_photons.ms<ms_max)]
    return photons_loc
    

def undrift(photons_index, drift_file, offset, integration_time=200):
    '''
    IN: 
    - photon_index - list of all photons (x, y, dt, ms) as pandas dataframe
    - drift_file - picasso generated
    - integration_time
    OUT: 
    undrifted photons_index as dataframe
    
    
    Note: drift array is subtracted from locs to get undrifted coordinates
    0.53125 is added to coordinates to convert coordinates 
    from camera pixels (LINCAM) to TIFfile to picasso coordinates.
    
    Formula: Picasso_coord = LIN_coord + 0.5 + (1/(2*Binning))
    
    For 16x Binning: P_c = L_c + 0.5 +(1/(2*16)) = L_c + 0.53125
    
    '''
    # create frame array
    #frame_arr = np.ones(len(photons_index))
    ms_index = np.copy(photons_index.ms)
    frames = np.floor(ms_index/(integration_time)).astype(int)
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
        #if i == 0: print('start undrifting')
        #elif i %10000000 == 0: print('10mio undrifted')
    #apply drift and shift of 0.53125 to photons -> synchron in position 
    #with Localizations
    undrifted_x += (0.53125-drift_x_array)
    undrifted_y += (0.53125-drift_y_array)
    #create and return new dataframe 
    photons_index_undrifted = pd.DataFrame({'x': undrifted_x, 
        'y': undrifted_y, 'dt': photons_index.dt, 'ms': photons_index.ms})
    return photons_index_undrifted


