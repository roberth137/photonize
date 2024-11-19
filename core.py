#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:07:51 2024

@author: roberthollmann
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import shutil



def min_max_box(localizations, box_side_length=0):
    
    '''
    Returns x, y, boundaries + box 
    for a set of localizations as pd dataframe
    '''
    min_x = min(localizations.x)-(box_side_length/2)
    max_x = max(localizations.x)+(box_side_length/2)
    min_y = min(localizations.y)-(box_side_length/2)
    max_y = max(localizations.y)+(box_side_length/2)
    
    return min_x, max_x, min_y, max_y



def loc_boundaries(localization, offset, 
                            box_side_length, integration_time):
    '''
    Returns boundaries of a single localization (pd Series) 
    for photons filtering as a rectangular box 
    
    dimensions: 
        -x
        -y
        -ms

    '''
    
    x_min = localization.x - (box_side_length/2)
    x_max = x_min + box_side_length
    
    y_min = localization.y - (box_side_length/2)
    y_max = y_min + box_side_length
    
    ms_min = (localization.frame/offset) * integration_time
    ms_max = ms_min + integration_time
    
    return x_min, x_max, y_min, y_max, ms_min, ms_max



def crop_cylinder(localization, photons, offset, 
                  box_side_length, integration_time):
    '''
    Parameters
    ----------
    localization : single localization as pd Series
    photons : photons as pd DataFrame
    offset :
    box_side_length :
    integration_time : 

    Returns
    -------
    photons_cylinder : All photons from the current frame closer than 
    box_side_length/2 to the localization position 

    '''
    
    #x_pos = localization.x
    #y_pos = localization.y
    
    x_min, x_max, y_min, y_max, ms_min, ms_max = loc_boundaries(
        localization, offset, box_side_length, integration_time)
    
    photons_cropped = pd.DataFrame(data = crop_photons(
                                    photons, 
                                    x_min, x_max, 
                                    y_min, y_max, 
                                    ms_min, ms_max))
    #print('type of cropped photons: ', type(photons_cropped))
    
    x_distance = (photons_cropped['x'].to_numpy() - localization.x)
    y_distance = (photons_cropped['y'].to_numpy() - localization.y)
    
    total_distance_sq = np.square(x_distance) + np.square(y_distance)
    photons_cropped['distance'] = total_distance_sq
    
    radius_sq = ((0.5*box_side_length)**2)
    photons_cylinder = photons_cropped[
        photons_cropped.distance < radius_sq]
    
    return photons_cylinder



def crop_photons(photons, x_min=0, x_max=float('inf'), y_min=0, 
                 y_max=float('inf'), ms_min=0, ms_max=float('inf')):
    '''
    Parameters
    ----------
    photons : photons as pd dataframe
    x_min :
    x_max :
    y_min :
    y_max :
    ms_min : optional The default is None.
    ms_max : optional The default is None.

    Returns
    -------
    cropped photons as pd dataframe

    '''
    
    photons_cropped = photons[
        (photons.x>=x_min)
        &(photons.x<=x_max)
        &(photons.y>=y_min)
        &(photons.y<=y_max)
        &(photons.ms>=ms_min)
        &(photons.ms<=ms_max)]
    
    return photons_cropped 



def dataframe_to_picasso(dataframe, filename, extension='_lt'):
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
    yaml_new = (yaml_old[:-5] + extension + '.yaml')
    shutil.copyfile(yaml_old, yaml_new) 
    
    hf = h5py.File(path + '/' + filename[:-5] + extension +'.hdf5', 'w')
    hf.create_dataset('locs', data=locs)
    hf.close()
    print('dataframe succesfully saved in picasso format.')



def undrift(photons, drift, offset, integration_time=200):
    '''
    IN: 
    - photon_index - list of all photons (x, y, dt, ms) as pd dataframe
    - drift_file - picasso generated as pd DataFrame
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
    ms_index = np.copy(photons.ms)
    frames = np.floor((offset*ms_index)/integration_time).astype(int)
    drift_x = np.copy(drift.x)
    drift_y = np.copy(drift.y)
    
    #create numpy arrays to speed up
    number_photons = len(photons)
    undrifted_x = np.copy(photons.x)
    undrifted_y = np.copy(photons.y)
    drift_x_array = np.ones(number_photons)
    drift_y_array = np.ones(number_photons)
    
    for i in range(number_photons):
        frame = frames[i]
        drift_x_array[i] = drift_x[frame]
        drift_y_array[i] = drift_y[frame]
        if i == 0: print('start undrifting')
        elif i %10000000 == 0: print('100mio undrifted')
        
    #apply drift and shift of 0.53125 to photons -> synchron in position 
    #with Localizations
    undrifted_x += (0.53125-drift_x_array)
    undrifted_y += (0.53125-drift_y_array)
    
    #create and return new dataframe 
    photons_undrifted = pd.DataFrame({'x': undrifted_x, 
        'y': undrifted_y, 'dt': photons.dt, 'ms': photons.ms})
    
    return photons_undrifted

