#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:37:09 2024

@author: roberthollmann
"""

import pandas as pd
import h5py
from pathlib import Path
import shutil


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
    locs = df_picasso.to_records(index=False)

    # Saving data
    yaml_old = (path + '/' + filename[:-4] + 'yaml')
    yaml_new = (yaml_old[:-5] + extension + '.yaml')
    shutil.copyfile(yaml_old, yaml_new)

    hf = h5py.File(path + '/' + filename[:-5] + extension + '.hdf5', 'w')
    hf.create_dataset('locs', data=locs)
    hf.close()
    print('\ndataframe succesfully saved in picasso format.')

def process_input(input_data, dataset):
    """
    Processes the input to determine if it is a filename or a pandas DataFrame.

    Parameters:
        input_data (str or pd.DataFrame): The input, which can be a string (filename) or a DataFrame.

    Returns:
        pd.DataFrame: The resulting DataFrame from the input.
    """
    if isinstance(input_data, str) and input_data.endswith(('.hdf5', '.txt')):
        # Attempt to read the HDF5 file
        try:
            if dataset == 'locs':
                data = pd.read_hdf(input_data, key=dataset)
            elif dataset == 'photons':
                data = pd.read_hdf(input_data, key=dataset)
            elif dataset == 'drift':
                data = pd.read_csv(input_data, delimiter=' ',names =['x','y']) 
            return data
        
            #if dataset == 'photons':
            #    photons = pd.read_hdf(input_data, key=dataset)
            #return photons
        except Exception as e:
            raise ValueError(f"Failed to read the HDF5 file: {e}")
    elif isinstance(input_data, pd.DataFrame):
        return input_data
    else:
        raise TypeError("Input must be a string ending with '.txt', '.hdf5' or a pandas DataFrame.")


def validate_columns(dataframe, required_columns):
    """
    Validates if required columns are in the DataFrame.
    Returns a tuple (missing_columns, has_all_columns).
    """
    missing = [col for col in required_columns if col not in dataframe.columns]
    if len(missing) > 0:
        print(missing)