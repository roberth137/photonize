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
from datetime import datetime
import os



def dataframe_to_picasso(dataframe, filename, extension=None, yaml_dump=None):
    '''

    Parameters
    ----------
    dataframe : dataframe in picasso format (with all necessary columns)
    filename : name with which the file will be saved
    extension : extension to filename
    add_content

    DO: takes a dataframe and saves it to picasso format
    The corresponding yaml file has to be in the same directory and will be copied
    '''

    path = Path.cwd()
    labels = list(dataframe.keys())
    df_picasso = dataframe.reindex(columns=labels, fill_value=1)
    locs = df_picasso.to_records(index=False)

    # Get the full path of the original YAML file
    yaml_old = Path(filename).with_suffix('.yaml')  # Ensures correct extension handling

    # Construct new YAML filename
    if extension:
        yaml_new = yaml_old.with_name(f"{yaml_old.stem}{extension}.yaml")
    else:
        yaml_new = yaml_old

    try:
        # Ensure the original YAML file exists
        if not yaml_old.exists():
            print(f"Error: The file '{yaml_old}' was not found.")
            return

        # Copy and optionally modify the YAML file
        shutil.copyfile(yaml_old, yaml_new)
        if yaml_dump:
            with open(yaml_new, 'a') as file:
                file.writelines([f"{line.strip()}\n" for line in yaml_dump.split('\n')])

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    hf = h5py.File(path / f"{filename[:-5]}{extension}.hdf5", 'w')
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

def calculate_total_photons(localizations, box_side_length):
    if {'photons', 'bg'}.issubset(localizations.columns):
        photons_arr = localizations['photons'].to_numpy()
        bg_arr = localizations['bg'].to_numpy()
        total_photons = photons_arr + (bg_arr * box_side_length ** 2)
        localizations.insert(4, 'total_photons', total_photons)
        return localizations
    else:
        raise ValueError("DataFrame must contain 'photons' and 'bg' columns.")

def create_append_message(function, **info):
    message = f"\n---\nGenerated by: {function}"
    for key,value in info.items():
        if 'file' in key.lower():
            try:
                full_path = os.path.abspath(info[key])
                message += f"\n{key} (full path): {full_path}"
            except Exception as e:
                message += f"\n{key}: {info[key]} (could not retrieve full path: {e})"
        else:
            message += f"\n{key}: {info[key]}"
        # Append the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message += f"\nTimestamp: {current_time}"

    return message

def crop_photons_file(new_photons_df, photons_file, extension='_20k'):
    '''

    Parameters
    ----------
    dataframe : dataframe in picasso format (with all necessary columns)
    filename : name with which the file will be saved

    DO: takes a dataframe and saves it to picasso format
    The corresponding yaml file has to be in the same directory and will be copied
    '''

    path = str(Path.cwd())
    labels = list(new_photons_df.keys())
    new_photons_df = new_photons_df.reindex(columns=labels, fill_value=1)
    photons = new_photons_df.to_records(index=False)


    hf = h5py.File(path + '/' + photons_file[:-5] + extension + '.hdf5', 'w')
    hf.create_dataset('photons', data=photons)
    hf.close()
    print('\nphotons successfully saved.')

def crop_drift(drift_file):
    """
    Modify a drift file to use only for specific frames
    """
    # Load the data from drift.txt
    drift_file = pd.read_csv(drift_file, delimiter=' ', names=['x', 'y'])

    # Keep only the first 20,000 rows
    drift_20k = drift_file.iloc[120000:140000]

    # Save the reduced dataset to drift_20k.txt
    drift_20k.to_csv('drift_L20.txt', sep=' ', index=False, header=False)

    print("Saved first 20k rows to drift_20k.txt")