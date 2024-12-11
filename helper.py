#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:37:09 2024

@author: roberthollmann
"""

import pandas as pd

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