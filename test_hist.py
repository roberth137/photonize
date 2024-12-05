#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 22:47:23 2024

@author: roberthollmann
"""

import pandas as pd
import numpy as np 
import tifffile as tiff
from numba import njit, prange



@njit(parallel=True)
def compute_2d_histogram(positions, resolution):
    """
    Compute a 2D histogram using Numba for speed.
    
    Parameters:
        positions (np.ndarray): Array of shape (N, 2), where each row contains (x, y).
        resolution (int): Number of bins for the histogram along each axis.
    
    Returns:
        np.ndarray: 2D histogram.
    """
    histogram = np.zeros((resolution, resolution), dtype=np.uint64)
    for i in prange(len(positions)):
        x, y = positions[i]
        if 0 <= x < resolution and 0 <= y < resolution:
            histogram[int(x), int(y)] += 1
    return histogram

def create_2d_histogram_to_tiff_c(positions, output_filename, resolution=4096):
    """
    Creates a 2D histogram from photon positions and saves it as a TIFF file.
    
    Parameters:
        positions (np.ndarray): Array of shape (N, 2), where each row contains (x, y).
        output_filename (str): Path to save the resulting TIFF file.
        resolution (int): Number of bins for the histogram along each axis.
    """
    # Compute the histogram using the compiled function
    histogram = compute_2d_histogram(positions, resolution)
    
    # Convert to uint16 for TIFF compatibility
    histogram = np.clip(histogram, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    
    # Save the histogram as a TIFF file
    tiff.imwrite(output_filename, histogram)
    print(f"2D histogram saved as {output_filename}.")
