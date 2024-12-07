#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:54:44 2024

@author: roberthollmann and chatgpt 
"""
import numpy as np
cimport numpy as np
from math import floor

cdef inline int map_to_bin(int value, int bin_size):
    """
    Map a coordinate value (0 to 2358) to a grid bin index.
    """
    return floor(value / bin_size)

def sort_photons(
    np.ndarray[np.int16_t, ndim=1] x_coords,
    np.ndarray[np.int16_t, ndim=1] y_coords,
    np.ndarray[np.int16_t, ndim=1] dt_values,
    np.ndarray[np.int32_t, ndim=1] ms_values,
    int num_bins_x=148,
    int num_bins_y=148
):
    """
    Sort photons into a 148x148 grid based on their x and y positions.
    Parameters:
        x_coords: ndarray of int16, photon x-coordinates (0 to 2358).
        y_coords: ndarray of int16, photon y-coordinates (0 to 2358).
        dt_values: ndarray of int16, photon dt values.
        ms_values: ndarray of int32, photon ms values.
    Returns:
        A 148x148 grid where each cell contains a list of tuples (x, y, dt, ms).
    """
    cdef int num_photons = x_coords.shape[0]
    cdef int i, x_bin, y_bin
    cdef int bin_size = 16  # Fixed bin size

    # Initialize the grid dynamically
    grid = [[[] for _ in range(num_bins_y)] for _ in range(num_bins_x)]

    # Iterate through the photons and bin them
    for i in range(num_photons):
        x_bin = map_to_bin(x_coords[i], bin_size)
        y_bin = map_to_bin(y_coords[i], bin_size)

        # Add the photon to the corresponding bin as a tuple
        grid[x_bin][y_bin].append((x_coords[i], y_coords[i], dt_values[i], ms_values[i]))

    return grid