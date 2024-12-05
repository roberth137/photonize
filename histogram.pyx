#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:49:38 2024

@author: roberthollmann
"""

import numpy as np
cimport numpy as np
from math import floor

cdef struct Photon:
    float x
    float y
    int dt
    int ms

cdef inline int map_to_cell(float value, int max_value, int num_bins):
    """
    Maps a value to a grid cell index using the floor function.
    """
    cell_size = max_value / num_bins
    return floor(value / cell_size)

def create_2d_histogram(np.ndarray[np.float64_t, ndim=2] photons, int num_bins=148):
    """
    Create a 2D histogram for photons.
    """
    cdef int i, x_cell, y_cell
    cdef int num_photons = photons.shape[0]

    # Initialize the grid
    # Create a 2D list of lists dynamically
    cdef list histogram = [[[] for _ in range(num_bins)] for _ in range(num_bins)]


    cdef Photon p
    for i in range(num_photons):
        x_cell = map_to_cell(photons[i, 0], 148, num_bins)
        y_cell = map_to_cell(photons[i, 1], 148, num_bins)

        p.x = photons[i, 0]
        p.y = photons[i, 1]
        p.dt = int(photons[i, 2])
        p.ms = int(photons[i, 3])
        histogram[x_cell][y_cell].append(p)

    return histogram
