#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 21:40:07 2024

@author: roberthollmann
"""

# Import necessary modules
import numpy as np
cimport numpy as np
from libc.math cimport floor

# Define the optimized undrift function
def undrift(
    np.ndarray[np.float64_t, ndim=1] photons_x,
    np.ndarray[np.float64_t, ndim=1] photons_y,
    np.ndarray[np.float64_t, ndim=1] photons_ms,
    np.ndarray[np.float64_t, ndim=1] drift_x,
    np.ndarray[np.float64_t, ndim=1] drift_y,
    double offset,
    double int_time=200.0
):
    cdef Py_ssize_t number_photons = photons_x.shape[0]
    cdef Py_ssize_t max_frame_drift = drift_x.shape[0]
    cdef Py_ssize_t i, frame
    cdef double frame_offset = offset / int_time
    cdef double shift = 0.53125
    cdef np.ndarray[np.int32_t, ndim=1] frames = np.empty(number_photons, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] undrifted_x = np.empty(number_photons, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] undrifted_y = np.empty(number_photons, dtype=np.float64)
    
    # Calculate frames
    for i in range(number_photons):
        frame = int(floor(frame_offset * photons_ms[i]))
        if frame >= max_frame_drift:
            frame = max_frame_drift - 1  # Clip frame to max_frame_drift
        frames[i] = frame

    # Undrift photons
    for i in range(number_photons):
        frame = frames[i]
        undrifted_x[i] = photons_x[i] + (shift - drift_x[frame])
        undrifted_y[i] = photons_y[i] + (shift - drift_y[frame])

    return undrifted_x, undrifted_y