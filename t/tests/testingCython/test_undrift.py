#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 21:51:45 2024

@author: roberthollmann
"""

import numpy as np
import pandas as pd
from undrift import undrift

# Example data
#photons = pd.DataFrame({
#    'x': np.random.rand(1000000),
#    'y': np.random.rand(1000000),
#    'ms': np.random.rand(1000000) * 1000,
#    'dt': np.random.rand(1000000),
#})
#drift = pd.DataFrame({
#    'x': np.random.rand(100),
#    'y': np.random.rand(100),
#})

photons = pd.read_hdf('photons.hdf5', key='photons')
drift = pd.read_csv('drift.txt', delimiter=' ',names =['x','y'])


offset = 1.0
int_time = 200.0

# Call the accelerated undrift function
undrifted_x, undrifted_y = undrift(
    photons.x.to_numpy(),
    photons.y.to_numpy(),
    photons.ms.to_numpy(),
    drift.x.to_numpy(),
    drift.y.to_numpy(),
    offset,
    int_time
)

# Combine results into a DataFrame
photons_undrifted = pd.DataFrame({
    'x': undrifted_x,
    'y': undrifted_y,
    'dt': photons.dt,
    'ms': photons.ms,
})
print(photons_undrifted.head())