#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:10:41 2024

@author: roberthollmann
"""

import pandas as pd
import core
import resi
import matplotlib.pyplot as plt
import numpy as np


#photons = pd.read_hdf('80mhz_10k_index.hdf5', key='photons')
#drift = pd.read_csv('drift10k_80.txt', delimiter=' ',names =['x','y'])
#phot_und = core.undrift(photons, drift, 20)
localizations = pd.read_hdf('orig2_event_based.hdf5', key='locs')

#phot_locs = resi.photons_of_one_localization(localizations.iloc[48], 
#                                             phot_und, 20)

##

plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)

plt.hist(localizations.lifetime, bins=np.arange(200,450,5))