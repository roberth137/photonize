#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:08:39 2024

@author: roberthollmann
"""

import pandas as pd
import numpy as np

class event:
    def __init__(self,
                 x,
                 y,
                 number_locs,
                 frames,
                 start_ms, 
                 end_ms, 
                 num_photons,
                 bg):
        self.x = x
        self.y = y
        self.num_photons = num_photons
        self.frames = frames
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.num_photons = num_photons
        self.bg = bg
        
