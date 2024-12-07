#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 21:05:14 2024

@author: roberthollmann
"""

import numpy as np
from photon_sorter import sort_photons
import time 

# Generate test data
num_photons = 10000000  # Smaller for testing; increase for real use cases
x_coords = np.random.randint(0, 2359, size=num_photons, dtype=np.int16)
y_coords = np.random.randint(0, 2359, size=num_photons, dtype=np.int16)
dt_values = np.random.randint(0, 1000, size=num_photons, dtype=np.int16)
ms_values = np.random.randint(0, 1000000, size=num_photons, dtype=np.int32)


start_time = time.time()

# Sort photons into the grid
grid = sort_photons(x_coords, y_coords, dt_values, ms_values)

end_time = time.time()


elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")


# Check a specific bin
x_bin, y_bin = 10, 20
print(f"Bin ({x_bin}, {y_bin}) contains {len(grid[x_bin][y_bin])} photons")

print(len(grid[x_bin][y_bin]), 'photons in bin 10,20')
