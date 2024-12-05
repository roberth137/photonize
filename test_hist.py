import numpy as np
from histogram import create_2d_histogram
import pandas as pd

# Generate random test data
num_photons = 1000 # Smaller number for quick testing
#photons1 = np.random.rand(num_photons, 4) * [147, 147, 1000000, 1000]

photons = pd.read_hdf('photons.hdf5', key='photons')

photons_np = photons.to_numpy()

photons_np = photons_np[::1000]

# Test the histogram function
histogram = create_2d_histogram(photons_np, num_bins=148)
print("2D histogram created successfully!")