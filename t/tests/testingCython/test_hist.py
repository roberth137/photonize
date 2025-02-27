import numpy as np
from histogram import create_2d_histogram
import pandas as pd
import time

# Generate random test data
num_photons = 10000000 # Smaller number for quick testing
photons_rand = np.random.rand(num_photons, 4) * [147, 147, 1000000, 1000]

#photons = pd.read_hdf('photons.hdf5', key='photons')

#photons_np = photons.to_numpy()

#photons_np = photons_np[::10]
#print('photons read in. histogramming: ', len(photons_np), ' photons.')
#time.sleep(2)
start_time = time.time()

# Test the histogram function
histogram = create_2d_histogram(photons_rand, num_bins=148)

end_time = time.time()

print("2D histogram created successfully!")

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")