import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

photons = pd.read_hdf('t/orig58_index.hdf5', key='photons')

photon_coords = photons[['x', 'y']].to_numpy()
photon_tree = cKDTree(photon_coords)
