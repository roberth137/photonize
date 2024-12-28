import pandas as pd
import h5py
from pathlib import Path
import shutil

def crop_photons_file(new_photons_df, photons_file, extension='_20k'):
    '''

    Parameters
    ----------
    dataframe : dataframe in picasso format (with all necessary columns)
    filename : name with which the file will be saved

    DO: takes a dataframe and saves it to picasso format
    The corresponding yaml file has to be in the same directory and will be copied
    '''

    path = str(Path.cwd())
    labels = list(new_photons_df.keys())
    new_photons_df = new_photons_df.reindex(columns=labels, fill_value=1)
    photons = new_photons_df.to_records(index=False)


    hf = h5py.File(path + '/' + photons_file[:-5] + extension + '.hdf5', 'w')
    hf.create_dataset('photons', data=photons)
    hf.close()
    print('\nphotons successfully saved.')

def crop_drift(drift_file):
    """
    Modify a drift file to use only for specific frames
    """
    # Load the data from drift.txt
    drift_file = pd.read_csv(drift_file, delimiter=' ', names=['x', 'y'])

    # Keep only the first 20,000 rows
    drift_20k = drift_file.iloc[120000:140000]

    # Save the reduced dataset to drift_20k.txt
    drift_20k.to_csv('drift_L20.txt', sep=' ', index=False, header=False)

    print("Saved first 20k rows to drift_20k.txt")
