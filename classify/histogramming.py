import numpy as np
import pandas as pd
from utilities import helper
import fitting
import get_photons


def create_histogram_dataframe(input_events, input_photons, drift_file,
                               offset, diameter, int_time, bin_size=20):
    """
    Process events and photons to compute histograms for each event.

    Parameters
    ----------
    input_events : str
        Path to the events file (e.g. an HDF5 file with event localizations).
    input_photons : str
        Path to the photons file (e.g. an HDF5 file with photon data).
    drift_file : str
        Path to the drift correction file.
    offset : int or float
        Offset value used in selecting the correct photons.
    diameter : float
        Diameter used for spatial cropping.
    int_time : int or float
        Integration time (used for photon selection).
    bin_size : int or float, optional
        Bin size for the histogram (default is 20, in the same units as dt).

    Returns
    -------
    histograms : pd.DataFrame
        A DataFrame containing histogram values (log-scaled) for each event.
        The columns are named "bin_0", "bin_1", ..., "bin_N" and "event".
    """
    # Process input files
    events = helper.process_input(input_events, 'locs')
    photons = helper.process_input(input_photons, 'photons')
    drift = helper.process_input(drift_file, 'drift')

    # Calibrate and define bins for the histograms
    peak_arrival_time = fitting.calibrate_peak_events(photons[:1000000])
    max_dt = max(photons[:1000000].dt)
    bins = np.arange(peak_arrival_time, max_dt, bin_size)
    print(f'peak arrival time: {peak_arrival_time}')
    print(f'first bins: {bins[:5]}, . . . last bins: {bins[-5:]}')

    num_bins = len(bins) - 1  # np.histogram returns one more edge than bins

    # Create column names for each bin and the event id
    column_names = [f'bin_{i}' for i in range(num_bins)] + ['event']

    histogram_list = []  # will hold dictionaries for each event's histogram
    delta_phot = 0

    # Process events grouped by their 'group' field
    for group in events['group'].unique():
        events_group = events[events.group == group]
        print('_______________')
        print(f'Histogramming group: {group} with {len(events_group)} events.')
        # Get the set of photons corresponding to these events
        pick_photons = get_photons.get_pick_photons(
            events_group, photons, drift, offset,
            diameter=diameter, int_time=int_time
        )
        # Process each event in the group
        for i, event in events_group.iterrows():
            event_photons = get_photons.crop_event(event, pick_photons, diameter)
            delta_phot += (len(event_photons) - event.photons)
            # Compute histogram for the event based on the dt values of photons
            hist, _ = np.histogram(event_photons.dt, bins=bins)
            hist = np.log1p(hist) / 3  # apply log scaling as in your original code
            # Append the event id to the histogram vector
            hist = np.append(hist, event.event)
            # Create a dictionary mapping column names to histogram values
            histogram_list.append(dict(zip(column_names, hist)))

    # Create a DataFrame from the list of histogram dictionaries
    histograms = pd.DataFrame(histogram_list, columns=column_names)
    histograms['event'] = histograms['event'].astype(np.int32)

    print(histograms.head())
    print(f'Average photon difference per event: {delta_phot / len(events)}')

    return histograms


# Example usage:
folder = '/Users/roberthollmann/Desktop/resi-flim/ml/event_data/'
input_events = f'{folder}cy3_200ms_fp_event_f.hdf5'
input_photons = f'{folder}cy3_59_index.hdf5'
drift_file = f'{folder}cy3_200ms_drift.txt'
fluorophore_name = 'cy3_1402_4p5'
offset = 10
diameter = 4.5
int_time = 200

if __name__ == "__main__":
    hist_df = create_histogram_dataframe(input_events, input_photons, drift_file,
                                         offset, diameter, int_time, bin_size=20)