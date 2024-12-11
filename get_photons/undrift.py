import numpy as np
import pandas as pd

def undrift_photons(photons, drift, offset, int_time=200):
    '''
    IN:
    - photon_index - list of all photons (x, y, dt, ms) as pd dataframe
    - drift_file - picasso generated as pd DataFrame
    - integration_time
    OUT:
    undrifted photons_index as dataframe


    Note: drift array is subtracted from locs to get undrifted coordinates
    0.53125 is added to coordinates to convert coordinates
    from camera pixels (LINCAM) to TIFfile to picasso coordinates.

    Formula: Picasso_coord = LIN_coord + 0.5 + (1/(2*Binning))

    For 16x Binning: P_c = L_c + 0.5 +(1/(2*16)) = L_c + 0.53125

    '''
    # create frame array
    ms_index = np.copy(photons.ms)
    frames = np.floor((offset * ms_index) / int_time).astype(int)
    max_frame_photons = max(frames)
    max_frame_drift = len(drift.x)
    if max_frame_photons > max_frame_drift - 1:
        overhang_ms = 0
        for i in range(len(frames)):
            if frames[i] > max_frame_drift - 1:
                frames[i] = max_frame_drift - 1
                overhang_ms += 1
        print('overhang for ', overhang_ms, ' number of photons.')
    drift_x = np.copy(drift.x)
    drift_y = np.copy(drift.y)

    # create numpy arrays to speed up
    number_photons = len(photons)
    undrifted_x = np.copy(photons.x)
    undrifted_y = np.copy(photons.y)
    drift_x_array = np.ones(number_photons)
    drift_y_array = np.ones(number_photons)
    print('length of drift_x: ', len(drift_x))
    print('max frame: ', max(frames))
    for i in range(number_photons):
        frame = frames[i]
        drift_x_array[i] = drift_x[frame]
        drift_y_array[i] = drift_y[frame]
        if i == 0:
            print('start undrifting')
        elif i % 10000000 == 0:
            print('100mio undrifted')

    # apply drift and shift of 0.53125 to photons -> synchron in position
    # with Localizations
    undrifted_x += (0.53125 - drift_x_array)
    undrifted_y += (0.53125 - drift_y_array)

    # create and return new dataframe
    photons_undrifted = pd.DataFrame({'x': undrifted_x,
                                      'y': undrifted_y, 'dt': photons.dt, 'ms': photons.ms})

    return photons_undrifted

