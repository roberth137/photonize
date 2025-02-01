import numpy as np
import pandas as pd
import numba


@numba.njit
def apply_drift_correction(x, y, frames, drift_x, drift_y, num_photons, max_frame_drift):
    undrifted_x = np.empty(num_photons)
    undrifted_y = np.empty(num_photons)

    for i in range(num_photons):
        frame = frames[i]
        if frame >= max_frame_drift:
            frame = max_frame_drift - 1  # Prevent out-of-bounds access

        # Apply drift and fixed correction offset of 0.46875
        undrifted_x[i] = x[i] - (0.46875 + drift_x[frame])
        undrifted_y[i] = y[i] - (0.46875 + drift_y[frame])

    return undrifted_x, undrifted_y


def undrift_photons(photons, drift, offset, int_time=200):
    # Convert DataFrame columns to NumPy arrays
    ms_index = photons['ms'].to_numpy()
    x_photons = photons['x'].to_numpy()
    y_photons = photons['y'].to_numpy()

    # Calculate frame indices for each photon
    frames = np.floor((offset * ms_index) / int_time).astype(np.int32)

    # Handle drift arrays
    drift_x = drift['x'].to_numpy()
    drift_y = drift['y'].to_numpy()

    # Get necessary dimensions
    num_photons = len(photons)
    max_frame_drift = len(drift_x)

    # Apply drift correction using the Numba function
    undrifted_x, undrifted_y = apply_drift_correction(
        x_photons, y_photons, frames, drift_x, drift_y, num_photons, max_frame_drift
    )

    # Create a new DataFrame with undrifted coordinates
    photons_undrifted = pd.DataFrame({
        'x': undrifted_x,
        'y': undrifted_y,
        'dt': photons['dt'].to_numpy(),
        'ms': ms_index
    })

    print('Finished undrifting!')
    return photons_undrifted


def old_undrift_photons(photons, drift, offset, int_time=200):
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
    #print('length ms_index is: ', len(ms_index))
    #print('length frames_array is: ', len(frames))
    #print('highes ms is: ', max(ms_index))
    max_frame_photons = max(frames)
    max_frame_drift = len(drift.x)
    if max_frame_photons > max_frame_drift - 1:
        overhang_ms = 0
        for i in range(len(frames)):
            if frames[i] > max_frame_drift - 1:
                frames[i] = max_frame_drift - 1
                overhang_ms += 1
        #print('overhang for ', overhang_ms, ' number of photons.')
    drift_x = np.copy(drift.x)
    drift_y = np.copy(drift.y)

    # create numpy arrays to speed up
    number_photons = len(photons)
    undrifted_x = np.copy(photons.x)
    undrifted_y = np.copy(photons.y)
    drift_x_array = np.ones(number_photons)
    drift_y_array = np.ones(number_photons)
    #print('length of drift_x: ', len(drift_x))
    #print('max frame: ', max(frames))
    for i in range(number_photons):
        frame = frames[i]
        drift_x_array[i] = drift_x[frame]
        drift_y_array[i] = drift_y[frame]
        if i == 0:
            print(f'start undrifting {number_photons} photons')
        elif i % 10000000 == 0:
            print('100mio undrifted')

    # apply drift and shift of 0.53125 to photons -> synchron in position
    # with Localizations
    undrifted_x -= (0.46875 + drift_x_array)
    undrifted_y -= (0.46875 + drift_y_array)
    # create and return new dataframe
    photons_undrifted = pd.DataFrame({'x': undrifted_x,
                                      'y': undrifted_y, 'dt': photons.dt, 'ms': photons.ms})
    print('finished undrifting!')
    return photons_undrifted

