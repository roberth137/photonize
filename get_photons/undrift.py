import numpy as np
import pandas as pd
import numba


@numba.njit
def apply_drift_correction(x, y, frames, drift_x, drift_y, num_photons, max_frame_drift):
    """
    Numba optimized function that substracts drift and aligns photons with picasso localizations

    Alingment Formula:
    For 16x Binning: P_c = L_c - (0.5 - (1/(2*Binning))) = L_c - 0.46875

    Reason: If binning 16 individual pixels to a TIF, the first 16 pixel (values 0-15)
    are binned into the 0 pixel. This means the average value of the 0 pixel is 7.5/16 which is 0.46875
    """
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
    """
    subtracts drift and aligns photons to picasso coordinates
    Using Numba optimized apply drift correction function
    Input:
        photons - pd dataframe with x_array,y_array,dt,ms coordinates
        drift - pd dataframe
        offset -
        int_time -
    Returns:
        undrifted and aligned photons
    """
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