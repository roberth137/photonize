import numpy as np
import numba

#@numba.njit
def localize_com(x_photons, y_photons, return_sd=True):
    #fit_area = np.pi * ((diameter / 2) ** 2)
    total_photons = len(x_photons)

    pos_x = (np.sum(x_photons) / total_photons)
    pos_y = np.sum(y_photons) / total_photons

    if return_sd:
        sd_x = calculate_sd(x_photons, pos_x, total_photons)
        sd_y = calculate_sd(y_photons, pos_y, total_photons)
        return (pos_x.astype(np.float32),
                pos_y.astype(np.float32),
                sd_x.astype(np.float32),
                sd_y.astype(np.float32))
    else:
        sd_x, sd_y = 0.0, 0.0
        return (pos_x.astype(np.float32),
                pos_y.astype(np.float32),
                sd_x,
                sd_y)

#numba.njit
def calculate_sd(photon_positions, mean_position, total_photons):
    return np.sqrt(np.sum((photon_positions - mean_position) ** 2) / total_photons)
