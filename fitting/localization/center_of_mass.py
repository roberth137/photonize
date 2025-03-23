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

#@numba.njit
def calculate_sd(photon_positions, mean_position, total_photons):
    return np.sqrt(np.sum((photon_positions - mean_position) ** 2) / total_photons)



def com_position(x_phot, y_phot, x_ref, y_ref, bg_count=None):

    num_photons = len(x_phot)

    if bg_count:
        num_photons -= bg_count

    if bg_count is not None:
        pos_x = (np.sum(x_phot) - bg_count * x_ref) / num_photons
        pos_y = (np.sum(y_phot) - bg_count * y_ref) / num_photons
    else:
        pos_x = np.sum(x_phot) / num_photons
        pos_y = np.sum(y_phot) / num_photons

    return pos_x.astype(np.float32), pos_y.astype(np.float32)


def event_position_cons_bg(event, phot_event, diameter, return_sd=True):

    x_photons = phot_event['x'].to_numpy()
    y_photons = phot_event['y'].to_numpy()

    fit_area = np.pi * ((diameter / 2) ** 2)
    total_photons = len(phot_event)
    number_phot = (total_photons - fit_area * event.bg) #fit area in pixel^2 #bg per pixel
    bg = event.bg * fit_area
    #bg = 1 * fit_area

    pos_x = (np.sum(x_photons) - bg * event.x) / number_phot
    pos_y = (np.sum(y_photons) - bg * event.y) / number_phot

    if return_sd:
        sd_x = calculate_sd_cons_bg(x_photons, pos_x, number_phot, bg, diameter)
        sd_y = calculate_sd_cons_bg(y_photons, pos_y, number_phot, bg, diameter)

        return pos_x, pos_y, sd_x, sd_y

    else:
        return pos_x, pos_y

def calculate_sd_cons_bg(positions, pos_fit, number_phot, bg_total, diameter):
        """
        Calculates 1d std for a center of mass fit:
        positions: array with photons positions
        pos_fit: fitted position
        number_photons: total_photons - bg
        bg_total: total number of background photons
        diameter: diameter of roi
        """
        bg_var = ((diameter / 4) ** 2) / 2  # s.d. in 1d of a random distribution on disk
        var = (np.sum((positions - pos_fit) ** 2) - (bg_var * bg_total)) / number_phot
        return 10 if var <= 0 else np.sqrt(var)

