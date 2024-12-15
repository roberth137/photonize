import numpy as np


def avg_of_roi(localization, bg_pixel, phot_locs, box_side_length, return_sd=False):
    """
    Parameters
    ----------
    phot_locs : photons of one localization as pd dataframe
    Returns
    -------
    - x position
    - y position
    - sd_x
    - sd_y

    Position is calculated by adding up all photons in the circular
    surrounding of the localization.
    Background gets subtracted
    """

    x_photons = phot_locs['x'].to_numpy()
    y_photons = phot_locs['y'].to_numpy()

    fit_area = np.pi * ((box_side_length / 2) ** 2)
    total_photons = len(phot_locs)
    bg_photons = bg_pixel * fit_area
    number_phot = (total_photons - bg_photons)

    pos_x = (np.sum(x_photons) - bg_photons * localization.x) / number_phot
    pos_y = (np.sum(y_photons) - bg_photons * localization.y) / number_phot

    if return_sd:

        sd_x = calculate_sd(x_photons, pos_x, number_phot, bg_photons, box_side_length)
        sd_y = calculate_sd(y_photons, pos_y, number_phot, bg_photons, box_side_length)

        return pos_x, pos_y, sd_x, sd_y

    else:
        return pos_x, pos_y



def calculate_sd(positions, pos_fit, number_phot, bg_total, diameter):
    """
    Calculates 1d std for a center of mass fit:
    positions: array with photons positions
    pos_fit: fitted position
    number_photons: total_photons - bg
    bg_total: total number of background photons
    diameter: diameter of roi
    """
    bg_var = ((diameter/4) ** 2) / 4  # s.d. in 1d of a random distribution on disk
    var = (np.sum((positions - pos_fit) ** 2) - (bg_var * bg_total)) / number_phot
    return 10 if var <= 0 else np.sqrt(var)


def localization_precision(photons, sigma, bg):
    """
    Calculates the theoretical localization precision according to
    Mortensen et al., Nat Meth, 2010 for a 2D unweighted Gaussian fit.
    Copied from picasso
    """
    sigma2 = sigma**2
    sigma_a2 = sigma2 + 1 / 12
    v = sigma_a2 * (16 / 9 + (8 * np.pi * sigma_a2 * bg) / photons) / photons
    with np.errstate(invalid="ignore"):
        return np.sqrt(v)


def event_position(event, phot_event, diameter, return_sd=True):

    x_photons = phot_event['x'].to_numpy()
    y_photons = phot_event['y'].to_numpy()

    fit_area = np.pi * ((diameter / 2) ** 2)
    total_photons = len(phot_event)
    number_phot = (total_photons - fit_area * event.bg)
    bg = event.bg * fit_area

    pos_x = (np.sum(x_photons) - bg * event.x) / number_phot
    pos_y = (np.sum(y_photons) - bg * event.y) / number_phot

    if return_sd:
        sd_x = calculate_sd(x_photons, pos_x, number_phot, bg, diameter)
        sd_y = calculate_sd(y_photons, pos_y, number_phot, bg, diameter)

        return pos_x, pos_y, sd_x, sd_y

    else:
        return pos_x, pos_y