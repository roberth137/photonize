import numpy as np


def avg_of_roi(localization, phot_locs, box_side_length, return_sd=False):
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
    number_phot = (total_photons - fit_area * localization.bg)
    bg = localization.bg * fit_area

    pos_x = (np.sum(x_photons) - bg * localization.x) / number_phot
    pos_y = (np.sum(y_photons) - bg * localization.y) / number_phot

    if return_sd:

        sd_x = calculate_sd(x_photons, pos_x, number_phot, bg, box_side_length)
        sd_y = calculate_sd(y_photons, pos_y, number_phot, bg, box_side_length)

        return pos_x, pos_y, sd_x, sd_y

    else:
        return pos_x, pos_y


def event_position(event, phot_event, radius, return_sd=True):

    x_photons = phot_event['x'].to_numpy()
    y_photons = phot_event['y'].to_numpy()

    fit_area = np.pi * ((radius / 2) ** 2)
    total_photons = len(phot_event)
    number_phot = (total_photons - fit_area * event.bg)
    bg = event.bg * fit_area

    pos_x = (np.sum(x_photons) - bg * event.x) / number_phot
    pos_y = (np.sum(y_photons) - bg * event.y) / number_phot

    if return_sd:
        sd_x = calculate_sd(x_photons, pos_x, number_phot, bg, radius)
        sd_y = calculate_sd(y_photons, pos_y, number_phot, bg, radius)

        return pos_x, pos_y, sd_x, sd_y

    else:
        return pos_x, pos_y



def calculate_sd(positions, pos_fit, number_phot, bg, radius):
    """
    Calculates 1d std for a center of mass fit:
    positions: array with photons positions
    pos_fit: fitted position
    number_photons: total_photons - bg
    bg: number of background photons
    radius: radius of roi
    """
    bg_var = ((radius/2) ** 2) / 4  # s.d. in 1d of a random distribution on disk
    var = (np.sum((positions - pos_fit) ** 2) - (bg_var * bg)) / number_phot
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