import numpy as np


def avg_of_roi(localization, phot_locs, box_side_length):
    '''
    Parameters
    ----------
    phot_locs : photons of one localization as pd dataframe

    Returns
    -------
    - x position
    - y position

    Position is calculated by adding up all photons in the circular
    surrounding of the localization.
    Background gets subtracted

    '''
    fit_area = np.pi * ((box_side_length / 2) ** 2)
    total_photons = len(phot_locs)
    number_phot = (total_photons - fit_area * localization.bg)
    bg = localization.bg * fit_area
    bg_photons = fit_area * localization.bg
    ###
    #x_pos = np.mean(phot_locs.x) - (localization.x * bg_photons/total_photons)
    #y_pos = np.mean(phot_locs.y) - (localization.y * bg_photons/total_photons)

    x_pos = (np.sum(phot_locs.x) - bg * localization.x) / number_phot
    y_pos = (np.sum(phot_locs.y) - bg * localization.y) / number_phot
    return x_pos, y_pos

def standard_deviation(localization, phot_locs, box_side_length):
    """
    Calculated the standart deviation of all photons from the fitted
    """

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