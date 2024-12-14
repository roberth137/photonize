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
    #x_pos = np.sum(phot_locs.x)/total_photons - (localization.x * bg_photons/total_photons)
    #y_pos = np.sum(phot_locs.y)/total_photons - (localization.y * bg_photons/total_photons)

    x_pos = (np.sum(phot_locs.x) - bg * localization.x) / number_phot
    y_pos = (np.sum(phot_locs.y) - bg * localization.y) / number_phot
    return x_pos, y_pos



def event_position(event, phot_event, radius):

    x_photons = phot_event['x'].to_numpy()
    y_photons = phot_event['y'].to_numpy()
    fit_area = np.pi * ((radius / 2) ** 2)
    total_photons = len(phot_event)
    number_phot = (total_photons - fit_area * event.bg)
    bg = event.bg * fit_area

    pos_x = (np.sum(x_photons) - bg * event.x) / number_phot
    pos_y = (np.sum(y_photons) - bg * event.y) / number_phot

    bg_var = (radius**2)/4 # s.d. of a disc with random and homogenous
                                # distributed measurements
                                # in one dimension

    var_x = (np.sum((x_photons - pos_x) ** 2) - (bg_var * bg))/number_phot
    var_y = (np.sum((x_photons - pos_y) ** 2) - (bg_var * bg))/number_phot
    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)

    return pos_x, pos_y, std_x, std_y




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