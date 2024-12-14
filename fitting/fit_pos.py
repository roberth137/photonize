import numpy as np


def avg_of_roi(localization, phot_locs, box_side_length, return_sd=False):
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

    x_photons = phot_locs['x'].to_numpy()
    y_photons = phot_locs['y'].to_numpy()

    fit_area = np.pi * ((box_side_length / 2) ** 2)
    total_photons = len(phot_locs)
    number_phot = (total_photons - fit_area * localization.bg)
    bg = localization.bg * fit_area

    pos_x = (np.sum(phot_locs.x) - bg * localization.x) / number_phot
    pos_y = (np.sum(phot_locs.y) - bg * localization.y) / number_phot

    if return_sd:
        bg_var = ((box_side_length/2) ** 2) / 4  # s.d. in 1d of a random distr on disk
        tot_dev_x2 = np.sum((x_photons - pos_x) ** 2)
        tot_dev_y2 = np.sum((x_photons - pos_x) ** 2)
        var_x = (tot_dev_x2 - (bg_var * bg)) / number_phot
        var_y = (tot_dev_y2 - (bg_var * bg)) / number_phot
        print('variances are: tot_dev_x2, var_x, tot_dev_y2, var_y, bg_var, bg')
        print(tot_dev_x2, var_x, tot_dev_y2, var_y, bg_var, bg)
        std_x = np.sqrt(var_x)
        std_y = np.sqrt(var_y)

        return pos_x, pos_y, std_x, std_y

    else:
        return pos_x, pos_y


def calculate_sd(positions, pos_true, bg, radius, number_phot):
    bg_var = (radius ** 2) / 4  # s.d. in 1d of a random distr on disk
    var_x = (np.sum((positions - pos_true) ** 2) - (bg_var * bg)) / number_phot
    std_x = np.sqrt(var_x)
    return std_x



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