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
    fit_area = (box_side_length / 2) ** 2
    number_phot = (len(phot_locs) - fit_area * localization.bg)
    bg = localization.bg * fit_area
    x_pos = (np.sum(phot_locs.x) - bg * localization.x) / number_phot
    y_pos = (np.sum(phot_locs.y) - bg * localization.y) / number_phot
    return x_pos, y_pos

