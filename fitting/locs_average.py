import numpy as np

def avg_photon_weighted(localizations, column):
    '''

    Parameters
    ----------
    localizations : localizations where one column should be averaged over
    with photons weight
    column : column to be averaged, e.g. 'x' or 'lifetime'

    Returns
    -------
    average : sum over i: value[i]*photons[i]/total_photons

    '''
    localizations = localizations.reset_index(drop=True)
    column_sum = 0
    total_photons = 0
    for i in np.arange(len(localizations), dtype=int):
        loc_photons = localizations.loc[i, 'photons']
        column_sum += (localizations.loc[i, column] * loc_photons)
        total_photons += loc_photons
    average = column_sum/total_photons
    #print('average value is: ', average)
    return average