from get_photons.crop_photons import crop_rectangle
from get_photons.undrift import undrift_photons
from get_photons.boundaries import min_max_box

def get_pick_photons(
        locs_group, photons, drift, offset,
        diameter, int_time):
    """
    Parameters
    ----------
    locs_group : localizations of this pick (group) as pd dataframe
    photons : photons as pd dataframe
    drift : drift as pd dataframe
    offset:
    integration time: camera integration time
    diameter: size of the PSF in pixels
    int_time:

    Returns
    -------
    All driftcorrected photons in the area
    of the pick +- diameter/2
    """
    # set dimensions of the region and crop photons
    # -0.46875 because: -> see undrift (pixel conversion)
    print(type(drift))
    print(f'drift: {drift}')
    dr_x, dr_y = max(abs(drift.x)), max(abs(drift.y))
    print(f'drift.x: {drift.x}')
    print(type(drift))
    min_x, max_x, min_y, max_y = min_max_box(locs_group, diameter + 1)
    #crop photons of area of interest plus drift
    phot_cr = crop_rectangle(photons,
                           (min_x - 0.46875 - dr_x),
                           (max_x - 0.46875 + dr_x),
                           (min_y - 0.46875 - dr_y),
                           (max_y - 0.46875 + dr_y))
    # undrift and align photons
    phot_cr_und = undrift_photons(phot_cr, drift, offset, int_time)
    # crop photons again after drift
    phot_cr_und_cr = crop_rectangle(phot_cr_und,
                                  min_x, max_x, min_y, max_y)
    return phot_cr_und_cr