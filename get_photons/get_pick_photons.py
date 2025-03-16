from get_photons.crop_photons import crop_cuboid
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
    dr_x, dr_y = max(abs(drift.x)), max(abs(drift.y))
    min_x, max_x, min_y, max_y = min_max_box(locs_group, diameter + 1)
    phot_cr = crop_cuboid(photons,
                           (min_x - 0.46875 - dr_x),
                           (max_x - 0.46875 + dr_x),
                           (min_y - 0.46875 - dr_y),
                           (max_y - 0.46875 + dr_y))
    # print('number of cropped photons: ', len(phot_cr))
    # undrift photons
    phot_cr_und = undrift_photons(phot_cr, drift, offset, int_time)
    # crop photons again after drift
    phot_cr_und_cr = crop_cuboid(phot_cr_und,
                                  min_x, max_x, min_y, max_y)
    return phot_cr_und_cr


def crop_undrift_crop(
        locs_group, photons, drift, offset,
        box_side_length, int_time):
    """
    Parameters
    ----------
    locs_group : localizations of this pick (group) as pd dataframe
    photons : photons as pd dataframe
    drift : drift as pd dataframe
    integration time: camera integration time
    box_side_length: size of the PSF in pixels

    Returns
    -------
    All drift-corrected photons in the area
    of the pick +- diameter/2
    """
    # set dimensions of the region and crop photons
    # -0.53125 because: -> see undrift (pixel conversion)
    dr_x, dr_y = max(abs(drift.x)), max(abs(drift.y))
    min_x, max_x, min_y, max_y = min_max_box(locs_group, box_side_length)
    phot_cr = crop_cuboid(photons,
                           (min_x - 0.46875 - dr_x),
                           (max_x - 0.46875 + dr_x),
                           (min_y - 0.46875 - dr_y),
                           (max_y - 0.46875 + dr_y))
    print('number of cropped photons: ', len(phot_cr))
    # undrift photons
    phot_cr_und = (undrift_photons(phot_cr, drift, offset, int_time))
    # crop photons again after drift
    phot_cr_und_cr = crop_cuboid(phot_cr_und,
                                  min_x, max_x, min_y, max_y)
    print('number of cropped-undrifted-cropped photons: ',
          len(phot_cr_und_cr))
    return phot_cr_und_cr