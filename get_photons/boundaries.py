import numpy as np
import pandas as pd

def min_max_box(localizations, box_side_length=0):
    '''
    Returns x, y, boundaries + box
    for a set of localizations as pd dataframe
    '''
    min_x = min(localizations.x) - (box_side_length / 2)
    max_x = max(localizations.x) + (box_side_length / 2)
    min_y = min(localizations.y) - (box_side_length / 2)
    max_y = max(localizations.y) + (box_side_length / 2)

    return min_x, max_x, min_y, max_y


def spatial_boundaries(event, radius):
    x_min = event.x - (radius / 2)
    x_max = x_min + radius

    y_min = event.y - (radius / 2)
    y_max = y_min + radius

    return x_min, x_max, y_min, y_max


def loc_boundaries(localization, offset,
                   box_side_length, integration_time):
    '''
    Returns boundaries in x, y, ms of a single localization (pd Series)
    as a rectangular box

    '''

    x_min = localization.x - (box_side_length / 2)
    x_max = x_min + box_side_length

    y_min = localization.y - (box_side_length / 2)
    y_max = y_min + box_side_length

    ms_min = (localization.frame / offset) * integration_time
    ms_max = ms_min + integration_time

    return x_min, x_max, y_min, y_max, ms_min, ms_max


def crop_event(event, photons, radius):
    '''
    Parameters
    ----------
    localization : single localization as pd Series
    photons : photons as pd DataFrame
    offset :
    box_side_length :
    integration_time :

    Returns
    -------
    photons_cylinder : All photons from the current frame closer than
    box_side_length/2 to the localization position

    '''

    x_min, x_max, y_min, y_max = spatial_boundaries(event, radius)

    photons_cropped = pd.DataFrame(data=crop_photons(
        photons,
        x_min, x_max,
        y_min, y_max,
        event['start_ms'], event['end_ms']))

    #total_photons = len(photons_cropped)

    x_distance = (photons_cropped['x'].to_numpy() - event.x)
    y_distance = (photons_cropped['y'].to_numpy() - event.y)

    total_distance_sq = np.square(x_distance) + np.square(y_distance)
    photons_cropped['distance'] = total_distance_sq

    radius_sq = ((0.5 * radius) ** 2)
    photons_cylinder = photons_cropped[
        photons_cropped.distance < radius_sq]

    #bg_photons = total_photons - len(photons_cylinder)

    if len(photons_cylinder) < 30:
        print('\nlow photon count for crop_event: ')
        print('len(pick_photons) : ', len(photons_cylinder))
        print('\nthis is the event: \n', event)

    return photons_cylinder


def crop_cylinder(localization, photons, offset,
                  box_side_length, integration_time):
    '''
    Parameters
    ----------
    localization : single localization as pd Series
    photons : photons as pd DataFrame
    offset :
    box_side_length : in pixel
    integration_time :

    Returns
    -------
    photons_cylinder : All photons from the current frame closer than
    box_side_length/2 to the localization position

    '''

    x_min, x_max, y_min, y_max, ms_min, ms_max = loc_boundaries(
        localization, offset, box_side_length, integration_time)

    photons_cropped = pd.DataFrame(data=crop_photons(
        photons,
        x_min, x_max,
        y_min, y_max,
        ms_min, ms_max))

    x_distance = (photons_cropped['x'].to_numpy() - localization.x)
    y_distance = (photons_cropped['y'].to_numpy() - localization.y)

    total_distance_sq = np.square(x_distance) + np.square(y_distance)
    photons_cropped['distance'] = total_distance_sq

    radius_sq = ((0.5 * box_side_length) ** 2)
    photons_cylinder = photons_cropped[
        photons_cropped.distance < radius_sq]

    bg_per_pixel = ((len(photons_cropped) - len(photons_cylinder))/
                    ((box_side_length**2) * 0.21460183)) # 0.214... is (1-pi/4), (square - circle)

    return photons_cylinder, bg_per_pixel


def crop_photons(photons, x_min=0, x_max=float('inf'), y_min=0,
                 y_max=float('inf'), ms_min=0, ms_max=float('inf')):
    '''
    Parameters
    ----------
    photons : photons as pd dataframe
    x_min :
    x_max :
    y_min :
    y_max :
    ms_min : optional The default is None.
    ms_max : optional The default is None.

    Returns
    -------
    cropped photons as pd dataframe

    '''

    photons_cropped = photons[
        (photons.x >= x_min)
        & (photons.x <= x_max)
        & (photons.y >= y_min)
        & (photons.y <= y_max)
        & (photons.ms >= ms_min)
        & (photons.ms <= ms_max)]

    return photons_cropped