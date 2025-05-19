"""
Created on Wed Nov 27 10:19:22 2024

@author: roberthollmann
"""
from get_photons import boundaries
import pandas as pd
import numpy as np

def crop_rectangle(
    photons,
    x_min=0, x_max=float('inf'),
    y_min=0, y_max=float('inf')):
    """
        Takes a DataFrame of photons and returns a x and y filtered subset.

        Parameters
        ----------
        photons : pandas.DataFrame
            Input DataFrame, expected to have columns: x, y, ms.
        x_min, x_max : float, optional
            Bounds on x. If None, that bound is ignored.
        y_min, y_max : float, optional
            Bounds on y. If None, that bound is ignored.
        Returns
        -------
        pandas.DataFrame
            The subset of photons within the specified ranges.
        """
    mask = ((photons.x >= x_min)
        & (photons.x <= x_max)
        & (photons.y >= y_min)
        & (photons.y <= y_max))

    return photons[mask]


def crop_cuboid(photons, x_min=0, x_max=float('inf'), y_min=0,
                 y_max=float('inf'), ms_min=0, ms_max=float('inf')):
    """
    Takes photons as input and return 3d cropped photons as output
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

    """
    mask = ((photons.x >= x_min)
            & (photons.x <= x_max)
            & (photons.y >= y_min)
            & (photons.y <= y_max)
            & (photons.ms >= ms_min)
            & (photons.ms <= ms_max))
    return photons[mask]

def crop_event(event, photons, diameter, more_ms=0):
    '''
    Parameters
    ----------
    event : single event as pd Series
    photons : photons as pd DataFrame
    diameter : considered diameter around event position
    more_ms: additional timewindow to be considered before and after the event

    Returns
    -------
    photons_cylinder : All photons in the cylinder around this event

    '''

    x_min, x_max, y_min, y_max = boundaries.spatial_boundaries(event, diameter)
    if hasattr(event, 'start_ms') and hasattr(event, 'end_ms'):
        start, end = event.start_ms, event.end_ms
    elif hasattr(event, 'start_ms_fr') and hasattr(event, 'end_ms_fr'):
        start, end = event.start_ms_fr, event.end_ms_fr
    else:
        raise AttributeError("Required attributes are missing. Expected either 'start_ms', 'end_ms' or 'start_ms_fr', 'end_ms_fr'.")

    photons_cropped = pd.DataFrame(data=crop_cuboid(
        photons,
        x_min, x_max,
        y_min, y_max,
        (start-more_ms),
        (end+more_ms)))

    x_distance = (photons_cropped['x'].to_numpy() - event.x)
    y_distance = (photons_cropped['y'].to_numpy() - event.y)

    total_distance_sq = np.square(x_distance) + np.square(y_distance)
    photons_cropped['distance'] = total_distance_sq

    radius_sq = ((diameter/2) ** 2)
    photons_cylinder = photons_cropped[
        photons_cropped.distance <= radius_sq]

    if len(photons_cylinder) < 30:
        try: print(f'!!!!!!!!!!\nlow photon count for boundaries.crop_event():'
              f'\nlen(pick_photons): {len(photons_cylinder)}'
              f'\nevent_number: {event.event}')
        except:
            pass
    return photons_cylinder


def crop_loc(localization, photons, offset,
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
    diameter/2 to the localization position

    '''

    x_min, x_max, y_min, y_max, ms_min, ms_max = boundaries.loc_boundaries(
        localization, offset, box_side_length, integration_time)

    photons_cropped = pd.DataFrame(data=crop_cuboid(
        photons,
        x_min, x_max,
        y_min, y_max,
        ms_min, ms_max))

    x_distance = (photons_cropped['x'].to_numpy() - localization.x)
    y_distance = (photons_cropped['y'].to_numpy() - localization.y)

    total_distance_sq = np.square(x_distance) + np.square(y_distance)
    photons_cropped['distance'] = total_distance_sq

    radius_sq = ((box_side_length/2) ** 2)

    photons_cylinder = photons_cropped[photons_cropped.distance < radius_sq]

    return photons_cylinder