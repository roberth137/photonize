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

def spatial_boundaries(event, diameter):
    x_min = event.x - (diameter / 2)
    x_max = x_min + diameter

    y_min = event.y - (diameter / 2)
    y_max = y_min + diameter

    return x_min, x_max, y_min, y_max


def loc_boundaries(localization, offset,
                   diameter, integration_time):
    '''
    Returns boundaries in x, y, ms of a single localization (pd Series)
    as a rectangular box

    '''

    x_min = localization.x - (diameter / 2)
    x_max = x_min + diameter

    y_min = localization.y - (diameter / 2)
    y_max = y_min + diameter

    ms_min = (localization.frame / offset) * integration_time
    ms_max = ms_min + integration_time

    return x_min, x_max, y_min, y_max, ms_min, ms_max