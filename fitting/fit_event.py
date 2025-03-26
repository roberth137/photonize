# This is the main function for analysing a single event

import numpy as np
from fitting.on_off import get_on_off_dur
from fitting.localization import localize_com
from fitting.lifetime import avg_lifetime_weighted
from collections import namedtuple

# Define the named tuple once
EventResult = namedtuple('EventResult', [
    'x_fit', 'y_fit', 'lifetime',
    'start_ms', 'end_ms', 'duration_ms',
    'num_photons'
])

def fit_event(photons, dt_peak, diameter):
    """
    Analyze a single photon event.

    Returns:
        EventResult namedtuple with:
        - x_fit, y_fit: fitted x/y position
        - lifetime: weighted average lifetime
        - start_ms, end_ms: event start/end time
        - duration_ms: event duration
        - num_photons: number of photons in the event
    """
    # get start and end of event
    start_ms, end_ms, duration_ms = get_on_off_dur(photons)

    # filter photons within the event time window
    event_photons = photons[(photons.ms >= start_ms) & (photons.ms <= end_ms)]

    # extract coordinates and arrival times
    x_photons = np.copy(event_photons.x)
    y_photons = np.copy(event_photons.y)
    dt_photons = np.copy(event_photons.dt)

    # fit x and y position using center of mass
    x_fit, y_fit, sdx, sdy = localize_com(
        event_photons.x, event_photons.y, return_sd=True
    )

    # calculate distance from center
    distances = np.sqrt((x_photons - x_fit) ** 2 + (y_photons - y_fit) ** 2)

    # calculate lifetime using weighted averaging
    lifetime = avg_lifetime_weighted(
        dt_photons, distance=distances, peak=dt_peak, diameter=diameter
    )

    # pack into a named tuple and return
    return EventResult(
        x_fit=x_fit,
        y_fit=y_fit,
        lifetime=lifetime,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=duration_ms,
        num_photons=len(event_photons)
    )
