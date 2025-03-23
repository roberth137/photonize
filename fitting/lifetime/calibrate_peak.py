import numpy as np

def calibrate_peak_arrival(event_photons):
    """
    Estimates the peak arrival time of photons by finding the mode
    of the arrival time histogram.

    Parameters
    ----------
    event_photons : DataFrame or similar object
        Photon data for an event or acquisition. Must contain a `dt` attribute
        or column representing photon arrival times (e.g., in nanoseconds or milliseconds).
        It is recommended to use at most ~1,000,000 photons to avoid performance issues.

    Returns
    -------
    peak_time : int
        The time bin (in the same units as `dt`) corresponding to the peak of the
        arrival time histogram (i.e., the most frequent arrival time).

    Notes
    -----
    - This function creates a histogram of `dt` values using bins from 0 to 2500 (step = 1).
    - The function returns the bin center corresponding to the maximum count.
    - Useful for aligning or calibrating arrival time distributions in FLIM or similar photon arrival data.

    """
    min_dt = np.min(event_photons.dt)
    max_dt = np.max(event_photons.dt)
    counts, bins = np.histogram(event_photons.dt, bins=np.arange(min_dt, max_dt))
    print(f'len photons for calib_peak: {len(event_photons)}'
          f'min_dt: {min_dt}, max_dt: {max_dt}')
    return np.argmax(counts)
