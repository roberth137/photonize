import simulate as s
import numpy as np

def distance_to_point(x, y, x_ref, y_ref, max_dist=s.max_dist):
    """
    Calculate the distance from each (x[i], y[i]) to the given point.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates.
    y : np.ndarray
        1D array of y-coordinates.
    point : tuple or list of length 2
        The reference point (x0, y0).
    max_dist : max_dist to point

    Returns
    -------
    x: np.ndarray
    y: np.ndarray

    dist: 1D array of distances from each (x[i], y[i]) to 'point'.
    """

    dist = np.sqrt((x - x_ref)**2 + (y - y_ref)**2)

    if max_dist:
        mask = dist < max_dist
        return x[mask], y[mask], dist[mask]
    else:
        return x, y, dist